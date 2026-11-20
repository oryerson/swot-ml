from pathlib import Path
from typing import Optional

import numpy as np
from numpy import ceil, floor
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from pyproj import Geod


def resolve_raster(base_dir, basin) -> str:
    """Locate the per-pfaf2 flow raster, preferring a VRT mosaic over a GeoTIFF.

    The original per-basin GeoTIFF mosaics under MERIT-Hydro/processed/{dir,upg}
    no longer exist; equivalent VRTs can be rebuilt over the raw MERIT-Hydro
    tiles (see CONUS/preprocess/build_merit_vrts.py). Both layouts are accepted
    so existing GeoTIFF setups keep working unchanged.
    """
    base_dir = Path(base_dir)
    for ext in (".vrt", ".tif"):
        path = base_dir / f"{basin}{ext}"
        if path.is_file():
            return str(path)
    raise FileNotFoundError(
        f"No flow raster for basin {basin} in {base_dir} (looked for {basin}.vrt / {basin}.tif)"
    )


MIN_PIXELS = 5
MERIT_RES = 0.000833333
PIXEL_AREA = MERIT_RES * MERIT_RES


def _get_largest_polygon(geom):
    """Extract largest polygon from a geometry (handles MultiPolygon)"""
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda p: p.area)
    return geom


def get_pixel_area_m2(gauge, grid):
    g = Geod(ellps="WGS84")
    lon, lat = gauge["lng"], gauge["lat"]

    # Four corners of pixel around gauge
    x0, y0 = lon, lat
    x1 = lon + grid.affine[0]
    y1 = lat + grid.affine[4]  # negative

    area, _ = g.polygon_area_perimeter([x0, x1, x1, x0], [y0, y0, y1, y1])
    pixel_area_m2 = abs(area)

    return pixel_area_m2


def split_catchment_raster(
    gauge: dict,
    basin: int,
    catchment_poly: Polygon,
    flow_dir_path: Path,
    flow_acc_path: Path,
    skip_snap: bool = False,
) -> Optional[Polygon]:
    """
    Performs detailed pixel-scale raster-based delineation using pysheds.

    Since gauges are already snapped to MERIT polylines, we skip the snapping step
    and use the exact gauge location for delineation.

    Args:
        gauge_id: Gauge identifier
        basin: Pfafstetter basin code (tells us which raster files to open)
        lat: Latitude of gauge (already snapped to river)
        lng: Longitude of gauge (already snapped to river)
        catchment_poly: Shapely polygon of the catchment to split
        flow_dir_path: Path to flow direction rasters

    Returns:
        gauge_polygon: catchment for the gauge, split at gauge point if needed.

    """
    from pysheds.grid import Grid

    # Get bounding box and adjust to pixel boundaries
    bounds = catchment_poly.bounds
    bounds_list = [float(i) for i in bounds]

    # MERIT-Hydro has 3 arcsecond resolution (1/1200 degree)
    halfpix = MERIT_RES / 2  # Half pixel width

    # Adjust bounds to pixel centers
    bounds_list[0] = floor(bounds_list[0] * 1200) / 1200 - halfpix
    bounds_list[1] = floor(bounds_list[1] * 1200) / 1200 - halfpix
    bounds_list[2] = ceil(bounds_list[2] * 1200) / 1200 + halfpix
    bounds_list[3] = ceil(bounds_list[3] * 1200) / 1200 + halfpix
    bounding_box = tuple(bounds_list)

    # Load flow direction raster (windowed reading)
    flow_file = resolve_raster(flow_dir_path, basin)
    grid = Grid.from_raster(flow_file, window=bounding_box, nodata=0)
    fdir = grid.read_raster(flow_file, window=bounding_box, nodata=0)

    facc_file = resolve_raster(flow_acc_path, basin)
    facc = grid.read_raster(facc_file, window=bounding_box, nodata=0)
    mymask = grid.rasterize([catchment_poly])

    # Mask land outside catchment
    fdir[mymask == 0] = 0
    facc[mymask == 0] = 0

    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)  # ESRI flow direction standard

    # Get the minimum flow (in pixels) of the mainstem of this polygon.
    min_area_m2 = gauge["min_area_km2"] * 1e6 * 0.9
    pixel_area_m2 = get_pixel_area_m2(gauge, grid)
    facc_thresh = max(min_area_m2 / pixel_area_m2, 100)

    if skip_snap:
        x_snap, y_snap = gauge["lng"], gauge["lat"]
    else:
        if facc.max() < facc_thresh:
            print(f"flow accumulation threshold ({facc_thresh}) was greater than maxmimum flow accumulation in catchment ({facc.max()}) for gauge id {gauge['id']}")
            return None
        x_snap, y_snap = grid.snap_to_mask(facc > facc_thresh, (gauge["lng"], gauge["lat"]))

    # Validate that the pour point falls on a valid flow direction cell.
    # If the point is outside the catchment mask, fdir will be 0 and
    # grid.catchment() will segfault (crash the kernel, not a Python exception).
    # Use pysheds' own coordinate->index conversion, with the same snap mode
    # grid.catchment() defaults to, so this guard tests exactly the cell the
    # delineation will use. Deriving the index by hand disagreed at cell
    # boundaries: snap_to_mask returns a coordinate lying exactly on a boundary,
    # and (x - origin)/res then lands a floating-point hair below the integer
    # (26.999999999992 for 27), so floor picked the neighbouring cell -- normally
    # the one outside the catchment mask -- and a valid gauge was discarded.
    col_snap, row_snap = grid.nearest_cell(x_snap, y_snap, snap="corner")
    if (row_snap < 0 or row_snap >= fdir.shape[0] or
            col_snap < 0 or col_snap >= fdir.shape[1] or
            fdir[row_snap, col_snap] == 0):
        print(f"Warning: Pour point for {gauge['id']} at ({x_snap}, {y_snap}) "
              f"falls on an invalid flow direction cell. Skipping split.")
        return None

    # Delineate watershed using pysheds
    try:
        catch = grid.catchment(
            fdir=fdir,
            x=x_snap,
            y=y_snap,
            dirmap=dirmap,
            xytype="coordinate",
            recursionlimit=15000,
        )
        grid.clip_to(catch)
        clipped_catch = grid.view(catch, dtype=np.uint8)
    except Exception as e:
        print(f"Error during catchment delineation for gauge {gauge['id']}: {e}")
        return None

    # Convert raster to polygon
    shapes = grid.polygonize(clipped_catch)

    # Collect all polygons from pysheds
    shapely_polygons = []
    for shape, value in shapes:
        poly_coords = [[p[0], p[1]] for p in shape["coordinates"][0]]
        shapely_polygon = Polygon(poly_coords)
        shapely_polygons.append(shapely_polygon)

    # Dissolve multiple polygons if necessary
    if len(shapely_polygons) > 1:
        result_polygon = unary_union(shapely_polygons)
        result_polygon = _get_largest_polygon(result_polygon)
    elif len(shapely_polygons) == 1:
        result_polygon = shapely_polygons[0]
    else:
        print(f"Warning: No polygon generated for gauge {gauge['id']}")
        return None

    # By definition, pysheds.catchment finds the area UPSTREAM of the pour point.
    upstream_poly = result_polygon

    # The remainder is the downstream portion.
    downstream_poly = catchment_poly.difference(upstream_poly)

    # Check for slivers.
    # Only skip split if DOWNSTREAM portion is tiny (gauge at bottom of catchment).
    downstream_pixels = downstream_poly.area / PIXEL_AREA
    if downstream_pixels < MIN_PIXELS:
        # Gauge is at the very bottom - it gets the entire catchment.
        return catchment_poly

    # Return the delineated upstream polygon.
    return upstream_poly
