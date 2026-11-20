"""Compute the MERIT-Hydro upstream area at the outlet of every node of graph v3.5 (a static attribute of the model).

Definition: the MERIT upstream area (upa, km2) at the node's routed outlet, that is the end of its river reach, the
lake's pour point, or the gauge's location on the channel. It is not the largest upa found anywhere in the node's
polygon, and it does not depend on the graph's own routing, so it is an independent measure of how much land drains
through the node.

Reads (W = $SWOTGAT_WORK, E = W/CONUS_storage_fork/eval; inputs are only read)
  E/graph_v3/v35/subbasins_conus_v35.parquet    sub-basin table of graph v3.5
  E/graph_v3/gauges_v34/<basin>.csv             gauge tables used by the graph v3.4 delineation
  W/LakeSPvC_ts_for_Ted/SWOT_PLD_v106_reservoirs.gdb   lake outlines of the SWOT Prior Lake Database (PLD)
  W/CONUS/merit_rasters/dir, upg                MERIT flow direction and flow accumulation (upg, cell count) mosaics
  E/graph_review/_merit_upa_conus.vrt           MERIT upstream area (upa, km2) mosaic
  $SWOTGAT_NAS_ICE/data/MERIT_Basins/MERIT_Hydro_v07_Basins_v01_bugfix/pfaf_level_01/   MERIT-Basins river reaches
                                                (riv_pfaf_7) and unit catchments (cat_pfaf_7)
  $SWOTGAT_NAS_ICE/data/MERIT-Hydro/v1.0.1/upa_n00w090/n20w090_upa.tif   one raw tile, for the georeferencing check

Writes (in E/v35_inputs/statics/work)
  merit_uparea_v35_nodes.csv         one row per node: merit_uparea_km2, source (= node kind:status), the outlet cell
                                     (px, py, facc), check columns, and outlet_id, node_type, uparea_km2 from the graph.
                                     The script refuses to overwrite it.
  merit_upa_conus_tilegeoref.vrt     the upa mosaic with corrected georeferencing (see Raster frames), if not present

Usage
  python3.12 merit_uparea_v35.py [--workers 16]        (about 8 minutes on 16 CPUs)

Method, by node kind
  reservoir  upa at the lake's pour point: the cell with the largest flow accumulation (upg) inside the lake's PLD
             outline (lake_id = node id), found with the delineator's own find_reservoir_outlet (copied below).
  gauge      upa at the gauge's snapped pour point in the graph v3.4 delineation (graph v3.5 moves no gauge). The
             delineation output does not record the snapped point (its node tables hold the input location of the
             gauge), so the point is reproduced by replaying the delineator's code: the gauge tables go through
             set_gauge_area_range, then through split_catchment_raster in the same (COMID, position) order and with
             the same replace/split bookkeeping as the delineator's insert_all_gauges. The delineator inserts gauges
             first, on unit catchments, so the replay is exact. If the snap fails with snap_min_area_km2, it is
             retried with the original minimum area (status split_retry), as the delineator does.
  river      (every other node) the MERIT-Basins reach attribute uparea of the node's original_comid, which is upa at
             the reach outlet, if the downstream end of that reach lies within 1.5 cells of the node polygon (status
             reach_end). Otherwise the fallback: upa at the cell with the largest upg inside the node polygon (status
             max_upg_in_polygon).

Raster frames
  The delineator's dir and upg mosaics and the MERIT-Basins vectors use the MERIT tiles' own georeferencing (cell
  centres on multiples of 1/1200 degree). The upa mosaic _merit_upa_conus.vrt places the same tiles half a cell to
  the north-east of that (its origin is -130, 55), so reading it by coordinate is off by half a cell. This script
  reads it through a copy that has the tiles' own origin (same sources, same cell indices) and checks that copy
  against a raw MERIT tile at four points.

Checks on every outlet cell (reported in the log and kept as columns)
  upg_check              upg read back at the cell; it should equal facc, the value the delineator code saw
  upa_per_upg_vs_cell    upa / upg divided by the area of one cell at that latitude; it should be close to 1

merit_override_v35.py then replaces a few of these values (see there).
"""
from swotgat.paths import NAS_ICE, WORK
import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
import shapely
from numpy import ceil, floor
from pyproj import Geod
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

R = Path(f"{WORK}")
F = R/"CONUS_storage_fork/eval"
V3 = F/"graph_v3"
S = F/"v35_inputs/statics"
WORK = S/"work"
SUBS = V3/"v35/subbasins_conus_v35.parquet"
GAUGES = V3/"gauges_v34"
PLD = R/"LakeSPvC_ts_for_Ted/SWOT_PLD_v106_reservoirs.gdb"
MB1 = Path(f"{NAS_ICE}/data/MERIT_Basins/MERIT_Hydro_v07_Basins_v01_bugfix/pfaf_level_01")
FLOW_DIR = R/"CONUS/merit_rasters/dir"
FLOW_ACC = R/"CONUS/merit_rasters/upg"
UPA_SRC = F/"graph_review/_merit_upa_conus.vrt"
UPA = WORK/"merit_upa_conus_tilegeoref.vrt"
RAW_TILE = Path(f"{NAS_ICE}/data/MERIT-Hydro/v1.0.1/upa_n00w090/n20w090_upa.tif")
OUT = WORK/"merit_uparea_v35_nodes.csv"
REACH_TOL = 1.5 / 1200
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

# ---------------- verbatim copies from graph_v3/src_v34/graph_delineator (split_catchment_reservoirs.py, delineate_reservoirs_working.py)
MIN_PIXELS = 5
MERIT_RES = 0.000833333
PIXEL_AREA = MERIT_RES * MERIT_RES


def resolve_raster(base_dir, basin) -> str:
    """Return the path of the flow raster of one region: <base_dir>/<basin>.vrt if it exists, else the .tif.

    Raise FileNotFoundError if neither file exists.
    """
    base_dir = Path(base_dir)
    for ext in (".vrt", ".tif"):
        path = base_dir / f"{basin}{ext}"
        if path.is_file():
            return str(path)
    raise FileNotFoundError(f"No flow raster for basin {basin} in {base_dir}")


def _get_largest_polygon(geom):
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda p: p.area)
    return geom


def get_pixel_area_m2(gauge, grid):
    """Return the area (m2) of one raster cell at the gauge's location, on the WGS84 ellipsoid.

    gauge gives lng and lat; grid is the pysheds grid, whose affine transform gives the cell size.
    """
    g = Geod(ellps="WGS84")
    lon, lat = gauge["lng"], gauge["lat"]
    x0, y0 = lon, lat
    x1 = lon + grid.affine[0]
    y1 = lat + grid.affine[4]  # negative
    area, _ = g.polygon_area_perimeter([x0, x1, x1, x0], [y0, y0, y1, y1])
    return abs(area)


def split_catchment_raster(gauge, basin, catchment_poly, flow_dir_path, flow_acc_path, skip_snap=False):
    """Split a unit catchment at a gauge and return (polygon, pour).

    Copy of the delineator's split_catchment_reservoirs.split_catchment_raster. Unchanged except that it returns
    (polygon, pour) instead of the polygon alone, and that its prints are removed.

    gauge is a record with lng, lat and min_area_km2; basin is the two-digit Pfafstetter region that names the
    rasters; catchment_poly is the unit catchment. Unless skip_snap is set, the gauge is snapped to the nearest
    cell whose flow accumulation exceeds the larger of 100 cells and 0.9 times min_area_km2 (in cells).

    polygon is the part of the catchment upstream of the snapped cell, or the whole catchment if the part
    downstream would be smaller than 5 cells; it is None if the split fails. pour is the pour-point cell that
    grid.catchment() uses: (x of cell centre, y of cell centre, facc value there); it is None if no valid cell
    is found.
    """
    from pysheds.grid import Grid
    bounds = catchment_poly.bounds
    bounds_list = [float(i) for i in bounds]
    halfpix = MERIT_RES / 2
    bounds_list[0] = floor(bounds_list[0] * 1200) / 1200 - halfpix
    bounds_list[1] = floor(bounds_list[1] * 1200) / 1200 - halfpix
    bounds_list[2] = ceil(bounds_list[2] * 1200) / 1200 + halfpix
    bounds_list[3] = ceil(bounds_list[3] * 1200) / 1200 + halfpix
    bounding_box = tuple(bounds_list)
    flow_file = resolve_raster(flow_dir_path, basin)
    grid = Grid.from_raster(flow_file, window=bounding_box, nodata=0)
    fdir = grid.read_raster(flow_file, window=bounding_box, nodata=0)
    facc_file = resolve_raster(flow_acc_path, basin)
    facc = grid.read_raster(facc_file, window=bounding_box, nodata=0)
    mymask = grid.rasterize([catchment_poly])
    fdir[mymask == 0] = 0
    facc[mymask == 0] = 0
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    min_area_m2 = gauge["min_area_km2"] * 1e6 * 0.9
    pixel_area_m2 = get_pixel_area_m2(gauge, grid)
    facc_thresh = max(min_area_m2 / pixel_area_m2, 100)
    if skip_snap:
        x_snap, y_snap = gauge["lng"], gauge["lat"]
    else:
        if facc.max() < facc_thresh:
            return None, None
        x_snap, y_snap = grid.snap_to_mask(facc > facc_thresh, (gauge["lng"], gauge["lat"]))
    col_snap, row_snap = grid.nearest_cell(x_snap, y_snap, snap="corner")
    if (row_snap < 0 or row_snap >= fdir.shape[0] or col_snap < 0 or col_snap >= fdir.shape[1] or fdir[row_snap, col_snap] == 0):
        return None, None
    xc, yc = grid.affine * (col_snap + 0.5, row_snap + 0.5)                               # (added) pour-point cell centre
    pour = (float(xc), float(yc), float(facc[row_snap, col_snap]))
    try:
        catch = grid.catchment(fdir=fdir, x=x_snap, y=y_snap, dirmap=dirmap, xytype="coordinate", recursionlimit=15000)
        grid.clip_to(catch)
        clipped_catch = grid.view(catch, dtype=np.uint8)
    except Exception:
        return None, pour
    shapes = grid.polygonize(clipped_catch)
    shapely_polygons = []
    for shape, value in shapes:
        poly_coords = [[p[0], p[1]] for p in shape["coordinates"][0]]
        shapely_polygons.append(Polygon(poly_coords))
    if len(shapely_polygons) > 1:
        result_polygon = _get_largest_polygon(unary_union(shapely_polygons))
    elif len(shapely_polygons) == 1:
        result_polygon = shapely_polygons[0]
    else:
        return None, pour
    upstream_poly = result_polygon
    downstream_poly = catchment_poly.difference(upstream_poly)
    if downstream_poly.area / PIXEL_AREA < MIN_PIXELS:
        return catchment_poly, pour
    return upstream_poly, pour


def find_reservoir_outlet(res_poly, flow_dir_path, flow_acc_path, pfaf2_id):
    """Return the outlet of a reservoir polygon: the cell with the largest flow accumulation inside it.

    Copy of the delineator's delineate_reservoirs_working.find_reservoir_outlet. Unchanged except that it also
    returns the facc value at the outlet. The result is ((lng, lat) of the cell centre, facc value).
    """
    from pysheds.grid import Grid
    bounds = res_poly.bounds
    halfpix = MERIT_RES / 2
    bounding_box = (floor(bounds[0] * 1200) / 1200 - halfpix, floor(bounds[1] * 1200) / 1200 - halfpix,
                    ceil(bounds[2] * 1200) / 1200 + halfpix, ceil(bounds[3] * 1200) / 1200 + halfpix)
    facc_file = resolve_raster(flow_acc_path, pfaf2_id)
    grid = Grid.from_raster(facc_file, window=bounding_box, nodata=0)
    facc = grid.read_raster(facc_file, window=bounding_box, nodata=0)
    mask = grid.rasterize([res_poly])
    facc[mask == 0] = 0
    max_idx = np.unravel_index(facc.argmax(), facc.shape)
    outlet_lng = grid.affine[2] + (max_idx[1] + 0.5) * grid.affine[0]
    outlet_lat = grid.affine[5] + (max_idx[0] + 0.5) * grid.affine[4]
    return (outlet_lng, outlet_lat), float(facc[max_idx])


def set_gauge_area_range(gauges, rivers):
    """Return a copy of the gauge table with the upstream-area range of each gauge's reach.

    rivers is the MERIT-Basins reach table indexed by COMID. Columns added (km2): min_area_orig_km2, the summed
    uparea of the reaches that flow into the gauge's reach; min_area_km2, the same value raised to the gauge's
    snap_min_area_km2 where one is given; max_area_km2, the uparea of the gauge's own reach.
    """
    gauges = gauges.copy()
    gauges["min_area_km2"] = 0.0
    gauges["max_area_km2"] = 0.0
    for i, g in gauges.iterrows():
        min_area = rivers.loc[rivers["NextDownID"] == g["COMID"], "uparea"].sum()
        gauges.loc[i, "min_area_orig_km2"] = min_area
        if "snap_min_area_km2" in gauges.columns and pd.notna(g.get("snap_min_area_km2")):
            min_area = max(min_area, float(g["snap_min_area_km2"]))
        gauges.loc[i, "min_area_km2"] = min_area
        gauges.loc[i, "max_area_km2"] = rivers.loc[g["COMID"], "uparea"]
    return gauges
# ---------------- end of copies


def gauge_group(args):
    """Replay the delineator's gauge insertion for all gauges of one MERIT unit catchment.

    args is (COMID, gauge records, catchment polygon as WKB, pfaf2 region). The gauges are handled in the order
    of insert_all_gauges (by position, upstream first) and with its polygon bookkeeping: a gauge either splits
    the current polygon or replaces it. A split that fails with the raised minimum area is retried with the
    original one. Return one record per gauge with the pour-point cell (px, py, facc) and a status (split,
    replace, split_retry, replace_retry or failed).
    """
    comid, rows, cat_wkb, pfaf2 = args
    polys = {str(comid): shapely.from_wkb(cat_wkb)}; repl = {}; out = []
    for g in rows:
        cur = str(comid)
        while cur in repl:
            cur = repl[cur]
        cp = polys[cur]
        poly, pour = split_catchment_raster(g, pfaf2, cp, FLOW_DIR, FLOW_ACC); how = "split"
        if poly is None and g.get("min_area_orig_km2") is not None and g.get("min_area_km2") != g.get("min_area_orig_km2"):
            g2 = dict(g); g2["min_area_km2"] = g2["min_area_orig_km2"]
            poly, pour = split_catchment_raster(g2, pfaf2, cp, FLOW_DIR, FLOW_ACC); how = "split_retry"
        if poly is None:
            out.append(dict(node=g["id"], status="failed")); continue
        if poly == cp:
            repl[cur] = g["id"]; polys[g["id"]] = cp; how = how.replace("split", "replace")
        else:
            polys[g["id"]] = poly
            polys[cur] = cp.difference(poly).buffer(-MERIT_RES / 2).buffer(MERIT_RES / 2)
        out.append(dict(node=g["id"], status=how, px=pour[0], py=pour[1], facc=pour[2], comid=int(comid)))
    return out


def res_one(args):
    """Return the pour point of one lake node as a record with px, py and facc.

    args is (node id, PLD outline as WKB, pfaf2 region).
    """
    node, wkb, pfaf2 = args
    (x, y), fa = find_reservoir_outlet(shapely.from_wkb(wkb), FLOW_DIR, FLOW_ACC, pfaf2)
    return dict(node=node, status="pour_point", px=float(x), py=float(y), facc=fa)


def maxcell_one(args):
    """Return the fallback outlet cell of one river node as a record with px, py and facc.

    The cell is the one with the largest upg among the cells whose centre lies inside the node polygon (or that
    touch it, if no centre does), located in the upg mosaic's own frame. args is (node id, node polygon as WKB,
    pfaf2 region). The status is no_cell if no such cell has a positive upg.
    """
    node, wkb, pfaf2 = args
    from rasterio.features import geometry_mask
    from rasterio.windows import from_bounds
    g = shapely.from_wkb(wkb)
    with rasterio.open(resolve_raster(FLOW_ACC, pfaf2)) as ds:
        b = g.bounds
        w = from_bounds(b[0] - 2 / 1200, b[1] - 2 / 1200, b[2] + 2 / 1200, b[3] + 2 / 1200, transform=ds.transform).round_offsets().round_lengths()
        a = ds.read(1, window=w, boundless=True, fill_value=-9999).astype(float); t = ds.window_transform(w)
    m = geometry_mask([g], out_shape=a.shape, transform=t, invert=True)
    if not m.any():
        m = geometry_mask([g], out_shape=a.shape, transform=t, invert=True, all_touched=True)
    v = np.where(m & (a > 0), a, -1); i = np.unravel_index(int(np.argmax(v)), v.shape)
    if v[i] <= 0:
        return dict(node=node, status="no_cell")
    x, y = t * (i[1] + 0.5, i[0] + 0.5)
    return dict(node=node, status="max_upg_in_polygon", px=float(x), py=float(y), facc=float(v[i]))


def main():
    """Find the outlet cell of every node, read the MERIT upstream area there, and write the node table."""
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=16); A = ap.parse_args()
    assert not OUT.exists(), f"{OUT} exists; refusing to overwrite"
    # Write the upa mosaic with the tiles' own origin (half a cell south-west of the source VRT's origin), then check
    # it against a raw tile at four points.
    if not UPA.exists():
        t = UPA_SRC.read_text()
        old = "<GeoTransform>-130, 0.0008333333333333334, 0, 55, 0, -0.0008333333333333334</GeoTransform>"
        assert t.count(old) == 1
        UPA.write_text(t.replace(old, "<GeoTransform>-130.00041666666667, 0.0008333333333333334, 0, 54.999583333333334, 0, "
                                      "-0.0008333333333333334</GeoTransform>"))
    with rasterio.open(UPA) as a, rasterio.open(RAW_TILE) as tile:
        pts = [(-89.5, 24.5), (-85.123, 21.987), (-88.0004, 23.0001), (-87.25, 20.25)]
        va, vt = [float(v[0]) for v in a.sample(pts)], [float(v[0]) for v in tile.sample(pts)]
        assert va == vt, ("corrected upa VRT does not match the raw tile", va, vt)
        print("check", va, vt)
    log("upa VRT georeferencing check vs raw tile n20w090: ok")

    s = gpd.read_parquet(SUBS); s.index = s.index.astype(str)
    log(f"{len(s):,} subbasin rows, {s.outlet_id.nunique()} basins")
    basin_pfaf2, tabs = {}, {}
    for f in sorted(GAUGES.glob("*.csv")):
        t = pd.read_csv(f); tabs[f.stem] = t
        o = t[t["id"] == f.stem]
        if len(o):
            basin_pfaf2[f.stem] = int(o["COMID"].item() // 1e6)
    log(f"{len(tabs)} gauge tables; pfaf2 for {len(basin_pfaf2)} basins: {sorted(set(basin_pfaf2.values()))}")
    assert set(s.outlet_id) <= set(basin_pfaf2), set(s.outlet_id) - set(basin_pfaf2)

    riv = pyogrio.read_dataframe(MB1/"riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp", columns=["COMID", "NextDownID", "uparea"])
    riv = riv.set_index("COMID"); log(f"{len(riv):,} MERIT-Basins reaches")
    rows = []

    # ---- reservoirs
    pld = gpd.read_file(PLD); idc = [c for c in pld.columns if c.lower() in ("lake_id", "lakeid")][0]
    log("PLD crs", pld.crs.to_epsg(), len(pld))
    assert pld.crs.to_epsg() == 4326, "the delineator uses the PLD geometry as read; expected EPSG:4326"
    pld[idc] = pld[idc].astype("int64").astype(str); pld = pld.set_index(idc)
    res = s.index[s.is_reservoir]
    assert res.isin(pld.index).all() and not pld.index.duplicated().any()
    jobs = [(n, shapely.to_wkb(pld.geometry[n]), basin_pfaf2[s.outlet_id[n]]) for n in res]
    with Pool(A.workers) as p:
        rr = p.map(res_one, jobs, chunksize=8)
    rows += [dict(r, node_kind="reservoir") for r in rr]; log(f"reservoirs: {len(rr)} pour points")

    # ---- gauges (delineator order and bookkeeping, per MERIT catchment; every table gauge on those catchments is replayed)
    gn = set(s.index[s.is_gauge])
    gl = [t.assign(_basin=b) for b, t in tabs.items()]
    Gall = pd.concat(gl, ignore_index=True)
    Gall["id"] = Gall["id"].astype(str)
    assert (Gall.outlet_id == Gall._basin).all()
    found = gn & set(Gall["id"])
    log(f"{len(found):,} of {len(gn):,} gauge nodes found in the gauge tables")
    comids = Gall.loc[Gall["id"].isin(gn), "COMID"].unique()
    Gall = set_gauge_area_range(Gall[Gall.COMID.isin(comids)], riv)
    Gall["node_id"] = Gall["COMID"].astype(str)
    Gall = Gall.sort_values(["node_id", "position"])
    ids = ",".join(str(int(c)) for c in comids)
    cat = pyogrio.read_dataframe(MB1/"cat_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp", columns=["COMID"], where=f"COMID IN ({ids})").set_index("COMID")
    log(f"{len(cat):,} unit catchments read for {len(comids):,} gauge COMIDs")
    gjobs = []
    for (b, c), grp in Gall.groupby(["_basin", "COMID"], sort=False):
        recs = grp.drop(columns=["_basin"]).to_dict("records")
        recs = [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()} for r in recs]
        gjobs.append((int(c), recs, shapely.to_wkb(cat.geometry[int(c)]), basin_pfaf2[b]))
    gjobs.sort(key=lambda j: -len(j[1]))
    with Pool(A.workers) as p:
        gr = [x for lst in p.imap_unordered(gauge_group, gjobs, chunksize=2) for x in lst]
    gr = [dict(r, node_kind="gauge") for r in gr if r["node"] in gn]
    rows += gr; log(f"gauges: {len(gr)} processed, status {pd.Series([r['status'] for r in gr]).value_counts().to_dict()}")

    # ---- river nodes: reach end of original_comid
    rv = s[~s.is_gauge & ~s.is_reservoir]
    oc = rv.original_comid.astype("int64")
    first = {}
    for c in oc[oc.isin(riv.index)].unique():
        gg = riv.geometry[c]
        first[c] = Point((gg if gg.geom_type == "LineString" else gg.geoms[0]).coords[0])
    fb = []
    for n in rv.index:
        c = int(oc[n])
        if c in first and shapely.distance(rv.geometry[n], first[c]) <= REACH_TOL:
            rows.append(dict(node=n, node_kind="river", status="reach_end", reach_uparea=float(riv.uparea[c]), comid=c))
        else:
            fb.append(n)
    log(f"river nodes: {len(rv) - len(fb):,} at their reach end, {len(fb)} fallback")
    with Pool(A.workers) as p:
        fr = p.map(maxcell_one, [(n, shapely.to_wkb(rv.geometry[n]), int(oc[n] // 1000000)) for n in fb])
    for r in fr:
        c = int(oc[r["node"]])
        r.update(node_kind="river", comid=c, reach_uparea=float(riv.uparea[c]) if c in riv.index else np.nan,
                 reach_end_dist_cells=float(shapely.distance(rv.geometry[r["node"]], first[c]) * 1200) if c in first else np.nan)
    rows += fr

    # Read upa (and upg, as a check) at every outlet cell. Nodes at their reach end take the reach's uparea instead.
    D = pd.DataFrame(rows).set_index("node")
    assert not D.index.duplicated().any()
    D["pfaf2"] = [basin_pfaf2[s.outlet_id[n]] if D.node_kind[n] != "river" else int(s.original_comid[n] // 1000000) for n in D.index]
    pp = D.px.notna()
    with rasterio.open(UPA) as a:
        D.loc[pp, "upa_cell"] = [float(v[0]) for v in a.sample(list(zip(D.px[pp], D.py[pp])))]
    D["upg_check"] = np.nan
    for pf in sorted(D.pfaf2.unique()):
        k = pp & (D.pfaf2 == pf)
        if k.any():
            with rasterio.open(resolve_raster(FLOW_ACC, pf)) as u:
                D.loc[k, "upg_check"] = [float(v[0]) for v in u.sample(list(zip(D.px[k], D.py[k])))]
    cell_km2 = (6371.0072 * np.pi / 180 / 1200) ** 2 * np.cos(np.radians(D.py))
    D["upa_per_upg_vs_cell"] = D.upa_cell / D.upg_check / cell_km2
    D["merit_uparea_km2"] = np.where(D.status == "reach_end", D.reach_uparea, D.upa_cell)
    D["source"] = D.node_kind + ":" + D.status
    D = D.join(s[["outlet_id", "node_type", "uparea_km2"]])
    D.index.name = "node"
    D.to_csv(OUT)
    log(f"wrote {OUT}: {len(D):,} nodes; sources {D.source.value_counts().to_dict()}")
    bad = pp & (D.upg_check != D.facc)
    log(f"pour points whose upg read back differs from the delineator facc value: {int(bad.sum())}")
    log(f"upa/upg / cell area: {D.upa_per_upg_vs_cell.describe(percentiles=[.001, .01, .5, .99, .999]).round(4).to_dict()}")


if __name__ == "__main__":
    main()
