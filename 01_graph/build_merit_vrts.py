"""Mosaic the MERIT-Hydro flow-direction and flow-accumulation tiles, one mosaic per Pfafstetter level-2 region.

The graph delineator splits catchments on rasters and reads one mosaic per
two-digit Pfafstetter region (pfaf2):

    <flow_dir_path>/<pfaf2>.vrt      # pfaf2 = int(COMID // 1e6)
    <flow_acc_path>/<pfaf2>.vrt

MERIT-Hydro is distributed as 5x5-degree tiles. This script writes each mosaic
as a GDAL virtual raster (VRT): a small XML index that GDAL stitches on read.
A VRT costs kilobytes instead of a second copy of the tiles, and it supports
the windowed reads that the delineator (pysheds) makes.

Reads
  --raw      MERIT-Hydro v1.0.1 tiles, <var>_<30-degree group>/<tile>_<var>.tif
             (default $SWOTGAT_NAS_ICE/data/MERIT-Hydro/v1.0.1)
  --cat-dir  MERIT-Basins level-2 catchment shapefiles; only their bounding
             boxes are used, to choose the tiles of each region (default
             $SWOTGAT_NAS_ICE/data/MERIT_Basins/MERIT_Hydro_v07_Basins_v01_bugfix/pfaf_level_02)

Writes
  <out>/<var>/<pfaf2>.vrt   for each variable in --vars (default dir = flow
             direction, upg = flow accumulation as upstream cell count) under
             --out (default $SWOTGAT_WORK/CONUS/merit_rasters).
             An existing VRT is kept unless --overwrite is given.

Notes
  The VRT XML is written directly, so the script needs rasterio only (neither
  the gdalbuildvrt command nor the osgeo Python bindings).
  The VRTs store absolute tile paths, so rebuild them if the tiles move.
  The script stops if the tiles of a region differ in data type or do not lie
  on one 3-arcsecond grid.

Usage
-----
  python3.12 build_merit_vrts.py --pfaf2 73 74 75 77 78   # the regions of the paper's basins
  python3.12 build_merit_vrts.py --from-subbasins <subbasins.parquet>   # infer the regions from node ids
"""
from swotgat.paths import NAS_ICE, WORK
import argparse
import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import rasterio

MERIT_RAW = Path(f"{NAS_ICE}/data/MERIT-Hydro/v1.0.1")
CAT_L2 = Path(f"{NAS_ICE}/data/MERIT_Basins/MERIT_Hydro_v07_Basins_v01_bugfix/pfaf_level_02")
OUT = Path(f"{WORK}/CONUS/merit_rasters")

RES = 1.0 / 1200.0  # MERIT-Hydro 3 arcsec
TILE_DEG = 5
GROUP_DEG = 30

GDAL_DTYPE = {"uint8": "Byte", "int8": "Int8", "uint16": "UInt16", "int16": "Int16",
              "uint32": "UInt32", "int32": "Int32", "float32": "Float32", "float64": "Float64"}


def parse_args():
    """Parse the command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pfaf2", nargs="+", type=int, help="Pfafstetter level-2 basin ids, e.g. 73 78")
    g.add_argument("--from-subbasins", type=Path,
                   help="Infer needed pfaf2 ids from a subbasins parquet (numeric COMID index).")
    p.add_argument("--vars", nargs="+", default=["dir", "upg"])
    p.add_argument("--raw", type=Path, default=MERIT_RAW)
    p.add_argument("--cat-dir", type=Path, default=CAT_L2)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def tile_name(lat, lon):
    """Return the MERIT-Hydro name of the tile whose south-west corner is (lat, lon), e.g. n35w100.

    lat and lon are whole degrees. Tiles are 5 degrees wide and are stored in folders of 30-degree groups
    named <var>_<south-west corner of the group>.
    """
    ns, ew = ("n" if lat >= 0 else "s"), ("e" if lon >= 0 else "w")
    return f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"


def tile_path(raw, var, lat, lon):
    """Return the path, under folder raw, of the tile of variable var whose south-west corner is (lat, lon)."""
    glat = int(math.floor(lat / GROUP_DEG) * GROUP_DEG)
    glon = int(math.floor(lon / GROUP_DEG) * GROUP_DEG)
    group = f"{var}_{tile_name(glat, glon)}"
    return raw / group / f"{tile_name(lat, lon)}_{var}.tif"


def tiles_covering(raw, var, bounds):
    """List the existing tiles of one variable that cover a bounding box.

    bounds is (min_lon, min_lat, max_lon, max_lat). Tiles over open ocean do not exist and are skipped.
    """
    minx, miny, maxx, maxy = bounds
    out = []
    lat0 = int(math.floor(miny / TILE_DEG) * TILE_DEG)
    lat1 = int(math.ceil(maxy / TILE_DEG) * TILE_DEG)
    lon0 = int(math.floor(minx / TILE_DEG) * TILE_DEG)
    lon1 = int(math.ceil(maxx / TILE_DEG) * TILE_DEG)
    for lat in range(lat0, lat1, TILE_DEG):
        for lon in range(lon0, lon1, TILE_DEG):
            p = tile_path(raw, var, lat, lon)
            if p.is_file():
                out.append(p)
    return out


def write_vrt(out_path, tiles):
    """Write a VRT mosaic of the given tiles to out_path.

    The tiles must share one data type and lie on one 3-arcsecond grid, so that the pixel offsets are exact;
    otherwise the script exits. Return (width, height, number of tiles), the size in pixels.
    """
    infos = []
    for t in tiles:
        with rasterio.open(t) as s:
            infos.append({"path": t, "w": s.width, "h": s.height, "b": s.bounds,
                          "dtype": s.dtypes[0], "nodata": s.nodata,
                          "blockx": s.block_shapes[0][1], "blocky": s.block_shapes[0][0]})

    dtypes = {i["dtype"] for i in infos}
    if len(dtypes) != 1:
        sys.exit(f"mixed dtypes across tiles: {dtypes}")
    dtype = dtypes.pop()
    nodata = infos[0]["nodata"]

    minx = min(i["b"].left for i in infos)
    maxy = max(i["b"].top for i in infos)
    maxx = max(i["b"].right for i in infos)
    miny = min(i["b"].bottom for i in infos)
    width = round((maxx - minx) / RES)
    height = round((maxy - miny) / RES)

    lines = [
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">',
        '  <SRS>EPSG:4326</SRS>',
        f'  <GeoTransform>{minx:.17g}, {RES:.17g}, 0, {maxy:.17g}, 0, {-RES:.17g}</GeoTransform>',
        f'  <VRTRasterBand dataType="{GDAL_DTYPE[dtype]}" band="1">',
    ]
    if nodata is not None:
        lines.append(f'    <NoDataValue>{nodata:.17g}</NoDataValue>')

    for i in infos:
        xoff = (i["b"].left - minx) / RES
        yoff = (maxy - i["b"].top) / RES
        # Exactness matters: a fractional offset means the tiles are not on one
        # grid and the mosaic would be silently misregistered.
        if abs(xoff - round(xoff)) > 1e-6 or abs(yoff - round(yoff)) > 1e-6:
            sys.exit(f"tile {i['path'].name} is off-grid (xoff={xoff}, yoff={yoff})")
        lines += [
            '    <ComplexSource>',
            f'      <SourceFilename relativeToVRT="0">{escape(str(i["path"]))}</SourceFilename>',
            '      <SourceBand>1</SourceBand>',
            f'      <SourceProperties RasterXSize="{i["w"]}" RasterYSize="{i["h"]}" '
            f'DataType="{GDAL_DTYPE[dtype]}" BlockXSize="{i["blockx"]}" BlockYSize="{i["blocky"]}"/>',
            f'      <SrcRect xOff="0" yOff="0" xSize="{i["w"]}" ySize="{i["h"]}"/>',
            f'      <DstRect xOff="{round(xoff)}" yOff="{round(yoff)}" xSize="{i["w"]}" ySize="{i["h"]}"/>',
        ]
        if nodata is not None:
            lines.append(f'      <NODATA>{nodata:.17g}</NODATA>')
        lines.append('    </ComplexSource>')

    lines += ['  </VRTRasterBand>', '</VRTDataset>', '']
    out_path.write_text("\n".join(lines))
    return width, height, len(infos)


def pfaf2_bounds(cat_dir, pfaf2):
    """Return the bounding box of one level-2 region, from its MERIT-Basins catchment shapefile.

    Return None if the shapefile is missing.
    """
    import pyogrio
    shp = cat_dir / f"cat_pfaf_{pfaf2}_MERIT_Hydro_v07_Basins_v01_bugfix1.shp"
    if not shp.is_file():
        return None
    return pyogrio.read_info(shp)["total_bounds"]


def main():
    """Write one VRT per requested region and variable."""
    args = parse_args()

    if args.from_subbasins:
        import geopandas as gpd
        subs = gpd.read_parquet(args.from_subbasins)
        ids = {int(str(i)[:2]) for i in subs.index if str(i).isdigit()}
        pfaf2s = sorted(ids)
        print(f"inferred pfaf2 basins from {args.from_subbasins.name}: {pfaf2s}")
    else:
        pfaf2s = sorted(args.pfaf2)

    for var in args.vars:
        d = args.out / var
        d.mkdir(parents=True, exist_ok=True)

    for pfaf2 in pfaf2s:
        bounds = pfaf2_bounds(args.cat_dir, pfaf2)
        if bounds is None:
            print(f"pfaf2 {pfaf2}: no level-2 catchment shapefile, skipping")
            continue
        print(f"pfaf2 {pfaf2}: bounds {tuple(round(float(b), 3) for b in bounds)}")

        for var in args.vars:
            out_path = args.out / var / f"{pfaf2}.vrt"
            if out_path.is_file() and not args.overwrite:
                print(f"  {var}: exists, skipping ({out_path})")
                continue
            tiles = tiles_covering(args.raw, var, bounds)
            if not tiles:
                print(f"  {var}: NO TILES FOUND, skipping")
                continue
            w, h, n = write_vrt(out_path, tiles)
            print(f"  {var}: {n:>3} tiles -> {w}x{h} px, {out_path.stat().st_size/1024:.1f} KB")

    print(f"\nVRTs written under {args.out}/{{{','.join(args.vars)}}}/")


if __name__ == "__main__":
    main()
