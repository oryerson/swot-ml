"""Delineate one river basin into sub-basins, with one node per gauge and one per reservoir.

The work is done by the graph delineator (vendor/graph_delineator,
delineate_basins). This script chooses the basin, loads the reservoir polygons
around it and passes the settings of the paper's graph. Run it once per basin,
for example as a Slurm array over the rows of basin_manifest.csv.

Reads
  --manifest   basin_manifest.csv from build_conus_basin_list.py; used only
               with --array-index, to turn the row number into an outlet_id
               (default $SWOTGAT_WORK/CONUS/basin_lists/basin_manifest.csv)
  --gauge-dir  folder with the basin's gauge table <outlet_id>.csv
               (default $SWOTGAT_WORK/CONUS/basin_lists/gauges; the paper's
               graph v3.4 uses the re-snapped tables from snap_gauges_v34.py)
  --res-gdb    reservoir polygons of the SWOT Prior Lake Database (default
               $SWOTGAT_WORK/LakeSPvC_ts_for_Ted/SWOT_PLD_v106_reservoirs.gdb)
  MERIT-Basins vector data ($SWOTGAT_NAS_ICE/data/MERIT_Basins/
  MERIT_Hydro_v07_Basins_v01_bugfix/pfaf_level_01) and the MERIT-Hydro mosaics
  from build_merit_vrts.py ($SWOTGAT_WORK/CONUS/merit_rasters/dir and upg)

Writes
  <out-dir>/<outlet_id>/   the delineator's output for the basin; the file
               subbasins/<outlet_id>_subbasins.parquet marks the basin as done
               (default out-dir $SWOTGAT_WORK/CONUS/delineation)

Method
  - Sub-basins are at most --max-area (default 500) km2. Reservoirs of at
    least --min-reservoir-area (default 0.5) km2 become nodes.
  - Reservoir polygons are loaded only inside the bounding box of the basin's
    gauges plus 0.5 degrees, so the cost scales with basin size. The box covers
    the gauges of --gauge-dir and, if present, the same basin's original table
    in $SWOTGAT_WORK/CONUS/basin_lists/gauges. Re-snapped gauges can move
    inward; using both tables keeps the box, and so the set of reservoirs, the
    same as with the original gauges.
  - A basin that is already delineated is skipped unless --overwrite is given.

Usage
-----
  python delineate_basin_v34.py --outlet-id USGS-01578310
  python delineate_basin_v34.py --array-index 0 --out-dir <dir> --gauge-dir <dir>
"""
from swotgat.paths import NAS_ICE, WORK

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

import graph_delineator as gd  # reservoir-aware delineate_basins (see __init__.py)

CONUS = Path(f"{WORK}/CONUS")
MANIFEST = CONUS / "basin_lists" / "basin_manifest.csv"
GAUGE_DIR = CONUS / "basin_lists" / "gauges"
OUT_DIR = CONUS / "delineation"
RES_GDB = Path(
    f"{WORK}/LakeSPvC_ts_for_Ted/SWOT_PLD_v106_reservoirs.gdb"
)
# MERIT inputs of the delineator: the MERIT-Basins vector data, and the flow-direction
# and flow-accumulation rasters as VRT mosaics over the raw MERIT-Hydro tiles
# (written by build_merit_vrts.py).
MERIT_DIRS = {
    "basins": Path(f"{NAS_ICE}/data/MERIT_Basins/MERIT_Hydro_v07_Basins_v01_bugfix/pfaf_level_01"),
    "flow_dir": Path(f"{WORK}/CONUS/merit_rasters/dir"),
    "flow_acc": Path(f"{WORK}/CONUS/merit_rasters/upg"),
}

# Delineation settings of the paper's graph: reservoirs of at least 0.5 km2 become nodes,
# sub-basins are at most 500 km2, and reservoirs are loaded within 0.5 degrees of the gauges.
MIN_RESERVOIR_AREA_KM2 = 0.5
MAX_AREA = 500
BBOX_PAD_DEG = 0.5


def parse_args():
    """Parse the command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--outlet-id")
    g.add_argument("--array-index", type=int, help="Row of basin_manifest.csv (0-based).")
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--gauge-dir", type=Path, default=GAUGE_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--res-gdb", type=Path, default=RES_GDB)
    p.add_argument("--max-area", type=float, default=MAX_AREA)
    p.add_argument("--min-reservoir-area", type=float, default=MIN_RESERVOIR_AREA_KM2)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    """Delineate the chosen basin and report the number of sub-basins, gauges and reservoirs."""
    args = parse_args()

    if args.array_index is not None:
        manifest = pd.read_csv(args.manifest)
        if args.array_index >= len(manifest):
            sys.exit(f"array index {args.array_index} out of range ({len(manifest)} basins)")
        outlet_id = manifest.iloc[args.array_index]["outlet_id"]
    else:
        outlet_id = args.outlet_id

    gauges_csv = args.gauge_dir / f"{outlet_id}.csv"
    if not gauges_csv.is_file():
        sys.exit(f"no gauge file for {outlet_id}: {gauges_csv}")

    out_dir = args.out_dir / outlet_id
    done = out_dir / "subbasins" / f"{outlet_id}_subbasins.parquet"
    if done.is_file() and not args.overwrite:
        print(f"{outlet_id}: already delineated ({done}); use --overwrite to rebuild")
        return

    gauges_df = pd.read_csv(gauges_csv)
    # Gauges re-snapped by drainage area can move inward. Take the reservoir box from the original gauge table as well,
    # so that the same reservoirs are loaded as with the original gauges (without this, one run lost lake 7420074653).
    orig_csv = GAUGE_DIR / f"{outlet_id}.csv"
    box_df = pd.concat([gauges_df[["lng", "lat"]], pd.read_csv(orig_csv)[["lng", "lat"]]]) if orig_csv.is_file() else gauges_df
    print(f"=== {outlet_id}: {len(gauges_df)} gauges ===", flush=True)

    bbox = (
        box_df.lng.min() - BBOX_PAD_DEG,
        box_df.lat.min() - BBOX_PAD_DEG,
        box_df.lng.max() + BBOX_PAD_DEG,
        box_df.lat.max() + BBOX_PAD_DEG,
    )
    print(f"reservoir bbox: {bbox}", flush=True)
    reservoirs = gpd.read_file(args.res_gdb, bbox=bbox)
    print(f"loaded {len(reservoirs)} reservoirs in bbox", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    gd.delineate_basins(
        gauges_path=gauges_csv,
        merit_dirs=MERIT_DIRS,
        reservoirs=reservoirs,
        min_reservoir_area_km2=args.min_reservoir_area,
        max_area=args.max_area,
        output_dir=out_dir,
        save_plots=False,   # plots are the slow part and are not consumed downstream
        consolidate=True,
        overwrite=args.overwrite,
    )
    print(f"=== {outlet_id}: done in {(time.time() - t0) / 60:.1f} min ===", flush=True)

    if done.is_file():
        subs = gpd.read_parquet(done)
        print(f"subbasins={len(subs)} gauges={int(subs['is_gauge'].sum())} "
              f"reservoirs={int(subs['is_reservoir'].sum())}", flush=True)
    else:
        sys.exit(f"ERROR: delineation finished but {done} was not written")


if __name__ == "__main__":
    main()
