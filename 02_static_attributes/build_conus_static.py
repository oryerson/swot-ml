r"""Give every sub-basin its static attributes from BasinATLAS (HydroATLAS), level 9.

Reads
  --subbasins  sub-basin table (GeoParquet) with outlet_id, area_km2, uparea_km2
               (default: the file named by $V35S_SUBS)
  --gdb        BasinATLAS_v10.gdb, layer BasinATLAS_v10_lev09 (default
               $SWOTGAT_NAS_ICE/data/HydroSHEDS/HydroATLAS/BasinATLAS_v10.gdb)
  --props      text file with the BasinATLAS column names to keep, one per line
               (shipped as 02_static_attributes/inputs/properties.txt)

Writes
  --out        table indexed by (basin, subbasin) with area_km2, uparea_km2,
               the BasinATLAS columns and hybas_area_diff
               (default $V35S_WORK/static_conus_$V35S_TAG.parquet). This is the
               layout the model configuration reads as attributes_file.

Usage
  V35S_SUBS=... V35S_WORK=... V35S_TAG=v35 python build_conus_static.py \
      [--subbasins FILE] [--out FILE] [--props FILE] [--gdb FILE] [--dry-run]
  The three environment variables must be set even when --subbasins and --out
  are given, because the defaults are built from them when the script starts.

Method
  For each sub-basin, take its centroid, buffer it by 2500 m (in the equal-area
  projection EPSG:6933) and find the BasinATLAS level-9 polygons that the buffer
  touches. Keep the polygon whose upstream area (UP_AREA) agrees best with the
  sub-basin's uparea_km2, measured by
      hybas_area_diff = (UP_AREA - uparea_km2) / mean(UP_AREA, uparea_km2).
  The value of the kept polygon is written to the output, as a record of how
  good the match is. The no-data value -9999 of wet_cl_smj (wetland class) is
  replaced by class 13.
  The work is done basin by basin: each basin reads only the BasinATLAS polygons
  under its own bounding box plus 1 degree, so the cost scales with basin
  extent instead of reading all (about 508,000) global polygons for every basin.
"""
from swotgat.paths import NAS_ICE, NAS_WATER

import argparse
import os as _os  # file names come from the environment: V35S_SUBS, V35S_WORK, V35S_TAG
from pathlib import Path

import geopandas as gpd
import pandas as pd

DATASETS = Path(f"{NAS_ICE}/data")
GDB = DATASETS / "HydroSHEDS" / "HydroATLAS" / "BasinATLAS_v10.gdb"
PROPS_TXT = Path(
    f"{NAS_WATER}/cjgleason/ted/swot-ml/data/owen_reservoirs/Columbia/attributes/properties.txt"
)
SUBS = Path(_os.environ["V35S_SUBS"])
OUT = Path(_os.environ["V35S_WORK"]) / f"static_conus_{_os.environ['V35S_TAG']}.parquet"

PAD_DEG = 1.0  # comfortably beyond the 2500 m join buffer


def parse_args():
    """Parse the command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subbasins", type=Path, default=SUBS)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--props", type=Path, default=PROPS_TXT)
    p.add_argument("--gdb", type=Path, default=GDB)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def attrs_for_basin(subs: gpd.GeoDataFrame, gdb: Path) -> pd.DataFrame:
    """Match the sub-basins of one basin to BasinATLAS level-9 polygons.

    subs holds the sub-basins of one basin (index = node id) and gdb is the path of BasinATLAS_v10.gdb. See
    Method in the module docstring. Return (attrs, n): attrs has one row per sub-basin, indexed by site_id,
    with the columns of its best-matching polygon and hybas_area_diff; n is the number of BasinATLAS polygons
    read.
    """
    subs = subs.copy()
    subs.index = subs.index.astype(str)
    subs.index.name = "site_id"
    subs.geometry = subs.geometry.centroid

    minx, miny, maxx, maxy = subs.total_bounds
    bbox = (minx - PAD_DEG, miny - PAD_DEG, maxx + PAD_DEG, maxy + PAD_DEG)
    hydro = gpd.read_file(gdb, layer="BasinATLAS_v10_lev09", engine="pyogrio",
                          use_arrow=True, bbox=bbox)
    hydro.replace({"wet_cl_smj": -9999}, 13, inplace=True)
    hydro = hydro.set_index("HYBAS_ID")

    subs_buff = subs.to_crs("EPSG:6933")
    subs_buff.geometry = subs_buff.buffer(2500)
    pairs = gpd.sjoin(subs_buff, hydro.to_crs("EPSG:6933"), how="left")
    pairs["hybas_area_diff"] = (pairs["UP_AREA"] - pairs["uparea_km2"]) / (
        (pairs["UP_AREA"] + pairs["uparea_km2"]) / 2
    )

    def best(grp):
        return grp.loc[grp["hybas_area_diff"].abs().idxmin()]

    attrs = pairs.reset_index().groupby("site_id").apply(best, include_groups=False)
    return attrs, len(hydro)


def main():
    """Match every basin's sub-basins to BasinATLAS and write the combined attribute table."""
    args = parse_args()
    subs_all = gpd.read_parquet(args.subbasins)
    props = pd.read_csv(args.props, header=None)[0].to_list()
    basins = sorted(subs_all["outlet_id"].unique())
    print(f"{len(subs_all):,} subbasins across {len(basins)} basins")

    frames = []
    for outlet_id in basins:
        subs = subs_all[subs_all["outlet_id"] == outlet_id]
        attrs, n_hydro = attrs_for_basin(subs, args.gdb)
        print(f"  {outlet_id:<18} {len(subs):>6} subbasins  <- {n_hydro:>6} HydroATLAS polygons in bbox")
        frames.append(attrs)

    attrs = pd.concat(frames)

    keep = ["outlet_id", "area_km2", "uparea_km2"] + props + ["hybas_area_diff"]
    out = (
        attrs[keep]
        .reset_index()
        .rename(columns={"site_id": "subbasin", "outlet_id": "basin"})
        .astype({"subbasin": str, "basin": str})
        .set_index(["basin", "subbasin"])
    )

    print()
    print(f"COMBINED shape={out.shape}")
    print(f"  NaN cells: {int(out.isna().sum().sum())}  unmatched (all-NaN) rows: {int(out.isna().all(axis=1).sum())}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
