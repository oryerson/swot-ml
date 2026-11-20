"""Write the extracted ERA5-Land forcing of every sub-basin into the shared Zarr data store.

Reads
    --datalake   per-basin, per-year Parquet files written by flipped_extract_sub_basins.py
                 (default $SWOTGAT_WORK/CONUS/datalakes/training/dynamic_data/era5, with one
                 folder `basin=<outlet_id>` per basin)
    --subbasins  the sub-basin table; its index is the sub-basin (node) id and its column
                 `outlet_id` names the basin (default
                 $SWOTGAT_WORK/CONUS/metadata/subbasins_conus.parquet)

Writes
    --store      one group per basin in the Zarr store (default $SWOTGAT_WORK/CONUS/zarr_store),
                 through ZarrBasinStore.write_batch_data of the vendored swot_ml package

Usage
    python merge_era5_conus.py [--store DIR] [--datalake DIR] [--subbasins FILE]
                               [--outlet-id BASIN]
    --outlet-id merges only that basin. Without it, every basin in --subbasins is merged.

Method
    Sub-basins are written in batches of 1000 per basin. For each batch the script reads all
    yearly Parquet files of the basin, keeps the rows of the batch's sub-basins, drops duplicate
    (sub-basin, date) rows (the first one is kept) and writes the batch. A basin with no files
    is skipped with a warning. The script exits with an error if any batch fails to write.

Notes
    Each basin writes to its own group in the store, so different basins may be merged at the
    same time. Two concurrent merges of the same basin are not safe.
"""
from swotgat.paths import REPO, WORK
import argparse
import sys
from pathlib import Path

sys.path.insert(0, f"{REPO}/vendor/swot_ml/src")

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from data import ZarrBasinStore

CONUS = Path(f"{WORK}/CONUS")
BATCH_SIZE = 1000


def parse_args():
    """Parse the command line (--store, --datalake, --subbasins, --outlet-id)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", type=Path, default=CONUS / "zarr_store")
    p.add_argument("--datalake", type=Path,
                   default=CONUS / "datalakes" / "training" / "dynamic_data" / "era5")
    p.add_argument("--subbasins", type=Path, default=CONUS / "metadata" / "subbasins_conus.parquet")
    p.add_argument("--outlet-id", help="Merge only this basin (default: every basin in --subbasins).")
    return p.parse_args()


def main():
    """Merge the Parquet forcing of every selected basin into the Zarr store, batch by batch.

    A batch is 1000 sub-basins of one basin. Exits with an error if --outlet-id names no
    sub-basin or if any batch fails to write.
    """
    args = parse_args()

    subs_df = gpd.read_parquet(args.subbasins)
    subs_df.index = subs_df.index.astype(str)
    if args.outlet_id:
        subs_df = subs_df[subs_df["outlet_id"] == args.outlet_id]
        if subs_df.empty:
            sys.exit(f"no subbasins for outlet_id {args.outlet_id}")

    basin_subbasin_dict = (
        subs_df.groupby("outlet_id")
        .apply(lambda g: g.index.tolist(), include_groups=False)
        .to_dict()
    )

    store = ZarrBasinStore(args.store)

    batches = []
    for basin_id, basin_subs in basin_subbasin_dict.items():
        for i in range(0, len(basin_subs), BATCH_SIZE):
            batches.append((basin_id, basin_subs, basin_subs[i : i + BATCH_SIZE]))
    print(f"{len(subs_df):,} subbasins / {len(basin_subbasin_dict)} basin(s) -> {len(batches)} batch(es)")

    failed = {}
    for basin_id, all_basin_subs, batch_subs in tqdm(batches, desc="ERA5 batches"):
        root = args.datalake / f"basin={basin_id}"
        if not root.is_dir():
            print(f"WARNING: no datalake partition for basin={basin_id} ({root})")
            continue

        dfs = []
        for year_dir in sorted(root.iterdir()):
            parquets = list(year_dir.glob("*.parquet"))
            if not parquets:
                continue
            tmp = pd.concat([pd.read_parquet(f) for f in parquets])
            tmp = tmp.set_index(["subbasin", "date"])
            # Build the set of available sub-basins once. Testing membership against
            # tmp.index.get_level_values(...) inside the comprehension would recompute it
            # for every candidate and scan the whole index each time: about 1000 linear scans
            # over about 100 million rows for a basin the size of the Mississippi.
            avail = set(tmp.index.get_level_values("subbasin"))
            keep = [s for s in batch_subs if s in avail]
            if not keep:
                continue
            dfs.append(tmp.loc[pd.IndexSlice[keep, :], :])

        if not dfs:
            print(f"WARNING: no ERA5 data found for basin={basin_id}")
            continue

        df = pd.concat(dfs)
        df = df.loc[~df.index.duplicated(keep="first")]

        try:
            store.write_batch_data(basin_id, df, all_basin_subs, batch_subs)
            print(f"  wrote {len(batch_subs)} subs for {basin_id}")
        except Exception as e:
            failed[basin_id] = e
            print(f"  FAILED {basin_id}: {e}")

    if failed:
        sys.exit(f"\nFailed basins: {list(failed.keys())}")
    print("\nERA5 merge complete - all batches succeeded.")


if __name__ == "__main__":
    main()
