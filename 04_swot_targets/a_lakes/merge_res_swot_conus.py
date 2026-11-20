"""Write the version-C SWOT lake series (level and storage change) of every lake node into the data store.

This wrote the first lake targets of the project, from SWOT LakeSP version C. The paper's model trains on
version-D targets built by the other scripts of this folder, so none of the values written here reach the model.
The script is kept because the normalization statistics of the first store were computed with these values in it.

Usage: python merge_res_swot_conus.py [--store DIR] [--subbasins FILE] [--jida-dir DIR] [--outlet-id ID]
  --store      data store to write into (default $SWOTGAT_WORK/CONUS/zarr_store)
  --subbasins  sub-basin table (default $SWOTGAT_WORK/CONUS/metadata/subbasins_conus.parquet)
  --jida-dir   folder with the version-C daily series, one file basin_<region>_daily_timeseries.csv per
               Pfafstetter level-3 region (filtered with HALF and delivered by the HALF authors; the code
               calls them "Jida" files). Default: under the raw-data archive configured as NAS_WATER
  --outlet-id  process only the basin with this outlet id

Method
  A lake (reservoir) node's id is its lake_id in the SWOT Prior Lake Database, so series are matched by node id.
  The regions to load are the first three digits of the lake node ids. A region without a file is skipped with a
  warning. Days without a level (wse) are dropped, and so are the two 3D-LAKES columns. Every remaining column of
  the files (among them wse and dv_SWOTAE) is written for the lake node, with dates as UTC.
"""
from swotgat.paths import NAS_WATER, REPO, WORK
import argparse
import sys
from pathlib import Path

sys.path.insert(0, f"{REPO}/vendor/swot_ml/src")

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from data import ZarrBasinStore

CONUS = Path(f"{WORK}/CONUS")
JIDA_DIR = Path(f"{NAS_WATER}/cjgleason/ted/LakeSPvC_ts_for_Ted/daily_ts_from_LakeSPvC")


def parse_args():
    """Return the parsed command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", type=Path, default=CONUS / "zarr_store")
    p.add_argument("--subbasins", type=Path, default=CONUS / "metadata" / "subbasins_conus.parquet")
    p.add_argument("--jida-dir", type=Path, default=JIDA_DIR)
    p.add_argument("--outlet-id")
    return p.parse_args()


def main():
    """Read the version-C lake series and write them into the data store, lake node by lake node."""
    args = parse_args()

    subs_df = gpd.read_parquet(args.subbasins)
    subs_df.index = subs_df.index.astype(str)
    if args.outlet_id:
        subs_df = subs_df[subs_df["outlet_id"] == args.outlet_id]
        if subs_df.empty:
            sys.exit(f"no subbasins for outlet_id {args.outlet_id}")

    # original_comid in the parquet holds the HydroATLAS stream COMID, not the
    # lake_id of the lake series. For reservoir nodes the index is the lake_id, so override.
    subs_df["original_comid"] = subs_df.index

    basin_subbasin_dict = (
        subs_df.groupby("outlet_id")
        .apply(lambda g: g.index.tolist(), include_groups=False)
        .to_dict()
    )

    pfaf3s = subs_df[subs_df["is_reservoir"]].index.str[0:3].unique()
    print(f"SWOT pfaf3 regions needed: {pfaf3s.tolist()}")

    res_dfs = []
    for pfaf3 in pfaf3s:
        csv_path = args.jida_dir / f"basin_{pfaf3}_daily_timeseries.csv"
        if not csv_path.is_file():
            print(f"WARNING: no Jida CSV for pfaf3={pfaf3} ({csv_path}) - "
                  "reservoirs in this region will have no SWOT data")
            continue
        res_data = pd.read_csv(csv_path)
        res_data["lake_id"] = res_data["lake_id"].astype(str)
        res_data.dropna(axis=0, subset="wse", inplace=True)
        res_data.drop(columns=["dv_3DLAKES", "dv_e_3DLAKES"], inplace=True, errors="ignore")
        res_data.set_index(["lake_id", "date"], inplace=True)
        res_dfs.append(res_data)

    if not res_dfs:
        sys.exit("no SWOT reservoir CSVs could be loaded")

    res_df = pd.concat(res_dfs)
    print(f"Loaded {len(res_df):,} SWOT reservoir observations")

    store = ZarrBasinStore(args.store)

    def get_swot_res(lake_id):
        try:
            df = res_df.xs(lake_id, level="lake_id").copy()
        except KeyError:
            return None
        df.index = pd.to_datetime(df.index).tz_localize("UTC")
        df.index.name = "date"
        return df

    n_written = n_no_data = 0
    for basin_id, basin_group in tqdm(subs_df.groupby("outlet_id"), desc="reservoir basins"):
        all_basin_subs = basin_subbasin_dict[basin_id]
        res_rows = basin_group[basin_group["is_reservoir"]]

        for subbasin, row in tqdm(res_rows.iterrows(), total=len(res_rows),
                                  desc=f"  reservoirs in {basin_id}", leave=False):
            site_df = get_swot_res(str(row["original_comid"]))
            if site_df is None or site_df.empty:
                n_no_data += 1
                continue
            store.write_subbasin_data(basin_id, subbasin, all_basin_subs, site_df)
            n_written += 1

    print(f"\nSWOT reservoir merge complete. written={n_written} no_swot_data={n_no_data}")


if __name__ == "__main__":
    main()
