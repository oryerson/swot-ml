"""Match each river node to SWOT river reaches and write the reaches' RiverSP observations into the data store.

Usage: python merge_swot_river_conus.py [--store DIR] [--subbasins FILE] [--gauges FILE] [--crosswalk-dir DIR]
                                        [--mhs-dir DIR] [--reach-dir DIR] [--outlet-id ID]
                                        [--svs FILE | --no-svs] [--sword-version v16|v17b|v17c]
  --store          data store to write into (default $SWOTGAT_WORK/CONUS/zarr_store)
  --subbasins      sub-basin table (default $SWOTGAT_WORK/CONUS/metadata/subbasins_conus.parquet)
  --gauges         gauge table with columns site_id and COMID. The default file sits in a folder named after the
                   Columbia basin but is the global list (12,114 gauges)
  --crosswalk-dir  MERIT-Basins tables v07_to_bugfix_pfaf1_*.csv (COMID of the bug-fixed version -> COMID of v0.7)
  --mhs-dir        tables all_*_MHS.csv (MERIT-Basins v0.7 COMID -> SWOT reach_id); called the MHS table below
  --reach-dir      SWOT RiverSP reach product, version D, as parquet files
  --outlet-id      process only the basin with this outlet id
  --svs, --no-svs, --sword-version   see "SVS override"

Method
  Node -> SWOT reach is a three-step crosswalk:
      node id  --(gauge nodes: the gauge's COMID)-->  MERIT-Basins COMID, bug-fixed version
               --(v07_to_bugfix table)            -->  MERIT-Basins v0.7 COMID
               --(MHS table)                      -->  SWOT reach_id (one node can match several reaches)
  SVS override: the SWOT Validation Set (SVS) publishes its own gauge-to-reach matches. Where SVS has a match for
  a gauge node, its single reach replaces all reaches the crosswalk gave that node (svs_reach_overrides.py).
  Quality filters: fill values of wse are dropped; an observation is kept if reach_q_b < 2^18 (no high-order
  quality bit set) and the level uncertainty wse_u < 0.20 m.
  Anomalies are taken from each reach's own median over its kept observations:
      d_wse = wse - median(wse);  d_width = width - median(width);  d_area = d_wse x (width + median(width)) / 2
  Per node, the observations of all its reaches are put on one daily axis (time rounded to the day, UTC). If a
  day has several observations, the one with the lowest wse_u is kept. Every field in FIELDS and the three
  anomalies are written with the suffix `_river`.

Notes
  This creates new variables in the store. Run update_zarr_stats_conus.py (stage 03) afterwards so that the
  *_river variables get normalization statistics.
"""
from swotgat.paths import NAS_ICE, NAS_WATER, REPO, WORK
import argparse
import sys
from pathlib import Path

sys.path.insert(0, f"{REPO}/vendor/swot_ml/src")

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from data import ZarrBasinStore

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svs_reach_overrides import build_overrides, load_svs

CONUS = Path(f"{WORK}/CONUS")
GAUGES = Path(f"{WORK}/Columbia_testing/swot-ml-data/"
              "Columbia_05km2/metadata/gauges.parquet")  # the global gauge list (12,114 gauges), despite the folder name
CROSSWALK = Path(f"{NAS_ICE}/data/MERIT_Basins/v07_to_bugfix")
MHS_DIR = Path(f"{NAS_WATER}/cjgleason/ted/swot-ml/data/Heejin_MHS")
REACH_DIR = Path(f"{NAS_ICE}/data/SWOT/RiverSP_Reach_vD/parquet")

FIELDS = [
    "reach_id", "datetime",
    "wse", "wse_u", "wse_r_u",
    "slope", "slope_u", "slope_r_u",
    "slope2", "slope2_u", "slope2_r_u",
    "width", "width_u",
    "area_total", "area_tot_u", "area_detct", "area_det_u", "area_wse",
    "layovr_val", "node_dist", "loc_offset", "xtrk_dist", "reach_q", "reach_q_b",
    "dark_frac", "ice_clim_f", "ice_dyn_f", "partial_f", "obs_frac_n",
]

WSE_FILL = -999999999999.0
REACH_Q_B_MAX = 2 ** 18   # drop reaches with high-order quality bits set
WSE_U_MAX = 0.20          # metres
BATCH_SIZE = 10000


def parse_args():
    """Return the parsed command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", type=Path, default=CONUS / "zarr_store")
    p.add_argument("--subbasins", type=Path, default=CONUS / "metadata" / "subbasins_conus.parquet")
    p.add_argument("--gauges", type=Path, default=GAUGES)
    p.add_argument("--crosswalk-dir", type=Path, default=CROSSWALK)
    p.add_argument("--mhs-dir", type=Path, default=MHS_DIR)
    p.add_argument("--reach-dir", type=Path, default=REACH_DIR)
    p.add_argument("--outlet-id")
    p.add_argument("--svs", type=Path,
                   default=Path(f"{WORK}/SVS/SVS_v1_1_2.nc"),
                   help="SWOT Validation Set NetCDF; its published gauge->reach matches take "
                        "precedence over the crosswalk. Pass --no-svs to disable.")
    p.add_argument("--no-svs", dest="svs", action="store_const", const=None,
                   help="Ignore SVS and use the crosswalk everywhere.")
    p.add_argument("--sword-version", default="v17c", choices=["v16", "v17b", "v17c"],
                   help="SWORD version in SVS to read. Our MHS table is v17b/v17c aligned.")
    return p.parse_args()


def build_reach_lookup(args, subs_df):
    """Return the table of (node, SWOT reach_id) pairs, indexed by node id.

    Uses the crosswalk chain bug-fixed COMID -> v0.7 COMID -> MHS table, then the SVS override.
    Nodes without a match at any step are dropped.
    """
    # Reservoir/plain nodes are keyed by their own COMID; gauge nodes are keyed by
    # the gauge id, so substitute the gauge's matched COMID for those.
    subs_df = subs_df.copy()
    subs_df["original_comid"] = subs_df.index
    gauges = gpd.read_parquet(args.gauges).set_index("site_id")
    common = subs_df.index.intersection(gauges.index)
    subs_df.loc[common, "original_comid"] = gauges.loc[common, "COMID"].astype(str)
    print(f"  gauge COMID overrides applied: {len(common)}")

    files = sorted(args.crosswalk_dir.glob("v07_to_bugfix_pfaf1_*.csv"))
    if not files:
        sys.exit(f"no crosswalk CSVs in {args.crosswalk_dir}")
    bugfix_to_v07 = pd.concat([pd.read_csv(f, dtype=str) for f in files]).set_index("COMID_bugfix")
    subs_df = subs_df.merge(bugfix_to_v07, how="left", left_on="original_comid", right_index=True)

    n_before = len(subs_df)
    subs_df = subs_df.dropna(subset="COMID_v07")
    subs_df["COMID_v07"] = subs_df["COMID_v07"].astype(int)
    print(f"  matched to MERIT v0.7 COMID: {len(subs_df)}/{n_before}")

    mhs_files = sorted(args.mhs_dir.glob("all_*_MHS.csv"))
    if not mhs_files:
        sys.exit(f"no MHS CSVs in {args.mhs_dir}")
    mhs = pd.concat([pd.read_csv(f)[["COMID", "reach_id"]].drop_duplicates().dropna()
                     for f in mhs_files])

    merged = subs_df.reset_index().merge(mhs, left_on="COMID_v07", right_on="COMID", how="left")
    merged = merged.dropna(subset="reach_id").set_index("id")
    print(f"  matched to SWOT reach_id: {merged.index.nunique()} subbasins "
          f"({len(merged)} subbasin-reach pairs)")

    if args.svs and args.svs.is_file():
        merged = apply_svs_overrides(merged, args.svs, args.sword_version)
    elif args.svs:
        print(f"  SVS file not found ({args.svs}); keeping crosswalk assignments")

    return merged


def apply_svs_overrides(merged, svs_path, sword_version):
    """Use the gauge-to-reach match published by the SWOT Validation Set (SVS) where it has one.

    SVS matches gauges to SWORD reaches independently of the COMID -> v0.7 -> MHS
    chain, so its match is preferred for the gauges it covers. SVS only covers
    gauges, so every other node keeps its crosswalk reaches. The override changes
    a small minority of links.

    An override replaces all crosswalk reaches of a node with the single SVS
    reach. The two sources disagree only by picking adjacent segments of the same
    river, so keeping both would count one river twice rather than add coverage.
    """
    svs = load_svs(svs_path, sword_version)
    ov = build_overrides(merged.index.unique(), svs)
    if not ov:
        print("  SVS overrides: none applicable")
        return merged

    ov_ser = pd.Series(ov, dtype="int64")
    cur = merged.groupby(level=0)["reach_id"].apply(lambda s: set(int(x) for x in s))
    changed = [s for s, r in ov_ser.items() if s in cur.index and r not in cur[s]]

    keep = merged[~merged.index.isin(ov_ser.index)]
    repl = merged[merged.index.isin(ov_ser.index)].groupby(level=0).first().copy()
    repl["reach_id"] = ov_ser.reindex(repl.index).astype("int64")
    merged = pd.concat([keep, repl]).sort_index()

    print(f"  SVS overrides applied: {len(repl)} subbasin(s) "
          f"({len(changed)} actually changed reach; the rest already agreed)")
    return merged


def load_swot(args, reach_ids):
    """Read the RiverSP records of the given reaches, apply the quality filters, add the anomalies.

    Returns a table indexed by (reach_id, datetime), or an empty table if nothing passes.
    """
    filters = [("reach_id", "in", reach_ids), ("wse", "!=", WSE_FILL)]
    files = sorted(args.reach_dir.glob("*.parquet"))
    if not files:
        sys.exit(f"no SWOT parquet in {args.reach_dir}")

    swot = pd.concat([pd.read_parquet(f, columns=FIELDS, filters=filters) for f in files])
    print(f"  raw SWOT records: {len(swot):,}")

    swot = swot[swot["reach_q_b"] < REACH_Q_B_MAX].copy()
    swot = swot[swot["wse_u"] < WSE_U_MAX]
    print(f"  after quality filters: {len(swot):,}")
    if swot.empty:
        return swot

    # Anomalies relative to each reach's own median.
    med_wse = swot.groupby("reach_id")["wse"].transform("median")
    med_width = swot.groupby("reach_id")["width"].transform("median")
    swot["d_wse"] = swot["wse"] - med_wse
    swot["d_width"] = swot["width"] - med_width
    swot["d_area"] = swot["d_wse"] * (swot["width"] + med_width) / 2

    return swot.set_index(["reach_id", "datetime"])


def get_best_record(grp):
    """Return the record of grp with the lowest level uncertainty (wse_u)."""
    return grp.sort_values("wse_u").iloc[0]


def safe_xs(df, val, level):
    """Return the rows of df whose index level `level` equals val (an empty table if there are none)."""
    try:
        return df.xs(val, level=level)
    except KeyError:
        return pd.DataFrame()


def get_swot_for_reaches(reach_ids, swot):
    """Return the records of one node's reach or reaches, one row per day.

    Times are rounded to the day (UTC). If a day has several records, the one
    with the lowest wse_u is kept. Column names get the suffix `_river`.
    """
    frames = [safe_xs(swot, rid, "reach_id") for rid in reach_ids]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames)
    out.index = pd.to_datetime(out.index).round("D").tz_localize("UTC")
    out.index.name = "date"

    # Multiple observations on one day -> keep the lowest-uncertainty record.
    if out.index.duplicated().any():
        out = out.groupby(out.index.name).apply(get_best_record)

    return out.add_suffix("_river")


def main():
    """Build the node-to-reach crosswalk, read the SWOT records and write them into the store."""
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

    print(f"{len(subs_df):,} subbasins / {len(basin_subbasin_dict)} basin(s)")
    print("Building subbasin -> SWOT reach crosswalk ...")
    merged = build_reach_lookup(args, subs_df)
    if merged.empty:
        sys.exit("no subbasins matched a SWOT reach")

    reaches = merged["reach_id"].astype(float).astype(int).astype(str).unique().tolist()
    print(f"Loading SWOT records for {len(reaches):,} reaches ...")
    swot = load_swot(args, reaches)
    if swot.empty:
        sys.exit("no SWOT records survived filtering")

    store = ZarrBasinStore(args.store)

    batches = []
    for basin_id in merged["outlet_id"].unique():
        basin_subs = basin_subbasin_dict[basin_id]
        for i in range(0, len(basin_subs), BATCH_SIZE):
            batches.append((basin_id, basin_subs[i : i + BATCH_SIZE]))

    n_written = 0
    for basin_id, batch_subs in tqdm(batches, desc="SWOT river batches"):
        basin_subs = basin_subbasin_dict[basin_id]
        batch_df_list = []

        for subbasin in batch_subs:
            if subbasin not in merged.index:
                continue
            # .loc[[x]] always yields a DataFrame, avoiding the scalar/Series
            # ambiguity when a subbasin maps to exactly one reach.
            rids = merged.loc[[subbasin], "reach_id"].values
            rids = np.atleast_1d(rids).astype(float).astype(int).astype(str)

            swot_df = get_swot_for_reaches(rids, swot)
            if swot_df.empty:
                continue
            swot_df["subbasin"] = subbasin
            swot_df.index.name = "date"
            batch_df_list.append(swot_df)
            n_written += 1

        # Nothing to write for an empty batch: the store is already initialized with NaN.
        if batch_df_list:
            batch_df = pd.concat(batch_df_list)
            store.write_batch_data(basin_id, batch_df, basin_subs, batch_subs)

    print(f"\nSWOT river merge complete. subbasins written={n_written}")
    print("NOTE: re-run update_zarr_stats_conus.py so the *_river vars get stats.")


if __name__ == "__main__":
    main()
