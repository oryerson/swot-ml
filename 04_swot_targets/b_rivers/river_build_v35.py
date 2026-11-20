"""Build the daily river storage-change record on graph v3.5, up to the latest SWOT pass (first of two builds).

This build uses the reach areas of the first graph version, so river nodes whose id is new in graph v3.5 get no
dv_river here. Its record and its crosswalk are the inputs of reach_area_v35.py, which makes the reach-area table
for graph v3.5. river_build_v35ra.py then repeats the build with that table and gives the record used for the
targets. This script is river_build_v2.py with the graph-v3.5 sub-basin table and other output names.

Usage: python river_build_v35.py      (no arguments; command-line arguments are ignored)

Reads
  $SWOTGAT_WORK/CONUS_storage_fork/eval/graph_v3/v35/subbasins_conus_v35.parquet   graph v3.5 sub-basin table
  NA.parquet in the RiverSP folder of merge_swot_river_conus.py (--reach-dir default)   SWOT RiverSP version D,
                                                       North America, archived tables (to 2026-06-28)
  $SWOTGAT_WORK/CONUS_storage_fork/eval/swotext/hc_reach_vD_2026.parquet   Hydrocron download (hc_pull_reach.py);
                                                       only records later than the last archived record are used
  $SWOTGAT_WORK/CONUS/metadata/reach_area_conus.csv    reach areas of the first graph version
  $SWOTGAT_WORK/CONUS/basin_lists/swotqc/drop_river_paper_min.parquet and $SWOTGAT_WORK/CONUS/zarr_store
                                                       drop list and dv_river of the first store (checks only)

Writes (to $SWOTGAT_WORK/CONUS_storage_fork/eval/v35_inputs/forcing/work/river/)
  reach_lookup_v35.csv                        node -> SWOT reach crosswalk on graph v3.5
  riversp_vD_present_graph_v35.parquet        one row per node and day: dv_river (not masked), the SWOT fields used
                                              by the quality rules, and the flag drop_paper
  drop_river_paper_min_present_v35.parquet    (basin, subbasin, date) of the rows with drop_paper set
  It does not write to any data store.

Method
  Processing is that of merge_swot_river_conus.py, whose functions are imported: crosswalk with the gauge matches
  of the SWOT Validation Set taking precedence; keep reach_q_b < 2^18 and wse_u < 0.20 m; level anomaly from each
  reach's median over its whole record; per node and day the record with the lowest wse_u.
  dv_river = d_wse_river x the node's fixed median reach area / 1e6, in million cubic meters
  (build_storage_target_conus.py).
  Drop list: the rules of build_river_paper_mask.py applied to the same records, on rows with a finite dv_river.
  A row is dropped if xovr_cal_q > 1, or area_detct / area_total <= 0.5, or ice_clim_f > 1, or
  |wse - p_wse| >= 10 m. A 3-point Hampel filter on d_wse_river then runs per node on the rows that remain.
  Checks (printed): the drop list against the first store's drop list up to 2025-12-31, and dv_river against
  the first store's dv_river in the first 15 basins, after removing each series' median. The printed messages
  call the first store "production" and the archived tables "NAS".
"""
from swotgat.paths import WORK
import sys; sys.argv = [sys.argv[0]]
sys.path.insert(0, f"{WORK}/CONUS/preprocess")
import numpy as np, pandas as pd, geopandas as gpd, xarray as xr
import merge_swot_river_conus as M
C = f"{WORK}/CONUS"; OUT = f"{WORK}/CONUS_storage_fork/eval/v35_inputs/forcing/work/river"; SWX = f"{WORK}/CONUS_storage_fork/eval/swotext"
EXTRA = ["xovr_cal_q", "p_wse"]
args = M.parse_args(); from pathlib import Path as _P; args.subbasins = _P(f"{WORK}/CONUS_storage_fork/eval/graph_v3/v35/subbasins_conus_v35.parquet"); subs = gpd.read_parquet(args.subbasins); subs.index = subs.index.astype(str)
merged = M.build_reach_lookup(args, subs)
merged.rename_axis("subbasin").reset_index()[["subbasin", "reach_id", "outlet_id"]].to_csv(f"{OUT}/reach_lookup_v35.csv", index=False)
reaches = merged["reach_id"].astype(float).astype(int).astype(str).unique().tolist(); print("reaches", len(reaches), flush=True)
nas = pd.read_parquet(f"{args.reach_dir}/NA.parquet", columns=M.FIELDS + EXTRA, filters=[("reach_id", "in", reaches), ("wse", "!=", M.WSE_FILL)])
cut = nas.datetime.max(); print(f"NAS records {len(nas):,} to {cut}", flush=True)
hc = pd.read_parquet(f"{SWX}/hc_reach_vD_2026.parquet"); hc = hc[hc["_error"].isna()] if "_error" in hc else hc
hc["datetime"] = pd.to_datetime(hc.time_str, utc=True).dt.tz_convert(None)
for c in M.FIELDS + EXTRA:
    if c not in ("reach_id", "datetime"): hc[c] = pd.to_numeric(hc[c], errors="coerce")
hc = hc[(hc.datetime > cut) & (hc.wse != M.WSE_FILL) & hc.wse.notna()][M.FIELDS + EXTRA]
print(f"Hydrocron records after {cut}: {len(hc):,}, {hc.datetime.min()} .. {hc.datetime.max()}", flush=True)
swot = pd.concat([nas, hc], ignore_index=True); print(f"  raw SWOT records: {len(swot):,}", flush=True)
swot = swot[swot["reach_q_b"] < M.REACH_Q_B_MAX].copy(); swot = swot[swot["wse_u"] < M.WSE_U_MAX]; print(f"  after quality filters: {len(swot):,}", flush=True)
med_wse = swot.groupby("reach_id")["wse"].transform("median"); med_w = swot.groupby("reach_id")["width"].transform("median")
swot["d_wse"] = swot["wse"] - med_wse; swot["d_width"] = swot["width"] - med_w; swot["d_area"] = swot["d_wse"]*(swot["width"] + med_w)/2
swot = swot.set_index(["reach_id", "datetime"])
areas = pd.read_csv(f"{C}/metadata/reach_area_conus.csv", dtype={"subbasin": str}).set_index("subbasin").reach_area_m2
frames = []
for sb in merged.index.unique():
    rids = np.atleast_1d(merged.loc[[sb], "reach_id"].values).astype(float).astype(int).astype(str)
    df = M.get_swot_for_reaches(rids, swot)
    if df.empty: continue
    df = df.reset_index(); df["subbasin"] = sb; df["basin"] = subs.outlet_id.get(sb); df["dv_river"] = df.d_wse_river*areas.get(sb, np.nan)/1e6
    frames.append(df[["basin", "subbasin", "date", "dv_river", "d_wse_river", "wse_river", "wse_u_river", "xtrk_dist_river", "area_detct_river",
                      "area_total_river", "ice_clim_f_river", "xovr_cal_q_river", "p_wse_river"]])
R = pd.concat(frames, ignore_index=True); R["date"] = pd.to_datetime(R.date).dt.tz_localize(None)
# ---- drop list (rules of build_river_paper_mask.py), on records with a finite dv_river, as in that script
R = R.sort_values(["subbasin", "date"]).reset_index(drop=True); ok = np.isfinite(R.dv_river)
af = R.area_detct_river/R.area_total_river
drop = (R.xovr_cal_q_river > 1) | ~(af.fillna(0) > 0.5) | (R.ice_clim_f_river.fillna(0) > 1) | ((R.wse_river - R.p_wse_river).abs() >= 10)
drop &= ok
def hampel3(x):
    """Flag outliers in the series x with a Hampel filter on a centered 3-observation window.

    Returns a boolean array that is True where |x_t - median| > 3 x 1.4826 x MAD of the window. The
    first and last value have no full window and are never flagged.
    """
    out = np.zeros(len(x), bool)
    for i in range(1, len(x) - 1):
        w = x[i-1:i+2]; m = np.median(w); mad = np.median(np.abs(w - m))
        if abs(x[i] - m) > 3*1.4826*mad: out[i] = True
    return out
surv = R[ok & ~drop]
for sb, g in surv.groupby("subbasin"):
    if len(g) < 3: continue
    h = hampel3(g.d_wse_river.values); drop.loc[g.index[h]] = True
R["drop_paper"] = drop
R.to_parquet(f"{OUT}/riversp_vD_present_graph_v35.parquet", index=False)
M_ = R.loc[R.drop_paper, ["basin", "subbasin", "date"]]; M_.to_parquet(f"{OUT}/drop_river_paper_min_present_v35.parquet", index=False)
print(f"written {len(R):,} rows, {R.subbasin.nunique():,} subbasins, {R.date.min().date()} .. {R.date.max().date()}; rows after 2025-12-31 {int((R.date > '2025-12-31').sum()):,}; "
      f"after {cut.date()} {int((R.date > cut).sum()):,}; paper drops {int(R.drop_paper.sum()):,} ({R.drop_paper.mean():.1%} of rows with dv)", flush=True)
# ---- checks against the first store's drop list and dv_river (up to 2025-12-31)
old = pd.read_parquet(f"{C}/basin_lists/swotqc/drop_river_paper_min.parquet"); old["date"] = pd.to_datetime(old.date)
a = set(zip(old.subbasin.astype(str), old.date)); b = set(zip(M_.subbasin, M_.date)) ; b = {x for x in b if x[1] <= pd.Timestamp("2025-12-31")}
print(f"drop list vs production (<= 2025-12-31): production {len(a):,}, new {len(b):,}, both {len(a & b):,} (Jaccard {len(a & b)/len(a | b):.3f})", flush=True)
chk = []
for bsn in R.basin.dropna().unique()[:15]:
    z = xr.open_zarr(f"{C}/zarr_store/{bsn}", consolidated=False)["dv_river"].sel(date=slice("2023-01-01", "2025-12-31")).load()
    zs = z.subbasin.values.astype(str); rb = R[(R.basin == bsn) & (R.date <= "2025-12-31")]
    for sb, g in rb.groupby("subbasin"):
        if sb not in zs: continue
        s = z.sel(subbasin=z.subbasin.values[list(zs).index(sb)]).to_series(); s.index = pd.to_datetime(s.index)
        j = pd.concat([s.rename("zarr"), g.set_index("date").dv_river.rename("new")], axis=1).dropna()
        if len(j) >= 10: j = j - j.median(); chk.append(dict(same=float(np.mean(np.abs(j.zarr - j.new) < 1e-3*max(j.zarr.abs().median(), 1e-6) + 1e-6)), r=float(np.corrcoef(j.zarr, j.new)[0, 1])))
C_ = pd.DataFrame(chk); print(f"dv_river vs production (re-referenced), {len(C_)} subbasins: median share identical {C_.same.median():.3f}, median r {C_.r.median():.4f}", flush=True)
