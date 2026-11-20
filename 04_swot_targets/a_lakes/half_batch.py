"""Run the HALF v1 lake outlier filter on the stored SWOT LakeSP download, for every lake of the graph.

HALF v1 is the lake filter of Trudel et al. (code: Wang & Trudel 2026, Zenodo 10.5281/zenodo.20782248). The folder
half_v1/ next to the data holds the unmodified release (`vendor/half_v1` in this repository); this script imports
its functions.

Usage: python half_batch.py vC|vD [--start ISO] [--end ISO] [--lakes N]      (run with the HALF environment)
  vC|vD     LakeSP product version to process
  --start   start of the window, default 2023-07-21T05:33:45Z
  --end     end of the window, default 2025-05-18T21:36:19Z for vC and 2026-12-31T00:00:00Z for vD
  --lakes   process only the first N lakes (for testing); default 0 = all
  SLURM_CPUS_PER_TASK sets the number of worker processes (default 8).

Reads (in $SWOTGAT_WORK/CONUS_storage_fork/eval/swotext/)
  hc_all_<version>.parquet   LakeSP observations downloaded from Hydrocron by hc_pull.py

Writes (same folder)
  half_obs_<version>.parquet     every observation, with filter_flag (1 = kept by HALF), wse_adjusted,
                                 filter_attempt, intra_cycle_flag, n_while, n_while_r2
  half_thr_<version>.parquet     the thresholds HALF calibrated per lake
  half_daily_<version>.parquet   per lake and day: median wse_adjusted of the kept observations (column wse),
                                 median wse_u and wse_std, number of observations

Method
  Per lake the script repeats the release script half_v1_run.py step by step:
    Step 1  calibrate_heuristic_thresholds, with the same conservative selection (conservative_SQL) and the same
            stratification.
    Step 2  apply_customized_filter in strict mode. If that fails, lenient mode. If that fails, HALF's baseline
            Tukey filter. If that fails, no filtering. filter_attempt records which one was used (1_strict,
            2_lenient, 3_tukey, 4_none). The threshold bounds and FILTER_CONFIG are those of the release script.
    Step 3  sp_cycle_adjustment, HALF's bias correction between passes. Its result is wse_adjusted.
  It differs from the release script in three ways, none of them in the filter: the input comes from the stored
  download instead of a live Hydrocron query (same fields), no per-lake figures are drawn, and lakes are processed
  in parallel.
  The daily value is the median of wse_adjusted over the kept observations of the day, as in the HALF authors'
  own daily series ("daily median of valid observations after filtering and corrections").

Notes
  release_replay.py and compare_replay.py check that this script gives the same result as the release script.
  A lake whose processing raises an exception is skipped and counted in the final printed line.
"""
from swotgat.paths import WORK
import os, sys, io, contextlib, warnings, argparse, time as _t
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np, pandas as pd
from multiprocessing import Pool
D = f"{WORK}/CONUS_storage_fork/eval/swotext"; sys.path.insert(0, f"{D}/half_v1")
FILL = -999999999999
NUMERIC = ["time", "wse", "wse_u", "wse_std", "xtrk_dist", "quality_f", "xovr_cal_q", "ice_clim_f", "cycle_id", "pass_id", "p_ref_area", "area_total",
           "dark_frac", "partial_f", "area_tot_u", "ds1_l", "ds1_q", "p_ref_wse"]
FILTER_CONFIG = dict(apply_low_pass_filter="yes", evaluating_at_full_data="no", r2_filter="yes", filter_type="savgol", z_score_thresholds=[2.576, 3.5],
                     maximum_residual_spreads=[0.07, 0.05], show_filtering_evolution="no")
STRICT = dict(max_gap=90, min_range=365); LENIENT = dict(max_gap=90, min_range=365)
def one_lake(args):
    """Filter the observations of one lake with HALF.

    args = (lake id, its observations, window start, window end). Returns (observations with HALF's
    flags, threshold summary), None if the lake has no usable data, or ("error", lake id, message)
    if HALF raised an exception.
    """
    lake, df, start, end = args
    from half_v1_functions import calibrate_heuristic_thresholds, apply_customized_filter, apply_baseline_tukey_filter, sp_cycle_adjustment
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = df.copy(); df["index_col"] = range(len(df))
            df = df.loc[(df["time"] != FILL) & (df["wse"] != FILL) & df["time"].notna() & df["wse"].notna()].copy()
            if df.empty: return None
            for c in ("wse_u", "wse_std", "xtrk_dist"): df[c] = df[c].mask(df[c] == FILL, np.nan)
            df["filter_flag"] = 1; df = df.sort_values("time", kind="mergesort")
            df["datetime"] = pd.to_datetime(df["time"], unit="s", origin=pd.Timestamp("2000-01-01 00:00:00"))
            df_eval = df.drop_duplicates(subset="time", keep="first").copy()
            nv = '(crid.str.endswith("C2") | crid.str.match(".*D.$"))'
            conservative_SQL = f'((xovr_cal_q < 1) & (   (quality_f < 1) | ((quality_f == 1) & {nv})))'
            thr = calibrate_heuristic_thresholds(df_eval, conservative_SQL, by_crid_scenario=[False, False, False], by_pass_id=[False, False, True], by_ice=[True, True, True])
            rules = ["ice-free", "ice-free", "not apply"]
            common = dict(rules_for_ice_free_data=rules, rules_for_ice_covered_data=rules, gauge_df=None, plot_period=[start, end], **FILTER_CONFIG)
            attempt = "1_strict"
            out = apply_customized_filter(df_eval, thr, wse_std_threshold_bounds=[0, 3], wse_u_threshold_bounds=[0, 0.5], xtrk_dist_threshold_bounds=[0, 75000],
                                          wse_std_ice_min=3, wse_u_ice_min=0.1, allow_major_gap="no", max_temporal_gap=STRICT["max_gap"], min_temporal_range=STRICT["min_range"], **common)
            df_f, nwhile, status, thr_sum, obs_thr = out
            if status == "no data": return None
            if status == "fail":
                attempt = "2_lenient"
                df_f, nwhile, status, thr_sum, obs_thr = apply_customized_filter(df_eval, thr, wse_std_threshold_bounds=[0.5, 5], wse_u_threshold_bounds=[0.1, 0.5],
                                          xtrk_dist_threshold_bounds=[0, 75000], wse_std_ice_min=5, wse_u_ice_min=0.1, allow_major_gap="yes",
                                          max_temporal_gap=LENIENT["max_gap"], min_temporal_range=LENIENT["min_range"], **common)
            if status == "fail":
                attempt = "3_tukey"
                df_f, nw, status = apply_baseline_tukey_filter(df_eval, "(quality_f == 0) & (xovr_cal_q == 0) & (ice_clim_f < 2)", multiplier=3, lower_q=0.25, upper_q=0.75, iteration_n=5)
                nwhile = [nw, -2]
            if status == "fail":
                attempt = "4_none"; nwhile = [0, 0]
            if attempt in ("3_tukey", "4_none"): thr_sum = thr.copy()
            _, _, df_f = sp_cycle_adjustment(df_f)
            icf = int((df_f["wse"].notna() & df_f["wse_adjusted"].notna() & (df_f["wse"] != df_f["wse_adjusted"])).any())
            df.loc[~df["index_col"].isin(df_f["index_col"]), "filter_flag"] = 0
            df = df.merge(df_f[["index_col", "wse_adjusted"]], on="index_col", how="left")
            df["filter_attempt"] = attempt; df["intra_cycle_flag"] = icf; df["n_while"] = nwhile[0]; df["n_while_r2"] = nwhile[1]
            thr_sum = thr_sum.copy(); thr_sum["filter_attempt"] = attempt
            return df, thr_sum
    except Exception as e:
        return ("error", lake, f"{type(e).__name__}: {str(e)[:200]}")
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("version"); ap.add_argument("--start", default="2023-07-21T05:33:45Z")
    ap.add_argument("--end", default=None); ap.add_argument("--lakes", type=int, default=0); a = ap.parse_args(); t0 = _t.time()
    end = a.end or ("2025-05-18T21:36:19Z" if a.version == "vC" else "2026-12-31T00:00:00Z")
    h = pd.read_parquet(f"{D}/hc_all_{a.version}.parquet"); h = h[h["_error"].isna()] if "_error" in h else h
    h = h.drop(columns=[c for c in h.columns if c.startswith("_")])
    # parse every column as pd.read_csv would a Hydrocron CSV: numeric where all non-missing values are numeric, else plain object strings
    # (the stored download keeps everything as pandas "string", whose pd.NA breaks HALF's boolean tests)
    for c in h.columns:
        v = h[c].astype(object).where(h[c].notna(), np.nan)
        num = pd.to_numeric(v, errors="coerce")
        if num.notna().sum() == v.notna().sum(): h[c] = num.astype("int64") if (num.notna().all() and (num % 1 == 0).all() and c != "time") else num
        else: h[c] = v
    h["lake_id"] = h["lake_id"].astype("int64").astype(str) if h["lake_id"].dtype != object else h["lake_id"]
    for c in ("crid", "collection_shortname", "time_str"): h[c] = h[c].astype(str)
    ts = pd.to_datetime(h.time_str, errors="coerce", utc=True); h = h[(ts >= pd.Timestamp(a.start)) & (ts <= pd.Timestamp(end))]
    lakes = sorted(h.lake_id.unique());  lakes = lakes[:a.lakes] if a.lakes else lakes
    groups = [(l, g, a.start, end) for l, g in h[h.lake_id.isin(lakes)].groupby("lake_id")]
    print(f"{a.version}: {len(groups)} lakes, {sum(len(g[1]) for g in groups):,} obs, window {a.start} .. {end}", flush=True)
    obs, thr, err = [], [], []
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)), maxtasksperchild=50) as p:
        for i, r in enumerate(p.imap_unordered(one_lake, groups, chunksize=4)):
            if r is None: continue
            if isinstance(r, tuple) and len(r) == 3 and r[0] == "error": err.append(r[1:]); continue
            obs.append(r[0]); thr.append(r[1])
            if (i + 1) % 500 == 0: print(f"  {i+1}/{len(groups)} lakes, {_t.time()-t0:.0f}s", flush=True)
    O = pd.concat(obs, ignore_index=True); T = pd.concat(thr, ignore_index=True)
    O.to_parquet(f"{D}/half_obs_{a.version}.parquet", index=False); T.astype({c: str for c in T.columns if T[c].dtype == object}).to_parquet(f"{D}/half_thr_{a.version}.parquet", index=False)
    K = O[O.filter_flag == 1].copy(); K["date"] = K.datetime.dt.normalize()
    Dd = K.groupby(["lake_id", "date"]).agg(wse=("wse_adjusted", "median"), wse_u=("wse_u", "median"), wse_std=("wse_std", "median"), n_obs=("wse_adjusted", "size")).reset_index()
    Dd.to_parquet(f"{D}/half_daily_{a.version}.parquet", index=False)
    lk = O.drop_duplicates("lake_id")
    print(f"done {_t.time()-t0:.0f}s: lakes {O.lake_id.nunique()}, errors {len(err)}, retained obs {K.shape[0]:,} of {len(O):,} ({K.shape[0]/len(O):.1%}); "
          f"attempts {lk.filter_attempt.value_counts().to_dict()}; cross-pass corrected lakes {int(lk.intra_cycle_flag.sum())}; daily lake-days {len(Dd):,}", flush=True)
    if err: print("error samples:", err[:5], flush=True)
