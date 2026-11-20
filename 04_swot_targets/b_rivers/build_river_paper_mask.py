"""List the SWOT river observations that fail the minimum quality baseline of Harlan et al.

Harlan et al. (SWOT version D against river gauges in the US and Canada; manuscript under review) name the
filters they found important for R2 > 0.5 and a 1-sigma error below 25 cm (their Discussion and Table 4). For the
reach product an observation is kept if
  xovr_cal_q <= 1                 crossover calibration good or suspect
  area_detct / area_total > 0.5   more than half of the water area was detected
  ice_clim_f <= 1                 not likely fully ice-covered
  |wse - p_wse| < 10 m            level within 10 m of the prior level of the SWORD river database
and if it passes a Hampel filter run on the observations that survive the four rules: 3-observation window,
flag |x_t - median| > 3 x 1.4826 x MAD(window).

Usage: python build_river_paper_mask.py      (no arguments; 12 worker processes)

Reads
  $SWOTGAT_WORK/CONUS/zarr_store/<basin>         the data store with the river observations (variables *_river)
  RiverSP_Reach_vD/parquet/NA.parquet            SWOT RiverSP version D, North America, in the raw-data archive
                                                 configured as NAS_ICE

Writes
  $SWOTGAT_WORK/CONUS/basin_lists/swotqc/drop_river_paper_min.parquet   one row (basin, subbasin, date) per
                                                                        observation to drop

Method
  The rules are applied to what the store already holds, for 2023-01-01 to 2025-12-31. When the store was filled
  (merge_swot_river_conus.py), only observations with reach_q_b < 2^18 and wse_u < 0.20 m were kept, which is
  stricter than the "summary quality better than bad" of Harlan et al.
  xovr_cal_q and p_wse are not in the store. Each river record of the store is therefore matched to its RiverSP
  row exactly on (UTC date rounded to the day, wse, xtrk_dist), all float64. A record without a match skips the
  two rules that need these fields; the share of matched records is printed.
  A missing area fraction counts as failing the area rule.
  The Hampel filter runs per node, in date order, on d_wse_river (the level anomaly from each reach's own
  median), so that nodes with several reaches are not penalized for level offsets between reaches. The first and
  last observation of a node cannot be flagged.
  Lake (reservoir) targets are not filtered here.
"""
from swotgat.paths import NAS_ICE, WORK
import numpy as np, pandas as pd, xarray as xr
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
C = Path(f"{WORK}/CONUS"); Z = C/"zarr_store"; OUT = C/"basin_lists/swotqc"
T0, T1 = "2023-01-01", "2025-12-31"
SP = pd.read_parquet(f"{NAS_ICE}/data/SWOT/RiverSP_Reach_vD/parquet/NA.parquet", columns=["datetime", "wse", "xtrk_dist", "xovr_cal_q", "p_wse"],
                     filters=[("wse", "!=", -999999999999.0)])
SP["date"] = pd.to_datetime(SP.datetime).dt.round("D").dt.tz_localize(None).values.astype("datetime64[D]")
SP = SP.drop(columns="datetime").drop_duplicates(["date", "wse", "xtrk_dist"])

def hampel3(x):
    """Flag outliers in the series x with a Hampel filter on a centered 3-observation window.

    Returns a boolean array that is True where |x_t - median| > 3 x 1.4826 x MAD of the window. The
    first and last value have no full window and are never flagged.
    """
    out = np.zeros(len(x), bool)
    for i in range(1, len(x) - 1):
        w = x[i-1:i+2]; m = np.median(w); mad = np.median(np.abs(w - m))
        if abs(x[i] - m) > 3 * 1.4826 * mad: out[i] = True
    return out

def basin(b):
    """Apply the rules to the river observations of basin b of the store.

    Returns (basin, list of (basin, subbasin, date) to drop, counts per rule, error message or None).
    """
    try: ds = xr.open_zarr(Z/b, consolidated=True).sel(date=slice(T0, T1))
    except Exception as e: return b, [], {}, str(e)
    st = dict(n=0, matched=0, xovr=0, afrac=0, ice=0, prior=0, hampel=0)
    if not ("dv_river" in ds and "xtrk_dist_river" in ds): return b, [], st, None
    dates = ds.date.values.astype("datetime64[D]"); subs = ds.subbasin.values.astype(str)
    R = lambda v: ds[v].transpose("date", "subbasin").values
    DV, W, D, XT, AD, AT, ICE = R("dv_river"), R("wse_river"), R("d_wse_river"), R("xtrk_dist_river"), R("area_detct_river"), R("area_total_river"), R("ice_clim_f_river")
    ii, jj = np.nonzero(np.isfinite(DV))
    if not len(ii): return b, [], st, None
    rec = pd.DataFrame(dict(i=ii, j=jj, date=dates[ii], wse=W[ii, jj], xtrk_dist=XT[ii, jj]))
    rec = rec.merge(SP, on=["date", "wse", "xtrk_dist"], how="left")
    st["n"] = len(rec); st["matched"] = int(rec.xovr_cal_q.notna().sum())
    af = AD[rec.i, rec.j] / AT[rec.i, rec.j]
    c_x = (rec.xovr_cal_q > 1).values                                      # NaN (unmatched) -> False -> kept
    c_a = ~(np.nan_to_num(af, nan=0.0) > 0.5)
    c_i = np.nan_to_num(ICE[rec.i, rec.j], nan=0) > 1
    c_p = (np.abs(rec.wse - rec.p_wse) >= 10).values
    for k, c in (("xovr", c_x), ("afrac", c_a), ("ice", c_i), ("prior", c_p)): st[k] = int(c.sum())
    rec["drop"] = c_x | c_a | c_i | c_p
    # Hampel on the survivors, per node, in date order
    for j, g in rec[~rec["drop"]].sort_values("i").groupby("j"):
        if len(g) < 3: continue
        h = hampel3(D[g.i.values, j]); st["hampel"] += int(h.sum()); rec.loc[g.index[h], "drop"] = True
    d = rec[rec["drop"]]
    return b, [(b, subs[j], dates[i]) for i, j in zip(d.i, d.j)], st, None

if __name__ == "__main__":
    basins = sorted(p.name for p in Z.iterdir() if (p/"zarr.json").is_file())
    rows, S = [], []
    with ProcessPoolExecutor(12) as ex:
        for b, r, st, err in ex.map(basin, basins):
            if err: print(b, err, flush=True); continue
            rows += r; S.append(st)
    S = pd.DataFrame(S).sum(); n = S.n
    X = pd.DataFrame(rows, columns=["basin", "subbasin", "date"]); X["date"] = pd.to_datetime(X.date)
    X.to_parquet(OUT/"drop_river_paper_min.parquet", index=False)
    print(f"river records {int(n):,} | matched to RiverSP {S.matched/n:.2%}")
    print(f"criterion alone: xovr_cal_q>1 {S.xovr/n:.1%} | area frac<=0.5 {S.afrac/n:.1%} | ice_clim_f>1 {S.ice/n:.1%} | |wse-p_wse|>=10 m {S.prior/n:.1%} | "
          f"Hampel on survivors {S.hampel/n:.1%}")
    print(f"total dropped {len(X):,} ({len(X)/n:.1%})  [v9qc river list drops 33.0% for comparison]")
