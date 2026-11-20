"""Seasonal-cycle irregularity of each reservoir's gauge record over calendar years 2000-2020.

The definition is exactly that of irreg_8120 in pa_ops8120.py (water years 1981-2020), applied to
2000-01-01 to 2020-12-31 with calendar years. No model output is used.

Usage: bin/run 07_evaluation/c_scoring/drivers/irreg0020_g.py      (no arguments; after pa_ops8120.py)

Reads (through pa_lib.py)
  the reservoir list (ids from an earlier model configuration's score table; 482 reservoirs)
  the 1980-2020 gauge test set, copy in eval/g4_fix/testsets: truth_clean.parquet, days with the
      `use` flag set
  out/ops_8120.parquet (pa_ops8120.py)   only to print the correlation with irreg_8120

Method
  A calendar year counts when it has at least 292 gauge days. A reservoir needs at least 5 such
  years; only their days are used. The mean seasonal cycle is the day-of-year mean (day 366 folded
  into 365; at least 330 days of the year present; gaps interpolated; 31-day circular running
  mean). With the anomaly = storage minus its mean,
    irregularity = sum((anomaly - cycle anomaly)^2) / sum(anomaly^2),
  which is 1 - R2 of storage against its mean seasonal cycle.

Writes irreg_0020.parquet next to this script (columns irreg_0020 and n_yr_0020, the number of
years used). Prints the coverage and the Spearman correlation with irreg_8120.
The figure preparation (08_figures/prepare_kit.py) copies this table, but no panel of the paper
uses it.
"""
from swotgat.paths import WORK
import os, sys; sys.path.insert(0, f"{WORK}/CONUS_storage_fork/eval/perf_attrib")
import numpy as np, pandas as pd
from pathlib import Path
import pa_lib as L
OUT = Path(__file__).resolve().parent
ids = L.site_ids(); A0, A1 = "2000-01-01", "2021-01-01"
TR = pd.read_parquet(L.T3/"truth_clean.parquet"); TR["date"] = pd.to_datetime(TR.date); TR = TR[TR.use.fillna(False).astype(bool) & TR.DAM_ID.isin(ids)]
def cycle(s):
    """Return the mean seasonal cycle of a daily series, indexed by day of year 1-365.

    The cycle is the day-of-year mean (day 366 folded into 365), with gaps interpolated and a 31-day
    circular running mean applied. Return None when fewer than 330 days of the year have data.
    The function is the same as in pa_ops8120.py.
    """
    doy = np.minimum(s.index.dayofyear.values, 365); m = pd.Series(s.values).groupby(doy).mean().reindex(range(1, 366))
    if m.notna().sum() < 330: return None
    m = m.interpolate(limit_direction="both"); e = np.r_[m.values[-15:], m.values, m.values[:15]]
    return pd.Series(np.convolve(e, np.ones(31)/31, "valid"), index=range(1, 366))
F = {"irreg_0020": {}, "n_yr_0020": {}}
for d, g in TR.groupby("DAM_ID"):
    s = g.set_index("date").storage_mcm.sort_index(); a = s[(s.index >= A0) & (s.index < A1)]; yr = a.index.year; n = a.groupby(yr).size(); good = n[n >= 292].index
    F["n_yr_0020"][d] = len(good)
    if len(good) < 5: continue
    a = a[yr.isin(good)]; cy = cycle(a)
    if cy is not None:
        an = a - a.mean(); cv = cy.reindex(np.minimum(a.index.dayofyear, 365)).values; cv = cv - cv.mean()
        F["irreg_0020"][d] = float(((an.values - cv)**2).sum()/(an.values**2).sum())
out = pd.DataFrame({k: pd.Series(v) for k, v in F.items()}).reindex(ids); out.index.name = "DAM_ID"; out["n_yr_0020"] = out.n_yr_0020.fillna(0).astype(int)
out.to_parquet(OUT/"irreg_0020.parquet")
O = pd.read_parquet(L.OUT/"ops_8120.parquet").reindex(ids)
x = out.irreg_0020; print("irreg_0020: n =", int(x.notna().sum()), "of", len(ids), "| years>=292d distribution:", out.n_yr_0020.describe().round(1).to_dict())
print("no value: <5 good years", int((out.n_yr_0020 < 5).sum()), "| cycle gap", int(((out.n_yr_0020 >= 5) & x.isna()).sum()))
print("p10/p50/p90", np.round(np.nanpercentile(x, [10, 50, 90]), 3), "| tertiles", np.round(np.nanquantile(x, [0, 1/3, 2/3, 1]), 3))
print("Spearman vs irreg_8120:", round(float(x.corr(O.irreg_8120, method="spearman")), 3), "on", int((x.notna() & O.irreg_8120.notna()).sum()))
