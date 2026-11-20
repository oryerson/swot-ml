"""Independent spot check of the rule-model ensemble (RULE-ENS) scores for 1981-1990.

Recomputes NSE at 8 reservoirs with code that shares nothing with the scorer and prints it next to
the scorer's value. Writes no file.

Usage: bin/run 07_evaluation/c_scoring/full_record/spot_ens.py      (no arguments; after full.py)

Reads (paths relative to $SWOTGAT_WORK/CONUS_storage_fork)
  eval/grl_figs/intermediate/c5_o50_20261113/score/score_full/fullrecord_site_window.csv
  predictions/grades_<tag>/full_series.parquet for gdrom3c_g_y1980, istarf3c_g_y1980 and
      modromcal3c_g_y1980   the three rule-model runs started in 1980
  testsets/regime3_full_1980_2020/truth_clean.parquet   the 1980-2020 gauge test set
  eval/overlap50/fig1_o50.json   only to print the single rule models' median NSE for comparison

Method
  The 8 reservoirs are a fixed random sample (random_state=0) of the High- and Medium-tier
  reservoirs that have a RULE-ENS row for 1981-1990. Because the scorer wrote those rows, model
  and gauge share at least 50% of the window's days there.
  Ensemble = daily mean of the three rule models' storage on days when all three have a value.
  Gauge = days with the `use` flag set in 1981-01-01 to 1990-12-31. On the shared days both series
  have their own mean removed, and NSE = 1 - sum((model - gauge)^2) / sum(gauge^2).
  Printed per reservoir: DAM_ID, number of shared days, the recomputed NSE, the scorer's NSE and
  their absolute difference.
"""
from swotgat.paths import WORK
import json, numpy as np, pandas as pd
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork"); B = F/"eval/grl_figs/intermediate/c5_o50_20261113"; P = F/"predictions"
SF = pd.read_csv(B/"score/score_full/fullrecord_site_window.csv"); E = SF[(SF.window == "1981-1990") & (SF.model == "RULE-ENS") & SF.site_tier.isin(["High", "Medium"])].set_index("DAM_ID")
ids = list(E.sample(8, random_state=0).index)
R = {}
for m, t in (("GDROM", "gdrom3c_g_y1980"), ("ISTARF", "istarf3c_g_y1980"), ("MODROM-cal", "modromcal3c_g_y1980")):
    d = pd.read_parquet(P/f"grades_{t}"/"full_series.parquet"); d["date"] = pd.to_datetime(d.date); d = d[d.DAM_ID.isin(ids)]
    R[m] = {int(k): g.set_index("date").storage_mcm for k, g in d.groupby("DAM_ID")}
TR = pd.read_parquet(F/"testsets/regime3_full_1980_2020/truth_clean.parquet"); TR["date"] = pd.to_datetime(TR.date); TR = TR[TR.use.fillna(False).astype(bool) & TR.DAM_ID.isin(ids)]
for d in ids:
    o = TR[TR.DAM_ID == d].set_index("date").storage_mcm.loc["1981-01-01":"1990-12-31"]
    e = pd.concat([R[m][d] for m in R], axis=1).dropna().mean(axis=1)
    j = pd.concat([o.rename("o"), e.rename("p")], axis=1).dropna(); oc, pc = j.o - j.o.mean(), j.p - j.p.mean()
    nse = 1 - ((pc - oc)**2).sum()/(oc**2).sum(); print(d, len(j), round(nse, 6), round(float(E.nse[d]), 6), "diff", abs(nse - E.nse[d]))
J = json.load(open(F/"eval/overlap50/fig1_o50.json")); print("fig1_o50 8190 medians", {k: round(v, 3) for k, v in J["8190"]["med"].items()}, "2023-25", {k: round(v, 3) for k, v in J["2023-25"]["med"].items()})
