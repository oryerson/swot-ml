"""Check every downloaded 2023-2025 series for physical plausibility, before the merge.

Reads (current folder, which must be $SWOTGAT_WORK/CONUS_storage_fork)
  agency_storage_2023_2025_union.csv   an earlier download of agency storage
  harvested*.csv                       outputs of harvest.py, harvest_v2.py, manual_override.py
                                       (files with "legacy_gated" in the name are skipped)
  resops_targets_full.csv              dam names
  $SWOTGAT_WORK/CONUS/attributes/static_conus_storage.parquet   capacity per node

Writes
  qa_harvest.csv   one row per file and node: n, sd, cap, ac1, flag, method

Usage
  python qa_harvest.py     (no arguments; prints the flagged series)

Method
  Each series is put on a daily axis. Flags, joined by ";":
    negative     more than 1% of the days are below zero (absolute storage only; not for series
                 whose method contains "static PLD" or "anomaly")
    ac1=...      lag-1 autocorrelation below 0.8. Storage is smooth from day to day; a flow or
                 a daily change series is not, so this catches a wrong variable.
    jump=...     median absolute day-to-day change above 5% of the series range
    flat         standard deviation below 0.1% of capacity
    range=...    range above 2 times capacity
  The last two need a known capacity.

Notes
  This is a screening report for inspection. It removes nothing. qa_truth.py applies the same
  checks to the merged table.
"""
from swotgat.paths import WORK
import pandas as pd, numpy as np, glob
from pathlib import Path
F = Path(".")
C = Path(f"{WORK}/CONUS")
cap = pd.read_parquet(C/"attributes/static_conus_storage.parquet", columns=["dam_cap_mcm"]).reset_index()
cap["subbasin"] = cap.subbasin.astype(str); cap = cap.drop_duplicates("subbasin").set_index("subbasin").dam_cap_mcm
names = pd.read_csv("resops_targets_full.csv", dtype={"node": str}).drop_duplicates("node").set_index("node").DAM_NAME
files = ["agency_storage_2023_2025_union.csv"] + sorted(glob.glob("harvested*.csv"))
files = [f for f in files if "legacy_gated" not in f]
rows = []
for f in files:
    h = pd.read_csv(f, dtype={"node": str})
    vcol = "storage_mcm" if "storage_mcm" in h else [c for c in h.columns if "stor" in c.lower()][0]
    for n, g in h.groupby("node"):
        s = g.set_index(pd.to_datetime(g.date))[vcol].astype(float).sort_index().asfreq("D")
        meth = str(g["method"].iat[0]) if "method" in g else ""
        anomaly = "static PLD" in meth or "anomaly" in meth
        k = cap.get(n, np.nan); rng = s.max() - s.min()
        ac = s.autocorr(1); jump = s.diff().abs().median() / rng if rng > 0 else np.nan
        fl = []
        if not anomaly and (s < 0).mean() > 0.01: fl.append("negative")
        if ac < 0.8: fl.append(f"ac1={ac:.2f}")
        if jump > 0.05: fl.append(f"jump={jump:.2f}")
        if np.isfinite(k) and k > 0 and s.std() < 1e-3*k: fl.append("flat")
        if np.isfinite(k) and k > 0 and rng > 2*k: fl.append(f"range={rng/k:.1f}xcap")
        rows.append(dict(file=f, node=n, dam=names.get(n, ""), n=s.notna().sum(), sd=s.std(), cap=k, ac1=ac, flag=";".join(fl), method=meth[:60]))
q = pd.DataFrame(rows); q.to_csv("qa_harvest.csv", index=False)
print(q.groupby("file").size().to_string()); nf = (q.flag != "").sum(); print(f"\nflagged {nf} / {len(q)}")
print(q[q.flag != ""][["file","dam","n","sd","cap","flag","method"]].round(3).to_string())
