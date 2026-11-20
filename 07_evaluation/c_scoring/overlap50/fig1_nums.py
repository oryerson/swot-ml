"""Matched reservoir sets and headline NSE numbers of three windows, from one folder of score tables.

Usage: fig1_nums.py <folder> <out.json> [--check <reference.json>]
  For the paper, with E = $SWOTGAT_WORK/CONUS_storage_fork/eval:
      <folder> = $E/overlap50/score     <out.json> = $E/overlap50/fig1_o50.json
  --check   compare n and the NSE median, 90th percentile and share below 0 with a reference file
            of the same layout (tolerance 1e-12); the exit status is 1 on any mismatch.

Reads from <folder>
  score_j/fullrecord_site_window.csv   from decade.py; the rows of window 1981-1990 (key "8190")
  sites_2010_g.csv                     from score_2010.py; 2010-2020 (key "2010-20")
  sites_2325span_g.csv                 from score_x26span.py; the SWOT Period, where each reservoir
                                       is scored over its SWOT span, July 2023 to June 2026. The
                                       table calls this PERIOD "2023-25" (key "2023-25").

Writes <out.json>. Per window: the matched 4-model set (`n` and `ids`), and per model the median,
the 90th percentile and the share below 0 of NSE, and the median nRMSE. Other scripts read the
sets from this file (analyze.py, text_nums_o50.py, picks.py, picks_d.py).

Method
  Only reservoirs of confidence tier High or Medium are used; in the 2010-2020 and SWOT Period
  tables, only rows with an nRMSE value. The matched 4-model set of a window holds the reservoirs
  that all four of SWOT-GAT (label `v10pm`, or `SWOT-GNN` in the 1981-1990 table), MODROM-cal,
  GDROM and ISTARF score. NSE is read from the scorers' column `nse`
  (= 1 - sum((S - O)^2) / sum(O^2) on mean-removed model S and gauge O); it is not recomputed.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
D, OUT = Path(sys.argv[1]), Path(sys.argv[2]); CHK = sys.argv[sys.argv.index("--check") + 1] if "--check" in sys.argv else None
M4 = ["v10pm", "MODROM-cal", "GDROM", "ISTARF"]; LAB = {"v10pm": "SWOT-GAT", "MODROM-cal": "MODROM-cal", "GDROM": "GDROM", "ISTARF": "ISTARF"}
SJ = pd.read_csv(D/"score_j/fullrecord_site_window.csv"); SJ = SJ[(SJ.window == "1981-1990") & SJ.site_tier.isin(["High", "Medium"])].copy()
assert "nse" in SJ and SJ.nse.notna().all(), "needs the direct nse column (eval/nse_direct scorers)"   # the scorers must have written `nse`
SJ["MODEL"] = SJ.model.map({"SWOT-GNN": "v10pm", "MODROM-cal": "MODROM-cal", "GDROM": "GDROM", "ISTARF": "ISTARF"}); SJ = SJ[SJ.MODEL.notna()]
S_ = pd.concat([pd.read_csv(D/"sites_2010_g.csv"), pd.read_csv(D/"sites_2325span_g.csv")], ignore_index=True)
S_ = S_[S_.tier.isin(["High", "Medium"]) & S_.nrmse.notna()].copy(); assert "nse" in S_ and S_.nse.notna().all(), "needs the direct nse column"
NUM = {}
for per, d in (("8190", SJ), ("2010-20", S_[S_.PERIOD == "2010-20"]), ("2023-25", S_[S_.PERIOD == "2023-25"])):
    ids = set.intersection(*[set(d[d.MODEL == m].DAM_ID) for m in M4]); d = d[d.DAM_ID.isin(ids)]
    NUM[per] = dict(n=len(ids), ids=sorted(int(i) for i in ids), med={}, p90={}, share_lt0={}, med_nrmse={})
    for m in M4:
        x = d[d.MODEL == m]; v = x.nse.dropna().values
        NUM[per]["med"][m] = float(np.median(v)); NUM[per]["p90"][m] = float(np.percentile(v, 90)); NUM[per]["share_lt0"][m] = float((v < 0).mean())
        NUM[per]["med_nrmse"][m] = float(np.median(x.nrmse.dropna().values))
if CHK:
    R = json.load(open(CHK)); bad = []
    for p in ("8190", "2010-20", "2023-25"):
        if R[p]["n"] != NUM[p]["n"]: bad.append((p, "n", R[p]["n"], NUM[p]["n"]))
        for k in ("med", "p90", "share_lt0"):
            for m in M4:
                if abs(R[p][k][m] - NUM[p][k][m]) > 1e-12: bad.append((p, k, m, R[p][k][m], NUM[p][k][m]))
    NUM["check"] = dict(against=str(CHK), ok=not bad, mismatches=bad[:20]); print("CHECK", "PASSED" if not bad else f"FAILED {bad[:6]}")
json.dump(NUM, open(OUT, "w"), indent=1, default=float)
for p in ("8190", "2010-20", "2023-25"):
    print(p, "n", NUM[p]["n"], {LAB[m]: (round(NUM[p]["med"][m], 3), round(NUM[p]["med_nrmse"][m], 4)) for m in M4})
sys.exit(1 if CHK and not NUM["check"]["ok"] else 0)
