"""Check that the scorers in full_record/ reproduce the scores of overlap50/ for every shared model.

The full_record scorers add the rule-model ensemble (RULE-ENS) and the 1981-2020 window. This check
proves that adding them changed no other row.

Usage: bin/run 07_evaluation/c_scoring/full_record/check_score.py      (no arguments)
  Run it after full.py and full_record/score_x26span.py (ENS=1, BASE=0) and after the three
  overlap50 scorers (BASE=1).

Reads (paths relative to $SWOTGAT_WORK/CONUS_storage_fork/eval)
  grl_figs/intermediate/c5_o50_20261113/score/score_full/fullrecord_site_window.csv   (full.py)
  grl_figs/intermediate/c5_o50_20261113/score/sites_2325span_g.csv   (full_record/score_x26span.py)
  overlap50/score/score_j/fullrecord_site_window.csv                 (overlap50/decade.py)
  overlap50/score/sites_2325span_g.csv                               (overlap50/score_x26span.py)

Checks
  1. Window 1981-1990, models SWOT-GNN (the label for SWOT-GAT), GDROM, ISTARF, MODROM-cal and
     normal_year: the rows of the two full-record tables are equal.
  2. SWOT Period: the full_record table without its RULE-ENS rows equals the overlap50 table
     without its baseline rows (SWOT-cycle, normal_year).
  "Equal" means the same rows and columns, every numeric column identical with the same pattern of
  missing values, and every text column identical.
  The script also counts the High- and Medium-tier rows of RULE-ENS (1981-1990, 1981-2020, SWOT
  Period) and of SWOT-GAT (1981-2020).

Writes grl_figs/intermediate/c5_o50_20261113/score/check_score.json and prints
"SCORE GATE PASSED" or "SCORE GATE FAILED". The exit status is 1 on failure.
"""
from swotgat.paths import WORK
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
B = Path(f"{WORK}/CONUS_storage_fork/eval/grl_figs/intermediate/c5_o50_20261113"); O = Path(f"{WORK}/CONUS_storage_fork/eval/overlap50/score")
J = {}; ok = True
def cmp(a, b, keys):
    """Compare two tables after sorting both by the key columns.

    Return a dict with ok (True when the tables are equal), the number of rows and, per column, the
    largest absolute numeric difference, the number of mismatches of missing values and the number
    of mismatches of text (only columns that differ are listed). When the columns, the number of
    rows or the key values differ, the dict holds only ok = False, the two row counts and whether
    the columns agree.
    """
    a = a.sort_values(keys).reset_index(drop=True); b = b.sort_values(keys).reset_index(drop=True)
    if list(a.columns) != list(b.columns) or len(a) != len(b) or not (a[keys].values == b[keys].values).all(): return dict(ok=False, rows=(len(a), len(b)), cols_equal=list(a.columns) == list(b.columns))
    num = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c]) and c not in keys]; oth = [c for c in a.columns if c not in num and c not in keys]
    d = {c: float(np.nanmax(np.abs(a[c].astype(float) - b[c].astype(float)))) if b[c].notna().any() else 0.0 for c in num}
    nan = {c: int((a[c].isna() != b[c].isna()).sum()) for c in num}; txt = {c: int((a[c].astype(str) != b[c].astype(str)).sum()) for c in oth}
    good = all(v == 0 for v in d.values()) and not any(nan.values()) and not any(txt.values())
    return dict(ok=good, rows=len(a), max_abs_diff={k: v for k, v in d.items() if v}, nan_diff={k: v for k, v in nan.items() if v}, text_diff={k: v for k, v in txt.items() if v})
F_ = pd.read_csv(B/"score/score_full/fullrecord_site_window.csv"); R = pd.read_csv(O/"score_j/fullrecord_site_window.csv")
M = ["SWOT-GNN", "GDROM", "ISTARF", "MODROM-cal", "normal_year"]
a = F_[(F_.window == "1981-1990") & F_.model.isin(M)]; b = R[(R.window == "1981-1990") & R.model.isin(M)]
J["decade_1981_1990"] = cmp(a, b, ["DAM_ID", "window", "model"]); ok &= J["decade_1981_1990"]["ok"]
X = pd.read_csv(B/"score/sites_2325span_g.csv"); Y = pd.read_csv(O/"sites_2325span_g.csv")
J["swot_period"] = cmp(X[X.MODEL != "RULE-ENS"], Y[~Y.MODEL.isin(["SWOT-cycle", "normal_year"])], ["MODEL", "DAM_ID"]); ok &= J["swot_period"]["ok"]
HM = ["High", "Medium"]
J["rows"] = dict(full=F_.groupby(["window", "model"]).size().astype(int).to_dict().__repr__(),
                 ens_1981_1990_hm=int(((F_.window == "1981-1990") & (F_.model == "RULE-ENS") & F_.site_tier.isin(HM)).sum()),
                 ens_1981_2020_hm=int(((F_.window == "1981-2020") & (F_.model == "RULE-ENS") & F_.site_tier.isin(HM)).sum()),
                 gat_1981_2020_hm=int(((F_.window == "1981-2020") & (F_.model == "SWOT-GNN") & F_.site_tier.isin(HM)).sum()),
                 ens_swot_hm=int(((X.MODEL == "RULE-ENS") & X.tier.isin(HM) & X.nrmse.notna()).sum()))
J["ok"] = bool(ok); json.dump(J, open(B/"score/check_score.json", "w"), indent=1, default=str); print(json.dumps(J, indent=1, default=str))
print("SCORE GATE", "PASSED" if ok else "FAILED"); sys.exit(0 if ok else 1)
