"""Score SWOT-GAT, the rule models and their ensemble against gauges over 1981-1990 and 1981-2020.

Method
  This script holds no scoring code of its own. It reads the score_fullrecord.py next to it as text,
  replaces the two lines that define that scorer's window list (WIN) with the windows 1981-1990 and
  1981-2020, and executes the result with a fixed command line. The script stops if either of the
  two lines is not found exactly once. Everything else is score_fullrecord.py unchanged: the
  metrics (NSE, nRMSE, r and VR, in the column `amp`, on series with their own mean removed) and
  the 50%-of-days overlap rule (a reservoir is scored in a window only if the gauge alone, and
  gauge and model together, have values on at least 50% of the window's days).
  The scorer in this folder is the one of overlap50/ plus the rule-model ensemble.

Usage (after `source 07_evaluation/c_scoring/env.sh`; no command-line arguments):
    ENS=1 bin/run 07_evaluation/c_scoring/full_record/full.py
  IP_RUNS   the two prediction runs (seeds S1, S2), comma-separated. Their predictions are averaged
            per node and day and scored under the label `SWOT-GNN`, which means SWOT-GAT.
  IP_STEPS  each run's checkpoint tag (read by ipcfg.py, which the scorer imports).
  ENS=1     adds the model RULE-ENS: the daily mean of GDROM, ISTARF and MODROM-cal storage on days
            when all three have a value, at reservoirs that have all three.
  BASE=1    also scores the SWOT-cycle baseline (bl.py). The paper's run of this script used BASE=0.

Reads (paths relative to $SWOTGAT_WORK/CONUS_storage_fork)
  predictions/<run>/<step>_<tag>.parquet for the six tags x8085, x8691, x9297, x9803, x0409 and
      h10, which together cover 1981-2020
  predictions/grades_<tag>/full_series.parquet, the rule-model runs started in 1980:
      GDROM gdrom3c_g_y1980, ISTARF istarf3c_g_y1980, MODROM-cal modromcal3c_g_y1980
  testsets/regime3_full_1980_2020/   the 1980-2020 gauge test set

Writes score_full/ next to this script: fullrecord_site_window.csv (one row per reservoir, window
and model) and the scorer's two summary tables, blocks_all.csv and blocks_balanced.csv.
The script refuses to run if score_full/ already exists; delete or rename the folder to score again.
"""
import os, sys
from pathlib import Path
D = Path(__file__).resolve().parent; sys.path.insert(0, str(D)); OUT = D/"score_full"; assert not OUT.exists(), "score_full exists already"
RA, RB = os.environ["IP_RUNS"].split(",")
src = (D/"score_fullrecord.py").read_text()
W1 = 'WIN = [(f"{y}-{y+4}", f"{y}-01-01", f"{y+4}-12-31", "block") for y in range(1981, 2017, 5)]'
W2 = 'WIN += [(str(y), f"{y}-01-01", f"{y}-12-31", "year") for y in range(1981, 2021)] + [("2015-2020", "2015-01-01", "2020-12-31", "frozen")]'
assert src.count(W1) == 1 and src.count(W2) == 1
src = src.replace(W1, 'WIN = [("1981-1990", "1981-01-01", "1990-12-31", "block"), ("1981-2020", "1981-01-01", "2020-12-31", "block")]').replace(W2, "")
sys.argv = ["score_fullrecord.py", "--out", str(OUT), "--gnn", f"SWOT-GNN={RA},{RB}", "--chunks", "x8085,x8691,x9297,x9803,x0409,h10",
            "--rule", "GDROM=gdrom3c_g_y1980", "--rule", "ISTARF=istarf3c_g_y1980", "--rule", "MODROM-cal=modromcal3c_g_y1980"]
exec(compile(src, "score_fullrecord.py (c5_o50 windows)", "exec"), {"__name__": "__main__", "__file__": str(D/"score_fullrecord.py")})
