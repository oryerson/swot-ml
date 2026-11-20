"""Score SWOT-GAT and the rule models against gauges in four fixed windows between 1981 and 2020.

Windows: 1981-1985, 1986-1990, 1981-1990 and 2010-2020.

Method
  This script holds no scoring code of its own. It reads the score_fullrecord.py next to it as text,
  replaces the two lines that define that scorer's window list (WIN) with the four windows above,
  and executes the result with a fixed command line. The script stops if either of the two lines is
  not found exactly once. Everything else is score_fullrecord.py unchanged: the metrics (NSE, nRMSE,
  r and VR, in the column `amp`, on series with their own mean removed) and the 50%-of-days overlap
  rule (a reservoir is scored in a window only if the gauge alone, and gauge and model together,
  have values on at least 50% of the window's days).

Usage (after `source 07_evaluation/c_scoring/env.sh`; no command-line arguments):
    [BASE=1] bin/run 07_evaluation/c_scoring/overlap50/decade.py
  IP_RUNS   the two prediction runs (seeds S1, S2), comma-separated. Their predictions are averaged
            per node and day and scored under the label `SWOT-GNN`, which means SWOT-GAT.
  IP_STEPS  each run's checkpoint tag (read by ipcfg.py, which the scorer imports).
  BASE=1    also scores the SWOT-cycle baseline (bl.py). The gauge normal year is always scored.

Reads (paths relative to $SWOTGAT_WORK/CONUS_storage_fork)
  predictions/<run>/<step>_<tag>.parquet for the tags x8085, x8691 and h10
      (1981-1985, 1986-1991 and 2010-2020)
  predictions/grades_<tag>/full_series.parquet, the rule-model runs started in 1980:
      GDROM gdrom3c_g_y1980, ISTARF istarf3c_g_y1980, MODROM-cal modromcal3c_g_y1980
  testsets/regime3_full_1980_2020/   the 1980-2020 gauge test set

Writes score_j/ next to this script: fullrecord_site_window.csv (one row per reservoir, window and
model) and the scorer's two summary tables, blocks_all.csv and blocks_balanced.csv.
The script refuses to run if score_j/ already exists; delete or rename the folder to score again.
"""
import os, sys
from pathlib import Path
D = Path(__file__).resolve().parent; sys.path.insert(0, str(D)); OUT = D/"score_j"; assert not OUT.exists(), "score_j exists already"
RA, RB = os.environ["IP_RUNS"].split(",")
src = (D/"score_fullrecord.py").read_text()
W1 = 'WIN = [(f"{y}-{y+4}", f"{y}-01-01", f"{y+4}-12-31", "block") for y in range(1981, 2017, 5)]'
W2 = 'WIN += [(str(y), f"{y}-01-01", f"{y}-12-31", "year") for y in range(1981, 2021)] + [("2015-2020", "2015-01-01", "2020-12-31", "frozen")]'
assert src.count(W1) == 1 and src.count(W2) == 1
NEW = ('WIN = [("1981-1985", "1981-01-01", "1985-12-31", "block"), ("1986-1990", "1986-01-01", "1990-12-31", "block"), '
       '("1981-1990", "1981-01-01", "1990-12-31", "block"), ("2010-2020", "2010-01-01", "2020-12-31", "block")]')
src = src.replace(W1, NEW).replace(W2, "")
sys.argv = ["score_fullrecord.py", "--out", str(OUT), "--gnn", f"SWOT-GNN={RA},{RB}", "--chunks", "x8085,x8691,h10",
            "--rule", "GDROM=gdrom3c_g_y1980", "--rule", "ISTARF=istarf3c_g_y1980", "--rule", "MODROM-cal=modromcal3c_g_y1980"]
exec(compile(src, "score_fullrecord.py (decade windows)", "exec"), {"__name__": "__main__", "__file__": str(D/"score_fullrecord.py")})
