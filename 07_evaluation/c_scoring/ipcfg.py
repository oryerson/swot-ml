"""Run settings shared by the scoring scripts, read from environment variables.

Set them with `source 07_evaluation/c_scoring/env.sh`, which exports the paper's values.

  IP_RUNS      -> RUNS    Comma-separated names of the prediction runs to average per node and day
                          (the paper: the two seeds, S1 and S2).
  IP_STEPS     -> STEP    Comma-separated `run:step` pairs: the checkpoint tag of each run. The
                          prediction files are predictions/<run>/<step>_<window tag>.parquet.
                          Default: step_050000 for every run.
  IP_HIND      -> HIND    Comma-separated window tags of the prediction files that cover 2010-2020
                          (the paper: h10).
  IP_SERIES    -> SERIES  Path of the series file that ipx.py reads (written by ip_series.py).
  IP_NOTE_TEX  -> NOTE_TEX  Free text; no scoring script in this folder uses it.

Note: the defaults in the code (the two `v10pm` runs, windows `x1014,c2`, `series_g.npz`) belong to
an earlier model configuration, not to the paper's model. Always set the variables explicitly.

Copies of this file with the same code sit next to the scripts that import it.
"""
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
V10 = "era5_conus_v10pm_S1_storage_20260930_143855,era5_conus_v10pm_S2_storage_s1_20260930_143620"
RUNS = tuple(os.environ.get("IP_RUNS", V10).split(","))
STEP = dict(x.split(":") for x in os.environ["IP_STEPS"].split(",")) if os.environ.get("IP_STEPS") else {r: "step_050000" for r in RUNS}
HIND = os.environ.get("IP_HIND", "x1014,c2").split(",")
SERIES = os.environ.get("IP_SERIES", "series_g.npz")
NOTE_TEX = os.environ.get("IP_NOTE_TEX", "")
