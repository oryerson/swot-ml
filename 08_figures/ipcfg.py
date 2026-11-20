"""Run settings shared by the scoring and figure scripts, read from environment variables.

For the paper's run, 07_evaluation/c_scoring/env.sh sets the variables. Each one fills a module constant:

  IP_RUNS      RUNS      comma-separated names of the prediction runs (the two seeds, S1 and S2). The scripts
                         average the runs' predictions per node and day.
  IP_STEPS     STEP      comma-separated `run:step` pairs: the checkpoint tag of each run. If it is not set,
                         every run uses step_050000.
  IP_HIND      HIND      comma-separated tags of the prediction files that cover the hindcast window. A
                         prediction file is predictions/<run>/<step>_<tag>.parquet.
  IP_SERIES    SERIES    path of the daily series file (.npz) of the SWOT Period scorer (see ipx.py).
  IP_NOTE_TEX  NOTE_TEX  optional text; none of the paper's figure scripts uses it.

Notes: the defaults (the run pair named `v10pm`, the tags x1014 and c2, the file series_g.npz) belong to an
earlier model configuration. A run that does not source env.sh would read that configuration's files, so
always source env.sh first. HERE is the folder of this file.
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
