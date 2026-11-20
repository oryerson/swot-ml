"""Run the unmodified HALF v1.0 release script on the stored version-D lake download.

HALF is the lake outlier filter of Trudel et al. The pipeline filters lakes with half_batch.py, which calls HALF's
functions lake by lake. This script runs HALF's own release script (half_v1_run.py) instead, so that
compare_replay.py can test whether half_batch.py reproduces it.

Usage: python release_replay.py SHARD NSHARDS [--start ISO] [--end ISO] [--window-key range_start_time|time_str]
                                [--tag replay] [--src FILE] [--collection NAME]     (run with the HALF environment)
  SHARD NSHARDS   process every NSHARDS-th lake of the sorted lake list, starting at SHARD (for a Slurm array)
  --start, --end  window passed to HALF (defaults 2023-07-21T05:33:45Z and 2026-12-31T00:00:00Z)
  --window-key    column the window is applied to, default range_start_time (the key Hydrocron sorts by)
  --tag           name of the output folder, default "replay"
  --src           stored download, default eval/swotext/hc_all_vD.parquet (hc_pull.py)
  --collection    collection name the script expects in the queries, default SWOT_L2_HR_LakeSP_D

Reads (under $SWOTGAT_WORK/CONUS_storage_fork/eval/)
  half_vd_review/release_copy/half_v1_run.py and half_v1_functions.py   the HALF v1.0 release (md5 of the run
                                                                        script 5a2bff6f..., same as vendor/half_v1)
  swotext/hc_all_vD.parquet                                             the stored Hydrocron download

Writes (to half_vd_review/<tag>/shard_<SHARD>/)
  the output files of the release script (df_Hydrocron_HALF_v1_LakeSP_vD.csv among them)
  half_v1_run_configured.py   the release script as it was run, so that it can be compared with the original
  stdout.txt                  what the release script printed

Method
  Two things are replaced, both outside HALF's logic:
    1. The `requests` module is a stub. Its get() serves the stored Hydrocron CSV text of the queried lake
       (the stored download keeps Hydrocron's strings verbatim), limited to the queried window and to the
       requested fields that were downloaded. All columns HALF needs are present. lake_name, p_lon, p_lat,
       range_end_time, geometry and PLD_version were not downloaded and HALF does not use them.
    2. plt.savefig does nothing, so no per-lake figures are written.
  The RUN_CONFIG block of the release script is changed by text substitution only (lake_ids,
  collection_shortname, start_time, end_time, work_dir), and each substitution is asserted to match exactly
  once. FILTER_CONFIG and every call into half_v1_functions.py are untouched.
"""
from swotgat.paths import WORK
import os, sys, io, json, argparse, contextlib, warnings, urllib.parse, logging
os.environ["MPLBACKEND"] = "Agg"
import numpy as np, pandas as pd
R = f"{WORK}/CONUS_storage_fork/eval/half_vd_review"
S = f"{WORK}/CONUS_storage_fork/eval/swotext"
ap = argparse.ArgumentParser(); ap.add_argument("shard", type=int); ap.add_argument("nshards", type=int)
ap.add_argument("--start", default="2023-07-21T05:33:45Z"); ap.add_argument("--end", default="2026-12-31T00:00:00Z")
ap.add_argument("--window-key", default="range_start_time"); ap.add_argument("--tag", default="replay"); ap.add_argument("--src", default=f"{S}/hc_all_vD.parquet")
ap.add_argument("--collection", default="SWOT_L2_HR_LakeSP_D")
a = ap.parse_args()
H = pd.read_parquet(a.src); H = H[H["_error"].isna()] if "_error" in H else H
H = H.drop(columns=[c for c in H.columns if c.startswith("_")])
lakes = sorted(H.lake_id.dropna().unique()); mine = lakes[a.shard::a.nshards]
G = {l: g for l, g in H[H.lake_id.isin(mine)].groupby("lake_id", sort=False)}
out = f"{R}/{a.tag}/shard_{a.shard:02d}"; os.makedirs(out, exist_ok=True)

import types
requests = types.ModuleType("requests")                 # the HALF environment has no requests package; nothing here touches the network anyway
class RequestException(Exception):
    """Stand in for requests.RequestException in the stub module."""

    pass
class HTTPError(RequestException):
    """Stand in for requests.HTTPError in the stub module."""

    pass
requests.RequestException, requests.HTTPError = RequestException, HTTPError
sys.modules["requests"] = requests
class _Resp:
    def __init__(self, code, payload): self.status_code, self._p = code, payload
    def raise_for_status(self):
        if self.status_code != 200: raise requests.HTTPError(f"{self.status_code}")
    def json(self): return self._p
served = []
def fake_get(url, timeout=None, **kw):
    """Answer a Hydrocron time-series query from the stored download.

    The lake, the window and the requested fields are read from the query string of `url`. Returns a
    response with status 200 and the stored rows of the lake inside the window as CSV text (only the
    requested fields that were downloaded), or status 400 with Hydrocron's message if the lake is
    not in the download or has no row in the window. Each served lake is recorded in `served` with
    its number of rows.
    """
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    lake, t0, t1 = q["feature_id"][0], pd.Timestamp(q["start_time"][0]), pd.Timestamp(q["end_time"][0])
    assert q["feature"][0] == "PriorLake" and q["collection_name"][0] == a.collection, url
    fields = q["fields"][0].split(",")
    g = G.get(lake)
    if g is None: return _Resp(400, {"error": "400: Results with the specified Feature ID were not found."})
    t = pd.to_datetime(g[a.window_key], errors="coerce", utc=True); g = g[(t >= t0) & (t <= t1)]
    if g.empty: return _Resp(400, {"error": "400: No results"})
    cols = [f for f in fields if f in g.columns]; served.append((lake, len(g)))
    return _Resp(200, {"status": "200 OK", "results": {"csv": g[cols].to_csv(index=False), "geojson": {}}})
requests.get = fake_get
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.savefig = lambda *x, **k: None
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

src = open(f"{R}/release_copy/half_v1_run.py").read()
subs = [('"lake_ids": [7240054132, 2510236842, 4340980733],', f'"lake_ids": {json.dumps([int(l) for l in mine])},'),
        ('"collection_shortname": "SWOT_L2_HR_LakeSP_2.0",', f'"collection_shortname": "{a.collection}",'),
        ('"start_time": "2023-07-21T05:33:45Z",', f'"start_time": "{a.start}",'),
        ('"end_time": "2025-05-18T21:36:19Z",', f'"end_time": "{a.end}",'),
        ('"work_dir": ".", ', f'"work_dir": "{out}", ')]
for old, new in subs:
    assert src.count(old) == 1, old; src = src.replace(old, new)
open(f"{out}/half_v1_run_configured.py", "w").write(src)
sys.path.insert(0, f"{R}/release_copy")
with open(f"{out}/stdout.txt", "w") as f, contextlib.redirect_stdout(f), warnings.catch_warnings():
    warnings.simplefilter("ignore")
    exec(compile(src, f"{out}/half_v1_run_configured.py", "exec"), {"__name__": "__main__"})
print(f"shard {a.shard}/{a.nshards}: {len(mine)} lakes asked, {len(served)} served, {sum(n for _, n in served):,} rows", flush=True)
