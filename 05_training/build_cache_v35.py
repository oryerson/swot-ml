"""Build the training cache of one or more model configurations on CPU and verify it.

The model's data loader trains from a cache: the normalized inputs and targets of every basin, written once per
configuration and keyed by a hash of the configuration (config_hash in the model source). Configurations that
differ only in the random seed share one cache. For the paper, 05_training/submit_training.sh runs this script
once with the two C5 configurations (seeds S1 and S2) before the training jobs start.

Usage: python build_cache_v35.py cfg.json [cfg.json ...] [--min-gb 30] [--workers 16]
  cfg.json   model configuration files; all must have the same cache hash
  --min-gb   free disk space required in the cache folder before a new build
  --workers  processes used to build the cache
  The model source is found through v35lib.src() (V35_SRC, or the file v35_launch/SRC written by tools/deploy.py).

Writes
  <cache_dir>/<hash>/                  the cache (cache_dir comes from the configuration)
  <cache_dir>/<hash>/V35_BUILT.json    what was built and the result of the checks; "ok": true if all passed.
                                       train.sh refuses to start without it. Exit status 1 if a check fails.

Method
  Before building
    - every configuration must give the same hash;
    - the cache must be new. If the folder <cache_dir>/<hash> exists, it must hold a V35_BUILT.json with ok=true
      for the same hash from an earlier run of this script; the run then only repeats the checks. This rules out
      training on a stale cache;
    - the cache folder may hold no other cache that this script did not build;
    - the cache is keyed by the path of the data store, not by its content. The script therefore records a
      content digest of the training store and of the long-memory inputs (v35lib.data_digests) and refuses to
      reuse a cache that was built from other content. A changed store must get a new name.
  The cache of the training subset is built with the first configuration.
  After building
    - no target (dv_river, dv_SWOTAE, dv_staticPLD, or the fallback target dv_SWOTAE__fb) has a value before
      the start of train_date_range in any basin of the cache;
    - the first training sequence is complete: from the first input date of the cache to the earliest sequence
      end there are at least sequence_length days (the cache starts input_history_days before the training
      period);
    - there is at least one training sample;
    - the store digests are the same as before the build.

Notes
  About 14 minutes on 16 CPUs for the paper's configuration.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v35lib as V   # noqa: E402
S = str(V.src()); sys.path.insert(0, S)
from config import Config                                     # noqa: E402
import data.cached_basingraphdataset as cbd                   # noqa: E402
from data.cached_basingraphdataset import DynamicCacheManager  # noqa: E402
assert S in cbd.__file__, cbd.__file__

ap = argparse.ArgumentParser(); ap.add_argument("cfgs", nargs="+"); ap.add_argument("--min-gb", type=float, default=30); ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()
cfgs = [Config.from_file(Path(p)) for p in a.cfgs]
hs = {cbd.config_hash(c) for c in cfgs}; assert len(hs) == 1, f"configs do not share one cache: {hs}"
h = hs.pop(); root = Path(cfgs[0].cache_dir)/h; marker = root/"V35_BUILT.json"
Path(cfgs[0].cache_dir).mkdir(parents=True, exist_ok=True)
# The cache is keyed by the store's path, so a build is reused only for the same store content (digest), and the cache folder
# may hold nothing this script did not build (it was empty before the first build).
foreign = [p.name for p in Path(cfgs[0].cache_dir).iterdir() if p.is_dir() and not (p/"V35_BUILT.json").exists() and p != root]
if foreign: sys.exit(f"FATAL {cfgs[0].cache_dir} holds caches this script did not build: {foreign[:5]}")
dig = V.data_digests(json.load(open(a.cfgs[0])))
prior = json.load(open(marker)) if marker.exists() else None
if root.exists() and not (prior and prior.get("ok") and prior.get("hash") == h):
    sys.exit(f"FATAL {root} exists without a valid V35_BUILT.json: refusing to train on a cache this script did not build")
if prior and prior.get("digests") != dig:
    sys.exit(f"FATAL {root} was built from other store content ({prior.get('digests')} vs now {dig}): refusing to reuse it; give the "
             f"changed store a new name (checker B F4)")
free = shutil.disk_usage(cfgs[0].cache_dir).free / 1e9
print(f"hash {h}  cache {root}  free {free:.0f} GB  src {S}  {'re-check of an existing build' if prior else 'fresh build'}", flush=True)
assert root.exists() or free >= a.min_gb, f"only {free:.0f} GB free (< {a.min_gb})"
t0 = time.time(); c = cfgs[0]
d = DynamicCacheManager(c).create_cache("train", num_workers=a.workers)
ix = d/"_indices"/f"sequence={c.sequence_length}"
sl = np.load(ix/"sample_list.npy"); lk = json.load(open(ix/"basin_lookup.json"))
t_start = np.datetime64(pd.Timestamp(c.train_date_range[0])); hist = int(getattr(c, "input_history_days", 0) or 0)
early, first = {}, []
for g in sorted(p for p in (root/"train").iterdir() if (p/"zarr.json").is_file() or (p/".zmetadata").is_file()):
    z = xr.open_zarr(g, consolidated=True); first.append(z.date.values[0])
    for v in ("dv_river", "dv_SWOTAE", "dv_staticPLD", "dv_SWOTAE__fb"):
        if v in z and np.isfinite(z[v].sel(date=slice(None, t_start - np.timedelta64(1, "D"))).values).any(): early[g.name] = v
first_input = min(first); end_min = int(min(int(t) for _, t in sl))
full = (np.datetime64(end_min, "D") - np.datetime64(first_input, "D")).astype(int) + 1 >= c.sequence_length
if V.data_digests(json.load(open(a.cfgs[0]))) != dig: sys.exit("FATAL the store content changed during the cache build")
info = dict(hash=h, cache=str(root), src=S, configs=list(a.cfgs), digests=dig, samples=int(len(sl)), basins=len(lk), first_input_date=str(first_input)[:10],
            first_end_date=str(np.datetime64(end_min, "D")), train_start=str(t_start)[:10], input_history_days=hist,
            extra_dynamic_vars=list(getattr(c, "extra_dynamic_vars", None) or []), targets_before_train_start=early, full_first_window=bool(full),
            built_s=round(time.time() - t0), size_gb=round(sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 1e9, 2),
            built_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
info["ok"] = bool(not early and full and len(sl) > 0); json.dump(info, open(marker, "w"), indent=1, default=str)
print(json.dumps(info, indent=1, default=str)); sys.exit(0 if info["ok"] else 1)
