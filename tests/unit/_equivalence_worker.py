"""Worker for test_release_equivalence.py: one real C5 batch through one copy of the model source.

Run in its own process, because the release copy and the original staged copy have the same
package names (config, data, models, ...).

    python _equivalence_worker.py --src DIR --ckpt RUN/checkpoints/step_050000 --out OUT.npz
                                  [--batch BATCH.npz] [--audit AUDIT.json]

Without --batch it builds a batch of two real basins from the paper's training cache
(read-only) and saves it in OUT.npz. With --batch it runs the model on that batch instead.
Everything is read-only; nothing is written except OUT.npz and AUDIT.json.

It records:
  * the model's GMM outputs for both targets, seq2seq, inference mode (no dropout), key 0;
  * the training loss of the batch (compute_loss with the checkpoint's step_kwargs);
  * the last-day outputs as the two prediction runners compute them: full runner (REMAT_LEVEL 0,
    seq2seq, last day taken) and low-memory runner (REMAT_LEVEL 1, seq2seq switched off);
  * every file opened (audit hook + wrappers of the parquet/zarr/numpy readers).
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("PYTHONHASHSEED", "0")

OPENED = []


def _audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        OPENED.append(os.fsdecode(args[0]))


sys.addaudithook(_audit)

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--batch"); ap.add_argument("--audit")
a = ap.parse_args()
sys.path.insert(0, a.src)

import numpy as np                                       # noqa: E402
import pandas as pd                                      # noqa: E402
import pyarrow.parquet as pq                             # noqa: E402
import xarray as xr                                      # noqa: E402
import zarr                                              # noqa: E402


def _wrap(mod, name):
    f = getattr(mod, name)

    def g(path, *args, **kw):
        OPENED.append(str(path)); return f(path, *args, **kw)
    setattr(mod, name, g)


for m, n in ((pd, "read_parquet"), (pd, "read_csv"), (pq, "read_table"), (pq, "ParquetFile"), (xr, "open_zarr"),
             (xr, "open_dataset"), (zarr, "open"), (zarr, "open_group"), (np, "load")):
    _wrap(m, n)

import equinox as eqx                                    # noqa: E402
import jax                                               # noqa: E402
import jax.numpy as jnp                                  # noqa: E402
import config, data, models                              # noqa: E402,E401
import models.sparse_fusion_gann as sfg                  # noqa: E402
from config import Config                                # noqa: E402
from data import CachedBasinGraphDataset, DynamicCacheManager, GraphBatch   # noqa: E402
from data.cached_basingraphdataloader import padding_collate_fn             # noqa: E402
from train.loss import compute_loss                      # noqa: E402
from evaluate.inference import _gmm_values               # noqa: E402

for mod in (config, data, models, sfg):
    assert str(Path(mod.__file__).resolve()).startswith(str(Path(a.src).resolve())), mod.__file__
assert jax.devices()[0].platform == "cpu"

ck = Path(a.ckpt)
cfg = Config.from_file(ck / "config.json")
mgr = DynamicCacheManager(cfg)
cache = mgr.cache_dir / "train"
assert (cache / "_indices" / f"sequence={cfg.sequence_length}" / "sample_list.npy").is_file(), cache
ds = CachedBasinGraphDataset(cfg, cache, "train")
res = {"cache": str(cache)}

if a.batch:
    z = np.load(a.batch, allow_pickle=False)
    dyn = {k[len("dyn_"):]: z[k] for k in z.files if k.startswith("dyn_")}
    y = {k[len("y_"):]: z[k] for k in z.files if k.startswith("y_")}
    batch = GraphBatch(dynamic=dyn, graph_edges=z["graph_edges"], static=z["static"], node_mask=z["node_mask"],
                       edge_mask=z["edge_mask"], graph_idx=z["graph_idx"], y=y)
    meta = dict(basins=list(z["basins"]), dates=list(z["dates"]))
else:
    # two small basins with a sample: deterministic choice
    sizes = ds.basin_subbasin_counts
    cand = sorted((sizes[b], b) for b in ds.basin_index_map if 25 <= sizes.get(b, 0) <= 70)
    picks = []
    for _, b in cand:
        idx = ds.basin_index_map[b]
        picks.append(idx[len(idx) // 2])
        if len(picks) == 2:
            break
    items = [ds[i] for i in picks]
    n = sum(it[3].static.shape[0] for it in items)
    _, _, _, batch = padding_collate_fn(items, target_nodes_per_batch=n + 7)
    meta = dict(basins=[it[0] for it in items], dates=[str(it[2]) for it in items])

# ---- model from the checkpoint (as run_predict_ext.py loads it)
with open(ck / "model.eqx", "rb") as f:
    _, template = models.make(cfg)
    model = eqx.tree_deserialise_leaves(f, template)
model = eqx.nn.inference_mode(model)
assert model.sparse_sources == [] and model.seq2seq
key = jax.random.PRNGKey(0)
jb = batch.to_jax()

sfg.REMAT_LEVEL = 0
out = eqx.filter_jit(lambda m, b: m(b, key))(model, jb)
for t, d in out.items():
    for q, v in d.items():
        res[f"out_{t}_{q}"] = np.asarray(v)

loss = compute_loss(model, None, jb, ds.denormalize, key, **cfg.step_kwargs.model_dump())
res["loss"] = np.asarray(loss)

# the two prediction runners, last day of the window
full_last = {t: {q: np.asarray(v[-1]) for q, v in d.items()} for t, d in out.items()}
sfg.REMAT_LEVEL = 1
low = eqx.filter_jit(lambda m, b: m(b, key))(eqx.tree_at(lambda m: m.seq2seq, model, False), jb)
sfg.REMAT_LEVEL = 0
nm = np.asarray(batch.node_mask)
for t in out:
    sc = ds.d_scale[t]
    for name, src in (("full", full_last[t]), ("lowmem", {q: np.asarray(v) for q, v in low[t].items()})):
        mean, std = _gmm_values(src, nm, sc["scale"], sc["offset"], sc["log_norm"])
        res[f"pred_{name}_{t}"] = mean; res[f"pred_{name}_{t}_std"] = std

for k, v in batch.dynamic.items():
    res[f"dyn_{k}"] = np.asarray(v)
for k, v in batch.y.items():
    res[f"y_{k}"] = np.asarray(v)
for k in ("graph_edges", "static", "node_mask", "edge_mask", "graph_idx"):
    res[k] = np.asarray(getattr(batch, k))
res["basins"] = np.array(meta["basins"]); res["dates"] = np.array(meta["dates"])
np.savez(a.out, **res)
if a.audit:
    import re
    json.dump({"opened": sorted(set(OPENED)),
               "modules": sorted(m for m in sys.modules if re.search(r"gauge|resops|nwis", m, re.I))},
              open(a.audit, "w"), indent=0)
print("ok", meta, "loss", float(loss), "nodes", int(nm.sum()))
