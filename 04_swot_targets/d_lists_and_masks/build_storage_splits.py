"""Write the first training, validation and test node lists from the first data store.

Design choice: train wherever there is a SWOT storage target and test against all available gauge records. There
is no spatial hold-out. In the paper's pipeline the test list (river gauges and gauged reservoirs) and the
validation list descend from the lists written here, re-keyed to graph v3.5 (rekey_lists_v35.py); the training
list is rebuilt by build_lists_v35.py.

Usage: python build_storage_splits.py      (no arguments)

Reads (under $SWOTGAT_WORK)
  CONUS/zarr_store/<basin>                           which nodes have any dv_river or dv_SWOTAE value
  CONUS_storage_fork/river_insitu_nodes.csv          nodes of river gauges (column node), if the file exists
  CONUS_storage_fork/reservoir_insitu_final.csv      nodes of gauged reservoirs (rows with km <= 0.5), if it exists
  CONUS_storage_fork/addable_reservoirs.csv          gauged reservoirs without a SWOT target (column lid), called
                                                     "unsupervised" below

Writes (to $SWOTGAT_WORK/CONUS/basin_lists/storage/)
  train_nodes.csv, valid_nodes.csv, test_nodes.csv   columns basin, subbasin

Method
  valid : a random sample of 400 nodes (seed 0) among the nodes that have a target and are not gauge nodes. It
          exists only to give checkpoint selection an honest signal. Gauge nodes are excluded so that the
          evaluation is never scored on a node used for early stopping.
  train : every node of the store except the validation nodes, whether or not the node has a target.
  test  : every gauge node (the three files above). The test list overlaps the training list on purpose for
          gauged river reaches. The unsupervised reservoirs (98 when the lists were made) carry no target at all.
"""
from swotgat.paths import WORK
import os, warnings
from pathlib import Path
import numpy as np, pandas as pd, zarr
warnings.filterwarnings("ignore")

C = Path(f"{WORK}/CONUS")
F = Path(f"{WORK}/CONUS_storage_fork")
OUT = C / "basin_lists" / "storage"
OUT.mkdir(parents=True, exist_ok=True)
Z = C / "zarr_store"
N_VALID = 400
SEED = 0

rows, targets = [], {}
for b in sorted(os.listdir(Z)):
    if not (Z / b / "zarr.json").is_file():
        continue
    g = zarr.open_group(str(Z / b), mode="r")
    subs = [str(s) for s in g["subbasin"][:]]
    nS = len(subs)
    has = {v: np.zeros(nS, bool) for v in ("dv_river", "dv_SWOTAE")}
    for v in has:
        if not (Z / b / v).is_dir():
            continue
        for i in range(0, nS, 400):
            sl = slice(i, min(i + 400, nS))
            has[v][sl] = np.isfinite(g[v][:, sl]).sum(axis=0) > 0
    for j, s in enumerate(subs):
        rows.append((b, s))
        targets[s] = (bool(has["dv_river"][j]), bool(has["dv_SWOTAE"][j]))

allnodes = pd.DataFrame(rows, columns=["basin", "subbasin"])
allnodes["riv"] = allnodes.subbasin.map(lambda s: targets[s][0])
allnodes["res"] = allnodes.subbasin.map(lambda s: targets[s][1])
allnodes["tgt"] = allnodes.riv | allnodes.res
print(f"graph nodes {len(allnodes):,}")
print(f"  with dv_river : {int(allnodes.riv.sum()):,}")
print(f"  with dv_SWOTAE: {int(allnodes.res.sum()):,}")
print(f"  with either   : {int(allnodes.tgt.sum()):,}")

# gauge (in-situ) nodes: river gauges, gauged reservoirs (rows with km <= 0.5), gauged reservoirs without a SWOT target
gt = set()
p = F / "river_insitu_nodes.csv"
if p.is_file():
    gt |= set(pd.read_csv(p, dtype=str).node)
p = F / "reservoir_insitu_final.csv"
if p.is_file():
    d = pd.read_csv(p, dtype={"node": str})
    gt |= set(d[d.km <= 0.5].node)
unsup = set(pd.read_csv(F / "addable_reservoirs.csv", dtype={"lid": str}).lid) & set(allnodes.subbasin)
gt |= unsup
print(f"\nin-situ ground-truth nodes: {len(gt):,}  (of which unsupervised reservoirs: {len(unsup):,})")

tgt = allnodes[allnodes.tgt]
pool = tgt[~tgt.subbasin.isin(gt)]
valid = pool.sample(n=min(N_VALID, len(pool)), random_state=SEED)
train = allnodes[~allnodes.subbasin.isin(set(valid.subbasin))]
test = allnodes[allnodes.subbasin.isin(gt)]

for nm, d in [("train", train), ("valid", valid), ("test", test)]:
    d[["basin", "subbasin"]].to_csv(OUT / f"{nm}_nodes.csv", index=False)
    n_t = int(d.subbasin.map(lambda s: targets[s][0] or targets[s][1]).sum())
    print(f"{nm:6s} {len(d):6,} nodes  ({n_t:,} with a target)  -> {OUT/f'{nm}_nodes.csv'}")
ov = set(train.subbasin) & set(test.subbasin)
print(f"\ntrain/test overlap: {len(ov):,} nodes  (intended: gauged reaches trained and tested)")
print(f"test nodes with NO target (truly unsupervised): "
      f"{int((~test.subbasin.map(lambda s: targets[s][0] or targets[s][1])).sum()):,}")
print(f"train/valid overlap: {len(set(train.subbasin)&set(valid.subbasin)):,}  (must be 0)")
