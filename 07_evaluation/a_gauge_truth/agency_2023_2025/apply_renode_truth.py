"""Move the 2023-2025 reservoir gauge table onto the corrected graph nodes found by renode.py.

Reads (all in $SWOTGAT_WORK/CONUS_storage_fork)
  resops_node_map.csv            old node -> main-lake node, written by renode.py
  reservoir_truth_2023_2025.csv  daily gauge storage per node, written by merge_truth.py
  reservoir_truth_summary.csv    one row per reservoir, written by merge_truth.py

Writes
  The two truth tables, edited in place.

Usage
  python apply_renode_truth.py     (no arguments)

Method
  In both tables the column `node` is replaced by the main-lake node of the map and the previous
  value is kept in a new column `node_orig`. Nodes that are not in the map keep their value.
  The summary also receives `basin`, `member_nodes` (all graph nodes that belong to the
  reservoir, separated by ";", so that the parts of a split reservoir can be summed) and
  `renoded` (True where the node changed).

Notes
  Safe to run twice: a table that already has `node_orig` is left alone.
  These tables are test data only. Nothing is written into a data store used for training.
"""
from swotgat.paths import WORK
import pandas as pd
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork")
X = pd.read_csv(F/"resops_node_map.csv", dtype={"node": str, "old_node": str})
m = X.drop_duplicates("old_node").set_index("old_node")
T = pd.read_csv(F/"reservoir_truth_2023_2025.csv", dtype={"node": str})
if "node_orig" not in T:                      # skip if already applied
    T.insert(1, "node_orig", T.node); T["node"] = T.node_orig.map(m.node).fillna(T.node_orig)
    T.to_csv(F/"reservoir_truth_2023_2025.csv", index=False)
S = pd.read_csv(F/"reservoir_truth_summary.csv", dtype={"node": str})
if "node_orig" not in S:
    S.insert(1, "node_orig", S.node); S["node"] = S.node_orig.map(m.node).fillna(S.node_orig)
    S["basin"] = S.node_orig.map(m.basin); S["member_nodes"] = S.node_orig.map(m.member_nodes)
    S["renoded"] = S.node_orig.map(m.renoded).fillna(False)
    S.to_csv(F/"reservoir_truth_summary.csv", index=False)
print(f"truth rows {len(T):,}; reservoirs {S.shape[0]}; re-noded {int(S.renoded.sum())}; "
      f"split (>1 member) {(S.member_nodes.str.count(';') > 0).sum()}; duplicate nodes {S.node.duplicated().sum()}")
