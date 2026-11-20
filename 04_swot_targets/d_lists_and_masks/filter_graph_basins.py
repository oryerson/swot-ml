"""Remove from the node lists every basin that the graph file cannot supply.

The model's loader (_load_basin_graphs) splits the graph file into weakly connected
components and takes each component's single outlet node as the basin id. A list row
whose basin is not the outlet of a component makes the loader raise an error and
stops the run. This script keeps only rows of basins the graph supplies.

Usage: python filter_graph_basins.py      (no arguments)

Reads (under $SWOTGAT_WORK/CONUS)
  metadata/graphs_conus.json                                the graph of the first graph version
  basin_lists/storage/train_nodes_all.csv, valid_nodes_insample.csv, test_nodes_insitu.csv
                                                            the node lists (columns basin, subbasin)
  $SWOTGAT_WORK/CONUS_storage_fork/addable_reservoirs.csv   gauged reservoirs without a SWOT target, for the
                                                            report only

Writes (to basin_lists/storage/)
  train_nodes_g.csv, valid_nodes_g.csv, test_nodes_g.csv    the same lists without the dropped basins

Notes
  The outputs get new file names on purpose. The training cache is keyed by a hash that
  covers the path of the node list, not its content, so writing a new list to an old path
  would silently pair it with the cache built from the old list. Rebuilding the cache
  takes about 40 seconds.
  build_storage_splits.py writes train_nodes.csv, valid_nodes.csv and test_nodes.csv. The
  three input names read here differ from those; no script in this repository writes them.
"""
from swotgat.paths import WORK
import json, warnings
from pathlib import Path
import networkx as nx, pandas as pd
warnings.filterwarnings("ignore")
C = Path(f"{WORK}/CONUS")
O = C / "basin_lists" / "storage"

G = nx.readwrite.json_graph.node_link_graph(
    json.load(open(C / "metadata/graphs_conus.json")), edges="edges")
valid, multi = set(), 0
for nodes in nx.weakly_connected_components(G):
    sub = G.subgraph(nodes)
    outs = [n for n in sub.nodes if sub.out_degree(n) == 0]
    if len(outs) == 1: valid.add(outs[0])
    else: multi += 1
print(f"graph components: {len(valid)} usable basins, {multi} with != 1 outlet")

REN = {"train_nodes_all.csv": "train_nodes_g.csv",
       "valid_nodes_insample.csv": "valid_nodes_g.csv",
       "test_nodes_insitu.csv": "test_nodes_g.csv"}
for src, dst in REN.items():
    d = pd.read_csv(O / src, dtype=str)
    miss = sorted(set(d.basin) - valid)
    keep = d[d.basin.isin(valid)]
    keep.to_csv(O / dst, index=False)
    print(f"{src:26s} -> {dst}")
    print(f"    basins {d.basin.nunique():3d} -> {keep.basin.nunique():3d} | "
          f"nodes {len(d):6,} -> {len(keep):6,} | dropped basins: {miss}")

# report: how many gauged reservoirs without a SWOT target remain in the test list, and the size of the training list
F = Path(f"{WORK}/CONUS_storage_fork")
unsup = set(pd.read_csv(F / "addable_reservoirs.csv", dtype={"lid": str}).lid)
t = pd.read_csv(O / "test_nodes_g.csv", dtype=str)
print(f"\nunsupervised reservoirs still in test: {len(set(t.subbasin) & unsup)}")
tr = pd.read_csv(O / "train_nodes_g.csv", dtype=str)
print(f"supervised training nodes: {len(tr):,}")
