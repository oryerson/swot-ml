"""Remove from the re-keyed validation and test lists every basin that the graph v3.5 file cannot supply.

This is filter_graph_basins.py pointed at graph v3.5. The model's loader
(_load_basin_graphs) splits the graph file into weakly connected components and takes
each component's single outlet node as the basin id. A list row whose basin is not the
outlet of a component makes the loader raise an error and stops the run. This script
keeps only rows of basins the graph supplies.

Usage: python filter_graph_basins_v35.py      (no arguments)

Reads (under $SWOTGAT_WORK/CONUS_storage_fork)
  eval/graph_v3/v35/graphs_conus_v35.json                              graph v3.5
  eval/v35_inputs/lists/work/valid_rekeyed.csv, test_rekeyed.csv       lists re-keyed by rekey_lists_v35.py
  addable_reservoirs.csv                                               gauged reservoirs without a SWOT target,
                                                                       for the report only

Writes (to eval/v35_inputs/lists/)
  valid_nodes_g_v35.csv, test_nodes_g_v35.csv    the script refuses to overwrite them

Notes
  The training list is not handled here. build_lists_v35.py builds it by rule.
  The outputs get new file names on purpose. The training cache is keyed by a hash that
  covers the paths of the node lists, not their content, so writing a new list to an old
  path would silently pair it with the cache built from the old list.
"""
from swotgat.paths import WORK
import json, warnings
from pathlib import Path
import networkx as nx, pandas as pd
warnings.filterwarnings("ignore")
C = Path(f"{WORK}/CONUS")
CSF = Path(f"{WORK}/CONUS_storage_fork")
I = CSF / "eval/v35_inputs/lists/work"; O = CSF / "eval/v35_inputs/lists"   # inputs are read from I, outputs go to O

G = nx.readwrite.json_graph.node_link_graph(
    json.load(open(CSF / "eval/graph_v3/v35/graphs_conus_v35.json")), edges="edges")
valid, multi = set(), 0
for nodes in nx.weakly_connected_components(G):
    sub = G.subgraph(nodes)
    outs = [n for n in sub.nodes if sub.out_degree(n) == 0]
    if len(outs) == 1: valid.add(outs[0])
    else: multi += 1
print(f"graph components: {len(valid)} usable basins, {multi} with != 1 outlet")

REN = {"valid_rekeyed.csv": "valid_nodes_g_v35.csv",
       "test_rekeyed.csv": "test_nodes_g_v35.csv"}
assert not any((O / dst).exists() for dst in REN.values()), "refusing to overwrite"
for src, dst in REN.items():
    d = pd.read_csv(I / src, dtype=str)
    miss = sorted(set(d.basin) - valid)
    keep = d[d.basin.isin(valid)]
    keep.to_csv(O / dst, index=False)
    print(f"{src:26s} -> {dst}")
    print(f"    basins {d.basin.nunique():3d} -> {keep.basin.nunique():3d} | "
          f"nodes {len(d):6,} -> {len(keep):6,} | dropped basins: {miss}")

# report: how many gauged reservoirs without a SWOT target remain in the test list, and the size of the validation list
F = Path(f"{WORK}/CONUS_storage_fork")
unsup = set(pd.read_csv(F / "addable_reservoirs.csv", dtype={"lid": str}).lid)
t = pd.read_csv(O / "test_nodes_g_v35.csv", dtype=str)
print(f"\nunsupervised reservoirs still in test: {len(set(t.subbasin) & unsup)}")
va = pd.read_csv(O / "valid_nodes_g_v35.csv", dtype=str)
print(f"valid nodes: {len(va):,}")
