"""List the SWOT reach ids matched to the graph's river nodes (the list hc_pull_reach.py downloads).

Usage: python reach_list.py      (no arguments; command-line arguments are ignored)

Writes $SWOTGAT_WORK/CONUS_storage_fork/eval/swotext/reaches_graph.txt, one reach_id per line.

Method
  Calls build_reach_lookup of merge_swot_river_conus.py with that script's default arguments: the crosswalk from
  node to SWOT reach, with the gauge matches of the SWOT Validation Set taking precedence. The defaults point to
  the sub-basin table of the first graph version ($SWOTGAT_WORK/CONUS/metadata/subbasins_conus.parquet).
  merge_swot_river_conus.py is imported from $SWOTGAT_WORK/CONUS/preprocess, where it is deployed.
"""
from swotgat.paths import WORK
import sys; sys.argv = [sys.argv[0]]
sys.path.insert(0, f"{WORK}/CONUS/preprocess")
import geopandas as gpd, merge_swot_river_conus as M
args = M.parse_args(); subs = gpd.read_parquet(args.subbasins); subs.index = subs.index.astype(str)
merged = M.build_reach_lookup(args, subs)
r = sorted(merged["reach_id"].astype(float).astype(int).astype(str).unique())
open(f"{WORK}/CONUS_storage_fork/eval/swotext/reaches_graph.txt", "w").write("\n".join(r) + "\n"); print("reaches", len(r))
