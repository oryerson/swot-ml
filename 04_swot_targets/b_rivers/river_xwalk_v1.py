"""Save the node -> SWOT reach crosswalk of the first graph version ("v1"), for comparison with graph v3.5.

Usage: python river_xwalk_v1.py      (no arguments; command-line arguments are ignored)

Writes $SWOTGAT_WORK/CONUS_storage_fork/eval/v35_inputs/forcing/work/river/reach_lookup_v1.csv
(subbasin, reach_id, outlet_id; one row per node-reach pair). Nothing else is written.

Method
  Calls build_reach_lookup of merge_swot_river_conus.py with that script's default arguments, that is, on the
  sub-basin table of the first graph version ($SWOTGAT_WORK/CONUS/metadata/subbasins_conus.parquet; asserted).
  This is the crosswalk that river_build_v2.py used for the river targets of the earlier stores
  (zarr_store_vd2026 and zarr_store_vd2026_halfexact). reach_area_v35.py compares it with the graph-v3.5
  crosswalk to decide which reach areas can be carried over.
"""
from swotgat.paths import WORK
import sys; sys.argv = [sys.argv[0]]
sys.path.insert(0, f"{WORK}/CONUS/preprocess")
import geopandas as gpd
import merge_swot_river_conus as M
OUT = f"{WORK}/CONUS_storage_fork/eval/v35_inputs/forcing/work/river"
args = M.parse_args(); subs = gpd.read_parquet(args.subbasins); subs.index = subs.index.astype(str)
assert str(args.subbasins).endswith("CONUS/metadata/subbasins_conus.parquet"), args.subbasins
merged = M.build_reach_lookup(args, subs)
merged.rename_axis("subbasin").reset_index()[["subbasin", "reach_id", "outlet_id"]].to_csv(f"{OUT}/reach_lookup_v1.csv", index=False)
print("v1 lookup:", merged.index.nunique(), "subbasins,", len(merged), "pairs")
