"""Build the river drop list on graph v3.5, part 1 ("T9"): quality drops plus 34 gauge nodes.

The 34 gauge nodes carry a SWOT river series that is a copy of a much larger river's reach, so they get no
river target. The drop list is rebuilt on graph v3.5 rather than re-keyed from the earlier node set, so that it
matches the river record of the v3.5 store. This is c_template_store/make_t9_mask.py pointed at the graph v3.5
files; lines that differ from it end in "# v3.5".

Usage: python make_t9_mask_v35ra.py      (no arguments)

Reads (under $SWOTGAT_WORK/CONUS_storage_fork/eval)
  swotqc/duplicate_river_targets.csv                       gauge nodes with a duplicated series (dup_targets.py)
  v35_inputs/lists/t9_nodes_v35.csv                        the 34 nodes re-keyed to graph v3.5 (rekey_lists_v35.py)
  v35_inputs/forcing/work/river/drop_river_paper_min_present_v35ra.parquet
                                                           river quality drop list built with the graph v3.5
                                                           reach areas (river_build_v35ra.py); not modified
  v35_inputs/forcing/zarr_store_vd2026_halfexact_v35/<basin>   the graph v3.5 training store, variable dv_river
  graph_v3/v35/subbasins_conus_v35.parquet                 basin of each node

Writes (to v35_inputs/lists/)
  drop_river_paper_min_present_t9_v35ra.parquet   (basin, subbasin, date); the script stops if the file exists
  work/t9_nodes_v35ra.csv                         the 34 nodes with their twins, area ratio and days in the store

Method
  A gauge node is selected if its SWOT series is identical to another node's and its upstream area is less than
  half of that node's (up_self / up_twin < 0.5). The script asserts that this selects 34 nodes and that they are
  the nodes of t9_nodes_v35.csv (all 34 are gauges, whose ids did not change).
  New drop list = the quality drop list, plus every day with a finite dv_river at the 34 nodes in the store. The
  drop list is applied when the model loads the data.

Notes
  The training cache is keyed by a hash that includes the content of the drop list (sha256), so a new drop list
  gets its own cache. make_t9b_mask_v35ra.py adds the second part.
"""
from swotgat.paths import WORK
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

C = Path(f"{WORK}/CONUS"); F = Path(f"{WORK}/CONUS_storage_fork")
L = F/"eval/v35_inputs/lists"; V = L/"work"; OUT = L/"drop_river_paper_min_present_t9_v35ra.parquet"  # v3.5
assert not OUT.exists(), OUT
D = pd.read_csv(F/"eval/swotqc/duplicate_river_targets.csv", dtype={"node": str}); D["ratio"] = D.up_self/D.up_twin
T9 = D[D.ratio < 0.5].copy(); assert len(T9) == 34, len(T9)
assert set(T9.node) == set(pd.read_csv(L/"t9_nodes_v35.csv", dtype={"node": str}).node), "T9 != t9_nodes_v35.csv"  # v3.5
subs = pd.read_parquet(F/"eval/graph_v3/v35/subbasins_conus_v35.parquet", columns=["outlet_id"]); subs.index = subs.index.astype(str)  # v3.5
T9["basin"] = subs.outlet_id.reindex(T9.node).values; assert T9.basin.notna().all()
base = pd.read_parquet(F/"eval/v35_inputs/forcing/work/river/drop_river_paper_min_present_v35ra.parquet")  # v3.5
rows = []
for b, g in T9.groupby("basin"):
    z = xr.open_zarr(F/"eval/v35_inputs/forcing/zarr_store_vd2026_halfexact_v35"/b, consolidated=True); z = z.assign_coords(subbasin=z.subbasin.values.astype(str))  # v3.5
    for n in g.node:
        s = z.dv_river.sel(subbasin=n).to_series(); d = s.index[np.isfinite(s.values)]
        rows.append(pd.DataFrame({"basin": b, "subbasin": n, "date": d}))
        T9.loc[T9.node == n, "n_days_store"] = len(d)
add = pd.concat(rows, ignore_index=True)
M = pd.concat([base, add.astype({"basin": base.basin.dtype, "subbasin": base.subbasin.dtype})], ignore_index=True)
M["date"] = pd.to_datetime(M.date).astype(base.date.dtype)
M = M.drop_duplicates(["basin", "subbasin", "date"]).sort_values(["basin", "subbasin", "date"]).reset_index(drop=True)
assert list(M.columns) == list(base.columns) and M.dtypes.equals(base.dtypes)
M.to_parquet(OUT, index=False)
T9[["node", "basin", "site", "n_swot", "n_days_store", "twins", "up_self", "up_twin", "ratio", "in_train"]].to_csv(V/"t9_nodes_v35ra.csv", index=False)  # v3.5
print(f"T9: 34 nodes, {len(add):,} dv_river days ({int(T9.n_days_store.sum())} in store; csv n_swot {int(T9.n_swot.sum())}); "
      f"already in paper list {len(base) + len(add) - len(M):,}; base {len(base):,} -> {len(M):,} rows -> {OUT}")
