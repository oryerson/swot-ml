"""Extend the river drop list: 34 gauge nodes that carry a much larger river's SWOT reach get no target.

The mask is called "T9" in file names. It was built on the node set of the template store (the training store
before graph v3.5); make_t9_mask_v35ra.py in d_lists_and_masks/ is the same step on graph v3.5.

Usage: python make_t9_mask.py      (no arguments)

Reads (under $SWOTGAT_WORK)
  CONUS_storage_fork/eval/swotqc/duplicate_river_targets.csv   gauge nodes with a duplicated series (dup_targets.py)
  CONUS/basin_lists/swotqc/drop_river_paper_min_present.parquet   the river quality drop list; not modified.
                                                               river_build_v2.py writes a file of this name to
                                                               CONUS_storage_fork/eval/swotext/; it is expected
                                                               here
  CONUS/zarr_store_vd2026/<basin>                              store whose dv_river gives the days to drop; its
                                                               dv_river is bitwise equal to that of
                                                               zarr_store_vd2026_halfexact
  CONUS/metadata/subbasins_conus_v2.parquet                    basin of each node (graph version 2 table)

Writes
  CONUS/basin_lists/swotqc/drop_river_paper_min_present_t9.parquet   (basin, subbasin, date); the script stops if
                                                                     the file exists
  t9_nodes.csv next to this script                                   the 34 nodes with their twins and area ratio

Method
  A gauge node is selected if its SWOT series is identical to another node's and its upstream area is less than
  half of that node's (up_self / up_twin < 0.5). The script asserts that this selects 34 nodes.
  New drop list = the quality drop list, plus every day with a finite dv_river at the 34 nodes. The drop list is
  applied when the model loads the data, so these nodes keep their forcing but have no river target.

Notes
  The training cache is keyed by a hash that includes the content of the drop list (sha256), so a new drop list
  gets its own cache.
"""
from swotgat.paths import WORK
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

C = Path(f"{WORK}/CONUS"); F = Path(f"{WORK}/CONUS_storage_fork")
V = Path(__file__).resolve().parent; OUT = C/"basin_lists/swotqc/drop_river_paper_min_present_t9.parquet"
assert not OUT.exists(), OUT
D = pd.read_csv(F/"eval/swotqc/duplicate_river_targets.csv", dtype={"node": str}); D["ratio"] = D.up_self/D.up_twin
T9 = D[D.ratio < 0.5].copy(); assert len(T9) == 34, len(T9)
subs = pd.read_parquet(C/"metadata/subbasins_conus_v2.parquet", columns=["outlet_id"]); subs.index = subs.index.astype(str)
T9["basin"] = subs.outlet_id.reindex(T9.node).values; assert T9.basin.notna().all()
base = pd.read_parquet(C/"basin_lists/swotqc/drop_river_paper_min_present.parquet")
rows = []
for b, g in T9.groupby("basin"):
    z = xr.open_zarr(C/"zarr_store_vd2026"/b, consolidated=True); z = z.assign_coords(subbasin=z.subbasin.values.astype(str))
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
T9[["node", "basin", "site", "n_swot", "n_days_store", "twins", "up_self", "up_twin", "ratio", "in_train"]].to_csv(V/"t9_nodes.csv", index=False)
print(f"T9: 34 nodes, {len(add):,} dv_river days ({int(T9.n_days_store.sum())} in store; csv n_swot {int(T9.n_swot.sum())}); "
      f"already in paper list {len(base) + len(add) - len(M):,}; base {len(base):,} -> {len(M):,} rows -> {OUT}")
