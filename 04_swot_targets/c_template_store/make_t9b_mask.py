"""Extend the T9 river drop list to small non-gauge nodes that carry the SWOT reach of a gauged river ("T9b").

The T9 list (make_t9_mask.py, 34 nodes) came from a scan of gauge nodes only. Here every node is scanned. This
script works on the template store (the training store before graph v3.5); make_t9b_mask_v35ra.py in
d_lists_and_masks/ is the same step on graph v3.5.

Usage: python make_t9b_mask.py [SUBBASINS.parquet]
  SUBBASINS.parquet   sub-basin table that gives the upstream areas (column uparea_km2);
                      default $SWOTGAT_WORK/CONUS/metadata/subbasins_conus_v21.parquet (graph version 2.1)

Reads (under $SWOTGAT_WORK/CONUS)
  zarr_store_vd2026_halfexact/<basin>                              template store, variable dv_river
  basin_lists/swotqc/drop_river_paper_min_present_t9.parquet       the T9 drop list (not modified)
  basin_lists/storage/train_nodes_g.csv                            training nodes (for the column in_train)
  t9_nodes.csv next to this script                                 the T9 nodes

Writes
  CONUS/basin_lists/swotqc/drop_river_paper_min_present_t9b.parquet   (basin, subbasin, date); the script stops
                                                                      if the file exists
  t9b_nodes.csv, t9b_groups.csv next to this script                   the added nodes; all groups found

Method
  In each basin, nodes whose dv_river series are identical (same days with a value, same values) form a group.
  Only groups with at least one gauge node (id starting with "USGS-") are used. The group's owner is its gauge
  node with the largest upstream area. Every non-gauge node of the group whose upstream area is less than 0.5 x
  the owner's (the ratio rule of T9) is taken to carry the gauge's reach and is masked.
  New drop list = the T9 list, plus every day with a finite dv_river at those nodes.
  The script asserts that it adds at least one node and that none of them is already a T9 node.
"""
from swotgat.paths import WORK
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

C = Path(f"{WORK}/CONUS"); V = Path(__file__).resolve().parent
SUBS = Path(sys.argv[1]) if len(sys.argv) > 1 else C/"metadata/subbasins_conus_v21.parquet"
T9 = C/"basin_lists/swotqc/drop_river_paper_min_present_t9.parquet"; OUT = C/"basin_lists/swotqc/drop_river_paper_min_present_t9b.parquet"
assert not OUT.exists(), OUT
up = pd.read_parquet(SUBS, columns=["uparea_km2"]).uparea_km2; up.index = up.index.astype(str)
t9 = set(pd.read_csv(V/"t9_nodes.csv", dtype={"node": str}).node)
tr = set(pd.read_csv(C/"basin_lists/storage/train_nodes_g.csv", dtype=str).subbasin)
Z = C/"zarr_store_vd2026_halfexact"; rows, groups = [], []
for g in sorted(p for p in Z.iterdir() if (p/"zarr.json").is_file()):
    z = xr.open_zarr(g, consolidated=True); subs = z.subbasin.values.astype(str)
    a = z.dv_river.transpose("date", "subbasin").values; fin = np.isfinite(a)
    sig = {}
    for j in np.flatnonzero(fin.any(axis=0)):
        k = np.flatnonzero(fin[:, j]); h = hashlib.sha1(k.tobytes() + a[k, j].tobytes()).hexdigest()
        sig.setdefault(h, []).append(j)
    for h, js in sig.items():
        if len(js) < 2: continue
        nodes = [subs[j] for j in js]; gauges = [n for n in nodes if n.startswith("USGS-")]
        if not gauges: continue
        owner = max(gauges, key=lambda n: up.get(n, -1)); oa = float(up.get(owner, np.nan))
        groups.append(dict(basin=g.name, nodes=";".join(nodes), owner_gauge=owner, owner_uparea=oa, n_days=int(fin[:, js[0]].sum())))
        for j, n in zip(js, nodes):
            if n.startswith("USGS-"): continue
            r = float(up.get(n, np.nan)) / oa
            if r < 0.5:
                rows.append(dict(node=n, basin=g.name, owner_gauge=owner, uparea=float(up.get(n, np.nan)), owner_uparea=oa, ratio=r,
                                 n_days=int(fin[:, j].sum()), in_train=n in tr, already_t9=n in t9,
                                 dates=z.date.values[fin[:, j]]))
N = pd.DataFrame(rows); G = pd.DataFrame(groups)
assert len(N) and not N.already_t9.any(), "T9b must add non-gauge nodes only"
add = pd.concat([pd.DataFrame({"basin": r.basin, "subbasin": r.node, "date": r.dates}) for r in N.itertuples()], ignore_index=True)
base = pd.read_parquet(T9)
M = pd.concat([base, add.astype({"basin": base.basin.dtype, "subbasin": base.subbasin.dtype})], ignore_index=True)
M["date"] = pd.to_datetime(M.date).astype(base.date.dtype)
M = M.drop_duplicates(["basin", "subbasin", "date"]).sort_values(["basin", "subbasin", "date"]).reset_index(drop=True)
assert list(M.columns) == list(base.columns) and M.dtypes.equals(base.dtypes)
M.to_parquet(OUT, index=False)
N.drop(columns=["dates", "already_t9"]).sort_values(["basin", "owner_gauge", "node"]).to_csv(V/"t9b_nodes.csv", index=False)
G.to_csv(V/"t9b_groups.csv", index=False)
print(N.drop(columns=["dates"]).sort_values("ratio").to_string(index=False))
print(f"T9b: {len(N)} non-gauge nodes in {N.owner_gauge.nunique()} gauge-owned reaches, {len(add):,} dv_river days "
      f"({len(base) + len(add) - len(M):,} already masked); {int(N.in_train.sum())} in train_nodes_g; T9 {len(base):,} -> {len(M):,} rows -> {OUT}")
