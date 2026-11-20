"""Build the river drop list on graph v3.5, part 2 ("T9b"): add small non-gauge nodes.

The added nodes carry the SWOT reach of a gauged, larger river, so they get no river target. The result is the
river drop list of the paper's model. In the paper's run it adds 9 nodes to the 34 of part 1,
and 01_graph/install_v35.py installs the file as
$SWOTGAT_WORK/CONUS/basin_lists/swotqc/drop_river_paper_min_present_t9b_v35.parquet, the name in the model
configuration (`target_drop_masks`). This is c_template_store/make_t9b_mask.py pointed at the graph v3.5 files,
with one change of construction (see Method); lines that differ from it carry a "# v3.5" comment, and
"decided set" in those comments means the fixed set of nodes described under Method.

Usage: python make_t9b_mask_v35ra.py [SUBBASINS.parquet]
  SUBBASINS.parquet   sub-basin table that gives the upstream areas (column uparea_km2); default: the graph
                      v3.5 table, $SWOTGAT_WORK/CONUS_storage_fork/eval/graph_v3/v35/subbasins_conus_v35.parquet

Reads (L = $SWOTGAT_WORK/CONUS_storage_fork/eval/v35_inputs/lists)
  L/drop_river_paper_min_present_t9_v35ra.parquet, L/work/t9_nodes_v35ra.csv   part 1 (make_t9_mask_v35ra.py)
  L/t9b_nodes_v35.csv                the fixed set of nodes to add: the nodes of the template drop list
                                     (c_template_store/inputs/t9b_nodes.csv) re-keyed to graph v3.5 by
                                     rekey_lists_v35.py; rows without a v3.5 node are ignored
  L/train_nodes_g_v35.csv            training list (for the report)
  $SWOTGAT_WORK/CONUS_storage_fork/eval/v35_inputs/forcing/zarr_store_vd2026_halfexact_v35/<basin>
                                     the graph v3.5 training store, variable dv_river

Writes
  L/drop_river_paper_min_present_t9b_v35ra.parquet   (basin, subbasin, date); the script stops if the file exists
  L/t9b_nodes_v35ra.csv                              the added nodes, their days in the store, and whether the
                                                     scan on graph v3.5 also finds them
  L/work/t9b_scan_v35ra_nodes.csv, t9b_scan_v35ra_groups.csv   result of the scan

Method
  Scan (as in make_t9b_mask.py): in each basin of the store, nodes whose dv_river series are identical (same
  days with a value, same values) form a group. Only groups with at least one gauge node (id starting with
  "USGS-") are used. The group's owner is its gauge node with the largest upstream area. A non-gauge node of the
  group whose upstream area is less than 0.5 x the owner's is a candidate: it carries the gauge's reach.
  The change of construction: the nodes added to the drop list are the fixed set of t9b_nodes_v35.csv, not the
  candidates of the scan. The set was fixed by review on the earlier node set. A candidate outside it is printed
  and saved but not masked, because adding a node is a manual decision.
  New drop list = the part 1 list, plus every day with a finite dv_river at each node of the fixed set.
  The script asserts that every node of the fixed set is in the store and that none of them is a part 1 node.
"""
from swotgat.paths import WORK
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

C = Path(f"{WORK}/CONUS"); F = Path(f"{WORK}/CONUS_storage_fork")  # v3.5
L = F/"eval/v35_inputs/lists"; V = L/"work"  # v3.5
SUBS = Path(sys.argv[1]) if len(sys.argv) > 1 else F/"eval/graph_v3/v35/subbasins_conus_v35.parquet"  # v3.5
T9 = L/"drop_river_paper_min_present_t9_v35ra.parquet"; OUT = L/"drop_river_paper_min_present_t9b_v35ra.parquet"  # v3.5
assert not OUT.exists(), OUT
up = pd.read_parquet(SUBS, columns=["uparea_km2"]).uparea_km2; up.index = up.index.astype(str)
t9 = set(pd.read_csv(V/"t9_nodes_v35ra.csv", dtype={"node": str}).node)  # v3.5
tr = set(pd.read_csv(L/"train_nodes_g_v35.csv", dtype=str).subbasin)  # v3.5
Z = F/"eval/v35_inputs/forcing/zarr_store_vd2026_halfexact_v35"; rows, groups = [], []  # v3.5
DEC = pd.read_csv(L/"t9b_nodes_v35.csv", dtype={"node": str, "template_node": str}); DEC = DEC[DEC.node.notna() & (DEC.node != "")]  # v3.5: decided set
dec = set(DEC.node); drows = []  # v3.5
for g in sorted(p for p in Z.iterdir() if (p/"zarr.json").is_file()):
    z = xr.open_zarr(g, consolidated=True); subs = z.subbasin.values.astype(str)
    a = z.dv_river.transpose("date", "subbasin").values; fin = np.isfinite(a)
    for j in [k for k, n in enumerate(subs) if n in dec]:  # v3.5: every finite day of the decided T9b nodes
        drows.append(dict(node=subs[j], basin=g.name, dates=z.date.values[fin[:, j]], n_days=int(fin[:, j].sum())))  # v3.5
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
D_ = pd.DataFrame(drows); assert set(D_.node) == dec, f"decided T9b nodes not in the store: {sorted(dec - set(D_.node))}"  # v3.5
assert not (dec & t9), "a decided T9b node is a T9 node"  # v3.5
add = pd.concat([pd.DataFrame({"basin": r.basin, "subbasin": r.node, "date": r.dates}) for r in D_.itertuples()], ignore_index=True)  # v3.5: decided set
base = pd.read_parquet(T9)
M = pd.concat([base, add.astype({"basin": base.basin.dtype, "subbasin": base.subbasin.dtype})], ignore_index=True)
M["date"] = pd.to_datetime(M.date).astype(base.date.dtype)
M = M.drop_duplicates(["basin", "subbasin", "date"]).sort_values(["basin", "subbasin", "date"]).reset_index(drop=True)
assert list(M.columns) == list(base.columns) and M.dtypes.equals(base.dtypes)
M.to_parquet(OUT, index=False)
N["decided_t9b"] = N.node.isin(dec)  # v3.5 reporting
N.drop(columns=["dates", "already_t9"]).sort_values(["basin", "owner_gauge", "node"]).to_csv(V/"t9b_scan_v35ra_nodes.csv", index=False)  # v3.5
G.to_csv(V/"t9b_scan_v35ra_groups.csv", index=False)  # v3.5
DEC = DEC.merge(D_[["node", "n_days"]].rename(columns={"n_days": "n_days_v35_store"}), on="node", how="left")  # v3.5
DEC["detected_by_rule_v35"] = DEC.node.isin(set(N.node)); DEC.to_csv(L/"t9b_nodes_v35ra.csv", index=False)  # v3.5
newc = N[~N.node.isin(dec)]  # v3.5
print(f"T9b rule scan on v3.5: {len(N)} candidates; decided set {len(dec)} (detected by the rule {int(DEC.detected_by_rule_v35.sum())}); NEW candidates NOT added: {len(newc)}")  # v3.5
print(newc.drop(columns=["dates", "already_t9"]).to_string(index=False) if len(newc) else "  none")  # v3.5
print(N.drop(columns=["dates"]).sort_values("ratio").to_string(index=False))
print(f"T9b (decided set): {len(D_)} non-gauge nodes, {len(add):,} dv_river days "  # v3.5
      f"({len(base) + len(add) - len(M):,} already masked); {int(D_.node.isin(tr).sum())} in train_nodes_g_v35; T9 {len(base):,} -> {len(M):,} rows -> {OUT}")
