"""Find, for each ResOpsUS reservoir, the graph node that is the main body of its lake.

The first match of dams to graph nodes took the lake of the SWOT Prior Lake Database (PLD)
nearest to the dam point. That lake is often a small pool at the dam (Kentucky: a 0.52 km2
node, while the main lake polygon has 498 km2). This script proposes a better node.

Reads ($SWOTGAT_WORK/CONUS_storage_fork unless noted)
  resops_node_match.csv     the first dam-to-node match (grand_id, node, basin)
  resops_targets_full.csv   dam names
  $SWOTGAT_WORK/CONUS/attributes/static_conus_storage.parquet   graph nodes with lake area
  $SWOTGAT_WORK/CONUS/basin_lists/storage/train_nodes_g.csv     nodes that have a SWOT target
  $SWOTGAT_NAS_WATER/cjgleason/ted/GDW/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp
                            Global Dam Watch (GDW) reservoir polygons
  $SWOTGAT_NAS_WATER/cjgleason/ted/LakeSPvC_ts_for_Ted/SWOT_PLD_v106_reservoirs.gdb
                            PLD lake polygons

Writes
  resops_node_map.csv   one row per dam: grand_id, dam, old_node, old_basin, node, basin,
                        renoded, areas (old, new, GDW), member_nodes (";"-separated), n_members,
                        members_area, primary_share, new_supervised, old_supervised, and the
                        ratios of node area to GDW area

Usage
  python renode.py     (no arguments). apply_renode_truth.py then applies the map.

Method
  * Candidates for a dam are the PLD polygons whose `res_id` equals the dam's GRanD id, plus
    the PLD polygons that lie at least 20% inside the GDW reservoir polygon (areas computed in
    the equal-area projection EPSG:5070), plus the current node. Only polygons that are graph
    nodes count.
  * The primary node is the largest candidate. The dam is moved to it only when it is at least
    twice as large as the current node.
  * The basin stays the same if the new node belongs to it; otherwise the first basin that
    contains the new node is used.
  * All candidates are recorded as `member_nodes`, so that a reservoir split into several
    polygons (for example the arms of Shasta) can be summed. `primary_share` is the primary
    node's share of the member area.
  * `new_supervised` / `old_supervised` say whether the node is in the list of nodes with a
    SWOT training target.

Notes
  Only resops_node_map.csv is written. No other file is changed.
"""
from swotgat.paths import NAS_WATER, WORK
import pandas as pd, numpy as np, geopandas as gpd, pyogrio
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork"); C = Path(f"{WORK}/CONUS")
M = pd.read_csv(F/"resops_node_match.csv", dtype={"node": str})
t = pd.read_csv(F/"resops_targets_full.csv", dtype={"node": str}).drop_duplicates("grand_id").set_index("grand_id")
M["dam"] = M.grand_id.map(t.DAM_NAME)
st = pd.read_parquet(C/"attributes/static_conus_storage.parquet", columns=["lake_area_km2"]).reset_index()
st["subbasin"] = st.subbasin.astype(str)
graph = set(st.subbasin); basins_of = st.groupby("subbasin").basin.apply(list)
node_area = st.drop_duplicates("subbasin").set_index("subbasin").lake_area_km2
sup = set(pd.read_csv(C/"basin_lists/storage/train_nodes_g.csv", dtype=str).subbasin)

G = gpd.read_file(f"{NAS_WATER}/cjgleason/ted/GDW/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp",
                  columns=["GRAND_ID", "AREA_SKM", "RES_NAME"])
G["GRAND_ID"] = pd.to_numeric(G.GRAND_ID, errors="coerce")
G = G[G.GRAND_ID.isin(M.grand_id)].drop_duplicates("GRAND_ID").to_crs(4326)
P = pyogrio.read_dataframe(f"{NAS_WATER}/cjgleason/ted/LakeSPvC_ts_for_Ted/SWOT_PLD_v106_reservoirs.gdb",
                           columns=["lake_id", "res_id", "names", "poly_area"], bbox=tuple(G.total_bounds))
P["lake_id"] = P.lake_id.astype("int64").astype(str); P["poly_area"] = pd.to_numeric(P.poly_area, errors="coerce")
P["res_id"] = pd.to_numeric(P.res_id, errors="coerce")
P = P[P.lake_id.isin(graph)].copy()
print(f"GDW polygons {len(G)} of {M.grand_id.nunique()};  PLD polygons that are graph nodes in bbox: {len(P):,}")

# polygon overlap in an equal-area CRS
Ge, Pe = G.to_crs(5070), P.to_crs(5070)
Pe["pa"] = Pe.area
J = gpd.overlay(Pe[["lake_id", "pa", "geometry"]], Ge[["GRAND_ID", "geometry"]], how="intersection", keep_geom_type=True)
J["frac"] = J.area/J.pa
ov = J[J.frac >= 0.2].groupby("GRAND_ID").lake_id.apply(set)
byres = P.dropna(subset=["res_id"]).groupby("res_id").lake_id.apply(set)
area = P.set_index("lake_id").poly_area
gdw_area = G.set_index("GRAND_ID").AREA_SKM

rows = []
for _, r in M.iterrows():
    gid = r.grand_id
    cand = set(byres.get(float(gid), set())) | set(ov.get(gid, set()))
    cand.add(r.node) if r.node in graph else None
    mem = sorted(cand, key=lambda n: -area.get(n, node_area.get(n, 0)))
    prim = mem[0] if mem else r.node
    a_old = area.get(r.node, node_area.get(r.node, np.nan)); a_new = area.get(prim, np.nan)
    renode = prim != r.node and a_new >= 2*a_old
    new = prim if renode else r.node
    # basin: keep the old basin if the new node is in it, else the first basin that contains it
    b = r.basin if r.basin in basins_of.get(new, []) else basins_of.get(new, [r.basin])[0]
    tot = sum(area.get(n, 0) for n in mem)
    rows.append(dict(grand_id=gid, dam=r.dam, old_node=r.node, old_basin=r.basin, node=new, basin=b, renoded=renode,
                     old_area=a_old, new_area=area.get(new, a_old), gdw_area=gdw_area.get(gid, np.nan),
                     member_nodes=";".join(mem), n_members=len(mem), members_area=tot,
                     primary_share=area.get(new, a_old)/tot if tot else np.nan,
                     new_supervised=new in sup, old_supervised=r.node in sup))
X = pd.DataFrame(rows)
X["old_ratio"] = X.old_area/X.gdw_area; X["new_ratio"] = X.new_area/X.gdw_area; X["members_ratio"] = X.members_area/X.gdw_area
X.to_csv(F/"resops_node_map.csv", index=False)
dup = X[X.node.duplicated(keep=False)]
print(f"re-noded: {X.renoded.sum()} of {len(X)};  new node shared by >1 dam: {len(dup)}")
if len(dup): print(dup[["dam", "node", "new_area"]].to_string())
print(f"node/GDW area within 1/3-3: before {X.old_ratio.between(1/3,3).mean():.0%}  after {X.new_ratio.between(1/3,3).mean():.0%}")
print(f"split reservoirs (primary < 80% of member area): {(X.primary_share < .8).sum()}")
pd.set_option("display.width", 250)
print(X[X.renoded][["dam", "old_area", "new_area", "gdw_area", "n_members", "primary_share", "old_supervised", "new_supervised"]]
      .round(2).sort_values("gdw_area", ascending=False).to_string())
