"""Place MODROM's non-GRanD reservoirs on graph nodes, using MODROM's own coordinates.

MODROM and GDROM both give ids of 10000 and above to reservoirs that are not in GRanD, but they
number them differently. In the datasets used for the paper, 23 ids name a different reservoir in
the two datasets (for example, MODROM 10034 is Causey, Utah; GDROM 10034 is Ridgway, Colorado).
rule_reservoir_nodes.csv (from grades_cascade_prep.py) places every such id with GDROM's
coordinates, so MODROM's parameters for those ids would run on the wrong lake. This script places
MODROM's non-GRanD ids again with MODROM's coordinates. The simulators (grades_sim_v4*.py) use
its table for the MODROM runs.

Usage:
  python modrom_nongrand_nodes.py          (no arguments; run grades_cascade_prep.py nodes first)

Reads:
  $SWOTGAT_WORK/GRADES/modrom/data/reservoir_metadata.csv   MODROM reservoirs (id, name, lat, lon)
  $SWOTGAT_WORK/GRADES/rule_reservoir_nodes.csv             nodes already held by rule reservoirs
  $SWOTGAT_WORK/CONUS/metadata/subbasins_conus.parquet      nodes of the graph
  SWOT_PLD_v106_reservoirs.gdb (under $SWOTGAT_NAS_WATER)   SWOT Prior Lake Database (PLD)
                                                            reservoir polygons
  $SWOTGAT_WORK/CONUS_storage_fork/testsets/{regime1_2023_2025,regime2_2015_2020}/sites.csv
                                                            member nodes of the test-set dams

Writes:
  $SWOTGAT_WORK/GRADES/modrom_nongrand_nodes.csv   GRAND_ID, name, node, dist_m, pld_names, reject.
      `node` is empty when the reservoir is not placed; `reject` then gives the reason.
  The full table is also printed, with the number of ids whose node differs from the placement
  with GDROM's coordinates.

Method (the same rule as grades_cascade_prep.py): a reservoir goes to the nearest PLD reservoir
polygon that is a reservoir node of the graph, within 1.5 km of MODROM's dam point (distances in
EPSG:5070). The placement is rejected when that node is a member of a test-set dam, or when a GRanD
rule reservoir (id below 10000) already holds it in rule_reservoir_nodes.csv.
"""
from swotgat.paths import NAS_WATER, WORK
import numpy as np, pandas as pd, geopandas as gpd
from pathlib import Path
G = Path(f"{WORK}/GRADES"); C = Path(f"{WORK}/CONUS")
F = Path(f"{WORK}/CONUS_storage_fork"); J = Path(f"{NAS_WATER}/cjgleason/ted/LakeSPvC_ts_for_Ted")
O = pd.read_csv(G/"modrom/data/reservoir_metadata.csv"); O.columns = O.columns.str.strip()
O = O[O.GRAND_ID >= 10000][["GRAND_ID", "Dam name", "latitude", "longitude"]].rename(columns={"Dam name": "name"})
P = gpd.GeoDataFrame(O, geometry=gpd.points_from_xy(O.longitude, O.latitude), crs=4326).to_crs(5070)
s = pd.read_parquet(C/"metadata/subbasins_conus.parquet", columns=["node_type"]); s.index = s.index.astype(str)
nodes = set(s.index[s.node_type == "reservoir"])
L = gpd.read_file(J/"SWOT_PLD_v106_reservoirs.gdb", columns=["lake_id", "names"])
L = L[L.lake_id.notna()]; L["node"] = L.lake_id.astype("float64").astype("int64").astype(str); L = L[L.node.isin(nodes)].to_crs(5070)
j = gpd.sjoin_nearest(P, L[["node", "names", "geometry"]], how="left", max_distance=1500, distance_col="dist_m").drop_duplicates("GRAND_ID")
RN = pd.read_csv(G/"rule_reservoir_nodes.csv", dtype={"node": str})
held = set(RN.node[RN.node.notna() & (RN.GRAND_ID < 10000)])                     # GRanD rule reservoirs keep their nodes
S1 = pd.concat([pd.read_csv(F/"testsets"/x/"sites.csv", dtype={"members": str}) for x in ("regime1_2023_2025", "regime2_2015_2020")])
members = {m for ms in S1.members.astype(str) for m in ms.split(";") if m and m != "nan"}
j["reject"] = np.where(j.node.isna(), "no graph reservoir within 1.5 km", np.where(j.node.isin(members), "test dam member",
                       np.where(j.node.isin(held), "held by a GRanD rule reservoir", "")))
j.loc[j.reject != "", "node"] = np.nan
out = j[["GRAND_ID", "name", "node", "dist_m", "names", "reject"]].rename(columns={"names": "pld_names"})
out.to_csv(G/"modrom_nongrand_nodes.csv", index=False)
old = RN.set_index("GRAND_ID").node
out["old_node"] = out.GRAND_ID.map(old)
print(f"MODROM non-GRanD ids {len(out)} | placed {out.node.notna().sum()} | changed from the GDROM-coordinate placement "
      f"{int((out.node.fillna('') != out.old_node.fillna('')).sum())}")
print(out.to_string(index=False))
