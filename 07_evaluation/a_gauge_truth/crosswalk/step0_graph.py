"""Audit every reservoir node of the graph against the SWOT lake database and the dam database.

Step 0 of the dam-to-node crosswalk. The script changes no data. It prints four diagnostics and writes
one audit table per reservoir node, which later steps read.

Terms: PLD = SWOT Prior Lake Database (lake polygons; a lake node of the graph carries its PLD lake_id as
node id). GDW = Global Dam Watch (reservoir polygons and dam points). GRanD id = the dam identifier that
both databases share (PLD field res_id, GDW field GRAND_ID; 0 means "no id"). wse = SWOT water-surface
elevation.

Reads
  $SWOTGAT_WORK/CONUS/attributes/static_conus_storage.parquet    lake_area_km2 and dam_cap_mcm per node
  $SWOTGAT_WORK/CONUS/metadata/subbasins_conus.parquet           is_reservoir flag per node
  $SWOTGAT_WORK/CONUS_storage_fork/crosswalk/pld_wse_graph.parquet
      SWOT series per lake (columns date, lake_id, wse, dv_SWOTAE, dv_staticPLD); 2023-2025 is used
  under $SWOTGAT_NAS_WATER: the PLD v1.06 reservoir geodatabase (lake_id, res_id, poly_area) and the
      GDW v1.0 reservoir and barrier shapefiles (AREA_SKM, CAP_MCM, ELEV_MASL), clipped to the
      conterminous US

Writes
  $SWOTGAT_WORK/CONUS_storage_fork/crosswalk/graph_reservoir_node_audit.csv

Usage
  python step0_graph.py        (no arguments)

Method
  A reservoir node is a node flagged is_reservoir or with lake_area_km2 > 0.
  0a  Area. PLD polygon area / GDW reservoir area for every node with a GRanD id, counted in bins.
      A node is a "fragment" if it holds less than 5% of the PLD area of all graph nodes of its
      reservoir. The script also counts the reservoirs that are split over more than one node.
  0b  SWOT coverage. Among nodes with at least 10 SWOT wse observations: how many have no dv_SWOTAE
      value (the storage change computed with the SWOT area-elevation curve), how many of those have
      dv_staticPLD (the storage change computed with the static PLD area) instead, and the share
      without dv_SWOTAE by lake-area class.
  Level.  Median SWOT wse minus the GDW pool elevation (ELEV_MASL, read from a DEM), for nodes with both.
  Capacity.  The node's static attribute dam_cap_mcm divided by the GDW capacity of the node's own
      GRanD dam. This checks the catchment-intersection join that built the static attribute.
"""
from swotgat.paths import NAS_WATER, WORK
import numpy as np, pandas as pd, geopandas as gpd, pyogrio
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork"); C = Path(f"{WORK}/CONUS"); O = F/"crosswalk"
BB = (-125.5, 24.0, -66.5, 49.8)
st = pd.read_parquet(C/"attributes/static_conus_storage.parquet", columns=["lake_area_km2", "dam_cap_mcm"]).reset_index()
st["subbasin"] = st.subbasin.astype(str); st = st.drop_duplicates("subbasin").set_index("subbasin")
sb = pd.read_parquet(C/"metadata/subbasins_conus.parquet", columns=["is_reservoir"]); sb.index = sb.index.astype(str)
RES = set(sb.index[sb.is_reservoir == 1]) | set(st.index[st.lake_area_km2 > 0])
P = pyogrio.read_dataframe(f"{NAS_WATER}/cjgleason/ted/LakeSPvC_ts_for_Ted/SWOT_PLD_v106_reservoirs.gdb",
                           columns=["lake_id", "res_id", "poly_area"], bbox=BB, read_geometry=False)
P["lake_id"] = P.lake_id.astype("int64").astype(str); P["res_id"] = pd.to_numeric(P.res_id, errors="coerce").fillna(0).astype(int)
P["poly_area"] = pd.to_numeric(P.poly_area, errors="coerce")
P = P[P.lake_id.isin(RES)].drop_duplicates("lake_id").set_index("lake_id")
G = gpd.read_file(f"{NAS_WATER}/cjgleason/ted/GDW/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp", bbox=BB,
                  columns=["GRAND_ID", "AREA_SKM", "CAP_MCM"], ignore_geometry=True)
G["GRAND_ID"] = pd.to_numeric(G.GRAND_ID, errors="coerce"); G = G[G.GRAND_ID > 0].drop_duplicates("GRAND_ID").set_index("GRAND_ID")
B = gpd.read_file(f"{NAS_WATER}/cjgleason/ted/GDW/GDW_v1_0_shp/GDW_barriers_v1_0.shp", bbox=BB, columns=["GRAND_ID", "ELEV_MASL"], ignore_geometry=True)
B["GRAND_ID"] = pd.to_numeric(B.GRAND_ID, errors="coerce"); B = B[B.GRAND_ID > 0].drop_duplicates("GRAND_ID").set_index("GRAND_ID")
W = pd.read_parquet(O/"pld_wse_graph.parquet"); W = W[(W.date >= "2023-01-01") & (W.date <= "2025-12-31")]
agg = W.groupby("lake_id").agg(wse_n=("wse", "count"), wse_med=("wse", "median"), dvs_n=("dv_SWOTAE", "count"), dvp_n=("dv_staticPLD", "count"))
N = pd.DataFrame(index=sorted(RES)); N = N.join(P).join(agg).join(st)
N["grand"] = N.res_id.fillna(0).astype(int)
g = N[N.grand > 0].copy()
print(f"reservoir nodes in graph: {len(N):,}; PLD polygons: {N.poly_area.notna().sum():,}; with GRanD res_id: {len(g):,} "
      f"({g.grand.nunique():,} distinct GRanD reservoirs)")
# 0a: node area vs GDW area, node level and reservoir level
g["gdw_area"] = g.grand.map(G.AREA_SKM); g["ratio"] = g.poly_area/g.gdw_area
tot = g.groupby("grand").poly_area.sum(); g["share"] = g.poly_area/g.grand.map(tot)
bins = [0, .05, .2, 1/3, .5, 2, 3, np.inf]
print("\n0a) node PLD area / GDW area (nodes with a GRanD id):")
print(pd.cut(g.ratio, bins).value_counts().sort_index().to_string())
print(f"  nodes that are <5% of their reservoir's graph polygons (fragments): {(g.share < .05).sum():,}")
print(f"  GRanD reservoirs split over >1 graph node: {(g.groupby('grand').size() > 1).sum():,} of {g.grand.nunique():,}")
# 0b: SWOT wse present but no dv_SWOTAE
N["wse_n"] = N.wse_n.fillna(0); N["dvs_n"] = N.dvs_n.fillna(0); N["dvp_n"] = N.dvp_n.fillna(0)
hasw = N[N.wse_n >= 10]
miss = hasw[hasw.dvs_n == 0]
print(f"\n0b) nodes with >=10 SWOT wse obs: {len(hasw):,}; of those NO dv_SWOTAE: {len(miss):,} ({len(miss)/len(hasw):.0%}); "
      f"with dv_staticPLD instead: {(miss.dvp_n > 0).sum():,}")
ab = pd.cut(hasw.poly_area, [0, 1, 10, 50, 100, 1e5])
print(" share with NO dv_SWOTAE by lake area (km2):")
print((hasw.dvs_n == 0).groupby(ab).agg(["mean", "size"]).round(2).to_string())
print(f"  PLD area held by nodes with wse but no dv_SWOTAE: {miss.poly_area.sum():,.0f} km2 of {hasw.poly_area.sum():,.0f} km2 observed")
# SWOT level vs GDW DEM pool elevation
g2 = g.join(agg[[]], how="left"); g["dz_gdw"] = g.wse_med - g.grand.map(B.ELEV_MASL)
z = g.dz_gdw.dropna()
print(f"\nSWOT wse - GDW pool elevation (nodes with both, n={len(z):,}): median {z.median():+.1f} m; "
      f"|dz|<=5 {(z.abs()<=5).mean():.0%}; >+15 {(z>15).sum()}; <-15 {(z<-15).sum()}")
# static dam_cap_mcm vs the node's own GRanD capacity (catchment-intersects join audit)
g["own_cap"] = g.grand.map(G.CAP_MCM); g["cap_ratio"] = g.dam_cap_mcm/g.own_cap
print(f"\nstatic dam_cap_mcm vs own GRanD CAP_MCM (n={g.cap_ratio.notna().sum():,}): within 0.9-1.1 {g.cap_ratio.between(.9,1.1).mean():.0%}; "
      f">3x {(g.cap_ratio>3).sum()}; <1/3 {(g.cap_ratio<1/3).sum()}")
N.join(g[["gdw_area", "ratio", "share", "dz_gdw", "own_cap", "cap_ratio"]]).to_csv(O/"graph_reservoir_node_audit.csv")
