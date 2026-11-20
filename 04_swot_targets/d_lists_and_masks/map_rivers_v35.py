"""Map every listed river node that no longer exists in graph v3.5 to the v3.5 node that holds its reach.

Step 1 of re-keying the node lists to graph v3.5 (step 2: rekey_lists_v35.py; step 3:
filter_graph_basins_v35.py). The "template" lists and river drop list are those of an earlier model configuration
(A180), which used graph version 2.2. Gauge and lake ids all survive into graph v3.5; only plain river nodes
(MERIT-Basins reach ids) can disappear.

Usage: python map_rivers_v35.py      (no arguments)

Reads (under $SWOTGAT_WORK; nothing is modified)
  CONUS/basin_lists/storage/train_nodes_g.csv, valid_nodes_g.csv, test_nodes_g.csv,
  CONUS/basin_lists/fallback/dv_SWOTAE_vd2026_fb_g.csv            the template lists
  CONUS/basin_lists/swotqc/drop_river_paper_min_present_t9b.parquet   the template river drop list
  CONUS/metadata/subbasins_conus_v22.parquet, graphs_conus_v22.json   the template node set (graph v2.2)
  CONUS_storage_fork/eval/graph_v3/v35/subbasins_conus_v35.parquet, graphs_conus_v35.json   graph v3.5 (md5 checked)
  riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp               MERIT-Basins river lines, in the raw-data
                                                                  archive configured as NAS_ICE

Writes
  CONUS_storage_fork/eval/v35_inputs/lists/work/map_rivers_v35.csv   one row per old id; the script stops if the
                                                                     file exists

Method
  For every id in the template lists or drop list that is not a v3.5 graph node:
    midpoint  Take the river line of the old node (its original_comid) and the point at half its length. The
              v3.5 node of the same basin (outlet_id) whose polygon covers that point is the match
              (method "midpoint_in_polygon"). If the point lies on a shared boundary and several polygons cover
              it, the one with the larger overlap is taken.
    overlap   The v3.5 node of the same basin whose polygon has the largest intersection with the old polygon
              (areas in EPSG:5070). It is the match only when no polygon covers the midpoint (method
              "max_overlap"). It is always reported as a cross-check.
  The midpoint is also located among all v3.5 nodes, to show whether a reach has left its basin.
  Drop-list ids that are not nodes of the template graph either (basins the graph never supplied) are not mapped.
"""
from swotgat.paths import NAS_ICE, WORK
import hashlib, json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely

ROOT = Path(f"{WORK}"); C = ROOT/"CONUS"; CSF = ROOT/"CONUS_storage_fork"
V35 = CSF/"eval/graph_v3/v35"; L = CSF/"eval/v35_inputs/lists"; W = L/"work"
SUBS35 = V35/"subbasins_conus_v35.parquet"; GRAPH35 = V35/"graphs_conus_v35.json"
SUBST = C/"metadata/subbasins_conus_v22.parquet"; GRAPHT = C/"metadata/graphs_conus_v22.json"
RIV = Path(f"{NAS_ICE}/data/MERIT_Basins/MERIT_Hydro_v07_Basins_v01_bugfix/pfaf_level_01/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp")
TEMPL = {"train": C/"basin_lists/storage/train_nodes_g.csv", "valid": C/"basin_lists/storage/valid_nodes_g.csv",
         "test": C/"basin_lists/storage/test_nodes_g.csv", "fb": C/"basin_lists/fallback/dv_SWOTAE_vd2026_fb_g.csv"}
MASK = C/"basin_lists/swotqc/drop_river_paper_min_present_t9b.parquet"
OUT = W/"map_rivers_v35.csv"; assert not OUT.exists(), OUT
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
assert md5(SUBS35) == "19a764d997a07c870b2edb0688dbf3a8" and md5(GRAPH35) == "c580577d7581bbfb975c261bea609051", "v3.5 graph md5"

g35 = {str(n["id"]) for n in json.load(open(GRAPH35))["nodes"]}; gT = {str(n["id"]) for n in json.load(open(GRAPHT))["nodes"]}
N = gpd.read_parquet(SUBS35); N.index = N.index.astype(str); N = N[N.index.isin(g35)]; assert len(N) == len(g35) == 24381
O = gpd.read_parquet(SUBST); O.index = O.index.astype(str)
mem = {}
for k, p in TEMPL.items():
    for b, s in pd.read_csv(p, dtype=str)[["basin", "subbasin"]].itertuples(index=False): mem.setdefault((b, s), []).append(k)
mk = pd.read_parquet(MASK, columns=["basin", "subbasin"]).drop_duplicates()
for b, s in mk.astype(str).itertuples(index=False): mem.setdefault((b, s), []).append("mask_t9b")
gone = sorted((b, s) for (b, s) in mem if s not in g35)
offg = [(b, s) for (b, s) in gone if s not in gT]              # not graph nodes in the template either (basins the graph never supplied)
assert all(mem[k] == ["mask_t9b"] for k in offg), "a list id outside the template graph"
print(f"template ids (lists + t9b mask): {len(mem):,}; not v3.5 graph nodes: {len(gone)} "
      f"({len(offg)} mask-only rows outside the template graph too, basins {sorted({b for b, _ in offg})}: not mapped)", flush=True)
gone = [k for k in gone if k not in set(offg)]
for b, s in gone:
    assert s in gT and s in O.index and O.outlet_id[s] == b, (b, s)          # a template graph node, in the basin the list says
    assert s.isdigit() and len(s) == 8 and not O.is_gauge[s] and not O.is_reservoir[s], s   # MERIT river node (gauge / lake ids all survive)
ids = [s for _, s in gone]; comid = O.loc[ids, "original_comid"].astype("int64")
cl = ",".join(str(c) for c in sorted(set(comid)))
R = pyogrio.read_dataframe(RIV, where=f"COMID IN ({cl})", columns=["COMID", "uparea", "lengthkm"]).set_index("COMID")
assert set(comid) <= set(R.index), sorted(set(comid) - set(R.index))[:5]
mid = shapely.line_interpolate_point(R.geometry.loc[comid.values].values, 0.5, normalized=True)

Np = N.to_crs(5070); Op = O.loc[ids].to_crs(5070); midp = gpd.GeoSeries(mid, crs=4326).to_crs(5070).values
tree_all = shapely.STRtree(N.geometry.values); Nidx = N.index.values
rows = []
for k, (b, s) in enumerate(gone):
    pt = mid[k]
    cov = [Nidx[j] for j in tree_all.query(pt) if N.geometry.values[j].covers(pt)]
    cov_b = [n for n in cov if N.outlet_id[n] == b]
    cand = Np[Np.outlet_id == b]; inter = cand.geometry.intersection(Op.geometry[s]).area
    frac = (inter/Op.geometry[s].area).sort_values(ascending=False)
    ov1, f1 = (frac.index[0], float(frac.iloc[0])) if len(frac) and frac.iloc[0] > 0 else ("", 0.0)
    ov2, f2 = (frac.index[1], float(frac.iloc[1])) if len(frac) > 1 and frac.iloc[1] > 0 else ("", 0.0)
    if len(cov_b) == 1: v, how = cov_b[0], "midpoint_in_polygon"
    elif len(cov_b) > 1:                                        # midpoint on a shared boundary: the covering node with the larger overlap
        v = max(cov_b, key=lambda n: float(frac.get(n, 0.0))); how = "midpoint_in_polygon(boundary,max_overlap)"
    elif ov1: v, how = ov1, "max_overlap"
    else: v, how = "", "unmapped"
    rows.append(dict(old_id=s, basin=b, old_lists=";".join(sorted(mem[(b, s)])), old_node_type=O.node_type[s], old_area_km2=float(O.area_km2[s]),
                     old_uparea_km2=float(O.uparea_km2[s]), original_comid=int(comid.iloc[k]), merit_uparea_km2=float(R.uparea[comid.iloc[k]]),
                     merit_lengthkm=float(R.lengthkm[comid.iloc[k]]), mid_lon=float(pt.x), mid_lat=float(pt.y),
                     midpoint_node=";".join(cov_b), midpoint_node_any_basin=";".join(cov), overlap_node=ov1, overlap_frac=round(f1, 4),
                     overlap_node2=ov2, overlap_frac2=round(f2, 4), v35_id=v, method=how, midpoint_agrees_overlap=(v == ov1),
                     v35_node_type=(N.node_type[v] if v else ""), v35_is_new_id=(v not in gT) if v else False,
                     v35_area_km2=(float(N.area_km2[v]) if v else np.nan), v35_uparea_km2=(float(N.uparea_km2[v]) if v else np.nan)))
M = pd.DataFrame(rows); W.mkdir(exist_ok=True); M.to_csv(OUT, index=False)
print(M.method.value_counts().to_string()); print("midpoint agrees with max overlap:", int(M.midpoint_agrees_overlap.sum()), "of", len(M))
print("v35 node types:", M.v35_node_type.value_counts().to_dict(), "| new ids:", int(M.v35_is_new_id.sum()))
print("reach midpoint outside its basin's v3.5 nodes:", int((M.midpoint_node == "").sum()), "| midpoint in another basin:",
      int(((M.midpoint_node == "") & (M.midpoint_node_any_basin != "")).sum()))
print(f"-> {OUT}")
