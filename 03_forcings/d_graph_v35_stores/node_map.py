"""Decide, for every sub-basin of graph v3.5, whether its forcing is copied or extracted again.

The graph v3.5 stores are not extracted from scratch. A sub-basin whose polygon did not change
keeps the values of the existing store (the "template store", built on the earlier node set), bit
for bit. Only new or reshaped sub-basins get a new ERA5-Land extraction. This script writes the
table that every later script of this folder reads to tell the two cases apart.

Reads (paths relative to $SWOTGAT_WORK; all inputs are only read)
    CONUS_storage_fork/eval/graph_v3/v35/subbasins_conus_v35.parquet   graph v3.5 sub-basin polygons
    CONUS_storage_fork/eval/graph_v3/v35/graphs_conus_v35.json         graph v3.5 (its node list)
    CONUS/metadata/subbasins_conus.parquet    polygons of the earlier node set, from which the
                                              template store was extracted
    CONUS/zarr_store_vd2026_halfexact         the template store: the 2020-2026 training store on
                                              the earlier node set (04_swot_targets/c_template_store)
    CONUS_storage_fork/eval/v21_retrain/store/lake_targets_vd2026_halfexact.parquet
                                              lake targets, for the lake counts of the summary
    CONUS/metadata/graphs_conus_v22.json      the graph of an earlier graph version (v2.2), for
                                              the summary counts

Writes (<forcing> = CONUS_storage_fork/eval/v35_inputs/forcing)
    <forcing>/work/node_map.csv            one row per v3.5 sub-basin
    <forcing>/work/node_map_summary.json   counts (also printed)

Usage: python node_map.py        (no arguments)

Method
    Every row of the v3.5 sub-basin table is classified, not only the graph nodes, because a store
    carries one column per sub-basin row of its graph version (the template does the same).
    copy     The id is a column of the template store, in the same basin group, and the v3.5
             polygon is identical to the polygon the template was extracted from. Identical means:
             the raw WKB bytes are equal or, if the bytes differ, shapely.equals_exact with
             tolerance 0 holds (same geometry type, same coordinates in the same order).
    extract  Everything else: a new id, a retained id that moved to another basin group, or a
             retained id with any difference in the polygon. These get a new ERA5-Land extraction
             over the v3.5 polygon.
    Columns of node_map.csv: subbasin, outlet_id (basin), in_graph, template_group, in_template,
    same_group, wkb_equal, geom_equal, area_ratio_vs_template (v3.5 area / template area, for
    retained ids whose WKB bytes differ), action (copy or extract), kind (reservoir, gauge or plain).
    The script asserts that ids are unique, that both polygon tables are in EPSG:4326, and that
    the columns of the template store are exactly the rows of the earlier sub-basin table, each in
    the group of its basin.

    In the paper's run: 20,081 sub-basins are copied and 4,416 are extracted again.
"""
from swotgat.paths import WORK
import json, numpy as np, pandas as pd, pyarrow.parquet as pq, shapely, zarr
from pathlib import Path
ROOT = Path(f"{WORK}"); C = ROOT/"CONUS"; CSF = ROOT/"CONUS_storage_fork"
V35 = CSF/"eval/graph_v3/v35"; T = C/"zarr_store_vd2026_halfexact"; F = CSF/"eval/v35_inputs/forcing"
COLS = ["id", "outlet_id", "geometry", "is_gauge", "is_reservoir", "node_type", "area_km2"]
def load(p):
    """Read the sub-basin table p (the columns in COLS), indexed by sub-basin id as a string.

    The ids of the basins (outlet_id) are also converted to strings. Stops with an assertion
    error if a sub-basin id occurs twice.
    """
    t = pq.read_table(p, columns=COLS).to_pandas(ignore_metadata=True); t["id"] = t["id"].astype(str); t["outlet_id"] = t["outlet_id"].astype(str)
    assert t["id"].is_unique; return t.set_index("id")
N = load(V35/"subbasins_conus_v35.parquet"); O = load(C/"metadata/subbasins_conus.parquet")
for p in (V35/"subbasins_conus_v35.parquet", C/"metadata/subbasins_conus.parquet"):
    assert json.loads(pq.ParquetFile(p).schema_arrow.metadata[b"geo"])["columns"]["geometry"]["crs"]["id"] == {"authority": "EPSG", "code": 4326}, p
nodes = [str(n["id"]) for n in json.load(open(V35/"graphs_conus_v35.json"))["nodes"]]; gset = set(nodes)
# Columns of the template store, per basin group. The store itself is read, not the polygon table,
# because the store is what the values are copied from.
tb = sorted(p.name for p in T.iterdir() if (p/"zarr.json").is_file()); tcol = {}
for b in tb:
    for s in zarr.open_group(T/b, mode="r")["subbasin"][:]: tcol[str(s)] = b
assert set(tcol) == set(O.index) and all(O.outlet_id[s] == b for s, b in tcol.items()), "template columns != v1 subbasins rows"
M = pd.DataFrame(index=N.index); M["outlet_id"] = N.outlet_id; M["in_graph"] = M.index.isin(gset)
M["template_group"] = [tcol.get(s, "") for s in M.index]; M["in_template"] = M.template_group != ""
M["same_group"] = M.in_template & (M.template_group == M.outlet_id)
sh = M.index[M.in_template]
M["wkb_equal"] = False; M.loc[sh, "wkb_equal"] = [N.geometry[s] == O.geometry[s] for s in sh]
M["geom_equal"] = M.wkb_equal.copy()
cand = M.index[M.in_template & ~M.wkb_equal]
if len(cand):
    a = shapely.from_wkb(N.geometry[cand].values); b = shapely.from_wkb(O.geometry[cand].values)
    M.loc[cand, "geom_equal"] = shapely.equals_exact(a, b, tolerance=0.0)
    ar = shapely.area(a)/np.maximum(shapely.area(b), 1e-30)
    M["area_ratio_vs_template"] = np.nan; M.loc[cand, "area_ratio_vs_template"] = ar
M["action"] = np.where(M.same_group & M.geom_equal, "copy", "extract")
M["kind"] = np.where(N.is_reservoir.astype(bool), "reservoir", np.where(N.is_gauge.astype(bool), "gauge", "plain"))
M.index.name = "subbasin"; M.to_csv(F/"work/node_map.csv")
gm = M[M.in_graph]
L = pd.read_parquet(CSF/"eval/v21_retrain/store/lake_targets_vd2026_halfexact.parquet", columns=["subbasin"]); lakes = sorted(set(L.subbasin.astype(str)))
summ = dict(v35_rows=len(M), graph_nodes=len(nodes), rows_not_in_graph=int((~M.in_graph).sum()),
            graph_copy=int((gm.action == "copy").sum()), graph_extract=int((gm.action == "extract").sum()),
            graph_extract_new_id=int((~gm.in_template).sum()), graph_extract_same_id_polygon_differs=int((gm.in_template & ~gm.geom_equal).sum()),
            same_id_wkb_differs_but_geom_equal=int((M.in_template & ~M.wkb_equal & M.geom_equal).sum()),
            same_id_group_changed=int((M.in_template & ~M.same_group).sum()),
            nongraph_rows_copy=int(((~M.in_graph) & (M.action == "copy")).sum()), nongraph_rows_extract=int(((~M.in_graph) & (M.action == "extract")).sum()),
            template_ids_gone=int(len(set(O.index) - set(N.index))), template_graph_ids_gone_from_v35_graph=None,
            extract_by_basin_top10=gm[gm.action == "extract"].outlet_id.value_counts().head(10).to_dict(),
            extract_basins=int(gm[gm.action == "extract"].outlet_id.nunique()),
            extract_by_kind=gm[gm.action == "extract"].kind.value_counts().to_dict(),
            lakes_halfexact=len(lakes), lakes_in_v35_graph=int(sum(l in gset for l in lakes)),
            lakes_same_group=int(sum(l in M.index and M.same_group.get(l, False) for l in lakes)),
            lakes_polygon_changed=int(sum(l in M.index and not M.geom_equal.get(l, False) for l in lakes)))
v22 = json.load(open(C/"metadata/graphs_conus_v22.json")); v22n = {str(n["id"]) for n in v22["nodes"]}
summ["template_graph_ids_gone_from_v35_graph"] = len(v22n - gset); summ["v22_graph_nodes"] = len(v22n)
json.dump(summ, open(F/"work/node_map_summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))
