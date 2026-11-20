"""Carry the lake-inflow corrections of earlier graph versions (2 to 2.3) over to a re-delineated graph, by lake id.

Graph versions 2, 2.1, 2.2 and 2.3 corrected, by hand, links where a river that passes beside a lake was routed into
it. Each such edit took an inflow away from a lake (its old_nextdown is a reservoir node). The edits name node ids of
the first graph, and most of those nodes no longer exist after the re-delineation (graph v3.x). Lake ids exist in
both. This script therefore re-derives the corrections on the re-delineated graph, lake by lake, and uses graph v2.3
as the reference for where each inflow's water goes.

Reads (E = $SWOTGAT_WORK/CONUS_storage_fork/eval)
  --base-subs, --base-graph   sub-basin table and graph JSON of the re-delineated graph, relative to E/graph_v3
  from earlier graph versions:
    $SWOTGAT_WORK/CONUS/metadata/subbasins_conus_v23.parquet, graphs_conus_v23.json   graph v2.3
    E/graph_v2/edits_v2_combined.csv, E/graph_map/v21_build/edits_v21_applied.csv,
    E/graph_map/v22_build/edits_v22_approved.csv, E/graph_map/v23_build/edits_v23_approved.csv   the edit lists
    E/graph_map/v22_area_check/area_check_nodes.csv   lake names and MERIT upstream areas, for the report only
  $SWOTGAT_NAS_ICE/data/MERIT_Basins/MERIT_Hydro_v07_Basins_v01_bugfix/pfaf_level_01/riv_pfaf_7_*.shp
                              MERIT-Basins river reaches (COMID, NextDownID, uparea)

Writes (in E/graph_v3/<out-dir>, default v32; inputs are only read)
  subbasins_conus_<tag>.parquet   the base table with nextdown and uparea_km2 updated. uparea_km2 changes by the same
                                  amount as the area accumulated over the graph. Every other column is checked to be
                                  identical to the base table.
  carry_edits.csv                 one row per re-pointed inflow (node, old_nextdown, new_nextdown, lake, ...)
  carry_lakes.csv                 every lake considered, with the outcome (plus one row per inflow that found no
                                  common node below the lake)
  carry_summary.json              counts

Usage (the paper's graph v3.4)
  python carry_v2_fixes.py --base-subs subbasins_conus_v34base.parquet --base-graph graphs_conus_v34base.json
                           --tag v34 --out-dir v34          (one command line)
  Then build the graph JSON from the new table with build_conus_graphs_v2.py.

Method
For every lake L from which an earlier edit took an inflow away:
  1. L must exist in the re-delineated graph.
  2. Act only if the re-delineated graph sends more water through L than graph v2.3 does: the upstream area of L
     (own areas accumulated over the graph) is more than 1.05 times the v2.3 value and more than 1 km2 above it.
     The test compares the two graphs with each other, so that MERIT upstream area stays an independent reference
     for checking the result.
  3. For each inflow p of L, find the v2.3 node that holds p's water. If p is also a v2.3 node (same id), it is p.
     Otherwise take p's outlet reach (the MERIT reach with the largest MERIT upstream area whose midpoint lies in p)
     and the v2.3 node that holds that reach's midpoint. If that node is L itself, follow the MERIT main stem upstream
     (at each step, the inflowing reach with the largest upstream area) until a reach lies in another node. This is
     needed because the v2.3 lake polygons still contain the river corridors they absorbed in the first graph: the
     earlier edits re-pointed links, not polygons.
  4. If v2.3 routes that water through L, p is left as it is.
  5. Otherwise p is re-pointed past L, to the first node q below L in the re-delineated graph that lies on the v2.3
     downstream path of that water (q is a v2.3 node on the path, or q's outlet reach lies in one). If no such node is
     found, no edit is made and the lake is reported.
After the edits the graph must still be acyclic, with at most one link out of each node and the same number of basins.

Notes
  This script needs files of graph v2.3 and the edit lists of versions 2 to 2.3, which are not rebuilt by this
  repository (see docs/KNOWN_LIMITS.md). In the paper's run it re-pointed 106 inflows; that edit list is shipped as
  01_graph/inputs/carry_edits_v34.csv.
"""
from swotgat.paths import NAS_ICE, WORK
import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyogrio
import shapely

R = Path(f"{WORK}")
F = R/"CONUS_storage_fork/eval"
V3 = F/"graph_v3"
G = F/"graph_map"
MB1 = Path(f"{NAS_ICE}/data/MERIT_Basins/MERIT_Hydro_v07_Basins_v01_bugfix/pfaf_level_01")
V23 = (R/"CONUS/metadata/subbasins_conus_v23.parquet", R/"CONUS/metadata/graphs_conus_v23.json")
EDITS = {"v2": F/"graph_v2/edits_v2_combined.csv", "v2.1": G/"v21_build/edits_v21_applied.csv",
         "v2.2": G/"v22_build/edits_v22_approved.csv", "v2.3": G/"v23_build/edits_v23_approved.csv"}
MORE_FRAC, MORE_KM2 = 1.05, 1.0
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--base-subs", required=True); ap.add_argument("--base-graph", required=True)
ap.add_argument("--tag", required=True); ap.add_argument("--out-dir", default="v32")
A = ap.parse_args()
OUT = V3/A.out_dir; OUT.mkdir(parents=True, exist_ok=True)
BASE = (V3/A.base_subs, V3/A.base_graph)


def load(sp, gp):
    """Read a sub-basin table and its graph JSON.

    Return (s, J): the table from GeoParquet file sp, indexed by node id as text, and the networkx graph
    from node-link JSON file gp.
    """
    s = gpd.read_parquet(sp, columns=["geometry", "area_km2", "uparea_km2", "node_type", "outlet_id", "nextdown"]); s.index = s.index.astype(str)
    J = nx.readwrite.json_graph.node_link_graph(json.load(open(gp)), edges="edges")
    return s, J


def accumulate(J, s):
    """Return the upstream area of every node of graph J as a Series indexed by node id.

    The upstream area is the node's own area_km2 (from table s; 0 if the node has no row) plus that of
    everything upstream in J.
    """
    acc = {}
    for n in nx.topological_sort(J):
        acc[n] = float(s.area_km2.get(n, 0.0) or 0.0) + sum(acc[p] for p in J.predecessors(n))
    return pd.Series(acc)


def reach_nodes(s, J, riv, mid):
    """Match MERIT reaches to the nodes of graph J and return two dicts.

    The first maps a MERIT reach (COMID) to the graph node that holds the reach's midpoint. The second maps a
    graph node to its outlet reach (the reach with the largest MERIT upstream area among those whose midpoint
    lies in the node). s is the sub-basin table, riv the reach table and mid the reach midpoints.
    """
    sg = s.loc[s.index.isin(set(J.nodes))]
    tree = shapely.STRtree(sg.geometry.values)
    pi, qi = tree.query(mid, predicate="within")
    first = pd.Series(qi, index=pi); first = first[~first.index.duplicated()]
    node_of = pd.Series(sg.index.values[first.values], index=riv.COMID.values[first.index.values])
    up = pd.Series(riv.uparea.values, index=riv.COMID.values)
    t = pd.DataFrame({"node": node_of.values, "up": up.reindex(node_of.index).values}, index=node_of.index)
    outlet = t.sort_values("up").groupby("node").tail(1)
    return node_of.to_dict(), pd.Series(outlet.index.values, index=outlet.node.values).to_dict()


t0 = time.time()
s2, J2 = load(*V23)
s3, J3 = load(*BASE)
nd2 = {u: next(iter(J2.successors(u)), None) for u in J2}
nd3 = {u: next(iter(J3.successors(u)), None) for u in J3}
acc2, acc3 = accumulate(J2, s2), accumulate(J3, s3)
log(f"v2.3 {J2.number_of_nodes():,} nodes; base {J3.number_of_nodes():,} nodes ({BASE[1].name})")
bb = np.r_[np.minimum(s2.total_bounds[:2], s3.total_bounds[:2]), np.maximum(s2.total_bounds[2:], s3.total_bounds[2:])]
riv = pyogrio.read_dataframe(MB1/"riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp", columns=["COMID", "NextDownID", "uparea"], bbox=tuple(bb))
mid = shapely.line_interpolate_point(riv.geometry.values, 0.5, normalized=True)
node2_of, _ = reach_nodes(s2, J2, riv, mid)
_, outlet3 = reach_nodes(s3, J3, riv, mid)
log(f"{len(riv):,} MERIT reaches; {len(node2_of):,} placed in v2.3 nodes; outlet reach for {len(outlet3):,} base nodes")
# main-stem parent of each reach (the inflowing reach with the largest MERIT upstream area)
_r = riv[["COMID", "NextDownID", "uparea"]].sort_values("uparea")
main_up = _r.groupby("NextDownID").COMID.last().to_dict()


def v23_origin(c, L):
    """Return the graph v2.3 node that holds the water of MERIT reach c.

    The v2.3 lake polygons still contain the river corridors they absorbed in the first graph (the earlier edits
    re-pointed the river's inflows, not the polygons). So while c lies in the lake L in v2.3, step up the MERIT
    main stem to where the water comes from. Return L or None if the main stem ends before another node is
    found, and None after 200 steps.
    """
    for _ in range(200):
        n = node2_of.get(c)
        if n is not None and n != L:
            return n
        c = main_up.get(c)
        if c is None:
            return n
    return None

E = pd.concat([pd.read_csv(p, dtype=str)[["node", "old_nextdown", "new_nextdown"]].assign(edit_set=k) for k, p in EDITS.items()], ignore_index=True)
E = E[E.old_nextdown.map(lambda o: s2.node_type.get(o) == "reservoir")]
lakes = E.groupby("old_nextdown").edit_set.agg(lambda x: ",".join(sorted(set(x))))
log(f"{len(E)} v2-v2.3 edits took an inflow away from a lake; {len(lakes)} lakes")
X = pd.read_csv(G/"v22_area_check/area_check_nodes.csv", dtype={"node": str}).set_index("node")   # names only (and MERIT for the report)


def down_set(nd, n):
    """Return the set of nodes on the downstream path of n (n included), following the nextdown mapping nd."""
    out = set()
    while n is not None and n not in out:
        out.add(n); n = nd.get(n)
    return out


edits, lrows = [], []
for L, sets in lakes.items():
    row = dict(lake=L, name=X.name.get(L) if L in X.index else None, edit_sets=sets, km2_v23=round(float(acc2.get(L, np.nan)), 1))
    if L not in J3:
        lrows.append(dict(row, outcome="lake not in the re-delineation")); continue
    a2, a3 = float(acc2.get(L, np.nan)), float(acc3[L])
    row["km2_base"] = round(a3, 1)
    if not (a3 > MORE_FRAC * a2 and a3 - a2 > MORE_KM2):
        lrows.append(dict(row, outcome="re-delineation sends no more water through the lake than v2.3")); continue
    n_ed = 0
    for p in sorted(J3.predecessors(L)):
        c = outlet3.get(p)
        n2 = p if p in nd2 else (v23_origin(c, L) if c is not None else None)   # same id in v2.3: its v2.3 routing
        if n2 is None:
            continue
        if n2 == L or L in down_set(nd2, n2):
            continue                                                            # v2.3 also sends this water through L
        path2 = down_set(nd2, n2)
        q, hops = nd3.get(L), 0
        while q is not None and hops < 500:
            cq = outlet3.get(q)
            if q in path2 or (cq is not None and node2_of.get(cq) in path2):
                break
            q, hops = nd3.get(q), hops + 1
        if q is None or hops >= 500:
            lrows.append(dict(row, outcome=f"inflow {p}: no common node below the lake")); continue
        edits.append(dict(node=p, old_nextdown=L, new_nextdown=q, lake=L, lake_name=row["name"], node_km2=round(float(acc3[p]), 1),
                          outlet_reach=c, v23_node_of_reach=n2, v23_nextdown_of_that=nd2.get(n2), hops_below_lake=hops + 1, edit_sets=sets))
        n_ed += 1
    lrows.append(dict(row, outcome=f"{n_ed} inflow(s) re-pointed past the lake" if n_ed else "no inflow that v2.3 routes elsewhere"))
Ed = pd.DataFrame(edits)
log(f"{len(Ed)} carry-over edits at {Ed.lake.nunique() if len(Ed) else 0} lakes")

# ---- apply to the graph and the table
J = J3.copy()
for r in Ed.itertuples():
    J.remove_edge(r.node, r.old_nextdown); J.add_edge(r.node, r.new_nextdown)
assert nx.is_directed_acyclic_graph(J) and max(d for _, d in J.out_degree()) <= 1
assert nx.number_weakly_connected_components(J) == nx.number_weakly_connected_components(J3)
acc = accumulate(J, s3)
delta = (acc - acc3).reindex(s3.index).fillna(0.0)
moved = delta.index[(delta.abs() > 1e-9 * acc3.reindex(delta.index).abs().clip(lower=1e-9))]
new_up = s3.uparea_km2.astype(float).copy(); new_up[moved] = new_up[moved] + delta[moved]
new_nd = s3.nextdown.astype("string").copy()
for r in Ed.itertuples():
    assert str(new_nd[r.node]) == r.old_nextdown, (r.node, new_nd[r.node], r.old_nextdown)
    new_nd[r.node] = r.new_nextdown
T = pq.read_table(BASE[0])
ids = pd.Index(T.column("id").to_pylist() if "id" in T.schema.names else s3.index).astype(str)
assert (ids == s3.index).all()
T2 = T.set_column(T.schema.get_field_index("nextdown"), T.schema.field("nextdown"),
                  pa.array([None if pd.isna(v) or v == "" else str(v) for v in new_nd.reindex(ids)], type=T.schema.field("nextdown").type))
T2 = T2.set_column(T2.schema.get_field_index("uparea_km2"), T2.schema.field("uparea_km2"),
                   pa.array(new_up.reindex(ids).to_numpy(np.float64), type=T.schema.field("uparea_km2").type))
same = {c: T.column(c).equals(T2.column(c)) for c in T.schema.names}
assert all(v for c, v in same.items() if c not in ("nextdown", "uparea_km2")), same
pq.write_table(T2, OUT/f"subbasins_conus_{A.tag}.parquet")

L2 = pd.DataFrame(lrows)
L2["km2_after"] = L2.lake.map(lambda n: round(float(acc[n]), 1) if n in acc.index else np.nan)
L2["merit_km2"] = L2.lake.map(lambda n: round(float(X.merit[n]), 1) if n in X.index and pd.notna(X.merit[n]) else np.nan)
L2.to_csv(OUT/"carry_lakes.csv", index=False)
Ed.to_csv(OUT/"carry_edits.csv", index=False)
gauges_moved = [n for n in moved if s3.node_type.get(n) == "gauge"]
summ = dict(base=str(BASE[1].name), tag=A.tag, lakes_fixed_in_v2=len(lakes), outcomes=L2.outcome.str.replace(r"^inflow \S+:", "inflow:", regex=True)
            .str.replace(r"^\d+ inflow\(s\)", "n inflow(s)", regex=True).value_counts().to_dict(),
            edits=len(Ed), lakes_edited=int(Ed.lake.nunique()) if len(Ed) else 0, nodes_uparea_changed=len(moved), gauges_uparea_changed=len(gauges_moved),
            seconds=round(time.time() - t0))
(OUT/"carry_summary.json").write_text(json.dumps(summ, indent=1))
log(json.dumps(summ, indent=1))
pd.set_option("display.width", 250)
if len(Ed):
    print(L2[L2.outcome.str.contains("re-pointed")].sort_values("km2_base", ascending=False).head(40).to_string(index=False))
