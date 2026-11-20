"""Collect the evidence used to review the flagged lakes of graph v3.4: one table and a set of map panels.

A review of graph v3.4 compared the upstream area of every lake node with MERIT-Hydro and flagged the lakes that
disagree. The flagged lakes are the rows of the review item table whose `group` is one of:
  refov   the graph looks right and the MERIT reference at the lake's pour point looks wrong (reference override)
  edge    a river runs along the lake outline and the graph routes it through the lake node
  above   the lake node's upstream area is far above any MERIT value in or next to the outline
  inflow  the MERIT reach that feeds the lake lies in a node that drains elsewhere
This script gathers, for each of these lakes, the numbers and a map needed to rule on it. make_decisions_v35.py then
writes the rulings.

Reads (W = $SWOTGAT_WORK, V3 = W/CONUS_storage_fork/eval/graph_v3)
  V3/review_v34/items.csv                         the review items (shipped as 01_graph/inputs/review_v34_items.csv)
  V3/v34/subbasins_conus_v34.parquet, V3/v34/graphs_conus_v34.json   graph v3.4
  W/LakeSPvC_ts_for_Ted/SWOT_PLD_v106_reservoirs.gdb   lake outlines of the SWOT Prior Lake Database (PLD)
  V3/bothwrong/lake_outline_test.csv              per lake, MERIT upstream area: merit_ref (at the PLD pour point),
                                                  upa_inside, upa_buffered (maximum within one cell of the outline)
  W/CONUS_storage_fork/eval/graph_review/_merit_upa_conus.vrt   MERIT-Hydro upstream-area mosaic (panel background)
  W/GDW_v1_0/GDW_v1_0_shp/GDW_barriers_v1_0.shp   dams of Global Dam Watch (GDW)
  W/CONUS_storage_fork/eval/half_vd_review/halfexact_lakes_vD.csv   lakes kept as SWOT targets (column kept; stage 04a)
  W/CONUS/basin_lists/storage/test_nodes.csv, train_nodes_all.csv   test and training node lists that existed before
                                                  graph v3.5

Writes
  V3/v35_work/lake_evidence.csv    one row per lake (shipped as 01_graph/inputs/lake_evidence_v35.csv)
  V3/v35_work/panels/lakes_NN.png  map panels, six lakes per sheet

Columns of lake_evidence.csv
  graph_v34        upstream area of the lake node in graph v3.4 (km2)
  merit_ref, upa_inside, upa_buffered   copied from lake_outline_test.csv
  lake_km2         area of the PLD outline (degrees converted with 111 km per degree and the cosine of latitude)
  own_area, nextdown, n_inflows, inflows   the node's own area, its downstream node, and its inflows with their
                   upstream areas (first 8)
  holder, holder_upa   the node named in the item's proposed fix, and its upstream area
  gdw, gdw_catch_max   GDW dams within 0.015 degrees (about 1.5 km) of the outline, with CATCH_SKM, and the largest
                   CATCH_SKM. CATCH_SKM is the HydroSHEDS catchment area at the dam, a reference independent of MERIT.
  vd_target        the lake is kept as a SWOT version-D target after the HALF filter and its coverage rule
  test, train      the lake is in the test or training node list

Panels: grey shading = log10 of MERIT upstream area; grey lines = sub-basins; blue = the lake node; orange = the holder
node; dashed green = the node downstream; cyan = the lake outline; dotted = other PLD lakes; red triangles = GDW dams.

Usage
  python lake_evidence_v35.py

Notes
  The script only reads its inputs. Several of them come from the review of graph v3.4 and are not rebuilt by this
  repository, which is why the resulting table is shipped.
  The output column `sub` does not hold the item's sub-group: `it.sub` returns a pandas method, not the column, so
  the column contains that method's text. make_decisions_v35.py, which reads this table, does not use the column.
"""
from swotgat.paths import WORK
import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyogrio
import rasterio
import shapely
from rasterio.windows import from_bounds

R = Path(f"{WORK}")
V3 = R/"CONUS_storage_fork/eval/graph_v3"
OUT = V3/"v35_work"; (OUT/"panels").mkdir(parents=True, exist_ok=True)
UPA = R/"CONUS_storage_fork/eval/graph_review/_merit_upa_conus.vrt"
items = pd.read_csv(V3/"review_v34/items.csv", dtype={"node": str})
items = items[items.group.isin(["refov", "edge", "above", "inflow"])].copy()
print(len(items), "lake items", flush=True)

s4 = gpd.read_parquet(V3/"v34/subbasins_conus_v34.parquet", columns=["geometry", "area_km2", "uparea_km2", "nextdown", "node_type"])
s4.index = s4.index.astype(str)
J = nx.readwrite.json_graph.node_link_graph(json.load(open(V3/"v34/graphs_conus_v34.json")), edges="edges")
s4 = s4.loc[s4.index.isin(set(J.nodes))]
pld = gpd.read_file(R/"LakeSPvC_ts_for_Ted/SWOT_PLD_v106_reservoirs.gdb")
pld["lake_id"] = pld.lake_id.astype("int64").astype(str)
pld = pld.drop_duplicates("lake_id").set_index("lake_id").to_crs("EPSG:4326")
lt = pd.read_csv(V3/"bothwrong/lake_outline_test.csv", dtype={"node": str}).set_index("node")
vd = pd.read_csv(R/"CONUS_storage_fork/eval/half_vd_review/halfexact_lakes_vD.csv", dtype={"lake_id": str})
vd_kept = set(vd.loc[vd.kept.astype(bool), "lake_id"])
test = set(pd.read_csv(R/"CONUS/basin_lists/storage/test_nodes.csv", dtype=str).subbasin)
train = set(pd.read_csv(R/"CONUS/basin_lists/storage/train_nodes_all.csv", dtype=str).iloc[:, -1])
ds = rasterio.open(UPA)

rows, panels = [], []
for _, it in items.iterrows():
    n = it.node
    lake = pld.loc[n] if n in pld.index else None
    geom = lake.geometry if lake is not None else s4.geometry.get(n)
    fix = json.loads(it.fix)
    holder = (fix.get("node") or "").rstrip(";") or None
    b = geom.bounds; pad = max(b[2] - b[0], b[3] - b[1]) * 0.5 + 0.02
    bb = (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)
    gdw = pyogrio.read_dataframe(R/"GDW_v1_0/GDW_v1_0_shp/GDW_barriers_v1_0.shp", bbox=bb,
                                 columns=["GDW_ID", "DAM_NAME", "RES_NAME", "RIVER", "CATCH_SKM", "AREA_SKM", "CAP_MCM"])
    gdw = gdw.to_crs("EPSG:4326")
    near = gdw[gdw.geometry.distance(geom) < 0.015].copy()
    near["d_km"] = near.geometry.distance(geom) * 111
    gstr = "; ".join(f"{r.DAM_NAME or r.RES_NAME or r.GDW_ID} catch {r.CATCH_SKM:.1f} (d {r.d_km:.2f} km)" for r in near.itertuples())
    gr = float(s4.uparea_km2.get(n, np.nan))
    nxt = next(iter(J.successors(n)), None) if n in J else None
    preds = list(J.predecessors(n)) if n in J else []
    rows.append(dict(key=it.key, group=it.group, sub=it.sub, node=n, name=it["name"], graph_v34=gr,
                     merit_ref=lt.merit_ref.get(n), upa_inside=lt.upa_inside.get(n), upa_buffered=lt.upa_buffered.get(n),
                     lake_km2=float(lake.geometry.area * 111 * 111 * math.cos(math.radians(geom.centroid.y))) if lake is not None else np.nan,
                     own_area=float(s4.area_km2.get(n, np.nan)), nextdown=nxt, n_inflows=len(preds),
                     inflows=";".join(f"{p}:{s4.uparea_km2.get(p, np.nan):.1f}" for p in preds[:8]),
                     holder=holder, holder_upa=float(s4.uparea_km2.get(holder, np.nan)) if holder else np.nan,
                     gdw=gstr, gdw_catch_max=float(near.CATCH_SKM.max()) if len(near) else np.nan,
                     vd_target=n in vd_kept, test=n in test, train=n in train))
    # panel background: MERIT upstream area in the same window
    win = from_bounds(*bb, ds.transform)
    a = ds.read(1, window=win, boundless=True, fill_value=0).astype(float)
    panels.append((it, rows[-1], bb, a, geom, gdw, holder, nxt))

ev = pd.DataFrame(rows)
ev.to_csv(OUT/"lake_evidence.csv", index=False)
print(ev.drop(columns=["inflows"]).to_string(), flush=True)

# map panels, six lakes per sheet
for k in range(0, len(panels), 6):
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    for ax, (it, r, bb, a, geom, gdw, holder, nxt) in zip(axs.flat, panels[k:k + 6]):
        ax.imshow(np.log10(np.clip(a, 0.01, None)), extent=(bb[0], bb[2], bb[1], bb[3]), cmap="Greys", vmin=-1, vmax=4, origin="upper")
        box = shapely.box(*bb)
        sub = s4[s4.intersects(box)]
        sub.boundary.plot(ax=ax, color="0.55", lw=0.6)
        if r["node"] in s4.index: s4.loc[[r["node"]]].plot(ax=ax, color="tab:blue", alpha=0.35)
        if holder and holder in s4.index: s4.loc[[holder]].boundary.plot(ax=ax, color="tab:orange", lw=2)
        if nxt and nxt in s4.index: s4.loc[[nxt]].boundary.plot(ax=ax, color="tab:green", lw=1.2, ls="--")
        others = pld[pld.intersects(box)]
        others.boundary.plot(ax=ax, color="c", lw=0.6, ls=":")
        gpd.GeoSeries([geom]).boundary.plot(ax=ax, color="cyan", lw=1.6)
        if len(gdw): gdw.plot(ax=ax, color="red", marker="^", markersize=40)
        ax.set_xlim(bb[0], bb[2]); ax.set_ylim(bb[1], bb[3])
        ax.set_title(f"{it.key} {str(it['name'])[:28]}\ngraph {r['graph_v34']:.1f} ref {r['merit_ref']:.1f} buf {r['upa_buffered']:.1f} GDW {r['gdw_catch_max']:.1f}"
                     f"{' vD' if r['vd_target'] else ''}{' TEST' if r['test'] else ''}", fontsize=9)
        ax.tick_params(labelsize=6)
    for ax in axs.flat[len(panels[k:k + 6]):]: ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT/"panels"/f"lakes_{k // 6:02d}.png", dpi=70); plt.close(fig)
print("done", flush=True)
