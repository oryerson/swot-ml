"""Build the sub-basin table of graph v3.5 by applying the reviewed corrections to graph v3.4 as routing edits.

Graph v3.5 is the graph the paper uses. A review of graph v3.4 flagged lakes and gauges whose sub-basin did not match
the river network. The rulings are in decisions_v35.csv (written by make_decisions_v35.py), one row per item, with the
operation in column `op`. This script applies the rows with op == "conform" (lakes) and op == "passthrough" (gauges).
The corrections are routing edits. No node is added or removed, and no gauge or lake point is moved: the locations
are observations, so only the network around them is corrected.

Reads (relative to $SWOTGAT_WORK/CONUS_storage_fork/eval/graph_v3 unless a full path is given)
  --base-subs   sub-basin table of graph v3.4 (GeoParquet, one row per node)
  --base-graph  graph JSON of graph v3.4 (networkx node-link format)
  --decisions   the rulings (default v35_work/decisions_v35.csv)
  $SWOTGAT_WORK/LakeSPvC_ts_for_Ted/SWOT_PLD_v106_reservoirs.gdb   lake outlines of the SWOT Prior Lake Database (PLD)
  $SWOTGAT_WORK/CONUS/merit_rasters/dir/<pfaf2>.vrt, upg/<pfaf2>.vrt   MERIT-Hydro flow direction (D8) and flow
                accumulation, the same rasters the delineator uses; <pfaf2> is the first two characters of the node id

Writes (in <out-dir>, default v35; inputs are only read)
  subbasins_conus_v35.parquet   the base table with geometry, area_km2, uparea_km2 and nextdown updated; all other
                                columns and the schema are unchanged. The script refuses to overwrite this file.
  edits_v35.csv                 one row per re-pointed link (lake, node, old_nextdown, new_nextdown, why)
  conform_v35.csv               one row per conformed lake (status, outlet, catchment area, receiver, cut and spill)
  uparea_changes_v35.csv        nodes whose upstream area changed by more than 0.05 km2
  summary_v35.json              counts

Usage
  python apply_v35.py --base-subs v34/subbasins_conus_v34.parquet --base-graph v34/graphs_conus_v34.json --out-dir v35
  Then build the graph JSON from the new table with build_conus_graphs_v2.py.

Method: "conform" (lakes)
The lake node is made to hold exactly what drains to the lake's outlet on the MERIT flow-direction grid. The outlet
cell o is the cell with the largest flow accumulation inside the lake's PLD outline (the delineator's own rule,
find_reservoir_outlet). C is the set of all cells that drain to o (D8 trace upstream from o). The rasters are read in
a window around the lake (0.15 degrees of padding, enlarged by 2.5x up to twice if C touches the window edge).
  * upstream node = a node whose outlet cell (largest accumulation in its polygon) lies in C. It is kept whole. If its
                    nextdown is neither the lake nor another upstream node, it is re-pointed to the lake.
  * cut node      = any other node that overlaps C by at least 0.02 km2. It loses the overlap to the lake. Gauge nodes
                    may be cut. Another lake node may be cut only if it is the receiver (its own upstream total then
                    stays the same); otherwise an overlap of 0.1 km2 or more aborts this lake, and smaller slivers
                    (raster and polygon edge effects) are left alone. A cut that would leave a node under 0.05 km2
                    also aborts the lake.
  * spill         = the part of the old lake polygon outside C. Each piece of at least 0.02 km2 goes to the node its
                    water reaches (D8 trace downstream).
  * old inflow    = a node routed into the lake whose outlet is not in C. It is re-pointed to the node its water
                    reaches, or to the receiver if that cannot be traced.
  * receiver      = the node that holds the first cell downstream of o that is outside C and outside the old lake
                    polygon. It becomes the lake's new nextdown and must be in the same basin.
A lake that fails any check is skipped with status "aborted: <reason>" in conform_v35.csv; the other lakes continue.

Method: "passthrough" (gauges)
For a gauge whose node sits on a side channel beside the river it measures, while the main stem enters the next node
down (D): the main-stem inflow of D (its inflow with the largest uparea_km2, other than the gauge node) is routed
through the gauge node. No polygon changes.

Method: areas and accumulation
Polygons are cleaned like the delineator's remainders (buffer by minus, then plus, half a raster cell). area_km2 of
each changed polygon is computed in the Mollweide projection (ESRI:54009, the delineator's projection). uparea_km2 is
re-accumulated over the whole graph (own area plus everything upstream). The script stops if the edits create a cycle.

In the paper's run: 38 lakes reshaped, 4 gauges made pass-through, 13 links re-pointed.
"""
from swotgat.paths import WORK
import argparse
import json
import time
from collections import deque
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyproj
import rasterio
import shapely
from rasterio.features import geometry_mask, shapes
from rasterio.windows import from_bounds, Window
from shapely.ops import transform as stransform, unary_union

R = Path(f"{WORK}")
V3 = R/"CONUS_storage_fork/eval/graph_v3"
DIR, UPG = R/"CONUS/merit_rasters/dir", R/"CONUS/merit_rasters/upg"
RES = 1/1200
OFF = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}
MOLL = pyproj.Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True).transform
MIN_PIECE_KM2 = 0.02
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--base-subs", required=True); ap.add_argument("--base-graph", required=True)
ap.add_argument("--decisions", default="v35_work/decisions_v35.csv"); ap.add_argument("--out-dir", default="v35")
A = ap.parse_args()
OUT = V3/A.out_dir; OUT.mkdir(parents=True, exist_ok=True)
for f in ("subbasins_conus_v35.parquet",):
    assert not (OUT/f).exists(), f"{OUT/f} exists; refusing to overwrite"

subs = gpd.read_parquet(V3/A.base_subs); subs.index = subs.index.astype(str)
J0 = nx.readwrite.json_graph.node_link_graph(json.load(open(V3/A.base_graph)), edges="edges")
nd = {n: (None if pd.isna(v) else str(v)) for n, v in subs.nextdown.items()}
geom = dict(subs.geometry.items())
pld = gpd.read_file(R/"LakeSPvC_ts_for_Ted/SWOT_PLD_v106_reservoirs.gdb")
pld["lake_id"] = pld.lake_id.astype("int64").astype(str)
pld = pld.to_crs("EPSG:4326").dissolve("lake_id")
dec = pd.read_csv(V3/A.decisions, dtype={"node": str})
todo = dec[dec.op == "conform"].node.tolist()
log(len(todo), "lakes to conform")
km2 = lambda g: stransform(MOLL, g).area / 1e6 if g is not None and not g.is_empty else 0.0
clean = lambda g: g.buffer(-RES/2).buffer(RES/2) if g is not None and not g.is_empty else g
sidx = subs.sindex


class Win:
    """A window of the MERIT flow-direction and flow-accumulation rasters around one lake.

    Attributes d and u hold the D8 flow direction and the flow accumulation, T the affine transform of the
    window, and H and W its height and width in cells. The methods are the D8 helpers used below; (r, c) are
    row and column in the window.
    """

    def __init__(self, pf, bb):
        """Read both rasters in the window.

        pf is the two-digit Pfafstetter region (names the VRT files); bb is the lon/lat bounding box
        (west, south, east, north). Cells outside the rasters are filled with 0.
        """
        self.d_ds, self.u_ds = rasterio.open(DIR/f"{pf}.vrt"), rasterio.open(UPG/f"{pf}.vrt")
        w = from_bounds(*bb, self.d_ds.transform).round_offsets().round_lengths()
        self.w = w; self.T = self.d_ds.window_transform(w)
        self.d = self.d_ds.read(1, window=w, boundless=True, fill_value=0)
        self.u = self.u_ds.read(1, window=w, boundless=True, fill_value=0).astype("float64")
        self.H, self.W = self.d.shape

    def rc(self, x, y):
        """Return the (row, column) of the window cell that contains the point (x, y) in lon/lat."""
        c, r = ~self.T * (x, y); return int(np.floor(r)), int(np.floor(c))

    def xy(self, r, c):
        """Return the lon/lat (x, y) of the centre of cell (r, c)."""
        return self.T * (c + 0.5, r + 0.5)

    def mask(self, g):
        """Return the boolean cell mask of geometry g (True inside), with the shape of the window."""
        return geometry_mask([g], out_shape=self.d.shape, transform=self.T, invert=True)

    def outlet(self, m):
        """Return the outlet (r, c) of cell mask m: its cell with the largest flow accumulation.

        Return None if the mask is empty.
        """
        if not m.any(): return None
        u = np.where(m, self.u, -1); return np.unravel_index(int(np.argmax(u)), u.shape)

    def upstream(self, r0, c0):
        """Trace D8 upstream from cell (r0, c0).

        Return (m, edge): m is the mask of every cell that drains to (r0, c0), the cell itself included; edge is
        True if that catchment touches the window border.
        """
        m = np.zeros(self.d.shape, bool); m[r0, c0] = True; q = deque([(r0, c0)]); edge = False
        while q:
            r, c = q.popleft()
            if r in (0, self.H - 1) or c in (0, self.W - 1): edge = True
            for k, (dr, dc) in OFF.items():
                rr, cc = r - dr, c - dc
                if 0 <= rr < self.H and 0 <= cc < self.W and not m[rr, cc] and self.d[rr, cc] == k:
                    m[rr, cc] = True; q.append((rr, cc))
        return m, edge

    def down(self, r, c, stop, nmax=20000):
        """Follow D8 downstream from (r, c) until stop(r, c) is False.

        Return that cell as (r, c), or None if the path leaves the window, reaches a sink, or takes more than
        nmax steps.
        """
        for _ in range(nmax):
            k = int(self.d[r, c])
            if k not in OFF: return None
            r, c = r + OFF[k][0], c + OFF[k][1]
            if not (0 <= r < self.H and 0 <= c < self.W): return None
            if not stop(r, c): return (r, c)
        return None

    def poly(self, m):
        """Return the polygon (union of cell squares, in lon/lat) that covers the True cells of mask m."""
        return unary_union([shapely.geometry.shape(s) for s, v in shapes(m.astype("uint8"), mask=m, transform=self.T) if v == 1])


def node_at(x, y, exclude):
    """Return the id of the first node whose polygon intersects the point (x, y), or None if there is none.

    Nodes in `exclude` and nodes with an empty polygon are ignored.
    """
    p = shapely.Point(x, y)
    for i in sidx.query(p, predicate="intersects"):
        n = subs.index[i]
        if n not in exclude and geom[n] is not None and not geom[n].is_empty and geom[n].intersects(p):
            return n
    return None


edits, rows = [], []
for L in todo:
    row = dict(node=L, status="ok")
    try:
        lake = pld.geometry[L]; oldL = geom[L]
        b = lake.bounds; pad = 0.15
        for attempt in range(3):
            bb = (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)
            W = Win(L[:2], bb)
            o = W.outlet(W.mask(lake))
            C, edge = W.upstream(*o)
            if not edge: break
            pad *= 2.5
        assert not edge, "catchment reaches the window edge"
        Cp = W.poly(C)
        lat = W.xy(*o)[1]
        cellkm2 = (RES * 111.32 * np.cos(np.radians(lat))) * (RES * 110.57)
        row.update(outlet_x=W.xy(*o)[0], outlet_y=lat, C_km2=float(C.sum() * cellkm2), upg_o=float(W.u[o]))
        near = [subs.index[i] for i in sidx.query(shapely.box(*bb), predicate="intersects")]
        win_box = shapely.box(*bb)
        ups, cut = [], []
        for X in near:
            if X == L or geom[X] is None or geom[X].is_empty: continue
            gX = geom[X]
            if not gX.intersects(Cp): continue
            ov = km2(gX.intersection(Cp))
            if ov < MIN_PIECE_KM2: continue
            oX = W.outlet(W.mask(gX)) if win_box.contains(gX) else None
            if oX is not None and C[oX]: ups.append(X)
            else: cut.append((X, ov))
        upset = set(ups)
        # receiver of the lake: the first node below the outlet, outside C and the old lake polygon
        # (stepping over gaps between polygons)
        inLold = W.mask(oldL) if oldL is not None and not oldL.is_empty else np.zeros_like(C)
        cell = W.down(*o, stop=lambda r, c: C[r, c] or inLold[r, c] or node_at(*W.xy(r, c), exclude={L}) is None)
        assert cell is not None, "no receiver below the outlet in the window"
        Rn = node_at(*W.xy(*cell), exclude={L})
        assert Rn is not None and Rn not in upset, f"bad receiver {Rn}"
        # another lake node may be cut only where it is the receiver (its own upstream total is then unchanged); slivers of
        # other lake nodes (< 0.1 km2, raster/polygon edge effects) are left alone
        keep = []
        for X, ov in cut:
            if bool(subs.is_reservoir.get(X, False)) and X != Rn:
                if ov < 0.1: continue
                raise RuntimeError(f"would cut lake node {X} ({ov:.2f} km2)")
            keep.append((X, ov))
        cut = keep
        assert subs.outlet_id[Rn] == subs.outlet_id[L], "receiver in another basin"
        # new polygons
        upoly = unary_union([geom[X] for X in ups]) if ups else shapely.Polygon()
        newL = clean(Cp.difference(upoly))
        newgeo = {L: newL}
        for X, ov in cut:
            rem = clean(geom[X].difference(Cp))
            if km2(rem) < 0.05: raise RuntimeError(f"cut node {X} would vanish")
            newgeo[X] = rem
        spill = clean(oldL.difference(Cp)) if oldL is not None and not oldL.is_empty else None
        spill_to = {}
        if spill is not None and not spill.is_empty:
            parts = list(getattr(spill, "geoms", [spill]))
            for pc in parts:
                if km2(pc) < MIN_PIECE_KM2: continue
                m = W.mask(pc); op_ = W.outlet(m)
                tgt = Rn
                if op_ is not None:
                    cc = W.down(*op_, stop=lambda r, c: m[r, c] or inLold[r, c] and not C[r, c])
                    if cc is not None:
                        tgt = L if C[cc] else (node_at(*W.xy(*cc), exclude={L}) or Rn)
                if tgt == L: newgeo[L] = unary_union([newgeo[L], pc]); continue
                if tgt in upset: tgt = Rn
                base = newgeo.get(tgt, geom[tgt])
                newgeo[tgt] = unary_union([base, pc]); spill_to[tgt] = spill_to.get(tgt, 0) + km2(pc)
        # routing
        for X in ups:
            if nd[X] not in upset and nd[X] != L:
                edits.append(dict(lake=L, node=X, old_nextdown=nd[X], new_nextdown=L, why="drains to the lake outlet (MERIT D8)")); nd[X] = L
        for p in list(J0.predecessors(L)):
            if p in upset: continue
            gp = geom[p]; tgt = Rn
            if gp is not None and win_box.contains(gp):
                mp = W.mask(gp); op_ = W.outlet(mp)
                if op_ is not None:
                    cc = W.down(*op_, stop=lambda r, c: mp[r, c])
                    if cc is not None and C[cc]: continue  # MERIT: p does drain into the lake
                    if cc is not None:
                        t2 = node_at(*W.xy(*cc), exclude={L, p})
                        if t2 is not None and t2 not in upset: tgt = t2
            if nd[p] == L:
                edits.append(dict(lake=L, node=p, old_nextdown=L, new_nextdown=tgt, why="does not reach the lake (MERIT D8); routed past it")); nd[p] = tgt
        if nd[L] != Rn:
            edits.append(dict(lake=L, node=L, old_nextdown=nd[L], new_nextdown=Rn, why="lake outlet drains here (MERIT D8)")); nd[L] = Rn
        for X, g in newgeo.items(): geom[X] = g
        row.update(receiver=Rn, n_upstream=len(ups), upstream=";".join(ups[:20]), cut=";".join(f"{X}:{ov:.2f}" for X, ov in cut),
                   spill=";".join(f"{k}:{v:.2f}" for k, v in spill_to.items()), newL_own_km2=km2(newgeo[L]))
    except Exception as e:  # noqa: BLE001 - one bad lake must not stop the others; reported in conform_v35.csv
        row["status"] = f"aborted: {e}"
    rows.append(row); log(L, row["status"], row.get("C_km2"), row.get("receiver"))

# pass-through (gauges whose node sits on a side channel beside the river they measure, with the main stem entering
# the next node down): the main-stem inflow M of that node is routed through the gauge node; no polygon changes.
for r in dec[dec.op == "passthrough"].itertuples():
    g = r.node; D = nd[g]
    preds = [p for p, d in nd.items() if d == D and p != g]
    M = max(preds, key=lambda p: float(subs.uparea_km2[p]))
    edits.append(dict(lake="", node=M, old_nextdown=D, new_nextdown=g, why=f"main stem routed through gauge node {g} (side channel in MERIT)"))
    nd[M] = g
    log("passthrough", g, "<-", M, f"{float(subs.uparea_km2[M]):.1f}")

# areas, accumulation, checks
changed = {n for n in geom if geom[n] is not subs.geometry[n]}
area = subs.area_km2.copy()
for n in changed: area[n] = km2(geom[n])
G = nx.DiGraph(); G.add_nodes_from(subs.index)
G.add_edges_from((n, d) for n, d in nd.items() if d is not None and d in G)
assert nx.is_directed_acyclic_graph(G), "cycle after edits"
up = {}
for n in nx.topological_sort(G): up[n] = float(area[n]) + sum(up[p] for p in G.predecessors(n))
# The base table's uparea_km2 should equal this accumulation over the base graph, so nodes away from the edits keep
# their value. uparea_changes_v35.csv (below) lists every node that differs by more than 0.05 km2.
# The table is rewritten with pyarrow so that the schema and the untouched columns stay exactly as in the base file.
tbl = pq.read_table(V3/A.base_subs)
df = tbl.to_pandas()
idcol = "id" if "id" in df.columns else df.columns[0]
ids = (df[idcol] if idcol in df.columns and idcol != "geometry" else df.index).astype(str).tolist()
new_nd = [nd[i] for i in ids]; new_up = [up[i] for i in ids]; new_area = [float(area[i]) for i in ids]
geoms = [shapely.to_wkb(geom[i]) if i in changed else w for i, w in zip(ids, df["geometry"])]
cols = {c: tbl.column(c) for c in tbl.column_names}
cols["geometry"] = pa.array(geoms, type=tbl.schema.field("geometry").type)
cols["area_km2"] = pa.array(new_area, type=tbl.schema.field("area_km2").type)
cols["uparea_km2"] = pa.array(new_up, type=tbl.schema.field("uparea_km2").type)
cols["nextdown"] = pa.array(new_nd, type=tbl.schema.field("nextdown").type)
out = pa.table([cols[c] for c in tbl.column_names], schema=tbl.schema)
pq.write_table(out, OUT/"subbasins_conus_v35.parquet")
ev = pd.DataFrame(edits); ev.to_csv(OUT/"edits_v35.csv", index=False)
cf = pd.DataFrame(rows)
cf["uparea_v34"] = cf.node.map(subs.uparea_km2); cf["uparea_v35"] = cf.node.map(up)
cf.to_csv(OUT/"conform_v35.csv", index=False)
dup = {n: abs(up[n] - float(subs.uparea_km2[n])) for n in subs.index}
moved = {n: (float(subs.uparea_km2[n]), up[n]) for n, v in dup.items() if v > 0.05}
pd.DataFrame([(n, a, b, bool(subs.is_gauge[n]), bool(subs.is_reservoir[n])) for n, (a, b) in moved.items()],
             columns=["node", "uparea_v34", "uparea_v35", "is_gauge", "is_reservoir"]).to_csv(OUT/"uparea_changes_v35.csv", index=False)
summ = dict(lakes=len(todo), ok=int((cf.status == "ok").sum()), aborted=cf.loc[cf.status != "ok", ["node", "status"]].values.tolist(),
            edits=len(ev), polygons_changed=len(changed), uparea_changed=len(moved),
            gauges_uparea_changed=int(sum(bool(subs.is_gauge[n]) for n in moved)),
            lakes_uparea_changed=int(sum(bool(subs.is_reservoir[n]) for n in moved)))
json.dump(summ, open(OUT/"summary_v35.json", "w"), indent=1)
log(summ)
log(cf[["node", "status", "C_km2", "uparea_v34", "uparea_v35", "receiver", "cut", "spill"]].to_string())
