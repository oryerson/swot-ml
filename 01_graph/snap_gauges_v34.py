"""Re-snap USGS gauges to the MERIT-Hydro channel whose upstream area matches the USGS drainage area.

A gauge placed on the wrong channel (for example a small side channel next to the river it measures) gets a sub-basin
of the wrong size. This script finds such gauges by comparing the upstream area they have in the previous graph
(v3.3) with the USGS drainage area, and moves them to a nearby MERIT cell that fits. It writes new gauge tables for
the delineation of graph v3.4. All other gauges are left exactly as they are.

Reads (W = $SWOTGAT_WORK, V3 = W/CONUS_storage_fork/eval/graph_v3; inputs are only read)
  W/CONUS/basin_lists/gauges/<basin>.csv           the gauge tables from build_conus_basin_list.py
  W/global_gauges_data/usgs/site_metadata/data.parquet   USGS site location and total drainage area
  V3/bothwrong/usgs_contrib_area.csv               USGS contributing drainage area (contrib_drain_area_km2)
  from the previous graph version, v3.3:
    V3/acceptance/v33/gauges.csv                   upstream area of each gauge in graph v3.3 (graph_km2)
    V3/v33/subbasins_conus_v33.parquet, V3/v33/graphs_conus_v33.json
    V3/bothwrong/classes_v33.csv                   review class of each gauge (cls_final), for the summary only
  W/CONUS_storage_fork/eval/graph_review/_merit_upa_conus.vrt   MERIT-Hydro upstream area (km2) mosaic
  $SWOTGAT_NAS_ICE/data/MERIT_Basins/MERIT_Hydro_v07_Basins_v01_bugfix/pfaf_level_01/
                                                   MERIT-Basins catchments (cat_pfaf_7) and river lines (riv_pfaf_7)

Writes
  V3/gauges_v34/<basin>.csv     the gauge tables for the delineation: the input columns plus snap_min_area_km2;
                                lng, lat, position and COMID are changed for moved gauges only
  V3/bothwrong/snap_report.csv  every gauge considered, with the best cell at every search radius
  V3/bothwrong/snap_summary.json  number of gauges moved and fixed at every radius (a sensitivity test of the radius)

Usage
  python snap_gauges_v34.py [--radius-km 2]     the radius must be one of 0.5, 1, 2 or 5; the paper's graph uses 2

Method
  1. Mismatch of an area a against the USGS references: min over the references of |log(a / reference)|. The
     references are the USGS total and contributing drainage areas, so a gauge counts as matching if it fits either.
  2. Candidates: USGS gauges whose graph v3.3 upstream area is more than 5% off both references.
     A gauge directly below a lake that supplies at least 80% of its upstream area (a gauge at a dam, e.g. Structure
     P-11 below Lake Hancock, Florida) is not a candidate: there USGS and MERIT disagree about the lake itself.
     Gauges of other agencies are not candidates.
  3. Search: among the MERIT cells within the radius of the USGS site location, take the cell with the smallest
     mismatch. Cells within 0.005 (TIE_LOG) of the smallest mismatch count as ties, and the nearest one wins.
  4. Move rule: the gauge moves to that cell only if the cell's mismatch is less than half the current one and
     smaller by more than log(1.05) (about 5 percentage points), and the cell lies in the same model basin.
  5. A moved gauge gets: lng, lat = the cell centre; COMID = the MERIT-Basins catchment that holds the cell;
     position = the fraction along that catchment's river line, measured from the upstream end as in the original
     tables (1 minus the normalized distance along the stored line); snap_min_area_km2 = 0.8 times the cell's
     upstream area. The delineator snaps a gauge to the nearest cell above an accumulation threshold inside its
     catchment; with snap_min_area_km2 it keeps the gauge on the chosen channel instead of a 1 km2 side channel next
     to it (the case of the Pootatuck River).
  6. Flags in snap_report.csv: a move of more than 1 km, or a remaining mismatch above 10%, is marked "review";
     a move that cannot be made (cell outside the gauge's basin, or no MERIT catchment or river line) is marked
     "not applied".

Notes
  The script compares against graph v3.3, which this repository does not rebuild (see docs/KNOWN_LIMITS.md). Its
  output, the 102 gauge tables of the paper's run, is shipped in 01_graph/inputs/gauges_v34/.
"""
from swotgat.paths import NAS_ICE, WORK
import argparse
import glob
import json
import os
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
import shapely
from rasterio.windows import from_bounds
from shapely.geometry import Point

R = Path(f"{WORK}")
V3 = R/"CONUS_storage_fork/eval/graph_v3"
UPA = R/"CONUS_storage_fork/eval/graph_review/_merit_upa_conus.vrt"
MB1 = Path(f"{NAS_ICE}/data/MERIT_Basins/MERIT_Hydro_v07_Basins_v01_bugfix/pfaf_level_01")
RADII = (0.5, 1.0, 2.0, 5.0)
TIE_LOG = 0.005
ap = argparse.ArgumentParser(); ap.add_argument("--radius-km", type=float, default=2.0); A = ap.parse_args()
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

M = pd.read_parquet(R/"global_gauges_data/usgs/site_metadata/data.parquet", columns=["site_id", "area", "latitude", "longitude"]).drop_duplicates("site_id").set_index("site_id")
CA = pd.read_csv(V3/"bothwrong/usgs_contrib_area.csv").set_index("site_id")
G33 = pd.read_csv(V3/"acceptance/v33/gauges.csv", dtype={"node": str}).set_index("node")
CL = pd.read_csv(V3/"bothwrong/classes_v33.csv", dtype={"node": str}).set_index("node")
tabs = {Path(f).stem: pd.read_csv(f) for f in sorted(glob.glob(str(R/"CONUS/basin_lists/gauges/*.csv")))}
allg = pd.concat([t.assign(gbasin=b) for b, t in tabs.items()], ignore_index=True)
log(f"{len(allg):,} gauges in {len(tabs)} basin tables")


def err(a, refs):
    """Return the mismatch of area a against the reference areas refs: the smallest |log(a / ref)|.

    Non-positive or missing values are skipped; the result is infinite if no comparison is possible.
    """
    v = [abs(np.log(a / r)) for r in refs if r and r > 0 and a and a > 0]
    return min(v) if v else np.inf


# Gauges at a dam: gauge nodes of graph v3.3 with an inflowing lake node that carries at least 80% of the gauge's
# upstream area (areas accumulated over the v3.3 graph). They are never moved.
import json as _json, networkx as _nx
_s =pd.read_parquet(V3/"v33/subbasins_conus_v33.parquet", columns=["area_km2", "node_type"]); _s.index = _s.index.astype(str)
_J = _nx.readwrite.json_graph.node_link_graph(_json.load(open(V3/"v33/graphs_conus_v33.json")), edges="edges")
_acc = {}
for _n in _nx.topological_sort(_J): _acc[_n] = float(_s.area_km2.get(_n, 0) or 0) + sum(_acc[_p] for _p in _J.predecessors(_n))
at_dam = {g for g in _J if _s.node_type.get(g) == "gauge" and any(_s.node_type.get(p) == "reservoir" and _acc[p] >= 0.8 * _acc[g] for p in _J.predecessors(g))}
log(f"{len(at_dam):,} gauges sit directly below a lake that supplies >= 80% of their area: not moved")
cand = []
for r in allg.itertuples():
    if r.id not in M.index or r.id not in G33.index or r.id in at_dam: continue
    refs = [M.area.get(r.id), CA.contrib_drain_area_km2.get(r.id) if r.id in CA.index else None]
    refs = [x for x in refs if x is not None and np.isfinite(x) and x > 0]
    cur = err(G33.graph_km2[r.id], refs)
    if cur <= np.log(1.05): continue
    cand.append((r.Index, r.id, r.gbasin, refs, cur))
log(f"{len(cand):,} USGS gauges more than 5% off both references in v3.3")

# For each candidate, read the MERIT upstream area in a window of the largest radius around the USGS site and record
# the best-matching cell within every radius (dist = distance of each cell centre from the site, in km).
ds = rasterio.open(UPA)
rows = []
for k, gid, basin, refs, cur in cand:
    lat, lon = float(M.latitude[gid]), float(M.longitude[gid])
    dlat = max(RADII) / 110.57; dlon = max(RADII) / (111.32 * np.cos(np.radians(lat)))
    w = from_bounds(lon - dlon, lat - dlat, lon + dlon, lat + dlat, transform=ds.transform).round_offsets().round_lengths()
    a = ds.read(1, window=w, boundless=True, fill_value=-9999).astype(float); t = ds.window_transform(w)
    rr, cc = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    x = t.c + (cc + 0.5) * t.a; y = t.f + (rr + 0.5) * t.e
    dist = np.hypot((x - lon) * 111.32 * np.cos(np.radians(lat)), (y - lat) * 110.57)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.min(np.stack([np.abs(np.log(np.where(a > 0, a, np.nan) / ref)) for ref in refs]), axis=0)
    e = np.where(np.isfinite(e), e, np.inf)
    row = dict(gauge=gid, basin=basin, graph_v33=round(float(G33.graph_km2[gid]), 1), usgs_total=refs[0], usgs_contrib=refs[1] if len(refs) > 1 else None,
               err_now=round(float(cur), 4), cls_v33=CL.cls_final.get(gid), lat=lat, lon=lon)
    for rad in RADII:
        m = dist <= rad
        if not m.any() or not np.isfinite(e[m]).any():
            row.update({f"err_{rad}": None, f"dist_{rad}": None, f"upa_{rad}": None}); continue
        emin = e[m].min()
        ties = m & (e <= emin + TIE_LOG)
        i = np.unravel_index(np.argmin(np.where(ties, dist, np.inf)), dist.shape)
        row.update({f"err_{rad}": round(float(e[i]), 4), f"dist_{rad}": round(float(dist[i]), 3), f"upa_{rad}": round(float(a[i]), 1),
                    f"x_{rad}": float(x[i]), f"y_{rad}": float(y[i])})
    rows.append(row)
S = pd.DataFrame(rows)
log("windows read")

# ---- decisions for every radius (the sensitivity test) and the move list for RADIUS_KM
summ = {}
for rad in RADII:
    e1 = S[f"err_{rad}"]
    move = e1.notna() & (e1 < S.err_now / 2) & (e1 < S.err_now - np.log(1.05))
    S[f"move_{rad}"] = move
    summ[str(rad)] = dict(considered=len(S), moved=int(move.sum()), fixed_within_5pct=int((move & (e1 <= np.log(1.05))).sum()),
                         fixed_within_10pct=int((move & (e1 <= np.log(1.10))).sum()), moved_over_1km=int((move & (S[f"dist_{rad}"] > 1)).sum()),
                         by_class={c: int(((S.cls_v33 == c) & move).sum()) for c in ("A", "B", "C", "D", "E", "N")},
                         fixed_by_class={c: int(((S.cls_v33 == c) & move & (e1 <= np.log(1.10))).sum()) for c in ("A", "B", "C", "D", "E", "N")})
log(json.dumps(summ, indent=1))

# ---- new COMID and position for the moves at RADIUS_KM; check the cell lies in the same model basin
rad = A.radius_km
mv = S[S[f"move_{rad}"]].copy()
bb = (mv[f"x_{rad}"].min() - 0.1, mv[f"y_{rad}"].min() - 0.1, mv[f"x_{rad}"].max() + 0.1, mv[f"y_{rad}"].max() + 0.1)
cat = pyogrio.read_dataframe(MB1/"cat_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp", columns=["COMID"], bbox=bb)
tree = shapely.STRtree(cat.geometry.values)
sub = gpd.read_parquet(V3/"v33/subbasins_conus_v33.parquet", columns=["geometry", "outlet_id"])
stree = shapely.STRtree(sub.geometry.values)
pts = shapely.points(mv[f"x_{rad}"].values, mv[f"y_{rad}"].values)
ci = tree.query(pts, predicate="within"); comid = pd.Series(cat.COMID.values[ci[1]], index=ci[0]); comid = comid[~comid.index.duplicated()]
si = stree.query(pts, predicate="within"); sbas = pd.Series(sub.outlet_id.values[si[1]], index=si[0]); sbas = sbas[~sbas.index.duplicated()]
mv["new_comid"] = [int(comid.get(i)) if i in comid.index else None for i in range(len(mv))]
mv["cell_basin"] = [sbas.get(i) for i in range(len(mv))]
mv["ok_basin"] = mv.cell_basin == mv.basin
ids = ",".join(str(int(c)) for c in mv.new_comid.dropna().unique())
riv = pyogrio.read_dataframe(MB1/"riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp", columns=["COMID"], where=f"COMID IN ({ids})").set_index("COMID") if ids else None
def position(c, x, y):
    """Return the position of point (x, y) along the river line of MERIT-Basins catchment c.

    The position is 1 minus the normalized distance along the stored line, rounded to 6 decimals. Return None
    if the catchment has no river line.
    """
    if riv is None or c is None or c not in riv.index: return None
    return round(1.0 - riv.geometry[c].project(Point(x, y), normalized=True), 6)
mv["new_position"] = [position(c, x, y) for c, x, y in zip(mv.new_comid, mv[f"x_{rad}"], mv[f"y_{rad}"])]
mv["applied"] = mv.ok_basin & mv.new_comid.notna() & mv.new_position.notna()
mv["flag"] = np.where(~mv.applied, "not applied: " + np.where(~mv.ok_basin, "cell outside the gauge's model basin", "no MERIT catchment / river"),
                      np.where((mv[f"dist_{rad}"] > 1) | (mv[f"err_{rad}"] > np.log(1.10)), "review: " + np.where(mv[f"dist_{rad}"] > 1, "moved over 1 km", "") +
                               np.where((mv[f"dist_{rad}"] > 1) & (mv[f"err_{rad}"] > np.log(1.10)), "; ", "") + np.where(mv[f"err_{rad}"] > np.log(1.10), "still more than 10% off", ""), ""))
S = S.merge(mv[["gauge", "new_comid", "new_position", "cell_basin", "applied", "flag"]], on="gauge", how="left")
S.to_csv(V3/"bothwrong/snap_report.csv", index=False)
summ["applied_at_radius"] = dict(radius_km=rad, moves=int(mv.applied.sum()), not_applied=int((~mv.applied).sum()),
                                 flagged_for_review=int(mv.flag.str.startswith("review").sum()))
json.dump(summ, open(V3/"bothwrong/snap_summary.json", "w"), indent=1)
# ---- the v3.4 gauge tables
out = V3/"gauges_v34"; out.mkdir(exist_ok=True)
ap_ = mv[mv.applied].set_index("gauge")
nmov = 0
for b, t in tabs.items():
    t = t.copy(); t["snap_min_area_km2"] = np.nan
    for i, r in t.iterrows():
        if r.id in ap_.index:
            q = ap_.loc[r.id]
            t.loc[i, ["lng", "lat", "position", "COMID"]] = [round(q[f"x_{rad}"], 7), round(q[f"y_{rad}"], 7), q.new_position, int(q.new_comid)]
            t.loc[i, "snap_min_area_km2"] = round(0.8 * float(q[f"upa_{rad}"]), 3)
            nmov += 1
    t["COMID"] = t.COMID.astype("int64")
    t.to_csv(out/f"{b}.csv", index=False)
log(f"wrote {out} ({len(tabs)} tables, {nmov} gauges moved at {rad} km); {summ['applied_at_radius']}")
