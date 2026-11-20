"""
Simplified Watershed Delineation Module
========================================

Store all data in the graph to avoid synchronization issues.
Each node contains its polygon, river geometry, and all attributes.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import geopandas as gpd
import pandas as pd
import pyproj
from numpy import ceil, floor
from pathlib import Path
from tqdm import tqdm
from shapely import set_precision
from shapely.geometry import Point, LineString, Polygon
from shapely import STRtree
from shapely.ops import transform, unary_union
from typing import Dict, List, Tuple, Any

from .split_catchment_reservoirs import resolve_raster, split_catchment_raster
from .dataclasses_reservoirs import DelineationResult

MERIT_RES = 0.000833333  # 3 arc second resolution in degrees
TINY_AREA_THRESHOLD_KM2 = 0.5

# =============================================================================
# GAUGE UTILS
# =============================================================================
def load_gauges(csv_file: str, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Load gauge locations from CSV.

    CSV must have columns: id, lat, lng
    Optional column: basin (Pfafstetter basin code)
    """
    df = pd.read_csv(csv_file)

    required_cols = ["id", "lat", "lng", "outlet_id"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    geometry = [Point(lng, lat) for lng, lat in zip(df["lng"], df["lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)

    return gdf


def get_pfaf1_outlet_dict(
    gauges_gdf: gpd.GeoDataFrame,
) -> dict[int, dict[str, list[str]]]:
    """
    Identify which Pfafstetter basins contain the gauges.

    Args:
        gauges_gdf: GeoDataFrame with gauge points (must have 'id' column)

    Returns:
        Dict mapping {megabasin_id: {outlet_id: [gauge_ids]}}
    """
    gauges_gdf["pfaf1"] = (gauges_gdf["COMID"] // 1e7).astype(int)

    megabasin_dict = {}
    for megabasin, group in gauges_gdf.groupby("pfaf1")[["outlet_id", "id"]]:
        megabasin_dict[megabasin] = (
            group.groupby("outlet_id")["id"].apply(list).to_dict()
        )

    return megabasin_dict


def get_network_comids(outlet_comid: int, rivers: gpd.GeoDataFrame) -> set:
    """
    Find all COMIDs in the watershed network containing this outlet.

    Uses recursive network traversal to find all upstream catchments from the outlet.
    """

    def addnode(B: list, node_id):
        """Recursively assemble list of upstream unit catchments."""
        B.append(node_id)
        for up_field in ["up1", "up2", "up3", "up4"]:
            up = rivers[up_field].loc[node_id]
            if up != 0:
                addnode(B, up)

    network_comids = []
    addnode(network_comids, outlet_comid)

    return network_comids


# =============================================================================
# GRAPH BUILDING WITH GEOMETRIES
# =============================================================================


def build_graph_with_geometries(
    catchments: gpd.GeoDataFrame, rivers: gpd.GeoDataFrame
) -> nx.DiGraph:
    """
    Build graph with all geometries stored in nodes.

    Each node contains:
    - polygon: catchment geometry
    - river_geom: river line geometry
    - area_km2: catchment area
    - length_km: river length
    - original_comid: original catchment ID
    """
    G = nx.DiGraph()

    # Project to equal area for accurate area calculation
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)

    for comid in catchments.index:
        if comid not in rivers.index:
            continue

        catchment = catchments.loc[comid]
        river = rivers.loc[comid]

        # Calculate area in km²
        poly_projected = transform(transformer.transform, catchment.geometry)
        area_km2 = poly_projected.area / 1e6

        # Add node with complete data
        G.add_node(
            str(comid),  # Use string IDs for consistency
            polygon=catchment.geometry,
            river_geom=river.geometry,
            area_km2=area_km2,
            length_km=river["lengthkm"],
            node_type="original",
            is_gauge=False,
            original_comid=comid,
            nextdown=str(river["NextDownID"]) if river["NextDownID"] != "0" else None,
        )

    # Add edges based on river network
    for node_id in G.nodes():
        nextdown = G.nodes[node_id].get("nextdown")
        if nextdown and nextdown in G.nodes:
            G.add_edge(node_id, nextdown)

    return G


# =============================================================================
# GAUGE INSERTION WITH POLYGON SPLITTING
# =============================================================================


# Rule 5 (graph v3): when a catchment is split, a predecessor goes to the remainder only if its river clearly ends there.
SPLIT_ROUTE_TOL_DEG = 2 * MERIT_RES
# ... and (v3.1) only if MERIT D8 flow from its river end does not pass the lake outlet.
_D8_STEP = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}   # ESRI


def _passes_outlet(outlet_d8, start, bounds):
    """True if MERIT D8 flow from `start` passes within one cell of the lake outlet before leaving `bounds` (+0.01 deg);
    False if it does not; None if the raster cannot be read or the start cell has no flow direction."""
    try:
        import rasterio
        from rasterio.windows import from_bounds
        flow_dir_path, pfaf2_id, olng, olat = outlet_d8
        m = 0.01
        bounds = (min(bounds[0], start.x, olng), min(bounds[1], start.y, olat),
                  max(bounds[2], start.x, olng), max(bounds[3], start.y, olat))   # v3.2: the window holds the start point
        with rasterio.open(resolve_raster(flow_dir_path, pfaf2_id)) as ds:
            w = from_bounds(bounds[0] - m, bounds[1] - m, bounds[2] + m, bounds[3] + m,
                            transform=ds.transform).round_offsets().round_lengths()
            d = ds.read(1, window=w, boundless=True, fill_value=0)
            inv = ~ds.window_transform(w)
        c, r = inv * (start.x, start.y)
        oc, orow = inv * (olng, olat)
        r, c, orow, oc = int(r), int(c), int(orow), int(oc)
        H, Wd = d.shape
        for k in range(H * Wd):
            if abs(r - orow) <= 1 and abs(c - oc) <= 1:
                return True
            if not (0 <= r < H and 0 <= c < Wd):
                return None if k == 0 else False
            st = _D8_STEP.get(int(d[r, c]))
            if st is None:
                return None if k == 0 else False      # v3.2: no flow direction at the start point = cannot tell
            r += st[0]; c += st[1]
        return False
    except Exception:
        return None


def _line_parts(g):
    if g is None or g.is_empty:
        return []
    return [p for p in (g.geoms if hasattr(g, "geoms") else [g]) if p.geom_type == "LineString" and not p.is_empty]


def _assign_river_parts(first, second, upstream_poly, remainder_poly):
    """(v3.2) Each line part goes to the piece whose polygon holds more of it (ties: the upstream piece)."""
    up, down = [], []
    for p in _line_parts(first) + _line_parts(second):
        try:
            a, b = p.intersection(upstream_poly).length, p.intersection(remainder_poly).length
        except Exception:
            a, b = 1.0, 0.0
        (up if a >= b else down).append(p)
    mk = lambda parts: None if not parts else (parts[0] if len(parts) == 1 else unary_union(parts))
    return mk(up), mk(down)


def _pred_outlet(G: nx.DiGraph, node, near):
    """(v3.2) Where `node`'s water leaves it: the stored outlet of a gauge or reservoir node (the gauge location, the lake
    outlet cell), else the downstream end of its river."""
    nd = G.nodes[node]
    if (nd.get("is_gauge") or nd.get("is_reservoir")) and nd.get("lat") is not None and nd.get("lng") is not None:
        return Point(float(nd["lng"]), float(nd["lat"]))
    return _river_end(G, node, near)


def _river_end(G: nx.DiGraph, node, near):
    """Downstream end of `node`'s river: of all line ends, the one nearest `near` (the split catchment). MERIT-Basins lines
    start at their downstream end (sibling inflows share their first point), and a merged node's river is several lines,
    so the end is found by position, not by orientation."""
    geom = G.nodes[node].get("river_geom")
    if geom is None or geom.is_empty:
        return None
    parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    ends = [Point(xy) for p in parts if p.geom_type == "LineString" and len(p.coords) for xy in (p.coords[0], p.coords[-1])]
    return min(ends, key=lambda q: q.distance(near)) if ends else None


def _drains_into_remainder(G: nx.DiGraph, pred, upstream_poly, remainder_poly, outlet_d8=None) -> bool:
    if remainder_poly is None or remainder_poly.is_empty or upstream_poly is None or upstream_poly.is_empty:
        return False
    end = _pred_outlet(G, pred, upstream_poly.union(remainder_poly))
    if end is None:
        return False
    d_up, d_rem = end.distance(upstream_poly), end.distance(remainder_poly)
    if not (d_up > SPLIT_ROUTE_TOL_DEG and d_rem < d_up):
        return False
    # MERIT D8 check (v3.1): keep the original rewiring when the flow passes the lake outlet, or when it cannot be traced
    if outlet_d8 is None:
        return False
    return _passes_outlet(outlet_d8, end, upstream_poly.union(remainder_poly).bounds) is False


def insert_gauge_into_graph(
    G: nx.DiGraph,
    gauge_id: str,
    lat: float,
    lng: float,
    split_polygon: Polygon,
    remainder_polygon: Polygon,
    original_node_id: str,
    route_by_river_end: bool = False,
    outlet_d8: tuple = None,
) -> bool:
    """
    Insert a gauge by splitting a node's polygon.

    The gauge gets the upstream polygon, the original node keeps the downstream remainder.
    """
    if original_node_id not in G.nodes:
        print(f"Warning: Node {original_node_id} not in graph")
        return False

    original_data = G.nodes[original_node_id]

    # Calculate areas
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)
    gauge_area = transform(transformer.transform, split_polygon).area / 1e6
    remainder_area = transform(transformer.transform, remainder_polygon).area / 1e6

    # Split the river geometry at the gauge point
    gauge_point = Point(lng, lat)
    river_geom = original_data["river_geom"]

    # Find split point on river
    if river_geom:
        distance_along = river_geom.project(gauge_point)
        total_length = river_geom.length
        split_ratio = distance_along / total_length if total_length > 0 else 0.5

        # Helper to split a single LineString
        def split_linestring(line, pt):
            dist = line.project(pt)
            coords = list(line.coords)
            # Find segment index
            for i in range(len(coords) - 1):
                p1, p2 = Point(coords[i]), Point(coords[i+1])
                seg = LineString([p1, p2])
                if seg.distance(pt) < 1e-8:
                     # Interpolate point exactly on line to ensure connectivity
                     split_pt = line.interpolate(dist)
                     
                     # Check if split point captures full line or none
                     if dist <= 1e-8: return None, line
                     if dist >= line.length - 1e-8: return line, None
                     
                     up = LineString(coords[:i+1] + [split_pt.coords[0]])
                     down = LineString([split_pt.coords[0]] + coords[i+1:])
                     return up, down
            # Fallback if projection issues
            return line, None

        # Split river into two segments
        if split_ratio > 0 and split_ratio < 1:
            split_pt_geom = river_geom.interpolate(distance_along)
            
            if river_geom.geom_type == 'MultiLineString':
                # Find the component containing the split point
                target_part = None
                other_parts = []
                best_dist = float('inf')
                
                for part in river_geom.geoms:
                    d = part.distance(split_pt_geom)
                    if d < best_dist:
                        best_dist = d
                        target_part = part
                    other_parts.append(part)
                
                # Remove target from others if added (it was added in loop)
                other_parts.remove(target_part)

                # Split the target part
                up_part, down_part = split_linestring(target_part, split_pt_geom)
                
                # Assemble
                up_list = other_parts
                if up_part: up_list.append(up_part)
                
                if not up_list: upstream_river = None
                elif len(up_list) == 1: upstream_river = up_list[0]
                else: upstream_river = unary_union(up_list) # Or MultiLineString(up_list)
                
                downstream_river = down_part
                
            else:
                # Simple LineString
                upstream_river, downstream_river = split_linestring(river_geom, split_pt_geom)
        else:
            upstream_river = river_geom
            downstream_river = river_geom
    else:
        upstream_river = None
        downstream_river = None
        split_ratio = 0.5

    # v3.2: MERIT-Basins lines start at their DOWNSTREAM end,
    # so the split above hands the gauge (upstream) piece the river BELOW the gauge; rules 2 and 5 read
    # river_geom, so they measured a split piece's river in the wrong place.
    if river_geom and 0 < split_ratio < 1:
        upstream_river, downstream_river = _assign_river_parts(upstream_river, downstream_river, split_polygon, remainder_polygon)
        len_up = upstream_river.length if upstream_river is not None else 0.0
        len_down = downstream_river.length if downstream_river is not None else 0.0
        if len_up + len_down > 0:
            split_ratio = len_up / (len_up + len_down)

    # Get network connections
    predecessors = list(G.predecessors(original_node_id))

    # Create gauge node with upstream polygon
    G.add_node(
        gauge_id,
        polygon=split_polygon,
        river_geom=upstream_river,
        area_km2=gauge_area,
        length_km=original_data["length_km"] * split_ratio,
        node_type="gauge",
        is_gauge=True,
        lat=lat,
        lng=lng,
        original_comid=original_data["original_comid"],
    )

    # Update original node with downstream polygon.
    # Preserve existing node_type if the node is already a gauge or reservoir;
    # only label as "split_downstream" if it has no special role.
    existing_type = original_data.get("node_type", "original")
    new_type = existing_type if existing_type in ("gauge", "reservoir") else "split_downstream"

    G.nodes[original_node_id].update(
        {
            "polygon": remainder_polygon,
            "river_geom": downstream_river,
            "area_km2": remainder_area,
            "length_km": original_data["length_km"] * (1 - split_ratio),
            "node_type": new_type,
        }
    )

    # Rewire edges. Each predecessor drains into the piece where its river ends (rule 5): normally the upstream (gauge)
    # piece, but a river that enters the catchment below the split point stays with the remainder. Handing every
    # predecessor to the upstream piece routed whole rivers through a lake beside them (graph v2: Lake Oahe).
    # Only at lake-outlet splits (route_by_river_end), and never for a piece of the same MERIT catchment (an earlier split
    # of this reach, upstream on it by construction).
    to_remainder = {p for p in predecessors if route_by_river_end
                    and G.nodes[p].get("original_comid") != original_data.get("original_comid")
                    and _drains_into_remainder(G, p, split_polygon, remainder_polygon, outlet_d8)}
    for pred in predecessors:
        if pred not in to_remainder:
            G.remove_edge(pred, original_node_id)

    # Add new edges: predecessors -> gauge -> original
    for pred in predecessors:
        if pred not in to_remainder:
            G.add_edge(pred, gauge_id)
    G.add_edge(gauge_id, original_node_id)

    return True


def insert_all_gauges(
    G: nx.DiGraph,
    gauges: gpd.GeoDataFrame,
    flow_dir_path: Path,
    flow_acc_path: Path,
    pfaf2_id: int,
) -> Dict[str, Any]:
    """
    Insert all gauges into the graph.
    If a gauge is at the catchment outlet, it replaces the node.
    Otherwise, it splits the node.
    """
    gauge_info = {}
    # Track node replacements: old_node_id -> new_gauge_id
    node_replacements = {}

    # COMID is what we assigned during matchups.
    # node_id links this gauge into the graph and will change as we split and merge polys.
    gauges["node_id"] = gauges["COMID"].astype(str)

    # Process from upstream (0.0) to downstream (1.0)
    gauges = gauges.sort_values(["node_id", "position"])

    for _, gauge in gauges.iterrows():
        gauge_id = str(gauge["id"])

        # Follow the replacement chain to find current node
        original_node_id = gauge["node_id"]
        current_node_id = original_node_id
        while current_node_id in node_replacements:
            current_node_id = node_replacements[current_node_id]

        # Get the current polygon (may have been split already)
        current_polygon = G.nodes[current_node_id]["polygon"]

        gauge_polygon = split_catchment_raster(
            gauge=gauge.to_dict(),
            basin=pfaf2_id,
            catchment_poly=current_polygon,
            flow_dir_path=flow_dir_path,
            flow_acc_path=flow_acc_path,
        )

        if gauge_polygon is None and gauge.get("min_area_orig_km2") is not None and gauge.get("min_area_km2") != gauge.get("min_area_orig_km2"):
            # v3.4: the hinted threshold found no cell in this catchment; retry with the original one rather than lose the gauge
            print(f"  v3.4: {gauge_id}: retrying without snap_min_area_km2")
            g2 = gauge.to_dict(); g2["min_area_km2"] = g2["min_area_orig_km2"]
            gauge_polygon = split_catchment_raster(gauge=g2, basin=pfaf2_id, catchment_poly=current_polygon,
                                                   flow_dir_path=flow_dir_path, flow_acc_path=flow_acc_path)

        if gauge_polygon is None:
            # Failed, reason will be logged by fn
            continue

        if gauge_polygon == current_polygon:
            # No split needed. Convert the node to a gauge
            convert_node_to_gauge(
                G, gauge_id, gauge["lat"], gauge["lng"], current_node_id
            )
            # Track that this node was replaced
            node_replacements[current_node_id] = gauge_id
            gauge_info[gauge_id] = {
                "original_node": original_node_id,
                "current_node": current_node_id,
                "method": "replace",
            }
            continue

        # Calculate remainder
        remainder_polygon = current_polygon.difference(gauge_polygon)
        remainder_polygon = remainder_polygon.buffer(-MERIT_RES / 2).buffer(
            MERIT_RES / 2
        )

        insert_gauge_into_graph(
            G,
            gauge_id,
            gauge["lat"],
            gauge["lng"],
            gauge_polygon,
            remainder_polygon,
            current_node_id,
        )
        gauge_info[gauge_id] = {
            "original_node": original_node_id,
            "current_node": current_node_id,
            "method": "split",
        }

    return gauge_info


def convert_node_to_gauge(
    G: nx.DiGraph,
    gauge_id: str,
    lat: float,
    lng: float,
    original_node_id: str,
):
    """
    Converts an existing node to a gauge by updating its
    attributes and relabeling its ID in the graph.
    """
    # 1. Update the attributes in-place
    G.nodes[original_node_id].update(
        {
            "node_type": "gauge",
            "is_gauge": True,
            "lat": lat,
            "lng": lng,
        }
    )

    # 2. Relabel the node (NetworkX handles all edge rewiring)
    mapping = {original_node_id: gauge_id}
    nx.relabel_nodes(G, mapping, copy=False)


# =============================================================================
# RESERVOIR INSERTION
# =============================================================================



def find_reservoir_outlet(
    res_poly: Polygon,
    flow_dir_path: Path,
    flow_acc_path: Path,
    pfaf2_id: int,
) -> tuple:
    """
    Find the downstream outlet of a reservoir polygon using flow accumulation.

    Loads the flow accumulation raster, masks it to the reservoir footprint,
    and returns the (lng, lat) of the pixel with maximum flow accumulation.

    """
    from pysheds.grid import Grid

    bounds = res_poly.bounds
    halfpix = MERIT_RES / 2
    bounding_box = (
        floor(bounds[0] * 1200) / 1200 - halfpix,
        floor(bounds[1] * 1200) / 1200 - halfpix,
        ceil(bounds[2] * 1200) / 1200 + halfpix,
        ceil(bounds[3] * 1200) / 1200 + halfpix,
    )

    facc_file = resolve_raster(flow_acc_path, pfaf2_id)
    grid = Grid.from_raster(facc_file, window=bounding_box, nodata=0)
    facc = grid.read_raster(facc_file, window=bounding_box, nodata=0)

    # Mask to reservoir footprint
    mask = grid.rasterize([res_poly])
    facc[mask == 0] = 0

    # The outlet is the cell with maximum flow accumulation
    max_idx = np.unravel_index(facc.argmax(), facc.shape)
    outlet_lng = grid.affine[2] + (max_idx[1] + 0.5) * grid.affine[0]
    outlet_lat = grid.affine[5] + (max_idx[0] + 0.5) * grid.affine[4]

    return (outlet_lng, outlet_lat)

def find_downstream_node(G: nx.DiGraph, nodes: list) -> str:
    """Return the most-downstream node from a subset of graph nodes."""
    sub = G.subgraph(nodes)
    return list(nx.topological_sort(sub))[-1]


def split_at_reservoir_outlet(
    G: nx.DiGraph,
    outlet_node: str,
    outlet_lng: float,
    outlet_lat: float,
    res_id: str,
    flow_dir_path: Path,
    flow_acc_path: Path,
    pfaf2_id: int,
) -> str:
    """
    Split the catchment at the reservoir outlet using the same code path
    as gauge splits (no skip_snap, normal snap_to_mask).

    Returns the ID of the new reservoir node, or outlet_node if split fails.
    """
    node_data = G.nodes[outlet_node]
    catchment_poly = node_data["polygon"]

    # Same dict format as gauges — NO skip_snap
    synthetic_gauge = {
        "id": res_id,
        "lat": outlet_lat,
        "lng": outlet_lng,
        "min_area_km2": 0.0,
        "max_area_km2": node_data.get("area_km2", 0),
    }

    upstream_poly = split_catchment_raster(
        gauge=synthetic_gauge,
        basin=pfaf2_id,
        catchment_poly=catchment_poly,
        flow_dir_path=flow_dir_path,
        flow_acc_path=flow_acc_path,
    )

    if upstream_poly is None:
        return outlet_node

    if upstream_poly.equals(catchment_poly):
        return outlet_node

    # Calculate remainder
    remainder_poly = catchment_poly.difference(upstream_poly)
    remainder_poly = remainder_poly.buffer(-MERIT_RES / 2).buffer(MERIT_RES / 2)


    # Reuse the gauge insertion machinery for edge rewiring
    insert_gauge_into_graph(
        G,
        gauge_id=res_id,
        lat=outlet_lat,
        lng=outlet_lng,
        split_polygon=upstream_poly,
        remainder_polygon=remainder_poly,
        original_node_id=outlet_node,
        route_by_river_end=True,
        outlet_d8=(flow_dir_path, pfaf2_id, outlet_lng, outlet_lat),
    )

    return res_id



# Rule 2 (graph v3, lake entry): a catchment that overlaps the reservoir's PLD polygon is absorbed only if the polygon
# covers at least MIN_OVERLAP_FRAC of it AND either its river runs at least MIN_RIVER_IN_LAKE_M inside the polygon or the
# polygon covers at least FULL_OVERLAP_FRAC of it. A bare touch is not enough (graph v2.2: Bluestone Lake touched MERIT
# catchment 74045389 by 0.036 km2 and cut off the Glen Lyn gauge), and neither is a river that runs past the lake.
MIN_OVERLAP_FRAC = 0.02
FULL_OVERLAP_FRAC = 0.5
MIN_RIVER_IN_LAKE_M = 100.0
_GEOD = pyproj.Geod(ellps="WGS84")


def _river_in_lake_m(G: nx.DiGraph, node, res_poly):
    riv = G.nodes[node].get("river_geom")
    if riv is None or riv.is_empty:
        return None
    inter = riv.intersection(res_poly)
    return 0.0 if inter.is_empty else float(_GEOD.geometry_length(inter))


# Rule 6 (graph v3.3): a lake inside the catchment piece of the gauge below its dam .
LAKE_IN_GAUGE_FRAC = 0.2       # the gauge's piece holds at least a fifth of the lake polygon (and more than the outlet node) ...
OUTLET_NODE_MAX_FRAC = 0.25    # ... and the node holding the outlet cell less than a quarter


def _lake_inside_gauge(G: nx.DiGraph, res_poly, outlet_node, intersecting, outlet_d8, flow_dir_path, flow_acc_path, pfaf2_id):
    """(v3.3) Return (gauge node, outlet lng, outlet lat) when the lake should be split out of a gauge's piece instead of outlet_node,
    else None. See make_patch_v33.py."""
    if res_poly is None or res_poly.is_empty or res_poly.area <= 0:
        return None
    def share(n):
        p = G.nodes[n].get("polygon")
        return 0.0 if p is None else p.intersection(res_poly).area / res_poly.area
    gauges = [n for n in intersecting if n in G and n != outlet_node and G.nodes[n].get("is_gauge", False)
              and G.nodes[n].get("lat") is not None and G.nodes[n].get("lng") is not None]
    if not gauges:
        return None
    g = max(gauges, key=share)
    sx = share(outlet_node) if outlet_node in G else 0.0
    if share(g) < LAKE_IN_GAUGE_FRAC or share(g) <= sx or sx >= OUTLET_NODE_MAX_FRAC:
        return None
    if outlet_node in G and not nx.has_path(G, g, outlet_node):
        return None                       # MERIT-Basins must put the gauge above the outlet node (the contradiction)
    nd = G.nodes[g]
    start = Point(float(nd["lng"]), float(nd["lat"]))
    if _passes_outlet(outlet_d8, start, res_poly.union(nd["polygon"]).bounds) is not False:
        return None                       # the gauge is above or at the outlet, or it cannot be told: as before
    inner = res_poly.intersection(nd["polygon"])
    if inner.is_empty:
        return None
    lng, lat = find_reservoir_outlet(inner, flow_dir_path, flow_acc_path, pfaf2_id)
    if not nd["polygon"].buffer(MERIT_RES).contains(Point(lng, lat)):
        return None
    print(f"  rule 6: gauge {g} below the dam holds {share(g):.0%} of the lake (outlet node {outlet_node} {sx:.0%})")
    return g, lng, lat


def insert_reservoirs(
    G: nx.DiGraph,
    reservoirs: gpd.GeoDataFrame,
    flow_dir_path: Path,
    flow_acc_path: Path,
    pfaf2_id: int,
) -> dict:
    """
    Insert reservoir polygons into the graph.

    For each reservoir:
    1. Find which graph node contains the reservoir outlet
    2. Split the catchment at the pour point (dam)
    3. Merge same-COMID, hydraulically connected, non-gauge fragments
       into the reservoir node
    4. Flag as is_reservoir=True to protect from consolidation

    Args:
        G: The river network graph
        reservoirs: GeoDataFrame with columns: reservoir_id, geometry (Polygon)
        flow_dir_path: Path to flow direction rasters
        flow_acc_path: Path to flow accumulation rasters
        pfaf2_id: Pfafstetter level-2 basin ID for raster file selection

    Returns:
        Dict mapping reservoir_id to info about the insertion
    """

    reservoir_info = {}

    # Callers select reservoirs with a bounding box around the basin's gauges, but
    # a basin is an irregular watershed occupying only part of that rectangle, so
    # most candidates belong to neighbouring basins. Partition them up front
    # instead of discovering it one reservoir at a time: it keeps the per-reservoir
    # warning meaningful (a reservoir that IS in this basin but fails to insert is
    # a real problem worth seeing) and avoids scanning every node for reservoirs
    # that were never going to match.
    #
    # Safe to do once, before the loop: inserting a reservoir splits and merges
    # catchments within the basin's existing footprint, so it never extends the
    # basin's spatial extent. A reservoir outside the footprint now cannot begin
    # intersecting it later.
    node_ids = [n for n in G.nodes if G.nodes[n].get("polygon") is not None]
    node_polys = [G.nodes[n]["polygon"] for n in node_ids]
    footprint_tree = STRtree(node_polys) if node_polys else None

    in_basin, n_outside = [], 0
    for _, res in reservoirs.iterrows():
        if footprint_tree is None:
            n_outside += 1
            continue
        if len(footprint_tree.query(res.geometry, predicate="intersects")):
            in_basin.append(res)
        else:
            n_outside += 1

    if n_outside:
        print(f"  {n_outside} of {len(reservoirs)} candidate reservoirs fall outside this "
              f"basin's catchments (bounding-box selection, not an error)")

    for res in in_basin:
        res_poly = res.geometry
        # Handle lake_id which might be float in the GDB
        try:
             video_id_val = res.get("lake_id", res.get("reservoir_id", 0))
             res_id = str(int(video_id_val))
        except (ValueError, TypeError):
             res_id = str(video_id_val)

        # 1. Find the reservoir outlet using flow accumulation
        #    Check that at least one graph node intersects the reservoir
        intersecting = [
            n for n in G.nodes
            if G.nodes[n].get("polygon") is not None
            and G.nodes[n]["polygon"].intersects(res_poly)
        ]
        if not intersecting:
            # Reached only if the graph changed under us, since this reservoir was
            # pre-screened against the basin footprint -- so it is worth surfacing.
            print(f"Warning: reservoir {res_id} is inside the basin footprint but "
                  "matched no catchment during insertion")
            reservoir_info[res_id] = {"status": "no_intersections"}
            continue

        outlet_lng, outlet_lat = find_reservoir_outlet(
            res_poly, flow_dir_path, flow_acc_path, pfaf2_id
        )
        outlet_point = Point(outlet_lng, outlet_lat)

        # 2. Determine which graph node contains the outlet
        outlet_node = next(
            (n for n in G.nodes
             if G.nodes[n].get("polygon") is not None
             and G.nodes[n]["polygon"].contains(outlet_point)),
            None,
        )

        if outlet_node is None:
            # Outlet at a polygon boundary — use most-downstream intersecting node
            outlet_node = find_downstream_node(G, intersecting)

        # Rule 6 (v3.3): the lake lies inside the piece of the gauge below its dam (see _lake_inside_gauge)
        alt = _lake_inside_gauge(G, res_poly, outlet_node, intersecting, (flow_dir_path, pfaf2_id, outlet_lng, outlet_lat),
                                 flow_dir_path, flow_acc_path, pfaf2_id)

        # 3. Split the catchment at the dam pour point
        target = None
        if alt is not None:
            # rule 6: split the lake out of the gauge's piece; the gauge keeps the part below the dam
            g6, lng6, lat6 = alt
            t6 = split_at_reservoir_outlet(G, g6, lng6, lat6, res_id, flow_dir_path, flow_acc_path, pfaf2_id)
            if t6 != g6:
                outlet_node, outlet_lng, outlet_lat, target = g6, lng6, lat6, t6
                outlet_point = Point(outlet_lng, outlet_lat)
                print(f"  rule 6: {res_id} split out of gauge {g6}'s piece; the lake drains into the gauge")
        if target is None:
            had6 = alt is not None and G.has_edge(alt[0], outlet_node)
            target = split_at_reservoir_outlet(
                G, outlet_node, outlet_lng, outlet_lat, res_id,
                flow_dir_path, flow_acc_path, pfaf2_id,
            )
            if had6 and target != outlet_node and G.has_edge(alt[0], outlet_node):
                # rule 6, gauge at the lake outlet: its piece ends inside the lake, so nothing of it lies below the dam. The lake takes
                # the gauge's piece (the lake body) and every inflow of both pieces; the gauge takes the small piece at the dam and
                # drains the lake into the outlet node's remainder.
                g6 = alt[0]
                pa, pl = [q for q in G.predecessors(g6)], [q for q in G.predecessors(target)]
                for k in ("polygon", "river_geom", "area_km2", "length_km", "original_comid"):
                    G.nodes[g6][k], G.nodes[target][k] = G.nodes[target].get(k), G.nodes[g6].get(k)
                for q in pa: G.remove_edge(q, g6)
                for q in pl: G.remove_edge(q, target)
                G.remove_edge(g6, outlet_node)
                G.remove_edge(target, outlet_node)
                for q in pa + pl: G.add_edge(q, target)
                G.add_edge(target, g6)
                G.add_edge(g6, outlet_node)
                print(f"  rule 6: gauge {g6} sits at the outlet of {res_id}: the lake takes its piece, the gauge the piece at the dam below it")

        # 4. Merge upstream catchments into the reservoir node if they:
        #    (a) spatially overlap the reservoir polygon and the lake holds their river or most of them (rule 2), OR
        #    (b) share the same original COMID (even without overlap).
        #    Only catchments UPSTREAM of the target (draining into the outlet) qualify (rule 1), and only if their path to
        #    the outlet does not run through a gauge or another reservoir (rule 3).
        #    Gauges and other reservoirs are always protected.
        target_comid = G.nodes[target].get("original_comid")
        merged_nodes = []

        # Find all upstream nodes reachable from target via reversed edges
        # (i.e., nodes that drain INTO the target).
        upstream_of_target = nx.ancestors(G, target)
        
        # Consider both intersecting nodes and upstream nodes
        candidates = set(intersecting) | set(upstream_of_target)

        for node in list(candidates):
            if node not in G:
                continue
            if node == target:
                continue
            # IMPORTANT: never merge outlet_node if it was split and holds the downstream remainder!
            if node == outlet_node and target != outlet_node:
                continue
            # Never merge gauge nodes
            if G.nodes[node].get("is_gauge", False):
                continue
            # Never merge other reservoir nodes
            if G.nodes[node].get("is_reservoir", False):
                continue
            # Only catchments that drain INTO the reservoir outlet (rule 1). A node below the pour point that merely
            # touches the lake polygon (e.g. just under the dam) is downstream of the target; absorbing it stranded every
            # node between the dam and it without an outlet, and build_conus_graphs then wired those dead ends to the basin
            # outlet (graph v1 long links: upper Columbia -> Port Westward, Tongue River Reservoir -> Baton Rouge).
            if node not in upstream_of_target:
                continue
            # (a) Spatial overlap with reservoir polygon
            node_poly = G.nodes[node].get("polygon")
            overlaps_reservoir = node_poly is not None and node_poly.intersects(res_poly)
            if overlaps_reservoir and node_poly.area > 0:
                frac = node_poly.intersection(res_poly).area / node_poly.area
                if frac < MIN_OVERLAP_FRAC:
                    overlaps_reservoir = False
                elif frac < FULL_OVERLAP_FRAC:
                    in_lake_m = _river_in_lake_m(G, node, res_poly)
                    overlaps_reservoir = in_lake_m is None or in_lake_m >= MIN_RIVER_IN_LAKE_M
            # (b) Same original COMID as the reservoir node (and is an upstream neighbor)
            same_comid = (
                target_comid is not None
                and G.nodes[node].get("original_comid") == target_comid
                and node in upstream_of_target
            )
            if not (overlaps_reservoir or same_comid):
                continue
            # Never absorb across a protected node (rule 3): if this catchment's water reaches the outlet through a gauge
            # or another reservoir, absorbing it would hand its inflows to the reservoir and leave that gauge or reservoir
            # as an island with only its local area (graph v2.2: New River at Glen Lyn inside Bluestone Lake).
            try:
                between = nx.shortest_path(G, node, target)[1:-1]
            except nx.NetworkXNoPath:
                continue
            if any(G.nodes[x].get("is_gauge", False) or G.nodes[x].get("is_reservoir", False) for x in between):
                continue
            reason = "reservoir_overlap_merge" if overlaps_reservoir else "same_comid_merge"
            info = merge_nodes(G, source=node, target=target, reason=reason)
            if info:
                merged_nodes.append(info["source"])

        # 5. Flag the node so consolidation won't consume it
        G.nodes[target]["is_reservoir"] = True
        G.nodes[target]["node_type"] = "reservoir"
        G.nodes[target]["is_gauge"] = False
        G.nodes[target]["lat"] = outlet_lat
        G.nodes[target]["lng"] = outlet_lng

        # 6. Relabel to reservoir ID if not already
        if target != res_id:
            nx.relabel_nodes(G, {target: res_id}, copy=False)
            target = res_id

        reservoir_info[res_id] = {
            "status": "inserted",
            "outlet_node": target,
            "merged_nodes": merged_nodes,
            "outlet_lat": outlet_lat,
            "outlet_lng": outlet_lng,
        }
        print(f"  Reservoir {res_id}: merged {len(merged_nodes)} nodes into {target}")

    return reservoir_info


# =============================================================================
# NETWORK CONSOLIDATION
# =============================================================================
def consolidate_graph(
    G: nx.DiGraph,
    target_area: float = 500,
    preserve_gauges: bool = True,
    max_iterations: int = 100,
) -> list[dict]:
    """
    Consolidate river network using a sequential, multi-rule approach
    """
    # We initialize a list to hold polygons so we don't union them every step
    for n, data in G.nodes(data=True):
        if "polygon" in data:
            data["component_polygons"] = [data["polygon"]]

    calculate_stream_orders(G)

    merge_history = []

    # Iterative consolidation loop
    for i in range(max_iterations):
        initial_nodes = G.number_of_nodes()
        merges_in_round = 0

        # Merge small leaves downstream
        merges_in_round += merge_leaves(G, target_area, preserve_gauges, merge_history)

        # Merge stems downstream
        merges_in_round += collapse_stems(
            G, target_area, preserve_gauges, merge_history
        )

        # Check for Convergence
        final_nodes = G.number_of_nodes()
        if final_nodes == initial_nodes:
            break

    # Final step: Merge stems upstream 
    merge_stems_upstream(G, target_area, preserve_gauges, merge_history)

    # Final recomputation of attributes
    finalize_geometry(G)
    calculate_stream_orders(G)
    calculate_upstream_areas(G)

    return merge_history


def can_be_source(G: nx.DiGraph, n: Any, preserve_gauges: bool) -> bool:
    """Node can be merged INTO another node."""
    if preserve_gauges and G.nodes[n].get("is_gauge", False):
        return False
    if G.nodes[n].get("is_reservoir", False):
        return False
    return True

def merge_nodes(G: nx.DiGraph, source: str, target: str, reason: str) -> dict:
    """Merge `source` node INTO `target` node with immediate geometry optimization.

    Refuses (ValueError) a merge that would break the river tree, instead of silently doing it:
      - a predecessor of `source` that lies between `target` and `source` (target -> ... -> pred -> source) cannot be
        re-pointed to `target` without a loop; skipping it left that node with no nextdown (an interior dead end);
      - `target` would end up with more than one downstream node (upstream area then gets counted twice).
    Both only arise when a node is merged into a target UPSTREAM of it; insert_reservoirs no longer does that.
    """
    if source not in G or target not in G:
        return None
    stranded = [p for p in G.predecessors(source) if p != target and nx.has_path(G, target, p)]
    new_exits = {s for s in G.successors(source) if s != target and not nx.has_path(G, s, target)}
    exits = (set(G.successors(target)) - {source}) | new_exits
    if stranded or len(exits) > 1:
        raise ValueError(
            f"merge_nodes({source!r} -> {target!r}, {reason}) would break the river tree: "
            + (f"{stranded} would be left without an outlet" if stranded else "")
            + ("; " if stranded and len(exits) > 1 else "")
            + (f"{target!r} would drain to {sorted(map(str, exits))}" if len(exits) > 1 else ""))

    s = G.nodes[source]
    t = G.nodes[target]

    # Union the catchment geometries
    if "polygon" in s and "polygon" in t:
        combined = unary_union([s["polygon"], t["polygon"]])
        combined = set_precision(combined, grid_size=1e-6)
        t["polygon"] = combined.simplify(1e-7)

    # Sum up the area attributes
    t["area_km2"] = t.get("area_km2", 0) + s.get("area_km2", 0)

    # Union the river geometries
    if s.get("river_geom") and t.get("river_geom"):
        t["river_geom"] = unary_union([s["river_geom"], t["river_geom"]])

    # --- Rewire edges ---
    for pred in list(G.predecessors(source)):
        if pred != target and not nx.has_path(G, target, pred):
            G.add_edge(pred, target)
            
    for succ in list(G.successors(source)):
        if succ != target and not nx.has_path(G, succ, target):
            G.add_edge(target, succ)

    G.remove_node(source)

    return {"source": source, "target": target, "reason": reason}


def finalize_geometry(G: nx.DiGraph):
    """Final cleanup of geometries using Mitre joins to prevent vertex explosion."""
    for n in G.nodes:
        if "polygon" in G.nodes[n]:
            poly = G.nodes[n]["polygon"]
            
            # Use Mitre joins (style 2) to avoid creating arcs at every corner
            refined = poly.buffer(MERIT_RES, join_style=2).buffer(-MERIT_RES, join_style=2)
            
            # Snap to grid and simplify one last time
            refined = set_precision(refined, grid_size=1e-6)
            G.nodes[n]["polygon"] = refined.simplify(1e-7)

            if "river_geom" in G.nodes[n]:
                G.nodes[n]["river_geom"] = set_precision(G.nodes[n]["river_geom"], 1e-6)


def merge_leaves(
    G: nx.DiGraph,
    target_area: float,
    preserve_gauges: bool,
    merge_history: List[Dict[str, Any]],
):
    """Mimics trim_clusters / prune_leaves: merges small leaves into their downstream neighbor."""

    # Pre-filter nodes by degree first (fastest check) before checking attributes
    # This prevents attribute lookups on nodes that are topologically ineligible
    candidates = [
        n for n in G.nodes
        if G.out_degree(n) == 1 # Must be a leaf (or flow-through)
        and can_be_source(G, n, preserve_gauges) # Gauge check
        and G.nodes[n].get("area_km2", 0) < target_area # Area check
    ]

    merges_in_step = 0
    for source in candidates:
        if source not in G: 
            continue # Might have been merged in this loop
        
        succs = list(G.successors(source))
        if not succs: 
            continue
        target = succs[0]
        
        # Check merge validity
        if target in G and (
            G.nodes[source].get("area_km2", 0) + G.nodes[target].get("area_km2", 0) < target_area
        ):
            # Target is valid if it's a normal mergeable node, a gauge,
            # or a reservoir IF source shares the same original COMID.
            target_ok = (
                can_be_source(G, target, preserve_gauges)
                or (preserve_gauges and G.nodes[target].get("is_gauge", False))
                or (
                    G.nodes[target].get("is_reservoir", False)
                    and G.nodes[source].get("original_comid")
                    == G.nodes[target].get("original_comid")
                )
            )
            if target_ok:
                info = merge_nodes(G, source, target, "leaf_to_downstream")
                if info:
                    merge_history.append(info)
                    merges_in_step += 1
    return merges_in_step


def collapse_stems(
    G: nx.DiGraph,
    target_area: float,
    preserve_gauges: bool,
    merge_history: List[Dict[str, Any]],
):
    """
    Mimics collapse_stems: merges stems downstream (source=stem, target=successor).
    UPDATED: Now includes junction preservation logic.
    """

    # Stems are nodes with in-degree < 2 (or in-degree == 1).
    stems = [
        n for n in G.nodes
        if G.in_degree(n) < 2 
        and G.out_degree(n) == 1
        and can_be_source(G, n, preserve_gauges)
        and G.nodes[n].get("area_km2", 0) < target_area
    ]

    # Sort stems by area, small to large to merge these first
    stems.sort(key=lambda n: G.nodes[n].get("area_km2", 0))

    merges_in_step = 0
    for source in stems:
        if source not in G.nodes:
            continue
        
        # Get the downstream neighbor
        succs = list(G.successors(source))
        if not succs:
            continue
        target = succs[0]

        # Target must be present in graph
        if target not in G.nodes:
            continue

        # If the target is a junction (has more than 1 incoming river), 
        # do NOT merge the current stem into it. This keeps the confluence point fixed.
        if G.in_degree(target) > 1:
            continue

        # Check area thresholds
        if (
            G.nodes[source].get("area_km2", 0) + G.nodes[target].get("area_km2", 0)
            < target_area
        ):
            # Target is valid if it's a normal mergeable node, a gauge,
            # or a reservoir IF source shares the same original COMID.
            target_ok = (
                can_be_source(G, target, preserve_gauges)
                or (preserve_gauges and G.nodes[target].get("is_gauge", False))
                or (
                    G.nodes[target].get("is_reservoir", False)
                    and G.nodes[source].get("original_comid")
                    == G.nodes[target].get("original_comid")
                )
            )
            if target_ok:
                info = merge_nodes(G, source, target, "stem_to_downstream")
                if info:
                    merge_history.append(info)
                    merges_in_step += 1

    return merges_in_step


def merge_stems_upstream(
    G: nx.DiGraph,
    target_area: float,
    preserve_gauges: bool,
    merge_history: List[Dict[str, Any]],
):
    """Mimics last_merge: merges stems upstream (source=predecessor, target=stem)."""

    # Candidates are nodes with in-degree == 1 (stems).
    candidates = [
        n
        for n in list(G.nodes)
        if G.in_degree(n) == 1 and len(list(G.successors(n))) == 1
    ]

    # Sort candidates by area, small to large (to favor small merges first)
    candidates.sort(key=lambda n: G.nodes[n].get("area_km2", 0))

    merges_in_step = 0
    for target in candidates:  # Target is the downstream node, it receives the merge
        if target not in G.nodes:
            continue

        # Target must be able to receive merges (gauges can absorb;
        # reservoirs can absorb only same-COMID sources)
        is_valid_target = (
            can_be_source(G, target, preserve_gauges)
            or (preserve_gauges and G.nodes[target].get("is_gauge", False))
        )
        if not is_valid_target:
            # Check same-COMID reservoir exception
            source = list(G.predecessors(target))[0]
            if not (
                G.nodes[target].get("is_reservoir", False)
                and G.nodes[source].get("original_comid")
                == G.nodes[target].get("original_comid")
            ):
                continue

        # Source is the predecessor
        source = list(G.predecessors(target))[0]

        # Source must be eligible to be merged out
        if not can_be_source(G, source, preserve_gauges):
            continue

        if (
            G.nodes[source].get("area_km2", 0) + G.nodes[target].get("area_km2", 0)
            < target_area
        ):
            info = merge_nodes(G, source, target, "stem_to_upstream")
            if info:
                merge_history.append(info)
                merges_in_step += 1

    return merges_in_step


def calculate_stream_orders(G: nx.DiGraph):
    """Calculate Strahler and Shreve stream orders"""
    # Sort nodes topologically (upstream to downstream)
    try:
        sorted_nodes = list(nx.topological_sort(G))
    except nx.NetworkXError:
        # Graph has cycles, use all nodes
        sorted_nodes = list(G.nodes())

    for node in sorted_nodes:
        predecessors = list(G.predecessors(node))

        if not predecessors:
            # Headwater node
            G.nodes[node]["strahler_order"] = 1
            G.nodes[node]["shreve_order"] = 1
        else:
            # Calculate Shreve (sum of upstream orders)
            shreve = sum(G.nodes[p].get("shreve_order", 1) for p in predecessors)
            G.nodes[node]["shreve_order"] = shreve

            # Calculate Strahler
            upstream_orders = [
                G.nodes[p].get("strahler_order", 1) for p in predecessors
            ]
            max_order = max(upstream_orders)
            count_max = upstream_orders.count(max_order)

            if count_max > 1:
                G.nodes[node]["strahler_order"] = max_order + 1
            else:
                G.nodes[node]["strahler_order"] = max_order


def calculate_upstream_areas(G: nx.DiGraph):
    """Calculate cumulative upstream area for each node"""
    # Sort nodes topologically (upstream to downstream)
    try:
        sorted_nodes = list(nx.topological_sort(G))
    except nx.NetworkXError:
        sorted_nodes = list(G.nodes())

    for node in sorted_nodes:
        predecessors = list(G.predecessors(node))

        if not predecessors:
            # Headwater - upstream area is just its own area
            G.nodes[node]["uparea_km2"] = G.nodes[node].get("area_km2", 0)
        else:
            # Sum all upstream areas plus own area
            upstream_total = sum(G.nodes[p].get("uparea_km2", 0) for p in predecessors)
            G.nodes[node]["uparea_km2"] = upstream_total + G.nodes[node].get(
                "area_km2", 0
            )


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================


def delineate_basins(
    gauges_path: str,
    merit_dirs: Dict[str, Path],
    reservoirs: gpd.GeoDataFrame = None,
    min_reservoir_area_km2: float = 0.0,
    max_area: float = 500,
    consolidate: bool = True,
    preserve_gauges: bool = True,
    output_dir: Path = None,
    save_plots: bool = False,
    overwrite: bool = False,
) -> Dict[str, DelineationResult]:
    """
    Main function to delineate watersheds.

    Args:
        gauges_csv: Path to CSV with columns: id, lat, lng, outlet_id
        merit_dirs: Dictionary with paths to MERIT data
        reservoirs: GeoDataFrame of reservoir polygons
        min_reservoir_area_km2: Minimum area (km2) to include a reservoir
        max_area: Maximum area for consolidation (km²)
        consolidate: Whether to merge small catchments
        preserve_gauges: Keep gauge nodes during consolidation

    Returns:
        Dictionary mapping outlet_id to DelineationResult
    """
    gauges_path = Path(gauges_path)
    if output_dir is None:
        output_dir = (gauges_path.parent / gauges_path.stem).resolve()

    # Filter reservoirs
    if reservoirs is not None and not reservoirs.empty and min_reservoir_area_km2 > 0:
        if "area_geodesic_km2" in reservoirs.columns:
            initial_count = len(reservoirs)
            reservoirs = reservoirs[reservoirs["area_geodesic_km2"] >= min_reservoir_area_km2]
            print(f"Filtered reservoirs by area >= {min_reservoir_area_km2} km2: {initial_count} -> {len(reservoirs)}")
        else:
            print(f"Warning: 'area_geodesic_km2' not found in reservoir data. Skipping area filter.")

    processed_outlets = []
    if not overwrite:
        processed_outlets = [
            f.stem.split('_gauges')[0] 
            for f in (output_dir / 'gauges').glob('*.parquet')
        ]

    print("\n" + "=" * 80)
    print("GRAPH-BASED WATERSHED DELINEATION")
    results = {}   
    print(f"Data output directory: {output_dir}")
    print("=" * 80)

    all_gauges = load_gauges(gauges_path)
    if all_gauges.empty:
        print("Error: No gauges loaded!")
        return {}

    existing_count = len(processed_outlets)

    all_gauges = all_gauges[~all_gauges['outlet_id'].isin(processed_outlets)]
    remaining_count = len(all_gauges)
    
    print(f"Loaded {remaining_count + existing_count} gauges total.")
    print(f"Skipping {existing_count} already processed outlets.")
    print(f"Processing {remaining_count} remaining gauges.")
    
    if remaining_count == 0:
        print("No new gauges to delineate.")
        print("COMPLETE")
        print(f"{'=' * 80}")
        return {} # Early exit

    pfaf1_outlets = get_pfaf1_outlet_dict(all_gauges)

    for pfaf1_id, outlet_dict in pfaf1_outlets.items():
        # Load basin data
        pfaf1_catchments, pfaf1_rivers = load_basin_data(pfaf1_id, merit_dirs)

        # Process each outlet in this basin
        for outlet_id, outlet_gauge_ids in tqdm(outlet_dict.items(), desc=f'Basin {pfaf1_id}'):
            gauges = all_gauges[all_gauges["id"].isin(outlet_gauge_ids)]
            gauges = set_gauge_area_range(gauges, pfaf1_rivers)

            final_comid = gauges[gauges["id"] == outlet_id]["COMID"].item()
            comids = get_network_comids(final_comid, pfaf1_rivers)

            catchments = pfaf1_catchments.loc[pfaf1_catchments.index.isin(comids)]
            rivers = pfaf1_rivers.loc[pfaf1_rivers.index.isin(comids)]

            # Build graph with all geometries
            G = build_graph_with_geometries(catchments, rivers)

            # Insert gauges
            pfaf2_id = int(final_comid // 1e6)
            gauge_info = insert_all_gauges(
                G, gauges, merit_dirs["flow_dir"], merit_dirs["flow_acc"], pfaf2_id
            )

            # Trim network at outlet if it's a gauge
            if outlet_id in G.nodes:
                trim_at_outlet(G, outlet_id)

            # Insert reservoir polygons if provided
            if reservoirs is not None and not reservoirs.empty:
                reservoir_info = insert_reservoirs(
                    G, reservoirs, merit_dirs["flow_dir"], merit_dirs["flow_acc"], pfaf2_id
                )

            # Consolidate if requested
            if consolidate:
                # starting_nodes = len(G.nodes)
                merge_history = consolidate_graph(
                    G, target_area=max_area, preserve_gauges=preserve_gauges
                )
                # ending_nodes = len(G.nodes)
                # print(f"  Consolidated graph from {starting_nodes} to {ending_nodes} nodes")

            # Store result
            result = DelineationResult(graph=G, outlet_id=outlet_id)
            save_result(outlet_id, result, output_dir, save_plots)
            results[outlet_id] = result

    print(f"\n{'=' * 80}")
    print("DELINEATION SUMMARY")
    for outlet_id, result in results.items():
        G = result.graph
        n_gauges = sum(1 for n, d in G.nodes(data=True) if d.get("is_gauge", False))
        n_reservoirs = sum(1 for n, d in G.nodes(data=True) if d.get("is_reservoir", False))
        n_other = len(G.nodes) - n_gauges - n_reservoirs
        print(f"  Outlet {outlet_id}: {n_gauges} Gauges, {n_reservoirs} Reservoirs, {n_other} Catchment/Other Nodes")
    print("COMPLETE")
    print(f"{'=' * 80}")

    return results


def set_gauge_area_range(gauges, rivers):
    gauges = gauges.copy()
    gauges["min_area_km2"] = 0.0
    gauges["max_area_km2"] = 0.0

    for i, g in gauges.iterrows():
        min_area = rivers.loc[rivers["NextDownID"] == g["COMID"], "uparea"].sum()
        gauges.loc[i, "min_area_orig_km2"] = min_area
        # v3.4: a gauge re-snapped by drainage area carries the area of its target cell; snap to a channel at least that big
        if "snap_min_area_km2" in gauges.columns and pd.notna(g.get("snap_min_area_km2")):
            min_area = max(min_area, float(g["snap_min_area_km2"]))
        gauges.loc[i, "min_area_km2"] = min_area
        gauges.loc[i, "max_area_km2"] = rivers.loc[g["COMID"], "uparea"]

    return gauges


def trim_at_outlet(G: nx.DiGraph, outlet_id: str):
    """Remove all nodes downstream of the outlet"""
    if outlet_id not in G.nodes:
        return

    # Find all downstream nodes
    downstream = list(nx.descendants(G, outlet_id))

    if downstream:
        # Trim any nodes downstream of the lowest gauge.
        for node in downstream:
            G.remove_node(node)

    # Mark as outlet
    G.nodes[outlet_id]["is_outlet"] = True


def load_basin_data(
    basin_id: int, merit_dirs: Dict[str, Path]
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load MERIT catchment and river data for a basin"""
    basins_dir = Path(merit_dirs["basins"])

    cat_file = (
        basins_dir / f"cat_pfaf_{int(basin_id)}_MERIT_Hydro_v07_Basins_v01_bugfix1.shp"
    )
    riv_file = (
        basins_dir / f"riv_pfaf_{int(basin_id)}_MERIT_Hydro_v07_Basins_v01_bugfix1.shp"
    )

    catchments = gpd.read_file(cat_file).set_index("COMID").set_crs("EPSG:4326")
    rivers = gpd.read_file(riv_file).set_index("COMID").set_crs("EPSG:4326")

    return catchments, rivers


def save_result(
    outlet_id: str, result: DelineationResult, output_dir: Path, save_plots: bool
):
    """Save all watershed results"""

        # Export to GeoDataFrames
    subbasins, nodes = result.to_geodataframes()

    if not subbasins.empty:
        subbasins_dir = output_dir / "subbasins"
        subbasins_dir.mkdir(parents=True, exist_ok=True)
        # subbasins.to_file(subbasins_dir / f"{outlet_id}.gpkg", driver='GPKG')
        subbasins.to_parquet(subbasins_dir / f"{outlet_id}_subbasins.parquet")
    if not nodes.empty:
        nodes_dir = output_dir / "nodes"
        nodes_dir.mkdir(parents=True, exist_ok=True)
        # gauges.to_file(gauges_dir / f"{outlet_id}.gpkg", driver='GPKG')
        nodes.to_parquet(nodes_dir / f"{outlet_id}_nodes.parquet")

    if save_plots and not subbasins.empty and not nodes.empty:
        ids = pd.concat(
            [subbasins.loc[subbasins["is_gauge"], "id"], nodes["id"]]
        ).unique()

        cmap = plt.get_cmap("tab20")
        id_to_color = {id_val: cmap(i % cmap.N) for i, id_val in enumerate(ids)}

        sub_color = subbasins[subbasins["is_gauge"]]["id"].map(id_to_color)

        # Color nodes by type: gauges=red, reservoirs=blue, others=black
        def get_node_color(row):
            if row.get("is_gauge"):
                return "red"
            elif row.get("is_reservoir"):
                return "blue"
            else:
                return "black"

        node_color = nodes.apply(get_node_color, axis=1)

        # import numpy as np
        # subbasins['log_area'] = np.log10(subbasins['uparea_km2'])

        fig, ax = plt.subplots(figsize=(6, 6))
        subbasins.plot(color="grey", ax=ax)
        subbasins[subbasins["is_gauge"]].plot(
            color=sub_color, ax=ax, edgecolor="black", linewidth=0.5
        )
        nodes.plot(color=node_color, ax=ax, edgecolor="black", linewidth=0.8)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Gage node',
                   markerfacecolor='red', markeredgecolor='black', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Reservoir node',
                   markerfacecolor='blue', markeredgecolor='black', markersize=8),
        ]
        ax.legend(handles=legend_elements, loc='upper left')

        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_dir / f"{outlet_id}.png", dpi=300)

