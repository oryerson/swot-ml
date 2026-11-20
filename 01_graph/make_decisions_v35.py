"""Write the ruling on every review item of graph v3.4 to decisions_v35.csv, the table that apply_v35.py applies.

The review of graph v3.4 listed gauges and lakes whose sub-basin disagrees with a reference (USGS drainage area for
gauges, MERIT-Hydro upstream area for lakes), each with a proposed fix. This script encodes the ruling on each item
as one row: `decision` (accept or reject the proposed fix), `op` (the operation that follows) and `why`.
One principle holds throughout: no gauge and no lake point is moved. Corrections are made by routing only.

Reads (V3 = $SWOTGAT_WORK/CONUS_storage_fork/eval/graph_v3; both files are only read)
  V3/review_v34/items.csv        the review items (shipped as 01_graph/inputs/review_v34_items.csv)
  V3/v35_work/lake_evidence.csv  evidence per flagged lake, from lake_evidence_v35.py (only the column gdw_catch_max,
                                 the Global Dam Watch catchment area at the dam, is used, in the `why` text)

Writes
  V3/v35_work/decisions_v35.csv  columns key, group, node, name, decision, op, why
                                 (the paper's table is shipped as 01_graph/inputs/decisions_v35.csv)

Rulings by item group
  hand     gauges proposed to be moved by hand: rejected, op none (gauges are not moved)
  snap     gauges that the automatic re-snap of graph v3.4 moved (snap_gauges_v34.py): the re-snap is kept, op none
  explain  differences with a known explanation: op none
  refov    lakes where the graph is right and the MERIT reference at the pour point is wrong: op override_reference
           (the graph is kept). Lake 7421024112 is kept with low confidence.
  above    lakes whose node holds a river corridor that MERIT does not run through the lake: op conform
  edge     lakes with a river routed through them that passes beside them: op conform
  inflow   lakes that do not receive the reach that feeds them: op conform, except the two lakes in KEEP_INFLOW, where
           the dam's catchment area in Global Dam Watch supports the graph (op none)
  gauge2x  four gauges added here (PASS), whose node sits on a side channel while the main stem enters the next node
           down: op passthrough (the main stem is routed through the gauge node)
"conform" sets the lake node to exactly what drains to the lake outlet on the MERIT flow-direction grid; see
apply_v35.py, which applies the ops conform and passthrough.

In the paper's run: 240 rows, of which 38 conform, 4 passthrough, 14 override_reference and 184 none.

Usage
  python make_decisions_v35.py

Note: the `why` texts written to the table quote the review as it was recorded, including the reviewer's name.
"""
from swotgat.paths import WORK
import json
import pandas as pd
V3 = f"{WORK}/CONUS_storage_fork/eval/graph_v3/"
it = pd.read_csv(V3 + "review_v34/items.csv", dtype={"node": str})
ev = pd.read_csv(V3 + "v35_work/lake_evidence.csv", dtype={"node": str}).set_index("key")
# the two 'inflow' lakes that keep their graph v3.4 routing, with the reason
KEEP_INFLOW = {"7320175082": "GDW catchment at the dam (83 km2) supports the graph (89.3) over the MERIT pour point (98.3)",
               "7720028983": "Giroux tailings impoundment: GDW catchment (34 km2) supports the graph (36.4) over MERIT (51.4)"}
rows = []
for r in it.itertuples():
    e = ev.loc[r.key] if r.key in ev.index else None
    gd = "" if e is None or pd.isna(e.gdw_catch_max) else f"; GDW dam catchment {e.gdw_catch_max:.1f} km2"
    if r.group == "hand":
        d, op, why = "reject", "none", "not moved: Owen 2026-10-22 'do not move the physical location of any gauges'"
    elif r.group == "snap":
        d, op, why = "accept", "none", "v3.4 re-snap kept (Owen 2026-10-22: 'ok keep these changes')"
    elif r.group == "explain":
        d, op, why = "accept", "none", "explained, no routing fix"
    elif r.group == "refov":
        d, op = "accept", "override_reference"
        why = ("graph right: the outlet cell one cell beyond the PLD outline is where a second arm or tributary of the impoundment "
               "joins at the dam" + gd)
        if r.node == "7421024112":
            why = "graph kept, LOW confidence: GDW dam catchment 4.0 km2 is below both the graph (8.3) and the pour point (5.9)"
    elif r.group == "above":
        d, op, why = "reject", "conform", "Owen rejected 'keep': the river corridor is taken out of the lake node (lake = its MERIT catchment)"
    elif r.group == "edge":
        d, op = "reject", "conform"
        why = ("river routed past the lake, as for Owen's 'graph above the lake' rejections: "
               + ("a side-arm fragment of Dannelly Reservoir; the Alabama River goes to the main reservoir node" if r.node == "7320138252"
                  else "a small Polk County (phosphate district) pond; the creek beside it does not flow through it"))
    elif r.group == "inflow":
        if r.node in KEEP_INFLOW:
            d, op, why = "reject", "none", "kept: " + KEEP_INFLOW[r.node]
        else:
            d, op = "accept", "conform"
            why = "lake node set to what drains to the lake outlet (MERIT D8)" + gd + " (an on-stream lake, not a fragment of a split lake)"
    rows.append(dict(key=r.key, group=r.group, node=r.node, name=r.name, decision=d, op=op, why=why))
# the four pass-through gauges, with the upstream areas (km2) that support each ruling
PASS = {"USGS-08062700": "Trinity River at Trinidad TX: graph 49.8 km2, USGS 22,113; main stem through the gauge node gives 24,741 (MERIT cell 3.4 km away: 24,614)",
        "USGS-05227500": "Mississippi River at Aitkin MN: graph 449, USGS 15,902; main stem through the gauge node gives 15,192",
        "USGS-02223056": "Oconee River at Avant Mine GA: graph 228, USGS 8,208; main stem through the gauge node gives 8,252",
        "USGS-07368000": "Boeuf River near Girard LA: graph 56, USGS 3,175; main stem through the gauge node gives 4,002 (flat terrain; MERIT's main stem is already 24% over USGS)"}
for g, why in PASS.items():
    rows.append(dict(key="gauge2x-" + g, group="gauge2x", node=g, name="", decision="accept", op="passthrough",
                     why="routing, no move: " + why))
df = pd.DataFrame(rows)
df.to_csv(V3 + "v35_work/decisions_v35.csv", index=False)
print(df.groupby(["group", "decision", "op"]).size().to_string())
