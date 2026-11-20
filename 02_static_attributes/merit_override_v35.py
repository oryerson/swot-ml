"""Use the graph's upstream area in place of MERIT's where the MERIT outlet cell is on the wrong river.

merit_uparea_v35.py reads the MERIT upstream area at each node's outlet cell. At a few nodes that cell lies on a
different river than the one the node represents, so the value is wrong by a large factor. For these nodes
merit_uparea_km2 is set to the graph v3.5 value uparea_km2, the source is set to "graph uparea (override)" and the
reason is recorded.

Reads   $SWOTGAT_WORK/CONUS_storage_fork/eval/v35_inputs/statics/work/merit_uparea_v35_nodes.csv
Writes  merit_uparea_v35_nodes_r2.csv in the same folder (the input is not modified; an existing output is not
        overwritten). New columns: merit_raw_km2 (the value before the override), ratio_raw (= merit_raw_km2 /
        uparea_km2) and override_reason (empty where nothing was overridden).

Overrides
  - The four pass-through gauges of graph v3.5 (USGS-08062700, USGS-07368000, USGS-02223056, USGS-05227500). The
    graph routes the main stem through the gauge node, while the delineator's snapped cell is on a MERIT side
    channel; the gauge measures the main stem. The script checks that these four are exactly the gauges whose
    ratio is above 2 or below 0.5.
  - Every river node (neither lake nor gauge) whose ratio is above 2 or below 0.5: the end of its original_comid
    reach carries a much larger or smaller river than the node holds. The script checks that this rule covers at
    least the four nodes 78013722, 73011224, 74053646 and 74030313.
  - Lake nodes are never overridden: their value is the upstream area at the pour point by definition (this
    includes Heidecke Lake, 7421056723, and fragments of split lakes).

In the paper's run: 8 overrides.

Usage
  python merit_override_v35.py
"""
from swotgat.paths import WORK
from pathlib import Path

import numpy as np
import pandas as pd

W = Path(f"{WORK}/CONUS_storage_fork/eval/v35_inputs/statics/work")
IN, OUT = W/"merit_uparea_v35_nodes.csv", W/"merit_uparea_v35_nodes_r2.csv"
assert not OUT.exists(), f"{OUT} exists; refusing to overwrite"
GAUGES = ["USGS-08062700", "USGS-07368000", "USGS-02223056", "USGS-05227500"]
RIVERS = ["78013722", "73011224", "74053646", "74030313"]
M = pd.read_csv(IN, dtype={"node": str}, float_precision="round_trip").set_index("node")
ratio = M.merit_uparea_km2 / M.uparea_km2
gauge_out = set(M.index[(M.node_kind == "gauge") & ((ratio > 2) | (ratio < 0.5))])
assert gauge_out == set(GAUGES), ("gauges outside 0.5-2x differ from the named four", sorted(gauge_out))
river_rule = M.index[(M.node_kind == "river") & ((ratio > 2) | (ratio < 0.5))]
assert set(RIVERS) <= set(river_rule), set(RIVERS) - set(river_rule)
M["merit_raw_km2"] = M.merit_uparea_km2
M["ratio_raw"] = ratio
M["override_reason"] = ""
for n in GAUGES:
    M.loc[n, "override_reason"] = ("gauge: v3.5 routes the main stem through this gauge node; the delineator's snapped cell is on a MERIT side channel "
                                   f"(MERIT {M.merit_raw_km2[n]:,.1f} vs graph {M.uparea_km2[n]:,.1f} km2); the gauge measures the main stem")
for n in river_rule:
    M.loc[n, "override_reason"] = (f"river: original_comid {int(M.comid[n])} reach end carries {M.merit_raw_km2[n]:,.1f} km2, "
                                   f"{ratio[n]:.3g}x the {M.uparea_km2[n]:,.1f} km2 the node holds in v3.5 (rule: ratio > 2 or < 0.5)")
ov = M.override_reason != ""
M.loc[ov, "merit_uparea_km2"] = M.loc[ov, "uparea_km2"].astype("float64")
M.loc[ov, "source"] = "graph uparea (override)"
M.to_csv(OUT)
print(f"overrides: {int(ov.sum())} ({len(GAUGES)} gauges, {len(river_rule)} river nodes by the 2x rule)")
print(M.loc[ov, ["node_type", "outlet_id", "merit_raw_km2", "uparea_km2", "ratio_raw", "override_reason"]].to_string())
print("wrote", OUT)
