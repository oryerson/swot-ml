"""Assign each reservoir of the driver tables to a spatial block of hydrologically connected reservoirs.

Blocks group reservoirs that drain to the same part of a river network, so that statistics can
treat a block, not a single reservoir, as the independent unit. The script reads no model score.

Usage: bin/run 07_evaluation/c_scoring/drivers/pa_blocks.py      (no arguments; run before pa_vars.py)

Reads
  the reservoir list, the test-set sites table and the graph through pa_lib.py. Note that the
  list comes from an earlier model configuration's score table (ids only) and that the network is
  graph v2.2, an earlier graph version than the paper's v3.5.
  $SWOTGAT_WORK/CONUS_storage_fork/ResOpsUS/attributes/reservoir_attributes.csv   LAT and LONG

Method (T = 50,000 km2 of upstream area)
  1. From its node, each reservoir walks downstream to the last node whose upstream area is at
     most T. That node names its block ("<outlet>:<node>").
  2. A reservoir whose own upstream area is above T joins the block "<outlet>_main", so that
     cascades on a main stem stay together.
  3. A block with fewer than 3 reservoirs is merged into the block of the nearest downstream
     reservoir: the first reservoir of the list on the downstream path whose block has at least 3
     members. If there is none, it is merged into "<outlet>_main".

Writes out/blocks.csv next to pa_lib.py (DAM_ID, node, outlet, block, lat, lon) and prints the
block sizes. In the paper's run the list has 482 reservoirs.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import pa_lib as L

T = 50_000.0
ids = L.site_ids(); S3 = L.sites(L.T3).loc[ids]; g = L.Graph(); up = g.sb.uparea_km2
att = pd.read_csv(L.F/"ResOpsUS/attributes/reservoir_attributes.csv").set_index("DAM_ID")
rows = []
for d in ids:
    n = S3.node[d]; out = g.sb.outlet_id.get(n, S3.basin[d])
    if up.get(n, np.nan) > T: b = f"{out}_main"
    else:
        b, path = n, g.down_path(n)
        for x in path:
            if up.get(x, np.inf) > T: break
            b = x
        b = f"{out}:{b}"
    rows.append(dict(DAM_ID=d, node=n, outlet=out, block0=b, lat=att.LAT.get(d), lon=att.LONG.get(d)))
B = pd.DataFrame(rows).set_index("DAM_ID"); B["block"] = B.block0
size = B.block.value_counts(); small = set(size[size < 3].index); node2dam = {B.node[d]: d for d in B.index}
for d in B.index:
    if B.block[d] not in small: continue
    new = f"{B.outlet[d]}_main"
    for x in g.down_path(B.node[d]):
        if x in node2dam and B.block0[node2dam[x]] not in small: new = B.block0[node2dam[x]]; break
    B.loc[d, "block"] = new
B.drop(columns="block0").to_csv(L.OUT/"blocks.csv")
vc = B.block.value_counts()
print(f"{len(B)} reservoirs, {B.outlet.nunique()} graph basins, {len(vc)} blocks; sizes: max {vc.max()}, median {vc.median()}, "
      f"blocks < 3: {(vc < 3).sum()}; size-weighted mean block size {float((vc**2).sum()/vc.sum()):.1f}")
print(vc.head(15).to_string())
print("Mississippi (USGS-07374000) blocks:", B[B.outlet == "USGS-07374000"].block.nunique(), "| largest outlet sizes:",
      B.outlet.value_counts().head(5).to_dict())
