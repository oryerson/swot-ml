"""Check the inputs before, and the outputs after, one prediction run on graph v3.5.

06_predictions/predict.sh uses this script with the low-memory runner pred_v35_lowmem.py.

Why: if a graph node is missing from a basin group of the prediction store, the data loader fills it with NaN.
The NaN spreads through the graph attention layers to the whole basin, and the scoring scripts then drop those
reservoirs without notice. These checks stop the run instead.

Usage: python pred_assert_v35.py pre  <checkpoint> <store> <nodes.csv> <start> <end>
       python pred_assert_v35.py post <checkpoint> <parquet> <nodes.csv> <start> <end>
  <checkpoint>  folder <run>/checkpoints/step_NNNNNN with config.json
  <store>       prediction forcing store, one group per basin
  <parquet>     prediction file written by the runner
  <nodes.csv>   columns basin, subbasin: the nodes to predict
  <start> <end> prediction window (YYYY-MM-DD)
  Exit status 1 if any check fails. 06_predictions/predict.sh runs both modes around the runner.
  v35lib.py is imported from the parent of this script's folder (the script is deployed to
  CONUS_storage_fork/eval/v35_launch/eval/, v35lib.py to v35_launch/).

Checks in mode `pre`
  The graph file named in the checkpoint has the md5 of graph v3.5. Then, for every basin of the node list:
  - the basin is the outlet of a connected component of the graph, and the store has its group;
  - the group's normalization_stats equal those of the training store, so prediction inputs are scaled as in
    training;
  - every node of the basin's graph component is a column of the group;
  - the model's ERA5-Land inputs are in the group;
  - the dates cover the inputs of the window, from <start> minus sequence_length days to <end>;
  - the forcing is finite at every graph node of the basin over those dates;
  - for a model with long-memory inputs: the long-memory store named in the checkpoint has the basin, with the
    same node set, the same date coverage and the model's long-memory variables.

Checks in mode `post`
  - every requested node has a row on every day of the window;
  - pred_dv_SWOTAE and pred_dv_river are finite in every such row.
  If a check fails, the parquet and its .json file are renamed to *.FAILED_CHECK so that no scoring script reads
  them.
"""
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr

MODE, CK, TARGET, NODES, START, END = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5], sys.argv[6]
cfg = json.load(open(CK/"config.json")); want = pd.read_csv(NODES, dtype=str); bad = []


def zopen(g):
    """Open the basin group g of a store, with consolidated metadata if the group has it."""
    try: return xr.open_zarr(g, consolidated=True)
    except Exception: return xr.open_zarr(g, consolidated=False)   # groups without consolidated metadata (the older stores)


if MODE == "pre":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import v35lib as V   # noqa: E402
    gm = V.md5(cfg["graph_network_file"])
    if gm != V.GRAPH_MD5: bad.append(f"checkpoint graph md5 {gm} is not v3.5's {V.GRAPH_MD5}: not a v3.5 checkpoint")
    J = nx.readwrite.json_graph.node_link_graph(json.load(open(cfg["graph_network_file"])), edges="edges")
    comp = {[n for n in c if J.out_degree(n) == 0][0]: c for c in nx.weakly_connected_components(J)}
    w0 = (pd.Timestamp(START) - pd.Timedelta(days=int(cfg["sequence_length"]))).strftime("%Y-%m-%d")
    era5 = [v for v in cfg["features"]["dynamic"]["era5"] if not v.startswith("lm_")]
    lm = cfg.get("extra_dynamic_vars") or []
    for b in sorted(want.basin.unique()):
        if b not in comp: bad.append(f"{b}: not a basin of {Path(cfg['graph_network_file']).name}"); continue
        g = TARGET/b
        if not g.exists(): bad.append(f"{b}: no group in {TARGET.name}"); continue
        z = zopen(g); ss = set(z.subbasin.values.astype(str)); miss = sorted(set(comp[b]) - ss)
        nt = zopen(Path(cfg["zarr_dir"])/b).attrs.get("normalization_stats")      # the prediction store must scale inputs exactly as the training store does
        if z.attrs.get("normalization_stats") is None or z.attrs.get("normalization_stats") != nt:
            bad.append(f"{b}: normalization_stats differ from the training store {Path(cfg['zarr_dir']).name}")
        if miss: bad.append(f"{b}: {len(miss)} of {len(comp[b])} graph nodes are not columns of the store group (e.g. {miss[:4]})")
        nv = [v for v in era5 if v not in z]
        if nv: bad.append(f"{b}: no {nv}")
        d0, d1 = str(z.date.values[0])[:10], str(z.date.values[-1])[:10]
        if d0 > w0 or d1 < END: bad.append(f"{b}: store dates {d0}..{d1}, the window needs {w0}..{END}")
        elif not miss and not nv:   # forcing must be finite at every graph node of the basin over the window's inputs
            zz = z.assign_coords(subbasin=z.subbasin.values.astype(str)).sel(date=slice(w0, END)).sel(subbasin=sorted(comp[b]))
            nf = {v: int((~np.isfinite(np.asarray(zz[v].values))).sum()) for v in era5}
            nf = {v: k for v, k in nf.items() if k}
            if nf: bad.append(f"{b}: non-finite forcing in {w0}..{END}: {nf}")
        if lm:
            e = Path(cfg["extra_dynamic_store"])/b
            if not e.exists(): bad.append(f"{b}: no long-memory group"); continue
            x = zopen(e)
            if set(x.subbasin.values.astype(str)) != ss: bad.append(f"{b}: long-memory node set != the prediction store's")
            if str(x.date.values[0])[:10] > w0 or str(x.date.values[-1])[:10] < END: bad.append(f"{b}: long-memory dates do not cover {w0}..{END}")
            nv = [v for v in lm if v not in x]
            if nv: bad.append(f"{b}: long-memory store lacks {nv}")
    msg = f"{len(want)} nodes in {want.basin.nunique()} basins; store {TARGET}"
else:
    x = pd.read_parquet(TARGET, columns=["basin", "subbasin", "date", "pred_dv_SWOTAE", "pred_dv_river"])
    x["subbasin"] = x.subbasin.astype(str); x["date"] = pd.to_datetime(x.date)
    x = x[(x.date >= START) & (x.date <= END)]
    days = len(pd.date_range(START, END)); n = x.groupby("subbasin").date.nunique()
    short = sorted(set(want.subbasin) - set(n[n == days].index))
    if short: bad.append(f"{len(short)} requested nodes without a row on every one of {days} days (e.g. {short[:5]})")
    x = x[x.subbasin.isin(set(want.subbasin))]
    for c in ("pred_dv_SWOTAE", "pred_dv_river"):
        k = ~np.isfinite(x[c].values)
        if k.any(): bad.append(f"{c} non-finite in {int(k.sum()):,} rows at {x.subbasin[k].nunique()} nodes (e.g. {sorted(x.subbasin[k].unique())[:5]})")
    msg = f"{len(x):,} rows at {x.subbasin.nunique()} of {len(want)} requested nodes, {START}..{END}"
    if bad:
        for f in (TARGET, TARGET.with_suffix(".json")):
            if f.exists(): f.rename(f.with_name(f.name + ".FAILED_CHECK"))
print(f"pred_assert {MODE}: {msg}"); [print("FAIL", b) for b in bad]
print(f"PRED ASSERT {MODE.upper()}", "PASSED" if not bad else f"FAILED ({len(bad)})"); sys.exit(1 if bad else 0)
