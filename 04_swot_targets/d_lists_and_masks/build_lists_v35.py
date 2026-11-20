"""Build the training node list and the lake-target node list for graph v3.5 by a fixed rule.

Design choice: there is no spatial hold-out. Every node with a SWOT target is trained on, and the model is
tested out of sample in time only (1981-2020), at all locations, river gauges and reservoirs alike.

Usage: python build_lists_v35.py --store <store dir> [--graph G] [--subbasins S] [--valid V] [--test T]
                                 [--out-dir D] [--suffix _v35] [--start 2023-01-01] [--end 2026-09-30]
                                 [--compare-train FILE] [--compare-fb FILE] [--compare-map FILE]
  --store          training store to scan for targets (required), one group per basin
  --graph, --subbasins   graph v3.5 JSON and sub-basin table (defaults under
                   $SWOTGAT_WORK/CONUS_storage_fork/eval/graph_v3/v35/; their md5 sums are checked)
  --valid, --test  validation and test lists on graph v3.5 (defaults valid_nodes_g_v35.csv and
                   test_nodes_g_v35.csv in $SWOTGAT_WORK/CONUS_storage_fork/eval/v35_inputs/lists/, written by
                   rekey_lists_v35.py and filter_graph_basins_v35.py). They are inputs and are not rewritten.
  --out-dir, --suffix    where to write and the suffix of the output names (defaults: that lists folder, _v35)
  --start, --end   period scanned for targets
  --compare-train, --compare-fb   the training and lake-target lists of an earlier model configuration (A180),
                   used only for the difference report
  --compare-map    table old_id,v35_id that translates the ids of those lists to graph v3.5

Writes (to --out-dir; the script refuses to overwrite any of them)
  train_nodes_g<suffix>.csv            the training list (basin, subbasin)
  dv_SWOTAE_vd2026_fb_g<suffix>.csv    the lake-target list, called the fallback list in the code (basin,
                                       subbasin). The model configuration reads it as
                                       target_basin_files.dv_SWOTAE.
  node_targets<suffix>.csv             per graph node: number of finite days of each target, flags, list membership
  build_lists<suffix>.json             counts, gates and the comparison
  diff_train<suffix>.csv, diff_fb<suffix>.csv   nodes added or removed relative to the compared lists, with a reason

Method
  Training list = the union, with nothing subtracted, of
    - every graph node with at least one finite value of dv_river, dv_SWOTAE or dv_staticPLD in --store between
      --start and --end (the whole period, 2026 included);
    - every node of the test list;
    - every node of the validation list.
  Lake-target list = every lake node (is_reservoir in the sub-basin table) with at least one finite dv_SWOTAE or
  dv_staticPLD in the same period, plus every lake node of the test list.
  A node's basin is the outlet of its connected component in the graph. The script asserts that this equals
  outlet_id in the sub-basin table and the store group that holds the node.
  Gates (exit status 1 if one fails): every test and validation node is in the training list, every node with a
  target is in the training list, every lake with a target and every test lake is in the lake-target list.

Notes
  Lists always get new file names. The training cache is keyed by a hash that covers the paths of the lists, not
  their content, so a list must never be rewritten in place.
"""
from swotgat.paths import WORK
import argparse, hashlib, json, re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import zarr

ROOT = Path(f"{WORK}"); C = ROOT/"CONUS"; CSF = ROOT/"CONUS_storage_fork"
V35 = CSF/"eval/graph_v3/v35"; L = CSF/"eval/v35_inputs/lists"
MD5 = {str(V35/"subbasins_conus_v35.parquet"): "19a764d997a07c870b2edb0688dbf3a8", str(V35/"graphs_conus_v35.json"): "c580577d7581bbfb975c261bea609051"}
# three gauges whose list membership is reported explicitly in the JSON (key "owen_gauges")
OWEN_GAUGES = ["USGS-02312667", "USGS-13210831", "USGS-02264495"]
ap = argparse.ArgumentParser()
ap.add_argument("--store", required=True, type=Path)
ap.add_argument("--graph", type=Path, default=V35/"graphs_conus_v35.json"); ap.add_argument("--subbasins", type=Path, default=V35/"subbasins_conus_v35.parquet")
ap.add_argument("--valid", type=Path, default=L/"valid_nodes_g_v35.csv"); ap.add_argument("--test", type=Path, default=L/"test_nodes_g_v35.csv")
ap.add_argument("--out-dir", type=Path, default=L); ap.add_argument("--suffix", default="_v35")
ap.add_argument("--start", default="2023-01-01"); ap.add_argument("--end", default="2026-09-30")
ap.add_argument("--compare-train", type=Path, default=C/"basin_lists/storage/train_nodes_g.csv")
ap.add_argument("--compare-fb", type=Path, default=C/"basin_lists/fallback/dv_SWOTAE_vd2026_fb_g.csv")
ap.add_argument("--compare-map", type=Path, default=None, help="old_id,v35_id table to translate the compare lists' ids (v3.5 run)")
a = ap.parse_args()
OUT_T = a.out_dir/f"train_nodes_g{a.suffix}.csv"; OUT_F = a.out_dir/f"dv_SWOTAE_vd2026_fb_g{a.suffix}.csv"
OUT_N = a.out_dir/f"node_targets{a.suffix}.csv"; OUT_J = a.out_dir/f"build_lists{a.suffix}.json"
OUT_D = [a.out_dir/f"diff_train{a.suffix}.csv", a.out_dir/f"diff_fb{a.suffix}.csv"]
assert not any(p.exists() for p in (OUT_T, OUT_F, OUT_N, OUT_J, *OUT_D)), "refusing to overwrite"
for p in (a.graph, a.subbasins):
    if str(p) in MD5: assert hashlib.md5(open(p, "rb").read()).hexdigest() == MD5[str(p)], f"md5 {p}"

# ---- graph: node -> basin (component outlet) ----
G = nx.readwrite.json_graph.node_link_graph(json.load(open(a.graph)), edges="edges"); gb = {}
for comp in nx.weakly_connected_components(G):
    outs = [n for n in comp if G.out_degree(n) == 0]; assert len(outs) == 1, "component with != 1 outlet"
    for n in comp: gb[str(n)] = str(outs[0])
S = pd.read_parquet(a.subbasins, columns=["outlet_id", "is_reservoir", "is_gauge"]); S.index = S.index.astype(str)
assert all(S.outlet_id.get(n) == b for n, b in gb.items()), "graph basin != subbasins outlet_id"
lake = {n for n in gb if bool(S.is_reservoir[n])}

# ---- store: any finite target per node over the window ----
t0, t1 = pd.Timestamp(a.start), pd.Timestamp(a.end); t26 = pd.Timestamp("2026-01-01")
VARS = ["dv_river", "dv_SWOTAE", "dv_staticPLD"]; rec = []; groups = sorted(p.name for p in a.store.iterdir() if (p/"zarr.json").is_file())
for b in groups:
    g = zarr.open_group(str(a.store/b), mode="r"); subs = [str(s) for s in g["subbasin"][:]]; da = g["date"]
    m = re.fullmatch(r"days since (\d{4}-\d{2}-\d{2})(?:[ T]00:00:00)?", da.attrs.get("units", "")); assert m and da.attrs.get("calendar", "proleptic_gregorian") in ("proleptic_gregorian", "standard", "gregorian"), (b, dict(da.attrs))
    dates = pd.Timestamp(m.group(1)) + pd.to_timedelta(np.asarray(da[:], dtype="int64"), unit="D")
    i = np.flatnonzero((dates >= t0) & (dates <= t1)); i0, i1 = int(i[0]), int(i[-1]) + 1; assert (np.diff(i) == 1).all()
    k26 = int(np.searchsorted(dates[i0:i1], t26))
    cnt = {}
    for v in VARS:
        n_all = np.zeros(len(subs), int); n_pre = np.zeros(len(subs), int)
        if v in g.array_keys():
            arr = g[v]; assert arr.shape == (len(dates), len(subs)), (b, v, arr.shape)
            for j in range(0, len(subs), 2000):
                x = np.isfinite(arr[i0:i1, j:j + 2000]); n_all[j:j + 2000] = x.sum(0); n_pre[j:j + 2000] = x[:k26].sum(0)
        cnt[v] = (n_all, n_pre)
    for j, s in enumerate(subs):
        rec.append(dict(group=b, subbasin=s, **{f"n_{v}": int(cnt[v][0][j]) for v in VARS}, **{f"n_{v}_2023_25": int(cnt[v][1][j]) for v in VARS}))
N = pd.DataFrame(rec); assert N.subbasin.is_unique, "a subbasin in two store groups"
N["in_graph"] = N.subbasin.isin(gb.keys()); N["basin"] = N.subbasin.map(gb)
assert (N.basin[N.in_graph] == N.group[N.in_graph]).all(), "store group != graph basin"
miss = sorted(set(gb) - set(N.subbasin)); assert not miss, f"graph nodes not in the store: {len(miss)} e.g. {miss[:5]}"
N["is_lake"] = N.subbasin.isin(lake); N["is_gauge"] = N.subbasin.map(lambda s: bool(S.is_gauge.get(s, False)))
N["any_target"] = (N[[f"n_{v}" for v in VARS]] > 0).any(axis=1); N["lake_target"] = (N.n_dv_SWOTAE > 0) | (N.n_dv_staticPLD > 0)
N["target_2026_only"] = N.any_target & ~(N[[f"n_{v}_2023_25" for v in VARS]] > 0).any(axis=1)
Ng = N[N.in_graph].copy()

# ---- lists ----
def rd(p):
    """Read the node list in file p (columns basin, subbasin) and check it.

    Asserts that no sub-basin is listed twice and that every id is a graph node of the stated basin.
    Returns the list as a table.
    """
    d = pd.read_csv(p, dtype=str); assert list(d.columns) == ["basin", "subbasin"] and d.subbasin.is_unique, p
    bad = d[(d.subbasin.map(gb) != d.basin)]; assert bad.empty, f"{p}: {len(bad)} ids not graph nodes of the stated basin, e.g. {bad.head(3).values.tolist()}"
    return d
VA, TE = rd(a.valid), rd(a.test)
tr = set(Ng.subbasin[Ng.any_target]) | set(TE.subbasin) | set(VA.subbasin)
fb = set(Ng.subbasin[Ng.is_lake & Ng.lake_target]) | (set(TE.subbasin) & lake)
out = lambda ids: pd.DataFrame({"basin": [gb[s] for s in ids], "subbasin": list(ids)}).sort_values(["basin", "subbasin"]).reset_index(drop=True)
TR, FB = out(tr), out(fb)
a.out_dir.mkdir(parents=True, exist_ok=True); TR.to_csv(OUT_T, index=False); FB.to_csv(OUT_F, index=False)
Ng["in_train"] = Ng.subbasin.isin(tr); Ng["in_fb"] = Ng.subbasin.isin(fb); Ng["in_valid"] = Ng.subbasin.isin(set(VA.subbasin)); Ng["in_test"] = Ng.subbasin.isin(set(TE.subbasin))
Ng.drop(columns=["in_graph", "group"]).to_csv(OUT_N, index=False)

# ---- gates ----
R = dict(store=str(a.store), graph=str(a.graph), subbasins=str(a.subbasins), window=[a.start, a.end], store_groups=len(groups),
         graph_basins=len(set(gb.values())), graph_nodes=len(gb), store_columns=len(N), store_columns_outside_graph=int((~N.in_graph).sum()),
         store_columns_outside_graph_with_target=int((~N.in_graph & N.any_target).sum()))
R["counts"] = {"train": len(TR), "train_basins": int(TR.basin.nunique()), "fallback": len(FB), "fallback_basins": int(FB.basin.nunique()),
               "valid": len(VA), "test": len(TE), "nodes_with_any_target": int(Ng.any_target.sum()), "river_target_nodes": int((Ng.n_dv_river > 0).sum()),
               "lake_target_nodes": int((Ng.is_lake & Ng.lake_target).sum()), "test_lakes": len(set(TE.subbasin) & lake),
               "target_2026_only": int(Ng.target_2026_only.sum())}
R["gates"] = {"test_minus_train": len(set(TE.subbasin) - tr), "valid_minus_train": len(set(VA.subbasin) - tr),
              "target_nodes_not_in_train": int((Ng.any_target & ~Ng.in_train).sum()),
              "lake_target_nodes_not_in_fallback": int((Ng.is_lake & Ng.lake_target & ~Ng.in_fb).sum()),
              "test_lakes_not_in_fallback": len((set(TE.subbasin) & lake) - fb),
              "nonlake_nodes_with_dv_SWOTAE_or_dv_staticPLD": int((~Ng.is_lake & Ng.lake_target).sum()),
              "fallback_nonlake_rows": int((~FB.subbasin.isin(lake)).sum())}
R["owen_gauges"] = {n: dict(in_graph=n in gb, any_target=bool(Ng.any_target[Ng.subbasin == n].any()), in_train=n in tr, in_test=n in set(TE.subbasin),
                            in_valid=n in set(VA.subbasin), in_fallback=n in fb) for n in OWEN_GAUGES}
# ---- comparison with the lists of an earlier model configuration (A180); report only ----
mp = {}
if a.compare_map:
    m_ = pd.read_csv(a.compare_map, dtype=str); mp = dict(zip(m_.old_id, m_.v35_id))
def cmp(p):
    """Read the compared list in file p and translate its ids to graph v3.5.

    Returns (the set of translated ids, the number of rows, the number of ids that are not graph
    nodes either as listed or after the translation).
    """
    d = pd.read_csv(p, dtype=str); ids = {mp.get(s, s) for s in d.subbasin}
    return ids, len(d), len({s for s in d.subbasin if s not in gb and mp.get(s, s) not in gb})
def kind(s):
    """Return the kind of node s: "lake", "gauge" or "river"."""
    return "lake" if s in lake else ("gauge" if s.startswith(("USGS-", "ECCC-")) or bool(S.is_gauge.get(s, False)) else "river")
info = Ng.set_index("subbasin")
def why_added(s):
    """Return the reason why node s is in the training list.

    The reason is reported for the nodes that are absent from the compared list: the node is only in
    the test or validation list and has no target, or it has a target in 2026 only, or it has a
    target in 2023-2025 (named by dv_river first, then dv_SWOTAE, else dv_staticPLD only).
    """
    r = info.loc[s]
    if not r.any_target: return "test_only_no_target" if r.in_test else ("valid_only_no_target" if r.in_valid else "?")
    if r.target_2026_only: return "target_2026_only"
    return "target_2023_25_dv_river" if r.n_dv_river_2023_25 > 0 else ("target_2023_25_dv_SWOTAE" if r.n_dv_SWOTAE_2023_25 > 0 else "target_2023_25_dv_staticPLD_only")
if a.compare_train and a.compare_train.exists():
    old, n_old, n_unres = cmp(a.compare_train); add = sorted(tr - old); rem = sorted(old - tr)
    A = pd.DataFrame({"subbasin": add}); A["kind"] = A.subbasin.map(kind); A["why"] = A.subbasin.map(why_added)
    Rm = pd.DataFrame({"subbasin": rem}); Rm["kind"] = Rm.subbasin.map(kind)
    Rm["why"] = Rm.subbasin.map(lambda s: "not a graph node" if s not in gb else ("no finite target in store, not in test/valid" if not info.any_target.get(s, False) else "?"))
    R["vs_compare_train"] = dict(file=str(a.compare_train), rows=n_old, ids_unresolved=n_unres, kept=len(tr & old), added=len(add), removed=len(rem),
                                 added_by_why_kind=A.groupby(["why", "kind"]).size().rename("n").reset_index().to_dict("records"),
                                 removed_by_why_kind=Rm.groupby(["why", "kind"]).size().rename("n").reset_index().to_dict("records") if len(Rm) else [],
                                 test_minus_compare_train=len(set(TE.subbasin) - old))
    pd.concat([A.assign(change="added"), Rm.assign(change="removed")]).assign(list="train").to_csv(a.out_dir/f"diff_train{a.suffix}.csv", index=False)
if a.compare_fb and a.compare_fb.exists():
    old, n_old, n_unres = cmp(a.compare_fb); oldl = {s for s in old if s in lake}; add = sorted(fb - old); rem = sorted(old - fb)
    Rm = pd.DataFrame({"subbasin": rem}); Rm["kind"] = Rm.subbasin.map(kind)
    Rm["why"] = Rm.subbasin.map(lambda s: "not a graph node" if s not in gb else ("not a lake node (rule: lakes only)" if s not in lake else "lake without finite dv_SWOTAE/dv_staticPLD, not a test lake"))
    A = pd.DataFrame({"subbasin": add}); A["kind"] = A.subbasin.map(kind)
    A["why"] = A.subbasin.map(lambda s: "test lake without target" if not info.lake_target.get(s, False) else ("target_2026_only" if info.target_2026_only.get(s, False) else "target 2023-25"))
    R["vs_compare_fb"] = dict(file=str(a.compare_fb), rows=n_old, compare_lake_rows=len(oldl), ids_unresolved=n_unres, kept=len(fb & old), added=len(add), removed=len(rem),
                              added_by_why_kind=A.groupby(["why", "kind"]).size().rename("n").reset_index().to_dict("records") if len(A) else [],
                              removed_by_why_kind=Rm.groupby(["why", "kind"]).size().rename("n").reset_index().to_dict("records") if len(Rm) else [])
    pd.concat([A.assign(change="added"), Rm.assign(change="removed")]).assign(list="fallback").to_csv(a.out_dir/f"diff_fb{a.suffix}.csv", index=False)
json.dump(R, open(OUT_J, "w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))
ok = R["gates"]["test_minus_train"] == 0 and R["gates"]["valid_minus_train"] == 0 and R["gates"]["target_nodes_not_in_train"] == 0 \
     and R["gates"]["lake_target_nodes_not_in_fallback"] == 0 and R["gates"]["test_lakes_not_in_fallback"] == 0
print(f"GATES {'PASS' if ok else 'FAIL'} -> {OUT_T} ({len(TR):,}), {OUT_F} ({len(FB):,})"); raise SystemExit(0 if ok else 1)
