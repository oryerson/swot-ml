"""Re-key the validation list, the test list and the river drop list of the earlier node set to graph v3.5.

Step 2 of the list building for graph v3.5. The "template" lists and drop list are those of an earlier model
configuration (A180, graph version 2.2). Only the validation list, the test list and the river drop list (the
"T9b" list of c_template_store/make_t9b_mask.py) are re-keyed. The training list and the lake-target (fallback)
list are built by rule in build_lists_v35.py.

Usage: python rekey_lists_v35.py      (no arguments; the script stops if any output exists)

Reads (under $SWOTGAT_WORK)
  CONUS/basin_lists/storage/train_nodes_g.csv, valid_nodes_g.csv, test_nodes_g.csv,
  CONUS/basin_lists/fallback/dv_SWOTAE_vd2026_fb_g.csv                 the template lists
  CONUS/basin_lists/swotqc/drop_river_paper_min_present_t9b.parquet    the template river drop list
  CONUS_storage_fork/eval/graph_v3/v35/graphs_conus_v35.json, subbasins_conus_v35.parquet   graph v3.5
  CONUS_storage_fork/eval/v35_inputs/lists/work/map_rivers_v35.csv     old river id -> v3.5 node (map_rivers_v35.py)
  CONUS_storage_fork/eval/v21_retrain/store/t9_nodes.csv, t9b_nodes.csv   node tables of the template drop list
                                                                       (shipped in c_template_store/inputs/)

Writes (L = $SWOTGAT_WORK/CONUS_storage_fork/eval/v35_inputs/lists)
  L/work/valid_rekeyed.csv, test_rekeyed.csv        re-keyed lists; filter_graph_basins_v35.py (step 3) turns
                                                    them into valid_nodes_g_v35.csv and test_nodes_g_v35.csv
  L/node_map_lists_v35.csv                          every template id with its v3.5 id and what happened to it
  L/removed_gauges_v35.csv                          the three gauges taken out of the validation and test lists
  L/drop_river_paper_min_present_t9b_v35.parquet    the re-keyed drop list
  L/t9_nodes_v35.csv, t9b_nodes_v35.csv             the node tables of the drop list, re-keyed
  L/work/rekey_stats.json                           counts

Rules
  * An id that is a v3.5 graph node keeps its rows unchanged, in the same basin (checked). All gauge and lake
    ids are such ids.
  * A river id that no longer exists is mapped to the v3.5 node Y that now holds its reach (map_rivers_v35.py:
    midpoint of the MERIT-Basins reach in a v3.5 polygon of the same basin).
      - If Y is a plain river node (node type original or split_downstream) with no role in any template list
        or in the drop list (a new v3.5 id, or a surviving id without a role), Y takes over what is common to
        every old id mapped to it (validation entry, drop-list rows). The rest is dropped and reported.
      - If Y has a template role (training, validation, test or fallback list, or drop-list rows of its own) or
        is a gauge or reservoir node, Y keeps exactly its template entries. The old id's validation entry and
        drop-list rows are dropped, because its reach is now inside Y, which has its own series.
    So no node enters the validation or test list unless an old node it replaces was there.
  * Drop-list rows moved to Y keep their dates (the same days stay masked). Drop-list rows in basins outside
    the graph are kept unchanged.
  * Three small gauges that sit inside a much larger graph node (USGS-02312667, USGS-13210831, USGS-02264495;
    drainage areas in the table REMOVE) are taken out of the validation and test lists, so they are not scored.
    build_lists_v35.py does not take them out of training. Their drop-list rows are not touched: removing rows
    from a drop list would bring masked days back. The nodes stay in the graph, and so do basin USGS-02264495
    and its other nodes.

Notes
  The drop list the paper's model uses is not the re-keyed one written here. make_t9_mask_v35ra.py and
  make_t9b_mask_v35ra.py rebuild it on graph v3.5; they use the re-keyed node tables t9_nodes_v35.csv and
  t9b_nodes_v35.csv written here.
"""
from swotgat.paths import WORK
import json
from pathlib import Path

import pandas as pd

ROOT = Path(f"{WORK}"); C = ROOT/"CONUS"; CSF = ROOT/"CONUS_storage_fork"
V35 = CSF/"eval/graph_v3/v35"; L = CSF/"eval/v35_inputs/lists"; W = L/"work"; VS = CSF/"eval/v21_retrain/store"
TEMPL = {"train": C/"basin_lists/storage/train_nodes_g.csv", "valid": C/"basin_lists/storage/valid_nodes_g.csv",
         "test": C/"basin_lists/storage/test_nodes_g.csv", "fb": C/"basin_lists/fallback/dv_SWOTAE_vd2026_fb_g.csv"}
MASK = C/"basin_lists/swotqc/drop_river_paper_min_present_t9b.parquet"
REKEY = ["valid", "test", "mask"]; ALL = ["train", "valid", "test", "fb", "mask"]
REMOVE = pd.DataFrame([("USGS-02312667", "SHADY BROOK NEAR SUMTERVILLE FL", 136, 20.7),
                       ("USGS-13210831", "SOUTH MIDDLETON DRAIN AT MIDDLETON ID", 125, 11.7),
                       ("USGS-02264495", "SHINGLE CREEK AT CAMPBELL FL", 940, 466)], columns=["node", "name", "graph_km2", "usgs_km2"])
OUTM = L/"drop_river_paper_min_present_t9b_v35.parquet"
outs = [W/"valid_rekeyed.csv", W/"test_rekeyed.csv", L/"node_map_lists_v35.csv", L/"removed_gauges_v35.csv", OUTM,
        L/"t9_nodes_v35.csv", L/"t9b_nodes_v35.csv"]
assert not any(p.exists() for p in outs), [p for p in outs if p.exists()]

g35 = {str(n["id"]) for n in json.load(open(V35/"graphs_conus_v35.json"))["nodes"]}
S = pd.read_parquet(V35/"subbasins_conus_v35.parquet", columns=["outlet_id", "node_type", "is_gauge", "is_reservoir"]); S.index = S.index.astype(str)
T = {k: pd.read_csv(p, dtype=str) for k, p in TEMPL.items()}
for k, d in T.items(): assert list(d.columns) == ["basin", "subbasin"] and d.subbasin.is_unique, k
base = pd.read_parquet(MASK); assert list(base.columns) == ["basin", "subbasin", "date"]
mem = {}
for k in T:
    for s in T[k].subbasin: mem.setdefault(s, set()).add(k)
for s in base.subbasin.astype(str).unique(): mem.setdefault(s, set()).add("mask")
pat = lambda ks: ";".join(k for k in ALL if k in ks)

M = pd.read_csv(W/"map_rivers_v35.csv", dtype={"old_id": str, "v35_id": str, "overlap_node": str})
bas = {}
for k in T: bas.update({s: b for b, s in T[k][["basin", "subbasin"]].itertuples(index=False)})
for b_, s_ in base[["basin", "subbasin"]].astype(str).drop_duplicates().itertuples(index=False): bas.setdefault(s_, b_)
nomap = {s for s in mem if s not in g35} - set(M.old_id)                       # drop-list-only ids in basins the graph never supplied
assert all(mem[s] == {"mask"} for s in nomap) and not any(bas[s] in set(S.outlet_id[list(g35)]) for s in nomap), sorted(nomap)[:5]
assert {s for k in T for s in T[k].subbasin if s not in g35} <= set(M.old_id), "every gone list id has a mapping row"
assert (M.method != "unmapped").all() and (M.v35_id.map(S.outlet_id) == M.basin).all(), "mapped in the same basin"
M["P"] = M.old_id.map(lambda s: frozenset(mem[s]) & frozenset(REKEY))          # what the old id brings that is re-keyed
M["E"] = M.v35_id.map(lambda s: frozenset(mem.get(s, set())))                     # the target's own template role (any list or mask)
M["plain"] = M.v35_id.map(lambda s: S.node_type[s] in ("original", "split_downstream") and not S.is_gauge[s] and not S.is_reservoir[s])
takeover = {y: frozenset.intersection(*g.P) for y, g in M.groupby("v35_id") if g.plain.iloc[0] and not g.E.iloc[0]}
def decide(r):
    """Decide what the old river id in row r of the mapping table passes on to its v3.5 node.

    Returns a Series with action ("mapped": the v3.5 node takes over all re-keyed entries of the old
    id, "mapped_partial": only those common to every old id mapped to it, "absorbed": none, the v3.5
    node keeps its own entries), rekeyed (the entries taken over), not_transferred (the entries
    dropped) and reason.
    """
    if r.v35_id in takeover:
        keep = takeover[r.v35_id]; act = "mapped" if keep == r.P else "mapped_partial"
        why = "" if keep == r.P else f"other old ids mapped to {r.v35_id} are not in {pat(r.P - keep)}"
    else:
        keep = frozenset(); act = "absorbed"
        why = (f"{r.v35_id} ({'plain' if r.plain else S.node_type[r.v35_id]}; own template role: {pat(r.E) or 'none'}) keeps its own entries; "
               f"old {pat(r.P) or 'train/fallback-only'} entries not transferred")
    return pd.Series(dict(action=act, rekeyed=pat(keep), not_transferred=pat(r.P - keep), reason=why))
M = pd.concat([M, M.apply(decide, axis=1)], axis=1)
sub = dict(zip(M.old_id, M.v35_id)); kept = {o: set(x.split(";")) - {""} for o, x in zip(M.old_id, M.rekeyed)}
rm = set(REMOVE.node); removed_from = {n: [] for n in rm}; stats = {}
for k in ("valid", "test"):
    d = T[k].copy(); n0 = len(d); gone = ~d.subbasin.isin(g35); ok = gone & d.subbasin.map(lambda s: k in kept.get(s, set()))
    d = d[~gone | ok].copy(); d["subbasin"] = d.subbasin.map(lambda s: sub.get(s, s) if s not in g35 else s)
    n1 = len(d); d = d.drop_duplicates("subbasin", keep="first"); dup = n1 - len(d)
    hit = d.subbasin.isin(rm)
    for n in d.subbasin[hit]: removed_from[n].append(f"{TEMPL[k].name} -> {k}_nodes_g_v35.csv")
    d = d[~hit]
    assert d.subbasin.isin(g35).all() and (d.subbasin.map(S.outlet_id) == d.basin).all(), k
    stats[k] = dict(template=n0, gone_from_v35=int(gone.sum()), rekeyed=int(ok.sum()), not_transferred=int((gone & ~ok).sum()),
                    merged_duplicates=dup, owen_removed=int(hit.sum()), after=len(d))
    d.to_csv(W/f"{k}_rekeyed.csv", index=False)
# river mask: same dates, ids re-keyed
m = base.copy(); ms = m.subbasin.astype(str); gone = ~ms.isin(g35) & ms.isin(sub.keys())
outside = ~ms.isin(g35) & ~ms.isin(sub.keys())                                   # basins outside the graph (template has them too)
ok = gone & ms.map(lambda s: "mask" in kept.get(s, set()))
m = m[~gone | ok].copy(); m.loc[ok[~gone | ok], "subbasin"] = ms[ok].map(sub).astype(base.subbasin.dtype).values
n1 = len(m); m = m.drop_duplicates(["basin", "subbasin", "date"]).sort_values(["basin", "subbasin", "date"]).reset_index(drop=True)
assert list(m.columns) == list(base.columns) and m.dtypes.equals(base.dtypes)
m.to_parquet(OUTM, index=False)
stats["mask_t9b"] = dict(template_rows=len(base), template_nodes=int(ms.nunique()), rows_at_gone_ids=int(gone.sum()),
                         nodes_gone=int(ms[gone].nunique()), rows_rekeyed=int(ok.sum()), nodes_rekeyed=int(ms[ok].nunique()),
                         rows_not_transferred=int((gone & ~ok).sum()), nodes_not_transferred=int(ms[gone & ~ok].nunique()),
                         rows_merged_duplicates=n1 - len(m), rows_outside_graph_kept=int(outside.sum()),
                         nodes_outside_graph_kept=int(ms[outside].nunique()), after_rows=len(m), after_nodes=int(m.subbasin.nunique()))
for nm in ("t9_nodes", "t9b_nodes"):                                              # the T9 / T9b node tables of the template drop list, re-keyed
    t = pd.read_csv(VS/f"{nm}.csv", dtype={"node": str}); t.insert(1, "template_node", t.node)
    t["node"] = t.node.map(lambda s: s if s in g35 else (sub[s] if "mask" in kept.get(s, set()) else ""))
    t["rekey"] = ["identity" if a == b else ("mapped" if a else "not_transferred") for a, b in zip(t.node, t.template_node)]
    assert set(t.node[t.node != ""]) <= set(m.subbasin.astype(str)), nm
    t.to_csv(L/f"{nm}_v35.csv", index=False); stats[nm] = t.rekey.value_counts().to_dict()
# full old -> v3.5 table: identity rows for every surviving template id, mapping rows for the ids that no longer exist
rows = []
for s, ks in mem.items():
    if s in sub or s not in g35: continue
    b = bas[s]
    own = s in rm
    rows.append(dict(old_id=s, basin=b, old_lists=pat(ks), v35_id=s, method="identity",
                     action="removed_from_valid_test_owen_20261022" if own and ks & {"valid", "test"} else "kept",
                     rekeyed=pat(ks & set(REKEY)), not_transferred="", reason=""))
M["old_lists"] = M.old_id.map(lambda s: pat(mem[s])); M["v35_template_role"] = M.E.map(pat)
cols = ["old_id", "basin", "old_lists", "v35_id", "method", "action", "rekeyed", "not_transferred", "reason", "v35_node_type", "v35_is_new_id",
        "v35_template_role", "old_node_type", "original_comid", "overlap_node", "overlap_frac", "midpoint_agrees_overlap"]
T2 = pd.concat([M[cols], pd.DataFrame(rows).reindex(columns=cols)], ignore_index=True)
T2["v35_node_type"] = T2.v35_id.map(S.node_type); T2.to_csv(L/"node_map_lists_v35.csv", index=False)
R = REMOVE.copy(); R["reason"] = "Owen 2026-10-22: small gauge inside a much bigger node"
R["in_v35_graph"] = R.node.isin(g35); R["v35_basin"] = R.node.map(S.outlet_id); R["in_template_lists_or_mask"] = R.node.map(lambda n: pat(mem.get(n, set())) or "none")
R["removed_from"] = R.node.map(lambda n: "; ".join(removed_from[n]) or "none needed: not in any A180 list, mask or test set (template or v3.5)")
R.to_csv(L/"removed_gauges_v35.csv", index=False)
json.dump(dict(stats=stats, actions=M.action.value_counts().to_dict(), takeover_targets=len(takeover)), open(W/"rekey_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1)); print(M.action.value_counts().to_string())
print(M[(M.action != "mapped") & (M.P != frozenset({"mask"}))][["old_id", "basin", "old_lists", "v35_id", "v35_node_type", "action", "not_transferred"]].to_string(index=False))
