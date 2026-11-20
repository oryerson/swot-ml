"""Build the reach-area table for graph v3.5, so that river nodes with new ids also get a storage target.

The river target is level anomaly x reach area. The reach-area table of the first graph version ("v1";
build_storage_target_conus.py -> CONUS/metadata/reach_area_conus.csv) has no rows for node ids that are new in
graph v3.5 (the paper's graph). This script writes a table for graph v3.5 that keeps every v1 value it can and
computes the rest.

Usage: python reach_area_v35.py      (no arguments)

Reads (W = $SWOTGAT_WORK/CONUS_storage_fork/eval/v35_inputs/forcing/work; nothing is modified)
  $SWOTGAT_WORK/CONUS/zarr_store                 first data store; its area_total_river is the record the v1
                                                 table was computed from
  $SWOTGAT_WORK/CONUS/metadata/reach_area_conus.csv   the v1 table
  W/river/reach_lookup_v1.csv, reach_lookup_v35.csv   node -> SWOT reach crosswalks of the two graph versions
                                                      (river_xwalk_v1.py, river_build_v35.py)
  W/river/riversp_vD_present_graph_v35.parquet   daily river record rebuilt on graph v3.5 (river_build_v35.py)
  W/node_map.csv                                 the nodes of graph v3.5, in sub-basin table order

Writes (to W/river/)
  reach_area_conus_v35.csv             same format as the v1 table: subbasin, reach_area_m2, n_area_obs, cv_area.
                                       The script stops if this file already exists.
  reach_area_v35_all_reach_nodes.csv   every v3.5 node with a reach: source of its value, number of reaches,
                                       number of area observations, row written or not
  reach_area_v35_summary.json          the counts printed at the end

Method
  Rule (as in build_storage_target_conus.py): a node with at least 10 finite area_total_river observations gets
  reach_area_m2 = round(median, 1), n_area_obs, and cv_area = round(standard deviation / mean, 4).
  A  The rule is applied to the first store's record for every v1 node and compared with the v1 table. This
     tests the implementation of the rule; the result is reported, and it should reproduce the table exactly.
  B  The rule is applied to the rebuilt v3.5 record, limited to the date range of the first store's record, for
     every v3.5 node with records. The agreement with the v1 table on retained ids is reported.
  Final table, per v3.5 node with a reach:
    - a retained id (same id, identical set of reaches) takes its v1 row unchanged, or no row if v1 had none;
    - a new id whose set of reaches equals that of exactly one v1 node that no longer exists takes that node's
      v1 row unchanged (the reach's area exactly as the v1 node had it);
    - any other new id takes its rule-B row, if it has at least 10 observations.
  The script asserts that no retained id has a changed crosswalk.

Notes
  In this script the standard deviation of rule-B rows is computed on a nullable pandas column, which uses
  ddof=1, while build_storage_target_conus.py uses numpy (ddof=0). Only cv_area of rule-B rows is affected.
  reach_area_v35_fix.py is the corrected run; its table is the one to use.
"""
from swotgat.paths import WORK
import json, numpy as np, pandas as pd, xarray as xr
from pathlib import Path
ROOT = Path(f"{WORK}"); C = ROOT/"CONUS"; F = ROOT/"CONUS_storage_fork/eval/v35_inputs/forcing"; W = F/"work"
MIN_AREA_OBS = 10
def rule(a):
    """Apply the reach-area rule to the area observations a of one node.

    Returns None if a has fewer than 10 finite values. Otherwise returns a dict with reach_area_m2
    (the median of the finite values, rounded to 0.1 m2), n_area_obs (their number) and cv_area
    (their standard deviation divided by their mean, rounded to 4 decimals).
    """
    fa = np.isfinite(a)
    if fa.sum() < MIN_AREA_OBS: return None
    return dict(reach_area_m2=round(float(np.median(a[fa])), 1), n_area_obs=int(fa.sum()), cv_area=round(float(a[fa].std()/a[fa].mean()), 4))
V1 = pd.read_csv(C/"metadata/reach_area_conus.csv", dtype={"subbasin": str}); V1i = V1.set_index("subbasin")
# ---- A: the rule on the first store's record (also finds the date range of that record)
Z = C/"zarr_store"; A = {}; dmin, dmax = None, None
for b in sorted(p.name for p in Z.iterdir() if (p/"zarr.json").is_file()):
    if not (Z/b/"area_total_river").is_dir(): continue
    ds = xr.open_zarr(Z/b, consolidated=False); a = ds["area_total_river"].transpose("date", "subbasin").values; d = pd.DatetimeIndex(ds.date.values)
    fin = np.isfinite(a).any(1)
    if fin.any():
        lo, hi = d[fin].min(), d[fin].max(); dmin = lo if dmin is None else min(dmin, lo); dmax = hi if dmax is None else max(dmax, hi)
    for j, s in enumerate(ds.subbasin.values.astype(str)):
        r = rule(a[:, j])
        if r: A[s] = r
A = pd.DataFrame.from_dict(A, orient="index")
com = A.index.intersection(V1i.index)
eqA = (A.loc[com] == V1i.loc[com, ["reach_area_m2", "n_area_obs", "cv_area"]]).all(1)
resA = dict(v1_rows=len(V1), rule_rows=len(A), common=len(com), exact=int(eqA.sum()), only_rule=len(A.index.difference(V1i.index)), only_v1=len(V1i.index.difference(A.index)),
            production_record_window=[str(dmin.date()), str(dmax.date())])
print("A", json.dumps(resA), flush=True)
# ---- B: the rule on the rebuilt v3.5 record, within the date range of the first store's record
R = pd.read_parquet(W/"river/riversp_vD_present_graph_v35.parquet", columns=["subbasin", "date", "area_total_river"]); R["subbasin"] = R.subbasin.astype(str)
Rw = R[(R.date >= dmin) & (R.date <= dmax)]
B = {s: r for s, g in Rw.groupby("subbasin") if (r := rule(g.area_total_river.values))}
B = pd.DataFrame.from_dict(B, orient="index")
NM = pd.read_csv(W/"node_map.csv", dtype={"subbasin": str, "outlet_id": str}, keep_default_na=False).set_index("subbasin")
def xw(p):
    """Read the node -> SWOT reach crosswalk in file p.

    Returns a dict that maps each node id to the sorted tuple of its distinct reach ids.
    """
    x = pd.read_csv(p, dtype={"subbasin": str}); x["reach_id"] = x.reach_id.astype(float).astype(np.int64)
    return x.groupby("subbasin").reach_id.agg(lambda s: tuple(sorted(set(s)))).to_dict()
X1, X35 = xw(W/"river/reach_lookup_v1.csv"), xw(W/"river/reach_lookup_v35.csv")
ret = [s for s in X35 if s in X1 and X1[s] == X35[s]]; new = [s for s in X35 if s not in X1]; chg = [s for s in X35 if s in X1 and X1[s] != X35[s]]
assert not chg, f"{len(chg)} retained ids with a changed crosswalk"
retv1 = [s for s in ret if s in V1i.index]; cb = [s for s in retv1 if s in B.index]
eqB = (B.loc[cb] == V1i.loc[cb, ["reach_area_m2", "n_area_obs", "cv_area"]]).all(1)
relB = (B.loc[cb, "reach_area_m2"]/V1i.loc[cb, "reach_area_m2"] - 1).abs()
resB = dict(retained_ids_with_v1_row=len(retv1), rule_B_rows_for_them=len(cb), exact=int(eqB.sum()), within_0p1pct=int((relB < 1e-3).sum()), within_1pct=int((relB < 1e-2).sum()),
            n_obs_equal=int((B.loc[cb, "n_area_obs"] == V1i.loc[cb, "n_area_obs"]).sum()), retained_ids_without_v1_row_but_rule_B=int(sum(s in B.index for s in ret if s not in V1i.index)))
print("B", json.dumps(resB), flush=True)
# ---- v1 nodes that no longer exist in v3.5, grouped by their set of reaches; then the final table
gone = [s for s in X1 if s not in NM.index]; byset = {}
for s in gone:
    if s in V1i.index: byset.setdefault(X1[s], []).append(s)
rows, src = [], {}
for s in NM.index:                                   # order of the v3.5 sub-basin table
    if s not in X35: continue
    if s in ret:
        if s in V1i.index: rows.append(dict(subbasin=s, **V1i.loc[s, ["reach_area_m2", "n_area_obs", "cv_area"]].to_dict())); src[s] = "v1_row_same_id"
        else: src[s] = "same_id_no_v1_row"
        continue
    g = byset.get(X35[s], [])
    if len(g) == 1:
        rows.append(dict(subbasin=s, **V1i.loc[g[0], ["reach_area_m2", "n_area_obs", "cv_area"]].to_dict())); src[s] = f"v1_row_of_gone:{g[0]}"
    elif s in B.index:
        rows.append(dict(subbasin=s, **B.loc[s].to_dict())); src[s] = "rule_B" + (f"_ambiguous_gone:{'|'.join(g)}" if len(g) > 1 else "")
    else: src[s] = "fewer_than_10_obs_or_no_records"
T = pd.DataFrame(rows); T["n_area_obs"] = T.n_area_obs.astype(int)
out = W/"river/reach_area_conus_v35.csv"; assert not out.exists(), out
T.to_csv(out, index=False)
allr = pd.DataFrame({"subbasin": list(src), "source": list(src.values())}); allr["n_reaches"] = [len(X35[s]) for s in allr.subbasin]
allr["n_area_obs_rule_B"] = [int(B.n_area_obs.get(s, 0)) if s in B.index else 0 for s in allr.subbasin]
allr["row_written"] = allr.subbasin.isin(set(T.subbasin)); allr["in_template"] = [s in X1 for s in allr.subbasin]
allr.to_csv(W/"river/reach_area_v35_all_reach_nodes.csv", index=False)
newsrc = allr[~allr.in_template].source.str.split(":").str[0].value_counts().to_dict()
# new ids that take the row of a v1 node that no longer exists: does rule B agree with that row?
gn = [(s, src[s].split(":")[1]) for s in new if src.get(s, "").startswith("v1_row_of_gone")]
agree = sum(1 for s, g in gn if s in B.index and (B.loc[s] == V1i.loc[g, ["reach_area_m2", "n_area_obs", "cv_area"]]).all())
res = dict(A=resA, B=resB, rows_v35=len(T), rows_v1=len(V1), v35_nodes_with_reach=len(X35), retained_ids=len(ret), new_ids_with_reach=len(new),
           new_id_sources=newsrc, new_ids_taking_gone_row=len(gn), of_which_rule_B_identical=agree,
           retained_rows_identical_to_v1=bool((T.set_index("subbasin").loc[[s for s in retv1]] == V1i.loc[retv1, ["reach_area_m2", "n_area_obs", "cv_area"]]).all().all()))
json.dump(res, open(W/"river/reach_area_v35_summary.json", "w"), indent=1); print(json.dumps(res, indent=1))
