"""Write the final static-attribute table of graph v3.5 and its check report.

This is the last step of the static attributes: it adds the MERIT upstream-area column, runs the consistency
checks, and writes the table.

The final table is the output of apply_g4_overrides.py plus one column, merit_uparea_km2 (km2, float64), taken from
merit_override_v35.py. Checks (a) to (g) compare the table with the graph, with the attribute tables of earlier graph
versions and with its own inputs, and write the results to CHECKS.md.

Reads (W = $SWOTGAT_WORK, E = W/CONUS_storage_fork/eval, S = E/v35_inputs/statics)
  S/work/static_conus_storage_fixgdw_v35r2frz_g4.parquet   the table from apply_g4_overrides.py
  S/work/merit_uparea_v35_nodes_r2.csv                     MERIT upstream area per node, after the overrides
  E/graph_v3/v35/subbasins_conus_v35.parquet, graphs_conus_v35.json   graph v3.5
  E/v35_inputs/forcing/work/river/reach_area_conus_v35.csv            reach areas on graph v3.5 (stage 04b)
  E/g4_fix/g4_overrides.csv                                the hand-checked dam assignments
  E/g4_fix/testsets/*_g4/sites.csv, W/CONUS_storage_fork/testsets/regime*/sites.csv   the test reservoirs
  W/global_gauges_data/usgs/site_metadata/data.parquet     USGS drainage areas
  raw MERIT-Hydro upa tiles under $SWOTGAT_NAS_ICE/data/MERIT-Hydro/v1.0.1 and the BasinATLAS column list
  (properties.txt under $SWOTGAT_NAS_WATER)
  for comparison, files of earlier graph versions and earlier builds (this repository does not rebuild them):
    W/CONUS/metadata/subbasins_conus_v22.parquet, subbasins_conus_v23.parquet    sub-basin tables of graphs v2.2, v2.3
    W/CONUS/attributes/static_conus_storage_fixgdw_v22_g4.parquet                attribute table of graph v2.2
    W/CONUS/attributes/static_conus_storage_fixgdw_v23.parquet                   attribute table of graph v2.3
    S/work/ref_fixgdw_v23_g4.parquet            the v2.3 table with the same overrides applied (apply_g4_overrides.py)
    S/work/static_conus_storage_fixgdw_v35r2_g4.parquet   the same build chain without freeze_dam_v35.py
    S/static_conus_storage_fixgdw_v35_g4.parquet          the first build of this table ("r1"; this script writes "r2")
    W/CONUS/metadata/reach_area_conus.csv                 the first reach-area table

Writes (in S; the script refuses to overwrite the table, CHECKS.md or checks_v35.json)
  static_conus_storage_fixgdw_v35_g4_r2.parquet   the final attribute table
  CHECKS.md, checks_v35.json                      the check report, as text and as JSON (with md5 of inputs and output)
  merit_uparea_v35_nodes_r2.csv                   a copy of the per-node MERIT table
  checks_c_differences.csv, checks_d_exceptions.csv (only if there are any), checks_f_test_reservoirs_gt1pct.csv,
  checks_merit_vs_graph.csv                       details of checks (c), (d), (f) and of the new column

Checks
  (a) One row per node of graph v3.5: no duplicate id, no graph node without a row, the rows are exactly those of
      the sub-basin table, and the basin index level equals outlet_id. Rows of basins that are not in the graph are
      kept and counted (116 rows in the paper's run).
  (b) Same columns, in the same order and with the same types, as the graph v2.2 table, plus the new column.
  (c) For nodes whose polygon and uparea_km2 are identical to graph v2.3, every column equals the v2.3 reference
      table. Accepted differences: area_km2 (rounding of a recomputed area), and reach_area_km2 / water_area_km2
      where the value is the one from the graph v3.5 reach-area table.
  (d) No missing value in a column where the earlier tables have none.
  (e) Every node of the override table is in the graph, carries the 11 overridden dam cells, and has
      dam_cap_per_uparea = dam_cap_mcm / uparea_km2 of graph v3.5.
  (f) Report only: for six named test dams (TEST_DAMS, by DAM_ID) and for all test reservoir nodes, the columns that
      changed by more than 1% against the graph v2.2 table, grouped by cause.
  New column: coverage by node type, agreement with the graph's uparea_km2 (within 5% and 20%), the 20 largest
      disagreements, agreement with USGS drainage areas at gauges, and a re-read of upa from the raw MERIT tiles at
      400 random outlet cells and every fallback cell.
  (g) reach_area_km2 equals reach_area_m2 / 1e6 and is positive at every node with a row in the reach-area table, is
      0 elsewhere, and water_area_km2 is the lake area at lakes and the reach area elsewhere.
  Finally the differences between the first build (r1) and this one are counted per column.

Usage
  python assemble_checks_v35_r2.py

Notes
  The checks are reported, not enforced: the table is written even if a check prints FAIL. Read CHECKS.md.
  The report text is written by this script as it was for the paper's run; its "Decisions" section and some labels
  refer to the project's internal review.
"""
from swotgat.paths import NAS_ICE, NAS_WATER, WORK
import hashlib
import json
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

R = Path(f"{WORK}"); C = R/"CONUS"; F = R/"CONUS_storage_fork"; E = F/"eval"
S = E/"v35_inputs/statics"; W = S/"work"
SUBS35 = E/"graph_v3/v35/subbasins_conus_v35.parquet"; GRAPH35 = E/"graph_v3/v35/graphs_conus_v35.json"
SUBS23 = C/"metadata/subbasins_conus_v23.parquet"; SUBS22 = C/"metadata/subbasins_conus_v22.parquet"
V22G4 = C/"attributes/static_conus_storage_fixgdw_v22_g4.parquet"; V23 = C/"attributes/static_conus_storage_fixgdw_v23.parquet"
REF = W/"ref_fixgdw_v23_g4.parquet"
CHAIN = W/"static_conus_storage_fixgdw_v35r2frz_g4.parquet"
PURE = W/"static_conus_storage_fixgdw_v35r2_g4.parquet"   # the build chain without freeze_dam_v35.py (imputation refitted)
MERIT = W/"merit_uparea_v35_nodes_r2.csv"
OUT = S/"static_conus_storage_fixgdw_v35_g4_r2.parquet"
R1 = S/"static_conus_storage_fixgdw_v35_g4.parquet"
REACH35 = E/"v35_inputs/forcing/work/river/reach_area_conus_v35.csv"; REACH1 = C/"metadata/reach_area_conus.csv"
NEWCOL = "merit_uparea_km2"
G4 = E/"g4_fix/g4_overrides.csv"
TEST_DAMS = [508, 931, 929, 1752, 1595, 420]
md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()
assert not OUT.exists(), f"{OUT} exists; refusing to overwrite"
for p in (S/"CHECKS.md", S/"checks_v35.json"):
    assert not p.exists(), f"{p} exists; refusing to overwrite"


def same(a, b):
    """Return the element-wise equality of a and b, with two missing values counted as equal."""
    a, b = np.asarray(a), np.asarray(b)
    return (a == b) | (pd.isna(a) & pd.isna(b))


L = []                                                     # CHECKS.md lines
J = {"inputs": {str(p): md5(p) for p in (SUBS35, GRAPH35, SUBS23, V22G4, V23, REF, CHAIN, MERIT, REACH35, R1)}}
P = lambda *a: (print(*a, flush=True), L.append(" ".join(str(x) for x in a)))

# ---------------- assemble
st = pd.read_parquet(CHAIN)
M = pd.read_csv(MERIT, dtype={"node": str}, float_precision="round_trip").set_index("node")
sub = st.index.get_level_values("subbasin").astype(str)
assert NEWCOL not in st.columns
out = st.copy()
out[NEWCOL] = M[NEWCOL].reindex(sub).astype("float64").values

s35 = gpd.read_parquet(SUBS35); s35.index = s35.index.astype(str)
s23 = gpd.read_parquet(SUBS23); s23.index = s23.index.astype(str)
s22 = pd.read_parquet(SUBS22, columns=["uparea_km2"]); s22.index = s22.index.astype(str)
g = json.load(open(GRAPH35)); gnodes = [str(n["id"]) for n in g["nodes"]]
gset = set(gnodes)
v22 = pd.read_parquet(V22G4); v23 = pd.read_parquet(V23); ref = pd.read_parquet(REF)
for t in (v22, v23, ref): t.index = pd.MultiIndex.from_arrays([t.index.get_level_values(0).astype(str), t.index.get_level_values(1).astype(str)], names=t.index.names)
v22s = v22.droplevel("basin"); v23s = v23.droplevel("basin"); refs = ref.droplevel("basin"); outs = out.droplevel("basin")

P("# Graph v3.5 statics: hard gates (r2)")
P("")
P(f"Output r2: `{OUT}`  md5 `@@MD5@@`  shape {out.shape[0]:,} x {out.shape[1]}")
P(f"Output r1 (kept for comparison): `{R1}`  md5 `{md5(R1)}`; its gate report: CHECKS_r1.md")
P("")
P("## Decisions (coordinator, 2026-10-22)")
P("1. freeze_dam_v35 ACCEPTED: dam/use values per lake come from fixgdw_v23 where the dam match is unchanged (all 3,466 lakes, asserted), "
  "dam_cap_per_uparea recomputed with v3.5 uparea. Reason: keeps C1 vs A180 differences down to graph and upstream area (the hub's design); "
  "a plain re-run moves ~1,290 lakes' dam statics only through the GDW builder's polygon-dependent imputation constants and tie-breaking.")
P("2. merit_uparea_km2: 8 nodes overridden to the graph's v3.5 uparea_km2 (source `graph uparea (override)`, reason in merit_uparea_v35_nodes_r2.csv): "
  "the 4 gauges whose main stem v3.5 routes through the gauge node while MERIT's snapped cell is a side channel, and every non-reservoir, "
  "non-gauge node with merit/graph > 2 or < 0.5 (exactly the 4 named river nodes; no others). Reservoirs stay at their pour point by definition.")
P("3. reach_area_km2 / water_area_km2: chain rerun from build_storage_static with the forcing track's v3.5 reach-area table "
  f"`{REACH35}` (md5 {md5(REACH35)}); new gate (g).")
P("4. The 116 non-graph rows are kept, as v22_g4 does.")
P("")
P(f"Built by the production chain (copies in statics/src, paths repointed only) on subbasins_conus_v35.parquet (md5 {md5(SUBS35)}): "
  "build_conus_static -> build_conus_gdw_static (--impute area) -> build_storage_static (reach table = v3.5, V35S_REACH) -> fix_gdw_static -> freeze_dam_v35 -> apply_g4_overrides, + `merit_uparea_km2` (r2).")
P("freeze_dam_v35 (the one addition): reservoir dam/use cells taken from fixgdw_v23 by lake id, dam_cap_per_uparea recomputed with v3.5 uparea. "
  "Reason: the GDW builder matches dams by SUBBASIN polygon and fits its imputation on that set, and fix_gdw_static keeps the builder's row where it "
  "agrees; 1,077 lake polygons change in v3.5, so a plain re-run moves the dam statics of ~1,290 lakes whose dam decision is unchanged (imputed lakes: "
  "refitted constants, cap/depth -1.8% median, dor median fill 95.30 -> 95.65, or the other imputation path, up to ~5%; 2 matched lakes: another GDW "
  "record of equal capacity, depth/dor differ). freeze_dam_v35 asserts these are the only differences (dam_has_gdw, use_*, matched cap/hgt identical "
  "for all 3,466 lakes; work/freeze_dam_report_v35r2.csv). The plain re-run is kept as work/static_conus_storage_fixgdw_v35r2_g4.parquet "
  f"(md5 {md5(PURE)}).")
P("Control: the same copies on subbasins_conus_v23.parquet reproduce CONUS static_conus_storage_v23 and static_conus_storage_fixgdw_v23 "
  "exactly (all 23,458 rows x 213 columns, dtypes, index order), so a fresh build on a graph's own uparea = the production chain + the "
  "v2.x uparea patches (statics_v2x.py only re-select BasinATLAS / recompute dam_cap_per_uparea on nodes whose uparea changed since "
  "the base table; the v3.5 build starts from v3.5 uparea, so they have nothing to do).")
P("")

# ---------------- (a) one row per graph node
P("## (a) one row per v3.5 graph node, no missing, no duplicate")
dup = int(pd.Index(sub).duplicated().sum())
missing = sorted(gset - set(sub)); extra = sorted(set(sub) - gset)
basin_ok = (out.index.get_level_values("basin").astype(str) == s35.outlet_id.reindex(sub).astype(str).values).all()
gb = {str(n["id"]): n.get("basin", n.get("outlet_id")) for n in g["nodes"]}
rows_eq_subs = set(sub) == set(s35.index) and len(sub) == len(s35)
ok_a = dup == 0 and not missing and rows_eq_subs and basin_ok
P(f"- graph nodes {len(gset):,} (JSON node list {len(gnodes):,}); table rows {len(out):,}; duplicate subbasin ids {dup}; graph nodes without a row {len(missing)}")
P(f"- rows = subbasins_conus_v35 rows exactly: {rows_eq_subs}; basin index level = outlet_id for every row: {basin_ok}")
xb = s35.loc[extra, "outlet_id"].value_counts().to_dict() if extra else {}
P(f"- {len(extra)} rows are not graph nodes: {xb}. Same convention as v22_g4 (its 23,458 rows include the 110 rows of the 2 basins "
  "outside graph v2.2); the loader reads statics with .sel per graph basin and normalises on train nodes only, so these rows are inert. "
  "Drop them if a strict one-row-per-graph-node table is wanted.")
P(f"- **(a) {'PASS' if ok_a else 'FAIL'}**")
J["a"] = dict(graph_nodes=len(gset), rows=len(out), dup=dup, missing=len(missing), extra_rows=len(extra), extra_by_basin=xb, pass_=bool(ok_a))
P("")

# ---------------- (b) column set
P("## (b) same columns as v22_g4 + the new column")
cols_ok = list(out.columns) == list(v22.columns) + [NEWCOL]
dt_ok = all(out[c].dtype == v22[c].dtype for c in v22.columns)
idx_ok = list(out.index.names) == list(v22.index.names) and all(out.index.get_level_values(i).dtype == v22.index.get_level_values(i).dtype for i in range(2))
order_ok = out.index.get_level_values("basin").is_monotonic_increasing
P(f"- columns = v22_g4 columns ({v22.shape[1]}) in the same order + `{NEWCOL}` (float64): {cols_ok}")
P(f"- dtypes of the {v22.shape[1]} shared columns identical: {dt_ok}; index (basin, subbasin), object/str: {idx_ok}; rows sorted by basin then subbasin as the builder writes them: {order_ok}")
bad_dt = {c: (str(out[c].dtype), str(v22[c].dtype)) for c in v22.columns if out[c].dtype != v22[c].dtype}
if bad_dt: P(f"- dtype differences: {bad_dt}")
ok_b = cols_ok and dt_ok and idx_ok
P(f"- **(b) {'PASS' if ok_b else 'FAIL'}**")
J["b"] = dict(cols_ok=cols_ok, dtypes_ok=dt_ok, index_ok=idx_ok, pass_=bool(ok_b))
P("")

# ---------------- (c) identical polygon AND uparea -> identical row
P("## (c) nodes whose polygon AND uparea equal v2.3: every column equals fixgdw_v23 (+G4)")
shared = s35.index.intersection(s23.index)
wkb35 = shapely.to_wkb(s35.geometry.reindex(shared).values); wkb23 = shapely.to_wkb(s23.geometry.reindex(shared).values)
poly_eq = pd.Series(wkb35 == wkb23, index=shared)
exact_eq = pd.Series(shapely.equals_exact(s35.geometry.reindex(shared).values, s23.geometry.reindex(shared).values, tolerance=0), index=shared)
up_eq = pd.Series(s35.uparea_km2.reindex(shared).values == s23.uparea_km2.reindex(shared).values, index=shared)
ident = shared[poly_eq & up_eq]
P(f"- shared ids {len(shared):,}; polygon identical (WKB) {int(poly_eq.sum()):,} (equals_exact {int(exact_eq.sum()):,}); uparea identical {int(up_eq.sum()):,}; both: {len(ident):,}")
a = outs.loc[ident, v22.columns]; b = refs.reindex(ident)[v22.columns]
assert b.notna().any(axis=1).all(), "reference rows missing"
basin_same = (out.loc[(slice(None), ident), :].index.get_level_values("basin") if False else s35.outlet_id.reindex(ident).values) == ref.reset_index().set_index("subbasin").basin.reindex(ident).values
D = {c: ~same(a[c].values, b[c].values) for c in v22.columns}
dcols = {c: int(v.sum()) for c, v in D.items() if v.any()}
drows = sorted(set(np.asarray(ident)[np.any(np.column_stack(list(D.values())), axis=1)]))
P(f"- basin identical for all of them: {bool(np.all(basin_same))}")
P(f"- columns that differ: {dcols if dcols else 'none'}; rows that differ: {len(drows)}")
LC = []
for c, n in dcols.items():
    for node in np.asarray(ident)[D[c]]:
        LC.append(dict(node=node, column=c, ref_v23g4=refs.at[node, c], v35=outs.at[node, c]))
LC = pd.DataFrame(LC, columns=["node", "column", "ref_v23g4", "v35"]); LC.to_csv(S/"checks_c_differences.csv", index=False)
if len(LC):
    P("- details: checks_c_differences.csv; first rows:")
    for r in LC.head(20).itertuples(): P(f"  - {r.node} {r.column}: v23+G4 {r.ref_v23g4} -> v35 {r.v35}")
J["c"] = dict(shared=len(shared), poly_identical=int(poly_eq.sum()), uparea_identical=int(up_eq.sum()), both=len(ident), diff_cols=dcols, diff_rows=len(drows))
# context: nodes with changed polygon or uparea (not part of the gate)
chg = shared.difference(ident)
P(f"- (context) {len(chg):,} shared nodes have a changed polygon or uparea and {len(s35.index.difference(s23.index)):,} ids are new in v3.5 "
  f"({len(s23.index.difference(s35.index)):,} v2.3 ids are gone); their rows are rebuilt from their own polygon/uparea by the same code.")
pu = pd.read_parquet(PURE).droplevel("basin").loc[ident, v22.columns]
pdc = {c: int((~same(pu[c].values, b[c].values)).sum()) for c in v22.columns}; pdc = {c: n for c, n in pdc.items() if n}
P(f"- (for comparison) the plain re-run without freeze_dam_v35 differs on this node set in: {pdc}")
if "area_km2" in dcols:
    nn = np.asarray(ident)[D["area_km2"]]
    rel = np.abs(outs.loc[nn, "area_km2"].values / refs.loc[nn, "area_km2"].values - 1)
    s23a = np.abs(s35.area_km2[nn].values / s23.area_km2[nn].values - 1)
    P(f"- area_km2: the {len(nn)} differences are the subbasins' own area_km2 (statics copy it): |rel| max {rel.max():.2e}; "
      f"subbasins_conus_v35 vs v23 area_km2 at these nodes |rel| max {s23a.max():.2e} (float rounding of a re-computed area, polygon identical)")
rt35 = pd.read_csv(REACH35, dtype={"subbasin": str}).set_index("subbasin").reach_area_m2 / 1e6      # parsed as build_storage_static parses it
rt1 = pd.read_csv(REACH1, dtype={"subbasin": str}).set_index("subbasin").reach_area_m2 / 1e6
explained = True
for c in ("reach_area_km2", "water_area_km2"):
    if c not in dcols: continue
    nn = np.asarray(ident)[D[c]]
    if c == "reach_area_km2":
        ok = np.array([outs.at[n, c] == rt35.get(n, 0.0) and refs.at[n, c] == rt1.get(n, 0.0) for n in nn])
    else:
        ok = np.array([outs.at[n, c] == (outs.at[n, "lake_area_km2"] if outs.at[n, "lake_area_km2"] > 0 else rt35.get(n, 0.0)) for n in nn])
    explained &= bool(ok.all())
    rr = outs.loc[nn, c].values / np.where(refs.loc[nn, c].values > 0, refs.loc[nn, c].values, np.nan)
    P(f"- {c}: {len(nn)} differences, {int(ok.sum())} of them = the v3.5 reach-area table value where the v1 table had another "
      f"(new/old ratio median {np.nanmedian(rr):.3f}, range [{np.nanmin(rr):.3f}, {np.nanmax(rr):.3f}]; {int(np.isnan(rr).sum())} were 0 in v2.3); "
      "reason: decision 3, the forcing track's v3.5 reach areas replace the v1 ones, so these cells change by design")
J["c"]["plain_rerun_diff_cols"] = pdc
J["c"]["pass_"] = set(dcols) <= {"area_km2", "reach_area_km2", "water_area_km2"} and explained
_why = [w for c, w in (("area_km2", "area_km2 float rounding inherited from the v3.5 subbasins"),
                        ("reach_area_km2", "reach_area_km2 from the v3.5 reach-area table (decision 3)"),
                        ("water_area_km2", "water_area_km2 following it")) if c in dcols]
P(f"- **(c) {'PASS' if not dcols else (('PASS except: ' + '; '.join(_why) + ' (above)') if J['c']['pass_'] else 'FAIL (see above)')}**")
P("")

# ---------------- (d) NaN
P("## (d) no NaN where v2.3 / v22_g4 have none")
exc = []
for c in v22.columns:
    n_new = int(out[c].isna().sum())
    if n_new == 0: continue
    ref_nan_any = int(v22[c].isna().sum()) + int(v23[c].isna().sum())
    shared_bad = [n for n in outs.index[outs[c].isna()] if n in v23s.index and pd.notna(v23s.at[n, c])] + \
                 [n for n in outs.index[outs[c].isna()] if n in v22s.index and pd.notna(v22s.at[n, c]) and n not in v23s.index]
    if ref_nan_any == 0 or shared_bad:
        nodes = list(outs.index[outs[c].isna()])
        exc.append(dict(column=c, n_nan_v35=n_new, n_nan_v22g4=int(v22[c].isna().sum()), n_nan_v23=int(v23[c].isna().sum()),
                        nan_where_v23_has_value=len(shared_bad), nodes=";".join(nodes[:50])))
nan_cols = {c: int(out[c].isna().sum()) for c in v22.columns if out[c].isna().any()}
P(f"- columns with any NaN in v3.5 (shared columns): {nan_cols if nan_cols else 'none'}")
P(f"- NaN counts in the references: v22_g4 {int(v22.isna().sum().sum())} cells, v23 {int(v23.isna().sum().sum())} cells")
if exc:
    pd.DataFrame(exc).to_csv(S/"checks_d_exceptions.csv", index=False)
    for e in exc: P(f"  - exception {e['column']}: {e['n_nan_v35']} NaN (v22_g4 {e['n_nan_v22g4']}, v23 {e['n_nan_v23']}); nodes {e['nodes'][:300]}")
nm = int(out[NEWCOL].isna().sum()); nmg = int(outs[NEWCOL].reindex(list(gset)).isna().sum())
P(f"- new column `{NEWCOL}`: NaN in {nm} rows ({nmg} graph nodes)")
P(f"- **(d) {'PASS' if not exc else 'EXCEPTIONS listed'}**")
J["d"] = dict(nan_cols=nan_cols, exceptions=exc, newcol_nan=nm, newcol_nan_graph=nmg, pass_=not exc)
P("")

# ---------------- (e) G4
P("## (e) G4 override nodes present and overridden")
O = pd.read_csv(G4, dtype={"node": str})
COLS = ["dam_has_gdw", "dam_cap_mcm", "dam_hgt_m", "dam_depth_m", "dam_dor_pc", "use_elec", "use_irri", "use_fcon", "use_supp", "use_recr", "use_navi"]
ok_e = True
for o in O.itertuples():
    pres = o.node in outs.index and o.node in gset
    vals_ok = pres and all(outs.at[o.node, c] == np.asarray(getattr(o, c)).astype(out[c].dtype) for c in COLS)
    cpu = (np.float32(outs.at[o.node, "dam_cap_mcm"]) / outs.at[o.node, "uparea_km2"]).astype("float32") if pres else np.nan
    cpu_ok = pres and outs.at[o.node, "dam_cap_per_uparea"] == cpu
    up_eq22 = pres and o.node in s22.index and s35.uparea_km2[o.node] == s22.uparea_km2[o.node]
    eq22 = pres and o.node in v22s.index and all(same([outs.at[o.node, c]], [v22s.at[o.node, c]])[0] for c in COLS)
    ok_e &= bool(pres and vals_ok and cpu_ok)
    P(f"- {o.node} ({o.decision}, {o.label}): in graph {pres}; 11 dam/use cells = override {vals_ok}; dam_cap_per_uparea = cap/v3.5 uparea {cpu_ok} "
      f"({outs.at[o.node, 'dam_cap_per_uparea']:.6g}); dam/use cells = v22_g4 {eq22}; uparea v3.5 {s35.uparea_km2[o.node]:,.2f} vs v2.2 {s22.uparea_km2.get(o.node, np.nan):,.2f}")
P(f"- **(e) {'PASS' if ok_e else 'FAIL'}**")
J["e"] = dict(pass_=bool(ok_e))
P("")

# ---------------- (f) test reservoirs
P("## (f) test reservoirs")
sites = []
for d in sorted((E/"g4_fix/testsets").glob("*_g4")) + sorted(p for p in (F/"testsets").glob("regime*") if p.is_dir()):
    f = d/"sites.csv"
    if f.exists():
        t = pd.read_csv(f, dtype={"node": str, "members": str}); t["set"] = d.name; sites.append(t[["DAM_ID", "DAM_NAME", "node", "members", "basin", "set"]])
sites = pd.concat(sites, ignore_index=True)
num = [c for c in v22.columns if np.issubdtype(v22[c].dtype, np.number)]


def rel_changes(n):
    """Return the relative change of every numeric column at node n, from the graph v2.2 table to this table.

    The result is a Series indexed by column: |new - old| / |old|, 0 where the values are equal or both missing,
    and infinite where only one side is missing or the old value is 0. Return None if the node is not in both
    tables.
    """
    if n not in v22s.index or n not in outs.index: return None
    o, nw = v22s.loc[n, num].astype(float), outs.loc[n, num].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(o == nw, 0.0, np.where(o != 0, np.abs(nw - o) / np.abs(o), np.inf))
    r = pd.Series(r, index=num); r[o.isna() & nw.isna()] = 0.0; r[o.isna() ^ nw.isna()] = np.inf
    return r


P("Test-site numbering = DAM_ID in the test sets (g4_fix/testsets/*_g4 and testsets/regime*).")
six = []
for dam in TEST_DAMS:
    t = sites[sites.DAM_ID == dam]
    if not len(t): P(f"- DAM_ID {dam}: not found in the test sets"); continue
    nodes = sorted({x for m in t.members.dropna() for x in str(m).split(";")} | set(t.node.dropna()))
    for n in nodes:
        r = rel_changes(n); big = r[r > 0.01].sort_values(ascending=False) if r is not None else pd.Series(dtype=float)
        pe = bool(n in s23.index and n in s35.index and shapely.to_wkb(s35.geometry[n]) == shapely.to_wkb(s23.geometry[n]))
        line = (f"- DAM_ID {dam} {t.DAM_NAME.iloc[0]} -> node {n} ({t.basin.iloc[0]}; sets {sorted(t.set.unique())}): polygon = v2.3 {pe}; "
                f"uparea v2.2 {s22.uparea_km2.get(n, np.nan):,.1f} / v2.3 {s23.uparea_km2.get(n, np.nan):,.1f} / v3.5 {s35.uparea_km2.get(n, np.nan):,.1f}; "
                f"{NEWCOL} {outs.at[n, NEWCOL]:,.1f}; columns changed >1% vs v22_g4: "
                + (", ".join(f"{c} {v22s.at[n, c]:.4g}->{outs.at[n, c]:.4g}" for c in big.index[:12]) + (f" (+{len(big) - 12} more)" if len(big) > 12 else "") if len(big) else "none"))
        P(line); six.append(dict(dam=dam, node=n, n_cols_gt1pct=len(big), cols=";".join(big.index)))
tn = sorted({x for m in sites.members.dropna() for x in str(m).split(";")} | set(sites.node.dropna()))
T = []
for n in tn:
    r = rel_changes(n)
    if r is None: T.append(dict(node=n, status="not in v22_g4 or v3.5")); continue
    big = r[r > 0.01]
    if len(big):
        dams = sorted(set(sites.DAM_ID[(sites.node == n) | sites.members.fillna("").str.split(";").apply(lambda m: n in m)]))
        T.append(dict(node=n, dam_ids=";".join(map(str, dams)), n_cols=len(big), cols=";".join(big.sort_values(ascending=False).index),
                      uparea_v22=s22.uparea_km2.get(n, np.nan), uparea_v35=s35.uparea_km2.get(n, np.nan)))
T = pd.DataFrame(T); T.to_csv(S/"checks_f_test_reservoirs_gt1pct.csv", index=False)
n_T_all = len(T)
P(f"- all test reservoir nodes (sites + members, {len(tn)} nodes over {sites.DAM_ID.nunique()} DAM_IDs): {len(T)} change by > 1% in at least one column vs v22_g4 "
  "(checks_f_test_reservoirs_gt1pct.csv). By what changed:")
if len(T):
    props = set(pd.read_csv(f"{NAS_WATER}/cjgleason/ted/swot-ml/data/owen_reservoirs/Columbia/attributes/properties.txt", header=None)[0])
    cs = T.cols.fillna("").str.split(";").apply(set)
    only_area = cs.apply(lambda c: c <= {"area_km2"})
    up = cs.apply(lambda c: "uparea_km2" in c)
    hyb = cs.apply(lambda c: bool(c & props))
    dam = cs.apply(lambda c: bool(c & {"dam_cap_mcm", "dam_hgt_m", "dam_depth_m", "dam_dor_pc", "dam_has_gdw", "dam_cap_per_uparea"}))
    P(f"  - only area_km2 (the lake node's catchment polygon changed by > 1%, uparea within 1%): {int(only_area.sum())}")
    P(f"  - uparea_km2 > 1% (graph routing change): {int(up.sum())}; of these with dam_cap_per_uparea: {int((up & dam).sum())}")
    P(f"  - BasinATLAS attributes re-selected (new centroid or uparea): {int(hyb.sum())}")
    P(f"  - dam columns other than via uparea: {int((dam & ~up).sum())}")
    P("  - nodes with changes beyond area_km2:")
    T = T[~only_area.values]
for r in T.itertuples():
    P(f"    - node {r.node} (DAM_ID {getattr(r, 'dam_ids', '')}): {getattr(r, 'n_cols', '')} columns [{str(getattr(r, 'cols', ''))[:200]}]; uparea v2.2 {getattr(r, 'uparea_v22', np.nan):,.1f} -> v3.5 {getattr(r, 'uparea_v35', np.nan):,.1f}")
J["f"] = dict(six=six, n_test_nodes=len(tn), n_changed_gt1pct=n_T_all, n_changed_beyond_area=len(T))
P("")

# ---------------- the new column
P(f"## New column `{NEWCOL}` (km2, not log-transformed)")
Mg = M.reindex(list(gset))
P("Definition (hub BOARD.md, Owen 2026-10-20): MERIT-Hydro upstream area at the node's routed outlet, graph-independent:")
P("- reservoir nodes: MERIT upa at the lake pour point = the largest-flow-accumulation (upg) cell inside the SWOT PLD v106 outline (lake_id = node id), "
  "the delineator's find_reservoir_outlet (verbatim copy, delineator rasters CONUS/merit_rasters/upg).")
P("- gauge nodes: MERIT upa at the gauge's snapped pour point of the v3.4 delineation (v3.5 moves no gauge). Not recorded in any output "
  "(delineation_v34/*/nodes lat/lng = the gauge-table input point, not the snap), so it is replayed with the delineator's code: gauges_v34 tables, "
  "set_gauge_area_range (incl. snap_min_area_km2), insert_all_gauges order/bookkeeping, split_catchment_raster's snap_to_mask cell.")
P("- other river nodes: MERIT-Basins `uparea` of original_comid (riv_pfaf_7 bugfix1) when the reach's downstream end lies at the node polygon (<= 1.5 cells); "
  "otherwise MERIT upa at the max-upg cell inside the node polygon (fallback, flagged in merit_uparea_v35_nodes_r2.csv).")
P("- r2 overrides (decision 2): graph uparea_km2 at the 8 nodes below; merit_raw_km2 keeps the MERIT value.")
P("- upa is read from the MERIT upa tiles via work/merit_upa_conus_tilegeoref.vrt: eval/graph_review/_merit_upa_conus.vrt is half a cell NE of the tiles' "
  "own georeferencing (which the delineator rasters and MERIT-Basins use), so sampling it at delineator coordinates lands on cell corners "
  "(test: a gauge pour point with upa 574.6 km2 reads 0.07 km2 there). Every pour point re-reads the delineator's upg value exactly and has upa/upg ~ cell area.")
P("")
ovr = M[M.source == "graph uparea (override)"]
P("| node | type | basin | MERIT value km2 | graph uparea_km2 (used) | reason |")
P("|---|---|---|---|---|---|")
for n, r in ovr.iterrows():
    P(f"| {n} | {r.node_type} | {r.outlet_id} | {r.merit_raw_km2:,.2f} | {r.uparea_km2:,.2f} | {r.override_reason} |")
J["merit_overrides"] = ovr[["merit_raw_km2", "uparea_km2", "override_reason"]].reset_index().to_dict("records")
P("")
P("Coverage (graph nodes), after the overrides:")
P("")
P("| node type | nodes | value | source counts | within 5% of graph uparea_km2 | within 20% |")
P("|---|---|---|---|---|---|")
Mg = Mg.join(s35[["node_type"]], rsuffix="_s")
Mg["ratio"] = Mg[NEWCOL] / Mg.uparea_km2; Mg["abs_log"] = np.abs(np.log(Mg.ratio.where(Mg.ratio > 0)))
cov = {}
for nt, grp in Mg.groupby("node_type_s"):
    v = grp[NEWCOL].notna() & (grp[NEWCOL] > 0)
    w5 = int((grp.abs_log <= np.log(1.05)).sum()); w20 = int((grp.abs_log <= np.log(1.20)).sum())
    cov[nt] = dict(nodes=len(grp), value=int(v.sum()), within5=w5, within20=w20, sources=grp.source.value_counts().to_dict())
    P(f"| {nt} | {len(grp):,} | {int(v.sum()):,} | {grp.source.value_counts().to_dict()} | {w5:,} ({w5 / len(grp):.1%}) | {w20:,} ({w20 / len(grp):.1%}) |")
w5 = int((Mg.abs_log <= np.log(1.05)).sum())
P(f"| all | {len(Mg):,} | {int(Mg[NEWCOL].notna().sum()):,} | | {w5:,} ({w5 / len(Mg):.1%}) | {int((Mg.abs_log <= np.log(1.2)).sum()):,} |")
P("")
P("20 largest disagreements with the graph's uparea_km2 after the overrides (|log ratio|):")
P("")
P("| node | type | basin | merit_uparea_km2 | graph uparea_km2 | ratio | source |")
P("|---|---|---|---|---|---|---|")
for n, r in Mg.sort_values("abs_log", ascending=False).head(20).iterrows():
    P(f"| {n} | {r.node_type_s} | {r.outlet_id} | {r[NEWCOL]:,.2f} | {r.uparea_km2:,.2f} | {r.ratio:.3g} | {r.source} |")
Mg.sort_values("abs_log", ascending=False).to_csv(S/"checks_merit_vs_graph.csv")
usgs = pd.read_parquet(R/"global_gauges_data/usgs/site_metadata/data.parquet", columns=["site_id", "area"]).drop_duplicates("site_id").set_index("site_id").area
gg = Mg[(Mg.node_type_s == "gauge") & Mg.index.isin(usgs.index)].copy(); gg["usgs"] = usgs.reindex(gg.index).values; gg = gg[gg.usgs > 0]
wm = int((np.abs(np.log(gg[NEWCOL] / gg.usgs)) <= np.log(1.05)).sum()); wg = int((np.abs(np.log(gg.uparea_km2 / gg.usgs)) <= np.log(1.05)).sum())
P("")
P(f"Gauges vs USGS published drainage area ({len(gg):,} gauge nodes with one): within 5% {wm:,} for {NEWCOL}, {wg:,} for graph uparea_km2. "
  "Where the delineator snapped a gauge onto a side channel (e.g. USGS-08062700) the column inherits that snap, by definition.")
J.setdefault("merit", {})["usgs_within5"] = dict(n=len(gg), merit=wm, graph=wg)
chk = M[M.px.notna()]
P("")
P(f"Pour-point cell checks: {len(chk):,} raster pour points; upg read back == delineator facc value at {int((chk.upg_check == chk.facc).sum()):,}; "
  f"upa/upg/cell-area median {chk.upa_per_upg_vs_cell.median():.4f}, 0.1-99.9% [{chk.upa_per_upg_vs_cell.quantile(.001):.3f}, {chk.upa_per_upg_vs_cell.quantile(.999):.3f}].")
J.setdefault("merit", {}).update(coverage=cov, within5_all=w5, graph_nodes=len(Mg))
# independent georeferencing check: re-read upa from the raw MERIT tiles (their own transform) at a sample of
# pour points
import math
import rasterio
rng = np.random.default_rng(0)
samp = chk.index[rng.choice(len(chk), size=min(400, len(chk)), replace=False)].tolist() + chk.index[~chk.source.isin(["reservoir:pour_point", "gauge:split", "gauge:replace", "gauge:split_retry", "gauge:replace_retry"])].tolist()
nok = nbad = 0
for n in dict.fromkeys(samp):
    x, y = float(chk.px[n]), float(chk.py[n])
    la0 = math.floor((y + 1 / 2400) / 5) * 5; lo0 = math.floor((x + 1 / 2400) / 5) * 5
    fo = f"upa_{'n' if la0 >= 0 else 's'}{abs(math.floor(la0 / 30) * 30):02d}{'w' if lo0 < 0 else 'e'}{abs(math.floor(lo0 / 30) * 30):03d}"
    tf = Path(f"{NAS_ICE}/data/MERIT-Hydro/v1.0.1/{fo}/{'n' if la0 >= 0 else 's'}{abs(la0):02d}{'w' if lo0 < 0 else 'e'}{abs(lo0):03d}_upa.tif")
    with rasterio.open(tf) as t:
        r, c = t.index(x, y); v = float(t.read(1, window=rasterio.windows.Window(c, r, 1, 1))[0, 0])
    if v == float(chk.upa_cell[n]) and v == float(out[NEWCOL].values[sub.get_loc(n)] if chk.source[n] not in ("river:reach_end", "graph uparea (override)") else v): nok += 1
    else: nbad += 1
P(f"Raw-tile re-read (tiles' own GeoTransform, {nok + nbad} pour points incl. every fallback): identical upa at {nok}, different at {nbad}.")
J["merit"]["raw_tile_check"] = dict(ok=nok, bad=nbad)
P("")

# ---------------- (g) reach area
P("## (g) reach_area_km2 > 0 at every v3.5 node with a row in reach_area_conus_v35.csv")
rt = pd.read_csv(REACH35, dtype={"subbasin": str})
dupr = int(rt.subbasin.duplicated().sum())
inrows = pd.Index(rt.subbasin.unique()).intersection(outs.index); notin = sorted(set(rt.subbasin) - set(outs.index))
bad_pos = [n for n in inrows if not outs.at[n, "reach_area_km2"] > 0]
bad_val = [n for n in inrows if outs.at[n, "reach_area_km2"] != rt35[n]]
nonrow_nonzero = int((outs.loc[outs.index.difference(inrows), "reach_area_km2"] != 0).sum())
lake = outs.lake_area_km2 > 0
w_ok = bool((outs.water_area_km2[lake] == outs.lake_area_km2[lake]).all() and (outs.water_area_km2[~lake] == outs.reach_area_km2[~lake]).all())
src_bad = rt[~(rt.reach_area_m2 > 0)]
sh1 = rt35.index.intersection(rt1.index)
P(f"- table rows {len(rt):,} (duplicate ids {dupr}; rows with reach_area_m2 <= 0 or NaN {len(src_bad)}); ids in the v3.5 table {len(inrows):,}; ids not in v3.5 {len(notin)}")
P(f"- reach_area_km2 > 0 at all {len(inrows):,}: {not bad_pos} (failures {len(bad_pos)}: {bad_pos[:20]}); value = reach_area_m2 / 1e6 exactly: {not bad_val} ({len(bad_val)} mismatches)")
P(f"- nodes without a row have reach_area_km2 = 0: {nonrow_nonzero == 0}; water_area_km2 = lake area at lakes, else reach area: {w_ok}")
P(f"- vs the v1 table: {len(sh1):,} ids in both, identical value at {int((rt35[sh1] == rt1[sh1]).sum()):,}; {len(rt35.index.difference(rt1.index)):,} ids new "
  f"in the v3.5 table; {len(rt1.index.difference(rt35.index)):,} v1 ids absent from it ({int(rt1.index.difference(rt35.index).isin(outs.index).sum())} of those are still v3.5 nodes, now 0)")
nr = outs.reach_area_km2 > 0
P(f"- nodes with reach_area_km2 > 0: r2 {int(nr.sum()):,} (graph nodes {int(nr.reindex(list(gset)).sum()):,}); r1 {int((pd.read_parquet(R1).droplevel('basin').reach_area_km2 > 0).sum()):,}")
ok_g = not bad_pos and not bad_val and nonrow_nonzero == 0 and w_ok and dupr == 0
P(f"- **(g) {'PASS' if ok_g else 'FAIL'}**")
J["g"] = dict(rows=len(rt), ids_in_v35=len(inrows), not_in_v35=len(notin), fail_pos=len(bad_pos), fail_val=len(bad_val), pass_=bool(ok_g))
P("")

# ---------------- r1 vs r2
P("## r1 -> r2 differences")
r1 = pd.read_parquet(R1)
assert r1.index.equals(out.index) and list(r1.columns) == list(out.columns)
dd = {c: int((~same(r1[c].values, out[c].values)).sum()) for c in out.columns}; dd = {c: n for c, n in dd.items() if n}
P(f"- columns that differ (cells): {dd}; rows {int(np.any(np.column_stack([~same(r1[c].values, out[c].values) for c in out.columns]), axis=1).sum()):,}")
J["r1_vs_r2"] = dd
P("")

# ---------------- write (only now, after every check ran)
out.to_parquet(OUT)
back = pd.read_parquet(OUT)
assert back.index.equals(out.index) and list(back.columns) == list(out.columns) and all(same(back[c], out[c]).all() for c in out.columns)
assert all(back[c].dtype == out[c].dtype for c in out.columns)
J["output"] = {"path": str(OUT), "md5": md5(OUT), "shape": list(out.shape)}
shutil.copyfile(MERIT, S/"merit_uparea_v35_nodes_r2.csv")
(S/"CHECKS.md").write_text("\n".join(L).replace("@@MD5@@", J["output"]["md5"]) + "\n")
json.dump(J, open(S/"checks_v35.json", "w"), indent=1, default=str)
print("wrote", OUT, J["output"]["md5"])
