"""Keep the dam attributes of every lake exactly as in the graph v2.3 attribute table.

This stops the attributes from drifting between graph versions. The script runs between fix_gdw_static.py and
apply_g4_overrides.py.

Why
  The dam and use attributes of a lake row should depend only on the lake (its PLD res_id or outline against Global
  Dam Watch, GDW), not on the graph version. In the build chain they do depend on it slightly:
  build_conus_gdw_static.py matches GDW dams by sub-basin polygon and fits its imputation (log10 capacity on log10
  lake area; medians for height, depth and DOR) on that matched set, and fix_gdw_static.py keeps the builder's row
  wherever the two agree. Graph v3.5 changes 1,077 lake sub-basin polygons, so a plain re-run moves the dam
  attributes of about 1,290 lakes whose dam is unchanged:
    1. imputed lakes get refitted constants (capacity and depth -1.8% in the median, DOR median 95.30 -> 95.65) or
       are imputed by the other script's formula (up to about 5%);
    2. two matched lakes get another GDW record with the same CAP_MCM, picked through the new polygon (depth and
       DOR differ, e.g. DOR 9.8 -> 10000).

Reads
  IN.parquet   static table from fix_gdw_static.py (graph v3.5)
  $SWOTGAT_WORK/CONUS/attributes/static_conus_storage_fixgdw_v23.parquet   the attribute table of the earlier graph
               v2.3 (an archived file; this repository does not rebuild it, see docs/KNOWN_LIMITS.md)

Writes (refuses to overwrite either file)
  OUT.parquet  IN with the dam cells of lake rows replaced
  REPORT.csv   one row per changed cell: node, column, chain_v35 (value in IN), frozen_v23 (value written),
               dam_has_gdw, why

Method
  The lake rows of IN and of the v2.3 table must be the same set of lakes. For every lake row, the 11 cells
  dam_has_gdw, dam_cap_mcm, dam_hgt_m, dam_depth_m, dam_dor_pc, use_elec, use_irri, use_fcon, use_supp, use_recr and
  use_navi are copied from the v2.3 table by lake id. Before copying, the script checks that IN differs from v2.3
  only by the drift described above:
    - dam_has_gdw and the use columns are equal for every lake;
    - where dam_has_gdw == 1, capacity and height are equal;
    - where dam_has_gdw == 0, height is equal (the constant median) and capacity, depth and DOR differ by less
      than 6%.
  Any other difference would be a real change of dam match and stops the script.
  dam_cap_per_uparea is then recomputed as in build_conus_gdw_static.py, float32(dam_cap_mcm / uparea_km2), with
  the uparea_km2 of IN (graph v3.5). Rows of other nodes and all other columns are checked to be unchanged.

Usage
  python freeze_dam_v35.py IN.parquet OUT.parquet REPORT.csv
"""
from swotgat.paths import WORK
import sys
from pathlib import Path

import numpy as np
import pandas as pd

C = Path(f"{WORK}/CONUS")
V23 = C/"attributes/static_conus_storage_fixgdw_v23.parquet"
COLS = ["dam_has_gdw", "dam_cap_mcm", "dam_hgt_m", "dam_depth_m", "dam_dor_pc", "use_elec", "use_irri", "use_fcon", "use_supp", "use_recr", "use_navi"]
inp, out, rep = map(Path, sys.argv[1:4])
assert not out.exists() and not rep.exists(), "refusing to overwrite"
st = pd.read_parquet(inp); ref = pd.read_parquet(V23)
sid = pd.Index(st.index.get_level_values("subbasin").astype(str)); rsid = pd.Index(ref.index.get_level_values("subbasin").astype(str))
rref = ref.set_axis(rsid)
isr = (st.is_reservoir == 1).values
res = sid[isr]
assert res.isin(rsid).all() and (rref.loc[res, "is_reservoir"] == 1).all(), "a v3.5 reservoir is not a v2.3 reservoir row"
assert int((rref.is_reservoir == 1).sum()) == len(res), "reservoir sets differ"
A = st.set_axis(sid).loc[res, COLS]; B = rref.loc[res, COLS]
eq = lambda c: (A[c].values == B[c].values) | (pd.isna(A[c].values) & pd.isna(B[c].values))
for c in ["dam_has_gdw", "use_elec", "use_irri", "use_fcon", "use_supp", "use_recr", "use_navi"]:
    assert eq(c).all(), f"{c} differs for {int((~eq(c)).sum())} lakes: a real dam-match change, not imputation drift"
m = (B.dam_has_gdw == 1).values
assert eq("dam_cap_mcm")[m].all() and eq("dam_hgt_m")[m].all(), "a matched lake changed capacity/height"
assert eq("dam_hgt_m")[~m].all(), "imputed hgt differs (expected the constant median)"
for c in ["dam_cap_mcm", "dam_depth_m", "dam_dor_pc"]:
    r = np.abs(A[c].values.astype(float) / B[c].values.astype(float) - 1)
    worst = np.nanmax(np.where(eq(c) | m, 0, r))
    assert worst < 0.06, f"{c}: imputed-lake drift {worst:.3f} above 6%"
# Why a lake differs (build_conus_gdw_static.py matches GDW by sub-basin polygon; fix_gdw_static.py keeps that row
# when it agrees):
#   matched lake, capacity and height equal, depth or DOR differ
#       -> the sub-basin polygon picked another GDW record with the same CAP_MCM (a tie) and fix_gdw_static.py kept it
#          as "same dam as before"
#   imputed lake
#       -> the imputation was refitted on a changed matched set, or the lake was imputed by build_conus_gdw_static.py
#          in one build and by fix_gdw_static.py (its own fit and medians) in the other
L = []
for c in COLS:
    for n in res[~eq(c)]:
        why = ("matched: builder tie, another GDW record with equal capacity" if B.at[n, "dam_has_gdw"] == 1 else "imputed: constants / imputation path")
        L.append(dict(node=n, column=c, chain_v35=A.at[n, c], frozen_v23=B.at[n, c], dam_has_gdw=int(B.at[n, "dam_has_gdw"]), why=why))
L = pd.DataFrame(L)
new = st.copy()
pos = np.flatnonzero(isr)
for c in COLS:
    new.iloc[pos, new.columns.get_loc(c)] = rref.loc[res, c].values.astype(st[c].dtype)
with np.errstate(divide="ignore", invalid="ignore"):
    cpu = (new.dam_cap_mcm.values[pos] / new.uparea_km2.values[pos]).astype("float32")
k = new.columns.get_loc("dam_cap_per_uparea")
old_cpu = st.dam_cap_per_uparea.values[pos]
new.iloc[pos, k] = cpu
ch = ~((cpu == old_cpu) | (np.isnan(cpu) & np.isnan(old_cpu)))
for n, o, v in zip(res[ch], old_cpu[ch], cpu[ch]):
    L = pd.concat([L, pd.DataFrame([dict(node=n, column="dam_cap_per_uparea", chain_v35=o, frozen_v23=v, dam_has_gdw=int(rref.at[n, "dam_has_gdw"]), why="recomputed: cap / v3.5 uparea")])], ignore_index=True)
# nothing else changed
diff = ~((new.values == st.values) | (pd.isna(new).values & pd.isna(st).values))
assert not diff[~isr].any(), "non-reservoir rows changed"
assert set(st.columns[diff.any(axis=0)]) <= set(COLS) | {"dam_cap_per_uparea"}
assert list(new.columns) == list(st.columns) and new.index.equals(st.index) and (new.dtypes == st.dtypes).all()
new.to_parquet(out); L.to_csv(rep, index=False)
print(f"reservoir rows {len(res)}; rows changed {int(diff.any(axis=1).sum())}; cells by column {L.column.value_counts().to_dict() if len(L) else {}}")
print("lakes by reason:", L[L.column != "dam_cap_per_uparea"].drop_duplicates("node").why.value_counts().to_dict())
print(L[L.why.str.startswith("matched")].to_string())
print(f"wrote {out}")
