"""Apply the hand-checked dam assignments to a static-attribute table and write the result as a new table.

For a few lakes the automatic match between a lake and its dam in Global Dam Watch (GDW) is wrong. The override
table g4_overrides.csv lists these lakes, one row per lake node, with the dam attributes they should have: those of
another GDW dam (decision "replace") or imputed values for a lake with no GDW dam (decision "no_dam"). The table also
holds the evidence for each ruling. In the paper's run it has 6 lakes.

Reads
  IN.parquet    static-attribute table; the index has a level `subbasin` (the node id)
  --overrides   the override table (default: g4_overrides.csv in this script's folder; shipped as
                02_static_attributes/inputs/g4_overrides.csv)

Writes
  OUT.parquet   the new table. The input is never modified, and an existing OUT is not overwritten.
  --report      optional CSV with one row per override node and column: old value, new value, changed

Method
  For each override node, which must be a lake row (is_reservoir == 1) that occurs exactly once, the columns
  dam_has_gdw, dam_cap_mcm, dam_hgt_m, dam_depth_m, dam_dor_pc and use_elec, use_irri, use_fcon, use_supp, use_recr,
  use_navi are set from the override table, cast to the table's own types. dam_cap_per_uparea is then recomputed as
  dam_cap_mcm / uparea_km2 (float32) from the input table's own uparea_km2, the same definition as in
  build_conus_gdw_static.py and fix_gdw_static.py. The same override table can therefore be applied to tables of
  different graph versions. The script checks that no other row, column, type or index entry changed.

Usage
  python apply_g4_overrides.py IN.parquet OUT.parquet [--overrides g4_overrides.csv] [--report REPORT.csv]
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
COLS = ["dam_has_gdw", "dam_cap_mcm", "dam_hgt_m", "dam_depth_m", "dam_dor_pc",
        "use_elec", "use_irri", "use_fcon", "use_supp", "use_recr", "use_navi"]
ap = argparse.ArgumentParser(); ap.add_argument("inp", type=Path); ap.add_argument("out", type=Path)
ap.add_argument("--overrides", type=Path, default=HERE/"g4_overrides.csv"); ap.add_argument("--report", type=Path)
a = ap.parse_args()
assert a.inp.exists(), a.inp
assert not a.out.exists(), f"refusing to overwrite {a.out}"
assert a.out.resolve() != a.inp.resolve()
st = pd.read_parquet(a.inp)
sid = pd.Index(st.index.get_level_values("subbasin").astype(str))
O = pd.read_csv(a.overrides, dtype={"node": str})
new = st.copy(); rep = []
for o in O.itertuples():
    k = np.flatnonzero(sid == o.node); assert len(k) == 1, (o.node, len(k)); k = int(k[0])
    assert st.iloc[k]["is_reservoir"] == 1, f"{o.node} is not a reservoir row"
    for c in COLS:
        new.iloc[k, new.columns.get_loc(c)] = np.asarray(getattr(o, c)).astype(st[c].dtype)
    cpu = (np.float32(new.iloc[k][["dam_cap_mcm"]].values[0]) / st.iloc[k]["uparea_km2"]).astype("float32")
    new.iloc[k, new.columns.get_loc("dam_cap_per_uparea")] = cpu
    for c in COLS + ["dam_cap_per_uparea"]:
        rep.append(dict(node=o.node, decision=o.decision, column=c, old=st.iloc[k][c], new=new.iloc[k][c]))
# nothing else may change
eq = (new.values == st.values) | (pd.isna(new).values & pd.isna(st).values)
rows_ch = set(sid[np.flatnonzero(~eq.all(axis=1))]); cols_ch = set(st.columns[np.flatnonzero(~eq.all(axis=0))])
assert rows_ch <= set(O.node), f"unexpected rows changed: {rows_ch - set(O.node)}"
assert cols_ch <= set(COLS) | {"dam_cap_per_uparea"}, f"unexpected columns changed: {cols_ch}"
assert list(new.columns) == list(st.columns) and new.index.equals(st.index) and (new.dtypes == st.dtypes).all()
new.to_parquet(a.out)
md5 = hashlib.md5(a.out.read_bytes()).hexdigest()
R = pd.DataFrame(rep); R["changed"] = ~((R.old == R.new) | (pd.isna(R.old) & pd.isna(R.new)))
if a.report: R.to_csv(a.report, index=False)
pd.set_option("display.width", 200)
print(R[R.changed].to_string(index=False))
print(f"rows changed: {len(rows_ch)} {sorted(rows_ch)}\ncolumns changed: {sorted(cols_ch)}\nwrote {a.out} md5 {md5}")
