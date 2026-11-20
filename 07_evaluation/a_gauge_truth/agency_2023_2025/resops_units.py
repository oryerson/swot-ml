"""Find ResOpsUS storage records that are in the wrong unit, using reservoir capacity as a referee.

Three numbers are compared per reservoir: the ResOpsUS median storage R, the median A of the
downloaded 2023-2025 agency series, and the capacity K from Global Dam Watch (GDW), which is
independent of both. Median storage should lie between about 2% and 120% of capacity. A source
that is a clean unit factor away from that range is the one that is wrong.

Reads ($SWOTGAT_WORK/CONUS_storage_fork)
  resops_targets_full.csv     ResOpsUS dams with their graph node
  harvest.py                  imported from this folder for `resops_hist` (ResOpsUS 2010-2020)
  harvested_legacy_gated.csv, harvested.csv, harvested_pass*.csv   downloaded series, if present
  $SWOTGAT_WORK/CONUS/attributes/static_conus_storage.parquet   dam_cap_mcm, dam_has_gdw

Writes
  resops_unit_audit.csv   node, dam, agency, R_resops, A_harvest, K_capacity, gdw_catalogued,
                          R_over_K, A_over_K, harvest_method, verdict

Usage
  python resops_units.py     (no arguments; prints the suspect records)

Method
  R is the ResOpsUS median of 2010-2020 (needs more than 200 values). If R is outside 2% to
  120% of K, the verdict is "RESOPS SUSPECT" followed by every unit factor that would bring R
  into that range: 1000, 1233.48 (acre-feet), 1e6 or their inverses, or "(no clean factor)".
  A and A/K are reported for comparison but do not enter the verdict.

Notes
  This script only reports. It changes no data.
"""
from swotgat.paths import WORK
import sys, glob, math
from pathlib import Path
import numpy as np, pandas as pd
F = Path(f"{WORK}/CONUS_storage_fork")
C = Path(f"{WORK}/CONUS")
sys.path.insert(0, str(F)); import harvest as H
t = pd.read_csv(F/"resops_targets_full.csv", dtype={"node": str}).drop_duplicates("node")
st = pd.read_parquet(C/"attributes/static_conus_storage.parquet",
                     columns=["dam_cap_mcm", "dam_has_gdw"]).reset_index()
st["subbasin"] = st.subbasin.astype(str); st = st.drop_duplicates("subbasin").set_index("subbasin")
parts = []
for f in [F/"harvested_legacy_gated.csv", *sorted(F.glob("harvested.csv")), *sorted(F.glob("harvested_pass*.csv"))]:
    if f.is_file(): parts.append(pd.read_csv(f, dtype={"node": str}, usecols=["node", "storage_mcm", "method"]))
harv = pd.concat(parts) if parts else pd.DataFrame(columns=["node", "storage_mcm", "method"])
A_med = harv.groupby("node").storage_mcm.median()
A_meth = harv.groupby("node").method.first()
FACT = {1.0: "same", 1000.0: "x1000", 1/1000.0: "/1000", 1233.48: "x1233(af)", 1/1233.48: "/1233(af)",
        1e6: "x1e6", 1e-6: "/1e6"}
def plaus(x, K):
    """Return True if storage `x` lies within 2% to 120% of a positive capacity `K`."""
    return K > 0 and 0.02*K <= x <= 1.2*K
rows = []
for _, d in t.iterrows():
    hist = H.resops_hist(d.DAM_ID)
    R = hist.storage.median() if hist is not None and hist.storage.notna().sum() > 200 else np.nan
    K = st.dam_cap_mcm.get(d.node, np.nan); gdw = st.dam_has_gdw.get(d.node, np.nan)
    A = A_med.get(d.node, np.nan)
    verdict = ""
    if np.isfinite(R) and np.isfinite(K) and K > 0:
        if not plaus(R, K):
            # which unit factor would make the ResOpsUS median plausible?
            fix = [lab for f, lab in FACT.items() if f != 1.0 and plaus(R*f, K)]
            verdict = "RESOPS SUSPECT " + (",".join(fix) if fix else "(no clean factor)")
    rows.append(dict(node=d.node, dam=d.DAM_NAME, agency=d.AGENCY_CODE, R_resops=R, A_harvest=A,
                     K_capacity=K, gdw_catalogued=gdw, R_over_K=R/K if K else np.nan,
                     A_over_K=A/K if K else np.nan, harvest_method=A_meth.get(d.node, ""), verdict=verdict))
D = pd.DataFrame(rows); D.to_csv(F/"resops_unit_audit.csv", index=False)
sus = D[D.verdict.str.startswith("RESOPS")]
print(f"reservoirs audited: {len(D)} | ResOps median implausible vs GDW capacity: {len(sus)}")
print(sus.verdict.str.replace("RESOPS SUSPECT ", "").value_counts().to_string())
print("\nexamples (R and A in MCM, K = GDW capacity):")
print(sus.sort_values("R_over_K").head(20)[["dam","agency","R_resops","A_harvest","K_capacity","gdw_catalogued","verdict"]]
      .to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
