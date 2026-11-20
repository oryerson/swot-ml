"""Grade every ResOpsUS record (link L1) by ruling on each flag that step 1 raised.

Step 1b of the crosswalk. It runs after step23_crosswalk.py, because it uses the area of the dam's graph
node(s) in the SWOT Prior Lake Database (PLD) as independent evidence.

Reads (under $SWOTGAT_WORK/CONUS_storage_fork)
  crosswalk/resops_record_qa.csv        flags from step1_resops_qa.py
  crosswalk/dam_node_crosswalk.csv      from step23_crosswalk.py: members_area (used by the rules);
                                        dz_resops and resops_elev_msl (SWOT level evidence, copied to
                                        the output for the reader)
  ResOpsUS/time_series_all/ResOpsUS_<DAM_ID>.csv

Writes
  crosswalk/resops_record_verdict.csv   the step-1 rows with has_data, plus L1_grade and L1_reason

Usage
  python step1b_resops_verdict.py        (no arguments)

Method
  Grades: A = clean. B = flagged, and the flag is explained. C = usable but doubtful. X = rejected
  (the record is not this dam's, or it is corrupt). Every rule that fires is written to L1_reason.
    - Three records are rejected by hand (table MANUAL; the evidence is quoted in each entry).
    - status ok -> A. status fixable (a unit factor from step 1) -> B.
    - Any other record starts at B and is tested flag by flag. "Physics holds" below means: the
      storage-elevation slope dS/dh divided by the PLD area of the dam's member nodes lies in 1/3-3
      and the storage-elevation correlation is at least 0.8.
        duplicate         noted; the grade does not change. The partner record of each pair is
                          expected to be one of the hand rejections.
        median/capacity   stays B if physics holds (then the GDW capacity is the likely error);
                          otherwise C.
        unit jump         stays B if the largest sustained shift in 2013-2020 (see max_jump) is below
                          a factor of 100, because then the jump is a drawdown and refill; otherwise C.
        storage-elevation inconsistent   the ResOpsUS elevation series is judged unreliable and
                          storage is judged on its own: C unless median/capacity lies in 0.02-1.3.
        dS/dh vs GDW area (without a median/capacity flag)   the slope is compared with the PLD area
                          instead. C only if the correlation is at least 0.8 and dS/dh / PLD area is
                          outside 0.2-5. A slope counts as evidence only when the elevation series is
                          coherent with storage.
"""
from swotgat.paths import WORK
import numpy as np, pandas as pd
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork"); O = F/"crosswalk"
Q = pd.read_csv(O/"resops_record_qa.csv")
X = pd.read_csv(O/"dam_node_crosswalk.csv", dtype={"node": str}).set_index("DAM_ID")
Q = Q[Q.status == "has_data"].copy()
Q["pld_area"] = Q.DAM_ID.map(X.members_area)
Q["dSdh_over_pld"] = Q.dSdh/Q.pld_area
Q["dz_resops"] = Q.DAM_ID.map(X.dz_resops); Q["resops_elev_msl"] = Q.DAM_ID.map(X.resops_elev_msl)

MANUAL = {  # hand rulings on record identity; each entry quotes its evidence
    1125: ("X", "Narrows record is DeGray's: storage series identical to 1121 (r>0.999) and ResOps elevation 123 m = DeGray pool (SWOT/GDW Greeson 165-170 m)"),
    223:  ("X", "Coyote record duplicates Coyote Valley/Mendocino (169): r>0.999; median 3x Coyote capacity"),
    1585: ("X", "Beltzville storage flat at 1,840 while elevation varies; dS/dh < 0"),
}
def max_jump(dam):
    """Return the largest sustained shift of the storage level in 2013-2020, in log10.

    dam is the ResOpsUS DAM_ID. For every possible change point, the median of the two yearly medians
    before it is compared with the median of the two after it. A change of units persists across such
    two-year blocks. A drawdown and refill (Antero 2015-16, Horseshoe) does not.
    """
    t = pd.read_csv(F/f"ResOpsUS/time_series_all/ResOpsUS_{dam}.csv"); t["date"] = pd.to_datetime(t.date, errors="coerce")
    x = pd.to_numeric(t.set_index("date").storage, errors="coerce").loc["2013":"2020"]   # only shifts in or just before the 2015-2020 test window
    ym = x.groupby(x.index.year).median().clip(lower=1e-6)
    best = 0.0
    for k in range(2, len(ym) - 1):
        a, b = np.median(ym.iloc[k-2:k]), np.median(ym.iloc[k:k+2])
        best = max(best, abs(np.log10(a/b)))
    return best
def grade(q):
    """Return (L1_grade, L1_reason) for one flagged-record row, by the rules of the module docstring.

    q is a row of the step-1 table with pld_area and dSdh_over_pld added. Hand rulings (MANUAL) come
    first, then "fixable" -> B and "ok" -> A. Any other record starts at B and each flag named in its
    record_note is ruled on; a flag that is not explained lowers the grade to C. The reason joins the
    rulings, or repeats the step-1 note if no rule fired.
    """
    if q.DAM_ID in MANUAL: return MANUAL[q.DAM_ID]
    if q.record_status == "fixable": return "B", f"fixed x{q.fix_factor:g}: {q.record_note}"
    if q.record_status == "ok": return "A", ""
    notes = str(q.record_note); why = []; g = "B"
    phys_ok = pd.notna(q.dSdh_over_pld) and 1/3 <= q.dSdh_over_pld <= 3 and q.r_SH >= 0.8
    if "duplicate" in notes:                      # the surviving member of a duplicate pair
        why.append("duplicate partner rejected; this record matches its own dam")
    if "median/capacity" in notes:
        if phys_ok: why.append("GDW capacity inconsistent but dS/dh matches PLD area (GDW capacity likely wrong)")
        elif pd.notna(q.med_over_cap) and q.med_over_cap < 0.02: g = "C"; why.append("storage median <2% of capacity and no physics support")
        else: g = "C"; why.append("magnitude vs capacity unexplained")
    if "unit jump" in notes:
        # a unit change shifts the level for good; a drawdown/refill does not (max_jump tests this)
        if max_jump(q.DAM_ID) < 2.0: why.append("between-year shift is a drawdown/refill (not sustained)")
        else: g = "C"; why.append("sustained >=100x level shift (unit change)")
    if "storage-elevation inconsistent" in notes:
        why.append("ResOps elevation series unreliable (storage judged separately)")
        if not (pd.notna(q.med_over_cap) and 0.02 <= q.med_over_cap <= 1.3): g = "C"
    if "dS/dh / GDW area" in notes and "median/capacity" not in notes:
        why.append("dS/dh vs PLD area " + (f"{q.dSdh_over_pld:.2f} ok" if phys_ok else f"{q.dSdh_over_pld:.2f} (GDW/PLD extent mismatch)"))
        # dS/dh is only evidence when the ResOps elevation series itself is coherent with storage
        if not phys_ok and pd.notna(q.dSdh_over_pld) and q.r_SH >= 0.8 and not (0.2 <= q.dSdh_over_pld <= 5): g = "C"
    return g, "; ".join(why) or notes
V = Q.apply(grade, axis=1, result_type="expand"); V.columns = ["L1_grade", "L1_reason"]
Q = pd.concat([Q, V], axis=1)
Q.to_csv(O/"resops_record_verdict.csv", index=False)
print(Q.L1_grade.value_counts().sort_index().to_string())
pd.set_option("display.width", 250)
print(Q[Q.L1_grade.isin(["C", "X"])][["DAM_ID", "DAM_NAME", "STATE", "med_over_cap", "dSdh_over_pld", "L1_grade", "L1_reason"]].round(3).to_string(max_colwidth=110))
