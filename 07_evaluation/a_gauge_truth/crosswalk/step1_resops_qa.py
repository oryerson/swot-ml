"""Screen every ResOpsUS storage record: is it a clean record of its own dam, in million cubic meters (MCM)?

Step 1 of the crosswalk. Record quality is link "L1" of the test-set chain (L2 = dam to graph node,
L4 = agency gauge to dam). This script only raises flags; step1b_resops_verdict.py rules on them and
assigns the final L1 grade.

Reads
  $SWOTGAT_WORK/CONUS_storage_fork/ResOpsUS/attributes/reservoir_attributes.csv
  $SWOTGAT_WORK/CONUS_storage_fork/ResOpsUS/time_series_all/ResOpsUS_<DAM_ID>.csv
  under $SWOTGAT_NAS_WATER: the Global Dam Watch (GDW) v1.0 reservoir shapefile (CAP_MCM, CAP_MAX,
      AREA_SKM). The ResOpsUS DAM_ID is used as the GDW GRAND_ID.

Writes
  $SWOTGAT_WORK/CONUS_storage_fork/crosswalk/resops_record_qa.csv      one row per ResOpsUS dam

Usage
  python step1_resops_qa.py        (no arguments)

Method
  A dam is tested if it has at least 365 days of storage in 2015-2020 (status has_data; otherwise
  short or no_file). For each tested dam:
    capacity     median 2015-2020 storage / GDW capacity. Flagged outside 0.02-1.3.
    unit jump    largest change of the yearly median storage from one year to the next in 1990-2020,
                 in log10. Flagged at 2 or more (a factor of 100), the sign of a change of units.
    physics      slope dS/dh of a straight line fitted to paired storage and elevation, 2010-2020
                 (needs more than 200 pairs and an elevation standard deviation above 0.05 m). The
                 slope is an area (MCM per m = km2). Flagged if dS/dh <= 0, if the storage-elevation
                 correlation is below 0.8, or if dS/dh / GDW area is outside 1/3-3.
    duplicates   pairs of dams whose 2010-2020 daily storage correlates above 0.999 over at least
                 365 shared days (two DAM_IDs for one water body, or a copied series).
    zeros, spikes   counted only (see clean_flags); they do not change the status.
  record_status is ok (no flag), review (at least one flag; the reasons are in record_note), fixable
  or suspect. The last two come from two hand-checked tables in this file: KNOWN_FIX gives the factor
  that converts a record to MCM (Center Hill, x1000 because the record is in km3 from 1996 on) and
  KNOWN_BAD lists records that cannot be used (Beltzville).

Notes
  All unit fix factors are defined here (column fix_factor). Later steps read them from the output
  table; no other script hard-codes a factor.
"""
from swotgat.paths import NAS_WATER, WORK
import numpy as np, pandas as pd, geopandas as gpd
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork"); O = F/"crosswalk"
TSA = F/"ResOpsUS/time_series_all"
R = pd.read_csv(F/"ResOpsUS/attributes/reservoir_attributes.csv")
G = gpd.read_file(f"{NAS_WATER}/cjgleason/ted/GDW/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp",
                  columns=["GRAND_ID", "CAP_MCM", "CAP_MAX", "AREA_SKM", "AREA_MAX"], ignore_geometry=True)
G["GRAND_ID"] = pd.to_numeric(G.GRAND_ID, errors="coerce"); G = G[G.GRAND_ID > 0].drop_duplicates("GRAND_ID").set_index("GRAND_ID")
KNOWN_FIX = {1777: (1000.0, "Center Hill: km3 from 1996 (CWMS ~1,418 MCM)")}
KNOWN_BAD = {1585: "Beltzville: storage flat 1,840 while elevation varies; dS/dh<0",
             }

def clean_flags(s):
    """Count dropouts and isolated spikes in a storage series, with the rules of merge_truth.py.

    A dropout is an exact zero in a series whose median exceeds 1. A spike is a day that departs from
    the 5-day centered median by more than 10 robust scales, where the scale is 1.4826 x the median
    absolute day-to-day change plus 0.1% of the series range. Returns (zeros, spikes).
    """
    x = s.values.astype(float); med = np.nanmedian(x)
    zeros = int(((x == 0) & (med > 1)).sum())
    m = s.rolling(5, center=True, min_periods=3).median()
    scale = 1.4826*np.nanmedian(np.abs(np.diff(x))) + 1e-3*(np.nanmax(x)-np.nanmin(x))
    spikes = int(((s-m).abs() > 10*scale).sum())
    return zeros, spikes

rows, wide = [], {}
for _, r in R.iterrows():
    f = TSA/f"ResOpsUS_{int(r.DAM_ID)}.csv"
    d = dict(DAM_ID=int(r.DAM_ID), DAM_NAME=r.DAM_NAME, STATE=r.STATE)
    if not f.is_file(): rows.append({**d, "status": "no_file"}); continue
    t = pd.read_csv(f); t["date"] = pd.to_datetime(t.date, errors="coerce"); t = t.dropna(subset=["date"]).set_index("date")
    t = t.apply(pd.to_numeric, errors="coerce")
    t = t[~t.index.duplicated()].sort_index()
    s = t.storage.where(t.storage.notna()); w = s.loc["2015":"2020"].dropna()
    d["n_2015_2020"] = len(w)
    if len(w) < 365: rows.append({**d, "status": "short"}); continue
    wide[int(r.DAM_ID)] = s.loc["2010":"2020"]
    cap = G.CAP_MCM.get(r.DAM_ID, np.nan); capmax = G.CAP_MAX.get(r.DAM_ID, np.nan)
    d.update(median=w.median(), sd=w.std(), cap=cap, cap_max=capmax, med_over_cap=w.median()/cap if cap > 0 else np.nan)
    # unit jumps: year-median level shifts >= 100x (log10 >= 2) anywhere 1990-2020
    ym = s.loc["1990":"2020"].groupby(s.loc["1990":"2020"].index.year).median()
    ym = ym[ym > 0]
    lj = np.abs(np.log10(ym).diff()).max() if len(ym) > 2 else np.nan
    d["max_year_jump_log10"] = lj
    # physics: dS/dh from paired storage-elevation 2010-2020
    pr = t.loc["2010":"2020", ["storage", "elevation"]].dropna()
    if len(pr) > 200 and pr.elevation.std() > 0.05:
        b = np.polyfit(pr.elevation, pr.storage, 1)[0]; rr = pr.storage.corr(pr.elevation)
        d.update(dSdh=b, r_SH=rr, dSdh_over_area=b/G.AREA_SKM.get(r.DAM_ID, np.nan))
    d["zeros"], d["spikes"] = clean_flags(w)
    rows.append(d)
Q = pd.DataFrame(rows)

# duplicate records (two DAM_IDs, one water body / copied series)
W = pd.DataFrame(wide)
C = W.corr(min_periods=365)
dups = [(a, b, C.loc[a, b]) for i, a in enumerate(C.columns) for b in C.columns[i+1:] if C.loc[a, b] > 0.999]
dup_of = {}
for a, b, c in dups:
    dup_of.setdefault(a, []).append(b); dup_of.setdefault(b, []).append(a)
Q["dup_with"] = Q.DAM_ID.map(lambda i: ";".join(map(str, dup_of.get(i, []))))

def verdict(q):
    """Return (record_status, fix_factor, record_note) for one row of the flag table.

    A dam without a file or with a short record keeps its status as record_status. A dam in KNOWN_BAD is
    "suspect" (factor NaN) and a dam in KNOWN_FIX is "fixable" with its factor to MCM. Any other dam is
    "review" if at least one flag is raised (median/capacity outside 0.02-1.3, unit jump of 2 or more in
    log10, dS/dh <= 0 or storage-elevation correlation below 0.8, dS/dh / GDW area outside 1/3-3, a
    duplicate record) and "ok" otherwise, with factor 1. The note lists the flags raised.
    """
    if q.status in ("no_file", "short"): return q.status, 1.0, ""
    if q.DAM_ID in KNOWN_BAD: return "suspect", np.nan, KNOWN_BAD[q.DAM_ID]
    if q.DAM_ID in KNOWN_FIX: return "fixable", KNOWN_FIX[q.DAM_ID][0], KNOWN_FIX[q.DAM_ID][1]
    why = []
    if pd.notna(q.med_over_cap) and not (0.02 <= q.med_over_cap <= 1.3): why.append(f"median/capacity {q.med_over_cap:.3g}")
    if pd.notna(q.max_year_jump_log10) and q.max_year_jump_log10 >= 2: why.append(f"unit jump 10^{q.max_year_jump_log10:.1f} between years")
    if pd.notna(q.get("dSdh")) and (q.dSdh <= 0 or q.r_SH < 0.8): why.append(f"storage-elevation inconsistent (dS/dh {q.dSdh:.3g}, r {q.r_SH:.2f})")
    if pd.notna(q.get("dSdh_over_area")) and not (1/3 <= q.dSdh_over_area <= 3): why.append(f"dS/dh / GDW area {q.dSdh_over_area:.2f}")
    if q.dup_with: why.append(f"duplicate of {q.dup_with}")
    return ("review" if why else "ok"), 1.0, "; ".join(why)
V = Q.apply(verdict, axis=1, result_type="expand"); V.columns = ["record_status", "fix_factor", "record_note"]
Q = pd.concat([Q, V], axis=1)
Q.loc[Q.status.isna(), "status"] = "has_data"
Q.to_csv(O/"resops_record_qa.csv", index=False)
q = Q[Q.status == "has_data"]
print(f"ResOps dams: {len(Q)}; with >=365 d in 2015-2020: {len(q)}")
print(q.record_status.value_counts().to_string())
print(f"duplicate pairs (r>0.999): {len(dups)}  {[(a,b) for a,b,_ in dups][:10]}")
pd.set_option("display.width", 250)
print(q[q.record_status != "ok"][["DAM_ID", "DAM_NAME", "STATE", "median", "cap", "dSdh_over_area", "r_SH", "zeros", "spikes", "record_status", "record_note"]]
      .round(2).to_string(max_colwidth=70))
