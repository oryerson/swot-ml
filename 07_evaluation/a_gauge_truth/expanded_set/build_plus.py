"""Build the expanded reservoir test sets: the frozen test sets plus the newly found gauge records.

Step 4 of the expanded gauge set. The expanded sets are test data only; they are never written into a
training target. The frozen sets are read and never modified, and their rows are carried over
unchanged. All rules below were fixed before any model was scored on the new sites.

Folder names call the test sets "regimes": regime 3 = 1980-2020 (ResOpsUS), regime 1x = 2023-2026
(agency gauges). "truth" in file names means the gauge reference series. A "site" is one dam with its
graph node(s). L2 is the dam-to-node tier of the crosswalk; GDW = Global Dam Watch.

Reads (under $SWOTGAT_WORK/CONUS_storage_fork)
  gauges_expand/discovery/{dams.csv, series_fetched.csv, native.parquet}   steps 2 and 3
  gauges_expand/crosswalk_plus/dam_node_crosswalk_plus.csv    dam-to-node crosswalk of the new dams, built
                                                              with the L2 rules of step23_crosswalk.py
  crosswalk/resops_record_verdict.csv                         ResOpsUS unit fix factors
  eval/g4_fix/testsets/regime3_full_1980_2020_g4/, eval/g4_fix/testsets/regime1x_2023_2026_g4/
  testsets/regime1x_2023_2026/candidate_additions/            sites added earlier, kept apart so far
  under $SWOTGAT_NAS_WATER: the GDW v1.0 reservoir shapefile (YEAR_DAM, CAP_MCM)
  It imports screen from the copy of gauge_stat_screen.py in gauges_expand/stage, fit_map from
  harvest_2026.py, and gx_common.py.

Writes (under $SWOTGAT_WORK/CONUS_storage_fork/gauges_expand/testsets)
  regime3_plus_1980_2020/     the frozen 1980-2020 set (507 sites in the paper's run) + new dams:
      sites.csv, truth_all.parquet, truth_clean.parquet, truth_new_detail.parquet, qa_site_year.csv,
      predict_nodes.csv, unit_runs_new.csv, version.json
  regime1x_plus_2023_2026/    the frozen 2023-2026 set (457 sites) + the extended candidate additions
      (18) + new dams: sites.csv, truth.parquet, truth_clean_v1x.parquet, predict_nodes.csv, version.json
  build_log_dams.csv          dams that did not enter, with the reason
  Column added_by of sites.csv is frozen_g4, candidate_additions or gauges_expand.

Usage
  python build_plus.py        (no arguments; stops if an output folder already exists)

Method
  New dams. Group "new": rule-model GRanD dams outside the frozen crosswalk. Group "gap23", for
  2023-2026 only: dams that are High or Medium in regime 3 but have no High or Medium regime-1x series.

  Dams with a ResOpsUS record:
    1980-2020  The series is the ResOpsUS record times its fix factor, treated as in build_regime3.py,
      with one change. These records have no 2015-2020 data, so every "2015-2020" reference of that
      script (the median for unit runs, the maximum for the scale ratio) uses the dam's last six
      calendar years with at least 300 days. Unit runs are runs of at least 30 days at 10^2.5 to
      10^3.5 times the reference median or its inverse; runs that are too large are dropped and runs
      that are too small are kept. The gauge grade is A if record maximum / GDW capacity lies in
      0.5-1.5, B if in 1/3-3, otherwise X (dropped). At least 365 days are needed.
    2023-2026  An agency series is mapped onto the ResOpsUS record over their common days (at least
      120, in any years; the most recent 2,192 at most) with fit_map of harvest_2026.py and its
      grade A and B rules (storage: linear map; elevation: monotone rating). The series needs at
      least 180 days in 2023-07-01 to 2026-06-12. The grade-A series with the smallest relative
      error is used; if there is none, the best grade-B series.

  Dams without a ResOpsUS record:
    Only storage series are used, because an elevation series needs a rating curve that no agency
    service publishes. Acre-feet are converted to MCM (x 0.00123348). A series is admissible if its
    99th percentile is 0.25-2 times the GDW capacity (this catches unit errors and wrong reservoirs)
    and it has at least 365 days in 1980-2020, or 180 days in 2023-07-01 to 2026-06-12.
    The primary series is the admissible one with the most days in the period. It is "linked" to the
    dam if the gauge is at most 2 km away, or it is a TWDB or GDROM v2 series (matched by name or
    id), or the name similarity is at least 0.6.
    Grade: B if linked, C if not. A if a second admissible series from another source agrees over at
    least 365 shared days (r >= 0.995 and RMS residual of the linear map at most 1% of the range).
    C if a second series disagrees (r < 0.95). In 1980-2020, agreeing series also fill the days the
    primary lacks, after the linear map.

  Tier. Gauge grades A, B, C count as High, Medium, Low. site_tier is the weaker of the L2 node tier
  and the gauge tier.

  Cleaning. Days on which more than half of the raw values are marked bad by the agency are dropped.
  clean7 removes zero dropouts and isolated spikes. gauge_stat_screen.screen then flags spike and
  negative days. These days stay in truth_all.parquet (with removed = True) and in truth.parquet,
  and are left out of the truth_clean files.

  Site-year check, 1980-2020 (qa_site_year.csv). A year is used for scoring (use = True) if it has
  at least 300 days; its maximum is 1/3 to 3 times the maximum of the reference years; it is not flat
  (zero standard deviation, or one value on more than 80% of days); it is not within the first fill
  (up to two years after GDW YEAR_DAM); and it does not duplicate another dam's series that year
  (r > 0.999, among series with at least 300 days in the year).

  A dam enters a set only if it has data in that period: at least one usable year in 1980-2020, or
  at least 180 clean days in 2023-07-01 to 2026-06-12. No new site may share its primary node with a
  frozen site (asserted).
"""
from swotgat.paths import NAS_WATER, WORK
import json, sys, shutil, hashlib, numpy as np, pandas as pd
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork"); X = F/"gauges_expand"; G4 = F/"eval/g4_fix/testsets"
sys.path.insert(0, str(F/"eval")); sys.path.insert(0, str(F/"gauges2026")); sys.path.insert(0, str(X))
sys.path.insert(0, str(X/"stage"))          # staged gauge_stat_screen: its audit runs only under __main__, so importing writes nothing
from gauge_stat_screen import screen
assert "gauges_expand/stage" in sys.modules["gauge_stat_screen"].__file__
import harvest_2026 as H
from gx_common import AF, sim
OUT3 = X/"testsets/regime3_plus_1980_2020"; OUT1 = X/"testsets/regime1x_plus_2023_2026"
for o in (OUT3, OUT1): assert not o.exists(), f"{o} exists: build into a new folder"
for o in (OUT3, OUT1): o.mkdir(parents=True)
TIER = {"High": 0, "Medium": 1, "Low": 2}; INV = {v: k for k, v in TIER.items()}
def tmin(*t):
    """Return the weakest of the given tiers, or "Reject" if any is not High, Medium or Low."""
    return INV[max(TIER[x] for x in t)] if all(x in TIER for x in t) else "Reject"

# ---------------- inputs
DM = pd.read_csv(X/"discovery/dams.csv"); SF = pd.read_csv(X/"discovery/series_fetched.csv", dtype={"sid": str})
NV = pd.read_parquet(X/"discovery/native.parquet"); NV["date"] = pd.to_datetime(NV.date)
XP = pd.read_csv(X/"crosswalk_plus/dam_node_crosswalk_plus.csv", dtype={"node": str, "members": str, "basin": str}).set_index("DAM_ID")
XF = pd.read_csv(F/"crosswalk/resops_record_verdict.csv").drop_duplicates("DAM_ID").set_index("DAM_ID")
R3 = pd.read_csv(G4/"regime3_full_1980_2020_g4/sites.csv", dtype={"node": str, "members": str, "basin": str})
import geopandas as gpd
_g = gpd.read_file(f"{NAS_WATER}/cjgleason/ted/GDW/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp", columns=["GRAND_ID", "YEAR_DAM", "CAP_MCM"], ignore_geometry=True)
for c in ("GRAND_ID", "YEAR_DAM", "CAP_MCM"): _g[c] = pd.to_numeric(_g[c], errors="coerce")
_g = _g[_g.GRAND_ID > 0].drop_duplicates("GRAND_ID").set_index("GRAND_ID"); DAMYR = _g.YEAR_DAM.where(_g.YEAR_DAM > 1800); CAPG = _g.CAP_MCM
NVG = {k: g for k, g in NV.groupby(["source", "sid", "kind"])}
def series(src, sid, kind):
    """Return (values, bad_frac) of one downloaded series, both indexed by date.

    The series is looked up by source, series id and kind ("storage" or "elevation") in the table of
    downloaded daily values. Days on which more than half of the raw values are marked bad by the
    agency are left out of the values, and storage in acre-feet is converted to MCM. bad_frac is the
    share of bad raw values on every day. Returns (None, None) if the series was not downloaded.
    """
    d = NVG.get((src, str(sid), kind))
    if d is None or d.empty: return None, None
    d = d.set_index("date").sort_index(); d = d[~d.index.duplicated()]
    unit = d.unit.iloc[0]; v = d.v.where(d.bad_frac <= 0.5)
    if kind == "storage" and unit == "ac-ft": v = v*AF
    return v.dropna(), d.bad_frac
def clean7(s):
    """Remove dropouts and isolated spikes from a daily storage series and return what is left.

    The rules are those of clean7 in build_regime3.py and clean in step7_build_testsets.py. A dropout is
    an exact zero in a series whose median exceeds 1. A spike is a day more than 10 robust scales from
    the 5-day centered median, where the scale is 1.4826 x the median absolute day-to-day change plus
    0.1% of the range.
    """
    s = s.astype(float).copy(); med = s.median()
    if med > 1: s[s == 0] = np.nan
    m = s.rolling(5, center=True, min_periods=3).median()
    scale = 1.4826*np.nanmedian(np.abs(np.diff(s.dropna().values))) + 1e-3*(s.max() - s.min())
    s[(s - m).abs() > 10*scale] = np.nan
    return s.dropna()
def ref_years(s):
    """Return the part of series s in its last six calendar years that have at least 300 days.

    These years stand in for the 2015-2020 reference period of build_regime3.py. If no year has 300
    days, all of s is returned.
    """
    n = s.groupby(s.index.year).size(); yrs = sorted(n[n >= 300].index)[-6:]
    return s[s.index.year.isin(yrs)] if yrs else s
def unit_runs(v, ref_med):
    """Drop the runs of days that were recorded about 1,000 times too large, and log every run found.

    v is a daily storage series and ref_med the median of its reference years. This is unit_runs of
    build_regime3.py without its test on elevation. Values that are not positive are dropped first. A run
    is at least 30 consecutive days at 10^2.5 to 10^3.5 times ref_med, or at its inverse. A run above
    ref_med is dropped; a run below it is kept, because it can be a real low level. Returns (series, list
    of dicts with start, end, n and action).
    """
    v = v.dropna(); v = v[v > 0]; logs = []
    if v.empty or not np.isfinite(ref_med) or ref_med <= 0: return v, logs
    L = np.log10(v/ref_med); off = (L.abs() >= 2.5) & (L.abs() <= 3.5); grp = (off != off.shift()).cumsum(); out = v.copy()
    for g, idx in L.groupby(grp).groups.items():
        if not off.loc[idx[0]] or len(idx) < 30: continue
        big = float(np.median(L.loc[idx])) > 0
        if big: out = out.drop(idx); logs.append(dict(start=idx[0].date(), end=idx[-1].date(), n=len(idx), action="dropped (too large)"))
        else: logs.append(dict(start=idx[0].date(), end=idx[-1].date(), n=len(idx), action="kept (low level)"))
    return out, logs
def agree(a, b):
    """Measure how well two series agree on their shared days.

    Returns None if they share fewer than 365 days or if either is constant there. Otherwise returns a
    dict: r (Pearson correlation), k and c (slope and intercept of the linear map a = k*b + c), rel (RMS
    residual of that map divided by the range of a) and n (number of shared days).
    """
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(j) < 365 or j.a.std() == 0 or j.b.std() == 0: return None
    r = float(np.corrcoef(j.a, j.b)[0, 1]); k, c = np.polyfit(j.b, j.a, 1); rel = float(np.sqrt(np.mean((j.a - (k*j.b + c))**2))/(j.a.max() - j.a.min()))
    return dict(r=r, rel=rel, n=len(j), k=k, c=c)

# ---------------- per dam
W81, W1x = ("1980-01-01", "2020-12-31"), ("2023-01-01", "2026-09-30")
rows3, rows1, tr3, tr1, qa3, units, glog = [], [], [], [], [], [], []
for _, d in DM.iterrows():
    dam = int(d.DAM_ID); cap = float(CAPG.get(dam, np.nan)); sf = SF[SF.DAM_ID == dam]; info = dict(DAM_ID=dam, DAM_NAME=d.DAM_NAME, STATE=d.STATE)
    if d.group == "new":
        if dam not in XP.index or XP.tier.get(dam) not in TIER: glog.append({**info, "status": f"L2 {XP.tier.get(dam, 'missing')}"}); continue
        x = XP.loc[dam]; L2 = x.tier
        base = dict(node=x.node, members=x.members, basin=x.basin, L2_tier=L2, supervised_2023_25=bool(x.supervised), primary_share=x.primary_share,
                    area_ratio=x.area_ratio, weaknesses=x.weaknesses, conflicts=x.conflicts)
    else:
        x = R3[R3.DAM_ID == dam].iloc[0]; L2 = x.L2_tier
        base = dict(node=x.node, members=x.members, basin=x.basin, L2_tier=L2, supervised_2023_25=bool(x.supervised_2023_25), primary_share=x.primary_share,
                    area_ratio=x.area_ratio, weaknesses=x.weaknesses, conflicts=x.conflicts)
    ref = None
    if d.resops:
        rs, _ = series("RESOPS", str(dam), "storage")
        if rs is not None and len(rs) > 365:
            fixf = float(XF.fix_factor.get(dam, 1.0)) if np.isfinite(XF.fix_factor.get(dam, np.nan)) else 1.0
            ref = rs*fixf
    # ======== regime 3 (1981-2020): new dams only (gap23 dams are already in regime 3)
    if d.group == "new":
        cand = []
        if ref is not None:
            v = ref.loc[W81[0]:W81[1]]; v2, lg = unit_runs(v, ref_years(v).median()); units += [{**info, **l} for l in lg]
            mx = float(ref_years(v2).max()) if len(v2) else np.nan; r_ = mx/cap if cap > 0 else np.nan
            L1 = "A" if 0.5 <= r_ <= 1.5 else ("B" if 1/3 <= r_ <= 3 else "X")
            if L1 != "X" and len(v2) >= 365: cand.append(("RESOPS", str(dam), v2, L1, f"ResOpsUS record; max/GDW cap {r_:.2f}"))
            g3 = {"A": "High", "B": "Medium"}.get(L1)
        else:
            adm = []
            for _, s in sf[(sf.kind == "storage") & (sf.source != "RESOPS")].iterrows():
                v, _ = series(s.source, s.sid, "storage")
                if v is None or len(v.loc[W81[0]:W81[1]]) < 365: continue
                p99 = float(v.quantile(.99)); r_ = p99/cap if cap > 0 else np.nan
                if not (0.25 <= r_ <= 2): glog.append({**info, "status": f"r3 {s.source}:{s.sid} cap ratio {r_:.2f} rejected"}); continue
                link = (s.km <= 2) if np.isfinite(s.km) else (s.source in ("TWDB", "GDROMV2"))
                link = link or sim(d.DAM_NAME, s.note) >= 0.6
                adm.append((s.source, s.sid, v.loc[W81[0]:W81[1]], link, r_))
            if adm:
                adm.sort(key=lambda z: -len(z[2])); p = adm[0]; grade = "B" if p[3] else "C"; note = f"cap ratio {p[4]:.2f}"
                fills = []
                for q in adm[1:]:
                    if q[0] == p[0]: continue
                    a = agree(p[2], q[2])
                    if a is None: continue
                    if a["r"] >= 0.995 and a["rel"] <= 0.01: grade = "A"; fills.append((q, a)); note += f"; agrees with {q[0]}:{q[1]} r={a['r']:.4f} rel={a['rel']:.4f} n={a['n']}"
                    elif a["r"] < 0.95: grade = "C"; note += f"; DISAGREES with {q[0]}:{q[1]} r={a['r']:.3f}"; break
                    elif a["r"] >= 0.99 and a["rel"] <= 0.025 and grade == "C" and p[3]: grade = "B"
                v = p[2].copy()
                for q, a in fills:
                    add = (a["k"]*q[2] + a["c"]); add = add[~add.index.isin(v.index)]; v = pd.concat([v, add]).sort_index()
                cand.append((p[0], p[1], v, grade, note))
            g3 = {"A": "High", "B": "Medium", "C": "Low"}.get(cand[0][3]) if cand else None
        if cand:
            src, sid, v, grade, note = cand[0]
            e = clean7(v.dropna()) if len(v.dropna()) >= 5 else v.dropna()
            flags, _ = screen(e, "res", cap, True) if len(e) >= 30 else ([], {})
            fl = pd.DataFrame(flags, columns=["date", "flag"]) if flags else pd.DataFrame(columns=["date", "flag"])
            rm = set(pd.to_datetime(fl[fl.flag.isin(["spike", "negative"])].date)); sc = e[~e.index.isin(rm)]
            tr3.append(pd.DataFrame(dict(DAM_ID=dam, date=e.index, storage_mcm=e.values, removed=e.index.isin(rm), source=src, sid=sid)))
            ref_max = float(ref_years(sc).max()) if len(sc) else np.nan
            for y, g in sc.groupby(sc.index.year):
                dy = float(DAMYR.get(dam, np.nan))
                row = dict(DAM_ID=dam, year=int(y), n_days=len(g), ymax=float(g.max()), ymed=float(g.median()), scale_ratio=float(g.max()/ref_max) if ref_max > 0 else np.nan,
                           flat=bool(g.std() == 0 or g.value_counts(normalize=True).iloc[0] > 0.8), dup_with=None, n_removed=int(sum(1 for z in rm if z.year == y)),
                           dam_year=dy, first_fill=bool(np.isfinite(dy) and y <= dy + 2))
                row["scale_bad"] = bool(np.isfinite(row["scale_ratio"]) and not (1/3 <= row["scale_ratio"] <= 3)); qa3.append(row)
            rows3.append({**info, **base, "site_tier": tmin(L2, g3), "gauge_grade": grade, "source": src, "sid": sid, "evidence": note,
                          "record_status": d.record_status, "added_by": "gauges_expand"})
    # ======== regime 1x (2023-26)
    pick = None
    if ref is not None:                                            # map an agency series onto the ResOps record
        best = {}
        for _, s in sf[sf.source != "RESOPS"].iterrows():
            v, _ = series(s.source, s.sid, s.kind)
            if v is None or len(v.loc["2023-07-01":"2026-06-12"]) < 180: continue
            j = pd.concat([v.rename("x"), ref.rename("y")], axis=1).dropna()
            if len(j) < 120: continue
            H.OV0, H.OV1 = str(j.index[-min(len(j), 2192)].date()), str(j.index[-1].date())
            m = H.fit_map(v, ref, s.kind, False)
            if m and m["ok"] and (m["grade"] not in best or (best[m["grade"]][1]["rel_rmse"] > m["rel_rmse"])): best[m["grade"]] = ((s.source, s.sid, s.kind, v), m)
        for g_ in ("A", "B"):
            if g_ in best:
                (src, sid, kind, v), m = best[g_]
                w = v.loc[W1x[0]:W1x[1]]; vals = m["f"](w.values)
                note = f"{m['desc']}; overlap {H.OV0}..{H.OV1} r={m['r']:.4f} rel_rmse={m['rel_rmse']:.4f} n={m['n']} lag={m['lag']} grade {g_} vs ResOpsUS"
                pick = (src, sid, kind, pd.Series(vals, index=w.index + pd.Timedelta(days=m["lag"])), g_, note); break
        g1 = {"A": "High", "B": "Medium"}.get(pick[4]) if pick else None
    else:
        adm = []
        for _, s in sf[(sf.kind == "storage") & (sf.source != "RESOPS")].iterrows():
            v, _ = series(s.source, s.sid, "storage")
            if v is None or len(v.loc["2023-07-01":"2026-06-12"]) < 180: continue
            p99 = float(v.quantile(.99)); r_ = p99/cap if cap > 0 else np.nan
            if not (0.25 <= r_ <= 2): continue
            link = ((s.km <= 2) if np.isfinite(s.km) else (s.source in ("TWDB", "GDROMV2"))) or sim(d.DAM_NAME, s.note) >= 0.6
            adm.append((s.source, s.sid, v, link, r_))
        if adm:
            adm.sort(key=lambda z: -len(z[2].loc[W1x[0]:W1x[1]])); p = adm[0]; grade = "B" if p[3] else "C"; note = f"cap ratio {p[4]:.2f}"
            for q in adm[1:]:
                if q[0] == p[0]: continue
                a = agree(p[2], q[2])
                if a is None: continue
                if a["r"] >= 0.995 and a["rel"] <= 0.01: grade = "A"; note += f"; agrees with {q[0]}:{q[1]} r={a['r']:.4f} rel={a['rel']:.4f}"
                elif a["r"] < 0.95: grade = "C"; note += f"; DISAGREES with {q[0]}:{q[1]} r={a['r']:.3f}"; break
            pick = (p[0], p[1], "storage", p[2].loc[W1x[0]:W1x[1]], grade, note)
        g1 = {"A": "High", "B": "Medium", "C": "Low"}.get(pick[4]) if pick else None
    if pick:
        src, sid, kind, v, grade, note = pick
        e = clean7(v.dropna()) if len(v.dropna()) >= 5 else v.dropna()
        flags, _ = screen(e, "res", cap, True) if len(e) >= 30 else ([], {})
        rm = {pd.Timestamp(a) for a, f in flags if f in ("spike", "negative")}; ec = e[~e.index.isin(rm)]
        if len(ec.loc["2023-07-01":"2026-06-12"]) >= 180:
            tr1.append(pd.DataFrame(dict(DAM_ID=dam, date=e.index.strftime("%Y-%m-%d"), storage_mcm=e.values, is_anomaly=False, source=f"{src}:{kind}", sid=sid,
                                         method=f"gauges_expand: {note}", removed=e.index.isin(rm))))
            rows1.append({**info, **base, "site_tier": tmin(L2, g1), "ext_grade": grade, "ext_source": src, "ext_sid": sid, "kind": kind, "evidence": note,
                          "n_days_2326": int(len(ec)), "record_status": d.record_status, "group": d.group, "added_by": "gauges_expand"})
    if not ((rows3 and rows3[-1]["DAM_ID"] == dam) or (rows1 and rows1[-1]["DAM_ID"] == dam)): glog.append({**info, "status": "no admissible series"})

# ---------------- regime 3 plus
S3 = pd.DataFrame(rows3); Q3 = pd.DataFrame(qa3); T3 = pd.concat(tr3, ignore_index=True)
# duplicate years: a new dam whose truth equals another dam's truth (new or frozen) that year
FT = pd.read_parquet(G4/"regime3_full_1980_2020_g4/truth_clean.parquet")[["DAM_ID", "date", "storage_mcm"]]; FT["date"] = pd.to_datetime(FT.date)
ALL = pd.concat([FT, T3[~T3.removed][["DAM_ID", "date", "storage_mcm"]]]).pivot_table(index="date", columns="DAM_ID", values="storage_mcm")
newids = set(S3.DAM_ID); dup = {}
for y in range(1980, 2021):
    Xy = ALL.loc[str(y)]; Xy = Xy.loc[:, (Xy.notna().sum() >= 365*0.82) & (Xy.std() > 0)]
    if Xy.shape[1] < 2: continue
    Cm = np.corrcoef(Xy.ffill().bfill().values.T); a, b = np.where(np.triu(Cm > 0.999, 1))
    for i, j in zip(a, b):
        da, db = int(Xy.columns[i]), int(Xy.columns[j])
        if da in newids: dup[(da, y)] = db
        if db in newids: dup[(db, y)] = da
Q3["dup_with"] = [dup.get((a, y)) for a, y in zip(Q3.DAM_ID, Q3.year)]
Q3["use"] = (Q3.n_days >= 300) & ~Q3.scale_bad & ~Q3.flat & ~Q3.first_fill & Q3.dup_with.isna()
good = Q3[Q3.use].groupby("DAM_ID").size(); S3 = S3[S3.DAM_ID.isin(good.index)].copy(); S3["n_headline_years"] = S3.DAM_ID.map(good)
S3["low_signal"] = False; S3["L1_grade"] = S3.gauge_grade
F3 = pd.read_csv(G4/"regime3_full_1980_2020_g4/sites.csv", dtype={"node": str, "members": str, "basin": str}); F3["added_by"] = "frozen_g4"
assert not set(S3.node) & set(F3.node), "a new site shares a primary node with a frozen site"
pd.concat([F3, S3], ignore_index=True).to_csv(OUT3/"sites.csv", index=False)
T3 = T3[T3.DAM_ID.isin(S3.DAM_ID)]
TA = pd.read_parquet(G4/"regime3_full_1980_2020_g4/truth_all.parquet")
pd.concat([TA, T3[["DAM_ID", "date", "storage_mcm", "removed"]]], ignore_index=True).to_parquet(OUT3/"truth_all.parquet", index=False)
TC = T3[~T3.removed].drop(columns="removed"); TC = TC.merge(Q3[["DAM_ID", "year", "use"]], left_on=["DAM_ID", TC.date.dt.year], right_on=["DAM_ID", "year"], how="left")
TC = TC.drop(columns=[c for c in ("key_1", "year") if c in TC.columns])
FC = pd.read_parquet(G4/"regime3_full_1980_2020_g4/truth_clean.parquet")
pd.concat([FC, TC[["DAM_ID", "date", "storage_mcm", "use"]]], ignore_index=True).to_parquet(OUT3/"truth_clean.parquet", index=False)
TC[["DAM_ID", "date", "storage_mcm", "use", "source", "sid"]].to_parquet(OUT3/"truth_new_detail.parquet", index=False)
FQ = pd.read_csv(G4/"regime3_full_1980_2020_g4/qa_site_year.csv")
pd.concat([FQ, Q3[Q3.DAM_ID.isin(S3.DAM_ID)]], ignore_index=True).to_csv(OUT3/"qa_site_year.csv", index=False)
pn = pd.read_csv(G4/"regime3_full_1980_2020_g4/predict_nodes.csv", dtype=str)
add = pd.DataFrame([(b, m) for b, ms in zip(S3.basin, S3.members) for m in str(ms).split(";")], columns=["basin", "subbasin"])
pd.concat([pn, add]).drop_duplicates().to_csv(OUT3/"predict_nodes.csv", index=False)
pd.DataFrame(units).to_csv(OUT3/"unit_runs_new.csv", index=False)
# ---------------- regime 1x plus
S1 = pd.DataFrame(rows1); T1 = pd.concat(tr1, ignore_index=True) if tr1 else pd.DataFrame()
F1 = pd.read_csv(G4/"regime1x_2023_2026_g4/sites.csv", dtype={"node": str, "members": str, "basin": str}); F1["added_by"] = "frozen_g4"
CA = pd.read_csv(F/"testsets/regime1x_2023_2026/candidate_additions/sites.csv", dtype={"node": str, "members": str, "basin": str})
CA = CA[CA.ext_status == "extended"].copy(); CA["added_by"] = "candidate_additions"
S1 = S1[~S1.DAM_ID.isin(set(CA.DAM_ID) | set(F1.DAM_ID))]
assert not set(S1.node) & set(F1.node) and not set(CA.node) & set(F1.node)
pd.concat([F1, CA, S1], ignore_index=True).to_csv(OUT1/"sites.csv", index=False)
FT1 = pd.read_parquet(G4/"regime1x_2023_2026_g4/truth_clean_v1x.parquet"); CAT = pd.read_parquet(F/"testsets/regime1x_2023_2026/candidate_additions/truth_clean.parquet")
T1 = T1[T1.DAM_ID.isin(S1.DAM_ID)]
pd.concat([FT1, CAT, T1[~T1.removed].drop(columns="removed")], ignore_index=True).to_parquet(OUT1/"truth_clean_v1x.parquet", index=False)
pd.concat([pd.read_parquet(G4/"regime1x_2023_2026_g4/truth.parquet"), pd.read_parquet(F/"testsets/regime1x_2023_2026/candidate_additions/truth.parquet"),
           T1.drop(columns="removed")], ignore_index=True).to_parquet(OUT1/"truth.parquet", index=False)
pn = pd.read_csv(G4/"regime1x_2023_2026_g4/predict_nodes.csv", dtype=str)
add = pd.DataFrame([(b, m) for b, ms in zip(pd.concat([CA.basin, S1.basin]), pd.concat([CA.members, S1.members])) for m in str(ms).split(";")], columns=["basin", "subbasin"])
pd.concat([pn, add]).drop_duplicates().to_csv(OUT1/"predict_nodes.csv", index=False)
pd.DataFrame(glog).to_csv(X/"testsets/build_log_dams.csv", index=False)
# ---------------- version + summary
hm = lambda s: s.isin(["High", "Medium"])
def h(p):
    """Return the MD5 hash (hex digits) of the file at path p."""
    return hashlib.md5(open(p, "rb").read()).hexdigest()
for o, base_, S_, fz in ((OUT3, "eval/g4_fix/testsets/regime3_full_1980_2020_g4", S3, F3), (OUT1, "eval/g4_fix/testsets/regime1x_2023_2026_g4 + candidate_additions", pd.concat([CA, S1]), F1)):
    v = dict(built=pd.Timestamp.now().isoformat(), base=base_, script="gauges_expand/build_plus.py", frozen_sites=int(len(fz)), frozen_HM=int(hm(fz.site_tier).sum()),
             added_sites=int(len(S_)), added_by_tier=S_.site_tier.value_counts().to_dict(), added_by_source=S_.get("source", S_.get("ext_source")).value_counts().to_dict(),
             files={f.name: h(f) for f in o.iterdir() if f.is_file()})
    json.dump(v, open(o/"version.json", "w"), indent=1, default=str); print(o.name, json.dumps({k: v[k] for k in v if k != "files"}, default=str))
