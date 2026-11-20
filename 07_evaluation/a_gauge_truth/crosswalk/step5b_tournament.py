"""Look for a better gauge series at dams whose 2023-2025 series does not pass the overlap test.

Step 5b of the crosswalk (the "tournament"). A dam enters if its current series did not reach r >= 0.95
with an amplitude ratio of 0.8-1.25 against the dam's ResOpsUS record in step 5. All alternative series
near the dam are scored the same way, and the best one replaces the current series if it is clearly
better. The choice uses agreement with ResOpsUS in 2015-2020 only, never any model output.

Reads (under $SWOTGAT_WORK/CONUS_storage_fork)
  crosswalk/gauge_overlap.csv                    step 5 scores
  catalogs/cwms_unfiltered.csv, catalogs/{nwis,nrcs,rise,codwr}.csv    series catalogs
  obsfilter.py                                   the function is_obs (is a CWMS series an observation?)
  ResOpsUS/                                      dam coordinates and records
  reservoir_truth_2023_2025.csv, resops_targets_full.csv
  crosswalk/dam_node_crosswalk.csv               PLD area of the dam's member nodes
  crosswalk/resops_record_qa.csv                 unit fix factor per dam

Writes
  crosswalk/gauge_tournament.csv                 the current series and every candidate, with scores
  crosswalk/tournament_winners.csv               the replacements
  reservoir_truth_2023_2025_v2.csv               the 2023-2025 table with the winners swapped in
                                                 (the input table is not modified)

Usage
  python step5b_tournament.py        (no arguments; 8 download threads; run after step 5)

Method
  * Candidates per dam: every observation series at the same CWMS location, except rule curves and
    forecasts; a total-storage composite (Stor-Conservation Pool + Stor-Flood Pool of the same type,
    interval and version); the other USGS storage and elevation parameter codes at the same site; and
    every catalogued series within 3 km of the dam.
  * Each candidate is scored against the dam's 2015-2020 ResOpsUS record with step 5's one() (storage
    against storage, elevation against elevation).
  * Winner: the highest r among candidates with at least 180 overlap days and an amplitude ratio of
    0.67-1.5 that step 5 does not label "averaged". Storage is preferred on ties. The winner replaces the
    current series only if its r is at least 0.95, or at least 0.10 above the current r.
  * The winner is then downloaded for 2023-2025 and needs at least 250 days there. A storage series is
    used directly. An elevation series is converted to storage:
      - With the dam's ResOpsUS rating curve, if the 2010-2020 ResOpsUS storage-elevation pairs are
        physical (slope dS/dh / PLD member area in 0.67-1.5 and correlation >= 0.9). The gauge datum is
        aligned with the median gauge - ResOpsUS elevation difference over 2015-2020. The curve is the
        median storage per 0.01 m of elevation, made monotone, and extended linearly at both ends. It
        is accepted if at least 90% of the days lie inside its elevation range and its day-to-day
        variability is 0.5-2 times that of the area estimate below.
      - Otherwise as a storage anomaly, (elevation - median elevation) x PLD member area, marked
        is_anomaly.

Notes
  Downloads use the fetch functions and cache of harvest_v2.py through swotgat.http (generic client
  name, at most one request per 1.1 s per host). The 2015-2020 and 2023-2025 requests use separate
  cache tags. The download window is module state, so the winners are fetched one after another.
"""
from swotgat.paths import WORK
import sys, re, json, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
F = Path(f"{WORK}/CONUS_storage_fork"); O = F/"crosswalk"
sys.path.insert(0, str(O)); sys.path.insert(0, str(F))
import step5_gauge_overlap as S5          # sets H window to 2015-2020 with a separate cache tag
H = S5.H
RULE_VER = re.compile(r"(?i)\bRUL\b|rule|guide|target|fcst|forecast")   # LRH ...Elev.Inst.1Day.0.RUL = rule curve

GO = pd.read_csv(O/"gauge_overlap.csv")
truth = GO[GO.role.str.startswith("truth")].drop_duplicates("DAM_ID")
need = truth[~((truth.status == "ok") & (truth.ovl_r >= 0.95) & truth.amp_ratio.between(0.8, 1.25))]
print(f"dams needing a tournament: {len(need)} of {len(truth)}")

CW = pd.read_csv(F/"catalogs/cwms_unfiltered.csv", dtype=str)
exec(open(F/"obsfilter.py").read().split("U = F")[0])       # is_obs()
CW["loc"] = CW.sid.str.split(".").str[0]
cats = []
for c in ("nwis", "nrcs", "rise", "codwr"):
    k = pd.read_csv(F/f"catalogs/{c}.csv", dtype=str); cats.append(k)
cats.append(CW[CW.sid.map(is_obs)].drop(columns="loc"))
K = pd.concat(cats, ignore_index=True)
K["lat"] = pd.to_numeric(K.lat, errors="coerce"); K["lon"] = pd.to_numeric(K.lon, errors="coerce")
R = pd.read_csv(F/"ResOpsUS/attributes/reservoir_attributes.csv").set_index("DAM_ID")

def candidates(row):
    """Return the sorted list of alternative (source key, series id) pairs for one dam.

    row is the step-5 row of the dam's current series (DAM_ID, source, sid). The list holds: at the same
    CWMS location, every observation series that is not a rule curve or forecast, and the composite
    conservation pool + flood pool where both exist (key CWMSSUM, the two ids joined by "+"); at the
    same USGS site, the storage and elevation parameter codes 00054, 72036, 62614, 62615 and 00062; and
    every catalogued series within 3 km of the dam's ResOpsUS coordinate. The current series is left out.
    """
    out = set()
    sid, src = str(row.sid), row.source
    if src == "CWMS" and "|" in sid:
        loc = sid.split(".")[0]
        grp = CW[CW["loc"] == loc]
        for s in grp.sid:
            if is_obs(s) and not RULE_VER.search(s.split(".")[-1]): out.add(("CWMS", s))
            if "Stor-Conservation Pool" in s:
                f = s.replace("Stor-Conservation Pool", "Stor-Flood Pool")
                if f in set(grp.sid): out.add(("CWMSSUM", s + "+" + f))
    if src in ("NWIS", "NWISANY"):
        site = sid.split("|")[0]
        for code in ("00054", "72036", "62614", "62615", "00062"):
            out.add(("NWISANY", f"{site}|{code}"))
    if row.DAM_ID in R.index:
        la, lo = R.LAT[row.DAM_ID], R.LONG[row.DAM_ID]
        p1, p2 = np.radians(la), np.radians(K.lat.values)
        d = 2*6371*np.arcsin(np.sqrt(np.sin((p2-p1)/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(np.radians(K.lon.values-lo)/2)**2))
        near = K[np.nan_to_num(d, nan=1e9) <= 3.0]
        for _, k in near.iterrows():
            out.add(("NWISANY" if k.source == "NWIS" else k.source, k.sid if k.source != "NWIS" else "|".join(k.sid.split("|")[:2])))
    out.discard((src, sid))
    return sorted(out)

def fetch_sum(sid):
    """Download two CWMS series, whose ids are joined by "+" in sid, and return their sum.

    Returns (sum on the common time steps, unit of the first series), or (None, None) if either
    series is missing or empty.
    """
    a, b = sid.split("+")
    s1, u1 = S5.FETCH["CWMS"](a); s2, u2 = S5.FETCH["CWMS"](b)
    if s1 is None or s2 is None or not len(s1) or not len(s2): return None, None
    j = pd.concat([s1.rename("c"), s2.rename("f")], axis=1).dropna()
    return (j.c + j.f), u1
S5.FETCH["CWMSSUM"] = fetch_sum

def kind_of(src, sid):
    """Return "storage" or "elevation" for a candidate series.

    A USGS series is an elevation for parameter codes 62614, 62615, 00062 and 00065. A CWMS series is an
    elevation if its parameter (second part of the id) starts with "Elev". Any other series takes the
    kind listed in the catalog, and "storage" if the catalog gives neither kind.
    """
    if src == "NWISANY": return "elevation" if sid.split("|")[1] in ("62614", "62615", "00062", "00065") else "storage"
    if src in ("CWMS", "CWMSSUM"):
        p = sid.split("|", 1)[-1].split(".")
        return "elevation" if p[1].startswith("Elev") else "storage"
    k = K[K.sid == sid]
    return (k.kind.iat[0] if len(k) and k.kind.iat[0] in ("storage", "elevation") else "storage")

jobs = []
for _, r in need.iterrows():
    for src, sid in candidates(r):
        jobs.append((int(r.DAM_ID), r.DAM_NAME, src, sid, kind_of(src, sid), "candidate"))
print(f"candidate series: {len(jobs)}", flush=True)
with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(S5.one, jobs))
C = pd.DataFrame(res)
C = pd.concat([need.assign(role="current"), C], ignore_index=True)
C.to_csv(O/"gauge_tournament.csv", index=False)

# ---- winners
win = []
for dam, g in C.groupby("DAM_ID"):
    cur = g[g.role == "current"].iloc[0]
    ok = g[(g.role == "candidate") & (g.status == "ok") & (g.ovl_n >= 180) & g.amp_ratio.between(0.67, 1.5)
           & ~g.cls.fillna("").str.contains("averaged")]
    if ok.empty: continue
    b = ok.sort_values(["ovl_r", "kind"], ascending=[False, False]).iloc[0]     # prefer storage on ties
    cur_r = cur.ovl_r if cur.status == "ok" else -1
    if b.ovl_r >= 0.95 or b.ovl_r >= cur_r + 0.10:
        win.append(dict(DAM_ID=dam, DAM_NAME=cur.DAM_NAME, old_source=cur.source, old_sid=cur.sid, old_r=cur_r,
                        source=b.source, sid=b.sid, kind=b.kind, ovl_r=b.ovl_r, amp_ratio=b.amp_ratio, level_ratio=b.level_ratio))
W = pd.DataFrame(win); W.to_csv(O/"tournament_winners.csv", index=False)
print(f"winners replacing the current series: {len(W)}")
print(W.round(3).to_string(max_colwidth=55) if len(W) else "")

# ---- fetch winners for 2023-2025 (sequential: window is module state) and swap into the v2 table
H.START, H.END = "2023-01-01", "2025-12-31"
S5.LO, S5.HI = "2023-01-01", "2025-12-31"
H.get = lambda url, tag, headers=None, ttl=30: S5._get(url, tag + "_w2325", headers, ttl)   # own cache tag for 2023-25, so a response cached under the plain tag is never reused
TT = pd.read_csv(F/"reservoir_truth_2023_2025.csv", dtype={"node": str, "node_orig": str})
TGT = pd.read_csv(F/"resops_targets_full.csv", dtype={"node": str}).drop_duplicates("node").set_index("node")
TT["DAM_ID"] = TT.node_orig.map(TGT.DAM_ID)
XW = pd.read_csv(O/"dam_node_crosswalk.csv", dtype={"node": str}).set_index("DAM_ID")
VER = pd.read_csv(O/"resops_record_qa.csv").set_index("DAM_ID")
new = []
for _, w in W.iterrows():
    s, u = S5.FETCH[w.source](w.sid)
    if s is None or not len(s): print("  no 2023-25 data:", w.DAM_NAME, w.sid); continue
    if w.source in ("NWIS", "NWISANY"): u = u or S5.NWIS_UNIT.get(w.sid.split("|")[1])
    s = s[~s.index.duplicated()].sort_index(); s.index = pd.to_datetime(s.index)
    s = s.resample("D").mean().dropna().loc["2023":"2025"]
    v = S5.clean(H.to_storage_units(s, u, w.kind))
    if w.kind == "elevation":
        area = XW.members_area.get(w.DAM_ID, np.nan)
        rr = pd.read_csv(F/f"ResOpsUS/time_series_all/ResOpsUS_{w.DAM_ID}.csv"); rr["date"] = pd.to_datetime(rr.date, errors="coerce")
        rr = rr.dropna(subset=["date"]).drop_duplicates("date").set_index("date").apply(pd.to_numeric, errors="coerce").loc["2010":"2020"]
        pair = rr[["elevation", "storage"]].dropna()
        pair["storage"] *= VER.fix_factor.get(w.DAM_ID, 1.0) if "fix_factor" in VER else 1.0
        dsdh = np.polyfit(pair.elevation, pair.storage, 1)[0] if len(pair) > 200 and pair.elevation.std() > 0.05 else np.nan
        phys = area > 0 and pd.notna(dsdh) and 0.67 <= dsdh/area <= 1.5 and pair.storage.corr(pair.elevation) >= 0.9
        if phys:
            # datum offset gauge - ResOps from the 2015-20 overlap (cached fetch)
            S5.LO, S5.HI = "2015-01-01", "2020-12-31"; H.START, H.END = S5.LO, S5.HI; H.get = lambda url, tag, headers=None, ttl=30: S5._get(url, tag + "_ovl1520", headers, ttl)
            o, uo = S5.FETCH[w.source](w.sid)
            S5.LO, S5.HI = "2023-01-01", "2025-12-31"; H.START, H.END = S5.LO, S5.HI; H.get = lambda url, tag, headers=None, ttl=30: S5._get(url, tag + "_w2325", headers, ttl)
            if w.source in ("NWIS", "NWISANY"): uo = uo or S5.NWIS_UNIT.get(w.sid.split("|")[1])
            o = H.to_storage_units(o.resample("D").mean().dropna(), uo, "elevation")
            off = (o - rr.elevation).dropna().median()
            b = pair.sort_values("elevation"); b = b.groupby(b.elevation.round(2)).storage.median()
            x, y = b.index.values, np.maximum.accumulate(b.values)
            k = min(5, len(x) - 1); slo = (y[k]-y[0])/max(x[k]-x[0], 1e-6); shi = (y[-1]-y[-1-k])/max(x[-1]-x[-1-k], 1e-6)
            e = v.values - off
            st = np.interp(e, x, y); st = np.where(e < x[0], y[0] + (e-x[0])*slo, st); st = np.where(e > x[-1], y[-1] + (e-x[-1])*shi, st)
            cover = float(np.mean((e >= x[0]) & (e <= x[-1])))
            area_est = (v - v.median())*area
            rat = pd.Series(st, index=v.index)
            var_ratio = rat.diff().std()/area_est.diff().std() if area_est.diff().std() > 0 else np.nan
            if cover >= 0.9 and 0.5 <= var_ratio <= 2.0:
                v = rat; meth, anom = f"tournament: elevation -> ResOps rating curve (datum offset {off:+.2f} m, dS/dh/area {dsdh/area:.2f}, coverage {cover:.0%})", False
            else:
                v = area_est; meth, anom = (f"tournament: elevation x PLD member area ({area:.1f} km2; rating rejected: coverage {cover:.0%}, "
                                            f"variability ratio {var_ratio:.2f})"), True
        else:
            if not area > 0: print("  no area for elevation winner:", w.DAM_NAME); continue
            v = (v - v.median())*area; meth, anom = f"tournament: elevation x PLD member area ({area:.1f} km2; ResOps rating not physical)", True
    else: meth, anom = "tournament: direct storage", False
    if len(v) < 250: print("  short 2023-25:", w.DAM_NAME, len(v)); continue
    base = TT[TT.DAM_ID == w.DAM_ID].iloc[0]
    new.append(pd.DataFrame(dict(node=base.node, node_orig=base.node_orig, date=v.index.strftime("%Y-%m-%d"), storage_mcm=v.values,
                                 is_anomaly=anom, source=f"{w.source}:{w.kind}", sid=w.sid,
                                 method=f"{meth}; overlap r={w.ovl_r:.3f} amp={w.amp_ratio:.2f}", **{"pass": "tournament"}, DAM_ID=w.DAM_ID)))
N = pd.concat(new, ignore_index=True) if new else TT.iloc[:0]
T2 = pd.concat([TT[~TT.DAM_ID.isin(N.DAM_ID)], N], ignore_index=True).drop(columns="DAM_ID")
T2.to_csv(F/"reservoir_truth_2023_2025_v2.csv", index=False)
print(f"truth v2: {T2.node_orig.nunique()} reservoirs; swapped series for {N.DAM_ID.nunique() if len(N) else 0}")
