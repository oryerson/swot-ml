"""Test that each agency gauge series belongs to its dam, by comparing it with the dam's ResOpsUS record.

Step 5 of the crosswalk (link L4, gauge to dam). ResOpsUS ends in 2020 and the agency series are the
2023-2025 reference ("truth"), so the two cannot be compared in the test period. Instead, every agency
series is downloaded again for 2015-2020 and correlated day by day with the dam's own ResOpsUS record
(storage against storage, elevation against elevation). A series from a different reservoir cannot
track the dam's daily record at r >= 0.95, and the amplitude and level ratios expose wrong units.

Reads (under $SWOTGAT_WORK/CONUS_storage_fork)
  reservoir_truth_summary.csv, reservoir_truth_2023_2025.csv   the 2023-2025 series per reservoir
  resops_targets_full.csv                                      node -> ResOpsUS DAM_ID
  crosswalk/legacy_rederive_check.csv, harvested_legacyre.csv, harvested_legacyTWDB.csv,
  harvested_legacyCDEC.csv                                     recovered series ids (see Method)
  crosswalk/resops_record_qa.csv                               unit fix factor per dam (step 1)
  ResOpsUS/time_series_all/ResOpsUS_<DAM_ID>.csv

Writes
  crosswalk/gauge_overlap.csv      one row per dam and series

Usage
  python step5_gauge_overlap.py        (no arguments; 8 download threads)
  step5b_tournament.py imports this module for its fetch functions and for one().

Method
  * Series. Each 2023-2025 series is identified by its source and series id (sid). Rows from an earlier
    download (pass "agency_storage_2023_2025_union") did not store a sid. For those, the sid found by a
    later re-harvest is used if that re-harvest reproduces the stored values (r >= 0.99 and ratio
    0.95-1.05 in legacy_rederive_check.csv); otherwise the sid of a re-harvest restricted to the
    original source (Texas or California service); otherwise the series cannot be tested. A re-harvest
    that does not reproduce the stored values is still tested, with role "alternative".
  * Download. Fetch functions come from harvest_v2.py, plus five defined here (Colorado telemetry,
    any USGS parameter code, Reclamation Lower Colorado and Hydromet, NRCS monthly). The window is
    2015-01-01 to 2020-12-31. Responses are cached under tags ending in "_ovl1520", so they never mix
    with the cached 2023-2025 responses. Requests go through swotgat.http (generic client name, at most
    one request per 1.1 s per host).
  * Comparison. The series is averaged to days and converted to million cubic meters or meters. The
    ResOpsUS storage is multiplied by the dam's fix factor. NRCS monthly series are compared at month
    ends. Both series are cleaned (see clean). The score is the best Pearson r over lags of -2 to +2
    days, on at least 60 common days. amp_ratio is the ratio of standard deviations and level_ratio the
    ratio of medians (series / ResOpsUS), both at zero lag.
  * Class. A series is labelled averaged (a CWMS average over a duration other than one day),
    conservation-pool, raw, or monthly. top_pinned_frac is the share of days at the series maximum
    (within 0.1%); a high value marks a series truncated at the top of the conservation pool.

Notes
  The module changes the download window and the cache tag of harvest_v2 when it is imported.
"""
from swotgat.paths import WORK
import sys, io, json, re, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
F = Path(f"{WORK}/CONUS_storage_fork"); O = F/"crosswalk"; sys.path.insert(0, str(F))
import harvest_v2 as H
LO, HI = "2015-01-01", "2020-12-31"
H.START, H.END = LO, HI
_get = H.get
H.get = lambda url, tag, headers=None, ttl=30: _get(url, tag + "_ovl1520", headers, ttl)   # own cache tag: never reuse 2023-25 responses

def fetch_codwr(sid):
    """Download the 2015-2020 daily storage of a Colorado telemetry station (abbreviation sid).

    The dates are fixed in the request; the window LO to HI is not used. Returns (series indexed by date,
    "af"), or (None, None) if the response cannot be read.
    """
    t = H.get("https://dwr.state.co.us/Rest/GET/api/v2/telemetrystations/telemetrytimeseriesday/?format=json"
              f"&abbrev={sid}&parameter=STORAGE&startDate=01/01/2015&endDate=12/31/2020&pageSize=5000", f"codwr_{sid}")
    try:
        L = json.loads(t).get("ResultList", [])
        return pd.Series({pd.Timestamp(x["measDate"]): x.get("measValue") for x in L}, dtype=float).dropna(), "af"
    except Exception: return None, None
def fetch_nwis_any(sid):
    """Download the USGS daily values of any parameter code in the window LO to HI; sid is "site|code".

    Returns (series indexed by date, None): the caller sets the unit from the parameter code. Returns
    (None, None) if the response holds no data rows or no value column for the code.
    """
    site, code = sid.split("|")[:2]
    t = H.get(f"https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites={site}&parameterCd={code}&startDT={LO}&endDT={HI}", f"nwisany_{site}_{code}")
    L = [l.split("\t") for l in t.splitlines() if l and not l.startswith("#")]
    if len(L) < 3: return None, None
    col = next((i for i, c in enumerate(L[0]) if f"_{code}_" in c and not c.endswith("_cd")), None)
    if col is None: return None, None
    return pd.Series({pd.Timestamp(r[2]): pd.to_numeric(r[col], errors="coerce") for r in L[2:] if len(r) > col}, dtype=float).dropna(), None
def fetch_lchdb(sdi):
    """Download a daily series of the Reclamation Lower Colorado database in the window LO to HI.

    sdi is the site-datatype id of the series. Returns (series indexed by date, "ac-ft"), or (None, None)
    if the response cannot be read.
    """
    t = H.get(f"https://www.usbr.gov/pn-bin/hdb/hdb.pl?svr=lchdb&sdi={sdi}&tstp=DY&t1={LO}T00:00&t2={HI}T00:00&table=R&mrid=0&format=json", f"lchdb_{sdi}")
    try:
        d = json.loads(t)["Series"][0]["Data"]
        return pd.Series({pd.Timestamp(x["t"]).normalize(): pd.to_numeric(x["v"], errors="coerce") for x in d}, dtype=float).dropna(), "ac-ft"
    except Exception: return None, None
def fetch_nrcs_monthly(sid):
    """Download the monthly reservoir storage of an NRCS station (triplet sid) in the window LO to HI.

    Each value is dated at the end of its month. Returns (series, "ac-ft"), or (None, None) if the
    response cannot be read.
    """
    t = H.get("https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data?stationTriplets=%s&elements=RESC"
              "&duration=MONTHLY&beginDate=%s&endDate=%s" % (sid, LO, HI), f"nrcs_m_{sid}")
    try:
        v = json.loads(t)[0]["data"][0]["values"]
        return pd.Series({pd.Timestamp(year=x["year"], month=x["month"], day=1) + pd.offsets.MonthEnd(0): x.get("value") for x in v}, dtype=float).dropna(), "ac-ft"
    except Exception: return None, None
def fetch_hydromet(code):
    """Download the 2015-2020 daily storage of a Reclamation Hydromet station (station code).

    The dates are fixed in the request; the window LO to HI is not used. Returns (series indexed by date,
    "ac-ft"), or (None, None) if the response cannot be read.
    """
    t = H.get(f"https://www.usbr.gov/pn-bin/daily.pl?parameter={code}%20af&syer=2015&smnth=1&sdy=1&eyer=2020&emnth=12&edy=31&format=csv", f"hydromet_{code}")
    try:
        d = pd.read_csv(io.StringIO(t)); d.columns = ["d", "v"] + list(d.columns[2:])
        return pd.Series(pd.to_numeric(d.v, errors="coerce").values, index=pd.to_datetime(d.d, errors="coerce")).dropna(), "ac-ft"
    except Exception: return None, None
FETCH = dict(H.FETCH, CODWR=fetch_codwr, NWISANY=fetch_nwis_any, LCHDB=fetch_lchdb, NRCSM=fetch_nrcs_monthly, HYDROMET=fetch_hydromet)
NWIS_UNIT = {"00054": "ac-ft", "72036": "kaf", "62614": "ft", "62615": "ft", "00062": "ft", "00065": "ft"}

def series_class(src, sid):
    """Return the class labels of a series, joined by ";" ("" if none applies).

    src is the source key and sid the series id. A CWMS id is labelled averaged(<duration>) if its
    third part (the type) starts with "ave" and its fifth part (the duration) is not one day,
    conservation-pool if it names the conservation pool, and raw if its last part contains "raw". An
    NRCS monthly series is labelled monthly.
    """
    s = str(sid)
    c = []
    if src == "CWMS":
        p = s.split("|", 1)[-1].split(".")
        if len(p) >= 5 and p[2].lower().startswith("ave") and p[4] not in ("0", "1Day", "~1Day"): c.append(f"averaged({p[4]})")
        if re.search(r"(?i)conservation pool|stor-con\b|stor-cons", s): c.append("conservation-pool")
        if re.search(r"(?i)raw", s.split(".")[-1]): c.append("raw")
    if src in ("NRCSM",): c.append("monthly")
    return ";".join(c)

def resops(dam_id):
    """Return the ResOpsUS table of a dam in the window LO to HI, numeric and indexed by date."""
    t = pd.read_csv(F/f"ResOpsUS/time_series_all/ResOpsUS_{dam_id}.csv"); t["date"] = pd.to_datetime(t.date, errors="coerce")
    t = t.dropna(subset=["date"]).set_index("date").apply(pd.to_numeric, errors="coerce")
    t = t[~t.index.duplicated()].sort_index().loc[LO:HI]
    return t

def clean(s):
    """Remove dropouts and isolated spikes, with the rules of merge_truth.py and step7_build_testsets.py.

    Exact zeros are removed where the median exceeds 1. A day is removed if it departs from the 5-day
    centered median by more than 10 robust scales (1.4826 x the median absolute day-to-day change plus
    0.1% of the range). Series shorter than 10 values are returned unchanged.
    """
    s = s.astype(float).dropna().copy()
    if len(s) < 10: return s
    if s.median() > 1: s[s == 0] = np.nan
    m = s.rolling(5, center=True, min_periods=3).median()
    scale = 1.4826*np.nanmedian(np.abs(np.diff(s.dropna().values))) + 1e-3*(s.max() - s.min())
    s[(s - m).abs() > 10*scale] = np.nan
    return s.dropna()

def overlap(a, b):
    """Return (r, lag, n, amp, lvl) for series a against reference b, both cleaned first.

    r is the best Pearson r over lags of -2 to +2 days (at least 60 common days), with its lag and
    number of days. amp is sd(a)/sd(b) and lvl is median(a)/median(b), both at zero lag and on more
    than 60 common days.
    """
    a, b = clean(a), clean(b)
    best = (np.nan, 0, 0)
    for lag in range(-2, 3):
        j = pd.concat([a.shift(lag, freq="D").rename("a"), b.rename("b")], axis=1).dropna()
        if len(j) < 60 or j.a.std() == 0 or j.b.std() == 0: continue
        r = j.a.corr(j.b)
        if np.isnan(best[0]) or r > best[0]: best = (r, lag, len(j))
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    amp = j.a.std()/j.b.std() if len(j) > 60 and j.b.std() > 0 else np.nan
    lvl = j.a.median()/j.b.median() if len(j) > 60 and j.b.median() not in (0, np.nan) else np.nan
    return best[0], best[1], best[2], amp, lvl

QA = pd.read_csv(O/"resops_record_qa.csv").set_index("DAM_ID")
def one(job):
    """Download one agency series for 2015-2020 and compare it with the dam's ResOpsUS record.

    job is (DAM_ID, dam name, source key, series id, kind, role), where kind is "storage" or "elevation".
    The series is averaged to days (NRCS monthly series stay monthly), converted to million cubic meters
    or meters, and scored against the ResOpsUS storage (times the dam's fix factor) or elevation with
    overlap(). Returns a dict with the job fields, the series class, status, ovl_r, ovl_lag, ovl_n,
    amp_ratio, level_ratio, top_pinned_frac (share of days within 0.1% of the series maximum), the number
    of values and the unit. status says when there is no 2015-2020 data or no overlap.
    """
    dam, name, src, sid, kind, role = job
    out = dict(DAM_ID=dam, DAM_NAME=name, source=src, sid=sid, kind=kind, role=role, cls=series_class(src, sid))
    try:
        s, u = FETCH[src](sid)
    except Exception as e:
        s, u = None, None; out["err"] = f"{type(e).__name__}"
    if s is None or len(s) == 0: out["status"] = "no 2015-2020 data"; return out
    if src == "NWIS" or src == "NWISANY": u = u or NWIS_UNIT.get(sid.split("|")[1], None)
    s = s[~s.index.duplicated()].sort_index()
    s.index = pd.to_datetime(s.index).tz_localize(None) if getattr(s.index, "tz", None) is not None else pd.to_datetime(s.index)
    s = s.resample("D").mean().dropna() if src != "NRCSM" else s
    v = H.to_storage_units(s, u, "storage" if kind == "storage" else "elevation")
    R = resops(dam)
    ref = R.storage*QA.fix_factor.get(dam, 1.0) if kind == "storage" else R.elevation
    ref = ref.dropna()
    if src == "NRCSM": ref = ref.resample("ME").last(); v = v.resample("ME").last()
    r, lag, n, amp, lvl = overlap(v, ref)
    # truncation: share of days pinned at the series maximum (conservation-pool caps)
    top = (v >= v.max() - 1e-3*max(abs(v.max()), 1)).mean() if len(v) else np.nan
    out.update(status="ok" if n else "no overlap with ResOps", ovl_n=n, ovl_r=r, ovl_lag=lag, amp_ratio=amp, level_ratio=lvl,
               top_pinned_frac=top, n_2015_2020=len(v), unit=u)
    return out

if __name__ == "__main__":
    S = pd.read_csv(F/"reservoir_truth_summary.csv", dtype={"node": str, "node_orig": str})
    T = pd.read_csv(F/"resops_targets_full.csv", dtype={"node": str}).drop_duplicates("node").set_index("node")
    S["DAM_ID"] = S.node_orig.map(T.DAM_ID)
    T2 = pd.read_csv(F/"reservoir_truth_2023_2025.csv", dtype={"node_orig": str}, usecols=["node_orig", "sid", "source"]).drop_duplicates("node_orig").set_index("node_orig")
    S["sid"] = S.node_orig.map(T2.sid)
    # series ids of the earlier-download rows, recovered by a re-harvest that reproduces the stored values
    # (r >= 0.99) or by a re-harvest restricted to the original source
    chk = pd.read_csv(O/"legacy_rederive_check.csv", dtype={"node": str}).set_index("node")
    lre = pd.read_csv(F/"harvested_legacyre.csv", dtype={"node": str}).drop_duplicates("node").set_index("node")
    alt = pd.concat([pd.read_csv(F/f, dtype={"node": str}).drop_duplicates("node") for f in ("harvested_legacyTWDB.csv", "harvested_legacyCDEC.csv")]).set_index("node")
    jobs = []
    for _, r in S.iterrows():
        if pd.isna(r.DAM_ID): continue
        dam = int(r.DAM_ID); src_full = str(r.source)
        if r.pass_ == "agency_storage_2023_2025_union":
            if src_full.startswith("HYDROMET:"):
                jobs.append((dam, r.DAM_NAME, "HYDROMET", src_full.split(":")[1], "storage", "truth")); continue
            n = r.node_orig
            if n in chk.index and chk.r.get(n, 0) >= .99 and .95 <= chk.ratio.get(n, 0) <= 1.05:
                jobs.append((dam, r.DAM_NAME, lre.source[n].split(":")[0], lre.sid[n], lre.source[n].split(":")[1] if ":" in lre.source[n] else "storage", "truth(legacy sid recovered)"))
            elif n in alt.index:
                jobs.append((dam, r.DAM_NAME, alt.source[n].split(":")[0], alt.sid[n], "storage", "truth(legacy sid by source re-harvest)"))
            else:
                jobs.append((dam, r.DAM_NAME, src_full, "", "storage", "truth(legacy sid unknown)"))
            if n in lre.index and not (chk.r.get(n, 0) >= .99):   # also test the re-harvest alternative
                jobs.append((dam, r.DAM_NAME, lre.source[n].split(":")[0], lre.sid[n], "storage", "alternative"))
        else:
            src, kind = (src_full.split(":") + ["storage"])[:2]
            if kind not in ("storage", "elevation"): kind = "storage"
            jobs.append((dam, r.DAM_NAME, src, r.sid, kind, "truth"))
    jobs = [j for j in jobs if j[3]]
    print(f"series to test: {len(jobs)} for {len(set(j[0] for j in jobs))} dams", flush=True)
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, jobs))
    X = pd.DataFrame(res); X.to_csv(O/"gauge_overlap.csv", index=False)
    t = X[X.role.str.startswith("truth")]
    print(t.status.value_counts().to_string())
    ok = t[t.status == "ok"]
    print(f"with overlap: {len(ok)} | r>=0.95: {(ok.ovl_r>=.95).sum()} | 0.8-0.95: {ok.ovl_r.between(.8,.95,inclusive='left').sum()} | <0.8: {(ok.ovl_r<.8).sum()}")
    print(f"storage series amp ratio in 0.8-1.25: {ok[ok.kind=='storage'].amp_ratio.between(.8,1.25).mean():.0%}")
