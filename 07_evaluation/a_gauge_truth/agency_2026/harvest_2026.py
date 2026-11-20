"""Download 2026 agency gauge data and map it onto the frozen 2023-2025 reservoir test set.

"Regime 1" is the reservoir test set of agency gauge storage for 2023-2025
(testsets/regime1_2023_2025). It was frozen before any model was scored. This script extends
each of its dams through 2026-09-30, the end of the SWOT Period, without changing any existing
value. build_regime1x.py then assembles the extended test set.

Reads ($SWOTGAT_WORK/CONUS_storage_fork)
  testsets/regime1_2023_2025/truth.parquet, sites.csv   the frozen test set
  crosswalk/gauge_final.csv        the validated gauge series of each dam
  crosswalk/gauge_tournament.csv   all gauge series that were compared with ResOpsUS 2015-2020
  gauges2026/extra_candidates.csv, extra_manual.csv     further candidate series (optional;
                                   columns DAM_ID, source, sid, kind)
  with --plus: testsets/regime2_2015_2020/truth.parquet, testsets/regime3_full_1980_2020/
               sites.csv, gauges2026/plus_dams.csv, gauges2026/extra_plus.csv

Writes ($SWOTGAT_WORK/CONUS_storage_fork/gauges2026)
  harvest_2026<suffix>.csv   one row per dam: status (extended | not extended | error), grade,
                             chosen series, overlap statistics, days added, everything tried
  ext_2026<suffix>.parquet   DAM_ID, date, storage_mcm, native (value in the agency's unit),
                             bad_frac, flag, extrapolated, source, sid, map, grade, ovl;
                             2026-01-01 to 2026-09-30
  cache/<tag>.txt            every HTTP response. Cached files never expire.

Usage
  python harvest_2026.py [--dams 1615,1585] [--workers 8] [--crosscheck]
                         [--start 2023-01-01 --ov0 2023-01-01 --suffix _x23] [--plus]
  --dams        comma-separated DAM_IDs (default: all sites)
  --crosscheck  also compare the 2026 values with an accepted series from another source
  --start, --ov0, --suffix   for dams whose record ends before mid-2024 and so has no 120-day
                overlap in 2024-2025: fetch from 2023 and use a 2023-2025 overlap. Cache tags
                are then prefixed with the start date, so responses of the two windows stay
                apart.
  --plus        dams of the 1980-2020 test set that are not in regime 1 (plus_dams.csv). The
                reference is their ResOpsUS storage of 2015-2020, candidates come only from
                extra_plus.csv, the overlap is 2015-2020, and the output covers 2023-01-01 to
                2026-09-30 (suffix _plus).

Method
  Candidates per dam, in this order:
    1. the series its 2023-2025 values were built from (source and sid in truth.parquet);
    2. its series in gauge_final.csv (some early rows of the test set carry no sid);
    3. other series of gauge_tournament.csv with status ok and correlation of at least 0.9
       with ResOpsUS, best first;
    4. series listed in extra_candidates.csv and extra_manual.csv.
  Each candidate is fetched again from its agency for 2024-01-01 to 2026-10-15, with the same
  services and daily aggregation as harvest_v2.py: USACE CWMS (means over UTC days), USGS NWIS
  daily or instantaneous values, NRCS reservoir storage, California CDEC sensor 15, the Texas
  Water Development Board per-reservoir tables, Reclamation RISE, Hydromet and Lower Colorado
  database, and Colorado DWR. A separate cache is used because the tags of the 2023-2025 cache
  carry no dates and would return the old responses.

  The fetched series, in its native unit, is then mapped onto the dam's existing values
  (million cubic meters) over their 2024-2025 overlap. The 2026 values therefore carry the
  same unit, storage-level curve, datum and anomaly convention as the frozen 2023-2025 values.
    storage    y = a*x + b by ordinary least squares.
    elevation  y = f(x), a piecewise-linear curve through the median storage of up to 200
               level bins, forced to be non-decreasing. Levels outside the overlap range are
               extrapolated with the end slopes and flagged `extrapolated`.
    lag        shifts of -1, 0 and +1 day are tried (agencies stamp once-daily values
               differently, and local days differ from UTC days); the best is kept.
  The overlap must have at least 120 days (12 months for monthly series). Acceptance, grade A
  (grade B in parentheses; B is used only when no candidate of the dam reaches A):
    storage    Pearson r >= 0.995 (0.99) and RMS residual of the linear map <= 1% (2.5%) of
               the range of the overlap values.
    elevation  A storage-level curve is monotone but not linear, and it is flat where storage
               is capped at the top of a pool, so a correlation is the wrong test. Instead:
               the RMS residual of the best monotone (isotonic) fit must be <= 1% (2.5%) of
               the range, and so must a leave-one-out residual, in which each overlap day is
               predicted from its two neighbouring levels (a curve through every level would
               fit its own points exactly). Spearman rho >= 0.5 is a minimum sanity check
               only, because ties from a capped or empty pool lower rho.
  The first grade-A candidate is the dam's 2026 series. If it does not cover all of January to
  September 2026, later grade-A candidates fill only the missing days. See `one`.

Notes
  Downloads go through swotgat.http: a generic client name and at most one request per 1.1 s
  per host, however many workers are used.
  Nothing in testsets/regime1_2023_2025 is modified.
  Agencies revise their records, so a new download will differ from the paper's.
  The Colorado DWR request has fixed dates (2024-01-01 to 2026-10-15) and the Hydromet request
  a fixed end date; --start does not move them.
"""
from swotgat.paths import WORK
import io, json, re, sys, time, argparse, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests

F = Path(f"{WORK}/CONUS_storage_fork"); OUT = F/"gauges2026"; CACHE = OUT/"cache"
R1 = F/"testsets/regime1_2023_2025"
START, END = "2024-01-01", "2026-10-15"; OV0, OV1 = "2024-01-01", "2025-12-31"; E0, E1 = "2026-01-01", "2026-09-30"
from swotgat.http import PoliteSession; SES = PoliteSession()

def get(url, tag, headers=None):
    """Fetch a URL through the file cache and return the response text.

    `tag` names the cache file; it is prefixed with the fetch start date when that is not
    2024-01-01. A cached file is always reused. Returns "" after a 400/404 (also cached) or four
    failed tries.
    """
    if START != "2024-01-01": tag = f"s{START}_" + tag                     # cache tags carry no dates; keep other fetch windows apart
    f = CACHE/(re.sub(r"[^A-Za-z0-9_.-]", "_", tag)[:180] + ".txt")
    if f.is_file(): return f.read_text()
    for a in range(4):
        try:
            r = SES.get(url, timeout=180, headers=headers)
            if r.status_code == 200: f.write_text(r.text); return r.text
            if r.status_code in (400, 404): f.write_text(""); return ""
        except Exception: pass
        time.sleep(4*(a + 1))
    return ""

def daily(s):
    """Return the mean per calendar day of the index (UTC days, as in harvest_v2.py).

    Returns None if the series has no value.
    """
    s = s.dropna()
    return s.groupby(s.index.normalize()).mean() if len(s) else None

# ---- fetchers: return DataFrame(date index; v, bad_frac, flag) in native units -------------------------------------------
# v = daily value; bad_frac = share of that day's raw values the agency marks as bad (CWMS: quality
# code questionable, rejected or missing; NWIS: 1 if a value carries the qualifier Eqp, equipment
# malfunction; other sources: 0); flag = the agency's qualifier text (NWIS only). None if no data.
def f_cwms(sid):
    """Fetch a USACE CWMS time series (`sid` = "office|series name") as means over UTC days.

    bad_frac is the share of the day's values whose quality code marks them as questionable,
    rejected or missing.
    """
    off, ts = sid.split("|", 1)
    t = get("https://cwms-data.usace.army.mil/cwms-data/timeseries?office=%s&name=%s&begin=%sT00:00:00Z&end=%sT00:00:00Z&page-size=2000000"
            % (off, requests.utils.quote(ts, safe=""), START, END), f"cwms_{off}_{ts}")
    j = json.loads(t); rows = [(pd.to_datetime(v[0], unit="ms"), v[1], v[2] if len(v) > 2 else 0) for v in j.get("values", []) if v[1] is not None]
    if not rows: return None
    d = pd.DataFrame(rows, columns=["ts", "v", "q"]); d["bad"] = ((d.q.fillna(0).astype(int) & 30).isin([4, 8, 16]))
    d["day"] = d.ts.dt.normalize(); g = d.groupby("day")
    return pd.DataFrame({"v": g.v.mean(), "bad_frac": g.bad.mean(), "flag": ""})
def f_cwmssum(sid):
    """Fetch the sum of several CWMS series, whose ids are joined by "+" in `sid`.

    Returns None if any of them has no data. bad_frac is the largest of the parts' values.
    """
    parts = [f_cwms(s) for s in sid.split("+")]
    if any(p is None for p in parts): return None
    v = sum(p.v for p in parts).dropna(); return pd.DataFrame({"v": v, "bad_frac": pd.concat([p.bad_frac for p in parts], axis=1).max(axis=1).reindex(v.index), "flag": ""})
def f_nwis(sid):
    """Fetch a USGS NWIS series (`sid` = "site|parameter|dv or iv", dv if omitted) as daily means.

    Values at or below -9e5 (missing-data codes) are dropped. flag holds the day's qualifiers and
    bad_frac is 1 when they include Eqp (equipment malfunction).
    """
    site, pc, dt = (sid.split("|") + ["dv"])[:3]
    svc = "dv" if dt == "dv" else "iv"
    t = get(f"https://waterservices.usgs.gov/nwis/{svc}/?format=json&sites={site}&parameterCd={pc}&startDT={START}&endDT={END}", f"nwis_{svc}_{site}_{pc}")
    j = json.loads(t); out, fl = {}, {}
    for ts in j["value"]["timeSeries"]:
        for blk in ts["values"]:
            for v in blk["value"]:
                x = float(v["value"])
                if x > -9e5:
                    k = pd.Timestamp(v["dateTime"]).tz_localize(None); out[k] = x; fl[k] = ",".join(v.get("qualifiers", []))
    if not out: return None
    s = pd.Series(out, dtype=float); q = pd.Series(fl)
    v = daily(s); f_ = q.groupby(q.index.normalize()).agg(lambda z: ",".join(sorted(set(",".join(z).split(","))))).reindex(v.index).fillna("")
    return pd.DataFrame({"v": v, "bad_frac": f_.str.contains("Eqp").astype(float), "flag": f_})
def f_nwisany(sid):
    """Fetch USGS daily values of one site and parameter code (`sid` = "site|parameter").

    Takes whichever daily statistic the site has.
    """
    site, code = sid.split("|")[:2]
    t = get(f"https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites={site}&parameterCd={code}&startDT={START}&endDT={END}", f"nwisany_{site}_{code}")
    L = [l.split("\t") for l in t.splitlines() if l and not l.startswith("#")]
    if len(L) < 3: return None
    col = next((i for i, c in enumerate(L[0]) if f"_{code}_" in c and not c.endswith("_cd")), None)
    if col is None: return None
    v = pd.Series({pd.Timestamp(r[2]): pd.to_numeric(r[col], errors="coerce") for r in L[2:] if len(r) > col}, dtype=float).dropna()
    return pd.DataFrame({"v": v, "bad_frac": 0.0, "flag": ""}) if len(v) else None
def _nrcs(sid, dur):
    t = get("https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data?stationTriplets=%s&elements=RESC&duration=%s&beginDate=%s&endDate=%s"
            % (sid, dur, START, END), f"nrcs_{dur}_{sid}")
    vals = json.loads(t)[0]["data"][0]["values"]
    if dur == "DAILY": v = pd.Series({pd.Timestamp(x["date"]): x.get("value") for x in vals}, dtype=float).dropna()
    else: v = pd.Series({pd.Timestamp(year=x["year"], month=x["month"], day=1) + pd.offsets.MonthEnd(0): x.get("value") for x in vals}, dtype=float).dropna()
    return pd.DataFrame({"v": v, "bad_frac": 0.0, "flag": ""}) if len(v) else None
def f_cdec(sid):
    """Fetch daily reservoir storage (sensor 15) of a California CDEC station, in acre-feet."""
    t = get(f"https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations={sid}&SensorNums=15&dur_code=D&Start={START}&End={END}", f"cdec_{sid}")
    d = pd.read_csv(io.StringIO(t)); d["v"] = pd.to_numeric(d.VALUE, errors="coerce")
    d["d"] = pd.to_datetime(d["DATE TIME"], format="%Y%m%d %H%M", errors="coerce")
    v = d.dropna(subset=["v", "d"]).set_index("d").v; v = v.groupby(v.index.normalize()).mean()
    return pd.DataFrame({"v": v, "bad_frac": 0.0, "flag": ""}) if len(v) else None
def f_cdech(sid):
    """Fetch hourly reservoir storage (sensor 15) of a CDEC station, averaged per day.

    The CDEC daily values (`f_cdec`) are midnight readings.
    """
    t = get(f"https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations={sid}&SensorNums=15&dur_code=H&Start={START}&End={END}", f"cdech_{sid}")
    d = pd.read_csv(io.StringIO(t)); d["v"] = pd.to_numeric(d.VALUE, errors="coerce")
    d["d"] = pd.to_datetime(d["DATE TIME"], format="%Y%m%d %H%M", errors="coerce")
    v = d.dropna(subset=["v", "d"]).set_index("d").v; v = v.groupby(v.index.normalize()).mean()
    return pd.DataFrame({"v": v, "bad_frac": 0.0, "flag": ""}) if len(v) else None
def f_twdb(sid):
    """Fetch the daily storage of a Texas (TWDB) reservoir for START..END, in acre-feet.

    `sid` is the reservoir's short name. Total reservoir storage is used if the file has it, else
    conservation storage.
    """
    for slug in (re.sub(r"(?<!^)(?=[A-Z])", "-", sid).lower(), sid.lower()):
        t = get(f"https://www.waterdatafortexas.org/reservoirs/individual/{slug}.csv", f"twdb_{slug}")
        if not t.strip(): continue
        L = [l for l in t.splitlines() if not l.startswith("#")]
        try: d = pd.read_csv(io.StringIO("\n".join(L)))
        except Exception: continue
        vc = next((c for c in ["reservoir_storage", "conservation_storage"] if c in d.columns), None)
        if not vc or "date" not in d.columns: continue
        v = pd.Series(pd.to_numeric(d[vc], errors="coerce").values, index=pd.to_datetime(d.date, errors="coerce")).dropna()
        v = v[(v.index >= START) & (v.index <= END)]
        return pd.DataFrame({"v": v, "bad_frac": 0.0, "flag": ""}) if len(v) else None
    return None
def f_rise(sid):
    """Fetch the results of a Reclamation RISE catalog item (`sid` = item id), one value per day.

    Reads at most 11 pages of 500 values.
    """
    H = {"Accept": "application/vnd.api+json"}; out = {}
    for p in range(1, 12):
        t = get("https://data.usbr.gov/rise/api/result?itemId=%s&dateTime%%5Bafter%%5D=%s&dateTime%%5Bbefore%%5D=%s&itemsPerPage=500&page=%d"
                % (sid, START, END, p), f"rise_{sid}_{p}", H)
        try: d = json.loads(t)["data"]
        except Exception: break
        for x in d:
            a = x["attributes"]; out[pd.Timestamp(a["dateTime"]).tz_localize(None).normalize()] = a.get("result")
        if len(d) < 500: break
    v = pd.Series(out, dtype=float).dropna()
    return pd.DataFrame({"v": v, "bad_frac": 0.0, "flag": ""}) if len(v) else None
def f_hydromet(code):
    """Fetch daily storage (acre-feet) of a station from the Reclamation Hydromet daily service.

    `code` is the station code. The request starts on 1 January of the START year and ends on
    2026-10-15.
    """
    t = get(f"https://www.usbr.gov/pn-bin/daily.pl?parameter={code}%20af&syer={START[:4]}&smnth=1&sdy=1&eyer=2026&emnth=10&edy=15&format=csv", f"hydromet_{code}")
    d = pd.read_csv(io.StringIO(t)); d.columns = ["d", "v"] + list(d.columns[2:])
    v = pd.Series(pd.to_numeric(d.v, errors="coerce").values, index=pd.to_datetime(d.d, errors="coerce")).dropna()
    return pd.DataFrame({"v": v, "bad_frac": 0.0, "flag": ""}) if len(v) else None
def f_hydromet_gp(code):
    """Fetch daily storage (acre-feet) of a station from the Reclamation Great Plains Hydromet archive.

    `code` is the station code. Values at or below -9e5 (missing-data codes) are dropped.
    """
    y0, m0, d0 = START.split("-"); y1, m1, d1 = END.split("-")
    t = get(f"https://www.usbr.gov/gp-bin/webarccsv.pl?parameter={code}%20AF&syer={y0}&smnth={int(m0)}&sdy={int(d0)}&eyer={y1}&emnth={int(m1)}&edy={int(d1)}&format=2",
            f"hydrometgp_{code}")
    if "BEGIN DATA" not in t: return None
    L = [l.split(",") for l in t.split("BEGIN DATA")[1].split("END DATA")[0].strip().splitlines()[1:]]
    v = pd.Series({pd.to_datetime(a.strip(), format="%m/%d/%Y", errors="coerce"): pd.to_numeric(b.strip(), errors="coerce") for a, b, *_ in L if b.strip()}, dtype=float)
    v = v[v.index.notna()].dropna(); v = v[v > -9e5]
    return pd.DataFrame({"v": v, "bad_frac": 0.0, "flag": ""}) if len(v) else None
def f_lchdb(sdi):
    """Fetch daily values from the Reclamation Lower Colorado hydrologic database.

    `sdi` is the site-datatype id of the series.
    """
    t = get(f"https://www.usbr.gov/pn-bin/hdb/hdb.pl?svr=lchdb&sdi={sdi}&tstp=DY&t1={START}T00:00&t2={END}T00:00&table=R&mrid=0&format=json", f"lchdb_{sdi}")
    d = json.loads(t)["Series"][0]["Data"]
    v = pd.Series({pd.Timestamp(x["t"]).normalize(): pd.to_numeric(x["v"], errors="coerce") for x in d}, dtype=float).dropna()
    return pd.DataFrame({"v": v, "bad_frac": 0.0, "flag": ""}) if len(v) else None
def f_codwr(sid):
    """Fetch daily telemetry storage of a Colorado DWR station (`sid` = station abbreviation).

    The request dates are fixed: 2024-01-01 to 2026-10-15.
    """
    t = get("https://dwr.state.co.us/Rest/GET/api/v2/telemetrystations/telemetrytimeseriesday/?format=json&abbrev=%s&parameter=STORAGE"
            "&startDate=01/01/2024&endDate=10/15/2026&pageSize=5000" % sid, f"codwr_{sid}")
    L = json.loads(t).get("ResultList", [])
    v = pd.Series({pd.Timestamp(x["measDate"]): x.get("measValue") for x in L}, dtype=float).dropna()
    return pd.DataFrame({"v": v, "bad_frac": 0.0, "flag": ""}) if len(v) else None
FETCH = dict(CWMS=f_cwms, CWMSSUM=f_cwmssum, NWIS=f_nwis, NWISANY=f_nwisany, NRCS=lambda s: _nrcs(s, "DAILY"), NRCSM=lambda s: _nrcs(s, "MONTHLY"),
             CDEC=f_cdec, CDECH=f_cdech, TWDB=f_twdb, RISE=f_rise, HYDROMET=f_hydromet, HYDROMETGP=f_hydromet_gp, LCHDB=f_lchdb, CODWR=f_codwr)

def pava(y):
    """Return the isotonic (non-decreasing) least-squares fit of `y`, by pool-adjacent-violators."""
    v, w, n = [], [], []
    for x in y:
        v.append(float(x)); w.append(1.0); n.append(1)
        while len(v) > 1 and v[-2] > v[-1]:
            tw = w[-2] + w[-1]; v[-2] = (v[-2]*w[-2] + v[-1]*w[-1])/tw; w[-2] = tw; n[-2] += n[-1]; v.pop(); w.pop(); n.pop()
    return np.repeat(v, n)
# Acceptance thresholds: r = minimum Pearson correlation (storage), rel = maximum residual as a
# fraction of the overlap range.
GRADE = dict(A=dict(r=0.995, rel=0.01), B=dict(r=0.99, rel=0.025))
def grade(m):
    """Return the grade of a fitted map `m` (see `fit_map`): "A", "B", or None if it fails both.

    Storage needs the Pearson r and the relative RMS residual of GRADE; elevation needs a
    Spearman rho of at least 0.5 and both the isotonic and the leave-one-out residual within the
    limit of GRADE.
    """
    for g, t in GRADE.items():
        if m["kind"] == "storage" and m["r"] >= t["r"] and m["rel_rmse"] <= t["rel"]: return g
        if m["kind"] == "elevation" and m["rho"] >= 0.5 and m["rel_iso"] <= t["rel"] and m["rel_loo"] <= t["rel"]: return g
    return None

# ---- overlap mapping ----------------------------------------------------------------------------------------------------
def fit_map(x, y, kind, monthly):
    """Fit the map from a fetched series to the dam's existing storage over their overlap.

    x: fetched series in its native unit (daily, or month-end if `monthly`). y: the dam's existing
    daily storage (MCM). kind: "storage" (linear map) or "elevation" (monotone curve).
    Returns a dict with the map `f`, its valid level range (lo, hi), the chosen lag, the overlap
    statistics (n, r, rho, rel_rmse, rel_iso, rel_loo, range) and the grade; or None when the
    overlap is too short or has no variation. For elevation, `r` holds the Spearman rank
    correlation used to choose the lag.
    """
    best = None
    for lag in ([0] if monthly else [0, -1, 1]):
        xs = x.copy(); xs.index = xs.index + pd.Timedelta(days=lag)
        yy = y.resample("ME").last() if monthly else y
        j = pd.concat([xs.rename("x"), yy.rename("y")], axis=1).dropna()
        j = j[(j.index >= OV0) & (j.index <= OV1)]
        if len(j) < (12 if monthly else 120) or j.x.std() == 0 or j.y.std() == 0: continue
        r = float(np.corrcoef(j.x, j.y)[0, 1]) if kind == "storage" else float(j.x.rank().corr(j.y.rank()))
        if best is None or r > best["r"]: best = dict(lag=lag, r=r, j=j)
    if best is None: return None
    j = best["j"]; rng = j.y.max() - j.y.min()
    if kind == "storage":
        a, b = np.polyfit(j.x, j.y, 1); f = lambda v: a*v + b; desc = f"linear a={a:.6g} b={b:.4g}"; lo, hi = -np.inf, np.inf
    else:
        jj = j.sort_values(["x", "y"]); xv, yv = jj.x.values.astype(float), jj.y.values.astype(float)
        if len(np.unique(xv)) < 5: return None
        rho = float(pd.Series(xv).rank().corr(pd.Series(yv).rank()))
        rel_iso = float(np.sqrt(np.mean((yv - pava(yv))**2))/rng) if rng > 0 else np.inf      # departure from any monotone storage-level curve
        # leave-one-out: each interior point is predicted by linear interpolation between its
        # two neighbours in level order
        x0, x1, y0, y1 = xv[:-2], xv[2:], yv[:-2], yv[2:]
        w = np.where(x1 > x0, (xv[1:-1] - x0)/np.where(x1 > x0, x1 - x0, 1), 0.5); loo = yv[1:-1] - (y0 + w*(y1 - y0))
        rel_loo = float(np.sqrt(np.mean(loo**2))/rng) if rng > 0 else np.inf
        nb = min(200, len(np.unique(xv))); qb = pd.qcut(pd.Series(xv).rank(method="first"), nb, labels=False)
        g = pd.DataFrame({"x": xv, "y": yv, "b": qb.values}).groupby("b").median().sort_values("x")
        xs_, ys_ = g.x.values, np.maximum.accumulate(g.y.values)
        k = max(3, len(xs_)//7)
        sl_lo = np.polyfit(xs_[:k], ys_[:k], 1)[0]; sl_hi = np.polyfit(xs_[-k:], ys_[-k:], 1)[0]
        lo, hi = xs_[0], xs_[-1]
        def f(v, xs_=xs_, ys_=ys_, lo=lo, hi=hi, sl_lo=sl_lo, sl_hi=sl_hi):
            v = np.asarray(v, float); out = np.interp(v, xs_, ys_)
            return np.where(v < lo, ys_[0] + sl_lo*(v - lo), np.where(v > hi, ys_[-1] + sl_hi*(v - hi), out))
        desc = f"monotone rating from {len(xs_)} binned overlap levels {lo:.2f}-{hi:.2f}, end slopes {sl_lo:.4g}/{sl_hi:.4g}"
        res = j.y - f(j.x.values); rel = float(np.sqrt(np.mean(res**2))/rng) if rng > 0 else np.inf
        m = dict(f=f, lo=lo, hi=hi, lag=best["lag"], r=best["r"], rho=rho, n=len(j), rel_rmse=rel, rel_loo=rel_loo, rel_iso=rel_iso, desc=desc,
                 range=float(rng), kind=kind)
        m["grade"] = grade(m); m["ok"] = m["grade"] is not None; return m
    res = j.y - f(j.x.values); rel = float(np.sqrt(np.mean(res**2))/rng) if rng > 0 else np.inf
    m = dict(f=f, lo=lo, hi=hi, lag=best["lag"], r=best["r"], rho=np.nan, n=len(j), rel_rmse=rel, rel_loo=np.nan, rel_iso=np.nan, desc=desc,
             range=float(rng), kind=kind)
    m["grade"] = grade(m); m["ok"] = m["grade"] is not None; return m

def candidates(dam, T1, GF, TOUR):
    """Return the candidate series of a dam as tuples (role, source, sid[, kind]).

    T1 is the table of existing test-set values, GF the table gauge_final.csv and TOUR the table
    gauge_tournament.csv. The order is that of the module docstring; duplicates and sources that
    have no fetcher are left out.
    """
    out = []
    t = T1[T1.DAM_ID == dam].drop_duplicates(["source", "sid"]) if "source" in T1 else T1.iloc[0:0]
    for _, r in t.iterrows():
        src = str(r.source).split(":")[0]; sid = r.sid
        if str(r.source).startswith("HYDROMET:") and (pd.isna(sid) or sid == ""): sid = str(r.source).split(":")[1]
        if pd.notna(sid) and str(sid) != "": out.append(("truth", src, str(sid)))
    for _, r in GF[GF.DAM_ID == dam].iterrows(): out.append(("gauge_final", r.source, str(r.sid)))
    tt = TOUR[(TOUR.DAM_ID == dam) & (TOUR.status == "ok") & (TOUR.ovl_r >= 0.9)].sort_values("ovl_r", ascending=False)
    for _, r in tt.iterrows(): out.append(("tournament", r.source, str(r.sid)))
    if EXTRA is not None:
        for _, r in EXTRA[EXTRA.DAM_ID == dam].iterrows(): out.append(("discovered", r.source, str(r.sid), r.kind))
    seen, uniq = set(), []
    for c in out:
        if (c[1], c[2]) not in seen and c[1] in FETCH: seen.add((c[1], c[2])); uniq.append(c)
    return uniq
EXTRA = pd.concat([pd.read_csv(OUT/f) for f in ("extra_candidates.csv", "extra_manual.csv") if (OUT/f).exists()] or [pd.DataFrame(columns=["DAM_ID", "source", "sid", "kind"])],
                  ignore_index=True)

def kind_of(src, sid, GF, TOUR):
    """Return "storage" or "elevation" for a series.

    The kind comes from the gauge tables (GF, TOUR) if the series is listed there, else from the
    USGS parameter code or the CWMS series name, else it is storage.
    """
    m = pd.concat([GF, TOUR])[lambda d: (d.source == src) & (d.sid.astype(str) == sid)]
    if len(m): return m.kind.iloc[0]
    if src == "NWIS" or src == "NWISANY": return "elevation" if sid.split("|")[1] in ("62614", "62615", "00062", "00065") else "storage"
    if src == "CWMS": return "elevation" if ".Elev" in sid else "storage"
    return "storage"

def one(dam, T1, GF, TOUR):
    """Extend one dam and return (summary row, DataFrame of new days or None).

    The first grade-A candidate is the dam's 2026 series. If it leaves days of January to
    September 2026 uncovered, later grade-A candidates fill only the missing days; each is mapped
    onto the same frozen values on its own. The search stops once all but two days are covered.
    Grade-B candidates are used only when no candidate reaches grade A, best first (ext_grade = B
    in the output).
    With --crosscheck the search continues until a grade-A series from another source is found.
    Its mapped values are compared with the primary series on their common days (at least 30):
    xc_r is their correlation, xc_rel their RMS difference and xc_bias their mean difference, both
    as a fraction of the overlap range.
    """
    y = T1[T1.DAM_ID == dam].set_index("date").storage_mcm.sort_index(); tried = []; acc = {"A": [], "B": []}; xc = None
    ndays = len(pd.date_range(E0, E1)); have = set()
    for c in candidates(dam, T1, GF, TOUR):
        if acc["A"] and len(have) >= ndays - 2 and (not XCHECK or xc is not None): break
        role, src, sid = c[:3]
        try: d = FETCH[src](sid)
        except Exception as e: tried.append(f"{src}:{sid} error {type(e).__name__}"); continue
        if d is None or not len(d): tried.append(f"{src}:{sid} empty"); continue
        d = d[~d.index.duplicated()].sort_index()
        n26 = int(((d.index >= E0) & (d.index <= E1)).sum())
        if n26 == 0: tried.append(f"{src}:{sid} no 2026 data (last {d.index.max().date()})"); continue
        kind = c[3] if len(c) > 3 else kind_of(src, sid, GF, TOUR); monthly = src == "NRCSM"
        m = fit_map(d.v, y, kind, monthly)
        if m is None: tried.append(f"{src}:{sid} no usable {OV0[:4]}-{OV1[2:4]} overlap with the truth"); continue
        stat = (f"r={m['r']:.4f} rel_rmse={m['rel_rmse']:.4f}" if kind == "storage" else
                f"rho={m['rho']:.4f} rel_iso={m['rel_iso']:.4f} rel_loo={m['rel_loo']:.4f}")
        tried.append(f"{src}:{sid} [{kind}] {stat} n={m['n']} lag={m['lag']} {'grade ' + m['grade'] if m['ok'] else 'reject'}")
        if not m["ok"]: continue
        e = d[(d.index >= pd.Timestamp(E0) - pd.Timedelta(days=1)) & (d.index <= pd.Timestamp(E1) + pd.Timedelta(days=1))].copy()
        e.index = e.index + pd.Timedelta(days=m["lag"]); e = e[(e.index >= E0) & (e.index <= E1)]
        x = pd.DataFrame({"DAM_ID": dam, "date": e.index, "storage_mcm": m["f"](e.v.values), "native": e.v.values, "bad_frac": e.bad_frac.values,
                          "flag": e.flag.values, "extrapolated": (e.v.values < m["lo"]) | (e.v.values > m["hi"]),
                          "source": f"{src}:{kind}", "sid": sid, "map": m["desc"], "grade": m["grade"],
                          "ovl": f"overlap {OV0}..{OV1} {stat} n={m['n']} lag={m['lag']} grade {m['grade']}"})
        acc[m["grade"]].append((role, src, sid, kind, m, x))
        if m["grade"] == "A":
            if len(acc["A"]) == 1: have = set(x.date)
            else:
                p_ = acc["A"][0]
                if xc is None and src != p_[1]:   # cross-check: an accepted series from another source, mapped on its own, against the primary
                    jj = p_[5].set_index("date").storage_mcm.to_frame("p").join(x.set_index("date").storage_mcm.rename("q"), how="inner")
                    if len(jj) >= 30:
                        xc = dict(xc_source=src, xc_sid=sid, xc_n=len(jj), xc_r=float(jj.p.corr(jj.q)),
                                  xc_rel=float(np.sqrt(np.mean((jj.p - jj.q)**2))/m["range"]), xc_bias=float((jj.q - jj.p).mean()/m["range"]))
                have |= set(x.date)
    use = acc["A"] or sorted(acc["B"], key=lambda t: t[4]["rel_rmse"] if t[3] == "storage" else max(t[4]["rel_iso"], t[4]["rel_loo"]))
    if not use: return dict(DAM_ID=dam, status="not extended", tried=" | ".join(tried) if tried else "no candidates"), None
    role, src, sid, kind, m, ext = use[0]; fills = []
    for t in use[1:]:
        add = t[5][~t[5].date.isin(ext.date)]
        if len(add): ext = pd.concat([ext, add], ignore_index=True).sort_values("date"); fills.append(f"{t[1]}:{t[2]} +{len(add)} d")
    return dict(DAM_ID=dam, status="extended", ext_grade=m["grade"], role=role, source=src, sid=sid, kind=kind, ovl_n=m["n"], ovl_r=m["r"],
                rel_rmse=m["rel_rmse"], ovl_rho=m["rho"], rel_iso=m["rel_iso"], rel_loo=m["rel_loo"], lag=m["lag"], map=m["desc"],
                n_2026=len(ext), n_primary=int((ext.sid == sid).sum()), n_fill=int((ext.sid != sid).sum()), fill=" | ".join(fills),
                last=str(ext.date.max().date()), n_extrap=int(ext.extrapolated.sum()), tried=" | ".join(tried), **(xc or {})), ext

XCHECK = False
def main():
    """Parse the options, extend every requested dam in parallel and write the two output files."""
    global START, OV0, OV1, E0, XCHECK, EXTRA
    ap = argparse.ArgumentParser(); ap.add_argument("--dams"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--start", default=START, help="fetch start; with --ov0 for dams whose truth ends before mid-2024")
    ap.add_argument("--ov0", default=OV0); ap.add_argument("--suffix", default="")
    ap.add_argument("--crosscheck", action="store_true", help="also fetch the next accepted series and compare its 2026 values")
    ap.add_argument("--plus", action="store_true", help="the regime-2/3 dams absent from regime 1 (plus_dams.csv): truth = frozen regime-2 "
                    "ResOpsUS 2015-20, candidates = extra_plus.csv only, overlap 2015-20, output 2023-01-01..2026-09-30 (suffix _plus)"); a = ap.parse_args()
    START, OV0, XCHECK = a.start, a.ov0, a.crosscheck
    if a.plus: START, OV0, OV1, E0 = "2015-01-01", "2015-01-01", "2020-12-31", "2023-01-01"; a.suffix = a.suffix or "_plus"
    OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
    T1 = pd.read_parquet(R1/"truth.parquet"); T1["date"] = pd.to_datetime(T1.date)
    S = pd.read_csv(R1/"sites.csv"); GF = pd.read_csv(F/"crosswalk/gauge_final.csv"); TOUR = pd.read_csv(F/"crosswalk/gauge_tournament.csv")
    if a.plus:
        T1 = pd.read_parquet(F/"testsets/regime2_2015_2020/truth.parquet")[["DAM_ID", "date", "storage_mcm"]]; T1["date"] = pd.to_datetime(T1.date)
        S = pd.read_csv(F/"testsets/regime3_full_1980_2020/sites.csv"); GF, TOUR = GF.iloc[0:0], TOUR.iloc[0:0]
        EXTRA = pd.read_csv(OUT/"extra_plus.csv"); S = S[S.DAM_ID.isin(pd.read_csv(OUT/"plus_dams.csv").DAM_ID)]
    dams = [int(x) for x in a.dams.split(",")] if a.dams else sorted(S.DAM_ID.astype(int))
    rows, exts, t0 = [], [], time.time()
    def run(d):
        try: return one(d, T1, GF, TOUR)
        except Exception: return dict(DAM_ID=d, status="error", tried=traceback.format_exc(limit=2)), None
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (r, e) in enumerate(ex.map(run, dams)):
            rows.append(r); (exts.append(e) if e is not None else None)
            if (i + 1) % 25 == 0: print(f"{i + 1}/{len(dams)} dams, {sum(x['status'] == 'extended' for x in rows)} extended ({time.time() - t0:.0f}s)", flush=True)
    R = pd.DataFrame(rows).merge(S[["DAM_ID", "DAM_NAME", "STATE", "site_tier"]], on="DAM_ID", how="left")
    R["overlap_window"] = f"{OV0}..{OV1}"
    R.to_csv(OUT/f"harvest_2026{a.suffix}.csv", index=False)
    if exts: pd.concat(exts, ignore_index=True).to_parquet(OUT/f"ext_2026{a.suffix}.parquet", index=False)
    print(R.status.value_counts().to_string()); print(R[R.status == "extended"].groupby("source").size().to_string())
    print("extended by tier:", R[R.status == "extended"].site_tier.value_counts().to_dict(), "| all:", R.site_tier.value_counts().to_dict())

if __name__ == "__main__":
    main()
