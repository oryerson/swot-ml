"""Download every series listed by discover_expand.py, as daily values in the units of its source.

Step 3 of the expanded gauge set. The window is 1980-01-01 to 2026-10-15. No series is judged or
converted here; build_plus.py does that.

Reads (under $SWOTGAT_WORK/CONUS_storage_fork unless noted)
  gauges_expand/discovery/series.csv             from discover_expand.py
  ResOpsUS/time_series_all/ResOpsUS_<DAM_ID>.csv
  $SWOTGAT_WORK/GRADES/gdrom_v2/file_list.json   URLs of the GDROM v2 "collected data" files

Writes (gauges_expand/discovery/)
  series_fetched.csv   series.csv after the CWMS selection below
  native.parquet       source, sid, kind, unit, date, v, bad_frac (one row per series and day)
  harvest_log.csv      one row per series: ok (with number of days, first and last day, unit),
                       empty, or error
  and every response in gauges_expand/cache/

Usage
  python harvest_expand.py        (no arguments)

Method
  Endpoints and daily aggregation follow harvest_2026.py:
    CWMS     USACE time series in English units (acre-feet, feet), requested in 5-year pieces. The
             daily value is the mean over the UTC day. bad_frac is the share of the day's values
             whose quality code marks them as questionable, rejected or missing.
    NWISANY  USGS daily values. Parameter 00054 is in acre-feet, elevation in feet. bad_frac is 1 on
             days whose qualifier contains Eqp, Ice or Dis.
    NRCS     AWDB element RESC, daily, acre-feet
    CDEC     California sensor 15, daily, acre-feet
    TWDB     Texas per-reservoir csv (reservoir_storage, or conservation_storage if that is absent),
             acre-feet
    RISE     Reclamation results of a catalog item, acre-feet
    CODWR    Colorado daily telemetry, parameter STORAGE, acre-feet
    GDROMV2  the GDROM v2 "collected data" csv on HydroShare, acre-feet
    RESOPS   the local ResOpsUS file: storage in million cubic meters (MCM), elevation in meters
  CWMS selection, to keep the number of requests small: at most two storage series and one elevation
  series per dam. Daily series come first, then hourly, then finer intervals; within those, revised
  or best versions first, then the longest record. Series that end before 1981 are skipped.
  One thread runs per service, so services are queried in parallel but each one in sequence.

Notes
  All requests go through gx_common.get: swotgat.http (generic client name, at most one request per
  1.1 s per host, 5 s for HydroShare, its published crawl delay) and a cache of every response. The
  robots.txt files of CWMS, CDEC, TWDB and RISE disallow automated access to the paths requested here
  (see size_candidates.py). Check each service's current terms before you run this again. Agencies
  revise their records, so a new download will differ from the archived one.
"""
import io, json, re, sys, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); from gx_common import *
O = X/"discovery"; S = pd.read_csv(O/"series.csv", dtype={"sid": str}); START, END = "1980-01-01", "2026-10-15"
def ser(d, unit):
    """Return (d, unit), or None if the table d is missing or empty.

    Every fetch function below ends with this call. d is indexed by day and has the columns v (the daily
    value in the unit of the source) and bad_frac (share of the day's raw values marked bad, 0 to 1).
    """
    return None if d is None or len(d) == 0 else (d, unit)
def daily_mean(s):
    """Return the mean per calendar day of a time-indexed series, missing values left out."""
    s = s.dropna(); return s.groupby(s.index.normalize()).mean()
def f_resops(sid, kind):
    """Read the local ResOpsUS record of dam sid: storage in MCM, or elevation in m for that kind.

    bad_frac is 0 on every day. Returns (table, unit) as described for ser, or None if there is no data.
    """
    t = pd.read_csv(F/f"ResOpsUS/time_series_all/ResOpsUS_{sid}.csv"); t["date"] = pd.to_datetime(t.date, errors="coerce")
    t = t.dropna(subset=["date"]).drop_duplicates("date").set_index("date").sort_index()
    v = pd.to_numeric(t.storage if kind == "storage" else t.get("elevation"), errors="coerce").dropna()
    return ser(pd.DataFrame({"v": v, "bad_frac": 0.0}), "MCM" if kind == "storage" else "m")
def f_cwms(sid, kind, b=None, e=None):
    """Download a USACE CWMS time series in English units and average it to UTC days.

    sid is "office|time-series name". b and e are the begin and end of the period of record from the
    catalog; they limit the years requested (1980-2026 at most, in 5-year pieces). bad_frac is the share
    of a day's values whose quality code marks them as questionable, rejected or missing. The unit is
    "ac-ft" for storage and "ft" for elevation. Returns (table, unit) as described for ser, or None if
    there is no data.
    """
    off, ts = sid.split("|", 1); parts = []
    y0 = max(1980, pd.Timestamp(b).year if isinstance(b, str) and b[:4].isdigit() else 1980)
    y1 = min(2026, pd.Timestamp(e).year if isinstance(e, str) and e[:4].isdigit() else 2026)
    for y in range(y0, y1 + 1, 5):
        a, z = f"{y}-01-01", (f"{min(y + 5, 2026)}-01-01" if y + 5 <= 2026 else END)
        t = get("https://cwms-data.usace.army.mil/cwms-data/timeseries?office=%s&name=%s&begin=%sT00:00:00Z&end=%sT00:00:00Z&unit=EN&page-size=5000000"
                % (off, requests.utils.quote(ts, safe=""), a, z), f"gxcwms_{off}_{ts}_{y}")
        try: j = json.loads(t) if t else {}
        except Exception: continue
        rows = [(v[0], v[1], v[2] if len(v) > 2 else 0) for v in j.get("values", []) if v[1] is not None]
        if rows: parts.append(pd.DataFrame(rows, columns=["ts", "v", "q"]))
    if not parts: return None
    d = pd.concat(parts); d["ts"] = pd.to_datetime(d.ts, unit="ms"); d["bad"] = (d.q.fillna(0).astype(int) & 30).isin([4, 8, 16])
    d["day"] = d.ts.dt.normalize(); g = d.groupby("day")
    unit = "ac-ft" if kind == "storage" else "ft"
    return ser(pd.DataFrame({"v": g.v.mean(), "bad_frac": g.bad.mean()}), unit)
def f_nwisany(sid, kind, *a):
    """Download the USGS daily values of one parameter code; sid is "site|code".

    bad_frac is 1 on days whose qualifier contains Eqp, Ice or Dis, and 0 otherwise. The unit is "ac-ft"
    for parameter 00054 and "ft" for any other code. Returns (table, unit) as described for ser, or None
    if there is no data.
    """
    site, code = sid.split("|")[:2]
    t = get(f"https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites={site}&parameterCd={code}&startDT={START}&endDT={END}", f"gxnwis_{site}_{code}")
    L = [l.split("\t") for l in t.splitlines() if l and not l.startswith("#")]
    if len(L) < 3: return None
    col = next((i for i, c in enumerate(L[0]) if f"_{code}_" in c and not c.endswith("_cd")), None)
    if col is None: return None
    cd = col + 1 if col + 1 < len(L[0]) and L[0][col + 1].endswith("_cd") else None
    v = {pd.Timestamp(r[2]): (pd.to_numeric(r[col], errors="coerce"), (r[cd] if cd and len(r) > cd else "")) for r in L[2:] if len(r) > col}
    d = pd.DataFrame.from_dict(v, orient="index", columns=["v", "cd"]).dropna(subset=["v"])
    return ser(pd.DataFrame({"v": d.v, "bad_frac": d.cd.astype(str).str.contains("Eqp|Ice|Dis").astype(float)}), "ac-ft" if code == "00054" else "ft")
def f_nrcs(sid, kind, *a):
    """Download the daily reservoir storage (element RESC, acre-feet) of an NRCS station triplet.

    bad_frac is 0 on every day. Returns (table, unit) as described for ser, or None if there is no data.
    """
    t = get("https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data?stationTriplets=%s&elements=RESC&duration=DAILY&beginDate=%s&endDate=%s" % (sid, START, END), f"gxnrcs_D_{sid}")
    try: vals = json.loads(t)[0]["data"][0]["values"]
    except Exception: return None
    v = pd.Series({pd.Timestamp(x["date"]): x.get("value") for x in vals}, dtype=float).dropna()
    return ser(pd.DataFrame({"v": v, "bad_frac": 0.0}), "ac-ft")
def f_cdec(sid, kind, *a):
    """Download the daily storage (sensor 15, acre-feet) of a California CDEC station.

    Values at or below -900,000 (the missing-value codes) are dropped. bad_frac is 0 on every day.
    Returns (table, unit) as described for ser, or None if there is no data.
    """
    t = get(f"https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations={sid}&SensorNums=15&dur_code=D&Start={START}&End={END}", f"gxcdec_{sid}")
    if not t.strip(): return None
    d = pd.read_csv(io.StringIO(t)); d["v"] = pd.to_numeric(d.VALUE, errors="coerce"); d["d"] = pd.to_datetime(d["DATE TIME"], format="%Y%m%d %H%M", errors="coerce")
    v = daily_mean(d.dropna(subset=["v", "d"]).set_index("d").v); v = v[v > -9e5]
    return ser(pd.DataFrame({"v": v, "bad_frac": 0.0}), "ac-ft")
def f_twdb(sid, kind, *a):
    """Download the storage (acre-feet) of a Texas reservoir from its TWDB csv file.

    sid is the name of the reservoir's page. The column reservoir_storage is used, or
    conservation_storage if that is absent. Days before
    1980-01-01 are dropped and bad_frac is 0 on every day. Returns (table, unit) as described for ser, or
    None if there is no data.
    """
    t = get(f"https://www.waterdatafortexas.org/reservoirs/individual/{sid}.csv", f"gxtwdb_{sid}")
    L = [l for l in t.splitlines() if not l.startswith("#")]
    try: d = pd.read_csv(io.StringIO("\n".join(L)))
    except Exception: return None
    vc = next((c for c in ["reservoir_storage", "conservation_storage"] if c in d.columns), None)
    if not vc or "date" not in d.columns: return None
    v = pd.Series(pd.to_numeric(d[vc], errors="coerce").values, index=pd.to_datetime(d.date, errors="coerce")).dropna()
    return ser(pd.DataFrame({"v": v[v.index >= START], "bad_frac": 0.0}), "ac-ft")
def f_rise(sid, kind, *a):
    """Download the results (acre-feet) of a Reclamation RISE catalog item, one value per day.

    Pages of 1,000 results are followed, 39 at most. bad_frac is 0 on every day. Returns (table, unit)
    as described for ser, or None if there is no data.
    """
    Hh = {"Accept": "application/vnd.api+json"}; out = {}
    for p in range(1, 40):
        t = get("https://data.usbr.gov/rise/api/result?itemId=%s&dateTime%%5Bafter%%5D=%s&dateTime%%5Bbefore%%5D=%s&itemsPerPage=1000&page=%d"
                % (sid, START, END, p), f"gxrise_{sid}_{p}", Hh)
        try: d = json.loads(t)["data"]
        except Exception: break
        for x in d:
            at = x["attributes"]; out[pd.Timestamp(at["dateTime"]).tz_localize(None).normalize()] = at.get("result")
        if len(d) < 1000: break
    v = pd.Series(out, dtype=float).dropna(); return ser(pd.DataFrame({"v": v, "bad_frac": 0.0}), "ac-ft")
def f_codwr(sid, kind, *a):
    """Download the daily storage (parameter STORAGE, acre-feet) of a Colorado telemetry station.

    sid is the station abbreviation. bad_frac is 0 on every day. Returns (table, unit) as described for
    ser, or None if there is no data.
    """
    rows = []
    for p in range(1, 10):
        t = get("https://dwr.state.co.us/Rest/GET/api/v2/telemetrystations/telemetrytimeseriesday/?format=json&abbrev=%s&parameter=STORAGE"
                "&startDate=01/01/1980&endDate=10/15/2026&pageSize=50000&pageIndex=%d" % (sid, p), f"gxcodwr_{sid}_{p}")
        try: j = json.loads(t)
        except Exception: break
        rows += j.get("ResultList", [])
        if p >= j.get("PageCount", 1): break
    v = pd.Series({pd.Timestamp(x["measDate"]): x.get("measValue") for x in rows}, dtype=float).dropna()
    return ser(pd.DataFrame({"v": v, "bad_frac": 0.0}), "ac-ft")
FL = {int(m.group(1)): x["url"].replace("http://", "https://") for x in json.load(open(F.parent/"GRADES/gdrom_v2/file_list.json"))
      if "collected" in x["url"] for m in [re.search(r"/(\d+)\.csv$", x["url"])] if m}
def f_gdromv2(sid, kind, *a):
    """Download the storage (acre-feet) in the GDROM v2 "collected data" file of dam sid.

    The date and storage columns are found by the start of their names. bad_frac is 0 on every day.
    Returns (table, unit) as described for ser, or None if the file has no storage column or no data.
    """
    t = get(FL[int(sid)], f"gxgdromv2_{sid}")
    d = pd.read_csv(io.StringIO(t)); dc = next(c for c in d.columns if c.lower().startswith("date")); sc = next((c for c in d.columns if c.lower().startswith("storage")), None)
    if sc is None: return None
    v = pd.Series(pd.to_numeric(d[sc], errors="coerce").values, index=pd.to_datetime(d[dc], errors="coerce")).dropna()
    return ser(pd.DataFrame({"v": v, "bad_frac": 0.0}), "ac-ft")
FETCH = dict(RESOPS=f_resops, CWMS=f_cwms, NWISANY=f_nwisany, NRCS=f_nrcs, CDEC=f_cdec, TWDB=f_twdb, RISE=f_rise, CODWR=f_codwr, GDROMV2=f_gdromv2)
HOST = dict(RESOPS="local", CWMS="cwms", NWISANY="usgs", NRCS="nrcs", CDEC="cdec", TWDB="twdb", RISE="rise", CODWR="codwr", GDROMV2="hydroshare")
# CWMS: keep request volume small - per dam at most 2 storage and 1 elevation series, coarsest interval first (1Day < 1Hour < finer),
# revised/best versions first, then the longest extent; series whose extent lies wholly outside 1980-2026 are skipped
cw = S[S.source == "CWMS"].copy()
iv = cw.sid.str.split(".").str[3].fillna("")
cw["r_iv"] = np.select([iv.str.contains("1Day|~1Day"), iv.str.contains("1Hour|~1Hour")], [0, 1], 2)
cw["r_ver"] = (~cw.sid.str.contains(r"Rev|Best|Ccp-Rev|Final", case=False)).astype(int)
cw["ext"] = (pd.to_datetime(cw.end, errors="coerce", utc=True) - pd.to_datetime(cw.begin, errors="coerce", utc=True)).dt.days.fillna(0)
cw = cw[~(pd.to_datetime(cw.end, errors="coerce", utc=True) < pd.Timestamp("1981-01-01", tz="UTC"))]
cw = cw.sort_values(["DAM_ID", "kind", "r_iv", "r_ver", "ext"], ascending=[True, True, True, True, False])
cw = cw.groupby(["DAM_ID", "kind"], group_keys=False).apply(lambda g: g.head(2 if g.name[1] == "storage" else 1))
S = pd.concat([S[S.source != "CWMS"], cw[S.columns]], ignore_index=True); S.to_csv(O/"series_fetched.csv", index=False)
U = S.drop_duplicates(["source", "sid", "kind"]); out, log = [], []
print("series to fetch", len(U), U.source.value_counts().to_dict(), flush=True)
def run(rows):
    """Download the series of one service, one after another, and collect the results.

    rows is the part of the series table that one service holds. Every series adds a row to the global
    list log (status ok, empty or error) and, if it has data, its daily values within the download window
    to the global list out.
    """
    for _, r in rows.iterrows():
        try:
            res = FETCH[r.source](r.sid, r.kind, r.begin, r.end) if r.source == "CWMS" else FETCH[r.source](r.sid, r.kind)
        except Exception as e:
            log.append(dict(source=r.source, sid=r.sid, kind=r.kind, status=f"error {type(e).__name__}: {str(e)[:80]}")); continue
        if res is None: log.append(dict(source=r.source, sid=r.sid, kind=r.kind, status="empty")); continue
        d, unit = res; d = d[(d.index >= START) & (d.index <= END)]
        log.append(dict(source=r.source, sid=r.sid, kind=r.kind, status="ok", n=len(d), first=d.index.min(), last=d.index.max(), unit=unit))
        out.append(pd.DataFrame({"source": r.source, "sid": r.sid, "kind": r.kind, "unit": unit, "date": d.index, "v": d.v.values.astype(float),
                                 "bad_frac": d.bad_frac.values.astype(float)}))
groups = [g for _, g in U.groupby(U.source.map(HOST))]
with ThreadPoolExecutor(len(groups)) as ex: list(ex.map(run, groups))
L = pd.DataFrame(log); L.to_csv(O/"harvest_log.csv", index=False)
N = pd.concat(out, ignore_index=True); N.to_parquet(O/"native.parquet", index=False)
print(L.groupby(["source", "status"]).size().unstack(fill_value=0).to_string())
print("rows", len(N), "| series", N.groupby(["source", "sid", "kind"]).ngroups, "| cache MB", round(sum(f.stat().st_size for f in CACHE.iterdir())/1e6, 1))
