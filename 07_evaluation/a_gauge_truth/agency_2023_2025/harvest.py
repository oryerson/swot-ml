"""Download 2023-2025 daily storage from agency services for ResOpsUS reservoirs (first version).

This is the first harvester. harvest_v2.py is the later version with more checks (reservoir
capacity, alternative names); it imports nothing from this file. resops_units.py imports
`resops_hist` from here.

Reads ($SWOTGAT_WORK/CONUS_storage_fork unless noted)
  resops_targets_full.csv              ResOpsUS dams with their graph node (node, DAM_ID,
                                       DAM_NAME, LAT, LONG, STATE, AGENCY_CODE)
  agency_storage_2023_2025_union.csv   reservoirs that an earlier download already covers
  harvested*.csv                       results of earlier runs of this script
  catalogs/{nrcs,cwms,rise,nwis,codwr,twdb}.csv   series catalogs (build_catalogs.py and others)
  agency_cache/cdec_stations.txt       saved HTML station table of California CDEC
  ResOpsUS/time_series_all/ResOpsUS_<DAM_ID>.csv  the dam's own historical record
  $SWOTGAT_WORK/CONUS/attributes/static_conus_storage.parquet   lake area per node (from the
                                       SWOT Prior Lake Database, PLD)

Writes
  harvested<tag>.csv     node, date, storage_mcm (million cubic meters), source, sid, method
  harvest_log<tag>.csv   one row per dam: what was tried, what passed and why
  agency_cache/<tag>.txt every HTTP response, reused for 14 days

Usage
  python harvest.py [SOURCE]
  SOURCE (optional) restricts the catalogs to one source: NRCS, CWMS, RISE, NWIS, CODWR, TWDB
  or CDEC. <tag> is the environment variable HARVEST_TAG if set, else "_SOURCE", else empty.

Method
  Only dams that no earlier file covers are processed. Each one is matched against every
  catalog, not only the agency ResOpsUS used:
  * Candidates: catalog series within 30 km that are either within 5 km or have a name
    similarity of at least 0.6; plus series without coordinates whose name similarity is at
    least 0.85. Name similarity is the larger of the shared-word fraction and the character
    match ratio, after generic words (lake, dam, reservoir, ...) are removed.
  * Order: storage series before elevation series, then better name match, then shorter
    distance. At most 10 candidates are tried. The first that passes validation is kept.
  * Each series is fetched for 2023-01-01 to 2025-12-31, averaged to days where it is
    sub-daily, and converted to million cubic meters (storage) or meters (elevation).
  * Validation uses the dam's own ResOpsUS record of 2010-2020. A series needs at least 250
    values and must not be constant.
    storage    The 2023-2025 median must be within 1/3 to 3 times the ResOpsUS median, either
               as it is or after dividing by one unit factor: 1000, 810.7 (acre-feet per
               million cubic meters, 1/AF in the code), 1e6, or their inverses. A factor is
               skipped when the corrected 5-95% range does not overlap the ResOpsUS 1-99%
               range, unless the ratio is within 0.5 to 2. Among the factors that fit, the one
               that brings the ratio closest to 1 wins (a factor other than 1 carries a small
               penalty) and is written to `method`. Without at least 200 ResOpsUS storage
               values the series is kept and marked unvalidated.
    elevation  The level is recentered on the ResOpsUS median elevation, which removes datum
               offsets, and converted with a storage-elevation curve built from the paired
               ResOpsUS daily values (median storage per 0.01 level step, forced to be
               non-decreasing, extended linearly beyond its ends). Rejected when the range
               exceeds 80 m, when the standard deviation exceeds 5 times that of ResOpsUS
               plus 0.5 m, or when the median differs from the ResOpsUS median by more than
               15 whether the ResOpsUS value is read as meters or as feet. The last rule is
               waived for a gauge on a local datum (median below half the ResOpsUS median)
               that is a close match (within 3 km, name similarity at least 0.8) and whose
               standard deviation is within 1/3 to 3 times that of ResOpsUS. If no curve can
               be built, storage is (level minus its median) times the PLD lake area. That
               gives a storage anomaly, not an absolute storage.

Notes
  Downloads go through swotgat.http: a generic client name and at most one request per 1.1 s
  per host, even though 8 dams are processed in parallel.
  Agencies revise their records, so a new download will differ from the paper's.
  Reclamation RISE is read for at most 7 pages of 500 values.
"""
from swotgat.paths import WORK
import io, json, math, re, sys, time, difflib, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests

F = Path(f"{WORK}/CONUS_storage_fork")
C = Path(f"{WORK}/CONUS")
CACHE = F/"agency_cache"; CAT = F/"catalogs"; TSA = F/"ResOpsUS/time_series_all"
START, END = "2023-01-01", "2025-12-31"
AF, FT = 1233.48/1e6, 0.3048
from swotgat.http import PoliteSession; S = PoliteSession()

def get(url, tag, headers=None, ttl=14):
    """Fetch a URL through the file cache and return the response text.

    `tag` names the cache file. A cached file younger than `ttl` days is returned without a
    request. Returns "" after a 400/404 (also cached) or four failed tries.
    """
    f = CACHE/(re.sub(r"[^A-Za-z0-9_.-]", "_", tag)[:180]+".txt")
    if f.is_file() and time.time()-f.stat().st_mtime < ttl*86400: return f.read_text()
    for a in range(4):
        try:
            r = S.get(url, timeout=150, headers=headers)
            if r.status_code == 200: f.write_text(r.text); return r.text
            if r.status_code in (400, 404): f.write_text(""); return ""
        except Exception: pass
        time.sleep(3*(a+1))
    return ""

STOP = set("lake lk dam reservoir res rsvr the and of at nr near pool project main headwater hw basin "
           "storage stor elev elevation daily time series data af ft ws water surface conservation cons "
           "tot total inst ave dcp rev ccp raw usbr usace nr at".split())
def key(s):
    """Return the distinctive words of a name.

    The words are lowercase; generic words (STOP) and single letters are removed.
    """
    s = str(s).lower().replace("_", " ").replace("mtn", "mountain").replace(" ck", " creek")
    return [w for w in re.findall(r"[a-z]+", s) if w not in STOP and len(w) > 1]
def sim(a, b):
    """Return the similarity of two names in [0, 1].

    It is the larger of the shared-word fraction (Jaccard) and the character match ratio of the
    distinctive words (see `key`), and 0 when either name has no distinctive word.
    """
    ka, kb = key(a), key(b)
    if not ka or not kb: return 0.0
    j = len(set(ka) & set(kb)) / len(set(ka) | set(kb))
    s = difflib.SequenceMatcher(None, "".join(ka), "".join(kb)).ratio()
    return max(j, s)
def km(a, b, c, d):
    """Return the great-circle distance in km between (lat a, lon b) and (lat c, lon d).

    Returns 1e9 if any coordinate is missing.
    """
    if any(pd.isna(x) for x in (a, b, c, d)): return 1e9
    R = 6371.0; p1, p2 = math.radians(a), math.radians(c)
    return 2*R*math.asin(math.sqrt(math.sin((p2-p1)/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(math.radians(d-b)/2)**2))

# ------------------------------------------------------------------ fetch --
# Each fetch_<source>(sid) returns (daily series, unit) for START..END, or (None, None).
# A unit of None means "use the unit listed in the catalog".
def fetch_nrcs(sid):
    """Fetch daily reservoir storage of an NRCS AWDB station (`sid` = station triplet), in acre-feet."""
    t = get("https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data?stationTriplets=%s&elements=RESC"
            "&duration=DAILY&beginDate=%s&endDate=%s" % (sid, START, END), f"nrcs_d_{sid}")
    try:
        j = json.loads(t); vals = j[0]["data"][0]["values"]
        s = pd.Series({pd.Timestamp(v["date"]): v.get("value") for v in vals}, dtype=float)
        return s.dropna(), "ac-ft"
    except Exception: return None, None

def fetch_cwms(sid):
    """Fetch a USACE CWMS time series (`sid` = "office|series name"), averaged to days.

    The unit is the one the service reports.
    """
    off, ts = sid.split("|", 1)
    t = get("https://cwms-data.usace.army.mil/cwms-data/timeseries?office=%s&name=%s&begin=%sT00:00:00Z"
            "&end=%sT00:00:00Z&page-size=900000" % (off, requests.utils.quote(ts, safe=""), START, END),
            f"cwms_ts_{off}_{ts}")
    try:
        j = json.loads(t); u = j.get("units", "")
        s = pd.Series({pd.to_datetime(v[0], unit="ms"): v[1] for v in j.get("values", []) if v[1] is not None}, dtype=float)
        return s.resample("D").mean().dropna(), u
    except Exception: return None, None

def fetch_rise(sid):
    """Fetch the results of a Reclamation RISE catalog item (`sid` = item id), one value per day.

    Reads at most 7 pages of 500 values. The unit is None (taken from the catalog).
    """
    H = {"Accept": "application/vnd.api+json"}; out = {}
    for p in range(1, 8):
        t = get("https://data.usbr.gov/rise/api/result?itemId=%s&dateTime%%5Bafter%%5D=%s&dateTime%%5Bbefore%%5D=%s"
                "&itemsPerPage=500&page=%d" % (sid, START, END, p), f"rise_r_{sid}_{p}", H)
        try: d = json.loads(t)["data"]
        except Exception: break
        for x in d:
            a = x["attributes"]; out[pd.Timestamp(a["dateTime"]).tz_localize(None).normalize()] = a.get("result")
        if len(d) < 500: break
    s = pd.Series(out, dtype=float).dropna()
    return (s if len(s) else None), None

def fetch_nwis(sid):
    """Fetch a USGS NWIS series (`sid` = "site|parameter|dv or iv"), averaged to days.

    Values at or below -9e5 (missing-data codes) are dropped. The unit is None (taken from the
    catalog).
    """
    site, pc, dt = sid.split("|")
    svc = "dv" if dt == "dv" else "iv"
    t = get("https://waterservices.usgs.gov/nwis/%s/?format=json&sites=%s&parameterCd=%s&startDT=%s&endDT=%s"
            % (svc, site, pc, START, END), f"nwis_{svc}_{site}_{pc}")
    try:
        j = json.loads(t); out = {}
        for ts in j["value"]["timeSeries"]:
            for blk in ts["values"]:
                for v in blk["value"]:
                    x = float(v["value"])
                    if x > -9e5: out[pd.Timestamp(v["dateTime"]).tz_localize(None)] = x
        s = pd.Series(out, dtype=float)
        return (s.resample("D").mean().dropna() if len(s) else None), None
    except Exception: return None, None

def fetch_cdec(sid):
    """Fetch daily reservoir storage (sensor 15) of a California CDEC station, in acre-feet."""
    t = get(f"https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations={sid}&SensorNums=15&dur_code=D"
            f"&Start={START}&End={END}", f"cdec_{sid}")
    try:
        d = pd.read_csv(io.StringIO(t)); d["v"] = pd.to_numeric(d.VALUE, errors="coerce")
        d["d"] = pd.to_datetime(d["DATE TIME"], format="%Y%m%d %H%M", errors="coerce")
        s = d.dropna(subset=["v","d"]).set_index("d").v
        return (s if len(s) else None), "ac-ft"
    except Exception: return None, None

def fetch_codwr(sid):
    """Fetch daily telemetry storage of a Colorado DWR station (`sid` = station abbreviation).

    The unit is the one the service reports, acre-feet if it reports none.
    """
    t = get("https://dwr.state.co.us/Rest/GET/api/v2/telemetrystations/telemetrytimeseriesday/?format=json"
            "&abbrev=%s&parameter=STORAGE&startDate=01/01/2023&endDate=12/31/2025&pageSize=5000" % sid,
            f"codwr_{sid}")
    try:
        L = json.loads(t).get("ResultList", [])
        s = pd.Series({pd.Timestamp(x["measDate"]): x.get("measValue") for x in L}, dtype=float)
        u = (L[0].get("measUnit") or "af") if L else "af"
        return (s.dropna() if len(s) else None), u
    except Exception: return None, None

def fetch_twdb(sid):
    """Fetch the daily storage record of a Texas (TWDB) reservoir, in acre-feet.

    `sid` is the reservoir's short name. The whole record is returned (the caller cuts it to
    START..END); total reservoir storage is used if the file has it, else conservation storage.
    """
    # URL slug hyphenates CamelCase then lowercases: JimChapman -> jim-chapman
    for slug in (re.sub(r"(?<!^)(?=[A-Z])", "-", sid).lower(), sid.lower()):
        t = get(f"https://www.waterdatafortexas.org/reservoirs/individual/{slug}.csv", f"twdb_{slug}")
        if not t.strip(): continue
        L = [l for l in t.splitlines() if not l.startswith("#")]
        try: d = pd.read_csv(io.StringIO("\n".join(L)))
        except Exception: continue
        vc = next((c for c in ["reservoir_storage", "conservation_storage"] if c in d.columns), None)
        if not vc or "date" not in d.columns: continue
        s = pd.Series(pd.to_numeric(d[vc], errors="coerce").values, index=pd.to_datetime(d.date, errors="coerce"))
        return s.dropna(), "ac-ft"
    return None, None

FETCH = dict(NRCS=fetch_nrcs, CWMS=fetch_cwms, RISE=fetch_rise, NWIS=fetch_nwis, CDEC=fetch_cdec, CODWR=fetch_codwr, TWDB=fetch_twdb)

# ------------------------------------------------------------- per dam data --
def resops_hist(dam_id):
    """Return the dam's ResOpsUS record for 2010-2020 (columns storage, elevation, ...), or None."""
    f = TSA/f"ResOpsUS_{int(dam_id)}.csv"
    if not f.is_file(): return None
    d = pd.read_csv(f, parse_dates=["date"]).dropna(subset=["date"]).set_index("date")
    # Some ResOpsUS per-dam files repeat dates or are out of order. Slicing by date would
    # then raise, and the per-dam try/except would silently skip the dam.
    d = d[~d.index.duplicated(keep="first")].sort_index()
    d = d.loc["2010-01-01":"2020-12-31"]
    return d

def to_storage_units(v, unit, kind):
    """Convert storage to million cubic meters (MCM) or elevation to meters.

    `kind` is "storage" or anything else for elevation. An unrecognized `unit` is read as
    acre-feet (storage) or feet (elevation).
    """
    u = (unit or "").lower()
    if kind == "storage":
        if u in ("m3",): return v*1e-6
        if u in ("mcm",): return v
        if "kaf" in u or u in ("1000 ac-ft",): return v*1000*AF
        return v*AF            # ac-ft, af, acre-ft, default for US agencies
    else:
        if u in ("m",): return v
        return v*FT            # ft, default

UNITFIX = [(1.0, "ok"), (1000., "x1000"), (1/1000., "/1000"), (1/AF, "/af"), (AF, "*af"), (1e6, "x1e6"), (1e-6, "/1e6")]

def validate_storage(s_mcm, hist):
    """Check a storage series (MCM) against the dam's ResOpsUS record `hist`.

    Returns (series, note), with a unit factor applied if one was needed, or (None, reason).
    The rules are listed in the module docstring.
    """
    if s_mcm is None or len(s_mcm) < 250: return None, f"short({0 if s_mcm is None else len(s_mcm)})"
    if s_mcm.std() == 0: return None, "constant"
    if hist is None or hist.storage.notna().sum() < 200: return s_mcm, "unvalidated(no ResOps storage)"
    M = hist.storage.median(); A = s_mcm.median()
    if M <= 0 or A <= 0: return None, "nonpositive"
    # Several unit factors can fall inside the tolerance at once (1000 and the acre-foot
    # factor differ by only 23%), so take the one closest to a ratio of 1, not the first
    # that fits. Taking the first would pick x1000 for a series in acre-feet and leave its
    # amplitude 23% too small.
    best = None
    lo, hi = hist.storage.quantile([.01, .99])
    for f, tag in UNITFIX:
        r = (A/f)/M
        if not (1/3 <= r <= 3): continue
        fixed = s_mcm/f
        ov = min(hi, fixed.quantile(.95)) - max(lo, fixed.quantile(.05))
        if ov <= 0 and not (0.5 <= r <= 2): continue
        score = abs(math.log(r)) + (0 if tag == "ok" else 0.25)   # mild prior for no correction
        if best is None or score < best[0]: best = (score, fixed, r, tag)
    if best:
        return best[1], f"ratio={best[2]:.2f} unit:{best[3]}"
    return None, f"magnitude mismatch A={A:.1f} vs ResOps {M:.1f} MCM"

def validate_elevation(e_m, hist, lake_km2, tight=False):
    """Check a water-level series (meters) and convert it to storage (MCM).

    `hist` is the dam's 2010-2020 ResOpsUS record, `lake_km2` the PLD lake area, and `tight`
    marks a close match by distance and name. Returns (series, note) or (None, reason). The
    rules are listed in the module docstring.
    """
    if e_m is None or len(e_m) < 250: return None, "short"
    if e_m.std() == 0: return None, "constant"
    if e_m.max()-e_m.min() > 80: return None, "elev range >80 m (not a reservoir level)"
    he = None if hist is None else hist.elevation.dropna()
    if he is not None and len(he) > 200:
        gap = abs(e_m.median()-he.median())
        if gap > 15 and abs((e_m.median()/FT)-he.median()) > 15:
            # A gage height on a local datum sits far below the sea-level elevation
            # of the same lake. Its anomaly is still valid; recentering removes the
            # datum. Accept only a tight match whose spread agrees with ResOpsUS.
            local = e_m.median() < 0.5*he.median()
            ratio = e_m.std()/max(he.std(), 1e-6)
            if not (local and tight and 1/3 <= ratio <= 3):
                return None, f"elev median {e_m.median():.1f} vs ResOps {he.median():.1f} m"
        if e_m.std() > 5*he.std() + 0.5: return None, "elev spread incompatible"
        pair = hist[["elevation", "storage"]].dropna()
        centred = e_m - e_m.median() + he.median()        # strips datum offsets
        if len(pair) > 200 and pair.elevation.max()-pair.elevation.min() > 0.5:
            pair = pair.sort_values("elevation")
            b = pair.groupby(pair.elevation.round(2)).storage.median()
            x, y = b.index.values, np.maximum.accumulate(b.values)
            slope_lo = (y[min(5,len(y)-1)]-y[0])/max(x[min(5,len(x)-1)]-x[0], 1e-6)
            slope_hi = (y[-1]-y[max(-6,-len(y))])/max(x[-1]-x[max(-6,-len(x))], 1e-6)
            v = centred.values
            st = np.interp(v, x, y)
            st = np.where(v < x[0], y[0] + (v-x[0])*slope_lo, st)
            st = np.where(v > x[-1], y[-1] + (v-x[-1])*slope_hi, st)
            return pd.Series(st, index=e_m.index), "rating curve (ResOps elev-storage)"
    if lake_km2 and lake_km2 > 0:
        return (e_m - e_m.median())*lake_km2, "static PLD area x elev anomaly"
    return None, "no rating and no area"

# ------------------------------------------------------------------ main ---
def main():
    """Harvest a 2023-2025 storage series for every ResOpsUS dam that no earlier file covers.

    Writes harvested<tag>.csv and harvest_log<tag>.csv (see the module docstring).
    """
    tgt = pd.read_csv(F/"resops_targets_full.csv", dtype={"node": str})
    have = set(pd.read_csv(F/"agency_storage_2023_2025_union.csv", dtype={"node": str}).node)
    for hf in F.glob("harvested*.csv"):          # skip dams that earlier runs already found
        have |= set(pd.read_csv(hf, dtype={"node": str}, usecols=["node"]).node)
    miss = tgt[~tgt.node.isin(have)].drop_duplicates("node").copy()
    only = sys.argv[1] if len(sys.argv) > 1 else None
    st = pd.read_parquet(C/"attributes/static_conus_storage.parquet", columns=["lake_area_km2"]).reset_index()
    st["subbasin"] = st.subbasin.astype(str); area = st.drop_duplicates("subbasin").set_index("subbasin").lake_area_km2

    cats = []
    for f in ["nrcs", "cwms", "rise", "nwis", "codwr", "twdb"]:
        p = CAT/f"{f}.csv"
        if p.is_file(): cats.append(pd.read_csv(p, dtype=str))
    try:
        html = (CACHE/"cdec_stations.txt").read_text()
        cd = max(pd.read_html(io.StringIO(html)), key=len); cd.columns = [str(c).strip().upper() for c in cd.columns]
        idc = next(c for c in cd.columns if "ID" in c); nmc = next(c for c in cd.columns if "NAME" in c)
        la = next(c for c in cd.columns if "LAT" in c); lo = next(c for c in cd.columns if "LON" in c)
        cats.append(pd.DataFrame(dict(source="CDEC", sid=cd[idc].astype(str), name=cd[nmc].astype(str),
                                      lat=cd[la], lon=cd[lo], kind="storage", unit="ac-ft", meta="")))
    except Exception as e: print("cdec catalog skipped:", e)
    K = pd.concat(cats, ignore_index=True)
    K["lat"] = pd.to_numeric(K.lat, errors="coerce"); K["lon"] = pd.to_numeric(K.lon, errors="coerce")
    if only: K = K[K.source == only]
    print(f"missing dams: {len(miss)}   catalog series: {len(K):,}  "
          f"{K.source.value_counts().to_dict()}", flush=True)
    Klat, Klon = K.lat.values, K.lon.values

    def candidates(d):
        dl = np.array([km(d.LAT, d.LONG, a, b) if not (np.isnan(a) or np.isnan(b)) else 1e9
                       for a, b in zip(Klat, Klon)])
        near = np.where(dl <= 30)[0]
        rows = []
        for i in near:
            ns = sim(d.DAM_NAME, K.name.iat[i])
            if dl[i] <= 5 or ns >= 0.6:
                rows.append((i, dl[i], ns))
        # catalog entries without coordinates: matched by name only, with a stricter threshold
        nocoord = np.where(np.isnan(Klat))[0]
        for i in nocoord:
            ns = sim(d.DAM_NAME, K.name.iat[i])
            if ns >= 0.85: rows.append((i, 999.0, ns))
        rows.sort(key=lambda r: (K.kind.iat[r[0]] != "storage", -round(r[2], 1), r[1]))
        return rows[:10]

    def work(d):
        log = dict(node=d.node, dam=d.DAM_NAME, state=d.STATE, agency=d.AGENCY_CODE, tried=0,
                   ok=False, source="", sid="", method="", note="")
        try:
            hist = resops_hist(d.DAM_ID)
            cands = candidates(d)
            log["n_cands"] = len(cands)
            for i, dist, ns in cands:
                src, sid, kind, unit = K.source.iat[i], K.sid.iat[i], K.kind.iat[i], K.unit.iat[i]
                log["tried"] += 1
                s, u2 = FETCH[src](sid)
                if s is None or len(s) == 0:
                    log["note"] += f" | {src}:{sid[:30]} empty"; continue
                unit_eff = u2 or unit
                if src == "RISE" and not unit_eff:
                    unit_eff = "af" if kind == "storage" else "ft"
                v = to_storage_units(s, unit_eff, kind)
                v = v[(v.index >= START) & (v.index <= END)]
                if kind == "storage":
                    out, note = validate_storage(v, hist)
                else:
                    out, note = validate_elevation(v, hist, area.get(d.node),
                                                   tight=(dist <= 3 and ns >= 0.8))
                if out is None:
                    log["note"] += f" | {src}:{sid[:30]} {note}"; continue
                log.update(ok=True, source=src, sid=sid, method=f"{kind}: {note}",
                           note=f"{dist:.1f} km, name sim {ns:.2f}")
                return log, pd.DataFrame(dict(node=d.node, date=out.index.date, storage_mcm=out.values,
                                              source=f"{src}:{kind}", sid=sid, method=note))
        except Exception as e:
            log["note"] = f"error {type(e).__name__}: {e}"
        return log, None

    logs, frames = [], []
    with ThreadPoolExecutor(8) as ex:
        for k, (lg, df) in enumerate(ex.map(work, [r for _, r in miss.iterrows()])):
            logs.append(lg)
            if df is not None: frames.append(df)
            flag = "OK " if lg["ok"] else "-- "
            print(f"{k+1:>4}/{len(miss)} {flag}{str(lg['dam'])[:26]:<26} {str(lg['agency'])[:8]:<8} "
                  f"{lg['source'] or '':<5} {lg['method'][:44]:<44} {lg['note'][:50]}", flush=True)
    import os
    tag = os.environ.get("HARVEST_TAG") or (f"_{only}" if only else "")
    pd.DataFrame(logs).to_csv(F/f"harvest_log{tag}.csv", index=False)
    if frames:
        H = pd.concat(frames); H.to_csv(F/f"harvested{tag}.csv", index=False)
        print(f"\nHARVESTED {H.node.nunique()} new reservoirs, {len(H):,} daily values")
        print(pd.DataFrame(logs).query("ok").source.value_counts().to_string())
    else:
        print("\nnothing harvested")

if __name__ == "__main__":
    main()
