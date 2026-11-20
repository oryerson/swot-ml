"""Download 2023-2025 daily storage from agency services for ResOpsUS reservoirs (second version).

For every ResOpsUS dam that no earlier file covers, the script searches the catalogs of all
agency services for a storage or water-level series of that reservoir, downloads it, and keeps
the first series that is consistent with the dam's own ResOpsUS history and with its capacity.
Compared with harvest.py it adds a capacity check, alternative reservoir names, a different
candidate order and two more ways to accept a storage series. manual_override.py imports this
module for its fetchers and validators.

Reads ($SWOTGAT_WORK/CONUS_storage_fork unless noted)
  resops_targets_full.csv              ResOpsUS dams with their graph node (node, DAM_ID,
                                       grand_id, DAM_NAME, LAT, LONG, STATE, AGENCY_CODE)
  agency_storage_2023_2025_union.csv   reservoirs that an earlier download already covers
  harvested*.csv                       results of earlier runs
  catalogs/{nrcs,cwms,rise,nwis,codwr,twdb}.csv   series catalogs (build_catalogs.py and others)
  agency_cache/cdec_stations.txt       saved HTML station table of California CDEC
  ResOpsUS/time_series_all/ResOpsUS_<DAM_ID>.csv  the dam's own historical record
  $SWOTGAT_WORK/CONUS/attributes/static_conus_storage.parquet   per node: lake_area_km2 and
                                       dam_cap_mcm (reservoir capacity)
  $SWOTGAT_NAS_WATER/cjgleason/ted/GDW/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp
                                       Global Dam Watch (GDW) reservoir names
  $SWOTGAT_NAS_WATER/cjgleason/ted/LakeSPvC_ts_for_Ted/SWOT_PLD_v106_reservoirs.gdb
                                       SWOT Prior Lake Database (PLD) lake names

Writes
  harvested<tag>.csv     node, date, storage_mcm (million cubic meters), source, sid, method
  harvest_log<tag>.csv   one row per dam: what was tried, what passed and why
  agency_cache/<tag>.txt every HTTP response, reused for 14 days

Usage
  [HARVEST_TAG=_pass2] [HARVEST_NODES=node1,node2] python harvest_v2.py [SOURCE]
  SOURCE (optional) restricts the catalogs to one source: NRCS, CWMS, RISE, NWIS, CODWR, TWDB
  or CDEC. <tag> is HARVEST_TAG if set, else "_SOURCE", else empty. HARVEST_NODES processes the
  named nodes even if an earlier file already has them. merge_truth.py expects the files
  harvested.csv and harvested_pass2.csv to harvested_pass5.csv.

Method
  * Names. A dam is known by its ResOpsUS name, its GDW reservoir and alternative names, and
    the PLD names of its node. The name similarity to a catalog series is the best over these
    names. Similarity is the larger of the shared-word fraction and the character match ratio,
    after generic words (lake, dam, reservoir, ...) are removed.
  * Candidates. Catalog series within 30 km that are either within 5 km or have a name
    similarity of at least 0.6; plus series without coordinates whose similarity is at least
    0.85 (Texas Water Development Board series only for dams inside a box around Texas).
  * Order. Series with coordinates first, then better name match, then shorter distance, then
    series whose id marks them as revised or observed before raw sensor feeds. The first 10
    storage series are tried, then the first 10 elevation series. The first that passes is kept.
  * Each series is fetched for 2023-01-01 to 2025-12-31, averaged to days where it is
    sub-daily, and converted to million cubic meters (storage) or meters (elevation).
  * A match is "tight" when it is within 3 km with similarity at least 0.6, or within 1 km, or
    within 15 km with similarity at least 0.9.
  * Validation uses the dam's ResOpsUS record of 2010-2020 and its capacity. A series needs at
    least 250 values and must not be constant. "Plausible against capacity" means that the
    median is at least 0.5% of capacity and the 95th percentile at most 130% (always true when
    the capacity is unknown).
    storage    Accepted in one of four ways (see `validate_storage`): the median agrees with
               ResOpsUS; it agrees after one unit factor; a tight match lies inside the range
               of the whole ResOpsUS record (drought); or a tight match is plausible against
               capacity while ResOpsUS is not.
    elevation  The level is recentered on the ResOpsUS median elevation, which removes datum
               offsets, and converted with a storage-elevation curve built from the paired
               ResOpsUS daily values (median storage per 0.01 level step, forced to be
               non-decreasing, extended linearly beyond its ends). Rejected when the range
               exceeds 80 m, when the standard deviation exceeds 5 times that of ResOpsUS
               plus 0.5 m, or when the median differs from the ResOpsUS median by more than
               15 whether the ResOpsUS value is read as meters or as feet. The last rule has
               two exceptions for tight matches: a gauge on a local datum (median below half
               the ResOpsUS median) with a standard deviation within 1/3 to 3 times that of
               ResOpsUS; and a gauge whose level change times the PLD lake area has a standard
               deviation within 1/3 to 3 times that of ResOpsUS storage, which is then
               converted with the lake area instead of the curve. If no curve can be built,
               storage is (level minus its median) times the PLD lake area. Lake-area
               conversions give a storage anomaly, not an absolute storage; their `method`
               text contains "static PLD".

Notes
  Downloads go through swotgat.http: a generic client name and at most one request per 1.1 s
  per host, even though 8 dams are processed in parallel.
  Agencies revise their records, so a new download will differ from the paper's.
  Reclamation RISE is read for at most 7 pages of 500 values.
"""
from swotgat.paths import NAS_WATER, WORK
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
# FULL_STOR keeps each dam's whole ResOpsUS storage record, keyed by id() of the 2010-2020
# frame that resops_hist returns. validate_storage looks it up for the drought check.
FULL_STOR = {}
def resops_hist(dam_id):
    """Return the dam's ResOpsUS record for 2010-2020 (columns storage, elevation, ...), or None.

    The storage of the whole record is saved in FULL_STOR under id() of the returned frame.
    """
    f = TSA/f"ResOpsUS_{int(dam_id)}.csv"
    if not f.is_file(): return None
    d = pd.read_csv(f, parse_dates=["date"]).dropna(subset=["date"]).set_index("date")
    # Some ResOpsUS per-dam files repeat dates or are out of order. Slicing by date would
    # then raise, and the per-dam try/except would silently skip the dam.
    d = d[~d.index.duplicated(keep="first")].sort_index()
    full = d.copy()                     # whole record, for the drought range check
    d = d.loc["2010-01-01":"2020-12-31"].copy()
    FULL_STOR[id(d)] = full.storage.dropna()   # not stored in d.attrs: pandas compares attrs on concat,
    return d                                    # and a DataFrame-valued attr made that comparison raise

def to_storage_units(v, unit, kind):
    """Convert storage to million cubic meters (MCM) or elevation to meters.

    `kind` is "storage" or anything else for elevation. An unrecognized `unit` is read as
    acre-feet (storage) or feet (elevation).
    """
    u = (unit or "").lower()
    if kind == "storage":
        if u in ("m3",): return v*1e-6
        if u in ("mcm",): return v
        if "kaf" in u or u in ("1000 ac-ft", "kac-ft", "kac_ft"): return v*1000*AF
        return v*AF            # ac-ft, af, acre-ft, default for US agencies
    else:
        if u in ("m",): return v
        return v*FT            # ft, default

UNITFIX = [(1.0, "ok"), (1000., "x1000"), (1/1000., "/1000"), (1/AF, "/af"), (AF, "*af"), (1e6, "x1e6"), (1e-6, "/1e6")]

def validate_storage(s_mcm, hist, cap=None, tight=False):
    """Check a storage series (MCM) against ResOpsUS history, capacity, and the series itself.

    `hist` is the dam's 2010-2020 ResOpsUS record, `cap` its capacity (MCM), `tight` marks a
    close match by distance and name. Returns (series, note) or (None, reason).

    A series is "plausible against capacity" when its median is at least 0.5% of capacity and
    its 95th percentile at most 130% of capacity (always true if the capacity is unknown).
    Without at least 200 ResOpsUS storage values, a plausible series is accepted as unvalidated.
    Otherwise the series is accepted in the first of these cases that applies:
      1. Its median is within 1/3 to 3 times the ResOpsUS median: accepted as it is.
      2. Dividing by a unit factor (1000, 810.7 = acre-feet per MCM, 1e6, or their inverses)
         brings the median within 1/3 to 3 and the corrected series is plausible against
         capacity: accepted corrected. If cases 1 and 2 both hold, the factor that brings the
         ratio closest to 1 wins, with a small penalty for any factor other than 1.
      3. Tight match, series plausible against capacity, and its median inside 0.7 times the
         0.5th percentile to 1.3 times the 99.5th percentile of the dam's whole ResOpsUS
         record: accepted. This covers droughts and changes of operation (Medina Lake was at
         about 3% of capacity in 2023).
      4. Tight match, series plausible against capacity, and the ResOpsUS median outside 2% to
         120% of capacity: the series is accepted and the note marks ResOpsUS as suspect
         (Center Hill: ResOpsUS holds acre-feet divided by 1e6 where it should hold MCM).
    """
    if s_mcm is None or len(s_mcm) < 250: return None, f"short({0 if s_mcm is None else len(s_mcm)})"
    if s_mcm.std() == 0: return None, "constant"
    A = s_mcm.median()
    capok = lambda x: cap is None or not (cap > 0) or (0.005*cap <= x.median() and x.quantile(.95) <= 1.3*cap)
    if hist is None or hist.storage.notna().sum() < 200:
        return (s_mcm, "unvalidated(no ResOps), capacity-plausible") if capok(s_mcm) else (None, "no ResOps and implausible vs capacity")
    M = hist.storage.median()
    if M <= 0 or A <= 0: return None, "nonpositive"
    best = None
    for f, tag in UNITFIX:
        r = (A/f)/M
        if not (1/3 <= r <= 3): continue
        fixed = s_mcm/f
        if tag != "ok" and not capok(fixed): continue
        score = abs(math.log(r)) + (0 if tag == "ok" else 0.25)
        if best is None or score < best[0]: best = (score, fixed, r, tag)
    if best:
        return best[1], f"ratio={best[2]:.2f} unit:{best[3]}"
    fs = FULL_STOR.get(id(hist), hist.storage.dropna())
    if tight and capok(s_mcm):
        lo, hi = fs.quantile(.005), fs.quantile(.995)
        if 0.7*lo <= A <= 1.3*hi:
            return s_mcm, f"drought/regime shift: median {A:.1f} in ResOps range [{lo:.1f},{hi:.1f}]"
        if cap and cap > 0 and not (0.02*cap <= M <= 1.2*cap):
            return s_mcm, f"ResOps SUSPECT (median {M:.2f} vs capacity {cap:.0f}); raw kept"
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
                # The ResOpsUS elevation itself can be on another datum or wrong (Narrows:
                # 165 m at the gauge, 123 m in ResOpsUS). For a tight match, compare the
                # amplitude instead: level anomaly x lake area against the ResOpsUS storage
                # standard deviation.
                hs = hist.storage.dropna()
                if tight and lake_km2 and lake_km2 > 0 and len(hs) > 200 and hs.std() > 0:
                    amp = (e_m.std()*lake_km2)/hs.std()
                    if 1/3 <= amp <= 3:
                        return (e_m - e_m.median())*lake_km2, f"static PLD area (ResOps elev off; amplitude ratio {amp:.2f})"
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

    With HARVEST_NODES set, the named nodes are processed instead. Writes harvested<tag>.csv and
    harvest_log<tag>.csv (see the module docstring).
    """
    tgt = pd.read_csv(F/"resops_targets_full.csv", dtype={"node": str})
    have = set(pd.read_csv(F/"agency_storage_2023_2025_union.csv", dtype={"node": str}).node)
    for hf in F.glob("harvested*.csv"):          # skip dams that earlier runs already found
        have |= set(pd.read_csv(hf, dtype={"node": str}, usecols=["node"]).node)
    miss = tgt[~tgt.node.isin(have)].drop_duplicates("node").copy()
    import os
    if os.environ.get("HARVEST_NODES"):         # process the named nodes even if an earlier run accepted them
        miss = tgt[tgt.node.isin(os.environ["HARVEST_NODES"].split(","))].drop_duplicates("node").copy()
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

    import geopandas as gpd
    gdw = gpd.read_file(f"{NAS_WATER}/cjgleason/ted/GDW/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp",
                        columns=["GRAND_ID", "RES_NAME", "ALT_NAME"], ignore_geometry=True)
    gdw["GRAND_ID"] = pd.to_numeric(gdw.GRAND_ID, errors="coerce")
    gnames = gdw.dropna(subset=["GRAND_ID"]).drop_duplicates("GRAND_ID").set_index("GRAND_ID")
    pld = gpd.read_file(f"{NAS_WATER}/cjgleason/ted/LakeSPvC_ts_for_Ted/SWOT_PLD_v106_reservoirs.gdb",
                        columns=["lake_id", "names"], ignore_geometry=True)
    pld["node"] = pd.to_numeric(pld.lake_id, errors="coerce").dropna().astype("int64").astype(str)
    pnames = pld.dropna(subset=["node"]).drop_duplicates("node").set_index("node").names
    capx = pd.read_parquet(C/"attributes/static_conus_storage.parquet", columns=["dam_cap_mcm"]).reset_index()
    capx["subbasin"] = capx.subbasin.astype(str); cap = capx.drop_duplicates("subbasin").set_index("subbasin").dam_cap_mcm
    def aliases(d):
        al = [d.DAM_NAME]
        try:
            g = gnames.loc[float(d.grand_id)]
            al += [x for x in (g.RES_NAME, g.ALT_NAME) if isinstance(x, str) and x.strip()]
        except Exception: pass
        x = pnames.get(d.node)
        if isinstance(x, str) and x.strip(): al += [n for n in re.split(r"[;,/]", x) if n.strip()]
        return al
    def asim(al, other): return max(sim(a, other) for a in al)

    def candidates(d):
        AL = aliases(d)
        dl = np.array([km(d.LAT, d.LONG, a, b) if not (np.isnan(a) or np.isnan(b)) else 1e9
                       for a, b in zip(Klat, Klon)])
        near = np.where(dl <= 30)[0]
        rows = []
        for i in near:
            ns = asim(AL, K.name.iat[i])
            if dl[i] <= 5 or ns >= 0.6:
                rows.append((i, dl[i], ns))
        # catalog entries without coordinates: matched by name only, with a stricter threshold
        nocoord = np.where(np.isnan(Klat))[0]
        in_tx = 25.5 <= d.LAT <= 37.0 and -107.0 <= d.LONG <= -93.3   # TWDB covers Texas only (without the box, Wateree matched the Texas Cedar Creek)
        for i in nocoord:
            if K.source.iat[i] == "TWDB" and not in_tx: continue
            ns = asim(AL, K.name.iat[i])
            if ns >= 0.85: rows.append((i, 999.0, ns))
        def quality(i):
            # Among series that tie on name and distance, the observed or revised one is
            # often a single series among many raw sensor feeds (Willamette: 1 of about 40).
            sid = str(K.sid.iat[i]).lower()
            if re.search(r"rev|obs|best|mixed|comp|ccp|man-", sid): return 0
            if "raw" in sid: return 2
            return 1
        rows.sort(key=lambda r: (r[1] > 30, -round(r[2], 1), round(r[1]), quality(r[0])))   # series with coordinates first (by name alone, Willow Creek in Oregon matched Wills Creek in Ohio)
        stor = [r for r in rows if K.kind.iat[r[0]] == "storage"][:10]
        elev = [r for r in rows if K.kind.iat[r[0]] != "storage"][:10]
        return stor + elev          # storage first, but elevation series always get their own 10 slots

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
                tight = (dist <= 3 and ns >= 0.6) or (dist <= 1.0) or (ns >= 0.9 and dist <= 15)
                if kind == "storage":
                    out, note = validate_storage(v, hist, cap.get(d.node), tight)
                else:
                    out, note = validate_elevation(v, hist, area.get(d.node), tight=tight)
                if out is None:
                    log["note"] += f" | {src}:{sid[:30]} {note}"; continue
                log.update(ok=True, source=src, sid=sid, method=f"{kind}: {note}",
                           note=f"{dist:.1f} km, name sim {ns:.2f}")
                return log, pd.DataFrame(dict(node=d.node, date=out.index.date, storage_mcm=out.values,
                                              source=f"{src}:{kind}", sid=sid, method=note))
        except Exception as e:
            import traceback; tb = [f for f in traceback.extract_tb(e.__traceback__) if f.filename.endswith("harvest_v2.py")][-1]
            log["note"] = f"error {type(e).__name__} at line {tb.lineno}: {e}"
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
