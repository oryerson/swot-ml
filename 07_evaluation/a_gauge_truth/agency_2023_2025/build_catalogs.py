"""List every reservoir storage and water-level series that four US agency services offer.

The harvest scripts (harvest.py, harvest_v2.py) match each ResOpsUS dam against these catalogs
by distance and name. All sources are searched for every dam, whichever agency ResOpsUS took
its own record from.

Writes ($SWOTGAT_WORK/CONUS_storage_fork)
  catalogs/nrcs.csv, cwms.csv, rise.csv, nwis.csv, all with the same columns:
    source, sid (series id), name, lat, lon, kind (storage | elevation), unit, meta
  agency_cache/<tag>.txt   every HTTP response, reused for 7 days

Usage
  python build_catalogs.py [nrcs] [cwms] [rise] [nwis]     (no argument: all four)

Method
  nrcs  NRCS AWDB station list for the element RESC (reservoir storage), inactive stations
        included. Unit acre-feet.
  cwms  USACE CWMS Data API. For every district office (codes LR?, MV?, NA?, NW?, PO?, SA?, SP?,
        SW? and LCRA, NWDP, NWDM), the time-series catalog entries whose parameter starts with
        Stor or Elev. Coordinates come from the office's location list (exact location name, else
        the part before the first "-"). A series whose latest value is before 2023-06-01 is
        skipped. `meta` holds office, earliest date and latest date.
  rise  Reclamation RISE catalog items whose title contains "daily" and "Reservoir Storage"
        (storage) or a reservoir elevation phrase (elevation), and whose record ends on or
        after 2023-06-01. Coordinates need two more requests per item (first result, then its
        location).
  nwis  USGS NWIS site service, for each of the 48 conterminous states, parameter codes 00054
        (reservoir storage, acre-feet) and 00062 (reservoir elevation, feet), daily (dv) and
        instantaneous (iv) series that end on or after 2023-06-01. The series id is
        "site|parameter|dv or iv".

Notes
  Downloads go through swotgat.http: a generic client name and at most one request per 1.1 s
  per host, however many threads are used.
  nwis_v2.py and rise_by_name.py write catalogs/nwis.csv and catalogs/rise.csv again with
  different rules (more USGS parameter codes; RISE without the coordinate requests). Whichever
  runs last decides which catalog the harvest reads.
  The Colorado catalog is built by cat_codwr.py.
"""
from swotgat.paths import WORK
import json, re, time, sys, io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, requests
F = Path(f"{WORK}/CONUS_storage_fork")
CACHE = F/"agency_cache"; CACHE.mkdir(exist_ok=True)
CAT = F/"catalogs"; CAT.mkdir(exist_ok=True)
from swotgat.http import PoliteSession; S = PoliteSession()

def get(url, tag, headers=None, ttl=7):
    """Fetch a URL through the file cache and return the response text.

    `tag` names the cache file. A cached file younger than `ttl` days is returned without a
    request. Returns "" after a 400/404 or four failed tries.
    """
    f = CACHE/(re.sub(r"[^A-Za-z0-9_.-]", "_", tag)[:180]+".txt")
    if f.is_file() and time.time()-f.stat().st_mtime < ttl*86400: return f.read_text()
    for a in range(4):
        try:
            r = S.get(url, timeout=120, headers=headers)
            if r.status_code == 200: f.write_text(r.text); return r.text
            if r.status_code in (400, 404): return ""
        except Exception: pass
        time.sleep(3*(a+1))
    return ""

def nrcs():
    """Return the catalog of NRCS AWDB reservoir storage stations (acre-feet), active or not."""
    txt = get("https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/stations?elements=RESC&activeOnly=false", "nrcs_resc_all")
    rows = [dict(source="NRCS", sid=s["stationTriplet"], name=s["name"], lat=s.get("latitude"),
                 lon=s.get("longitude"), kind="storage", unit="ac-ft", meta=s.get("networkCode",""))
            for s in json.loads(txt)]
    return pd.DataFrame(rows)

def cwms():
    """Return the catalog of USACE CWMS storage and elevation series, over all district offices.

    Coordinates and names come from each office's location list. A series whose latest value is
    before 2023-06-01 is skipped.
    """
    offs = [o["name"] for o in json.loads(get("https://cwms-data.usace.army.mil/cwms-data/offices", "cwms_offices"))]
    dist = [o for o in offs if re.fullmatch(r"(LR|MV|NA|NW|PO|SA|SP|SW)[A-Z]|LCRA|NWDP|NWDM", o)]
    def one(o):
        loc = get(f"https://cwms-data.usace.army.mil/cwms-data/locations?office={o}&page-size=20000", f"cwms_loc_{o}")
        cat = get(f"https://cwms-data.usace.army.mil/cwms-data/catalog/TIMESERIES?office={o}&page-size=20000&like=.*%5C.(Stor|Elev)%5B%5E.%5D*%5C..*",
                  f"cwms_catSE_{o}")
        try: L = json.loads(loc); L = L if isinstance(L, list) else L.get("locations", [])
        except Exception: L = []
        try: E = json.loads(cat).get("entries", [])
        except Exception: E = []
        ll = {x["name"]: (x.get("latitude"), x.get("longitude"), x.get("public-name") or x.get("long-name") or x["name"]) for x in L}
        out = []
        for e in E:
            ts = e["name"]; locname = ts.split(".")[0]; param = ts.split(".")[1]
            base = locname.split("-")[0]
            la, lo, pn = ll.get(locname, ll.get(base, (None, None, locname)))
            latest = max([x.get("latest-time","") for x in e.get("extents",[])] or [""])
            earliest = min([x.get("earliest-time","9999") for x in e.get("extents",[])] or ["9999"])
            if latest and latest < "2023-06-01": continue
            kind = "storage" if param.startswith("Stor") else "elevation"
            out.append(dict(source=f"CWMS", sid=f"{o}|{ts}", name=pn, lat=la, lon=lo, kind=kind,
                            unit=e.get("units",""), meta=f"{o}|{earliest[:10]}|{latest[:10]}"))
        return o, out, len(L), len(E)
    rows = []
    with ThreadPoolExecutor(8) as ex:
        for o, out, nl, ne in ex.map(one, dist):
            print(f"   CWMS {o:<5} locations {nl:>6}  Stor/Elev series {ne:>5}  live in window {len(out):>5}", flush=True)
            rows += out
    return pd.DataFrame(rows)

def rise():
    """Return the catalog of Reclamation RISE daily reservoir storage and elevation items.

    Only items whose record ends on or after 2023-06-01 are kept. Coordinates and names come from
    two more requests per item; an item whose location cannot be read is dropped.
    """
    H = {"Accept": "application/vnd.api+json"}
    first = json.loads(get("https://data.usbr.gov/rise/api/catalog-item?itemsPerPage=100&page=1", "rise_items_p1", H))
    npages = -(-first["meta"]["totalItems"] // 100)
    items = []
    def page(p):
        t = get(f"https://data.usbr.gov/rise/api/catalog-item?itemsPerPage=100&page={p}", f"rise_items_p{p}", H)
        return json.loads(t)["data"] if t else []
    with ThreadPoolExecutor(6) as ex:
        for d in ex.map(page, range(1, npages+1)): items += d
    keep = []
    for it in items:
        a = it["attributes"]; t = a.get("itemTitle", "")
        if not re.search(r"daily", t, re.I): continue
        if re.search(r"Reservoir Storage", t): kind = "storage"
        elif re.search(r"Reservoir (Elevation|WS Elevation)|Lake/Reservoir Elevation", t): kind = "elevation"
        else: continue
        if (a.get("temporalEndDate") or "") < "2023-06-01": continue
        unit = "af" if "-af" in t else ("ft" if "-ft" in t else "")
        keep.append((a["_id"], t, kind, unit, a.get("locationSourceCode","")))
    print(f"   RISE items {len(items):,}; daily storage/elevation live in window: {len(keep):,}", flush=True)
    def loc(k):
        iid = k[0]
        r = get(f"https://data.usbr.gov/rise/api/result?itemId={iid}&itemsPerPage=1", f"rise_res1_{iid}", H)
        try: lid = json.loads(r)["data"][0]["attributes"]["locationId"]
        except Exception: return None
        L = get(f"https://data.usbr.gov/rise/api/location/{lid}", f"rise_loc_{lid}", H)
        try:
            at = json.loads(L)["data"]["attributes"]; c = at.get("locationCoordinates") or {}
            co = c.get("coordinates") if isinstance(c, dict) else None
            lon, lat = (co[0], co[1]) if co and len(co) >= 2 else (None, None)
            return dict(source="RISE", sid=str(iid), name=at.get("locationName") or k[1], lat=lat, lon=lon,
                        kind=k[2], unit=k[3], meta=f"{k[4]}|{k[1][:80]}")
        except Exception: return None
    rows = []
    with ThreadPoolExecutor(8) as ex:
        for r in ex.map(loc, keep):
            if r: rows.append(r)
    return pd.DataFrame(rows)

def nwis():
    """Return the catalog of USGS NWIS reservoir storage (00054) and elevation (00062) series.

    Covers the 48 conterminous states, daily and instantaneous series that end on or after
    2023-06-01, one row per series id.
    """
    STATES = ["al","az","ar","ca","co","ct","de","fl","ga","id","il","in","ia","ks","ky","la","me","md","ma","mi",
              "mn","ms","mo","mt","ne","nv","nh","nj","nm","ny","nc","nd","oh","ok","or","pa","ri","sc","sd","tn",
              "tx","ut","vt","va","wa","wv","wi","wy"]
    rows = []
    def one(args):
        st, pc, dt = args
        t = get(f"https://waterservices.usgs.gov/nwis/site/?format=rdb&stateCd={st}&seriesCatalogOutput=true"
                f"&outputDataTypeCd={dt}&parameterCd={pc}", f"nwis_{st}_{pc}_{dt}")
        out = []
        if not t.strip(): return out
        L = [l for l in t.splitlines() if l and not l.startswith("#")]
        try: d = pd.read_csv(io.StringIO("\n".join(L)), sep="\t", dtype=str)
        except Exception: return out
        d = d[d.site_no.notna() & (d.site_no != "5s") & (d.parm_cd == pc)]
        for _, r in d.iterrows():
            if str(r.end_date) < "2023-06-01": continue
            out.append(dict(source="NWIS", sid=f"{r.site_no}|{pc}|{dt}", name=r.station_nm,
                            lat=pd.to_numeric(r.dec_lat_va, errors="coerce"), lon=pd.to_numeric(r.dec_long_va, errors="coerce"),
                            kind="storage" if pc=="00054" else "elevation", unit="ac-ft" if pc=="00054" else "ft", meta=dt))
        return out
    jobs = [(s, p, d) for s in STATES for p in ("00054","00062") for d in ("dv","iv")]
    with ThreadPoolExecutor(6) as ex:
        for o in ex.map(one, jobs): rows += o
    d = pd.DataFrame(rows).drop_duplicates(["sid"])
    return d

if __name__ == "__main__":
    which = sys.argv[1:] or ["nrcs","cwms","rise","nwis"]
    for w in which:
        t0 = time.time(); print(f"== {w}", flush=True)
        d = globals()[w]()
        d.to_csv(CAT/f"{w}.csv", index=False)
        print(f"   {w}: {len(d):,} series ({d.lat.notna().sum():,} with coords) in {time.time()-t0:.0f}s\n", flush=True)
