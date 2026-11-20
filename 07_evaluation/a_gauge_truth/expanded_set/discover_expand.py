"""List the agency series that may record storage at each dam of the expanded gauge set.

Step 2 of the expanded set. For every dam it searches the agency catalogs by position and by name for
daily storage series. Elevation series are listed only for dams that have a ResOpsUS record, because
an elevation series needs that record to be converted to storage. Nothing is judged here:
harvest_expand.py downloads the listed series and build_plus.py decides which to use.

Folder names call the test sets "regimes": regime 3 = 1980-2020 (ResOpsUS), regime 1x = 2023-2026
(agency gauges).

Reads (under $SWOTGAT_WORK/CONUS_storage_fork unless noted)
  gauges_expand/sizing/candidates.csv            from size_candidates.py
  gauges_expand/crosswalk_plus/dams_in.csv       rule-model GRanD dams outside the frozen crosswalk
                                                 (in the paper's run: 947)
  testsets/regime1x_2023_2026/candidate_additions/sites.csv
  gauges_expand/cache/nwissite_*                 USGS site catalogs cached by size_candidates.py
  gauges2026/cache/cwmsloc_*.txt                 USACE CWMS location catalogs cached by the 2026 harvest
  $SWOTGAT_WORK/GRADES/gdrom_v2/file_list.json   the files of the GDROM v2 data set on HydroShare
  ResOpsUS/time_series_all/                      to see which dams have a record
  under $SWOTGAT_NAS_WATER: the Global Dam Watch (GDW) v1.0 reservoir shapefile (alternative names)

Writes (gauges_expand/discovery/)
  dams.csv       the dams, in two groups (see Method), with names, capacity and flags
  series.csv     one row per dam and candidate series: source, series id, kind (storage or
                 elevation), distance, period of record, note

Usage
  python discover_expand.py        (no arguments)

Method
  Dams. Group "new": the dams of dams_in.csv. Group "gap23": dams with a GRanD id (GRAND_ID below
  10000) that are High or Medium in regime 3 but have no High or Medium regime-1x series, and are not
  among the extended candidate additions (ext_status "extended"; 18 in the paper's run).
  Sources and search rules:
    RESOPS   the dam's own ResOpsUS record, if it has one
    NWISANY  USGS daily series of parameter 00054 (storage) or 62614, 62615, 00062, 72020 (elevation)
             within 6 km
    NRCS     AWDB stations with element RESC (reservoir storage) within 8 km
    CODWR    Colorado telemetry stations with parameter STORAGE within 5 km
    CDEC     California stations with daily sensor 15 (storage) within 5 km, from the station search
             page
    RISE     Reclamation catalog items whose title contains "Storage" and "daily" (not live, active
             or dead storage, not change) at RISE locations within 5 km
    TWDB     Texas reservoirs (in the paper's run: 122), matched by name for dams in Texas: name
             similarity of at least 0.8, the best two
    GDROMV2  the "collected data" file of GDROM v2 on HydroShare, for dams whose GDROM storage source
             is an observation (usace, gdromv1 or resopsus)
    CWMS     USACE locations within 4 km, except tailwater, spillway, gate and similar locations. The
             time-series catalog of each location is requested. Kept: storage parameters other than
             flood, conservation, inactive, dead, exclusive or surcharge storage, and pool elevation
             parameters. Forecast, model, rule-curve and zone series are dropped.

Notes
  All requests go through gx_common.get: swotgat.http (generic client name, at most one request per
  1.1 s per host, 5 s for HydroShare) and a cache of every response. The robots.txt files of CWMS,
  CDEC, TWDB and RISE disallow automated access to the paths requested here (see size_candidates.py).
  Check each service's current terms before you run this again.
"""
from swotgat.paths import NAS_WATER
import io, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); from gx_common import *
O = X/"discovery"; O.mkdir(exist_ok=True)
C = pd.read_csv(X/"sizing/candidates.csv").drop_duplicates("GRAND_ID").set_index("GRAND_ID")
D = pd.read_csv(X/"crosswalk_plus/dams_in.csv"); D["group"] = "new"
CA = pd.read_csv(F/"testsets/regime1x_2023_2026/candidate_additions/sites.csv"); ext = set(CA[CA.ext_status == "extended"].DAM_ID)
hm = lambda s: s.isin(["High", "Medium"])
g23 = C[hm(C.r3_tier) & ~hm(C.r1x_tier) & ~C.index.isin(ext) & (C.index < 10000)]
D = pd.concat([D, pd.DataFrame(dict(DAM_ID=g23.index, DAM_NAME=g23.name, STATE=g23.st, LAT=g23.lat, LONG=g23.lon, record_status="resops_tested", group="gap23"))], ignore_index=True)
import geopandas as gpd
_gw = gpd.read_file(f"{NAS_WATER}/cjgleason/ted/GDW/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp", columns=["GRAND_ID", "RES_NAME", "DAM_NAME", "ALT_NAME"], ignore_geometry=True)
_gw["GRAND_ID"] = pd.to_numeric(_gw.GRAND_ID, errors="coerce"); _gw = _gw[_gw.GRAND_ID > 0].drop_duplicates("GRAND_ID").set_index("GRAND_ID")
D["names"] = [";".join(str(x) for x in [n] + ([_gw.at[i, c] for c in ("RES_NAME", "DAM_NAME", "ALT_NAME")] if i in _gw.index else []) if isinstance(x, str) and x.strip())
              for i, n in zip(D.DAM_ID, D.DAM_NAME)]
D["cap_mcm"] = D.DAM_ID.map(C.cap_mcm); D["gdrom_src"] = D.DAM_ID.map(C.gdrom_src).fillna("")
D["resops"] = D.DAM_ID.map(lambda d: (F/f"ResOpsUS/time_series_all/ResOpsUS_{d}.csv").exists())
D.to_csv(O/"dams.csv", index=False); print("dams", len(D), D.group.value_counts().to_dict(), flush=True)
S = []
def add(dam, source, sid, kind, km=np.nan, b=None, e=None, note=""):
    """Append one candidate series of a dam to the list S.

    kind is "storage" or "elevation", km the distance from the dam to the station, b and e the begin and
    end of the period of record, and note a free text (station name or how the series was matched).
    """
    S.append(dict(DAM_ID=dam, source=source, sid=sid, kind=kind, km=km, begin=b, end=e, note=note))

# ---------------- ResOps
for _, d in D[D.resops].iterrows(): add(d.DAM_ID, "RESOPS", str(d.DAM_ID), "storage", 0.0, note="ResOpsUS record")
# ---------------- NWIS (sizing catalog)
nw = []
for f in sorted(CACHE.glob("nwissite_*.txt.gz")) + sorted((X/"cache").glob("nwissite_*.txt")):
    t = gzip.decompress(f.read_bytes()).decode() if f.suffix == ".gz" else f.read_text()
    L = [l for l in t.splitlines() if l and not l.startswith("#")]
    if len(L) > 2: nw.append(pd.read_csv(io.StringIO("\n".join([L[0]] + L[2:])), sep="\t", dtype=str))
NW = pd.concat(nw).drop_duplicates(); NW = NW[(NW.data_type_cd == "dv") & NW.parm_cd.isin(["00054", "62614", "62615", "00062", "72020"])]
NW["lat"] = NW.dec_lat_va.astype(float); NW["lon"] = NW.dec_long_va.astype(float)
print("NWIS series", len(NW), flush=True)
# ---------------- NRCS / CO DWR
def jl(tag):
    """Return the cached response stored under a tag, compressed or not ("" if there is none)."""
    for f in (CACHE/(tag + ".txt.gz"), X/"cache"/(tag + ".txt")):
        if f.exists(): return gzip.decompress(f.read_bytes()).decode() if f.suffix == ".gz" else f.read_text()
    return ""
NR = pd.DataFrame(json.loads(get("https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/stations?elements=RESC&returnReservoirMetadata=true&activeOnly=false",
                                 "nrcs_stations_resc") or "[]"))
cw = []
for p in range(1, 4):
    t = get(f"https://dwr.state.co.us/Rest/GET/api/v2/telemetrystations/telemetrystation/?format=json&parameter=STORAGE&pageSize=50000&pageIndex={p}", f"codwr_stations_{p}")
    if not t: break
    j = json.loads(t); cw += j.get("ResultList", [])
    if p >= j.get("PageCount", 1): break
CD = pd.DataFrame(cw); CD = CD[CD.parameter.astype(str).str.upper().str.contains("STOR")] if len(CD) else CD
# ---------------- CDEC sensor-15 daily stations
t = get("https://cdec.water.ca.gov/dynamicapp/staSearch?sta=&sensor_chk=on&sensor=15&collect=NONE+SPECIFIED&dur_chk=on&dur=D&active=&lon1=&lon2=&lat1=&lat2="
        "&elev1=-5&elev2=99000&nearby=&basin=NONE+SPECIFIED&hydro=NONE+SPECIFIED&county=NONE+SPECIFIED&agency_num=160&display=sta", "cdec_stasearch_s15_D")
rows = []
for tr in re.findall(r"<tr.*?</tr>", t, flags=re.S):
    sid = re.search(r"station_id=([A-Z0-9]{3})", tr); tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S)
    if sid and len(tds) >= 6:
        txt = [re.sub(r"<[^>]+>", "", x).strip() for x in tds]
        nums = [x for x in txt if re.fullmatch(r"-?\d+\.\d+", x)]
        if len(nums) >= 2: rows.append(dict(sid=sid.group(1), name=txt[1], lon=float(nums[0]), lat=float(nums[1])))
CE = pd.DataFrame(rows).drop_duplicates("sid"); print("CDEC s15 daily stations", len(CE), flush=True)
# ---------------- RISE locations + daily storage catalog items
H = {"Accept": "application/vnd.api+json"}
def rise_pages(kind):
    """Return every record of one RISE API collection (kind is "location" or "catalog-item").

    RISE returns at most 100 records per page, so the pages are followed to the last one.
    """
    out, p = [], 1
    while True:
        j = json.loads(get(f"https://data.usbr.gov/rise/api/{kind}?itemsPerPage=100&page={p}", f"rise_{kind}_100_p{p}", H) or '{"data":[]}')
        out += j.get("data", [])
        if not j.get("data") or "next" not in (j.get("links") or {}): return out
        p += 1
RL = rise_pages("location")
RL = pd.DataFrame([dict(loc=int(x["attributes"]["_id"]), name=x["attributes"]["locationName"], lon=x["attributes"]["locationCoordinates"]["coordinates"][0],
                        lat=x["attributes"]["locationCoordinates"]["coordinates"][1]) for x in RL
                   if (x["attributes"].get("locationCoordinates") or {}).get("type") == "Point"])
items = []
for x in rise_pages("catalog-item"):
    a = x["attributes"]; rel = x.get("relationships", {})
    loc = None
    for k in ("catalogRecord", "location"):
        dd = (rel.get(k) or {}).get("data")
        if isinstance(dd, dict) and "id" in dd: loc = (k, dd["id"])
    items.append(dict(item=int(a["_id"]), title=a.get("itemTitle", ""), b=a.get("temporalStartDate"), e=a.get("temporalEndDate"), rel=loc,
                      locsrc=a.get("locationSourceCode")))
RI = pd.DataFrame(items); RI = RI[RI.title.str.contains("Storage", case=False) & RI.title.str.contains("daily", case=False) & ~RI.title.str.contains("Live|Active|Dead|Change", case=False)]
print("RISE locations", len(RL), "| daily storage items", len(RI), flush=True)
def rec_loc(rid):
    """Return the RISE location id of a catalog record, or None if the record names no location.

    rid is the API path of the record. One request is made per record that a storage item uses.
    """
    j = json.loads(get(f"https://data.usbr.gov/rise{rid}" if rid.startswith("/api") else f"https://data.usbr.gov{rid}", "rise_rec_" + rid.replace("/", "_"), H) or "{}")
    d = ((j.get("data") or {}).get("relationships") or {}).get("location", {}).get("data") or {}
    m = re.search(r"(\d+)$", str(d.get("id", ""))); return int(m.group(1)) if m else None
RI["loc"] = [int(re.search(r"(\d+)$", r[1]).group(1)) if r and r[0] == "location" else (rec_loc(r[1]) if r else None) for r in RI.rel]
RI = RI.merge(RL, on="loc", how="left")
# ---------------- TWDB reservoir slugs
t = get("https://www.waterdatafortexas.org/reservoirs/statewide", "twdb_statewide")
TW = pd.DataFrame({"slug": sorted(set(re.findall(r"reservoirs/individual/([a-z0-9-]+)", t)))}); TW["name"] = TW.slug.str.replace("-", " ")
# ---------------- GDROM v2 collected series
FL = json.load(open(F.parent/"GRADES/gdrom_v2/file_list.json"))
GV = {int(m.group(1)): x["url"] for x in FL if "collected%20data" in x["url"] or "collected data" in x["url"] for m in [re.search(r"/(\d+)\.csv$", x["url"])] if m}
print("GDROM v2 collected files", len(GV), flush=True)
# ---------------- CWMS locations (cached) -> timeseries catalog
cl = []
for f in (F/"gauges2026/cache").glob("cwmsloc_*.txt"):
    try: cl += json.loads(f.read_text()).get("entries", [])
    except Exception: pass
CL = pd.DataFrame(cl); CL = CL[(CL.latitude.abs() > 1) & (CL.longitude.abs() > 1)]
BAD_LOC = re.compile(r"Tailwater|Tail|Spillway|Gate|Rereg|Downstream|Outlet|Intake|Powerhouse|-TW|_TW|Unit|Lock", re.I)
BAD_VER = re.compile(r"Crest|Fcst|Forecast|RAS|M0D0R0|Model|Guide|Rule|Target|Flood|Bottom|Top|Zone|Avail|Space|Encroach|Pct|Percent", re.I)
ELEV_OK = ("Elev", "Elev-Pool", "Elev-Forebay", "Elev-Lake", "Elev-Headwater", "Elev-Reservoir")
def cwms_ts(office, loc):
    """Return the entries of the CWMS time-series catalog for one location of a USACE office.

    Returns an empty list if the response cannot be read.
    """
    like = requests.utils.quote(re.escape(loc) + r"\..*", safe="")
    t = get(f"https://cwms-data.usace.army.mil/cwms-data/catalog/TIMESERIES?office={office}&like={like}&page-size=1000",
            f"cattsx_{office}_{loc}", {"Accept": "application/json;version=2"})
    try: return json.loads(t).get("entries", [])
    except Exception: return []
for _, d in D.iterrows():
    dam, lat, lon = int(d.DAM_ID), float(d.LAT), float(d.LONG); ref = bool(d.resops)
    if not np.isfinite(lat): continue
    n = NW[(NW.lat - lat).abs().lt(0.08) & (NW.lon - lon).abs().lt(0.1)].copy(); n["km"] = hav(lat, lon, n.lat, n.lon)
    for _, s in n[n.km <= 6].iterrows():
        k = "storage" if s.parm_cd == "00054" else "elevation"
        if k == "elevation" and not ref: continue
        add(dam, "NWISANY", f"{s.site_no}|{s.parm_cd}", k, s.km, s.begin_date, s.end_date, s.station_nm)
    m = NR[(NR.latitude - lat).abs().lt(0.08) & (NR.longitude - lon).abs().lt(0.1)].copy(); m["km"] = hav(lat, lon, m.latitude, m.longitude)
    for _, s in m[m.km <= 8].iterrows(): add(dam, "NRCS", s.stationTriplet, "storage", s.km, s.beginDate, s.endDate, s["name"])
    if len(CD):
        m = CD[(CD.latitude.astype(float) - lat).abs().lt(0.06) & (CD.longitude.astype(float) - lon).abs().lt(0.08)].copy()
        m["km"] = hav(lat, lon, m.latitude.astype(float), m.longitude.astype(float))
        for _, s in m[m.km <= 5].iterrows(): add(dam, "CODWR", str(s.abbrev), "storage", s.km, s.get("stationPorStart"), s.get("stationPorEnd"), s.stationName)
    if len(CE):
        m = CE[(CE.lat - lat).abs().lt(0.06) & (CE.lon - lon).abs().lt(0.08)].copy(); m["km"] = hav(lat, lon, m.lat, m.lon)
        for _, s in m[m.km <= 5].iterrows(): add(dam, "CDEC", s.sid, "storage", s.km, note=s["name"])
    m = RI[(RI.lat - lat).abs().lt(0.06) & (RI.lon - lon).abs().lt(0.08)].copy()
    if len(m):
        m["km"] = hav(lat, lon, m.lat, m.lon)
        for _, s in m[m.km <= 5].iterrows(): add(dam, "RISE", str(s["item"]), "storage", s.km, s["b"], s["e"], s["title"][:80])
    if str(d.STATE) in ("TX", "Texas"):
        sc = TW.assign(s=[max(sim(nm, x) for nm in str(d.names).split(";")) for x in TW.name]).sort_values("s", ascending=False)
        for _, s in sc[sc.s >= 0.8].head(2).iterrows(): add(dam, "TWDB", s.slug, "storage", np.nan, note=f"name sim {s.s:.2f}")
    if dam in GV and re.search("usace|gdromv1|resopsus", d.gdrom_src or ""): add(dam, "GDROMV2", str(dam), "storage", 0.0, note=d.gdrom_src)
    m = CL[(CL.latitude - lat).abs().lt(0.04) & (CL.longitude - lon).abs().lt(0.05)].copy()
    if len(m):
        m["km"] = hav(lat, lon, m.latitude, m.longitude)
        for _, l in m[m.km <= 4].iterrows():
            if BAD_LOC.search(str(l["name"])) or BAD_LOC.search(str(l.get("long-name", ""))): continue
            for e in cwms_ts(l.office, l["name"]):
                nm = e.get("name", ""); parts = nm.split(".")
                if len(parts) < 6 or BAD_VER.search(parts[5]) or BAD_VER.search(parts[1]): continue
                par = parts[1]
                if par.startswith("Stor") and not re.search(r"Flood|Cons|Inactive|Dead|Excl|Surch", par): k = "storage"
                elif par in ELEV_OK and ref: k = "elevation"
                else: continue
                ext = (e.get("extents") or [{}])[0]
                add(dam, "CWMS", f"{l.office}|{nm}", k, l.km, ext.get("earliest-time"), ext.get("latest-time"), parts[3] + "/" + parts[4])
SR = pd.DataFrame(S).drop_duplicates(["DAM_ID", "source", "sid"]); SR.to_csv(O/"series.csv", index=False)
print(SR.groupby(["source", "kind"]).DAM_ID.agg(["count", "nunique"]).to_string())
print("dams with >=1 non-ResOps storage series:", SR[(SR.source != "RESOPS") & (SR.kind == "storage")].DAM_ID.nunique(), "of", len(D))
