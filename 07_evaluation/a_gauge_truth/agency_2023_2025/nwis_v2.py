"""List the USGS NWIS reservoir storage and lake-level series (wider search than build_catalogs.py).

Writes ($SWOTGAT_WORK/CONUS_storage_fork)
  catalogs/nwis.csv        source (NWIS), sid ("site|parameter|dv or iv"), name, lat, lon, kind,
                           unit, meta (begin date|end date). This replaces the nwis.csv written
                           by `build_catalogs.py nwis`.
  agency_cache/<tag>.txt   every HTTP response, reused for 14 days

Usage
  python nwis_v2.py     (no arguments)

Method
  For each of the 48 conterminous states the USGS site service is asked for series of five
  parameter codes: 00054 (reservoir storage, acre-feet), 72036 (reservoir storage, thousand
  acre-feet), 00062 (reservoir elevation, feet), and 62614 and 62615 (lake elevation above
  NGVD29 and NAVD88, feet). build_catalogs.py asks only for 00054 and 00062, but USGS files
  most lake levels under 62614 and 62615 (Texas alone had 156 active lake gauges under these
  two codes).
  A series is kept when its record ends on or after 2023-06-01 and begins on or before
  2025-06-30, so that gauges installed after the window are left out. When a site has both a
  daily (dv) and an instantaneous (iv) series of the same parameter, the daily one is kept,
  because USGS has already computed the daily means.

Notes
  Downloads go through swotgat.http: a generic client name and at most one request per 1.1 s
  per host, however many threads are used.
"""
from swotgat.paths import WORK
import io, time, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, requests
F = Path(f"{WORK}/CONUS_storage_fork"); CACHE = F/"agency_cache"
from swotgat.http import PoliteSession; S = PoliteSession()
PARAMS = {"00054": ("storage", "ac-ft"), "72036": ("storage", "kaf"),
          "00062": ("elevation", "ft"), "62614": ("elevation", "ft"), "62615": ("elevation", "ft")}
STATES = ["al","az","ar","ca","co","ct","de","fl","ga","id","il","in","ia","ks","ky","la","me","md","ma","mi",
          "mn","ms","mo","mt","ne","nv","nh","nj","nm","ny","nc","nd","oh","ok","or","pa","ri","sc","sd","tn",
          "tx","ut","vt","va","wa","wv","wi","wy"]
def get(url, tag):
    """Fetch a URL through the file cache and return the response text.

    `tag` names the cache file. A cached file younger than 14 days is returned without a
    request. Returns "" after a 400/404 (also cached) or four failed tries.
    """
    f = CACHE/(re.sub(r"[^A-Za-z0-9_.-]", "_", tag)[:180]+".txt")
    if f.is_file() and time.time()-f.stat().st_mtime < 14*86400: return f.read_text()
    for a in range(4):
        try:
            r = S.get(url, timeout=150)
            if r.status_code == 200: f.write_text(r.text); return r.text
            if r.status_code in (400, 404): f.write_text(""); return ""
        except Exception: pass
        time.sleep(3*(a+1))
    return ""
def one(job):
    """Return the catalog rows of one (state, parameter code) pair.

    Each row is a daily or instantaneous series whose record ends on or after 2023-06-01 and
    begins on or before 2025-06-30.
    """
    st, pc = job
    t = get(f"https://waterservices.usgs.gov/nwis/site/?format=rdb&stateCd={st}&seriesCatalogOutput=true"
            f"&outputDataTypeCd=iv,dv&parameterCd={pc}", f"nwis2_{st}_{pc}")
    L = [l for l in t.splitlines() if l and not l.startswith("#")]
    if len(L) < 3: return []
    d = pd.read_csv(io.StringIO("\n".join(L)), sep="\t", dtype=str)
    d = d[d.site_no.notna() & (d.site_no != "5s") & (d.parm_cd == pc)]
    d = d[(d.end_date.astype(str) >= "2023-06-01") & (d.begin_date.astype(str) <= "2025-06-30")]
    out = []
    for _, r in d.iterrows():
        dt = "dv" if r.data_type_cd == "dv" else "iv"
        kind, unit = PARAMS[pc]
        out.append(dict(source="NWIS", sid=f"{r.site_no}|{pc}|{dt}", name=r.station_nm,
                        lat=r.dec_lat_va, lon=r.dec_long_va, kind=kind, unit=unit, meta=f"{r.begin_date}|{r.end_date}"))
    return out
rows = []
with ThreadPoolExecutor(6) as ex:
    for o in ex.map(one, [(s, p) for s in STATES for p in PARAMS]): rows += o
d = pd.DataFrame(rows).drop_duplicates("sid")
# prefer dv over iv for the same site and parameter (USGS has already computed the daily means)
d["site_pc"] = d.sid.str.rsplit("|", n=1).str[0]
d = d.sort_values("sid", key=lambda s: s.str.endswith("|iv")).drop_duplicates("site_pc").drop(columns="site_pc")
d.to_csv(F/"catalogs/nwis.csv", index=False)
d["pc"] = d.sid.str.split("|").str[1]
print(f"NWIS v2: {len(d)} series at {d.sid.str.split('|').str[0].nunique()} sites")
print(d.groupby(["pc", "kind"]).size().to_string())
