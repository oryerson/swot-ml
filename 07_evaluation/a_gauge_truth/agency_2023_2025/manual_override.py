"""Fetch hand-picked agency series for dams that the automatic matching got wrong or missed.

Reads (current folder, which must be $SWOTGAT_WORK/CONUS_storage_fork)
  resops_targets_full.csv   ResOpsUS dams with their graph node
  harvest_v2.py             imported for its fetchers, unit conversion and validators
  $SWOTGAT_WORK/CONUS/attributes/static_conus_storage.parquet   lake area and capacity per node
  ResOpsUS/time_series_all/ResOpsUS_<DAM_ID>.csv   (through harvest_v2.resops_hist)

Writes
  harvested_manual.csv   node, date, storage_mcm, source, sid, method; `method` ends with
                         "[manual pin]". merge_truth.py gives this file the highest priority.

Usage
  python manual_override.py     (no arguments)

Method
  `PIN` lists, per dam name, one or more series (source, series id, kind, unit) in order of
  preference. Two groups of dams are pinned:
  * Dams for which the automatic match picked a neighbouring reservoir: Kentucky (matched
    Barkley), Dry Falls (Grand Coulee), Dexter (Fall Creek), Hopkinton (Everett).
  * Dams the harvest missed because the series was not in a catalog or needed another service:
    Hoover, Beaver, C.J. Strike, Horseshoe, Alvin Wirtz, Max Starcke, Antero, Eleven Mile
    Canyon, Conconully.
  Each pinned series must still pass the harvest_v2 validators, called as a tight match. The
  first series that passes is kept. Three fetchers are added to those of harvest_v2:
    NRCSM    NRCS monthly reservoir storage (month-end dates). Validated by `validate_monthly`:
             at least 24 values and a median within 1/3 to 3 times the ResOpsUS median.
    NWISANY  USGS daily values of any statistic for one site and parameter code.
    LCHDB    Reclamation Lower Colorado hydrologic database, daily values.
  `FORCE` lists series that are accepted even when validation fails, with the reason. It holds
  one series: the Tennessee Valley Authority storage of Kentucky Lake (see the comment at
  `FORCE`).

Notes
  Downloads go through harvest_v2.get, and therefore through swotgat.http: a generic client
  name, at most one request per 1.1 s per host, responses cached in agency_cache/.
"""
from swotgat.paths import WORK
import sys, pandas as pd; sys.path.insert(0, ".")
import harvest_v2 as H
C = f"{WORK}/CONUS/attributes/static_conus_storage.parquet"
st = pd.read_parquet(C, columns=["lake_area_km2", "dam_cap_mcm"]).reset_index()
st["subbasin"] = st.subbasin.astype(str); st = st.drop_duplicates("subbasin").set_index("subbasin")
t = pd.read_csv("resops_targets_full.csv", dtype={"node": str}).drop_duplicates("node").set_index("DAM_NAME")
PIN = {"Kentucky": [("CWMS", "LRN|KYDK2-KENTUCKY.Stor.Inst.1Hour.0.tva-raw", "storage", "m3"),
                    ("CWMS", "LRD|Kentucky-Lake.Stor.Inst.1Hour.0.Best-LRD", "storage", "m3")],
       "Dry Falls": [("RISE", "129465", "storage", "af"), ("CWMS", "NWDP|BNK.Elev-Lake.Inst.~1Day.0.CBT-REV", "elevation", "m")],
       "Dexter": [("CWMS", "NWDP|DEX.Elev-Forebay.Inst.0.0.MIXED-REV", "elevation", "m"),
                  ("CWMS", "NWDP|DEX.Elev-Forebay.Inst.1Hour.0.CBT-REV", "elevation", "m")],
       "Hopkinton Flood Control Dam": [("CWMS", "NAE|HKD.Stor.Inst.15Minutes.0.DCP-rev", "storage", "m3"),
                                       ("CWMS", "NAE|HKD.Elev-Pool.Inst.15Minutes.0.DCP-rev", "elevation", "m")]}
# Kentucky: the Tennessee Valley Authority (TVA) storage fails the ResOpsUS median ratio (2.41),
# but ResOpsUS is the inconsistent record. Its storage sd (161 MCM) over an elevation sd of 0.71 m
# implies a 227 km2 lake. Kentucky Lake is about 550-650 km2, and the TVA sd (347 MCM over 0.59 m)
# agrees with that. The TVA series is therefore accepted on physical grounds.
FORCE = {"LRN|KYDK2-KENTUCKY.Stor.Inst.1Hour.0.tva-raw": "TVA own storage; ResOps Kentucky amplitude inconsistent with lake area"}
# Dams the harvest missed although data exist: the series was absent from the catalogs or needed
# a fetcher that harvest_v2 does not have.
PIN.update({
    # USGS 09421500 parameter 00065 is not the lake level (5 ft range; Lake Mead varies by about
    # 25 ft), so use the Reclamation Lower Colorado database storage instead
    "Hoover": [("LCHDB", "1721", "storage", "ac-ft")],
    "Beaver": [("CWMS", "SWL|Beaver_Dam-Headwater.Stor-Res.Inst.1Hour.0.CCP-Comp", "storage", "m3"),
               ("CWMS", "SWL|Beaver_Dam-Headwater.Elev.Inst.1Hour.0.Decodes-rev", "elevation", "m")],
    "C.J. Strike": [("NWISANY", "13171500|00054", "storage", "ac-ft"), ("NWISANY", "13171500|62614", "elevation", "ft"),
                    ("NWISANY", "13171500|00062", "elevation", "ft"), ("NWISANY", "13171500|00065", "elevation", "ft")],
    "Horseshoe": [("NWISANY", "09509501|00054", "storage", "ac-ft"), ("NWISANY", "09509501|00065", "elevation", "ft")],
    "Alvin Wirtz Dam": [("TWDB", "LyndonBJohnson", "storage", "ac-ft")],          # LBJ Lake
    "Max Starcke Dam": [("TWDB", "MarbleFalls", "storage", "ac-ft")],             # Lake Marble Falls
    "Antero": [("NRCSM", "06016010:CO:BOR", "storage", "ac-ft")],
    "Eleven Mile Canyon": [("NRCSM", "06016100:CO:BOR", "storage", "ac-ft")],
    "Conconully": [("NRCSM", "12446480:WA:BOR", "storage", "ac-ft")],
})
def fetch_nrcs_monthly(sid):
    """Fetch monthly reservoir storage of an NRCS AWDB station, dated at month ends, in acre-feet.

    `sid` is the station triplet. Returns (series, unit) or (None, None).
    """
    t = H.get("https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data?stationTriplets=%s&elements=RESC"
              "&duration=MONTHLY&beginDate=%s&endDate=%s" % (sid, H.START, H.END), f"nrcs_m_{sid}")
    try:
        v = H.json.loads(t)[0]["data"][0]["values"]
        return pd.Series({pd.Timestamp(year=x["year"], month=x["month"], day=1) + pd.offsets.MonthEnd(0): x.get("value")
                          for x in v}, dtype=float).dropna(), "ac-ft"
    except Exception: return None, None
H.FETCH["NRCSM"] = fetch_nrcs_monthly
def fetch_nwis_any(sid):
    """Fetch USGS daily values of one site and parameter code (`sid` = "site|parameter").

    Takes whichever daily statistic the site has (00003 daily mean or 32400 midnight
    observation; Horseshoe has only the latter). Returns (series, None), where None means "use
    the pinned unit", or (None, None).
    """
    site, code = sid.split("|")
    t = H.get("https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites=%s&parameterCd=%s&startDT=%s&endDT=%s"
              % (site, code, H.START, H.END), f"nwisany_{site}_{code}")
    L = [l.split("\t") for l in t.splitlines() if l and not l.startswith("#")]
    if len(L) < 3: return None, None
    hdr = L[0]; col = next((i for i, c in enumerate(hdr) if f"_{code}_" in c and not c.endswith("_cd")), None)
    if col is None: return None, None
    s = pd.Series({pd.Timestamp(r[2]): pd.to_numeric(r[col], errors="coerce") for r in L[2:] if len(r) > col}, dtype=float)
    return s.dropna(), None
def fetch_lchdb(sdi):
    """Fetch daily values from the Reclamation Lower Colorado hydrologic database, in acre-feet.

    `sdi` is the site-datatype id of the series. Returns (series, unit) or (None, None).
    """
    t = H.get("https://www.usbr.gov/pn-bin/hdb/hdb.pl?svr=lchdb&sdi=%s&tstp=DY&t1=%sT00:00&t2=%sT00:00&table=R&mrid=0&format=json"
              % (sdi, H.START, H.END), f"lchdb_{sdi}")
    try:
        d = H.json.loads(t)["Series"][0]["Data"]
        return pd.Series({pd.Timestamp(x["t"]).normalize(): pd.to_numeric(x["v"], errors="coerce") for x in d}, dtype=float).dropna(), "ac-ft"
    except Exception: return None, None
H.FETCH["NWISANY"] = fetch_nwis_any; H.FETCH["LCHDB"] = fetch_lchdb
def validate_monthly(v, hist, cap):
    """Check a monthly storage series (MCM) against the dam's ResOpsUS record `hist`.

    Same ResOpsUS median-ratio rule as daily storage (within 1/3 to 3), but a monthly series
    needs only 24 values. `cap` is not used. Returns (series, note) or (None, reason).
    """
    if v is None or len(v) < 24 or v.std() == 0: return None, "short/constant monthly"
    if hist is None or hist.storage.notna().sum() < 200: return None, "no ResOps"
    r = v.median()/hist.storage.median()
    return (v, f"ratio={r:.2f} unit:ok monthly") if 1/3 <= r <= 3 else (None, f"monthly ratio {r:.2f}")
out = []
for dam, cands in PIN.items():
    r = t.loc[dam]; hist = H.resops_hist(r.DAM_ID); n = r.node
    for src, sid, kind, unit in cands:
        s, u2 = H.FETCH[src](sid)
        if s is None or not len(s): print(f"{dam:28s} {sid}: empty"); continue
        v = H.to_storage_units(s, u2 or unit, kind); v = v[(v.index >= H.START) & (v.index <= H.END)]
        if src == "NRCSM": o, note = validate_monthly(v, hist, st.dam_cap_mcm.get(n))
        elif kind == "storage": o, note = H.validate_storage(v, hist, st.dam_cap_mcm.get(n), True)
        else: o, note = H.validate_elevation(v, hist, st.lake_area_km2.get(n), tight=True)
        if o is None and sid in FORCE: o, note = v, FORCE[sid]
        print(f"{dam:28s} {sid[:48]:48s} n={len(v):4d} -> {note}")
        if o is not None:
            out.append(pd.DataFrame(dict(node=n, date=o.index.date, storage_mcm=o.values, source=f"{src}:{kind}",
                                         sid=sid, method=note + " [manual pin]"))); break
pd.concat(out).to_csv("harvested_manual.csv", index=False); print("pinned", len(out))
