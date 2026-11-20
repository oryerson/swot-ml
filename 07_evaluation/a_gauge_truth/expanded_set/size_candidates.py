"""Size the gauge expansion: which rule-model reservoirs lack a gauge record, and where one probably exists.

Step 1 of the expanded gauge set. The aim of the expansion is more reservoirs at which SWOT-GAT and the
rule models (GDROM, ISTARF, MODROM) can be compared with gauges, in 1981-2020 and in 2023-2026. This
script only counts. It lists the candidate reservoirs, the rule models that can run at each, the test
sets that already hold it, and the evidence that an agency keeps a daily record there.

Reads (under $SWOTGAT_WORK)
  GRADES/rule_reservoir_nodes.csv                rule reservoirs placed on graph nodes
  GRADES/gdrom_v2/reservoir_metadata.csv, GRADES/ISTARF-CONUS.csv,
  GRADES/modrom/results/calibrated_parameters.json, GRADES/modrom/data/reservoir_norm_info.json,
  GRADES/modrom_nongrand_nodes.csv               rule-model coverage and reservoir attributes
  CONUS_storage_fork/eval/g4_fix/testsets/regime3_full_1980_2020_g4/sites.csv,
  CONUS_storage_fork/testsets/{regime1x_2023_2026,regime2_2015_2020}/sites.csv,
  CONUS_storage_fork/testsets/regime1x_2023_2026/candidate_additions/sites.csv (if present),
  three testsets/*/predict_nodes.csv             the existing test sets and their tiers
  CONUS_storage_fork/crosswalk/dam_node_crosswalk.csv, crosswalk/graph_reservoir_node_audit.csv
  CONUS_storage_fork/ResOpsUS/attributes/reservoir_attributes.csv
  CONUS_storage_fork/gauges2026/cache/cwmsloc_*.txt    USACE CWMS location catalogs cached earlier

Writes (under $SWOTGAT_WORK/CONUS_storage_fork/gauges_expand)
  sizing/candidates.csv     one row per candidate reservoir, with all evidence columns
  sizing/sizing.csv         counts per model set and period
  cache/<tag>.txt           every downloaded response

Usage
  python size_candidates.py        (no arguments)

Method
  * Candidates: every rule reservoir placed on a graph node (in the paper's run: 1,506 of the GDROM v2
    2,017 and ISTARF-CONUS 1,930 GRanD dams), plus the reservoirs of calibrated MODROM that have no
    GRanD id. A MODROM reservoir counts as calibrated if all its monthly modes are 2 or 3, its
    parameters are finite and it has normalization data.
  * A candidate is "new" for a period if the test set of that period (1980-2020 or 2023-2026) does
    not hold it at tier High or Medium. ResOpsUS dams that already have a tier from the frozen
    crosswalk are counted separately, because their tier is fixed.
  * Evidence of a gauge record:
      GDROM v2 metadata  STORAGE_SOURCE and YEAR_RANGE. Sources usace, gdromv1 and resopsus are
                 observations. "reconstruction" is a fusion of satellite and streamflow data, not a
                 gauge.
      USGS NWIS  site service: daily series of parameter 00054 (storage) and 62614, 62615, 00062,
                 72020 (elevation) within 6 km of the dam, with their periods of record
      NRCS AWDB  stations with element RESC (reservoir storage) within 8 km, with dates
      Colorado DWR  telemetry stations with parameter STORAGE within 0.06 degrees of latitude and
                 0.08 degrees of longitude of the dam
      USACE CWMS  locations within 4 km, from catalogs cached earlier by the 2026 harvest. Tailwater,
                 spillway, gate and similar locations are left out.
      California and Texas  a flag only (column blocked_agency).
    Evidence is counted as years of overlap with 1981-2020, 2010-2020 and 2023-07-01 to 2026-06-12.
  * Model sets counted: GDROM + ISTARF + calibrated MODROM + SWOT-GAT, GDROM + ISTARF + SWOT-GAT,
    ISTARF + SWOT-GAT, and SWOT-GAT alone.

Notes
  This step downloads only from three services whose robots.txt allowed automated access when the
  script was written: waterservices.usgs.gov and wcc.sc.egov.usda.gov (no robots.txt) and
  dwr.state.co.us (allowed path). get() refuses any other host. Requests go through swotgat.http
  (generic client name, at most one request per 1.1 s per host) and are cached.
  The robots.txt of CWMS disallows /catalog and /timeseries, so this step sends CWMS no request and
  uses cached catalogs only. The robots.txt files of the California (CDEC), Texas (TWDB) and Bureau of
  Reclamation (RISE, Hydromet) services disallow their data paths, so this step sends them no request
  and only flags the reservoirs in California and Texas. The later steps discover_expand.py and
  harvest_expand.py do request CWMS, CDEC, TWDB and RISE.
"""
from swotgat.paths import WORK
import io, json, re, time, math
from pathlib import Path
import numpy as np, pandas as pd, requests
R = Path(f"{WORK}"); F = R/"CONUS_storage_fork"; G = R/"GRADES"; X = F/"gauges_expand"
OUT = X/"sizing"; CACHE = X/"cache"; OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
from swotgat.http import PoliteSession; SES = PoliteSession()
ALLOWED = ("waterservices.usgs.gov", "wcc.sc.egov.usda.gov", "dwr.state.co.us")
def get(url, tag):
    """Return the text of a URL from the cache, or download it from an allowed host.

    tag names the cache file (cache/<tag>.txt). A URL whose host is not in ALLOWED stops the script. A
    download is tried at most four times, each after a pause of 1 s: status 200 is cached and returned;
    400 or 404 is cached as empty and returns "". Returns "" if every attempt fails.
    """
    f = CACHE/(re.sub(r"[^A-Za-z0-9_.-]", "_", tag)[:180] + ".txt")
    if f.is_file(): return f.read_text()
    assert any(h in url for h in ALLOWED), url
    for a in range(4):
        try:
            time.sleep(1.0); r = SES.get(url, timeout=300)
            if r.status_code == 200: f.write_text(r.text); return r.text
            if r.status_code in (400, 404): f.write_text(""); return ""
        except Exception: pass
        time.sleep(5*(a + 1))
    return ""
def hav(a, b, c, d):
    """Return the great-circle distance in km between (lat a, lon b) and (lat c, lon d).

    All coordinates are in degrees; c and d may be arrays. The Earth radius is 6371 km.
    """
    p = math.pi/180; x = np.sin((c - a)*p/2)**2 + np.cos(a*p)*np.cos(c*p)*np.sin((d - b)*p/2)**2
    return 12742*np.arcsin(np.sqrt(x))
ST = dict(AL="Alabama", AZ="Arizona", AR="Arkansas", CA="California", CO="Colorado", CT="Connecticut", DE="Delaware", FL="Florida", GA="Georgia",
          ID="Idaho", IL="Illinois", IN="Indiana", IA="Iowa", KS="Kansas", KY="Kentucky", LA="Louisiana", ME="Maine", MD="Maryland", MA="Massachusetts",
          MI="Michigan", MN="Minnesota", MS="Mississippi", MO="Missouri", MT="Montana", NE="Nebraska", NV="Nevada", NH="New Hampshire", NJ="New Jersey",
          NM="New Mexico", NY="New York", NC="North Carolina", ND="North Dakota", OH="Ohio", OK="Oklahoma", OR="Oregon", PA="Pennsylvania",
          RI="Rhode Island", SC="South Carolina", SD="South Dakota", TN="Tennessee", TX="Texas", UT="Utah", VT="Vermont", VA="Virginia",
          WA="Washington", WV="West Virginia", WI="Wisconsin", WY="Wyoming")

# ---------------- candidate reservoirs and rule coverage (same family logic as grades_sim_v4.py)
RN = pd.read_csv(G/"rule_reservoir_nodes.csv", dtype={"node": str})
MD = pd.read_csv(G/"gdrom_v2/reservoir_metadata.csv").set_index("GRAND_ID")
IP = pd.read_csv(G/"ISTARF-CONUS.csv").set_index("GRanD_ID")
PRM = {int(x["GRAND_ID"]): x for x in json.load(open(G/"modrom/results/calibrated_parameters.json"))}
NRM = {int(k): v for k, v in json.load(open(G/"modrom/data/reservoir_norm_info.json")).items()}
MC = {g for g, x in PRM.items() if all(m in (2, 3) for m in x["MonthlyModes"]) and np.all(np.isfinite(np.asarray(sum(x["ModeParameters"], []), float)))} & set(NRM)
MNG = pd.read_csv(G/"modrom_nongrand_nodes.csv", dtype={"node": str})
C = RN.dropna(subset=["node"]).drop_duplicates("GRAND_ID").copy()
mng = MNG[~MNG.GRAND_ID.isin(C.GRAND_ID) & MNG.reject.isna()].dropna(subset=["node"])
C = pd.concat([C, pd.DataFrame({"GRAND_ID": mng.GRAND_ID, "node": mng.node, "gdrom": False, "istarf": False, "test": False,
                                 "lon": mng.get("lon", np.nan), "lat": mng.get("lat", np.nan), "STATE": mng.get("STATE", np.nan)})], ignore_index=True)
C["gdrom"] = C.GRAND_ID.isin(MD.index); C["istarf"] = C.GRAND_ID.isin(IP.index); C["modrom_cal"] = C.GRAND_ID.isin(MC)
C["gdrom_cat"] = C.GRAND_ID.map(MD.CATEGORY); C["istarf_fit"] = C.GRAND_ID.map(IP.fit)
C["name"] = C.GRAND_ID.map(MD.DAM_NAME).fillna(C.GRAND_ID.map(IP.GRanD_NAME)); C["cap_mcm"] = C.GRAND_ID.map(IP.GRanD_CAP_MCM)
fill = C.lat.isna() & C.GRAND_ID.isin(MD.index); C.loc[fill, "lat"] = C.GRAND_ID[fill].map(MD.LATITUDE); C.loc[fill, "lon"] = C.GRAND_ID[fill].map(MD.LONGITUDE)
C["st"] = C.STATE.fillna(C.GRAND_ID.map(MD.STATE))
# GDROM v2 observed-record evidence
src = C.GRAND_ID.map(MD.STORAGE_SOURCE).fillna("")
C["gdrom_src"] = src; C["gdrom_obs"] = src.str.contains("usace|gdromv1|resopsus")
yr = C.GRAND_ID.map(MD.YEAR_RANGE).fillna("").str.findall(r"\d{4}")
C["gdrom_y0"] = [int(v[0]) if len(v) == 2 else np.nan for v in yr]; C["gdrom_y1"] = [int(v[1]) if len(v) == 2 else np.nan for v in yr]

# ---------------- existing truth
R3 = pd.read_csv(F/"eval/g4_fix/testsets/regime3_full_1980_2020_g4/sites.csv"); R1 = pd.read_csv(F/"testsets/regime1x_2023_2026/sites.csv")
R2 = pd.read_csv(F/"testsets/regime2_2015_2020/sites.csv"); CA_ = pd.read_csv(F/"testsets/regime1x_2023_2026/candidate_additions/sites.csv") \
    if (F/"testsets/regime1x_2023_2026/candidate_additions/sites.csv").exists() else pd.DataFrame(columns=["DAM_ID", "site_tier"])
for k, T in (("r3", R3), ("r1x", R1), ("r2", R2), ("r1x_cand", CA_)):
    C[f"{k}_tier"] = C.GRAND_ID.map(T.drop_duplicates("DAM_ID").set_index("DAM_ID").site_tier)
XW = pd.read_csv(F/"crosswalk/dam_node_crosswalk.csv").drop_duplicates("DAM_ID").set_index("DAM_ID")
C["xw_tier"] = C.GRAND_ID.map(XW.tier); C["resops"] = C.GRAND_ID.isin(pd.read_csv(F/"ResOpsUS/attributes/reservoir_attributes.csv").DAM_ID)
AU = pd.read_csv(F/"crosswalk/graph_reservoir_node_audit.csv", index_col=0); AU.index = AU.index.astype(str)
C["node_grand"] = C.node.map(AU.grand); C["node_is_lake"] = C.node.isin(AU.index); C["node_wse_n"] = C.node.map(AU.wse_n)
C["node_id_match"] = C.node_grand == C.GRAND_ID
PN = set()
for p in [F/"testsets/regime3_full_1980_2020/predict_nodes.csv", F/"testsets/regime1x_2023_2026/predict_nodes.csv", F/"testsets/regime2_2015_2020/predict_nodes.csv"]:
    PN |= set(pd.read_csv(p, dtype=str).subbasin)
C["node_predicted"] = C.node.isin(PN)

# ---------------- NWIS (dv storage / elevation, with period of record)
rows = []
for ab in ST:
    t = get(f"https://waterservices.usgs.gov/nwis/site/?format=rdb&stateCd={ab}&parameterCd=00054,62614,62615,00062,72020&hasDataTypeCd=dv"
            f"&seriesCatalogOutput=true&siteStatus=all", f"nwissite_{ab}")
    L = [l for l in t.splitlines() if l and not l.startswith("#")]
    if len(L) < 3: continue
    d = pd.read_csv(io.StringIO("\n".join([L[0]] + L[2:])), sep="\t", dtype=str); rows.append(d)
NW = pd.concat(rows); NW = NW[(NW.data_type_cd == "dv") & NW.parm_cd.isin(["00054", "62614", "62615", "00062", "72020"])].copy()
NW["lat"] = NW.dec_lat_va.astype(float); NW["lon"] = NW.dec_long_va.astype(float)
NW["b"] = pd.to_datetime(NW.begin_date, errors="coerce"); NW["e"] = pd.to_datetime(NW.end_date, errors="coerce")
NW["kind"] = np.where(NW.parm_cd == "00054", "storage", "elevation")
print("NWIS dv series", len(NW), "sites", NW.site_no.nunique(), flush=True)
# ---------------- NRCS AWDB reservoir storage
t = get("https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/stations?elements=RESC&returnReservoirMetadata=true&activeOnly=false", "nrcs_stations_resc")
NR = pd.DataFrame(json.loads(t)) if t else pd.DataFrame()
NR["b"] = pd.to_datetime(NR.beginDate, errors="coerce"); NR["e"] = pd.to_datetime(NR.endDate.str.replace("2100-01-01", "2026-10-01"), errors="coerce")
print("NRCS RESC stations", len(NR), flush=True)
# ---------------- CO DWR telemetry storage stations
cw = []
for p in range(1, 4):
    t = get(f"https://dwr.state.co.us/Rest/GET/api/v2/telemetrystations/telemetrystation/?format=json&parameter=STORAGE&pageSize=50000&pageIndex={p}", f"codwr_stations_{p}")
    if not t: break
    j = json.loads(t); cw += j.get("ResultList", [])
    if p >= j.get("PageCount", 1): break
CD = pd.DataFrame(cw)
if len(CD): CD = CD[CD.parameter.astype(str).str.upper().str.contains("STOR")]
print("CO DWR storage stations", len(CD), flush=True)
# ---------------- CWMS locations (cached by gauges2026 only)
cl = []
for f in (F/"gauges2026/cache").glob("cwmsloc_*.txt"):
    try: cl += json.loads(f.read_text()).get("entries", [])
    except Exception: pass
CL = pd.DataFrame(cl); CL = CL[(CL.latitude.abs() > 1) & (CL.longitude.abs() > 1)]
BAD = re.compile(r"Tailwater|Tail|Spillway|Gate|Rereg|Downstream|Outlet|Intake|Powerhouse|-TW|_TW|Unit|Lock", re.I)
CL = CL[~CL.name.astype(str).str.contains(BAD) & ~CL["long-name"].astype(str).str.contains(BAD)]
print("CWMS cached locations", len(CL), flush=True)

# ---------------- per candidate evidence
W81 = (pd.Timestamp("1981-01-01"), pd.Timestamp("2020-12-31")); W10 = (pd.Timestamp("2010-01-01"), pd.Timestamp("2020-12-31"))
W23 = (pd.Timestamp("2023-07-01"), pd.Timestamp("2026-06-12"))
def ov(b, e, w):
    """Return the years that a period of record (begin b, end e) shares with the window w = (start, end).

    The result is 0 if the two do not overlap or if b or e is missing.
    """
    return max(0.0, (min(e, w[1]) - max(b, w[0])).days/365.25) if pd.notna(b) and pd.notna(e) else 0.0
ev = []
for _, c in C.iterrows():
    r = dict(GRAND_ID=c.GRAND_ID)
    if np.isfinite(c.lat):
        n = NW[(NW.lat - c.lat).abs().lt(0.08) & (NW.lon - c.lon).abs().lt(0.1)].copy(); n["km"] = hav(c.lat, c.lon, n.lat.values, n.lon.values); n = n[n.km <= 6]
        for k in ("storage", "elevation"):
            m = n[n.kind == k]
            r[f"nwis_{k}_sites"] = ";".join(sorted(set(m.site_no)))
            r[f"nwis_{k}_y81"] = max([ov(b, e, W81) for b, e in zip(m.b, m.e)] or [0]); r[f"nwis_{k}_y10"] = max([ov(b, e, W10) for b, e in zip(m.b, m.e)] or [0])
            r[f"nwis_{k}_y23"] = max([ov(b, e, W23) for b, e in zip(m.b, m.e)] or [0])
        m = NR[(NR.latitude - c.lat).abs().lt(0.08) & (NR.longitude - c.lon).abs().lt(0.1)].copy()
        m = m[hav(c.lat, c.lon, m.latitude.values, m.longitude.values) <= 8]
        r["nrcs"] = ";".join(m.stationTriplet); r["nrcs_y81"] = max([ov(b, e, W81) for b, e in zip(m.b, m.e)] or [0]); r["nrcs_y23"] = max([ov(b, e, W23) for b, e in zip(m.b, m.e)] or [0])
        if len(CD):
            m = CD[(CD.latitude.astype(float) - c.lat).abs().lt(0.06) & (CD.longitude.astype(float) - c.lon).abs().lt(0.08)]
            r["codwr"] = ";".join(m.abbrev.astype(str))
        m = CL[(CL.latitude - c.lat).abs().lt(0.04) & (CL.longitude - c.lon).abs().lt(0.05)].copy()
        m = m[hav(c.lat, c.lon, m.latitude.values, m.longitude.values) <= 4]
        r["cwms_locs"] = ";".join((m.office + "/" + m.name).head(6)); r["cwms_n"] = len(m)
    ev.append(r)
C = C.merge(pd.DataFrame(ev), on="GRAND_ID", how="left")
C["blocked_agency"] = np.select([C.st.isin(["CA", "California"]), C.st.isin(["TX", "Texas"])], ["CDEC (robots: disallow all)", "TWDB (robots: csv/api disallowed)"], "")
C.to_csv(OUT/"candidates.csv", index=False)

# ---------------- counts
hm = lambda s: s.isin(["High", "Medium"])
C["have81"] = hm(C.r3_tier); C["have23"] = hm(C.r1x_tier)
C["cand"] = ~C.resops | C.r3_tier.isna()          # ResOps dams already went through the frozen crosswalk: their tier is fixed
obs81 = C.gdrom_obs & (C.gdrom_y0 <= 2018) & (C.gdrom_y1 >= 1983)
C["ev81_allowed"] = (C.nwis_storage_y10 >= 2) | (C.nrcs_y81 >= 2) | (obs81 & ~C.resops)
C["ev23_allowed"] = (C.nwis_storage_y23 >= 0.5) | (C.nrcs_y23 >= 0.5) | (C.codwr.fillna("") != "")
C["ev_cwms"] = C.cwms_n.fillna(0) > 0
C["ev_elev_only81"] = ~C.ev81_allowed & (C.nwis_elevation_y10 >= 2); C["ev_elev_only23"] = ~C.ev23_allowed & (C.nwis_elevation_y23 >= 0.5)
C.to_csv(OUT/"candidates.csv", index=False)
sets = {"4-model (GDROM+ISTARF+MODROM-cal+SWOT-GAT)": C.gdrom & C.istarf & C.modrom_cal, "3-model (GDROM+ISTARF+SWOT-GAT)": C.gdrom & C.istarf,
        "SWOT-GAT+ISTARF": C.istarf, "SWOT-GAT alone (rule-model reservoirs only)": pd.Series(True, index=C.index)}
L = []
for nm, s in sets.items():
    for per, have, evA, evE in (("1981-2020", "have81", "ev81_allowed", "ev_elev_only81"), ("2023-26", "have23", "ev23_allowed", "ev_elev_only23")):
        new = s & ~C[have] & C.cand
        L.append(dict(set=nm, period=per, placed=int(s.sum()), have_HM_now=int((s & C[have]).sum()), not_yet_in_any_test_set=int(new.sum()),
                      new_resops_lowtier_or_unmatched=int((s & ~C[have] & ~C.cand).sum()),
                      ev_storage_allowed=int((new & C[evA]).sum()), ev_elev_only=int((new & C[evE]).sum()),
                      ev_cwms_loc=int((new & C.ev_cwms & ~C[evA]).sum()), ca_tx_blocked=int((new & (C.blocked_agency != "") & ~C[evA]).sum()),
                      gdrom_obs_record=int((new & C.gdrom_obs).sum())))
S = pd.DataFrame(L); S.to_csv(OUT/"sizing.csv", index=False); print(S.to_string())
print("ResOps dams placed but no H/M in regime3:", int((C.resops & ~C.have81).sum()), C[C.resops & ~C.have81].r3_tier.value_counts(dropna=False).to_dict())
print("gdrom_src among non-ResOps candidates:", C[~C.resops].gdrom_src.str.split(",").explode().value_counts().to_dict())
