"""Prepare two inputs of the routed rule-model runs: statewide PDSI and the rule-reservoir nodes.

The rule-model benchmarks route water through chains of reservoirs (a "cascade"): the outflow of
an upstream rule reservoir is part of the inflow of the next one. For that, every reservoir that
has published rules must sit on a reservoir node of the graph. This script places them, and it
downloads the drought index that GDROM's rules use. Paths below are relative to
`$SWOTGAT_WORK/GRADES/` unless they start with `$`.

Usage (one step per call):
  python grades_cascade_prep.py pdsi
  python grades_cascade_prep.py nodes

Steps:
  pdsi    NOAA Climate at a Glance statewide monthly Palmer Drought Severity Index (PDSI),
          1999-2025, for the 48 conterminous states.
          Writes  pdsi_state_monthly.csv (state, yyyymm, pdsi).
          This is the index GDROM's rules were trained with. grades_prep_early.py checks it:
          Washington, January 1980 is 0.54, as in GDROM's cleaned training record for Ross Dam
          (GRanD id 41).
  nodes   Place every GDROM v2 and ISTARF-CONUS reservoir on a reservoir node of the graph.
          Reads   gdrom_v2/reservoir_metadata.csv, ISTARF-CONUS.csv, testset_dams.csv (from
                  grades_map.py),
                  $SWOTGAT_WORK/GDW_v1_0/GDW_v1_0.gdb (Global Dam Watch, GDW; barrier points),
                  SWOT_PLD_v106_reservoirs.gdb (SWOT Prior Lake Database, PLD; reservoir polygons;
                  under $SWOTGAT_NAS_WATER),
                  $SWOTGAT_WORK/CONUS/metadata/subbasins_conus.parquet (nodes of the graph),
                  $SWOTGAT_WORK/CONUS_storage_fork/testsets/{regime1_2023_2025,regime2_2015_2020}/
                  sites.csv (member nodes of the test-set dams).
          Writes  rule_reservoir_nodes.csv: GRAND_ID, gdrom, istarf, test, STATE, CATEGORY, lon,
                  lat, node, dist_m. `node` is empty for a reservoir that could not be placed.

Placement rules (step `nodes`):
  - Dam point: the GDW barrier point with the same GRanD id. GDROM also has reservoirs that are not
    in GRanD (ids of 10000 and above); these use GDROM's own coordinates.
  - A test-set dam always keeps its inflow node from testset_dams.csv (the member node with the
    largest upstream area).
  - Every other rule reservoir goes to the nearest PLD reservoir polygon that is a reservoir node
    of the graph, within 1.5 km of the dam point (distances in EPSG:5070).
  - A node that is a member of a test-set dam is never given to another dam.
  - If several dams claim one node, a test-set dam wins, then the nearest dam; the others stay
    unplaced.
  - Unplaced reservoirs stay out of the cascade: their reach passes natural flow.
  - State (used to pick the PDSI series): GDROM's metadata state; for reservoirs that only ISTARF
    has, the GDW administrative unit.

Note: MODROM numbers its non-GRanD reservoirs differently from GDROM. modrom_nongrand_nodes.py
places those separately, and the simulators use its table for MODROM.
"""
from swotgat.paths import NAS_WATER, WORK
import sys, re, time, numpy as np, pandas as pd, requests
from swotgat.http import polite_get
from pathlib import Path
G = Path(f"{WORK}/GRADES"); C = Path(f"{WORK}/CONUS")
step = sys.argv[1]
ABBR = {"Alabama": "AL", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
        "Florida": "FL", "Georgia": "GA", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY",
        "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
        "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
        "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
        "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
        "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"}

if step == "pdsi":
    U = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/statewide/time-series/{}/pdsi/1/0/1999-2025/data.csv"
    rows = []
    for code in range(1, 49):
        for k in range(4):
            try:
                txt = polite_get(U.format(code), timeout=60).text; break
            except Exception: time.sleep(5 * (k + 1))
        name = re.search(r'"?([A-Za-z ]+) Palmer Drought', txt.replace("&quot;", '"')).group(1).strip()
        for line in txt.splitlines():
            m = re.match(r"^(\d{6}),(-?[\d.]+)", line)
            if m and float(m.group(2)) > -99: rows.append((ABBR[name], int(m.group(1)), float(m.group(2))))
        time.sleep(0.5)
    P = pd.DataFrame(rows, columns=["state", "yyyymm", "pdsi"]); P.to_csv(G/"pdsi_state_monthly.csv", index=False)
    print(f"states {P.state.nunique()} | months {P.yyyymm.min()}-{P.yyyymm.max()} | WA 198001 check skipped (series starts 1999)")
    print(P.groupby("state").yyyymm.max().value_counts().head().to_string())

elif step == "nodes":
    import geopandas as gpd
    GDW = f"{WORK}/GDW_v1_0/GDW_v1_0.gdb"
    J = Path(f"{NAS_WATER}/cjgleason/ted/LakeSPvC_ts_for_Ted")
    M = pd.read_csv(G/"gdrom_v2/reservoir_metadata.csv"); I = pd.read_csv(G/"ISTARF-CONUS.csv")
    ids = pd.DataFrame({"GRAND_ID": sorted(set(M.GRAND_ID) | set(I.GRanD_ID))})
    ids["gdrom"] = ids.GRAND_ID.isin(M.GRAND_ID); ids["istarf"] = ids.GRAND_ID.isin(I.GRanD_ID)
    ids = ids.merge(M[["GRAND_ID", "STATE", "LONGITUDE", "LATITUDE", "CATEGORY"]], on="GRAND_ID", how="left")
    bar = gpd.read_file(GDW, layer="GDW_barriers_v1_0", columns=["GRAND_ID", "ADMIN_UNIT"])
    bar["GRAND_ID"] = pd.to_numeric(bar.GRAND_ID, errors="coerce"); bar = bar[bar.GRAND_ID > 0].drop_duplicates("GRAND_ID").set_index("GRAND_ID")
    pts = []
    for r in ids.itertuples():
        if r.GRAND_ID in bar.index and bar.geometry[r.GRAND_ID] is not None and not bar.geometry[r.GRAND_ID].is_empty:
            g = bar.geometry[r.GRAND_ID]; pts.append((g.x, g.y))
        else: pts.append((r.LONGITUDE, r.LATITUDE))
    ids["lon"], ids["lat"] = zip(*pts)
    # states for ISTARF-only dams (GDROM gives its own): GDW admin unit
    st = ids.STATE.isna() & ids.GRAND_ID.isin(bar.index)
    ids.loc[st, "STATE"] = ids.loc[st, "GRAND_ID"].map(bar.ADMIN_UNIT).map(ABBR)
    ids = ids.dropna(subset=["lon", "lat"])
    P = gpd.GeoDataFrame(ids, geometry=gpd.points_from_xy(ids.lon, ids.lat), crs=4326).to_crs(5070)
    s = pd.read_parquet(C/"metadata/subbasins_conus.parquet", columns=["node_type", "uparea_km2"]); s.index = s.index.astype(str)
    nodes = set(s.index[s.node_type == "reservoir"])
    L = gpd.read_file(J/"SWOT_PLD_v106_reservoirs.gdb", columns=["lake_id"])
    L = L[L.lake_id.notna()]; L["node"] = L.lake_id.astype("float64").astype("int64").astype(str); L = L[L.node.isin(nodes)].to_crs(5070)
    j = gpd.sjoin_nearest(P, L[["node", "geometry"]], how="left", max_distance=1500, distance_col="dist_m").sort_values("dist_m")
    j = j.drop_duplicates("GRAND_ID")
    # test dams: their own inflow node, always
    T = pd.read_csv(G/"testset_dams.csv", dtype={"inflow_node": str}).dropna(subset=["inflow_node"]).drop_duplicates("DAM_ID")
    S1 = pd.concat([pd.read_csv(f"{WORK}/CONUS_storage_fork/testsets/{x}/sites.csv", dtype={"members": str})
                    for x in ("regime1_2023_2025", "regime2_2015_2020")]).drop_duplicates("DAM_ID")
    member_of = {m: d for d, ms in zip(S1.DAM_ID, S1.members.astype(str)) for m in ms.split(";") if m and m != "nan"}
    tmap = dict(zip(T.DAM_ID, T.inflow_node))
    j["test"] = j.GRAND_ID.isin(tmap)
    j.loc[j.test, "node"] = j.loc[j.test, "GRAND_ID"].map(tmap); j.loc[j.test, "dist_m"] = 0.0
    bad = (~j.test) & j.node.map(lambda n: n in member_of)      # a test dam's member polygon is that dam, not another
    j.loc[bad, "node"] = np.nan
    j = j.sort_values(["test", "dist_m"], ascending=[False, True])
    dup = j.node.notna() & j.duplicated("node")
    j.loc[dup, "node"] = np.nan
    out = j[["GRAND_ID", "gdrom", "istarf", "test", "STATE", "CATEGORY", "lon", "lat", "node", "dist_m"]]
    out.to_csv(G/"rule_reservoir_nodes.csv", index=False)
    print(f"rule reservoirs {len(out)} (GDROM {out.gdrom.sum()}, ISTARF {out.istarf.sum()}) | placed on a graph node {out.node.notna().sum()} "
          f"(GDROM {(out.gdrom & out.node.notna()).sum()}, ISTARF {(out.istarf & out.node.notna()).sum()}) | test dams {out.test.sum()} "
          f"| dropped as another dam's member {int(bad.sum())}, as duplicate node {int(dup.sum())} | missing state {out.STATE.isna().sum()}")
