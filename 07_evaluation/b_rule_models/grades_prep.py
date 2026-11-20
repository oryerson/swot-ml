"""Download 2000-2025 inputs of the rule-model benchmarks: GDROM rules, PDSI and GRADES discharge.

The rule-model benchmarks drive published reservoir operating rules (GDROM, ISTARF, MODROM) with
river discharge simulated by GRADES-hydroDL. This script fetches three of their inputs. All paths
below are relative to `$SWOTGAT_WORK/GRADES/`.

Usage (one step per call):
  python grades_prep.py rules
  python grades_prep.py pdsi
  python grades_prep.py grades

Steps:
  rules   GDROM v2 operation rules: the `modules` and `module_conditions` text files, about 9,500
          small files.
          Reads   gdrom_v2/file_list.json, the file listing of the GDROM v2 dataset (url and size
                  of every file).
          Writes  gdrom_v2/<path of the file in the dataset>, for every file whose path contains
                  "Operation Rules".
          A file already on disk with the listed size is skipped. Each download is tried up to
          4 times.
  pdsi    gridMET Palmer Drought Severity Index (PDSI, a pentad product), 2000-01-01 to 2025-12-31,
          at the inflow node of every test-set dam.
          Reads   testset_dams.csv (from grades_map.py),
                  `$SWOTGAT_WORK/CONUS/metadata/subbasins_conus.parquet`, and the gridMET PDSI
                  aggregate on the Northwest Knowledge Network THREDDS server.
          Writes  pdsi_testset_nodes.parquet (node, day, pdsi).
          The value is taken at the grid cell nearest to a point inside the node's sub-basin
          polygon (`representative_point`). If that cell has no data (coast or mask), the mean of
          the cells within 0.1 degrees is used.
  grades  GRADES-hydroDL v2.0 daily discharge (`Qout`, m3/s) for Pfafstetter region 7 (North
          America), 2000-2025, at the MERIT-Basins v0.7 river reach of every reservoir node.
          Reads   reservoir_grades_map.csv (from grades_map.py; column comid_v07) and
                  gdrive_listing.json (Google Drive file id of each GRADES-hydroDL year file; a
                  copy ships in this folder's `inputs/`).
          Writes  qout/qout_<year>.parquet (rows = days, columns = v0.7 reach ids (COMID)).
          One year file (about 770 MB) is downloaded at a time with `gdown` (expected at
          `.venv/bin/gdown` under the GRADES folder), reduced to the wanted reaches and deleted.
          Years already extracted are skipped.

Notes:
  - Run grades_map.py before the `pdsi` and `grades` steps.
  - The simulators in this folder (grades_sim_v4.py, grades_sim_v4_hindcast.py) read the NOAA
    statewide PDSI made by grades_cascade_prep.py, not pdsi_testset_nodes.parquet.
  - grades_prep_early.py adds 1980-1999 and grades_prep_2026.py adds 2026.
"""
from swotgat.paths import WORK
import sys, json, urllib.parse, time, subprocess, numpy as np, pandas as pd, xarray as xr, requests
from swotgat.http import polite_get
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
G = Path(f"{WORK}/GRADES"); YEARS = range(2000, 2026)
step = sys.argv[1]

if step == "rules":
    files = json.load(open(G/"gdrom_v2/file_list.json"))
    todo = [f for f in files if "Operation%20Rules" in f["url"] or "Operation Rules" in urllib.parse.unquote(f["url"])]
    def get(f):
        p = urllib.parse.unquote(f["url"].split("/data/contents/")[1]); out = G/"gdrom_v2"/p
        if out.exists() and out.stat().st_size == f["size"]: return 0
        out.parent.mkdir(parents=True, exist_ok=True)
        for k in range(4):
            try:
                r = polite_get(f["url"].replace("http://", "https://"), timeout=60)
                if r.ok: out.write_bytes(r.content); return 1
            except Exception: time.sleep(2 * (k + 1))
        return -1
    with ThreadPoolExecutor(12) as ex: res = list(ex.map(get, todo))
    print(f"rule files {len(todo)} | downloaded {res.count(1)} | already present {res.count(0)} | failed {res.count(-1)}")

elif step == "pdsi":
    import geopandas as gpd
    D = pd.read_csv(G/"testset_dams.csv", dtype={"inflow_node": str}).dropna(subset=["inflow_node"])
    nodes = sorted(set(D.inflow_node))
    sub = gpd.read_parquet(f"{WORK}/CONUS/metadata/subbasins_conus.parquet", columns=["geometry"])
    sub.index = sub.index.astype(str); sub = sub.loc[nodes].to_crs(4326)
    cen = sub.geometry.representative_point()
    U = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_met_pdsi_1979_CurrentYear_CONUS.nc"
    ds = xr.open_dataset(U)["daily_mean_palmer_drought_severity_index"].sel(day=slice("2000-01-01", "2025-12-31"))
    def one(n):
        for k in range(4):
            try:
                p = cen[n]; x = ds.sel(lat=p.y, lon=p.x, method="nearest").load()
                if np.isnan(x.values).all():          # coastal / masked cell: mean of the cells within 0.1 degrees
                    w = ds.sel(lat=slice(p.y + 0.1, p.y - 0.1), lon=slice(p.x - 0.1, p.x + 0.1)).load().mean(["lat", "lon"])
                    x = w
                return pd.DataFrame({"node": n, "day": x.day.values, "pdsi": x.values})
            except Exception: time.sleep(3 * (k + 1))
        return pd.DataFrame(columns=["node", "day", "pdsi"])
    with ThreadPoolExecutor(6) as ex: parts = list(ex.map(one, nodes))
    P = pd.concat(parts); P.to_parquet(G/"pdsi_testset_nodes.parquet", index=False)
    print(f"PDSI: {P.node.nunique()} of {len(nodes)} nodes, {len(P):,} rows, missing share {P.pdsi.isna().mean():.2%}")

elif step == "grades":
    M = pd.read_csv(G/"reservoir_grades_map.csv", index_col=0)
    want = np.array(sorted(set(M.comid_v07.dropna().astype("int64"))))
    listing = dict(json.load(open(G/"gdrive_listing.json")))
    (G/"raw").mkdir(exist_ok=True); (G/"qout").mkdir(exist_ok=True)
    for y in YEARS:
        out = G/"qout"/f"qout_{y}.parquet"
        if out.exists(): continue
        key = f"{y}/GRADES-hydroDL_V2.0_pfaf_07_{y}.nc"; nc = G/"raw"/f"pfaf_07_{y}.nc"
        if key not in listing: print("missing on Drive:", key); continue
        for k in range(3):
            r = subprocess.run([str(G/".venv/bin/gdown"), "-q", f"https://drive.google.com/uc?id={listing[key]}", "-O", str(nc)])
            if r.returncode == 0 and nc.exists() and nc.stat().st_size > 1e8: break
            time.sleep(30)
        ds = xr.open_dataset(nc); riv = ds.rivid.values.astype("int64")
        idx = np.flatnonzero(np.isin(riv, want))
        q = ds.Qout.isel(rivid=idx).load()
        df = pd.DataFrame(q.values, index=pd.to_datetime(q.time.values), columns=riv[idx]); df.index.name = "date"
        df.to_parquet(out); ds.close(); nc.unlink()
        print(f"{y}: {len(idx)} of {len(want)} reaches, {len(df)} days", flush=True)
    print("GRADES extraction done")
