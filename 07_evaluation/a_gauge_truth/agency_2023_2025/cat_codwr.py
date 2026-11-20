"""List the Colorado Division of Water Resources (DWR) telemetry stations that report storage.

Writes ($SWOTGAT_WORK/CONUS_storage_fork)
  catalogs/codwr_raw.csv   every station the service returns for the parameter STORAGE
  catalogs/codwr.csv       the reservoir stations, in the catalog columns of build_catalogs.py:
                           source (CODWR), sid (station abbreviation), name, lat, lon, kind
                           (storage), unit, meta

Usage
  python cat_codwr.py     (no arguments; the folder catalogs/ must exist)

Method
  Pages through the DWR telemetry station service (5000 stations per page). A station is kept
  when its name contains RES, LAKE, RSVR or RESERVOIR and, if the service returns a measurement
  date (column measDateTime or measDate), that date is June 2023 or later.

Notes
  Downloads go through swotgat.http: a generic client name and at most one request per 1.1 s
  per host. Responses are not cached by this script.
"""
from swotgat.paths import WORK
import json, requests, pandas as pd
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork")
from swotgat.http import PoliteSession; S = PoliteSession()
B = "https://dwr.state.co.us/Rest/GET/api/v2/telemetrystations/telemetrystation/"
rows, page = [], 1
while True:
    r = S.get(B, params=dict(format="json", parameter="STORAGE", pageSize=5000, pageIndex=page), timeout=120).json()
    rows += r.get("ResultList", [])
    if page >= r.get("PageCount", 1): break
    page += 1
d = pd.DataFrame(rows)
print("fields:", list(d.columns)[:30])
print(f"STORAGE-reporting stations: {len(d)}")
d.to_csv(F/"catalogs/codwr_raw.csv", index=False)
res = d[d.stationName.str.contains(r"RES|LAKE|RSVR|RESERVOIR", case=False, na=False)]
print(f"  name looks like a reservoir: {len(res)}")
active = res
if "measDateTime" in res.columns or "measDate" in res.columns:
    c = "measDateTime" if "measDateTime" in res.columns else "measDate"
    active = res[res[c].astype(str) >= "2023-06"]
print(f"  of those with recent measurements: {len(active)}")
out = pd.DataFrame(dict(source="CODWR", sid=active["abbrev"], name=active["stationName"],
                        lat=active["latitude"], lon=active["longitude"], kind="storage",
                        unit=active.get("units", "af"), meta=active.get("dataSourceAbbrev", "")))
out.to_csv(F/"catalogs/codwr.csv", index=False)
print(out.head(8).to_string(index=False))
