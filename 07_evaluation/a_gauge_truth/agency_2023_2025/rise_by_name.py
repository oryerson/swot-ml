"""Build the Reclamation RISE series catalog from cached pages, without coordinates.

`build_catalogs.py rise` needs two slow requests per catalog item to find its coordinates. The
item title already names the reservoir (for example "Lake Berryessa and Monticello Dam Daily
Lake/Reservoir Storage-af Time Series Data"), so this script takes the name from the title and
leaves the coordinates empty. The harvest then matches these series by name alone, with its
stricter name threshold, and its comparison with the dam's ResOpsUS storage rejects a series of
the wrong reservoir however the series was found.

Reads ($SWOTGAT_WORK/CONUS_storage_fork)
  agency_cache/rise_items_p*.txt   RISE catalog-item pages saved by `build_catalogs.py rise`

Writes
  catalogs/rise.csv   source (RISE), sid (item id), name, lat and lon (empty), kind, unit, meta.
                      This replaces the rise.csv written by build_catalogs.py.

Usage
  python rise_by_name.py     (no arguments; makes no HTTP request)

Method
  An item is kept when its title contains "daily" and "Reservoir Storage" (kind storage) or
  "Reservoir Elevation" / "Reservoir WS Elevation" (kind elevation), and its record ends on or
  after 2023-06-01. The name is the title up to the word "Daily". The unit is read from the
  title ("-af" or "-ft"); if neither is present it is acre-feet for storage and feet for
  elevation.
"""
from swotgat.paths import WORK
import json, re
from pathlib import Path
import pandas as pd
F = Path(f"{WORK}/CONUS_storage_fork"); CACHE = F/"agency_cache"
rows = []
for f in sorted(CACHE.glob("rise_items_p*.txt")):
    try: d = json.loads(f.read_text())["data"]
    except Exception: continue
    for it in d:
        a = it["attributes"]; t = a.get("itemTitle", "")
        if "daily" not in t.lower(): continue
        if re.search(r"Reservoir Storage", t): kind = "storage"
        elif re.search(r"Reservoir (WS )?Elevation", t): kind = "elevation"
        else: continue
        if (a.get("temporalEndDate") or "") < "2023-06-01": continue
        name = re.split(r"\s+Daily\s+", t, flags=re.I)[0]
        unit = "af" if "-af" in t else ("ft" if "-ft" in t else ("af" if kind == "storage" else "ft"))
        rows.append(dict(source="RISE", sid=str(a["_id"]), name=name, lat="", lon="",
                         kind=kind, unit=unit, meta=a.get("locationSourceCode", "")))
d = pd.DataFrame(rows).drop_duplicates("sid")
d.to_csv(F/"catalogs/rise.csv", index=False)
print(f"RISE catalog by name: {len(d)} series  {d.kind.value_counts().to_dict()}")
print(d.head(8)[["sid","name","kind","unit"]].to_string(index=False))
