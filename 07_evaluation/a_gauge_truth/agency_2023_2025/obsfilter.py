"""Keep only the USACE CWMS catalog series that are observations of reservoir storage or pool level.

The CWMS catalog also lists rule curves, pool limits, forecasts and model output under the
parameters Stor and Elev. This script removes them from the catalog, and removes from the first
two harvest files the reservoirs whose accepted series was one of them.

Reads ($SWOTGAT_WORK/CONUS_storage_fork)
  catalogs/cwms_unfiltered.csv   the catalog as build_catalogs.py wrote it, if this script has
                                 run before; otherwise catalogs/cwms.csv
  harvested.csv, harvested_pass2.csv   results of harvest.py / harvest_v2.py (read from the
                                 current folder for the audit, so run the script there)

Writes
  catalogs/cwms_unfiltered.csv   copy of the catalog before filtering (first run only)
  catalogs/cwms.csv              the catalog with observation series only
  harvested.csv, harvested_pass2.csv   without the reservoirs whose accepted CWMS series fails
                                 the filter; a later harvest pass looks for them again

Usage
  python obsfilter.py     (no arguments; run in $SWOTGAT_WORK/CONUS_storage_fork)

Method
  A CWMS series id is "office|Location.Parameter.Type.Interval.Duration.Version". The filter
  works on these fields and not on the whole string (a search of the whole id flagged Minidoka,
  location code MIN, as a minimum). A series is an observation when
  * the id has at least six fields,
  * the parameter is Stor or Elev,
  * the type is not Const,
  * the sub-parameter (the part of the parameter after "-") names no limit or reference level
    (top, target, guide, rule, flood, max, min, normal, zone, spillway, crest, tailwater, dead,
    inactive, capacity, ...; see `BAD_SUB`),
  * the version names no forecast, rule curve, plan, scenario, simulation or model (`BAD_VER`).
  The filter always starts from the unfiltered catalog, never from an already filtered one.
"""
from swotgat.paths import WORK
import re
from pathlib import Path
import pandas as pd
F = Path(f"{WORK}/CONUS_storage_fork")
BAD_SUB = re.compile(r"(?i)top|target|guide|rule|authori|allowable|flood|max|min\b|minimum|normal|zone|spill|crest|"
                     r"toc|pz\d|piezo|avail|tail|downstream|design|capacity|dead|inactive|objective|limit|required|surcharge")
BAD_VER = re.compile(r"(?i)fcst|forecast|guide|rule|\bRUL\b|target|top of|plan|scenario|simul|model")  # LRH .RUL = rule curve
def is_obs(sid):
    """Return True if a CWMS series id is an observed storage or pool-level series.

    `sid` is "office|series name". The rules are listed in the module docstring. An id without
    "|" (not a CWMS id) returns True.
    """
    try: off, ts = sid.split("|", 1)
    except ValueError: return True
    p = ts.split(".")
    if len(p) < 6: return False
    loc, param, typ, interval, dur, ver = p[0], p[1], p[2], p[3], p[4], ".".join(p[5:])
    base, _, sub = param.partition("-")
    if base not in ("Stor", "Elev"): return False
    if typ.lower() == "const": return False
    if sub and BAD_SUB.search(sub): return False
    if BAD_VER.search(ver): return False
    return True
U = F/"catalogs/cwms_unfiltered.csv"          # always filter from the raw catalog, never re-filter
K = pd.read_csv(U if U.is_file() else F/"catalogs/cwms.csv", dtype=str)
K["obs"] = K.sid.map(is_obs)
print(f"CWMS catalog {len(K):,} series -> observations {int(K.obs.sum()):,}  (dropped {int((~K.obs).sum()):,})")
dropped = K[~K.obs].sid.str.split("|").str[1].str.split(".").str[1].str.partition("-")[2]
print("most common dropped sub-parameters:", dropped.value_counts().head(10).to_dict())
K[K.obs].drop(columns="obs").to_csv(F/"catalogs/cwms.csv", index=False)
if not U.is_file(): K.drop(columns="obs").to_csv(U, index=False)

# audit what's already accepted
h = pd.concat([pd.read_csv(f, dtype={"node": str}) for f in ["harvested.csv", "harvested_pass2.csv"] if (F/f).is_file()])
acc = h.drop_duplicates("node")[["node", "source", "sid"]].copy()
acc["obs"] = acc.apply(lambda r: is_obs(r.sid) if str(r.source).startswith("CWMS") else True, axis=1)
bad = acc[~acc.obs]
print(f"\naccepted reservoirs failing the observation filter: {len(bad)}")
print(bad.to_string(index=False))
for f in ["harvested.csv", "harvested_pass2.csv"]:
    d = pd.read_csv(F/f, dtype={"node": str})
    d[~d.node.isin(set(bad.node))].to_csv(F/f, index=False)
print(f"\nremoved from harvested files; they return to the missing pool for pass 3")
