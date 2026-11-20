"""Verify the template store (zarr_store_vd2026_halfexact) against the store it was derived from.

The other store is zarr_store_vd2026 (03_forcings/c_extension_2026/build_store_vd2026.py). The two stores
should differ only in the lake targets: zarr_store_vd2026_halfexact holds the lake record after
HALF's temporal-coverage rule (HALF = the lake outlier filter of Trudel et al.; build_halfexact.py), which
removes some lakes.

Usage: python verify_store_hx.py      (no arguments; exit status 1 if any check fails)

Reads (both stores are only read)
  $SWOTGAT_WORK/CONUS/zarr_store_vd2026_halfexact, $SWOTGAT_WORK/CONUS/zarr_store_vd2026
  lakes_removed_vs_vd2026.csv next to this script   the removed lakes (make_lake_targets_hx.py)

Writes
  verify_store_hx.json next to this script: totals, the number of problems and the first 20 of them

Checks, per basin group
  - the group exists in both stores, with the same date axis and the same sub-basin order (in the paper's run:
    102 groups, 2020-01-01 to 2026-09-30);
  - the ten ERA5-Land variables (*_mean) and dv_river are bitwise identical (same builder, same inputs);
  - normalization_stats are identical (both are copies of the first store's statistics);
  - dv_SWOTAE and dv_staticPLD are identical at every node that is not a removed lake, and all NaN at the removed
    lakes (153 lakes in the paper's run);
  - no target has a value before 2023-01-01, so a longer input window adds forcing only, never targets.
"""
from swotgat.paths import WORK
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

C = Path(f"{WORK}/CONUS"); N, O = C/"zarr_store_vd2026_halfexact", C/"zarr_store_vd2026"
V = Path(__file__).resolve().parent
VARS = [f"{v}_mean" for v in ["ssrd", "strd", "ro", "sro", "e", "tp", "sp", "d2m", "t2m", "fal"]]
gone = set(pd.read_csv(V/"lakes_removed_vs_vd2026.csv", dtype=str).subbasin)
basins = sorted(p.name for p in O.iterdir() if (p/"zarr.json").is_file())
bad, tot = [], dict(groups=len(basins), removed_lakes_in_store=0, removed_lake_days_old=0)
for b in basins:
    if not (N/b/"zarr.json").is_file(): bad.append((b, "missing")); continue
    n, o = xr.open_zarr(N/b, consolidated=True), xr.open_zarr(O/b, consolidated=True)
    if not np.array_equal(n.date.values, o.date.values): bad.append((b, "dates"))
    if list(n.subbasin.values.astype(str)) != list(o.subbasin.values.astype(str)): bad.append((b, "subbasin order"))
    for v in VARS + ["dv_river"]:
        if not np.array_equal(n[v].values, o[v].values, equal_nan=True): bad.append((b, f"{v} differs"))
    if dict(zarr.open_group(N/b, mode="r").attrs)["normalization_stats"] != dict(zarr.open_group(O/b, mode="r").attrs)["normalization_stats"]:
        bad.append((b, "normalization_stats"))
    subs = pd.Index(n.subbasin.values.astype(str)); rm = subs.isin(gone)
    for v in ("dv_SWOTAE", "dv_staticPLD"):
        a, q = n[v].transpose("date", "subbasin").values, o[v].transpose("date", "subbasin").values
        if not np.array_equal(a[:, ~rm], q[:, ~rm], equal_nan=True): bad.append((b, f"{v} differs at kept nodes"))
        if np.isfinite(a[:, rm]).any(): bad.append((b, f"{v} finite at a removed lake"))
        tot[f"{v}_finite"] = tot.get(f"{v}_finite", 0) + int(np.isfinite(a).sum())
        tot[f"{v}_finite_vd2026"] = tot.get(f"{v}_finite_vd2026", 0) + int(np.isfinite(q).sum())
    tot["removed_lakes_in_store"] += int(rm.sum())
    for v in ("dv_SWOTAE", "dv_staticPLD", "dv_river"):
        early = n[v].sel(date=slice(None, "2022-12-31")).values
        if np.isfinite(early).any(): bad.append((b, f"{v} finite before 2023"))
tot["problems"] = len(bad); tot["first_problems"] = bad[:20]
json.dump(tot, open(V/"verify_store_hx.json", "w"), indent=1)
print(json.dumps(tot, indent=1)); sys.exit(1 if bad else 0)
