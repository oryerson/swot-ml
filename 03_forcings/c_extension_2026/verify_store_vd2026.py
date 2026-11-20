"""Check the 2020-2026 store (zarr_store_vd2026) against the 2000-2025 "production" store.

Reads
    $SWOTGAT_WORK/CONUS/zarr_store           production store (a_production_2000_2025)
    $SWOTGAT_WORK/CONUS/zarr_store_vd2026    store written by build_store_vd2026.py

Writes nothing. Prints the number of basin groups and problems, the first 20 problems, and a JSON
line of target counts. The exit code is 1 if there is any problem, else 0.

Usage: python verify_store_vd2026.py        (no arguments)

Checks, for every basin group of the production store
    - the group exists in the new store;
    - its dates are consecutive days;
    - its sub-basins are the same, in the same order;
    - each of the 10 ERA5-Land means is bitwise equal to production from 2020-01-01 to
      2025-12-31 (NaN counts as equal to NaN);
    - the ERA5-Land values from 2026-01-01 on are finite at every sub-basin where production is
      finite on 2025-12-31;
    - `normalization_stats` is identical.
It also counts the finite values of the targets dv_SWOTAE, dv_staticPLD and dv_river, in total
and from 2026-01-01 on (reported, not tested).
"""
from swotgat.paths import WORK
import sys, json, numpy as np, pandas as pd, xarray as xr, zarr
from pathlib import Path
C = Path(f"{WORK}/CONUS"); Z, N = C/"zarr_store", C/"zarr_store_vd2026"
VARS = ["ssrd", "strd", "ro", "sro", "e", "tp", "sp", "d2m", "t2m", "fal"]
basins = sorted(p.name for p in Z.iterdir() if (p/"zarr.json").is_file()); bad = []; tot = {}
for b in basins:
    if not (N/b/"zarr.json").is_file(): bad.append((b, "missing")); continue
    n = xr.open_zarr(N/b, consolidated=True); p = xr.open_zarr(Z/b, consolidated=True)
    d = pd.to_datetime(n.date.values)
    if not (np.diff(d.values).astype("timedelta64[D]").astype(int) == 1).all(): bad.append((b, "dates not daily"))
    if list(n.subbasin.values.astype(str)) != list(p.subbasin.values.astype(str)): bad.append((b, "subbasin order"))
    for v in VARS:
        a = n[f"{v}_mean"].sel(date=slice("2020-01-01", "2025-12-31")).values; q = p[f"{v}_mean"].sel(date=slice("2020-01-01", "2025-12-31")).values
        if not np.array_equal(a, q, equal_nan=True): bad.append((b, f"{v} 2020-25 differs"))
        fin25 = np.isfinite(p[f"{v}_mean"].sel(date="2025-12-31").values); n26 = n[f"{v}_mean"].sel(date=slice("2026-01-01", None)).values
        if not np.isfinite(n26[:, fin25]).all(): bad.append((b, f"{v} 2026 NaN where 2025 finite"))
    if dict(zarr.open_group(N/b, mode="r").attrs)["normalization_stats"] != dict(zarr.open_group(Z/b, mode="r").attrs)["normalization_stats"]: bad.append((b, "norm stats"))
    for t in ("dv_SWOTAE", "dv_staticPLD", "dv_river"):
        tot[t] = tot.get(t, 0) + int(np.isfinite(n[t].values).sum()); tot[t + "_2026"] = tot.get(t + "_2026", 0) + int(np.isfinite(n[t].sel(date=slice("2026-01-01", None)).values).sum())
print(f"groups {len(basins)}; problems {len(bad)}"); print(bad[:20]); print(json.dumps(tot)); sys.exit(1 if bad else 0)
