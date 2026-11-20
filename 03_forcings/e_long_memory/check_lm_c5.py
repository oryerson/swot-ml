"""Check the 1980-2026 long-memory store of build_lm_c5.py before any prediction reads it.

Reads (paths relative to $SWOTGAT_WORK/CONUS_storage_fork)
    eval/c5_paper/longmem_v35_c5_8026   the store to check (build_lm_c5.py)
    features/longmem_v35                the 2009-2026 reference store (build_longmem_v35.py)

Writes
    eval/c5_paper/longmem_v35_c5_8026/CHECK.json   (also printed, followed by `LM CHECK PASSED`
    or `LM CHECK FAILED`)

Usage: python check_lm_c5.py        (no arguments; exit code 1 on failure)

The check fails unless
    - every basin group of longmem_v35 has a group in the new store with the same `subbasin`
      coordinate (values and order) and with daily dates from 1980-01-01 to 2026-09-30 without
      gaps;
    - lm_up_ssi3 and lm_up_ssi12 are bit-identical (float16) to longmem_v35 from 2009-01-01 to
      2026-09-30 at every node;
    - every value is finite.
It also reports, per input, the number of days among the first 366 on which all nodes of a group
are 0, summed over the groups. These are the days written as 0 because the accumulation window
reaches back before 1980-01-01 (the first 90 days for ssi3, the first 364 for ssi12). The count
is reported, not tested.
"""
from swotgat.paths import WORK
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
F = Path(f"{WORK}/CONUS_storage_fork"); NEW = F/"eval/c5_paper/longmem_v35_c5_8026"; OLD = F/"features/longmem_v35"
VARS = ["lm_up_ssi3", "lm_up_ssi12"]; want = pd.date_range("1980-01-01", "2026-09-30")
groups = sorted(p.name for p in OLD.iterdir() if (p/"zarr.json").exists() or (p/".zmetadata").exists())
fails, zero_fill, n_cmp = [], {v: 0 for v in VARS}, 0
for b in groups:
    if not (NEW/b).exists(): fails.append(f"{b}: no group"); continue
    n, o = xr.open_zarr(NEW/b, consolidated=True), xr.open_zarr(OLD/b, consolidated=True)
    if not np.array_equal(n.subbasin.values.astype(str), o.subbasin.values.astype(str)): fails.append(f"{b}: subbasin coordinate differs"); continue
    if not np.array_equal(pd.DatetimeIndex(n.date.values), want): fails.append(f"{b}: dates are not 1980-01-01..2026-09-30 daily"); continue
    on = n.sel(date=slice("2009-01-01", "2026-09-30")); oo = o.sel(date=slice("2009-01-01", "2026-09-30"))
    for v in VARS:
        a, c = np.asarray(on[v].values), np.asarray(oo[v].values); n_cmp += a.size
        if a.dtype != c.dtype or a.shape != c.shape or not np.array_equal(a.view(np.uint16), c.view(np.uint16)): fails.append(f"{b}:{v} differs from longmem_v35 on 2009-2026")
        full = np.asarray(n[v].values)
        if not np.isfinite(full).all(): fails.append(f"{b}:{v} non-finite values")
        early = full[:366]; zero_fill[v] += int((early == 0).all(axis=1).sum()) if early.ndim == 2 else 0
out = dict(ok=not fails, groups=len(groups), values_compared_2009_2026=n_cmp, fails=fails[:50],
           days_all_zero_in_1980_summed_over_groups=zero_fill, note="ssi3 is zero-filled for its first 90 days and ssi12 for its first 364 (no 1979 forcing)")
json.dump(out, open(NEW/"CHECK.json", "w"), indent=1); print(json.dumps(out, indent=1)); print("LM CHECK", "PASSED" if not fails else "FAILED")
sys.exit(0 if not fails else 1)
