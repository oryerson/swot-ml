"""Verify the extended forcing store (1980-2025) against the 2000-2025 "production" store.

Reads
    $SWOTGAT_WORK/CONUS/zarr_store            production store (a_production_2000_2025)
    $SWOTGAT_WORK/CONUS/zarr_store_ext1980    extended store written by build_store.py
    $SWOTGAT_WORK/CONUS_storage_fork/eval/overnight/era5ext/years/{1985,1999}.npy and
        subbasins.csv                         the extraction (extract.py, weights.py)
    $SWOTGAT_WORK/CONUS/era5_conus_v10pm_S1_storage_20260930_143855/config.json
        the training configuration of an earlier model configuration (v10pm), used only to
        call the data loader's own statistics routine

Writes nothing. Prints the problems found and `VERIFY PASS` or `VERIFY FAILED`; the exit code is
0 on pass and 1 on failure.

Usage: python verify_store.py        (no arguments; run after all build_store.py jobs)

Checks
    1. Every basin group of the production store exists in the extended store, with identical
       `normalization_stats` and identical `date` and `subbasin` coordinates.
    2. Every variable of the extended store is bitwise identical to production from 2000-01-01
       on (day index 7305 onward; NaN counts as equal to NaN). Before 2000, the non-ERA5
       variables are also identical to production, and the ERA5-Land means of 1985 and 1999
       equal the extraction files exactly (a spot check of two years).
    3. The data loader of the vendored swot_ml package (`get_train_ds_stats`) returns identical
       scaling of the dynamic inputs (`d_scale`) whether its `zarr_dir` points to the production
       store or to the extended store.
"""
from swotgat.paths import REPO, WORK
import sys, json, numpy as np, pandas as pd, xarray as xr, zarr
from pathlib import Path
sys.path.insert(0, f"{REPO}/vendor/swot_ml/src")
Z = Path(f"{WORK}/CONUS/zarr_store"); EXT = Path(f"{WORK}/CONUS/zarr_store_ext1980")
OUT = Path(f"{WORK}/CONUS_storage_fork/eval/overnight/era5ext")
VARS = ["ssrd", "strd", "ro", "sro", "e", "tp", "sp", "d2m", "t2m", "fal"]
SB = pd.read_csv(OUT/"subbasins.csv", dtype=str); pos = {s: i for i, s in enumerate(SB.subbasin)}
probs = []; nb = 0
for p in sorted(Z.iterdir()):
    if not (p/"zarr.json").is_file(): continue
    nb += 1; b = p.name
    if not (EXT/b/"zarr.json").is_file(): probs.append(f"{b}: missing"); continue
    a0 = dict(zarr.open_group(Z/b, mode="r").attrs)["normalization_stats"]; a1 = dict(zarr.open_group(EXT/b, mode="r").attrs)["normalization_stats"]
    if json.dumps(a0, sort_keys=True) != json.dumps(a1, sort_keys=True): probs.append(f"{b}: stats differ")
    s0 = xr.open_zarr(Z/b, consolidated=False); s1 = xr.open_zarr(EXT/b, consolidated=False)
    if not (s0.date.values == s1.date.values).all() or not (s0.subbasin.values == s1.subbasin.values).all(): probs.append(f"{b}: coords differ")
    for v in s1.data_vars:
        x0 = s0[v].transpose("date", "subbasin").values; x1 = s1[v].transpose("date", "subbasin").values
        if not np.array_equal(x0[7305:], x1[7305:], equal_nan=True): probs.append(f"{b} {v}: post-2000 differs")
        if v.endswith("_mean") and v[:-5] in VARS:
            pass
        elif not np.array_equal(x0[:7305], x1[:7305], equal_nan=True): probs.append(f"{b} {v}: pre-2000 target differs")
    # spot check: pre-2000 ERA5-Land against the extraction files, for 1985 and 1999
    idx = np.array([pos[str(s)] for s in s1.subbasin.values])
    for y in (1985, 1999):
        A = np.load(OUT/f"years/{y}.npy", mmap_mode="r"); t0 = (pd.Timestamp(f"{y}-01-01") - pd.Timestamp("1980-01-01")).days
        for vi, v in enumerate(VARS):
            x1 = s1[f"{v}_mean"].transpose("date", "subbasin").values[t0:t0 + A.shape[0]]
            if not np.array_equal(x1, A[:, idx, vi], equal_nan=True): probs.append(f"{b} {v}: {y} != extraction")
print(f"basins checked {nb}; problems {len(probs)}"); [print("  ", x) for x in probs[:40]]
# (3) the data loader's own normalization statistics, computed from each store
from config import Config
import data.cached_basingraphdataset as cbd
cfg = Config.from_file(f"{WORK}/CONUS/era5_conus_v10pm_S1_storage_20260930_143855/config.json")
st0 = cbd.get_train_ds_stats(cfg, static=False)
cfg.zarr_dir = EXT
st1 = cbd.get_train_ds_stats(cfg, static=False)
same = json.dumps(st0["d_scale"], sort_keys=True, default=str) == json.dumps(st1["d_scale"], sort_keys=True, default=str)
print("loader d_scale identical:", same)
print("VERIFY PASS" if (not probs and same) else "VERIFY FAILED")
sys.exit(0 if (not probs and same) else 1)
