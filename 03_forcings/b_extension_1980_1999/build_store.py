"""Build one basin group of the extended forcing store, which adds ERA5-Land for 1980-1999.

Reads
    $SWOTGAT_WORK/CONUS/zarr_store/<basin>   the 2000-2025 store of a_production_2000_2025 (called
        the "production" store in this stage). It is only read, never written.
    $SWOTGAT_WORK/CONUS_storage_fork/eval/overnight/era5ext/years/<year>.npy and <year>_dates.npy
        for 1980-1999, written by extract.py (array of days x sub-basins x 10 variables)
    .../era5ext/subbasins.csv   the sub-basin order of those arrays, written by weights.py

Writes
    $SWOTGAT_WORK/CONUS/zarr_store_ext1980/<basin>   (Zarr format 3, chunks of 365 days x all
        sub-basins of the basin, metadata not consolidated)

Usage
    python build_store.py <basin_index>
    basin_index (0..101 for the paper's 102 basins) indexes the sorted basin groups of the
    production store. Run one job per basin, for example as a Slurm array.

Method
    The new group has the same `date` and `subbasin` coordinates as the production group. The
    production date axis already starts on 1980-01-01, but its ERA5-Land values before
    2000-01-01 are all NaN (asserted here).
    - The 10 ERA5-Land means (<variable>_mean for ssrd, strd, ro, sro, e, tp, sp, d2m, t2m, fal)
      are copied, and their first 7305 days (1980-01-01 to 1999-12-31) are filled from the
      yearly extraction files. Each yearly file must start on 1 January and have 365 or 366 days.
    - The target-side variables that the swot_ml data loader may read (dv_SWOTAE, dv_staticPLD,
      dv_river, dv_e_SWOTAE, dv_e_staticPLD, wse, wse_u) are copied unchanged where present.
    - The group attribute `normalization_stats` is copied unchanged, so the extended store is
      scaled exactly like the production store.
    The script prints one JSON line with the number of finite pre-2000 values per variable.
"""
from swotgat.paths import WORK
import sys, json, numpy as np, pandas as pd, xarray as xr, zarr
from pathlib import Path
Z = Path(f"{WORK}/CONUS/zarr_store")
EXT = Path(f"{WORK}/CONUS/zarr_store_ext1980")
OUT = Path(f"{WORK}/CONUS_storage_fork/eval/overnight/era5ext")
VARS = ["ssrd", "strd", "ro", "sro", "e", "tp", "sp", "d2m", "t2m", "fal"]
TARGETS = ["dv_SWOTAE", "dv_staticPLD", "dv_river", "dv_e_SWOTAE", "dv_e_staticPLD", "wse", "wse_u"]
N_EARLY = 7305                                                     # 1980-01-01 .. 1999-12-31
basins = sorted(p.name for p in Z.iterdir() if (p/"zarr.json").is_file())
b = basins[int(sys.argv[1])]
SB = pd.read_csv(OUT/"subbasins.csv", dtype=str); pos = {s: i for i, s in enumerate(SB.subbasin)}
src = xr.open_zarr(Z/b, consolidated=False)
subs = [str(s) for s in src.subbasin.values]; idx = np.array([pos[s] for s in subs])
dates = pd.to_datetime(src.date.values); assert dates[0] == pd.Timestamp("1980-01-01") and dates[N_EARLY] == pd.Timestamp("2000-01-01")
# 1980-1999 block (days x sub-basins x variables) from the yearly extraction files (memory-mapped)
early = np.full((N_EARLY, len(subs), len(VARS)), np.nan); r = 0
for y in range(1980, 2000):
    A = np.load(OUT/f"years/{y}.npy", mmap_mode="r"); D = pd.to_datetime(np.load(OUT/f"years/{y}_dates.npy"))
    assert D[0] == pd.Timestamp(f"{y}-01-01") and len(D) == (366 if y % 4 == 0 else 365), (y, D[0], len(D))
    early[r:r + len(D)] = A[:, idx, :]; r += len(D)
assert r == N_EARLY
out = xr.Dataset(coords={"date": src.date, "subbasin": src.subbasin})
enc = {"date": {"units": "days since 1980-01-01 00:00:00", "calendar": "proleptic_gregorian", "dtype": "int64"}}
out.to_zarr(EXT/b, mode="w", zarr_format=3, encoding=enc, consolidated=False)
log = {}
for v in [f"{x}_mean" for x in VARS] + [t for t in TARGETS if t in src]:
    a = src[v].load().transpose("date", "subbasin")
    if v.endswith("_mean") and v[:-5] in VARS:
        pre = a.values[:N_EARLY]
        assert np.isnan(pre).all(), f"{b} {v}: production has values before 2000"
        vals = a.values.copy(); vals[:N_EARLY] = early[:, :, VARS.index(v[:-5])]
        a = a.copy(data=vals); log[v] = int(np.isfinite(early[:, :, VARS.index(v[:-5])]).sum())
    xr.Dataset({v: a}).to_zarr(EXT/b, mode="a", zarr_format=3, consolidated=False,
                               encoding={v: {"chunks": (365, len(subs))}})
g = zarr.open_group(EXT/b, mode="r+"); g.attrs["normalization_stats"] = dict(zarr.open_group(Z/b, mode="r").attrs)["normalization_stats"]
print(json.dumps(dict(basin=b, subbasins=len(subs), early_finite=log)))
