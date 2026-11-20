"""Build one basin group of the template training store (forcing 2020-2026 and SWOT version-D targets).

The "template store" is the training store on the node set of the first graph version, with the HALF-exact lake
targets (HALF = the lake outlier filter of Trudel et al.; see 04_swot_targets/a_lakes/build_halfexact.py). The
paper's training store on graph v3.5 is assembled from it: 03_forcings/d_graph_v35_stores/build_store_v35.py
copies the lake targets, and the forcing and river targets of unchanged nodes, from this store.

This file is 03_forcings/c_extension_2026/build_store_vd2026.py with two changes: the output store
(zarr_store_vd2026_halfexact) and the lake-target table (the HALF-exact one). make_builder.py records how it was
derived.

Usage: python build_store_vd2026_halfexact.py <basin_index 0..101> [--start 2020-01-01] [--end 2026-09-30]
  basin_index indexes the sorted basin groups of the first store. Run one job per basin.

Reads (under $SWOTGAT_WORK)
  CONUS/zarr_store/<basin>                                        the first data store (only read)
  CONUS_storage_fork/eval/overnight/era5_2026/years/2026_raw.npy, 2026_raw_dates.npy
                                                                  ERA5-Land means extracted for 2026
  CONUS_storage_fork/eval/overnight/era5_2026/susq_map.json       linear mapping for two sub-basins (see Method)
  CONUS_storage_fork/eval/overnight/era5ext/subbasins.csv         sub-basin order of the extraction arrays
  CONUS_storage_fork/eval/v21_retrain/store/lake_targets_vd2026_halfexact.parquet
                                                                  lake targets (make_lake_targets_hx.py)
  CONUS_storage_fork/eval/swotext/riversp_vD_present_graph.parquet   river record (river_build_v2.py)

Writes
  CONUS/zarr_store_vd2026_halfexact/<basin>, and prints one JSON line of counts

Layout
  Dates --start to --end, daily. Every variable is a float64 (date, subbasin) array in chunks of 365 days, as in
  the first store. Sub-basins and their order are those of the first store's group.

Method
  ERA5-Land means (10 variables <variable>_mean): values of the first store up to 2025-12-31, then 2026 from
      2026_raw.npy. The 2026 extraction uses the same weights and arithmetic as the first store (it reproduced
      December 2025 to floating-point noise). Two Susquehanna sub-basins have an older polygon in the first
      store; their 2026 values are mapped onto the first store's values with the linear fit a + b x value of
      susq_map.json.
  dv_SWOTAE: lake storage change from the SWOT area-elevation curve, else from the 3D-LAKES curve.
  dv_staticPLD: lake level change x the constant lake area of the SWOT Prior Lake Database; the model's fallback
      target.
  dv_river: the river record extended to the latest SWOT pass, not masked. The quality drop list is applied when
      the model loads the data (`target_drop_masks` in the model configuration).
  The group attribute normalization_stats is copied unchanged from the first store, so that every store scales
      its variables the same way. Do not run update_zarr_stats on this store.

Notes
  Forcing up to 2025-12-31 is identical to the first store's.
"""
from swotgat.paths import WORK
import sys, json, argparse, numpy as np, pandas as pd, xarray as xr, zarr
from pathlib import Path
C = Path(f"{WORK}/CONUS"); Z = C/"zarr_store"; NEW = C/"zarr_store_vd2026_halfexact"
E = Path(f"{WORK}/CONUS_storage_fork/eval/overnight/era5_2026"); EXT = E.parent/"era5ext"
S = Path(f"{WORK}/CONUS_storage_fork/eval/swotext")
VARS = ["ssrd", "strd", "ro", "sro", "e", "tp", "sp", "d2m", "t2m", "fal"]
ap = argparse.ArgumentParser(); ap.add_argument("i", type=int); ap.add_argument("--start", default="2020-01-01"); ap.add_argument("--end", default="2026-09-30")
a = ap.parse_args()
basins = sorted(p.name for p in Z.iterdir() if (p/"zarr.json").is_file()); b = basins[a.i]
src = xr.open_zarr(Z/b, consolidated=True); subs = [str(s) for s in src.subbasin.values]
dates = pd.date_range(a.start, a.end, freq="D"); old = dates[dates <= "2025-12-31"]; new = dates[dates > "2025-12-31"]
SB = pd.read_csv(EXT/"subbasins.csv", dtype=str); pos = {s: i for i, s in enumerate(SB.subbasin)}; idx = np.array([pos[s] for s in subs])
A26 = np.load(E/"years/2026_raw.npy", mmap_mode="r"); D26 = pd.to_datetime(np.load(E/"years/2026_raw_dates.npy"))
r26 = np.array([np.flatnonzero(D26 == d)[0] if (D26 == d).any() else -1 for d in new]); assert (r26 >= 0).all(), f"2026 forcing missing for {new[r26 < 0][:3]}"
SM = json.load(open(E/"susq_map.json"))
out = xr.Dataset(coords={"date": dates, "subbasin": src.subbasin})
enc = {"date": {"units": "days since 2020-01-01 00:00:00", "calendar": "proleptic_gregorian", "dtype": "int64"}}
out.to_zarr(NEW/b, mode="w", zarr_format=3, encoding=enc, consolidated=False)
def put(name, arr):
    """Write the (date, subbasin) array arr to the new store as the float64 variable `name`.

    The variable is stored in chunks of 365 days that span all sub-basins.
    """
    da = xr.DataArray(arr.astype("float64"), dims=("date", "subbasin"), coords={"date": dates, "subbasin": src.subbasin})
    xr.Dataset({name: da}).to_zarr(NEW/b, mode="a", zarr_format=3, consolidated=False, encoding={name: {"chunks": (365, len(subs))}})
log = dict(basin=b, subbasins=len(subs), dates=f"{dates[0].date()}..{dates[-1].date()}")
for k, v in enumerate(VARS):
    p = src[f"{v}_mean"].sel(date=old).transpose("date", "subbasin").values
    q = np.array(A26[r26][:, idx, k], dtype="float64")
    for s, m in SM.items():
        if s in subs: j = subs.index(s); q[:, j] = m[v]["a"] + m[v]["b"]*q[:, j]
    put(f"{v}_mean", np.concatenate([p, q]))
    log[f"{v}_nan_2026"] = int(np.isnan(q).sum()); log[f"{v}_nan_old"] = int(np.isnan(p).sum())
def grid(df, col):
    """Return column `col` of the table df as a (date, subbasin) array on the axes of the new store.

    df has one row per sub-basin and day (columns subbasin, date). Rows whose date or sub-basin is
    not on the axes are skipped, and cells without a row are NaN.
    """
    g = np.full((len(dates), len(subs)), np.nan)
    if len(df):
        di = dates.get_indexer(pd.to_datetime(df.date)); si = pd.Index(subs).get_indexer(df.subbasin.astype(str)); ok = (di >= 0) & (si >= 0)
        g[di[ok], si[ok]] = df[col].values[ok]
    return g
L = pd.read_parquet(Path(f"{WORK}/CONUS_storage_fork/eval/v21_retrain/store/lake_targets_vd2026_halfexact.parquet")); L = L[L.subbasin.isin(subs)]
R = pd.read_parquet(S/"riversp_vD_present_graph.parquet", columns=["subbasin", "date", "dv_river"]); R = R[R.subbasin.astype(str).isin(subs)]
for name, df, col in (("dv_SWOTAE", L, "dv_SWOTAE"), ("dv_staticPLD", L, "dv_staticPLD"), ("dv_river", R, "dv_river")):
    g = grid(df, col); put(name, g); log[f"{name}_finite"] = int(np.isfinite(g).sum())
    log[f"{name}_prod_finite_2023_25"] = int(np.isfinite(src[name].sel(date=slice("2023-01-01", "2025-12-31")).values).sum()) if name in src else -1
grp = zarr.open_group(NEW/b, mode="r+"); grp.attrs["normalization_stats"] = dict(zarr.open_group(Z/b, mode="r").attrs)["normalization_stats"]
zarr.consolidate_metadata(str(NEW/b))
print(json.dumps(log))
