"""Build one basin group of the 2020-2026 training store (zarr_store_vd2026).

"vD" is version D of the SWOT lake and river products. Paths below are relative to $SWOTGAT_WORK;
<swotext> = CONUS_storage_fork/eval/swotext, <era5_2026> = CONUS_storage_fork/eval/overnight/
era5_2026 and <era5ext> = CONUS_storage_fork/eval/overnight/era5ext.

Reads
    CONUS/zarr_store/<basin>          the 2000-2025 "production" store (a_production_2000_2025).
                                      It is only read, never written.
    <era5_2026>/years/2026_raw.npy and 2026_raw_dates.npy   the 2026 extraction (prep_2026.py)
    <era5ext>/subbasins.csv           the sub-basin order of that array (weights.py)
    <era5_2026>/susq_map.json         correction for two Susquehanna sub-basins
                                      (store_inputs_vd.py)
    <swotext>/lake_targets_vd2026.parquet          lake targets (store_inputs_vd.py)
    <swotext>/riversp_vD_present_graph.parquet     river target

Writes
    CONUS/zarr_store_vd2026/<basin>   (Zarr format 3, float64, chunks of 365 days x all
                                      sub-basins of the basin as in production, consolidated
                                      metadata)

Usage
    python build_store_vd2026.py <basin_index> [--start 2020-01-01] [--end 2026-09-30]
    basin_index (0..101 for the paper's 102 basins) indexes the sorted basin groups of the
    production store. Run one job per basin.

Contents (dates --start to --end)
    - The 10 ERA5-Land means (<variable>_mean): copied from production up to 2025-12-31, and
      taken from 2026_raw.npy after that. The 2026 extraction uses the same weights and
      arithmetic as production (verified on December 2025 to floating-point noise). The two
      Susquehanna sub-basins whose production forcing came from an older polygon are mapped onto
      production with the fitted lines of susq_map.json.
    - dv_SWOTAE: lake storage change from the version-D SWOT area-elevation curve after the HALF
      outlier filter, otherwise from 3D-LAKES. dv_staticPLD: the same lake series with the static
      PLD lake area (the model's fallback target).
    - dv_river: the version-D river record extended to the present, not masked. The training
      configuration's `target_drop_masks` applies the paper's list
      (drop_river_paper_min_present.parquet).
    - The group attribute `normalization_stats` is copied unchanged from production, so the store
      is scaled like the first store. Statistics are not recomputed here.
    The script prints one JSON line with NaN and finite counts per variable.

Notes
    The hindcast and test predictions read ERA5-Land from the production store
    (run_predict_ext.py --zarr), which is identical to this store up to 2025-12-31.
"""
from swotgat.paths import WORK
import sys, json, argparse, numpy as np, pandas as pd, xarray as xr, zarr
from pathlib import Path
C = Path(f"{WORK}/CONUS"); Z = C/"zarr_store"; NEW = C/"zarr_store_vd2026"
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
    """Append the array arr (dates x sub-basins) to the new group as the float64 variable `name`."""
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
    """Return column col of the long table df as a (dates, sub-basins) array, NaN where no row.

    df has the columns date and subbasin; rows whose date or sub-basin is not in the store are
    ignored.
    """
    g = np.full((len(dates), len(subs)), np.nan)
    if len(df):
        di = dates.get_indexer(pd.to_datetime(df.date)); si = pd.Index(subs).get_indexer(df.subbasin.astype(str)); ok = (di >= 0) & (si >= 0)
        g[di[ok], si[ok]] = df[col].values[ok]
    return g
L = pd.read_parquet(S/"lake_targets_vd2026.parquet"); L = L[L.subbasin.isin(subs)]
R = pd.read_parquet(S/"riversp_vD_present_graph.parquet", columns=["subbasin", "date", "dv_river"]); R = R[R.subbasin.astype(str).isin(subs)]
for name, df, col in (("dv_SWOTAE", L, "dv_SWOTAE"), ("dv_staticPLD", L, "dv_staticPLD"), ("dv_river", R, "dv_river")):
    g = grid(df, col); put(name, g); log[f"{name}_finite"] = int(np.isfinite(g).sum())
    log[f"{name}_prod_finite_2023_25"] = int(np.isfinite(src[name].sel(date=slice("2023-01-01", "2025-12-31")).values).sum()) if name in src else -1
grp = zarr.open_group(NEW/b, mode="r+"); grp.attrs["normalization_stats"] = dict(zarr.open_group(Z/b, mode="r").attrs)["normalization_stats"]
zarr.consolidate_metadata(str(NEW/b))
print(json.dumps(log))
