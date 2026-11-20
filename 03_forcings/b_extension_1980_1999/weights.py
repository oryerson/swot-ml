"""Compute the ERA5-Land coverage weights of every sub-basin (1980-1999 extension, step 1).

Reads (paths relative to $SWOTGAT_WORK unless noted)
    CONUS/metadata/subbasins_conus.parquet                sub-basin polygons
    $SWOTGAT_NAS_ICE/data/ERA5-Land/average/2000_01.nc    any archive file, for the 0.1 degree grid
    CONUS/preprocess/flipped_extract_sub_basins.py        imported for its two weighting functions
        (the deployed copy of 03_forcings/a_production_2000_2025/flipped_extract_sub_basins.py)

Writes (<era5ext> = CONUS_storage_fork/eval/overnight/era5ext)
    <era5ext>/weights.npz     sparse weight matrix in CSR form, one row per sub-basin: `ptr` (row
                              pointers), `cell` (index of the grid cell in the flattened global
                              grid, latitude index * nlon + longitude index), `w` (fraction of the
                              cell covered by the polygon), and the grid itself (`nlat`, `nlon`,
                              `lat`, `lon`)
    <era5ext>/subbasins.csv   row order of the matrix: subbasin, outlet_id (basin),
                              fallback_exact (1 if the exact method was used), n_cells

Usage
    python weights.py
    Uses one process per CPU given by the environment variable SLURM_CPUS_PER_TASK (default 8).

Method
    The weights come from the two functions of the 2000-2025 production extraction:
    get_weight_matrix_fast (rasterization on a grid 10 times finer than ERA5-Land) and, where
    that gives zero coverage, get_weight_matrix (exact polygon-cell intersection). Each sub-basin
    is weighted on its own bounding box, padded by 0.1 degree and rounded outward to 0.1 degree.
    The weights depend only on the polygon and the 0.1 degree grid, because the fine raster is
    aligned to the edges of the grid cells wherever the box starts. They therefore equal the
    weights of the production run, which used one box per batch of sub-basins. In the original
    run a separate script (validate.py) checked this against the values of the Zarr store.
"""
from swotgat.paths import NAS_ICE, REPO, WORK
import sys, os, numpy as np, pandas as pd, geopandas as gpd, xarray as xr
from multiprocessing import Pool
sys.path.insert(0, f"{REPO}/vendor/swot_ml/src")
sys.path.insert(0, f"{WORK}/CONUS/preprocess")
from flipped_extract_sub_basins import get_weight_matrix_fast, get_weight_matrix   # noqa: E402
OUT = f"{WORK}/CONUS_storage_fork/eval/overnight/era5ext"
REF = f"{NAS_ICE}/data/ERA5-Land/average/2000_01.nc"
_ref = xr.open_dataset(REF, engine="h5netcdf")
LAT = _ref.latitude.values.copy(); LON = _ref.longitude.values.copy(); _ref.close()
G = None

def one(i):
    """Compute the coverage weights of sub-basin i (row i of the polygon table G).

    The polygon is weighted on its own bounding box, padded by 0.1 degree and rounded outward to
    0.1 degree, with the fast rasterization method and, if that gives zero coverage, with the
    exact method. Returns (latitude indices, longitude indices, weights, fallback flag) for the
    cells with a weight above 0. The indices refer to the global grid; the flag is 1 if the exact
    method was used.
    """
    row = G.iloc[[i]]
    minx, miny, maxx, maxy = row.total_bounds
    b = (np.floor((minx - .1) * 10) / 10, np.floor((miny - .1) * 10) / 10, np.ceil((maxx + .1) * 10) / 10, np.ceil((maxy + .1) * 10) / 10)
    li = np.flatnonzero((LAT <= b[3]) & (LAT >= b[1])); lj = np.flatnonzero((LON >= b[0]) & (LON <= b[2]))
    sub = xr.Dataset(coords={"latitude": LAT[li], "longitude": LON[lj]})
    w = get_weight_matrix_fast(sub, row); fb = 0
    if float(w.sum()) <= 0:
        w = get_weight_matrix(sub, row); fb = 1
    wv = np.asarray(w.values, float); k = np.nonzero(wv > 0)
    return li[k[0]], lj[k[1]], wv[k], fb

if __name__ == "__main__":
    G = gpd.read_parquet(f"{WORK}/CONUS/metadata/subbasins_conus.parquet")
    G.index = G.index.astype(str)
    if G.crs is None or G.crs.to_epsg() != 4326: G = G.to_crs(4326)
    print(f"{len(G):,} sub-basins, crs {G.crs}", flush=True)
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as p:
        res = p.map(one, range(len(G)), chunksize=50)
    nlon = len(LON)
    ptr = np.zeros(len(res) + 1, np.int64); ptr[1:] = np.cumsum([len(r[2]) for r in res])
    cell = np.concatenate([r[0] * nlon + r[1] for r in res]).astype(np.int64)
    wts = np.concatenate([r[2] for r in res])
    np.savez(f"{OUT}/weights.npz", ptr=ptr, cell=cell, w=wts, nlat=len(LAT), nlon=nlon, lat=LAT, lon=LON)
    pd.DataFrame({"subbasin": G.index, "outlet_id": G.outlet_id.astype(str).values,
                  "fallback_exact": [r[3] for r in res], "n_cells": np.diff(ptr)}).to_csv(f"{OUT}/subbasins.csv", index=False)
    print(f"done: {ptr[-1]:,} weights, zero-coverage sub-basins {int((np.diff(ptr) == 0).sum())}, "
          f"exact fallbacks {sum(r[3] for r in res)}", flush=True)
