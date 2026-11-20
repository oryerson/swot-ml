# Same code as b_extension_1980_1999/extract.py with one change: the working folder is
# <forcing>/work. (make_scripts.py records how this file was derived.)
"""Extract one year of daily ERA5-Land means for every graph v3.5 sub-basin.

Reads (<forcing> = $SWOTGAT_WORK/CONUS_storage_fork/eval/v35_inputs/forcing)
    <forcing>/work/weights.npz      coverage weights of all v3.5 sub-basins (weights_v35.py)
    $SWOTGAT_NAS_ICE/data/ERA5-Land/{accumulated,average}/<year>_<month>.nc   the ERA5-Land archive

Writes
    <forcing>/work/years/<year>.npy         float64 array (days, sub-basins, 10 variables); the
                                            sub-basin order is that of <forcing>/work/subbasins.csv
    <forcing>/work/years/<year>_dates.npy   the dates of the first axis

Usage
    python extract_v35.py <year> [months]
    months is an optional comma-separated list (for example 1,2,3). With it, the output files are
    named <year>_m<months joined by "-">. The training store needs the years 2020 to 2025
    (build_store_v35.py), one job per year.

Method
    The 10 variables are ssrd, strd, ro, sro, e, tp (from the "accumulated" files) and sp, d2m,
    t2m, fal (from the "average" files), in this order on the last axis. The daily value of a
    sub-basin is the coverage-weighted mean of its grid cells. Cells that are NaN are left out
    of both the weighted sum and the sum of weights. This is what xarray's weighted mean does and
    what the 2000-2025 production extraction used (get_results_df). A sub-basin with no finite
    cell gets NaN. All arithmetic is in float64.
"""
from swotgat.paths import NAS_ICE, WORK
import sys, numpy as np, pandas as pd, xarray as xr, scipy.sparse as sp
OUT = f"{WORK}/CONUS_storage_fork/eval/v35_inputs/forcing/work"
E = f"{NAS_ICE}/data/ERA5-Land"
VARS = ["ssrd", "strd", "ro", "sro", "e", "tp", "sp", "d2m", "t2m", "fal"]
year = int(sys.argv[1]); months = range(1, 13) if len(sys.argv) < 3 else [int(m) for m in sys.argv[2].split(",")]
Z = np.load(f"{OUT}/weights.npz"); ptr, cell, w, nlon = Z["ptr"], Z["cell"], Z["w"], int(Z["nlon"])
ii, jj = cell // nlon, cell % nlon
i0, i1, j0, j1 = ii.min(), ii.max() + 1, jj.min(), jj.max() + 1           # grid box that holds every weighted cell
loc = (ii - i0) * (j1 - j0) + (jj - j0)
nsub = len(ptr) - 1
W = sp.csr_matrix((w, loc, ptr), shape=(nsub, (i1 - i0) * (j1 - j0)))
outs, dates = [], []
for m in months:
    parts = []
    for k in ("accumulated", "average"):
        ds = xr.open_dataset(f"{E}/{k}/{year}_{m:02d}.nc", engine="h5netcdf").isel(latitude=slice(i0, i1), longitude=slice(j0, j1))
        parts.append(ds)
    t = pd.to_datetime(parts[0].valid_time.values); assert (t == pd.to_datetime(parts[1].valid_time.values)).all()
    arr = np.full((len(t), nsub, len(VARS)), np.nan)
    for v_i, v in enumerate(VARS):
        src = parts[0] if v in parts[0] else parts[1]
        X = src[v].values.astype(np.float64).reshape(len(t), -1).T          # cells x days
        M = np.isfinite(X); X0 = np.where(M, X, 0.0)
        num = W @ X0; den = W @ M.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            arr[:, :, v_i] = (num / den).T
    for ds in parts: ds.close()
    outs.append(arr); dates.append(t)
    print(f"{year}-{m:02d}: {len(t)} days", flush=True)
A = np.concatenate(outs); D = np.concatenate([d.values for d in dates])
tag = f"{year}" if len(sys.argv) < 3 else f"{year}_m{sys.argv[2].replace(',', '-')}"
np.save(f"{OUT}/years/{tag}.npy", A); np.save(f"{OUT}/years/{tag}_dates.npy", D)
print(f"saved {tag}: {A.shape}, NaN share {np.isnan(A).mean():.4f}", flush=True)
