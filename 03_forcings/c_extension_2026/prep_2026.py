"""Prepare the ERA5-Land forcing after 2025: merge, validate, check and extract.

download_2026.py fetches the files; this script turns them into daily sub-basin means. Run it in
the Python environment of the vendored swot_ml package. Paths below are relative to
$SWOTGAT_WORK; <era5_2026> = CONUS_storage_fork/eval/overnight/era5_2026 and
<era5ext> = CONUS_storage_fork/eval/overnight/era5ext.

Usage
    python prep_2026.py merge
    python prep_2026.py validate
    python prep_2026.py check
    python prep_2026.py extract YYYY [M,M] [--src nas|raw] [--freq F]

    merge      Turns every <era5_2026>/raw/average_<freq>/YYYY_MM.zip into YYYY_MM.nc, with the
               files of the single variables merged as in the archive.
    validate   Compares the downloads of December 2025 (<era5_2026>/raw/val) with the archive
               ($SWOTGAT_NAS_ICE/data/ERA5-Land) on the same grid points: the accumulated file,
               and the average file of each sampling frequency. Prints which frequency
               reproduces the archive and writes <era5_2026>/validate.json.
    check      Extracts December 2025 from the archive files through this script's
               coordinate-matched path and compares the result with the production store
               (CONUS/zarr_store, read only) in every basin group. This proves the path before
               any new month is trusted.
    extract    Writes the daily sub-basin means of the 10 store variables to
               <era5_2026>/years/<year>_<src>.npy (and <year>_<src>_dates.npy). M,M is an
               optional list of months (default 1 to 12); the loop stops at the first month that
               is not downloaded. --src is raw (default, the downloads) or nas (the archive);
               --freq is the sampling frequency of the downloaded daily means (default 6_hourly).
               Same weights (<era5ext>/weights.npz), NaN handling and float64 arithmetic as
               b_extension_1980_1999/extract.py.

Notes
    Nothing here writes to CONUS/zarr_store or CONUS/zarr_store_ext1980.
"""
from swotgat.paths import NAS_ICE, WORK
import sys, json, glob, zipfile, tempfile, os, numpy as np, pandas as pd, xarray as xr, scipy.sparse as sp
from pathlib import Path
D = Path(f"{WORK}/CONUS_storage_fork/eval/overnight/era5_2026"); RAW = D/"raw"
EXT = Path(f"{WORK}/CONUS_storage_fork/eval/overnight/era5ext")
NAS = Path(f"{NAS_ICE}/data/ERA5-Land"); ZS = Path(f"{WORK}/CONUS/zarr_store")
VARS = ["ssrd", "strd", "ro", "sro", "e", "tp", "sp", "d2m", "t2m", "fal"]
Z = np.load(EXT/"weights.npz"); ptr, cell, w, nlon = Z["ptr"], Z["cell"], Z["w"], int(Z["nlon"]); LAT, LON = Z["lat"], Z["lon"]
ii, jj = cell // nlon, cell % nlon; i0, i1, j0, j1 = ii.min(), ii.max() + 1, jj.min(), jj.max() + 1
loc = (ii - i0)*(j1 - j0) + (jj - j0); nsub = len(ptr) - 1
W = sp.csr_matrix((w, loc, ptr), shape=(nsub, (i1 - i0)*(j1 - j0))); BLAT, BLON = LAT[i0:i1], LON[j0:j1]
def box(ds):
    """Cut a dataset to the grid box of the weights.

    Cells are matched by coordinate, to 1e-3 degree (the archive files are global, the downloaded
    files cover only the conterminous US).
    """
    lat = ds.latitude.values; lon = ds.longitude.values
    ia = np.array([np.argmin(np.abs(lat - v)) for v in BLAT]); ja = np.array([np.argmin(np.abs(lon - v)) for v in BLON])
    assert np.abs(lat[ia] - BLAT).max() < 1e-3 and np.abs(lon[ja] - BLON).max() < 1e-3, "grid points missing from file"
    return ds.isel(latitude=ia, longitude=ja)
def files(src, y, m, freq):
    """Return the paths of the two files of one month (accumulated, average).

    src "nas" selects the archive; any other value selects the downloads, whose daily means have
    the sampling frequency freq.
    """
    if src == "nas": return NAS/"accumulated"/f"{y}_{m:02d}.nc", NAS/"average"/f"{y}_{m:02d}.nc"
    return RAW/"accumulated"/f"{y}_{m:02d}.nc", RAW/f"average_{freq}"/f"{y}_{m:02d}.nc"
def tname(ds):
    """Return the name of the time coordinate of ds: "valid_time" if present, otherwise "time"."""
    return "valid_time" if "valid_time" in ds.coords else "time"
def month_means(src, y, m, freq):
    """Compute the daily coverage-weighted means of one month.

    NaN cells are left out of both the weighted sum and the sum of weights.
    Returns (dates, array of days x sub-basins x variables).
    """
    parts = [box(xr.open_dataset(f)) for f in files(src, y, m, freq)]
    t = pd.to_datetime(parts[0][tname(parts[0])].values); t1 = pd.to_datetime(parts[1][tname(parts[1])].values).normalize()
    assert (t.normalize() == t1).all(), (t[:3], t1[:3])
    arr = np.full((len(t), nsub, len(VARS)), np.nan)
    for k, v in enumerate(VARS):
        s = parts[0] if v in parts[0] else parts[1]
        X = s[v].values.astype(np.float64).reshape(len(t), -1).T; M = np.isfinite(X)
        with np.errstate(invalid="ignore", divide="ignore"):
            arr[:, :, k] = ((W @ np.where(M, X, 0.0))/(W @ M.astype(np.float64))).T
    for p in parts: p.close()
    return t.normalize(), arr
def merge_zip(z):
    """Turn one downloaded file into <stem>.nc.

    CDS may return a zip of one file per variable or a single NetCDF file, whatever name was
    requested. A zip is merged into one file; a NetCDF file with another suffix is renamed.
    """
    out = z.with_suffix(".nc")
    if out.is_file() and out != z: return
    if not zipfile.is_zipfile(z):
        if z.suffix != ".nc": z.rename(out); print("renamed (was NetCDF)", out, flush=True)
        return
    with tempfile.TemporaryDirectory(dir=D) as td:
        zipfile.ZipFile(z).extractall(td); fs = sorted(glob.glob(f"{td}/**/*.nc", recursive=True))
        tmp = out.with_suffix(".tmp.nc"); xr.open_mfdataset(fs, combine="by_coords").load().to_netcdf(tmp)
    if z == out: z.unlink()
    tmp.rename(out); print("merged", out, flush=True)
def normalise_all(root):
    """Turn every downloaded file below root into a NetCDF file named YYYY_MM.nc (see merge_zip)."""
    for f in sorted(root.glob("**/accumulated/*.nc")): merge_zip(f)          # an "unarchived" request can still come back zipped
    for f in sorted(root.glob("**/average_*/*.zip")): merge_zip(f)
if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "merge":
        normalise_all(RAW)
    elif cmd == "validate":
        normalise_all(RAW/"val")
        res = {}
        ref_a = xr.open_dataset(NAS/"accumulated/2025_12.nc"); ref_v = xr.open_dataset(NAS/"average/2025_12.nc")
        for name, f, ref in [("accumulated", RAW/"val/accumulated/2025_12.nc", ref_a)] + \
                            [(p.parent.name, p, ref_v) for p in sorted((RAW/"val").glob("average_*/2025_12.nc"))]:
            a, r = box(xr.open_dataset(f)), box(ref)
            res[name] = {v: float(np.nanmax(np.abs(a[v].values - r[v].values))/max(np.nanstd(r[v].values), 1e-12)) for v in a.data_vars if v in r}
        json.dump(res, open(D/"validate.json", "w"), indent=1); print(json.dumps(res, indent=1))
        ok = [k for k, d in res.items() if k.startswith("average") and max(d.values()) < 1e-6]
        print("accumulated reproduces archive:", max(res["accumulated"].values()) < 1e-6, "| average frequency reproducing archive:", ok or "NONE")
    elif cmd == "check":
        t, A = month_means("nas", 2025, 12, None)
        SB = pd.read_csv(EXT/"subbasins.csv", dtype=str); pos = {s: i for i, s in enumerate(SB.subbasin)}; worst, n = 0.0, 0
        for b in sorted(p.name for p in ZS.iterdir() if (p/"zarr.json").is_file()):
            z = xr.open_zarr(ZS/b, consolidated=False); subs = [str(s) for s in z.subbasin.values]; idx = np.array([pos[s] for s in subs])
            for k, v in enumerate(VARS):
                P = z[f"{v}_mean"].sel(date=slice("2025-12-01", "2025-12-31")).transpose("date", "subbasin").values.astype(np.float64)
                Q = A[:, idx, k]; sd = np.nanstd(P) if np.nanstd(P) > 0 else 1.0
                d = np.nanmax(np.abs(P - Q))/sd; worst = max(worst, d); n += np.isfinite(P).sum()
                if d > 1e-3: print(f"  {b} {v}: max |diff| / sd = {d:.2e}", flush=True)
        print(f"check vs production zarr, 2025-12, all groups: {n:,} values, worst max|diff|/sd = {worst:.2e}", flush=True)
    elif cmd == "extract":
        y = int(sys.argv[2]); ms = range(1, 13) if len(sys.argv) < 4 or sys.argv[3].startswith("--") else [int(m) for m in sys.argv[3].split(",")]
        src = sys.argv[sys.argv.index("--src") + 1] if "--src" in sys.argv else "raw"; freq = sys.argv[sys.argv.index("--freq") + 1] if "--freq" in sys.argv else "6_hourly"
        ts, As = [], []
        for m in ms:
            if not files(src, y, m, freq)[0].is_file(): print(f"{y}-{m:02d}: not downloaded, stop", flush=True); break
            t, A = month_means(src, y, m, freq); ts.append(t); As.append(A); print(f"{y}-{m:02d}: {len(t)} days, NaN share {np.isnan(A).mean():.4f}", flush=True)
        (D/"years").mkdir(exist_ok=True); tag = f"{y}_{src}"
        np.save(D/"years"/f"{tag}.npy", np.concatenate(As)); np.save(D/"years"/f"{tag}_dates.npy", np.concatenate([t.values for t in ts]))
        print("saved", tag, sum(len(t) for t in ts), "days", flush=True)
