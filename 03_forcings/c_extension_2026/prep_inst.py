"""Average the 6-hourly ERA5-Land fields of download_inst.py into daily means.

Reads
    <era5_2026>/raw/inst/YYYY_MM.nc     6-hourly d2m, t2m, sp, fal (download_inst.py)
    $SWOTGAT_NAS_ICE/data/ERA5-Land/average/2025_12.nc     the archive, for `check` only
    where <era5_2026> = $SWOTGAT_WORK/CONUS_storage_fork/eval/overnight/era5_2026

Writes
    <era5_2026>/raw/average_6_hourly/YYYY_MM.nc    in the layout of the CDS daily-statistics
        product (variables d2m, t2m, sp, fal; daily `valid_time`), so that prep_2026.py reads it
        like a file from download_2026.py

Usage
    python prep_inst.py check                        compare December 2025 with the archive
    python prep_inst.py make 2026-03 [2026-06 ...]   write the daily files of these months

Method
    The daily mean is the plain mean of the four samples of each UTC day (00, 06, 12, 18). A
    NaN in any of the four samples gives NaN (skipna=False), and the script stops if any day
    does not have exactly four samples.
    `check` cuts both the result and the archive to the grid box of the sub-basin weights and
    prints, per variable, the largest absolute difference (also divided by the standard
    deviation of the archive field) and whether the NaN patterns match.
    `make` does not overwrite an existing output file.

Notes
    The script imports prep_2026.py as a module from <era5_2026> (the folder where the scripts
    of this step were deployed), for its paths and its grid-box and time-name helpers. Importing
    it loads the weights file era5ext/weights.npz. The command-line arguments are hidden from
    that import.
"""
from swotgat.paths import WORK
import sys, zipfile, numpy as np, pandas as pd, xarray as xr
sys.argv_cmd = sys.argv[1:]; sys.argv = ["x"]
sys.path.insert(0, f"{WORK}/CONUS_storage_fork/eval/overnight/era5_2026")
import prep_2026 as P
def daily(f):
    """Return the daily means of d2m, t2m, sp and fal from one file f of 6-hourly fields.

    A NaN in any sample of a day gives NaN. Stops with an assertion error unless every day has
    exactly four samples. The time coordinate of the result is named valid_time.
    """
    ds = xr.open_dataset(f); tn = P.tname(ds)
    t = pd.to_datetime(ds[tn].values); ds = ds.assign_coords({tn: t})
    out = ds[["d2m", "t2m", "sp", "fal"]].resample({tn: "1D"}).mean(skipna=False)
    n = ds[tn].to_series().groupby(t.normalize()).size()
    assert (n == 4).all(), f"not 4 samples per day: {n.value_counts().to_dict()}"
    return out.rename({tn: "valid_time"}) if tn != "valid_time" else out
cmd = sys.argv_cmd
if cmd[0] == "check":
    a = P.box(daily(P.RAW/"inst/2025_12.nc")); r = P.box(xr.open_dataset(P.NAS/"average/2025_12.nc"))
    for v in ("d2m", "t2m", "sp", "fal"):
        x, y = a[v].values, r[v].values
        print(v, "max|diff|/sd", float(np.nanmax(np.abs(x - y))/np.nanstd(y)), "| max|diff|", float(np.nanmax(np.abs(x - y))), "| NaN match", bool((np.isnan(x) == np.isnan(y)).all()), flush=True)
else:
    for ym in cmd[1:]:
        y, m = ym.split("-"); o = P.RAW/f"average_6_hourly/{y}_{m}.nc"
        if o.is_file(): print("exists", o); continue
        d = daily(P.RAW/f"inst/{y}_{m}.nc"); d.to_netcdf(o); print("wrote", o, dict(d.sizes), flush=True)
