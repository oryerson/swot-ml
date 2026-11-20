"""Download the daily-mean ERA5-Land variables (d2m, t2m, sp, fal) as 6-hourly fields instead.

This is a faster route for months that wait too long in the queue of the CDS dataset
`derived-era5-land-daily-statistics`, which download_2026.py uses (in the paper's run that queue
held March, June and September 2026 for 2 to 10 hours). The script fetches the same four
variables from `reanalysis-era5-land`, whose queue is fast, at 00, 06, 12 and 18 UTC.
prep_inst.py then averages them per UTC day. That average equals the derived product's
`daily_mean` at frequency 6_hourly and time zone utc+00:00; `prep_inst.py check` tests this on
December 2025 against the archive before any month is used.

Writes
    $SWOTGAT_WORK/CONUS_storage_fork/eval/overnight/era5_2026/raw/inst/YYYY_MM.nc

Usage
    python download_inst.py 2025-12 [2026-03 ...]        (one or more months as YYYY-MM)

Notes
    Needs the `cdsapi` package and a personal CDS key in ~/.cdsapirc. Same request box as
    download_2026.py. A file that already exists and is not empty is skipped. A failed request
    is retried up to 5 times; if all 5 fail the script moves on to the next month without an
    error exit, so check the log.
"""
from swotgat.paths import WORK
import sys, time
from pathlib import Path
import cdsapi
OUT = Path(f"{WORK}/CONUS_storage_fork/eval/overnight/era5_2026/raw/inst"); OUT.mkdir(parents=True, exist_ok=True)
AREA = [53.0, -124.2, 26.0, -67.6]
c = cdsapi.Client()
for ym in sys.argv[1:]:
    y, m = ym.split("-"); t = OUT/f"{y}_{m}.nc"
    if t.is_file() and t.stat().st_size > 0: print("have", t, flush=True); continue
    for k in range(5):
        try:
            c.retrieve("reanalysis-era5-land", dict(variable=["2m_dewpoint_temperature", "2m_temperature", "surface_pressure", "forecast_albedo"], year=y, month=m,
                       day=[f"{d:02d}" for d in range(1, 32)], time=["00:00", "06:00", "12:00", "18:00"], data_format="netcdf", download_format="unarchived", area=AREA), str(t))
            print("got", t, flush=True); break
        except Exception as e:
            print(f"attempt {k+1} failed: {e}", flush=True); time.sleep(30*2**k)
