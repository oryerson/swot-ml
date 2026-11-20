"""Download ERA5-Land for months after 2025 from the Copernicus Climate Data Store (CDS).

The ERA5-Land archive used for 1980-2025 ($SWOTGAT_NAS_ICE/data/ERA5-Land/{accumulated,average}/
YYYY_MM.nc, global) ends in December 2025. This script fetches later months in the same layout,
but only over the box that holds the sub-basin weights of the conterminous US, so a month is
tens of MB instead of about 1.2 GB.

Writes (raw files only; prep_2026.py merges, validates and extracts them)
    $SWOTGAT_WORK/CONUS_storage_fork/eval/overnight/era5_2026/raw/
        accumulated/YYYY_MM.nc          CDS dataset `reanalysis-era5-land`, variables ssrd, strd,
                                        ro, sro, e, tp at 00:00 UTC of every day (the value at
                                        00:00 is the previous day's total, as in the archive)
        average_<freq>/YYYY_MM.zip      CDS dataset `derived-era5-land-daily-statistics`,
                                        `daily_mean` of d2m, t2m, sp, fal in UTC (usually a zip
                                        of one file per variable)
        val/...                         the same two files for 2025-12, with --validate

Usage
    python download_2026.py --validate
    python download_2026.py --months 2026-01:2026-09 [--freq 6_hourly]
    --validate   downloads December 2025 with both sampling frequencies (6_hourly and 1_hourly)
                 into raw/val/. `prep_2026.py validate` then compares them with the archive and
                 shows which frequency reproduces it.
    --months     first and last month to download, as YYYY-MM:YYYY-MM
    --freq       sampling frequency of the daily mean (default 6_hourly)

Notes
    Needs the `cdsapi` package, a personal CDS key in ~/.cdsapirc, and the ERA5-Land licences
    accepted on the CDS website for that account.
    The request box is [53.0 N, 124.2 W, 26.0 N, 67.6 W]: the box of the weights
    (26.1..52.9 N, 124.1..67.7 W) plus one grid cell on each side.
    A file that already exists and is not empty is not downloaded again. A failed request is
    retried up to 5 times, waiting 30, 60, 120, 240 and 480 s.
    CDS may return different data today than on the download date of the paper's run, so the
    downloaded files of the paper are archived (see the README).
"""
from swotgat.paths import WORK
import argparse, time
from pathlib import Path
import cdsapi
OUT = Path(f"{WORK}/CONUS_storage_fork/eval/overnight/era5_2026/raw")
AREA = [53.0, -124.2, 26.0, -67.6]                  # N W S E: box of the weights (26.1..52.9, -124.1..-67.7) plus one cell
ACC = ["surface_solar_radiation_downwards", "surface_thermal_radiation_downwards", "runoff", "surface_runoff", "total_evaporation", "total_precipitation"]
AVG = ["2m_dewpoint_temperature", "2m_temperature", "surface_pressure", "forecast_albedo"]
DAYS = [f"{d:02d}" for d in range(1, 32)]
def get(c, ds, req, target, tries=5):
    """Download one CDS request to the file target, unless a non-empty file is already there.

    c is the cdsapi client, ds the name of the CDS dataset and req the request. A failed request
    is retried up to `tries` times, waiting 30 s after the first failure and twice as long after
    each further one. If every attempt fails, a RuntimeError is raised.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0: print("have", target, flush=True); return
    for k in range(tries):
        try:
            c.retrieve(ds, req, str(target)); print("got", target, flush=True); return
        except Exception as e:
            print(f"attempt {k+1} failed for {target.name}: {e}", flush=True); time.sleep(30*2**k)
    raise RuntimeError(f"failed: {target}")
def month(c, y, m, freq, sub=""):
    """Download the two files of month m of year y: accumulated variables and daily means.

    freq is the sampling frequency of the daily mean; sub is an optional sub-folder of the output
    folder ("val" for the validation download).
    """
    get(c, "reanalysis-era5-land", dict(variable=ACC, year=str(y), month=f"{m:02d}", day=DAYS, time=["00:00"], data_format="netcdf",
        download_format="unarchived", area=AREA), OUT/sub/"accumulated"/f"{y}_{m:02d}.nc")
    get(c, "derived-era5-land-daily-statistics", dict(variable=AVG, year=str(y), month=f"{m:02d}", day=DAYS, daily_statistic="daily_mean",
        time_zone="utc+00:00", frequency=freq, area=AREA), OUT/sub/f"average_{freq}"/f"{y}_{m:02d}.zip")
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--validate", action="store_true"); ap.add_argument("--months", default="")
    ap.add_argument("--freq", default="6_hourly"); a = ap.parse_args(); c = cdsapi.Client()
    if a.validate:
        for f in ("6_hourly", "1_hourly"): month(c, 2025, 12, f, sub="val")
    if a.months:
        s, e = a.months.split(":"); y, m = map(int, s.split("-")); ye, me = map(int, e.split("-"))
        while (y, m) <= (ye, me):
            month(c, y, m, a.freq); y, m = (y + 1, 1) if m == 12 else (y, m + 1)
