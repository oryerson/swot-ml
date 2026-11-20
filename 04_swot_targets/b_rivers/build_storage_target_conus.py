"""Write the river storage-change target (dv_river) into the data store and save each node's reach area.

    dv_river = d_wse_river [m] x reach_area [m2] / 1e6      (million cubic meters, MCM)

d_wse_river is the SWOT water-level anomaly of the node's river reach (merge_swot_river_conus.py).

Usage: python build_storage_target_conus.py [--store DIR] [--out-areas FILE] [--dry-run]
  --store      data store (default $SWOTGAT_WORK/CONUS/zarr_store)
  --out-areas  reach-area table to write (default $SWOTGAT_WORK/CONUS/metadata/reach_area_conus.csv)
  --dry-run    compute and report, write nothing

Writes
  variable dv_river in every basin of the store
  the reach-area table: subbasin, reach_area_m2, n_area_obs, cv_area (standard deviation / mean of the area)

Method
  reach_area is one number per river node: the median of the node's finite area_total_river observations
  (SWOT total water area of the reach). The median is used rather than the mean because the area distribution
  is right-skewed and high flows pull the mean up (in the paper's run: mean 2.88e6 m2, median 1.62e6 m2).
  A node with fewer than 10 area observations gets no reach area and no target.
  The unit matches the lake target dv_SWOTAE, which is (level anomaly) x (lake area) in MCM. This was verified
  by fitting dv_SWOTAE = k x (wse - median wse): R2 0.9993, with k equal to the lake area in km2.

Notes
  dv_river is created in every basin, also in those without SWOT river data (43 in the paper's run). The model's
  data loader silently drops a variable that is missing from a basin (the `available_columns` intersection in
  _load_and_normalize_basin), which would leave such a basin with no target instead of an empty one.
  Run update_zarr_stats_conus.py (stage 03) afterwards: dv_river needs normalization statistics, and training
  stops without them.
"""
from swotgat.paths import REPO, WORK
import argparse, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, zarr, xarray as xr

sys.path.insert(0, f"{REPO}/vendor/swot_ml/src")
from data import ZarrBasinStore
warnings.filterwarnings("ignore")

CONUS = Path(f"{WORK}/CONUS")
MIN_AREA_OBS = 10        # nodes with fewer area observations get no target
BLOCK = 400              # contiguous subbasins per region write


def parse_args():
    """Return the parsed command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", type=Path, default=CONUS / "zarr_store")
    p.add_argument("--out-areas", type=Path,
                   default=CONUS / "metadata" / "reach_area_conus.csv")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    """Compute each node's reach area, write dv_river into the store and save the reach-area table."""
    args = parse_args()
    store = ZarrBasinStore(args.store)
    basins = sorted(p.name for p in args.store.iterdir()
                    if (p / "zarr.json").is_file())
    print(f"{len(basins)} basins in {args.store}")

    areas, n_written, n_init = [], 0, 0
    for bi, b in enumerate(basins):
        bp = args.store / b
        g = zarr.open_group(str(bp), mode="r")
        subs = [str(s) for s in g["subbasin"][:]]
        # The date coordinate is "days since 1980-01-01", not a Unix epoch. Decoding it
        # as Unix time shifts every value by 3,653 days (10 years) into 2013-2015,
        # outside the training window, and leaves the target empty in the window that
        # matters. Let xarray decode it from the stored units attribute.
        with xr.open_zarr(bp, consolidated=True) as _ds:
            dates = pd.to_datetime(_ds["date"].values)

        if not (bp / "area_total_river").is_dir():
            # no SWOT river data here: create the variable empty so every basin
            # has the same schema
            if not args.dry_run:
                store.init_missing_vars(bp, subs, pd.DataFrame(columns=["dv_river"]))
            n_init += 1
            continue

        nT, nS = g["area_total_river"].shape
        for i in range(0, nS, BLOCK):
            sl = slice(i, min(i + BLOCK, nS))
            blk = subs[sl]
            A = g["area_total_river"][:, sl]
            S = g["d_wse_river"][:, sl]
            frames = []
            for j, nd in enumerate(blk):
                a, s = A[:, j], S[:, j]
                fa = np.isfinite(a)
                if fa.sum() < MIN_AREA_OBS:
                    continue
                area_m2 = float(np.median(a[fa]))
                areas.append({"subbasin": nd, "reach_area_m2": round(area_m2, 1),
                              "n_area_obs": int(fa.sum()),
                              "cv_area": round(float(a[fa].std() / a[fa].mean()), 4)})
                fs = np.isfinite(s)
                if not fs.any():
                    continue
                frames.append(pd.DataFrame(
                    {"dv_river": s[fs] * area_m2 / 1e6, "subbasin": nd},
                    index=pd.Index(dates[fs], name="date")))
            if args.dry_run:
                n_written += len(frames)
                continue
            if frames:
                store.write_batch_data(b, pd.concat(frames), subs, blk)
                n_written += len(frames)
            else:
                store.init_missing_vars(bp, subs, pd.DataFrame(columns=["dv_river"]))
        if bi % 10 == 0:
            print(f"  {bi}/{len(basins)}  reaches written so far {n_written:,}", flush=True)

    df = pd.DataFrame(areas)
    if not args.dry_run:
        df.to_csv(args.out_areas, index=False)
    print(f"\nreaches with a reach area (>= {MIN_AREA_OBS} obs): {len(df):,}")
    print(f"reaches given a dv_river series                  : {n_written:,}")
    print(f"basins initialised empty (no SWOT river data)    : {n_init}")
    if len(df):
        print("\nreach_area_m2:")
        print(df.reach_area_m2.describe(percentiles=[.25,.5,.75]).round(0).to_string())
        print(f"wrote {args.out_areas}")
    print("\nNEXT: python preprocess/update_zarr_stats_conus.py  (dv_river needs stats)")


if __name__ == "__main__":
    main()
