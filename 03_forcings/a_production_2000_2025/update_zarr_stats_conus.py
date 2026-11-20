"""Compute the normalization statistics of every variable and store them in the Zarr data store.

Reads and writes
    --store   the Zarr store (default $SWOTGAT_WORK/CONUS/zarr_store). The statistics are written
              into each basin group by ZarrBasinStore.compute_and_store_stats of the vendored
              swot_ml package.

Usage
    python update_zarr_stats_conus.py [--store DIR]

Notes
    Run this after all merge steps (forcing and targets) for every basin in the store.
    Statistics are stored per basin. At training time the data loader aggregates them over the
    basins of the training split, so a basin merged later needs a re-run. Existing statistics
    are overwritten.
    Follow with fix_empty_stats.py, which adds the entries this step leaves out for variables
    that are entirely NaN in a basin.
    Every later store of this stage (1980-1999 extension, 2026 extension, graph v3.5 stores)
    copies these statistics unchanged, so this is the only place where they are computed.
"""
from swotgat.paths import REPO, WORK
import argparse
import sys
from pathlib import Path

sys.path.insert(0, f"{REPO}/vendor/swot_ml/src")

from data import ZarrBasinStore

CONUS = Path(f"{WORK}/CONUS")

# compute_and_store_stats starts a Dask LocalCluster. Its spawned workers import this module
# again, so the code that creates the cluster must stay behind the __main__ guard.
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", type=Path, default=CONUS / "zarr_store")
    args = p.parse_args()

    store = ZarrBasinStore(args.store)
    print(f"Computing normalisation stats for {args.store} ...")
    # overwrite=True so that variables added after an earlier statistics run (for example the
    # SWOT targets) are recomputed rather than skipped.
    store.compute_and_store_stats(overwrite=True)
    print("Done.")
