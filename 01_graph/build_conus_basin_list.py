"""Select the river basins of the conterminous US (CONUS) to build, and write one gauge table per basin.

A basin is defined by its outlet gauge. The global gauge table (all_matchups.csv,
shipped with the graph delineator) already assigns every gauge to a basin: it
has 12,114 gauges in 1,472 basins, and the column outlet_id names the gauge at
the basin's outlet. This script does not re-derive that assignment. It selects
the CONUS basins, applies a size filter, and writes the gauge tables that the
delineator's delineate_basins() expects.

Reads
  --matchups   global gauge table with columns id, lng, lat, outlet_id
               (default $SWOTGAT_WORK/graph_delineator/all_matchups.csv)

Writes (under --out-dir, default $SWOTGAT_WORK/CONUS/basin_lists)
  basin_manifest.csv        one row per selected basin (outlet_id, n_gauges,
                            agencies, bounding box of its gauges, outlet
                            position), sorted by number of gauges. The row
                            number is the Slurm array index of the basin.
  gauges/<outlet_id>.csv    gauge table of one basin (one delineation task each)
  conus_gauges.csv          all selected basins in one table; delineate_basins()
                            groups by outlet_id itself, so this file also works
                            for a single serial run.

Method
  1. Locate each basin's outlet: the position of the gauge named by outlet_id,
     or the mean position of the basin's gauges if that gauge has no row.
  2. Keep basins whose outlet lies inside --bbox (default: CONUS, without
     Alaska and Hawaii) and whose outlet gauge belongs to one of
     --outlet-agencies (default: USGS). The test is on the outlet only, so a
     basin keeps its gauges in Canada or Mexico and its gauges of other agencies.
  3. Keep basins with at least --min-gauges gauges (default 5).
  4. With --max-basins N, keep only the N basins with the most gauges.

In the paper's run (all defaults): 102 basins.

Usage
-----
  # see what would be selected, write nothing
  python build_conus_basin_list.py --dry-run

  # top 50 basins by gauge count
  python build_conus_basin_list.py --max-basins 50

  # every basin with at least 5 gauges (the default)
  python build_conus_basin_list.py --min-gauges 5
"""
from swotgat.paths import WORK

import argparse
from pathlib import Path

import pandas as pd

MATCHUPS = Path(
    f"{WORK}/graph_delineator/all_matchups.csv"
)
OUT_DIR = Path(f"{WORK}/CONUS/basin_lists")

# Conterminous US. Excludes Alaska (HUC 19) and Hawaii, which the training data
# does not cover. Applied to the *outlet* location so that basins draining to
# the US keep their headwater gauges across the border (the Columbia, for
# example, keeps its 58 Canadian ECCC gauges).
CONUS_BBOX = (-125.0, 24.4, -66.9, 49.4)  # min_lon, min_lat, max_lon, max_lat


def parse_args():
    """Parse the command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matchups", type=Path, default=MATCHUPS)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument(
        "--min-gauges",
        type=int,
        default=5,
        help="Skip basins with fewer than this many gauges. Most of the 718 CONUS "
        "outlets are single-gauge coastal slivers; the default trims those.",
    )
    p.add_argument(
        "--max-basins",
        type=int,
        default=None,
        help="Keep only the N largest basins by gauge count (applied after --min-gauges).",
    )
    p.add_argument(
        "--outlet-agencies",
        nargs="+",
        default=["USGS"],
        help="Agency prefixes allowed for the *outlet* gauge. Member gauges from "
        "any agency are always kept, so transboundary basins stay intact.",
    )
    p.add_argument("--bbox", nargs=4, type=float, default=CONUS_BBOX,
                   metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    p.add_argument("--dry-run", action="store_true", help="Report the selection, write nothing.")
    return p.parse_args()


def outlet_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Return the location of each basin's outlet.

    outlet_id names a gauge, which is usually present as its own row in df. Where it
    is not, fall back to the mean position of the basin's gauges so the basin
    still gets a fair shot at the bbox test rather than being dropped silently.

    Return a table indexed by outlet_id with columns lng, lat and outlet_resolved
    (True where the outlet gauge had its own row).
    """
    by_id = df.drop_duplicates("id").set_index("id")[["lng", "lat"]]
    outlets = df["outlet_id"].drop_duplicates()

    located = outlets.map(lambda o: o in by_id.index)
    coords = pd.DataFrame(index=outlets, columns=["lng", "lat"], dtype=float)
    coords.loc[located.values] = by_id.reindex(outlets[located.values]).values

    missing = outlets[~located.values]
    if len(missing):
        fallback = df[df["outlet_id"].isin(missing)].groupby("outlet_id")[["lng", "lat"]].mean()
        coords.loc[fallback.index] = fallback.values

    coords["outlet_resolved"] = located.values
    return coords


def main():
    """Select the basins, print the selection, and write the manifest and the gauge tables."""
    args = parse_args()
    min_lon, min_lat, max_lon, max_lat = args.bbox

    df = pd.read_csv(args.matchups)
    print(f"matchups: {len(df):,} gauges, {df['outlet_id'].nunique():,} outlets")

    coords = outlet_locations(df)
    n_fallback = int((~coords["outlet_resolved"]).sum())
    if n_fallback:
        print(f"  note: {n_fallback} outlet(s) not present as a gauge row; used member-gauge centroid")

    in_bbox = (
        coords["lng"].between(min_lon, max_lon) & coords["lat"].between(min_lat, max_lat)
    )
    agency_ok = coords.index.str.split("-").str[0].isin(args.outlet_agencies)
    keep = coords.index[in_bbox.values & agency_ok]
    print(f"CONUS outlets (bbox + outlet agency {args.outlet_agencies}): {len(keep):,}")

    conus = df[df["outlet_id"].isin(keep)].copy()
    counts = conus["outlet_id"].value_counts()

    counts = counts[counts >= args.min_gauges]
    print(f"after --min-gauges {args.min_gauges}: {len(counts):,} basins")
    if args.max_basins:
        counts = counts.head(args.max_basins)
        print(f"after --max-basins {args.max_basins}: {len(counts):,} basins")

    selected = conus[conus["outlet_id"].isin(counts.index)].copy()

    manifest = (
        selected.groupby("outlet_id")
        .agg(
            n_gauges=("id", "size"),
            agencies=("id", lambda s: "|".join(sorted(set(s.str.split("-").str[0])))),
            min_lon=("lng", "min"), max_lon=("lng", "max"),
            min_lat=("lat", "min"), max_lat=("lat", "max"),
        )
        .join(coords[["lng", "lat"]].rename(columns={"lng": "outlet_lng", "lat": "outlet_lat"}))
        .sort_values("n_gauges", ascending=False)
        .reset_index()
    )

    print()
    print(f"SELECTED {len(manifest):,} basins covering {len(selected):,} gauges")
    print(manifest.head(15).to_string(index=False))
    if len(manifest) > 15:
        print(f"  ... and {len(manifest) - 15} more")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    gauge_dir = args.out_dir / "gauges"
    gauge_dir.mkdir(parents=True, exist_ok=True)

    for outlet_id, grp in selected.groupby("outlet_id"):
        grp.to_csv(gauge_dir / f"{outlet_id}.csv", index=False)

    manifest.to_csv(args.out_dir / "basin_manifest.csv", index=False)
    selected.to_csv(args.out_dir / "conus_gauges.csv", index=False)

    print(f"\nwrote {args.out_dir / 'basin_manifest.csv'}")
    print(f"wrote {args.out_dir / 'conus_gauges.csv'}")
    print(f"wrote {len(manifest):,} per-basin files to {gauge_dir}/")
    print(f"\nSLURM array range for per-basin delineation: 0-{len(manifest) - 1}")


if __name__ == "__main__":
    main()
