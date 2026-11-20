r"""Join the per-basin delineation results into one sub-basin table.

Reads
  --delineation-dirs  one or more delineation output folders. Each holds a
                      subbasins/ folder (or is one) with the delineator's
                      files <outlet_id>_subbasins.parquet, one per basin. Each
                      file has the node id as column 'id' and no basin column.

Writes
  --out               one GeoParquet table of all basins with 'id' as the
                      index and a new column outlet_id naming the basin, which
                      is the layout the model code expects
                      (default $SWOTGAT_WORK/CONUS/metadata/subbasins_conus.parquet)

Method
  Node ids are MERIT-Basins catchment ids (COMID, unique within a two-digit
  Pfafstetter region), lake ids of the SWOT Prior Lake Database, or gauge ids
  (globally unique), so an id should belong to one basin only. The script checks
  this, because a shared id would merge two basins in the graph. The realistic
  case is a lake near a divide that is inserted into both neighbouring basins.
  Such a duplicate is resolved by keeping the copy with the larger upstream
  area and dropping the other, but only if the dropped copy has no node
  draining into it. Otherwise the script stops with an error. It also stops if
  the same outlet_id is found twice or if the basins' coordinate systems differ.

Usage
-----
  python build_conus_subbasins.py --delineation-dirs <dir> [<dir> ...] \
      --out <table.parquet> [--dry-run]
"""
from swotgat.paths import WORK

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

OUT = Path(f"{WORK}/CONUS/metadata/subbasins_conus.parquet")


def parse_args():
    """Parse the command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--delineation-dirs", nargs="+", type=Path, required=True,
                   help="Delineation run dirs (each containing subbasins/) or subbasins/ dirs themselves.")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def find_subbasin_files(dirs):
    """Collect the files <outlet_id>_subbasins.parquet under dirs.

    Each entry of dirs is a subbasins/ folder or a folder that holds one. Return a dict that maps outlet_id
    to the file path. Raise an error if a folder or its files are missing or an outlet_id is found twice.
    """
    found = {}
    for d in dirs:
        sub_dir = d if d.name == "subbasins" else d / "subbasins"
        if not sub_dir.is_dir():
            raise FileNotFoundError(f"no subbasins/ directory under {d}")
        files = sorted(sub_dir.glob("*_subbasins.parquet"))
        if not files:
            raise FileNotFoundError(f"no *_subbasins.parquet in {sub_dir}")
        for f in files:
            outlet_id = f.name.replace("_subbasins.parquet", "")
            if outlet_id in found:
                raise ValueError(f"outlet_id {outlet_id} found twice: {found[outlet_id]} and {f}")
            found[outlet_id] = f
    return found


def main():
    """Read the per-basin tables, resolve duplicate node ids, and write the joined table."""
    args = parse_args()
    files = find_subbasin_files(args.delineation_dirs)
    print(f"found {len(files)} basin(s)")

    frames, crs = [], None
    for outlet_id, path in sorted(files.items()):
        gdf = gpd.read_parquet(path)
        if crs is None:
            crs = gdf.crs
        elif gdf.crs != crs:
            raise ValueError(f"CRS mismatch: {outlet_id} is {gdf.crs}, expected {crs}")

        gdf["id"] = gdf["id"].astype(str)
        gdf = gdf.set_index("id")
        gdf.index.name = "id"
        gdf["outlet_id"] = outlet_id

        n_res = int(gdf["is_reservoir"].sum()) if "is_reservoir" in gdf else 0
        n_gauge = int(gdf["is_gauge"].sum()) if "is_gauge" in gdf else 0
        print(f"  {outlet_id:<18} {len(gdf):>6} subbasins  {n_gauge:>4} gauges  {n_res:>4} reservoirs")
        frames.append(gdf)

    # Ids must be unique: a collision would fuse two basins into one component in
    # the graph step. Reservoirs are the realistic source of collisions, because a
    # reservoir polygon near a divide can clip catchments in a neighbouring basin's
    # traced network, so the same lake_id is inserted into both -- meaningfully in
    # one basin, as a sliver in the other.
    #
    # Resolve those by keeping the claim with the larger upstream area (where the
    # node is hydrologically real) and dropping the sliver, but only when the
    # sliver is a leaf: if anything drains into it, dropping would leave dangling
    # edges, so fail loudly instead of silently corrupting the topology.
    owners = {}
    for gdf in frames:
        for sid in gdf.index:
            owners.setdefault(sid, []).append(gdf)

    drop = {}  # outlet_id -> set of ids to drop
    for sid, claims in owners.items():
        if len(claims) == 1:
            continue
        keep = max(claims, key=lambda g: float(g.loc[sid, "uparea_km2"]))
        for g in claims:
            if g is keep:
                continue
            oid = g["outlet_id"].iloc[0]
            has_upstream = (g["nextdown"].astype(str) == str(sid)).any()
            if has_upstream:
                raise ValueError(
                    f"subbasin id {sid!r} is claimed by "
                    f"{[c['outlet_id'].iloc[0] for c in claims]} and the copy in "
                    f"{oid} has upstream nodes, so it cannot be dropped safely. "
                    "Resolve this basin pair manually."
                )
            drop.setdefault(oid, set()).add(sid)
            print(f"  duplicate {sid}: keeping {keep['outlet_id'].iloc[0]} "
                  f"(uparea {float(keep.loc[sid, 'uparea_km2']):.1f} km2), dropping leaf copy "
                  f"in {oid} (uparea {float(g.loc[sid, 'uparea_km2']):.1f} km2)")

    if drop:
        frames = [
            g.drop(index=list(drop.get(g["outlet_id"].iloc[0], set())), errors="ignore")
            for g in frames
        ]
        print(f"  resolved {sum(len(v) for v in drop.values())} duplicate id(s)")

    combined = pd.concat(frames)
    assert not combined.index.duplicated().any(), "duplicate ids remain after resolution"
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=crs)

    print()
    print(f"UNIONED {len(combined):,} subbasins across {combined['outlet_id'].nunique()} basins")
    print(f"  gauges: {int(combined['is_gauge'].sum()):,}  reservoirs: {int(combined['is_reservoir'].sum()):,}")
    print(f"  terminal nodes (nextdown NaN): {int(combined['nextdown'].isna().sum())}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
