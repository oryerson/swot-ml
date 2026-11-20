"""Build gauge -> SWORD reach overrides from the SWOT Validation Set (SVS).

SVS (Saemian & Tourian 2026, DOI 10.18419/DARUS-5843, CC BY 4.0) matches in-situ
gauges to reaches of the SWORD river database. It does so independently of the
crosswalk chain of merge_swot_river_conus.py (MERIT-Basins COMID -> v0.7 COMID ->
MHS table -> reach), so it is a published reference to prefer where it has a match.
merge_swot_river_conus.py imports load_svs and build_overrides from this module.
Run as a script, the module writes the overrides to a table for inspection.

Usage: python svs_reach_overrides.py [--subbasins FILE] [--svs FILE]
                                     [--sword-version v16|v17b|v17c] [--out FILE]
  --subbasins      sub-basin table (default $SWOTGAT_WORK/CONUS/metadata/subbasins_conus.parquet)
  --svs            SVS NetCDF file (default $SWOTGAT_WORK/SVS/SVS_v1_1_2.nc)
  --sword-version  SWORD version of the reach ids to read from SVS (default v17c)
  --out            output table with columns subbasin, reach_id
                   (default $SWOTGAT_WORK/CONUS/metadata/svs_reach_overrides.csv)

Scope (counts on the first graph version). The override is narrow by design:
  * SVS covers gauges only. 18,522 of 23,458 sub-basins are not gauges, and SVS
    has nothing to say about them.
  * Only 1,181 of 4,936 gauge sub-basins (23.9%) appear in SVS at all.
  * The crosswalk chain already agrees with SVS on 2,194 of 2,204 links (99.55%).
    The 10 differences are adjacent reaches on the same river, not wrong rivers.
So the override changes about 0.45% of gauge links.

Two details that matter for correctness:
  * Join on (station_id, agency), not on station_id alone. Bare ids collide across
    agencies: USGS-12370000 and BRANA-12370000 both reduce to "12370000" and would
    cross-match, silently pairing a US gauge with a Brazilian reach.
  * Use reach_id_v17c. The MHS table is aligned with SWORD v17b/v17c (99.9% of ids
    overlap). v16 overlaps only 65.2%, so mixing versions would create
    disagreements that are not real.
"""
from swotgat.paths import WORK
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

CONUS = Path(f"{WORK}/CONUS")
SVS_NC = Path(f"{WORK}/SVS/SVS_v1_1_2.nc")

# agency prefix of the gauge ids used in this project -> SVS `source` agency code
AGENCY = {"USGS": "USGS", "ECCC": "HYDAT", "BRANA": "ANA", "UKEA": "EFRA", "ABOM": "BOM"}


def _decode(ds, var):
    return ["".join(c.decode() for c in col if c != b" ").strip() for col in ds[var].values.T]


def load_svs(path: Path = SVS_NC, sword_version: str = "v17c") -> pd.DataFrame:
    """Read the SVS gauge -> SWORD reach matches as a table (station_id, source, svs_reach, key).

    Stations without a reach are left out. `key` is station_id without leading zeros.
    """
    ds = xr.open_dataset(path)
    field = f"reach_id_{sword_version}"
    if field not in ds.variables:
        raise KeyError(f"{field} not in {path.name}; available: "
                       f"{[v for v in ds.variables if v.startswith('reach_id')]}")
    svs = pd.DataFrame({
        "station_id": _decode(ds, "station_id"),
        "source": _decode(ds, "source"),
        "svs_reach": ds[field].values.astype("int64"),
    })
    svs = svs[svs.svs_reach > 0].copy()
    svs["key"] = svs.station_id.str.lstrip("0")
    return svs


def build_overrides(gauge_ids, svs: pd.DataFrame | None = None) -> dict:
    """Return {gauge id -> SVS reach_id} for the gauges that SVS has a match for.

    gauge_ids use the form '<agency>-<station>', e.g. 'USGS-14246900' or 'ECCC-08NB016'.
    Ids that are not of this form, or whose agency is not in AGENCY, are ignored.
    The join is on (station number without leading zeros, agency).
    """
    if svs is None:
        svs = load_svs()

    g = pd.DataFrame({"id": list(gauge_ids)})
    g["agency"] = g.id.str.split("-").str[0]
    g["svs_source"] = g.agency.map(AGENCY)
    g["key"] = g.id.str.split("-", n=1).str[1].str.lstrip("0")
    g = g.dropna(subset="svs_source")

    j = g.merge(svs, left_on=["key", "svs_source"], right_on=["key", "source"], how="inner")
    return dict(zip(j.id, j.svs_reach))


def main():
    """Write the SVS overrides of the gauge sub-basins to a table and print how many there are."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subbasins", type=Path, default=CONUS / "metadata" / "subbasins_conus.parquet")
    p.add_argument("--svs", type=Path, default=SVS_NC)
    p.add_argument("--sword-version", default="v17c", choices=["v16", "v17b", "v17c"])
    p.add_argument("--out", type=Path, default=CONUS / "metadata" / "svs_reach_overrides.csv")
    args = p.parse_args()

    import geopandas as gpd
    subs = gpd.read_parquet(args.subbasins)
    subs.index = subs.index.astype(str)
    gauges = subs[subs.is_gauge].index.tolist()

    svs = load_svs(args.svs, args.sword_version)
    ov = build_overrides(gauges, svs)

    print(f"gauge subbasins            : {len(gauges):,}")
    print(f"SVS stations with a reach  : {len(svs):,}")
    print(f"overrides available        : {len(ov):,} "
          f"({100*len(ov)/max(len(gauges),1):.1f}% of gauge subbasins, "
          f"{100*len(ov)/len(subs):.1f}% of all subbasins)")

    out = pd.DataFrame({"subbasin": list(ov), "reach_id": list(ov.values())})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
