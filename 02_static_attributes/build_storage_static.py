"""Add the water-surface area of each node (lake area or river reach area) to the static-attribute table.

Why: the storage-change target is a water-level anomaly times a water-surface
area (lake area at lakes, SWOT reach area at rivers). Without an area attribute
the model cannot know the scale of storage at a reservoir that has no training
target, and it must predict at such nodes. The only other size attribute is
dam_cap_mcm, a volume, which relates to area only through a depth that varies
by an order of magnitude. The existing column area_km2 is no substitute: it is
the catchment area of the node, not the lake. For lake nodes its median is
18.6 times the lake polygon, and only 0.1% are within 10% of it.

Reads
  $V35S_WORK/static_conus_gdw_<tag>.parquet   static table from build_conus_gdw_static.py; <tag> is $V35S_GDW_TAG
                                              if set, else $V35S_TAG
  $SWOTGAT_NAS_WATER/cjgleason/ted/LakeSPvC_ts_for_Ted/SWOT_PLD_v106_reservoirs.gdb
                                              SWOT Prior Lake Database (PLD): lake_id, area_geodesic_km2
  $V35S_REACH                                 reach-area table with columns subbasin, reach_area_m2 (stage 04b);
                                              default $SWOTGAT_WORK/CONUS/metadata/reach_area_conus.csv

Writes
  $V35S_WORK/static_conus_storage_$V35S_TAG.parquet   the static table plus three columns

Usage
  V35S_WORK=... V35S_TAG=v35r2 V35S_GDW_TAG=v35 V35S_REACH=... python build_storage_static.py

Columns added (km2; a node without the water body gets 0, which stays 0 after log1p normalization and reads as
"no open water here" instead of a missing value)
  lake_area_km2    PLD geodesic lake area, for nodes whose id is a PLD lake_id
  reach_area_km2   reach area from the reach-area table
  water_area_km2   the lake area if the node has one, otherwise the reach area
"""
from swotgat.paths import NAS_WATER, WORK
import sys, warnings
import os as _os  # file names come from the environment: V35S_WORK, V35S_TAG, V35S_GDW_TAG, V35S_REACH
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
warnings.filterwarnings("ignore")

CONUS = Path(f"{WORK}/CONUS")
PLD = Path(f"{NAS_WATER}/cjgleason/ted/LakeSPvC_ts_for_Ted/SWOT_PLD_v106_reservoirs.gdb")
SRC = Path(_os.environ["V35S_WORK"]) / f"static_conus_gdw_{_os.environ.get('V35S_GDW_TAG', _os.environ['V35S_TAG'])}.parquet"  # the input table's tag may differ from the output tag
DST = Path(_os.environ["V35S_WORK"]) / f"static_conus_storage_{_os.environ['V35S_TAG']}.parquet"

st = pd.read_parquet(SRC)
print(f"source static table: {st.shape[0]:,} nodes x {st.shape[1]} columns")
idx = st.index.to_frame(index=False)
idx["subbasin"] = idx.subbasin.astype(str)

pld = gpd.read_file(PLD, columns=["lake_id", "area_geodesic_km2", "ref_area"])
pld["lake_id"] = pd.to_numeric(pld.lake_id, errors="coerce")
pld = pld[pld.lake_id.notna()].copy()
pld["subbasin"] = pld.lake_id.astype("int64").astype(str)
lake = (pld.drop_duplicates("subbasin")
           .set_index("subbasin")["area_geodesic_km2"].astype(float))

reach = pd.read_csv(_os.environ.get("V35S_REACH", CONUS / "metadata" / "reach_area_conus.csv"), dtype={"subbasin": str})  # for graph v3.5, V35S_REACH names its reach-area table
reach_km2 = (reach.set_index("subbasin")["reach_area_m2"] / 1e6)

wa = idx.subbasin.map(lake).fillna(idx.subbasin.map(reach_km2))
st["water_area_km2"] = wa.values
st["lake_area_km2"] = idx.subbasin.map(lake).values
st["reach_area_km2"] = idx.subbasin.map(reach_km2).values

res = st.is_reservoir == 1
print(f"  lake_area_km2  filled: {int(st.lake_area_km2.notna().sum()):,} "
      f"({int((res & st.lake_area_km2.notna()).sum()):,} of {int(res.sum()):,} reservoir nodes)")
print(f"  reach_area_km2 filled: {int(st.reach_area_km2.notna().sum()):,}")
print(f"  water_area_km2 filled: {int(st.water_area_km2.notna().sum()):,} "
      f"({100*st.water_area_km2.notna().mean():.1f}% of all nodes)")

# Nodes with no water body get 0, which stays 0 after log1p normalization and reads as
# "no open water here" rather than as a missing value.
st["water_area_km2"] = st.water_area_km2.fillna(0.0)
st["lake_area_km2"] = st.lake_area_km2.fillna(0.0)
st["reach_area_km2"] = st.reach_area_km2.fillna(0.0)
st.to_parquet(DST)
print(f"\nwrote {DST}   ({st.shape[1]} columns)")
print(st[["water_area_km2", "lake_area_km2", "reach_area_km2"]]
      .describe(percentiles=[.5, .9]).round(3).to_string())
