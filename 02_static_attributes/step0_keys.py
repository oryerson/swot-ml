"""Test which dam numbering two datasets use, before lakes, dams and gauges are joined by id.

Global Dam Watch (GDW) carries two ids per dam: its own GDW_ID and, for dams that are also in GRanD, a GRAND_ID.
Two other datasets refer to dams by a number without saying which system it is:
  A) the field res_id of the SWOT Prior Lake Database (PLD): is it a GRAND_ID or a GDW_ID?
  B) the DAM_ID of ResOpsUS (the reservoir gauge records): is it a GRAND_ID for every dam?
The evidence under each hypothesis is whether the id exists in GDW, how far the GDW dam or reservoir is from the
lake or dam, and how similar the names are.

Reads
  $SWOTGAT_NAS_WATER/cjgleason/ted/GDW/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp, GDW_barriers_v1_0.shp
  $SWOTGAT_NAS_WATER/cjgleason/ted/LakeSPvC_ts_for_Ted/SWOT_PLD_v106_reservoirs.gdb   (lake_id, res_id, names, lon, lat)
  $SWOTGAT_WORK/CONUS_storage_fork/ResOpsUS/attributes/reservoir_attributes.csv
  It also imports the name-similarity function sim() from harvest_v2.py in $SWOTGAT_WORK/CONUS_storage_fork
  (07_evaluation/a_gauge_truth/agency_2023_2025/harvest_v2.py in this repository).

Writes (in $SWOTGAT_WORK/CONUS_storage_fork/crosswalk)
  step0_pld_resid_key.csv     test A: one row per PLD lake with a res_id and per hypothesis (key = GRAND_ID or GDW_ID):
                              found, dist_km (PLD lake point to the GDW reservoir polygon), name_sim.
                              fix_gdw_static.py reads the GRAND_ID rows as its lake_id -> res_id table.
  step0_resops_damid_key.csv  test B: one row per ResOpsUS dam: in_gdw, dam_dist_km (to the GDW barrier with that
                              GRAND_ID), res_dist_km (to the GDW reservoir polygon), name_sim, the best-named other
                              GDW barrier within 5 km (alt_*), and two verdicts:
                                key_ok       = in GDW, and (dam within 2 km or reservoir within 1 km), and
                                               name similarity >= 0.5
                                key_ok_loose = in GDW, and (dam within 5 km or reservoir within 2 km)

Usage
  python step0_keys.py      (prints a summary of both tests and lists the ResOpsUS dams that fail key_ok)

Notes
  Only features inside the box 125.5 W to 66.5 W, 24.0 N to 49.8 N are read. Distances are computed in the Albers
  projection EPSG:5070.
"""
from swotgat.paths import NAS_WATER, WORK
import sys, numpy as np, pandas as pd, geopandas as gpd, pyogrio
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork"); sys.path.insert(0, str(F))
from harvest_v2 import sim
GDWD = Path(f"{NAS_WATER}/cjgleason/ted/GDW/GDW_v1_0_shp")
PLD = f"{NAS_WATER}/cjgleason/ted/LakeSPvC_ts_for_Ted/SWOT_PLD_v106_reservoirs.gdb"
O = F/"crosswalk"
BB = (-125.5, 24.0, -66.5, 49.8)
def nm(*xs):
    """Return the list of non-empty names (non-blank strings) among the arguments."""
    return [x for x in xs if isinstance(x, str) and x.strip()]
def best_sim(a_list, b_list):
    """Return the highest name similarity between two lists of names, or 0.0 if either list is empty."""
    return max([sim(a, b) for a in a_list for b in b_list] or [0.0])

GR = gpd.read_file(GDWD/"GDW_reservoirs_v1_0.shp", bbox=BB,
                   columns=["GDW_ID", "GRAND_ID", "RES_NAME", "DAM_NAME", "ALT_NAME"]).to_crs(5070)
GB = gpd.read_file(GDWD/"GDW_barriers_v1_0.shp", bbox=BB,
                   columns=["GDW_ID", "GRAND_ID", "RES_NAME", "DAM_NAME", "ALT_NAME"]).to_crs(5070)
for g in (GR, GB):
    g["GDW_ID"] = pd.to_numeric(g.GDW_ID, errors="coerce"); g["GRAND_ID"] = pd.to_numeric(g.GRAND_ID, errors="coerce")
print(f"GDW reservoirs {len(GR):,} (GRAND_ID {GR.GRAND_ID.notna().sum():,}); barriers {len(GB):,} (GRAND_ID {GB.GRAND_ID.notna().sum():,})")
print("GDW_ID == GRAND_ID where both exist:",
      f"{(GR.dropna(subset=['GRAND_ID']).eval('GDW_ID == GRAND_ID')).mean():.1%}")

# ---- A) PLD res_id -------------------------------------------------------------
P = pyogrio.read_dataframe(PLD, columns=["lake_id", "res_id", "names", "lon", "lat"], read_geometry=False)
P["res_id"] = pd.to_numeric(P.res_id, errors="coerce")
for c in ("lon", "lat"): P[c] = pd.to_numeric(P[c], errors="coerce")
P = P[P.res_id.notna() & P.lon.between(BB[0], BB[2]) & P.lat.between(BB[1], BB[3])].copy()
P["lake_id"] = P.lake_id.astype("int64").astype(str)
Pp = gpd.GeoDataFrame(P, geometry=gpd.points_from_xy(P.lon, P.lat), crs=4326).to_crs(5070)
rows = []
for key in ("GRAND_ID", "GDW_ID"):
    G = GR.dropna(subset=[key]).drop_duplicates(key).set_index(key)
    for _, p in Pp.iterrows():
        if p.res_id not in G.index: rows.append(dict(key=key, lake_id=p.lake_id, res_id=p.res_id, found=False)); continue
        g = G.loc[p.res_id]
        rows.append(dict(key=key, lake_id=p.lake_id, res_id=p.res_id, found=True,
                         dist_km=g.geometry.distance(p.geometry)/1e3,
                         name_sim=best_sim(nm(*str(p.names).split(";")), nm(g.RES_NAME, g.DAM_NAME, g.ALT_NAME))))
A = pd.DataFrame(rows); A.to_csv(O/"step0_pld_resid_key.csv", index=False)
print(f"\nA) PLD res_id hypothesis test  ({P.lake_id.nunique():,} CONUS PLD polygons with res_id)")
for key, a in A.groupby("key"):
    f = a[a.found]
    print(f"  res_id as {key:8s}: found {a.found.mean():6.1%} | dist median {f.dist_km.median():6.2f} km, "
          f"<=1 km {(f.dist_km<=1).mean():6.1%} | name sim>=0.6 {(f.name_sim>=.6).mean():6.1%}")

# ---- B) ResOps DAM_ID -----------------------------------------------------------
R = pd.read_csv(F/"ResOpsUS/attributes/reservoir_attributes.csv")
Rp = gpd.GeoDataFrame(R, geometry=gpd.points_from_xy(R.LONG, R.LAT), crs=4326).to_crs(5070)
GBg = GB.dropna(subset=["GRAND_ID"]).drop_duplicates("GRAND_ID").set_index("GRAND_ID")
GRg = GR.dropna(subset=["GRAND_ID"]).drop_duplicates("GRAND_ID").set_index("GRAND_ID")
out = []
for _, r in Rp.iterrows():
    d = dict(DAM_ID=r.DAM_ID, DAM_NAME=r.DAM_NAME, STATE=r.STATE)
    if r.DAM_ID in GBg.index:
        b = GBg.loc[r.DAM_ID]
        d.update(in_gdw=True, dam_dist_km=b.geometry.distance(r.geometry)/1e3,
                 name_sim=best_sim([r.DAM_NAME], nm(b.DAM_NAME, b.RES_NAME, b.ALT_NAME)), gdw_dam=b.DAM_NAME, gdw_res=b.RES_NAME)
        if r.DAM_ID in GRg.index: d["res_dist_km"] = GRg.loc[r.DAM_ID].geometry.distance(r.geometry)/1e3
    else:
        d.update(in_gdw=False)
    # best alternative GDW barrier within 5 km (catches DAM_ID != GRAND_ID)
    near = GB[GB.distance(r.geometry) <= 5000]
    if len(near):
        s = near.apply(lambda b: best_sim([r.DAM_NAME], nm(b.DAM_NAME, b.RES_NAME, b.ALT_NAME)), axis=1)
        k = s.idxmax(); d.update(alt_grand=near.GRAND_ID[k], alt_gdw=near.GDW_ID[k], alt_name=near.DAM_NAME[k],
                                 alt_sim=s[k], alt_dist_km=near.distance(r.geometry)[k]/1e3)
    out.append(d)
B = pd.DataFrame(out)
B["key_ok"] = B.in_gdw & ((B.dam_dist_km <= 2) | (B.res_dist_km.fillna(99) <= 1)) & (B.name_sim >= 0.5)
B["key_ok_loose"] = B.in_gdw & ((B.dam_dist_km <= 5) | (B.res_dist_km.fillna(99) <= 2))
B.to_csv(O/"step0_resops_damid_key.csv", index=False)
print(f"\nB) ResOps DAM_ID as GRAND_ID ({len(B)} dams)")
print(f"  found in GDW barriers : {B.in_gdw.mean():.1%}")
print(f"  strict ok (<=2 km dam or <=1 km reservoir, name>=0.5): {B.key_ok.mean():.1%}")
print(f"  location-only ok (<=5 km / <=2 km)                     : {B.key_ok_loose.mean():.1%}")
bad = B[~B.key_ok]
print(f"  to review: {len(bad)}")
pd.set_option("display.width", 250)
print(bad[["DAM_ID", "DAM_NAME", "STATE", "in_gdw", "dam_dist_km", "res_dist_km", "name_sim", "gdw_dam", "alt_grand", "alt_name", "alt_sim", "alt_dist_km"]]
      .round(2).to_string(max_colwidth=22))
