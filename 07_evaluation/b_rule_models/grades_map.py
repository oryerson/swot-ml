"""Match every reservoir node of the graph and every test-set dam to its GRADES-hydroDL river reach.

GRADES-hydroDL gives simulated daily discharge per MERIT-Basins river reach (COMID). The rule-model
benchmarks need that discharge at each reservoir, so this script finds the reach of every reservoir
node and the inflow node of every test-set dam. It also reports how many test-set dams have GDROM v2
rules.

Usage:
  python grades_map.py          (no arguments)

Reads:
  $SWOTGAT_WORK/CONUS/metadata/subbasins_conus.parquet    nodes of the graph (upstream area,
                                                          original_comid, node_type, outlet_id)
  $SWOTGAT_NAS_ICE/data/MERIT_Basins/MERIT_Hydro_v07_Basins_v01/pfaf_level_02/pfaf_7*/riv*.shp
                                                          MERIT-Basins v0.7 reaches (COMID, uparea)
  $SWOTGAT_NAS_ICE/data/MERIT_Basins/v07_to_bugfix/v07_to_bugfix_pfaf1_7.csv
                                                          reach ids, v0.7 against the bugfix version
  $SWOTGAT_WORK/GRADES/gdrom_v2/file_list.json            file listing of the GDROM v2 dataset
  $SWOTGAT_WORK/CONUS_storage_fork/testsets/{regime1_2023_2025,regime2_2015_2020}/sites.csv
                                                          the two test-set site tables

Writes (in $SWOTGAT_WORK/GRADES/):
  reservoir_grades_map.csv   one row per reservoir node: outlet_id, uparea_km2, comid_bugfix,
                             comid_v07, reach_uparea_v07, area_ratio
  testset_dams.csv           one row per dam and test set (regime r1 or r2): name, tier, n_members,
                             inflow_node, comid_v07, area_ratio, gdrom (True if the dam has GDROM v2
                             rules). A copy of the paper's file ships in this folder's `inputs/`.

Method:
  - Reach ids. The graph's `original_comid` is a reach id of the MERIT-Basins *bugfix* version.
    GRADES-hydroDL uses MERIT-Basins v0.7, so each id is translated with the v07_to_bugfix table
    distributed with MERIT-Basins. An id missing from the table but present in v0.7 is kept as is
    (ids unchanged between the versions). Evidence that the graph ids are bugfix ids, in the paper's
    run: the ratio of node to reach upstream area is within 0.8-1.2 for 60% of reservoir nodes after
    translation, against 7% when the ids are read as v0.7 ids.
  - area_ratio = upstream area of the node / upstream area of its v0.7 reach. The dam usually sits
    partway along the reach, so the simulators (grades_sim_v4*.py) multiply the reach discharge by
    this ratio.
  - Inflow node of a dam = the member node with the largest upstream area, which is the most
    downstream part of the reservoir. A dam with no listed members uses its own node.
  - GDROM coverage: a dam has rules if the dataset listing contains module_conditions/<dam id>.txt.
"""
from swotgat.paths import NAS_ICE, WORK
import glob, json, urllib.parse, numpy as np, pandas as pd, pyogrio
from pathlib import Path
G = Path(f"{WORK}/GRADES"); M = f"{NAS_ICE}/data/MERIT_Basins/"
F = Path(f"{WORK}/CONUS_storage_fork")
v07 = pd.concat([pyogrio.read_dataframe(f, read_geometry=False, columns=["COMID", "uparea"])
                 for f in sorted(glob.glob(M + "MERIT_Hydro_v07_Basins_v01/pfaf_level_02/pfaf_7*/riv*.shp"))]).set_index("COMID").uparea
cw = pd.read_csv(M + "v07_to_bugfix/v07_to_bugfix_pfaf1_7.csv")
b2v = cw.drop_duplicates("COMID_bugfix").set_index("COMID_bugfix").COMID_v07
s = pd.read_parquet(f"{WORK}/CONUS/metadata/subbasins_conus.parquet",
                    columns=["uparea_km2", "original_comid", "node_type", "outlet_id"])
r = s[s.node_type == "reservoir"].copy(); r.index = r.index.astype(str)
r["comid_bugfix"] = r.original_comid.astype("int64")
r["comid_v07"] = r.comid_bugfix.map(b2v)
same = r.comid_v07.isna() & r.comid_bugfix.isin(v07.index)          # ids unchanged between versions
r.loc[same, "comid_v07"] = r.loc[same, "comid_bugfix"]
r["reach_uparea_v07"] = r.comid_v07.map(v07)
r["area_ratio"] = r.uparea_km2 / r.reach_uparea_v07
r[["outlet_id", "uparea_km2", "comid_bugfix", "comid_v07", "reach_uparea_v07", "area_ratio"]].to_csv(G/"reservoir_grades_map.csv")
print(f"reservoir nodes {len(r):,} | mapped to a v0.7 reach {r.comid_v07.notna().mean():.1%} | area ratio median {r.area_ratio.median():.3f}, "
      f"0.8-1.2 {r.area_ratio.between(0.8, 1.2).mean():.1%}, 0.5-2 {r.area_ratio.between(0.5, 2).mean():.1%}")

rules = set()
for f in json.load(open(G/"gdrom_v2/file_list.json")):
    p = urllib.parse.unquote(f["url"].split("/data/contents/")[1])
    if "module_conditions/" in p: rules.add(int(p.split("/")[-1].split(".")[0]))
rows = []
for reg, sub in (("r1", "regime1_2023_2025"), ("r2", "regime2_2015_2020")):
    S = pd.read_csv(F/"testsets"/sub/"sites.csv", dtype={"node": str, "members": str})
    for _, x in S.iterrows():
        mem = [m for m in str(x.members).split(";") if m and m != "nan"] or [str(x.node)]
        mm = r.reindex(mem).dropna(subset=["uparea_km2"])
        if mm.empty: rows.append(dict(regime=reg, DAM_ID=x.DAM_ID, name=x.DAM_NAME, tier=x.site_tier, gdrom=x.DAM_ID in rules)); continue
        k = mm.uparea_km2.idxmax()
        rows.append(dict(regime=reg, DAM_ID=x.DAM_ID, name=x.DAM_NAME, tier=x.site_tier, n_members=len(mem), inflow_node=k,
                         comid_v07=mm.loc[k, "comid_v07"], area_ratio=mm.loc[k, "area_ratio"], gdrom=x.DAM_ID in rules))
D = pd.DataFrame(rows); D.to_csv(G/"testset_dams.csv", index=False)
for reg, g in D.groupby("regime"):
    hm = g[g.tier.isin(["High", "Medium"])]
    print(f"{reg}: dams {len(g)} | with GRADES reach {g.comid_v07.notna().mean():.1%} | with GDROM v2 rules {g.gdrom.mean():.1%} "
          f"| High+Medium {len(hm)}, of which with rules {hm.gdrom.mean():.1%}")
print(f"GDROM v2 rule sets: {len(rules)}")
