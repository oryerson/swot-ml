"""Correct the dam attributes of lake nodes by matching each lake to its own dam in Global Dam Watch (GDW).

Why: build_conus_gdw_static.py gives each lake node the largest GDW reservoir that intersects the node's sub-basin
polygon. A lake just downstream of a big reservoir, whose sub-basin touches that reservoir, then inherits the
upstream dam (for example Diablo took Ross, Oxbow took Brownlee, Hauser took Canyon Ferry; 44 of 1,572 lakes with a
GRanD id in an early run). This script matches by the lake itself and changes only the rows whose dam changes.

Reads
  $V35S_SUBS                                          sub-basin table (is_reservoir, uparea_km2)
  $V35S_WORK/static_conus_storage_$V35S_TAG.parquet   static table from build_storage_static.py
  $SWOTGAT_WORK/GDW_v1_0/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp       GDW reservoir polygons and attributes
  $SWOTGAT_WORK/LakeSPvC_ts_for_Ted/SWOT_PLD_v106_reservoirs.gdb    lake outlines of the SWOT Prior Lake Database (PLD)
  $SWOTGAT_WORK/CONUS_storage_fork/crosswalk/step0_pld_resid_key.csv   PLD lake_id -> res_id, from step0_keys.py

Writes
  $V35S_WORK/static_conus_storage_fixgdw_$V35S_TAG.parquet   the corrected table (the input table is not modified)
  $V35S_WORK/static_fix_changes_$V35S_TAG.csv                the lake nodes that changed, with old and new capacity

Usage
  V35S_SUBS=... V35S_WORK=... V35S_TAG=... python fix_gdw_static.py

Method
  Matching, for every lake node:
    1. by id: the PLD field res_id is a GRanD id (step0_keys.py tests this); it is matched exactly to GDW GRAND_ID.
    2. by overlap, for the lakes left: the GDW reservoir polygon with the largest overlap with the lake's own PLD
       polygon, not counting GDW dams already taken in step 1.
    3. otherwise the lake has no GDW dam and its values are imputed as in build_conus_gdw_static.py (mode "area"):
       capacity from a straight-line fit of log10(capacity) on log10(PLD area) over the matched lakes, depth =
       capacity / area, and the median of the lake nodes for anything still missing.
  Column definitions, the -99 no-data rule, the use ordinals and dam_cap_per_uparea are those of
  build_conus_gdw_static.py.
  Minimal change: a lake row is rewritten only if its dam changes.
    - matched now, and the input row has dam_has_gdw == 1 with the same capacity (within 0.1%): same dam, row kept
    - matched now, otherwise: row replaced ("rematched")
    - not matched now, but matched in the input: the old dam did not touch the lake, row replaced by imputed values
    - not matched now or before: row kept (the input's imputed values stay)
  Rows of nodes that are not lakes must not change; the script checks this.
"""
from swotgat.paths import WORK
import numpy as np, pandas as pd, geopandas as gpd
from pathlib import Path
import os as _os  # file names come from the environment: V35S_SUBS, V35S_WORK, V35S_TAG
_W = Path(_os.environ["V35S_WORK"]); _T = _os.environ["V35S_TAG"]
R = Path(f"{WORK}"); C = R/"CONUS"; F = R/"CONUS_storage_fork"
USE_COLS = {"use_elec": "USE_ELEC", "use_irri": "USE_IRRI", "use_fcon": "USE_FCON", "use_supp": "USE_SUPP", "use_recr": "USE_RECR", "use_navi": "USE_NAVI"}
USE_ORDINAL = {"Major": 3, "Main": 2, "Sec": 1}; NUM = ["dam_cap_mcm", "dam_hgt_m", "dam_depth_m", "dam_dor_pc"]
subs = gpd.read_parquet(_os.environ["V35S_SUBS"]).to_crs(4326); subs.index = subs.index.astype(str)
res = subs[subs["is_reservoir"]].copy(); print("reservoir nodes", len(res))
gdw = gpd.read_file(R/"GDW_v1_0/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp", bbox=tuple(res.total_bounds))
gdw = gdw[["GDW_ID", "GRAND_ID", "DAM_NAME", "CAP_MCM", "DAM_HGT_M", "DEPTH_M", "DOR_PC", *USE_COLS.values(), "geometry"]].to_crs(4326)
for c in ("GRAND_ID", "CAP_MCM", "DAM_HGT_M", "DEPTH_M", "DOR_PC"): gdw[c] = pd.to_numeric(gdw[c], errors="coerce")
K = pd.read_csv(F/"crosswalk/step0_pld_resid_key.csv"); K = K[K.key == "GRAND_ID"].copy(); K["lake_id"] = K.lake_id.astype("int64").astype(str)
rid = K.dropna(subset=["res_id"]).groupby("lake_id").res_id.first().astype(int)
byg = gdw[gdw.GRAND_ID > 0].drop_duplicates("GRAND_ID").set_index("GRAND_ID")
# (1) match by id: PLD res_id == GDW GRAND_ID. pick = node -> GDW record, how = node -> how it was matched
pick = {}; how = {}
for n in res.index:
    d = rid.get(n)
    if d is not None and d in byg.index: pick[n] = byg.loc[d]; how[n] = "res_id"
claimed = {int(v.name) for v in pick.values()}
# (2) match by overlap: PLD lake polygons of the remaining nodes against the GDW polygons not taken in (1)
rest = [n for n in res.index if n not in pick]
pld = gpd.read_file(R/"LakeSPvC_ts_for_Ted/SWOT_PLD_v106_reservoirs.gdb", bbox=tuple(res.total_bounds))
idc = [c for c in pld.columns if c.lower() in ("lake_id", "lakeid")][0]
pld[idc] = pld[idc].astype("int64").astype(str); pld = pld[pld[idc].isin(set(rest))].to_crs(6933).set_index(idc)
g2 = gdw.to_crs(6933); g2 = g2[~g2.GRAND_ID.isin(claimed)]
j = gpd.sjoin(pld[["geometry"]].reset_index(names="node"), g2, how="inner", predicate="intersects")
if len(j):
    j["ovl"] = [pld.geometry[n].intersection(g2.geometry[i]).area for n, i in zip(j.node, j.index_right)]
    j = j.sort_values("ovl", ascending=False).drop_duplicates("node")
    for _, r in j.iterrows():
        pick[r.node] = gdw.loc[r.index_right]; how[r.node] = "lake_overlap"
print({k: sum(1 for v in how.values() if v == k) for k in ("res_id", "lake_overlap")}, "unmatched", len(res) - len(pick))
out = pd.DataFrame(index=res.index)
out["is_reservoir"] = 1; out["dam_has_gdw"] = [1 if n in pick else 0 for n in res.index]
for col, src in (("dam_cap_mcm", "CAP_MCM"), ("dam_hgt_m", "DAM_HGT_M"), ("dam_depth_m", "DEPTH_M"), ("dam_dor_pc", "DOR_PC")):
    out[col] = [pick[n][src] if n in pick else np.nan for n in res.index]
    out.loc[out[col] <= -99, col] = np.nan
out.loc[out.dam_hgt_m <= 0, "dam_hgt_m"] = np.nan
for new, src in USE_COLS.items():
    out[new] = [USE_ORDINAL.get(pick[n][src], 0) if n in pick else 0 for n in res.index]
# (3) imputation as in build_conus_gdw_static.py (mode "area"): log10(capacity) fitted on log10(PLD area)
#     over the matched nodes
pla = gpd.read_file(R/"LakeSPvC_ts_for_Ted/SWOT_PLD_v106_reservoirs.gdb", bbox=tuple(res.total_bounds))
pla[idc] = pla[idc].astype("int64").astype(str); pla = pla[pla[idc].isin(set(res.index))].to_crs("EPSG:6933")
areas = pla.assign(a=pla.geometry.area / 1e6).set_index(idc)["a"].reindex(out.index)
miss = out.dam_cap_mcm.isna(); fit_ok = (~miss) & areas.notna() & (out.dam_cap_mcm > 0) & (areas > 0)
x = np.log10(areas[fit_ok].values); y = np.log10(out.loc[fit_ok, "dam_cap_mcm"].values); slope, icept = np.polyfit(x, y, 1)
est_ok = miss & areas.notna() & (areas > 0); est = 10 ** (slope*np.log10(areas[est_ok].values) + icept)
out.loc[est_ok, "dam_cap_mcm"] = est; out.loc[est_ok, "dam_depth_m"] = est / areas[est_ok].values
print(f"area imputation: slope {slope:.3f} intercept {icept:.3f} n_fit {int(fit_ok.sum())} imputed {int(est_ok.sum())}")
for col in NUM: out[col] = out[col].fillna(out[col].median()).astype("float32")
with np.errstate(divide="ignore", invalid="ignore"):
    out["dam_cap_per_uparea"] = (out.dam_cap_mcm / res.uparea_km2).astype("float32")
st = pd.read_parquet(_W/f"static_conus_storage_{_T}.parquet")
sid = st.index.get_level_values("subbasin").astype(str)
cols = ["is_reservoir", "dam_has_gdw", *NUM, "dam_cap_per_uparea", *USE_COLS]
old = st[cols].copy(); old.index = sid
new = st.copy(); isr = (st.is_reservoir == 1).values
# minimal change: only nodes whose matched dam changes get new values; every other row is kept exactly as in the input
rows = []
for k, n in enumerate(sid):
    if not isr[k]: continue
    o = old.loc[n]; h = how.get(n, "unmatched")
    if h in ("res_id", "lake_overlap"):
        if np.isfinite(out.at[n, "dam_cap_mcm"]) and np.isclose(out.at[n, "dam_cap_mcm"], o.dam_cap_mcm, rtol=1e-3) and o.dam_has_gdw == 1:
            continue                                                     # same dam as before
        act = f"rematched ({h})"
    elif o.dam_has_gdw == 1:
        act = "old match did not touch the lake -> imputed"              # out[] already holds the area-imputed value
    else:
        continue                                                         # imputed before and now: keep the original value
    for c in cols:
        new.iloc[k, new.columns.get_loc(c)] = np.asarray(out.at[n, c]).astype(st[c].dtype)
    rows.append(dict(subbasin=n, action=act, cap_old=float(o.dam_cap_mcm), cap_new=float(out.at[n, "dam_cap_mcm"]),
                     has_gdw_old=int(o.dam_has_gdw), has_gdw_new=int(out.at[n, "dam_has_gdw"])))
ch = pd.DataFrame(rows); ch.to_csv(_W/f"static_fix_changes_{_T}.csv", index=False)
print(f"reservoir nodes changed: {len(ch)} of {int(isr.sum())}:", ch.action.value_counts().to_dict() if len(ch) else {})
nonres_same = (new.loc[~isr].values == st.loc[~isr].values) | (pd.isna(new.loc[~isr]).values & pd.isna(st.loc[~isr]).values)
assert nonres_same.all(), "non-reservoir rows changed"
assert list(new.columns) == list(st.columns) and new.index.equals(st.index)
new.to_parquet(_W/f"static_conus_storage_fixgdw_{_T}.parquet"); print("wrote", _W/f"static_conus_storage_fixgdw_{_T}.parquet")
