"""Match every ResOpsUS dam to its graph node(s) and basin, and grade the match (link L2).

Steps 2-4 of the crosswalk: find candidate lake polygons for each dam, choose the node(s) that hold the
reservoir, and assign a confidence tier from the evidence. Only High and Medium tiers are scored in the
paper.

Terms: PLD = SWOT Prior Lake Database (lake polygons; a lake node of the graph carries its PLD lake_id
as node id). GDW = Global Dam Watch (reservoir polygons and dam "barrier" points). wse = SWOT
water-surface elevation.

Keys (checked by 02_static_attributes/step0_keys.py): the ResOpsUS DAM_ID is the GRanD dam id (true for
98.8% of dams in the paper's data), and so is the PLD field res_id (100% found in GDW, 97% within 1 km).
GDW GRAND_ID == 0 means "no GRanD id" and is never used as a key.

Reads (under $SWOTGAT_WORK unless noted)
  CONUS_storage_fork/crosswalk/resops_record_qa.csv           dams with status has_data (step 1)
  CONUS_storage_fork/crosswalk/step0_resops_damid_key.csv     key check per dam (step0_keys.py)
  CONUS_storage_fork/crosswalk/pld_wse_graph.parquet          SWOT wse per lake; 2023-2025 is used
  CONUS_storage_fork/ResOpsUS/                                dam coordinates; 2010-2020 pool elevation
  CONUS_storage_fork/resops_node_match.csv                    an earlier dam-node table, compared at the end
  CONUS/attributes/static_conus_storage.parquet               graph nodes, their basins, drainage areas
  CONUS/basin_lists/storage/train_nodes_g.csv                 training nodes and their basin
  under $SWOTGAT_NAS_WATER: the PLD v1.06 reservoir geodatabase and the GDW v1.0 reservoir and barrier
      shapefiles, clipped to the conterminous US
  It imports the name-similarity function sim from harvest_v2.py.

Writes (CONUS_storage_fork/crosswalk/)
  dam_node_candidates.csv        one row per dam and candidate polygon, with all evidence
  dam_node_crosswalk.csv         one row per dam: primary node, members, basin, tier, weaknesses, conflicts
  dam_node_crosswalk_meta.json   the thresholds (table TH) and the tier counts

Usage
  python step23_crosswalk.py        (no arguments)

Method
  Candidates of a dam are the PLD polygons that
    (a) carry res_id equal to the dam's GRanD id (id_match), or
    (b) lie at least 20% inside the dam's GDW reservoir polygon (in_frac >= 0.2), or
    (c) are within 1 km of a dam point (the GDW barrier point or the ResOpsUS coordinate).
  Evidence per candidate: id_match, id_conflict (the polygon carries another dam's id), in_frac (share
  of the polygon inside the GDW outline), cover_frac (share of the GDW outline it covers), distance to
  the dam, name similarity, median SWOT wse against the dam's own ResOpsUS pool elevation and against
  the GDW pool elevation, SWOT observation counts, and the share inside other GDW reservoirs.

  Eligible polygons are graph nodes with id_match or in_frac >= 0.5, and without id_conflict. The
  primary node is the largest eligible polygon. Every other eligible polygon is
    a minor member   if it is smaller than 5% of the primary (listed, not a member);
    disconnected     if its median SWOT wse differs from the primary's by more than 2 m when both are
                     observed (one pool has one level, so this polygon is another water body);
    a member         otherwise.
  A dam with no eligible polygon gets tier Reject, with the reason: its water sits in a polygon that
  PLD assigns to another dam, or its largest eligible polygon is not a graph node, or there is none.

  Each match collects weaknesses and conflicts:
    id          primary without id_match (geometry only): weakness.
    outline     primary less than 80% inside the GDW outline: weakness.
    distance    dam more than 1 km from the nearest member: weakness; more than 3 km: conflict, unless
                the reservoir is a regulated natural lake (GDW LAKE_CTRL) with id_match.
    area        member area / GDW area outside 1/3-3: weakness if above 3, or if below 1/3 for a single
                id-matched polygon at least 80% inside the outline (PLD low-water extent); otherwise
                conflict.
    level       median SWOT wse against the ResOpsUS 2010-2020 median pool elevation. The tolerance is
                half the 5th-95th percentile operating range plus 3 m (agreement) or plus 10 m (beyond
                it: conflict; in between: weakness). ResOpsUS elevation is used only when it is on a
                sea-level datum, that is, within 30 m of the GDW pool elevation or in agreement with
                SWOT; several operators report local datums. Otherwise SWOT is compared with the GDW
                elevation (15 m agreement, above 40 m conflict). A ResOpsUS conflict is reduced to a
                weakness when SWOT agrees with GDW within 15 m.
    other dam   primary at least 20% inside other GDW reservoirs: conflict.
    name        name similarity below 0.6 without id_match: weakness.
    split       primary below 80% of the connected members' area: weakness.
    off graph   a larger id-matched polygon that is not a graph node: conflict.
    excluded    one or more disconnected polygons were excluded: weakness.
    key         the DAM_ID key check failed for this dam: conflict.
  Tier: Low if there is any conflict; otherwise High with no weakness, Medium with one, Low with more.
  The flag swot_level_high marks a SWOT level above the ResOpsUS pool: more than 10 m above its 95th
  percentile, or a positive ResOpsUS conflict that was reduced to a weakness because SWOT agrees with
  GDW. It points to a suspected height bias of the SWOT lake product and does not change the tier.

  Basin: the training basin if the primary is a training node, otherwise the basin with the largest
  outlet drainage area among the basins that contain the node.
  Two dams that resolve to one primary node are collapsed: the dam with record_status ok and then the
  larger GDW capacity keeps the node; the others get tier Collapsed.

Notes
  The script asserts that node ids are strings, that no node is primary for two live dams, and that
  every member and basin exists in the graph. It then prints a comparison with the earlier dam-node
  table (columns old_node, old_match_verdict).
"""
from swotgat.paths import NAS_WATER, WORK
import sys, json, numpy as np, pandas as pd, geopandas as gpd, pyogrio
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork"); C = Path(f"{WORK}/CONUS")
O = F/"crosswalk"; sys.path.insert(0, str(F))
from harvest_v2 import sim
GDWD = Path(f"{NAS_WATER}/cjgleason/ted/GDW/GDW_v1_0_shp")
PLD = f"{NAS_WATER}/cjgleason/ted/LakeSPvC_ts_for_Ted/SWOT_PLD_v106_reservoirs.gdb"
BB = (-125.5, 24.0, -66.5, 49.8)
EA = 5070
TH = dict(msl_tol=30.0, gdw_ok=15.0, gdw_conflict=40.0, other_gdw=0.2, cand_in=0.2, member_in=0.5, geo_in=0.8, near_km=1.0, far_km=3.0, name=0.6,
          elev_ok=3.0, elev_conflict=10.0, conn_dz=2.0, minor=0.05, split=0.8)

# ---------------- inputs --------------------------------------------------------------------------
Q = pd.read_csv(O/"resops_record_qa.csv")
dams = Q[Q.status == "has_data"].copy()                               # >= 365 d in 2015-2020
R = pd.read_csv(F/"ResOpsUS/attributes/reservoir_attributes.csv").set_index("DAM_ID")
dams = dams.join(R[["LAT", "LONG"]], on="DAM_ID")
dams["GRAND"] = dams.DAM_ID.astype(int)
key_chk = pd.read_csv(O/"step0_resops_damid_key.csv").set_index("DAM_ID")

GR = gpd.read_file(GDWD/"GDW_reservoirs_v1_0.shp", bbox=BB,
                   columns=["GRAND_ID", "RES_NAME", "DAM_NAME", "ALT_NAME", "AREA_SKM", "AREA_MAX", "CAP_MCM", "LAKE_CTRL"])
GR["GRAND_ID"] = pd.to_numeric(GR.GRAND_ID, errors="coerce"); GR = GR[GR.GRAND_ID > 0].drop_duplicates("GRAND_ID")
GB = gpd.read_file(GDWD/"GDW_barriers_v1_0.shp", bbox=BB, columns=["GRAND_ID", "ELEV_MASL"])
GB["GRAND_ID"] = pd.to_numeric(GB.GRAND_ID, errors="coerce"); GB = GB[GB.GRAND_ID > 0].drop_duplicates("GRAND_ID")
GRe = GR.to_crs(EA).set_index("GRAND_ID"); GBe = GB.to_crs(EA).set_index("GRAND_ID")

st = pd.read_parquet(C/"attributes/static_conus_storage.parquet", columns=["lake_area_km2", "uparea_km2"]).reset_index()
st["subbasin"] = st.subbasin.astype(str)
GRAPH = set(st.subbasin)
basins_of = st.groupby("subbasin").basin.apply(list)
basin_size = st.groupby("basin").uparea_km2.max()                      # outlet drainage area per basin
tr = pd.read_csv(C/"basin_lists/storage/train_nodes_g.csv", dtype=str)
train_basin = tr.drop_duplicates("subbasin").set_index("subbasin").basin
SUP = set(tr.subbasin)

P = pyogrio.read_dataframe(PLD, columns=["lake_id", "res_id", "names", "poly_area"], bbox=BB)
P["lake_id"] = P.lake_id.astype("int64").astype(str)
P["res_id"] = pd.to_numeric(P.res_id, errors="coerce").fillna(0).astype(int)
P["poly_area"] = pd.to_numeric(P.poly_area, errors="coerce")
P["in_graph"] = P.lake_id.isin(GRAPH)
Pe = P.to_crs(EA); Pe["pa"] = Pe.area/1e6
print(f"dams {len(dams)}; GDW polygons {len(GR):,}; PLD polygons {len(P):,} ({P.in_graph.sum():,} are graph nodes)")

# SWOT wse per lake (2023-2025) from extract_pld_wse.py
W = pd.read_parquet(O/"pld_wse_graph.parquet")
W = W[(W.date >= "2023-01-01") & (W.date <= "2025-12-31")]
wse_med = W.groupby("lake_id").wse.median(); wse_n = W.groupby("lake_id").wse.count()
dvs_n = W.groupby("lake_id").dv_SWOTAE.count(); dvp_n = W.groupby("lake_id").dv_staticPLD.count()
wse_wide = W.pivot_table(index="date", columns="lake_id", values="wse")

def resops_elev(dam_id):
    """Return (median, 5th percentile, 95th percentile) of a dam's ResOpsUS pool elevation in 2010-2020.

    dam_id is the ResOpsUS DAM_ID. The three values are NaN when the record holds 100 or fewer elevation
    values in that period.
    """
    f = F/f"ResOpsUS/time_series_all/ResOpsUS_{dam_id}.csv"
    t = pd.read_csv(f); t["date"] = pd.to_datetime(t.date, errors="coerce")
    e = pd.to_numeric(t.set_index("date").elevation, errors="coerce").dropna()
    e = e.loc["2010":"2020"]
    return (e.median(), e.quantile(.05), e.quantile(.95)) if len(e) > 100 else (np.nan, np.nan, np.nan)

def names_of(s):
    """Return the list of names in a field, split at commas and semicolons (empty if missing)."""
    return [x.strip() for x in str(s).replace(",", ";").split(";") if x.strip() and x.strip().lower() != "nan"]

# ---------------- candidates -----------------------------------------------------------------------
cand_rows = []
Pidx = Pe.sindex; GRidx = GRe.sindex
for _, d in dams.iterrows():
    g = d.GRAND
    gpoly = GRe.geometry.get(g); gdw = GRe.loc[g] if g in GRe.index else None
    pts = [p for p in (GBe.geometry.get(g),
                       gpd.GeoSeries(gpd.points_from_xy([d.LONG], [d.LAT]), crs=4326).to_crs(EA).iloc[0]) if p is not None]
    aliases = [d.DAM_NAME] + ([x for x in (gdw.RES_NAME, gdw.DAM_NAME, gdw.ALT_NAME) if isinstance(x, str)] if gdw is not None else [])
    zr, zlo, zhi = resops_elev(g)
    idx = set(Pe.index[Pe.res_id == g])
    if gpoly is not None: idx |= set(Pidx.query(gpoly, predicate="intersects"))
    for p in pts: idx |= set(Pidx.query(p.buffer(TH["near_km"]*1e3), predicate="intersects"))
    for i in idx:
        c = Pe.loc[i]
        inter = c.geometry.intersection(gpoly).area/1e6 if gpoly is not None else 0.0
        in_frac = inter/c.pa if c.pa > 0 else 0.0
        cover = inter/(gpoly.area/1e6) if gpoly is not None else np.nan
        dist = min(c.geometry.distance(p) for p in pts)/1e3 if pts else np.nan
        idm, idc = c.res_id == g, (c.res_id > 0 and c.res_id != g)
        if not (idm or in_frac >= TH["cand_in"] or dist <= TH["near_km"]): continue
        wm = wse_med.get(c.lake_id, np.nan)
        others = GRe.iloc[list(GRidx.query(c.geometry, predicate="intersects"))]
        others = others[others.index != g]
        other_frac = sum(c.geometry.intersection(o).area for o in others.geometry)/1e6/c.pa if len(others) and c.pa > 0 else 0.0
        other_ids = ";".join(str(int(i)) for i in others.index)
        cand_rows.append(dict(DAM_ID=g, DAM_NAME=d.DAM_NAME, lake_id=c.lake_id, in_graph=c.in_graph,
            id_match=idm, id_conflict=idc, other_res_id=c.res_id if idc else 0,
            in_frac=in_frac, cover_frac=cover, dam_dist_km=dist, poly_area=c.poly_area,
            name_sim=max([sim(a, b) for a in aliases for b in names_of(c.names)] or [0.0]),
            pld_names=str(c.names)[:60], wse_med=wm, wse_n=int(wse_n.get(c.lake_id, 0)),
            dv_SWOTAE_n=int(dvs_n.get(c.lake_id, 0)), dv_staticPLD_n=int(dvp_n.get(c.lake_id, 0)),
            resops_elev_med=zr, resops_elev_p05=zlo, resops_elev_p95=zhi,
            dz_resops=wm - zr if pd.notna(wm) and pd.notna(zr) else np.nan,
            gdw_elev_masl=GBe.ELEV_MASL.get(g, np.nan),
            dz_gdw=wm - GBe.ELEV_MASL.get(g, np.nan) if pd.notna(wm) else np.nan,
            other_gdw_frac=other_frac, other_gdw_ids=other_ids,
            supervised=c.lake_id in SUP))
CAND = pd.DataFrame(cand_rows)
CAND.to_csv(O/"dam_node_candidates.csv", index=False)
print(f"candidate rows {len(CAND):,}; dams with >=1 candidate {CAND.DAM_ID.nunique()}")

# ---------------- assignment + tiers ---------------------------------------------------------------
def dz_ok(dz, lo, hi):
    """Compare the SWOT level with the ResOpsUS pool elevation (dz = SWOT median wse - ResOpsUS median).

    lo and hi are the 5th and 95th percentiles of the 2010-2020 pool elevation. Returns True (agree) if
    |dz| is within half that operating range plus 3 m, False (conflict) beyond half the range plus 10 m,
    and None (inconclusive) in between or when dz is missing. The 3 m allow for a datum difference and
    for a different pool level in 2023-2025. Without lo and hi, the limits are 3 m and 10 m.
    """
    if pd.isna(dz): return None
    if pd.notna(lo) and pd.notna(hi):
        halfr = (hi - lo)/2
        if abs(dz) <= halfr + TH["elev_ok"]: return True
        if abs(dz) > halfr + TH["elev_conflict"]: return False
        return None
    return True if abs(dz) <= TH["elev_ok"] else (False if abs(dz) > TH["elev_conflict"] else None)

out = []
for g, cs in CAND.groupby("DAM_ID"):
    d = dams.set_index("DAM_ID").loc[g]
    gdw_area = GRe.AREA_SKM.get(g, np.nan)
    elig = cs[cs.in_graph & (cs.id_match | (cs.in_frac >= TH["member_in"])) & ~cs.id_conflict].sort_values("poly_area", ascending=False)
    row = dict(DAM_ID=g, DAM_NAME=d.DAM_NAME, STATE=d.STATE, gdw_area=gdw_area, record_status=d.record_status,
               n_candidates=len(cs), n_candidates_not_in_graph=int((~cs.in_graph & (cs.id_match | (cs.in_frac >= TH["member_in"]))).sum()))
    if elig.empty:
        shared = cs[cs.in_graph & cs.id_conflict & (cs.in_frac >= TH["geo_in"])].sort_values("poly_area", ascending=False)
        offg = cs[~cs.in_graph & (cs.id_match | (cs.in_frac >= TH["member_in"]))].sort_values("poly_area", ascending=False)
        if len(shared):     # the reservoir's water sits in a polygon PLD assigns to another dam (Lake Helena -> Hauser)
            s0 = shared.iloc[0]
            row.update(tier="Reject", shared_with=int(s0.other_res_id), node_if_shared=s0.lake_id,
                       conflicts=f"only polygon ({s0.lake_id}, {s0.poly_area:.1f} km2, {s0.in_frac:.0%} inside GDW) is PLD-assigned to GRanD {int(s0.other_res_id)}")
        elif len(offg):
            row.update(tier="Reject", conflicts=f"largest eligible PLD {offg.lake_id.iat[0]} ({offg.poly_area.iat[0]:.1f} km2) is not a graph node")
        else:
            row.update(tier="Reject", conflicts="no eligible PLD polygon (none id-matched or >=50% inside GDW)")
        out.append(row); continue
    p = elig.iloc[0]
    # hydraulic connectivity of other members to the primary (same pool => same SWOT level)
    mem, minor, discon = [p.lake_id], [], []
    for _, m in elig.iloc[1:].iterrows():
        if m.poly_area < TH["minor"]*p.poly_area: minor.append(m.lake_id); continue
        if pd.notna(m.wse_med) and pd.notna(p.wse_med) and abs(m.wse_med - p.wse_med) > TH["conn_dz"]:
            discon.append(m.lake_id); continue
        mem.append(m.lake_id)
    mem_area = elig.set_index("lake_id").poly_area.reindex(mem).sum()
    # disagreement: an id-matched polygon larger than the geometric primary that is not eligible? (not in graph)
    bigger_offgraph = cs[~cs.in_graph & cs.id_match & (cs.poly_area > p.poly_area)]
    idm_other = elig[elig.id_match & (elig.lake_id != p.lake_id)]
    weak, conf = [], []
    if not p.id_match: weak.append("primary res_id blank/other (geometry-only)")
    if p.in_frac < TH["geo_in"]:
        if p.in_frac < TH["member_in"] and not p.id_match: conf.append(f"primary only {p.in_frac:.0%} inside GDW polygon, no id match")
        else: weak.append(f"primary only {p.in_frac:.0%} inside GDW polygon (GDW outline smaller than PLD)")
    lake_ctrl = isinstance(GRe.LAKE_CTRL.get(g), str) and GRe.LAKE_CTRL.get(g).strip() not in ("", "0")
    dmin = elig[elig.lake_id.isin(mem)].dam_dist_km.min()
    if dmin > TH["near_km"]:
        if dmin > TH["far_km"] and not (lake_ctrl and p.id_match): conf.append(f"dam {dmin:.1f} km from nearest member")
        else: weak.append(f"dam {dmin:.1f} km from nearest member" + (" (regulated natural lake)" if lake_ctrl else ""))
    ar = mem_area/gdw_area if gdw_area > 0 else np.nan
    low_water = pd.notna(ar) and ar < 1/3 and len(mem) == 1 and p.id_match and p.in_frac >= TH["geo_in"]
    if pd.notna(ar) and not (1/3 <= ar <= 3):
        if low_water: weak.append(f"area {ar:.2f}x GDW (single polygon: PLD low-water extent or GDW area)")
        elif ar > 3: weak.append(f"area {ar:.2f}x GDW")
        else: conf.append(f"members cover only {ar:.2f}x GDW area")
    # Level identity test, datum-aware. ResOps elevation is used only if it is on an MSL-like datum
    # (within 30 m of GDW's DEM pool elevation); Duke/Utah Lake etc. report local datums.
    agrees_gdw = pd.notna(p.resops_elev_med) and pd.notna(p.gdw_elev_masl) and abs(p.resops_elev_med - p.gdw_elev_masl) <= TH["msl_tol"]
    agrees_swot = pd.notna(p.dz_resops) and dz_ok(p.dz_resops, p.resops_elev_p05, p.resops_elev_p95) is True
    resops_msl = agrees_gdw or agrees_swot
    level_src, swot_high = "", False
    if pd.notna(p.wse_med):
        if resops_msl:
            ez = dz_ok(p.dz_resops, p.resops_elev_p05, p.resops_elev_p95); level_src = "ResOps"
            if ez is False and pd.notna(p.dz_gdw) and abs(p.dz_gdw) <= TH["gdw_ok"]:
                ez = None; swot_high = p.dz_resops > 0
        elif pd.notna(p.dz_gdw):
            level_src = "GDW (ResOps elev not MSL)" if pd.notna(p.resops_elev_med) else "GDW"
            ez = True if abs(p.dz_gdw) <= TH["gdw_ok"] else (False if abs(p.dz_gdw) > TH["gdw_conflict"] else None)
        else: ez = None
        if ez is False: conf.append(f"SWOT level {p.dz_resops if resops_msl else p.dz_gdw:+.1f} m from {level_src} pool")
        if ez is None and level_src: weak.append(f"SWOT level {p.dz_resops if resops_msl else p.dz_gdw:+.1f} m from {level_src} pool")
    # SWOT above the reservoir's full pool while ResOps says it sits lower: suspected height bias of the
    # SWOT lake product (LakeSP)
    if pd.notna(p.wse_med) and pd.notna(p.resops_elev_p95) and resops_msl and p.wse_med > p.resops_elev_p95 + 10: swot_high = True
    if p.other_gdw_frac >= TH["other_gdw"]: conf.append(f"primary polygon {p.other_gdw_frac:.0%} inside OTHER GDW reservoir(s) {p.other_gdw_ids}")
    if p.name_sim < TH["name"] and not p.id_match: weak.append(f"name sim {p.name_sim:.2f}")
    share = p.poly_area/mem_area if mem_area else np.nan
    if share < TH["split"]: weak.append(f"split: primary {share:.0%} of connected members")
    if len(bigger_offgraph): conf.append(f"larger id-matched polygon {bigger_offgraph.lake_id.iat[0]} ({bigger_offgraph.poly_area.iat[0]:.0f} km2) not in graph")
    if discon: weak.append(f"{len(discon)} id/geometry member(s) at a different SWOT level excluded")
    if key_chk.key_ok.get(g, True) is False: conf.append("ResOps DAM_ID location/name disagrees with GRanD")
    tier = "Low" if conf else ("High" if not weak else ("Medium" if len(weak) == 1 else "Low"))
    # basin: training basin for supervised nodes; else the basin with the largest outlet drainage area
    bl = basins_of.get(p.lake_id, [])
    basin = train_basin.get(p.lake_id) if p.lake_id in train_basin.index else max(bl, key=lambda b: basin_size.get(b, 0))
    row.update(node=p.lake_id, basin=basin, n_basins=len(bl), members=";".join(mem), minor_members=";".join(minor),
               disconnected=";".join(discon), n_members=len(mem), members_area=mem_area, primary_area=p.poly_area,
               primary_share=share, area_ratio=ar, low_water_extent=low_water,
               primary_id_match=p.id_match, primary_in_frac=p.in_frac, primary_dam_km=p.dam_dist_km, member_dam_km=dmin,
               primary_name_sim=p.name_sim, dz_resops=p.dz_resops, dz_gdw=p.dz_gdw, resops_elev_msl=resops_msl,
               swot_level_high=swot_high, other_gdw_frac=p.other_gdw_frac, swot_wse_n=p.wse_n,
               dv_SWOTAE_n=p.dv_SWOTAE_n, dv_staticPLD_n=p.dv_staticPLD_n, supervised=p.supervised,
               tier=tier, weaknesses=" | ".join(weak), conflicts=" | ".join(conf))
    out.append(row)
X = pd.DataFrame(out)
for g in set(dams.GRAND) - set(X.DAM_ID):
    d = dams.set_index("DAM_ID").loc[g]
    X = pd.concat([X, pd.DataFrame([dict(DAM_ID=g, DAM_NAME=d.DAM_NAME, STATE=d.STATE, tier="Reject",
                                          conflicts="no PLD candidate at all")])], ignore_index=True)

# multi-dam reservoirs: two dams resolving to one primary node -> collapse to one entity
X["collapsed_into"] = ""
for n, grp in X[X.node.notna()].groupby("node"):
    if len(grp) < 2: continue
    rank = grp.assign(ok=grp.record_status.eq("ok"), cap=grp.DAM_ID.map(GRe.CAP_MCM)).sort_values(["ok", "cap"], ascending=False)
    keep = rank.DAM_ID.iat[0]
    for g in rank.DAM_ID.iloc[1:]:
        k = X.DAM_ID == g
        X.loc[k, "collapsed_into"] = str(keep)
        X.loc[k, "tier"] = "Collapsed"
    X.loc[X.DAM_ID == keep, "weaknesses"] = X.loc[X.DAM_ID == keep, "weaknesses"].fillna("") + f" | shares node with {list(rank.DAM_ID.iloc[1:])}"

# ---------------- invariants ----------------------------------------------------------------------
live = X[X.tier.isin(["High", "Medium", "Low"])]
assert live.node.map(type).eq(str).all(), "node ids must be str"
assert not live.node.duplicated().any(), "a node is primary for two live dams"
assert live.node.isin(GRAPH).all(), "primary not a graph node"
for _, r in live.iterrows():
    for m in r.members.split(";"): assert m in GRAPH, (r.DAM_ID, m)
    assert r.basin in basins_of.get(r.node, []), (r.DAM_ID, r.basin)
X.to_csv(O/"dam_node_crosswalk.csv", index=False)
json.dump(dict(thresholds=TH, n=len(X), tiers=X.tier.value_counts().to_dict()), open(O/"dam_node_crosswalk_meta.json", "w"), indent=1)

old = pd.read_csv(F/"resops_node_match.csv", dtype={"node": str}).set_index("grand_id").node
X["old_node"] = X.DAM_ID.map(old)
rid = P.set_index("lake_id").res_id; parea = P.set_index("lake_id").poly_area
X["old_node_res_id"] = X.old_node.map(rid)
X["old_node_area"] = X.old_node.map(parea)
def old_verdict(r):
    """Return a label that relates the dam's node in the earlier dam-node table to the new match.

    r is a crosswalk row with old_node and old_node_res_id. The label is "" (dam absent from the earlier
    table), "same", wrong reservoir (the old node carries another dam's GRanD id), old node was a member,
    old node was a fragment (a minor member), now Reject, or old node not a member.
    """
    if pd.isna(r.old_node): return ""
    if r.old_node == r.node: return "same"
    if pd.notna(r.old_node_res_id) and r.old_node_res_id > 0 and r.old_node_res_id != r.DAM_ID: return f"WRONG RESERVOIR (old node is GRanD {int(r.old_node_res_id)})"
    if isinstance(r.members, str) and r.old_node in r.members.split(";"): return "old node was a member (smaller piece)"
    if isinstance(r.minor_members, str) and r.old_node in r.minor_members.split(";"): return "old node was a fragment (<5% of primary)"
    if r.tier == "Reject": return "now Reject"
    return "old node not a member (unrelated/disconnected)"
X["old_match_verdict"] = X.apply(old_verdict, axis=1)
print("\nORIGINAL MATCH AUDIT:"); print(X.old_match_verdict.str.replace(r"\d+", "#", regex=True).value_counts().to_string())
print("\nTIERS:", X.tier.value_counts().to_dict())
print(f"primary differs from ORIGINAL match: {(X.node.notna() & X.old_node.notna() & (X.node != X.old_node)).sum()}"
      f" | original-matched dams now Reject: {(X.old_node.notna() & (X.tier == 'Reject')).sum()}"
      f" | newly matched (not in original 502): {(X.old_node.isna() & X.tier.isin(['High','Medium','Low'])).sum()}")
print(f"split reservoirs: {(X.primary_share < TH['split']).sum()} | low-water single polygons: {X.low_water_extent.fillna(False).astype(bool).sum()}"
      f" | disconnected members excluded: {(X.disconnected.fillna('') != '').sum()} dams | collapsed: {(X.tier == 'Collapsed').sum()}")
X.to_csv(O/"dam_node_crosswalk.csv", index=False)
