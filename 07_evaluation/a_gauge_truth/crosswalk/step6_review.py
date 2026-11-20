"""Draw one review sheet per sampled dam, for the visual audit of the dam-to-node crosswalk.

Step 6 of the crosswalk. A reviewer looks at each sheet and judges whether the chosen node is the dam's
reservoir. The labels are recorded in step6c_audit_labels.py.

Reads (under $SWOTGAT_WORK/CONUS_storage_fork unless noted)
  crosswalk/dam_node_crosswalk.csv, crosswalk/dam_node_candidates.csv     from step23_crosswalk.py
  crosswalk/gauge_final.csv (or gauge_overlap.csv if it does not exist yet)
  crosswalk/pld_wse_graph.parquet                 SWOT water-surface elevation (wse) per lake
  catalogs/{cwms_unfiltered,nwis,nrcs,rise,codwr}.csv                     gauge station coordinates
  ResOpsUS/                                       dam coordinates and records
  under $SWOTGAT_NAS_WATER: the SWOT Prior Lake Database (PLD) v1.06 reservoir geodatabase and the
      Global Dam Watch (GDW) v1.0 reservoir and barrier shapefiles

Writes
  crosswalk/review/<tier>_<DAM_ID>.png            one sheet per dam
  crosswalk/crosswalk_audit.csv                   one row per sheet; label, reviewer and note are empty

Usage
  python step6_review.py        (no arguments)

Method
  Sample: every Low-tier dam of the crosswalk, plus 30 High-tier and 30 Medium-tier dams drawn at
  random without replacement (seed 20260923).
  Left panel, a map: the GDW reservoir outline; the candidate PLD polygons colored by role (primary,
  member, minor, different level, other dam, not in graph, candidate); the GDW dam point; the
  ResOpsUS dam point; the gauge station. The weaknesses and conflicts of the match are printed below.
  Right panel: ResOpsUS storage and elevation for 2015-2020, and the SWOT wse at the primary node on
  the elevation axis. The title gives the gauge series and its step-5 overlap scores (r, amplitude
  ratio, level ratio, class).
"""
from swotgat.paths import NAS_WATER, WORK
import sys, numpy as np, pandas as pd, geopandas as gpd, pyogrio, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork"); O = F/"crosswalk"; RV = O/"review"; RV.mkdir(exist_ok=True)
sys.path.insert(0, str(O)); sys.path.insert(0, str(F))
GDWD = Path(f"{NAS_WATER}/cjgleason/ted/GDW/GDW_v1_0_shp")
PLD = f"{NAS_WATER}/cjgleason/ted/LakeSPvC_ts_for_Ted/SWOT_PLD_v106_reservoirs.gdb"
X = pd.read_csv(O/"dam_node_crosswalk.csv", dtype={"node": str})
CAND = pd.read_csv(O/"dam_node_candidates.csv", dtype={"lake_id": str})
GO = pd.read_csv(O/"gauge_final.csv") if (O/"gauge_final.csv").is_file() else pd.read_csv(O/"gauge_overlap.csv")
rng = np.random.default_rng(20260923)
pick = [X[X.tier == "Low"]]
for t, k in (("High", 30), ("Medium", 30)):
    s = X[X.tier == t]; pick.append(s.iloc[rng.choice(len(s), min(k, len(s)), replace=False)])
S = pd.concat(pick)
R = pd.read_csv(F/"ResOpsUS/attributes/reservoir_attributes.csv").set_index("DAM_ID")
# station coordinates for gauges
cat = pd.concat([pd.read_csv(F/f"catalogs/{c}.csv", dtype=str) for c in ("cwms_unfiltered", "nwis", "nrcs", "rise", "codwr")])
cat = cat.dropna(subset=["lat", "lon"]).drop_duplicates("sid").set_index("sid")
W = pd.read_parquet(O/"pld_wse_graph.parquet")
EA = 5070
BB = (-125.5, 24.0, -66.5, 49.8)
GRall = gpd.read_file(GDWD/"GDW_reservoirs_v1_0.shp", bbox=BB, columns=["GRAND_ID"]).to_crs(EA)
GBall = gpd.read_file(GDWD/"GDW_barriers_v1_0.shp", bbox=BB, columns=["GRAND_ID"]).to_crs(EA)
for G_ in (GRall, GBall): G_["GRAND_ID"] = pd.to_numeric(G_.GRAND_ID, errors="coerce")
PLDall = pyogrio.read_dataframe(PLD, columns=["lake_id", "res_id", "names"], bbox=BB)
PLDall["lake_id"] = PLDall.lake_id.astype("int64").astype(str); PLDall = PLDall.to_crs(EA)
GOF = F/"crosswalk"/"gauge_final.csv"
def pts(lat, lon):
    """Return the point at (lat, lon), in degrees, projected to the equal-area CRS EPSG:5070."""
    return gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs=4326).to_crs(EA).iloc[0]
rows = []
for _, d in S.iterrows():
    g = int(d.DAM_ID); cs = CAND[CAND.DAM_ID == g]
    gdw = GRall[GRall.GRAND_ID == g]; bar = GBall[GBall.GRAND_ID == g]
    pl = PLDall[PLDall.lake_id.isin(set(cs.lake_id))].copy()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw=dict(width_ratios=[1.1, 1]))
    a = ax[0]
    if len(gdw): gdw.boundary.plot(ax=a, color="k", lw=1.2, label="GDW reservoir")
    mem = set(str(d.members).split(";")) if isinstance(d.members, str) else set()
    minor = set(str(d.minor_members).split(";")) if isinstance(d.minor_members, str) else set()
    disc = set(str(d.disconnected).split(";")) if isinstance(d.disconnected, str) else set()
    if pl is not None and len(pl):
        pl = pl.merge(cs, on="lake_id")
        def role(r):
            """Return the (legend label, color) of a candidate polygon by its role in the match."""
            if r.lake_id == d.node: return "primary", "#1f77b4"
            if r.lake_id in mem: return "member", "#6baed6"
            if r.lake_id in minor: return "minor", "#c6dbef"
            if r.lake_id in disc: return "diff. level", "#fdae6b"
            if r.id_conflict: return f"other dam {int(r.other_res_id)}", "#e6550d"
            if not r.in_graph: return "not in graph", "#d9d9d9"
            return "candidate", "#a1d99b"
        for (lab, col), grp in pl.assign(rc=pl.apply(role, axis=1)).groupby("rc"):
            grp.plot(ax=a, color=col, alpha=.75, edgecolor="none", label=lab)
    if len(bar): bar.plot(ax=a, color="red", marker="^", markersize=60, label="GDW dam")
    if g in R.index:
        p = pts(R.LAT[g], R.LONG[g]); a.scatter([p.x], [p.y], marker="x", color="m", s=60, label="ResOps point")
    go = GO[GO.DAM_ID == g] if len(GO) else GO
    gt = go[go.role.str.startswith("truth")] if len(go) else go
    if len(gt) and str(gt.sid.iat[0]) in cat.index:
        c = cat.loc[str(gt.sid.iat[0])]; p = pts(float(c.lat), float(c.lon))
        a.scatter([p.x], [p.y], marker="o", facecolor="none", edgecolor="green", s=80, lw=2, label="gauge station")
    a.set_title(f"{d.DAM_NAME} ({d.STATE}) GRanD {g}\n{d.tier}: node {d.node}", fontsize=9)
    a.legend(fontsize=6, loc="best"); a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")
    txt = f"weak: {d.weaknesses}\nconflict: {d.conflicts}"
    a.text(0, -0.02, txt[:400], transform=a.transAxes, fontsize=6, va="top", wrap=True)
    b = ax[1]
    try:
        rr = pd.read_csv(F/f"ResOpsUS/time_series_all/ResOpsUS_{g}.csv", parse_dates=["date"]).set_index("date").loc["2015":"2020"]
        b.plot(rr.index, rr.storage, color="k", lw=.8, label="ResOps storage (MCM)")
        b2 = b.twinx(); b2.plot(rr.index, rr.elevation, color="0.6", lw=.6, label="ResOps elev (m)")
        if isinstance(d.node, str):
            w = W[W.lake_id == d.node].dropna(subset=["wse"])
            if len(w): b2.scatter(w.date, w.wse, s=6, color="#1f77b4", label="SWOT wse @ primary")
        b2.legend(fontsize=6, loc="lower right")
    except Exception: pass
    if len(gt):
        r0 = gt.iloc[0]
        b.set_title(f"gauge {r0.source} {str(r0.sid)[:45]}\noverlap r={r0.get('ovl_r', np.nan):.3f} amp={r0.get('amp_ratio', np.nan):.2f} "
                    f"level={r0.get('level_ratio', np.nan):.2f} cls={r0.get('cls', '')}", fontsize=8)
    b.legend(fontsize=6, loc="upper left")
    fig.tight_layout(); fn = RV/f"{d.tier}_{g}.png"; fig.savefig(fn, dpi=90); plt.close(fig)
    rows.append(dict(DAM_ID=g, DAM_NAME=d.DAM_NAME, tier=d.tier, png=fn.name, label="", reviewer="", note=""))
pd.DataFrame(rows).to_csv(O/"crosswalk_audit.csv", index=False)
print("review sheets:", len(rows))
