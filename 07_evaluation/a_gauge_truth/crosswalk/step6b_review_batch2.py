"""Draw a second batch of review sheets: more High-tier dams for the visual audit of the crosswalk.

Step 6b of the crosswalk. The first batch (step6_review.py) holds 30 High-tier dams. This script adds
up to 95 more, so that the High-tier error rate computed in step6c_audit_labels.py rests on a larger
sample.

Reads
  The same inputs as step6_review.py, plus crosswalk/crosswalk_audit.csv (the dams already drawn).

Writes (under $SWOTGAT_WORK/CONUS_storage_fork)
  crosswalk/review/batch2_<NN>.png                four dams per image
  crosswalk/crosswalk_audit_batch2.csv            one row per dam; label, reviewer and note are empty

Usage
  python step6b_review_batch2.py        (no arguments; run after step6_review.py)

Method
  Sample: 95 High-tier dams that are not in the first batch, drawn at random without replacement
  (seed 20260924), or all of them if fewer remain.
  Each dam gets the same two panels as in step6_review.py: a map of the reservoir outline, candidate
  polygons, dam points and gauge station; and the 2015-2020 ResOpsUS record with the SWOT
  water-surface elevation at the primary node. Four dams share one image (2 rows x 4 panels).
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
done = set(pd.read_csv(O/"crosswalk_audit.csv").DAM_ID)
rng = np.random.default_rng(20260924)
s = X[(X.tier == "High") & ~X.DAM_ID.isin(done)]
S = s.iloc[rng.choice(len(s), min(95, len(s)), replace=False)]
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
for i, (_, d) in enumerate(S.iterrows()):
    g = int(d.DAM_ID); cs = CAND[CAND.DAM_ID == g]
    gdw = GRall[GRall.GRAND_ID == g]; bar = GBall[GBall.GRAND_ID == g]
    pl = PLDall[PLDall.lake_id.isin(set(cs.lake_id))].copy()
    k = i % 4
    if k == 0:
        fig, AX = plt.subplots(2, 4, figsize=(20, 9)); names = []
    ax = AX[k//2, (k%2)*2:(k%2)*2+2]
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
    a.text(0, -0.02, txt[:200], transform=a.transAxes, fontsize=6, va="top", wrap=True)
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
    names.append(g)
    fn = RV/f"batch2_{i//4:02d}.png"
    if k == 3 or i == len(S) - 1:
        fig.tight_layout(); fig.savefig(fn, dpi=75); plt.close(fig)
    rows.append(dict(DAM_ID=g, DAM_NAME=d.DAM_NAME, tier=d.tier, png=fn.name, label="", reviewer="", note=""))
pd.DataFrame(rows).to_csv(O/"crosswalk_audit_batch2.csv", index=False)
print("review sheets:", len(rows))
