"""Convert the HALF-filtered version-D lake levels of the graph's lakes to daily storage change.

Covers 2023-07-21 to the end of the download. Works on tables only; it does not write to any data store.

Usage: python build_vd_half.py      (no arguments)

Reads (in $SWOTGAT_WORK/CONUS_storage_fork/eval/swotext/)
  half_daily_vC.parquet, half_daily_vD.parquet   daily median levels after HALF (half_batch.py)
  half_obs_vC.parquet, half_obs_vD.parquet       single observations with HALF's keep flag (half_batch.py)
  jida_graph_lakes.parquet                       the reference series: version-C daily levels (wse) and storage
                                                 change (dv, by three methods) delivered by the HALF authors
                                                 (lake_lists.py). The file name and the printed messages call
                                                 it "Jida".

Writes (same folder)
  lakesp_vD_half_graph.parquet   one row per lake and day: wse, wse_vD, the three dv columns, dv_best, dv_source,
                                 out_of_range, offset columns
  version_offsets_half.csv       per lake: level offset between versions D and C, and how it was obtained

Method (the numbers match the printed blocks)
  1. Check. The version-C output of half_batch.py is compared with the reference series on all graph lakes (share
     of days found, share of equal values). In the paper's run HALF v1 reproduces the reference series.
  2. Level frame. Version D levels sit at a slightly different height than version C. Offset per lake = median of
     (wse_adjusted of D - wse_adjusted of C) over the passes (cycle_id, pass_id) that HALF kept in both versions.
     A lake with fewer than 5 shared passes gets the median offset of the lakes in its Pfafstetter level-3 region
     (first three digits of the lake id), or the overall median if the region has none.
     wse = version-D level - offset. This puts the levels in the version-C frame, so that the reference storage
     curves and their reference level apply. wse_vD keeps the native version-D level.
  3. Storage change (dv, million cubic meters). In the reference series each dv is a fixed function of wse, so
     the three methods are refitted per lake as functions of wse:
       static PLD   level change x the constant lake area of the SWOT Prior Lake Database (PLD): linear fit
                    (reproduces the reference values exactly)
       SWOT-AE      SWOT area-elevation curve: quadratic fit (exact) inside the reference wse range; outside it,
                    a straight line with the curve's slope at the range boundary
       3D-LAKES     the 3D-LAKES curve: linear interpolation of the reference (wse, dv) pairs inside their range;
                    outside it, the slope of the end segment
     A method is fitted only if the lake has enough reference points: static PLD 2 points with different wse;
     SWOT-AE 5 points spanning more than 0.1 m; 3D-LAKES 3 distinct levels spanning more than 0.05 m.
     Lakes that are not in the reference series get static PLD = (wse - first valid wse) x LakeSP p_ref_area
     (m x km2 = million cubic meters). Where both exist, the fitted static-PLD slope is checked against p_ref_area.
     dv_best takes SWOT-AE if present, else 3D-LAKES, else static PLD; dv_source records which.
     out_of_range marks days whose wse lies outside the lake's range in the reference series (2023-2025), where
     the curves are extrapolated.
  4. Cross-check. dv_best is compared with LakeSP version D's own storage change ds1_q (observed area against
     the PLD reference) on days with fully observed passes (partial_f == 0), for lakes with at least 10 such days.

Notes
  This record keeps every lake HALF returned observations for. build_halfexact.py then applies HALF's
  temporal-coverage rule to it.
"""
from swotgat.paths import WORK
import numpy as np, pandas as pd
D = f"{WORK}/CONUS_storage_fork/eval/swotext"
hC, hD = pd.read_parquet(f"{D}/half_daily_vC.parquet"), pd.read_parquet(f"{D}/half_daily_vD.parquet")
oC, oD = pd.read_parquet(f"{D}/half_obs_vC.parquet"), pd.read_parquet(f"{D}/half_obs_vD.parquet")
for x in (hC, hD, oC, oD): x["lake_id"] = x.lake_id.astype(str)
J = pd.read_parquet(f"{D}/jida_graph_lakes.parquet"); J["date"] = pd.to_datetime(J.date); Jw = J[J.wse.notna()].copy(); Jw["lake_id"] = Jw.lake_id.astype(str)
# ---- 1. HALF version-C daily series vs the reference series (J)
jl = set(Jw.lake_id); c = hC[hC.lake_id.isin(jl)]; jd = set(zip(Jw.lake_id, Jw.date)); cd = set(zip(c.lake_id, c.date)); m = c.merge(Jw, on=["lake_id", "date"], suffixes=("", "_j"))
print(f"[1] HALF vC vs Jida, {c.lake_id.nunique()} of her {len(jl)} graph lakes: day recall {len(cd & jd)/len(jd):.4f}, precision {len(cd & jd)/len(cd):.4f}, "
      f"exact values {np.mean(np.abs(m.wse - m.wse_j) < 1e-3):.4f}, within 1 cm {np.mean(np.abs(m.wse - m.wse_j) < 0.01):.4f}", flush=True)
lost = jl - set(hC.lake_id); print(f"    Jida lakes with no HALF-vC output: {len(lost)}", flush=True)
# ---- 2. version D - C level offsets from the passes HALF kept in both versions
k = ["lake_id", "cycle_id", "pass_id"]
kc = oC[oC.filter_flag == 1].groupby(k).wse_adjusted.median(); kd = oD[oD.filter_flag == 1].groupby(k).wse_adjusted.median()
P = pd.concat([kc.rename("C"), kd.rename("Dv")], axis=1).dropna().reset_index()
O = P.assign(d=P.Dv - P.C).groupby("lake_id").d.agg(off="median", n_shared="size")
lakes = sorted(set(hD.lake_id)); O = O.reindex(lakes); good = O.n_shared >= 5
reg = O[good].groupby(lambda l: l[:3]).off.median(); allmed = O[good].off.median()
O.loc[~good, "off"] = [reg.get(l[:3], allmed) for l in O.index[~good]]; O["off_flag"] = np.where(good, "lake", "regional")
print(f"[2] D-C offset (HALF passes): lake-level {int(good.sum())} lakes, median {O[good].off.median():.4f} m, IQR {O[good].off.quantile([.25, .75]).round(4).tolist()}; regional {int((~good).sum())}", flush=True)
X = hD.merge(O[["off", "off_flag"]], left_on="lake_id", right_index=True, how="left"); X["wse_vD"] = X.wse; X["wse"] = X.wse - X.off
# ---- 3. storage change
pref = oD.groupby("lake_id").p_ref_area.median()
recs = []; chk = []
for l, g in X.groupby("lake_id"):
    g = g.sort_values("date").copy(); w = g.wse.values; jg = Jw[Jw.lake_id == l]
    g["dv_staticPLD"] = np.nan; g["dv_SWOTAE"] = np.nan; g["dv_3DLAKES"] = np.nan; g["out_of_range"] = False
    if len(jg):
        lo, hi = jg.wse.min(), jg.wse.max(); g["out_of_range"] = (w < lo - 1e-6) | (w > hi + 1e-6)
        s = jg[jg.dv_staticPLD.notna()]
        if len(s) >= 2 and np.ptp(s.wse) > 0:
            b, a = np.polyfit(s.wse, s.dv_staticPLD, 1); g["dv_staticPLD"] = b*w + a
            if np.isfinite(pref.get(l, np.nan)): chk.append(b/pref[l])
        s = jg[jg.dv_SWOTAE.notna()]
        if len(s) >= 5 and np.ptp(s.wse) > 0.1:
            q = np.polyfit(s.wse, s.dv_SWOTAE, 2); l2, h2 = s.wse.min(), s.wse.max(); sl = lambda x: 2*q[0]*x + q[1]
            core = np.polyval(q, np.clip(w, l2, h2)); g["dv_SWOTAE"] = np.where(w < l2, core + sl(l2)*(w - l2), np.where(w > h2, core + sl(h2)*(w - h2), core))
        s = jg[jg.dv_3DLAKES.notna()].groupby("wse").dv_3DLAKES.median().sort_index()
        if len(s) >= 3 and np.ptp(s.index) > 0.05:
            xs, ys = s.index.values, s.values; core = np.interp(w, xs, ys)
            sl0 = (ys[1] - ys[0])/(xs[1] - xs[0]) if xs[1] > xs[0] else 0; sl1 = (ys[-1] - ys[-2])/(xs[-1] - xs[-2]) if xs[-1] > xs[-2] else 0
            g["dv_3DLAKES"] = np.where(w < xs[0], ys[0] + sl0*(w - xs[0]), np.where(w > xs[-1], ys[-1] + sl1*(w - xs[-1]), core))
    elif np.isfinite(pref.get(l, np.nan)):
        g["dv_staticPLD"] = (w - w[0])*pref[l]                    # MCM = m x km2; reference = first valid HALF day, as in the reference series
    recs.append(g)
X = pd.concat(recs, ignore_index=True)
X["dv_best"] = X.dv_SWOTAE.fillna(X.dv_3DLAKES).fillna(X.dv_staticPLD)
X["dv_source"] = np.select([X.dv_SWOTAE.notna(), X.dv_3DLAKES.notna(), X.dv_staticPLD.notna()], ["SWOTAE", "3DLAKES", "staticPLD"], "none")
print(f"[3] Jida static-PLD slope / LakeSP p_ref_area: median {np.median(chk):.4f}, within 1% {np.mean(np.abs(np.array(chk) - 1) < 0.01):.3f} (n {len(chk)})", flush=True)
lk = X.groupby("lake_id").dv_source.agg(lambda s: s.mode().iat[0] if len(s) else "none")
print(f"    lakes by dv_best source: {lk.value_counts().to_dict()}; days outside Jida's wse range: {int(X.out_of_range.sum()):,}", flush=True)
# ---- 4. dv_best vs LakeSP's own storage change ds1_q on fully observed passes
q = oD[(oD.filter_flag == 1) & (oD.partial_f == 0) & oD.ds1_q.notna() & (oD.ds1_q > -9e11)].copy(); q["date"] = q.datetime.dt.normalize()
qd = q.groupby(["lake_id", "date"]).ds1_q.median().rename("ds1_q").reset_index()
Z = X.merge(qd, on=["lake_id", "date"]); st = []
for l, g in Z.groupby("lake_id"):
    if len(g) < 10 or g.dv_best.std() == 0 or g.ds1_q.std() == 0: continue
    st.append((np.corrcoef(g.dv_best, g.ds1_q)[0, 1], g.ds1_q.std()/g.dv_best.std(), g.dv_source.iat[0]))
st = pd.DataFrame(st, columns=["r", "sd_ratio", "src"])
print(f"[4] dv_best vs LakeSP ds1_q (partial_f = 0 days), {len(st)} lakes: median r {st.r.median():.3f}, median sd ratio ds1_q/dv_best {st.sd_ratio.median():.3f}; "
      f"by source r {st.groupby('src').r.median().round(3).to_dict()}", flush=True)
X = X.rename(columns={"n_obs": "n_obs"}); X["source"] = "hydrocron_vD_HALF"
X.to_parquet(f"{D}/lakesp_vD_half_graph.parquet", index=False); O.to_csv(f"{D}/version_offsets_half.csv")
cut = Jw.date.max(); ov = X[(X.date <= cut) & X.lake_id.isin(jl)]
print(f"[5] written: {len(X):,} lake-days, {X.lake_id.nunique()} lakes, {X.date.min().date()} .. {X.date.max().date()}; "
      f"to {cut.date()} on Jida's lakes: HALF-vD {len(ov):,} lake-days vs Jida {len(jd):,} ({len(ov)/len(jd)-1:+.1%}); after: {int((X.date > cut).sum()):,}", flush=True)
mm = ov.merge(Jw, on=["lake_id", "date"], suffixes=("", "_j"))
print(f"    shared days {len(mm):,}: median |wse - Jida| {np.median(np.abs(mm.wse - mm.wse_j)):.4f} m, within 5 cm {np.mean(np.abs(mm.wse - mm.wse_j) < 0.05):.3f}", flush=True)
