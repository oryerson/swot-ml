"""Figure 3 of the paper: the variability ratio of SWOT-GAT's 1981-2020 hindcast.

The panels show the ratio and what it relates to (figure set version d).

The variability ratio (VR) is the model's standard deviation (sd) divided by the gauge's sd. The tables
call it `amp`. A VR below 1 means the model varies less than the gauge.

What each panel shows
  a  Map of SWOT-GAT's VR over the 1981-2020 hindcast. One dot per reservoir whose dam, gauge and graph
     node are linked with High or Medium confidence (419 in the paper's run). Colors (cmaps_d.VR_CMAP,
     viridis reversed, light = low) are on a logarithmic scale that spans the 10th to 90th percentile.
     Dot area grows with the logarithm of capacity. A note gives the median and the share below 1. The
     base map is plain grey. The panel shows SWOT-GAT alone, so it keeps the reservoirs excluded in
     excl_d.py.
  b  1981-2020 NSE of the four models (SWOT-GAT, MODROM-cal, GDROM, ISTARF), with the reservoirs grouped
     into thirds of gauge storage irregularity. Irregularity (`irr_harm`) is the sd of the gauge around
     its fitted annual + semiannual harmonic, divided by the harmonic's sd, over water years 1981-2020
     (07_evaluation/c_scoring/drivers/irr_harm.py). The edges of the thirds are the tertiles over every
     reservoir of ops_8120.parquet that has a value. Reservoirs: the 1981-2020 four-model set, scored by
     all four models (293 in the paper's run). Boxes: 25th to 75th percentile, black line at the median,
     whiskers from the 5th to the 95th percentile, all clipped at the axis limits.
  c  SWOT-GAT's 1981-2020 VR (y) against the training variability ratio (x): the sd of the SWOT storage
     the model was trained on in the SWOT Period, divided by the gauge's sd over 1981-2020. Both ratios
     have the same denominator, so on the dashed 1:1 line the hindcast varies as much as the SWOT
     observations did, and below the line it varies less. Both axes are logarithmic. Points are colored by
     SWOT-GAT's 1981-2020 NSE on the color scale of Figure 2a (10th to 90th percentile of all reservoirs
     of panel a; color bar at the bottom of the figure). The header gives Spearman's rho with a 95%
     block-bootstrap interval (kit_j.rho_ci) and n. Reservoirs: those of the skill-driver analysis
     (kit_j.IDS) with both ratios finite and positive (401 in the paper's run). The panel shows SWOT-GAT
     alone, so the excluded reservoirs stay.
  d  Cumulative distributions of VR for SWOT-GAT and MODROM-cal. Solid: 1981-2020 hindcast on the
     1981-2020 four-model set. Dashed: SWOT Period on the SWOT Period four-model set (267 reservoirs in
     the paper's run). A small table gives the four medians.

Reads
  "Kit folder" is the folder of this script; "figure-set folder" is its parent. Other paths are relative to
  CONUS_storage_fork/ under $SWOTGAT_WORK.
  Kit folder:
    score_full_d.csv              full-record scores; the rows of window 1981-2020 are used (prepare_kit.py)
    sites_2325span_g.csv          SWOT Period scores, loaded by kit_h.py (panel d)
    and the other files kit_h.py and kit_j.py load when they are imported
  Figure-set folder:
    score/sd_g8120.csv            per reservoir: sd of SWOT storage in the SWOT Period (sd_swot) and sd of
                                  the gauge over 1981-2020 (sd_g8120); written by full_record/sd8120.py
  eval/overlap50/fig3c_sd/all_sites_harm.csv   irregularity irr_harm, and sd_swot for a consistency check
  eval/perf_attrib/out/ops_8120.parquet        its index is the reservoir list for the tertile edges
  eval/perf_attrib/out/vars.parquet, blocks.csv   reservoir list and spatial blocks (through kit_j.py)
  eval/overlap50/fig1_o50.json    reference: the reservoir sets all four models score
  eval/grl_figs/sites_v5_g.csv    reservoir location and capacity (a table from an earlier model
                                  configuration; only these columns are used)
  gauges_expand/sizing/candidates.csv   GRanD location and capacity for a reservoir without a row there

Writes (kit folder)
  c5d_fig3.pdf, c5d_fig3.png (200 dpi)
  num_c5d_fig3.json  every number drawn, and a few that go to the caption only:
    "a"   n, median VR, 10th and 90th percentile, share below 1, color-bar ticks, color scale, ids located
          from GRanD
    "b"   per third ("0", "1", "2") each model's median NSE and its 5th, 25th, 75th and 95th percentile,
          and n; the tertile edges; the size of the set; the number with an irregularity value
    "c"   rho with its interval, n, the partial rho given irregularity, the share of reservoirs below the
          1:1 line, the number of points outside the axes, the medians of x, of y and of y/x, the limits
          of the NSE color scale. The partial rho and the share below 1:1 are not printed on the figure.
    "d"   per period n and the two medians
    "qa"  messages of the layout check

Usage
  source 07_evaluation/c_scoring/env.sh; bin/run 08_figures/fig3_c5d.py     (after prepare_kit.py)
  No arguments.

Method
  - Scores come from the stage 07 scorers and are not recomputed here.
  - Checks (assertions): every mapped reservoir has coordinates; sd_swot in sd_g8120.csv equals the value
    in all_sites_harm.csv exactly; the SWOT Period four-model set equals the "2023-25" set of
    fig1_o50.json without the excluded reservoirs; in panel d every reservoir of a set has a VR for both
    models.
  - The rule-model rows of the excluded reservoirs are missing from the kit tables, so panels b and d do
    not contain those reservoirs.

Notes
  In the tables, the model labels `SWOT-GNN` (full-record table) and `v10pm` both mean SWOT-GAT (the
  paper's model: mean of two seeds).
"""
from swotgat.paths import WORK
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np, matplotlib as mpl, pandas as pd
from pathlib import Path
from matplotlib.lines import Line2D
import kit_h as K, kit_j as KJ, cmaps_d as CB
from kit_h import jw, G, plt, FS, M4, CLR, TXT, LAB, C
B = K.OUT.parent; O50 = Path(f"{WORK}/CONUS_storage_fork/eval/overlap50"); J50 = json.load(open(O50/"fig1_o50.json")); HM = ["High", "Medium"]
MF = {"SWOT-GNN": "v10pm", "MODROM-cal": "MODROM-cal", "GDROM": "GDROM", "ISTARF": "ISTARF"}
META = pd.read_csv(f"{WORK}/CONUS_storage_fork/eval/grl_figs/sites_v5_g.csv").drop_duplicates("DAM_ID").set_index("DAM_ID")
_CAND = pd.read_csv(f"{WORK}/CONUS_storage_fork/gauges_expand/sizing/candidates.csv").drop_duplicates("GRAND_ID").set_index("GRAND_ID")
FILLED = sorted(set(_CAND.index) - set(META.index))   # dams with no sites_v5_g row (in the paper's run: Bonny CO, 958): GRanD lon/lat/capacity from candidates.csv
META = META.combine_first(_CAND[["lon", "lat", "cap_mcm"]].rename(columns={"cap_mcm": "cap"}))
FULL = pd.read_csv(K.OUT/"score_full_d.csv"); FULL = FULL[(FULL.window == "1981-2020") & FULL.site_tier.isin(HM) & FULL.model.isin(list(MF))].assign(MODEL=lambda x: x.model.map(MF))
GAT = FULL[FULL.MODEL == "v10pm"].join(META[["lon", "lat", "cap"]], on="DAM_ID").set_index("DAM_ID")
assert GAT.lon.notna().all() and GAT.lat.notna().all(); FILLED_HERE = sorted(int(i) for i in set(GAT.index) & set(FILLED))
IDS4 = sorted(set.intersection(*[set(FULL[FULL.MODEL == m].DAM_ID) for m in M4]))
AS = pd.read_csv(O50/"fig3c_sd/all_sites_harm.csv").set_index("DAM_ID"); SD = pd.read_csv(B/"score/sd_g8120.csv").set_index("DAM_ID")
n10, n90 = np.percentile(GAT.nse, [10, 90]); NNORM = mpl.colors.Normalize(n10, n90)                       # Fig 2a's NSE scale
FH = 6.45; fig = plt.figure(figsize=(jw.COL2, FH)); I = lambda inch: inch/FH; NUM = {}
# ---------------- (a) VR map, 1981-2020, grey basemap
MH, MW = 2.30, 0.475; aa = fig.add_axes([0.0, I(3.95), MW, I(MH)]); jw.basemap(aa); aa.set_xlim(-0.372, 0.357); aa.set_ylim(0.045, 0.50)
amp = GAT.amp.values.astype(float); a10, a90 = np.percentile(amp, [10, 90]); AT = [t for t in (0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0) if a10 - 1e-9 <= t <= a90 + 1e-9]
cap = pd.to_numeric(GAT.cap, errors="coerce").values; mx, my = G.albers(GAT.lon.values, GAT.lat.values); o = np.argsort(-np.nan_to_num(cap, nan=0.0))
sc = aa.scatter(mx[o], my[o], c=np.log2(np.clip(amp, 1e-6, None))[o], cmap=CB.VR_CMAP, norm=mpl.colors.Normalize(np.log2(a10), np.log2(a90)),
                s=G.cap_size(cap, lo=2.0, hi=20)[o], lw=0.3, edgecolors="#4d4d4d", zorder=3)
aa.text(0.045, 1.0, f"SWOT-GAT variability ratio · 1981–2020 hindcast · n = {len(GAT)}", transform=aa.transAxes, fontsize=FS, fontweight="bold", va="bottom")
cbr = [0.05, I(3.61), 0.17, I(0.07)]; cbx = fig.add_axes(cbr); cb = fig.colorbar(sc, cax=cbx, orientation="horizontal", extend="both")
cb.set_ticks([np.log2(t) for t in AT]); cb.set_ticklabels([f"{t:g}" for t in AT]); cb.ax.tick_params(labelsize=FS, length=1.8, pad=1); cb.outline.set_linewidth(0.4)
cbx.set_title("SWOT-GAT variability ratio", fontsize=FS, pad=2, color=jw.INK2, loc="left")
fig.text(cbr[0] + cbr[2] + 0.014, cbr[1] - I(0.063), f"median {np.median(amp):.2f} · below 1 at {100*(amp < 1).mean():.0f}%", fontsize=FS, color=jw.INK2, va="center")
K.cap_key(aa); jw.letter(aa, "a", x=0.035, y=1.0)
NUM["a"] = dict(coords_from_grand=FILLED_HERE, n=int(len(amp)), p10=float(a10), p90=float(a90), median=float(np.median(amp)), share_below_1=float((amp < 1).mean()), ticks=AT, cmap=CB.VR_PICK)
# ---------------- (b) NSE by harmonic-irregularity third, 1981-2020 4-model set
OPS = pd.read_parquet(KJ.PA/"ops_8120.parquet"); x8 = AS.irr_harm.reindex(OPS.index).astype(float); EDG8 = np.quantile(x8.dropna().values, [0, 1/3, 2/3, 1])
xi = x8.reindex(IDS4); th8 = pd.Series(np.where(xi.notna(), np.clip(np.searchsorted(EDG8, xi.fillna(0).values, side="right") - 1, 0, 2), -1), index=IDS4)
GT = [list(th8[th8 == k].index) for k in range(3)]; LT = [f"{K.NM3[k]}\n{EDG8[k]:.2f}–{EDG8[k + 1]:.2f}\nn = {len(GT[k])}" for k in range(3)]
PN = FULL[FULL.DAM_ID.isin(IDS4)].pivot(index="DAM_ID", columns="MODEL", values="nse")[M4]
ab = fig.add_axes([0.585, I(4.12), 0.395, I(1.62)]); NUM["b"] = K.model_boxes(ab, PN, GT, LT, "nse")
NUM["b"]["edges"] = [float(e) for e in EDG8]; NUM["b"]["n_set"] = len(IDS4); NUM["b"]["n_with_value"] = int(sum(len(g_) for g_ in GT))
fig.text(0.515, I(6.25), "b", fontsize=9, fontweight="bold", va="baseline")
fig.text(0.543, I(6.25), "NSE by storage irregularity third, 1981–2020 hindcast", fontsize=FS, fontweight="bold", va="baseline")
K.model_key(fig, 0.543, I(6.11), tail=f"· n = {len(IDS4)}")
fig.text(0.543, I(3.60), "irregularity: sd around fitted seasonal harmonic ÷ its sd (1981–2020)", fontsize=FS, color=jw.INK2, va="baseline")
# ---------------- (c) 1981-2020 VR against sd(SWOT)/sd(gauge 1981-2020), coloured by NSE
ac = fig.add_axes([0.085, I(0.98), 0.375, I(1.86)])
IDS = KJ.IDS; x = (SD.sd_swot/SD.sd_g8120).reindex(IDS).values.astype(float); y = GAT.amp.reindex(IDS).values.astype(float); z = GAT.nse.reindex(IDS).values.astype(float)
assert np.allclose(SD.sd_swot.reindex(AS.index).dropna(), AS.sd_swot.dropna(), rtol=0, atol=0)
ok = np.isfinite(x) & (x > 0) & np.isfinite(y) & (y > 0)
r_ = KJ.rho_ci(np.where(ok, x, np.nan), np.where(ok, y, np.nan)); pr = KJ.prho(np.where(ok, x, np.nan), y, AS.irr_harm.reindex(IDS).values.astype(float))
oo = np.argsort(z[ok]); sc3 = ac.scatter(x[ok][oo], y[ok][oo], c=z[ok][oo], cmap=CB.NSE_CMAP, norm=NNORM, s=9, edgecolor="#4d4d4d", linewidth=0.25, zorder=2)
LO, HI = 0.015, 10; ac.plot([LO, HI], [LO, HI], color=jw.INK, lw=1.0, ls=(0, (5, 3)), alpha=0.4, zorder=3)
TKS = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]; TL = ["0.02", "0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10"]
ac.set_xscale("log"); ac.set_yscale("log"); ac.set_xlim(LO, HI); ac.set_ylim(LO, HI); ac.minorticks_off(); ac.set_aspect("equal", adjustable="box", anchor="W")
ac.set_xticks(TKS); ac.set_xticklabels(TL); ac.set_yticks(TKS); ac.set_yticklabels(TL)
ac.set_xlabel("training variability ratio (SWOT sd ÷ 1981–2020 gauge sd)"); ac.set_ylabel("SWOT-GAT variability ratio")
fr = lambda v: f"{v:+.2f}".replace("-", "−"); below = float((y[ok] < x[ok]).mean()); outside = int((ok & ((y < LO) | (y > HI) | (x < LO) | (x > HI))).sum())
ac.text(0.0, 1.11, f"ρ = {fr(r_[0])} [{fr(r_[1])}, {fr(r_[2])}] · n = {r_[3]}", transform=ac.transAxes, fontsize=FS, color=jw.INK2, va="bottom")
ac.text(0.0, 1.20, "SWOT-GAT hindcast variability vs SWOT training variability", transform=ac.transAxes, fontsize=FS, fontweight="bold", va="bottom")
jw.letter(ac, "c", x=-0.17, y=1.20)
NUM["c"] = dict(rho=[float(v) for v in r_[:3]], n=int(r_[3]), partial_irr_harm=float(pr), share_below_1to1=below, n_outside_axes=outside,
                median_x=float(np.median(x[ok])), median_y=float(np.median(y[ok])), median_y_over_x=float(np.median(y[ok]/x[ok])),
                n_ids=int(len(IDS)), n_gat_not_in_ids=int(len(set(GAT.index) - set(IDS))), nse_scale=[float(n10), float(n90)])
cbn = fig.add_axes([0.085, I(0.20), 0.30, I(0.07)]); cbb = fig.colorbar(sc3, cax=cbn, orientation="horizontal", extend="both")
TKN = [n10] + [t for t in (0.2, 0.4, 0.6) if n10 + 0.05 < t <= n90]; cbb.set_ticks(TKN); cbb.set_ticklabels([f"{t:.2f}".replace("-", "−") if t == n10 else f"{t:g}" for t in TKN])
cbb.ax.tick_params(labelsize=FS, length=1.8, pad=1); cbb.outline.set_linewidth(0.4)
cbn.set_title("points in c: SWOT-GAT NSE, 1981–2020 (colours as Fig 2a)", fontsize=FS, pad=2, color=jw.INK2, loc="left")
# ---------------- (d) VR CDFs: 1981-2020 hindcast (solid) and SWOT Period (dashed)
YD, HD = 0.98, 1.78; YT = 0.98 + 1.86*1.20; YS = 0.98 + 1.86*1.08
ad = fig.add_axes([0.605, I(YD), 0.375, I(HD)]); M2 = ["v10pm", "MODROM-cal"]; NUM["d"] = {}; LS = {"8120": "-", "2023-25": (0, (3.2, 1.6))}
S26 = C["S_"][C["S_"].PERIOD == "2023-25"]; ID26 = set.intersection(*[set(S26[S26.MODEL == m].DAM_ID) for m in M4]); assert sorted(ID26) == [i for i in J50["2023-25"]["ids"] if i not in __import__("excl_d").EXCL]
SRC = {"8120": (FULL[FULL.DAM_ID.isin(IDS4)], len(IDS4)), "2023-25": (S26[S26.DAM_ID.isin(ID26)], len(ID26))}
for per, ls in LS.items():
    s, n_ = SRC[per]; NUM["d"][per] = dict(n=n_)
    for m in M2[::-1]:
        v = s[s.MODEL == m].amp.dropna().values; assert len(v) == n_; vv, Fv = jw.ecdf(v); gg = m == "v10pm"
        ad.step(vv, Fv, where="post", color=CLR[m], lw=1.6 if gg else 1.1, ls=ls, zorder=4 if gg else 3); NUM["d"][per][m] = float(np.median(v))
ad.axvline(1, color=jw.INK3, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
ad.set_xscale("log"); ad.set_xlim(1/16, 8); ad.minorticks_off(); ad.set_xticks([1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8]); ad.set_xticklabels(["1/16", "⅛", "¼", "½", "1", "2", "4", "8"])
ad.set_ylim(0, 1); ad.set_yticks([0, 0.5, 1]); ad.set_yticklabels(["0", "0.5", "1"]); ad.set_ylabel("share of reservoirs"); ad.set_xlabel("variability ratio (model sd ÷ gauge sd)")
ad.text(0.585, 0.40, "median", transform=ad.transAxes, fontsize=FS, color=jw.INK3, va="center")
for xc, per in ((0.875, "8120"), (0.965, "2023-25")): ad.plot([xc - 0.035, xc + 0.025], [0.40, 0.40], transform=ad.transAxes, color=jw.INK, lw=1.1, ls=LS[per], clip_on=False)
for k, m in enumerate(M2):
    fw = "bold" if m == "v10pm" else "normal"; yk = 0.28 - 0.11*k
    ad.text(0.585, yk, LAB[m], transform=ad.transAxes, fontsize=FS, color=TXT[m], fontweight=fw, va="center")
    for xc, per in ((0.905, "8120"), (0.995, "2023-25")): ad.text(xc, yk, f"{NUM['d'][per][m]:.2f}", transform=ad.transAxes, fontsize=FS, color=TXT[m], fontweight=fw, ha="right", va="center")
fig.text(0.605, I(YT), "Variability ratio in two periods: SWOT-GAT and MODROM-cal", fontsize=FS, fontweight="bold", va="bottom")
for yb, per, lab, col in ((YS, "8120", f"1981–2020 hindcast · n = {NUM['d']['8120']['n']}", jw.INK2), (YS - 0.15, "2023-25", f"SWOT Period 2023–26, training · n = {NUM['d']['2023-25']['n']}", KJ.TRAIN_T)):
    fig.add_artist(Line2D([0.605, 0.640], [I(yb + 0.035)]*2, transform=fig.transFigure, color=jw.INK, lw=1.1, ls=LS[per])); fig.text(0.648, I(yb), lab, fontsize=FS, color=col, va="bottom")
jw.letter(ad, "d", x=-0.17, y=(YT - YD)/HD)
STEM = "c5d_fig3"; H = K.save(fig, STEM); json.dump(dict(NUM, qa=H), open(K.OUT/f"num_{STEM}.json", "w"), indent=1, default=float)
print(STEM, "qa issues:", len(H), H); print(json.dumps({"a": NUM["a"], "b": {k: ({m: round(v[m], 3) for m in M4} | {"n": v["n"]}) if isinstance(v, dict) else v for k, v in NUM["b"].items()}, "c": NUM["c"], "d": NUM["d"]}, default=float, indent=1))
