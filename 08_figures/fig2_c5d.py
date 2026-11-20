"""Figure 2 of the paper: a map of SWOT-GAT's NSE, and NSE distributions by reservoir purpose.

The distributions are those of SWOT-GAT and of the rule-model ensemble (figure set version d).

What each panel shows
  a  Map of SWOT-GAT's NSE over the full 1981-2020 hindcast. One dot per reservoir whose dam, gauge and
     graph node are linked with High or Medium confidence (419 in the paper's run). Dot area grows with
     the logarithm of capacity; large reservoirs are drawn first so that small ones stay visible. The
     colors (cmaps_d.NSE_CMAP, YlGnBu, light = low) span the 10th to 90th percentile of NSE, and the end
     triangles of the color bar hold the values beyond. The color-bar title gives n and the median.
     The panel shows SWOT-GAT alone, so it keeps the reservoirs excluded in excl_d.py.
  b  1981-2020 hindcast: cumulative distributions of NSE, one color per main purpose (irrigation, water
     supply, hydropower, flood control, recreation). Solid lines: SWOT-GAT. Dashed lines: the rule-model
     ensemble (table label RULE-ENS: the daily mean of GDROM, ISTARF and MODROM-cal storage, scored like
     any model). NSE below -1 is drawn at -1.
  c  The same for the SWOT Period (each reservoir scored on its SWOT span, July 2023 to June 2026), which
     is SWOT-GAT's training period.
  Under b and c, a table gives n and the median NSE of SWOT-GAT and of the ensemble, per purpose and for
  all reservoirs. Reservoirs with another main purpose count in the "all" row only; a last line gives
  their number.
  Panels b and c use the reservoirs where both SWOT-GAT and the ensemble are scored. The excluded
  reservoirs are not among them, because prepare_kit.py removed their rule-model rows from the kit tables.
  In the paper's run: 293 reservoirs in b and 267 in c.

Reads
  "Kit folder" is the folder of this script. Other paths are relative to CONUS_storage_fork/ under
  $SWOTGAT_WORK.
  Kit folder (written by prepare_kit.py):
    score_full_d.csv              full-record scores; the rows of window 1981-2020 are used (panels a, b)
    sites_2325span_ens_d.csv      SWOT Period scores, including the ensemble (panel c)
    and the files kit_h.py loads when it is imported (see kit_h.py)
  eval/grl_figs/sites_v5_g.csv    reservoir location, capacity and main purpose (a table from an earlier
                                  model configuration; only these columns are used)
  gauges_expand/sizing/candidates.csv   GRanD location and capacity for a reservoir that has no row in
                                  sites_v5_g.csv (in the paper's run: Bonny, CO, DAM_ID 958)
  eval/overlap50/fig1_o50.json    reference: the reservoir sets all four models score
  eval/overlap50/tables/per_reservoir.csv   reference: SWOT-GAT's SWOT Period NSE per reservoir

Writes (kit folder)
  c5d_fig2.pdf, c5d_fig2.png (200 dpi)
  num_c5d_fig2.json  every number drawn:
    "a"        n, median NSE, the 10th and 90th percentile, the shares below and above them, the color
               scale, the ids shown and the ids located from GRanD
    "8120"     panel b: n; for all reservoirs the median, the 90th percentile and the share below 0 of
               each of the two, and the share of reservoirs where the ensemble has the higher NSE; per
               purpose n and the two medians; the number with another main purpose
    "2023-25"  the same for panel c
    "qa"       messages of the layout check

Usage
  source 07_evaluation/c_scoring/env.sh; bin/run 08_figures/fig2_c5d.py     (after prepare_kit.py)
  No arguments.

Method
  - Scores come from the stage 07 scorers and are not recomputed here. They follow the overlap50 rule: a
    reservoir is scored in a window only if model and gauge share at least 50% of the window's days.
  - Checks (assertions). Panel a: every reservoir has coordinates. Panel b: the reservoirs with both a
    SWOT-GAT and an ensemble score are exactly the 1981-2020 four-model set (scored by all of SWOT-GAT,
    MODROM-cal, GDROM and ISTARF). Panel c: the reservoir set equals the "2023-25" set of fig1_o50.json
    without the excluded reservoirs, and each reservoir's SWOT-GAT NSE equals per_reservoir.csv to 5e-6.

Notes
  In the tables, the model labels `SWOT-GNN` (full-record table) and `v10pm` (SWOT Period table) both mean
  SWOT-GAT (the paper's model: mean of two seeds).
"""
from swotgat.paths import WORK
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np, pandas as pd, matplotlib as mpl
from pathlib import Path
from matplotlib.lines import Line2D
import kit_h as K, cmaps_d as CB
from kit_h import jw, G, plt, FS
B = K.OUT.parent; O50 = Path(f"{WORK}/CONUS_storage_fork/eval/overlap50"); J50 = json.load(open(O50/"fig1_o50.json"))
PURP = [("Irrigation", "irrigation", "#008300"), ("Water supply", "water supply", "#6250d6"), ("Hydroelectricity", "hydropower", "#e34948"),
        ("Flood control", "flood control", "#eda100"), ("Recreation", "recreation", "#1baf7a")]
HM = ["High", "Medium"]; M4F = {"SWOT-GNN": "v10pm", "MODROM-cal": "MODROM-cal", "GDROM": "GDROM", "ISTARF": "ISTARF"}
META = pd.read_csv(f"{WORK}/CONUS_storage_fork/eval/grl_figs/sites_v5_g.csv").drop_duplicates("DAM_ID").set_index("DAM_ID")
_CAND = pd.read_csv(f"{WORK}/CONUS_storage_fork/gauges_expand/sizing/candidates.csv").drop_duplicates("GRAND_ID").set_index("GRAND_ID")
FILLED = sorted(set(_CAND.index) - set(META.index))   # dams with no sites_v5_g row (in the paper's run: Bonny CO, 958): GRanD lon/lat/capacity from candidates.csv
META = META.combine_first(_CAND[["lon", "lat", "cap_mcm"]].rename(columns={"cap_mcm": "cap"}))
FULL = pd.read_csv(K.OUT/"score_full_d.csv"); FULL = FULL[(FULL.window == "1981-2020") & FULL.site_tier.isin(HM)]
MW = 5.80; MH = MW*(0.50 - 0.045)/(0.357 + 0.372); YM = 4.20; FH = YM + MH + 0.20
fig = plt.figure(figsize=(jw.COL2, FH)); I = lambda inch: inch/FH; NUM = {}
# ---------------- (a) NSE map, 1981-2020
ax = fig.add_axes([(jw.COL2 - MW)/2/jw.COL2, I(YM), MW/jw.COL2, I(MH)]); jw.basemap(ax)
s = FULL[FULL.model == "SWOT-GNN"].join(META[["lon", "lat", "cap"]], on="DAM_ID"); assert s.lon.notna().all() and s.lat.notna().all()
cap = pd.to_numeric(s.cap, errors="coerce"); x, y = G.albers(s.lon.values, s.lat.values); o = np.argsort(-cap.fillna(0).values)
n10, n90 = np.percentile(s.nse, [10, 90]); NORM = mpl.colors.Normalize(vmin=n10, vmax=n90)
sc = ax.scatter(x[o], y[o], c=s.nse.values[o], cmap=CB.NSE_CMAP, norm=NORM, s=G.cap_size(cap.values, lo=2.5, hi=30)[o], lw=0.3, edgecolors="#4d4d4d", zorder=3)
bb = ax.get_position(); cbx = fig.add_axes([bb.x0 + 0.015*bb.width, bb.y0 + 0.06*bb.height, 0.19*bb.width, I(0.07)])
cb = fig.colorbar(sc, cax=cbx, orientation="horizontal", extend="both")
TK = [n10] + [t for t in (0.2, 0.4, 0.6) if n10 + 0.05 < t <= n90]; cb.set_ticks(TK); cb.set_ticklabels([f"{t:.2f}".replace("-", "−") if t == n10 else f"{t:g}" for t in TK])
cb.ax.tick_params(labelsize=FS, length=1.8, pad=1); cb.outline.set_linewidth(0.4)
cbx.set_title("SWOT-GAT NSE, 1981–2020\n" + f"n = {len(s)} · median " + f"{s.nse.median():.2f}".replace("-", "−"), fontsize=FS, pad=2, color=jw.INK2, loc="left")
NUM["a"] = dict(coords_from_grand=sorted(int(i) for i in set(s.DAM_ID) & set(FILLED)), n=int(len(s)), median_nse=float(s.nse.median()), p10=float(n10), p90=float(n90), share_below_p10=float((s.nse < n10).mean()),
                share_above_p90=float((s.nse > n90).mean()), cmap=CB.NSE_PICK, ids=sorted(int(i) for i in s.DAM_ID))
KC = [100, 1000, 10000]; ky = [0.33, 0.265, 0.195]
ax.scatter([0.985]*3, ky, s=G.cap_size(np.array(KC, float), lo=2.5, hi=30), facecolors="none", edgecolors="#4d4d4d", lw=0.5, transform=ax.transAxes, zorder=5, clip_on=False)
for c_, y_ in zip(KC, ky): ax.text(0.958, y_, f"{c_:,}", transform=ax.transAxes, fontsize=FS, color=jw.INK2, ha="right", va="center")
ax.text(1.0, 0.37, "capacity\n(MCM)", transform=ax.transAxes, fontsize=FS, color=jw.INK2, ha="right", va="bottom", linespacing=1.0)
jw.letter(ax, "a", x=0.02, y=1.0); ax.text(0.03, 1.0, "1981–2020 hindcast", transform=ax.transAxes, fontsize=FS, fontweight="bold", va="bottom")
# ---------------- (b, c) NSE CDFs by main purpose, SWOT-GAT solid, rule ensemble dashed
use = META.use
SF = FULL.assign(MODEL=FULL.model.map({"SWOT-GNN": "GAT", "RULE-ENS": "ENS"})).dropna(subset=["MODEL"])
M4_8120 = set.intersection(*[set(FULL[FULL.model == m].DAM_ID) for m in M4F])
SX = pd.read_csv(K.OUT/"sites_2325span_ens_d.csv"); SX = SX[SX.tier.isin(HM) & SX.nrmse.notna()]; SX = SX.assign(MODEL=SX.MODEL.map({"v10pm": "GAT", "RULE-ENS": "ENS"})).dropna(subset=["MODEL"])
PR = pd.read_csv(O50/"tables/per_reservoir.csv")
PER = [("8120", SF, "Full hindcast, 1981–2020", ""), ("2023-25", SX, "SWOT Period, Jul 2023–Jun 2026", "SWOT-GAT's training period")]
XC, WC, YB, PH = [0.075, 0.555], 0.395, 1.35, 2.05; LSD = (0, (4.0, 2.2))
for j, (per, d, ttl, sub) in enumerate(PER):
    ids = set(d[d.MODEL == "GAT"].DAM_ID) & set(d[d.MODEL == "ENS"].DAM_ID); d = d[d.DAM_ID.isin(ids)]
    pv = d.pivot(index="DAM_ID", columns="MODEL", values="nse")
    if per == "8120":
        assert ids == M4_8120, (len(ids), len(M4_8120))                                              # = the 1981-2020 4-model set
    else:
        assert sorted(int(i) for i in ids) == [i for i in J50[per]["ids"] if i not in __import__("excl_d").EXCL], (per, len(ids))
        pr = PR[(PR.window == "SWOT Period") & (PR.m == "SWOT-GAT")].set_index("DAM_ID").nse.reindex(pv.index)
        assert np.allclose(pr.values, pv.GAT.values, rtol=0, atol=5e-6), per                 # checked per reservoir: after the exclusion the set differs from the reference set, so its median does too
    u = use.reindex(pv.index); NUM[per] = dict(n=len(ids), all=dict(GAT=float(pv.GAT.median()), ENS=float(pv.ENS.median()), GAT_p90=float(np.percentile(pv.GAT, 90)),
                                                                  ENS_p90=float(np.percentile(pv.ENS, 90)), GAT_lt0=float((pv.GAT < 0).mean()), ENS_lt0=float((pv.ENS < 0).mean()),
                                                                  ENS_better=float((pv.ENS > pv.GAT).mean())), purpose={}, n_other=int((~u.isin([p for p, _, _ in PURP])).sum()))
    fig.text(XC[j] - 0.004, I(YB + PH + 0.36), "bc"[j], fontsize=9, fontweight="bold", va="baseline")
    fig.text(XC[j] + 0.018, I(YB + PH + 0.36), ttl, fontsize=FS, fontweight="bold", color=jw.INK, va="baseline")
    fig.text(XC[j] + 0.018, I(YB + PH + 0.22), f"{sub} · n = {len(ids)}" if sub else f"n = {len(ids)}", fontsize=FS, color=jw.INK2, va="baseline")
    a = fig.add_axes([XC[j], I(YB), WC, I(PH)])
    for p, lab, col in PURP[::-1]:
        g = list(u[u == p].index)
        for mdl, ls, lw in (("ENS", LSD, 1.15), ("GAT", "-", 1.5)):
            v = pv.loc[g, mdl].dropna().values; vv, Fv = jw.ecdf(np.clip(v, -1.0, 1.0)); a.step(vv, Fv, where="post", color=col, lw=lw, ls=ls, zorder=4 if mdl == "GAT" else 3)
        NUM[per]["purpose"][p] = dict(n=len(g), GAT=float(pv.loc[g, "GAT"].median()), ENS=float(pv.loc[g, "ENS"].median()))
    a.axvline(0, color=jw.INK3, lw=0.6, ls=(0, (3, 2))); a.set_xlim(-1, 1); a.set_xticks([-1, -0.5, 0, 0.5, 1]); a.set_xticklabels(["≤ −1", "−0.5", "0", "0.5", "1"])
    a.set_ylim(0, 1); a.set_yticks([0, 0.5, 1]); a.set_yticklabels(["0", "0.5", "1"]); a.set_xlabel("NSE")
    if j == 0: a.set_ylabel("share of reservoirs")
    yt = YB - 0.40; cx = [0.205, 0.285, 0.385]
    for xr, h in zip(cx, ("n", "SWOT-GAT", "rule ensemble")): fig.text(XC[j] + xr, I(yt), h, fontsize=FS, color=jw.INK3, ha="right", va="baseline")
    fig.text(XC[j], I(yt), "median NSE", fontsize=FS, color=jw.INK3, va="baseline")
    rows_ = [(p, col, NUM[per]["purpose"][p]) for p, _, col in PURP] + [("all reservoirs", None, dict(n=len(ids), **NUM[per]["all"]))]
    for k, (p, col, v) in enumerate(rows_):
        yk = yt - 0.118*(k + 1)
        if col: fig.add_artist(Line2D([XC[j], XC[j] + 0.022], [I(yk + 0.03)]*2, transform=fig.transFigure, color=col, lw=1.5))
        fig.text(XC[j] + (0.028 if col else 0.0), I(yk), p, fontsize=FS, color=jw.INK, va="baseline", fontweight="normal" if col else "bold")
        for xr, val in zip(cx, (f"{v['n']}", f"{v['GAT']:.2f}", f"{v['ENS']:.2f}")):
            fig.text(XC[j] + xr, I(yk), val.replace("-", "−"), fontsize=FS, color=jw.INK, ha="right", va="baseline", fontweight="normal" if col else "bold")
    if NUM[per]["n_other"]: fig.text(XC[j], I(yt - 0.118*7), f"other main purposes (in “all”, no line): {NUM[per]['n_other']}", fontsize=FS, color=jw.INK2, va="baseline")
yk = YB + PH + 0.66
fig.add_artist(Line2D([0.075, 0.115], [I(yk + 0.03)]*2, transform=fig.transFigure, color=jw.INK, lw=1.5)); fig.text(0.121, I(yk), "SWOT-GAT", fontsize=FS, color="#1f66c1", fontweight="bold", va="baseline")
fig.add_artist(Line2D([0.215, 0.255], [I(yk + 0.03)]*2, transform=fig.transFigure, color=jw.INK, lw=1.15, ls=LSD))
fig.text(0.261, I(yk), "rule-model ensemble: daily mean of GDROM, ISTARF and MODROM-cal storage · colour = main purpose", fontsize=FS, color=jw.INK2, va="baseline")
STEM = "c5d_fig2"; H = K.save(fig, STEM); json.dump(dict(NUM, qa=H), open(K.OUT/f"num_{STEM}.json", "w"), indent=1, default=float)
print(STEM, "qa issues:", len(H), H); print(json.dumps({k: v for k, v in NUM.items() if k != "a"}, default=float, indent=1)); print({k: v for k, v in NUM["a"].items() if k != "ids"})
