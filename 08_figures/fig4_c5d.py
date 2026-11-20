"""Figure 4 of the paper: the 2010-2020 hindcast of the four models by main reservoir purpose.

Figure set version d.

What each panel shows
  a  NSE of SWOT-GAT, MODROM-cal, GDROM and ISTARF, four boxes side by side for each main purpose (flood
     control, irrigation, hydropower, water supply, recreation). Boxes: 25th to 75th percentile, black
     line at the median, whiskers from the 5th to the 95th percentile, all clipped at the axis limits.
  b  The same for the variability ratio (VR = model standard deviation / gauge standard deviation; the
     tables call it `amp`), on a base-2 logarithmic axis with a guide line at 1 (kit_h.box_axis "amp").
  c  Stacked bars: the share of reservoirs where each model has the highest NSE. The first bar is for all
     reservoirs, the others for each purpose. Shares of 7% or more are printed in the bar.
  Reservoirs in all three panels: the 2010-2020 four-model set, that is, the reservoirs all four models
  score in 2010-2020, without the reservoirs excluded in excl_d.py (301 in the paper's run). Reservoirs
  with another main purpose count only in the "All" bar; the footnote gives their number.

Reads
  "Kit folder" is the folder of this script. Other paths are relative to CONUS_storage_fork/ under
  $SWOTGAT_WORK.
  Kit folder:
    sites_2010_g.csv                        2010-2020 scores per reservoir and model (written by
                                            prepare_kit.py; loaded through kit_h.py as K.S10)
    and the other files kit_h.py loads when it is imported (see kit_h.py)
  eval/overlap50/fig1_o50.json              reference: the reservoir sets all four models score
  eval/overlap50/score/sites_2010_g.csv     reference: the scorer's own 2010-2020 table

Writes (kit folder)
  c5d_fig4.pdf, c5d_fig4.png (200 dpi)
  num_c5d_fig4.json  every number drawn:
    "nse", "amp"  per purpose group ("0" to "4", in the order above) each model's median and its 5th,
                  25th, 75th and 95th percentile, and n
    "c"           per bar n and each model's share
    "c_same_winner_lowest_nrmse"  the share of reservoirs where the model with the highest NSE also has the
                  lowest nRMSE
    "best"        per purpose, the model with the highest median NSE and the model whose median VR is
                  closest to 1 (on a logarithmic scale)
    "qa"          messages of the layout check

Usage
  source 07_evaluation/c_scoring/env.sh; bin/run 08_figures/fig4_c5d.py     (after prepare_kit.py)
  No arguments.

Method
  - Scores come from the stage 07 scorer (score_2010.py) and are not recomputed here. Only reservoirs
    with High or Medium link confidence and a valid nRMSE are used.
  - Checks (assertions): the reservoir set equals the "2010-20" set of fig1_o50.json without the excluded
    reservoirs; every group median equals the median computed from the scorer's own table to 1e-12; the
    best model per reservoir is the same in both tables; every reservoir of the set has an NSE for all
    four models.

Notes
  In the tables the model label `v10pm` means SWOT-GAT (the paper's model: mean of two seeds). The name
  K.IDS314 keeps an earlier count.
"""
from swotgat.paths import WORK
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np, pandas as pd, matplotlib as mpl
from pathlib import Path
import kit_h as K
from kit_h import jw, plt, FS, M4, CLR, TXT, LAB, HL
O50 = Path(f"{WORK}/CONUS_storage_fork/eval/overlap50"); J50 = json.load(open(O50/"fig1_o50.json"))
ids = K.IDS314; assert ids == [i for i in J50["2010-20"]["ids"] if i not in __import__("excl_d").EXCL]; U = K.S10[K.S10.MODEL == "v10pm"].set_index("DAM_ID").use.reindex(ids)
R10 = pd.read_csv(O50/"score/sites_2010_g.csv"); R10 = R10[R10.tier.isin(["High", "Medium"]) & R10.nrmse.notna() & R10.DAM_ID.isin(ids)]
PURP = [("Flood control", "flood control"), ("Irrigation", "irrigation"), ("Hydroelectricity", "hydropower"), ("Water supply", "water supply"), ("Recreation", "recreation")]
GP = [list(U[U == u].index) for u, _ in PURP]; LP = [f"{l}\nn = {len(g)}" for (_, l), g in zip(PURP, GP)]
P = {f: K.S10[K.S10.DAM_ID.isin(ids)].pivot(index="DAM_ID", columns="MODEL", values=f)[M4] for f in ("nse", "amp")}
FH = 5.60; fig = plt.figure(figsize=(jw.COL2, FH)); I = lambda inch: inch/FH; NUM = {}; BEST = {}
for i, (f, x0) in enumerate((("nse", 0.075), ("amp", 0.575))):
    a = fig.add_axes([x0, I(3.15), 0.41, I(1.70)]); NUM[f] = K.model_boxes(a, P[f], GP, LP, f)
    RP = R10.pivot(index="DAM_ID", columns="MODEL", values=f)
    for gi in range(len(GP)):
        med = {m: NUM[f][str(gi)][m] for m in M4}
        b = max(M4, key=lambda m: med[m]) if f == "nse" else min(M4, key=lambda m: abs(np.log(med[m])))   # best: highest NSE / VR closest to 1
        BEST.setdefault(f, {})[PURP[gi][1]] = LAB[b]
        assert all(abs(med[m] - float(RP.loc[GP[gi], m].median())) < 1e-12 for m in M4), (f, gi)
    jw.letter(a, "ab"[i], x=-0.13, y=1.03)
# ---------------- (c) share of reservoirs where each model has the highest NSE
W = P["nse"].idxmax(axis=1); assert (P["nse"].notna().all(axis=1)).all()
NUM["c_same_winner_lowest_nrmse"] = float((W == K.S10[K.S10.DAM_ID.isin(ids)].pivot(index="DAM_ID", columns="MODEL", values="nrmse")[M4].idxmin(axis=1)).mean())
ROWS = [(f"All {len(ids)} reservoirs", list(ids))] + [(f"{l}  ({len(g)})", g) for (l, _), g in zip(PURP, GP)]
ac = fig.add_axes([0.205, I(0.62), 0.775, I(1.62)]); NUM["c"] = {}
for r_, (lab, g) in enumerate(ROWS):
    yy = r_ + (0.35 if r_ else 0.0); sh = {m: float((W.loc[g] == m).mean()) for m in M4}; NUM["c"][lab] = dict(n=len(g), share=sh); left = 0.0
    for m in M4:
        ac.barh(yy, sh[m], left=left, height=0.62, color=CLR[m], edgecolor="white", lw=0.6)
        if sh[m] >= 0.07: ac.text(left + sh[m]/2, yy, f"{100*sh[m]:.0f}%", fontsize=FS, color="white" if m == "v10pm" else jw.INK, ha="center", va="center",
                                  fontweight="bold" if m == "v10pm" else "normal")
        left += sh[m]
ac.set_yticks([r_ + (0.35 if r_ else 0.0) for r_ in range(len(ROWS))]); ac.set_yticklabels([r[0] for r in ROWS]); ac.tick_params(axis="y", length=0)
ac.get_yticklabels()[0].set_fontweight("bold")
ac.set_ylim(len(ROWS) - 0.3, -0.5); ac.set_xlim(0, 1); ac.set_xticks([0, 0.25, 0.5, 0.75, 1]); ac.set_xticklabels(["0", "25", "50", "75", "100%"])
ac.spines["left"].set_visible(False); ac.set_xlabel("share of reservoirs where the model has the highest NSE")
ac.text(0.0, 1.10, "Which model scores best", transform=ac.transAxes, fontsize=FS, fontweight="bold", va="bottom")
jw.letter(ac, "c", x=-0.215, y=1.10)
K.model_key(fig, 0.075, I(5.45))
fig.text(0.075, I(5.32), f"{HL} · the {len(ids)} reservoirs all four models score · by main purpose", fontsize=FS, color=jw.INK2, va="baseline")
fig.text(0.075, I(0.05), f"boxes: 25th–75th percentile, black line median, whiskers 5th–95th, drawn at the axis limit beyond it · not in a purpose group: {len(ids) - sum(len(g) for g in GP)}",
         fontsize=FS, color=jw.INK2, va="baseline")
Wt = R10.pivot(index="DAM_ID", columns="MODEL", values="nse")[M4].reindex(ids).idxmax(axis=1); assert (Wt == W).all()
H = K.save(fig, "c5d_fig4"); json.dump(dict(NUM, best=BEST, qa=H), open(K.OUT/"num_c5d_fig4.json", "w"), indent=1)
print("c5d_fig4 qa issues:", len(H), H); print(json.dumps(BEST)); print(json.dumps({k: {LAB[m]: round(v["share"][m], 3) for m in M4} for k, v in NUM["c"].items()}))
print(json.dumps({f: {str(g): {LAB[m]: round(NUM[f][str(g)][m], 3) for m in M4} | {"n": NUM[f][str(g)]["n"]} for g in range(len(GP))} for f in ("nse", "amp")}))
