"""Figure 1 of the paper: daily storage hydrographs at eight reservoirs (figure set version d).

Each reservoir's hindcast is drawn beside its SWOT Period.

What each panel shows
  The figure has eight rows, one reservoir each, and two panels per row.
    rows a-d  left: the 2010-2020 hindcast. right: the SWOT Period (axis July 2023 to June 2026).
              Row a is the pick stored in score/picks_d.json (in the paper's run: Hyrum, UT). It replaces
              Black Butte, one of the reservoirs excluded in excl_d.py. Rows b-d are Norfork, Rathbun and
              Trinity (DAM_ID 1042, 924, 131).
    rows e-h  left: the full 1981-2020 hindcast. right: the SWOT Period.
              The reservoirs are the picks stored in score/picks.json: the reservoirs nearest the 90th,
              60th, 35th and 10th percentile of SWOT-GAT's 1981-2020 NSE, among those with a nearly
              complete gauge record and SWOT data (rules in 07_evaluation/c_scoring/full_record/picks.py).
  Every panel shows storage anomaly in million cubic meters (MCM):
    - the gauge (black) and SWOT-GAT (blue). SWOT-GAT is the mean of the two seeds, summed over the
      reservoir's member lake nodes on days when every member has a value.
    - the rule models MODROM-cal, GDROM and ISTARF as thin pale lines. A rule line is drawn only when it
      has more than 300 daily values in the hindcast window (more than 100 in the SWOT Period).
    - right panels only: SWOT observations as open dark-blue rings. SWOT gives storage change, not
      absolute storage, so the observations are shifted by the median of their difference from the gauge
      anomaly on the pass dates. Observations beyond the y range are drawn as triangles at the panel edge.
  Each model series is centered on its mean over the days it shares with the gauge (the gauge on its mean
  over the days it shares with SWOT-GAT). Lines are 7-day centered running means, for display only. All
  scores use the daily values.
  The right panels cover only each reservoir's SWOT span (first to last SWOT pass, ending no later than
  12 June 2026, where the rule-model runs end) and only the days on which every rule model that has the
  reservoir has a value.
  Above each left panel: name, state, main purpose, capacity and ecoregion, and SWOT-GAT's NSE with its
  percentile ("pct"), nRMSE and VR (variability ratio = model standard deviation / gauge standard
  deviation). Above each right panel: SWOT-GAT's NSE, nRMSE and VR in the SWOT Period.
  The percentile ranks the reservoir's hindcast NSE: in rows a-d among the reservoirs that all four models
  score in 2010-2020, without the excluded reservoirs (301 in the paper's run); in rows e-h among every
  reservoir where SWOT-GAT is scored over 1981-2020 (419 in the paper's run).

Reads
  "Kit folder" is the folder of this script; "figure-set folder" is its parent. Other paths are relative to
  CONUS_storage_fork/ under $SWOTGAT_WORK.
  Kit folder:
    jw_fig_tt_e.py                           its data-loading code is run from here (see Method)
    sites_2010_g.csv, sites_2325span_g.csv   kit score tables; they give the 2010-2020 percentile set
    eco_l1_sites.csv                         ecoregion group of each reservoir
    excl_d.py                                the excluded reservoirs
  Figure-set folder:
    score/picks_d.json, score/picks.json     the reservoirs of row a and of rows e-h
    score/score_full/fullrecord_site_window.csv   full-record scores (window 1981-2020)
  Series, loaded by jw_fig_tt_e.py's code:
    predictions/<run>/<step>_<tag>.parquet   SWOT-GAT predictions of the runs in IP_RUNS. Tags: IP_HIND
                                             for 2010-2020; x8085, x8691, x9297, x9803, x0409 and h10 for
                                             1981-2020; x26 for the SWOT Period
    predictions/grades_<rule>_g_y1980/full_series.parquet   rule-model runs started in 1980
    predictions/grades_<rule>_g26h0/full_series.parquet     rule-model runs of the SWOT Period
                                             (<rule> = gdrom3c, istarf3c, modromcal3c)
    testsets/regime3_full_1980_2020/         gauge storage 1980-2020 (sites.csv, truth_clean.parquet)
    testsets/regime1x_2023_2026/             gauge storage 2023-2026 (sites.csv, truth_clean_v1x.parquet)
    eval/score2026/spans_2023_2026.csv       each reservoir's SWOT span
    eval/half_vd_review/lakesp_vD_halfexact_graph.parquet   SWOT lake storage after the HALF outlier filter
    eval/grl_figs/sites_v5_g.csv             reservoir name, state, main purpose and capacity (a table
                                             from an earlier model configuration; only these columns are used)
  Reference values for the checks:
    eval/overlap50/score/sites_2010_g.csv, sites_2325span_g.csv   scorer tables
    eval/overlap50/fig1_o50.json             the reservoir sets all four models score

Writes (kit folder)
  c5d_fig1.pdf, c5d_fig1.png (200 dpi)
  num_c5d_fig1.json  every number drawn: the reservoir ids of both row groups; the sizes of the two
                     percentile sets; and per reservoir the row letter, window, name, state, purpose,
                     capacity, ecoregion, hindcast NSE, nRMSE, VR and percentile, SWOT Period NSE, nRMSE
                     and VR, the number of hindcast days scored and the number of SWOT passes; plus the
                     messages of the layout check ("qa")

Usage
  source 07_evaluation/c_scoring/env.sh; bin/run 08_figures/fig1_c5d.py     (after prepare_kit.py)
  No arguments. Through ipcfg.py the script reads IP_RUNS, IP_STEPS and IP_HIND, which env.sh sets.

Method
  - Loading. The script reads jw_fig_tt_e.py as text and executes the part before that script's own
    figure, once per row group, after replacing its reservoir list. For rows e-h it also replaces the
    window by 1981-2020 and the prediction tags by the six listed above. The four text lines it searches
    for must each occur exactly once in jw_fig_tt_e.py, so that file must not be edited.
  - Scores. NSE, nRMSE and VR are recomputed here from the daily series that are drawn (scores() in
    jw_fig_tt_e.py: both series centered on their shared days; nRMSE = RMSE / gauge range; VR = model
    sd / gauge sd; NSE = 2*VR*r - VR^2, the NSE of the centered series). Each value is asserted equal to
    the scorers' tables: rows a-d hindcast against eval/overlap50/score/sites_2010_g.csv, rows e-h
    hindcast against score/score_full (window 1981-2020), right panels against
    eval/overlap50/score/sites_2325span_g.csv.
  - The tolerance of these checks is 1e-6 because the predictions are float32. The figure's numpy
    arithmetic stays in float32, while the scorers' pandas arithmetic is float64 (the SWOT Period scorer
    also sums SWOT-GAT in float32). Printed values have two decimals.
  - Also asserted: the percentiles of rows e-h equal those stored in picks.json, and the 2010-2020
    percentile set has the size of the "2010-20" set of fig1_o50.json without the excluded reservoirs.
  - y range of a row: the 0.2nd to 99.8th percentile of the gauge and SWOT-GAT values of both panels,
    widened to the 2nd-98th percentile of each smoothed rule line, plus 8% padding.
  - Line widths: the 40-year panels use thinner lines (gauge and SWOT-GAT 0.6 pt, rule models 0.35 pt)
    than the other panels (1.3 pt and 0.6 pt).
  - All text is 7 pt. jw.qa() reports smaller text, overlapping text and anything off the canvas.

Notes
  - In the tables and in the code, the labels `v10pm` and `SWOT-GNN` mean SWOT-GAT (the paper's model:
    mean of two seeds).
  - Only reservoirs whose dam, gauge and graph node are linked with High or Medium confidence are used.
  - Only SWOT-GAT's scores are printed. The rule models appear as lines only.
"""
from swotgat.paths import WORK
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np, pandas as pd, matplotlib as mpl
from pathlib import Path
OUTD = Path(__file__).resolve().parent; B = OUTD.parent; O50 = Path(f"{WORK}/CONUS_storage_fork/eval/overlap50")
EX1 = [int(json.load(open(B/"score/picks_d.json"))["pick"][0]["DAM_ID"]), 1042, 924, 131]   # rows a-d; row a's pick replaces Black Butte, where ISTARF is pinned at full or empty (excl_d.py)
PK = json.load(open(B/"score/picks.json")); EX2 = [int(PK["targets"][t][0]["DAM_ID"]) for t in ("90", "60", "35", "10")]
_src = (OUTD/"jw_fig_tt_e.py").read_text(); _MARK = "fig = plt.figure(figsize=(jw.COL2, 7.75))"; _EX = "EX = [144, 1275, 1042, 924, 131, 965, 601, 1035]"
_W = 'W0, W1 = pd.Timestamp("2010-01-01"), pd.Timestamp("2020-12-31")'; _CH = '    for c in __import__("ipcfg").HIND:\n'
for t_ in (_MARK, _EX, _W, _CH): assert _src.count(t_) == 1, t_
HEAD = _src.split(_MARK)[0]
def load(ex, full):
    """Run jw_fig_tt_e.py's loading code for the reservoirs `ex` and return its namespace.

    ex is a list of DAM_IDs. full=True switches the window to 1981-2020 and the prediction files to the
    six that cover it; full=False keeps the 2010-2020 window. The namespace is a dict of every name the
    loading code defines (loaders, score helpers, tables).
    """
    s = HEAD.replace(_EX, f"EX = {ex}")
    if full: s = s.replace(_W, 'W0, W1 = pd.Timestamp("1981-01-01"), pd.Timestamp("2020-12-31")').replace(_CH, '    for c in ["x8085", "x8691", "x9297", "x9803", "x0409", "h10"]:\n')
    ns = {"__file__": str(OUTD/"jw_fig_tt_e.py"), "__name__": "jw_fig_tt_e_head"}; exec(compile(s, f"jw_fig_tt_e.py (head, {'1981-2020' if full else '2010-2020'})", "exec"), ns); return ns
NS1, NS2 = load(EX1, False), load(EX2, True)
jw, pe = NS1["jw"], NS1["pe"]; plt = jw.plt
FS = 7.0; mpl.rcParams.update({"xtick.labelsize": FS, "ytick.labelsize": FS, "axes.labelsize": FS})
# ---------------- reference tables (asserts) and the 1981-2020 percentile
HM = ["High", "Medium"]; S10 = pd.read_csv(O50/"score/sites_2010_g.csv"); S10 = S10[(S10.MODEL == "v10pm") & S10.tier.isin(HM) & S10.nrmse.notna()].set_index("DAM_ID")
S26 = pd.read_csv(O50/"score/sites_2325span_g.csv"); S26 = S26[(S26.MODEL == "v10pm") & S26.tier.isin(HM) & S26.nrmse.notna()].set_index("DAM_ID")
FR = pd.read_csv(B/"score/score_full/fullrecord_site_window.csv"); FR = FR[(FR.window == "1981-2020") & FR.site_tier.isin(HM) & (FR.model == "SWOT-GNN")].set_index("DAM_ID")
# PCT2: percentile of SWOT-GAT's 1981-2020 NSE. N305 and N419 are the sizes of the two percentile sets
# (2010-2020 four-model set without the excluded reservoirs; all 1981-2020 reservoirs). The names keep
# earlier counts: in the paper's run the sizes are 301 and 419.
PCT2 = 100*FR.nse.rank(pct=True); N305 = len([i for i in json.load(open(O50/"fig1_o50.json"))["2010-20"]["ids"] if i not in __import__("excl_d").EXCL]); N419 = len(FR)
assert all(abs(PCT2[d] - PK["targets"][t][0]["pct"]) < 1e-9 for d, t in zip(EX2, ("90", "60", "35", "10")))
FH = 8.76; fig = plt.figure(figsize=(jw.COL2, FH)); I = lambda inch: inch/FH; RING = "#0d366b"
L0, GAPX, WL, WR = 0.085, 0.022, 0.62, 0.27; ROW_H, GAPY = 0.54, 0.42; TOPS = (7.90, 3.65)
TR_ = lambda a: mpl.transforms.blended_transform_factory(a.transData, a.transAxes)
out = {}
def row(ns, dam, j, y0, full, last, letter):
    """Draw one reservoir's row, check its scores and store its numbers in `out`.

    The hindcast panel aL is on the left and the SWOT Period panel aR on the right. ns is the namespace
    returned by load(); dam is the DAM_ID; j is the row's index within its group (not used); y0 is the
    bottom of the row in inches from the figure bottom; full is True for the 1981-2020 window; last is
    True for the last row of a group, which gets the x tick labels; letter is the row letter.
    SWOT-GAT's NSE, nRMSE and VR are asserted equal to the scorers' tables. Returns (aL, aR).
    """
    hind, x26_pair, span26, rules_ok26, rule26, swot26, scores, sm, tint = (ns[k] for k in ("hind", "x26_pair", "span26", "rules_ok26", "rule26", "swot26", "scores", "sm", "tint"))
    RL, RCLR, T26, LINEB, SWOT_RING, meta, USE, ECO, ECN, ordn = (ns[k] for k in ("RL", "RCLR", "T26", "LINEB", "SWOT_RING", "meta", "USE", "ECO", "ECN", "ordn"))
    aL = fig.add_axes([L0, I(y0), WL, I(ROW_H)]); aR = fig.add_axes([L0 + WL + GAPX, I(y0), WR, I(ROW_H)])
    d = hind(dam); o1, p1 = d.gauge.values, d["SWOT-GNN"].values; k = np.isfinite(o1) & np.isfinite(p1)
    o1, p1 = o1 - np.nanmean(o1[k]), p1 - np.nanmean(p1[k])
    t2 = T26; o2, p2 = x26_pair(dam); span = span26(dam) & rules_ok26(dam)
    o2, p2 = np.where(span, o2, np.nan), np.where(span, p2, np.nan); k2 = np.isfinite(o2) & np.isfinite(p2)
    o2, p2 = o2 - np.nanmean(o2[k2]), p2 - np.nanmean(p2[k2])
    extra = []
    for m in RL:                                                  # faint rule-model lines, centered like SWOT-GAT in each period
        r1 = d[m].values if m in d else None
        if r1 is not None and np.isfinite(r1).sum() > 300:
            kk = np.isfinite(o1) & np.isfinite(r1); r1 = r1 - np.nanmean(r1[kk]); aL.plot(d.index, sm(r1), color=tint(RCLR[m]), lw=0.35 if full else 0.6, zorder=2); extra.append(r1)
        o3, r2 = rule26(m, dam)
        if r2 is not None: o3, r2 = np.where(span, o3, np.nan), np.where(span, r2, np.nan)
        if r2 is not None and np.isfinite(r2).sum() > 100:
            kk = np.isfinite(o3) & np.isfinite(r2); r2 = r2 - np.nanmean(r2[kk]); aR.plot(t2, sm(r2), color=tint(RCLR[m]), lw=0.6, zorder=2); extra.append(r2)
    lwm = 0.6 if full else 1.3                                    # thinner lines in the compressed 40-year panel
    for a_, t, o, p, lw in ((aL, d.index, o1, p1, lwm), (aR, t2, o2, p2, 1.3)):
        halo = [pe.Stroke(linewidth=lw + (0.6 if lw < 1 else 1.1), foreground="white"), pe.Normal()]
        a_.plot(t, sm(o), color=jw.INK, lw=lw, zorder=4, path_effects=halo); a_.plot(t, sm(p), color=LINEB, lw=lw, zorder=5, path_effects=halo)
    sw_ = swot26(dam); swp = None
    if sw_ is not None and len(sw_):
        ts = sw_.index; gi = pd.Series(o2, index=t2).reindex(ts).values; kk = np.isfinite(gi)
        if kk.sum(): swp = (ts, sw_.values - np.median(sw_.values[kk] - gi[kk]))
    allv = np.r_[o1[np.isfinite(o1)], p1[np.isfinite(p1)], o2[np.isfinite(o2)], p2[np.isfinite(p2)]]; lo_, hi_ = np.percentile(allv, [0.2, 99.8])
    for e_ in extra:
        if np.isfinite(e_).sum() > 50: q_ = np.nanpercentile(sm(e_), [2, 98]); lo_, hi_ = min(lo_, q_[0]), max(hi_, q_[1])
    pad = 0.08*(hi_ - lo_)
    for a_ in (aL, aR): a_.set_ylim(lo_ - pad, hi_ + pad); a_.tick_params(axis="both", labelsize=FS, length=1.8, pad=1)
    if swp is not None:
        y0_, y1_ = lo_ - pad, hi_ + pad; v_ = swp[1]; inr = (v_ >= y0_) & (v_ <= y1_); hm, lm = v_ > y1_, v_ < y0_
        aR.scatter(swp[0][inr], v_[inr], s=8, facecolor="none", edgecolor=RING, lw=0.6, zorder=6)
        if hm.any(): aR.scatter(swp[0][hm], np.full(hm.sum(), y1_ - 0.05*(y1_ - y0_)), s=11, marker="^", color=RING, lw=0, zorder=7)
        if lm.any(): aR.scatter(swp[0][lm], np.full(lm.sum(), y0_ + 0.05*(y1_ - y0_)), s=11, marker="v", color=RING, lw=0, zorder=7)
    if full:
        aL.set_xlim(pd.Timestamp("1981-01-01"), pd.Timestamp("2020-12-31")); XT = pd.date_range("1985-01-01", "2020-01-01", freq="5YS"); aL.set_xticks(XT); aL.set_xticklabels([])
    else:
        aL.set_xlim(pd.Timestamp("2010-01-01"), pd.Timestamp("2020-12-31")); aL.set_xticks(pd.date_range("2010-01-01", "2020-01-01", freq="YS")); aL.set_xticklabels([])
    aR.set_xlim(pd.Timestamp("2023-07-01"), pd.Timestamp("2026-07-01")); RT = pd.to_datetime(["2023-07-01", "2024-07-01", "2025-07-01"]); aR.set_xticks(RT); aR.set_xticklabels([])
    if last:
        if full:
            for t_ in XT[:-1]: aL.text(t_, -0.10, str(t_.year), transform=TR_(aL), fontsize=FS, ha="center", va="top")   # 2020 tick unlabelled: it would meet 'Jul 2023'
        else:
            for yy in range(2010, 2021): aL.text(pd.Timestamp(f"{yy}-07-01"), -0.10, str(yy), transform=TR_(aL), fontsize=FS, ha="center", va="top")
        for t_, l_ in zip(RT, ["Jul 2023", "Jul 2024", "Jul 2025"]): aR.text(t_, -0.10, l_, transform=TR_(aR), fontsize=FS, ha="center", va="top")
    aL.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3, min_n_ticks=3))
    aL.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v/1000:g}k" if abs(v) >= 1000 else f"{v:g}"))
    aR.set_yticks([]); aR.spines["left"].set_visible(False)
    r = meta.loc[dam]; nm = ns["K"]._clean_name(str(r["name"])).replace(" Reservoir", "").replace(" Dam And", "").replace(" Dam", ""); st = ns["K"].ST.get(str(r.state), str(r.state))
    jw.seq_text_ax(aL, 0.0, 1.0 + 0.23/ROW_H, [(f"{nm}, {st}", jw.INK, "bold"), (f"{USE.get(str(r.use), str(r.use).lower())} · {r.cap:,.0f} MCM · {ECN.get(ECO.get(dam), '')}", jw.INK2, "normal")], fs=FS, gap=0.012)
    (n1, e1, v1), (n2, e2, v2) = scores(o1, p1), scores(o2, p2)
    pct = float(PCT2[dam]) if full else float(ns["PCT"][dam])
    ref1 = FR.loc[dam] if full else S10.loc[dam]; ref2 = S26.loc[dam]
    for got, want in ((e1, ref1.nse), (n1, ref1.nrmse), (v1, ref1.amp)): assert abs(got - want) < 1e-6, (dam, got, want)   # float32 predictions: the figure's numpy arithmetic stays float32
    for got, want in ((e2, ref2.nse), (n2, ref2.nrmse), (v2, ref2.amp)): assert abs(got - want) < 1e-6, (dam, got, want)   # the SWOT Period scorer also sums SWOT-GAT in float32
    out[int(dam)] = dict(row=letter, window="1981-2020" if full else "2010-2020", name=nm, state=st, use=str(r.use), cap=float(r.cap), eco=str(ECO.get(dam)),
                         nse_hind=float(e1), nrmse_hind=float(n1), vr_hind=float(v1), pct=pct, nse_swot=float(e2), nrmse_swot=float(n2), vr_swot=float(v2),
                         n_days_hind=int(k.sum()), n_swot_passes=0 if sw_ is None else int(len(sw_)))
    fe = lambda v: f"{v:.2f}".replace("-", "−")
    aL.text(1.0, 1.0 + 0.04/ROW_H, f"NSE {fe(e1)} ({ordn(int(round(pct)))} pct) · nRMSE {100*n1:.0f}% · VR {v1:.2f}", transform=aL.transAxes, fontsize=FS, color="#1f66c1", ha="right", va="bottom", fontweight="bold")
    aR.text(1.0, 1.0 + 0.04/ROW_H, f"NSE {fe(e2)} · nRMSE {100*n2:.0f}% · VR {v2:.2f}", transform=aR.transAxes, fontsize=FS, color=jw.BLUE8[7], ha="right", va="bottom", fontweight="bold")
    jw.letter(aL, letter, x=-0.075, y=1.0 + 0.21/ROW_H)
    return aL, aR
for g, (ns, ex, full, top) in enumerate(((NS1, EX1, False, TOPS[0]), (NS2, EX2, True, TOPS[1]))):
    for j, dam in enumerate(ex):
        y0 = top - (j + 1)*ROW_H - j*GAPY; aL, aR = row(ns, dam, j, y0, full, j == len(ex) - 1, "abcdefgh"[4*g + j])
        if j == 0:
            aL.text(0.5, 1.0 + 0.47/ROW_H, "1981–2020 hindcast (no SWOT)" if full else "2010–2020 hindcast (no SWOT)", transform=aL.transAxes, fontsize=FS, color=jw.INK, ha="center", va="bottom", fontweight="bold")
            aR.text(0.5, 1.0 + 0.47/ROW_H, "SWOT Period, Jul 2023–Jun 2026", transform=aR.transAxes, fontsize=FS, color=jw.INK, ha="center", va="bottom", fontweight="bold")
    fig.text(0.012, I(top - (4*ROW_H + 3*GAPY)/2), "storage anomaly (MCM)", rotation=90, fontsize=FS, va="center", ha="left")
from matplotlib.lines import Line2D                          # key: a line sample (or a ring) in the drawn color, then the name in a matching color
tint, RCLR = NS1["tint"], NS1["RCLR"]; x = L0; yk = I(FH - 0.16)
for lab, col, fw, smp in (("gauge", jw.INK, "bold", ("line", jw.INK, 1.3)), ("SWOT-GAT", "#1f66c1", "bold", ("line", NS1["LINEB"], 1.3)),
                          ("MODROM-cal", "#a87200", "normal", ("line", tint(RCLR["MODROM-cal"]), 1.1)), ("GDROM", "#c9501f", "normal", ("line", tint(RCLR["GDROM"]), 1.1)),
                          ("ISTARF", "#11875c", "normal", ("line", tint(RCLR["ISTARF"]), 1.1)), ("SWOT observations", RING, "normal", ("ring", RING, 0.7))):
    if smp[0] == "line": fig.add_artist(Line2D([x, x + 0.024], [yk + I(0.032)]*2, transform=fig.transFigure, color=smp[1], lw=smp[2], solid_capstyle="butt"))
    else: fig.add_artist(Line2D([x + 0.012], [yk + I(0.032)], transform=fig.transFigure, marker="o", ms=3.4, mfc="none", mec=smp[1], mew=smp[2], ls="none"))
    t = fig.text(x + 0.030, yk, lab, fontsize=FS, color=col, fontweight=fw, va="baseline"); fig.canvas.draw()
    x = t.get_window_extent().transformed(fig.transFigure.inverted()).x1 + 0.020
assert x < 1.0, x
assert len(NS1["PCT"]) == N305; stem = "c5d_fig1"; Hq = jw.qa(fig, stem, min_font=FS); fig.savefig(OUTD/f"{stem}.pdf"); fig.savefig(OUTD/f"{stem}.png", dpi=200)
json.dump(dict(rows_2010=EX1, rows_full=EX2, n_pct_2010=N305, n_pct_full=N419, series=out, qa=Hq), open(OUTD/f"num_{stem}.json", "w"), indent=1)
print(stem, "qa issues:", len(Hq), Hq); print(json.dumps(out, indent=1))
