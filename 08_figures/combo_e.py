"""Shared data section for the figure kit (and, below the split point, an earlier combined figure).

kit_h.py reads this file as text, keeps everything above the comment line that starts the "(a-d, v1)" part,
forces MODE to "v2" and executes it. That first part builds the tables every figure script uses:

  S_, HK_   the site score table (jw.sites_table: 2010-2020 and SWOT Period rows) and its hindcast period key
  SR        SWOT-GAT's 2010-2020 rows with coordinates, for maps
  EG, EN, EC, E9   Level I ecoregion groups, their names and colours, and each reservoir's group
  x9, EDG, b9      the irregularity of each gauge record (1981-2020) and its thirds
  HL_, HS_, SUF_   further values returned by jw.sites_table; mix() lightens a colour

Reads (under $SWOTGAT_WORK/CONUS_storage_fork/eval): perf_attrib/out/vars.parquet and ops_8120.parquet, the
kit folder's sites_2010_g.csv, sites_2325span_g.csv, eco_l1_sites.csv and outline files, grl_figs/sites_v5_g.csv.

The panel functions defined here (panel_e, panel_g, panel_f) and all code after the split point draw a
combined figure of an earlier figure set. The paper's four figures do not call them. Do not change the
`MODE = ...` line or the "(a-d, v1)" comment line: kit_h.py searches for both.
"""
from swotgat.paths import WORK
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np, pandas as pd, matplotlib as mpl
from pathlib import Path
MODE = os.environ.get("MODE", "v1"); OUTD = Path(__file__).resolve().parent
if MODE == "v1":                                       # Fig 5's loading code (everything before its figure), EX = the four reservoirs
    _src = (OUTD/"jw_fig_tt_e.py").read_text(); _MARK = "fig = plt.figure(figsize=(jw.COL2, 7.75))"; _EX = "EX = [144, 1275, 1042, 924, 131, 965, 601, 1035]"
    assert _src.count(_MARK) == 1 and _src.count(_EX) == 1
    exec(compile(_src.split(_MARK)[0].replace(_EX, "EX = [131, 924, 1042, 144]"), "jw_fig_tt_e.py (head)", "exec"))
import jw, grl_common as G
from jw import plt
import matplotlib.patheffects as pe
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, Polygon as MplPolygon, Rectangle
from matplotlib.colors import to_rgb
jw.use_style(); F = Path(f"{WORK}/CONUS_storage_fork"); PA = F/"eval/perf_attrib/out"; NUM = {}
S_, HK_, HL_, HS_, SUF_ = jw.sites_table("2010"); S_ = S_[S_.tier.isin(["High", "Medium"]) & S_.nrmse.notna()].copy(); assert S_.nse.notna().all()   # c5_o50: the direct NSE column
MODELS = ["SWOT-GNN", "MODROM-cal", "GDROM", "ISTARF"]; MK = {"SWOT-GNN": "v10pm", "MODROM-cal": "MODROM-cal", "GDROM": "GDROM", "ISTARF": "ISTARF"}
CLR = {"SWOT-GNN": "#2a78d6", "GDROM": "#eb6834", "ISTARF": "#1baf7a", "MODROM-cal": "#eda100"}
TXT = {"SWOT-GNN": "#1f66c1", "MODROM-cal": "#a87200", "GDROM": "#c9501f", "ISTARF": "#11875c"}
LAB = {"SWOT-GNN": "SWOT-GAT", "MODROM-cal": "MODROM-cal", "GDROM": "GDROM", "ISTARF": "ISTARF"}
TRAIN = "#f6e6cc"; TRAIN_T = "#a8641a"
mix = lambda c, f: tuple(np.array(to_rgb(c)) + (1 - np.array(to_rgb(c)))*f)
# ---------------- (e) data: Fig 6d's block medians + the SWOT-span point
FR = PANEL = SP = SPAN_IDS = WINS = None   # c5_o50: the 5-year-block thru panel is in none of the four pages
yc = lambda w: (int(w[:4]) + int(w[-4:]))/2
def panel_e(ax):
    """Draw the four models' median NSE through time: 5-year blocks (lines) and the SWOT span (diamonds).

    Not used by the paper's figures. It needs the block medians BM and the span medians SPM, which this
    file does not define, and WINS, PANEL and SPAN_IDS, which are None, so it cannot be called as the
    code stands.
    """
    t0, t1 = 2023.5, 2026 + 5.4/12
    ax.axvspan(t0, t1, color=TRAIN, lw=0, zorder=0)
    ax.text((t0 + t1)/2, 0.38, "SWOT training period", transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes), rotation=90,
            ha="center", va="bottom", fontsize=6.0, color=TRAIN_T)
    ax.axhline(0, color=jw.INK3, lw=0.5, ls=(0, (2, 2)), zorder=0.5)
    for m in MODELS[::-1]:
        g = m == "SWOT-GNN"; y = BM.xs(m, level="model").reindex(WINS).values
        ax.plot([yc(w) for w in WINS], y, color=CLR[m], lw=1.5 if g else 0.85, marker="o", ms=2.6 if g else 1.8, mec="white", mew=0.3, zorder=4 if g else 3)
        ax.plot([(t0 + t1)/2], [SPM[m]], marker="D", ms=3.6 if g else 2.8, color=CLR[m], mec="white", mew=0.4, ls="none", zorder=5 if g else 4)
    ax.text((t0 + t1)/2, 0.03, f"n = {len(SPAN_IDS)}", transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes), ha="center", va="bottom",
            fontsize=6.0, color=TRAIN_T)
    ax.set_xlim(1980, 2027.5); ax.set_xticks([1980, 1990, 2000, 2010, 2020]); ax.set_ylim(-0.15, 0.85); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_ylabel("median NSE")
    jw.seq_text_ax(ax, 0.0, 1.13, [(LAB[m], TXT[m], "bold" if m == "SWOT-GNN" else "normal") for m in MODELS], fs=6.0, gap=0.03)
    ax.text(0.0, 1.03, f"blocks: {len(PANEL)} reservoirs in every 5-year block · ◆ SWOT span: {len(SPAN_IDS)} of them", transform=ax.transAxes,
            fontsize=6.0, color=jw.INK2, va="bottom")
# ---------------- (g) data: as Fig 9a (pa_irreg_eco.py)
V = pd.read_parquet(PA/"vars.parquet"); ids9 = V.index
OPS = pd.read_parquet(PA/"ops_8120.parquet").reindex(ids9)
x9 = OPS.irreg_8120.values.astype(float)
EG = ["ETF", "GP", "NWFM", "NAD", "PAC"]; EN = ["Eastern Temperate Forests", "Great Plains", "Northwestern Forested Mountains", "North American Deserts", "Pacific coast"]
EC = ["#1baf7a", "#eda100", "#2a78d6", "#c0392b", "#e87ba4"]
E9 = pd.read_csv(OUTD/"eco_l1_sites.csv").set_index("DAM_ID").egroup.reindex(ids9); ge = E9.map({g: i for i, g in enumerate(EG)}).values
okx = np.isfinite(x9); EDG = np.quantile(x9[okx], [0, 1/3, 2/3, 1]); b9 = np.full(len(x9), -1); b9[okx] = np.clip(np.searchsorted(EDG, x9[okx], side="right") - 1, 0, 2)
g482 = S_[(S_.PERIOD == HK_) & (S_.MODEL == "v10pm")].set_index("DAM_ID").nse.reindex(ids9)
y9 = g482.values.astype(float); NUM["g"] = dict(edges=[float(e) for e in EDG])
def panel_g(ax):
    """Draw SWOT-GAT's hindcast NSE by storage-irregularity third, one box per ecoregion group.

    A group with at least 5 reservoirs in a third gets a box (25th to 75th percentile, black line at the
    median, whiskers from the 5th to the 95th percentile); a smaller group is drawn as points. The grey
    number under each box is its reservoir count. Counts and medians go to NUM["g"]["boxes"].
    """
    Gn = len(EG); W = 0.84/Gn; off = (np.arange(Gn) - (Gn - 1)/2)*W; NUM["g"]["boxes"] = {}
    for k in range(3):
        for g in range(Gn):
            m = (b9 == k) & (ge == g) & np.isfinite(y9); v = y9[m]; xp = k + off[g]
            if len(v) >= 5:
                q5, q25, q50, q75, q95 = np.percentile(v, [5, 25, 50, 75, 95])
                ax.add_patch(Rectangle((xp - W*0.42, q25), W*0.84, q75 - q25, facecolor=EC[g], alpha=0.85, edgecolor="white", lw=0.4, zorder=3))
                ax.vlines(xp, q5, q25, color=EC[g], lw=0.8, zorder=2); ax.vlines(xp, q75, q95, color=EC[g], lw=0.8, zorder=2)
                ax.hlines(q50, xp - W*0.42, xp + W*0.42, color=jw.INK, lw=1.1, zorder=4); NUM["g"]["boxes"][f"{k}_{EG[g]}"] = dict(n=int(m.sum()), median=float(q50))
            elif len(v):
                ax.scatter(np.full(len(v), xp), v, s=7, color=EC[g], edgecolor=jw.INK, linewidth=0.4, zorder=4); NUM["g"]["boxes"][f"{k}_{EG[g]}"] = dict(n=int(m.sum()))
            ax.text(xp, -0.03, str(int(m.sum())), transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.0, color=jw.INK3)
    for k in (0.5, 1.5): ax.axvline(k, color="#e6e5df", lw=0.8, zorder=1)
    ax.axhline(0, color=jw.INK3, ls=(0, (3, 2)), lw=0.6, zorder=1); ax.set_ylim(-1.04, 1.0); ax.set_xlim(-0.5, 2.5)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1]); ax.set_yticklabels(["≤ −1", "−0.5", "0", "0.5", "1"]); ax.set_ylabel("hindcast NSE")
    nm3 = ["most regular third", "middle third", "most irregular third"]
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{nm3[k]}  ({EDG[k]:.2f}–{EDG[k + 1]:.2f})" for k in range(3)]); ax.tick_params(axis="x", length=0, pad=11)
    ax.text(0.0, 1.03, "SWOT-GAT hindcast NSE by storage irregularity third, within EPA Level I ecoregions", transform=ax.transAxes, fontsize=6.0, color=jw.INK2, va="bottom")
    ax.text(1.0, 1.03, "grey numbers: reservoirs per box", transform=ax.transAxes, fontsize=6.0, color=jw.INK3, ha="right", va="bottom")
def epa_key(fig, x, y):
    """Draw the key to the ecoregion colors, with reservoir counts, and a title line above it.

    x and y are the figure fractions of the lower-left corner of the first swatch.
    """
    ng = [int(((ge == g) & (b9 >= 0)).sum()) for g in range(len(EG))]; x0 = x
    for g in range(len(EG)):
        fig.add_artist(Rectangle((x0, y), 0.013, 0.010*7.48/fig.get_figheight(), transform=fig.transFigure, facecolor=EC[g], edgecolor="none"))
        t = fig.text(x0 + 0.017, y + 0.005*7.48/fig.get_figheight(), f"{EN[g]} ({ng[g]})", fontsize=6.0, color=jw.INK2, va="center"); fig.canvas.draw()
        x0 = t.get_window_extent().transformed(fig.transFigure.inverted()).x1 + 0.016
    fig.text(x - 0.002, y + 0.025*7.48/fig.get_figheight(), "EPA Level I ecoregion (grouped) · panel " + ("g boxes, panel f background" if MODE == "v1" else "c boxes, panel b background")
             + " · irregularity: 1 − R² of storage vs its mean seasonal cycle, WY1981–2020",
             fontsize=6.0, color=jw.INK3, va="bottom")
# ---------------- (f) data: Fig 3b
def cpath(rings):
    """Return one matplotlib Path that holds every ring (a list of x, y vertices) as a closed polygon."""
    v, c = [], []
    for r in rings: r = np.asarray(r, float); v += r.tolist(); c += [MPath.MOVETO] + [MPath.LINETO]*(len(r) - 2) + [MPath.CLOSEPOLY]
    return MPath(v, c)
ECO_J = json.load(open(OUTD/"eco_l1_conus_albers.json")); _O = json.load(open(OUTD/"conus_outline_albers.json"))
OUTL = cpath([p["ext"] for p in _O] + [h for p in _O for h in p["holes"]])
SR = S_[(S_.PERIOD == HK_) & (S_.MODEL == "v10pm")]
def panel_f(fig, ax, cb_rect):
    """Draw the map of SWOT-GAT's correlation r of daily anomalies in the 2010-2020 hindcast.

    Dots lie on pale ecoregion tints with white state lines; dot area grows with the logarithm of
    capacity, and the colors span the 10th to 90th percentile of r. cb_rect is the color bar's rectangle
    in figure fractions. The numbers drawn go to NUM["f"].
    """
    jw.basemap(ax, states=False); clip = PathPatch(OUTL, transform=ax.transData, facecolor="none", edgecolor="none")
    for reg in ECO_J:
        if reg["group"] in EG and reg["rings"]:
            pp = PathPatch(cpath(reg["rings"]), facecolor=mix(EC[EG.index(reg["group"])], 0.80), edgecolor="none", zorder=0.6); ax.add_patch(pp); pp.set_clip_path(clip)
    for ring in json.load(open(G.STATES_JSON)):
        a_ = np.asarray(ring, float); xs, ys = G.albers(a_[:, 0], a_[:, 1])
        ax.add_patch(MplPolygon(np.column_stack([xs, ys]), closed=True, facecolor="none", edgecolor="white", lw=0.4, zorder=0.7))
    ax.set_xlim(-0.372, 0.357); ax.set_ylim(0.045, 0.50)
    cap = pd.to_numeric(SR.cap, errors="coerce"); x, y = G.albers(SR.lon.values, SR.lat.values); o = np.argsort(-cap.fillna(0).values)
    r10, r90 = np.percentile(SR.r, [10, 90])
    sc = ax.scatter(x[o], y[o], c=SR.r.values[o], cmap=jw.MAP_SEQ_R, norm=mpl.colors.Normalize(r10, r90), s=G.cap_size(cap.values, lo=2.0, hi=20)[o], lw=0.3, edgecolors="#4d4d4d", zorder=3)
    ax.text(0.045, 1.0, f"SWOT-GAT, {HL_} · n = {len(SR)}", transform=ax.transAxes, fontsize=6.5, fontweight="bold", va="bottom")
    cbx = fig.add_axes(cb_rect); cb = fig.colorbar(sc, cax=cbx, orientation="horizontal", extend="both")
    tk = [round(t, 1) for t in np.arange(-1, 1.01, 0.1) if r10 - 1e-9 <= t <= r90 + 1e-9]; cb.set_ticks(tk); cb.set_ticklabels([f"{t:.1f}" for t in tk])
    cb.ax.tick_params(labelsize=6.0, length=1.8, pad=1); cb.outline.set_linewidth(0.4)
    cbx.set_title("correlation r of daily anomalies, 10th–90th percentile", fontsize=6.0, pad=2, color=jw.INK2, loc="left")
    fig.text(cb_rect[0] + cb_rect[2] + 0.012, cb_rect[1] + cb_rect[3]/2, f"median {SR.r.median():.2f}", fontsize=6.0, color=jw.INK2, va="center")
    NUM["f"] = dict(n=int(len(SR)), p10=float(r10), p90=float(r90), median=float(SR.r.median()))
# ---------------- (a-d, v1) the Fig 5 series, compressed (Fig 5's loop with thinner lines and shorter rows)
def ts_rows(fig, I, top, ROW_H=0.50, GAPY=0.25, L0=0.085, GAPX=0.022, WL=0.62, WR=0.27, LW=0.95):
    """Draw one hydrograph row per reservoir in EX and return SWOT-GAT's scores per DAM_ID.

    Each row has the 2010-2020 hindcast on the left and the SWOT span on the right. The loaders it
    calls (hind, x26_pair, span26 ...) exist only in the v1 mode, which takes them from jw_fig_tt_e.py.
    I converts inches to a fraction of the figure height; top is the top of the first row in inches.
    ROW_H and GAPY are the row height and the gap between rows in inches; L0, GAPX, WL and WR are the
    left edge, the gap between the two panels and their widths in figure fractions; LW is the line
    width of the gauge and SWOT-GAT lines. Returns, per DAM_ID, the hindcast and SWOT-span nRMSE, NSE
    and VR, the NSE percentile and the reservoir name.
    """
    out = {}
    for j, dam in enumerate(EX):
        y0 = top - (j + 1)*ROW_H - j*GAPY
        aL = fig.add_axes([L0, I(y0), WL, I(ROW_H)]); aR = fig.add_axes([L0 + WL + GAPX, I(y0), WR, I(ROW_H)])
        d = hind(dam); o1, p1 = d.gauge.values, d["SWOT-GNN"].values; k = np.isfinite(o1) & np.isfinite(p1)
        o1, p1 = o1 - np.nanmean(o1[k]), p1 - np.nanmean(p1[k])
        t2 = T26; o2, p2 = x26_pair(dam); span = span26(dam) & rules_ok26(dam)
        o2, p2 = np.where(span, o2, np.nan), np.where(span, p2, np.nan); k2 = np.isfinite(o2) & np.isfinite(p2)
        o2, p2 = o2 - np.nanmean(o2[k2]), p2 - np.nanmean(p2[k2])
        extra = []
        if RULES:
            for m in RL:
                r1 = d[m].values if m in d else None
                if r1 is not None and np.isfinite(r1).sum() > 300:
                    kk = np.isfinite(o1) & np.isfinite(r1); r1 = r1 - np.nanmean(r1[kk]); aL.plot(d.index, sm(r1), color=tint(RCLR[m]), lw=0.45, zorder=2); extra.append(r1)
                o3, r2 = rule26(m, dam)
                if r2 is not None: o3, r2 = np.where(span, o3, np.nan), np.where(span, r2, np.nan)
                if r2 is not None and np.isfinite(r2).sum() > 100:
                    kk = np.isfinite(o3) & np.isfinite(r2); r2 = r2 - np.nanmean(r2[kk]); aR.plot(t2, sm(r2), color=tint(RCLR[m]), lw=0.45, zorder=2); extra.append(r2)
        for a_, t, o, p in ((aL, d.index, o1, p1), (aR, t2, o2, p2)):
            halo = [pe.Stroke(linewidth=LW + 0.9, foreground="white"), pe.Normal()]
            a_.plot(t, sm(o), color=jw.INK, lw=LW, zorder=4, path_effects=halo); a_.plot(t, sm(p), color=LINEB, lw=LW, zorder=5, path_effects=halo)
        sw_ = swot26(dam); swp = None
        if sw_ is not None and len(sw_):
            ts = sw_.index; gi = pd.Series(o2, index=t2).reindex(ts).values; kk = np.isfinite(gi)
            if kk.sum(): swp = (ts, sw_.values - np.median(sw_.values[kk] - gi[kk]))
        allv = np.r_[o1[np.isfinite(o1)], p1[np.isfinite(p1)], o2[np.isfinite(o2)], p2[np.isfinite(p2)]]; lo_, hi_ = np.percentile(allv, [0.2, 99.8])
        for e_ in extra:
            if np.isfinite(e_).sum() > 50: q_ = np.nanpercentile(sm(e_), [2, 98]); lo_, hi_ = min(lo_, q_[0]), max(hi_, q_[1])
        pad = 0.08*(hi_ - lo_)
        for a_ in (aL, aR): a_.set_ylim(lo_ - pad, hi_ + pad); a_.tick_params(axis="both", labelsize=6.0, length=1.6, pad=1)
        if swp is not None:
            y0_, y1_ = lo_ - pad, hi_ + pad; v_ = swp[1]; inr = (v_ >= y0_) & (v_ <= y1_); hm, lm = v_ > y1_, v_ < y0_
            aR.scatter(swp[0][inr], v_[inr], s=4.5, facecolor="none", edgecolor=SWOT_RING, lw=0.5, zorder=6)
            if hm.any(): aR.scatter(swp[0][hm], np.full(hm.sum(), y1_ - 0.05*(y1_ - y0_)), s=8, marker="^", color=SWOT_RING, lw=0, zorder=7)
            if lm.any(): aR.scatter(swp[0][lm], np.full(lm.sum(), y0_ + 0.05*(y1_ - y0_)), s=8, marker="v", color=SWOT_RING, lw=0, zorder=7)
        aL.set_xlim(pd.Timestamp("2010-01-01"), pd.Timestamp("2020-12-31")); aR.set_xlim(pd.Timestamp("2023-07-01"), pd.Timestamp("2026-07-01"))
        aL.set_xticks(pd.date_range("2010-01-01", "2020-01-01", freq="YS")); aL.set_xticklabels([])
        RT = pd.to_datetime(["2023-07-01", "2024-07-01", "2025-07-01"]); aR.set_xticks(RT); aR.set_xticklabels([])
        if j == len(EX) - 1:
            for yy in range(2010, 2021): aL.text(pd.Timestamp(f"{yy}-07-01"), -0.20, str(yy), transform=mpl.transforms.blended_transform_factory(aL.transData, aL.transAxes), fontsize=6.0, ha="center", va="top")
            for t_, l_ in zip(RT, ["Jul 2023", "Jul 2024", "Jul 2025"]): aR.text(t_, -0.20, l_, transform=mpl.transforms.blended_transform_factory(aR.transData, aR.transAxes), fontsize=6.0, ha="center", va="top")
        aL.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3, min_n_ticks=3))
        aL.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v/1000:g}k" if abs(v) >= 1000 else f"{v:g}"))
        aR.set_yticks([]); aR.spines["left"].set_visible(False)
        r = meta.loc[dam]; nm = K._clean_name(str(r["name"])).replace(" Reservoir", ""); st = K.ST.get(str(r.state), str(r.state))
        jw.seq_text_ax(aL, 0.0, 1.09, [(f"{nm}, {st}", jw.INK, "bold"), (f"{USE.get(str(r.use), str(r.use).lower())} · {r.cap:,.0f} MCM · {ECN.get(ECO.get(dam), '')}", jw.INK2, "normal")], fs=6.0, gap=0.012)
        (n1, e1, v1), (n2, e2, v2) = scores(o1, p1), scores(o2, p2)
        out[int(dam)] = dict(nrmse_hind=float(n1), nrmse_swot=float(n2), nse_hind=float(e1), nse_swot=float(e2), vr_hind=float(v1), vr_swot=float(v2), pct=float(PCT[dam]), name=nm)
        fe = lambda v: f"{v:.2f}".replace("-", "−")
        aL.text(1.0, 1.09, f"NSE {fe(e1)} ({ordn(int(round(PCT[dam])))} pct) · nRMSE {100*n1:.0f}% · VR {v1:.2f}", transform=aL.transAxes, fontsize=6.0, color="#1f66c1", ha="right", va="bottom", fontweight="bold")
        aR.text(1.0, 1.09, f"NSE {fe(e2)} · nRMSE {100*n2:.0f}% · VR {v2:.2f}", transform=aR.transAxes, fontsize=6.0, color=jw.BLUE8[7], ha="right", va="bottom", fontweight="bold")
        jw.letter(aL, "abcd"[j], x=-0.075, y=1.0)
        if j == 0:
            aL.text(0.5, 1.50, "2010–2020 hindcast (no SWOT)", transform=aL.transAxes, fontsize=6.5, color=jw.INK, ha="center", va="bottom", fontweight="bold")
            aR.text(0.5, 1.50, "SWOT span (trained on), Jul 2023–Jun 2026", transform=aR.transAxes, fontsize=6.5, color=jw.INK, ha="center", va="bottom", fontweight="bold")
    return out
# ---------------- layout
base = 3.98 if MODE == "v1" else 0.50; FH = base + 4.98
fig = plt.figure(figsize=(jw.COL2, FH)); I = lambda inch: inch/FH
L = dict(v1="efg", v2="abc")[MODE]
if MODE == "v1":
    KEY = [("gauge", jw.INK, "bold"), ("SWOT-GAT", "#1f66c1", "bold"), ("MODROM-cal", "#a87200", "normal"), ("GDROM", "#c9501f", "normal"), ("ISTARF", "#11875c", "normal"),
           ("○  SWOT observations (HALF-exact)", SWOT_RING, "normal"), ("· 7-day means; SWOT-GAT scores on daily values, VR = variability ratio", jw.INK2, "normal")]
    jw.seq_text(fig, 0.085, I(FH - 0.16), KEY, fs=6.0, gap=0.012)
    NUM["series"] = ts_rows(fig, I, FH - 0.66)
    fig.text(0.012, I(FH - 0.66 - 1.37), "storage anomaly (MCM)", rotation=90, fontsize=6.5, va="center", ha="left")
    ref = json.load(open(OUTD/"num_e_fig_tt.json"))["nrmse"]
    for dam, e in NUM["series"].items():
        for k_ in ("nse_hind", "nse_swot", "nrmse_hind", "nrmse_swot", "vr_hind", "vr_swot"): assert abs(e[k_] - ref[str(dam)][k_]) < 1e-9, (dam, k_)
ae = fig.add_axes([0.075, I(FH - base - 2.02), 0.38, I(1.72)]); panel_e(ae); jw.letter(ae, L[0], x=-0.13, y=1.13)
af = fig.add_axes([0.50, I(FH - base - 2.10), 0.49, I(2.10)]); panel_f(fig, af, [0.55, I(FH - base - 2.43), 0.20, I(0.07)]); jw.letter(af, L[1], x=0.035, y=1.0)
ag = fig.add_axes([0.075, I(FH - base - 4.30), 0.905, I(1.48)]); panel_g(ag); jw.letter(ag, L[2], x=-0.045, y=1.03)
epa_key(fig, 0.075, I(0.10))
stem = f"e_combo_{MODE}"; jw.save(fig, OUTD, stem); json.dump(NUM, open(OUTD/f"num_{stem}.json", "w"), indent=1, default=float)
print(stem, json.dumps({k: v for k, v in NUM.items() if k != "series"}, default=float)[:1500])
