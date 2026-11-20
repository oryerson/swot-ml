"""Shared data and drawing code for the figure scripts. Figures 2, 3 and 4 import it.

Nothing is run or scored again here: the module only loads tables that stage 07 wrote and draws from them.

What happens at import
  1. load_combo() reads combo_e.py as text, keeps the part before the comment line that opens its
     "(a-d, v1)" part (that is, its data section), sets MODE to "v2" and executes it. The names that
     code defines are in the dict C: score tables, ecoregion polygons, colors. Because of this text
     search, combo_e.py must not be edited.
  2. style() sets the plot style (jw.use_style, all text 7 pt).
  3. S10 is the 2010-2020 part of the kit's score table C["S_"]. IDS314 is the 2010-2020 four-model set:
     the reservoirs scored by all of SWOT-GAT, MODROM-cal, GDROM and ISTARF. An assertion checks that it
     equals the "2010-20" set of fig1_o50.json without the reservoirs excluded in excl_d.py. The name
     keeps an earlier count; in the paper's run the set has 301 reservoirs.
  4. X20 and EDG20 are the 2000-2020 gauge storage irregularity and its tertile edges over the
     reservoirs that have a value (477 in the paper's run). None of the paper's four figures uses them.

Reads
  "Kit folder" is the folder of this file. Other paths are relative to CONUS_storage_fork/ under
  $SWOTGAT_WORK.
  Kit folder:
    combo_e.py                               its data section is executed (step 1)
    sites_2010_g.csv, sites_2325span_g.csv   scores of the 2010-2020 hindcast and of the SWOT Period
                                             (read by jw.sites_table; rows with High or Medium link
                                             confidence and a valid nRMSE are kept)
    eco_l1_sites.csv, eco_l1_conus_albers.json, conus_outline_albers.json   ecoregions and map outlines
    irreg_0020.parquet                       2000-2020 irregularity (step 4)
    excl_d.py                                the excluded reservoirs
  eval/grl_figs/sites_v5_g.csv               read by jw.sites_table (a table from an earlier model
                                             configuration)
  eval/perf_attrib/out/vars.parquet, ops_8120.parquet   tables of the skill-driver analysis
  eval/overlap50/fig1_o50.json               reference: the reservoir sets all four models score

Writes: save(fig, stem) writes <stem>.pdf and <stem>.png (200 dpi) to the kit folder.

Model labels
  In the tables, `v10pm` and `SWOT-GNN` mean SWOT-GAT (the paper's model: mean of two seeds). M4 lists the
  four models by their label in the site tables, KEY gives their label in the full-record tables, LAB
  their printed name, CLR their line color and TXT a darker shade for text.

What the paper's figures use
  save(), model_boxes() with box() and box_axis() (Figures 3b, 4a, 4b), model_key(), cap_key() (the key
  to the capacity dot sizes in Figure 3a), NM3, S10, IDS314 and the constants above.
  Boxes show the 25th to 75th percentile, a black line at the median and whiskers from the 5th to the 95th
  percentile, all clipped at the axis limits in LIM.

Not used by the paper's figures
  map_panel(), map_irreg(), map_vr(), map_r(), epa_key(), thru_panel(), third20(), nums() and SEQ draw
  panels of earlier versions of the figure set. thru_panel() needs C["FR"], C["PANEL"], C["SP"],
  C["SPAN_IDS"] and C["WINS"], which combo_e.py sets to None, so it cannot be called as the code stands.
"""
from swotgat.paths import WORK
import os, sys, json, numpy as np, pandas as pd, matplotlib as mpl
from pathlib import Path
OUT = Path(__file__).resolve().parent; FS = 7.0
def load_combo():
    """Execute the data section of combo_e.py and return the names it defines as a dict.

    The data section is the text of combo_e.py before the comment line that opens its "(a-d, v1)" part.
    Its MODE line is replaced so that MODE is "v2". The function stops with an assertion error unless
    both the comment line and the MODE line occur exactly once in combo_e.py.
    """
    src = (OUT/"combo_e.py").read_text(); CM = "# ---------------- (a-d, v1)"; MO = 'MODE = os.environ.get("MODE", "v1")'
    assert src.count(CM) == 1 and src.count(MO) == 1
    C = {"__file__": str(OUT/"combo_e.py"), "__name__": "combo_e_data"}; exec(compile(src.split(CM)[0].replace(MO, 'MODE = "v2"'), "combo_e.py (data)", "exec"), C)
    return C
C = load_combo()
import jw, grl_common as G
from jw import plt
from matplotlib.patches import PathPatch, Polygon as MplPolygon, Rectangle
def style():
    """Set the plot style: jw.use_style(), with tick labels, axis labels and legend text at FS (7 pt)."""
    jw.use_style(); mpl.rcParams.update({"xtick.labelsize": FS, "ytick.labelsize": FS, "axes.labelsize": FS, "legend.fontsize": FS})
style()
M4 = ["v10pm", "MODROM-cal", "GDROM", "ISTARF"]; KEY = {"v10pm": "SWOT-GNN", "MODROM-cal": "MODROM-cal", "GDROM": "GDROM", "ISTARF": "ISTARF"}   # label in the site tables -> label in the full-record tables
CLR = {"v10pm": "#2a78d6", "GDROM": "#eb6834", "ISTARF": "#1baf7a", "MODROM-cal": "#eda100"}
TXT = {"v10pm": "#1f66c1", "MODROM-cal": "#a87200", "GDROM": "#c9501f", "ISTARF": "#11875c"}
LAB = {"v10pm": "SWOT-GAT", "MODROM-cal": "MODROM-cal", "GDROM": "GDROM", "ISTARF": "ISTARF"}
EC, EG, EN, mix = C["EC"], C["EG"], C["EN"], C["mix"]; HL = C["HL_"]; SR = C["SR"]
S10 = C["S_"][C["S_"].PERIOD == C["HK_"]]
IDS314 = sorted(set.intersection(*[set(S10[S10.MODEL == m].DAM_ID) for m in M4]))
assert IDS314 == sorted(i for i in json.load(open(f'{WORK}/CONUS_storage_fork/eval/overlap50/fig1_o50.json'))['2010-20']['ids'] if i not in __import__('excl_d').EXCL)   # the excluded reservoirs have no rule-model rows in the kit tables
X20 = pd.read_parquet(OUT/"irreg_0020.parquet").irreg_0020; EDG20 = np.nanquantile(X20.values, [0, 1/3, 2/3, 1])
def third20(ids):
    """Return the 2000-2020 irregularity third of each id (0, 1, 2; -1 = no value) as a Series."""
    x = X20.reindex(ids); return pd.Series(np.where(x.notna(), np.clip(np.searchsorted(EDG20, x.fillna(0).values, side="right") - 1, 0, 2), -1), index=ids)
NM3 = ["most regular", "middle", "most irregular"]
SEQ = mpl.colors.LinearSegmentedColormap.from_list("blue250_700", ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"])   # sequential light -> dark blue; the fifth color, #2a78d6, is SWOT-GAT's line color
def nums(name):
    """Return the content of the JSON file `name` in the kit folder."""
    return json.load(open(OUT/name))
def save(fig, stem):
    """Check the layout (jw.qa) and write <stem>.pdf and <stem>.png (200 dpi) to the kit folder.

    Returns the list of messages of the layout check.
    """
    H = jw.qa(fig, stem, min_font=FS); fig.savefig(OUT/f"{stem}.pdf"); fig.savefig(OUT/f"{stem}.png", dpi=200); return H
# ---------------- maps (fig4_f.py / fig4alt_g.py)
CAP = pd.to_numeric(SR.cap, errors="coerce")
def map_panel(fig, I, ax, cb_rect, sel, val, cmap, norm, ticks, tlabels, head, cbtitle, note):
    """Draw a map of one value per reservoir on pale ecoregion tints, with a horizontal color bar.

    I converts inches to a fraction of the figure height. sel is a boolean mask over the rows of SR and
    val holds the values of the selected reservoirs; cmap and norm color them. Dot area grows with the
    logarithm of capacity, and large reservoirs are drawn first. cb_rect is the color bar's rectangle in
    figure fractions, ticks and tlabels its ticks, cbtitle its title and note the text to its right;
    head is the bold line above the map. Returns the number of selected reservoirs with coordinates.
    Not used by the paper's figures.
    """
    jw.basemap(ax, states=False); clip = PathPatch(C["OUTL"], transform=ax.transData, facecolor="none", edgecolor="none")
    for reg in C["ECO_J"]:
        if reg["group"] in EG and reg["rings"]:
            pp = PathPatch(C["cpath"](reg["rings"]), facecolor=mix(EC[EG.index(reg["group"])], 0.80), edgecolor="none", zorder=0.6); ax.add_patch(pp); pp.set_clip_path(clip)
    for ring in json.load(open(G.STATES_JSON)):
        a_ = np.asarray(ring, float); xs, ys = G.albers(a_[:, 0], a_[:, 1])
        ax.add_patch(MplPolygon(np.column_stack([xs, ys]), closed=True, facecolor="none", edgecolor="white", lw=0.4, zorder=0.7))
    ax.set_xlim(-0.372, 0.357); ax.set_ylim(0.045, 0.50)
    cap = CAP[sel].values; mx, my = G.albers(SR.lon.values[sel], SR.lat.values[sel]); o = np.argsort(-np.nan_to_num(cap, nan=0.0))
    sc = ax.scatter(mx[o], my[o], c=val[o], cmap=cmap, norm=norm, s=G.cap_size(cap, lo=2.0, hi=20)[o], lw=0.3, edgecolors="#4d4d4d", zorder=3)
    ax.text(0.045, 1.0, head, transform=ax.transAxes, fontsize=FS, fontweight="bold", va="bottom")
    cbx = fig.add_axes(cb_rect); cb = fig.colorbar(sc, cax=cbx, orientation="horizontal", extend="both")
    cb.set_ticks(ticks); cb.set_ticklabels(tlabels); cb.ax.tick_params(labelsize=FS, length=1.8, pad=1); cb.outline.set_linewidth(0.4)
    cbx.set_title(cbtitle, fontsize=FS, pad=2, color=jw.INK2, loc="left")
    fig.text(cb_rect[0] + cb_rect[2] + 0.014, cb_rect[1] - I(0.063), note, fontsize=FS, color=jw.INK2, va="center")
    return int(np.isfinite(mx).sum())
def map_irreg(fig, I, ax, cb_rect):
    """Draw the map of 2000-2020 gauge storage irregularity.

    Irregularity is 1 - R^2 of storage against its mean seasonal cycle; the colors span its 10th to 90th
    percentile. Returns the numbers drawn. Not used by the paper's figures.
    """
    xa = X20.reindex(SR.DAM_ID.values).values.astype(float); sel = np.isfinite(xa) & np.isfinite(SR.lon.values.astype(float)) & np.isfinite(SR.lat.values.astype(float))
    i10, i90 = np.nanpercentile(X20.values, [10, 90]); it = [t for t in (0.3, 0.5, 0.7, 0.9) if i10 - 1e-9 <= t <= i90 + 1e-9]
    na = map_panel(fig, I, ax, cb_rect, sel, xa[sel], jw.MAP_SEQ, mpl.colors.Normalize(i10, i90), it, [f"{t:g}" for t in it],
                   f"Gauge storage irregularity · 2000–2020 · n = {int(np.isfinite(X20.values).sum())}", "1 − R² of storage vs its mean seasonal cycle, 10th–90th percentile",
                   f"median {np.nanmedian(X20.values):.2f}")
    return dict(n=int(np.isfinite(X20.values).sum()), n_mapped=na, n_of=int(len(X20)), p10=float(i10), p90=float(i90), median=float(np.nanmedian(X20.values)),
                share_below_p10=float(np.nanmean(X20.values < i10)), share_above_p90=float(np.nanmean(X20.values > i90)))
def map_vr(fig, I, ax, cb_rect):
    """Draw the map of SWOT-GAT's variability ratio in the 2010-2020 hindcast.

    The color scale is logarithmic and spans the 10th to 90th percentile. Returns the numbers drawn.
    Not used by the paper's figures (Figure 3a draws its own map).
    """
    amp = SR.amp.values.astype(float); a10, a90 = np.percentile(amp, [10, 90]); AT = [t for t in (0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0) if a10 - 1e-9 <= t <= a90 + 1e-9]
    map_panel(fig, I, ax, cb_rect, np.ones(len(SR), bool), np.log2(np.clip(amp, 1e-6, None)), SEQ, mpl.colors.Normalize(np.log2(a10), np.log2(a90)),
              [np.log2(t) for t in AT], [f"{t:g}" for t in AT], f"SWOT-GAT variability ratio · {HL} · n = {len(SR)}", "model sd ÷ gauge sd",
              f"median {np.median(amp):.2f} · below 1 at {100*(amp < 1).mean():.0f}%")
    return dict(n=int(len(amp)), p10=float(a10), p90=float(a90), median=float(np.median(amp)), share_below_1=float((amp < 1).mean()),
                share_below_p10=float((amp < a10).mean()), share_above_p90=float((amp > a90).mean()), ticks=AT)
def map_r(fig, I, ax, cb_rect):
    """Draw the map of SWOT-GAT's correlation r of daily anomalies in the 2010-2020 hindcast.

    The colors span the 10th to 90th percentile. Returns the numbers drawn. Not used by the paper's
    figures.
    """
    r10, r90 = np.percentile(SR.r, [10, 90]); rt = [round(t, 1) for t in np.arange(-1, 1.01, 0.1) if r10 - 1e-9 <= t <= r90 + 1e-9]
    map_panel(fig, I, ax, cb_rect, np.ones(len(SR), bool), SR.r.values, jw.MAP_SEQ_R, mpl.colors.Normalize(r10, r90), rt, [f"{t:.1f}" for t in rt],
              f"SWOT-GAT correlation r · {HL} · n = {len(SR)}", "correlation r of daily anomalies, 10th–90th percentile", f"median {SR.r.median():.2f}")
    return dict(n=int(len(SR)), p10=float(r10), p90=float(r90), median=float(SR.r.median()))
def cap_key(ax):
    """Draw the key to the capacity dot sizes (100, 1,000 and 10,000 MCM) at the right edge of a map."""
    KC = [100, 1000, 10000]; ky = [0.30, 0.235, 0.165]
    ax.scatter([1.0]*3, ky, s=G.cap_size(np.array(KC, float), lo=2.0, hi=20), facecolors="none", edgecolors="#4d4d4d", lw=0.5, transform=ax.transAxes, zorder=5, clip_on=False)
    for c_, y_ in zip(KC, ky): ax.text(0.97, y_, f"{c_:,}", transform=ax.transAxes, fontsize=FS, color=jw.INK2, ha="right", va="center")
    ax.text(1.015, 0.335, "capacity\n(MCM)", transform=ax.transAxes, fontsize=FS, color=jw.INK2, ha="right", va="bottom", linespacing=1.0)
def epa_key(fig, I, y, title, tint=True, counts=None, x0=0.032):
    """Draw one row of swatches for the EPA Level I ecoregion groups, with a title line above.

    The row starts at figure fraction x0, y inches above the figure bottom (I converts inches to a
    fraction of the figure height). tint=True draws the pale map tints, tint=False the box colors;
    counts, if given, adds a number to each group name. Stops with an assertion error if the row runs
    off the canvas.
    """
    fig.text(x0, I(y + 0.17), title, fontsize=FS, color=jw.INK3, va="baseline"); fig.canvas.draw()
    fx = lambda t_: t_.get_window_extent().transformed(fig.transFigure.inverted()).x1; x = x0
    for g in range(len(EG)):
        lab = EN[g] + (f" ({counts[g]})" if counts is not None else "")
        t = fig.text(x + 0.017, I(y), lab, fontsize=FS, color=jw.INK2, va="center"); fig.canvas.draw()
        fig.add_artist(Rectangle((x, I(y - 0.04)), 0.013, I(0.08), transform=fig.transFigure, facecolor=mix(EC[g], 0.80) if tint else EC[g],
                                 edgecolor="#bdbdbd" if tint else "none", lw=0.3)); end = fx(t); x = end + 0.014
    assert end < 0.99, f"EPA key runs off the canvas ({end:.3f})"
# ---------------- median metric through time (fig1_f.py panel c; fig4alt_g.py panel d)
def thru_panel(ax, metric, label="inside", xstep=5):
    """Draw the four models' median of a metric through time: 5-year blocks and the SWOT span.

    metric is "nse" or the variability ratio column. label="inside" writes the name of the SWOT
    training period and n inside its band, any other value writes the name above the band; xstep is the
    spacing of the year ticks. Returns the counts and medians drawn. Not used by the paper's figures:
    the tables it needs are set to None in combo_e.py, so it cannot be called as the code stands.
    """
    FR, PANEL, SP, SPAN_IDS, WINS, yc, MK = C["FR"], C["PANEL"], C["SP"], C["SPAN_IDS"], C["WINS"], C["yc"], C["MK"]
    B = FR[FR.DAM_ID.isin(PANEL)].groupby(["window", "model"])[metric].median()
    SV = {m: float(SP[(SP.MODEL == MK[KEY[m]]) & SP.DAM_ID.isin(SPAN_IDS)][metric].median()) for m in M4}
    t0, t1 = 2023.5, 2026 + 5.4/12; tc = (t0 + t1)/2; DODGE = {"GDROM": -0.3, "ISTARF": 0.3}; TR = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.axvspan(t0, t1, color=C["TRAIN"], lw=0, zorder=0)
    if metric == "nse": ax.axhline(0, color=jw.INK3, lw=0.5, ls=(0, (2, 2)), zorder=0.5)
    else: ax.axhline(1.0, color=jw.INK3, lw=0.5, ls=(0, (3, 2)), zorder=0.5)
    if label == "inside":
        ax.text(tc, 0.60, "SWOT\ntraining\nperiod", transform=TR, ha="center", va="center", fontsize=FS, color=C["TRAIN_T"], linespacing=1.0)
        ax.text(tc, 0.03, f"n = {len(SPAN_IDS)}", transform=TR, ha="center", va="bottom", fontsize=FS, color=C["TRAIN_T"])
    else:
        ax.text(t1, 1.015, "SWOT training period", transform=TR, ha="right", va="bottom", fontsize=FS, color=C["TRAIN_T"])
    for m in M4[::-1]:
        g = m == "v10pm"; yv = B.xs(KEY[m], level="model").reindex(WINS).values; xs_ = tc + DODGE.get(m, 0.0)
        ax.plot([yc(w) for w in WINS], yv, color=CLR[m], lw=1.5 if g else 0.85, marker="o", ms=2.6 if g else 1.8, mec="white", mew=0.3, zorder=4 if g else 3)
        ax.plot([yc(WINS[-1]), xs_], [yv[-1], SV[m]], color=CLR[m], lw=1.1 if g else 0.8, ls=(0, (1, 1.6)), zorder=3.5 if g else 2.5)
        ax.plot([xs_], [SV[m]], marker="D", ms=3.6 if g else 2.8, color=CLR[m], mec="white", mew=0.4, ls="none", zorder=5 if g else 4)
    ax.set_xlim(1980, 2027.5); ax.set_xticks(range(1980, 2026, xstep))
    if metric == "nse": ax.set_ylim(-0.15, 0.85); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8]); ax.set_ylabel("median NSE")
    else: ax.set_ylim(0.4, 1.2); ax.set_yticks([0.4, 0.6, 0.8, 1.0, 1.2]); ax.set_ylabel("median variability ratio")
    return dict(panel=len(PANEL), span_n=len(SPAN_IDS), span_median=SV, blocks={w: {m: float(B[(w, KEY[m])]) for m in M4} for w in WINS},
                connectors={m: [float(B[(WINS[-1], KEY[m])]), SV[m]] for m in M4})
def model_key(fig, x, y, tail=None):
    """Write the four model names in their text colors, SWOT-GAT in bold, from figure position (x, y).

    tail is an optional grey text after the names.
    """
    jw.seq_text(fig, x, y, [(LAB[m], TXT[m], "bold" if m == "v10pm" else "normal") for m in M4] + ([(tail, jw.INK2, "normal")] if tail else []), fs=FS, gap=0.014)
# ---------------- boxes (fig_box_f.py)
def box(ax, v, lim, pos, color, width=0.17):
    """Draw one box of the values v at x position pos.

    The box spans the 25th to 75th percentile, with a black line at the median and whiskers from the
    5th to the 95th percentile; all five are clipped to the axis limits lim = (low, high).
    """
    q5, q25, q50, q75, q95 = np.clip(np.percentile(v, [5, 25, 50, 75, 95]), *lim)
    ax.add_patch(Rectangle((pos - width/2, q25), width, q75 - q25, facecolor=color, alpha=0.85, edgecolor="white", lw=0.4, zorder=3))
    ax.vlines(pos, q5, q25, color=color, lw=0.8, zorder=2); ax.vlines(pos, q75, q95, color=color, lw=0.8, zorder=2)
    ax.hlines(q50, pos - width/2, pos + width/2, color=jw.INK, lw=1.1, zorder=4)
LIM = {"nse": (-1.0, 1.0), "nrmse": (0.0, 0.6), "amp": (1/16, 8.0)}
def box_axis(ax, f):
    """Set the y axis of a box panel for metric f.

    "nse": -1 to 1 with a guide line at 0. "nrmse": 0 to 60%. Any other value (the variability ratio,
    "amp"): base-2 logarithmic axis from 1/16 to 8 with a guide line at 1. The limits are those of LIM.
    """
    if f == "nse":
        ax.set_ylim(*LIM[f]); ax.set_yticks([-1, -0.5, 0, 0.5, 1]); ax.set_yticklabels(["≤ −1", "−0.5", "0", "0.5", "1"]); ax.axhline(0, color=jw.INK3, lw=0.5, ls=(0, (3, 2)), zorder=1); ax.set_ylabel("NSE")
    elif f == "nrmse":
        ax.set_ylim(*LIM[f]); ax.set_yticks([0, 0.2, 0.4, 0.6]); ax.set_yticklabels(["0", "20", "40", "≥ 60%"]); ax.set_ylabel("nRMSE")
    else:
        ax.set_yscale("log", base=2); ax.set_ylim(*LIM[f]); ax.set_yticks([1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8]); ax.set_yticklabels(["≤ 1/16", "1/8", "1/4", "1/2", "1", "2", "4", "≥ 8"])
        ax.yaxis.set_minor_locator(mpl.ticker.NullLocator()); ax.axhline(1.0, color=jw.INK3, lw=0.5, ls=(0, (3, 2)), zorder=1); ax.set_ylabel("variability ratio")
def model_boxes(ax, P, groups, labels, f):
    """Draw four boxes side by side (one per model in M4) for each group of reservoirs.

    P is a table of metric f ("nse", "nrmse" or "amp" = variability ratio) with one row per reservoir
    and one column per model; groups is a list of reservoir-id lists; labels are the x tick labels.
    Returns, per group index, each model's median, its 5th, 25th, 75th and 95th percentile, and n.
    """
    num = {}; off = np.array([-0.33, -0.11, 0.11, 0.33])
    for gi, g in enumerate(groups):
        for j, m in enumerate(M4):
            v = P.loc[g, m].dropna().values.astype(float); box(ax, v, LIM[f], gi + off[j], CLR[m])
            num.setdefault(str(gi), {})[m] = float(np.median(v)); num[str(gi)][m + "_p5_25_75_95"] = [float(q) for q in np.percentile(v, [5, 25, 75, 95])]
        num[str(gi)]["n"] = len(g)
    for k in np.arange(0.5, len(groups) - 0.5, 1.0): ax.axvline(k, color="#e6e5df", lw=0.6, zorder=0)
    ax.set_xlim(-0.5, len(groups) - 0.5); ax.set_xticks(range(len(groups))); ax.set_xticklabels(labels, linespacing=1.05); ax.tick_params(axis="x", length=0)
    box_axis(ax, f); return num
