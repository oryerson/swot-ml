"""Panel functions, labels and colors of an earlier generation of the figures.

The paper's figures use two small pieces of it.

What the paper's figures use
  fig1_c5d.py runs the loading code of jw_fig_tt_e.py, which imports this module. Figure 1 then calls
  _clean_name() to tidy dam names for the row titles, and ST to turn state names into postal
  abbreviations. Importing this module also sets grl_common.SERIES_FILE to series{GRL_TAG}.npz (see
  grl_tag.py); jw_fig_tt_e.py sets it again afterwards.

Everything else
  The other functions draw panels for figure sets of an earlier model configuration: maps (p_map,
  map_colorbar), cumulative distributions (p_cdf), paired differences (p_forest), medians by reservoir
  purpose (p_dots), hydrographs (p_hydro, p_hydro2, fig_examples, pick_range) and 5-year-block scores of
  the full gauge record (fullrecord, med_ci_block, drift_quartiles). These panels show nRMSE and r computed
  on series with their mean removed (anomalies); none of them shows NSE.
  They read, under CONUS_storage_fork/eval/ in $SWOTGAT_WORK: grl_figs/sites_v2{GRL_TAG}.csv and the series
  file (through grl_common), compare_weekend/storage_atlas_v10pm/ (SWOT points) and
  overnight/fullrecord_v1/ (block scores).

Labels and colors
  The model label `v10pm`, and the printed name "SWOT–GNN" (OURS), stand for the SWOT model, which is
  SWOT-GAT in the paper (mean of two seeds). The periods "2015-20" and "2023-25" are those of the earlier
  figure sets.
  Each model keeps one color in every panel (CLR); TXT holds darker shades for text.
  nRMSE colors use a square-root scale (NORM) on the reversed RdYlBu colormap, so that 25% sits at the
  yellow middle of the color bar and 100% at the red end.
"""
from swotgat.paths import WORK
import sys
sys.path.insert(0, f"{WORK}/CONUS_storage_fork/eval/grl_figs")
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import grl_common as G
from grl_tag import TAG, tagged

OUT = G.OUT
G.SERIES_FILE = tagged("series.npz"); COL1, COL2 = G.COL1, G.COL2
def use_style():
    """Set the older plot style (grl_common.use_style)."""
    G.use_style()

OURS = "SWOT–GNN"
MODELS = ["v10pm", "MODROM-cal", "GDROM", "ISTARF", "MODROM-tr"]
RULES = MODELS[1:]
LABEL = {"v10pm": OURS, "GDROM": "GDROM", "ISTARF": "ISTARF", "MODROM-cal": "MODROM-cal", "MODROM-tr": "MODROM-tr"}
CLR = {"v10pm": "#2a78d6", "GDROM": "#eb6834", "ISTARF": "#1baf7a", "MODROM-cal": "#eda100", "MODROM-tr": "#e87ba4"}
LW = {"v10pm": 2.1}
PERIODS = ["2015-20", "2023-25"]
PERLAB = {"2015-20": "2015–20 hindcast (held out)", "2023-25": "2023–25 training period"}
PERSHORT = {"2015-20": "2015–20", "2023-25": "2023–25"}
TYPES = ["Flood control", "Irrigation", "Water supply", "Hydroelectricity", "Recreation"]
TYPELAB = {"Flood control": "Flood control", "Irrigation": "Irrigation", "Water supply": "Water supply",
           "Hydroelectricity": "Hydropower", "Recreation": "Recreation"}
GAUGE = dict(color="black", lw=1.35, solid_capstyle="butt", solid_joinstyle="miter", zorder=1)
panel = G.panel

NORM = PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0)      # sqrt scale: 25 % sits at the colour-bar middle
CMAP = mpl.colormaps["RdYlBu_r"]
PCT_TICKS = [0.05, 0.10, 0.25, 0.50, 1.0]
pct = lambda v: f"{100*v:g}%"

def sites(fname="sites_v2.csv"):
    """Return a site score table from OUT: the file `fname` with GRL_TAG inserted before its extension.

    A missing main purpose (column `use`), or the text "None", becomes "Unknown".
    """
    S = pd.read_csv(OUT/tagged(fname))       # sites_v*{GRL_TAG}.csv
    S["use"] = S.use.fillna("Unknown").replace({"None": "Unknown"})
    return S

def common(S, period, models=MODELS):
    """Return the set of DAM_IDs for which every listed model has an nRMSE in that period."""
    s = S[(S.PERIOD == period) & S.nrmse.notna()]
    return set.intersection(*[set(s[s.MODEL == m].DAM_ID) for m in models])

def med_ci(v, n=3000, seed=0):
    """Return the median of v and its 95% bootstrap interval, as (median, lower, upper).

    Non-finite values are dropped; the interval is the 2.5th to 97.5th percentile of the medians of n
    resamples. Returns three NaNs for fewer than 3 values.
    """
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 3: return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = np.median(rng.choice(v, (n, len(v)), replace=True), axis=1)
    return float(np.median(v)), *np.percentile(b, [2.5, 97.5])

# ---------------------------------------------------------------- sqrt axis
def sqrt_x(ax, lo=0.02, hi=1.0, ticks=(0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)):
    """Give the x axis a square-root scale from lo to hi, with the ticks labeled in percent."""
    ax.set_xscale("function", functions=(lambda x: np.sqrt(np.clip(x, 0, None)), lambda x: x**2))
    ax.set_xlim(lo, hi); ax.set_xticks(list(ticks)); ax.set_xticklabels([pct(t) for t in ticks])
    ax.minorticks_off()

# ---------------------------------------------------------------- map ------
def p_map(ax, S, period, model="v10pm", ids=None, field="nrmse", title=None, cbar=True,
          key=True, cbar_label="nRMSE", sub=None, s_scale=1.0, key_dx=0.115, title_x=0.085):
    """Draw a map of one model's metric at each reservoir, on the nRMSE color scale (CMAP, NORM).

    S is a site score table; period and model choose its rows and field the column, whose values are
    clipped to 0-1 for the colors. ids restricts the reservoirs, and sub is an optional function that
    returns a row mask. Dot area grows with the logarithm of capacity (times s_scale); large reservoirs
    are drawn first. cbar adds a color bar under the map and key a key to the dot sizes (key_dx apart);
    title is written at axes fraction title_x. Returns (the scatter artist, the rows drawn).
    """
    G.basemap(ax)
    s = S[(S.MODEL == model) & (S.PERIOD == period)].dropna(subset=[field, "lat"])
    if ids is not None: s = s[s.DAM_ID.isin(ids)]
    if sub is not None: s = s[sub(s)]
    x, y = G.albers(s.lon.values, s.lat.values)
    o = np.argsort(-pd.to_numeric(s.cap, errors="coerce").fillna(0).values)
    sc = ax.scatter(x[o], y[o], c=np.clip(s[field].values[o], 0, 1), cmap=CMAP, norm=NORM,
                    s=G.cap_size(s.cap.values)[o]*s_scale, linewidths=0.3, edgecolors="#333", zorder=3)
    ttl = ax.text(title_x, 1.0, title or "", transform=ax.transAxes, ha="left", va="bottom", fontsize=8)
    ax._map_title = ttl
    if cbar:
        cax = ax.inset_axes([0.50, -0.075, 0.46, 0.032])
        cb = ax.figure.colorbar(sc, cax=cax, orientation="horizontal", extend="max")
        cb.set_ticks(PCT_TICKS); cb.set_ticklabels([pct(t) for t in PCT_TICKS])
        cb.set_label(cbar_label, labelpad=1.5, fontsize=7.5)
        cb.outline.set_linewidth(.6); cb.ax.tick_params(width=.6, length=2.5, labelsize=6.8)
    if key:
        kx, ky = 0.06, -0.06
        ax.text(kx - 0.04, ky + 0.035, "capacity (MCM)", transform=ax.transAxes, fontsize=6.5,
                color="#333", ha="left", va="bottom")
        for k, (cap, lab) in enumerate([(30, "30"), (1000, "1 000"), (30000, "30 000")]):
            xx = kx + key_dx*k
            ax.scatter([xx], [ky], s=G.cap_size(np.array([cap]))[0]*s_scale, transform=ax.transAxes,
                       facecolor="white", edgecolor="#333", lw=.55, zorder=4, clip_on=False)
            ax.text(xx, ky - 0.035, lab, transform=ax.transAxes, ha="center", va="top",
                    fontsize=6.2, color="#333")
    return sc, s

def map_colorbar(fig, mappable, rect, label="nRMSE"):
    """Add a horizontal color bar with percent ticks in the figure rectangle rect, and return it."""
    cax = fig.add_axes(rect)
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", extend="max")
    cb.set_ticks(PCT_TICKS); cb.set_ticklabels([pct(t) for t in PCT_TICKS])
    cb.set_label(label, labelpad=1.5, fontsize=7.5)
    cb.outline.set_linewidth(.6); cb.ax.tick_params(width=.6, length=2.5, labelsize=6.8)
    return cb

# ---------------------------------------------------------------- CDF ------
def cdf_line(ax, v, **kw):
    """Draw the cumulative distribution of the finite values of v as a step line (nothing if none)."""
    v = np.sort(np.asarray(v, float)); v = v[np.isfinite(v)]
    if len(v): ax.step(v, np.arange(1, len(v) + 1)/len(v), where="post", **kw)

def p_cdf(ax, S, period, field, models=MODELS, ids=None, sub=None, fmt="{:.1%}", legend=True,
          loc="lower right", xlab=None, title=None, show_n=False):
    """Draw the cumulative distribution of a metric for each model, with the medians in the legend.

    S is a site score table; period chooses its rows and field the column. ids restricts the reservoirs,
    and sub is an optional function that returns a row mask. fmt formats the medians; show_n adds each
    model's n to its legend entry.
    """
    s = S[S.PERIOD == period]
    if ids is not None: s = s[s.DAM_ID.isin(ids)]
    if sub is not None: s = s[sub(s)]
    h = []
    for m in models[::-1]:
        v = s.loc[s.MODEL == m, field]
        cdf_line(ax, v, color=CLR[m], lw=LW.get(m, 1.15), zorder=5 if m == "v10pm" else 3)
    for m in models:
        v = s.loc[s.MODEL == m, field].dropna()
        lab = f"{LABEL[m]}  {fmt.format(v.median())}" + (f" (n={len(v)})" if show_n else "")
        h.append(Line2D([], [], color=CLR[m], lw=LW.get(m, 1.15), label=lab))
    ax.set_ylim(0, 1); ax.set_ylabel("cumulative fraction")
    if xlab: ax.set_xlabel(xlab)
    if title: ax.set_title(title, pad=3)
    if legend: ax.legend(handles=h, loc=loc, fontsize=6.5, borderaxespad=0.3)

# ---------------------------------------------------------------- forest ---
def p_forest(ax, S, field="nrmse", ids_by_period=None, models=RULES, better="lower", fmt=pct):
    """Draw the median paired difference between the SWOT model and each rule model, for both periods.

    Per period and rule model, the difference in `field` (SWOT model minus rule model) is taken at the
    reservoirs both score; its median is a dot and the 95% bootstrap interval a line. ids_by_period
    optionally maps each period to the reservoirs to use. better ("lower" or "higher") says which sign
    counts as the SWOT model doing better. Returns one tuple per row: (y position, period, model, median
    difference, lower, upper, n, share of reservoirs where the SWOT model is better).
    """
    rows, y = [], 0
    for per in PERIODS:
        s = S[S.PERIOD == per]
        if ids_by_period: s = s[s.DAM_ID.isin(ids_by_period[per])]
        o = s[s.MODEL == "v10pm"].set_index("DAM_ID")[field]
        for m in models:
            b = s[s.MODEL == m].set_index("DAM_ID")[field]
            j = pd.concat([o, b], axis=1, keys=["o", "b"]).dropna()
            d, lo, hi = med_ci(j.o - j.b)
            w = (j.o < j.b).mean() if better == "lower" else (j.o > j.b).mean()
            rows.append((y, per, m, d, lo, hi, len(j), w)); y += 1
        y += 0.8
    for yy, per, m, d, lo, hi, n, w in rows:
        ax.plot([lo, hi], [yy, yy], color=CLR[m], lw=1.6, solid_capstyle="butt")
        ax.plot([d], [yy], "o", color=CLR[m], ms=4.3, mec="white", mew=.5, zorder=4)
        ax.text(max(hi, lo) + 0.002, yy, f"  {OURS} lower at {100*w:.0f}%", va="center", fontsize=6.2,
                color="#555")
    ax.axvline(0, color="#333", lw=.7)
    ax.set_yticks([r[0] for r in rows]); ax.set_yticklabels([f"vs {LABEL[r[2]]}" for r in rows], fontsize=6.9)
    ax.invert_yaxis(); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    for per in PERIODS:
        yy = [r[0] for r in rows if r[1] == per]
        ax.text(0.0, min(yy) - 0.85, f"{PERLAB[per]}  (n = {rows[[r[1] for r in rows].index(per)][6]})",
                transform=ax.get_yaxis_transform(), fontsize=6.9, fontweight="bold", va="center")
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{100*v:+.0f}" if v else "0"))
    ax.set_xlabel(f"paired difference in nRMSE, {OURS} − benchmark (percentage points)")
    return rows

# ---------------------------------------------------------- dot summary ---
def p_dots(ax, S, period, field, models=MODELS, ids=None, by="use", cats=TYPES, ref=None,
           xlab=None, xlim=None, catlab=TYPELAB, show_n=True, xfmt=None):
    """Draw the median and 95% CI of a metric for each category (rows) and model (dots)."""
    s = S[S.PERIOD == period]
    if ids is not None: s = s[s.DAM_ID.isin(ids)]
    nm = len(models); off = np.linspace(-0.28, 0.28, nm) if nm > 1 else [0]
    for i, c in enumerate(cats):
        for k, m in enumerate(models):
            v = s[(s[by] == c) & (s.MODEL == m)][field]
            md, lo, hi = med_ci(v)
            if not np.isfinite(md): continue
            ax.plot([lo, hi], [i + off[k]]*2, color=CLR[m], lw=1.1, solid_capstyle="butt")
            ax.plot([md], [i + off[k]], "o", color=CLR[m], ms=3.6 if m != "v10pm" else 4.4,
                    mec="white", mew=.4, zorder=4)
    if ref is not None: ax.axvline(ref, color="#555", lw=.7, ls=(0, (4, 2)))
    n = [((s[by] == c) & (s.MODEL == models[0])).sum() for c in cats]
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([f"{catlab.get(c, c)}" + (f"  ({nn})" if show_n else "") for c, nn in zip(cats, n)],
                       fontsize=6.8)
    ax.set_ylim(len(cats) - 0.5, -0.5)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    for i in range(len(cats) - 1): ax.axhline(i + 0.5, color="#e6e6e6", lw=.6, zorder=0)
    if xlab: ax.set_xlabel(xlab)
    if xlim: ax.set_xlim(*xlim)
    if xfmt: ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(xfmt))

def model_legend(fig_or_ax, models=MODELS, gauge=False, dashed_rules=False, **kw):
    """Add a legend with one line sample per model to a figure or axes, and return it.

    gauge=True puts the gauge line first; dashed_rules=True draws the rule models' samples dashed.
    Other keyword arguments go to legend().
    """
    h = [Line2D([], [], **GAUGE, label="gauge")] if gauge else []
    h += [Line2D([], [], color=CLR[m], lw=LW.get(m, 1.15),
                 ls="-" if (m == "v10pm" or not dashed_rules) else (0, (3.2, 1.4)), label=LABEL[m])
          for m in models]
    return fig_or_ax.legend(handles=h, **kw)

# ---------------------------------------------------------- hydrograph ----
SWOT_DOT = dict(marker="o", ls="", ms=3.0, mfc="none", mec="#f07f00", mew=0.75, zorder=5)   # hollow

_SWOT = None
def swot_points(dam):
    """Return the SWOT storage-change points (2023-25) of one reservoir that the model trained on.

    They are read from the series atlas of an earlier model configuration
    (eval/compare_weekend/storage_atlas_v10pm: reservoirs.json and swot.json).
    Returns (dates, values in MCM, kind) or None when the reservoir has no points.
    kind 1 = the dv_SWOTAE storage estimate, 0 = the static-PLD fallback (PLD = SWOT Prior Lake
    Database).
    """
    global _SWOT
    import json
    if _SWOT is None:
        A = G.F/"eval/compare_weekend/storage_atlas_v10pm"
        idx = {g["id"]: g["i"] for g in json.load(open(A/"reservoirs.json"))["reservoirs"]}
        _SWOT = (idx, json.load(open(A/"swot.json")))
    idx, sw = _SWOT
    rec = sw.get(str(idx.get(dam, -1)))
    if not rec: return None
    d0 = pd.Timestamp("2023-01-01")
    dd = np.array([d0 + pd.Timedelta(days=int(k)) for k, _ in rec["d"]]); vv = np.array([v for _, v in rec["d"]], float)
    return dd, vv, int(rec["k"])

def p_hydro(ax, S, period, dam, models=("v10pm",), stats="nrmse", ylab=True, title=None, fs=6.0, swot=False):
    """Draw one reservoir's daily storage anomaly: the gauge and the listed models.

    The gauge is centered on its own mean, each model on its mean over the days it shares with the
    gauge. swot=True adds the SWOT points in 2023-25, shifted to the gauge's mean on the SWOT dates.
    The header gives the name, state, main purpose and capacity from the site table S (or `title`);
    if stats is true, each model's nRMSE and r are printed in the panel at font size fs.
    """
    t = G.dates(period)
    o, _ = G.get_series("v10pm", period, dam)
    oc = o - np.nanmean(o)
    ax.plot(t, oc, **GAUGE)
    if swot and period == "2023-25":
        sp = swot_points(dam)
        if sp is not None:
            dd, vv, _ = sp
            g = pd.Series(oc, index=t).reindex(pd.DatetimeIndex(dd))
            ok = g.notna().values
            # SWOT has no absolute datum: shift it to the gauge's mean on the SWOT dates
            shift = (g[ok].mean() - vv[ok].mean()) if ok.sum() >= 3 else -np.nanmean(vv)
            ax.plot(dd, vv + shift, **SWOT_DOT)
    for m in models:
        oo, pp = G.get_series(m, period, dam)
        if pp is None or not np.isfinite(pp).any(): continue
        ok = np.isfinite(oo) & np.isfinite(pp)
        ax.plot(t, pp - np.nanmean(pp[ok]), color=CLR[m], lw=0.95 if m == "v10pm" else 0.8,
                ls="-", solid_capstyle="butt", solid_joinstyle="miter", zorder=3 if m == "v10pm" else 2)
    q = S[(S.PERIOD == period) & (S.DAM_ID == dam)].set_index("MODEL")
    if len(q):
        g0 = q.iloc[0]
        nm = str(g0["name"]); nm = nm.title() if nm.isupper() else nm
        for long_, short in ((" Dam And Powerhouse", " Dam"), (" Lake Dam And Powerhouse", " Lake"),
                             (" And Powerhouse", ""), (" Dam And Reservoir", " Dam"), (" And Reservoir", "")):
            nm = nm.replace(long_, short)
        ttl = title or f"{nm}, {g0['state']}"
        ax.text(0.0, 1.135, ttl, transform=ax.transAxes, fontsize=7.0, fontweight="bold", va="bottom")
        use = g0["use"] if isinstance(g0["use"], str) and g0["use"] != "Unknown" else "use n/a"
        sub = f"{TYPELAB.get(use, use)} \u00b7 {g0['cap']:,.0f} MCM"
        ax.text(0.0, 1.03, sub, transform=ax.transAxes, fontsize=5.9, va="bottom", ha="left", color="#555")
        if stats:
            lines = []
            for m in models:
                if m in q.index and np.isfinite(q.loc[m, "nrmse"]):
                    lines.append((m, f"nRMSE {100*q.loc[m,'nrmse']:.0f}%  r {q.loc[m,'r']:.2f}"))
            for k, (m, txt) in enumerate(lines):
                ax.text(0.99, 0.96 - 0.12*k, txt, transform=ax.transAxes, fontsize=fs, ha="right", va="top",
                        color=CLR[m], fontweight="bold" if m == "v10pm" else None,
                        bbox=dict(fc="white", ec="none", alpha=.75, pad=0.4), zorder=6)
    ax.set_xlim(t[0], t[-1])
    ys = [np.asarray(l.get_ydata(), float) for l in ax.get_lines()]
    lo = np.nanmin([np.nanmin(y) for y in ys]); hi = np.nanmax([np.nanmax(y) for y in ys])
    R = hi - lo if hi > lo else 1
    ax.set_ylim(lo - 0.06*R, hi + (0.18 + 0.12*len(models))*R)
    if ylab: ax.set_ylabel("ΔS (MCM)", fontsize=7)
    ax.tick_params(labelsize=6.5)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator(1 if period == "2023-25" else 2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

def pick_range(S, period, k=12, ids=None, field="nrmse", min_days=900, types=None):
    """Return k reservoirs at evenly spaced quantiles of the SWOT model's `field` (label v10pm).

    The reservoirs are sorted ascending (so best to worst for nRMSE) and restricted to records of at
    least min_days days, so that each panel is legible. Returns (list of DAM_ID, their quantile
    positions). The argument types is not used.
    """
    s = S[(S.MODEL == "v10pm") & (S.PERIOD == period) & S[field].notna()]
    if ids is not None: s = s[s.DAM_ID.isin(ids)]
    s = s[s.n_days >= min_days].sort_values(field)
    idx = np.unique(np.round(np.linspace(0, len(s) - 1, k)).astype(int))
    return list(s.DAM_ID.iloc[idx]), [ (i+0.5)/len(s) for i in idx ]

# ------------------------------------------------- legible hydrograph (Fig. 4 v2) ----
ST = {"Alabama": "AL", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO", "Connecticut": "CT",
      "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
      "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
      "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
      "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
      "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
      "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
      "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"}
USE_SHORT = {"Flood control": "flood control", "Irrigation": "irrigation", "Water supply": "water supply",
             "Hydroelectricity": "hydropower", "Recreation": "recreation", "Navigation": "navigation"}

def _clean_name(nm):
    """Return a tidy dam name for titles.

    A name in capitals is put in title case; endings such as " And Powerhouse" and " And Reservoir" and
    a final " Dam" are removed.
    """
    nm = str(nm); nm = nm.title() if nm.isupper() else nm
    for a, b in ((" Lake Dam And Powerhouse", " Lake"), (" Dam And Powerhouse", " Dam"), (" And Powerhouse", ""),
                 (" Dam And Reservoir", " Dam"), (" And Reservoir", ""), (" And ", " and "), (" De Terre", " de Terre")):
        nm = nm.replace(a, b)
    return nm[:-4] if nm.endswith(" Dam") else nm      # captions name reservoirs without "Dam"

# darker shades of the model colours for text, so coloured labels stay readable on white
TXT = {"v10pm": "#1f66c1", "MODROM-cal": "#a87200", "GDROM": "#c9501f", "ISTARF": "#11875c", "MODROM-tr": "#c4547f"}

def p_hydro2(ax, S, period, dam, rule, letter, smooth=7, swot=False, fs=7.6, lw=(1.2, 1.25, 1.05), xlabels=True):
    """Draw a hydrograph of the gauge, the SWOT model and one rule model, legible at print size.

    Lines are running means over `smooth` days, for display (scores use the daily series); the header
    is kept outside the data, every year is named, and there are three y ticks. rule is the label of the
    rule model to draw and letter the panel letter. The header prints both models' nRMSE from the site
    table S. swot=True adds the SWOT points in 2023-25.
    """
    import matplotlib.ticker as mt, matplotlib.dates as mdates
    t = G.dates(period)
    sm = (lambda x: pd.Series(x, index=t).rolling(smooth, center=True, min_periods=(smooth + 1)//2).mean().values
          if smooth and smooth > 1 else np.asarray(x))
    o, _ = G.get_series("v10pm", period, dam); oc = o - np.nanmean(o)
    ax.axhline(0, color="#e0e0e0", lw=0.6, zorder=0)
    ax.plot(t, sm(oc), color=GAUGE["color"], lw=lw[0], solid_capstyle="butt", solid_joinstyle="round", zorder=1)
    for m, w, z in (("v10pm", lw[1], 3), (rule, lw[2], 2)):
        oo, pp = G.get_series(m, period, dam)
        ok = np.isfinite(oo) & np.isfinite(pp)
        ax.plot(t, sm(pp - np.nanmean(pp[ok])), color=CLR[m], lw=w, solid_capstyle="butt", solid_joinstyle="round",
                zorder=z)
    if swot and period == "2023-25":
        sp = swot_points(dam)
        if sp is not None:
            dd, vv, _ = sp
            g = pd.Series(oc, index=t).reindex(pd.DatetimeIndex(dd)); ok = g.notna().values
            shift = (g[ok].mean() - vv[ok].mean()) if ok.sum() >= 3 else -np.nanmean(vv)
            ax.plot(dd, vv + shift, **SWOT_DOT)
    q = S[(S.PERIOD == period) & (S.DAM_ID == dam)].set_index("MODEL"); g0 = q.iloc[0]
    st = ST.get(str(g0["state"]), str(g0["state"]))
    use = g0["use"] if isinstance(g0["use"], str) and g0["use"] != "Unknown" else "use n/a"
    hd = dict(xycoords="axes fraction", textcoords="offset points", va="baseline", annotation_clip=False)
    ax.annotate(f"{letter}  {_clean_name(g0['name'])}, {st}", (0, 1), xytext=(0, 12.5), fontsize=fs + 0.4,
                fontweight="bold", ha="left", **hd)
    ax.annotate(f"{USE_SHORT.get(use, use)} · {g0['cap']:,.0f} MCM", (0, 1), xytext=(0, 3.2), fontsize=fs - 0.6,
                color="#555", ha="left", **hd)
    ax.annotate(f"{OURS} {100*q.loc['v10pm', 'nrmse']:.0f}%", (1, 1), xytext=(0, 12.5), fontsize=fs,
                color=TXT["v10pm"], fontweight="bold", ha="right", **hd)
    ax.annotate(f"{LABEL[rule]} {100*q.loc[rule, 'nrmse']:.0f}%", (1, 1), xytext=(0, 3.2), fontsize=fs,
                color=TXT[rule], ha="right", **hd)
    ax.set_xlim(t[0], t[-1] + pd.Timedelta(days=1))
    ax.xaxis.set_major_locator(mdates.YearLocator(1)); ax.xaxis.set_major_formatter(mt.NullFormatter())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=7, bymonthday=1))     # year names centred in their year
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%Y") if xlabels else mt.NullFormatter())
    ax.tick_params(axis="x", which="major", length=3, width=0.6)
    ax.tick_params(axis="x", which="minor", length=0, labelsize=fs - 0.6, pad=2.5)
    ax.yaxis.set_major_locator(mt.MaxNLocator(nbins=3, min_n_ticks=3))
    ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v/1000:g}k" if abs(v) >= 1000 else f"{v:g}"))
    ax.tick_params(axis="y", labelsize=fs - 0.6, length=2.5, pad=2)
    ax.margins(y=0.08)

def fig_examples(S, period, dams, rules, swot=False, h=8.2, smooth=7, xlabels="all", fs=7.6):
    """Draw the twelve-reservoir example grid (6 x 2) and return (figure, list of axes).

    The grid is laid out in inches so that headers, tick labels and legend never collide. dams lists
    the reservoirs and rules the rule model to draw with each; h is the figure height in inches.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    W = COL2; fig = plt.figure(figsize=(W, h))
    L, R, T, B = 0.50, 0.06, 0.66, 0.22          # margins: y label + ticks, right, legend + first header, x labels
    HD, XL = 0.31, 0.19                           # header block above a panel; x tick labels below one
    nr, nc = 6, 2
    gap = HD + (XL if xlabels == "all" else 0.07); wgap = 0.44
    ph = (h - T - B - gap*(nr - 1))/nr; pw = (W - L - R - wgap*(nc - 1))/nc
    axs = []
    for k, dam in enumerate(dams):
        i, j = divmod(k, nc)
        x0 = L + j*(pw + wgap); y0 = h - T - (i + 1)*ph - i*gap
        ax = fig.add_axes([x0/W, y0/h, pw/W, ph/h]); axs.append(ax)
        p_hydro2(ax, S, period, dam, rules[k], "abcdefghijkl"[k], smooth=smooth, swot=swot, fs=fs,
                 xlabels=(xlabels == "all" or i == nr - 1))
    fig.text(0.10/W, (B + (h - T - B)/2)/h, "storage anomaly (MCM)", rotation=90, va="center", ha="left",
             fontsize=fs + 0.6)
    yl = (h - T + HD + 0.07)/h
    ha = [Line2D([], [], color=GAUGE["color"], lw=1.6, label="gauge"),
          Line2D([], [], color=CLR["v10pm"], lw=1.9, label=OURS)]
    if swot: ha.append(Line2D([], [], **{**SWOT_DOT, "ms": 4.6, "mew": 1.0}, label="SWOT observations"))
    kw = dict(fontsize=fs, handlelength=1.7, handletextpad=0.5, columnspacing=1.3, borderaxespad=0, borderpad=0,
              frameon=False)
    fig.legend(handles=ha, loc="lower left", bbox_to_anchor=(L/W, yl), ncol=len(ha), **kw)
    hb = [Line2D([], [], color=CLR[m], lw=1.7, label=LABEL[m]) for m in ("MODROM-cal", "GDROM", "ISTARF")]
    lb = fig.legend(handles=hb, loc="lower right", bbox_to_anchor=(1 - R/W, yl), ncol=3, **kw)
    bb = lb.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    fig.text(bb.x0 - 0.008, (bb.y0 + bb.y1)/2, "best rule model:", ha="right", va="center", fontsize=fs, color="#333")
    return fig, axs


# ------------------------------------------------- full ResOps record, 1981-2020 (Fig. 5) ----
# Tables of an earlier full-record scoring, in eval/overnight/fullrecord_v1: scores in 5-year blocks,
# each series centered within its block, and a table of score drift by operation similarity.
# There the label SWOT-GNN is the hindcast of an earlier model configuration (v10pm) from the extended
# ERA5-Land store; the rule models are the runs started from gauge storage on 1980-01-01
# (grades_*_g_y1980).
FR = G.F/"eval/overnight/fullrecord_v1"
NY = "normal year"
FRMAP = {"SWOT-GNN": "v10pm", "GDROM": "GDROM", "ISTARF": "ISTARF", "MODROM-cal": "MODROM-cal",
         "MODROM-tr": "MODROM-tr", "normal_year": NY}
CLR[NY], TXT[NY], LABEL[NY] = "#6b6b6b", "#555555", "normal year"
NY_LS = (0, (3.4, 1.6))

def fullrecord():
    """Return the 5-year-block scores on a fixed set of reservoirs.

    The set holds the reservoirs with High or Medium link confidence that every model scores in every
    5-year block. Reads fullrecord_site_window.csv from FR.
    Returns (rows of that set, block names, all High/Medium rows).
    """
    R = pd.read_csv(FR/"fullrecord_site_window.csv"); R = R[R.site_tier.isin(["High", "Medium"])].copy()
    R["MODEL"] = R.model.map(FRMAP)
    B = R[R.kind == "block"]; wins = sorted(B.window.unique())
    cnt = B.groupby(["DAM_ID", "MODEL"]).window.nunique().unstack()
    panel = cnt.index[(cnt == len(wins)).all(axis=1)]
    return B[B.DAM_ID.isin(panel)].copy(), wins, R

def med_ci_block(v, g, n=2000, seed=0):
    """Return the median of v with a 95% block-bootstrap interval, as (median, lower, upper).

    g gives each value's group (for example its river basin); whole groups are resampled with
    replacement, n times. Returns NaNs for fewer than 5 values.
    """
    x = pd.Series(np.asarray(v, float)); g = pd.Series(np.asarray(g)); ok = x.notna().values; x, g = x[ok], g[ok]
    if len(x) < 5: return np.nan, np.nan, np.nan
    bl = [a.values for _, a in x.groupby(g.values)]; rng = np.random.default_rng(seed)
    bs = [np.median(np.concatenate([bl[k] for k in rng.integers(0, len(bl), len(bl))])) for _ in range(n)]
    return float(np.median(x)), *np.percentile(bs, [2.5, 97.5])

def drift_quartiles():
    """Return the table ops_drift_quartiles.csv (in FR), with a MODEL column added.

    The table gives the median change in r from 2016-20, by quartile of seasonal-operation similarity.
    """
    Q = pd.read_csv(FR/"ops_drift_quartiles.csv"); Q["MODEL"] = Q.model.map(FRMAP)
    return Q
