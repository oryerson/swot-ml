"""Map projection, marker sizes, paths and an older plot style, shared by the figure modules.

What the paper's four figures use (directly or through jw.py and kit_h.py):
  albers(lon, lat)   Albers equal-area conic projection of the conterminous US (standard parallels 29.5
                     and 45.5 N, origin 96 W, 23 N); returns x, y on a unit sphere
  cap_size(cap)      marker area from the logarithm of reservoir capacity in million cubic meters (MCM),
                     on a fixed scale from CAP_LO = 10 to CAP_HI = 30,000 MCM
  STATES_JSON        the state outlines, CONUS_testing/data/us_states_rings.json under $SWOTGAT_WORK
  F, OUT             CONUS_storage_fork/ and CONUS_storage_fork/eval/grl_figs/ under $SWOTGAT_WORK
  SERIES_FILE        name of the daily series file; grl2.py and jw_fig_tt_e.py reset it

The rest serves figure sets of an earlier model configuration and is kept because other modules import
this one: use_style() (the paper's figures use jw.use_style instead), the model lists, labels and colors,
sites(), rescale(), series(), get_series(), dates(), basemap(), cdf(), med_ci() and caption_page().
These read sites.csv, rescale.csv and the series file from OUT. In the labels, `v10pm` and "SWOT–GNN"
stand for the SWOT model, which is SWOT-GAT in the paper.

Style of use_style(), following AGU/GRL practice: sans-serif throughout, 7-9 pt type, thin
spines, no top/right spines on Cartesian panels, Okabe-Ito colors (safe for
all common color-vision deficiencies), Type-42 fonts so the PDF text stays
editable, and single/double column widths of 95 mm and 190 mm (COL1, COL2).

Notes: importing this module selects matplotlib's non-interactive Agg backend.
"""
from swotgat.paths import WORK
import base64, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

F   = Path(f"{WORK}/CONUS_storage_fork")
OUT = F/"eval"/"grl_figs"
STATES_JSON = Path(f"{WORK}/CONUS_testing/data/us_states_rings.json")

COL1, COL2 = 3.74, 7.48          # inches: 95 mm and 190 mm

def use_style():
    """Set matplotlib's defaults to the older plot style described in the module docstring."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2,
        "legend.frameon": False, "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5, "legend.labelspacing": 0.32,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.solid_capstyle": "round",
        "savefig.dpi": 400, "figure.dpi": 130, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    })

# ---- models -----------------------------------------------------------------
OURS = "SWOT–GNN"
MODELS = ["v10pm", "MODROM-cal", "ISTARF", "GDROM", "MODROM-tr"]
LABEL  = {"v10pm": OURS, "GDROM": "GDROM", "ISTARF": "ISTARF",
          "MODROM-cal": "MODROM-cal", "MODROM-tr": "MODROM-tr"}
CLR    = {"v10pm": "#000000", "GDROM": "#0072B2", "ISTARF": "#009E73",
          "MODROM-cal": "#CC79A7", "MODROM-tr": "#9A9A9A"}
LW     = {"v10pm": 2.0}
PERIODS = ["2015-20", "2023-25"]
PERLAB  = {"2015-20": "2015–20  held-out backtest",
           "2023-25": "2023–25  SWOT era"}

def panel(ax, s, dx=-0.16, dy=1.04, **kw):
    """Write the bold panel label s at axes fraction (dx, dy), by default above the top-left corner."""
    ax.text(dx, dy, s, transform=ax.transAxes, fontweight="bold",
            fontsize=9, va="bottom", ha="left", **kw)

# ---- data -------------------------------------------------------------------
def sites():
    """Return the rows of sites.csv (in OUT) that belong to the models in MODELS."""
    S = pd.read_csv(OUT/"sites.csv")
    return S[S.MODEL.isin(MODELS)].copy()

def common_set(S, period, models):
    """Return the set of DAM_IDs that have a row for every listed model in that period."""
    s = S[S.PERIOD == period]
    ids = None
    for m in models:
        k = set(s.loc[s.MODEL == m, "DAM_ID"])
        ids = k if ids is None else (ids & k)
    return ids or set()

def rescale():
    """Return the table rescale.csv (in OUT)."""
    return pd.read_csv(OUT/"rescale.csv")

_SER = None
SERIES_FILE = "series.npz"      # grl2.py points this at series{GRL_TAG}.npz (see grl_tag.py)
def series():
    """Return the daily series file OUT/SERIES_FILE, opened on the first call and kept."""
    global _SER
    if _SER is None: _SER = np.load(OUT/SERIES_FILE)
    return _SER

def get_series(model, period, dam):
    """Return the daily gauge and model storage of one reservoir from the series file.

    period "2015-20" reads the "P2" arrays, any other period the "P1" arrays (2023-2025). Returns
    (gauge, prediction), or (None, None) when the reservoir is not in the file.
    """
    Z = series(); dams = list(Z["dams"])
    if dam not in dams: return None, None
    i = dams.index(dam); pk = "P2" if period == "2015-20" else "P1"
    return Z[f"{model}|{pk}|obs"][i], Z[f"{model}|{pk}|pred"][i]

def dates(period):
    """Return the days of a period: 2,192 from 2015-01-01 for "2015-20", else 1,096 from 2023-01-01."""
    return (pd.date_range("2015-01-01", periods=2192, freq="D") if period == "2015-20"
            else pd.date_range("2023-01-01", periods=1096, freq="D"))

# ---- metric geometry --------------------------------------------------------
# r = Pearson correlation, a = variability ratio (model sd / gauge sd), both for series with their mean removed
def nse_of(r, a):
    """Return the NSE of the mean-removed series from correlation r and variability ratio a."""
    return 2*a*r - a**2
def kge_of(r, a):
    """Return 1 minus the distance of the point (r, a) from (1, 1)."""
    return 1 - np.sqrt((r-1)**2 + (a-1)**2)
def sep_of(r):
    """Return 2r - 1, the NSE at a = 1.

    Matching the gauge's variance (a = 1) raises the NSE of a model with a < 1 only if a < 2r - 1.
    """
    return 2*r - 1

# ---- Albers equal-area conic, CONUS -----------------------------------------
_D2R = np.pi/180; _p1, _p2 = 29.5*_D2R, 45.5*_D2R; _lon0, _lat0 = -96*_D2R, 23*_D2R
_n = (np.sin(_p1) + np.sin(_p2))/2
_C = np.cos(_p1)**2 + 2*_n*np.sin(_p1)
_rho0 = np.sqrt(_C - 2*_n*np.sin(_lat0))/_n
def albers(lon, lat):
    """Return x, y of the Albers equal-area conic projection of the conterminous US, on a unit sphere.

    lon and lat are in degrees (east and north positive). Standard parallels 29.5 and 45.5 N, origin
    96 W, 23 N.
    """
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    t = _n*(lon*_D2R - _lon0); rho = np.sqrt(_C - 2*_n*np.sin(lat*_D2R))/_n
    return rho*np.sin(t), _rho0 - rho*np.cos(t)

def basemap(ax, lw=0.4, fc="#f2f2f2", ec="#b8b8b8"):
    """Draw the states of the conterminous US (fill fc, outline ec of width lw) and set the map limits."""
    rings = json.load(open(STATES_JSON))
    for ring in rings:
        a = np.asarray(ring, float)
        x, y = albers(a[:, 0], a[:, 1])
        ax.add_patch(MplPolygon(np.column_stack([x, y]), closed=True,
                                facecolor=fc, edgecolor=ec, lw=lw, zorder=0))
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.375, 0.36); ax.set_ylim(0.042, 0.505)   # full CONUS incl. FL/TX

CAP_LO, CAP_HI = 10.0, 30000.0       # MCM; fixed so maps and legends share one scale
def cap_size(cap, lo=3.0, hi=34.0):
    """Return marker areas scaled by the logarithm of reservoir capacity (MCM) on a fixed scale.

    Capacities from CAP_LO to CAP_HI map linearly in log10 to areas from lo to hi; values outside are
    clipped, and a missing capacity gets lo. The scale is fixed so that the few very large reservoirs
    do not swamp the map and every legend agrees.
    """
    c = np.asarray(pd.to_numeric(cap, errors="coerce"), float)
    z = (np.log10(np.clip(c, CAP_LO, CAP_HI)) - np.log10(CAP_LO)) / \
        (np.log10(CAP_HI) - np.log10(CAP_LO))
    return lo + np.nan_to_num(z, nan=0.0)*(hi - lo)

# ---- small helpers ----------------------------------------------------------
def cdf(ax, v, **kw):
    """Draw the cumulative distribution of the finite values of v as a step line (nothing if none)."""
    v = np.sort(np.asarray(v, float)); v = v[np.isfinite(v)]
    if not len(v): return
    ax.step(v, np.arange(1, len(v)+1)/len(v), where="post", **kw)

def med_ci(v, n=4000, seed=0):
    """Return the median of v and its 95% bootstrap interval, as (median, lower, upper).

    Non-finite values are dropped; the interval is the 2.5th to 97.5th percentile of the medians of n
    resamples. Returns three NaNs for fewer than 3 values.
    """
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 3: return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = np.median(rng.choice(v, (n, len(v)), replace=True), axis=1)
    return np.median(v), *np.percentile(b, [2.5, 97.5])

def caption_page(pdf, title, body, width=COL2):
    """Add a page with a title and caption text to an open PDF, so that the PDF carries its captions."""
    fig = plt.figure(figsize=(width, 2.6)); fig.patch.set_facecolor("white")
    fig.text(0.02, 0.93, title, fontsize=9.5, fontweight="bold", va="top")
    fig.text(0.02, 0.80, body, fontsize=8, va="top", wrap=True,
             linespacing=1.55)
    pdf.savefig(fig); plt.close(fig)
