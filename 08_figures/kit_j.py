"""Shared helpers for the figure scripts: rank correlations with a block bootstrap, and layout pieces.

The module builds on kit_h.py.

Figure 3 (fig3_c5d.py) uses PA, IDS, rho_ci(), prho() and TRAIN_T. The other pieces come from earlier
versions of the figure set.

Reads at import, from CONUS_storage_fork/eval/perf_attrib/out/ under $SWOTGAT_WORK (written by the
scripts in 07_evaluation/c_scoring/drivers/):
  vars.parquet      one row per reservoir of the skill-driver analysis; only its index is used (IDS)
  ops_8120.parquet  operations variables over water years 1981-2020 (read into _OPS, not used further here)
  blocks.csv        the spatial block of each reservoir (pa_blocks.py): reservoirs that drain to the same
                    river node of at most 50,000 km2 upstream area share a block (blocks of fewer than
                    three reservoirs are merged into a neighbor)

Method:
  - Block bootstrap. DRAWS holds 1,000 resamples of the reservoirs, made once with random seed 0. Each
    resample draws as many blocks as there are, with replacement, and takes every reservoir of each drawn
    block. Whole blocks are resampled because reservoirs on one river are not independent.
  - rho_ci(x, y) returns (Spearman rho, lower bound, upper bound, n). x and y are ordered like IDS. rho uses
    the reservoirs where both are finite; the bounds are the 2.5th and 97.5th percentile of rho over DRAWS.
  - prho(x, y, z) returns the partial Spearman correlation of x and y given z: all three are ranked, the
    ranks of x and of y are each regressed on the ranks of z, and the two residuals are correlated.
  - same(a, b, tol) compares two nested structures of dicts, lists and numbers (for example a panel's
    numbers and a stored JSON file); numbers may differ by tol relative to max(1, |b|).
  - head_band() draws a header band over a panel: tan for SWOT-GAT's training period (the SWOT Period),
    grey for a hindcast, with an optional panel letter, a bold title and a second line.
  - cdf_medians() writes the four models' medians as colored labels inside a cumulative-distribution
    panel. The table label `v10pm` is SWOT-GAT (the paper's model: mean of two seeds).
"""
from swotgat.paths import WORK
import json, numpy as np, pandas as pd, matplotlib as mpl
from pathlib import Path
from scipy.stats import spearmanr, rankdata
import kit_h as K
from kit_h import jw, plt, FS, SR, EC, EG, M4, TXT, LAB, C
from matplotlib.patches import Rectangle
PA = Path(f"{WORK}/CONUS_storage_fork/eval/perf_attrib/out")
TRAIN, TRAIN_T, TRAIN_BG, HIND_BAND = C["TRAIN"], C["TRAIN_T"], "#fcf5ea", "#ecebe7"
def same(a, b, tol=1e-12):
    """Return True when two nested structures of dicts, lists and numbers agree.

    Dict keys are compared as strings (so a structure can be checked against its JSON copy). Two numbers
    agree when they differ by at most tol relative to max(1, |b|); anything else must be equal.
    """
    if isinstance(a, dict): return isinstance(b, dict) and set(map(str, a)) == set(map(str, b)) and all(same(a[k], b[str(k)] if str(k) in b else b[k], tol) for k in a)
    if isinstance(a, (list, tuple)): return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float, np.floating, np.integer)) and isinstance(b, (int, float)): return abs(float(a) - float(b)) <= tol*max(1.0, abs(float(b)))
    return a == b
def head_band(fig, I, x, w, y, title, sub, train, letter=None):
    """Draw the period header band over a panel.

    The band lies over the panel that starts at figure fraction x and is w wide; its lower edge is y
    inches above the figure bottom and it is 0.36 inches high (I converts inches to a fraction of the
    figure height). It is tan when train is true (SWOT-GAT's training period, the SWOT Period) and grey
    for a hindcast. title is written in bold and sub on a second line; an optional panel letter goes
    before the title.
    """
    fig.add_artist(Rectangle((x - 0.012, I(y)), w + 0.012, I(0.36), transform=fig.transFigure, facecolor=TRAIN if train else HIND_BAND, edgecolor="none", zorder=0))
    tx = x
    if letter: fig.text(x - 0.004, I(y + 0.215), letter, fontsize=9, fontweight="bold", va="baseline"); tx = x + 0.018
    fig.text(tx, I(y + 0.215), title, fontsize=FS, fontweight="bold", color=TRAIN_T if train else jw.INK, va="baseline")
    fig.text(tx, I(y + 0.075), sub, fontsize=FS, color=TRAIN_T if train else jw.INK2, va="baseline")
def cdf_medians(ax, med, x=0.03, fmt="{:.2f}"):
    """Write the four models' medians as colored labels inside a cumulative-distribution panel.

    med maps each model label of M4 to its median; x is the left edge of the labels in axes fractions
    and fmt the number format. SWOT-GAT's line (`v10pm`) is bold.
    """
    ax.text(x, 0.96, "median", transform=ax.transAxes, fontsize=FS, color=jw.INK3, va="top")
    for k, m in enumerate(M4):
        ax.text(x, 0.85 - 0.105*k, f"{LAB[m]} {fmt.format(med[m])}".replace("-", "−"), transform=ax.transAxes, color=TXT[m], fontsize=FS,
                fontweight="bold" if m == "v10pm" else "normal", va="top")
# ---------------- explanatory page 8 scatter, recomputed with pa_casey_extra.py's code (fig4_i.py panel c)
_V = pd.read_parquet(PA/"vars.parquet"); IDS = _V.index
_OPS = pd.read_parquet(PA/"ops_8120.parquet").reindex(IDS); _BL = pd.read_csv(PA/"blocks.csv").set_index("DAM_ID").reindex(IDS)
_blocks = _BL.block.values; _ub = np.unique(_blocks); _bi = {b: np.where(_blocks == b)[0] for b in _ub}; _rng = np.random.default_rng(0)
DRAWS = [np.concatenate([_bi[b] for b in _rng.choice(_ub, len(_ub), replace=True)]) for _ in range(1000)]
def rho_ci(x, y):
    """Return Spearman's rho of x and y with a block-bootstrap 95% interval.

    x and y are ordered like IDS. rho uses the reservoirs where both are finite. Returns
    (rho, lower bound, upper bound, n): the bounds are the 2.5th and 97.5th percentile of rho over the
    resamples in DRAWS, and n is the number of reservoirs used.
    """
    x, y = np.asarray(x, float), np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y); r = spearmanr(x[ok], y[ok])[0]; bs = []
    for d in DRAWS:
        s = d[ok[d]]; bs.append(spearmanr(x[s], y[s])[0])
    return r, np.percentile(bs, 2.5), np.percentile(bs, 97.5), int(ok.sum())
def prho(x, y, z):
    """Return the partial Spearman correlation of x and y given z.

    All three arrays are ranked over the entries where all are finite; the ranks of x and of y are each
    regressed on the ranks of z (with an intercept), and the two residuals are correlated.
    """
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z); rx, ry, rz = (rankdata(v[ok]) for v in (x, y, z)); Z = np.c_[np.ones(ok.sum()), rz]
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]; ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]; return float(np.corrcoef(ex, ey)[0, 1])
