"""Plot style, base map, text layout helpers and the site score table shared by all figure scripts.

The look follows the figures of Wang et al. (SWOT lake storage phenology): dark grey text and axes, two
hue families (blue and red), classed color ramps, labels written next to the data instead of legends, thin
paired box plots, uncertainty as translucent bands, bold lowercase panel letters. Sizes follow Geophysical
Research Letters: COL1 = 95 mm and COL2 = 190 mm wide.

Main pieces
  use_style()                 matplotlib settings: sans-serif 7 pt, thin axes, no top or right spine,
                              Type 42 fonts so that text in the PDF stays editable
  basemap(ax)                 pale map of the conterminous US (Albers projection), white state lines
  letter(), seq_text(), seq_text_ax()   panel letters and runs of text in several colors
  ecdf(v)                     sorted finite values and their cumulative shares
  qa(fig)                     layout check: overlapping text, text that is too small, anything off the canvas
  sites_table()               the per-reservoir score table (see its docstring)
  thin_box(), classed(), ladder(), donut(), save(), grow() and the MAP_* color ramps serve earlier figure
  versions; the paper's four figures do not call them.

Reads
  conus_outline_albers.json (next to this file) and the state outlines named by grl_common.STATES_JSON,
  both in basemap(). sites_table() reads sites_2010_g.csv and sites_2325span_g.csv next to this file, and
  CONUS_storage_fork/eval/grl_figs/sites_v5_g.csv under $SWOTGAT_WORK.

Environment (sites_table only): HIND ("2015" by default, or "2010") and ERA ("span" by default, or "full").

Notes: importing this module puts CONUS_storage_fork/eval/grl_figs under $SWOTGAT_WORK at the front of the
module search path, then imports grl_common. It also selects matplotlib's non-interactive Agg backend.
"""
from swotgat.paths import WORK
import sys; sys.path.insert(0, f"{WORK}/CONUS_storage_fork/eval/grl_figs")
import json, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
import grl_common as G

COL1, COL2 = 3.74, 7.48                       # GRL: 95 mm / 190 mm
INK, INK2, INK3, RULE = "#1a1a1a", "#4d4d4d", "#8a8a8a", "#d9d9d9"
LAND, BORDER = "#ededed", "#ffffff"
# color ramps after Wang et al., Fig. 1: 8 classes each (light -> dark), sampled from that figure
BLUE8 = ["#bae6f2", "#8fcfe9", "#6face4", "#4c90cb", "#3476c0", "#3252a8", "#24409f", "#2a2d88"]
RED8  = ["#f6e1e2", "#f4bbb8", "#e99a91", "#d4806c", "#cc6757", "#b24636", "#a53223", "#901619"]
BLUE, RED = "#3476c0", "#cc6757"              # line / mark colours (mid ramp steps)
BLUE_T, RED_T = "#8fcfe9", "#f4bbb8"          # light fills
# diverging ramp after Wang et al., Fig. 4: orange -> warm grey -> teal
DIV = ["#c05525", "#da7e2f", "#ef963e", "#f7b15b", "#f6d09a", "#ced0d1", "#9aced7", "#76bcc7", "#4a9eb3", "#2e87aa", "#1f6a8f"]
# categorical hues after Wang et al., Fig. 3
ARCH = {"navy": "#143d64", "teal": "#1e7ea2", "sky": "#3fb4e6", "pink": "#e4b3cf", "lime": "#a2c636", "green": "#36a24e", "orange": "#d28a2a"}

def use_style():
    """Set matplotlib's defaults for the figure set.

    Sans-serif 7 pt text (tick labels and legends 6.5 pt), thin dark axes without top or right spine,
    no legend frame, and Type 42 fonts so that text in the PDF stays editable.
    """
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "legend.frameon": False, "axes.linewidth": 0.55, "axes.edgecolor": INK,
        "xtick.major.width": 0.55, "ytick.major.width": 0.55, "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "xtick.minor.size": 1.3, "ytick.minor.size": 1.3, "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.labelpad": 2.0,
        "xtick.major.pad": 1.6, "ytick.major.pad": 1.6, "lines.linewidth": 0.9, "lines.solid_capstyle": "round",
        "savefig.dpi": 450, "figure.dpi": 150, "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "custom",
        "mathtext.it": "Nimbus Sans:italic", "mathtext.rm": "Nimbus Sans", "mathtext.bf": "Nimbus Sans:bold"})

def letter(ax, s, x=-0.02, y=1.0, **kw):
    """Write the bold panel letter s just outside the top-left corner of the axes."""
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom", **kw)

def basemap(ax, states=True):
    """Draw a pale map of the conterminous US with white state lines (states=False leaves the lines out).

    Rivers and graph nodes are not drawn, to keep the map quiet under the reservoir dots.
    The merged national outline is drawn first so that slivers between state polygons never show through.
    """
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch
    from pathlib import Path
    for prt in json.load(open(Path(__file__).parent/"conus_outline_albers.json")):
        verts, codes = [], []
        for ring in [prt["ext"]] + prt["holes"]:
            r = np.asarray(ring, float); verts += r.tolist(); codes += [MPath.MOVETO] + [MPath.LINETO]*(len(r) - 2) + [MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), facecolor=LAND, edgecolor="none", zorder=0))
    if states:
        for ring in json.load(open(G.STATES_JSON)):
            a_ = np.asarray(ring, float); x, y = G.albers(a_[:, 0], a_[:, 1])
            ax.add_patch(MplPolygon(np.column_stack([x, y]), closed=True, facecolor=LAND, edgecolor=BORDER, lw=0.4, zorder=0.5))
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-0.372, 0.357); ax.set_ylim(0.045, 0.50)

# One soft blue-to-red ramp for maps of earlier figure versions (the paper's maps use cmaps_d.py).
# MAP_SEQ: low = muted blue, high = muted brick red (for nRMSE: red = larger error); MAP_SEQ_R for metrics where
# higher is better (such as r); MAP_DIV is the same ramp used two-sided (variability ratio around 1, bias
# around 0): blue below, red above.
_SOFT = ["#3f6fae", "#6f9bc9", "#a8c5e0", "#dfe8ef", "#f3ebe3", "#efc6ae", "#df937c", "#c05a52"]
MAP_SEQ = mpl.colors.LinearSegmentedColormap.from_list("soft_br", _SOFT)
MAP_SEQ_R = MAP_SEQ.reversed()
MAP_DIV = MAP_SEQ
# pale tints from the same family for an ecoregion background (EPA Level I groups, in the order ETF, GP, NWFM,
# NAD, PAC: Eastern Temperate Forests, Great Plains, Northwestern Forested Mountains, North American Deserts,
# Pacific coast)
MAP_TINTS = ["#dce7f1", "#f3eedc", "#e2e1f0", "#f4e2cf", "#eed3dc"]

def classed(values, bins, colors):
    """Return one color per value: the color of the class that the bin edges `bins` put the value in.

    Values beyond the last class get the last color.
    """
    idx = np.clip(np.digitize(values, bins), 0, len(colors) - 1); return [colors[i] for i in idx]

def thin_box(ax, data, pos, color, light, width=0.22, horiz=False, median_color="white", lw=0.6):
    """Draw a thin box after Wang et al. at position pos (horiz=True draws it lying down).

    A filled box from the 25th to the 75th percentile without outline (color), a median line (white by
    default) and thin whiskers to the 5th and 95th percentile (light), no caps and no outliers. Draws
    nothing for fewer than 3 finite values.
    """
    d = np.asarray(data, float); d = d[np.isfinite(d)]
    if len(d) < 3: return
    q05, q25, q50, q75, q95 = np.percentile(d, [5, 25, 50, 75, 95])
    if not horiz:
        ax.plot([pos, pos], [q05, q25], color=light, lw=lw, zorder=1, solid_capstyle="butt"); ax.plot([pos, pos], [q75, q95], color=light, lw=lw, zorder=1, solid_capstyle="butt")
        ax.add_patch(Rectangle((pos - width/2, q25), width, q75 - q25, facecolor=color, edgecolor="none", zorder=2))
        ax.plot([pos - width/2, pos + width/2], [q50, q50], color=median_color, lw=0.9, zorder=3, solid_capstyle="butt")
    else:
        ax.plot([q05, q25], [pos, pos], color=light, lw=lw, zorder=1, solid_capstyle="butt"); ax.plot([q75, q95], [pos, pos], color=light, lw=lw, zorder=1, solid_capstyle="butt")
        ax.add_patch(Rectangle((q25, pos - width/2), q75 - q25, width, facecolor=color, edgecolor="none", zorder=2))
        ax.plot([q50, q50], [pos - width/2, pos + width/2], color=median_color, lw=0.9, zorder=3, solid_capstyle="butt")

def ecdf(v):
    """Return the sorted finite values of v and their cumulative shares (1/n, 2/n ... 1)."""
    v = np.sort(np.asarray(v, float)); v = v[np.isfinite(v)]; return v, np.arange(1, len(v) + 1)/len(v)

def ladder(ax, x, y, cols, labels, header=None, dx=0.034, dy=0.042, sw=0.03, sh=0.036, fs=6.0, header_rot=35):
    """Draw a legend "ladder" after Wang et al.

    cols holds one list of colors per hue family; each becomes a column of swatches, with the class
    labels to the right and an optional rotated header above each column. x, y is the lower-left corner;
    all positions and sizes are in axes fractions.
    """
    n = len(labels)
    for j, cl in enumerate(cols):
        for i, c in enumerate(cl):
            ax.add_patch(Rectangle((x + j*dx, y + (n - 1 - i)*dy), sw, sh, transform=ax.transAxes, facecolor=c, edgecolor="none", clip_on=False, zorder=5))
    for i, l in enumerate(labels):
        ax.text(x + len(cols)*dx + 0.008, y + (n - 1 - i)*dy + sh/2, l, transform=ax.transAxes, fontsize=fs, va="center", ha="left", color=INK, zorder=5)
    if header:
        for j, h in enumerate(header):
            ax.text(x + j*dx + sw/2, y + n*dy + 0.005, h, transform=ax.transAxes, fontsize=fs, rotation=header_rot, ha="left", va="bottom", color=INK2, zorder=5)

def donut(fig, rect, shares, colors, center_text, fs=6.0):
    """Add a ring chart of the shares in the figure rectangle rect, with text in its center.

    Returns the new axes.
    """
    a = fig.add_axes(rect)
    a.pie(shares, colors=colors, startangle=90, counterclock=False, wedgeprops=dict(width=0.30, edgecolor="white", lw=0.9))
    a.text(0, 0, center_text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.1); a.set_aspect("equal")
    return a

def seq_text(fig, x, y, parts, fs=6.0, gap=0.007, ha="left", va="baseline"):
    """Write (text, colour, weight) segments one after another in figure coordinates.

    Each segment's width is measured with the renderer, so multi-coloured keys never collide. The run
    starts at x (ha='right' ends it at x); gap is the space between segments in figure fractions.
    Returns the x position after the last segment.
    """
    r = fig.canvas.get_renderer(); W = fig.bbox.width
    tmp = [fig.text(0, 0, t, fontsize=fs, fontweight=w, alpha=0) for t, c, w in parts]
    widths = [t_.get_window_extent(renderer=r).width / W for t_ in tmp]
    for t_ in tmp: t_.remove()
    total = sum(widths) + gap*(len(parts) - 1); xx = x - total if ha == "right" else x
    for (t, c, w), wd in zip(parts, widths):
        fig.text(xx, y, t, fontsize=fs, color=c, fontweight=w, va=va); xx += wd + gap
    return xx

def seq_text_ax(ax, x, y, parts, fs=6.0, gap=0.012, va="baseline", ha="left"):
    """Write (text, colour, weight) segments one after another in axes-fraction coordinates.

    This is seq_text for an axes; widths are measured with the renderer. apply_aspect() is called first:
    an equal-aspect axes (every map) shrinks to its drawn box only at draw time, and measuring the
    undrawn, wider box squeezes the run. Returns the x position after the last segment.
    """
    fig = ax.figure; r = fig.canvas.get_renderer(); ax.apply_aspect(); W = ax.get_window_extent(renderer=r).width
    tt = [ax.text(0, y, t, transform=ax.transAxes, fontsize=fs, color=c, fontweight=w, va=va) for t, c, w in parts]
    wd = [t_.get_window_extent(renderer=r).width / W for t_ in tt]
    xx = x - (sum(wd) + gap*(len(parts) - 1)) if ha == "right" else x
    for t_, w_ in zip(tt, wd):
        t_.set_x(xx); xx += w_ + gap
    return xx

def qa(fig, tag="", min_font=6.0, shrink=0.6):
    """Check the layout of a hand-placed figure and return the issues found.

    Reports text boxes that overlap other text, scatter markers or connector lines; text under min_font
    pt; and anything drawn off the canvas (the canvas IS the GRL width, so nothing may spill). Text boxes
    are shrunk by `shrink` pt on every side before the overlap test. Give an artist gid='qa-skip' to
    exempt an intended overlap. Prints a summary line labeled with `tag` and one line per issue, and
    returns the issues as a list of strings.
    """
    from matplotlib.text import Text
    from matplotlib.collections import PathCollection
    from matplotlib.patches import ConnectionPatch
    fig.canvas.draw(); r = fig.canvas.get_renderer(); px = fig.dpi/72.0; FB = fig.bbox
    skip = set()
    for a in fig.axes:                                   # tick labels outside the view interval exist but are never drawn
        for axis in (a.xaxis, a.yaxis):
            lo, hi = sorted(axis.get_view_interval()); eps = 1e-9*max(1.0, abs(hi - lo))
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                loc = tk.get_loc()
                if loc is None or not (lo - eps <= loc <= hi + eps) or not tk.get_visible(): skip.update((id(tk.label1), id(tk.label2)))
    T = []
    for t in fig.findobj(Text):
        if id(t) in skip or t.get_gid() == "qa-skip" or not t.get_visible() or not t.get_text().strip(): continue
        if t.get_alpha() is not None and t.get_alpha() == 0: continue
        bb = t.get_window_extent(renderer=r)
        if bb.width >= 0.5 and bb.height >= 0.5: T.append((t, bb))
    hits = []; nm = lambda t: repr(t.get_text().replace("\n", " ")[:30])
    off = lambda bb: max(FB.x0 - bb.x0, bb.x1 - FB.x1, FB.y0 - bb.y0, bb.y1 - FB.y1)/fig.dpi
    for t, bb in T:
        if t.get_fontsize() < min_font - 1e-6: hits.append(f"font {t.get_fontsize():.1f} pt: {nm(t)}")
        if off(bb) > 0.005: hits.append(f"off canvas by {off(bb):.2f} in: {nm(t)}")
    sb = [(t, bb.padded(-shrink*px)) for t, bb in T]
    for i in range(len(sb)):
        for j in range(i + 1, len(sb)):
            if sb[i][1].overlaps(sb[j][1]): hits.append(f"text/text: {nm(sb[i][0])} x {nm(sb[j][0])}")
    for pc in fig.findobj(PathCollection):
        offs = np.asarray(pc.get_offsets(), float)
        if not pc.get_visible() or offs.size == 0 or pc.get_gid() == "qa-skip": continue
        xy = pc.get_offset_transform().transform(offs); sz = np.asarray(pc.get_sizes(), float)
        rad = np.sqrt(np.resize(sz if sz.size else np.array([36.0]), len(xy)))/2*px
        if pc.get_clip_on() and pc.axes is not None:
            ab = pc.axes.bbox; m = (xy[:, 0] >= ab.x0) & (xy[:, 0] <= ab.x1) & (xy[:, 1] >= ab.y0) & (xy[:, 1] <= ab.y1); xy, rad = xy[m], rad[m]
        for t, bb in sb:
            h = (xy[:, 0] + rad > bb.x0) & (xy[:, 0] - rad < bb.x1) & (xy[:, 1] + rad > bb.y0) & (xy[:, 1] - rad < bb.y1)
            if h.any(): hits.append(f"text/marker: {nm(t)} covers {int(h.sum())} marker(s)")
    for cp in fig.findobj(ConnectionPatch):
        try: v = cp._get_path_in_displaycoord()[0].vertices
        except Exception: continue
        seg = np.concatenate([np.linspace(v[k], v[k + 1], 80) for k in range(len(v) - 1)])
        for t, bb in sb:
            if ((seg[:, 0] > bb.x0) & (seg[:, 0] < bb.x1) & (seg[:, 1] > bb.y0) & (seg[:, 1] < bb.y1)).any(): hits.append(f"text/connector: {nm(t)}")
    for a in fig.axes:
        ab = a.get_tightbbox(r)
        if ab is not None and off(ab) > 0.005: hits.append(f"axes off canvas by {off(ab):.2f} in: {a.get_label() or type(a).__name__}")
    print(f"[qa {tag}] canvas {fig.get_figwidth():.2f} x {fig.get_figheight():.2f} in, {len(T)} texts, {len(hits)} issue(s)")
    for h in hits: print("   ", h)
    return hits

def save(fig, out, stem):
    """Check the layout, then save <stem>.pdf and a 200-dpi <stem>.png in the folder `out`.

    Both files have the exact canvas size (190 mm wide): no tight crop, so the page width cannot creep.
    """
    qa(fig, stem)
    fig.savefig(out/f"{stem}.pdf"); fig.savefig(out/f"{stem}.png", dpi=200)

def grow(fig, H0, HN):
    """Make room at the bottom of a finished layout by raising the canvas height from H0 to HN inches.

    Every axes and every figure-level text and patch keeps its size in inches and moves up by HN - H0,
    so a new row can go underneath without re-tuning the old fractions.
    """
    DY = HN - H0; S = H0/HN; Y = lambda y: (y*H0 + DY)/HN
    fig.set_size_inches(fig.get_figwidth(), HN)
    for a in fig.axes:
        p = a.get_position(original=True); a.set_position([p.x0, Y(p.y0), p.width, p.height*S])
    for t in fig.texts:
        x, y = t.get_position(); t.set_position((x, Y(y)))
    for p in fig.patches:
        if hasattr(p, "get_y") and hasattr(p, "set_height"): p.set_y(Y(p.get_y())); p.set_height(p.get_height()*S)
        elif hasattr(p, "get_y"): p.set_y(Y(p.get_y()))

ERA_S = "SWOT span, Jul 2023–Jun 2026"          # label for the SWOT Period (table key "2023-25") scored on each reservoir's SWOT span
def sites_table(hind=None):
    """Return the per-reservoir score table: one row per reservoir, model and period.

    hind == "2010" (argument, or environment variable HIND; what the paper's figure scripts pass): the
    table is sites_2010_g.csv (2010-2020 hindcast, scored by score_2010.py) followed by
    sites_2325span_g.csv (SWOT Period, each reservoir scored on its SWOT span), both read from this file's
    folder. With ERA=full the SWOT Period rows are instead the "2023-25" rows of sites_v5_g.csv, scored
    on all of 2023-2025.
    Otherwise (HIND=2015, the default): the table is CONUS_storage_fork/eval/grl_figs/sites_v5_g.csv, the
    2015-2020 and 2023-2025 scores of an earlier model configuration, computed with the same definitions.
    sites_v5_g.csv is read in every case.

    Returns (table, hindcast period key, hindcast label, short label, file suffix).
    """
    import os
    from pathlib import Path
    hind = hind or os.environ.get("HIND", "2015")
    base = Path(f"{WORK}/CONUS_storage_fork/eval/grl_figs")
    S = pd.read_csv(base/"sites_v5_g.csv")
    if hind == "2010":
        H = pd.read_csv(Path(__file__).resolve().parent/"sites_2010_g.csv")
        # by default the SWOT Period is scored on each reservoir's SWOT span (first to last SWOT pass); ERA=full takes all of 2023-25 from sites_v5_g.csv
        E = S[S.PERIOD == "2023-25"] if os.environ.get("ERA", "span") == "full" else pd.read_csv(Path(__file__).resolve().parent/"sites_2325span_g.csv")
        return pd.concat([H, E], ignore_index=True), "2010-20", "2010–20 hindcast", "2010–20", "_2010"
    return S, "2015-20", "2015–20 hindcast", "2015–20", ""
