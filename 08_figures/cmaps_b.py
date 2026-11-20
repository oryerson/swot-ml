"""Candidate color scales for the two maps, from an earlier version of the figure set.

The paper's figures take their color scales from cmaps_d.py. This module stays in the folder because
figs_only_c5d.py imports it (and uses nothing from it).

Method: every candidate is a sequential ramp through several hues, light at the low end and dark at the
high end, with a wide lightness range, so that low and high values are easy to tell apart. No candidate is
diverging. trunc(name, a, b) cuts the matplotlib colormap `name` to the fraction a..b of its range.

  NSE_CANDS  "viridis" (reversed), "ylgnbu" (YlGnBu from 0.15), "cividis" (reversed)
  VR_CANDS   for the variability ratio (VR): "plasma" (reversed, up to 0.92), "ylorrd" (YlOrRd from 0.15),
             "magma" (reversed, 0.05 to 0.88)

Usage: import it. The environment variables NSE_CMAP and VR_CMAP choose a candidate by its key;
the defaults are "viridis" and "ylorrd". The chosen scales are NSE_CMAP and VR_CMAP.
"""
import os, numpy as np, matplotlib as mpl
def trunc(name, a, b):
    """Return the matplotlib colormap `name` cut to the fraction a..b of its range (256 colors)."""
    c = mpl.colormaps[name]; return mpl.colors.LinearSegmentedColormap.from_list(f"{name}_{a:g}_{b:g}", c(np.linspace(a, b, 256)))
NSE_CANDS = {"viridis": trunc("viridis_r", 0.0, 1.0), "ylgnbu": trunc("YlGnBu", 0.15, 1.0), "cividis": trunc("cividis_r", 0.0, 1.0)}
VR_CANDS = {"plasma": trunc("plasma_r", 0.0, 0.92), "ylorrd": trunc("YlOrRd", 0.15, 1.0), "magma": trunc("magma_r", 0.05, 0.88)}
NSE_PICK = os.environ.get("NSE_CMAP", "viridis"); VR_PICK = os.environ.get("VR_CMAP", "ylorrd")   # defaults: NSE viridis, VR YlOrRd (two different scales, because one figure shows both)
NSE_CMAP, VR_CMAP = NSE_CANDS[NSE_PICK], VR_CANDS[VR_PICK]
