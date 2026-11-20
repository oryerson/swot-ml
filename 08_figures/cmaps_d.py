"""Color scales of the paper's maps (figure set version d).

  NSE_CMAP  matplotlib's YlGnBu cut to the fraction 0.25-1.0 of its range; light = low NSE.
            Used for the Figure 2a map and for the points of Figure 3c.
  VR_CMAP   matplotlib's viridis, reversed; light = low variability ratio (VR). Used for the Figure 3a map.

ramp(name, a, b) samples the matplotlib colormap `name` from fraction a to fraction b (b < a reverses it).
NSE_PICK and VR_PICK are the short names the figure scripts store in their JSON files.
NSE_LABEL and VR_LABEL are the names mk_captions_d.py prints in the captions.
"""
import numpy as np, matplotlib as mpl
def ramp(name, a, b):
    """Return the matplotlib colormap `name` sampled from fraction a to fraction b (b < a reverses it)."""
    return mpl.colors.LinearSegmentedColormap.from_list(f"{name}_{a:g}_{b:g}", mpl.colormaps[name](np.linspace(a, b, 256)))
NSE_PICK, VR_PICK = "ylgnbu", "viridis"
NSE_CMAP, VR_CMAP = ramp("YlGnBu", 0.25, 1.0), ramp("viridis", 1.0, 0.0)
NSE_LABEL, VR_LABEL = "YlGnBu 0.25–1.0", "viridis, reversed"
