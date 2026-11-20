"""Reader for the SWOT Period arrays of the series file (used by score_x26span.py).

Reads the .npz file named by ipcfg.SERIES (environment variable IP_SERIES), which must have been
written by `ip_series.py --p1x`. Its "P1x" arrays hold, for each model and dam, the daily gauge
storage ("<model>|P1x|obs") and the model's storage ("<model>|P1x|pred") on
T26 = 2023-01-01 to 2026-09-30. Values are raw, in million cubic meters (MCM), with no mean removed.
The model key "v10pm" is the label these files use for SWOT-GAT (the paper's model: mean of two
seeds).

get_x(model, dam) returns (gauge, prediction) as float arrays on T26, or (None, None) when the dam
is not in the file. A rule model's prediction is NaN after the last day of its run.

Copies of this file with the same code sit next to the scripts that import it.
"""
import numpy as np, pandas as pd, ipcfg
T26 = pd.date_range("2023-01-01", "2026-09-30", freq="D")
_Z = {}
def _z():
    """Open the series file once, check that its P1x dates equal T26, and index the dams by row."""
    if not _Z:
        z = np.load(ipcfg.SERIES); assert "P1x|dates" in z.files, f"{ipcfg.SERIES} has no P1x arrays (ip_series.py --p1x)"
        assert (pd.DatetimeIndex(z["P1x|dates"]) == T26).all()
        _Z.update(z=z, dams={int(d): i for i, d in enumerate(z["dams"])})
    return _Z
def get_x(model, dam):
    """Return (gauge, prediction) of one model at one dam as float arrays on T26.

    Both are None when the dam is not in the series file.
    """
    Z = _z(); i = Z["dams"].get(int(dam))
    if i is None: return None, None
    for k in (f"{model}|P1x|obs", f"{model}|P1x|pred"):          # an NpzFile re-reads a member on every access: keep each array once
        if k not in Z: Z[k] = Z["z"][k]
    return Z[f"{model}|P1x|obs"][i].astype(float), Z[f"{model}|P1x|pred"][i].astype(float)
