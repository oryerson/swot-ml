"""Read the daily 2023-2026 gauge and model storage series from the series file (ipcfg.SERIES).

Reads: the .npz file named by ipcfg.SERIES (environment variable IP_SERIES). It must hold the "P1x" arrays
that 07_evaluation/c_scoring/ip_series.py writes with its --p1x option: for every model, "<model>|P1x|obs"
(gauge storage) and "<model>|P1x|pred" (model storage), one row per reservoir of the array "dams" and one
column per day of T26 (2023-01-01 to 2026-09-30, checked against "P1x|dates").

Usage: get_x(model, dam) returns (gauge, prediction) as float arrays on T26, in million cubic meters (MCM)
and not centered, or (None, None) when the reservoir is not in the file. `model` is the label used in the
file: "v10pm" is SWOT-GAT (the paper's model: mean of two seeds); the rule models are "GDROM", "ISTARF"
and "MODROM-cal".

Notes:
  - The rule models' predictions stop before T26 ends and are NaN after their last day. That day depends on
    the rule runs the series file was built from: 2025-12-31 for the 2023-2025 runs, 2026-06-12 (the end of
    the GRADES-hydroDL inflow) for the 2026 runs the SWOT Period scorer uses.
  - The SWOT Period scorer (score_x26span.py, stage 07) calls this module. The figure scripts do not.
"""
import numpy as np, pandas as pd, ipcfg
T26 = pd.date_range("2023-01-01", "2026-09-30", freq="D")
_Z = {}
def _z():
    """Open the series file on first use and return the cache (the file and a DAM_ID -> row lookup)."""
    if not _Z:
        z = np.load(ipcfg.SERIES); assert "P1x|dates" in z.files, f"{ipcfg.SERIES} has no P1x arrays (ip_series.py --p1x)"
        assert (pd.DatetimeIndex(z["P1x|dates"]) == T26).all()
        _Z.update(z=z, dams={int(d): i for i, d in enumerate(z["dams"])})
    return _Z
def get_x(model, dam):
    """Return the daily gauge and model storage of one reservoir on T26.

    model is the label used in the series file ("v10pm" is SWOT-GAT; "GDROM", "ISTARF", "MODROM-cal");
    dam is the DAM_ID. Returns (gauge, prediction) as float arrays in million cubic meters, not centered,
    or (None, None) when the reservoir is not in the file.
    """
    Z = _z(); i = Z["dams"].get(int(dam))
    if i is None: return None, None
    for k in (f"{model}|P1x|obs", f"{model}|P1x|pred"):          # an NpzFile re-reads a member on every access: keep each array once
        if k not in Z: Z[k] = Z["z"][k]
    return Z[f"{model}|P1x|obs"][i].astype(float), Z[f"{model}|P1x|pred"][i].astype(float)
