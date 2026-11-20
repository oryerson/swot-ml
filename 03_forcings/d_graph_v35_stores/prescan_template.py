"""Count the non-finite ERA5-Land values of the template store, as the reference for gate (b).

The template store is the 2020-2026 training store on the earlier node set, from which
build_store_v35.py copies the forcing of unchanged sub-basins. Gate (b) of checks_v35.py requires
that the graph v3.5 store has no non-finite forcing value, and reports this scan next to its own
count.

Reads
    $SWOTGAT_WORK/CONUS/zarr_store_vd2026_halfexact    the template store (only read)

Writes
    $SWOTGAT_WORK/CONUS_storage_fork/eval/v35_inputs/forcing/work/template_nan_scan.json
    For each of the 10 variables <variable>_mean: `nan_values` (number of non-finite values),
    `nodes_any_nan` (list of "basin:sub-basin:count" for every sub-basin with at least one) and
    `dates_any_nan` (number of distinct date positions with at least one).

Usage: python prescan_template.py        (no arguments)
"""
from swotgat.paths import WORK
import json, numpy as np, xarray as xr
from pathlib import Path
T = Path(f"{WORK}/CONUS/zarr_store_vd2026_halfexact"); F = Path(f"{WORK}/CONUS_storage_fork/eval/v35_inputs/forcing")
VARS = [f"{v}_mean" for v in ["ssrd", "strd", "ro", "sro", "e", "tp", "sp", "d2m", "t2m", "fal"]]
out = {v: dict(nan_values=0, nodes_any_nan=[], dates_any_nan=0) for v in VARS}; dates_any = {v: set() for v in VARS}
for b in sorted(p.name for p in T.iterdir() if (p/"zarr.json").is_file()):
    ds = xr.open_zarr(T/b, consolidated=True); subs = ds.subbasin.values.astype(str)
    for v in VARS:
        a = ds[v].transpose("date", "subbasin").values; m = ~np.isfinite(a)
        out[v]["nan_values"] += int(m.sum()); out[v]["nodes_any_nan"] += [f"{b}:{s}:{int(k)}" for s, k in zip(subs[m.any(0)], m.sum(0)[m.any(0)])]
        dates_any[v] |= set(np.flatnonzero(m.any(1)).tolist())
for v in VARS: out[v]["dates_any_nan"] = len(dates_any[v])
json.dump(out, open(F/"work/template_nan_scan.json", "w"), indent=1); print(json.dumps({v: {k: (x if k != "nodes_any_nan" else x[:10]) for k, x in d.items()} for v, d in out.items()}, indent=1))
