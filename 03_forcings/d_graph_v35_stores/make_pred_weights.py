"""Select the rows of the graph v3.5 weight matrix that need an ERA5-Land extraction for 1980-2019.

Reads (<forcing> = $SWOTGAT_WORK/CONUS_storage_fork/eval/v35_inputs/forcing)
    <forcing>/work/weights.npz, <forcing>/work/subbasins.csv   weights of all v3.5 sub-basins
                                                               (weights_v35.py)
    <forcing>/work/node_map.csv                                copy or extract per sub-basin
                                                               (node_map.py)

Writes
    <forcing>/work_pred/weights.npz     the selected rows, unchanged, in the same format
    <forcing>/work_pred/subbasins.csv   the selected rows of subbasins.csv, with a column `role`
                                        ("extract" or "validation_copy")
    and creates the folder <forcing>/work_pred/years for extract_pred_v35.py

Usage: python make_pred_weights.py        (no arguments)

Method
    The selected rows are
    - every sub-basin with action "extract" in the node map; the prediction stores take its
      forcing from the new extraction;
    - 1,000 sub-basins with action "copy", drawn at random without replacement (NumPy generator,
      seed 1980). Their forcing is still copied from the template store. They are extracted only
      so that checks_pred_v35.py (gate d2) can compare the extraction with the template.
    The years 2020-2026 are not affected: they are already extracted for all sub-basins in
    <forcing>/work/years.
"""
from swotgat.paths import WORK
import numpy as np, pandas as pd
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork/eval/v35_inputs/forcing"); W = F/"work"; P = F/"work_pred"
(P/"years").mkdir(parents=True, exist_ok=True)
Z = np.load(W/"weights.npz"); SB = pd.read_csv(W/"subbasins.csv", dtype=str)
NM = pd.read_csv(W/"node_map.csv", dtype={"subbasin": str}, keep_default_na=False).set_index("subbasin")
assert list(SB.subbasin) == list(NM.index)
ex = np.flatnonzero((NM.action == "extract").values); cp = np.flatnonzero((NM.action == "copy").values)
val = np.sort(np.random.default_rng(1980).choice(cp, 1000, replace=False)); rows = np.sort(np.concatenate([ex, val]))
ptr, cell, w = Z["ptr"], Z["cell"], Z["w"]; cells, ws, nptr = [], [], [0]          # read each npz member once (indexing Z[...] in the loop would reload it)
for i in rows:
    cells.append(cell[ptr[i]:ptr[i + 1]]); ws.append(w[ptr[i]:ptr[i + 1]]); nptr.append(nptr[-1] + ptr[i + 1] - ptr[i])
np.savez(P/"weights.npz", ptr=np.array(nptr, np.int64), cell=np.concatenate(cells), w=np.concatenate(ws), nlat=Z["nlat"], nlon=Z["nlon"], lat=Z["lat"], lon=Z["lon"])
S = SB.iloc[rows].copy(); S["role"] = np.where(np.isin(rows, ex), "extract", "validation_copy"); S.to_csv(P/"subbasins.csv", index=False)
print(f"rows {len(rows)} = extract {len(ex)} + validation {len(val)}; weights {nptr[-1]:,}")
