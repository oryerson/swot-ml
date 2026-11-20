"""Report whether the rows of the re-keyed river drop list fall on days with a river target in the store.

The store is the graph v3.5 training store. This is a report only; it does not stop the pipeline. The re-keyed
drop list is the one written by rekey_lists_v35.py (the "T9b" list of the earlier node set, moved to graph v3.5
ids).

Usage: python mask_rekey_info_v35.py <v3.5 training store>

Reads (L = $SWOTGAT_WORK/CONUS_storage_fork/eval/v35_inputs/lists)
  L/drop_river_paper_min_present_t9b_v35.parquet   the re-keyed drop list
  L/node_map_lists_v35.csv                         which v3.5 nodes took over drop-list rows of a node that no
                                                   longer exists
  <store>/<basin>                                  variable dv_river

Writes
  L/work/mask_rekey_info_v35.json   (the script stops if the file exists)

Method
  For the nodes that took over rows ("rekeyed") and, as a reference, for all other nodes of the drop list
  ("surviving"): number of drop-list rows, number of those rows that fall on a day with a finite dv_river at the
  node, their ratio, and the number of nodes.
"""
from swotgat.paths import WORK
import json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

L = Path(f"{WORK}/CONUS_storage_fork/eval/v35_inputs/lists"); Z = Path(sys.argv[1])
OUT = L/"work/mask_rekey_info_v35.json"; assert not OUT.exists(), OUT
m = pd.read_parquet(L/"drop_river_paper_min_present_t9b_v35.parquet"); m["subbasin"] = m.subbasin.astype(str); m["basin"] = m.basin.astype(str)
NM = pd.read_csv(L/"node_map_lists_v35.csv", dtype=str)
rk = NM[NM.action.isin(["mapped", "mapped_partial"]) & NM.rekeyed.fillna("").str.contains("mask")]
took = set(rk.v35_id); res = {"rekeyed": [0, 0, 0], "surviving": [0, 0, 0]}   # rows, rows on a finite dv_river day, nodes
for b, g in m.groupby("basin"):
    if not (Z/b/"zarr.json").is_file(): continue
    zg = zarr.open_group(str(Z/b), mode="r"); subs = np.array([str(s) for s in zg["subbasin"][:]]); da = zg["date"]
    mm = re.fullmatch(r"days since (\d{4}-\d{2}-\d{2}).*", da.attrs["units"]); dates = pd.Timestamp(mm.group(1)) + pd.to_timedelta(np.asarray(da[:], "int64"), unit="D")
    pos = {s: j for j, s in enumerate(subs)}; dpos = {d: i for i, d in enumerate(dates)}
    for kind, gg in (("rekeyed", g[g.subbasin.isin(took)]), ("surviving", g[~g.subbasin.isin(took)])):
        cols = sorted({pos[s] for s in gg.subbasin if s in pos})
        if not cols: continue
        A = np.isfinite(zg["dv_river"].get_orthogonal_selection((slice(None), cols))); cj = {c: k for k, c in enumerate(cols)}
        hit = [bool(A[dpos[pd.Timestamp(d)], cj[pos[s]]]) for s, d in zip(gg.subbasin, gg.date) if s in pos and pd.Timestamp(d) in dpos]
        res[kind][0] += len(hit); res[kind][1] += int(sum(hit)); res[kind][2] += len(cols)
R = {k: dict(rows=v[0], rows_on_finite_dv_river_day=v[1], frac=round(v[1]/max(v[0], 1), 4), nodes=v[2]) for k, v in res.items()}
json.dump(R, open(OUT, "w"), indent=1); print(json.dumps(R, indent=1))
