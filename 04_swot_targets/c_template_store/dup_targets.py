"""Find river gauge nodes whose SWOT river series is identical to another node's series.

Two nodes can be matched to the same SWOT reach, for example a gauge node on a tributary that was given the
reach of the main stem. Such a node would be trained on another river's signal. This script lists the cases;
make_t9_mask.py turns the list into a mask.

Usage: python dup_targets.py      (no arguments)

Reads (under $SWOTGAT_WORK)
  CONUS_storage_fork/testsets/rivers/sites.csv    river test gauges and their nodes
  CONUS/zarr_store/<basin>                        first data store; variable d_wse_river (SWOT level anomaly)
  CONUS/metadata/subbasins_conus.parquet          upstream area of each node
  CONUS/basin_lists/storage/train_nodes_g.csv     training nodes (for the column in_train)

Writes
  CONUS_storage_fork/eval/swotqc/duplicate_river_targets.csv   one row per gauge node with at least 5 SWOT days:
      node, site, n_swot, n_twins, twins (up to 5 ids), up_self and up_twin (upstream area in km2 of the node and
      of its first twin), in_train

Method
  Period 2023-01-01 to 2025-12-31. A node's fingerprint is the set of its observed days together with the values
  on those days, rounded to 4 decimals. Each gauge node is compared with every other node of its basin; nodes
  with the same fingerprint are its twins. Nodes with fewer than 5 observed days are not compared.
  The script also prints the ratio of upstream areas, gauge node / twin. A ratio below 0.5 means the gauge node
  is much smaller than the river whose reach it carries.
"""
from swotgat.paths import WORK
import numpy as np, pandas as pd, xarray as xr
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork"); C = Path(f"{WORK}/CONUS")
S = pd.read_csv(F/"testsets/rivers/sites.csv", dtype={"site_no": str, "node": str, "basin": str})
sub = pd.read_parquet(C/"metadata/subbasins_conus.parquet", columns=["nextdown", "uparea_km2"]); sub.index = sub.index.astype(str)
train = set(pd.read_csv(C/"basin_lists/storage/train_nodes_g.csv").subbasin.astype(str))
rows = []
for b, g in S.groupby("basin"):
    z = C/"zarr_store"/b
    if not z.exists(): continue
    ds = xr.open_zarr(z, consolidated=(z/".zmetadata").exists())
    if "d_wse_river" not in ds: continue
    X = ds["d_wse_river"].sel(date=slice("2023-01-01", "2025-12-31")).load().transpose("date", "subbasin")
    names = X.subbasin.values.astype(str); V = X.values
    obs = np.isfinite(V); cnt = obs.sum(0)
    key = {}
    for j in np.flatnonzero(cnt >= 5):                     # fingerprint: observed days + rounded values
        k = (tuple(np.flatnonzero(obs[:, j])), tuple(np.round(V[obs[:, j], j], 4)))
        key.setdefault(hash(k), []).append(j)
    for n in g.node:
        if n not in set(names): continue
        j = int(np.flatnonzero(names == n)[0])
        if cnt[j] < 5: continue
        k = hash((tuple(np.flatnonzero(obs[:, j])), tuple(np.round(V[obs[:, j], j], 4))))
        twins = [names[i] for i in key.get(k, []) if i != j]
        rows.append(dict(node=n, site=S.station_nm[S.node == n].iloc[0], n_swot=int(cnt[j]), n_twins=len(twins), twins=";".join(twins[:5]),
                         up_self=sub.uparea_km2.get(n), up_twin=sub.uparea_km2.get(twins[0]) if twins else np.nan, in_train=n in train))
D = pd.DataFrame(rows); D.to_csv(F/"eval/swotqc/duplicate_river_targets.csv", index=False)
d = D[D.n_twins > 0]
print(f"gauge nodes with SWOT (>=5 days): {len(D)} | with a SWOT series identical to another node's: {len(d)} ({len(d)/len(D):.1%})")
if len(d):
    r = (d.up_self / d.up_twin)
    print(f"  upstream-area ratio gauge/twin: median {r.median():.2f}; < 0.5 (gauge much smaller, e.g. tributary on a mainstem reach): {int((r < 0.5).sum())}; 0.8-1.25 (same river): {int(r.between(0.8, 1.25).sum())}")
    print(d.sort_values("up_self").head(25).to_string(index=False))
