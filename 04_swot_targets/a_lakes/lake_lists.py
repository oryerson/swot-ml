"""List the graph's lake nodes and collect the version-C daily lake series for them.

Usage: python lake_lists.py      (no arguments)

Reads
  $SWOTGAT_WORK/CONUS/metadata/subbasins_conus.parquet   sub-basin table; lake nodes have is_reservoir set, and
                                                         a lake node's id is its lake_id in the SWOT Prior Lake
                                                         Database (PLD)
  daily_ts_from_LakeSPvC/basin_<region>_daily_timeseries.csv, in the raw-data archive configured as NAS_WATER:
      daily lake level (wse) and storage change (dv) from SWOT LakeSP version C, filtered with HALF and delivered
      by the HALF authors, one file per Pfafstetter level-3 region. The code names these files and their
      contents `jida`.

Writes (to $SWOTGAT_WORK/CONUS_storage_fork/eval/swotext/)
  lakes_all.txt              every lake node id, one per line (the list hc_pull.py downloads)
  jida_graph_lakes.parquet   the version-C daily rows of the graph's lakes (the reference series for
                             build_vd_half.py)
  lakes_sample.txt, jida_sample.parquet
                             a random sample of up to 80 lakes and their version-C rows, used to test that the
                             HALF filter run here matches the delivered series

Method
  The regions to read are the first three digits of the lake ids. A missing region file is skipped.
  The sample is drawn from lakes with more than 10 values of dv_SWOTAE (storage change from the SWOT
  area-elevation curve): up to 3 lakes per region, then 80 of those at random (fixed seeds).
"""
from swotgat.paths import NAS_WATER, WORK
import numpy as np, pandas as pd, glob
OUT = f"{WORK}/CONUS_storage_fork/eval/swotext"
J = f"{NAS_WATER}/cjgleason/ted/LakeSPvC_ts_for_Ted/daily_ts_from_LakeSPvC"
s = pd.read_parquet(f"{WORK}/CONUS/metadata/subbasins_conus.parquet", columns=["is_reservoir", "outlet_id"])
s.index = s.index.astype(str); lakes = s[s.is_reservoir.astype(bool)].index
open(f"{OUT}/lakes_all.txt", "w").write("\n".join(lakes)); print("graph lake nodes", len(lakes))
rows = []
for p3 in sorted(set(l[:3] for l in lakes)):
    f = f"{J}/basin_{p3}_daily_timeseries.csv"
    try: d = pd.read_csv(f, dtype={"lake_id": str})
    except FileNotFoundError: continue
    d["lake_id"] = d.lake_id.str.replace(r"\.0$", "", regex=True); rows.append(d[d.lake_id.isin(lakes)])
D = pd.concat(rows); D["date"] = pd.to_datetime(D.date)                 # the files hold ISO dates (their readme says mm/dd/yyyy)
have = D.groupby("lake_id").agg(n_wse=("wse", "count"), n_ae=("dv_SWOTAE", "count"))
print("graph lakes in Jida files", len(have), "| with dv_SWOTAE", int((have.n_ae > 0).sum()))
cand = have[(have.n_ae > 10)].index.to_series()
rng = np.random.default_rng(0); samp = cand.groupby(cand.str[:3]).apply(lambda g: g.sample(min(len(g), 3), random_state=0)).values
samp = rng.choice(samp, min(80, len(samp)), replace=False)
open(f"{OUT}/lakes_sample.txt", "w").write("\n".join(samp))
D[D.lake_id.isin(samp)].to_parquet(f"{OUT}/jida_sample.parquet", index=False); D.to_parquet(f"{OUT}/jida_graph_lakes.parquet", index=False)
print("sample", len(samp), "lakes from", len(set(x[:3] for x in samp)), "pfaf-3 basins")
