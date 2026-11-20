"""Find the reservoirs where ISTARF storage is pinned at capacity or at empty.

At a few reservoirs the published ISTARF parameters keep simulated storage full or empty almost
every day. The figures leave five such reservoirs out of every rule-model panel (EX; see
08_figures/excl_d.py). This script documents that with numbers for every dam.

Usage: bin/run 07_evaluation/c_scoring/full_record/pinned_check.py      (no arguments)

Reads
  $SWOTGAT_WORK/GRADES/ISTARF-CONUS.csv   capacity per dam (column GRanD_CAP_MCM)
  and, relative to $SWOTGAT_WORK/CONUS_storage_fork:
  predictions/grades_istarf3c_g_y1980/full_series.parquet   the ISTARF run used for 1981-2020
  predictions/grades_istarf3c_g26h0/full_series.parquet     the ISTARF run used for the SWOT Period
  eval/overlap50/score/sites_2010_g.csv                     only for the High/Medium reservoir list

Method
  For every dam of each run: share_full = share of days with storage at or above 99.5% of
  capacity; share_empty = share of days at or below 0.5% of capacity. A dam is `pinned` in a run
  when either share is at least 0.95.

Writes eval/grl_figs/intermediate/c5_o50_20261113/score/pinned_check.csv (run, DAM_ID, cap, n,
share_full, share_empty, smin, smax, pinned). Prints the rows of the five excluded reservoirs, the
pinned dams per run, and those among them of confidence tier High or Medium.
"""
from swotgat.paths import WORK
import pandas as pd, numpy as np
from pathlib import Path
R = Path(f"{WORK}"); F = R/"CONUS_storage_fork"; B = F/"eval/grl_figs/intermediate/c5_o50_20261113"
IC = pd.read_csv(R/"GRADES/ISTARF-CONUS.csv"); print("ISTARF-CONUS columns:", list(IC.columns)[:25])
gid = [c for c in IC.columns if c.upper() in ("GRAND_ID", "GRAND_ID_", "DAM_ID")][0]; cap = IC.drop_duplicates(gid).set_index(gid)["GRanD_CAP_MCM"]
rows = []
for tag in ("istarf3c_g_y1980", "istarf3c_g26h0"):
    d = pd.read_parquet(F/f"predictions/grades_{tag}/full_series.parquet", columns=["DAM_ID", "date", "storage_mcm"])
    for k, g in d.groupby("DAM_ID"):
        c = cap.get(k, np.nan); s = g.storage_mcm.astype(float)
        rows.append(dict(run=tag, DAM_ID=int(k), cap=c, n=len(s), share_full=float((s >= 0.995*c).mean()) if c > 0 else np.nan, share_empty=float((s <= 0.005*c).mean()) if c > 0 else np.nan,
                         smin=float(s.min()), smax=float(s.max())))
P = pd.DataFrame(rows); P["pinned"] = (P.share_full >= 0.95) | (P.share_empty >= 0.95); P.to_csv(B/"score/pinned_check.csv", index=False)
EX = [144, 1095, 1125, 1790, 1916]
print(P[P.DAM_ID.isin(EX)].round(3).to_string(index=False))
print("pinned (>=95% of days at either end):", {r: sorted(P[(P.run == r) & P.pinned].DAM_ID.tolist()) for r in P.run.unique()})
HM = pd.read_csv(F/"eval/overlap50/score/sites_2010_g.csv"); hm = set(HM[HM.tier.isin(["High", "Medium"])].DAM_ID)
print("pinned and High/Medium:", {r: sorted(set(P[(P.run == r) & P.pinned].DAM_ID) & hm) for r in P.run.unique()})
