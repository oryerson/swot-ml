"""Compare the output of the unmodified HALF release script with that of half_batch.py, row by row.

HALF is the lake outlier filter of Trudel et al. release_replay.py runs its unmodified release script on the stored
version-D download, in shards. This script joins the shards into one table and checks that half_batch.py, which
the pipeline uses, gives the same result.

Usage: python compare_replay.py TAG        (run with the HALF environment; TAG is the --tag given to
                                            release_replay.py, by default "replay")

Reads (under $SWOTGAT_WORK/CONUS_storage_fork/eval/)
  half_vd_review/TAG/shard_*/df_Hydrocron_HALF_v1_LakeSP_vD.csv   output of release_replay.py
  swotext/half_obs_vD.parquet                                     output of half_batch.py

Writes
  half_vd_review/TAG/release_obs_vD.parquet   all shards in one table (read by build_halfexact.py)

Method
  Rows are matched on (lake_id, index_col), where index_col is the row number HALF gives each observation of a
  lake. The script prints the share of matched rows with equal time, wse, filter_flag (HALF's keep flag) and
  wse_adjusted (level after HALF's cross-pass adjustment, tolerance 1e-9 m), and the share of lakes with equal
  filter_attempt (strict, lenient, Tukey or none), intra_cycle_flag and n_while. It lists the lakes with any
  difference. It only reports; it does not stop on a difference.
"""
from swotgat.paths import WORK
import sys, glob, numpy as np, pandas as pd
R = f"{WORK}/CONUS_storage_fork/eval/half_vd_review"; S = f"{WORK}/CONUS_storage_fork/eval/swotext"
tag = sys.argv[1]
parts = [pd.read_csv(f, low_memory=False) for f in sorted(glob.glob(f"{R}/{tag}/shard_*/df_Hydrocron_HALF_v1_LakeSP_vD.csv"))]
A = pd.concat(parts, ignore_index=True); A["lake_id"] = A.lake_id.astype("int64").astype(str); A["datetime"] = pd.to_datetime(A.datetime)
A.to_parquet(f"{R}/{tag}/release_obs_vD.parquet", index=False)
B = pd.read_parquet(f"{S}/half_obs_vD.parquet"); B["lake_id"] = B.lake_id.astype(str)
B = B[B.lake_id.isin(set(A.lake_id))]
print(f"rows: release {len(A):,} ({A.lake_id.nunique()} lakes) vs half_batch {len(B):,} ({B.lake_id.nunique()} lakes)")
k = ["lake_id", "index_col"]
M = A.merge(B, on=k, how="outer", suffixes=("_r", "_b"), indicator=True)
print("row match:", M._merge.value_counts().to_dict())
m = M[M._merge == "both"]
print("same time:", np.mean(np.isclose(m.time_r, m.time_b)), " same wse:", np.mean(np.isclose(m.wse_r, m.wse_b)))
print("filter_flag equal:", np.mean(m.filter_flag_r == m.filter_flag_b), " kept release", int(m.filter_flag_r.sum()), "kept batch", int(m.filter_flag_b.sum()))
wa = (m.wse_adjusted_r.isna() & m.wse_adjusted_b.isna()) | np.isclose(m.wse_adjusted_r, m.wse_adjusted_b, atol=1e-9, rtol=0)
print("wse_adjusted equal:", np.mean(wa), " max |diff|", np.nanmax(np.abs(m.wse_adjusted_r - m.wse_adjusted_b)) if wa.mean() < 1 else 0)
la = m.groupby("lake_id").agg(ar=("filter_attempt_r", "first"), ab=("filter_attempt_b", "first"), ir=("intra_cycle_flag_r", "first"), ib=("intra_cycle_flag_b", "first"),
                               nr=("n_while_r", "first"), nb=("n_while_b", "first"))
print("filter_attempt equal (lakes):", np.mean(la.ar == la.ab), " intra_cycle_flag equal:", np.mean(la.ir == la.ib), " n_while equal:", np.mean(la.nr == la.nb))
print("release attempts:", la.ar.value_counts().to_dict())
bad = m[(m.filter_flag_r != m.filter_flag_b) | ~wa]
if len(bad): print("lakes with any difference:", bad.lake_id.nunique(), bad.lake_id.value_counts().head(10).to_dict())
