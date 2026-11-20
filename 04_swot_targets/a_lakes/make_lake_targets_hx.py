"""Write the lake target table for the data stores from the HALF-exact version-D lake record.

"HALF-exact" is the lake record of build_halfexact.py: SWOT LakeSP version D filtered with HALF (the lake outlier
filter of Trudel et al.), including HALF's temporal-coverage rule.

Usage: python make_lake_targets_hx.py      (no arguments; writes next to this script, which is deployed to
                                            CONUS_storage_fork/eval/v21_retrain/store/)

Reads (under $SWOTGAT_WORK/CONUS_storage_fork/)
  eval/half_vd_review/lakesp_vD_halfexact_graph.parquet   the lake record (in the paper's run: 3,224 lakes,
                                                          311,091 lake-days); not modified
  eval/swotext/lake_targets_vd2026.parquet                the target table built before the coverage rule
                                                          (03_forcings/c_extension_2026/store_inputs_vd.py);
                                                          used only for the check

Writes (next to this script)
  lake_targets_vd2026_halfexact.parquet   columns subbasin (lake node id), date, dv_SWOTAE, dv_staticPLD,
                                          primary_source
  lakes_removed_vs_vd2026.csv             lakes that are in the earlier table but not in this one

Method
  Same construction as the earlier table. Storage change is in million cubic meters.
    dv_SWOTAE     storage change from the SWOT area-elevation curve where the lake has one, else from the
                  3D-LAKES curve; primary_source says which (SWOTAE, 3DLAKES or none)
    dv_staticPLD  level change x the constant lake area of the SWOT Prior Lake Database; the model's fallback
                  target where dv_SWOTAE is missing
  Rows with neither value are dropped.
  Checks (assertions): every row is in the earlier table with identical values and source, and a kept lake has
  lost no days. The only difference from the earlier table is therefore the set of lakes.

Notes
  No further quality screen is applied to lake targets: HALF alone decides which lake-days are used. The storage
  curves are the version-C curves of the HALF authors (see build_vd_half.py).
"""
from swotgat.paths import WORK
from pathlib import Path

import numpy as np
import pandas as pd

F = Path(f"{WORK}/CONUS_storage_fork"); OUT = Path(__file__).resolve().parent
X = pd.read_parquet(F/"eval/half_vd_review/lakesp_vD_halfexact_graph.parquet"); X["lake_id"] = X.lake_id.astype(str)
T = pd.DataFrame({"subbasin": X.lake_id, "date": pd.to_datetime(X.date), "dv_SWOTAE": X.dv_SWOTAE.fillna(X.dv_3DLAKES), "dv_staticPLD": X.dv_staticPLD,
                  "primary_source": np.where(X.dv_SWOTAE.notna(), "SWOTAE", np.where(X.dv_3DLAKES.notna(), "3DLAKES", "none"))})
T = T[T.dv_SWOTAE.notna() | T.dv_staticPLD.notna()].reset_index(drop=True)
old = pd.read_parquet(F/"eval/swotext/lake_targets_vd2026.parquet")
m = T.merge(old, on=["subbasin", "date"], how="left", suffixes=("", "_old"), indicator=True)
assert (m._merge == "both").all(), "a HALF-exact lake-day is not in the vD2026 targets"
for c in ("dv_SWOTAE", "dv_staticPLD"):
    assert np.array_equal(m[c].values, m[f"{c}_old"].values, equal_nan=True), c
assert (m.primary_source == m.primary_source_old).all()
gone = sorted(set(old.subbasin) - set(T.subbasin)); kept_old = old[old.subbasin.isin(set(T.subbasin))]
assert len(kept_old) == len(T), "a kept lake lost days"
T.to_parquet(OUT/"lake_targets_vd2026_halfexact.parquet", index=False)
pd.Series(gone, name="subbasin").to_csv(OUT/"lakes_removed_vs_vd2026.csv", index=False)
print(f"HALF-exact lake targets: {len(T):,} lake-days at {T.subbasin.nunique()} lakes ({T.date.min().date()}..{T.date.max().date()}); "
      f"vd2026 had {len(old):,} at {old.subbasin.nunique()}; removed {len(gone)} lakes / {len(old) - len(T):,} lake-days; kept rows identical")
print("by primary source:", T.groupby("subbasin").primary_source.agg(lambda s: s.mode().iat[0]).value_counts().to_dict())
