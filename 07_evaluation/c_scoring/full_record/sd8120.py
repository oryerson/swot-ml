"""Standard deviation of gauge storage over 1981-2020 per reservoir (denominator of training coverage).

Training coverage in Figure 3c is sd(SWOT storage targets) / sd(gauge storage). This script
computes the gauge standard deviation over 1981-01-01 to 2020-12-31.

Usage: bin/run 07_evaluation/c_scoring/full_record/sd8120.py      (no arguments)
  Run it after drivers/fig3c_sd.py and drivers/irr_harm.py.

Reads (paths relative to $SWOTGAT_WORK/CONUS_storage_fork)
  testsets/regime3_full_1980_2020/truth_clean.parquet   the 1980-2020 gauge test set
  eval/overlap50/fig3c_sd/all_sites_harm.csv            drivers/irr_harm.py: sd_g10 and sd_swot

Method
  sd_g8120 = sample standard deviation (pandas default) of the gauge storage on the days of the
  window whose `use` flag is set; NaN when the window has fewer than 10 such days. This is how
  drivers/fig3c_sd.py computes sd_g10 for 2010-2020.
  Check: the same function on 2010-01-01 to 2020-12-31 (sd_g10_rebuilt) must reproduce sd_g10 of
  all_sites_harm.csv at every reservoir, to 1e-9. The script stops with an error otherwise.

Writes eval/grl_figs/intermediate/c5_o50_20261113/score/sd_g8120.csv
(DAM_ID, sd_g8120, sd_g10_rebuilt, sd_g10, sd_swot).
"""
from swotgat.paths import WORK
import sys
from pathlib import Path
import numpy as np, pandas as pd
F = Path(f"{WORK}/CONUS_storage_fork"); B = F/"eval/grl_figs/intermediate/c5_o50_20261113"
AS = pd.read_csv(F/"eval/overlap50/fig3c_sd/all_sites_harm.csv").set_index("DAM_ID")
TR = pd.read_parquet(F/"testsets/regime3_full_1980_2020/truth_clean.parquet"); TR["date"] = pd.to_datetime(TR.date); TR = TR[TR.use.fillna(False).astype(bool)]
OBS = {int(k): g.set_index("date").storage_mcm.sort_index() for k, g in TR.groupby("DAM_ID")}
def sd(s, a, b):
    """Return the sample standard deviation of series s on the days from a to b (both included).

    The result is NaN when the window has fewer than 10 values.
    """
    w = s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))]; return float(w.std()) if len(w) >= 10 else np.nan
R = pd.DataFrame({d: dict(sd_g8120=sd(s, "1981-01-01", "2020-12-31"), sd_g10_rebuilt=sd(s, "2010-01-01", "2020-12-31")) for d, s in OBS.items()}).T.rename_axis("DAM_ID")
R = R.join(AS[["sd_g10", "sd_swot"]], how="outer")
k = R.sd_g10.notna(); d = (R.sd_g10_rebuilt - R.sd_g10)[k].abs()
print(f"sd_g10 rebuilt at {int(k.sum())} reservoirs: max |diff| {d.max():.3g}; NaN mismatch {int((R.sd_g10_rebuilt[k].isna()).sum())}")
assert d.max() < 1e-9 and R.sd_g10_rebuilt[k].notna().all(), "sd_g10 not reproduced"
R.to_csv(B/"score/sd_g8120.csv"); print("wrote", B/"score/sd_g8120.csv", "sd_g8120 finite at", int(R.sd_g8120.notna().sum()))
