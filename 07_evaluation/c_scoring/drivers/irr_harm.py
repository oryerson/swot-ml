"""Harmonic irregularity of each reservoir's gauge record, water years 1981-2020.

Harmonic irregularity measures how much of a reservoir's storage variability is not a regular
seasonal cycle:
    irr_harm = sd(gauge - fit) / sd(fit)
where fit is the least-squares fit of a constant plus an annual and a semiannual harmonic to
daily gauge storage. A value of 1 means that departures from the cycle vary as much as the cycle
itself. The fit has the same functional form as the SWOT-cycle baseline (bl.py).

Usage: bin/run 07_evaluation/c_scoring/drivers/irr_harm.py      (no arguments)
  Run it after fig3c_sd.py (which writes all_sites_sd.csv), pa_vars.py and score_2010.py.

Reads
  through pa_lib.py: the reservoir list (ids from an earlier model configuration's score table)
      and the 1980-2020 gauge test set, copy in eval/g4_fix/testsets (truth_clean.parquet, days
      with the `use` flag set); out/vars.parquet (pa_vars.py) for the upstream aridity index ai_up
  next to this script: all_sites_sd.csv (fig3c_sd.py)
  ../score/sites_2010_g.csv relative to this script (score_2010.py; in the work tree the script
      lies in eval/overlap50/fig3c_sd/): the 2010-2020 scores, label `v10pm` = SWOT-GAT

Method
  Days: 1980-10-01 to 2020-09-30. A water year counts when it has at least 292 gauge days; a
  reservoir needs at least 5 such years, and only their days enter the fit (the same selection
  as for irreg_8120 in pa_ops8120.py). The harmonics have periods of 365.25 days and half of it.
  harm_r2 is the R2 of the fit; n_wy the number of water years used.
  The numbers file relates irr_harm to SWOT-GAT's 2010-2020 scores at reservoirs of confidence
  tier High or Medium: Spearman correlations with NSE (also for the two older irregularity
  measures of fig3c_sd.py), with aridity and with training coverage (cov_sd = sd_swot / sd_g10);
  partial rank correlations given irr_harm, with a bootstrap interval (2,000 resamples, seed 0)
  for aridity and NSE; medians by reservoir purpose; and the values at two example reservoirs.

Writes, next to this script
  all_sites_harm.csv      all_sites_sd.csv plus the columns irr_harm, harm_r2 and n_wy; read by
                          08_figures/fig3_c5d.py and full_record/sd8120.py
  irr_harm_numbers.json   the numbers described above
"""
from swotgat.paths import WORK
import os, sys, json
sys.path.insert(0, f"{WORK}/CONUS_storage_fork/eval/perf_attrib")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import pa_lib as L
HERE = os.path.dirname(os.path.abspath(__file__))
ids = L.site_ids()
TR = pd.read_parquet(L.T3/"truth_clean.parquet"); TR["date"] = pd.to_datetime(TR.date); TR = TR[TR.use.fillna(False).astype(bool) & TR.DAM_ID.isin(ids)]
wyf = lambda ix: ix.year + (ix.month >= 10)
R = {}
for d, g in TR.groupby("DAM_ID"):
    s = g.set_index("date").storage_mcm.sort_index(); a = s[(s.index >= "1980-10-01") & (s.index < "2020-10-01")]
    wy = wyf(a.index); n = a.groupby(wy).size(); good = n[n >= 292].index
    if len(good) < 5: continue
    a = a[wy.isin(good)].dropna(); t = (a.index - pd.Timestamp("1980-01-01")).days.values/365.25*2*np.pi
    X = np.column_stack([np.ones_like(t), np.sin(t), np.cos(t), np.sin(2*t), np.cos(2*t)])
    b, *_ = np.linalg.lstsq(X, a.values, rcond=None); fit = X @ b; res = a.values - fit
    sf = fit.std()
    R[d] = dict(irr_harm=float(res.std()/sf) if sf > 0 else np.nan, harm_r2=float(1 - (res**2).sum()/((a.values - a.values.mean())**2).sum()), n_wy=len(good))
H = pd.DataFrame(R).T; H.index.name = "DAM_ID"
AS = pd.read_csv(HERE + "/all_sites_sd.csv").set_index("DAM_ID"); OUT = AS.join(H, how="outer"); OUT.to_csv(HERE + "/all_sites_harm.csv")
# Relate harmonic irregularity to SWOT-GAT's 2010-2020 scores (label v10pm; tiers High and Medium)
A = pd.read_csv(HERE + "/../score/sites_2010_g.csv"); A = A[(A.MODEL == "v10pm") & A.tier.isin(["High", "Medium"]) & A.nrmse.notna()].set_index("DAM_ID")
V = pd.read_parquet(L.OUT/"vars.parquet")
D = A[["nse", "amp", "name", "use", "state", "lon"]].join(OUT).join(V[["ai_up"]])
D["cov_sd"] = D.sd_swot/D.sd_g10
def prho(df, x, y, z):
    """Return the partial rank correlation of columns x and y of df given column z.

    It is the correlation of the rank residuals of x and y after a linear fit on the ranks of z.
    """
    rx, ry, rz = [df[c].rank().values for c in (x, y, z)]; res = lambda u: u - np.polyval(np.polyfit(rz, u, 1), rz)
    return float(np.corrcoef(res(rx), res(ry))[0, 1])
def boot_prho(df, x, y, z):
    """Return the 95% percentile bootstrap interval of prho as [low, high], rounded to 2 decimals.

    The rows of df (reservoirs) are resampled with replacement 2,000 times, seed 0.
    """
    rng = np.random.default_rng(0); bs = [prho(df.iloc[rng.integers(0, len(df), len(df))], x, y, z) for _ in range(2000)]
    return [round(float(q), 2) for q in np.percentile(bs, [2.5, 97.5])]
k = D.dropna(subset=["irr_harm", "nse"]); N = {}
N["n_irr"] = len(k); N["rho_harm_nse"] = round(spearmanr(k.irr_harm, k.nse)[0], 3); N["rho_old_nse"] = round(spearmanr(k.irr_old, k.nse)[0], 3)
N["rho_ratio_nse"] = round(spearmanr(k.irr_ratio, k.nse)[0], 3); N["rho_harm_old"] = round(spearmanr(k.irr_harm, k.irr_old)[0], 4)
N["med_irr_harm"] = round(float(k.irr_harm.median()), 3); N["share_harm_gt1"] = round(float((k.irr_harm > 1).mean()), 3)
N["harm_quartiles"] = [round(float(q), 3) for q in k.irr_harm.quantile([0.1, 0.25, 0.75, 0.9])]
ka = k.dropna(subset=["ai_up"]); N["n_ai"] = len(ka); N["rho_ai_harm"] = round(spearmanr(ka.ai_up, ka.irr_harm)[0], 3)
N["prho_ai_nse_given_harm"] = round(prho(ka, "ai_up", "nse", "irr_harm"), 3); N["prho_ai_nse_ci"] = boot_prho(ka, "ai_up", "nse", "irr_harm")
kc = k.dropna(subset=["cov_sd", "amp"]); kc = kc[(kc.cov_sd > 0) & (kc.amp > 0)]
N["n_cov"] = len(kc); N["rho_cov_vr"] = round(spearmanr(kc.cov_sd, kc.amp)[0], 3); N["rho_cov_harm"] = round(spearmanr(kc.cov_sd, kc.irr_harm)[0], 3)
N["prho_cov_vr_given_harm"] = round(prho(kc, "cov_sd", "amp", "irr_harm"), 3); N["prho_cov_nse_given_harm"] = round(prho(kc, "cov_sd", "nse", "irr_harm"), 3)
gp = k.groupby("use").agg(n=("nse", "size"), nse=("nse", "median"), irr=("irr_harm", "median")).round(3)
N["purpose"] = gp.loc[[u for u in ["Irrigation", "Water supply", "Hydroelectricity", "Flood control", "Recreation"] if u in gp.index]].to_dict("index")
q75 = k.irr_harm.quantile(0.75); nc = k[k.state == "North Carolina"]; low = nc[nc.nse < 0.25]
N["NC_low"] = dict(n=len(nc), n_low=len(low), in_top_quarter=int((low.irr_harm >= q75).sum()))
for nm in ("Norfork", "Rathbun"):
    m = D[D.name.str.contains(nm, case=False, na=False)]
    N[nm] = dict(irr_harm=round(float(m.irr_harm.iloc[0]), 3), irr_old=round(float(m.irr_old.iloc[0]), 3), cov_sd=round(float(m.cov_sd.iloc[0]), 3)) if len(m) else None
json.dump(N, open(HERE + "/irr_harm_numbers.json", "w"), indent=1); print(json.dumps(N, indent=1))
