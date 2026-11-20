"""Training coverage per reservoir: how much storage variability SWOT saw, relative to the gauge record.

Training coverage is defined like the variability ratio (VR), so that the two can be compared on
one scale:
    cov_sd = sd(SWOT storage targets, SWOT Period) / sd(gauge storage, 2010-2020)
    VR     = sd(SWOT-GAT, 2010-2020)               / sd(gauge storage, 2010-2020)
Both have the same denominator, so VR / cov_sd = sd(model) / sd(SWOT). On the 1:1 line of a plot of
VR against cov_sd, the model's storage varies exactly as much as the SWOT targets it was trained on.
(The figure uses the 1981-2020 gauge standard deviation instead, from full_record/sd8120.py.)

The script also computes two seasonal-cycle irregularity measures of the gauge record. The paper
uses the harmonic irregularity of irr_harm.py, which adds its column to this script's table.

Usage: bin/run 07_evaluation/c_scoring/drivers/fig3c_sd.py      (no arguments)
  Run it after score_2010.py and pa_ops8120.py.

Reads
  through pa_lib.py (see the note there on inputs from earlier model configurations):
    the reservoir list; the sites tables and truth_clean.parquet of the gauge test sets (copy in
    eval/g4_fix/testsets; days with the `use` flag set)
    CONUS/zarr_store_vd2026_halfexact   the store with the SWOT lake targets that the graph-v3.5
        training store copies its lake targets from (04_swot_targets/c_template_store)
    CONUS/basin_lists/fallback/dv_SWOTAE_vd2026_fb_g.csv   lake nodes that carry a SWOT target
    out/ops_8120.parquet (pa_ops8120.py)   irreg_8120, for comparison
    out/a365/lm_sites.parquet   column cov00, an older coverage measure, for comparison. This
        table was written for an earlier model configuration (a365) by a script that is not in
        this repository.
  ../score/sites_2010_g.csv relative to this script (score_2010.py; in the work tree the script
      lies in eval/overlap50/fig3c_sd/): VR is the column `amp` of the rows with the label
      `v10pm`, which means SWOT-GAT (the paper's model: mean of two seeds)

Method
  Gauge. sd_g10 = sample standard deviation of gauge storage on 2010-01-01 to 2020-12-31 (at
  least 10 days).
  SWOT. Per member node and day the value is dv_SWOTAE, else dv_staticPLD (level change times the
  static lake area), at nodes on the list of SWOT-target nodes; a reservoir with a member off the
  list gets no series. Values are summed over the members on days when every member has one,
  2023-03-01 to 2026-09-30. Members come from the 2023-2026 test set if the dam is there,
  otherwise from the 1980-2020 test set. With at least 10 such days: sd_swot (sample standard
  deviation), n_swot (days) and span_swot (98th minus 2nd percentile).
  Irregularity, on water years 1981-2020 with the same days and mean seasonal cycle as irreg_8120
  in pa_ops8120.py (departure = storage anomaly minus cycle anomaly):
    irr_old   = sum(departure^2) / sum(anomaly^2), the definition of irreg_8120
    irr_sqrt  = sqrt(irr_old) = rms(departure) / sd(gauge)
    irr_ratio = sd(departure) / sd(cycle); 1 means equal non-seasonal and seasonal variability

Writes, next to this script
  all_sites_sd.csv        every reservoir of the list: irr_old, irr_sqrt, irr_ratio, sd_g10,
                          sd_swot, n_swot, span_swot
  fig3c_sd_sites.csv      the reservoirs of confidence tier High or Medium that SWOT-GAT scores in
                          2010-2020: nse, vr, r, metadata, the columns above, cov00, irreg_8120
                          and cov_sd
  fig3c_sd_numbers.json   rank correlations between coverage, VR, NSE and irregularity, medians
                          and shares (also printed)
"""
from swotgat.paths import WORK
import os, sys, json
sys.path.insert(0, f"{WORK}/CONUS_storage_fork/eval/perf_attrib")
import numpy as np, pandas as pd, xarray as xr
from scipy.stats import spearmanr
import pa_lib as L
HERE = os.path.dirname(os.path.abspath(__file__))
STORE = L.C/"zarr_store_vd2026_halfexact"
ids = L.site_ids(); S3 = L.sites(L.T3).loc[ids]; S1X = L.sites(L.T1X)
TR = pd.read_parquet(L.T3/"truth_clean.parquet"); TR["date"] = pd.to_datetime(TR.date); TR = TR[TR.use.fillna(False).astype(bool) & TR.DAM_ID.isin(ids)]
OBS = {int(k): g.set_index("date").storage_mcm.sort_index() for k, g in TR.groupby("DAM_ID")}
wyf = lambda ix: ix.year + (ix.month >= 10)
def cycle(s):
    """Return the mean seasonal cycle of a daily series, indexed by day of year 1-365.

    The cycle is the day-of-year mean (day 366 folded into 365), with gaps interpolated and a 31-day
    circular running mean applied. Return None when fewer than 330 days of the year have data.
    """
    doy = np.minimum(s.index.dayofyear.values, 365); m = pd.Series(s.values).groupby(doy).mean().reindex(range(1, 366))
    if m.notna().sum() < 330: return None
    m = m.interpolate(limit_direction="both"); e = np.r_[m.values[-15:], m.values, m.values[:15]]
    return pd.Series(np.convolve(e, np.ones(31)/31, "valid"), index=range(1, 366))
rows = {}
for d, s in OBS.items():
    r = {}
    g10 = s[(s.index >= L.W0) & (s.index <= L.W1)]; r["sd_g10"] = float(g10.std()) if len(g10) >= 10 else np.nan
    a = s[(s.index >= "1980-10-01") & (s.index < "2020-10-01")]; wy = wyf(a.index); n = a.groupby(wy).size(); good = n[n >= 292].index
    if len(good) >= 5:
        a = a[wy.isin(good)]; cy = cycle(a)
        if cy is not None:
            an = a - a.mean(); cv = cy.reindex(np.minimum(a.index.dayofyear, 365)).values; cv = cv - cv.mean(); dep = an.values - cv
            r["irr_old"] = float((dep**2).sum()/(an.values**2).sum()); r["irr_sqrt"] = float(np.sqrt(r["irr_old"]))
            r["irr_ratio"] = float(dep.std()/cv.std()) if cv.std() > 0 else np.nan
    rows[d] = r
G = pd.DataFrame(rows).T
fb = set(pd.read_csv(L.FB).subbasin.astype(str))
mem26 = {d: (S1X.mem[d], S1X.basin[d]) if d in S1X.index else (S3.mem[d], S3.basin[d]) for d in ids}
SW = {}
for bas in sorted({b for _, b in mem26.values()}):
    try: ds = xr.open_zarr(STORE/bas, consolidated=False)
    except Exception as e: print("no store", bas, e); continue
    have = set(ds.subbasin.values.astype(str)); nd = sorted({m for d, (mm, b) in mem26.items() if b == bas for m in mm} & have)
    if not nd: continue
    sub = ds[["dv_SWOTAE", "dv_staticPLD"]].sel(subbasin=nd, date=slice(L.X0, L.X1)).load()
    ae = sub.dv_SWOTAE.to_pandas(); sp = sub.dv_staticPLD.to_pandas(); ae.columns = ae.columns.astype(str); sp.columns = sp.columns.astype(str)
    for d, (mm, b) in mem26.items():
        if b != bas or not all(m in ae.columns for m in mm): continue
        v = pd.DataFrame({m: ae[m].where(ae[m].notna(), sp[m]) if m in fb else np.nan for m in mm}, index=ae.index)
        s = v[v.notna().all(axis=1)].sum(axis=1)
        if len(s) >= 10: SW[d] = dict(sd_swot=float(s.std()), n_swot=len(s), span_swot=float(np.nanpercentile(s, 98) - np.nanpercentile(s, 2)))
SW = pd.DataFrame(SW).T
G.join(SW, how="outer").rename_axis("DAM_ID").to_csv(HERE + "/all_sites_sd.csv")   # every site: irregularity variants, sd_g10, SWOT sd
A = pd.read_csv(HERE + "/../score/sites_2010_g.csv"); A = A[(A.MODEL == "v10pm") & A.tier.isin(["High", "Medium"]) & A.nrmse.notna()].set_index("DAM_ID")
LM = pd.read_parquet(L.OUT/"a365/lm_sites.parquet"); OP = pd.read_parquet(L.OUT/"ops_8120.parquet")
D = A[["nse", "amp", "r", "name", "state", "use"]].rename(columns={"amp": "vr"}).join(G).join(SW).join(LM[["cov00"]]).join(OP[["irreg_8120"]])
D["cov_sd"] = D.sd_swot/D.sd_g10
D.to_csv(HERE + "/fig3c_sd_sites.csv")
k = D.dropna(subset=["cov_sd", "vr"]); k = k[(k.cov_sd > 0) & (k.vr > 0)]
def prho(df, x, y, z):
    """Return the partial rank correlation of columns x and y of df given column z.

    It is the correlation of the rank residuals of x and y after a linear fit on the ranks of z.
    """
    rx, ry, rz = [df[c].rank().values for c in (x, y, z)]; res = lambda u: u - np.polyval(np.polyfit(rz, u, 1), rz)
    return float(np.corrcoef(res(rx), res(ry))[0, 1])
kk = k.dropna(subset=["irr_ratio"])
N = dict(n=len(k), rho_covsd_vr=round(spearmanr(k.cov_sd, k.vr)[0], 3), rho_cov00_vr=round(spearmanr(k.cov00, k.vr, nan_policy="omit")[0], 3),
         rho_covsd_cov00=round(spearmanr(k.cov_sd, k.cov00, nan_policy="omit")[0], 3),
         med_cov_sd=round(float(k.cov_sd.median()), 3), med_vr=round(float(k.vr.median()), 3),
         med_vr_over_cov=round(float((k.vr/k.cov_sd).median()), 3), share_below_1to1=round(float((k.vr < k.cov_sd).mean()), 3),
         share_cov_lt1=round(float((k.cov_sd < 1).mean()), 3),
         rho_logratio_vs_cov=round(spearmanr(np.log(k.vr/k.cov_sd), k.cov_sd)[0], 3),
         n_irr=len(kk), rho_irrratio_irrold=round(spearmanr(kk.irr_ratio, kk.irr_old)[0], 4), rho_irrold_vs_irreg8120=round(spearmanr(kk.irr_old, kk.irreg_8120, nan_policy="omit")[0], 4),
         rho_irrratio_nse=round(spearmanr(kk.irr_ratio, kk.nse)[0], 3), rho_irrold_nse=round(spearmanr(kk.irr_old, kk.nse)[0], 3),
         med_irr_ratio=round(float(kk.irr_ratio.median()), 3), share_irr_ratio_gt1=round(float((kk.irr_ratio > 1).mean()), 3),
         rho_covsd_irr=round(spearmanr(kk.cov_sd, kk.irr_ratio)[0], 3), prho_covsd_vr_given_irr=round(prho(kk, "cov_sd", "vr", "irr_ratio"), 3),
         prho_covsd_nse_given_irr=round(prho(kk, "cov_sd", "nse", "irr_ratio"), 3))
json.dump(N, open(HERE + "/fig3c_sd_numbers.json", "w"), indent=1); print(json.dumps(N, indent=1))
for nm in ("Norfork", "Rathbun"):
    m = D[D.name.str.contains(nm, case=False, na=False)]; print(nm, m[["vr", "cov_sd", "cov00", "irr_ratio", "irr_old", "nse"]].round(3).to_dict("records"))
