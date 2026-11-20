"""Recompute the numbers quoted in the paper's Results text (sections 3.2 to 3.6) and Methods counts.

All numbers come from the score tables made with the 50%-of-days overlap rule.

Usage: bin/run 07_evaluation/c_scoring/overlap50/text_nums_o50.py      (no arguments, no switches)

Reads (paths relative to $SWOTGAT_WORK/CONUS_storage_fork/eval)
  overlap50/tables/per_reservoir.csv    analyze.py: scores per reservoir, model and window
  overlap50/fig1_o50.json               fig1_nums.py: the matched 4-model sets
  overlap50/score/sites_2010_g.csv      score_2010.py: 2010-2020 scores with reservoir metadata
  perf_attrib/out/vars.parquet          drivers/pa_vars.py: upstream aridity index `ai_up`
  perf_attrib/out/ops_8120.parquet      drivers/pa_ops8120.py: seasonal-cycle irregularity
                                        `irreg_8120` of the gauge record, water years 1981-2020
  perf_attrib/out/a365/lm_sites.parquet column `cov00`, a training-coverage measure. This table
                                        was written for an earlier model configuration (a365) by a
                                        script that is not in this repository.
  swot_vs_gauge/per_reservoir_plus.csv  only its column names are printed; no number uses it

Writes overlap50/text_nums_o50.json and prints it.

Method
  Per window (1981-1990, 2010-2020, SWOT Period):
    SWOT-GAT on every reservoir it scores: n, median NSE, share of NSE below 0, median r, VR, nRMSE.
    On the matched 4-model set, against each rule model (GDROM, ISTARF, MODROM-cal): median paired
      difference of r (SWOT-GAT minus rule) and of nRMSE (rule minus SWOT-GAT); share of
      reservoirs where SWOT-GAT's VR is the lower one; the rule model's median VR; and for the
      paired NSE difference (SWOT-GAT minus rule) the list [median, lower and upper bound of a 95%
      bootstrap interval, share above 0, sign-test p-value].
    Against the SWOT-cycle baseline, on every reservoir both score: the same paired NSE summary,
      and the baseline's median r and VR.
  Bootstrap: 2,000 resamples of reservoirs, random seed 0, percentile interval of the median.
  Sign test: two-sided binomial test on the non-zero differences.
  `hind_paired`: SWOT-GAT NSE in 1981-1990 against 2010-2020 at the reservoirs scored in both.
  `attrib_2010` (drivers of skill, SWOT-GAT, 2010-2020, tiers High and Medium): Spearman
    correlations of irregularity with NSE, of aridity with irregularity and NSE, and of training
    coverage with VR and irregularity; partial rank correlations given irregularity (ranks of both
    variables regressed on the rank of irregularity, then the correlation of the residuals), with a
    bootstrap interval for aridity and NSE; median NSE and irregularity by reservoir purpose; a
    Kruskal-Wallis test across purposes with at least 10 reservoirs, on the rank residuals of NSE
    after irregularity; median NSE of five regions; and the values at two example reservoirs.

Note
  The driver numbers here use `irreg_8120` and `cov00`. The harmonic irregularity and the
  standard-deviation coverage shown in Figure 3 are computed in drivers/irr_harm.py and
  drivers/fig3c_sd.py, which write their own number files.
"""
from swotgat.paths import WORK
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr, kruskal
E = Path(f"{WORK}/CONUS_storage_fork/eval"); O = E/"overlap50"
P = pd.read_csv(O/"tables/per_reservoir.csv"); J = json.load(open(O/"fig1_o50.json"))
K = {"1981-1990": "8190", "2010-2020": "2010-20", "SWOT Period": "2023-25"}; R = ["GDROM", "ISTARF", "MODROM-cal"]
def boot(d, f=np.median):
    """Return the 95% percentile bootstrap interval of the statistic f of d, rounded to 3 decimals.

    The values d are resampled with replacement 2,000 times, random seed 0.
    """
    rng = np.random.default_rng(0); b = [f(rng.choice(d, len(d))) for _ in range(2000)]; return [round(float(x), 3) for x in np.percentile(b, [2.5, 97.5])]
sg = lambda d: float(stats.binomtest(int((d > 0).sum()), int((d != 0).sum())).pvalue)
out = {}
for w, k in K.items():
    x = P[P.window == w]; g = x[x.m == "SWOT-GAT"]
    o = dict(n=len(g), nse=round(g.nse.median(), 4), lt0=round((g.nse < 0).mean(), 4), r=round(g.r.median(), 4), vr=round(g.vr.median(), 4),
             nrmse=round(g.nrmse.median(), 5))
    ids = set(J[k]["ids"]); xm = x[x.DAM_ID.isin(ids)]
    pv = {met: xm.pivot(index="DAM_ID", columns="m", values=met) for met in ("nse", "nrmse", "r", "vr")}
    for rm in R:
        o[f"dr_{rm}"] = round(float((pv["r"]["SWOT-GAT"] - pv["r"][rm]).dropna().median()), 4)
        o[f"dnrmse_{rm}"] = round(float((pv["nrmse"][rm] - pv["nrmse"]["SWOT-GAT"]).dropna().median()), 5)
        v = pv["vr"][["SWOT-GAT", rm]].dropna(); o[f"vr_lower_{rm}"] = round(float((v["SWOT-GAT"] < v[rm]).mean()), 4)
        o[f"vrmed_{rm}"] = round(float(v[rm].median()), 4)
        d = (pv["nse"]["SWOT-GAT"] - pv["nse"][rm]).dropna().values
        o[f"dnse_{rm}"] = [round(float(np.median(d)), 4)] + boot(d) + [round(float((d > 0).mean()), 4), sg(d)]
    o["vrmed_gat_matched"] = round(float(pv["vr"]["SWOT-GAT"].median()), 4)
    pc = x.pivot(index="DAM_ID", columns="m", values="nse"); d = (pc["SWOT-GAT"] - pc["SWOT cycle (a)"]).dropna().values
    o["cycle"] = dict(n=len(d), med=round(float(np.median(d)), 4), ci=boot(d), better=round(float((d > 0).mean()), 4), p=sg(d))
    c = x[x.m == "SWOT cycle (a)"]; o["cycle_r"] = round(c.r.median(), 4); o["cycle_vr"] = round(c.vr.median(), 4)
    out[w] = o
g8 = P[(P.window == "1981-1990") & (P.m == "SWOT-GAT")].set_index("DAM_ID").nse; g10 = P[(P.window == "2010-2020") & (P.m == "SWOT-GAT")].set_index("DAM_ID").nse
c = g8.index.intersection(g10.index); d = (g8[c] - g10[c]).values
out["hind_paired"] = dict(n=len(c), med81=round(float(g8[c].median()), 4), med10=round(float(g10[c].median()), 4), d=round(float(np.median(d)), 4), ci=boot(d))
# SWOT Period: the two tables below are loaded and the second one's columns printed; no number uses them
sp = P[(P.window == "SWOT Period") & (P.m == "SWOT-GAT")].set_index("DAM_ID")
pr = pd.read_csv(E/"swot_vs_gauge/per_reservoir_plus.csv"); print("plus cols", pr.columns.tolist()[:15])
# drivers of skill, 2010-2020 (SWOT-GAT rows, tiers High and Medium)
a = pd.read_csv(O/"score/sites_2010_g.csv"); a = a[(a.MODEL == "v10pm") & a.tier.isin(["High", "Medium"]) & a.nrmse.notna()].set_index("DAM_ID")
V = pd.read_parquet(E/"perf_attrib/out/vars.parquet"); OP = pd.read_parquet(E/"perf_attrib/out/ops_8120.parquet"); LM = pd.read_parquet(E/"perf_attrib/out/a365/lm_sites.parquet")
d = a.join(V[["ai_up"]]).join(OP[["irreg_8120"]]).join(LM[["cov00"]])
def prho(df, x, y, z):
    """Return the partial rank correlation of columns x and y of df given column z.

    It is the correlation of the rank residuals of x and y after a linear fit on the ranks of z.
    """
    rx, ry, rz = [df[cc].rank().values for cc in (x, y, z)]; res = lambda u: u - np.polyval(np.polyfit(rz, u, 1), rz)
    return float(np.corrcoef(res(rx), res(ry))[0, 1])
k = d.dropna(subset=["irreg_8120", "nse"]); A = {"n_irreg": len(k), "rho_irreg_nse": round(spearmanr(k.irreg_8120, k.nse)[0], 3)}
ka = k.dropna(subset=["ai_up"]); A["n_ai"] = len(ka); A["rho_ai_irreg"] = round(spearmanr(ka.ai_up, ka.irreg_8120)[0], 3); A["rho_ai_nse"] = round(spearmanr(ka.ai_up, ka.nse)[0], 3)
A["prho_ai_nse"] = round(prho(ka, "ai_up", "nse", "irreg_8120"), 3)
rng = np.random.default_rng(0); bs = [prho(ka.iloc[rng.integers(0, len(ka), len(ka))], "ai_up", "nse", "irreg_8120") for _ in range(2000)]
A["prho_ai_nse_ci"] = [round(float(q), 2) for q in np.percentile(bs, [2.5, 97.5])]
kc = d.dropna(subset=["cov00", "amp", "irreg_8120"]); kc = kc[kc.cov00 > 0]
A["n_cov"] = len(kc); A["rho_cov_vr"] = round(spearmanr(kc.cov00, kc.amp)[0], 3); A["rho_cov_irreg"] = round(spearmanr(kc.cov00, kc.irreg_8120)[0], 3)
A["prho_cov_vr"] = round(prho(kc, "cov00", "amp", "irreg_8120"), 3); A["prho_cov_nse"] = round(prho(kc, "cov00", "nse", "irreg_8120"), 3)
k = k.copy(); k["res"] = k.nse.rank() - np.polyval(np.polyfit(k.irreg_8120.rank(), k.nse.rank(), 1), k.irreg_8120.rank())
gp = k.groupby("use").agg(n=("nse", "size"), nse=("nse", "median"), irreg=("irreg_8120", "median")).round(3)
A["purpose"] = gp.loc[["Irrigation", "Water supply", "Hydroelectricity", "Flood control", "Recreation"]].to_dict("index")
big = [u for u, cc in k.use.value_counts().items() if cc >= 10]; A["kw_resid_p"] = round(float(kruskal(*[k[k.use == u].res for u in big]).pvalue), 4)
reg = {"NC": d.state == "North Carolina", "PNW": d.state.isin(["Washington", "Oregon", "Idaho"]), "CW_TX": (d.state == "Texas") & (d.lon < -97.5),
       "East_exNC": (d.lon > -90) & (d.state != "North Carolina"), "SW": d.state.isin(["Arizona", "New Mexico", "Nevada", "Utah"])}
A["regions"] = {r_: dict(n=int(m.sum()), nse=round(float(d[m].nse.median()), 3)) for r_, m in reg.items()}
q75 = d.irreg_8120.quantile(0.75); nc = d[reg["NC"]]; low = nc[nc.nse < 0.25]
A["NC_low"] = dict(n_low=len(low), in_top_quarter=int((low.irreg_8120 >= q75).sum()))
for nm in ("Norfork", "Rathbun"):
    m = d[d.name.str.contains(nm, case=False, na=False)]
    A[nm] = dict(nse=round(float(m.nse.iloc[0]), 3), vr=round(float(m.amp.iloc[0]), 3), irreg=round(float(m.irreg_8120.iloc[0]), 3), cov=round(float(m.cov00.iloc[0]), 3)) if len(m) else None
out["attrib_2010"] = A
json.dump(out, open(O/"text_nums_o50.json", "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
