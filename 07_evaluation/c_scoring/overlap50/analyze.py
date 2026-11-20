"""Summary statistics and paired tests of SWOT-GAT, the rule models and the two baselines.

Works on the score tables of the three scorers in overlap50/ (50%-of-days overlap rule), which must
have been run with BASE=1 so that the baselines are in the tables.

Usage: bin/run 07_evaluation/c_scoring/overlap50/analyze.py      (no arguments, no switches)

Reads (paths relative to $SWOTGAT_WORK/CONUS_storage_fork/eval)
  overlap50/score/score_j/fullrecord_site_window.csv   decade.py: windows 1981-1990 and 2010-2020
  overlap50/score/sites_2010_g.csv                     score_2010.py: 2010-2020
  overlap50/score/sites_2325span_g.csv                 score_x26span.py: SWOT Period
  overlap50/fig1_o50.json                              fig1_nums.py: the matched reservoir sets
  baselines_c5/score/swot_fit.csv                      bl.py: fits of the SWOT-cycle baseline

Writes to overlap50/tables/, which must exist already (this script does not create it)
  per_reservoir.csv         NSE, nRMSE, r, VR and number of scored days per reservoir, model, window
  model_stats.csv           per window, set and model, on the reservoirs that model scores: n,
                            NSE median, quartiles, 90th percentile and share below 0; median nRMSE,
                            r and VR
  model_stats_common.csv    the same on the reservoirs that every model of the set scores
  paired.csv                paired NSE differences, model minus baseline (see Method)
  exploratory_r_vr.csv      paired differences of r, VR and nRMSE, SWOT-GAT against each baseline
  analysis.json             set sizes; how many reservoirs have each baseline and why the SWOT
                            cycle is missing elsewhere; fit diagnostics; whether a baseline was
                            scored on the same number of days as SWOT-GAT; a consistency check of
                            the normal year; exploratory rank correlations

Method
  Windows: 1981-1990, 2010-2020 and the SWOT Period. Only reservoirs whose gauge link has
  confidence tier High or Medium are used. Model labels are unified: `v10pm` and `SWOT-GNN` both
  mean SWOT-GAT; `SWOT-cycle` becomes "SWOT cycle (a)" and `normal_year` "gauge normal year (b)".
  Two reservoir sets per window:
    (i)  SWOT-GAT full set: every reservoir that SWOT-GAT scores. Models compared: SWOT-GAT and
         the two baselines.
    (ii) matched 4-model set: the reservoirs that SWOT-GAT, MODROM-cal, GDROM and ISTARF all
         score. The script stops if this set differs from the one in fig1_o50.json. Models
         compared: SWOT-GAT, the three rule models and the two baselines.
  Paired test (paired.csv): for a model and a baseline, the NSE difference at every reservoir that
  both score; its median; a 95% percentile bootstrap interval of the median (2,000 resamples of
  reservoirs, random seed 0 for every comparison); the share of reservoirs with a positive
  difference; the number of ties. Verdict: "beats" if the median and the lower bound are above 0,
  "loses" if the upper bound is below 0, otherwise "no clear difference". In set (i) only SWOT-GAT
  is tested against the baselines; in set (ii) the rule models are tested too.
  Consistency check: the normal year that bl.py adds to score_2010.py is compared with the normal
  year that the full-record scorer scores itself in the 2010-2020 window (same reference record).
  The comparisons above were specified before the baselines were scored. The block marked
  "exploratory" in the code was added afterwards.
"""
from swotgat.paths import WORK
import json
from pathlib import Path
import numpy as np, pandas as pd
F = Path(f"{WORK}/CONUS_storage_fork"); B = F/"eval/baselines_c5"; D = F/"eval/overlap50/score"; O = F/"eval/overlap50/tables"   # the tables folder O must exist already; this script does not create it
C5J = json.load(open(F/"eval/overlap50/fig1_o50.json"))   # matched 4-model sets, written by fig1_nums.py
MK = {"v10pm": "SWOT-GAT", "SWOT-GNN": "SWOT-GAT", "MODROM-cal": "MODROM-cal", "GDROM": "GDROM", "ISTARF": "ISTARF", "SWOT-cycle": "SWOT cycle (a)", "normal_year": "gauge normal year (b)"}
GAT, A, Bn = "SWOT-GAT", "SWOT cycle (a)", "gauge normal year (b)"; RULES = ["MODROM-cal", "GDROM", "ISTARF"]; BASES = [A, Bn]
WIN = {"1981-1990": "8190", "2010-2020": "2010-20", "SWOT Period": "2023-25"}; HM = ["High", "Medium"]
J = {}


def frame(win):
    """Return one window's scores in a common layout, High- and Medium-tier reservoirs only.

    win is "1981-1990", "2010-2020" or "SWOT Period". The columns are DAM_ID, m (the model name
    used in this script), nse, nrmse, r, vr and n_days.
    """
    if win == "1981-1990":
        x = pd.read_csv(D/"score_j/fullrecord_site_window.csv"); x = x[(x.window == "1981-1990") & x.site_tier.isin(HM)].rename(columns={"n": "n_days"})
        x["m"] = x.model.map(MK)
    else:
        x = pd.read_csv(D/("sites_2010_g.csv" if win == "2010-2020" else "sites_2325span_g.csv")); x = x[x.tier.isin(HM) & x.nrmse.notna()]
        x["m"] = x.MODEL.map(MK)
    x = x[x.m.notna()].copy(); assert x.nse.notna().all(); x["vr"] = x.amp
    assert not x.duplicated(["DAM_ID", "m"]).any()
    return x[["DAM_ID", "m", "nse", "nrmse", "r", "vr", "n_days"]]


def stats(x):
    """Return summary statistics of the score rows x of one model, as a dict.

    The keys are n, the median, 25th, 75th and 90th percentile of NSE, the share of NSE below 0,
    and the medians of nRMSE, r and VR. An empty table gives only n = 0.
    """
    v = x.nse.values
    return dict(n=int(len(x)), nse_med=float(np.median(v)), nse_p90=float(np.percentile(v, 90)), nse_share_lt0=float((v < 0).mean()),
                nse_p25=float(np.percentile(v, 25)), nse_p75=float(np.percentile(v, 75)),
                nrmse_med=float(x.nrmse.median()), r_med=float(x.r.median()), vr_med=float(x.vr.median())) if len(x) else dict(n=0)


def paired(dv):
    """Summarize the per-reservoir paired differences dv.

    Returns the median difference, its 95% percentile bootstrap interval (2,000 resamples,
    default_rng(0) for every comparison), the share of differences above 0, the number of ties and
    a verdict: "beats" (median and lower bound above 0), "loses" (upper bound below 0) or
    "no clear difference".
    """
    d = np.asarray(dv, float); n = len(d)
    idx = np.random.default_rng(0).integers(0, n, (2000, n)); lo, hi = np.percentile(np.median(d[idx], axis=1), [2.5, 97.5]); med = float(np.median(d))
    verdict = "beats" if (med > 0 and lo > 0) else ("loses" if hi < 0 else "no clear difference")
    return dict(n=n, d_med=med, ci_lo=float(lo), ci_hi=float(hi), share_better=float((d > 0).mean()), n_ties=int((d == 0).sum()), verdict=verdict)


FR = {w: frame(w) for w in WIN}
P = pd.concat([f.assign(window=w) for w, f in FR.items()]); P.to_csv(O/"per_reservoir.csv", index=False, float_format="%.6g")
SETS = {}
for w, k in WIN.items():
    f = FR[w]; full = set(f[f.m == GAT].DAM_ID); m4 = set.intersection(*[set(f[f.m == m].DAM_ID) for m in [GAT] + RULES])
    assert m4 == set(C5J[k]["ids"]), (w, len(m4), C5J[k]["n"])
    SETS[w] = {"(i) SWOT-GAT full set": sorted(full), "(ii) matched 4-model set": sorted(C5J[k]["ids"])}
J["set_sizes"] = {w: {s: len(v) for s, v in S.items()} for w, S in SETS.items()}

rows, rows_c, prs = [], [], []
for w in WIN:
    f = FR[w]
    for s, ids in SETS[w].items():
        models = [GAT] + BASES if s.startswith("(i)") else [GAT] + RULES + BASES
        x = f[f.DAM_ID.isin(ids)]
        common = set(ids)
        for m in models: common &= set(x[x.m == m].DAM_ID)
        for m in models:
            rows.append(dict(window=w, set=s, n_set=len(ids), model=m, **stats(x[x.m == m])))
            rows_c.append(dict(window=w, set=s, n_set=len(ids), n_common=len(common), model=m, **stats(x[(x.m == m) & x.DAM_ID.isin(common)])))
        piv = x.pivot(index="DAM_ID", columns="m", values="nse")
        for m in ([GAT] if s.startswith("(i)") else [GAT] + RULES):
            for b in BASES:
                if b not in piv or m not in piv: continue
                pp = piv[[m, b]].dropna()
                prs.append(dict(window=w, set=s, n_set=len(ids), model=m, baseline=b, **paired(pp[m] - pp[b]),
                                model_nse_med=float(pp[m].median()), baseline_nse_med=float(pp[b].median())))
MS, MC, PR = pd.DataFrame(rows), pd.DataFrame(rows_c), pd.DataFrame(prs)
MS.to_csv(O/"model_stats.csv", index=False, float_format="%.6g"); MC.to_csv(O/"model_stats_common.csv", index=False, float_format="%.6g")
PR.to_csv(O/"paired.csv", index=False, float_format="%.6g")

# ---- availability of the baselines in each set, and why (a) is missing
FIT = pd.read_csv(B/"score/swot_fit.csv").set_index("DAM_ID")
J["fit"] = dict(dams=len(FIT), usable=int(FIT.usable.sum()), why=FIT[~FIT.usable].why.str.replace(r"[0-9]+", "#", regex=True).value_counts().to_dict())
AV = {}
for w in WIN:
    f = FR[w]
    for s, ids in SETS[w].items():
        has = {b: set(f[(f.m == b) & f.DAM_ID.isin(ids)].DAM_ID) for b in BASES}; miss_a = sorted(set(ids) - has[A]); miss_b = sorted(set(ids) - has[Bn])
        fa = FIT.reindex(miss_a)
        AV[f"{w} | {s}"] = dict(n=len(ids), with_a=len(has[A]), with_b=len(has[Bn]), with_both=len(has[A] & has[Bn]),
                                missing_a_why=fa.why.fillna("not in regime3/regime1x sites").str.replace(r"[0-9]+", "#", regex=True).value_counts().to_dict(),
                                missing_a_usable_fit=int(fa.usable.fillna(False).astype(bool).sum()), missing_b=miss_b[:30], n_missing_b=len(miss_b))
        if s.startswith("(i)"):
            u = FIT.reindex(sorted(has[A]))
            AV[f"{w} | {s}"]["fit_in_set"] = dict(n_days_med=float(u.n_days.median()), n_days_p10=float(u.n_days.quantile(.1)), span_med=float(u.span_days.median()),
                                                  cond_med=float(u.cond.median()), cond_max=float(u.cond.max()), r2_fit_med=float(u.r2_fit.median()),
                                                  share_all_swotae_med=float(u.share_all_swotae.median()))
J["availability"] = AV

# ---- day alignment: a baseline is scored on the same days as SWOT-GAT?
DA = {}
for w in WIN:
    f = FR[w]; g = f[f.m == GAT].set_index("DAM_ID").n_days
    for b in BASES:
        x = f[f.m == b].set_index("DAM_ID").n_days; c = x.index.intersection(g.index)
        DA[f"{w} | {b}"] = dict(n=len(c), same_n_days=int((x[c] == g[c]).sum()), max_abs_diff_days=int((x[c] - g[c]).abs().max()) if len(c) else 0,
                                median_n_days=float(x[c].median()) if len(c) else np.nan)
J["day_alignment"] = DA

# ---- consistency: the normal_year that bl.py adds to score_2010.py vs the decade scorer's own normal_year rows (2010-2020 window, same reference)
SJ = pd.read_csv(D/"score_j/fullrecord_site_window.csv"); s2 = SJ[(SJ.window == "2010-2020") & (SJ.model == "normal_year") & SJ.site_tier.isin(HM)].copy()
assert s2.nse.notna().all(); s2 = s2.set_index("DAM_ID"); n2 = FR["2010-2020"]; n2 = n2[n2.m == Bn].set_index("DAM_ID")
c = n2.index.intersection(s2.index); same_days = c[(n2.loc[c, "n_days"].values == s2.loc[c, "n"].values)]
J["check_normal_year_2010"] = dict(n_both=len(c), n_same_days=len(same_days), max_abs_dnse_same_days=float((n2.loc[same_days, "nse"] - s2.loc[same_days, "nse"]).abs().max()) if len(same_days) else None)

# ---- exploratory (added after the planned comparisons): paired r, VR and nRMSE differences, SWOT-GAT minus each baseline, on set (i)
EX = []
for w in WIN:
    f = FR[w]; ids = SETS[w]["(i) SWOT-GAT full set"]
    for b in BASES:
        for k in ("r", "vr", "nrmse"):
            pv = f[f.DAM_ID.isin(ids)].pivot(index="DAM_ID", columns="m", values=k)[[GAT, b]].dropna()
            EX.append(dict(window=w, baseline=b, metric=k, n=len(pv), gat_med=float(pv[GAT].median()), base_med=float(pv[b].median()), d_med=float((pv[GAT] - pv[b]).median())))
    # does SWOT-GAT's advantage over (a) shrink where (a) is good? Spearman of NSE(a) with dNSE
    pv = f[f.DAM_ID.isin(ids)].pivot(index="DAM_ID", columns="m", values="nse")
    if A in pv:
        q = pv[[GAT, A, Bn]].dropna() if Bn in pv else pv[[GAT, A]].dropna()
        J.setdefault("exploratory_corr", {})[w] = dict(n=len(q), spearman_gat_a=float(q[GAT].corr(q[A], method="spearman")),
                                                        spearman_gat_b=float(q[GAT].corr(q[Bn], method="spearman")) if Bn in q else None,
                                                        spearman_a_b=float(q[A].corr(q[Bn], method="spearman")) if Bn in q else None)
pd.DataFrame(EX).to_csv(O/"exploratory_r_vr.csv", index=False, float_format="%.6g")
json.dump(J, open(O/"analysis.json", "w"), indent=1, default=float)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(MS.round(3).to_string(index=False)); print(PR.round(3).to_string(index=False)); print(json.dumps(J, indent=1, default=float))
