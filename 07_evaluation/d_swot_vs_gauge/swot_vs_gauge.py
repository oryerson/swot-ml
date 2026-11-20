"""Compare SWOT storage with gauge storage at reservoirs on SWOT pass days (Results 3.1).

SWOT-GAT is trained on storage change observed by SWOT. This script checks those observations
themselves against reservoir gauges: no model output is involved. Paths below are relative to
`$SWOTGAT_WORK/CONUS_storage_fork/` unless they start with `$`.

Usage:
  python swot_vs_gauge.py          (no arguments; the output folder eval/swot_vs_gauge/ must exist)

Method:
  - SWOT series: the lake storage record filtered with HALF (the lake outlier filter of Trudel et
    al.), eval/half_vd_review/lakesp_vD_halfexact_graph.parquet. Per lake and day the value is
    dv_SWOTAE (from an area-elevation curve fitted to SWOT area and level pairs), else dv_3DLAKES
    (SWOT level with the 3D-LAKES lake geometry), else dv_staticPLD (SWOT level x the fixed lake
    area of the SWOT Prior Lake Database, PLD). A reservoir's storage is the sum over its member
    lakes, on days when every member has a value. This is the series Figure 2 plots as "SWOT
    observations" (08_figures/jw_fig_tt_e.py), and it is how SWOT-GAT's training target is built.
  - Gauge series: `storage_mcm` (million m3) of the test set's truth_clean_v1x.parquet on the same
    day.
  - Pairs: SWOT pass days from 2023-07-01 to 2026-06-30 that also have a gauge value. Both series
    are centered per reservoir over the matched days.
  - Per reservoir: number of matched days; Pearson r; nRMSE = RMSE / gauge range on the matched
    days; NSE of the centered series (equal to 2 VR r - VR^2); VR (variability ratio) = SWOT
    standard deviation / gauge standard deviation.
  - A reservoir is not scored, and the reason is written in column `drop`, when a member lake has
    no SWOT series, when the gauge has no data in the window, when there are fewer than 10 matched
    days, or when either series is constant on the matched days.
  - A gauge is labeled "elevation-based" when the test set's `source` or `method` column contains
    "elev"; otherwise "storage".
  - The summary also gives r over all pairs pooled (in MCM, and with each reservoir standardized by
    its own standard deviation), and splits by gauge kind and by SWOT source.

Two comparison sets:
  (1) The 272 High/Medium-tier reservoirs that all four models of an earlier figure set score in
      the SWOT Period (the set of that figure set's Figure 1d). The ids are read from that figure
      set's kit (`kit_h`, in eval/grl_figs/intermediate/a365_vDspan_20261030/figset_a365e/), which
      belongs to an earlier model configuration (a365). Sites and gauges:
      testsets/regime1x_2023_2026/.
      -> per_reservoir_fig1d.csv
  (2) The expanded gauge set gauges_expand/testsets/regime1x_plus_2023_2026/ (built by
      07_evaluation/a_gauge_truth/expanded_set/), High/Medium tier. Rule-model coverage comes from
      gauges_expand/sizing/candidates.csv, as in summarize_plus.py: the "3-model" set has GDROM
      and ISTARF rules (556 reservoirs), the "4-model" set also has calibrated MODROM (307).
      -> per_reservoir_plus.csv (columns in_3model, in_4model, added_by)
  The paper uses set (2). The script asserts the three counts (272, 307, 556) and stops if the
  inputs give other numbers.

Leakage check: the script also records what the model was trained on, to show that no gauge data
entered training. It reads config.json of two training runs of an earlier model configuration
(v22a365, seeds S1 and S2, under $SWOTGAT_WORK/CONUS/), not of the paper's configuration C5. It
lists the config keys that differ between the seeds, the targets and input features, the variables
of the training store, and any feature or variable name that contains "gauge", "truth",
"storage_mcm", "insitu" or "resops".

Reads:
  eval/half_vd_review/lakesp_vD_halfexact_graph.parquet
  testsets/regime1x_2023_2026/{sites.csv, truth_clean_v1x.parquet}
  gauges_expand/testsets/regime1x_plus_2023_2026/{sites.csv, truth_clean_v1x.parquet}
  gauges_expand/sizing/candidates.csv
  eval/grl_figs/intermediate/a365_vDspan_20261030/figset_a365e/kit_h.py and the tables it loads
      (earlier model configuration; the script changes directory there and sets HIND=2010 if the
      environment does not set it)
  $SWOTGAT_WORK/CONUS/<run>/config.json of the two v22a365 runs, and their training store

Writes (in eval/swot_vs_gauge/):
  per_reservoir_fig1d.csv, per_reservoir_plus.csv   one row per reservoir
  summary.json, summary.md                          medians, quartiles and counts per set
  leakage_check.json
The script only reads its inputs; it changes no figure, store or test-set file.
"""
from swotgat.paths import WORK
import os, sys, json, numpy as np, pandas as pd
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork"); OUT = F/"eval/swot_vs_gauge"
FIGD = F/"eval/grl_figs/intermediate/a365_vDspan_20261030/figset_a365e"; X = F/"gauges_expand"
CONUS = Path(f"{WORK}/CONUS")
RUNS = ["era5_conus_v22a365_S1_storage_20261020_034226", "era5_conus_v22a365_S2_storage_s1_20261020_034109"]
W0, W1, MINP = pd.Timestamp("2023-07-01"), pd.Timestamp("2026-06-30"), 10

# ---------------- SWOT: HALF-exact record, Fig 2's construction
HE = pd.read_parquet(F/"eval/half_vd_review/lakesp_vD_halfexact_graph.parquet", columns=["lake_id", "date", "dv_SWOTAE", "dv_3DLAKES", "dv_staticPLD", "dv_best", "dv_source"])
HE["lake_id"] = HE.lake_id.astype(str); HE["date"] = pd.to_datetime(HE.date)
assert not HE.duplicated(["lake_id", "date"]).any(), "duplicate lake-days in the HALF-exact record"
HE["v"] = HE.dv_SWOTAE.fillna(HE.dv_3DLAKES).fillna(HE.dv_staticPLD)
HE["src"] = np.select([HE.dv_SWOTAE.notna(), HE.dv_3DLAKES.notna()], ["SWOTAE", "3DLAKES"], "staticPLD")
HE_MIX = dict(rows=len(HE), finite=dict(SWOTAE=int(HE.dv_SWOTAE.notna().sum()), D3LAKES=int(HE.dv_3DLAKES.notna().sum()), staticPLD=int(HE.dv_staticPLD.notna().sum())),
              v_source=HE[HE.v.notna()].src.value_counts().to_dict(), dv_source=HE.dv_source.astype(str).value_counts().to_dict(),
              v_equals_dv_best=float(np.isclose(HE.v, HE.dv_best, equal_nan=True).mean()))
HE = HE[HE.v.notna()]; HEV = {k: g.set_index("date")[["v", "src"]] for k, g in HE.groupby("lake_id")}

def swot(members):
    """Return the SWOT storage of a reservoir in the comparison window, one row per day.

    The storage (column swot) is the sum over the member lakes `members`, on the days when every
    member has a value. The columns all_swotae and any_static flag the days on which every member
    uses SWOT-AE, or any member uses static PLD. Returns None when a member lake has no series in
    the HALF-filtered record.
    """
    if any(m not in HEV for m in members): return None
    v = pd.concat([HEV[m].v for m in members], axis=1).dropna(); s = pd.concat([HEV[m].src for m in members], axis=1).reindex(v.index)
    o = pd.DataFrame({"swot": v.sum(axis=1), "all_swotae": (s == "SWOTAE").all(axis=1), "any_static": (s == "staticPLD").any(axis=1)})
    return o[(o.index >= W0) & (o.index <= W1)]

def score(o, s):
    """Score SWOT storage (s) against gauge storage (o) on the matched days of one reservoir.

    Both series are centered on their mean over the matched days. Returns r, nRMSE (RMSE / gauge
    range), NSE, VR (SWOT standard deviation / gauge standard deviation), and the centered gauge
    and SWOT series.
    """
    oc, sc = o - o.mean(), s - s.mean(); r = float(np.corrcoef(oc, sc)[0, 1]); vr = float(sc.std()/oc.std())
    nse = float(1 - np.sum((sc - oc)**2)/np.sum(oc**2)); assert abs(nse - (2*vr*r - vr*vr)) < 1e-9
    return r, float(np.sqrt(np.mean((sc - oc)**2))/(oc.max() - oc.min())), nse, vr, oc, sc

def run(sites_csv, truth_parquet, ids):
    """Score every reservoir of `ids` against its gauge.

    sites_csv, truth_parquet: the test set's site table (member lakes, tier, gauge source and
    method) and gauge storage. Returns a table with one row per reservoir (the scores, or the
    reason in column `drop` when the reservoir is not scored) and a dict {DAM_ID: (centered gauge
    series, centered SWOT series)} of the scored reservoirs.
    """
    S = pd.read_csv(sites_csv, dtype={"members": str, "basin": str, "node": str}).drop_duplicates("DAM_ID").set_index("DAM_ID")
    T = pd.read_parquet(truth_parquet, columns=["DAM_ID", "date", "storage_mcm"]); T["date"] = pd.to_datetime(T.date)
    T = T[(T.date >= W0) & (T.date <= W1) & T.storage_mcm.notna()]; assert not T.duplicated(["DAM_ID", "date"]).any()
    TG = {int(k): g.set_index("date").storage_mcm.astype(float) for k, g in T.groupby("DAM_ID")}
    rows, CEN = [], {}
    for d in ids:
        r = S.loc[d]; mem = str(r.members).split(";"); sw = swot(mem); g = TG.get(int(d))
        kind = "elevation-based" if "elev" in f"{r.source} {r.method}".lower() else "storage"
        row = dict(DAM_ID=int(d), name=r.DAM_NAME, state=r.STATE, tier=r.site_tier, gauge_source=r.source, gauge_method=r.method, gauge_kind=kind, n_members=len(mem),
                   n_swot_days=0 if sw is None else len(sw), n_gauge_days=0 if g is None else len(g), n_pairs=0)
        if sw is None: row["drop"] = "no HALF-exact series for a member lake"
        elif g is None: row["drop"] = "no gauge data in the window"
        else:
            j = sw.join(g.rename("gauge"), how="inner"); row["n_pairs"] = len(j)
            if len(j) < MINP: row["drop"] = f"fewer than {MINP} matched days"
            elif np.std(j.gauge.values) == 0 or np.std(j.swot.values) == 0: row["drop"] = "constant series on the matched days"
            else:
                rr, nr, ns, vr, oc, sc = score(j.gauge.values, j.swot.values); CEN[int(d)] = (oc, sc)
                row.update(drop="", r=rr, nrmse=nr, nse=ns, vr=vr, share_swotae=float(j.all_swotae.mean()), share_staticpld=float(j.any_static.mean()), first=str(j.index.min().date()),
                           last=str(j.index.max().date()), median_gap_days=float(np.median(np.diff(j.index.values).astype("timedelta64[D]").astype(float))) if len(j) > 1 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows), CEN

def summ(df, CEN):
    """Summarize one comparison set as a dict.

    df, CEN: the table and the centered pairs returned by run(). The summary holds the counts of
    reservoirs (in the set, per reason for not being scored, scored), medians and quartiles of
    the scores, r over all pairs pooled (in MCM and standardized per reservoir), splits by gauge
    kind and by SWOT source, and the six reservoirs with the lowest r.
    """
    k = df[df["drop"] == ""]; q = lambda c, p: float(np.percentile(k[c], p))
    oc = np.concatenate([CEN[d][0] for d in k.DAM_ID]); sc = np.concatenate([CEN[d][1] for d in k.DAM_ID])
    zo = np.concatenate([CEN[d][0]/CEN[d][0].std() for d in k.DAM_ID]); zs = np.concatenate([CEN[d][1]/CEN[d][1].std() for d in k.DAM_ID])
    return dict(n_set=len(df), n_no_swot_series=int((df["drop"] == "no HALF-exact series for a member lake").sum()),
                n_no_gauge=int((df["drop"] == "no gauge data in the window").sum()),
                n_dropped_lt10=int((df["drop"] == f"fewer than {MINP} matched days").sum()), n_constant=int((df["drop"] == "constant series on the matched days").sum()),
                n_scored=len(k), pairs_total=int(k.n_pairs.sum()), pairs_median=float(k.n_pairs.median()),
                r_median=float(k.r.median()), r_q25=q("r", 25), r_q75=q("r", 75), n_r_ge_090=int((k.r >= 0.9).sum()), share_r_ge_090=float((k.r >= 0.9).mean()),
                n_r_ge_080=int((k.r >= 0.8).sum()), n_r_lt_050=int((k.r < 0.5).sum()),
                r_pooled_mcm=float(np.corrcoef(oc, sc)[0, 1]), r_pooled_standardised=float(np.corrcoef(zo, zs)[0, 1]),
                nrmse_median=float(k.nrmse.median()), nrmse_q25=q("nrmse", 25), nrmse_q75=q("nrmse", 75),
                nse_median=float(k.nse.median()), nse_q25=q("nse", 25), nse_q75=q("nse", 75), vr_median=float(k.vr.median()),
                share_swotae_median=float(k.share_swotae.median()), pairs_share_swotae=float((k.share_swotae*k.n_pairs).sum()/k.n_pairs.sum()),
                pairs_share_staticpld=float((k.share_staticpld*k.n_pairs).sum()/k.n_pairs.sum()), median_gap_days=float(k.median_gap_days.median()),
                by_gauge_kind={kd: dict(n=int(len(g)), r_median=float(g.r.median()), n_r_ge_090=int((g.r >= 0.9).sum()), nrmse_median=float(g.nrmse.median()))
                               for kd, g in k.groupby("gauge_kind")},
                by_swot_source={lab: dict(n=int(len(g)), r_median=float(g.r.median()), n_r_ge_090=int((g.r >= 0.9).sum()), nrmse_median=float(g.nrmse.median()))
                                for lab, g in (("SWOT-AE / 3D-LAKES only", k[k.share_staticpld == 0]), ("static PLD on some days", k[k.share_staticpld > 0]))},
                lowest_r=k.nsmallest(6, "r")[["DAM_ID", "name", "state", "n_pairs", "r", "vr"]].to_dict("records"))

# ---------------- set 1: Fig 1d's 272 (kit_h, as fig1_k.py)
os.environ.setdefault("HIND", "2010"); sys.path.insert(0, str(FIGD)); os.chdir(FIGD)
import kit_h as K
s_ = K.C["S_"][K.C["S_"].PERIOD == "2023-25"]; IDS1 = sorted(int(x) for x in set.intersection(*[set(s_[s_.MODEL == m].DAM_ID) for m in K.M4]))
assert len(IDS1) == 272
D1, C1 = run(F/"testsets/regime1x_2023_2026/sites.csv", F/"testsets/regime1x_2023_2026/truth_clean_v1x.parquet", IDS1)
assert D1.tier.isin(["High", "Medium"]).all()
# ---------------- set 2: regime1x_plus High/Medium, 4-model and 3-model (summarize_plus.py's rule coverage)
CA = pd.read_csv(X/"sizing/candidates.csv").drop_duplicates("GRAND_ID").set_index("GRAND_ID")
SP = pd.read_csv(X/"testsets/regime1x_plus_2023_2026/sites.csv").drop_duplicates("DAM_ID"); idx = pd.Index(SP.DAM_ID.values)
cov = lambda c: CA[c].reindex(idx, fill_value=False)                              # same sets as summarize_plus.py
hm = SP.site_tier.isin(["High", "Medium"]).values
IN4 = hm & (cov("gdrom") & cov("istarf") & cov("modrom_cal")).values.astype(bool); IN3 = hm & (cov("gdrom") & cov("istarf")).values.astype(bool)
assert IN4.sum() == 307 and IN3.sum() == 556 and not (IN4 & ~IN3).any()
D2, C2 = run(X/"testsets/regime1x_plus_2023_2026/sites.csv", X/"testsets/regime1x_plus_2023_2026/truth_clean_v1x.parquet", sorted(int(x) for x in SP.DAM_ID[IN3]))
D2["in_4model"] = D2.DAM_ID.isin(set(int(x) for x in SP.DAM_ID[IN4])); D2["in_3model"] = True
D2 = D2.merge(SP.set_index("DAM_ID").added_by.rename("added_by"), left_on="DAM_ID", right_index=True, how="left")
D1.to_csv(OUT/"per_reservoir_fig1d.csv", index=False, float_format="%.6g"); D2.to_csv(OUT/"per_reservoir_plus.csv", index=False, float_format="%.6g")
# reservoirs present in both sets: do both give the same r?
b = D1.merge(D2, on="DAM_ID", suffixes=("_1", "_2")); bb = b[(b.drop_1 == "") & (b.drop_2 == "")]
SUM = {"window": [str(W0.date()), str(W1.date())], "min_pairs": MINP, "half_exact_record": HE_MIX,
       "fig1d_272": summ(D1, C1), "plus_4model_307": summ(D2[D2.in_4model], C2), "plus_3model_556": summ(D2, C2),
       "both_sets": dict(n_common=len(b), n_scored_both=len(bb), max_abs_dr=float((bb.r_1 - bb.r_2).abs().max()), n_dr_gt_1e9=int(((bb.r_1 - bb.r_2).abs() > 1e-9).sum()))}
# ---------------- leakage: what SWOT-GAT trained on
cfg = [json.load(open(CONUS/r/"config.json")) for r in RUNS]
def flat(d, p=""):
    """Return a nested dict as one flat dict whose keys are the nested keys joined by dots."""
    o = {}
    for k_, v_ in d.items(): o.update(flat(v_, f"{p}{k_}.") if isinstance(v_, dict) else {f"{p}{k_}": v_})
    return o
f0, f1 = flat(cfg[0]), flat(cfg[1]); diff = sorted(k_ for k_ in set(f0) | set(f1) if f0.get(k_) != f1.get(k_))
feats = [x for k_, v_ in f0.items() if k_.startswith("features.") and isinstance(v_, list) for x in v_]
store = Path(f0["zarr_dir"]); svars = sorted({a.name for g in store.iterdir() if g.is_dir() for a in g.iterdir() if not a.name.startswith(".")})
bad = [x for x in feats + svars if any(w in x.lower() for w in ("gauge", "truth", "storage_mcm", "insitu", "resops"))]
LEAK = dict(runs=RUNS, config_keys_that_differ=diff, zarr_dir=str(store), n_store_groups=sum(1 for g in store.iterdir() if g.is_dir()), store_variables=svars,
            targets=f0.get("features.target"), fallback=f0.get("fallback_targets.dv_SWOTAE.source"), dynamic=f0.get("features.dynamic.era5"),
            extra_dynamic_vars=f0.get("extra_dynamic_vars"), n_static=len(f0.get("features.static", [])), static=f0.get("features.static"),
            attributes_file=f0.get("attributes_file"), train_date_range=f0.get("train_date_range"), use_obs_memory=f0.get("model_args.use_obs_memory"),
            names_matching_gauge_terms=bad)
SUM["leakage"] = {k_: LEAK[k_] for k_ in ("config_keys_that_differ", "targets", "fallback", "extra_dynamic_vars", "store_variables", "names_matching_gauge_terms")}
json.dump(LEAK, open(OUT/"leakage_check.json", "w"), indent=1); json.dump(SUM, open(OUT/"summary.json", "w"), indent=1, default=float)
print(json.dumps(SUM, indent=1, default=float))

# ---------------- summary.md
f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"; pc = lambda x: f"{100*x:.0f}%"
def row(lab, s):
    """Return the row of the results table of summary.md for the summary `s`, labeled `lab`."""
    return (f"| {lab} | {s['n_set']} | {s['n_no_swot_series']} | {s['n_dropped_lt10']} | {s['n_scored']} | {f3(s['r_median'])} | {f3(s['r_q25'])}–{f3(s['r_q75'])} | "
            f"{s['n_r_ge_090']} ({pc(s['share_r_ge_090'])}) | {f3(s['r_pooled_mcm'])} | {f3(s['r_pooled_standardised'])} | {f3(s['nrmse_median'])} ({f3(s['nrmse_q25'])}–{f3(s['nrmse_q75'])}) | "
            f"{f2(s['nse_median'])} | {f2(s['vr_median'])} | {s['pairs_total']:,} ({s['pairs_median']:.0f}) |")
A, B4, B3 = SUM["fig1d_272"], SUM["plus_4model_307"], SUM["plus_3model_556"]; HM = SUM["half_exact_record"]
E1 = D1[(D1["drop"] == "") & (D1.gauge_kind == "elevation-based")]; NEL = len(E1); NPLD = int(E1.gauge_method.astype(str).str.contains("PLD").sum())
L = ["# SWOT vs gauge storage anomalies on SWOT pass days", "",
     f"Owen 2026-11-04 (hub relay): \"run this corrilation betweeen SWOT and gauge test\", for the draft sentence \"SWOT-derived measurements of storage anomalies closely track gauge records where available (r = 0.9x)\". Built by `swot_vs_gauge.py` (run with `run_svg.sh`); numbers below are written from `summary.json`.", "",
     "## Method", "",
     "- **SWOT:** the series Figure 2 plots as \"SWOT observations\": the HALF-exact vD record (`eval/half_vd_review/lakesp_vD_halfexact_graph.parquet`), dv_SWOTAE, else dv_3DLAKES, else dv_staticPLD per lake (equal to the record's own `dv_best` on 100% of rows), summed over a reservoir's member lakes on days every member has a value. This is also how SWOT-GAT's training target is built (`CONUS/zarr_store_vd2026_halfexact/README.md`).",
     "- **Gauge:** `storage_mcm` from the set's `truth_clean_v1x.parquet`, same calendar day.",
     f"- **Pairing:** SWOT pass days only, {SUM['window'][0]} to {SUM['window'][1]}. Median gap between matched days: {A['median_gap_days']:.0f} days.",
     "- **Centring:** both series centred per reservoir over the matched days.",
     f"- **Statistics per reservoir:** Pearson r; nRMSE = RMSE ÷ gauge range on the matched days (the figure set's definition); NSE = 1 − SSE/SS on the centred series (= 2·VR·r − VR²); VR = SWOT sd ÷ gauge sd. At least {MINP} matched days, else dropped. Pooled r: all centred pairs pooled, in MCM (dominated by large reservoirs), and with each reservoir standardised by its own sd.",
     "- **Sets:**",
     "  - (1) the 272 High/Medium reservoirs of Fig 1d, which all four models score in the SWOT Period (`kit_h` C[\"S_\"], as `fig1_k.py`); truth and members from `testsets/regime1x_2023_2026/`.",
     "  - (2) `gauges_expand/testsets/regime1x_plus_2023_2026/` High/Medium, with rule coverage as `gauges_expand/summarize_plus.py`: 4-model (307) and 3-model (556; GDROM + ISTARF, which here is every High/Medium site).",
     "", "## Results", "",
     "| set | n | no SWOT series | dropped, < 10 days | scored | median r | r IQR | r ≥ 0.90 | pooled r (MCM) | pooled r (standardised) | median nRMSE (IQR) | median NSE | median VR | matched days (median per reservoir) |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
     row("(1) Fig 1d, SWOT Period", A), row("(2) regime1x_plus, 4-model", B4), row("(2) regime1x_plus, 3-model", B3), "",
     f"- The {SUM['both_sets']['n_scored_both']} reservoirs scored in both (1) and (2) give identical r (max |Δr| = {SUM['both_sets']['max_abs_dr']:.1g}).",
     f"- \"No SWOT series\": a member lake has no HALF-exact series (lakes HALF's Level-2 coverage check rejects, e.g. Glen Canyon / Lake Powell and El Vado). SWOT-GAT still predicts there, which is why they are in the Fig 1d set. Set (1): {A['n_no_swot_series']}; 4-model: {B4['n_no_swot_series']}; 3-model: {B3['n_no_swot_series']}.",
     f"- The 10-day rule drops none in set (1) and the 4-model set, and {B3['n_dropped_lt10']} in the 3-model set (Marsh Lake, MN, 2 days; Franklin County, TX, 9 days).",
     f"- r < 0.5 at {A['n_r_lt_050']} reservoirs in set (1): " + "; ".join(f"{x['name']}, {x['state']} (r = {x['r']:.2f}, VR {x['vr']:.1f}, {x['n_pairs']} days)" for x in A["lowest_r"]) + ". The four lowest are near-constant-level reservoirs (three of Texas's Highland Lakes and Shadow Mountain, CO): the gauge barely moves, so SWOT's noise dominates and VR is 3–16.",
     "", "**By SWOT source** (set 1): " + "; ".join(f"{lab}: n = {v['n']}, median r {f3(v['r_median'])}, r ≥ 0.90 at {v['n_r_ge_090']}, median nRMSE {f3(v['nrmse_median'])}" for lab, v in A["by_swot_source"].items()) + ".",
     f"Over all matched days of set 1, {pc(A['pairs_share_swotae'])} are SWOT-AE on every member lake and {pc(A['pairs_share_staticpld'])} use the static-PLD fallback (SWOT water level × the fixed PLD area) on at least one member. In the whole HALF-exact record: SWOT-AE {HM['v_source'].get('SWOTAE', 0):,}, 3D-LAKES {HM['v_source'].get('3DLAKES', 0):,}, static PLD {HM['v_source'].get('staticPLD', 0):,} of {HM['rows']:,} lake-days.",
     "", "**By gauge kind** (set 1): " + "; ".join(f"{kd}: n = {v['n']}, median r {f3(v['r_median'])}, r ≥ 0.90 at {v['n_r_ge_090']}" for kd, v in A["by_gauge_kind"].items()) + f". Of the {NEL} elevation-based gauges, {NEL - NPLD} are converted to storage with an elevation–storage rating curve and {NPLD} with gauge level × the PLD lake area (sites.csv `method`). Where both the gauge and the SWOT value use the PLD area, r compares SWOT water level with gauge level directly; a constant area factor does not change r.",
     "", "## Leakage check: did any gauge data go into SWOT-GAT training?", "",
     "No evidence of it:",
     f"- **Configs:** the two A365 seeds (`{RUNS[0]}`, `{RUNS[1]}`) differ only in `{', '.join(LEAK['config_keys_that_differ'])}`.",
     f"- **Targets:** {', '.join(LEAK['targets'])} (fallback {LEAK['fallback']}), from `{LEAK['zarr_dir']}`. The store's variables, over all {LEAK['n_store_groups']} basin groups: {', '.join(LEAK['store_variables'])}. None is a gauge record.",
     f"- **Inputs:** {len(LEAK['dynamic'])} ERA5 forcings ({', '.join(LEAK['dynamic'])}); extra dynamic vars {LEAK['extra_dynamic_vars']}; {LEAK['n_static']} static attributes (HydroATLAS-style catchment attributes plus GDW dam attributes: capacity, height, depth, degree of regulation, purposes). None comes from gauge storage.",
     f"- Feature and store names matching gauge / truth / storage_mcm / insitu / resops: {LEAK['names_matching_gauge_terms'] or 'none'}.",
     "- **Training code:** `swot-ml/src` has no reference to the gauge truth files, the test sets, `storage_mcm`, `gauges_expand` or ResOps (`leakage_code_grep.txt`).",
     "- **SWOT target construction:** SWOT-AE curves come from LakeSP area–elevation pairs, 3D-LAKES and static PLD from SWOT levels and lake geometry (`eval/half_vd_review/HALF_vD_conformance.md`). The gauge comparison there was \"diagnosis only; nothing was dropped or re-tiered on this basis\". The only load-time drop mask in the A365 config is for river reaches (`target_drop_masks.dv_river`).",
     "- **Caveat:** at these reservoirs SWOT-GAT did train on these same SWOT observations (the SWOT Period is its training period). That does not affect this test, which compares SWOT with gauges and involves no model output.",
     "", "## Files", "",
     "- `per_reservoir_fig1d.csv`, `per_reservoir_plus.csv`: one row per reservoir (`drop` gives the reason when not scored; `in_4model` / `in_3model` in the plus file).",
     "- `summary.json`, `leakage_check.json`, `leakage_code_grep.txt`; code `swot_vs_gauge.py`, `run_svg.sh`; logs in `logs/`.", ""]
(OUT/"summary.md").write_text("\n".join(L))
