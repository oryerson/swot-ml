"""Choose the four example reservoirs of Figure 1's 1981-2020 rows by a fixed rule.

The examples must have a complete gauge record over 1981-2020 and SWOT data, and they must span the
range of SWOT-GAT skill: one reservoir near each of the 90th, 60th, 35th and 10th percentile of
SWOT-GAT's 1981-2020 NSE. Choosing by rule keeps the examples from being hand-picked.

Usage: bin/run 07_evaluation/c_scoring/full_record/picks.py      (no arguments)
  Run it after full.py, full_record/score_x26span.py and overlap50/fig1_nums.py.

Rule
  Percentile. The rank of SWOT-GAT's 1981-2020 NSE among all reservoirs of confidence tier High or
  Medium that SWOT-GAT scores over 1981-2020 (419 in the paper's run).
  Pool. A reservoir is eligible if
    - its gauge has days with the `use` flag set on at least 95% of the 14,610 days from
      1981-01-01 to 2020-12-31;
    - GDROM, ISTARF, MODROM-cal and the rule-model ensemble are all scored there over 1981-2020;
    - it is in the matched 4-model set of the SWOT Period (fig1_o50.json, key "2023-25");
    - every member lake has a SWOT storage series (LakeSP version D after the HALF filter;
      dv_SWOTAE, else dv_3DLAKES, else dv_staticPLD), and the reservoir has at least 20 SWOT days
      (days on which every member has a value) within its scored span, which ends no later than
      2026-06-12;
    - it is not one of the four reservoirs already shown in the 2010-2020 rows (EXIST: Black
      Butte, Norfork, Rathbun, Trinity).
  Pick. For each target percentile (90, 60, 35, 10) the pool reservoir nearest in percentile,
  followed by the next two as alternates. A reservoir picked for one target is not offered again.

Reads (paths relative to $SWOTGAT_WORK/CONUS_storage_fork; B = eval/grl_figs/intermediate/c5_o50_20261113)
  B/score/score_full/fullrecord_site_window.csv   1981-2020 scores (label SWOT-GNN = SWOT-GAT)
  B/score/sites_2325span_g.csv                    SWOT Period scores (label v10pm = SWOT-GAT)
  eval/overlap50/fig1_o50.json                    matched sets
  testsets/regime3_full_1980_2020/truth_clean.parquet   the 1980-2020 gauge test set
  testsets/regime1x_2023_2026/sites.csv           member lakes
  eval/score2026/spans_2023_2026.csv              SWOT spans
  eval/half_vd_review/lakesp_vD_halfexact_graph.parquet   filtered LakeSP version D
  eval/grl_figs/sites_v5_g.csv, B/kit_d/eco_l1_sites.csv  metadata only (name, state, use,
                                                          capacity, ecoregion group)

Writes B/score/picks.json (the picks and alternates per target, with counts) and
B/score/picks_pool.csv (every candidate with its percentile, scores and the pool criteria).
"""
from swotgat.paths import WORK
import json
from pathlib import Path
import numpy as np, pandas as pd
F = Path(f"{WORK}/CONUS_storage_fork"); B = F/"eval/grl_figs/intermediate/c5_o50_20261113"; O50 = F/"eval/overlap50"
HM = ["High", "Medium"]; EXIST = [144, 1042, 924, 131]; TARGETS = [90, 60, 35, 10]
FR = pd.read_csv(B/"score/score_full/fullrecord_site_window.csv"); FR = FR[(FR.window == "1981-2020") & FR.site_tier.isin(HM)]
gat = FR[FR.model == "SWOT-GNN"].set_index("DAM_ID"); PCT = 100*gat.nse.rank(pct=True)
have = lambda m: set(FR[FR.model == m].DAM_ID)
T3 = F/"testsets/regime3_full_1980_2020"; TR = pd.read_parquet(T3/"truth_clean.parquet", columns=["DAM_ID", "date", "use"]); TR["date"] = pd.to_datetime(TR.date)
TR = TR[TR.use.fillna(False).astype(bool) & (TR.date >= "1981-01-01") & (TR.date <= "2020-12-31")]; ND = len(pd.date_range("1981-01-01", "2020-12-31"))
comp = TR.groupby("DAM_ID").date.nunique()/ND
J50 = json.load(open(O50/"fig1_o50.json")); SP4 = set(J50["2023-25"]["ids"])
X1 = pd.read_csv(F/"testsets/regime1x_2023_2026/sites.csv", dtype={"members": str}).set_index("DAM_ID")
SPX = pd.read_csv(F/"eval/score2026/spans_2023_2026.csv").set_index("DAM_ID")
HE = pd.read_parquet(F/"eval/half_vd_review/lakesp_vD_halfexact_graph.parquet", columns=["lake_id", "date", "dv_SWOTAE", "dv_3DLAKES", "dv_staticPLD"])
HE["lake_id"] = HE.lake_id.astype(str); HE["date"] = pd.to_datetime(HE.date); HE["v"] = HE.dv_SWOTAE.fillna(HE.dv_3DLAKES).fillna(HE.dv_staticPLD)
HEV = {k: g.set_index("date").v.dropna() for k, g in HE.groupby("lake_id")}
def n_swot(dam):
    """Return the number of SWOT days of a reservoir within its span, clipped at 2026-06-12.

    A SWOT day is a day on which every member lake has a value. The result is 0 when the dam is
    not in the sites table of the 2023-2026 test set, has no span, or has a member lake without a
    SWOT series.
    """
    if dam not in X1.index or dam not in SPX.index or pd.isna(SPX.vD_start[dam]): return 0
    mem = str(X1.members[dam]).split(";")
    if any(m not in HEV for m in mem): return 0
    v = pd.concat([HEV[m] for m in mem], axis=1).dropna().sum(axis=1)
    return int(((v.index >= pd.Timestamp(SPX.vD_start[dam])) & (v.index <= min(pd.Timestamp(SPX.vD_end[dam]), pd.Timestamp("2026-06-12")))).sum())
X26 = pd.read_csv(B/"score/sites_2325span_g.csv"); x26 = X26[(X26.MODEL == "v10pm") & X26.tier.isin(HM)].set_index("DAM_ID")
meta = pd.read_csv(F/"eval/grl_figs/sites_v5_g.csv").drop_duplicates("DAM_ID").set_index("DAM_ID")
eco = pd.read_csv(B/"kit_d/eco_l1_sites.csv").set_index("DAM_ID").egroup
rows = []
for d in gat.index:
    r = dict(DAM_ID=int(d), name=meta.name.get(d), state=meta.state.get(d), use=meta.use.get(d), cap=meta.cap.get(d), eco=eco.get(d), pct=float(PCT[d]),
             nse=float(gat.nse[d]), nrmse=float(gat.nrmse[d]), vr=float(gat.amp[d]), n_days=int(gat.n[d]), gauge_complete=float(comp.get(d, 0.0)),
             rules=all(d in have(m) for m in ("GDROM", "ISTARF", "MODROM-cal", "RULE-ENS")), in_swot4=d in SP4, n_swot=n_swot(int(d)),
             swot_nse=float(x26.nse.get(d, np.nan)), swot_vr=float(x26.amp.get(d, np.nan)))
    r["pool"] = r["gauge_complete"] >= 0.95 and r["rules"] and r["in_swot4"] and r["n_swot"] >= 20 and int(d) not in EXIST
    rows.append(r)
P = pd.DataFrame(rows).sort_values("pct", ascending=False); P.to_csv(B/"score/picks_pool.csv", index=False, float_format="%.6g")
pool = P[P["pool"]].copy(); used = set(); OUT = dict(n_scored_1981_2020=int(len(gat)), n_pool=int(len(pool)),
    pool_rule="High/Medium; gauge use-days >= 95% of 1981-01-01..2020-12-31; all three rules and the ensemble scored 1981-2020; in the SWOT Period 4-model set; >= 20 SWOT passes; not the existing four",
    counts=dict(complete95=int((P.gauge_complete >= 0.95).sum()), rules=int(P.rules.sum()), swot4=int(P.in_swot4.sum()), swot20=int((P.n_swot >= 20).sum())), targets={})
for t in TARGETS:
    c = pool[~pool.DAM_ID.isin(used)].assign(dist=lambda x: (x.pct - t).abs()).sort_values("dist").head(3)
    used |= set(c.DAM_ID.head(1)); OUT["targets"][str(t)] = c.drop(columns=["pool"]).to_dict("records")
json.dump(OUT, open(B/"score/picks.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in OUT.items() if k != "targets"}, indent=1))
for t, c in OUT["targets"].items():
    print(f"--- target {t}th percentile")
    for x in c: print(f"  {x['DAM_ID']:5d} {x['name']:<28} {x['state']:<14} {str(x['use']):<17} cap {x['cap']:>8.0f} {x['eco']:<5} pct {x['pct']:5.1f} NSE {x['nse']:+.2f} VR {x['vr']:.2f} nRMSE {x['nrmse']:.2f} | gauge {100*x['gauge_complete']:.1f}% | SWOT {x['n_swot']} passes, NSE {x['swot_nse']:+.2f} VR {x['swot_vr']:.2f}")
