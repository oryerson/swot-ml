"""Choose the example reservoir of Figure 1's first row (2010-2020, high SWOT-GAT skill) by a fixed rule.

The reservoir shown there before, Black Butte, is one of the five reservoirs removed from every
rule-model panel because ISTARF storage is pinned full or empty (EX5; 08_figures/excl_d.py). This
script picks its replacement: a reservoir near the 90th percentile of SWOT-GAT's 2010-2020 NSE.

Usage: bin/run 07_evaluation/c_scoring/full_record/picks_d.py      (no arguments)
  Run it after pinned_check.py and picks.py.

Rule
  Percentile. The rank of SWOT-GAT's 2010-2020 NSE within the matched 4-model set of 2010-2020
  (fig1_o50.json) after removing the five excluded reservoirs (`pct5`). The percentile after
  removing seven (the five plus dams 386 and 567) is recorded too (`pct7`).
  Pool. A reservoir of confidence tier High or Medium is eligible if
    - it is also in the matched 4-model set of the SWOT Period and not one of the five;
    - its gauge has days with the `use` flag set on at least 95% of 2010-01-01 to 2020-12-31;
    - every member lake has a SWOT storage series (LakeSP version D after the HALF filter) and
      the reservoir has at least 20 SWOT days within its scored span;
    - it is visibly seasonal: seas_frac >= 0.5, the share of the 2010-2020 gauge variance that
      its annual harmonic explains (column of score_2010.py);
    - it is not one of the other seven rows of Figure 1 (ROWS);
    - ISTARF is not pinned there: pinned_check.csv flags a dam when at least 95% of the days of
      either ISTARF run are at capacity or empty. In the paper's run 14 dams are flagged; they
      include the five excluded reservoirs and dams 386 and 567.
  Pick. The pool reservoir nearest the 90th percentile, followed by the next two as alternates.

Reads (paths relative to $SWOTGAT_WORK/CONUS_storage_fork; B = eval/grl_figs/intermediate/c5_o50_20261113)
  B/score/pinned_check.csv                 pinned_check.py
  B/score/picks_pool.csv                   picks.py (must exist; no value is taken from it)
  eval/overlap50/fig1_o50.json             matched sets
  eval/overlap50/score/sites_2010_g.csv, sites_2325span_g.csv   scores (label v10pm = SWOT-GAT)
  testsets/regime3_full_1980_2020/truth_clean.parquet           the 1980-2020 gauge test set
  testsets/regime1x_2023_2026/sites.csv, eval/score2026/spans_2023_2026.csv,
  eval/half_vd_review/lakesp_vD_halfexact_graph.parquet         member lakes, SWOT spans, SWOT series

Writes B/score/picks_d.json: set sizes, the pinned dams, and the pick with its two alternates.
"""
from swotgat.paths import WORK
import json
from pathlib import Path
import numpy as np, pandas as pd
F = Path(f"{WORK}/CONUS_storage_fork"); B = F/"eval/grl_figs/intermediate/c5_o50_20261113"; O50 = F/"eval/overlap50"
EX5 = [144, 1095, 1125, 1790, 1916]; EX7 = EX5 + [386, 567]; ROWS = [1042, 924, 131, 517, 1006, 384, 1279]
PIN = set(pd.read_csv(B/"score/pinned_check.csv").query("pinned").DAM_ID)
J = json.load(open(O50/"fig1_o50.json")); S = pd.read_csv(O50/"score/sites_2010_g.csv"); S = S[(S.MODEL == "v10pm") & S.tier.isin(["High", "Medium"]) & S.nrmse.notna()].set_index("DAM_ID")
def pct(ex):
    """Return SWOT-GAT's NSE percentiles within the matched 2010-2020 set without the dams in ex.

    The result is a pair: the percentile rank (0-100) of the 2010-2020 NSE per DAM_ID, and the
    number of reservoirs in the reduced set.
    """
    ids = [i for i in J["2010-20"]["ids"] if i not in ex]; return 100*S.loc[ids].nse.rank(pct=True), len(ids)
P5, N5 = pct(EX5); P7, N7 = pct(EX7)
TR = pd.read_parquet(F/"testsets/regime3_full_1980_2020/truth_clean.parquet", columns=["DAM_ID", "date", "use"]); TR["date"] = pd.to_datetime(TR.date)
TR = TR[TR.use.fillna(False).astype(bool) & (TR.date >= "2010-01-01") & (TR.date <= "2020-12-31")]; comp = TR.groupby("DAM_ID").date.nunique()/len(pd.date_range("2010-01-01", "2020-12-31"))
PP = pd.read_csv(B/"score/picks_pool.csv").set_index("DAM_ID")             # table of picks.py; loaded but not used below
X1 = pd.read_csv(F/"testsets/regime1x_2023_2026/sites.csv", dtype={"members": str}).set_index("DAM_ID"); SPX = pd.read_csv(F/"eval/score2026/spans_2023_2026.csv").set_index("DAM_ID")
HE = pd.read_parquet(F/"eval/half_vd_review/lakesp_vD_halfexact_graph.parquet", columns=["lake_id", "date", "dv_SWOTAE", "dv_3DLAKES", "dv_staticPLD"])
HE["lake_id"] = HE.lake_id.astype(str); HE["date"] = pd.to_datetime(HE.date); HE["v"] = HE.dv_SWOTAE.fillna(HE.dv_3DLAKES).fillna(HE.dv_staticPLD)
HEV = {k: g.set_index("date").v.dropna() for k, g in HE.groupby("lake_id")}
def n_swot(dam):
    """Return the number of SWOT days of a reservoir within its span, clipped at 2026-06-12.

    A SWOT day is a day on which every member lake has a value. The result is 0 when the dam is
    not in the sites table of the 2023-2026 test set, has no span, or has a member lake without a
    SWOT series. The function is the same as in picks.py.
    """
    if dam not in X1.index or dam not in SPX.index or pd.isna(SPX.vD_start[dam]): return 0
    mem = str(X1.members[dam]).split(";")
    if any(m not in HEV for m in mem): return 0
    v = pd.concat([HEV[m] for m in mem], axis=1).dropna().sum(axis=1)
    return int(((v.index >= pd.Timestamp(SPX.vD_start[dam])) & (v.index <= min(pd.Timestamp(SPX.vD_end[dam]), pd.Timestamp("2026-06-12")))).sum())
X26 = pd.read_csv(O50/"score/sites_2325span_g.csv"); x26 = X26[(X26.MODEL == "v10pm") & X26.tier.isin(["High", "Medium"])].set_index("DAM_ID")
rows = []
for d in P5.index:
    r = dict(DAM_ID=int(d), name=S.name.get(d), state=S.state.get(d), use=S.use.get(d), cap=float(S.cap.get(d)), pct5=float(P5[d]), pct7=float(P7.get(d, np.nan)),
             nse=float(S.nse[d]), vr=float(S.amp[d]), nrmse=float(S.nrmse[d]), seas_frac=float(S.seas_frac[d]), gauge_complete=float(comp.get(d, 0.0)),
             in_swot4=d in J["2023-25"]["ids"] and d not in EX5, n_swot=n_swot(int(d)), swot_nse=float(x26.nse.get(d, np.nan)), swot_vr=float(x26.amp.get(d, np.nan)))
    r["pool"] = (r["gauge_complete"] >= 0.95 and r["in_swot4"] and r["n_swot"] >= 20 and r["seas_frac"] >= 0.5 and int(d) not in ROWS and int(d) not in PIN)
    rows.append(r)
D = pd.DataFrame(rows); pool = D[D["pool"]].assign(dist=lambda x: (x.pct5 - 90).abs()).sort_values("dist")
OUT = dict(n_set_ex5=N5, n_set_ex7=N7, n_pool=int(len(pool)), pinned_excluded_from_pool=sorted(int(i) for i in PIN), pick=pool.head(3).drop(columns=["pool"]).to_dict("records"))
json.dump(OUT, open(B/"score/picks_d.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in OUT.items() if k != "pick"}, indent=1))
for x in OUT["pick"]: print(f"  {x['DAM_ID']:5d} {x['name']:<26} {x['state']:<12} {str(x['use']):<16} cap {x['cap']:>7.0f} pct {x['pct5']:.1f} (ex7 {x['pct7']:.1f}) NSE {x['nse']:+.2f} VR {x['vr']:.2f} nRMSE {x['nrmse']:.2f} seas {x['seas_frac']:.2f} gauge {100*x['gauge_complete']:.0f}% SWOT {x['n_swot']} passes NSE {x['swot_nse']:+.2f} VR {x['swot_vr']:.2f}")
