"""Score SWOT-GAT and the rule models against reservoir gauges in the SWOT Period.

Each reservoir is scored over its own SWOT span, with the 50%-of-days overlap rule.

Usage (after `source 07_evaluation/c_scoring/env.sh`):
    [BASE=1] bin/run 07_evaluation/c_scoring/overlap50/score_x26span.py
        [--check26 FILE:LABEL] [--check26r FILE:LABEL]
  IP_SERIES  the series file, read through ipx.py and ipcfg.py. Build it first with
             ip_series.py --p1x.
  BASE=1     adds the two baselines of bl.py (`SWOT-cycle`, `normal_year`), scored on the same days
             as SWOT-GAT. Their metadata columns are copied from the reservoir's SWOT-GAT row.
  --check26, --check26r   optional reproduction checks against tables of two earlier scorers that
             are not part of this repository (see Notes).

Reads (paths relative to $SWOTGAT_WORK/CONUS_storage_fork)
  the series file   daily gauge storage (the 2023-2026 gauge test set) and model storage on
      2023-01-01 to 2026-09-30
  eval/score2026/spans_2023_2026.csv   each reservoir's SWOT span (columns vD_start, vD_end,
      vD_n_pass), written by 07_evaluation/a_gauge_truth/span_vd.py
  eval/grl_figs/sites_v5_g.csv   a site table made for an earlier figure set. Only its rows with
      PERIOD "2023-25" are used, and only for the list of reservoir-model pairs and the metadata
      columns (name, state, lat, lon, cap, use, sup, tier, supervised_reservoir). No score is
      read from it.

Writes, next to this script
  sites_2325span_g.csv   one row per reservoir and model: nse, nrmse, r, amp (= VR), n_days,
      n_2026, first_day, last_day, flat, span_start, span_end, n_pass, plus the metadata.
      PERIOD stays "2023-25" and the SWOT-GAT label stays `v10pm` (the paper's model: mean of two
      seeds), so that the figure scripts read the table unchanged. The window is the SWOT span,
      not 2023-2025.
  num_x26span.json       counts and medians for the figure captions

Method
  Window. A reservoir's SWOT span runs from the first to the last day on which it has a SWOT
  storage target (LakeSP version D after the HALF filter) between 2023-01-01 and 2026-09-30. The
  span is clipped at LAST, the last day that every rule model reaches in the series file. In the
  paper's run LAST is 2026-06-12, where the GRADES-hydroDL inflow that drives the rule models ends.
  A reservoir without a span or without a SWOT-GAT prediction is skipped.
  Same days for every model. The scored days are the gauge days in the window on which SWOT-GAT
  and every rule model that has the reservoir have a value.
  Overlap rule. A model is scored if it shares with the gauge at least 50% of the days of the
  clipped span.
  Metrics. On the shared days both series have their own mean removed. Then
    nse = 1 - sum((model - gauge)^2) / sum(gauge^2),  nrmse = RMSE / (max - min of the gauge),
    r = Pearson correlation,  amp = VR = sd(model) / sd(gauge).
  A constant prediction gets r = amp = 0 and flat = True. A reservoir with a constant gauge is
  skipped.
  Rule models: GDROM, ISTARF and MODROM-cal. MODROM-tr is dropped because it has no run in 2026.
  SWOT-GAT predictions are converted to float32 before scoring, so that the scores reproduce an
  earlier scorer exactly.

Notes
  --check26 FILE:LABEL compares the SWOT-GAT rows with the rows of FILE that have WINDOW "span_vD",
  DAYS "all" and MODEL LABEL, reservoir by reservoir, and, if summary26.csv lies beside FILE, the
  High/Medium median. --check26r FILE:LABEL does the same for every model against the rows with
  WINDOW "span_vDg" and against summary26r.csv. A difference above 1e-9 stops the script.
"""
from swotgat.paths import WORK
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, ipx, bl
F = Path(f"{WORK}/CONUS_storage_fork"); OUT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--check26", default=""); ap.add_argument("--check26r", default=""); a = ap.parse_args()
V = pd.read_csv(F/"eval/grl_figs/sites_v5_g.csv"); V = V[(V.PERIOD == "2023-25") & (V.MODEL != "MODROM-tr")]
SP = pd.read_csv(F/"eval/score2026/spans_2023_2026.csv").set_index("DAM_ID")
META = ["MODEL", "PERIOD", "DAM_ID", "name", "state", "lat", "lon", "cap", "use", "sup", "tier", "supervised_reservoir"]
MIN, T = 180, ipx.T26
MIN_DAM = [180]   # minimum shared days; set per reservoir to 50% of its clipped span (overlap rule)
def score(o, q):
    """Return the scores of model series q against gauge series o on their shared days, or None.

    Both series have their own mean removed on the shared days. The result is a dict with nse,
    nrmse, r, amp (= VR), n_days, n_2026 (shared days in 2026), first_day, last_day and flat (True
    for a constant prediction, which gets r = amp = 0). None is returned when the series share
    fewer days than the current minimum (MIN_DAM[0], set per reservoir to 50% of its clipped span)
    or when the gauge is constant on the shared days.
    """
    j = pd.concat([o.rename("o"), q.rename("q")], axis=1).dropna()
    if len(j) < MIN_DAM[0] or j.o.std() == 0: return None
    oc, qc = j.o - j.o.mean(), j.q - j.q.mean(); rng = oc.max() - oc.min()
    if rng <= 0: return None
    flat = qc.std() < 1e-9*max(oc.std(), 1e-9)
    return dict(nse=float(1 - ((qc - oc)**2).sum()/(oc**2).sum()), nrmse=float(np.sqrt(((qc - oc)**2).mean())/rng), r=0.0 if flat else float(np.corrcoef(oc, qc)[0, 1]),
                amp=0.0 if flat else float(qc.std()/oc.std()), n_days=len(j), n_2026=int((j.index.year == 2026).sum()),
                first_day=j.index.min().date(), last_day=j.index.max().date(), flat=bool(flat))
RUL = ["GDROM", "ISTARF", "MODROM-cal"]
LAST = min(T[np.isfinite(np.vstack([ipx._z()["z"][f"{m}|P1x|pred"] for m in [r_]])).any(0)].max() for r_ in RUL)   # last day every rule model reaches (2026-06-12 in the paper's run)
def preds(dam):
    """Return the series of SWOT-GAT and the rule models at one dam, as {model: (gauge, prediction)}.

    gauge is the gauge array on the dates T; prediction is the model's storage as a Series by date
    with the missing days dropped (SWOT-GAT's is converted to float32). A model is left out when
    the dam is not in the series file or the model has no finite value there.
    """
    out = {}
    for m in ["v10pm"] + RUL:
        o, q = ipx.get_x(m, dam)
        if o is not None and np.isfinite(q).any():
            q_ = pd.Series(q, index=T).dropna(); out[m] = (o, q_.astype(np.float32) if m == "v10pm" else q_)   # SWOT-GAT in float32, to reproduce an earlier scorer exactly
    return out
rows, nospan = [], set()
for dam, vv in V.groupby("DAM_ID", sort=False):
    dam = int(dam)
    if dam not in SP.index or pd.isna(SP.at[dam, "vD_start"]): nospan.add(dam); continue
    Q = preds(dam)
    if "v10pm" not in Q: continue
    a0, b0 = pd.Timestamp(SP.at[dam, "vD_start"]), min(pd.Timestamp(SP.at[dam, "vD_end"]), LAST)
    MIN_DAM[0] = 0.5*((b0 - a0).days + 1)
    o = pd.Series(Q["v10pm"][0], index=T).loc[a0:b0]; common = o.index
    for m, (_, q_) in Q.items(): common = common.intersection(q_.loc[a0:b0].index)      # the same days for every model
    o = o.loc[common]
    for _, v in vv.iterrows():
        if v.MODEL not in Q: continue
        s = score(o.dropna(), Q[v.MODEL][1].loc[a0:b0])
        if s is None: continue
        rows.append({**{c: v[c] for c in META}, **s, "span_start": a0.date(), "span_end": b0.date(), "n_pass": int(SP.at[dam, "vD_n_pass"])})
    if bl.BASE:                                      # BASE=1: both baselines, scored on the same (common) days as SWOT-GAT
        v0 = vv[vv.MODEL == "v10pm"].iloc[0] if (vv.MODEL == "v10pm").any() else vv.iloc[0]
        for bm, q_ in bl.x26(dam, a0, b0, T).items():
            s = score(o.dropna(), q_.loc[a0:b0])
            if s is None: continue
            rows.append({**{c: v0[c] for c in META}, "MODEL": bm, **s, "span_start": a0.date(), "span_end": b0.date(), "n_pass": int(SP.at[dam, "vD_n_pass"])})
R = pd.DataFrame(rows); R.to_csv(OUT/"sites_2325span_g.csv", index=False)

M4 = ["v10pm", "MODROM-cal", "GDROM", "ISTARF"]; hm = R[R.tier.isin(["High", "Medium"])]; vh = V[V.tier.isin(["High", "Medium"])]
ids = set.intersection(*[set(hm[hm.MODEL == m].DAM_ID) for m in M4]); full = set.intersection(*[set(vh[vh.MODEL == m].DAM_ID) for m in M4])
gone = full - ids; never = sorted(gone & nospan); short = sorted(gone - nospan)
med = hm[hm.DAM_ID.isin(ids)].groupby("MODEL")[["nrmse", "r", "amp", "n_days"]].median()
print(f"vD span rows {len(R)} | 4-model High/Medium common {len(ids)} of {len(full)} (no vD span {len(never)}, too few scored days {len(short)} {short}) | "
      f"span start {hm.span_start.min()} to {hm.span_start.max()} | end {hm.span_end.min()} to {hm.span_end.max()} | "
      f"last scored day: SWOT-GNN {hm[hm.MODEL == 'v10pm'].last_day.max()}, rules {hm[hm.MODEL != 'v10pm'].last_day.max()}")
print(med.round(3).to_string())
num = dict(label="SWOT span, Jul 2023–Jun 2026", spans="eval/score2026/spans_2023_2026.csv (vD), clipped at 2026-06-12", n_common=len(ids), n_common_full=len(full),
           n_never=len(never), n_short=len(short), short=short, rules_end=str(LAST.date()),
           span_start=[str(hm.span_start.min()), str(hm.span_start.max())], span_end=[str(hm.span_end.min()), str(hm.span_end.max())],
           median={m: {k: float(med.at[m, k]) for k in med.columns} for m in med.index},
           n_vD_hm_swotgnn=int((hm.MODEL == "v10pm").sum()))

if a.check26:                                       # optional check of the SWOT-GAT rows against an earlier scorer's table (same predictions, same definitions)
    f26, lab = a.check26.rsplit(":", 1); f26 = Path(f26) if Path(f26).is_absolute() else F/f26
    S26 = pd.read_csv(f26); x = S26[(S26.WINDOW == "span_vD") & (S26.DAYS == "all")]
    b = x[x.MODEL == lab].set_index("DAM_ID"); m_ = R[R.MODEL == "v10pm"].set_index("DAM_ID")
    hm26 = b[b.tier.isin(["High", "Medium"])]
    j = m_[["nrmse", "r", "amp", "n_days"]].join(b[["nrmse", "r", "amp", "n_days"]], rsuffix="_26", how="inner")
    d = {k: float((j[k] - j[f"{k}_26"]).abs().max()) for k in ("nrmse", "r", "amp")}
    only26 = sorted(set(hm26.index) - set(m_.index)); onlyhere = sorted(set(m_[m_.tier.isin(["High", "Medium"])].index) - set(hm26.index))
    print(f"CHECK26 {lab}: {len(j)} reservoirs in both | max |d| {d} | n_days differ at {int((j.n_days != j.n_days_26).sum())} | "
          f"High/Medium only in score26 {only26} | only here {onlyhere}")
    ok = max(d.values()) <= 1e-9 and (j.n_days == j.n_days_26).all() and not only26 and not onlyhere
    sm = f26.parent/"summary26.csv"
    if sm.exists():
        U = pd.read_csv(sm); u = U[(U.WINDOW == "span_vD") & (U.DAYS == "all") & (U.GROUP == "High/Medium") & (U.SET == "gnn_common") & (U.MODEL == lab)].iloc[0]
        h = x[x.tier.isin(["High", "Medium"])]; gc = set.intersection(*[set(h[h.MODEL == mm].DAM_ID) for mm in h.MODEL.unique()])
        mine = m_.loc[sorted(gc & set(m_.index))]
        cmp = {k: (float(mine[k].median()), float(u[k])) for k in ("nrmse", "r", "amp")}
        print(f"CHECK26 summary26 High/Medium gnn_common: n here {len(mine)} vs {int(u.n)} | (here, summary26) {cmp}")
        ok = ok and len(mine) == int(u.n) and all(abs(p - q) <= 1e-9 for p, q in cmp.values())
        num["check26"] = dict(file=str(f26), label=lab, n_sites=len(j), max_abs_diff=d, gnn_common_n=len(mine), gnn_common=cmp)
    print("CHECK26", "PASSED" if ok else "FAILED", flush=True)
    if not ok: raise SystemExit("score_x26span: the SWOT-GNN rows do not reproduce score26")
if a.check26r:                                      # optional check of every model against an earlier scorer's table (same predictions, days and definitions)
    f26, lab = a.check26r.rsplit(":", 1); S26 = pd.read_csv(f26); x = S26[(S26.WINDOW == "span_vDg") & (S26.DAYS == "all")]
    ok = True; rep_ = {}
    for m in ["v10pm"] + RUL:
        b = x[x.MODEL == (lab if m == "v10pm" else m)].set_index("DAM_ID"); m_ = R[R.MODEL == m].set_index("DAM_ID")
        j = m_[["nrmse", "r", "amp", "n_days"]].join(b[["nrmse", "r", "amp", "n_days"]], rsuffix="_26", how="inner")
        d = {k: float((j[k] - j[f"{k}_26"]).abs().max()) for k in ("nrmse", "r", "amp")}
        hb, hm_ = set(b[b.tier.isin(["High", "Medium"])].index), set(m_[m_.tier.isin(["High", "Medium"])].index)
        good = max(d.values()) <= 1e-9 and (j.n_days == j.n_days_26).all() and hb == hm_
        print(f"CHECK26R {m}: {len(j)} in both | max |d| {d} | n_days differ {int((j.n_days != j.n_days_26).sum())} | H/M only in score26r {sorted(hb - hm_)[:8]} | only here {sorted(hm_ - hb)[:8]} -> {'ok' if good else 'MISMATCH'}")
        ok = ok and good; rep_[m] = dict(n=len(j), max_abs_diff=d)
    U = pd.read_csv(Path(f26).parent/"summary26r.csv")
    u = U[(U.WINDOW == "span_vDg") & (U.DAYS == "all") & (U.GROUP == "High/Medium") & (U.SET == "all_common")].set_index("MODEL")
    for m in ["v10pm"] + RUL:
        mine = hm[hm.DAM_ID.isin(ids) & (hm.MODEL == m)]; uu = u.loc[lab if m == "v10pm" else m]
        same = len(mine) == int(uu.n) and all(abs(float(mine[k].median()) - float(uu[k])) <= 1e-9 for k in ("nrmse", "r", "amp"))
        print(f"CHECK26R summary {m}: n {len(mine)} vs {int(uu.n)} | nrmse {mine.nrmse.median():.6f} vs {uu.nrmse:.6f} -> {'ok' if same else 'MISMATCH'}"); ok = ok and same
    num["check26r"] = dict(file=f26, label=lab, models=rep_)
    print("CHECK26R", "PASSED" if ok else "FAILED", flush=True)
    if not ok: raise SystemExit("score_x26span: does not reproduce score26r span_vDg")
json.dump(num, open(OUT/"num_x26span.json", "w"), indent=1, default=str)
