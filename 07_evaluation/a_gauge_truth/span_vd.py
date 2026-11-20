"""Find each gauged reservoir's SWOT observation span: the first and last day with a SWOT target.

The span of a reservoir is the period in which a model could be trained on SWOT storage change
at that reservoir. It is computed for the sites of the 2023-2026 agency gauge test set
(testsets/regime1x_2023_2026) and for two versions of the SWOT lake targets:
  vC  the SWOT LakeSP version C targets, 2023-01-01 to 2025-12-31
      (crosswalk/pld_wse_graph.parquet, node list dv_SWOTAE_v8_fb.csv). These targets and this
      node list belong to an earlier model configuration (run label v10pm), not to the paper's
      model.
  vD  the SWOT LakeSP version D targets after the HALF outlier filter of Trudel et al.,
      2023-01-01 to 2026-09-30 (eval/swotext/lake_targets_vd2026.parquet, node list
      dv_SWOTAE_vd2026_fb_g.csv; run label v12d).

Reads ($SWOTGAT_WORK/CONUS_storage_fork unless noted)
  testsets/regime1x_2023_2026/sites.csv        sites with their member nodes
  crosswalk/pld_wse_graph.parquet              vC lake targets (lake_id, date, dv_SWOTAE,
                                               dv_staticPLD)
  eval/swotext/lake_targets_vd2026.parquet     vD lake targets (subbasin, date, same columns)
  $SWOTGAT_WORK/CONUS/basin_lists/fallback/dv_SWOTAE_v8_fb.csv, dv_SWOTAE_vd2026_fb_g.csv
                                               lakes that have a storage target in each version
  eval/grl_figs/proto/sites_2325span_g.csv     the span table used by the scoring, for a check

Writes
  eval/score2026/spans_2023_2026.csv   per DAM_ID: name, tier, basin, vC_start, vC_end,
                                       vC_n_pass, vD_start, vD_end, vD_n_pass, vD_n_pass_le2025

Usage
  python span_vd.py     (no arguments)

Method
  * A reservoir can consist of several graph nodes (members). All members must be on the
    version's node list, otherwise the reservoir has no span.
  * The value of a member on a day is dv_SWOTAE (storage change from the SWOT area-elevation
    curve) or, where that is missing, dv_staticPLD (level change times the fixed lake area of
    the SWOT Prior Lake Database). This is what the training data set does: it uses the
    fallback only where the target is missing.
  * A day counts when every member has a value. The span runs from the first to the last such
    day; n_pass is the number of such days.
  * No observation masks are applied, because the two runs had none for lake targets.
  * The vC spans are computed in two forms. In the "lake" form a lake with any dv_SWOTAE value
    uses dv_SWOTAE only; this is the form written to the output. In the "day" form the fallback
    fills each missing day, as in training. Both are compared with span_start, span_end and
    n_pass of sites_2325span_g.csv, and the number of identical spans is printed. The vD spans
    use the "day" form.
"""
from swotgat.paths import WORK
import numpy as np, pandas as pd
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork"); C = Path(f"{WORK}/CONUS")
OUT = F/"eval/score2026"
S = pd.read_csv(F/"testsets/regime1x_2023_2026/sites.csv", dtype={"members": str, "basin": str, "node": str})

def passes(W, fb, t0, t1, by_lake_ae):
    """Return the span of each site: a table indexed by DAM_ID with start, end and n_pass.

    W is a lake target table (lake, date, dv_SWOTAE, dv_staticPLD), fb the set of nodes that have
    a target, and t0..t1 the window. by_lake_ae selects the "lake" form of the fallback, otherwise
    the "day" form is used (see the module docstring). Sites with a member outside fb, or with no
    day on which every member has a value, are left out.
    """
    W = W[(W.date >= t0) & (W.date <= t1) & W.lake.isin(fb)].copy()
    if by_lake_ae:      # "lake" form: a lake with any dv_SWOTAE value uses dv_SWOTAE only
        has = W.groupby("lake").dv_SWOTAE.transform(lambda s: s.notna().any())
        W["v"] = np.where(has, W.dv_SWOTAE, W.dv_staticPLD)
    else:               # "day" form, as in training: on each day the fallback fills a missing target
        W["v"] = W.dv_SWOTAE.where(W.dv_SWOTAE.notna(), W.dv_staticPLD)
    W = W.dropna(subset=["v"])
    days = W.groupby("lake").date.apply(lambda s: set(s))
    rows = []
    for s in S.itertuples():
        mem = [m for m in str(s.members).split(";") if m and m != "nan"]
        if not mem or any(m not in fb for m in mem): continue
        d = None
        for m in mem:
            k = days.get(m, set()); d = k if d is None else (d & k)
        if d: rows.append(dict(DAM_ID=int(s.DAM_ID), start=min(d).date(), end=max(d).date(), n_pass=len(d)))
    return pd.DataFrame(rows).set_index("DAM_ID")

T0 = pd.Timestamp("2023-01-01")
wc = pd.read_parquet(F/"crosswalk/pld_wse_graph.parquet", columns=["lake_id", "date", "dv_SWOTAE", "dv_staticPLD"]).rename(columns={"lake_id": "lake"})
wd = pd.read_parquet(F/"eval/swotext/lake_targets_vd2026.parquet", columns=["subbasin", "date", "dv_SWOTAE", "dv_staticPLD"]).rename(columns={"subbasin": "lake"})
for w in (wc, wd): w["lake"] = w.lake.astype(str); w["date"] = pd.to_datetime(w.date).dt.normalize()
fb8 = set(pd.read_csv(C/"basin_lists/fallback/dv_SWOTAE_v8_fb.csv").subbasin.astype(str))
fbd = set(pd.read_csv(C/"basin_lists/fallback/dv_SWOTAE_vd2026_fb_g.csv").subbasin.astype(str))
vc_lake = passes(wc, fb8, T0, pd.Timestamp("2025-12-31"), True)
vc_day = passes(wc, fb8, T0, pd.Timestamp("2025-12-31"), False)
vd = passes(wd, fbd, T0, pd.Timestamp("2026-09-30"), False)
vd25 = passes(wd, fbd, T0, pd.Timestamp("2025-12-31"), False)

# check the vC spans ("lake" form, then "day" form) against the span table used by the scoring
SP = pd.read_csv(F/"eval/grl_figs/proto/sites_2325span_g.csv").drop_duplicates("DAM_ID").set_index("DAM_ID")
j = SP[["span_start", "span_end", "n_pass"]].join(vc_lake.add_prefix("x_"), how="left")
ok = (j.span_start == j.x_start.astype(str)) & (j.span_end == j.x_end.astype(str))
print(f"vC spans (atlas form) vs sites_2325span_g: {int(ok.sum())}/{len(j)} identical start+end; "
      f"n_pass equal at {int((j.n_pass == j.x_n_pass).sum())}")
j2 = SP[["span_start", "span_end"]].join(vc_day, how="left")
print(f"vC spans (training per-day form) vs sites_2325span_g: {int(((j2.span_start == j2.start.astype(str)) & (j2.span_end == j2.end.astype(str))).sum())}/{len(j2)}")
bad = j[~ok]
if len(bad): print(bad.head(10).to_string())

R = S[["DAM_ID", "DAM_NAME", "site_tier", "supervised_2023_25", "basin"]].set_index("DAM_ID")
R = R.join(vc_lake.add_prefix("vC_")).join(vd.add_prefix("vD_")).join(vd25[["n_pass"]].rename(columns={"n_pass": "vD_n_pass_le2025"}))
R.to_csv(OUT/"spans_2023_2026.csv")
h = R[R.site_tier.isin(["High", "Medium"])]
print(f"reservoirs with a vC span {R.vC_start.notna().sum()} (High/Medium {h.vC_start.notna().sum()}); "
      f"with a vD span {R.vD_start.notna().sum()} (High/Medium {h.vD_start.notna().sum()})")
for c in ["vC_start", "vC_end", "vD_start", "vD_end"]:
    v = pd.to_datetime(h[c].dropna()); print(f"  {c}: median {v.median().date()}  range {v.min().date()} .. {v.max().date()}")
print("  median passes: vC", h.vC_n_pass.median(), "| vD", h.vD_n_pass.median(), "| vD to 2025", h.vD_n_pass_le2025.median())
print("  vD span but no vC span:", int((h.vD_start.notna() & h.vC_start.isna()).sum()), "| vC but no vD:", int((h.vC_start.notna() & h.vD_start.isna()).sum()))
