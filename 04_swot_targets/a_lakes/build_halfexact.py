"""Apply HALF's temporal-coverage rule to the version-D lake record; the result is the paper's lake record.

HALF is the lake outlier filter of Trudel et al. (HALF v1.0 release, `vendor/half_v1`). The record written by
build_vd_half.py (lakesp_vD_half_graph.parquet, called the "current" record in the printed messages) keeps every
lake for which HALF retained observations. HALF asks for one more step before its lenient-mode results are used:
a temporal-coverage check per lake. This script applies that check and changes nothing else, so that the lake
record follows the HALF framework exactly (hence "HALF-exact", `halfexact` and `hx` in file names).

Usage: python build_halfexact.py      (no arguments)

Method
  Source of every kept observation: the output of the unmodified HALF v1.0 release script, replayed on the stored
  download (release_replay.py and compare_replay.py -> replay/release_obs_vD.parquet). In the paper's run it is
  identical to the output of half_batch.py (half_obs_vD.parquet) at all 3,402 lakes.
  Coverage per lake, from the observations HALF retained (filter_flag == 1): number of observations, span in days
  from the first to the last, and the largest gap in days between consecutive observations.
  Tier per lake, first match wins:
    tier1_strict    HALF passed the lake in strict mode
    tier1_lenient   HALF passed it in lenient mode, and the retained observations span >= 365 days with no
                    gap > 120 days
    no_obs          no retained observation
    tier2           span >= 365 days and at least 5 retained observations
    below_tier2     everything else
  A lake is kept if it is tier 1. Lakes that only passed HALF's Tukey fallback are therefore never kept.
  The 365-day and 120-day limits are HALF's own: "A post-filtering temporal-coverage check (as in Trudel et al.
  (2026), min span 1 year and max gap 120 days) is recommended before using lenient results"
  (half_v1_functions.py lines 3184-3186; half_v1_run.py lines 911-913; tier 1 in the Methods of the HALF
  manuscript). HALF calls Tukey results "not recommended for phenology characterization" (half_v1_run.py
  line 927). Applied to version C, the rule reproduces the lake set of the HALF authors' own version-C series
  exactly (3,188 of 3,188 graph lakes, none extra, none missed).
  Everything else is taken unchanged from the record of build_vd_half.py: the daily median of wse_adjusted, the
  level offset between versions D and C, the level-to-storage curves and the out_of_range flag. Rows of kept
  lakes are copied as they are. The script asserts that the daily medians of the release output equal `wse_vD`
  of that record on every lake-day.

Reads (under $SWOTGAT_WORK/CONUS_storage_fork)
  eval/half_vd_review/replay/release_obs_vD.parquet
  eval/swotext/lakesp_vD_half_graph.parquet     record before the rule (build_vd_half.py)
  eval/swotext/jida_graph_lakes.parquet         version-C daily series from the HALF authors (lake_lists.py);
                                                "Jida" and `in_jida_vC` in the code refer to it
  eval/swotext/lake_targets_vd2026.parquet      lake targets built from the record before the rule
  eval/swotext/lake_qc1_days.csv and $SWOTGAT_WORK/CONUS/basin_lists/swotqc/drop_lake_vd_qc1.parquet
                                                lake-days flagged by an earlier quality check ("qc1"); only
                                                counted here, not applied
  testsets/regime1x_2023_2026/sites.csv, truth_clean_v1x.parquet
                                                gauge test set of the SWOT Period (diagnosis only)

Writes (to eval/half_vd_review/)
  lakesp_vD_halfexact_graph.parquet   the kept lakes, same columns as lakesp_vD_half_graph.parquet
                                      (in the paper's run: 3,224 lakes, 311,091 lake-days)
  halfexact_lakes_vD.csv              per lake: coverage, HALF attempt, tier, kept, in_jida_vC
  compare_summary.json, compare_lakes_removed.csv, compare_gauge.csv, powell_series.csv
                                      the record before and after the rule, compared

Notes
  The printed blocks [0] to [5] and the compare_* files are diagnostics: lake-days removed by tier, the effect on
  the store targets, on the qc1 list, on Lake Powell, and on the correlation with gauged storage. They do not
  change the record that is written.
"""
from swotgat.paths import WORK
import json, numpy as np, pandas as pd
R = f"{WORK}/CONUS_storage_fork/eval/half_vd_review"; F = f"{WORK}/CONUS_storage_fork"
S = f"{F}/eval/swotext"; C = f"{WORK}/CONUS"
SPAN, GAP = 365, 120
A = pd.read_parquet(f"{R}/replay/release_obs_vD.parquet", columns=["lake_id", "datetime", "filter_flag", "filter_attempt", "wse_adjusted"])
A["lake_id"] = A.lake_id.astype(str); K = A[A.filter_flag == 1].copy(); K["date"] = K.datetime.dt.normalize()
# ---- per-lake coverage of retained observations and the tier
def cov(g):
    """Return the temporal coverage of one lake's retained observations.

    g holds the times (datetime) and days (date) of the observations HALF retained at the lake. The
    result is a Series with the number of observations (n_obs) and of days (n_days), the first and
    last time, the span in days from the first to the last (span_d) and the largest gap in days
    between consecutive observations (max_gap_d, NaN if there is only one observation).
    """
    t = np.sort(g.datetime.values); gaps = np.diff(t) / np.timedelta64(1, "D")
    return pd.Series(dict(n_obs=len(t), n_days=g.date.nunique(), first=t[0], last=t[-1], span_d=(t[-1] - t[0]) / np.timedelta64(1, "D"),
                          max_gap_d=gaps.max() if len(gaps) else np.nan))
L = K.groupby("lake_id")[["datetime", "date"]].apply(cov).reindex(sorted(A.lake_id.unique()))
L["attempt"] = A.groupby("lake_id").filter_attempt.first()
L["tier"] = np.select([L.attempt == "1_strict", (L.attempt == "2_lenient") & (L.span_d >= SPAN) & (L.max_gap_d <= GAP),
                       L.n_obs.fillna(0) == 0, (L.span_d >= SPAN) & (L.n_obs >= 5)],
                      ["tier1_strict", "tier1_lenient", "no_obs", "tier2"], "below_tier2")
L["kept"] = L.tier.str.startswith("tier1")
J = pd.read_parquet(f"{S}/jida_graph_lakes.parquet", columns=["lake_id", "wse"]); L["in_jida_vC"] = L.index.isin(set(J[J.wse.notna()].lake_id.astype(str)))
L.to_csv(f"{R}/halfexact_lakes_vD.csv")
# ---- check: daily medians of the release output equal wse_vD of the record before the rule
Dd = K.groupby(["lake_id", "date"]).wse_adjusted.median().rename("wse_rel").reset_index()
X = pd.read_parquet(f"{S}/lakesp_vD_half_graph.parquet"); X["lake_id"] = X.lake_id.astype(str); X["date"] = pd.to_datetime(X.date)
m = X.merge(Dd, on=["lake_id", "date"], how="outer", indicator=True)
assert (m._merge == "both").all(), m._merge.value_counts()
assert np.allclose(m.wse_vD, m.wse_rel, atol=1e-9, rtol=0), "daily wse differs from the release output"
print(f"[0] release-script daily medians == current record wse_vD on all {len(m):,} lake-days", flush=True)
# how far the daily median used here is from the daily mean of HALF's validation utility (convert_to_daily_series)
n_day = K.groupby(["lake_id", "date"]).wse_adjusted.agg(["size", "median", "mean"])
multi = n_day[n_day["size"] >= 3]; dmm = (multi["median"] - multi["mean"]).abs()
print(f"[0b] lake-days with >= 3 kept passes (median != mean possible): {len(multi):,} of {len(n_day):,}; |median - mean| median {dmm.median():.4f} m, "
      f"p95 {dmm.quantile(.95):.3f} m, > 0.10 m at {int((dmm > 0.1).sum())}", flush=True)
# ---- the HALF-exact record: rows of the kept (tier 1) lakes, unchanged
keep = set(L.index[L.kept]); H = X[X.lake_id.isin(keep)].copy()
H.drop(columns=[]).to_parquet(f"{R}/lakesp_vD_halfexact_graph.parquet", index=False)
assert list(H.columns) == list(X.columns)
rm = X[~X.lake_id.isin(keep)]
print(f"[1] current: {len(X):,} lake-days, {X.lake_id.nunique()} lakes | HALF-exact: {len(H):,} lake-days, {H.lake_id.nunique()} lakes | "
      f"removed {len(rm):,} lake-days ({len(rm)/len(X):.1%}) at {rm.lake_id.nunique()} lakes; added 0", flush=True)
Lx = L[L.index.isin(set(X.lake_id))]
print("    lakes in the current record by tier:", Lx.tier.value_counts().to_dict(), flush=True)
print("    removed lake-days by tier:", rm.merge(L[["tier"]], left_on="lake_id", right_index=True).tier.value_counts().to_dict(), flush=True)
print("    all-lake tiers (3,402 with in-window obs):", L.tier.value_counts().to_dict(), "| Jida vC lakes not kept on vD:",
      int((L.in_jida_vC & ~L.kept).sum()), "of", int(L.in_jida_vC.sum()), flush=True)
t12 = set(L.index[L.tier.isin(["tier1_strict", "tier1_lenient", "tier2"])]); X12 = X[X.lake_id.isin(t12)]
print(f"    alternative tier1+tier2 (manuscript's wider set): {len(X12):,} lake-days, {X12.lake_id.nunique()} lakes", flush=True)
# ---- training-store view (store window 2020-01-01..2026-09-30, rows with a target)
T = pd.read_parquet(f"{S}/lake_targets_vd2026.parquet"); T["subbasin"] = T.subbasin.astype(str); T = T[pd.to_datetime(T.date) <= "2026-09-30"]
Tr = T[~T.subbasin.isin(keep)]
print(f"[2] lake_targets_vd2026 (to 2026-09-30): {len(T):,} lake-days at {T.subbasin.nunique()} lakes; from lakes HALF-exact drops: {len(Tr):,} "
      f"at {Tr.subbasin.nunique()}", flush=True)
# ---- diagnostics: the earlier qc1 drop list, out_of_range days, level outliers
Q =pd.read_parquet(f"{C}/basin_lists/swotqc/drop_lake_vd_qc1.parquet"); Q["subbasin"] = Q.subbasin.astype(str)
qd = pd.read_csv(f"{S}/lake_qc1_days.csv", dtype={"lake_id": str}); qd["in_removed"] = ~qd.lake_id.isin(keep)
print(f"[3] qc1 drop list: {len(Q)} lake-days at {Q.subbasin.nunique()} lakes; inside lakes HALF-exact removes: {int(qd.in_removed.sum())} days at "
      f"{qd[qd.in_removed].lake_id.nunique()} lakes ({qd[qd.in_removed].rule.value_counts().to_dict()}); left in kept lakes: {int((~qd.in_removed).sum())} "
      f"at {qd[~qd.in_removed].lake_id.nunique()}", flush=True)
def lev_out(Z):
    """Flag the level outliers of the lake record Z.

    A level outlier is a day whose level (wse) is more than max(3 m, 8 x the lake's median absolute
    deviation) from the lake median. Returns a boolean Series aligned with the rows of Z.
    """
    md = Z.groupby("lake_id").wse.transform("median"); mad = Z.groupby("lake_id").wse.transform(lambda s: (s - s.median()).abs().median())
    return (Z.wse - md).abs() > np.maximum(3, 8*mad)
X["lev_out"] = lev_out(X); H["lev_out"] = X.loc[H.index, "lev_out"]
print(f"    out_of_range days: current {int(X.out_of_range.sum()):,}, HALF-exact {int(H.out_of_range.sum()):,} | level-outlier days "
      f"(|wse - lake median| > max(3 m, 8 MAD)): current {int(X.lev_out.sum())} at {X[X.lev_out].lake_id.nunique()} lakes, HALF-exact "
      f"{int(H.lev_out.sum())} at {H[H.lev_out].lake_id.nunique()}", flush=True)
# ---- Lake Powell
P = "7720013333"; pw = X[X.lake_id == P][["date", "wse", "wse_vD", "n_obs", "dv_best", "dv_source", "out_of_range"]]
pw.assign(in_halfexact=P in keep).to_csv(f"{R}/powell_series.csv", index=False)
print(f"[4] Lake Powell {P}: {L.loc[P, ['attempt', 'n_obs', 'span_d', 'max_gap_d', 'tier']].to_dict()}; current {len(pw)} days "
      f"(dv_best range {pw.dv_best.min():.0f}..{pw.dv_best.max():.0f} MCM), HALF-exact {int((H.lake_id == P).sum())} days", flush=True)
# ---- gauges (diagnosis only): High- and Medium-confidence sites of the SWOT Period gauge test set
Sg = pd.read_csv(f"{F}/testsets/regime1x_2023_2026/sites.csv", dtype={"node": str, "members": str}); Sg = Sg[Sg.site_tier.isin(["High", "Medium"])]
Tg = pd.read_parquet(f"{F}/testsets/regime1x_2023_2026/truth_clean_v1x.parquet"); Tg["date"] = pd.to_datetime(Tg.date)
def score(Z):
    """Correlate the SWOT storage change of the lake record Z with gauged storage, per gauged site.

    A site's member lakes are summed on the days when all of them have dv_best. Returns a dict that
    maps the site's DAM_ID to (r, n): the correlation with the gauge storage and the number of
    shared days. r is NaN when there are fewer than 8 shared days, and the entry is (NaN, 0) when a
    member lake is missing from Z.
    """
    Zi = Z.set_index(["lake_id", "date"]).dv_best; have = set(Z.lake_id); out = {}
    for _, s in Sg.iterrows():
        mem = str(s.members).split(";")
        if not all(x in have for x in mem): out[s.DAM_ID] = (np.nan, 0); continue
        v = pd.concat([Zi.loc[x] for x in mem], axis=1).dropna().sum(axis=1)
        j = pd.concat([v.rename("s"), Tg[Tg.DAM_ID == s.DAM_ID].set_index("date").storage_mcm.rename("g")], axis=1).dropna()
        out[s.DAM_ID] = (j.s.corr(j.g) if len(j) >= 8 else np.nan, len(j))
    return out
a, b = score(X), score(H)
G = pd.DataFrame([dict(DAM_ID=k, name=Sg.set_index("DAM_ID").DAM_NAME.get(k), r_current=a[k][0], n_current=a[k][1], r_halfexact=b[k][0], n_halfexact=b[k][1])
                  for k in a])
G.to_csv(f"{R}/compare_gauge.csv", index=False)
sc, sh = G.r_current.notna(), G.r_halfexact.notna(); lost = G[sc & ~sh]
print(f"[5] gauged H/M sites scored (n >= 8): current {int(sc.sum())}, HALF-exact {int(sh.sum())}; median r current {G.r_current.median():.3f}, "
      f"HALF-exact {G.r_halfexact.median():.3f}; on the sites both score: identical r at {int(np.isclose(G[sc & sh].r_current, G[sc & sh].r_halfexact).sum())} "
      f"of {int((sc & sh).sum())}", flush=True)
print(f"    sites HALF-exact no longer scores: {len(lost)}; their current r median {lost.r_current.median():.3f}; r < 0.5 at {int((lost.r_current < 0.5).sum())}", flush=True)
print(lost.sort_values("r_current").round(3).to_string(index=False), flush=True)
rmL = L[L.index.isin(set(rm.lake_id))].join(rm.groupby("lake_id").size().rename("lake_days_removed"))
rmL.to_csv(f"{R}/compare_lakes_removed.csv")
json.dump(dict(current=dict(lake_days=len(X), lakes=int(X.lake_id.nunique())), halfexact=dict(lake_days=len(H), lakes=int(H.lake_id.nunique())),
               removed=dict(lake_days=len(rm), lakes=int(rm.lake_id.nunique())), added=dict(lake_days=0, lakes=0),
               tiers_all=L.tier.value_counts().to_dict(), tier12=dict(lake_days=len(X12), lakes=int(X12.lake_id.nunique())),
               store_lake_days_from_removed=len(Tr), qc1_days_in_removed=int(qd.in_removed.sum()), qc1_days_left=int((~qd.in_removed).sum()),
               gauge=dict(scored_current=int(sc.sum()), scored_halfexact=int(sh.sum()), median_r_current=float(G.r_current.median()),
                          median_r_halfexact=float(G.r_halfexact.median()), lost_sites=int(len(lost)))),
          open(f"{R}/compare_summary.json", "w"), indent=1, default=str)
