"""An earlier version of the full-record scorer, kept only as the source of one function.

bl.py (the baselines module of the scorers) cuts the function `normal_year()` out of this file's text and
uses it for the gauge normal-year baseline. Nothing else in this file is used by the paper's pipeline, and
the file is not run. Do not edit `normal_year()`.

The rest of the file differs from the current scorers (overlap50/score_fullrecord.py) in ways that matter:
it requires 70% (5-year blocks) or 80% (other windows) of a window's days instead of 50%, and it does not
write the `nse` column. The label `SWOT-GNN` means SWOT-GAT.
"""
from swotgat.paths import WORK
import argparse, numpy as np, pandas as pd
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork"); T3 = F/"testsets/regime3_full_1980_2020"; P = F/"predictions"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--gnn", action="append", default=[])
ap.add_argument("--rule", action="append", default=[]); ap.add_argument("--chunks", default="x8085,x8691,x9297,x9803,x0409,x1014,c2")
ap.add_argument("--step", default="step_050000")
ap.add_argument("--year-rule", default="headline", choices=["headline", "lenient"],
                help="headline: qa_site_year use flag; lenient (sensitivity): keep scale-break, flat and first-fill years, drop only "
                     "duplicate-record and < 300-day years")
a = ap.parse_args(); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
S = pd.read_csv(T3/"sites.csv", dtype={"node": str, "members": str, "basin": str})
TR = pd.read_parquet(T3/"truth_clean.parquet"); TR["date"] = pd.to_datetime(TR.date)
if a.year_rule == "headline":
    TR = TR[TR.use.fillna(False).astype(bool)]
else:
    QA = pd.read_csv(T3/"qa_site_year.csv"); ok = QA[(QA.n_days >= 300) & QA.dup_with.isna()][["DAM_ID", "year"]].assign(k=1)
    TR = TR.assign(year=TR.date.dt.year).merge(ok, on=["DAM_ID", "year"], how="inner").drop(columns=["year", "k"])
need = {(s.basin, m) for _, s in S.iterrows() for m in str(s.members).split(";")}
def gnn_run(run):
    """Read the SWOT-GAT predictions of one run at the member nodes of the scored reservoirs.

    The run's files of all time chunks (--chunks; checkpoint tag from ipcfg.STEP, else --step) are
    concatenated and duplicate node-days are dropped. Return the column `pred_dv_SWOTAE` as a
    Series indexed by (basin, subbasin, date).
    """
    parts = []
    for c in a.chunks.split(","):
        f = P/run/f"{__import__('ipcfg').STEP.get(run, a.step)}_{c}.parquet"
        d = pd.read_parquet(f, columns=["basin", "subbasin", "date", "pred_dv_SWOTAE"]); d["basin"] = d.basin.astype(str); d["subbasin"] = d.subbasin.astype(str)
        d["date"] = pd.to_datetime(d.date); parts.append(d[[(b, s) in need for b, s in zip(d.basin, d.subbasin)]])
    d = pd.concat(parts).drop_duplicates(["basin", "subbasin", "date"])
    return d.set_index(["basin", "subbasin", "date"]).pred_dv_SWOTAE
MODELS = {}
for x in a.gnn:
    name, runs = x.split("=", 1)
    rs = [gnn_run(r) for r in runs.split(",")]
    G = pd.concat(rs, axis=1).mean(axis=1, skipna=False).rename("p").sort_index()
    ser = {}
    for _, s in S.iterrows():
        mem = str(s.members).split(";")
        try: g = G.loc[s.basin].loc[mem].reset_index()
        except KeyError: continue
        if g.subbasin.nunique() < len(mem): continue
        k = g.groupby("date").subbasin.nunique(); ser[int(s.DAM_ID)] = g.groupby("date").p.sum()[k == len(mem)]
    MODELS[name] = ser; print(f"{name}: {len(ser)} sites", flush=True)
for x in a.rule:
    name, tag = x.split("=", 1)
    d = pd.read_parquet(P/f"grades_{tag}"/"full_series.parquet"); d["date"] = pd.to_datetime(d.date)
    MODELS[name] = {int(k): g.set_index("date").storage_mcm for k, g in d.groupby("DAM_ID")}; print(f"{name}: {len(MODELS[name])} dams", flush=True)
WIN = [(f"{y}-{y+4}", f"{y}-01-01", f"{y+4}-12-31", "block") for y in range(1981, 2017, 5)]
WIN += [(str(y), f"{y}-01-01", f"{y}-12-31", "year") for y in range(1981, 2021)] + [("2015-2020", "2015-01-01", "2020-12-31", "frozen")]
def normal_year(obs_all, w0, w1, dates):
    """Return the gauge's normal year on the given dates, built from its record outside [w0, w1].

    obs_all is the reservoir's daily gauge storage (date index). The days from w0 to w1 are left
    out; None is returned when fewer than 3 calendar years remain. The remaining days are turned
    into anomalies from each year's median, averaged by day of year (day 366 folded into 365, gaps
    interpolated) and smoothed with a 31-day circular running mean. The result is that cycle read
    at the day of year of each date in dates (a DatetimeIndex), as a Series indexed by dates.
    """
    ref = obs_all[(obs_all.index < pd.Timestamp(w0)) | (obs_all.index > pd.Timestamp(w1))]
    if ref.index.year.nunique() < 3: return None
    an = ref - ref.groupby(ref.index.year).transform("median")
    doy = an.groupby(np.minimum(an.index.dayofyear, 365)).mean().reindex(range(1, 366)).interpolate(limit_direction="both")
    sm = pd.concat([doy.iloc[-15:], doy, doy.iloc[:15]]).rolling(31, center=True).mean().iloc[15:-15]; sm.index = range(1, 366)
    return pd.Series(sm.reindex(np.minimum(dates.dayofyear, 365)).values, index=dates)
def metrics(o, p, c):
    """Return the scores of model series p against gauge series o (on the same days) as a dict.

    Both series have their own mean removed first. The keys are
      n                     number of days
      nrmse                 RMSE / (max - min of the gauge)
      r                     Pearson correlation; 0 for a constant model or gauge
      amp                   standard deviation of the model / of the gauge (the variability ratio, VR)
      lo_hit, hi_hit        share of the gauge's lowest (highest) 10% of days that the model also
                            puts in its lowest (highest) 10%; 0.1 for a constant model
      lo_spread, hi_spread  (P50 - P10) and (P90 - P50) of the model divided by the gauge's
      clim_ratio            model RMSE / normal-year RMSE; only when c, the normal-year series on the
                            same days, is not None
    """
    o = o - o.mean(); p = p - p.mean(); rng = o.max() - o.min()
    m = dict(n=len(o), nrmse=np.sqrt(((p - o)**2).mean())/rng if rng > 0 else np.nan)
    m["r"] = float(np.corrcoef(o, p)[0, 1]) if p.std() > 0 and o.std() > 0 else 0.0
    m["amp"] = float(p.std()/o.std()) if o.std() > 0 else np.nan
    lo, hi = o <= o.quantile(.1), o >= o.quantile(.9)
    if p.std() > 0:
        m["lo_hit"] = float((p[lo] <= p.quantile(.1)).mean()); m["hi_hit"] = float((p[hi] >= p.quantile(.9)).mean())
    else: m["lo_hit"] = m["hi_hit"] = 0.1
    so = (o.quantile(.5) - o.quantile(.1), o.quantile(.9) - o.quantile(.5)); sp = (p.quantile(.5) - p.quantile(.1), p.quantile(.9) - p.quantile(.5))
    m["lo_spread"] = sp[0]/so[0] if so[0] > 0 else np.nan; m["hi_spread"] = sp[1]/so[1] if so[1] > 0 else np.nan
    if c is not None:
        cc = c - c.mean(); rc = np.sqrt(((cc - o)**2).mean()); m["clim_ratio"] = np.sqrt(((p - o)**2).mean())/rc if rc > 0 else np.nan
    return m
rows = []
for _, s in S.iterrows():
    dam = int(s.DAM_ID); oa = TR[TR.DAM_ID == dam].set_index("date").storage_mcm.sort_index()
    if oa.empty: continue
    for wn, w0, w1, kind in WIN:
        ow = oa.loc[w0:w1]; ndays = len(pd.date_range(w0, w1)); cov = 0.7 if kind == "block" else 0.8   # blocks: >= 3.5 of 5 years
        if len(ow) < cov*ndays: continue
        nyc = normal_year(oa, w0, w1, ow.index)
        base = dict(DAM_ID=dam, window=wn, kind=kind, site_tier=s.site_tier, basin=s.basin)
        if nyc is not None and nyc.notna().all():
            rows.append({**base, "model": "normal_year", **metrics(ow, nyc, None)})
        for name, ser in MODELS.items():
            p = ser.get(dam)
            if p is None: continue
            j = pd.concat([ow.rename("o"), p.rename("p")], axis=1).dropna()
            if len(j) < cov*ndays: continue
            c = nyc.reindex(j.index) if nyc is not None else None
            rows.append({**base, "model": name, **metrics(j.o, j.p, c if (c is not None and c.notna().all()) else None)})
R = pd.DataFrame(rows); R.to_csv(OUT/"fullrecord_site_window.csv", index=False)
print(f"rows {len(R):,}; sites {R.DAM_ID.nunique()}", flush=True)
hm = R[R.site_tier.isin(["High", "Medium"])]
blk = hm[hm.kind == "block"]
for panel in ("all", "balanced"):
    d = blk
    if panel == "balanced":   # sites scored by EVERY model in EVERY block
        cnt = d.groupby("DAM_ID").apply(lambda g: g.groupby("model").window.nunique().min() if g.model.nunique() == d.model.nunique() else 0)
        d = d[d.DAM_ID.isin(cnt[cnt == d.window.nunique()].index)]
    tab = d.groupby(["window", "model"]).agg(n=("DAM_ID", "nunique"), nrmse=("nrmse", "median"), r=("r", "median"), amp=("amp", "median"),
                                            lo_hit=("lo_hit", "median"), hi_hit=("hi_hit", "median"), clim_ratio=("clim_ratio", "median"))
    tab.round(3).to_csv(OUT/f"blocks_{panel}.csv"); print(f"\n== 5-year blocks, High+Medium, {panel} panel ==\n{tab.round(3).to_string()}")
