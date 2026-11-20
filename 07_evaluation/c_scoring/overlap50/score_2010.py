"""Score SWOT-GAT and the rule models against reservoir gauges over 2010-2020 (50%-of-days overlap rule).

Usage (after `source 07_evaluation/c_scoring/env.sh`; no command-line arguments):
    [BASE=1] bin/run 07_evaluation/c_scoring/overlap50/score_2010.py
  IP_RUNS   the prediction runs to average (the two seeds); IP_STEPS their checkpoint tags;
  IP_HIND   the window tags of the prediction files that cover 2010-2020 (the paper: h10).
            All three are read through ipcfg.py.
  BASE=1    adds the two baselines of bl.py as models: `SWOT-cycle` and `normal_year` (the gauge's
            day-of-year cycle from its record outside 2010-2020).

Reads (paths relative to $SWOTGAT_WORK/CONUS_storage_fork)
  testsets/regime3_full_1980_2020/sites.csv, truth_clean.parquet   the 1980-2020 gauge test set;
      only gauge days whose `use` flag is set are scored
  predictions/<run>/<step>_<tag>.parquet   SWOT-GAT predictions (column pred_dv_SWOTAE)
  predictions/grades_<tag>/full_series.parquet   rule-model runs started from gauge storage in 1980:
      GDROM gdrom3c_g_y1980, ISTARF istarf3c_g_y1980, MODROM-cal modromcal3c_g_y1980 and
      MODROM-tr modromrf3c_g_y1980_pubnorm
  eval/grl_figs/sites_v5_g.csv   reservoir metadata only (name, state, lat, lon, capacity, use)

Writes sites_2010_g.csv next to this script: one row per reservoir and model, PERIOD = "2010-20".
The MODEL label `v10pm` means SWOT-GAT (the paper's model: mean of two seeds); the label is kept
because later scripts select on it. The script also prints the medians over the High- and
Medium-tier reservoirs that all four of SWOT-GAT, MODROM-cal, GDROM and ISTARF score.

Method
  SWOT-GAT series: the two runs are averaged per node and day (a day needs both). A reservoir's
  series is the sum over its member nodes on days when every member has a value; a reservoir with
  a member missing from the predictions is skipped.
  A reservoir is scored if its gauge has at least 730 days in the window and is not constant. A
  model is scored there if model and gauge share at least 50% of the window's days. On the shared
  days both series have their own mean removed (model and gauge use different reference levels).
    nse    = 1 - sum((model - gauge)^2) / sum(gauge^2)
    nrmse  = RMSE / (max - min of the mean-removed gauge)
    r      = Pearson correlation;  amp = VR = sd(model) / sd(gauge). A constant prediction gets
             r = amp = 0.
    lo_hit, hi_hit       share of the gauge's lowest (highest) 10% of days that are also in the
                         model's lowest (highest) 10%; a constant prediction gets 0.10
    lo_spread, hi_spread (P50 - P10) and (P90 - P50) of the model divided by the gauge's
    clim_ratio           RMSE(model) / RMSE(normal year). The normal year here is the gauge's
                         day-of-year cycle built leave-one-year-out within 2010-2020: anomalies
                         from each year's median, at least 3 other years, 31-day circular
                         smoothing. It is computed when the cycle covers at least 80% of the
                         shared days.
  Seasonal decomposition: an ordinary least-squares fit of mean + linear trend + annual harmonic
  (period 365.25 days) to each mean-removed series.
    seas_amp     harmonic amplitude of the model / of the gauge
    phase        day of the harmonic peak, model minus gauge, wrapped to +/- half a year
    r_ns, amp_ns correlation and sd ratio of the residuals left after removing each series' own
                 harmonic and trend
    seas_frac    share of the gauge's variance that its annual harmonic explains
    peak_doy_obs day of the gauge's harmonic peak, in days after 1 January
"""
from swotgat.paths import WORK
import time, numpy as np, pandas as pd
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork"); T3 = F/"testsets/regime3_full_1980_2020"; P = F/"predictions"; OUT = Path(__file__).resolve().parent
W0, W1 = pd.Timestamp("2010-01-01"), pd.Timestamp("2020-12-31"); t0 = time.time()
S = pd.read_csv(T3/"sites.csv", dtype={"node": str, "members": str, "basin": str})
TR = pd.read_parquet(T3/"truth_clean.parquet"); TR["date"] = pd.to_datetime(TR.date)
TR = TR[TR.use.fillna(False).astype(bool) & (TR.date >= W0) & (TR.date <= W1)]
need = {(str(r.basin), m) for r in S.itertuples() for m in str(r.members).split(";")}
def gnn_run(run):
    """Read the 2010-2020 SWOT-GAT predictions of one run at the member nodes of the scored reservoirs.

    The run's files of all window tags (ipcfg.HIND; checkpoint tag from ipcfg.STEP) are cut to the
    window, concatenated, and duplicate node-days are dropped. Return the column `pred_dv_SWOTAE`
    as a Series indexed by (basin, subbasin, date).
    """
    parts = []
    for c in __import__("ipcfg").HIND:
        d = pd.read_parquet(P/run/f"{__import__('ipcfg').STEP[run]}_{c}.parquet", columns=["basin", "subbasin", "date", "pred_dv_SWOTAE"])
        d["basin"] = d.basin.astype(str); d["subbasin"] = d.subbasin.astype(str); d["date"] = pd.to_datetime(d.date)
        d = d[(d.date >= W0) & (d.date <= W1)]; parts.append(d[[(b, s) in need for b, s in zip(d.basin, d.subbasin)]])
    return pd.concat(parts).drop_duplicates(["basin", "subbasin", "date"]).set_index(["basin", "subbasin", "date"]).pred_dv_SWOTAE
Gn = pd.concat([gnn_run(r) for r in __import__("ipcfg").RUNS], axis=1).mean(axis=1, skipna=False).rename("p").sort_index()
MOD = {"v10pm": {}}
for _, s in S.iterrows():
    mem = str(s.members).split(";")
    try: g = Gn.loc[str(s.basin)].loc[mem].reset_index()
    except KeyError: continue
    if g.subbasin.nunique() < len(mem): continue
    k = g.groupby("date").subbasin.nunique(); MOD["v10pm"][int(s.DAM_ID)] = g.groupby("date").p.sum()[k == len(mem)]
print(f"SWOT-GNN {len(MOD['v10pm'])} sites ({time.time()-t0:.0f}s)", flush=True)
for name, tag in (("GDROM", "gdrom3c_g_y1980"), ("ISTARF", "istarf3c_g_y1980"), ("MODROM-cal", "modromcal3c_g_y1980"), ("MODROM-tr", "modromrf3c_g_y1980_pubnorm")):
    d = pd.read_parquet(P/f"grades_{tag}"/"full_series.parquet"); d["date"] = pd.to_datetime(d.date); d = d[(d.date >= W0) & (d.date <= W1)]
    MOD[name] = {int(k): g.set_index("date").storage_mcm for k, g in d.groupby("DAM_ID")}; print(f"{name} {len(MOD[name])} dams", flush=True)
import bl                                            # BASE=1 adds the two baselines of bl.py as models
if bl.BASE:
    _d = pd.date_range(W0, W1); _ids = [int(x) for x in S.DAM_ID]
    MOD[bl.A_NAME] = bl.swot_cycles(_ids, _d)
    MOD[bl.B_NAME] = {d: s for d in _ids if (s := bl.gauge_cycle(d, W0, W1, _d)) is not None}
    print(f"baselines: {bl.A_NAME} {len(MOD[bl.A_NAME])} dams, {bl.B_NAME} {len(MOD[bl.B_NAME])} dams", flush=True)
def normal_year_loyo(o):
    """Return the gauge's normal year, built leave-one-year-out within the window.

    For each year, the day-of-year cycle of the other years (anomalies from each year's median,
    day 366 folded into 365, gaps interpolated, 31-day circular smoothing). A year stays NaN when
    fewer than 3 other years exist.
    """
    out = pd.Series(np.nan, index=o.index)
    for y in np.unique(o.index.year):
        ref = o[o.index.year != y]
        if ref.index.year.nunique() < 3: continue
        an = ref - ref.groupby(ref.index.year).transform("median")
        doy = an.groupby(np.minimum(an.index.dayofyear, 365)).mean().reindex(range(1, 366)).interpolate(limit_direction="both")
        sm = pd.concat([doy.iloc[-15:], doy, doy.iloc[:15]]).rolling(31, center=True).mean().iloc[15:-15]; sm.index = range(1, 366)
        k = o.index.year == y; out[k] = sm.reindex(np.minimum(o.index[k].dayofyear, 365)).values
    return out
Wf = 2*np.pi/365.25
def harm_fit(t, y):
    """Fit mean + linear trend + annual harmonic to the series y by least squares (t in days).

    Return the four coefficients (mean, trend per 365.25 days, cosine, sine) and the harmonic part
    of the fit.
    """
    X = np.column_stack([np.ones_like(t), t/365.25, np.cos(Wf*t), np.sin(Wf*t)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None); return b, X[:, 2:] @ b[2:]
rows = []
for _, s in S.iterrows():
    dam = int(s.DAM_ID); o_all = TR[TR.DAM_ID == dam].set_index("date").storage_mcm.sort_index()
    if len(o_all) < 730 or o_all.std() == 0: continue
    clim = normal_year_loyo(o_all)
    for name, ser in MOD.items():
        p = ser.get(dam)
        if p is None: continue
        j = pd.concat([o_all.rename("o"), p.rename("p"), clim.rename("c")], axis=1).dropna(subset=["o", "p"])
        if len(j) < 0.5*len(pd.date_range(W0, W1)): continue   # overlap rule: model and gauge share >= 50% of the window's days
        o = j.o - j.o.mean(); q = j.p - j.p.mean(); rng = o.max() - o.min()
        if rng <= 0: continue
        rmse = np.sqrt(((q - o)**2).mean()); flat = q.std() == 0
        m = dict(MODEL=name, PERIOD="2010-20", DAM_ID=dam, n_days=len(j), nrmse=rmse/rng, r=0.0 if flat else float(np.corrcoef(o, q)[0, 1]), amp=0.0 if flat else q.std()/o.std())
        m["nse"] = float(1 - ((q - o)**2).sum()/(o**2).sum())   # NSE on the mean-removed series o, q
        lo_o, hi_o = o <= np.percentile(o, 10), o >= np.percentile(o, 90)
        if flat: m["lo_hit"] = m["hi_hit"] = 0.10
        else:
            lo_q, hi_q = q <= np.percentile(q, 10), q >= np.percentile(q, 90); m["lo_hit"] = float((lo_o & lo_q).sum()/lo_o.sum()); m["hi_hit"] = float((hi_o & hi_q).sum()/hi_o.sum())
        p10o, p50o, p90o = np.percentile(o, [10, 50, 90]); p10q, p50q, p90q = np.percentile(q, [10, 50, 90])
        m["lo_spread"] = (p50q - p10q)/(p50o - p10o) if p50o > p10o else np.nan; m["hi_spread"] = (p90q - p50q)/(p90o - p50o) if p90o > p50o else np.nan
        t = ((j.index - W0).days).values.astype(float); oc, qc = o.values, q.values
        bo, ho = harm_fit(t, oc); Ao = np.hypot(bo[2], bo[3]); pk_o = (np.arctan2(bo[3], bo[2])/Wf) % 365.25
        ro = oc - ho - bo[1]*(t/365.25 - (t/365.25).mean()); m["seas_frac"] = np.var(ho)/np.var(oc); m["peak_doy_obs"] = pk_o
        if flat: m.update(seas_amp=0.0, phase=np.nan, r_ns=0.0, amp_ns=0.0)
        else:
            bq, hq = harm_fit(t, qc); Aq = np.hypot(bq[2], bq[3]); pk_q = (np.arctan2(bq[3], bq[2])/Wf) % 365.25
            rq = qc - hq - bq[1]*(t/365.25 - (t/365.25).mean())
            m.update(seas_amp=Aq/Ao if Ao > 0 else np.nan, phase=(pk_q - pk_o + 182.625) % 365.25 - 182.625,
                     r_ns=np.corrcoef(ro, rq)[0, 1] if ro.std() > 0 and rq.std() > 0 else 0.0, amp_ns=rq.std()/ro.std() if ro.std() > 0 else np.nan)
        c = j.c
        if c.notna().sum() >= 0.8*len(j):
            c = c.fillna(c.mean()); c = c - c.mean(); rc = np.sqrt(((c - o)**2).mean()); m["clim_ratio"] = rmse/rc if rc > 0 else np.nan
        rows.append(m)
R = pd.DataFrame(rows)
meta = pd.read_csv(F/"eval/grl_figs/sites_v5_g.csv").drop_duplicates("DAM_ID").set_index("DAM_ID")[["name", "state", "lat", "lon", "cap", "use"]]
R = R.join(meta, on="DAM_ID").merge(S[["DAM_ID", "site_tier", "basin"]].rename(columns={"site_tier": "tier"}), on="DAM_ID", how="left")
R.to_csv(OUT/"sites_2010_g.csv", index=False)
hm = R[R.tier.isin(["High", "Medium"])]; M4 = ["v10pm", "MODROM-cal", "GDROM", "ISTARF"]
c4 = set.intersection(*[set(hm[hm.MODEL == m].DAM_ID) for m in M4]); c5 = c4 & set(hm[hm.MODEL == "MODROM-tr"].DAM_ID)
print(f"rows {len(R)} | High/Medium per model {hm.groupby('MODEL').size().to_dict()} | 4-model common {len(c4)} | 5-model common {len(c5)} ({time.time()-t0:.0f}s)")
print(hm[hm.DAM_ID.isin(c4)].groupby("MODEL")[["nrmse", "r", "amp", "seas_amp", "phase", "r_ns", "amp_ns", "clim_ratio"]].median().round(3).to_string())
