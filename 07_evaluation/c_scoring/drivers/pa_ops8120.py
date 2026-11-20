"""Describe how each reservoir is operated, from its gauge record over water years 1981-2020.

The variables have the same definitions as group F of pa_vars.py, which uses water years
1981-2009. Here the window is 1980-10-01 to 2020-09-30. It overlaps the scored years 2010-2020,
so these variables describe the scored record itself. They are used for the scatter figures.
pa_vars.py's table (out/vars.parquet, 1981-2009) is not changed.

Usage: bin/run 07_evaluation/c_scoring/drivers/pa_ops8120.py      (no arguments; after pa_vars.py)

Reads (through pa_lib.py; paths relative to $SWOTGAT_WORK)
  the reservoir list (ids from an earlier model configuration's score table)
  the 1980-2020 gauge test set, copy in CONUS_storage_fork/eval/g4_fix/testsets: truth_clean.parquet,
      days with the `use` flag set
  GDW_v1_0/GDW_v1_0.gdb   Global Dam Watch (GDW) v1.0: capacity CAP_MCM, joined by GRAND_ID = DAM_ID
  CONUS_storage_fork/ResOpsUS/time_series_single_variable_table/DAILY_AV_OUTFLOW_CUMECS.csv
  out/vars.parquet        only to print rank correlations with the 1981-2009 variables

Method
  A water year (October to September) counts when it has at least 292 gauge days. A reservoir
  needs at least 5 such years; only their days are used.
    irreg_8120       1 - R2 of storage against its mean seasonal cycle:
                     sum((anomaly - cycle anomaly)^2) / sum(anomaly^2). The cycle is the
                     day-of-year mean (day 366 folded into 365; at least 330 days of the year
                     present; gaps interpolated; 31-day circular running mean).
    draw_8120        mean over water years of (max - min storage), divided by capacity
    iav_8120         standard deviation of the water-year mean storage, divided by capacity
    trend_8120       linear slope of the water-year means, per decade, divided by capacity
    full_frac_8120, empty_frac_8120   share of days at or above 95%, at or below 10%, of the
                     record's 99th percentile of storage
    n_wy_8120        number of water years used
    rel_cv_8120      coefficient of variation of daily release (ResOpsUS outflow), with the same
                     rule for water years

Writes out/ops_8120.parquet next to pa_lib.py, one row per reservoir of the list. Prints how many
reservoirs have each variable and the Spearman correlation with the 1981-2009 version.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import pa_lib as L

ids = L.site_ids(); A0, A1 = "1980-10-01", "2020-10-01"; wyf = lambda ix: ix.year + (ix.month >= 10)
import pyogrio
GD = pyogrio.read_dataframe(L.R/"GDW_v1_0/GDW_v1_0.gdb", layer="GDW_barriers_v1_0", columns=["GRAND_ID", "CAP_MCM"], read_geometry=False)
GD = GD[GD.GRAND_ID.notna()]; GD["GRAND_ID"] = GD.GRAND_ID.astype(int); cap = pd.to_numeric(GD.drop_duplicates("GRAND_ID").set_index("GRAND_ID").CAP_MCM, errors="coerce").reindex(ids)
TR = pd.read_parquet(L.T3/"truth_clean.parquet"); TR["date"] = pd.to_datetime(TR.date); TR = TR[TR.use.fillna(False).astype(bool) & TR.DAM_ID.isin(ids)]
def cycle(s):
    """Return the mean seasonal cycle of a daily series, indexed by day of year 1-365.

    The cycle is the day-of-year mean (day 366 folded into 365), with gaps interpolated and a 31-day
    circular running mean applied. Return None when fewer than 330 days of the year have data.
    """
    doy = np.minimum(s.index.dayofyear.values, 365); m = pd.Series(s.values).groupby(doy).mean().reindex(range(1, 366))
    if m.notna().sum() < 330: return None
    m = m.interpolate(limit_direction="both"); e = np.r_[m.values[-15:], m.values, m.values[:15]]
    return pd.Series(np.convolve(e, np.ones(31)/31, "valid"), index=range(1, 366))
F = {k: {} for k in ("irreg_8120", "draw_8120", "iav_8120", "full_frac_8120", "empty_frac_8120", "trend_8120", "n_wy_8120")}
for d, g in TR.groupby("DAM_ID"):
    s = g.set_index("date").storage_mcm.sort_index(); a = s[(s.index >= A0) & (s.index < A1)]; wy = wyf(a.index); n = a.groupby(wy).size(); good = n[n >= 292].index
    if len(good) < 5: continue
    a = a[wy.isin(good)]; F["n_wy_8120"][d] = len(good); cy = cycle(a); c_ = cap.get(d, np.nan)
    if cy is not None:
        an = a - a.mean(); cv = cy.reindex(np.minimum(a.index.dayofyear, 365)).values; cv = cv - cv.mean()
        F["irreg_8120"][d] = float(((an.values - cv)**2).sum()/(an.values**2).sum())
    gw = a.groupby(wyf(a.index)); p99 = a.quantile(0.99)
    if np.isfinite(c_) and c_ > 0:
        F["draw_8120"][d] = float((gw.max() - gw.min()).mean()/c_); F["iav_8120"][d] = float(gw.mean().std()/c_)
        yy = gw.mean(); F["trend_8120"][d] = float(np.polyfit(yy.index.values, yy.values, 1)[0]*10/c_)
    if p99 > 0: F["full_frac_8120"][d] = float((a >= 0.95*p99).mean()); F["empty_frac_8120"][d] = float((a <= 0.10*p99).mean())
out = pd.DataFrame({k: pd.Series(v) for k, v in F.items()}).reindex(ids); out.index.name = "DAM_ID"
for c, fn in (("rel_cv_8120", "OUTFLOW"),):
    OF = pd.read_csv(L.F/f"ResOpsUS/time_series_single_variable_table/DAILY_AV_{fn}_CUMECS.csv", usecols=lambda c: c == "date" or (c.isdigit() and int(c) in set(ids)))
    OF["date"] = pd.to_datetime(OF.date); OF = OF.set_index("date"); OF = OF[~OF.index.duplicated()].sort_index(); r = {}
    for k in OF.columns:
        o = OF[k][(OF.index >= A0) & (OF.index < A1)].dropna(); n = o.groupby(wyf(o.index)).size(); good = n[n >= 292].index
        if len(good) >= 5:
            o = o[wyf(o.index).isin(good)]
            if o.mean() > 0: r[int(k)] = float(o.std()/o.mean())
    out[c] = pd.Series(r).reindex(ids)
out.to_parquet(L.OUT/"ops_8120.parquet")
V = pd.read_parquet(L.OUT/"vars.parquet")
print("coverage:", out.notna().sum().to_dict())
print("Spearman 1981-2009 vs 1981-2020:", {k: round(float(V[k.replace("8120", "8109")].corr(out[k], method="spearman")), 3) for k in ("irreg_8120", "draw_8120", "iav_8120", "trend_8120")})
