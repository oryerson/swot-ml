"""Import-safe copy of gauge_stat_screen.py, used by build_plus.py.

Same code as ../../gauge_stat_screen.py, except that the audit runs only when the file is executed as a
script and its output folder can be set with the environment variable GSS_OUT. Importing `screen` from this
copy writes nothing. stage_gss.py records how the copy was made. The description of gauge_stat_screen.py
follows.

Flag statistical errors (spikes, steps, flat runs, ...) in every gauge series used for evaluation.

This script only finds and names suspect days. build_gauge_exclusions.py decides which flags
remove a day and also reads the agencies' own quality flags. build_regime3.py imports `screen`.

Reads ($SWOTGAT_WORK/CONUS_storage_fork)
  testsets/regime1_2023_2025/truth.parquet   reservoir storage from agency gauges, 2023-2025
                                             ("regime 1"; set name res_2023_25)
  testsets/regime2_2015_2020/truth.parquet   reservoir storage from ResOpsUS, 2015-2020
                                             ("regime 2"; set name res_2015_20)
  testsets/rivers/gauge_daily.parquet        river stage; years before 2022 form the set
                                             river_2015_20, later years river_2023_25
  crosswalk/resops_record_verdict.csv        reservoir capacity (`cap`)

Writes (eval/swotqc/audit)
  gauge_stat_flags.parquet   id, set, date, flag (one row per flagged day)
  gauge_stat_summary.csv     one row per series: n_days, noise, quantized, flat_share and the
                             number of days per flag

Usage
  python gauge_stat_screen.py     (no arguments)

Method (function `screen`; a series needs at least 30 values)
  noise = 1.4826 times the median absolute deviation of the day-to-day changes.
  range = 98th minus 2nd percentile of the series.
  Day flags:
    spike     One or two days jump away from the series and straight back, with no motion
              around them: the excursion from the mean of the day before and the day after
              exceeds max(8 x noise, floor); the levels before and after agree within 15% of
              the excursion; and the day-to-day changes just before and just after are below
              25% of it. A flood peak rises and recedes over several days and fails this test.
              floor = 2% of the range (reservoirs) or 0.15 m (rivers).
    rate      one-day change above 25% of capacity (reservoirs with a known capacity)
    negative  storage below zero (reservoirs; not for anomaly series, which have no true zero)
    over_cap  storage above 1.25 times capacity (the capacity itself may be wrong)
    flat_run  at least 21 identical consecutive values (reservoirs) or 10 (rivers), in a series
              that otherwise moves or where the run is shorter than half the record
    step      one-day shift above max(15 x noise, 5% of range) between quiet periods (changes
              on the 3 days either side below 10% of it) that does not revert (the medians of
              the 30 days before and after differ by more than 60% of it)
  Series flags, written to the summary:
    quantized   the two most common values cover more than 50% of the days, or the series has
                at most two distinct values
    flat_share  share of days that lie in flat runs (printed when above 50%)

Notes
  Downstream, build_gauge_exclusions.py removes `negative` days and reservoir `spike` days. All
  other flags, including river spikes and `rate`, are listed for review and the days are kept.
  The whole script runs when the module is imported (there is no main guard), so
  `from gauge_stat_screen import screen` in build_regime3.py repeats the screening and rewrites
  the two output files.
"""
from swotgat.paths import WORK
import numpy as np, pandas as pd
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork"); import os
OUT = Path(os.environ.get("GSS_OUT", str(F/"eval/swotqc/audit")))
cap = pd.read_csv(F/"crosswalk/resops_record_verdict.csv").drop_duplicates("DAM_ID").set_index("DAM_ID").cap

def rmad(x):
    """Return the robust standard deviation of the finite values: 1.4826 x median absolute deviation."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan

def screen(s, kind, capv=np.nan, absolute=True):
    """Screen one daily series with the statistical tests of the module docstring.

    kind: "res" (reservoir storage) or "river" (stage). capv: capacity in the unit of the series,
    if known. absolute: False for anomaly series, which skips the negative and over_cap tests.
    Returns ([(date, flag), ...], summary dict); both are empty below 30 values.
    """
    s = s.sort_index(); s = s[~s.index.duplicated()].asfreq("D"); x = s.values; n = len(x); out = []
    ok = np.isfinite(x)
    if ok.sum() < 30: return out, {}
    d = np.diff(x); noise = rmad(d[np.isfinite(d)]); rng = np.nanpercentile(x, 98) - np.nanpercentile(x, 2)
    floor = 0.02 * rng if kind == "res" else 0.15
    thr = max(8 * (noise if noise > 0 else 0), floor)
    for i in range(2, n - 3):
        for w in (1, 2):                                   # excursion of w days: x[i..i+w-1]
            pre, post = x[i-1], x[i+w]
            seg = x[i:i+w]
            if not (np.isfinite(pre) and np.isfinite(post) and np.isfinite(seg).all()): continue
            exc = seg - (pre + post) / 2
            if np.min(np.abs(exc)) <= thr or (w == 2 and exc[0] * exc[1] < 0): continue
            if abs(post - pre) > 0.15 * np.min(np.abs(exc)): continue
            around = [x[i-1] - x[i-2], x[i+w+1] - x[i+w]]
            if not np.isfinite(around).all() or max(abs(v) for v in around) > 0.25 * np.min(np.abs(exc)): continue
            out += [(s.index[j], "spike") for j in range(i, i + w)]; break
    if kind == "res" and np.isfinite(capv) and capv > 0:
        for i in np.flatnonzero(np.abs(np.nan_to_num(d)) > 0.25 * capv): out.append((s.index[i+1], "rate"))
        if absolute:
            for i in np.flatnonzero(ok & (x > 1.25 * capv)): out.append((s.index[i], "over_cap"))
    if kind == "res" and absolute:
        for i in np.flatnonzero(ok & (x < 0)): out.append((s.index[i], "negative"))
    # flat runs
    minrun = 21 if kind == "res" else 10; run = 1; flat_days = 0
    moves = np.nanmedian(np.abs(d[np.isfinite(d)])) > 0 if np.isfinite(d).any() else False
    for i in range(1, n + 1):
        if i < n and ok[i] and ok[i-1] and x[i] == x[i-1]: run += 1; continue
        if run >= minrun:
            flat_days += run
            if moves or run < 0.5 * ok.sum():
                out += [(s.index[j], "flat_run") for j in range(i - run, i)]
        run = 1
    # steps
    if noise > 0:
        for i in np.flatnonzero(np.abs(np.nan_to_num(d)) > max(15 * noise, 0.05 * rng)):
            before, after = x[max(0, i-29):i+1], x[i+1:i+31]
            quiet = np.r_[d[max(0, i-3):i], d[i+1:i+4]]
            if len(quiet) and np.nanmax(np.abs(quiet)) < 0.1 * abs(d[i]) and np.isfinite(before).sum() > 10 and np.isfinite(after).sum() > 10 \
               and abs(np.nanmedian(after) - np.nanmedian(before)) > 0.6 * abs(d[i]):
                out.append((s.index[i+1], "step"))
    vc = pd.Series(x[ok]).value_counts()
    summ = dict(n_days=int(ok.sum()), noise=noise, quantized=bool(vc.iloc[:2].sum() / ok.sum() > 0.5 and len(vc) > 2 or len(vc) <= 2),
                flat_share=flat_days / ok.sum())
    return out, summ

if __name__ == "__main__":
    rows, summ = [], []
    T1 = pd.read_parquet(F/"testsets/regime1_2023_2025/truth.parquet"); T1["date"] = pd.to_datetime(T1.date)
    for d, g in T1.groupby("DAM_ID"):
        o, sm = screen(g.set_index("date").storage_mcm, "res", cap.get(d, np.nan), absolute=not g.is_anomaly.any())
        rows += [(str(d), "res_2023_25", t, f) for t, f in o]; summ.append(dict(id=str(d), set="res_2023_25", source=g.source.iloc[0], **sm))
    T2 = pd.read_parquet(F/"testsets/regime2_2015_2020/truth.parquet"); T2["date"] = pd.to_datetime(T2.date)
    for d, g in T2.groupby("DAM_ID"):
        o, sm = screen(g.set_index("date").storage_mcm, "res", cap.get(d, np.nan))
        rows += [(str(d), "res_2015_20", t, f) for t, f in o]; summ.append(dict(id=str(d), set="res_2015_20", source="ResOpsUS", **sm))
    G = pd.read_parquet(F/"testsets/rivers/gauge_daily.parquet"); G["date"] = pd.to_datetime(G.date)
    for (site, w), g in G.groupby([G.site_no, G.date.dt.year < 2022]):
        lab = "river_2015_20" if w else "river_2023_25"
        o, sm = screen(g.set_index("date").stage_m, "river")
        rows += [(str(site), lab, t, f) for t, f in o]; summ.append(dict(id=str(site), set=lab, source=g.source.iloc[0], **sm))
    R = pd.DataFrame(rows, columns=["id", "set", "date", "flag"]).drop_duplicates(); R.to_parquet(OUT/"gauge_stat_flags.parquet", index=False)
    Sm = pd.DataFrame(summ)
    cnt = R.groupby(["set", "id", "flag"]).size().unstack(fill_value=0)
    Sm = Sm.merge(cnt.reset_index(), on=["set", "id"], how="left").fillna({c: 0 for c in cnt.columns}); Sm.to_csv(OUT/"gauge_stat_summary.csv", index=False)
    print(R.groupby(["set", "flag"]).agg(days=("date", "size"), series=("id", "nunique")).to_string())
    print("\nseries per set:", Sm.groupby("set").size().to_dict())
    Sm["quantized"] = Sm.quantized.fillna(False).astype(bool)
    print("quantized series:", Sm[Sm.quantized].groupby("set").id.apply(list).to_dict())
    print("series with flat share > 50%:", Sm[Sm.flat_share > .5].groupby("set").size().to_dict())
