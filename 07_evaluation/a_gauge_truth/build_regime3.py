"""Build the 1980-2020 reservoir test set from ResOpsUS ("regime 3").

The test sets are called regimes: regime 1 is agency gauge storage for 2023-2025, regime 2 is
ResOpsUS storage for 2015-2020, and regime 3, built here, is the whole ResOpsUS record from
1980 to 2020 for the same sites as regime 2 (508 in the paper's run). Regimes 1 and 2 are only
read. All rules below were fixed before any model was scored.

Reads ($SWOTGAT_WORK/CONUS_storage_fork unless noted)
  testsets/regime2_2015_2020/sites.csv, truth.parquet, truth_clean_v1.parquet, predict_nodes.csv
  crosswalk/resops_record_verdict.csv        per dam: unit fix factor and capacity
  ResOpsUS/time_series_all/ResOpsUS_<DAM_ID>.csv    daily storage and elevation
  ResOpsUS/attributes/time_series_inventory.csv     which dams have storage
  $SWOTGAT_NAS_WATER/cjgleason/ted/GDW/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp
                                             Global Dam Watch (GDW): year the dam was built
  eval/gauge_stat_screen.py                  imported for `screen`

Writes (testsets/regime3_full_1980_2020)
  truth_all.parquet    DAM_ID, date, storage_mcm, removed (True for days removed by a rule)
  truth_clean.parquet  the days that are not removed, with `use` (the site-year passes the
                       checks below)
  qa_site_year.csv     one row per dam and year with every check and `use`
  unit_runs.csv        each unit run that was found and what was done with it
  sites.csv, predict_nodes.csv   copies of the regime-2 files
  version.json         counts, input checksums and the comparison with regime 2

Usage
  python build_regime3.py     (no arguments)

Method
  Storage values. 2015-2020 is the frozen regime-2 series, with the same days removed as in its
  truth_clean_v1 table. 1980-2014 is ResOpsUS daily storage brought to the same scale:
  1. Raw storage times the dam's unit fix factor from the record verdict table, as in regime 2.
  2. Unit runs. Some records change unit for part of their length. Non-positive values are
     dropped first. A run is at least 30 consecutive values at 10^2.5 to 10^3.5 times the
     2015-2020 median, or the inverse. The factor is read from the dam's own elevation: the
     median ratio of storage on the run's days to the storage predicted from elevation by a
     quadratic storage-elevation fit of 2015-2020
     (used only when their correlation is at least 0.95). The run is rescaled when that ratio
     is within 5% of 1000 (cubic kilometers or thousand cubic meters against million cubic
     meters) or of 810.71 (acre-feet against million cubic meters). A run that is too large
     and has no such evidence is dropped, because a reservoir cannot hold 300 times its recent
     median. A run that is too small is kept, because near-empty and pre-filling periods are
     real; the site-year checks below decide whether it is used. Example: Center Hill is in
     cubic kilometers in 1991 and from 1996 on and in million cubic meters otherwise, so the
     single x1000 factor of regime 2 is right only from 1996 on.
  3. The cleaning of the regime 1 and 2 test sets (crosswalk/step7_build_testsets.py): exact
     zeros removed where the median exceeds 1, and isolated spikes removed (difference from
     the 5-day centered median above 10 times a robust scale).
  4. `gauge_stat_screen.screen` over 1980-2014: days flagged spike or negative are marked
     removed, as in truth_clean_v1. Other flags are only counted (n_flag_review).

  Site-year checks (qa_site_year.csv). A site-year is used for the main scores (`use` = True)
  when all of these hold:
    n_days >= 300;
    scale: the year's maximum is within 1/3 to 3 times the 2015-2020 maximum;
    not flat: the standard deviation is not zero and no single value covers more than 80% of
              the days;
    not first fill: the year is later than the GDW construction year plus 2 (before that
              there is no reservoir, or it is filling and not yet operated). Dams without a
              GDW construction year pass;
    no duplicate: the year's daily series does not correlate above 0.999 with that of another
              ResOpsUS dam. Dams with fewer than 300 values in the year or no variation are
              not compared; gaps are filled forward and backward before correlating.
  Listed but not excluded (`rating_flag`): the year's median residual from the 2015-2020
  storage-elevation fit (correlation above 0.9) exceeds 25% of the 2015-2020 storage range.
  This points to a new survey or a datum change and is meant for sensitivity tests.

Notes
  Importing gauge_stat_screen runs that whole script again and rewrites its two output files
  (it has no main guard).
  version.json reports how many 2015-2020 days differ from regime 2 (regime2_mismatch_days).
  The script does not stop if that number is not zero.
"""
from swotgat.paths import NAS_WATER, WORK
import hashlib, json, sys, numpy as np, pandas as pd
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork"); T = F/"testsets"; O = T/"regime3_full_1980_2020"
sys.path.insert(0, str(F/"eval")); sys.path.insert(0, str(F/"crosswalk"))
from gauge_stat_screen import screen                         # noqa: E402
O.mkdir(exist_ok=True)
S2 = pd.read_csv(T/"regime2_2015_2020/sites.csv", dtype={"node": str, "members": str, "basin": str})
L1 = pd.read_csv(F/"crosswalk/resops_record_verdict.csv").drop_duplicates("DAM_ID").set_index("DAM_ID")
CAP = L1.cap
W0, W1 = "1980-01-01", "2020-12-31"

def raw(dam):
    """Return one dam's ResOpsUS record as (storage, elevation or None).

    Both are series indexed by date, sorted, with duplicate and unreadable dates dropped.
    """
    t = pd.read_csv(F/f"ResOpsUS/time_series_all/ResOpsUS_{dam}.csv"); t["date"] = pd.to_datetime(t.date, errors="coerce")
    t = t.dropna(subset=["date"]).drop_duplicates("date").set_index("date").sort_index()
    st = pd.to_numeric(t.storage, errors="coerce"); el = pd.to_numeric(t.get("elevation"), errors="coerce") if "elevation" in t else None
    return st, el

def clean7(s):
    """Remove zero dropouts and isolated spikes from a storage series.

    Same rules as clean() in crosswalk/step7_build_testsets.py: exact zeros are removed where the
    median exceeds 1, then values that differ from the 5-day centered median by more than 10
    times a robust scale. Returns the series without the removed values.
    """
    s = s.astype(float).copy(); med = s.median()
    if med > 1: s[s == 0] = np.nan
    m = s.rolling(5, center=True, min_periods=3).median()
    scale = 1.4826*np.nanmedian(np.abs(np.diff(s.dropna().values))) + 1e-3*(s.max() - s.min())
    s[(s - m).abs() > 10*scale] = np.nan
    return s.dropna()

def unit_runs(v, ref_med, el=None):
    """Find and treat runs of at least 30 values at about 1000 times (or 1/1000 of) `ref_med`.

    v: storage before 2015 (fix factor applied). ref_med: the 2015-2020 median. el: elevation.
    The unit factor of a run (1000 or 810.71) is confirmed from the 2015-2020 storage-elevation
    fit, which uses the global FULLV (the dam's whole storage series). A confirmed run is
    rescaled, an unconfirmed large run is dropped, an unconfirmed small run is kept.
    Returns (series without non-positive values, list of log rows).
    """
    v = v.dropna(); v = v[v > 0]; logs = []
    if v.empty or not np.isfinite(ref_med) or ref_med <= 0: return v, logs
    fit = None
    if el is not None:
        pr = pd.concat([FULLV.rename("s"), el.rename("e")], axis=1).loc["2015":"2020"].dropna()
        if len(pr) > 200 and pr.e.std() > 0.05 and pr.s.corr(pr.e) >= 0.95: fit = np.polyfit(pr.e, pr.s, 2)
    L = np.log10(v / ref_med)
    off = (L.abs() >= 2.5) & (L.abs() <= 3.5)
    grp = (off != off.shift()).cumsum()
    out = v.copy()
    for g, idx in L.groupby(grp).groups.items():
        if not off.loc[idx[0]] or len(idx) < 30: continue
        seg = v.loc[idx]; lev = float(np.median(np.log10(seg / ref_med))); big = lev > 0
        ratio, fac = np.nan, None
        if fit is not None:
            e = el.reindex(idx).dropna()
            pred = pd.Series(np.polyval(fit, e.values), index=e.index); pred = pred[pred > 0]
            if len(pred) >= 15:
                ratio = float(np.median(seg.reindex(pred.index) / pred))
                for k in (1000.0, 810.71):
                    q = ratio if big else 1 / ratio
                    if abs(q / k - 1) < 0.05: fac = k
        if fac:
            out.loc[idx] = seg / fac if big else seg * fac; act = f"rescaled {'/' if big else 'x'}{fac:g} (elevation evidence)"
        elif big:
            out = out.drop(idx); act = "dropped (too large, no elevation evidence for a unit factor)"
        else:
            act = "kept (low level: real low storage or pre-impoundment)"
        logs.append(dict(start=idx[0].date(), end=idx[-1].date(), n=len(idx), level_log10=round(lev, 3),
                         storage_over_elev_pred=round(ratio, 2) if np.isfinite(ratio) else None, action=act))
    return out, logs

# ---- duplicate screen across all ResOpsUS dams with storage, by year: daily correlation > 0.999
#      between two dams that each have at least 300 values that year (gaps filled before correlating)
inv = pd.read_csv(F/"ResOpsUS/attributes/time_series_inventory.csv", encoding="utf-8-sig")
ALL = {}
for d in inv[inv.STORAGE == 1].DAM_ID:
    try: ALL[int(d)] = raw(int(d))[0].loc[W0:W1]
    except FileNotFoundError: pass
WD = pd.DataFrame(ALL)
dup_years = {}
for y in range(1980, 2021):
    X = WD.loc[str(y)]; X = X.loc[:, X.notna().sum() >= 300]
    X = X.loc[:, X.std() > 0]
    if X.shape[1] < 2: continue
    Cm = np.corrcoef(X.ffill().bfill().values.T)
    a, b = np.where(np.triu(Cm > 0.999, 1))
    for i, j in zip(a, b):
        da, db = int(X.columns[i]), int(X.columns[j])
        dup_years.setdefault(da, {})[y] = db; dup_years.setdefault(db, {})[y] = da
print(f"duplicate (dam, year) pairs found: {sum(len(v) for v in dup_years.values())//2}", flush=True)

# construction year of each dam, from Global Dam Watch (YEAR_DAM), keyed by GRanD id (= ResOpsUS DAM_ID)
import geopandas as gpd
_g = gpd.read_file(f"{NAS_WATER}/cjgleason/ted/GDW/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp", columns=["GRAND_ID", "YEAR_DAM"], ignore_geometry=True)
_g["GRAND_ID"] = pd.to_numeric(_g.GRAND_ID, errors="coerce"); _g["YEAR_DAM"] = pd.to_numeric(_g.YEAR_DAM, errors="coerce")
DAMYR = _g[(_g.GRAND_ID > 0) & (_g.YEAR_DAM > 1800)].drop_duplicates("GRAND_ID").set_index("GRAND_ID").YEAR_DAM

R2F = pd.read_parquet(T/"regime2_2015_2020/truth.parquet"); R2F["date"] = pd.to_datetime(R2F.date)
_c = pd.read_parquet(T/"regime2_2015_2020/truth_clean_v1.parquet"); _c["date"] = pd.to_datetime(_c.date)
_k = R2F.merge(_c[["DAM_ID", "date"]].assign(keep=1), on=["DAM_ID", "date"], how="left")
R2C_RM = {d: set(g.date) for d, g in _k[_k.keep.isna()].groupby("DAM_ID")}
FULLV = None
truth, qa, units = [], [], []
for _, r in S2.iterrows():
    dam = int(r.DAM_ID); st, el = raw(dam)
    fixf = float(L1.fix_factor.get(dam, 1.0)) if np.isfinite(L1.fix_factor.get(dam, np.nan)) else 1.0
    v = (st * fixf).loc[W0:W1]
    ref_med = v.loc["2015":"2020"].dropna().median()
    FULLV = v
    v2, logs = unit_runs(v.loc[:"2014-12-31"], ref_med, el.loc[W0:W1] if el is not None else None)
    for lg in logs: units.append(dict(DAM_ID=dam, DAM_NAME=r.DAM_NAME, **lg))
    e = clean7(v2.dropna()) if len(v2.dropna()) >= 5 else v2.dropna()
    flags, _ = screen(e, "res", CAP.get(dam, np.nan), True) if len(e) >= 30 else ([], {})
    fl = pd.DataFrame(flags, columns=["date", "flag"]) if flags else pd.DataFrame(columns=["date", "flag"])
    rm = set(pd.to_datetime(fl[fl.flag.isin(["spike", "negative"])].date))
    fz = R2F[R2F.DAM_ID == dam].set_index("date").storage_mcm                      # frozen 2015-2020 truth
    rmz = set(R2C_RM.get(dam, set()))                                            # its truth_clean_v1 removals
    s = pd.concat([e, fz]).sort_index(); s = s[~s.index.duplicated(keep="last")]
    rm = rm | rmz
    sc = s[~s.index.isin(rm)]
    truth.append(pd.DataFrame(dict(DAM_ID=dam, date=s.index, storage_mcm=s.values, removed=s.index.isin(rm))))
    # storage-elevation relation of 2015-2020 (quadratic fit) for the rating check
    rel = None
    if el is not None:
        p = pd.concat([v.rename("s"), el.loc[W0:W1].rename("e")], axis=1).dropna()
        pr = p.loc["2015":"2020"]
        if len(pr) > 200 and pr.e.std() > 0.05 and pr.s.corr(pr.e) > 0.9:
            b = np.polyfit(pr.e, pr.s, 2); rel = (b, float(pr.s.max() - pr.s.min()), p)
    ref_max = sc.loc["2015":"2020"].max()
    for y, g in sc.groupby(sc.index.year):
        row = dict(DAM_ID=dam, year=int(y), n_days=len(g), ymax=float(g.max()), ymed=float(g.median()),
                   scale_ratio=float(g.max() / ref_max) if ref_max and ref_max > 0 else np.nan,
                   flat=bool(g.std() == 0 or g.value_counts(normalize=True).iloc[0] > 0.8),
                   dup_with=dup_years.get(dam, {}).get(int(y)), n_removed=int(sum(1 for d in rm if d.year == y)),
                   n_flag_review=int(((pd.to_datetime(fl.date).dt.year == y) & ~fl.flag.isin(["spike", "negative"])).sum()) if len(fl) else 0)
        dy = float(DAMYR.get(dam, np.nan))
        row["dam_year"] = dy; row["first_fill"] = bool(np.isfinite(dy) and y <= dy + 2)
        if rel is not None:
            b, rng, p = rel; py = p[p.index.year == y]
            row["rating_resid"] = float(np.median(py.s - np.polyval(b, py.e)) / rng) if len(py) > 30 and rng > 0 else np.nan
        row["scale_bad"] = bool(np.isfinite(row["scale_ratio"]) and not (1/3 <= row["scale_ratio"] <= 3))
        row["use"] = bool(row["n_days"] >= 300 and not row["scale_bad"] and not row["flat"] and not row["first_fill"] and row["dup_with"] is None)
        qa.append(row)

TR = pd.concat(truth, ignore_index=True); QA = pd.DataFrame(qa); UN = pd.DataFrame(units)
QA["rating_flag"] = QA.get("rating_resid", pd.Series(np.nan, index=QA.index)).abs() > 0.25
TR.to_parquet(O/"truth_all.parquet", index=False)
TC = TR[~TR.removed].drop(columns="removed")
TC = TC.merge(QA[["DAM_ID", "year", "use"]], left_on=["DAM_ID", TC.date.dt.year], right_on=["DAM_ID", "year"], how="left").drop(columns=["key_1", "year"], errors="ignore")
TC.to_parquet(O/"truth_clean.parquet", index=False)
QA.to_csv(O/"qa_site_year.csv", index=False); UN.to_csv(O/"unit_runs.csv", index=False)
S2.to_csv(O/"sites.csv", index=False)
import shutil; shutil.copy(T/"regime2_2015_2020/predict_nodes.csv", O/"predict_nodes.csv")
# consistency with the frozen regime-2 values on 2015-2020 (they should match on every day);
# the counts go to version.json
R2 = pd.read_parquet(T/"regime2_2015_2020/truth.parquet"); R2["date"] = pd.to_datetime(R2.date)
j = R2.merge(TR, on=["DAM_ID", "date"], suffixes=("_r2", "_r3"))
_cl = pd.read_parquet(O/"truth_clean.parquet"); _cl["date"] = pd.to_datetime(_cl.date)
_jc = _c.merge(_cl, on=["DAM_ID", "date"], suffixes=("_v1", "_r3"))
dd = (j.storage_mcm_r2 - j.storage_mcm_r3).abs() / j.storage_mcm_r2.abs().clip(lower=1e-6)
ver = dict(built=pd.Timestamp.now().isoformat(), sites=int(S2.shape[0]), truth_days=int(len(TR)), clean_days=int(len(TC)),
           headline_site_years=int(QA.use.sum()), site_years=int(len(QA)),
           excluded=dict(scale=int(QA.scale_bad.sum()), flat=int(QA.flat.sum()), first_fill=int(QA.first_fill.sum()),
                         duplicate=int(QA.dup_with.notna().sum()), short=int((QA.n_days < 300).sum())),
           rating_flagged=int(QA.rating_flag.sum()), unit_runs=UN.action.value_counts().to_dict() if len(UN) else {},
           regime2_overlap_days=int(len(j)), regime2_mismatch_days=int((dd > 1e-6).sum()),
           regime2_days_missing_in_r3=int(len(R2) - len(j)),
           clean_v1_days=int(len(_c)), clean_v1_days_in_r3_clean=int(len(_jc)),
           inputs={p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16] for p in
                   [str(T/"regime2_2015_2020/sites.csv"), str(F/"crosswalk/resops_record_verdict.csv")]})
json.dump(ver, open(O/"version.json", "w"), indent=1, default=str)
print(json.dumps(ver, indent=1, default=str))
if len(UN): print(UN.to_string())
