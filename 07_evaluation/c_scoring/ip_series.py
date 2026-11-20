"""Build the series file of the SWOT Period scorer: daily gauge and model storage per reservoir.

Usage (after `source 07_evaluation/c_scoring/env.sh`):
    bin/run 07_evaluation/c_scoring/ip_series.py --out NPZ --span TAGS --hind TAGS [--p1x TAGS] [--check]
  The paper's file was built with:  --out $IP_SERIES --span x26 --hind h10 --p1x x26
  --out    the .npz file to write
  --span   window tags of the prediction files for the 2023-2025 arrays (default c1)
  --hind   window tags of the prediction files for the 2015-2020 arrays (default x1014,c2)
  --p1x    window tags for the SWOT Period arrays; without this option they are not written
  --check  stop unless the rebuilt arrays reproduce the template's (see Notes)
  IP_RUNS, IP_STEPS   the prediction runs to average and their checkpoint tags (through ipcfg.py)

Reads (paths relative to $SWOTGAT_WORK/CONUS_storage_fork)
  eval/grl_figs/series_g.npz   the template: the series file of an earlier figure set. Its list and
      order of dams (`dams`) and all its arrays are copied; only the arrays named below are
      replaced or added.
  predictions/<run>/<step>_<tag>.parquet   SWOT-GAT predictions (column pred_dv_SWOTAE)
  testsets/regime1_2023_2025/sites.csv, testsets/regime2_2015_2020/sites.csv and
      testsets/regime1x_2023_2026/sites.csv   the member nodes of each reservoir in three gauge
      test sets
  testsets/regime1x_2023_2026/truth_clean_v1x.parquet   gauge storage 2023-2026. Gauge data are
      used for scoring only, never for training.
  predictions/grades_<tag>/full_series.parquet for the tags gdrom3c_g26h0, istarf3c_g26h0 and
      modromcal3c_g26h0   the rule-model runs that extend into 2026

Writes the file given by --out. Array keys are "<model>|<period>|obs" and "<model>|<period>|pred",
with one row per dam. The model key "v10pm" is the label for SWOT-GAT (the paper's model: mean of
two seeds); the key is kept because the scorers and figure scripts select on it.

Method
  SWOT-GAT series. The runs are averaged per node and day (a day needs every run). A reservoir's
  series is the sum over its member nodes, kept only on days when every member is present.
  "v10pm|P1|pred" (2023-01-01 to 2025-12-31) and "v10pm|P2|pred" (2015-01-01 to 2020-12-31) are
  rebuilt from the --span and --hind prediction files. From each row the mean over the days it
  shares with the template's gauge array is removed (those gauge arrays are already mean-removed);
  if no day is shared, the row's own mean is removed. The arrays keep the template's data type.
  With --p1x, the "P1x" arrays are added for every model of the template, on 2023-01-01 to
  2026-09-30 (the dates are stored under the key "P1x|dates"):
    "<model>|P1x|obs"   gauge storage of the 2023-2026 test set (the same array for every model)
    "v10pm|P1x|pred"    SWOT-GAT at the reservoirs of that test set, from the --p1x prediction files
    rule models         GDROM, ISTARF and MODROM-cal: storage of the 2026 rule runs (run with a
                        fixed hash seed), NaN after their last day. Their inflow ends 2026-06-12,
                        the end of the GRADES-hydroDL file. MODROM-tr has no 2026 run and is all
                        NaN; it appears in no figure.
  The P1x arrays are float64 and have no mean removed. score_x26span.py removes the means itself.
  The P1 and P2 arrays are the same with or without --p1x.

Notes
  --check compares the rebuilt P1 and P2 arrays with the template's. It requires the same pattern
  of finite values and a difference below 1e-4 of each reservoir's range (the template's values
  were stored as 16-bit integers). It can pass only when IP_RUNS are the runs the template was
  built from, which belong to an earlier model configuration. Without --check the comparison is
  only printed.
  The script imports grl_common.py (08_figures in this repository, deployed to eval/grl_figs) and
  ipcfg.py, which must lie in the script's own folder or on PYTHONPATH.
"""
from swotgat.paths import WORK
import sys, argparse, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, f"{WORK}/CONUS_storage_fork/eval/grl_figs"); sys.path.insert(0, str(Path(__file__).resolve().parent))
import grl_common as G, ipcfg
F = Path(f"{WORK}/CONUS_storage_fork"); P = F/"predictions"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--span", default="c1"); ap.add_argument("--hind", default="x1014,c2")
ap.add_argument("--check", action="store_true"); ap.add_argument("--p1x", default=""); a = ap.parse_args()
Z = np.load(F/"eval/grl_figs/series_g.npz"); arr = {k: Z[k] for k in Z.files}; dams = [int(d) for d in Z["dams"]]

def per_dam(sites_csv, chunks, dates):
    """Return SWOT-GAT storage per dam on the given dates, one row per dam in the template's order.

    The runs of ipcfg.RUNS are read from the prediction files with the window tags in chunks and
    averaged per node and day (a day needs every run). A reservoir's series is the sum over the
    member nodes listed in sites_csv on days when every member is present; other days are NaN, and
    so are dams that are not in sites_csv or have a member missing from the predictions. The
    result is a float64 array of shape (number of dams, number of dates).
    """
    S = pd.read_csv(sites_csv, dtype={"node": str, "members": str, "basin": str})
    need = {(r.basin, m) for r in S.itertuples() for m in str(r.members).split(";")}
    w0, w1 = dates[0], dates[-1]; runs = []
    for run in ipcfg.RUNS:
        parts = []
        for c in chunks:
            x = pd.read_parquet(P/run/f"{ipcfg.STEP[run]}_{c}.parquet", columns=["basin", "subbasin", "date", "pred_dv_SWOTAE"])
            x["basin"] = x.basin.astype(str); x["subbasin"] = x.subbasin.astype(str); x["date"] = pd.to_datetime(x.date)
            x = x[(x.date >= w0) & (x.date <= w1)]; parts.append(x[[(b, s) in need for b, s in zip(x.basin, x.subbasin)]])
        runs.append(pd.concat(parts).drop_duplicates(["basin", "subbasin", "date"]).set_index(["basin", "subbasin", "date"]).pred_dv_SWOTAE)
    Gs = pd.concat(runs, axis=1).mean(axis=1, skipna=False).rename("p").sort_index()
    out = np.full((len(dams), len(dates)), np.nan, np.float64); idx = {d: i for i, d in enumerate(dams)}
    for s in S.itertuples():
        if int(s.DAM_ID) not in idx: continue
        mem = str(s.members).split(";")
        try: g = Gs.loc[s.basin].loc[mem].reset_index()
        except KeyError: continue
        if g.subbasin.nunique() < len(mem): continue
        k = g.groupby("date").subbasin.nunique(); q = g.groupby("date").p.sum()[k == len(mem)]
        out[idx[int(s.DAM_ID)]] = q.reindex(dates).values
    return out

new = {"v10pm|P1|pred": per_dam(F/"testsets/regime1_2023_2025/sites.csv", a.span.split(","), G.dates("2023-25")),
       "v10pm|P2|pred": per_dam(F/"testsets/regime2_2015_2020/sites.csv", a.hind.split(","), G.dates("2015-20"))}
for k, v in new.items():
    obs = arr[k.replace("|pred", "|obs")]
    for i in range(len(v)):                                      # remove the mean over the days shared with the gauge, as in the template
        sh = np.isfinite(v[i]) & np.isfinite(obs[i])
        if sh.any(): v[i] = v[i] - v[i][sh].mean()
        elif np.isfinite(v[i]).any(): v[i] = v[i] - np.nanmean(v[i])
    old = arr[k]; assert v.shape == old.shape, (k, v.shape, old.shape)
    both = np.isfinite(v) & np.isfinite(old)
    rng = np.nanmax(old, axis=1) - np.nanmin(old, axis=1)
    rel = np.nanmax(np.where(both, np.abs(v - old), np.nan), axis=1)/np.where(rng > 0, rng, 1)
    print(f"{k}: dams with data new {int(np.isfinite(v).any(1).sum())} / original {int(np.isfinite(old).any(1).sum())}; "
          f"max |new - original| / range over dams {np.nanmax(rel) if np.isfinite(rel).any() else float('nan'):.2e}; "
          f"finite-pattern mismatch {int((np.isfinite(v) != np.isfinite(old)).sum())}", flush=True)
    if a.check:
        assert np.array_equal(np.isfinite(v), np.isfinite(old)) and np.nanmax(rel) < 1e-4, f"{k} does not reproduce series_g"
    arr[k] = v.astype(old.dtype)
if a.p1x:                                                       # P1x: 2023-01..2026-09, raw values (no mean removed) for the SWOT Period scorer
    T26 = pd.date_range("2023-01-01", "2026-09-30", freq="D"); X1 = F/"testsets/regime1x_2023_2026"; idx = {d: i for i, d in enumerate(dams)}
    R26 = {"GDROM": "gdrom3c_g26h0", "ISTARF": "istarf3c_g26h0", "MODROM-cal": "modromcal3c_g26h0"}
    TX = pd.read_parquet(X1/"truth_clean_v1x.parquet"); TX["date"] = pd.to_datetime(TX.date)
    obs = np.full((len(dams), len(T26)), np.nan)
    for d, g in TX.groupby("DAM_ID"):
        if int(d) in idx: obs[idx[int(d)]] = g.set_index("date").storage_mcm.reindex(T26).values
    arr["P1x|dates"] = T26.values.astype("datetime64[D]")
    for m in sorted({k.split("|")[0] for k in arr if k.endswith("|P1|pred")}):
        if m == "v10pm": q = per_dam(X1/"sites.csv", a.p1x.split(","), T26)
        elif m in R26:                                       # 2026 rule runs, raw MCM, NaN after their last day
            q = np.full((len(dams), len(T26)), np.nan); x = pd.read_parquet(P/f"grades_{R26[m]}"/"full_series.parquet"); x["date"] = pd.to_datetime(x.date)
            for d, g in x[(x.date >= T26[0]) & (x.date <= T26[-1])].groupby("DAM_ID"):
                if int(d) in idx: q[idx[int(d)]] = g.set_index("date").storage_mcm.astype(float).reindex(T26).values
        else: q = np.full((len(dams), len(T26)), np.nan)      # MODROM-tr: no 2026 run, not in any figure
        arr[f"{m}|P1x|obs"], arr[f"{m}|P1x|pred"] = obs, q
        print(f"{m}|P1x: dams with predictions {int(np.isfinite(q).any(1).sum())}, last predicted day {T26[np.isfinite(q).any(0)].max().date()}; "
              f"dams with gauge days {int(np.isfinite(obs).any(1).sum())}", flush=True)
np.savez(a.out, **arr); print("wrote", a.out, flush=True)
