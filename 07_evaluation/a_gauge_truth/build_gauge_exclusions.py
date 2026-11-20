"""Remove gauge errors from the evaluation gauge tables (exclusion list, version 1).

The frozen test-set tables are not changed. A cleaned copy is written beside each one.

Reads ($SWOTGAT_WORK/CONUS_storage_fork)
  testsets/regime1_2023_2025/truth.parquet   reservoir storage from agency gauges, 2023-2025
  testsets/regime2_2015_2020/truth.parquet   reservoir storage from ResOpsUS, 2015-2020
  testsets/rivers/gauge_daily.parquet        river stage
  eval/swotqc/audit/gauge_stat_flags.parquet, gauge_stat_summary.csv   from gauge_stat_screen.py
  agency_cache/*.txt                         the raw agency responses saved by the harvest
                                             scripts (they hold the agencies' quality flags)

Writes
  testsets/gauge_exclusions_v1.parquet                 set, id, date, reason, action
  testsets/regime1_2023_2025/truth_clean_v1.parquet    without the REMOVE days
  testsets/regime2_2015_2020/truth_clean_v1.parquet    without the REMOVE days
  testsets/rivers/gauge_daily_clean_v1.parquet         without the REMOVE days
  eval/swotqc/audit/exclusion_summary.csv              days and series per set, action, reason
  eval/swotqc/audit/exclusion_examples_reservoirs.png, exclusion_examples_rivers.png
      12 randomly chosen series with removed days each (fixed seed), 20 days either side

Usage
  python build_gauge_exclusions.py     (no arguments; run gauge_stat_screen.py first)

Method
  Every flagged day gets the action REMOVE (dropped from the clean copies) or REVIEW (listed in
  the exclusion table, kept in the data).
  REMOVE
    agency_cwms   USACE CWMS: at least 50% of the day's raw values have a validity of
                  questionable, rejected or missing (quality code & 30 in {4, 8, 16})
    agency_eqp    USGS qualifier Eqp (equipment malfunction) on a reservoir series
    spike         reservoir series only, from gauge_stat_screen.py
    negative      from gauge_stat_screen.py
  REVIEW
    river spike, rate, over_cap, flat_run, step   from gauge_stat_screen.py
    quantized_series   series-level flag from gauge_stat_screen.py (no date)
    estimated          USGS qualifier e
    cdec_flag_<x>      any California CDEC data flag
    cwms_test_flag     at least 50% of the day's CWMS values are valid but failed one of the
                       agency's screening tests (quality code >= 32768)
  Why river spikes and `rate` are not removed: random examples of each flag were inspected.
  One- or two-day excursions in reservoir storage are sensor errors. River "spikes" are mostly
  real (hydropower release pulses, storm peaks, the December 2015 White River flood), and
  approved USGS river data has already been reviewed by a hydrographer. The rate rule caught
  real flood inflows at small dams.

Notes
  Agency flags can only be read for the 2023-2025 reservoir set, because only its raw responses
  are in the cache. A series whose cached response is missing gets no agency flags.
  River days with the USGS qualifiers Ice or Eqp were already dropped when the river gauge
  table was built.
"""
from swotgat.paths import WORK
import json, re, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork"); CA = F/"agency_cache"; TS = F/"testsets"; OUT = F/"eval/swotqc/audit"
tag = lambda t: CA/(re.sub(r"[^A-Za-z0-9_.-]", "_", t)[:180] + ".txt")
T1 = pd.read_parquet(TS/"regime1_2023_2025/truth.parquet"); T1["date"] = pd.to_datetime(T1.date)
T2 = pd.read_parquet(TS/"regime2_2015_2020/truth.parquet"); T2["date"] = pd.to_datetime(T2.date)
G = pd.read_parquet(TS/"rivers/gauge_daily.parquet"); G["date"] = pd.to_datetime(G.date)
rows = []
# ---- agency flags (2023-2025 reservoir set only: the only gauge table whose raw flags are in the cache)
for _, s in T1.drop_duplicates("DAM_ID").iterrows():
    ag, sid = str(s.source).split(":")[0], s.sid
    if not isinstance(sid, str): continue
    if ag in ("CWMS", "CWMSSUM"):
        vals = []
        for part in sid.split("+"):
            off, ts = part.split("|", 1) if "|" in part else (sid.split("|")[0], part)
            f = tag(f"cwms_ts_{off}_{ts}")
            if f.is_file() and f.stat().st_size: vals += json.loads(f.read_text()).get("values", [])
        if not vals: continue
        d = pd.DataFrame(vals, columns=["t", "v", "q"]); d["day"] = pd.to_datetime(d.t, unit="ms").dt.normalize()
        d["bad"] = (d.q.astype("int64") & 30).isin([4, 8, 16]); d["test"] = (d.q.astype("int64") >= 32768) & ~d.bad
        g = d.groupby("day").agg(bad=("bad", "mean"), test=("test", "mean"))
        rows += [("res_2023_25", str(s.DAM_ID), t, "agency_cwms", "REMOVE") for t in g.index[g.bad >= 0.5]]
        rows += [("res_2023_25", str(s.DAM_ID), t, "cwms_test_flag", "REVIEW") for t in g.index[(g.test >= 0.5) & (g.bad < 0.5)]]
    elif ag in ("NWIS", "NWISANY"):
        p = sid.split("|"); f = tag(f"nwis_{'dv' if len(p) > 2 and p[2] == 'dv' else 'iv'}_{p[0]}_{p[1]}")
        if not (f.is_file() and f.stat().st_size): continue
        for ts in json.loads(f.read_text())["value"]["timeSeries"]:
            for blk in ts["values"]:
                for v in blk["value"]:
                    day = pd.Timestamp(v["dateTime"]).tz_localize(None).normalize(); q = v.get("qualifiers", [])
                    if "Eqp" in q: rows.append(("res_2023_25", str(s.DAM_ID), day, "agency_eqp", "REMOVE"))
                    elif "e" in q: rows.append(("res_2023_25", str(s.DAM_ID), day, "estimated", "REVIEW"))
    elif ag == "CDEC":
        f = tag(f"cdec_{sid}")
        if f.is_file():
            d = pd.read_csv(f, dtype=str)
            for dt, fl in zip(d["DATE TIME"], d.DATA_FLAG.fillna("").str.strip()):
                if fl: rows.append(("res_2023_25", str(s.DAM_ID), pd.to_datetime(dt, format="%Y%m%d %H%M").normalize(), f"cdec_flag_{fl}", "REVIEW"))
# ---- statistical screen (flags from gauge_stat_screen.py; `act` gives the action per flag, and a
#      river spike is always REVIEW)
st = pd.read_parquet(OUT/"gauge_stat_flags.parquet")
act = {"spike": "REMOVE", "rate": "REVIEW", "negative": "REMOVE", "over_cap": "REVIEW", "flat_run": "REVIEW", "step": "REVIEW"}
rows += [(r.set, str(r.id), pd.Timestamp(r.date), r.flag, "REVIEW" if (r.flag == "spike" and r.set.startswith("river")) else act[r.flag])
         for r in st.itertuples()]
X = pd.DataFrame(rows, columns=["set", "id", "date", "reason", "action"]).drop_duplicates()
sm = pd.read_csv(OUT/"gauge_stat_summary.csv", dtype={"id": str})
Q = sm[sm.quantized.fillna(False).astype(bool)][["set", "id"]].assign(date=pd.NaT, reason="quantized_series", action="REVIEW")
X = pd.concat([X, Q], ignore_index=True); X.to_parquet(TS/"gauge_exclusions_v1.parquet", index=False)

def clean(df, key, idcol, setname, datecol="date"):
    """Drop the REMOVE days of one set from a gauge table.

    `df` is the table, `idcol` its series id column, `datecol` its date column and `setname` the
    set whose exclusions apply; `key` is not used. Returns (cleaned table, number of rows dropped).
    """
    rm = X[(X.set == setname) & (X.action == "REMOVE")][["id", "date"]].drop_duplicates().assign(rm=True)
    k = df.assign(_id=df[idcol].astype(str)).merge(rm, left_on=["_id", datecol], right_on=["id", "date"], how="left", suffixes=("", "_x"))
    keep = k.rm.isna().values
    return df[keep], int((~keep).sum())
c1, n1 = clean(T1, "DAM_ID", "DAM_ID", "res_2023_25"); c1.to_parquet(TS/"regime1_2023_2025/truth_clean_v1.parquet", index=False)
c2, n2 = clean(T2, "DAM_ID", "DAM_ID", "res_2015_20"); c2.to_parquet(TS/"regime2_2015_2020/truth_clean_v1.parquet", index=False)
Gs = G.assign(set=np.where(G.date.dt.year < 2022, "river_2015_20", "river_2023_25"))
nr = 0; parts = []
for sname, g in Gs.groupby("set"):
    c, n = clean(g.drop(columns="set"), "site_no", "site_no", sname); parts.append(c); nr += n
pd.concat(parts).to_parquet(TS/"rivers/gauge_daily_clean_v1.parquet", index=False)
S = X.groupby(["set", "action", "reason"]).agg(days=("date", "count"), series=("id", "nunique")).reset_index(); S.to_csv(OUT/"exclusion_summary.csv", index=False)
print(S.to_string(index=False))
print(f"\nrows removed: regime-1 truth {n1:,} of {len(T1):,} | regime-2 {n2:,} of {len(T2):,} | rivers {nr:,} of {len(G):,}")

# ---- example pages of REMOVE decisions (random, seed 0): 12 series per page, 20 days either side of a removed day
rng = np.random.default_rng(0)
series = {"res_2023_25": (T1, "DAM_ID", "storage_mcm", "MCM"), "res_2015_20": (T2, "DAM_ID", "storage_mcm", "MCM"),
          "river_2015_20": (G, "site_no", "stage_m", "m"), "river_2023_25": (G, "site_no", "stage_m", "m")}
for page, pick in enumerate([("res_2023_25", "res_2015_20"), ("river_2023_25", "river_2015_20")]):
    R = X[X.set.isin(pick) & (X.action == "REMOVE")].dropna(subset=["date"])
    R = R.drop_duplicates(["set", "id"]); R = R.iloc[rng.permutation(len(R))[:12]]
    fig, axs = plt.subplots(3, 4, figsize=(16, 9)); axs = axs.ravel()
    for ax, r in zip(axs, R.itertuples()):
        df, idc, vc, u = series[r.set]
        s = df[df[idc].astype(str) == r.id].set_index("date")[vc].sort_index()
        w = s[(s.index >= r.date - pd.Timedelta(days=20)) & (s.index <= r.date + pd.Timedelta(days=20))]
        flagged = X[(X.set == r.set) & (X.id == r.id) & (X.action == "REMOVE")].date
        ax.plot(w.index, w.values, "k.-", lw=0.8, ms=3)
        f = w[w.index.isin(flagged)]; ax.scatter(f.index, f.values, color="#d62728", s=30, zorder=3, label="removed")
        ax.set_title(f"{r.set} {r.id}: {r.reason}", fontsize=8); ax.tick_params(labelsize=6); ax.set_ylabel(u, fontsize=7)
    for ax in axs[len(R):]: ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT/f"exclusion_examples_{['reservoirs', 'rivers'][page]}.png", dpi=110); plt.close(fig)
print("example pages written")
