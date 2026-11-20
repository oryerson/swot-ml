"""Merge all 2023-2025 agency downloads into one table of daily gauge storage per reservoir.

Reads ($SWOTGAT_WORK/CONUS_storage_fork)
  harvested_manual.csv                              manual_override.py
  harvested_pass5.csv ... harvested_pass2.csv, harvested.csv   runs of harvest.py / harvest_v2.py
  agency_storage_2023_2025_union.csv                an earlier download of agency storage
  resops_targets_full.csv                           ResOpsUS dams with their graph node
  $SWOTGAT_WORK/CONUS/attributes/static_conus_storage.parquet   capacity per node

Writes
  reservoir_truth_2023_2025.csv   node, date, storage_mcm, is_anomaly, source, sid, method, pass
  reservoir_truth_summary.csv     one row per reservoir: days, first and last date, source,
                                  method, values removed as spikes, dam name, state, agency,
                                  resolution (daily | monthly), low_signal

Usage
  python merge_truth.py     (no arguments)

Method
  * Priority. The files are read in the order listed above and each node is taken from the
    first file that has it. `REJECT` removes four dams from the files in which the automatic
    match picked a neighbouring reservoir (Dexter, Hopkinton, Kentucky, Dry Falls); their
    series come from harvested_manual.csv or not at all.
  * Only 2023-01-01 to 2025-12-31 is kept.
  * Cleaning per node:
    - For absolute storage (not for anomaly series, whose method contains "static PLD"):
      exact zeros are removed when the median exceeds 1 MCM (sensor dropouts), and negative
      values smaller in size than 2% of the median (of 1 MCM if the median is below 1) are set
      to 0 (sensor noise at an empty pool).
    - Isolated spikes are removed: a value that differs from the 5-day centered median by more
      than 10 times a robust scale. The scale is 1.4826 times the median absolute day-to-day
      change plus 0.1% of the series range. On a steady rise or fall the centered median equals
      the value itself, so flood limbs are kept; only excursions of one or two days that come
      back are removed.
  * Dropped series: standard deviation below 0.05 MCM (no measurable signal); daily series
    with fewer than 180 values; monthly series with fewer than 24 values.
  * `low_signal` marks kept absolute series whose standard deviation is below 2% of their own
    median storage (constant-level and run-of-river reservoirs), so that an evaluation can
    treat them separately. The reference is the reservoir's own typical storage and not its
    capacity, because capacity can include a flood pool.

Notes
  This table is test data only. It is never written into the training target (the SWOT storage
  change variable dv_SWOTAE).
  `is_anomaly` marks series converted from water level with a fixed lake area. They are storage
  relative to an arbitrary level, not absolute storage.
"""
from swotgat.paths import WORK
import pandas as pd, numpy as np
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork")
C = Path(f"{WORK}/CONUS")
REJECT = {  # matches found to be a neighbouring reservoir; replaced by manual pins or dropped
    "harvested_pass5.csv": {"Dexter", "Hopkinton Flood Control Dam"},   # had matched Fall Creek / Everett
    "harvested.csv": {"Kentucky", "Dry Falls"},                          # had matched Barkley / Grand Coulee
}
order = ["harvested_manual.csv", "harvested_pass5.csv", "harvested_pass4.csv", "harvested_pass3.csv",
         "harvested_pass2.csv", "harvested.csv", "agency_storage_2023_2025_union.csv"]
t = pd.read_csv(F/"resops_targets_full.csv", dtype={"node": str}).drop_duplicates("node")
name2node = t.set_index("DAM_NAME").node
cap = pd.read_parquet(C/"attributes/static_conus_storage.parquet", columns=["dam_cap_mcm"]).reset_index()
cap["subbasin"] = cap.subbasin.astype(str); cap = cap.drop_duplicates("subbasin").set_index("subbasin").dam_cap_mcm

frames, taken = [], set()
for f in order:
    h = pd.read_csv(F/f, dtype={"node": str})
    for c, v in [("sid", ""), ("method", "legacy scraper (gated)")]:
        if c not in h: h[c] = v
    bad = {name2node[n] for n in REJECT.get(f, ()) if n in name2node}
    h = h[~h.node.isin(bad | taken)]
    h["pass"] = f.replace(".csv", "")
    taken |= set(h.node); frames.append(h)
    print(f"{f:38s} +{h.node.nunique():3d} nodes")
T = pd.concat(frames, ignore_index=True)
T["date"] = pd.to_datetime(T.date)
T = T[(T.date >= "2023-01-01") & (T.date <= "2025-12-31")]

def clean(g):
    """Clean the series of one node and return it without the removed values.

    `g` holds the rows of one node. Dates are sorted and made unique, sensor dropouts and
    isolated spikes are removed (rules in the module docstring), and the columns is_anomaly and
    n_despiked are added.
    """
    g = g.sort_values("date").drop_duplicates("date")
    s = g.storage_mcm.astype(float).values.copy()
    anomaly = g.method.iat[0].startswith("static PLD") or "static PLD" in g.method.iat[0]
    if not anomaly:
        med = np.nanmedian(s)
        if med > 1: s[s == 0] = np.nan                     # zero dropouts (example: Bucks)
        s = np.where((s < 0) & (s > -0.02*max(med, 1)), 0, s)  # sensor noise just below empty (example: Paonia)
    # Isolated spikes only: a 5-point centred median equals the value on any monotonic
    # rise/fall, so flood limbs survive; only 1-2 day excursions that come back are removed.
    x = pd.Series(s)
    m = x.rolling(5, center=True, min_periods=3).median()
    scale = 1.4826*np.nanmedian(np.abs(np.diff(s))) + 1e-3*(np.nanmax(s) - np.nanmin(s))
    spike = (x - m).abs() > 10*scale
    s[spike.values] = np.nan
    g = g.assign(storage_mcm=s, is_anomaly=anomaly, n_despiked=int(spike.sum()))
    return g.dropna(subset=["storage_mcm"])
T = T.groupby("node", group_keys=False).apply(clean)
sd = T.groupby("node").storage_mcm.std(); k = cap.reindex(sd.index)
# low_signal is judged against the reservoir's own typical storage, not its GDW (Global Dam
# Watch) capacity, which can include flood pools
lvl = T.groupby("node").storage_mcm.apply(lambda v: np.nanmedian(np.abs(v)))
anom = T.groupby("node").is_anomaly.first()
# Drop only series with no measurable signal. Constant-level lakes (Lake Austin, Lake LBJ) are
# kept and flagged low_signal, so that an evaluation can treat them separately instead of
# losing them without notice.
flat = sd[sd < 0.05].index
low_signal = set(sd[~anom.reindex(sd.index) & (sd < 0.02*lvl.reindex(sd.index))].index)
monthly = T.groupby("node").method.first().str.contains("monthly")
n_obs = T.groupby("node").size(); short = n_obs[(n_obs < 180) & ~monthly.reindex(n_obs.index)].index
short = short.union(n_obs[monthly.reindex(n_obs.index) & (n_obs < 24)].index)
print(f"\ndropped flat: {[t.set_index('node').DAM_NAME.get(n) for n in flat]}")
print(f"dropped <180 days: {[t.set_index('node').DAM_NAME.get(n) for n in short]}")
T = T[~T.node.isin(flat.union(short))]
T["date"] = T.date.dt.strftime("%Y-%m-%d")
cols = ["node", "date", "storage_mcm", "is_anomaly", "source", "sid", "method", "pass"]
T[cols].to_csv(F/"reservoir_truth_2023_2025.csv", index=False)
S = T.groupby("node").agg(n_days=("date", "size"), first=("date", "min"), last=("date", "max"),
                          source=("source", "first"), method=("method", "first"), pass_=("pass", "first"),
                          despiked=("n_despiked", "first"))
S = S.join(t.set_index("node")[["DAM_NAME", "STATE", "AGENCY_CODE"]])
S["resolution"] = np.where(S.method.str.contains("monthly"), "monthly", "daily")
S["low_signal"] = S.index.isin(low_signal)
S.to_csv(F/"reservoir_truth_summary.csv")
print("low_signal (constant-level / run-of-river):", sorted(S[S.low_signal].DAM_NAME))
print(f"\nFINAL: {S.shape[0]} of {t.node.nunique()} ResOps-matched reservoirs with 2023-2025 in-situ storage, {len(T):,} daily values")
print("by agency family:", S.source.str.split(":").str[0].value_counts().to_dict())
print("absolute storage:", int((~T.drop_duplicates('node').is_anomaly).sum()), " elevation x static area (anomaly):", int(T.drop_duplicates('node').is_anomaly.sum()))
print("median days per reservoir:", int(S.n_days.median()), "  despiked values total:", int(S.despiked.sum()))
