"""Build the two reservoir test sets from the crosswalk, with a confidence tier per site.

Step 7, the last step of the crosswalk. The folder names call the test sets "regimes":
  regime2_2015_2020   ResOpsUS storage, 2015-2020
  regime1_2023_2025   agency gauge storage, 2023-2025
A site is one dam with its graph node(s). Its reference series ("truth") is gauge storage in million
cubic meters (MCM). The test sets are never written into the training data store; models are scored
offline against these tables from their prediction output. The tier rules below were fixed before any
model was scored.

Reads (under $SWOTGAT_WORK/CONUS_storage_fork unless noted)
  crosswalk/dam_node_crosswalk.csv        link L2, dam to node: High / Medium / Low / Reject / Collapsed
  crosswalk/crosswalk_overrides.csv       review rulings on L2 (step 6c), if the file exists
  crosswalk/resops_record_verdict.csv     link L1, ResOpsUS record: A / B / C / X, and the unit fix factor
  crosswalk/gauge_final.csv               link L4 evidence, gauge to dam (steps 5-5c)
  crosswalk/audit_summary.json            hashed into the version file only
  reservoir_truth_2023_2025_v2.csv, reservoir_truth_summary.csv, resops_targets_full.csv,
  harvest_log*.csv                        the 2023-2025 gauge series and how each was found
  ResOpsUS/time_series_all/ResOpsUS_<DAM_ID>.csv
  $SWOTGAT_WORK/CONUS/basin_lists/storage/train_nodes_g.csv    training nodes (flag supervised_2023_25)

Writes (under $SWOTGAT_WORK/CONUS_storage_fork/testsets)
  regime2_2015_2020/{sites.csv, truth.parquet, predict_nodes.csv}
  regime1_2023_2025/{sites.csv, truth.parquet, predict_nodes.csv, excluded.csv}
  crosswalk_version.json                  build time, L4 thresholds, sha256 of the inputs, tier counts
  predict_nodes.csv lists every member node of every site with its basin: the nodes a model must predict.

Usage
  python step7_build_testsets.py        (no arguments)

Method
  Site tier = the weakest relevant link: min(L2, L1) for 2015-2020 and min(L2, L4) for 2023-2025, with
  L1 grades A, B, C, X read as High, Medium, Low, Reject. Review overrides replace the automatic L2
  tier and are logged in the conflicts column.

  2015-2020: every dam with L2 of at least Low and L1 of at least C. The series is ResOpsUS storage
  times the fix factor, cleaned (see clean), and needs at least 365 days. anomaly_mcm is storage minus
  its median. low_signal marks a standard deviation below 2% of the median.

  2023-2025: every reservoir of the 2023-2025 gauge table that has a ResOpsUS DAM_ID. Source, method,
  number of days and resolution are recomputed from the final table, because the tournament (step 5b)
  may have swapped the series. Sites with tier Reject go to excluded.csv.

  L4 tier (function l4; thresholds in TH4):
    With an overlap test (step 5 status ok and at least 180 overlap days):
      r >= 0.95        High if the amplitude ratio is in 0.8-1.25, else Medium
      0.80 <= r < 0.95 Medium if the amplitude ratio is in band, else Low
      r < 0.80         Reject if the ResOpsUS record is grade A or B; Low if the record itself is
                       doubtful (grade C or X), because then the reference cannot disprove the gauge
    Without an overlap test, from the harvest log: Medium if the gauge is within 1 km of the dam, the
      name similarity is at least 0.8 and the ratio recorded in the method string is in 0.67-1.5;
      else Low.
    Caps on top of that: a series labelled averaged is at most Low; a monthly series at most Medium;
      a conservation-pool series pinned at its maximum on more than 10% of days at most Medium; a
      series that holds one value on more than 50% of its days at most Low, on more than 30% at most
      Medium.

Notes
  The script asserts that DAM_ID and primary node are unique within each test set and that node ids
  are strings.
"""
from swotgat.paths import WORK
import hashlib, json, re, numpy as np, pandas as pd
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork"); C = Path(f"{WORK}/CONUS")
O = F/"crosswalk"; T = F/"testsets"; T.mkdir(exist_ok=True)
RANK = {"High": 3, "Medium": 2, "Low": 1, "Reject": 0, "Collapsed": 0}
L1R = {"A": 3, "B": 2, "C": 1, "X": 0}; INV = {3: "High", 2: "Medium", 1: "Low", 0: "Reject"}
TH4 = dict(pin_med=0.3, pin_low=0.5, r_hi=0.95, r_lo=0.80, amp_lo=0.8, amp_hi=1.25, min_ovl=180, pinned=0.10, dist=1.0, name=0.8, ratio_lo=0.67, ratio_hi=1.5)

XW = pd.read_csv(O/"dam_node_crosswalk.csv", dtype={"node": str})
L1 = pd.read_csv(O/"resops_record_verdict.csv").set_index("DAM_ID")
TR = set(pd.read_csv(C/"basin_lists/storage/train_nodes_g.csv", dtype=str).subbasin)
# review overrides (crosswalk_overrides.csv) replace the automatic L2 tier and are logged in the
# conflicts column
OV = O/"crosswalk_overrides.csv"
if OV.is_file():
    for _, o in pd.read_csv(OV).iterrows():
        k = XW.DAM_ID == o.DAM_ID
        if o.link == "L2" and k.any():
            XW.loc[k, "conflicts"] = XW.loc[k, "conflicts"].fillna("") + f" | OVERRIDE {o.new_tier}: {o.reason}"
            XW.loc[k, "tier"] = o.new_tier
XW["L2"] = XW.tier.map(RANK).fillna(0).astype(int)
XW["L1_grade"] = XW.DAM_ID.map(L1.L1_grade).fillna("X"); XW["L1"] = XW.L1_grade.map(L1R)
XW["supervised_2023_25"] = XW.node.isin(TR)

def clean(s):
    """Remove dropouts and isolated spikes from a daily storage series and return what is left.

    A dropout is an exact zero in a series whose median exceeds 1. A spike is a day more than 10 robust
    scales from the 5-day centered median, where the scale is 1.4826 x the median absolute day-to-day
    change plus 0.1% of the range.
    """
    s = s.astype(float).copy(); med = s.median()
    if med > 1: s[s == 0] = np.nan
    m = s.rolling(5, center=True, min_periods=3).median()
    scale = 1.4826*np.nanmedian(np.abs(np.diff(s.dropna().values))) + 1e-3*(s.max() - s.min())
    s[(s - m).abs() > 10*scale] = np.nan
    return s.dropna()

def node_list(sites):
    """Return a table (basin, subbasin) with every member node of every site, without repeats.

    sites needs the columns basin and members (node ids joined by ";").
    """
    rows = []
    for _, r in sites.iterrows():
        for m in str(r.members).split(";"):
            rows.append((r.basin, m))
    return pd.DataFrame(rows, columns=["basin", "subbasin"]).drop_duplicates()

# ================= Regime 2: 2015-2020 ResOps backtest =================
R2 = XW[(XW.L2 >= 1) & (XW.L1 >= 1)].copy()
R2["site_tier"] = np.minimum(R2.L2, R2.L1).map(INV)
truth2 = []
for _, r in R2.iterrows():
    t = pd.read_csv(F/f"ResOpsUS/time_series_all/ResOpsUS_{r.DAM_ID}.csv"); t["date"] = pd.to_datetime(t.date, errors="coerce")
    s = pd.to_numeric(t.dropna(subset=["date"]).drop_duplicates("date").set_index("date").storage, errors="coerce").sort_index().loc["2015":"2020"]
    s = clean(s.dropna()*L1.fix_factor.get(r.DAM_ID, 1.0))
    if len(s) < 365: continue
    truth2.append(pd.DataFrame(dict(DAM_ID=r.DAM_ID, date=s.index, storage_mcm=s.values, anomaly_mcm=(s - s.median()).values)))
truth2 = pd.concat(truth2, ignore_index=True)
R2 = R2[R2.DAM_ID.isin(truth2.DAM_ID)]
sd = truth2.groupby("DAM_ID").storage_mcm.std(); md = truth2.groupby("DAM_ID").storage_mcm.median()
R2["low_signal"] = R2.DAM_ID.map(sd < 0.02*md)
R2["n_days"] = R2.DAM_ID.map(truth2.groupby("DAM_ID").size())
d2 = T/"regime2_2015_2020"; d2.mkdir(exist_ok=True)
cols = ["DAM_ID", "DAM_NAME", "STATE", "node", "members", "basin", "site_tier", "tier", "L1_grade", "supervised_2023_25",
        "low_signal", "n_days", "primary_share", "area_ratio", "weaknesses", "conflicts"]
R2[cols].rename(columns={"tier": "L2_tier"}).to_csv(d2/"sites.csv", index=False)
truth2.to_parquet(d2/"truth.parquet", index=False)
node_list(R2).to_csv(d2/"predict_nodes.csv", index=False)

# ================= Regime 1: 2023-2025 agency gauges =================
S = pd.read_csv(F/"reservoir_truth_summary.csv", dtype={"node": str, "node_orig": str})
TGT = pd.read_csv(F/"resops_targets_full.csv", dtype={"node": str}).drop_duplicates("node").set_index("node")
S["DAM_ID"] = S.node_orig.map(TGT.DAM_ID)
GO = pd.read_csv(O/"gauge_final.csv").drop_duplicates("DAM_ID").set_index("DAM_ID")
# per-dam fields recomputed from the final 2023-2025 table (tournament swaps change source, method and
# resolution)
TT = pd.read_csv(F/"reservoir_truth_2023_2025_v2.csv", dtype={"node_orig": str})
TTg = TT.groupby("node_orig")
fin = pd.DataFrame(dict(source=TTg.source.first(), method=TTg.method.first(), n_days=TTg.size(),
                        sd=TTg.storage_mcm.std(), lvl=TTg.storage_mcm.apply(lambda v: v.abs().median()),
                        anom=TTg.is_anomaly.first()))
fin["resolution"] = fin.source.str.startswith("NRCSM").map({True: "monthly", False: "daily"})
fin["low_signal"] = (~fin.anom.astype(bool)) & (fin.sd < 0.02*fin.lvl)
# pinned/stuck series: a real gauge never repeats one value for months (conservation-pool caps, rating-curve floors)
def _pin(v):
    """Return mode_frac and longest_flat of a storage series, after rounding to 0.001 MCM.

    mode_frac is the share of days at the most frequent value and longest_flat the length of the longest
    run of days without a change.
    """
    v = v.round(3); runs = (v.diff() != 0).cumsum()
    return pd.Series(dict(mode_frac=v.value_counts(normalize=True).iloc[0], longest_flat=runs.value_counts().max()))
fin = fin.join(TT.sort_values("date").groupby("node_orig").storage_mcm.apply(_pin).unstack())
S = S.drop(columns=[c for c in ("source", "method", "n_days", "resolution", "low_signal") if c in S]).join(fin, on="node_orig")
logs = pd.concat([pd.read_csv(f, dtype={"node": str}) for f in sorted(F.glob("harvest_log*.csv"))], ignore_index=True)
logs = logs[logs.ok == True].drop_duplicates("node", keep="last").set_index("node")

def l4(r):
    """Return (L4_tier, L4_evidence, L4_flags) for one 2023-2025 site.

    The rules are those of the module docstring, with the thresholds of TH4.
    r is a row of the site table (DAM_ID, node_orig, resolution, mode_frac, method). The base tier comes
    from the 2015-2020 overlap test of the dam's gauge series (table GO) when it has at least 180 overlap
    days, and otherwise from the distance, name similarity and ratio recorded when the gauge was found.
    The caps for averaged, monthly and pinned series then limit it. The evidence text states the scores
    used, and the flags name every cap that applies.
    """
    g = GO.loc[r.DAM_ID] if r.DAM_ID in GO.index else None
    flags, cap = [], 3
    cls = str(g.cls) if g is not None and isinstance(g.cls, str) else ""
    if "averaged" in cls: flags.append(cls); cap = 1
    if "monthly" in cls or str(r.resolution) == "monthly": flags.append("monthly"); cap = min(cap, 2)
    if "conservation-pool" in cls and g is not None and g.top_pinned_frac > TH4["pinned"]: flags.append(f"conservation pool pinned at top {g.top_pinned_frac:.0%} of days"); cap = min(cap, 2)
    if r.mode_frac > 0.5: flags.append(f"series pinned at one value {r.mode_frac:.0%} of days"); cap = min(cap, 1)
    elif r.mode_frac > 0.3: flags.append(f"series pinned at one value {r.mode_frac:.0%} of days"); cap = min(cap, 2)
    ref_ok = L1.L1_grade.get(r.DAM_ID, "X") in ("A", "B")
    if g is not None and g.status == "ok" and g.ovl_n >= TH4["min_ovl"]:
        amp_ok = TH4["amp_lo"] <= g.amp_ratio <= TH4["amp_hi"]
        if g.ovl_r >= TH4["r_hi"]: base = 3 if amp_ok else 2
        elif g.ovl_r >= TH4["r_lo"]: base = 2 if amp_ok else 1
        else: base = 0 if ref_ok else 1
        how = f"overlap r={g.ovl_r:.3f} amp={g.amp_ratio:.2f} n={int(g.ovl_n)}"
        if not ref_ok and base < 2: base = max(base, 1); how += " (ResOps reference itself doubtful)"
    else:
        lg = logs.loc[r.node_orig] if r.node_orig in logs.index else None
        m = re.search(r"([\d.]+) km, name sim ([\d.]+)", str(lg.note)) if lg is not None else None
        rt = re.search(r"ratio=([\d.]+)", str(r.method))
        dist, ns = (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)
        ratio = float(rt.group(1)) if rt else np.nan
        good_meta = dist <= TH4["dist"] and ns >= TH4["name"] and TH4["ratio_lo"] <= ratio <= TH4["ratio_hi"]
        base = 2 if good_meta else 1
        how = f"no 2015-20 overlap ({g.status if g is not None else 'not tested'}); dist={dist} name={ns} ratio={ratio}"
    return INV[min(base, cap)], how, "; ".join(flags)
S = S[S.DAM_ID.notna()].copy(); S["DAM_ID"] = S.DAM_ID.astype(int)
V = S.apply(l4, axis=1, result_type="expand"); V.columns = ["L4_tier", "L4_evidence", "L4_flags"]
S = pd.concat([S, V], axis=1)
S = S.drop(columns=[c for c in ("basin", "member_nodes", "renoded", "node") if c in S])   # fields from the earlier renode.py matching; the crosswalk replaces them
R1 = S.merge(XW[["DAM_ID", "node", "members", "basin", "tier", "L2", "supervised_2023_25", "primary_share", "weaknesses", "conflicts"]]
             .rename(columns={"node": "node_xw"}), on="DAM_ID", how="left")
R1["L2"] = R1.L2.fillna(0).astype(int); R1["L4"] = R1.L4_tier.map(RANK)
R1["site_tier"] = np.minimum(R1.L2, R1.L4).map(INV)
excluded = R1[R1.site_tier == "Reject"]
R1 = R1[R1.site_tier != "Reject"]
TT = TT[TT.node_orig.isin(R1.node_orig)].merge(R1[["node_orig", "DAM_ID"]], on="node_orig")
TT = TT[["DAM_ID", "date", "storage_mcm", "is_anomaly", "source", "sid", "method"]]
d1 = T/"regime1_2023_2025"; d1.mkdir(exist_ok=True)
cols1 = ["DAM_ID", "DAM_NAME", "STATE", "node_xw", "members", "basin", "site_tier", "tier", "L4_tier", "L4_evidence", "L4_flags",
         "supervised_2023_25", "low_signal", "resolution", "source", "method", "n_days", "primary_share", "weaknesses", "conflicts"]
R1[cols1].rename(columns={"node_xw": "node", "tier": "L2_tier"}).to_csv(d1/"sites.csv", index=False)
excluded[["DAM_ID", "DAM_NAME", "tier", "L4_tier", "L4_evidence", "conflicts"]].to_csv(d1/"excluded.csv", index=False)
TT.to_parquet(d1/"truth.parquet", index=False)
node_list(R1.rename(columns={"node_xw": "node"})).to_csv(d1/"predict_nodes.csv", index=False)

# ================= version stamp + invariants =================
def h(p):
    """Return the first 16 hex digits of the SHA-256 hash of the file at path p."""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
inputs = [O/"crosswalk_overrides.csv", O/"audit_summary.json", O/"dam_node_crosswalk.csv", O/"resops_record_verdict.csv", O/"gauge_final.csv", F/"reservoir_truth_2023_2025_v2.csv"]
for d, sites in ((d2, R2), (d1, R1)):
    s = pd.read_csv(d/"sites.csv", dtype={"node": str})
    assert s.DAM_ID.is_unique and s.node.map(type).eq(str).all() and not s.node.duplicated().any(), d
summary = dict(built=pd.Timestamp.now().isoformat(), thresholds_L4=TH4, inputs={p.name: h(p) for p in inputs},
               regime2=R2.site_tier.value_counts().to_dict(), regime1=R1.site_tier.value_counts().to_dict(),
               regime1_excluded=int(len(excluded)))
json.dump(summary, open(T/"crosswalk_version.json", "w"), indent=1, default=str)
print(json.dumps(summary, indent=1, default=str))
print("\nregime 2 by supervision:", R2.groupby(["supervised_2023_25", "site_tier"]).size().to_dict())
print("regime 1 by supervision:", R1.groupby(["supervised_2023_25", "site_tier"]).size().to_dict())
print("\nregime 1 excluded:"); print(excluded[["DAM_NAME", "tier", "L4_tier", "L4_evidence", "conflicts"]].to_string(max_colwidth=80))
