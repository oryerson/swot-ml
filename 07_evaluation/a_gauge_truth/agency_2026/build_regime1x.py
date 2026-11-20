"""Build the SWOT Period reservoir test set: the 2023-2025 agency gauge set extended to 2026-09-30.

"Regime 1" is the frozen test set of agency gauge storage for 2023-2025
(testsets/regime1_2023_2025). "Regime 1x" (testsets/regime1x_2023_2026) is the same set with the
2026 days from harvest_2026.py appended. It is the reservoir gauge set for the SWOT Period.
Regime 1 is only read.

Reads ($SWOTGAT_WORK/CONUS_storage_fork)
  testsets/regime1_2023_2025/sites.csv, truth.parquet, truth_clean_v1.parquet, predict_nodes.csv
  gauges2026/harvest_2026.csv, ext_2026.parquet            output of harvest_2026.py
  gauges2026/harvest_2026_x23.csv, ext_2026_x23.parquet    its second run with a 2023-2025
                                                           overlap window (optional)

Writes (testsets/regime1x_2023_2026)
  truth.parquet            every regime-1 row unchanged, plus the 2026 rows. `method` says how
                           each 2026 series was mapped onto the 2023-2025 values.
  truth_clean_v1x.parquet  the rows of truth_clean_v1 unchanged, plus the 2026 rows that pass
                           the automated rules below. Later steps read this cleaned table.
  ext_2026_detail.parquet  all 2026 rows with their flags and the reason for removal
  sites.csv                the regime-1 sites plus extension columns (ext_status, ext_grade,
                           ext_source, ext_sid, n_2026, last_2026, overlap statistics, n_extrap,
                           n_2026_clean, ...)
  predict_nodes.csv        copied from regime 1
  version.json, README.md  counts, input checksums and the acceptance rule

Usage
  python build_regime1x.py     (no arguments)

Method
  * A dam that was extended only in the _x23 run takes its rows from that run.
  * The 2026 rows are flagged for removal by the same automated rules as the 2023-2025 set
    (merge_truth.py and build_gauge_exclusions.py):
    - zero dropout: an exact zero in an absolute storage series whose median exceeds 1 MCM;
    - negative values smaller in size than 2% of the median are set to 0 (not removed);
    - isolated spike: a value that differs from the 5-day centered median by more than 10
      times a robust scale (1.4826 times the median absolute day-to-day change plus 0.1% of
      the range). The dam's values from 2025-11-01 on are put in front of the 2026 values so
      that the scale is stable;
    - CWMS validity: at least 50% of the day's raw values are marked questionable, rejected or
      missing by the agency;
    - USGS Eqp: the day carries the qualifier for equipment malfunction.
  * Days whose level lies outside the 2024-2025 levels used to fit the storage-level curve are
    kept and marked "(extrapolated)" in `method`. Days that fill a gap of the primary series
    from another series are marked "(gap fill)".
  * Sites, confidence tiers and member nodes are not derived again. The extension only appends
    days to each dam's own validated series.
  * The script stops with an error if any 2023-2025 row of truth.parquet differs from regime 1.

Notes
  This is test data only. It is never written into a training target.
"""
from swotgat.paths import WORK
import json, hashlib, time, numpy as np, pandas as pd
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork"); R1 = F/"testsets/regime1_2023_2025"; G = F/"gauges2026"
O = F/"testsets/regime1x_2023_2026"; O.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
S = pd.read_csv(R1/"sites.csv"); T = pd.read_parquet(R1/"truth.parquet"); TC = pd.read_parquet(R1/"truth_clean_v1.parquet")
H = pd.read_csv(G/"harvest_2026.csv"); E = pd.read_parquet(G/"ext_2026.parquet")
if (G/"harvest_2026_x23.csv").exists():                      # dams whose gauge record ends before mid-2024: second run with a 2023-2025 overlap window
    HX = pd.read_csv(G/"harvest_2026_x23.csv"); hx = HX[HX.status == "extended"].DAM_ID
    hx = hx[~hx.isin(H[H.status == "extended"].DAM_ID)]
    H = pd.concat([H[~H.DAM_ID.isin(hx)], HX[HX.DAM_ID.isin(hx)]], ignore_index=True)
    if (G/"ext_2026_x23.parquet").exists(): EX = pd.read_parquet(G/"ext_2026_x23.parquet"); E = pd.concat([E, EX[EX.DAM_ID.isin(hx)]], ignore_index=True)
E["date"] = pd.to_datetime(E.date)
T["date"] = pd.to_datetime(T.date); TC["date"] = pd.to_datetime(TC.date)
anom = T.groupby("DAM_ID").is_anomaly.first()

def clean26(g, ctx):
    """Flag one dam's 2026 rows with the automated rules; the reason goes to the column `remove`.

    ctx holds the dam's storage values from 2025-11-01 to the end of 2025. They are used only to
    stabilize the median and the spike scale and are not returned.
    """
    g = g.sort_values("date").copy(); s = g.storage_mcm.astype(float).values.copy(); why = np.array([""]*len(g), dtype=object)
    if not bool(anom.get(g.DAM_ID.iat[0], False)):
        med = np.nanmedian(np.r_[ctx, s])
        z = (s == 0) & (med > 1); why[z] = "zero dropout"
        neg = (s < 0) & (s > -0.02*max(med, 1)); s[neg] = 0.0
    x = pd.Series(np.r_[ctx, s]); m = x.rolling(5, center=True, min_periods=3).median()
    v = x.values; scale = 1.4826*np.nanmedian(np.abs(np.diff(v[np.isfinite(v)]))) + 1e-3*(np.nanmax(v) - np.nanmin(v))
    spike = ((x - m).abs() > 10*scale).values[len(ctx):]
    why[spike & (why == "")] = "isolated spike"
    bad = (g.bad_frac.fillna(0).values >= 0.5) & (why == "")
    why[bad] = np.where(g.flag.astype(str).str.contains("Eqp").values[bad], "USGS Eqp", "CWMS validity")
    return g.assign(storage_mcm=s, remove=why)

rows = []
for d, g in E.groupby("DAM_ID"):
    ctx = T[(T.DAM_ID == d) & (T.date >= "2025-11-01")].sort_values("date").storage_mcm.values
    rows.append(clean26(g, ctx))
E2 = pd.concat(rows, ignore_index=True)
H1 = H.set_index("DAM_ID")
E2["is_anomaly"] = E2.DAM_ID.map(anom).fillna(False).astype(bool)
E2["method"] = "2026 extension: " + E2["map"].astype(str) + "; " + E2["ovl"].astype(str) + np.where(E2.sid != E2.DAM_ID.map(H1.sid), " (gap fill)", "")
E2["method"] = np.where(E2.extrapolated, E2.method + "; level outside fitted range (extrapolated)", E2.method)
cols = ["DAM_ID", "date", "storage_mcm", "is_anomaly", "source", "sid", "method"]
new_raw = E2[cols]; new_clean = E2[E2.remove == ""][cols]
TX = pd.concat([T[cols], new_raw], ignore_index=True).sort_values(["DAM_ID", "date"])
TCX = pd.concat([TC[cols], new_clean], ignore_index=True).sort_values(["DAM_ID", "date"])
assert TX[TX.date <= "2025-12-31"].reset_index(drop=True).equals(T[cols].sort_values(["DAM_ID", "date"]).reset_index(drop=True)), "2023-25 rows changed"
TX.assign(date=TX.date.dt.strftime("%Y-%m-%d")).to_parquet(O/"truth.parquet", index=False)
TCX.assign(date=TCX.date.dt.strftime("%Y-%m-%d")).to_parquet(O/"truth_clean_v1x.parquet", index=False)
E2.to_parquet(O/"ext_2026_detail.parquet", index=False)
ext = H.rename(columns={"status": "ext_status", "source": "ext_source", "sid": "ext_sid", "role": "ext_role", "last": "last_2026", "tried": "ext_tried"})
keep = ["DAM_ID", "ext_status", "ext_grade", "ext_role", "ext_source", "ext_sid", "kind", "overlap_window", "n_2026", "n_primary", "n_fill", "fill", "last_2026",
        "ovl_n", "ovl_r", "rel_rmse", "ovl_rho", "rel_iso", "rel_loo", "lag", "n_extrap",
        "xc_source", "xc_sid", "xc_n", "xc_r", "xc_rel", "xc_bias", "ext_tried"]
SX = S.merge(ext[[c for c in keep if c in ext]], on="DAM_ID", how="left")
SX["n_2026_clean"] = SX.DAM_ID.map(new_clean.groupby("DAM_ID").size()).fillna(0).astype(int)
SX.to_csv(O/"sites.csv", index=False)
for f in ("predict_nodes.csv",):
    if (R1/f).exists(): (O/f).write_bytes((R1/f).read_bytes())
rem = E2[E2.remove != ""].remove.value_counts().to_dict()
v = dict(built=time.strftime("%Y-%m-%d %H:%M"), period=["2023-01-01", "2026-09-30"], base="regime1_2023_2025 (unchanged rows)",
         inputs={str(p.relative_to(F)): sha(p) for p in [R1/"sites.csv", R1/"truth.parquet", R1/"truth_clean_v1.parquet", G/"harvest_2026.csv", G/"ext_2026.parquet",
                                                  G/"harvest_2026_x23.csv", G/"ext_2026_x23.parquet", G/"harvest_2026.py"] if p.exists()},
         sites=int(len(SX)), extended=int((SX.ext_status == "extended").sum()),
         extended_high_medium=int(((SX.ext_status == "extended") & SX.site_tier.isin(["High", "Medium"])).sum()),
         rows_2026_raw=int(len(new_raw)), rows_2026_clean=int(len(new_clean)), removed_2026=rem,
         extrapolated_days=int(E2.extrapolated.sum()), gap_fill_days=int((E2.sid != E2.DAM_ID.map(H1.sid)).sum()),
         acceptance=("overlap 2024-25 vs the frozen truth (2023-25 when the truth ends before mid-2024), n >= 120 days (12 months monthly); "
                     "grade A: storage Pearson r >= 0.995 and linear-map RMS residual <= 1% of range; elevation isotonic-fit and leave-one-out "
                     "neighbour residuals <= 1% of range (Spearman >= 0.5). Grade B (used only when no series reaches A): 0.99 / 2.5%"),
         extended_grade={k: int(v) for k, v in SX[SX.ext_status == "extended"].ext_grade.value_counts().items()})
json.dump(v, open(O/"version.json", "w"), indent=1)
by = SX[SX.ext_status == "extended"].groupby("ext_source").size().to_dict()
(O/"README.md").write_text(f"""# Regime 1x: agency gauge truth 2023-01-01 .. 2026-09-30

Built {v['built']} by `gauges2026/build_regime1x.py` from the frozen `regime1_2023_2025` set plus the `gauges2026` harvest
(`harvest_2026.py`). **TEST ONLY** - never written into any training target. `regime1_2023_2025` is unchanged.

* Every 2023-2025 row is byte-identical to regime 1 (asserted). 2026 rows are appended per dam.
* A dam's 2026 values come from the SAME validated gauge series as its 2023-25 truth when possible (else its gauge_final
  series, else another 2015-20 tournament winner), re-fetched from the agency and mapped onto the frozen truth over their
  2024-25 overlap (storage: linear; elevation: a monotone rating through the overlap's own level->storage pairs), accepted
  only if {v['acceptance']}. If that series stops early, later accepted series fill only the missing days
  ({v['gap_fill_days']} days; `method` says "gap fill").
* Sites, tiers, members and basins are those of regime 1 (no re-derivation). `sites.csv` adds the extension columns.
* `truth_clean_v1x.parquet` = `truth_clean_v1` + 2026 rows after the same automated rules (zero dropouts, isolated spikes,
  CWMS validity, USGS Eqp). Removed 2026 days by rule: {rem}.
* {v['extrapolated_days']} 2026 days have a level outside the 2024-25 range used for the rating; they are kept and marked in `method`.

Extended: {v['extended']} of {v['sites']} sites ({v['extended_high_medium']} High/Medium); by source {by}.
""")
print(json.dumps(v, indent=1)); print("by source:", by)
