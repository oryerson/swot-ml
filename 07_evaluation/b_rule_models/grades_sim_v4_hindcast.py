"""Run one reservoir rule model on GRADES-hydroDL inflow, routed through upstream reservoirs.

The rule-model benchmarks of the paper are published reservoir operating rules driven by river
discharge that GRADES-hydroDL simulates without reservoirs ("natural flow"), routed through the
upstream reservoirs that have rules. The engines are in resops_engines.py and are checked against
the authors' code by validate_engines.py. This is the version of the simulator that made the
paper's 1980-2020 runs. grades_sim_v4.py is the same simulator with three more options (--end,
--qbar-end, --unitfix); it made the SWOT Period runs.

Usage:
  python grades_sim_v4_hindcast.py <family> [--inflow grades|obs] [--no-cascade] [--s0 F]
         [--init frac|gauge] [--start DATE] [--pdsi FILE] [--norm gaugefree|published]
         [--qshift N] [--tag SUFFIX]

  The paper's 1980-2020 runs: `<family> --init gauge --start 1980-01-01` for gdrom, istarf and
  modrom_cal (run tags gdrom3c_g_y1980, istarf3c_g_y1980, modromcal3c_g_y1980).

Families (first argument):
  gdrom       GDROM v2 rules (Zheng et al. 2025).
  istarf      ISTARF-CONUS (Turner et al. 2021), in the daily form of the mosartwmpy model.
  modrom_cal  MODROM (Li & Villarini 2026) with the authors' calibrated monthly modes and their
              normalization maxima.
  modrom_rf   MODROM transferred to reservoirs it was not fitted on: modes and parameters come from
              random forests that never saw the reservoir (modrom/modrom_rf_heldout_params.json,
              made by modrom_rf_heldout.py, which is not part of this repository). Normalization
              without gauges: storage_max = GDW capacity, inflow_max = maximum of the simulated
              inflow, outflow_max set by bisection so that mean release = mean inflow.

Options:
  --inflow obs    Oracle run on observed inflow (see Method). Default: grades.
  --no-cascade    Simulate only the test-set dams, each on natural flow, without upstream
                  reservoirs.
  --s0 F          Initial storage as a fraction of capacity, for every family (sensitivity to
                  spin-up).
  --init gauge    Start every reservoir at its gauged storage (see Method). Default: frac.
  --start DATE    First simulated day (default 2000-01-01; the discharge files start in 1980). The
                  run ends on the last day of the discharge files.
  --pdsi FILE     Statewide PDSI table for GDROM (default: see Reads).
  --norm published   modrom_rf only: use the authors' gauge-based normalization maxima
                  (diagnostic). Default: gaugefree.
  --qshift N      Sensitivity test only: the natural flow of day t is the GRADES value of day t+N.
  --tag SUFFIX    Text appended to the run tag.

Run tag (TAG, names the outputs): gdrom3, istarf3, modromcal3 or modromrf3, then `c` for a cascade
run, `_obs` for observed inflow, `_g` for --init gauge, `_y<year>` when the start is not
2000-01-01, `_qs<N>` for --qshift, then the --tag suffix.

Method:
  - Natural flow at a node: GRADES-hydroDL `Qout` (m3/s) of the node's MERIT-Basins v0.7 reach
    (reservoir_grades_map.csv), with negative values set to 0 (in the paper's data: 73 values in 5
    reaches), missing values interpolated over up to 5 days and otherwise set to 0, and multiplied
    by the node/reach upstream-area ratio. A ratio outside 0.2-5 is not applied; the reservoir is
    then flagged (area_ratio_flag). The script stops if a day of the simulation window is missing
    from the discharge files.
  - Which reservoirs are simulated: the test-set dams of the family, plus every reservoir of the
    family that has a test-set dam downstream of it on the graph. Reservoirs are placed on nodes by
    rule_reservoir_nodes.csv. MODROM numbers its non-GRanD reservoirs (ids of 10000 and above)
    differently from GDROM, so the MODROM families place those with modrom_nongrand_nodes.csv.
  - Cascade: reservoirs are run from small to large upstream area. The inflow of a reservoir is
    its local flow (natural flow at its node minus the natural flow at the nearest simulated
    reservoirs upstream, not below 0) plus the simulated outflow of those reservoirs. An upstream
    reservoir that could not be simulated passes its natural flow.
  - GDROM capacity: the largest of the GDW capacity, GDROM's STORAGE_CAP and 1.05 x the largest
    storage threshold in the reservoir's rules.
  - GDROM drought index (PDSI): GDROM's rules must see the index they were fitted on, and GDROM's
    metadata gives some dams a neighboring or wrong state (for example Dillon Lake, Ohio, is listed
    in Colorado). The script therefore looks for the state whose NOAA statewide monthly series
    matches the PDSI column of GDROM's cleaned record from 1999 on (at least 90% of the months
    equal within 0.006). If none matches, it uses the metadata state. The choice is logged
    (pdsi_state, pdsi_state_src). Months after the end of the PDSI table repeat its last value.
  - ISTARF: capacity is GRanD_CAP_MCM. Releases are standardized by the mean of the simulated
    inflow (GRanD mean flow if that mean is 0).
  - MODROM: reservoirs are used only if all twelve monthly modes are 2 or 3 and all parameters are
    finite. With the authors' normalization, capacity = the larger of their storage_max and the
    GDW capacity.
  - Initial storage. `--init frac` (default): a fraction of capacity, 0.6 for GDROM and MODROM
    and the middle of the normal operating range at the turn of the year for ISTARF, or `--s0`
    for all families. `--init gauge`: the reservoir's real storage on the start day, taken from a
    gauge record: the value nearest the start day within 15 days; else the record's median within
    15 days of that day of year over all years (at least 15 values); else the `--init frac`
    value. Records: for GDROM and MODROM first GDROM's cleaned record (acre-feet, the scale of
    GDROM's thresholds and of MODROM's normalization), then ResOpsUS times the correction
    factor `fix_factor` of crosswalk/resops_record_qa.csv; for ISTARF the reverse order. MODROM
    ids of 10000 and above are not GDROM's reservoirs, so they keep the default. Source and
    fraction are logged (init_source, init_frac).
  - `--inflow obs` (oracle run): each test-set dam gets its own observed inflow from GDROM's
    cleaned record, and no cascade is run. Gaps of up to 5 days are interpolated; longer gaps are
    filled with GRADES flow rescaled to the observed mean. At least 365 observed days are needed.
  - GDROM fallbacks (no rule line matches, module file missing; see resops_engines.py) are counted
    per dam in the summary table.

Reads (relative to $SWOTGAT_WORK/GRADES/ unless the path starts with `$`):
  rule_reservoir_nodes.csv, modrom_nongrand_nodes.csv, reservoir_grades_map.csv, testset_dams.csv,
  qout/qout_*.parquet
  gdrom       gdrom_v2/Operation Rules - GDROMs/, gdrom_v2/reservoir_metadata.csv,
              pdsi_state_monthly.csv, or pdsi_state_monthly_1979_2025.csv when the start is before
              1999 (`--pdsi` overrides)
  istarf      ISTARF-CONUS.csv
  modrom_*    modrom/results/calibrated_parameters.json or modrom/modrom_rf_heldout_params.json,
              modrom/data/reservoir_norm_info.json
  gdrom_v2/cleaned/<id>.csv (GDROM cleaned records: PDSI, observed inflow, storage)
  $SWOTGAT_WORK/CONUS/metadata/subbasins_conus.parquet (downstream node and upstream area)
  $SWOTGAT_WORK/CONUS_storage_fork/: crosswalk/resops_record_verdict.csv (GDW capacity),
      crosswalk/resops_record_qa.csv, ResOpsUS/time_series_all/ResOpsUS_<id>.csv,
      testsets/{regime1_2023_2025,regime2_2015_2020}/sites.csv

Writes:
  $SWOTGAT_WORK/GRADES/v3/sim_<TAG>.csv   one row per simulated reservoir: status, capacity, mean
      natural flow, inflow and outflow, share of days full or empty, initial storage, and the
      family's diagnostics.
  $SWOTGAT_WORK/CONUS_storage_fork/predictions/grades_<TAG>/
      full_series.parquet       DAM_ID, date, storage_mcm: the whole simulated period for every
                                test-set dam. Storage is absolute, in million m3 (MCM). The
                                scoring scripts read this file.
      step_050000_c1.parquet    2023-2025, dams of test set regime1_2023_2025
      step_050000_c2.parquet    2015-2020, dams of test set regime2_2015_2020
                                Both have the layout of the SWOT-GAT prediction files (basin,
                                subbasin, date, pred_dv_SWOTAE, pred_dv_SWOTAE_std). The simulated
                                storage is on the dam's inflow node; its other member nodes hold 0.

Notes:
  - Set PYTHONHASHSEED (for example to 0) to make a run repeatable. The upstream reservoirs of a
    node are held in a Python set of strings, and their outflows are summed in set order, which
    depends on the hash seed of the process. Reruns then differ by floating-point round-off, which
    threshold rules can amplify at dams deep in a cascade. The paper's 1980-2020 runs were made
    without a fixed seed (see docs/KNOWN_LIMITS.md).
  - The node table subbasins_conus.parquet is the first graph version, not graph v3.5 (see
    docs/KNOWN_LIMITS.md).
  - `--s0 0` is read as "not set" for GDROM and MODROM (they then start at 0.6 of capacity).
"""
from swotgat.paths import WORK
import sys, json, argparse, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from resops_engines import GdromRules, run_gdrom, run_istarf, run_modrom, modrom_release_n, epiweek

ap = argparse.ArgumentParser(); ap.add_argument("family"); ap.add_argument("--inflow", default="grades"); ap.add_argument("--no-cascade", action="store_true")
ap.add_argument("--s0", type=float, default=None); ap.add_argument("--tag", default="")
ap.add_argument("--norm", default="gaugefree", help="modrom_rf only: 'published' uses the authors' gauge-based normalisation maxima (diagnostic)")
ap.add_argument("--init", choices=["frac", "gauge"], default="frac"); ap.add_argument("--start", default="2000-01-01"); ap.add_argument("--pdsi", default=None)
ap.add_argument("--qshift", type=int, default=0, help="SENSITIVITY ONLY: natural flow on day t = GRADES Qout of day t+N (N=1 advances GRADES one day)")
A = ap.parse_args()
G = Path(f"{WORK}/GRADES"); F = Path(f"{WORK}/CONUS_storage_fork")
C_ = Path(f"{WORK}/CONUS"); RD = G/"gdrom_v2/Operation Rules - GDROMs"; CL = G/"gdrom_v2/cleaned"; MR = G/"modrom"
AF = 1233.48184; DAY = 86400.0
FAM = A.family; OBS = A.inflow == "obs"; CASCADE = not (A.no_cascade or OBS)
TAG = {"gdrom": "gdrom3", "istarf": "istarf3", "modrom_cal": "modromcal3", "modrom_rf": "modromrf3"}[FAM] + ("c" if CASCADE else "") + ("_obs" if OBS else "") + ("_g" if A.init == "gauge" else "") + ("" if A.start == "2000-01-01" else "_y" + A.start[:4]) + (f"_qs{A.qshift}" if A.qshift else "") + A.tag
(G/"v3").mkdir(exist_ok=True)

# ---------------- reservoirs of the family, placed on graph nodes
RN = pd.read_csv(G/"rule_reservoir_nodes.csv", dtype={"node": str}).dropna(subset=["node"]).drop_duplicates("node")
GDW_CAP = pd.read_csv(F/"crosswalk/resops_record_verdict.csv").drop_duplicates("DAM_ID").set_index("DAM_ID").cap   # = GDW CAP_MCM
if FAM == "gdrom":
    fam_ids = set(RN.GRAND_ID[RN.gdrom])
elif FAM == "istarf":
    fam_ids = set(RN.GRAND_ID[RN.istarf])
else:
    PRM = {int(x["GRAND_ID"]): x for x in json.load(open(MR/("results/calibrated_parameters.json" if FAM == "modrom_cal" else "modrom_rf_heldout_params.json")))}
    NRM = {int(k): v for k, v in json.load(open(MR/"data/reservoir_norm_info.json")).items()}
    ok = {g for g, x in PRM.items() if all(m in (2, 3) for m in x["MonthlyModes"]) and np.all(np.isfinite(np.asarray(sum(x["ModeParameters"], []), float)))}
    fam_ids = ok & set(NRM) if (FAM == "modrom_cal" or A.norm == "published") else ok
if FAM.startswith("modrom"):
    # MODROM numbers its non-GRanD reservoirs differently from GDROM (e.g. 10034 = Causey UT vs
    # GDROM's Ridgway CO): place them with MODROM's own coordinates (modrom_nongrand_nodes.py),
    # never with GDROM's.
    MNG = pd.read_csv(G/"modrom_nongrand_nodes.csv", dtype={"node": str}).set_index("GRAND_ID").node
    RN0 = pd.read_csv(G/"rule_reservoir_nodes.csv", dtype={"node": str}).drop_duplicates("GRAND_ID")
    RN0 = RN0[RN0.GRAND_ID < 10000]
    RN = pd.concat([RN0, pd.DataFrame({"GRAND_ID": MNG.index, "node": MNG.values, "gdrom": True, "istarf": False, "test": False})])
    RN = RN.dropna(subset=["node"]).drop_duplicates("node")
RN = RN[RN.GRAND_ID.isin(fam_ids)].set_index("GRAND_ID")

s = pd.read_parquet(C_/"metadata/subbasins_conus.parquet", columns=["nextdown", "uparea_km2"]); s.index = s.index.astype(str)
def _nid(v):
    """Return a node id as a string, or None for "no downstream node" (0, -1, empty or missing)."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)): return None
    v = str(int(v)) if isinstance(v, (int, float, np.integer, np.floating)) else str(v)
    return None if v in ("0", "-1", "", "nan", "None") else v
nxt = {k: _nid(v) for k, v in s.nextdown.items()}
GM = pd.read_csv(G/"reservoir_grades_map.csv", index_col=0); GM.index = GM.index.astype(str)
Q = pd.concat([pd.read_parquet(f) for f in sorted((G/"qout").glob("qout_*.parquet"))]).sort_index(); Q.columns = Q.columns.astype("int64")
Q = Q.clip(lower=0.0)
if A.qshift: Q = Q.shift(-A.qshift)          # value labelled t+N used on day t (sensitivity)
days = pd.date_range(A.start, Q.index.max(), freq="D"); n = len(days)
_miss = days.difference(Q.index)
assert len(_miss) == 0, f"GRADES Qout missing {len(_miss)} days in the simulation window, first {_miss[:3]}"   # natural() would fill 0
_nat = {}
def natural(node):
    """Return the natural (no-reservoir) flow at a node in m3/s on every simulation day.

    This is the GRADES-hydroDL discharge of the node's reach, with gaps of up to 5 days
    interpolated and the remaining gaps set to 0, times the node/reach upstream-area ratio (ratio
    1 if it is outside 0.2-5). Returns None if the node has no GRADES reach. Results are cached.
    """
    if node in _nat: return _nat[node]
    if node not in GM.index or pd.isna(GM.comid_v07[node]) or int(GM.comid_v07[node]) not in Q.columns: _nat[node] = None; return None
    r = GM.area_ratio[node]; r = float(r) if np.isfinite(r) and 0.2 < r < 5 else 1.0
    _nat[node] = Q[int(GM.comid_v07[node])].reindex(days).interpolate(limit=5).fillna(0.0).values * r; return _nat[node]   # m3/s
def ratio_flag(node):
    """Return True if the node's area ratio is outside 0.2-5 (natural() then does not apply it)."""
    r = GM.area_ratio.get(node, np.nan); return bool(np.isfinite(r) and not (0.2 < r < 5))

T = pd.read_csv(G/"testset_dams.csv", dtype={"inflow_node": str}).dropna(subset=["inflow_node"]).drop_duplicates("DAM_ID")
test_nodes = {d: nd for d, nd in zip(T.DAM_ID, T.inflow_node) if d in RN.index}
node2dam = {nd: d for d, nd in zip(RN.index, RN.node)}
# keep = nodes to simulate: the test-set dams, plus (cascade runs) every family reservoir with a
# test-set dam downstream. down[nd] = the next simulated node downstream of nd; ups[nd] = the
# simulated nodes that drain directly into nd. `order` runs upstream reservoirs first.
if CASCADE:
    tset = set(test_nodes.values()); keep, path_of = set(tset), {}
    for nd in set(RN.node):
        p, cur, k = [], nxt.get(nd), 0
        while cur is not None and k < 20000: p.append(cur); cur = nxt.get(cur); k += 1
        path_of[nd] = p
        if any(x in tset for x in p): keep.add(nd)
    down = {nd: next((x for x in path_of.get(nd, []) if x in keep), None) for nd in keep}
else:
    keep, down = set(test_nodes.values()), {nd: None for nd in test_nodes.values()}
ups = {nd: [u for u in keep if down.get(u) == nd] for nd in keep}
order = sorted(keep, key=lambda nd: s.uparea_km2.get(nd, 0.0))
print(f"{TAG}: family reservoirs placed {len(RN)} | test dams {len(test_nodes)} | simulated {len(keep)} | "
      f"test dams with simulated upstream reservoirs {sum(1 for nd in test_nodes.values() if ups[nd])}", flush=True)

# ---------------- observed inflow (oracle)
def observed_inflow(dam, nat_m3s):
    """Return the observed inflow of a dam in m3/s, and the share of observed days in 2015-2020.

    The inflow comes from GDROM's cleaned record. Gaps of up to 5 days are interpolated; longer
    gaps are filled with the GRADES flow `nat_m3s` (m3/s) rescaled to the observed mean. Returns
    (None, nan) when the dam has no record or the record has fewer than 365 days in the
    simulation period.
    """
    f = CL/f"{dam}.csv"
    if not f.exists(): return None, np.nan
    o = pd.read_csv(f, parse_dates=["Date"]).set_index("Date").Inflow.reindex(days)          # acre-ft/day
    have = o.notna().values; o = o.interpolate(limit=5, limit_area="inside")
    if o.notna().sum() < 365: return None, np.nan
    g = pd.Series(nat_m3s * DAY / AF, index=days); k = o.notna() & np.isfinite(g)
    sc = o[k].mean() / g[k].mean() if g[k].mean() > 0 else 1.0
    I = o.fillna(g * sc).values * AF / DAY                                                       # back to m3/s
    return I, float(np.mean(have[(days >= "2015-01-01") & (days <= "2020-12-31")]))

# ---------------- family inputs
if FAM == "gdrom":
    MD = pd.read_csv(G/"gdrom_v2/reservoir_metadata.csv").set_index("GRAND_ID")
    PS = pd.read_csv(A.pdsi or (G/("pdsi_state_monthly_1979_2025.csv" if pd.Timestamp(A.start) < pd.Timestamp("1999-01-01") else "pdsi_state_monthly.csv")))
    assert PS.yyyymm.min() <= pd.Timestamp(A.start).year * 100 + pd.Timestamp(A.start).month, "PDSI table starts after the simulation"
    piv = PS.pivot(index="yyyymm", columns="state", values="pdsi")
    ym = days.year * 100 + days.month
    PDS = {st: piv[st].reindex(ym).ffill().fillna(0.0).values for st in piv.columns}
    def training_state(dam):
        """Return the state whose PDSI series the dam's rules were fitted on, and how it was found.

        This is the state whose NOAA monthly series equals the PDSI column of GDROM's cleaned
        record (from 1999 on, within 0.006) in at least 90% of the months; the second value is
        then "matched", or "matched-other" if the metadata names another state. If no state
        matches, or the record is missing or has fewer than 365 days from 1999 on, the state in
        the metadata is returned with "metadata".
        """
        f = CL/f"{dam}.csv"; st0 = MD.STATE.get(dam) if dam in MD.index else RN.STATE.get(dam)
        if not f.exists(): return st0, "metadata"
        d = pd.read_csv(f, parse_dates=["Date"]); d = d[d.Date >= "1999-01-01"]
        if len(d) < 365: return st0, "metadata"
        mm = d.assign(ym=d.Date.dt.year * 100 + d.Date.dt.month).groupby("ym").PDSI.first()
        j = piv.reindex(mm.index); agree = np.isclose(j.values, mm.values[:, None], atol=0.006).mean(0)
        i = int(np.argmax(agree))
        if agree[i] >= 0.9: return piv.columns[i], ("matched" if piv.columns[i] == st0 else "matched-other")
        return st0, "metadata"
    doy = days.dayofyear.values.astype(float)
elif FAM == "istarf":
    IP = pd.read_csv(G/"ISTARF-CONUS.csv").set_index("GRanD_ID"); EW = epiweek(days)
else:
    months = days.month.values

def closure_outmax(modes, prm, I_af, in_max, s_max, cap, s0):
    """Find the outflow_max for which the mean simulated release equals the mean inflow.

    Bisection on the logarithm of outflow_max (22 steps) between 0.05 x the mean and 50 x the
    maximum of the inflow I_af (acre-ft/day). Each step runs MODROM on I_af with the other
    arguments (see run_modrom). Returns outflow_max in acre-ft/day.
    """
    lo, hi = np.log(max(np.mean(I_af), 1e-6) * 0.05), np.log(max(np.max(I_af), 1e-6) * 50)
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        S, OUT, REL = run_modrom(modes, prm, I_af, months, in_max, s_max, np.exp(mid), cap, s0)
        if np.mean(REL) > np.mean(I_af): hi = mid
        else: lo = mid
    return float(np.exp(0.5 * (lo + hi)))

# ---------------- initial storage from the gauge record (--init gauge)
FIX = pd.read_csv(F/"crosswalk/resops_record_qa.csv").drop_duplicates("DAM_ID").set_index("DAM_ID").fix_factor
T0 = pd.Timestamp(A.start); _obs_cache = {}
def _obs_series(dam, src):
    """Return the daily observed storage of a dam in MCM, or None if there is no record.

    src 'gdrom' = GDROM v2 cleaned record; src 'resops' = ResOpsUS times the correction factor
    `fix_factor` of crosswalk/resops_record_qa.csv. Results are cached.
    """
    if (dam, src) in _obs_cache: return _obs_cache[(dam, src)]
    s = None
    if src == "gdrom":
        f = CL/f"{dam}.csv"
        if f.exists():
            d = pd.read_csv(f, usecols=["Date", "Storage"], parse_dates=["Date"]).dropna().set_index("Date").Storage
            s = d[~d.index.duplicated()].sort_index() * AF / 1e6
    elif dam < 10000:                                  # ResOps ids are GRanD ids
        f = F/f"ResOpsUS/time_series_all/ResOpsUS_{dam}.csv"
        if f.exists():
            t = pd.read_csv(f, usecols=["date", "storage"]); t["date"] = pd.to_datetime(t.date, errors="coerce")
            s = pd.to_numeric(t.dropna(subset=["date"]).drop_duplicates("date").set_index("date").storage, errors="coerce").dropna()
            s = s.sort_index() * float(FIX.get(dam, 1.0))
    _obs_cache[(dam, src)] = s; return s
def gauge_init_mcm(dam, order):
    """Return the gauged storage (MCM) to start a dam with, and its source label.

    `order` lists the records to try ("gdrom", "resops"). The first choice is the value nearest
    the start day, within 15 days, of the first record that has one (label "<record>_day"). The
    second is the median of the values within 15 days of that day of year over all years, of the
    first record that has at least 15 of them (label "<record>_doy_median"). Negative values are
    ignored. Returns (None, "default") if no record gives a value.
    """
    for src in order:                                  # 1) the record's value nearest the start day (within 15 d)
        s = _obs_series(dam, src)
        if s is None or s.empty: continue
        w = s[(s.index >= T0 - pd.Timedelta(days=15)) & (s.index <= T0 + pd.Timedelta(days=15))]
        w = w[w >= 0]
        if len(w): return float(w.iloc[int(np.argmin(np.abs((w.index - T0).days)))]), f"{src}_day"
    for src in order:                                  # 2) the record's median for this time of year
        s = _obs_series(dam, src)
        if s is None or s.empty: continue
        dd = np.abs(((s.index.dayofyear - T0.dayofyear + 182) % 365) - 182)
        w = s[(dd <= 15) & (s >= 0)]
        if len(w) >= 15: return float(w.median()), f"{src}_doy_median"
    return None, "default"
def S0AF(dam, cap_af, order):
    """Return the initial storage in acre-ft for GDROM and MODROM, and its source label.

    With `--init gauge`: the gauged storage from the records in `order`, limited to the range 0
    to cap_af (acre-ft). Without it, or if no record gives a value: capacity times `--s0`, or
    times 0.6 if `--s0` is not set, with the label "default".
    """
    if A.init == "gauge":
        v, src = gauge_init_mcm(dam, order)
        if v is not None: return float(np.clip(v * 1e6 / AF, 0.0, cap_af)), src
    return (A.s0 or 0.6) * cap_af, "default"
def MODROM_ORDER(dam):
    """Return the gauge records to try for a MODROM reservoir.

    None for ids of 10000 and above, which are not GDROM's reservoirs.
    """
    return ["gdrom", "resops"] if dam < 10000 else []

# Main loop, upstream to downstream. Inflow of a reservoir = local natural flow (natural flow at
# the node minus that at the simulated reservoirs directly upstream, not below 0) plus the simulated
# outflow of those reservoirs (their natural flow if they could not be simulated).
out_m3s, sims, summ = {}, {}, []
for nd in order:
    dam = node2dam[nd]; nat = natural(nd); info = dict(GRAND_ID=dam, node=nd, test=dam in test_nodes, area_ratio_flag=ratio_flag(nd))
    if nat is None: summ.append({**info, "status": "no GRADES reach"}); out_m3s[nd] = None; continue
    if OBS:
        I, fobs = observed_inflow(dam, nat)
        if I is None: summ.append({**info, "status": "no observed inflow"}); out_m3s[nd] = None; continue
        info["obs_frac_2015_20"] = fobs
    else:
        loc = nat.copy()
        for u in ups[nd]:
            nu = natural(u)
            if nu is not None: loc = loc - nu
        loc = np.maximum(loc, 0.0)
        I = loc + sum((out_m3s[u] if out_m3s.get(u) is not None else (natural(u) if natural(u) is not None else 0.0)) for u in ups[nd])
        I = np.asarray(I, float) if np.ndim(I) else np.full(n, float(I))
    try:
        if FAM == "gdrom":
            R = GdromRules(RD, dam); Ia = I * DAY / AF
            cap_af = np.nanmax([GDW_CAP.get(dam, np.nan) * 1e6 / AF, MD.STORAGE_CAP.get(dam, np.nan),
                                R.max_storage_threshold * 1.05 if np.isfinite(R.max_storage_threshold) else np.nan])
            if not np.isfinite(cap_af) or cap_af <= 0: raise ValueError("no capacity")
            st, how = training_state(dam); pdsi = PDS.get(st, np.zeros(n))
            s0_af, isrc = S0AF(dam, cap_af, ["gdrom", "resops"]); info.update(init_source=isrc, init_frac=s0_af / cap_af)
            S_, OUT_, _ = run_gdrom(R, Ia, doy, pdsi, cap_af, s0_af)
            S = S_ * AF / 1e6; OUT = OUT_ * AF / DAY; cap_mcm = cap_af * AF / 1e6
            info.update(pdsi_state=st, pdsi_state_src=how, nomatch_cond=R.n_nomatch_cond, nomatch_rel=R.n_nomatch_rel, missing_module=R.n_missing_module)
        elif FAM == "istarf":
            p = IP.loc[dam].copy(); p["Release_max"] = p.Release_max if np.isfinite(p.Release_max) else np.inf
            p["Release_min"] = p.Release_min if np.isfinite(p.Release_min) else -np.inf
            cap = float(p.GRanD_CAP_MCM) * 1e6
            if not np.isfinite(cap) or cap <= 0: raise ValueError("no capacity")
            Id = I * DAY; qbar = float(np.mean(Id)) if np.mean(Id) > 0 else float(p.GRanD_MEANFLOW_CUMECS) * DAY
            hi0 = min(p.NORhi_max, max(p.NORhi_min, p.NORhi_mu + p.NORhi_beta)); lo0 = min(p.NORlo_max, max(p.NORlo_min, p.NORlo_mu + p.NORlo_beta))
            s0 = cap * (A.s0 if A.s0 is not None else np.clip((hi0 + lo0) / 200, 0, 1)); isrc = "default"
            if A.init == "gauge":
                v, src = gauge_init_mcm(dam, ["resops", "gdrom"])
                if v is not None: s0, isrc = float(np.clip(v * 1e6, 0.0, cap)), src
            info.update(init_source=isrc, init_frac=s0 / cap)
            S_, OUT_, _ = run_istarf(p, Id, EW, cap, s0, qbar)
            S = S_ / 1e6; OUT = OUT_ / DAY; cap_mcm = cap / 1e6; info.update(istarf_fit=p.fit)
        else:
            x = PRM[dam]; Ia = I * DAY / AF
            if FAM == "modrom_cal" or A.norm == "published":
                nm = NRM[dam]; s_max, in_max, out_max = nm["storage_max"], nm["inflow_max"], nm["outflow_max"]
                cap_af = max(s_max, GDW_CAP.get(dam, 0.0) * 1e6 / AF)
            else:
                cap_af = GDW_CAP.get(dam, np.nan) * 1e6 / AF
                if not np.isfinite(cap_af) or cap_af <= 0: raise ValueError("no GDW capacity")
                s_max, in_max = cap_af, max(float(np.max(Ia)), 1e-6)
                out_max = closure_outmax(x["MonthlyModes"], x["ModeParameters"], Ia, in_max, s_max, cap_af, S0AF(dam, cap_af, MODROM_ORDER(dam))[0])
            s0_af, isrc = S0AF(dam, cap_af, MODROM_ORDER(dam)); info.update(init_source=isrc, init_frac=s0_af / cap_af)
            S_, OUT_, _ = run_modrom(x["MonthlyModes"], x["ModeParameters"], Ia, months, in_max, s_max, out_max, cap_af, s0_af)
            S = S_ * AF / 1e6; OUT = OUT_ * AF / DAY; cap_mcm = cap_af * AF / 1e6
            info.update(in_max=in_max, s_max=s_max, out_max=out_max)
    except Exception as e:
        summ.append({**info, "status": f"error: {e}"}); out_m3s[nd] = None; continue
    out_m3s[nd] = OUT; sims[dam] = pd.Series(S, index=days)
    w = (days >= "2015-01-01") & (days <= "2020-12-31"); w1 = days >= "2023-01-01"
    summ.append({**info, "status": "ok", "n_upstream": len(ups[nd]), "cap_mcm": cap_mcm, "mean_inflow_cms": float(I.mean()),
                 "mean_natural_cms": float(nat.mean()), "mean_outflow_cms": float(OUT.mean()),
                 "frac_full": float(np.mean(S >= cap_mcm * 0.999)), "frac_empty": float(np.mean(S <= cap_mcm * 1e-4)),
                 "frac_full_2015_20": float(np.mean(S[w] >= cap_mcm * 0.999)), "frac_empty_2015_20": float(np.mean(S[w] <= cap_mcm * 1e-4)),
                 "frac_full_2023_25": float(np.mean(S[w1] >= cap_mcm * 0.999)), "frac_empty_2023_25": float(np.mean(S[w1] <= cap_mcm * 1e-4)),
                 "sd_2015_20_mcm": float(S[w].std())})
SM = pd.DataFrame(summ); SM.to_csv(G/"v3"/f"sim_{TAG}.csv", index=False)
ok = SM[SM.status == "ok"]; tt = ok[ok.test]
print(SM.status.str.split(":").str[0].value_counts().to_string())
print(f"test dams ok {len(tt)} | >50% of days full or empty: {int(((tt.frac_full > .5) | (tt.frac_empty > .5)).sum())} "
      f"| flat in 2015-20 (sd < 0.1% cap): {int((tt.sd_2015_20_mcm < 1e-3 * tt.cap_mcm).sum())} | area-ratio flagged {int(tt.area_ratio_flag.sum())}")
if FAM == "gdrom":
    print(f"PDSI state source (test dams): {tt.pdsi_state_src.value_counts().to_dict()} | no-match days: cond {int(tt.nomatch_cond.sum())}, "
          f"release {int(tt.nomatch_rel.sum())}, missing module {int(tt.missing_module.sum())}")

# Storage of the test-set dams in the layout of the SWOT-GAT prediction files, one file per test
# set and its window: the series is written on the dam's inflow node, 0 on its other member nodes.
OUT_DIR = F/"predictions"/f"grades_{TAG}"; OUT_DIR.mkdir(parents=True, exist_ok=True)
for tag, sub, a, b in (("c1", "regime1_2023_2025", "2023-01-01", "2025-12-31"), ("c2", "regime2_2015_2020", "2015-01-01", "2020-12-31")):
    S_ = pd.read_csv(F/"testsets"/sub/"sites.csv", dtype={"node": str, "members": str, "basin": str})
    rows = []
    for _, x in S_.iterrows():
        if x.DAM_ID not in sims or x.DAM_ID not in test_nodes: continue
        ser = sims[x.DAM_ID].loc[a:b]
        for m in str(x.members).split(";"):
            rows.append(pd.DataFrame({"basin": x.basin, "subbasin": m, "date": ser.index,
                                      "pred_dv_SWOTAE": ser.values if m == test_nodes[x.DAM_ID] else 0.0, "pred_dv_SWOTAE_std": 0.0}))
    if not rows: print(f"{tag}: no test dams"); continue
    P = pd.concat(rows, ignore_index=True).drop_duplicates(["basin", "subbasin", "date"])
    P.to_parquet(OUT_DIR/f"step_050000_{tag}.parquet", index=False)
    print(f"{tag}: {P.subbasin.nunique()} nodes, {len(P):,} rows -> {OUT_DIR.name}")
# full simulated period for every test dam (full-record scoring)
FULL = [pd.DataFrame({"DAM_ID": d, "date": ser.index, "storage_mcm": ser.values.astype("float64")}) for d, ser in sims.items() if d in test_nodes]
if FULL:
    pd.concat(FULL, ignore_index=True).to_parquet(OUT_DIR/"full_series.parquet", index=False)
    print(f"full series: {len(FULL)} test dams, {days[0].date()}..{days[-1].date()} -> {OUT_DIR.name}/full_series.parquet")
if "init_source" in SM:
    print("init sources (test dams):", SM[SM.test & (SM.status == "ok")].init_source.value_counts().to_dict(),
          "| all reservoirs:", SM[SM.status == "ok"].init_source.value_counts().to_dict())
