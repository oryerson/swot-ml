"""Check the rule-model engines of resops_engines.py against the authors' code and published scores.

Every check uses OBSERVED inflow and runs before any GRADES-hydroDL benchmark. If a benchmark is
then weak, the cause is the simulated inflow or the rules themselves, not this implementation.

Usage:
  python validate_engines.py <step> [np=N]
     step  parser | gdrom | modrom | istarf | inflow
     np=N  number of worker processes (default 16)

Steps:
  parser  GDROM v2: the safe rule parser of resops_engines.py against the dataset's own script
          rule2model.py, on every rule set. Each rule set is evaluated at 4,000 random points
          (each variable drawn uniformly, or exactly at, just above or just below one of the rule
          set's thresholds) plus up to about 4,000 days of the observed record. Two releases are
          equal when |ours - reference| / max(1, |reference|) is below 1e-9. The step also counts
          the points where the reference script as distributed would stop with a NameError (some
          release expressions keep `Storage` and `Inflow` capitalized, names its generated
          functions do not define); the comparison defines both names so that the reference runs.
  gdrom   GDROM v2 on its cleaned daily records (453 records of categories Res-R and Res-L in the
          dataset used for the paper).
          realS: release computed from observed storage on every complete day, scored against the
          observed release and compared with the NSE GDROM reports in its metadata. Three variants:
          the engine's release (floored at 0), the unclipped release of the reference code, and the
          unclipped release on the last 20% of the days.
          iterS: storage simulated from observed inflow on the longest run of at least 365
          consecutive complete days, started from observed storage, with capacity = the largest
          observed storage. Scored on the whole run and on its last 20%; also after a restart from
          observed storage at the start of the last 20% (removes earlier drift), and for a
          reference-style simulation with no clipping and no spill.
  modrom  MODROM calibrated parameters on GDROM's cleaned daily records.
          realS: normalized release against normalized observed release. Its KGE is the authors'
          calibration metric and is compared with their published calibration KGE. The maxima of
          the record are compared with the authors' normalization maxima.
          iterS: storage on the longest run of at least 365 consecutive days. Also the outflow_max
          that closes the mass balance (mean release = mean inflow), against the authors' value.
  istarf  (a) Reproduce the weekly ISTARF simulation distributed with MODROM (their istarf.R) from
          the ISTARF-CONUS parameters. This checks parameter reading, harmonics, operating range
          and epiweek. Weeks are compared where the target release does not exceed the available
          water. The same check is repeated with their "fitted" simulation files.
          (b) The daily engine on GDROM's cleaned records. realS release, with the mean inflow
          taken from GRanD or from the record, compared with the ISTARF KGE published with MODROM;
          iterS storage on the longest run of at least 365 consecutive days.
  inflow  GRADES-hydroDL natural flow at every placed rule reservoir against observed inflow (GDROM
          cleaned records and MODROM daily records), from 2000 on: mean ratio, standard-deviation
          ratio, r and NSE on daily values, and r on monthly means, among others. The GRADES flow
          is scaled by the node/reach upstream-area ratio when that ratio is within 0.2-5, as in
          the simulators.

Reads (relative to $SWOTGAT_WORK/GRADES/):
  gdrom_v2/Operation Rules - GDROMs/{module_conditions,modules}/   GDROM rule text files
  gdrom_v2/cleaned/<id>.csv             GDROM cleaned records (Inflow, Storage, Release, PDSI, DOY)
  gdrom_v2/Scripts/rule2model.py        GDROM's reference script (step parser)
  gdrom_v2/reservoir_metadata.csv       category, published NSE, inflow source
  modrom/results/calibrated_parameters.json, modrom/data/reservoir_norm_info.json,
  modrom/results/calibration_metrics.csv, modrom/results/benchmark_metrics.csv,
  modrom/istarf_sim/istarf_simulation/..., modrom/daily/daily_records/<id>.csv
  ISTARF-CONUS.csv
  reservoir_grades_map.csv, rule_reservoir_nodes.csv, qout/qout_*.parquet   (step inflow)

Writes (in $SWOTGAT_WORK/GRADES/validation/), one row per reservoir (per reservoir and record
source for `inflow`), plus a printed summary of medians and interquartile ranges:
  parser  parser_equivalence.csv
  gdrom   gdrom_observed_inflow.csv
  modrom  modrom_observed_inflow.csv
  istarf  istarf_weekly_repro.csv, istarf_fitted_repro.csv, istarf_observed_inflow.csv
  inflow  grades_vs_observed_inflow.csv

The score functions are in resops_engines.py; nRMSE here is RMSE / standard deviation of the
observations. "realS" and "iterS" are explained there.

Notes:
  - Step `parser` executes part of the GDROM dataset's rule2model.py and the functions it generates
    from the rule files. Run it only on the dataset as downloaded from its authors.
  - Records are paired by id number. MODROM and GDROM number their non-GRanD reservoirs (ids of
    10000 and above) differently (see modrom_nongrand_nodes.py), so for those ids steps `modrom`
    and `inflow` can pair a MODROM id with a GDROM record or node of another reservoir.
"""
from swotgat.paths import WORK
import sys, json, re, glob, numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, str(Path(__file__).parent))
from resops_engines import GdromRules, run_gdrom, run_istarf, run_modrom, modrom_release_n, epiweek, nse, kge, nrmse_sd

G = Path(f"{WORK}/GRADES"); OUT = G/"validation"; OUT.mkdir(exist_ok=True)
RD = G/"gdrom_v2/Operation Rules - GDROMs"; CL = G/"gdrom_v2/cleaned"; MR = G/"modrom"
AF = 1233.48184; DAY = 86400.0
step = sys.argv[1]; NP = int(dict(a.split("=") for a in sys.argv[2:] if "=" in a).get("np", 16))

def segments(df, cols, min_len):
    """Split df into runs of consecutive days on which every column of `cols` is present.

    df is indexed by date. Returns the list of the runs of at least `min_len` days, each as a
    DataFrame.
    """
    ok = df[cols].notna().all(axis=1).values & np.r_[True, (df.index[1:] - df.index[:-1]).days == 1]
    good = df[cols].notna().all(axis=1).values
    brk = np.r_[True, ~ok[1:]] | ~good
    seg_id = np.cumsum(brk); out = []
    for k in np.unique(seg_id[good]):
        idx = np.where((seg_id == k) & good)[0]
        if len(idx) >= min_len: out.append(df.iloc[idx])
    return out

def score_block(o, s, tag):
    """Return the scores of simulated s against observed o as a dict with keys prefixed by `tag`.

    The keys are <tag>_nse, <tag>_kge and <tag>_nrmse (functions of resops_engines.py).
    """
    return {f"{tag}_nse": nse(o, s), f"{tag}_kge": kge(o, s), f"{tag}_nrmse": nrmse_sd(o, s)}

# =========================================================================================== parser equivalence
def ref_namespace():
    """Load the function definitions of GDROM's reference script into a fresh namespace.

    Executes the part of rule2model.py that comes before its rule-extraction cell and returns
    the resulting namespace (a dict).
    """
    src = (G/"gdrom_v2/Scripts/rule2model.py").read_text().split("#%% rule extraction")[0]
    ns = {}; exec(src, ns); return ns

def ref_release(ns, gid, inflow, storage, pdsi, doy):
    """Return the release of reservoir gid from the reference functions in namespace ns.

    The names Storage and Inflow are first defined in ns, so that release expressions that keep
    the capitalized names can run. The module is chosen with the reference's module-condition
    function if the reservoir has one, otherwise module 0 is used. Returns None if the reference
    has no function for the module.
    """
    ns["Storage"], ns["Inflow"] = storage, inflow          # patch for the reference's un-lower-cased "Release: <expr>" lines
    ct = ns.get(f"CT_{gid}")
    mf = ns.get(f"module_{gid}_{ct(inflow, pdsi, doy, storage)}") if ct else ns.get(f"module_{gid}_0")
    return mf(inflow, storage) if mf else None

def parser_one(gid):
    """Compare the engine's release with the reference on test points (step parser, one rule set).

    The points are 4,000 random draws (seeded with the reservoir id) plus up to about 4,000 days
    of the cleaned record. Returns one table row as a dict: the number of points, of equal
    releases, of mismatches and of points where the reference has no function, the largest
    relative difference with an example, the number of points where the unpatched reference
    raises NameError, and the engine's fallback counts. If the reference or the engine cannot
    read the rule set, the row holds only the id and a status text.
    """
    ns = ref_namespace()
    try:
        if (RD/"module_conditions"/f"{gid}.txt").exists(): ns["generate_ct_function"](gid, str(RD/"module_conditions"))
        ns["generate_module_functions"](gid, str(RD/"modules"))
    except Exception as e:
        return dict(GRAND_ID=gid, status=f"reference failed: {e}")
    try: R = GdromRules(RD, gid)
    except Exception as e: return dict(GRAND_ID=gid, status=f"ours failed: {e}")
    thr = {"Inflow": [], "Storage": [], "PDSI": [], "DOY": []}
    for txt in [RD/"module_conditions"/f"{gid}.txt"] + sorted((RD/"modules").glob(f"{gid}_*.txt")):
        if txt.exists():
            for v, _, x in re.findall(r"\(\s*(Inflow|Storage|PDSI|DOY)\s*(<=|>=|<|>)\s*(-?[\d.]+(?:[eE][-+]?\d+)?)\s*\)", txt.read_text()):
                thr[v].append(float(x))
    rng = np.random.default_rng(int(gid))
    lim = {"Inflow": (0, 1e3), "Storage": (0, 1e5), "PDSI": (-6, 6), "DOY": (1, 366)}
    def draw(v, k):
        t = np.array(thr[v]); lo, hi = lim[v]
        if len(t): lo, hi = min(lo, t.min() * 0.5 if t.min() > 0 else t.min() - 1), max(hi, t.max() * 1.5)
        u = rng.uniform(lo, hi, k)
        if len(t):
            pick = rng.choice(t, k); eps = np.abs(pick) * 1e-9 + 1e-9
            mode = rng.integers(0, 4, k)                       # 0 uniform, 1 exactly at threshold, 2 just above, 3 just below
            u = np.where(mode == 1, pick, np.where(mode == 2, pick + eps, np.where(mode == 3, pick - eps, u)))
        return np.round(u) if v == "DOY" else u
    k = 4000
    pts = pd.DataFrame({v: draw(v, k) for v in thr})
    f = CL/f"{gid}.csv"
    if f.exists():
        d = pd.read_csv(f).dropna(subset=["Inflow", "Storage", "PDSI", "DOY"])
        pts = pd.concat([pts, d[["Inflow", "Storage", "PDSI", "DOY"]].iloc[:: max(1, len(d) // 4000)]], ignore_index=True)
    n_eq = n_none = n_mis = 0; worst = 0.0; ex = ""
    ns0 = ref_namespace()        # unpatched copy, to count how often the reference script itself would crash
    try:
        if (RD/"module_conditions"/f"{gid}.txt").exists(): ns0["generate_ct_function"](gid, str(RD/"module_conditions"))
        ns0["generate_module_functions"](gid, str(RD/"modules"))
    except Exception: pass
    n_referr = 0
    for x in pts.itertuples(index=False):
        env = {"Inflow": x.Inflow, "Storage": x.Storage, "PDSI": x.PDSI, "DOY": x.DOY}
        try:
            ct = ns0.get(f"CT_{gid}"); mf = ns0.get(f"module_{gid}_{ct(x.Inflow, x.PDSI, x.DOY, x.Storage)}") if ct else ns0.get(f"module_{gid}_0")
            if mf: mf(x.Inflow, x.Storage)
        except NameError: n_referr += 1
        ref = ref_release(ns, gid, x.Inflow, x.Storage, x.PDSI, x.DOY)
        ours = R.release(env)
        if ref is None: n_none += 1; continue
        d_ = abs(ours - ref) / max(1.0, abs(ref))
        if d_ < 1e-9: n_eq += 1
        else:
            n_mis += 1
            if d_ > worst: worst, ex = d_, f"{env} ref={ref} ours={ours}"
    return dict(GRAND_ID=gid, status="ok", n=len(pts), ref_script_nameerror=n_referr, equal=n_eq, ref_none=n_none, mismatch=n_mis, worst_rel=worst, example=ex,
                ours_nomatch_cond=R.n_nomatch_cond, ours_nomatch_rel=R.n_nomatch_rel, ours_missing_module=R.n_missing_module)

# =========================================================================================== GDROM on cleaned records
def gdrom_one(gid):
    """Score GDROM on the cleaned record of one reservoir (step gdrom).

    Returns one table row as a dict: realS release scores (the engine's release, the unclipped
    release, and the unclipped release on the last 20% of the days), iterS storage scores on the
    longest run of at least 365 complete days (whole run, last 20%, restart at the last 20%,
    reference-style simulation without clipping or spill), the share of days full or empty, and
    the fallback counts. If the rule set cannot be read, the row holds only the id and a status
    text.
    """
    try: R = GdromRules(RD, gid)
    except Exception as e: return dict(GRAND_ID=gid, status=f"rules: {e}")
    d = pd.read_csv(CL/f"{gid}.csv", parse_dates=["Date"]).set_index("Date")
    out = dict(GRAND_ID=gid, status="ok", n_days=len(d))
    k = d[["Inflow", "Storage", "Release", "PDSI", "DOY"]].notna().all(axis=1); dd = d[k]
    # realS: release decided from observed storage, every observed day (no continuity needed)
    _, _, rel = run_gdrom(R, dd.Inflow.values, dd.DOY.values, dd.PDSI.values, np.inf, np.nan, S_obs=dd.Storage.values)
    out.update(score_block(dd.Release.values, rel, "rel_realS"))
    out.update(rel_realS_neg_frac=float(np.mean(rel <= 0)), nomatch_cond_realS=R.n_nomatch_cond, nomatch_rel_realS=R.n_nomatch_rel,
               missing_module_realS=R.n_missing_module)
    # unclipped reference-style release (their simulate_release) for an exact like-for-like number
    unc = np.array([R.release({"Inflow": a, "Storage": b, "DOY": c, "PDSI": p}) for a, b, c, p in
                    zip(dd.Inflow.values, dd.Storage.values, dd.DOY.values, dd.PDSI.values)])
    out.update(score_block(dd.Release.values, unc, "rel_realS_unclipped"))
    n80 = int(len(dd) * 0.8)
    out.update(score_block(dd.Release.values[n80:], unc[n80:], "rel_realS_last20"))
    # iterS: storage simulated from observed inflow on the longest contiguous segment, started from
    # observed storage
    segs = segments(d, ["Inflow", "Storage", "PDSI", "DOY"], 365)
    if segs:
        sg = max(segs, key=len); cap = max(float(d.Storage.max()), 1.0)
        R.n_nomatch_cond = R.n_nomatch_rel = R.n_missing_module = 0
        S, _, _ = run_gdrom(R, sg.Inflow.values, sg.DOY.values, sg.PDSI.values, cap, float(sg.Storage.iloc[0]))
        o = sg.Storage.values; m = int(len(o) * 0.8)
        out.update(seg_days=len(sg), seg_start=str(sg.index[0].date()), seg_end=str(sg.index[-1].date()))
        out.update(score_block(o, S, "S_iterS_full")); out.update(score_block(o[m:], S[m:], "S_iterS_last20"))
        out.update(frac_full=float(np.mean(S >= cap * 0.999)), frac_empty=float(np.mean(S <= cap * 1e-4)),
                   nomatch_cond_iterS=R.n_nomatch_cond, nomatch_rel_iterS=R.n_nomatch_rel)
        # restart at the last-20% start with the observed storage (removes drift accumulated before
        # the scored window)
        S2, _, _ = run_gdrom(R, sg.Inflow.values[m:], sg.DOY.values[m:], sg.PDSI.values[m:], cap, float(o[m]))
        out.update(score_block(o[m:], S2, "S_iterS_restart_last20"))
        # reference-style storage (their simulate_release_and_storage): no clipping, no spill
        s = float(sg.Storage.iloc[0]); Sr = np.empty(len(sg))
        for t, (a, c, p) in enumerate(zip(sg.Inflow.values, sg.DOY.values, sg.PDSI.values)):
            s = s + a - R.release({"Inflow": a, "Storage": s, "DOY": c, "PDSI": p}); Sr[t] = s
        out.update(score_block(o[m:], Sr[m:], "S_ref_unclipped_last20"))
    return out

# =========================================================================================== MODROM
def modrom_load():
    """Read MODROM's calibrated parameters and normalization maxima.

    Returns (P, N), both keyed by the reservoir id as text: P holds the entries of
    calibrated_parameters.json and N those of reservoir_norm_info.json.
    """
    P = {str(x["GRAND_ID"]): x for x in json.load(open(MR/"results/calibrated_parameters.json"))}
    N = json.load(open(MR/"data/reservoir_norm_info.json")); return P, N

def modrom_one(gid):
    """Score MODROM on GDROM's cleaned record of one reservoir (step modrom).

    Returns one table row as a dict: the maxima of the record next to the authors' normalization
    maxima, realS scores of the normalized release, iterS storage scores on the longest run of
    at least 365 consecutive days (whole run and last 20%), the share of days full or empty, and
    the outflow_max that makes the mean release equal to the mean inflow. Reservoirs without
    parameters or record, or with a monthly mode other than 2 or 3 or a parameter that is not
    finite, are returned with a status text only.
    """
    P, N = modrom_load()
    if gid not in P or gid not in N: return dict(GRAND_ID=gid, status="no params")
    modes, prm = P[gid]["MonthlyModes"], P[gid]["ModeParameters"]
    if any(m not in (2, 3) for m in modes) or any(np.any(~np.isfinite(np.asarray(p, float))) for p in prm):
        return dict(GRAND_ID=gid, status="incomplete params")
    if not (CL/f"{gid}.csv").exists(): return dict(GRAND_ID=gid, status="no observed record")
    d = (pd.read_csv(CL/f"{gid}.csv", parse_dates=["Date"]).rename(columns={"Date": "time", "Inflow": "inflow", "Release": "outflow",
                                                                         "Storage": "storage"}).set_index("time"))
    nm = N[gid]; out = dict(GRAND_ID=gid, status="ok")
    full = d.dropna(subset=["inflow", "outflow", "storage"]).copy()
    full["storage"] = full.storage.clip(lower=1); full["outflow"] = full.outflow.clip(lower=1)
    out.update(n_days=len(full), our_in_max=full.inflow.max(), our_s_max=full.storage.max(), our_out_max=full.outflow.max(), **nm)
    i_n, s_n, o_n = full.inflow / nm["inflow_max"], full.storage / nm["storage_max"], full.outflow / nm["outflow_max"]
    rn = np.array([modrom_release_n(modes[mo - 1], prm[mo - 1], a, b)
                   for mo, a, b in zip(full.index.month, i_n.values, s_n.values)])
    out.update(score_block(o_n.values, rn, "rel_realS_norm"))   # their calibration metric: KGE of normalised release
    segs = segments(d, ["inflow", "storage"], 365)
    if segs:
        sg = max(segs, key=len); cap = max(float(d.storage.max()), nm["storage_max"])
        S, OUTF, _ = run_modrom(modes, prm, sg.inflow.values, sg.index.month.values, nm["inflow_max"], nm["storage_max"],
                                nm["outflow_max"], cap, float(sg.storage.iloc[0]))
        o = sg.storage.values; m = int(len(o) * 0.8)
        out.update(seg_days=len(sg)); out.update(score_block(o, S, "S_iterS_full")); out.update(score_block(o[m:], S[m:], "S_iterS_last20"))
        out.update(frac_full=float(np.mean(S >= cap * 0.999)), frac_empty=float(np.mean(S <= cap * 1e-4)),
                   mean_out_over_in=float(OUTF.mean() / max(sg.inflow.mean(), 1e-9)))
        # mass-balance closure estimate of outflow_max (what a MODROM transferred to an ungauged dam
        # must do):
        # choose out_max so that mean simulated release (from observed storage) equals mean inflow
        rn2 = np.array([modrom_release_n(modes[mo - 1], prm[mo - 1], a, b)
                        for mo, a, b in zip(sg.index.month, sg.inflow.values / nm["inflow_max"], sg.storage.values / nm["storage_max"])])
        out.update(out_max_closure=float(sg.inflow.mean() / max(np.mean(np.maximum(rn2, 0)), 1e-12)))
    return out

# =========================================================================================== ISTARF
IP = None
def istarf_params():
    """Return the ISTARF-CONUS parameter table indexed by GRanD id (read once, then kept)."""
    global IP
    if IP is None: IP = pd.read_csv(G/"ISTARF-CONUS.csv").set_index("GRanD_ID")
    return IP

def istarf_weekly_repro(gid, f=None):
    """Recompute the weekly ISTARF release as MODROM's istarf.R does, line for line.

    Weekly, release only, GRanD mean flow, operating range from GRanD capacity. The result is
    compared with the release in their simulation file `f` (default: the reservoir's file of the
    "generalized" weekly simulation), on the weeks where the target release does not exceed
    inflow plus storage.

    Returns one table row as a dict: the number of weeks, of compared weeks and of compared weeks
    that are reproduced (equal within 1e-6, or their release is higher), the largest absolute
    difference (MCM), and the share of weeks where epiweek() agrees with their epi_week column.
    """
    f = f or MR/f"istarf_sim/istarf_simulation/generalized/istarf_weekly_simulation/{gid}.csv"
    p = istarf_params()
    if not f.exists() or int(gid) not in p.index: return dict(GRAND_ID=gid, status="missing")
    w = pd.read_csv(f, parse_dates=["time"]); q = p.loc[int(gid)]
    ew = w.epi_week.values.astype(float)
    def tgt(pre):
        v = q[f"{pre}_mu"] + q[f"{pre}_alpha"] * np.sin(2 * np.pi * ew / 52) + q[f"{pre}_beta"] * np.cos(2 * np.pi * ew / 52)
        return np.minimum(q[f"{pre}_max"], np.maximum(q[f"{pre}_min"], v))
    hi, lo = tgt("NORhi") * q.GRanD_CAP_MCM / 100, tgt("NORlo") * q.GRanD_CAP_MCM / 100
    mw = q.GRanD_MEANFLOW_CUMECS * 0.6048
    hr = (q.Release_alpha1 * np.sin(2 * np.pi * ew / 52) + q.Release_beta1 * np.cos(2 * np.pi * ew / 52)
          + q.Release_alpha2 * np.sin(4 * np.pi * ew / 52) + q.Release_beta2 * np.cos(4 * np.pi * ew / 52))
    ss = (w.storage_mcm.values - lo) / (hi - lo)
    target = (hr + q.Release_c + q.Release_p1 * ss + q.Release_p2 * (w.inflow_mcm.values - mw) / mw) * mw + mw
    avail = w.inflow_mcm.values + w.storage_mcm.values
    nb = (target <= avail) & np.isfinite(target) & np.isfinite(w.release_sim_mcm.values)    # weeks where only the lower bound could bind
    rs = w.release_sim_mcm.values
    match = np.isclose(target, rs, rtol=1e-6, atol=1e-6) | (rs > target + 1e-6)              # equal, or raised by the spill bound
    # our MMWR epiweek on the week-start dates vs their epi_week column
    ours_ew = epiweek(w.time.values)
    return dict(GRAND_ID=gid, status="ok", weeks=len(w), nonbinding=int(nb.sum()), repro_match=int((match & nb).sum()),
                max_abs_diff_nonbinding=float(np.nanmax(np.abs(target[nb] - rs[nb]))) if nb.any() else np.nan,
                epiweek_agree=float(np.mean(ours_ew == np.minimum(w.epi_week.values, 52))), epi_max=int(w.epi_week.max()))

def istarf_fitted_repro(gid):
    """Repeat the weekly check against the release of MODROM's "fitted" ISTARF simulation.

    The fitted daily files hold the weekly release spread evenly over 7 days. Their release
    times 7 replaces the release in the weekly simulation table (weeks present in both files),
    which is written to a temporary file in the output folder and passed to
    istarf_weekly_repro(). Returns that function's row.
    """
    fw_ = MR/f"istarf_sim/istarf_simulation/generalized/istarf_weekly_simulation/{gid}.csv"
    fd_ = MR/f"istarf_sim/istarf_simulation/fitted/istarf_daily_simulation/{gid}.csv"
    p = istarf_params()
    if not (fw_.exists() and fd_.exists()) or int(gid) not in p.index: return dict(GRAND_ID=gid, status="missing")
    w = pd.read_csv(fw_, parse_dates=["time"]).set_index("time")
    fd = pd.read_csv(fd_, parse_dates=["time"]).set_index("time")
    w = w.join(fd[["release_sim_mcm"]].rename(columns={"release_sim_mcm": "rel_fit"}), how="inner").reset_index()
    if not len(w): return dict(GRAND_ID=gid, status="no overlap")
    w["release_sim_mcm"] = w.rel_fit * 7.0          # the fitted daily files hold the weekly release spread evenly over 7 days
    f2 = OUT/f"_tmp_fit_{gid}.csv"; w.to_csv(f2, index=False)
    r = istarf_weekly_repro(gid, f2); f2.unlink(); return r

def istarf_daily_one(gid):
    """Score the daily ISTARF engine on GDROM's cleaned record of one reservoir (step istarf, b).

    Returns one table row as a dict: realS release scores with the mean inflow taken from GRanD
    and from the record, and iterS storage scores on the longest run of at least 365 consecutive
    days (whole run and last 20%), with capacity = the larger of the GRanD capacity and the
    largest observed storage. Reservoirs without parameters or record, or with fewer than 365
    complete days, are returned with a status text only.
    """
    p = istarf_params()
    if int(gid) not in p.index: return dict(GRAND_ID=gid, status="no params")
    q = p.loc[int(gid)].copy(); q["Release_max"] = q.Release_max if np.isfinite(q.Release_max) else np.inf
    q["Release_min"] = q.Release_min if np.isfinite(q.Release_min) else -np.inf
    f = CL/f"{gid}.csv"
    if not f.exists(): return dict(GRAND_ID=gid, status="no observed record")
    d = (pd.read_csv(f, parse_dates=["Date"]).rename(columns={"Date": "time", "Inflow": "inflow", "Release": "outflow",
                                                              "Storage": "storage"}).set_index("time"))
    cap = float(q.GRanD_CAP_MCM) * 1e6; out = dict(GRAND_ID=gid, status="ok", fit=q.fit)
    full = d.dropna(subset=["inflow", "outflow", "storage"])
    if len(full) < 365: return dict(GRAND_ID=gid, status="short")
    I = full.inflow.values * AF; Sobs = full.storage.values * AF; ew = epiweek(full.index.values)
    for tag, qbar in (("grandmean", q.GRanD_MEANFLOW_CUMECS * DAY), ("recmean", float(np.mean(I)))):
        _, rel, _ = run_istarf(q, I, ew, cap, np.nan, qbar, S_obs=Sobs)
        out.update(score_block(full.outflow.values * AF, rel, f"rel_realS_{tag}"))
    segs = segments(d, ["inflow", "storage"], 365)
    if segs:
        sg = max(segs, key=len); I2 = sg.inflow.values * AF; o = sg.storage.values * AF; m = int(len(o) * 0.8)
        cap2 = max(cap, float(np.nanmax(d.storage.values)) * AF)
        S, OUTF, _ = run_istarf(q, I2, epiweek(sg.index.values), cap2, float(o[0]), float(np.mean(d.inflow.dropna()) * AF))
        out.update(seg_days=len(sg), cap_grand_over_recmax=cap / max(float(np.nanmax(d.storage.values)) * AF, 1.0))
        out.update(score_block(o, S, "S_iterS_full")); out.update(score_block(o[m:], S[m:], "S_iterS_last20"))
        out.update(frac_full=float(np.mean(S >= cap2 * 0.999)), frac_empty=float(np.mean(S <= cap2 * 1e-4)))
    return out

# =========================================================================================== GRADES inflow vs observed
def inflow_check():
    """Compare GRADES-hydroDL flow with observed inflow at every placed rule reservoir (step inflow).

    Uses the days from 2000 on, and only records with at least 365 days of inflow. Returns a
    table with one row per reservoir and record source (GDROM cleaned record, MODROM daily
    record): mean and standard-deviation ratios, daily and monthly scores, and the reservoir's
    ISTARF mean flows.
    """
    GM = pd.read_csv(G/"reservoir_grades_map.csv", index_col=0); GM.index = GM.index.astype(str)
    RN = pd.read_csv(G/"rule_reservoir_nodes.csv", dtype={"node": str}).dropna(subset=["node"])
    Q = pd.concat([pd.read_parquet(f) for f in sorted((G/"qout").glob("qout_*.parquet"))]).sort_index(); Q.columns = Q.columns.astype("int64")
    days = pd.date_range("2000-01-01", Q.index.max(), freq="D")
    MD = pd.read_csv(G/"gdrom_v2/reservoir_metadata.csv").set_index("GRAND_ID")
    IP_ = istarf_params(); rows = []
    for r in RN.itertuples():
        nd = r.node
        if nd not in GM.index or pd.isna(GM.comid_v07[nd]) or int(GM.comid_v07[nd]) not in Q.columns: continue
        ratio = GM.area_ratio[nd]; used = float(ratio) if np.isfinite(ratio) and 0.2 < ratio < 5 else 1.0
        nat = pd.Series(Q[int(GM.comid_v07[nd])].reindex(days).interpolate(limit=5).values, index=days) * DAY / AF   # acre-ft/day
        for src, f, col, dcol in (("gdrom", CL/f"{r.GRAND_ID}.csv", "Inflow", "Date"),
                                  ("modrom", MR/f"daily/daily_records/{r.GRAND_ID}.csv", "inflow", "time")):
            if not f.exists(): continue
            o = pd.read_csv(f, parse_dates=[dcol]).set_index(dcol)[col]
            o = o[(o.index >= days[0]) & (o.index <= days[-1])].dropna()
            if len(o) < 365: continue
            s = nat.reindex(o.index) * used; k = s.notna()
            o, s = o[k], s[k]
            om, sm = o.resample("MS").mean(), s.resample("MS").mean()
            rows.append(dict(GRAND_ID=r.GRAND_ID, source=src, test=r.test, category=MD.CATEGORY.get(r.GRAND_ID, ""),
                             inflow_src=MD.INFLOW_SOURCE.get(r.GRAND_ID, ""), node=nd, area_ratio=ratio, ratio_used=used, n_days=len(o),
                             obs_mean=o.mean(), grades_mean=s.mean(), bias_ratio=s.mean() / o.mean() if o.mean() > 0 else np.nan,
                             r_daily=np.corrcoef(o, s)[0, 1], kge_daily=kge(o.values, s.values), nse_daily=nse(o.values, s.values),
                             r_monthly=np.corrcoef(om, sm)[0, 1] if len(om) > 3 else np.nan, kge_monthly=kge(om.values, sm.values),
                             neg_obs_frac=float(np.mean(o < 0)), sd_ratio=s.std() / o.std() if o.std() > 0 else np.nan,
                             istarf_obs_meanflow_cms=IP_.Obs_MEANFLOW_CUMECS.get(r.GRAND_ID, np.nan) if r.GRAND_ID in IP_.index else np.nan,
                             istarf_grand_meanflow_cms=IP_.GRanD_MEANFLOW_CUMECS.get(r.GRAND_ID, np.nan) if r.GRAND_ID in IP_.index else np.nan,
                             grades_mean_cms_all=float(nat.mean() * AF / DAY * used)))
    return pd.DataFrame(rows)

def q(x, c):
    """Return the text "median [25th percentile, 75th percentile]" of column c of table x."""
    return f"{np.nanmedian(x[c]):.3f} [{np.nanpercentile(x[c], 25):.3f}, {np.nanpercentile(x[c], 75):.3f}]"

if __name__ == "__main__":
    if step == "parser":
        ids = sorted({f.stem.split("_")[0] for f in (RD/"modules").glob("*.txt")}, key=int)
        with Pool(NP) as pool: rows = pool.map(parser_one, ids, chunksize=8)
        D = pd.DataFrame(rows); D.to_csv(OUT/"parser_equivalence.csv", index=False)
        ok = D[D.status == "ok"]
        print(f"rule sets {len(D)} | status:\n{D.status.str[:40].value_counts().to_string()}")
        print(f"points {ok.n.sum():,} | equal {ok.equal.sum():,} | reference returned None {ok.ref_none.sum():,} "
              f"(in {int((ok.ref_none > 0).sum())} rule sets) | MISMATCH {ok.mismatch.sum():,} (in {int((ok.mismatch > 0).sum())} rule sets)")
        print(f"reference script would crash (NameError, un-lower-cased Release expression): {ok.ref_script_nameerror.sum():,} points in "
              f"{int((ok.ref_script_nameerror > 0).sum())} rule sets")
        if (ok.mismatch > 0).any(): print(ok[ok.mismatch > 0].sort_values("mismatch", ascending=False).head(10).to_string())
    elif step == "gdrom":
        ids = sorted([f.stem for f in CL.glob("*.csv")], key=int)
        with Pool(NP) as pool: rows = pool.map(gdrom_one, ids, chunksize=4)
        D = pd.DataFrame(rows); MD = pd.read_csv(G/"gdrom_v2/reservoir_metadata.csv").set_index("GRAND_ID")
        D["CATEGORY"] = D.GRAND_ID.astype(int).map(MD.CATEGORY); D["paper_NSE"] = D.GRAND_ID.astype(int).map(MD.NSE)
        D["paper_PBIAS"] = D.GRAND_ID.astype(int).map(MD.PBIAS); D.to_csv(OUT/"gdrom_observed_inflow.csv", index=False)
        ok = D[D.status == "ok"]; print(D.status.str[:40].value_counts().to_string())
        for cat, x in ok.groupby("CATEGORY"):
            print(f"\n{cat} (n={len(x)})  median [IQR]")
            for c in ["paper_NSE", "rel_realS_unclipped_nse", "rel_realS_last20_nse", "rel_realS_nse", "rel_realS_kge",
                      "S_iterS_full_nse", "S_iterS_last20_nrmse", "S_iterS_restart_last20_nrmse", "S_ref_unclipped_last20_nrmse",
                      "S_iterS_last20_nse", "S_iterS_last20_kge", "frac_full", "frac_empty"]:
                if c in x: print(f"  {c:32s} {q(x, c)}")
        print(f"\nno-match events (realS, all dams): module-condition {ok.nomatch_cond_realS.sum():,} days in "
              f"{int((ok.nomatch_cond_realS > 0).sum())} dams; release-line {ok.nomatch_rel_realS.sum():,}; missing module {ok.missing_module_realS.sum():,}")
        print(f"days with release <= 0 (realS): median frac {ok.rel_realS_neg_frac.median():.4f}, dams with any {(ok.rel_realS_neg_frac > 0).sum()}")
        print(f"corr(paper NSE, our realS NSE unclipped) = {ok[['paper_NSE', 'rel_realS_unclipped_nse']].corr().iloc[0, 1]:.3f}")
    elif step == "modrom":
        P, _ = modrom_load(); ids = sorted(P, key=int)
        with Pool(NP) as pool: rows = pool.map(modrom_one, ids, chunksize=4)
        D = pd.DataFrame(rows); C = pd.read_csv(MR/"results/calibration_metrics.csv").rename(columns={"ID": "GRAND_ID", "kge": "paper_cal_kge"})
        D["GRAND_ID"] = D.GRAND_ID.astype(int); D = D.merge(C, on="GRAND_ID", how="left"); D.to_csv(OUT/"modrom_observed_inflow.csv", index=False)
        ok = D[D.status == "ok"]; print(D.status.value_counts().to_string())
        print(f"n {len(ok)}  median [IQR]")
        for c in ["paper_cal_kge", "rel_realS_norm_kge", "rel_realS_norm_nse", "S_iterS_full_nse", "S_iterS_last20_nse",
                  "S_iterS_last20_nrmse", "S_iterS_last20_kge", "frac_full", "frac_empty", "mean_out_over_in"]:
            if c in ok: print(f"  {c:28s} {q(ok, c)}")
        dk = (ok.rel_realS_norm_kge - ok.paper_cal_kge).abs()
        print(f"|our realS KGE - published calibration KGE|: median {dk.median():.4f}, 90th pct {dk.quantile(.9):.4f}, >0.05 in {(dk > 0.05).sum()} dams")
        for c, cc in (("our_in_max", "inflow_max"), ("our_s_max", "storage_max"), ("our_out_max", "outflow_max")):
            r_ = ok[c] / ok[cc]; print(f"  {c}/{cc}: median {r_.median():.4f}, frac within 1% {np.mean(np.abs(r_ - 1) < .01):.3f}")
        r_ = ok.out_max_closure / ok.outflow_max
        print(f"  closure outflow_max / true outflow_max: median {r_.median():.3f}, IQR [{r_.quantile(.25):.3f}, {r_.quantile(.75):.3f}]")
    elif step == "istarf":
        ids = sorted([f.stem for f in (MR/"istarf_sim/istarf_simulation/generalized/istarf_weekly_simulation").glob("*.csv")], key=int)
        with Pool(NP) as pool: A = pd.DataFrame(pool.map(istarf_weekly_repro, ids, chunksize=4))
        A.to_csv(OUT/"istarf_weekly_repro.csv", index=False); a = A[A.status == "ok"]
        print(f"(a) weekly reproduction: dams {len(a)} | non-binding weeks {a.nonbinding.sum():,} | reproduced {a.repro_match.sum():,} "
              f"({a.repro_match.sum() / max(a.nonbinding.sum(), 1):.4%}) | worst abs diff {a.max_abs_diff_nonbinding.max():.3g} MCM "
              f"| epiweek agreement median {a.epiweek_agree.median():.4f}, min {a.epiweek_agree.min():.4f}")
        with Pool(NP) as pool: A2 = pd.DataFrame(pool.map(istarf_fitted_repro, ids, chunksize=4))
        A2.to_csv(OUT/"istarf_fitted_repro.csv", index=False); a2 = A2[A2.status == "ok"].copy()
        a2["fit"] = a2.GRAND_ID.astype(int).map(istarf_params().fit); a2["frac"] = a2.repro_match / a2.nonbinding
        print(f"(a') fitted-parameter reproduction: dams {len(a2)} | non-binding weeks {a2.nonbinding.sum():,} | reproduced "
              f"{a2.repro_match.sum() / max(a2.nonbinding.sum(), 1):.4%}\n" + a2.groupby("fit").frac.describe().to_string())
        A["fit"] = A.GRAND_ID.astype(int).map(istarf_params().fit); A["frac"] = A.repro_match / A.nonbinding
        print("(a) generalized reproduction by ISTARF fit class:\n" + A[A.status == "ok"].groupby("fit").frac.describe().to_string())
        ids2 = sorted({f.stem for f in CL.glob("*.csv")}, key=int)
        with Pool(NP) as pool: B = pd.DataFrame(pool.map(istarf_daily_one, ids2, chunksize=4))
        BM = pd.read_csv(MR/"results/benchmark_metrics.csv"); BM.columns = ["GRAND_ID", "kge", "model", "kind"]
        print(BM.model.unique(), BM.kind.unique())
        for mdl in BM.model.unique():
            if "istarf" in str(mdl).lower():
                for kd in BM.kind.unique():
                    s_ = BM[(BM.model == mdl) & (BM.kind == kd)].set_index("GRAND_ID").kge
                    B[f"paper_{mdl}_{kd}_kge"] = B.GRAND_ID.astype(int).map(s_)
        B.to_csv(OUT/"istarf_observed_inflow.csv", index=False); b = B[B.status == "ok"]
        print(f"(b,c) daily engine, dams {len(b)}  median [IQR]")
        for c in [c for c in b.columns if c.startswith("paper_")] + ["rel_realS_grandmean_kge", "rel_realS_recmean_kge", "S_iterS_full_nse",
                  "S_iterS_last20_nse", "S_iterS_last20_nrmse", "S_iterS_last20_kge", "frac_full", "frac_empty", "cap_grand_over_recmax"]:
            if c in b: print(f"  {c:32s} {q(b, c)}")
    elif step == "inflow":
        D = inflow_check(); D.to_csv(OUT/"grades_vs_observed_inflow.csv", index=False)
        for (src, t), x in D.groupby(["source", "test"]):
            print(f"\n{src} records, test dam={t}: n={len(x)}  median [IQR]")
            for c in ["bias_ratio", "sd_ratio", "r_daily", "kge_daily", "nse_daily", "r_monthly", "kge_monthly", "neg_obs_frac"]:
                print(f"  {c:14s} {q(x, c)}")
        g = D[D.source == "gdrom"]
        if len(g): print("\nby GDROM inflow source:\n" + g.groupby("inflow_src").agg(n=("bias_ratio", "size"), bias=("bias_ratio", "median"),
                                                                                    kge=("kge_daily", "median")).to_string())
        ii = D.drop_duplicates("GRAND_ID").dropna(subset=["istarf_obs_meanflow_cms"])
        if len(ii): print(f"\nGRADES mean / ISTARF observed mean flow: median {np.median(ii.grades_mean_cms_all / ii.istarf_obs_meanflow_cms):.3f} (n={len(ii)})")
