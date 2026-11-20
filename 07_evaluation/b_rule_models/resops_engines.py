"""Daily simulation engines for three published reservoir rule models: GDROM, ISTARF and MODROM.

This module has no command line. grades_sim_v4.py and grades_sim_v4_hindcast.py import it to run
the rule-model benchmarks on GRADES-hydroDL inflow, and validate_engines.py imports it to check
the engines against the authors' code and published scores on observed inflow.

Daily step (all three engines):
  1. The release R_t is decided from the storage at the START of the day, as GDROM's reference
     function rule2model.simulate_release_and_storage does.
  2. R_t is clipped to [0, S_{t-1} + I_t].
  3. S_t = S_{t-1} + I_t - R_t. Water above capacity is spilled, and the spill counts as outflow.
Each engine returns three arrays: storage, outflow (release + spill) and the release decision
before it is limited to the available water.

Units are each rule model's own: GDROM and MODROM in acre-feet and acre-feet/day, ISTARF in m3
and m3/day.

Two modes, chosen with the `S_obs` argument:
  iterS  (S_obs=None) Storage is fed back from the simulation. This is what a model driven only
         by inflow does, and what the benchmarks use.
  realS  (S_obs given) The release of each day is computed from OBSERVED storage. This is the
         release-only evaluation the rule models' authors report. Storage is not simulated and is
         returned as NaN.

The rule models:
  GDROM v2  A decision tree per reservoir, stored as text files. A `module_conditions` file picks
            an operation module from Inflow, Storage, PDSI (Palmer Drought Severity Index) and DOY
            (day of year). Each `modules` file gives the release of one module as an expression of
            Inflow and Storage, chosen by conditions on the same four variables.
            The rule text is parsed, never executed: only numbers, the names Inflow and Storage,
            + - * /, unary minus and max/min are accepted in a release expression.
            Fallbacks: if no line of the module-condition file matches on a day, the LAST line is
            used. If the chosen module has no file, the first module is used. If no release line
            matches, the last one is used. Each event is counted (n_nomatch_cond,
            n_missing_module, n_nomatch_rel) so that its frequency can be reported.
  ISTARF    Turner et al. (2021), in the daily form of the mosartwmpy model. The upper and lower
            bounds of the normal operating range (NOR, in % of capacity) are harmonics of the MMWR
            epidemiological week (capped at 52), clipped to their fitted minimum and maximum. The
            forecast weekly inflow volume is 7 x today's inflow. Above the upper bound or below the
            lower bound, the weekly release is storage minus the storage at that bound, plus the
            forecast inflow. Inside the NOR the release is the mean inflow times (1 + release
            harmonic + c + p1 x position in the NOR + p2 x standardized forecast inflow). The
            release is then clamped to mean inflow x [1 + Release_min, 1 + Release_max].
  MODROM    Li & Villarini (2026). One operation mode per calendar month. Mode 2: R_n = k*I_n + c.
            Mode 3: R_n is piecewise linear in S_n (slopes m1 and m2, intercept c1, breakpoint
            `threshold`), continuous at the breakpoint. MODROM's mode 1 is mode 2 with k = 0.
            I_n = I/inflow_max, S_n = S/storage_max and R = R_n * outflow_max, exactly as the
            authors' fit_modrom.ipynb normalizes. The engine evaluates any mode other than 2 as
            mode 3, so callers must pass only modes 2 and 3 (both callers filter for this).
"""
import ast, operator, re, numpy as np, pandas as pd
from pathlib import Path

# ------------------------------------------------------------------ GDROM
COND = re.compile(r"\(\s*(Inflow|Storage|PDSI|DOY)\s*(<=|>=|<|>)\s*(-?[\d.]+(?:[eE][-+]?\d+)?)\s*\)")
OPS = {"<=": operator.le, ">=": operator.ge, "<": operator.lt, ">": operator.gt}
BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
FUN = {"max": max, "min": min}

def arith(expr):
    """Turn the text of a GDROM release expression into a function, without executing the text.

    The expression is parsed once into a syntax tree and evaluated once on a trial point, so that
    anything outside the allowed grammar (numbers, the names Inflow and Storage, + - * /, unary
    minus, max and min) raises ValueError here and not during a simulation.

    Returns a function of one argument, a dict with the keys "Inflow" and "Storage" (other keys
    are ignored), that returns the release as a float.
    """
    tree = ast.parse(expr.strip(), mode="eval").body
    def ev(n, env):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return float(n.value)
        if isinstance(n, ast.Name) and n.id in env: return env[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in BIN: return BIN[type(n.op)](ev(n.left, env), ev(n.right, env))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub): return -ev(n.operand, env)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUN and n.args and not n.keywords:
            return FUN[n.func.id](*(ev(a, env) for a in n.args))
        raise ValueError(f"unsupported rule expression: {expr!r}")
    f = lambda env: ev(tree, env); f({"Inflow": 1.0, "Storage": 1.0}); return f

def parse_rules(path, key):
    """Read one GDROM rule file into a list of (conditions, payload text), in file order.

    Each line of the file is a list of conditions "(Variable op number)" followed by
    "<key>: <payload>", where `key` is "module" or "Release". Each condition is returned as
    (variable name, comparison function, threshold). Empty lines are skipped. Raises ValueError
    if a line contains "nan" or has no `key` entry.
    """
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line: continue
        if re.search(r"\bnan\b", line, re.I): raise ValueError("rule set has nan thresholds/releases (unusable)")
        m = re.search(rf"{key}\s*[:=]\s*(.*)$", line)
        if not m: raise ValueError(f"cannot parse {path}: {line}")
        out.append(([(v, OPS[o], float(x)) for v, o, x in COND.findall(line[:m.start()])], m.group(1).strip()))
    return out

class GdromRules:
    """Rule set of one GDROM reservoir: module conditions and one list of release rules per module.

    A reservoir with no module-condition file has the single module "0".

    Attributes:
      cond                   module conditions, [(conditions, module payload), ...] in file order
      mods                   {module name: [(conditions, release function), ...]}
      max_storage_threshold  the largest Storage threshold in any condition (NaN if there is
                             none); the simulators use it as one of their estimates of capacity
      n_nomatch_cond, n_nomatch_rel, n_missing_module
                             number of calls that used each fallback (see the module docstring)
    """

    def __init__(self, rule_dir, rid):
        """Read the rule files of reservoir `rid` from the GDROM rule folder `rule_dir`.

        Module conditions come from module_conditions/<rid>.txt and the release rules of module
        <m> from modules/<rid>_<m>.txt. Raises ValueError if the reservoir has no module file or
        a file cannot be parsed.
        """
        rule_dir = Path(rule_dir); f = rule_dir/"module_conditions"/f"{rid}.txt"
        self.cond = parse_rules(f, "module") if f.exists() else [([], "0")]
        self.mods = {}
        for g in sorted((rule_dir/"modules").glob(f"{rid}_*.txt")):
            self.mods[g.stem.split("_")[-1]] = [(c, arith(p)) for c, p in parse_rules(g, "Release")]
        if not self.mods: raise ValueError("no modules")
        thr = [x for c, _ in self.cond for v, _, x in c if v == "Storage"] + \
              [x for m in self.mods.values() for c, _ in m for v, _, x in c if v == "Storage"]
        self.max_storage_threshold = max(thr) if thr else np.nan
        self.n_nomatch_cond = self.n_nomatch_rel = self.n_missing_module = 0

    def module(self, env):
        """Return the name of the operation module for one day.

        env: dict with the day's Inflow, Storage, PDSI and DOY. The module is that of the first
        line of the module-condition file whose conditions all hold, or of the last line if none
        does (counted in n_nomatch_cond).
        """
        for conds, pay in self.cond:
            if all(op(env[v], x) for v, op, x in conds): return pay.split(".")[0]
        self.n_nomatch_cond += 1; return self.cond[-1][1].split(".")[0]

    def release(self, env):
        """Return the release the rules give for one day (not limited to the available water).

        env: dict with the day's Inflow, Storage, PDSI and DOY. The module is chosen with
        module(); if it has no file, the first module is used (counted in n_missing_module). The
        release is that of the module's first line whose conditions all hold, or of its last line
        if none does (counted in n_nomatch_rel).
        """
        m = self.module(env)
        rules = self.mods.get(m)
        if rules is None: self.n_missing_module += 1; rules = self.mods[next(iter(self.mods))]
        for conds, f in rules:
            if all(op(env[v], x) for v, op, x in conds): return f(env)
        self.n_nomatch_rel += 1; return rules[-1][1](env)

def run_gdrom(rules, I, doy, pdsi, cap, s0, S_obs=None):
    """Run GDROM day by day and return storage, outflow and the release decision.

    rules: a GdromRules object. I: inflow, acre-ft/day. doy, pdsi: day of year and PDSI of each
    day. cap, s0: capacity and initial storage, acre-ft. S_obs: observed storage of each day
    (acre-ft) for realS mode, or None for iterS mode (see the module docstring).

    Returns three arrays: storage (acre-ft; NaN in realS mode), outflow (acre-ft/day; in realS
    mode the release decision) and the release decision, which is floored at 0 but not yet
    limited to the available water.
    """
    n = len(I); S = np.full(n, np.nan); OUT = np.empty(n); REL = np.empty(n); s = s0
    for t in range(n):
        st = S_obs[t] if S_obs is not None else s
        env = {"Inflow": I[t], "Storage": st, "DOY": doy[t], "PDSI": pdsi[t]}
        r = max(rules.release(env), 0.0); REL[t] = r
        if S_obs is not None: OUT[t] = r; continue
        r = min(r, s + I[t]); sn = s + I[t] - r; sp = 0.0
        if sn > cap: sp = sn - cap; sn = cap
        s = sn; S[t] = s; OUT[t] = r + sp
    return S, OUT, REL

# ------------------------------------------------------------------ ISTARF
def epiweek(dates):
    """Return the MMWR epidemiological week of each date, capped at 52 as mosartwmpy does.

    Weeks start on Sunday and week 1 is the week that contains January 4. Days before week 1 of
    their calendar year are counted from week 1 of the year before. Returns a float array.
    """
    out = np.empty(len(dates))
    for i, d in enumerate(pd.DatetimeIndex(dates)):
        j4 = pd.Timestamp(d.year, 1, 4); start = j4 - pd.Timedelta(days=(j4.dayofweek + 1) % 7)
        if d < start:
            j4 = pd.Timestamp(d.year - 1, 1, 4); start = j4 - pd.Timedelta(days=(j4.dayofweek + 1) % 7)
        out[i] = min((d - start).days // 7 + 1, 52)
    return out

def run_istarf(p, I, ew, cap, s0, qbar, S_obs=None):
    """Run ISTARF day by day and return storage, outflow and the release decision.

    p: the reservoir's row of ISTARF-CONUS.csv. I: inflow, m3/day. ew: epiweek of each day. cap,
    s0: capacity and initial storage, m3. qbar: mean inflow, m3/day, used to standardize
    releases. S_obs: observed storage of each day (m3) for realS mode, or None for iterS mode
    (see the module docstring).

    Returns three arrays: storage (m3; NaN in realS mode), outflow (m3/day; in realS mode the
    release decision) and the release decision, which is clamped to the release limits and
    floored at 0 but not yet limited to the available water.
    """
    def harm(pre):
        v = p[f"{pre}_mu"] + p[f"{pre}_alpha"] * np.sin(2*np.pi*ew/52) + p[f"{pre}_beta"] * np.cos(2*np.pi*ew/52)
        return np.minimum(p[f"{pre}_max"], np.maximum(p[f"{pre}_min"], v))
    hi, lo = harm("NORhi"), harm("NORlo")
    hr = (p["Release_alpha1"]*np.sin(2*np.pi*ew/52) + p["Release_alpha2"]*np.sin(4*np.pi*ew/52)
          + p["Release_beta1"]*np.cos(2*np.pi*ew/52) + p["Release_beta2"]*np.cos(4*np.pi*ew/52))
    rmax_p = p["Release_max"] if np.isfinite(p["Release_max"]) else np.inf
    rmin_p = p["Release_min"] if np.isfinite(p["Release_min"]) else -np.inf
    wk = 7 * qbar; rmin, rmax = wk * (1 + rmin_p) / 7, wk * (1 + rmax_p) / 7
    n = len(I); S = np.full(n, np.nan); OUT = np.empty(n); REL = np.empty(n); s = s0
    for t in range(n):
        st = S_obs[t] if S_obs is not None else s
        fw = 7 * I[t]; span = hi[t] - lo[t]
        av = (100 * st / cap - lo[t]) / span if span > 0 else 0.5
        if av > 1: r = (st - cap * hi[t] / 100 + fw) / 7
        elif av < 0: r = (st - cap * lo[t] / 100 + fw) / 7
        else: r = wk * (1 + hr[t] + p["Release_c"] + p["Release_p1"] * av + p["Release_p2"] * (fw / wk - 1)) / 7
        r = min(max(r, rmin), rmax); r = max(r, 0.0); REL[t] = r
        if S_obs is not None: OUT[t] = r; continue
        r = min(r, s + I[t]); sn = s + I[t] - r; sp = 0.0
        if sn > cap: sp = sn - cap; sn = cap
        s = sn; S[t] = s; OUT[t] = r + sp
    return S, OUT, REL

# ------------------------------------------------------------------ MODROM
def modrom_release_n(mode, prm, i_n, s_n):
    """Return the normalized MODROM release of one day.

    mode, prm: operation mode and parameters of the day's month. i_n, s_n: normalized inflow and
    storage. Mode 2 is linear in i_n (prm = slope k, intercept c). Any other mode is evaluated
    as mode 3: piecewise linear in s_n (prm = slopes m1 and m2, intercept c1, breakpoint),
    continuous at the breakpoint.
    """
    if mode == 2:
        k, c = prm[0], prm[1]; return k * i_n + c
    m1, m2, c1, thr = prm[0], prm[1], prm[2], prm[3]
    return m1 * s_n + c1 if s_n <= thr else m2 * s_n + (m1 * thr + c1 - m2 * thr)

def run_modrom(modes, params, I, months, in_max, s_max, out_max, cap, s0, S_obs=None):
    """Run MODROM day by day and return storage, outflow and the release decision.

    modes, params: 12 monthly entries. I: inflow, acre-ft/day. months: month (1-12) of each day.
    in_max, s_max, out_max: normalization maxima (acre-ft/day, acre-ft, acre-ft/day). cap, s0:
    capacity and initial storage, acre-ft. S_obs: observed storage of each day (acre-ft) for
    realS mode, or None for iterS mode (see the module docstring).

    Returns three arrays: storage (acre-ft; NaN in realS mode), outflow (acre-ft/day; in realS
    mode the release decision) and the release decision, which is floored at 0 but not yet
    limited to the available water.
    """
    n = len(I); S = np.full(n, np.nan); OUT = np.empty(n); REL = np.empty(n); s = s0
    for t in range(n):
        st = S_obs[t] if S_obs is not None else s
        mi = months[t] - 1
        r = max(modrom_release_n(modes[mi], params[mi], I[t] / in_max, st / s_max) * out_max, 0.0); REL[t] = r
        if S_obs is not None: OUT[t] = r; continue
        r = min(r, s + I[t]); sn = s + I[t] - r; sp = 0.0
        if sn > cap: sp = sn - cap; sn = cap
        s = sn; S[t] = s; OUT[t] = r + sp
    return S, OUT, REL

# ------------------------------------------------------------------ metrics
# Scores used by validate_engines.py (o = observed, s = simulated; days where either is missing are
# ignored). These are not the scoring functions of the paper's results; those are in ../c_scoring/.
def nse(o, s):
    """Return the Nash-Sutcliffe efficiency of simulated s against observed o.

    Days where either value is missing are ignored. Returns NaN with fewer than 3 paired days or
    constant observations.
    """
    o, s = np.asarray(o, float), np.asarray(s, float); k = np.isfinite(o) & np.isfinite(s); o, s = o[k], s[k]
    return 1 - np.sum((o - s) ** 2) / np.sum((o - o.mean()) ** 2) if len(o) > 2 and o.std() > 0 else np.nan

def kge(o, s):
    """Return the Kling-Gupta efficiency of simulated s against observed o.

    1 - sqrt((r - 1)^2 + (sd_s/sd_o - 1)^2 + (mean_s/mean_o - 1)^2). Days where either value is
    missing are ignored. Returns NaN with fewer than 3 paired days, a constant series or a zero
    observed mean. validate_engines.py uses it to compare with the scores the rule models'
    authors publish; it is not a metric of the paper.
    """
    o, s = np.asarray(o, float), np.asarray(s, float); k = np.isfinite(o) & np.isfinite(s); o, s = o[k], s[k]
    if len(o) < 3 or o.std() == 0 or s.std() == 0 or o.mean() == 0: return np.nan
    r = np.corrcoef(o, s)[0, 1]; return 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)

def nrmse_sd(o, s):
    """Return the RMSE of simulated s against observed o, divided by the standard deviation of o.

    Days where either value is missing are ignored. Returns NaN with fewer than 3 paired days or
    constant observations.
    """
    o, s = np.asarray(o, float), np.asarray(s, float); k = np.isfinite(o) & np.isfinite(s); o, s = o[k], s[k]
    return np.sqrt(np.mean((o - s) ** 2)) / o.std() if len(o) > 2 and o.std() > 0 else np.nan
