"""Rule-model engines (07_evaluation/b_rule_models/resops_engines.py): GDROM, ISTARF, MODROM.

Checked on hand-built inputs:
* water balance, every day, every engine: S_t = S_{t-1} + I_t - outflow_t; 0 <= S_t <= capacity;
  the executed release is min(decision, S_{t-1} + I_t); water is spilled only when the reservoir is
  full, and the spill counts as outflow;
* the release is decided from the storage at the START of the day: the release decisions of a
  simulation equal those of the release-only mode (realS) fed with the simulated start-of-day
  storage;
* GDROM: the rule-text parser equals Python arithmetic on random expressions and refuses anything
  else; module and release selection; the three documented fallbacks and their counters;
* ISTARF: normal operating range bounds (% of capacity), the three release branches, the clamp,
  by hand; the MMWR epidemiological week against an independent implementation;
* MODROM: modes 2 and 3 by hand, continuity at the breakpoint, normalisation,
  calendar month -> mode.
"""
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from conftest import REPO

import sys
sys.path.insert(0, str(REPO / "07_evaluation/b_rule_models"))
import resops_engines as E                                  # noqa: E402


# ------------------------------------------------------------------ helpers
def write_rules(d, rid, cond_lines, modules):
    """Write GDROM rule files under `d` and return the GdromRules read from them.

    The files are module_conditions/<rid>.txt (not written if cond_lines is None) and
    modules/<rid>_<m>.txt for each module m.
    """
    (d / "module_conditions").mkdir(parents=True, exist_ok=True); (d / "modules").mkdir(parents=True, exist_ok=True)
    if cond_lines is not None:
        (d / "module_conditions" / f"{rid}.txt").write_text("\n".join(cond_lines) + "\n")
    for m, lines in modules.items():
        (d / "modules" / f"{rid}_{m}.txt").write_text("\n".join(lines) + "\n")
    return E.GdromRules(d, rid)


def inflow_series(n=900, seed=0, scale=1000.0):
    """Return an inflow series with a dry spell and a flood (units of the engine)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    q = scale * (1 + 0.8 * np.sin(2 * np.pi * t / 365)) * rng.lognormal(0, 0.6, n)
    q[100:160] = 0.0                                          # drought: reservoir may empty
    q[400:420] *= 30                                          # flood: reservoir must spill
    return q


def check_water_balance(S, OUT, REL, I, cap, s0):
    """Assert the daily water balance of one simulation and return the spill.

    S_t = S_{t-1} + I_t - OUT_t and 0 <= S_t <= cap must hold. The spill is the outflow minus the
    executed release min(REL, S_{t-1} + I_t); it must not be negative and may occur only when
    the reservoir is full.
    """
    prev = np.r_[s0, S[:-1]]
    assert np.all(np.isfinite(S)) and np.all(S >= 0) and np.all(S <= cap * (1 + 1e-12))
    np.testing.assert_allclose(S - prev, I - OUT, rtol=0, atol=1e-9 * max(cap, 1.0))
    executed = np.minimum(REL, prev + I)
    spill = OUT - executed
    assert np.all(spill >= -1e-9 * cap)
    assert np.all(np.isclose(S[spill > 1e-9 * cap], cap, rtol=1e-12))     # spill only when full
    return spill


def engines(tmp_path):
    """Return one configured run of each engine (GDROM, ISTARF, MODROM).

    Each item is (name, run, I, cap, s0), where run(S_obs=None) returns (S, OUT, REL).
    """
    n = 900
    days = pd.date_range("2001-01-01", periods=n)
    I_af = inflow_series(n, 1)
    rules = write_rules(tmp_path / "g", 7, ["(DOY <= 180) module: 1", "(DOY > 180) module: 2"],
                        {"1": ["(Storage <= 30000) Release: 0.5*Inflow", "(Storage > 30000) Release: 0.5*Inflow + 0.004*Storage"],
                         "2": ["Release: max(0.8*Inflow, 0.01*Storage)"]})     # releases that depend on storage every day
    doy = days.dayofyear.values.astype(float); pdsi = np.zeros(n)
    g = lambda S_obs=None: E.run_gdrom(rules, I_af, doy, pdsi, 60000.0, 36000.0, S_obs)
    p = pd.Series(dict(NORhi_mu=80., NORhi_alpha=5., NORhi_beta=3., NORhi_min=60., NORhi_max=95.,
                       NORlo_mu=40., NORlo_alpha=4., NORlo_beta=-2., NORlo_min=20., NORlo_max=55.,
                       Release_alpha1=.1, Release_alpha2=.05, Release_beta1=-.1, Release_beta2=.02, Release_c=.0,
                       Release_p1=.3, Release_p2=.2, Release_min=-.6, Release_max=1.5))
    I_m3 = I_af * 1233.48184; ew = E.epiweek(days); cap_m3 = 8e7
    i = lambda S_obs=None: E.run_istarf(p, I_m3, ew, cap_m3, 0.5 * cap_m3, float(I_m3.mean()), S_obs)
    modes = [2, 2, 3, 3, 3, 2, 2, 3, 3, 3, 2, 2]
    prm = [[0.9, 0.05] if m == 2 else [0.2, 1.4, 0.05, 0.6] for m in modes]
    months = days.month.values
    m = lambda S_obs=None: E.run_modrom(modes, prm, I_af, months, float(I_af.max()), 60000.0, float(I_af.max()) * .5, 60000.0, 30000.0, S_obs)
    return [("gdrom", g, I_af, 60000.0, 36000.0), ("istarf", i, I_m3, cap_m3, 0.5 * cap_m3), ("modrom", m, I_af, 60000.0, 30000.0)]


# ------------------------------------------------------------------ water balance, timing, realS
def test_water_balance_every_day_every_engine(tmp_path):
    """Check the water balance of GDROM, ISTARF and MODROM on every day of a drought and a flood.

    The flood must make every engine spill, and outflow and release decisions must never be
    negative.
    """
    for name, run, I, cap, s0 in engines(tmp_path):
        S, OUT, REL = run()
        spill = check_water_balance(S, OUT, REL, I, cap, s0)
        assert (spill > 0).any(), f"{name}: the flood should fill the reservoir"     # the test exercises spilling
        assert np.all(OUT >= 0) and np.all(REL >= 0), name


def test_release_decided_from_start_of_day_storage(tmp_path):
    """Check that each engine decides the release from the storage at the start of the day.

    The release-only mode (realS), fed with the simulated start-of-day storage, must give the
    same decisions, no storage, and an outflow equal to the decision.
    """
    for name, run, I, cap, s0 in engines(tmp_path):
        S, OUT, REL = run()
        S2, OUT2, REL2 = run(S_obs=np.r_[s0, S[:-1]])                  # release-only mode on that storage
        assert np.all(np.isnan(S2)), name                                # realS does not simulate storage
        np.testing.assert_allclose(REL2, REL, rtol=1e-12, err_msg=name)
        np.testing.assert_array_equal(OUT2, REL2)                         # realS outflow = release decision


def test_storage_never_negative_in_drought(tmp_path):
    """Check that a release far above inflow empties the reservoir to exactly 0, never below."""
    rules = write_rules(tmp_path, 1, None, {"0": ["Release: 5000"]})     # releases far more than comes in
    I = np.r_[np.full(50, 100.0), np.zeros(50)]
    S, OUT, REL = E.run_gdrom(rules, I, np.arange(100.) + 1, np.zeros(100), 1e5, 2e4)
    check_water_balance(S, OUT, REL, I, 1e5, 2e4)
    assert S.min() == 0.0 and np.all(OUT[S == 0] <= np.r_[2e4, S[:-1]][S == 0] + I[S == 0] + 1e-9)


# ------------------------------------------------------------------ GDROM parser and rules
def _rand_expr(rng, depth=0):
    r = rng.random()
    if depth > 3 or r < 0.3:
        return rng.choice(["Inflow", "Storage", f"{rng.uniform(-50, 50):.4f}", f"{rng.randint(1, 9)}", "1.5e2"])
    op = rng.choice(["+", "-", "*", "/", "neg", "max", "min"])
    a, b = _rand_expr(rng, depth + 1), _rand_expr(rng, depth + 1)
    if op == "neg":
        return f"-({a})"
    if op in ("max", "min"):
        return f"{op}({a}, {b}, {_rand_expr(rng, depth + 1)})"
    if op == "/":
        b = f"({b}) * ({b}) + 1.0"                                        # keep away from zero
    return f"({a}) {op} ({b})"


def test_arith_equals_python_arithmetic_on_random_expressions():
    """Check the GDROM rule parser against Python arithmetic on 3,000 random expressions."""
    rng = random.Random(0)
    for _ in range(3000):
        ex = _rand_expr(rng)
        env = {"Inflow": rng.uniform(0, 5e4), "Storage": rng.uniform(0, 5e5)}
        want = eval(ex, {"__builtins__": {}, "max": max, "min": min}, dict(env))
        assert E.arith(ex)(env) == pytest.approx(want, rel=1e-12, abs=1e-9), ex


@pytest.mark.parametrize("bad", ["__import__('os').system('true')", "Inflow ** 2", "Inflow.real", "(lambda: 1)()",
                                 "abs(Inflow)", "Inflow if Storage else 0", "Inflow > 3", "[Inflow][0]",
                                 "max(Inflow, key=None)", "PDSI * 2", "open('x')"])
def test_arith_refuses_anything_else(bad):
    """Check that the rule parser refuses an expression outside its grammar."""
    with pytest.raises((ValueError, SyntaxError)):
        E.arith(bad)


def test_gdrom_module_and_release_selection_and_fallbacks(tmp_path):
    """Check GDROM's module and release selection and its three fallbacks with their counters.

    The module is chosen by the DOY/PDSI conditions and the release line by the Inflow/Storage
    conditions; the fallbacks are the last module line, the first module and the last release
    line.
    """
    r = write_rules(tmp_path, 5, ["(DOY <= 100) (PDSI > -2) module: 1.0", "(DOY > 100) module: 2.0", "(DOY <= 100) module: 9.0"],
                    {"1": ["(Inflow <= 10) (Storage < 500) Release: 1", "(Inflow > 10) Release: 2*Inflow", "(Storage >= 500) Release: 3"],
                     "2": ["(Storage <= 100) Release: min(Inflow, 7)", "(Storage > 100) (Storage <= 200) Release: Storage/100"]})
    rel = lambda **e: r.release({"Inflow": 0.0, "Storage": 0.0, "DOY": 1.0, "PDSI": 0.0, **e})
    assert rel(Inflow=5, Storage=10) == 1 and rel(Inflow=20) == 40 and rel(Inflow=5, Storage=600) == 3
    assert rel(DOY=200, Inflow=50, Storage=50) == 7 and rel(DOY=200, Storage=150) == 1.5
    assert (r.n_nomatch_cond, r.n_missing_module, r.n_nomatch_rel) == (0, 0, 0)
    # module 2, no release line matches (Storage > 200): the last release line is used and counted
    assert rel(DOY=200, Storage=300) == 3.0 and r.n_nomatch_rel == 1
    # PDSI <= -2 on DOY <= 100: line 3 matches -> module 9, which has no file: first module used, counted
    assert rel(DOY=50, PDSI=-3, Inflow=5, Storage=10) == 1 and r.n_missing_module == 1
    assert r.max_storage_threshold == 500
    r2 = write_rules(tmp_path / "b", 6, ["(DOY <= 10) module: 1"], {"1": ["Release: 4"], "2": ["Release: 8"]})
    assert r2.release({"Inflow": 0, "Storage": 0, "DOY": 50, "PDSI": 0}) == 4 and r2.n_nomatch_cond == 1   # last line's module


def test_gdrom_rule_set_with_nan_is_refused(tmp_path):
    """Check that a rule set with a NaN threshold is refused."""
    with pytest.raises(ValueError):
        write_rules(tmp_path, 3, None, {"0": ["(Storage <= nan) Release: 1"]})


# ------------------------------------------------------------------ ISTARF
P0 = dict(NORhi_mu=80., NORhi_alpha=0., NORhi_beta=0., NORhi_min=0., NORhi_max=100.,
          NORlo_mu=40., NORlo_alpha=0., NORlo_beta=0., NORlo_min=0., NORlo_max=100.,
          Release_alpha1=0., Release_alpha2=0., Release_beta1=0., Release_beta2=0., Release_c=0.1,
          Release_p1=0.3, Release_p2=0.2, Release_min=-0.5, Release_max=2.0)


def test_istarf_branches_by_hand():
    """Check ISTARF's release branches by hand.

    Storage above, inside and below the normal operating range, and the clamp at Release_min and
    Release_max.
    """
    p = pd.Series(P0); cap, qbar = 1e8, 2e5
    I = np.array([3e5, 3e5, 3e5, 3e5, 1e5])
    S_obs = np.array([0.81, 0.95, 0.2, 0.6, 0.5]) * cap
    _, _, REL = E.run_istarf(p, I, np.full(5, 10.0), cap, 0.5 * cap, qbar, S_obs)
    rmin, rmax = qbar * 0.5, qbar * 3.0
    above = lambda s, i: (s - 0.8 * cap + 7 * i) / 7
    assert rmin < above(S_obs[0], I[0]) < rmax
    assert REL[0] == pytest.approx(above(S_obs[0], I[0]))                            # above NOR
    assert REL[1] == pytest.approx(rmax)                                              # above NOR, clamped at Release_max
    assert REL[2] == pytest.approx(rmin)                                              # below NOR: (S - S_lo + 7I)/7 < 0 -> Release_min
    inside = lambda av, i: qbar * (1 + 0.1 + 0.3 * av + 0.2 * (7 * i / (7 * qbar) - 1))
    assert REL[3] == pytest.approx(inside(0.5, I[3]))                                 # middle of the NOR
    assert REL[4] == pytest.approx(inside(0.25, I[4]))


def test_istarf_nor_harmonics_and_bounds():
    """Check that the NOR bounds are the clipped weekly harmonics in % of capacity."""
    p = pd.Series({**P0, "NORhi_alpha": 10., "NORhi_beta": 4., "NORhi_max": 88., "NORlo_beta": -30., "NORlo_min": 25.})
    cap, qbar = 1e8, 2e5
    for w in (1.0, 13.0, 26.0, 40.0):
        hi = min(88., max(0., 80 + 10 * np.sin(2 * np.pi * w / 52) + 4 * np.cos(2 * np.pi * w / 52)))
        lo = min(100., max(25., 40 - 30 * np.cos(2 * np.pi * w / 52)))
        # storage just above / just below the upper bound decides the branch
        _, _, r_above = E.run_istarf(p, np.array([qbar]), np.array([w]), cap, 0, qbar, np.array([(hi + .5) / 100 * cap]))
        _, _, r_in = E.run_istarf(p, np.array([qbar]), np.array([w]), cap, 0, qbar, np.array([(hi - .5) / 100 * cap]))
        assert r_above[0] == pytest.approx(min(max((.005 * cap + 7 * qbar) / 7, qbar * .5), qbar * 3))
        av = (hi - .5 - lo) / (hi - lo)
        assert r_in[0] == pytest.approx(qbar * (1 + .1 + .3 * av))


def _mmwr(d: date) -> int:
    """Return the MMWR week by an independent rule.

    Weeks run Sunday-Saturday; week 1 is the week that contains 4 January.
    """
    def start(y):
        j4 = date(y, 1, 4); return j4 - timedelta(days=(j4.weekday() + 1) % 7)
    y = d.year
    if d >= start(y + 1):
        return 1
    if d < start(y):
        y -= 1
    return (d - start(y)).days // 7 + 1


def test_epiweek_known_dates():
    """Check epiweek() on known dates (MMWR week, capped at 52)."""
    got = E.epiweek(pd.to_datetime(["2023-01-01", "2023-01-07", "2023-01-08", "2021-01-02", "2021-01-03", "2020-12-31"]))
    assert list(got) == [1, 1, 2, 52, 1, 52]                 # 2021-01-02 and 2020-12-31 are MMWR 2020 week 53 -> capped at 52


def test_epiweek_matches_mmwr_except_year_end_week1():
    """Check epiweek() against an independent MMWR week (capped at 52) for 1979-2026.

    December days that belong to week 1 of the next MMWR year are left out here; the next test
    covers them.
    """
    d = pd.date_range("1979-01-01", "2026-12-31")
    got = E.epiweek(d)
    want = np.array([min(_mmwr(x.date()), 52) for x in d])
    nextyear_w1 = np.array([x.month == 12 and _mmwr(x.date()) == 1 for x in d])
    np.testing.assert_array_equal(got[~nextyear_w1], want[~nextyear_w1])


@pytest.mark.xfail(strict=True, reason=(
    "FINDING: late-December days that belong to week 1 of the NEXT MMWR year (e.g. 2024-12-29..31) get "
    "week 52 from epiweek(), against its own docstring (week 1 contains 4 January). 0-3 days in some years; "
    "the ISTARF harmonics then use week 52 instead of week 1 on those days."))
def test_epiweek_year_end_days_belong_to_next_years_week1():
    """Check epiweek() against the MMWR week on every day of 1979-2026 (expected failure)."""
    d = pd.date_range("1979-01-01", "2026-12-31")
    got = E.epiweek(d)
    want = np.array([min(_mmwr(x.date()), 52) for x in d])
    np.testing.assert_array_equal(got, want)


# ------------------------------------------------------------------ MODROM
def test_modrom_modes_by_hand_and_continuity():
    """Check MODROM's modes 2 and 3 by hand, and that mode 3 is continuous at its breakpoint."""
    assert E.modrom_release_n(2, [0.7, 0.1], 0.5, 0.9) == pytest.approx(0.7 * 0.5 + 0.1)
    prm = [0.2, 1.4, 0.05, 0.6]
    assert E.modrom_release_n(3, prm, 9.9, 0.3) == pytest.approx(0.2 * 0.3 + 0.05)          # below the breakpoint
    assert E.modrom_release_n(3, prm, 9.9, 0.8) == pytest.approx(1.4 * 0.8 + (0.2 * 0.6 + 0.05 - 1.4 * 0.6))
    a, b = E.modrom_release_n(3, prm, 0, 0.6), E.modrom_release_n(3, prm, 0, 0.6 + 1e-12)
    assert a == pytest.approx(b, abs=1e-10)                                                # continuous at the breakpoint


def test_modrom_normalisation_and_calendar_month():
    """Check MODROM's normalisation and its use of the calendar month.

    R = R_n(I/inflow_max, S/storage_max) x outflow_max, and each calendar month uses its own
    monthly mode (January the first).
    """
    modes = [2] * 12
    prm = [[0.0, 0.01 * (k + 1)] for k in range(12)]                 # month k releases 0.01 (k+1) x outflow_max
    days = pd.date_range("2003-01-01", "2003-12-31")
    I = np.full(len(days), 50.0)
    _, _, REL = E.run_modrom(modes, prm, I, days.month.values, 100.0, 1e4, 2000.0, 1e4, 5e3,
                             S_obs=np.full(len(days), 5e3))
    np.testing.assert_allclose(REL, 0.01 * days.month.values * 2000.0)
    _, _, REL = E.run_modrom([2] * 12, [[0.8, 0.0]] * 12, I, days.month.values, 100.0, 1e4, 2000.0, 1e4, 5e3,
                             S_obs=np.full(len(days), 5e3))
    np.testing.assert_allclose(REL, 0.8 * (50.0 / 100.0) * 2000.0)   # R = (k I/inflow_max + c) outflow_max
    _, _, REL = E.run_modrom([3] * 12, [[0.5, 0.5, 0.0, 0.5]] * 12, I, days.month.values, 100.0, 1e4, 2000.0, 1e4, 5e3,
                             S_obs=np.full(len(days), 4e3))
    np.testing.assert_allclose(REL, 0.5 * (4e3 / 1e4) * 2000.0)      # S normalised by storage_max
