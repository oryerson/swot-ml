"""7. The paper's scorers (07_evaluation/c_scoring) on hand-built series with known answers.

Definitions checked (on the shared days, each series minus its own mean over those days):
    NSE   = 1 - sum((S'-O')^2) / sum(O'^2)       (direct NSE)
    VR    = sd(S') / sd(O')                       (column `amp`)
    nRMSE = RMSE(S', O') / (max O' - min O')
    r     = Pearson correlation
    and the identity NSE = 2*VR*r - VR^2.
Overlap rule (overlap50): a model is scored only if it shares with the gauge at least 50% of
the window's days. Harmonic irregularity = sd(gauge - fit) / sd(fit), fit = least squares of a
constant + annual + semiannual harmonic on water years 1981-2020 with >= 292 days (>= 5 years).
Training coverage = sd(SWOT storage) / sd(gauge, 1981-2020).

The scorers are scripts that do their work at import time, so:
* `score()` (SWOT Period scorer) and `metrics()` (full-record scorer) are cut out of the
  source text and called directly;
* the full-record scorer and the 2010-2020 scorer are also run end to end, unchanged, in a
  subprocess on a synthetic work tree (gauges, two seeds of predictions, rule models).
"""
import ast
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import REPO, load_function, run_statements

C = REPO / "07_evaluation/c_scoring"
SCORE_FILES = [C / "overlap50/score_x26span.py", C / "full_record/score_x26span.py"]
METRIC_FILES = [C / "full_record/score_fullrecord.py", C / "overlap50/score_fullrecord.py"]
# (normal_year_source/score_fullrecord.py is an earlier scorer kept only as the source of normal_year())


def expected(o: pd.Series, s: pd.Series) -> dict:
    """Return the paper's metrics by an independent implementation.

    The result holds NSE, VR (`amp`), nRMSE, r and the number of shared days `n`.
    """
    j = pd.concat([o.rename("o"), s.rename("s")], axis=1).dropna()
    O = j.o.values - j.o.values.mean(); S = j.s.values - j.s.values.mean()
    return dict(nse=1 - ((S - O) ** 2).sum() / (O ** 2).sum(), amp=S.std(ddof=1) / O.std(ddof=1),
                nrmse=np.sqrt(((S - O) ** 2).mean()) / (O.max() - O.min()), r=np.corrcoef(O, S)[0, 1], n=len(j))


def _series(n=400, seed=0, start="2023-07-01"):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    o = 500 + 40 * np.sin(2 * np.pi * t / 365.25) + 10 * np.cos(4 * np.pi * t / 365.25) + rng.normal(0, 5, n)
    return pd.Series(o, index=pd.date_range(start, periods=n))


# ------------------------------------------------------------------ score() of the SWOT Period scorer
@pytest.fixture(params=SCORE_FILES, ids=lambda p: p.parent.name)
def score(request):
    """Return score() cut out of one copy of the SWOT Period scorer.

    The function's globals are attached to it as `ns`, so that a test can set the minimum
    number of shared days (MIN_DAM).
    """
    ns = {"np": np, "pd": pd, "MIN_DAM": [0]}
    f = load_function(request.param, "score", ns)
    f.ns = ns
    return f


def test_score_closed_forms(score):
    """Check score() on cases with known answers.

    Offset: NSE 1, VR 1, r 1. Doubled swing: NSE 0, VR 2. Mirror image: NSE -3, r -1. Constant
    prediction: flagged flat, r = VR = 0.
    """
    o = _series()
    oc = o - o.mean()
    s = score(o, o + 123.0)                                     # offset only: perfect after centring
    assert s["nse"] == pytest.approx(1) and s["amp"] == pytest.approx(1) and s["r"] == pytest.approx(1) and s["nrmse"] == pytest.approx(0, abs=1e-12)
    s = score(o, 2 * o)                                         # twice the swing
    assert s["nse"] == pytest.approx(0, abs=1e-12) and s["amp"] == pytest.approx(2) and s["r"] == pytest.approx(1)
    assert s["nrmse"] == pytest.approx(np.sqrt((oc ** 2).mean()) / (oc.max() - oc.min()))
    s = score(o, -o)                                            # mirror image
    assert s["nse"] == pytest.approx(-3) and s["amp"] == pytest.approx(1) and s["r"] == pytest.approx(-1)
    s = score(o, o * 0 + 7.0)                                   # constant prediction
    assert s["flat"] and s["r"] == 0 and s["amp"] == 0 and s["nse"] == pytest.approx(0, abs=1e-12)


def test_score_matches_definitions_and_identity(score):
    """Check NSE, VR, nRMSE and r against an independent implementation, and NSE = 2*VR*r - VR^2."""
    rng = np.random.default_rng(1)
    for k in range(20):
        o = _series(seed=k)
        sim = 0.3 + rng.uniform(0.2, 1.5) * (o - 500) + rng.normal(0, rng.uniform(1, 40), len(o)) + 1e3
        s, e = score(o, sim), expected(o, sim)
        for m in ("nse", "amp", "nrmse", "r"):
            assert s[m] == pytest.approx(e[m], rel=1e-10, abs=1e-12), m
        assert s["nse"] == pytest.approx(2 * s["amp"] * s["r"] - s["amp"] ** 2, rel=1e-9, abs=1e-12)


def test_score_overlap50_and_centring_over_shared_days(score):
    """Check the 50% overlap rule and that each series is centred over the shared days only.

    Exactly 50% shared days are scored and one day fewer is not; gauge days without a model
    value do not enter the means.
    """
    o = _series(200)
    sim = 0.7 * o + np.random.default_rng(2).normal(0, 5, 200)
    score.ns["MIN_DAM"][0] = 0.5 * 200                          # the scorer sets 50% of the clipped span
    assert score(o, sim.iloc[:100]) is not None                 # exactly 50% shared: scored
    assert score(o, sim.iloc[:99]) is None                      # one day short: not scored
    # gauge days without a model value do not enter the means
    o2 = o.copy(); o2.iloc[150:] += 1e4
    sim2 = sim.copy(); sim2.iloc[150:] = np.nan
    score.ns["MIN_DAM"][0] = 0
    a, b = score(o2, sim2), score(o.iloc[:150], sim.iloc[:150])
    for m in ("nse", "amp", "nrmse", "r"):
        assert a[m] == pytest.approx(b[m], rel=1e-12)


# ------------------------------------------------------------------ metrics() of the full-record scorer
@pytest.mark.parametrize("path", METRIC_FILES, ids=lambda p: p.parent.name)
def test_fullrecord_metrics(path):
    """Check the full-record metrics() against the definitions and NSE = 2*VR*r - VR^2."""
    metrics = load_function(path, "metrics", {"np": np, "pd": pd})
    rng = np.random.default_rng(3)
    for k in range(10):
        o = _series(3000, seed=k, start="1981-01-01")
        sim = rng.uniform(.2, 1.5) * o + rng.normal(0, 30, len(o)) + 77
        m, e = metrics(o, sim, None), expected(o, sim)
        for q in ("nse", "amp", "nrmse", "r"):
            assert m[q] == pytest.approx(e[q], rel=1e-10), q
        assert m["nse"] == pytest.approx(2 * m["amp"] * m["r"] - m["amp"] ** 2, rel=1e-9)


# ------------------------------------------------------------------ end-to-end on a synthetic work tree
DAYS = pd.date_range("1981-01-01", "2020-12-31")


def _write_tree(work: Path, one_seed_misses_a_day: bool = False):
    """Write gauges and predictions for five reservoirs with known overlap properties."""
    F = work / "CONUS_storage_fork"
    rng = np.random.default_rng(0)
    T = len(DAYS); t = np.arange(T)
    cyc = lambda a, ph: a * np.sin(2 * np.pi * t / 365.25 + ph)
    gauge = {d: 1000 + cyc(100 + 20 * d, d) + np.cumsum(rng.normal(0, 2, T)) for d in range(1, 6)}
    w1 = (DAYS >= "1981-01-01") & (DAYS <= "1990-12-31"); w2 = (DAYS >= "2010-01-01") & (DAYS <= "2020-12-31")
    use = {d: np.ones(T, bool) for d in gauge}
    use[2][rng.random(T) < .1] = False                         # dam 2: 10% gauge gaps
    use[4][rng.random(T) < .55] = False                        # dam 4: gauge on ~45% of days -> never scored
    TR = pd.concat([pd.DataFrame(dict(DAM_ID=d, date=DAYS, storage_mcm=g, use=use[d])) for d, g in gauge.items()])
    T3 = F / "testsets/regime3_full_1980_2020"; T3.mkdir(parents=True)
    TR.to_parquet(T3 / "truth_clean.parquet")
    members = {1: ["a"], 2: ["b", "c"], 3: ["d"], 4: ["e"], 5: ["f"]}
    pd.DataFrame(dict(DAM_ID=list(members), basin="B1", node=[m[0] for m in members.values()],
                      members=[";".join(m) for m in members.values()], site_tier="High")).to_csv(T3 / "sites.csv", index=False)
    # node-level predictions (MCM anomalies) of the two seeds
    node = {}
    for d, mem in members.items():
        g = gauge[d] - gauge[d].mean()
        share = rng.dirichlet(np.ones(len(mem)))
        for m, sh in zip(mem, share):
            node[m] = sh * (0.8 * g + rng.normal(0, 15, T))
    present = {m: np.ones(T, bool) for m in node}
    present["d"] = rng.random(T) < .45                          # dam 3: model on ~45% of days -> never scored
    for w in (w1, w2):                                          # dam 5: model on exactly 50% of each window's days
        idx = np.flatnonzero(w); present["f"][idx] = False; present["f"][idx[: (len(idx) + 1) // 2]] = True
    rows = {}
    for run, off in (("runA", 3.0), ("runB", -1.0)):
        recs = []
        for m, v in node.items():
            val = v + off + rng.normal(0, 2, T)
            keep = present[m].copy()
            if one_seed_misses_a_day and run == "runB" and m == "c":
                keep[np.flatnonzero(w1)[100]] = False; keep[np.flatnonzero(w2)[100]] = False   # one seed misses a day
            recs.append(pd.DataFrame(dict(basin="B1", subbasin=m, date=DAYS[keep], pred_dv_SWOTAE=val[keep].astype("float32"))))
        rows[run] = pd.concat(recs)
        for tag in ("c1", "h10"):
            p = F / "predictions" / run; p.mkdir(parents=True, exist_ok=True)
            rows[run].to_parquet(p / f"step_050000_{tag}.parquet")
    # rule models (2010-2020 scorer needs all four tags); dam 1 only: gauge + constant
    for tag in ("gdrom3c_g_y1980", "istarf3c_g_y1980", "modromcal3c_g_y1980", "modromrf3c_g_y1980_pubnorm"):
        p = F / "predictions" / f"grades_{tag}"; p.mkdir(parents=True)
        pd.DataFrame(dict(DAM_ID=1, date=DAYS, storage_mcm=gauge[1] + 42.0)).to_parquet(p / "full_series.parquet")
    g = F / "eval/grl_figs"; g.mkdir(parents=True)
    pd.DataFrame(dict(DAM_ID=list(members), name="x", state="y", lat=40.0, lon=-100.0, cap=1.0, use="Irrigation")).to_csv(g / "sites_v5_g.csv", index=False)
    nf = F / "eval/c5_paper/fig_c5"; nf.mkdir(parents=True)
    shutil.copy(C / "normal_year_source/score_fullrecord.py", nf / "score_fullrecord.py")
    return TR, rows, members


def _model_series(rows, members, dam):
    """Return the expected SWOT-GAT series of one reservoir.

    It is the mean of the seeds per node and day (both needed), summed over the members (all
    needed).
    """
    a = rows["runA"].set_index(["subbasin", "date"]).pred_dv_SWOTAE.astype(float)
    b = rows["runB"].set_index(["subbasin", "date"]).pred_dv_SWOTAE.astype(float)
    m = pd.concat([a, b], axis=1).dropna().mean(axis=1)
    per = pd.concat([m.loc[n] for n in members[dam]], axis=1).dropna()
    return per.sum(axis=1)


def _run(script_dir: Path, script: str, work: Path, args=(), patch=None):
    env = dict(os.environ, SWOTGAT_WORK=str(work), IP_RUNS="runA,runB", IP_HIND="h10", BASE="0",
               PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{script_dir}", PYTHONHASHSEED="0", PYTHONWARNINGS="ignore")
    env.pop("IP_STEPS", None); env.pop("ENS", None)
    runner = textwrap.dedent(f"""
        import sys
        src = open({str(script_dir / script)!r}).read()
        for a, b in {patch or []!r}:
            assert src.count(a) == 1, a
            src = src.replace(a, b)
        sys.argv = [{script!r}] + {list(args)!r}
        exec(compile(src, {script!r}, "exec"), {{"__name__": "__main__", "__file__": {str(script_dir / script)!r}}})
    """)
    r = subprocess.run([sys.executable, "-c", runner], env=env, cwd=script_dir, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """Write the synthetic work tree; return (work folder, gauges, predictions per run, members)."""
    work = tmp_path_factory.mktemp("work")
    return work, *_write_tree(work)


@pytest.fixture(scope="module")
def tree_gap(tmp_path_factory):
    """Write the synthetic work tree in which one seed misses a day of one member node.

    The return value is the same as that of `tree`.
    """
    work = tmp_path_factory.mktemp("work_gap")
    return work, *_write_tree(work, one_seed_misses_a_day=True)


FINDING_SEED_GAP = (
    "FINDING: the scorers average the seeds with mean(skipna=False) but then test 'every member present' with "
    "groupby(date).subbasin.nunique(), which counts a member whose seed mean is NaN, and sum the members with "
    "a NaN-skipping sum. On a day when one seed lacks one member node the reservoir value is the partial sum "
    "of the other members instead of being dropped, contrary to the docstrings. Both seeds' prediction files "
    "cover the same node-days in the paper's runs, so the paper's numbers are not affected.")


def _fullrecord(tree, tmp_path):
    work, TR, rows, members = tree
    sd = tmp_path / "scripts"; sd.mkdir()
    for f in ("score_fullrecord.py", "ipcfg.py", "bl.py"):
        shutil.copy(C / "full_record" / f, sd / f)
    W1 = 'WIN = [(f"{y}-{y+4}", f"{y}-01-01", f"{y+4}-12-31", "block") for y in range(1981, 2017, 5)]'
    W2 = 'WIN += [(str(y), f"{y}-01-01", f"{y}-12-31", "year") for y in range(1981, 2021)] + [("2015-2020", "2015-01-01", "2020-12-31", "frozen")]'
    out = tmp_path / "out"
    _run(sd, "score_fullrecord.py", work, ["--out", str(out), "--gnn", "SWOT-GNN=runA,runB", "--chunks", "c1"],
         patch=[(W1, 'WIN = [("1981-1990", "1981-01-01", "1990-12-31", "block")]'), (W2, "")])
    return pd.read_csv(out / "fullrecord_site_window.csv")


def test_fullrecord_scorer_end_to_end(tree, tmp_path):
    """Check the unchanged full-record scorer, run on the synthetic work tree.

    Seeds are averaged per node and day and members summed; reservoirs that share less than 50%
    of the days (model or gauge) are dropped, exactly 50% is kept, and the metrics are as defined.
    """
    _check_fullrecord(tree, _fullrecord(tree, tmp_path))


@pytest.mark.xfail(strict=True, reason=FINDING_SEED_GAP)
def test_fullrecord_scorer_drops_day_missing_in_one_seed(tree_gap, tmp_path):
    """Check that the full-record scorer drops a day that one seed lacks (expected failure)."""
    _check_fullrecord(tree_gap, _fullrecord(tree_gap, tmp_path))


def _check_fullrecord(tree, R):
    work, TR, rows, members = tree
    G = R[R.model == "SWOT-GNN"].set_index("DAM_ID")
    assert set(G.index) == {1, 2, 5}                            # dams 3 (model) and 4 (gauge) share < 50% of days
    for d in (1, 2, 5):
        o = TR[(TR.DAM_ID == d) & TR.use].set_index("date").storage_mcm.loc["1981-01-01":"1990-12-31"]
        e = expected(o, _model_series(rows, members, d))
        assert G.at[d, "n"] == e["n"]
        for q in ("nse", "amp", "nrmse", "r"):
            assert G.at[d, q] == pytest.approx(e[q], rel=1e-5), (d, q)
    assert G.at[5, "n"] == 1826 and 1826 >= 0.5 * 3652         # exactly half the window: kept


def _score2010(tree, tmp_path):
    work = tree[0]
    sd = tmp_path / "scripts"; sd.mkdir()
    for f in ("score_2010.py", "ipcfg.py", "bl.py"):
        shutil.copy(C / "overlap50" / f, sd / f)
    _run(sd, "score_2010.py", work)
    return pd.read_csv(sd / "sites_2010_g.csv")


def test_2010_scorer_end_to_end(tree, tmp_path):
    """Check the unchanged 2010-2020 scorer, run on the synthetic work tree.

    Same checks as for the full-record scorer, plus a rule model equal to gauge + constant
    scoring NSE 1.
    """
    _check_2010(tree, _score2010(tree, tmp_path))


@pytest.mark.xfail(strict=True, reason=FINDING_SEED_GAP)
def test_2010_scorer_drops_day_missing_in_one_seed(tree_gap, tmp_path):
    """Check that the 2010-2020 scorer drops a day that one seed lacks (expected failure)."""
    _check_2010(tree_gap, _score2010(tree_gap, tmp_path))


def _check_2010(tree, R):
    work, TR, rows, members = tree
    G = R[R.MODEL == "v10pm"].set_index("DAM_ID")
    assert set(G.index) == {1, 2, 5}
    for d in (1, 2, 5):
        o = TR[(TR.DAM_ID == d) & TR.use].set_index("date").storage_mcm.loc["2010-01-01":"2020-12-31"]
        e = expected(o, _model_series(rows, members, d))
        assert G.at[d, "n_days"] == e["n"]
        for q in ("nse", "amp", "nrmse", "r"):
            assert G.at[d, q] == pytest.approx(e[q], rel=1e-5), (d, q)
    rule = R[(R.MODEL == "GDROM")].set_index("DAM_ID")           # gauge + constant: perfect after centring
    assert rule.at[1, "nse"] == pytest.approx(1) and rule.at[1, "amp"] == pytest.approx(1)


# ------------------------------------------------------------------ harmonic irregularity
def _irr(TR):
    ns = run_statements(C / "drivers/irr_harm.py",
                        lambda s: s.startswith("wyf =") or s.startswith("R = {}") or s.startswith('for d, g in TR.groupby("DAM_ID")'),
                        {"np": np, "pd": pd, "TR": TR})
    return ns["R"]


def test_harmonic_irregularity():
    """Check the harmonic irregularity computed by irr_harm.py.

    It must be 0 for a pure annual + semiannual cycle and 0.5 when the residual sd is half the
    cycle sd; fewer than 5 water years give no value, and a water year with < 292 days is left
    out of the fit.
    """
    d = pd.date_range("1980-10-01", "2020-09-30")
    tt = (d - pd.Timestamp("1980-01-01")).days.values / 365.25 * 2 * np.pi
    fit = 300 + 50 * np.sin(tt) + 20 * np.cos(tt) + 15 * np.sin(2 * tt) - 5 * np.cos(2 * tt)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, len(d)); noise *= 0.5 * fit.std() / noise.std()
    short = np.ones(len(d), bool); short[(d.year < 1990)] = False
    TR = pd.concat([pd.DataFrame(dict(DAM_ID=1, date=d, storage_mcm=fit)),                        # pure cycle
                    pd.DataFrame(dict(DAM_ID=2, date=d, storage_mcm=fit + noise)),                # half as irregular
                    pd.DataFrame(dict(DAM_ID=3, date=d[(d >= "2016-10-01")], storage_mcm=fit[(d >= "2016-10-01")]))])  # 4 water years
    R = _irr(TR)
    assert R[1]["irr_harm"] == pytest.approx(0, abs=1e-8) and R[1]["harm_r2"] == pytest.approx(1)
    assert R[2]["irr_harm"] == pytest.approx(0.5, rel=0.02)
    assert R[1]["n_wy"] == 40 and 3 not in R                   # fewer than 5 water years: no value
    # a water year with < 292 gauge days is left out of the fit
    TR4 = pd.DataFrame(dict(DAM_ID=4, date=d, storage_mcm=fit))
    TR4 = TR4[~((TR4.date >= "1999-10-01") & (TR4.date < "2000-10-01") & (TR4.date.dt.day > 20))]   # WY2000 ~ 240 days
    assert _irr(TR4)[4]["n_wy"] == 39


# ------------------------------------------------------------------ training coverage
def test_training_coverage_definition():
    """Check the definition of training coverage.

    The gauge sd over 1981-2020 is the sample sd on the window's days (NaN under 10 days), and
    Figure 3c divides the SWOT sd by it.
    """
    sd = load_function(C / "full_record/sd8120.py", "sd", {"np": np, "pd": pd})
    s = _series(16000, start="1975-01-01")
    w = s.loc["1981-01-01":"2020-12-31"]
    assert sd(s, "1981-01-01", "2020-12-31") == pytest.approx(w.std(ddof=1), rel=1e-12)
    assert np.isnan(sd(s.iloc[:9], "1975-01-01", "2020-12-31"))       # fewer than 10 days
    # Figure 3c plots sd_swot / sd_g8120 (sd of SWOT storage over the sd of the 1981-2020 gauge record)
    src = (REPO / "08_figures/fig3_c5d.py").read_text()
    divs = [ast.unparse(n) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)]
    assert "SD.sd_swot / SD.sd_g8120" in divs, divs


# ------------------------------------------------------------------ gauge "normal year" baselines must not see the scored days
NY_FILES = [C / "full_record/score_fullrecord.py", C / "overlap50/score_fullrecord.py", C / "normal_year_source/score_fullrecord.py"]


def test_normal_year_is_the_same_function_in_every_copy():
    """Check that the three copies of normal_year() are the same text.

    bl.py executes the one cut from normal_year_source/score_fullrecord.py.
    """
    seg = lambda p: (lambda s: s[s.index("\ndef normal_year("):s.index("\ndef ", s.index("\ndef normal_year(") + 1)])(p.read_text())
    assert len({seg(p) for p in NY_FILES}) == 1


def _gauge(seed=0):
    d = pd.date_range("1981-01-01", "2000-12-31")
    rng = np.random.default_rng(seed)
    doy = np.minimum(d.dayofyear.values, 365)
    cyc = 30 * np.sin(2 * np.pi * doy / 365)
    level = np.repeat(rng.normal(500, 50, 20), [366 if y % 4 == 0 else 365 for y in range(1981, 2001)])
    return pd.Series(level + cyc, index=d), pd.Series(cyc, index=d)


def test_fullrecord_normal_year_uses_only_days_outside_the_window():
    """Check that the full-record normal-year baseline never sees the scored window.

    It must be unchanged when the window's gauge values change, recover a known seasonal cycle,
    and give no baseline with fewer than 3 reference years.
    """
    normal_year = load_function(C / "full_record/score_fullrecord.py", "normal_year", {"np": np, "pd": pd})
    o, cyc = _gauge()
    w0, w1 = "1991-01-01", "1995-12-31"
    dates = o.loc[w0:w1].index
    a = normal_year(o, w0, w1, dates)
    o2 = o.copy(); o2.loc[w0:w1] = o2.loc[w0:w1] * 3 + np.random.default_rng(1).normal(0, 100, len(dates))
    b = normal_year(o2, w0, w1, dates)
    pd.testing.assert_series_equal(a, b)                                    # the scored window never enters
    # it recovers the day-of-year anomaly cycle (31-day smoothing of a 365-day sine: ~1% amplitude loss)
    c = cyc.loc[w0:w1] - cyc.loc[w0:w1].median()
    assert np.corrcoef(a.values, c.values)[0, 1] > 0.999
    assert a.std() == pytest.approx(c.std(), rel=0.03)
    assert normal_year(o.loc["1990-01-01":"1996-12-31"], w0, w1, dates) is None   # < 3 reference years


def test_2010_normal_year_is_leave_one_year_out():
    """Check that the 2010-2020 normal year is leave-one-year-out.

    The normal year of year y must be unchanged when only year y changes, while those of the
    other years change.
    """
    loyo = load_function(C / "overlap50/score_2010.py", "normal_year_loyo", {"np": np, "pd": pd})
    o, _ = _gauge(2)
    o = o.loc["1990-01-01":"2000-12-31"]
    a = loyo(o)
    for y in (1992, 1997):
        o2 = o.copy(); k = o2.index.year == y
        o2[k] = o2[k] * 5 + 77.0                                          # change only year y
        b = loyo(o2)
        pd.testing.assert_series_equal(a[k], b[k])                         # year y's normal year never sees year y
        assert not np.allclose(a[~k].values, b[~k].values)                 # but the other years' do
