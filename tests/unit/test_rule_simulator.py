"""Rule-model simulators, run unchanged on a synthetic work tree.

The simulators are grades_sim_v4.py and grades_sim_v4_hindcast.py of 07_evaluation/b_rule_models.
The tree is one river of four nodes, 1001 -> 1002 -> 1003 -> 1004 (outlet), two years of GRADES-like
discharge, an upstream reservoir A (GRanD 100, node 1001, in GDROM and MODROM) and a test-set
reservoir B (GRanD 101, node 1003, in all three families). Each simulator output is compared with
the engines of resops_engines.py (tested in test_rule_engines.py) driven by inputs built here from
the documented method:

* natural flow = discharge clipped at 0, gaps of <= 5 days interpolated, times the node/reach
  upstream-area ratio (2.0 at B);
* cascade: B's inflow = max(natural_B - natural_A, 0) + outflow of A, when A is a reservoir of the
  same family; otherwise natural_B;
* units: GDROM and MODROM run in acre-feet (1 acre-foot = 1233.48184 m3), ISTARF in m3; the output
  `storage_mcm` is million m3;
* --init gauge: the start storage is the gauge value nearest the start day within 15 days (GDROM's
  cleaned record in acre-feet, or ResOpsUS in MCM times its correction factor); later gauge values
  are never used;
* without --end, --qbar-end and --unitfix, grades_sim_v4.py gives the same series as
  grades_sim_v4_hindcast.py (the script that made the paper's 1980-2020 runs).
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import REPO

B_DIR = REPO / "07_evaluation/b_rule_models"
sys.path.insert(0, str(B_DIR))
import resops_engines as E                                  # noqa: E402

AF, DAY = 1233.48184, 86400.0
DAYS = pd.date_range("2000-01-01", "2001-12-31")
GDROM_A = {"0": ["(Storage <= 3000) Release: 0.2*Inflow", "(Storage > 3000) Release: Inflow"]}
GDROM_B_COND = ["(DOY <= 180) module: 1", "(DOY > 180) module: 2"]
GDROM_B = {"1": ["(Storage <= 30000) Release: 0.5*Inflow", "(Storage > 30000) Release: 0.9*Inflow + 100"],
           "2": ["Release: max(0.8*Inflow, 400)"]}
ISTARF_B = dict(GRanD_ID=101, NORhi_mu=80., NORhi_alpha=5., NORhi_beta=3., NORhi_min=60., NORhi_max=95.,
                NORlo_mu=40., NORlo_alpha=4., NORlo_beta=-2., NORlo_min=20., NORlo_max=55.,
                Release_alpha1=.1, Release_alpha2=.05, Release_beta1=-.1, Release_beta2=.02, Release_c=.0,
                Release_p1=.3, Release_p2=.2, Release_min=-.6, Release_max=1.5,
                GRanD_CAP_MCM=100.0, GRanD_MEANFLOW_CUMECS=20.0, fit="synthetic")
MOD_MODES_B = [2, 2, 3, 3, 3, 2, 2, 3, 3, 3, 2, 2]
MOD_PRM_B = [[0.9, 0.05] if m == 2 else [0.2, 1.4, 0.05, 0.6] for m in MOD_MODES_B]
NRM = {"100": dict(storage_max=3500.0, inflow_max=2000.0, outflow_max=1500.0),
       "101": dict(storage_max=81000.0, inflow_max=4000.0, outflow_max=3000.0)}
MOD_A = dict(GRAND_ID=100, MonthlyModes=[2] * 12, ModeParameters=[[0.5 * 2000 / 1500, 0.0]] * 12)
GDW = {100: 4.0, 101: 100.0}                    # GDW capacity, MCM
FIX_B = 0.5                                     # ResOpsUS correction factor of B


def _discharge():
    rng = np.random.default_rng(0)
    t = np.arange(len(DAYS))
    q11 = 6 + 4 * np.sin(2 * np.pi * t / 365) + rng.gamma(2, 1.0, len(t))
    q13 = 1.3 * q11 + rng.gamma(2, 0.5, len(t))
    q13[40] = -5.0                              # negative discharge -> 0
    q13[200:203] = np.nan                       # 3-day gap -> interpolated
    q11[50:56] = 30.0                           # a flood upstream
    return pd.DataFrame({"11": q11, "13": q13}, index=DAYS)


def _write_tree(work: Path, gauge_scale_after: float = 1.0):
    G, C, F = work / "GRADES", work / "CONUS", work / "CONUS_storage_fork"
    for d in (G / "qout", G / "gdrom_v2/cleaned", G / "modrom/results", G / "modrom/data", C / "metadata",
              F / "crosswalk", F / "ResOpsUS/time_series_all", F / "testsets/regime1_2023_2025", F / "testsets/regime2_2015_2020"):
        d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(GRAND_ID=[100, 101], node=["1001", "1003"], gdrom=[True, True], istarf=[False, True],
                      test=[False, True], STATE=["CO", "CO"])).to_csv(G / "rule_reservoir_nodes.csv", index=False)
    pd.DataFrame(columns=["GRAND_ID", "node"]).to_csv(G / "modrom_nongrand_nodes.csv", index=False)
    pd.DataFrame(dict(comid_v07=[11, 13], area_ratio=[1.0, 2.0]), index=pd.Index(["1001", "1003"], name="node")).to_csv(G / "reservoir_grades_map.csv")
    pd.DataFrame(dict(DAM_ID=[101], inflow_node=["1003"])).to_csv(G / "testset_dams.csv", index=False)
    Q = _discharge()
    for y in (2000, 2001):
        Q[Q.index.year == y].to_parquet(G / "qout" / f"qout_{y}.parquet")
    pd.DataFrame(dict(nextdown=["1002", "1003", "1004", "0"], uparea_km2=[100., 150., 400., 500.]),
                 index=pd.Index(["1001", "1002", "1003", "1004"], name="node")).to_parquet(C / "metadata/subbasins_conus.parquet")
    rd = G / "gdrom_v2/Operation Rules - GDROMs"
    (rd / "module_conditions").mkdir(parents=True, exist_ok=True); (rd / "modules").mkdir(parents=True, exist_ok=True)
    for m, lines in GDROM_A.items():
        (rd / "modules" / f"100_{m}.txt").write_text("\n".join(lines) + "\n")
    (rd / "module_conditions" / "101.txt").write_text("\n".join(GDROM_B_COND) + "\n")
    for m, lines in GDROM_B.items():
        (rd / "modules" / f"101_{m}.txt").write_text("\n".join(lines) + "\n")
    pd.DataFrame(dict(GRAND_ID=[100, 101], STATE=["CO", "CO"], STORAGE_CAP=[3500.0, 80000.0])).to_csv(G / "gdrom_v2/reservoir_metadata.csv", index=False)
    ym = sorted({y * 100 + m for y in range(1999, 2002) for m in range(1, 13)})
    pd.DataFrame(dict(yyyymm=ym, state="CO", pdsi=np.linspace(-2, 2, len(ym)))).to_csv(G / "pdsi_state_monthly.csv", index=False)
    # gauges of B: GDROM cleaned record (acre-feet) and ResOpsUS (MCM); the value nearest 2000-01-01 is 50,000 af / 60 MCM
    gd = pd.date_range("1999-12-01", "2000-06-30")
    st_af = np.where(gd == "2000-01-01", 50000.0, 20000.0 + np.arange(len(gd)))
    st_mcm = np.where(gd == "2000-01-01", 60.0, 30.0 + 0.01 * np.arange(len(gd)))
    later = gd > "2000-01-18"
    st_af[later] *= gauge_scale_after; st_mcm[later] *= gauge_scale_after
    pd.DataFrame(dict(Date=gd, Storage=st_af, Inflow=1000.0, PDSI=0.0)).to_csv(G / "gdrom_v2/cleaned/101.csv", index=False)
    pd.DataFrame(dict(date=gd.strftime("%Y-%m-%d"), storage=st_mcm)).to_csv(F / "ResOpsUS/time_series_all/ResOpsUS_101.csv", index=False)
    pd.DataFrame([ISTARF_B]).to_csv(G / "ISTARF-CONUS.csv", index=False)
    json.dump([MOD_A, dict(GRAND_ID=101, MonthlyModes=MOD_MODES_B, ModeParameters=MOD_PRM_B)], open(G / "modrom/results/calibrated_parameters.json", "w"))
    json.dump(NRM, open(G / "modrom/data/reservoir_norm_info.json", "w"))
    pd.DataFrame(dict(DAM_ID=[100, 101], cap=[GDW[100], GDW[101]])).to_csv(F / "crosswalk/resops_record_verdict.csv", index=False)
    pd.DataFrame(dict(DAM_ID=[101], fix_factor=[FIX_B])).to_csv(F / "crosswalk/resops_record_qa.csv", index=False)
    for sub in ("regime1_2023_2025", "regime2_2015_2020"):
        pd.DataFrame(dict(DAM_ID=[101], node=["1003"], members=["1003"], basin=["1004"])).to_csv(F / "testsets" / sub / "sites.csv", index=False)


def _run(script, work, family, *extra):
    env = dict(os.environ, SWOTGAT_WORK=str(work), PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=str(REPO / "src"), PYTHONWARNINGS="ignore")
    r = subprocess.run([sys.executable, str(B_DIR / script), family, "--init", "gauge", *extra], env=env, cwd=work,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    tag = {"gdrom": "gdrom3c_g", "istarf": "istarf3c_g", "modrom_cal": "modromcal3c_g"}[family]
    fs = pd.read_parquet(work / "CONUS_storage_fork/predictions" / f"grades_{tag}" / "full_series.parquet")
    sm = pd.read_csv(work / "GRADES/v3" / f"sim_{tag}.csv").set_index("GRAND_ID")
    return fs, sm


# ------------------------------------------------------------------ expected series, built from the documented method
def _natural():
    Q = _discharge().clip(lower=0.0)
    nat = lambda c, r: Q[c].reindex(DAYS).interpolate(limit=5).fillna(0.0).values * r
    return nat("11", 1.0), nat("13", 2.0)


def _rules(tmp, rid, cond, mods):
    d = tmp / f"rules{rid}"
    (d / "module_conditions").mkdir(parents=True, exist_ok=True); (d / "modules").mkdir(parents=True, exist_ok=True)
    if cond:
        (d / "module_conditions" / f"{rid}.txt").write_text("\n".join(cond) + "\n")
    for m, lines in mods.items():
        (d / "modules" / f"{rid}_{m}.txt").write_text("\n".join(lines) + "\n")
    return E.GdromRules(d, rid)


def expected_gdrom(tmp):
    """Return B's expected GDROM storage (MCM) and its cascade inflow (m3/s).

    Reservoir A has no gauge and starts at 0.6 x capacity; B starts from its gauge value of
    50,000 acre-feet. The rule files are written under `tmp`.
    """
    nA, nB = _natural()
    doy = DAYS.dayofyear.values.astype(float); pdsi = np.zeros(len(DAYS))
    rA = _rules(tmp, 100, None, GDROM_A)
    capA = max(GDW[100] * 1e6 / AF, 3500.0, 1.05 * 3000)
    _, outA, _ = E.run_gdrom(rA, nA * DAY / AF, doy, pdsi, capA, 0.6 * capA)       # A has no gauge: 0.6 x capacity
    IB = np.maximum(nB - nA, 0.0) + outA * AF / DAY
    rB = _rules(tmp, 101, GDROM_B_COND, GDROM_B)
    capB = max(GDW[101] * 1e6 / AF, 80000.0, 1.05 * 30000)
    S, _, _ = E.run_gdrom(rB, IB * DAY / AF, doy, pdsi, capB, 50000.0)             # gauge start: 50,000 af
    return S * AF / 1e6, IB


def expected_istarf():
    """Return B's expected ISTARF storage (MCM) and its inflow (m3/s).

    A is not an ISTARF reservoir, so B's inflow is its natural flow. B starts from its ResOpsUS
    gauge value (60 MCM) times the correction factor.
    """
    _, nB = _natural()                                                              # A is not an ISTARF reservoir
    p = pd.Series(ISTARF_B); cap = 100.0 * 1e6; Id = nB * DAY
    S, _, _ = E.run_istarf(p, Id, E.epiweek(DAYS), cap, 60.0 * FIX_B * 1e6, float(Id.mean()))   # ResOpsUS x fix_factor
    return S / 1e6, nB


def expected_modrom():
    """Return B's expected MODROM storage (MCM) and its cascade inflow (m3/s).

    A starts at 0.6 x capacity; B starts from the GDROM gauge record (50,000 acre-feet), which
    MODROM uses first.
    """
    nA, nB = _natural(); months = DAYS.month.values
    a, b = NRM["100"], NRM["101"]
    capA = max(a["storage_max"], GDW[100] * 1e6 / AF)
    _, outA, _ = E.run_modrom(MOD_A["MonthlyModes"], MOD_A["ModeParameters"], nA * DAY / AF, months,
                              a["inflow_max"], a["storage_max"], a["outflow_max"], capA, 0.6 * capA)
    IB = np.maximum(nB - nA, 0.0) + outA * AF / DAY
    capB = max(b["storage_max"], GDW[101] * 1e6 / AF)
    S, _, _ = E.run_modrom(MOD_MODES_B, MOD_PRM_B, IB * DAY / AF, months, b["inflow_max"], b["storage_max"],
                           b["outflow_max"], capB, 50000.0)                         # GDROM record first for MODROM
    return S * AF / 1e6, IB


EXPECTED = {"gdrom": expected_gdrom, "istarf": lambda tmp: expected_istarf(), "modrom_cal": lambda tmp: expected_modrom()}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Run both simulators on synthetic work trees for GDROM, ISTARF and MODROM.

    Return {run: {family: (full series, summary table)}} for the runs "v4" (grades_sim_v4.py),
    "hindcast" (grades_sim_v4_hindcast.py) and "v4_gauge_x3" (grades_sim_v4.py with the later
    gauge values tripled).
    """
    out = {}
    for name, script, scale in (("v4", "grades_sim_v4.py", 1.0), ("hindcast", "grades_sim_v4_hindcast.py", 1.0),
                                ("v4_gauge_x3", "grades_sim_v4.py", 3.0)):
        work = tmp_path_factory.mktemp(name)
        _write_tree(work, gauge_scale_after=scale)
        out[name] = {fam: _run(script, work, fam) for fam in ("gdrom", "istarf", "modrom_cal")}
    return out


@pytest.mark.parametrize("family", ["gdrom", "istarf", "modrom_cal"])
def test_simulator_matches_engines_cascade_units_and_init(runs, family, tmp_path):
    """Check that the simulator's storage (MCM) equals the engine driven by the documented inputs.

    This covers the natural flow, the cascade inflow, the acre-foot conversion and the start
    storage taken from the gauge. Only the test-set reservoir is written; for GDROM and MODROM
    the upstream reservoir must be simulated and routed into it.
    """
    fs, sm = runs["v4"][family]
    want, inflow = EXPECTED[family](tmp_path)
    assert list(fs.DAM_ID.unique()) == [101]                                        # only test-set dams are written
    assert pd.DatetimeIndex(fs.date).equals(DAYS)
    np.testing.assert_allclose(fs.storage_mcm.values, want, rtol=1e-10, atol=1e-9)
    assert sm.at[101, "status"] == "ok"
    assert sm.at[101, "mean_inflow_cms"] == pytest.approx(inflow.mean(), rel=1e-10)
    assert sm.at[101, "init_source"] == ("resops_day" if family == "istarf" else "gdrom_day")
    if family != "istarf":
        assert sm.at[101, "n_upstream"] == 1 and sm.at[100, "status"] == "ok"         # A simulated and routed into B


@pytest.mark.parametrize("family", ["gdrom", "istarf", "modrom_cal"])
def test_gauge_enters_only_as_start_storage(runs, family):
    """Check that the gauge enters the simulation only as the start storage.

    Tripling every gauge value after 2000-01-18 (past the 15 days around the start day) must
    leave the simulated storage unchanged.
    """
    a, b = runs["v4"][family][0], runs["v4_gauge_x3"][family][0]
    np.testing.assert_array_equal(a.storage_mcm.values, b.storage_mcm.values)


@pytest.mark.parametrize("family", ["gdrom", "istarf", "modrom_cal"])
def test_v4_equals_hindcast_simulator(runs, family):
    """Check that grades_sim_v4.py gives the same series as grades_sim_v4_hindcast.py.

    grades_sim_v4.py is run without its three new options.
    """
    a, b = runs["v4"][family][0], runs["hindcast"][family][0]
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


@pytest.mark.parametrize("family", ["gdrom", "istarf", "modrom_cal"])
def test_simulated_storage_within_capacity(runs, family):
    """Check that 0 <= storage <= capacity on every day of the grades_sim_v4.py run."""
    fs, sm = runs["v4"][family]
    s = fs.storage_mcm.values
    assert np.all(s >= 0) and np.all(s <= sm.at[101, "cap_mcm"] * (1 + 1e-12))
