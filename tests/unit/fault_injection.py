#!/usr/bin/env python3
"""Fault injection: plant realistic bugs in a copy of the repository and check that the tests fail.

Each mutant changes one line of the model, data, loss, scoring or rule-model code in the way a real
mistake would (flow direction reversed, burn-in dropped, the scored year leaking into its own
baseline, acre-feet converted with a rounded constant, ...). The unit tests that cover that code are
run on the mutated copy. A mutant is CAUGHT when at least one of them fails. A mutant that survives
marks a gap in the tests.

The repository itself is never modified: everything happens in a temporary copy.

Usage (same Python environment as the tests, pytest importable):
    python tests/unit/fault_injection.py [--out FILE.md] [--only ID[,ID...]]
Runtime: about 20-30 minutes on 8 CPU cores (each mutant re-runs the test file that covers it).
Exit code 0 if every mutant is caught and the unmutated copy passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
M = "vendor/swot_ml/src/models/sparse_fusion_gann.py"
L = "vendor/swot_ml/src/train/loss.py"
D = "vendor/swot_ml/src/data/cached_basingraphdataset.py"
S = "07_evaluation/c_scoring"
R = "07_evaluation/b_rule_models"
T = "tests/unit/"

# (id, file, old text, new text, test files, what the bug is)
MUTANTS = [
    ("graph-direction", M, "src, dest = edge_index  # src=upstream, dest=downstream",
     "dest, src = edge_index  # src=upstream, dest=downstream", ["test_graph_attention.py"],
     "downstream aggregation reads the edges backwards"),
    ("one-hop-short", M, "for i in range(k_hops)", "for i in range(k_hops - 1)", ["test_graph_attention.py"],
     "the graph step has one layer fewer than k_hops"),
    ("carry-pre-gat-h", M, "new_state = (h_new, c_new)", "new_state = (h_temporal, c_new)", ["test_recurrent_and_fusion.py"],
     "the next day receives the LSTM's h, not the GAT-revised h"),
    ("gat-writes-c", M, "new_state = (h_new, c_new)", "new_state = (h_new, c_new + h_new - h_temporal)", ["test_recurrent_and_fusion.py"],
     "the graph step also changes the cell state"),
    ("area-after-dropout", M, "predictions = self._apply_area_scaling(predictions, raw_static)",
     "predictions = self._apply_area_scaling(predictions, data.static)", ["test_area_scaling_and_loss.py"],
     "area scaling reads the static vector after static dropout"),
    ("area-sigma", M, '"sigma": a * p["sigma"] / y_sc,', '"sigma": p["sigma"] / y_sc,', ["test_area_scaling_and_loss.py"],
     "the mixture spread is not scaled by the lake area"),
    ("no-burn-in", L, "v = jnp.where(t >= burn_in_steps, v, jnp.nan)", "v = jnp.where(t >= 0, v, jnp.nan)",
     ["test_area_scaling_and_loss.py"], "the 90-day burn-in is not excluded from the loss"),
    ("loss-padding", L, "valid_mask = ~jnp.isnan(y_true) & node_mask", "valid_mask = ~jnp.isnan(y_true)",
     ["test_area_scaling_and_loss.py"], "padding nodes count in the loss"),
    ("loss-weights", L, "loss = jnp.sum(jnp.stack(weighted_losses)) / total_weight",
     "loss = jnp.sum(jnp.stack(weighted_losses))", ["test_area_scaling_and_loss.py"], "target weights not normalised"),
    ("fallback-overrides", D, "np.isnan(target[t_name]), src, np.nan", "np.ones_like(target[t_name], dtype=bool), src, np.nan",
     ["test_data.py"], "the fallback is emitted where the primary target exists too"),
    ("median-whole-record", D, "med = ds_full[v].sel(date=rw)", "med = ds_full[v]", ["test_data.py"],
     "re-referencing median over the whole record instead of the window"),
    ("window-off-by-one", D, 'end_date = np.datetime64(int(date_int), "D")', 'end_date = np.datetime64(int(date_int) - 1, "D")',
     ["test_data.py"], "each sample ends one day before its date"),
    ("history-targets-kept", D, "ds[v] = ds[v].where(ds.date >= t0)", "ds[v] = ds[v]", ["test_data.py"],
     "targets inside the 365-day input history are not blanked"),
    ("nse-centring", f"{S}/overlap50/score_x26span.py", "oc, qc = j.o - j.o.mean(), j.q - j.q.mean(); rng = oc.max() - oc.min()",
     "oc, qc = j.o - j.o.mean(), j.q - j.o.mean(); rng = oc.max() - oc.min()", ["test_scoring.py"],
     "the model series is centred on the gauge mean"),
    ("nrmse-range", f"{S}/overlap50/score_x26span.py", "rng = oc.max() - oc.min()\n", "rng = qc.max() - qc.min()\n",
     ["test_scoring.py"], "nRMSE divided by the model's range"),
    ("overlap-40-fullrecord", f"{S}/full_record/score_fullrecord.py", "cov = 0.5", "cov = 0.4", ["test_scoring.py"],
     "full-record overlap rule at 40% of the window's days"),
    ("overlap-40-2010", f"{S}/overlap50/score_2010.py", "if len(j) < 0.5*len(pd.date_range(W0, W1)): continue",
     "if len(j) < 0.4*len(pd.date_range(W0, W1)): continue", ["test_scoring.py"], "2010-2020 overlap rule at 40%"),
    ("normal-year-leak", f"{S}/full_record/score_fullrecord.py",
     "ref = obs_all[(obs_all.index < pd.Timestamp(w0)) | (obs_all.index > pd.Timestamp(w1))]", "ref = obs_all",
     ["test_scoring.py"], "the normal-year baseline sees the scored window"),
    ("loyo-leak", f"{S}/overlap50/score_2010.py", "ref = o[o.index.year != y]", "ref = o", ["test_scoring.py"],
     "the leave-one-year-out normal year sees its own year"),
    ("irr-annual-only", f"{S}/drivers/irr_harm.py",
     "X = np.column_stack([np.ones_like(t), np.sin(t), np.cos(t), np.sin(2*t), np.cos(2*t)])",
     "X = np.column_stack([np.ones_like(t), np.sin(t), np.cos(t)])", ["test_scoring.py"],
     "harmonic irregularity without the semiannual harmonic"),
    ("coverage-denominator", "08_figures/fig3_c5d.py", "SD.sd_swot/SD.sd_g8120", "SD.sd_swot/SD.sd_g10", ["test_scoring.py"],
     "training coverage divided by the 2010-2020 gauge sd"),
    ("release-end-of-day", f"{R}/resops_engines.py", 'st = S_obs[t] if S_obs is not None else s\n        env = {',
     'st = S_obs[t] if S_obs is not None else s + I[t]\n        env = {', ["test_rule_engines.py"],
     "GDROM decides the release from end-of-day storage"),
    ("spill-not-outflow", f"{R}/resops_engines.py", "s = sn; S[t] = s; OUT[t] = r + sp", "s = sn; S[t] = s; OUT[t] = r",
     ["test_rule_engines.py"], "spilled water is not counted as outflow (first engine, GDROM)"),
    ("istarf-percent", f"{R}/resops_engines.py", "av = (100 * st / cap - lo[t]) / span", "av = (st / cap - lo[t]) / span",
     ["test_rule_engines.py"], "ISTARF compares a storage fraction with bounds in percent"),
    ("modrom-month", f"{R}/resops_engines.py", "mi = months[t] - 1", "mi = months[t] % 12", ["test_rule_engines.py"],
     "MODROM uses the next month's mode"),
    ("gdrom-fallback", f"{R}/resops_engines.py", "self.n_nomatch_rel += 1; return rules[-1][1](env)",
     "self.n_nomatch_rel += 1; return rules[0][1](env)", ["test_rule_engines.py"],
     "GDROM falls back to the first release line instead of the last"),
    ("parser-power", f"{R}/resops_engines.py", "ast.Div: operator.truediv}", "ast.Div: operator.truediv, ast.Pow: operator.pow}",
     ["test_rule_engines.py"], "the rule parser accepts more than the documented grammar"),
    ("acre-foot-rounded", f"{R}/grades_sim_v4.py", "AF = 1233.48184; DAY = 86400.0", "AF = 1233.0; DAY = 86400.0",
     ["test_rule_simulator.py"], "acre-feet converted with a rounded constant"),
    ("cascade-no-subtract", f"{R}/grades_sim_v4.py", "if nu is not None: loc = loc - nu", "if nu is not None: loc = loc",
     ["test_rule_simulator.py"], "upstream natural flow counted twice in the cascade"),
    ("init-last-gauge-day", f"{R}/grades_sim_v4.py", "return float(w.iloc[int(np.argmin(np.abs((w.index - T0).days)))]), f\"{src}_day\"",
     "return float(w.iloc[-1]), f\"{src}_day\"", ["test_rule_simulator.py"],
     "start storage taken from the gauge 15 days after the start"),
    ("area-ratio-ignored", f"{R}/grades_sim_v4.py", "r = GM.area_ratio[node]; r = float(r) if np.isfinite(r) and 0.2 < r < 5 else 1.0",
     "r = 1.0", ["test_rule_simulator.py"], "natural flow not scaled by the node/reach area ratio"),
    ("resops-fix-ignored", f"{R}/grades_sim_v4.py", "s = s.sort_index() * float(FIX.get(dam, 1.0))", "s = s.sort_index()",
     ["test_rule_simulator.py"], "ResOpsUS correction factor not applied"),
]


def run_pytest(root: Path, files, log: Path) -> int:
    """Run pytest on the given test files of the copy at `root` and return its exit code.

    The run's stdout and stderr are written to `log`.
    """
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    cmd = [sys.executable, "-m", "pytest", *[str(root / T / f) for f in files], "-q", "-p", "no:cacheprovider",
           "-W", "ignore::DeprecationWarning", "--basetemp", str(root.parent / "pt")]
    r = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=3600)
    log.write_text(r.stdout + r.stderr)
    return r.returncode


def main() -> int:
    """Plant each mutant in a temporary copy of the repository and report which ones are caught.

    Return 0 if the unmutated copy passes and every mutant makes a test fail, 1 otherwise.
    With --out the result table is also written as Markdown; --only restricts the run to the
    given mutant ids.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path); ap.add_argument("--only", default="")
    a = ap.parse_args()
    todo = [m for m in MUTANTS if not a.only or m[0] in a.only.split(",")]
    tmp = Path(tempfile.mkdtemp(prefix="swotgat_faults_"))
    root = tmp / "repo"
    shutil.copytree(REPO, root, ignore=shutil.ignore_patterns(".git", "__pycache__", "PYTEST_OUTPUT.txt"))
    rows, ok = [], True
    files = sorted({f for m in todo for f in m[4]})
    t0 = time.time()
    rc = run_pytest(root, files, tmp / "baseline.log")
    print(f"baseline (unmutated copy, {len(files)} test files): exit {rc} ({time.time() - t0:.0f} s)", flush=True)
    if rc != 0:
        print((tmp / "baseline.log").read_text()[-3000:]); return 1
    for mid, f, old, new, tests, what in todo:
        p = root / f; src = p.read_text()
        n = src.count(old)
        if n < 1:
            rows.append((mid, f, what, "MUTATION NOT APPLIED (text not found)", "")); ok = False; continue
        p.write_text(src.replace(old, new, 1))                        # first occurrence
        t0 = time.time()
        try:
            rc = run_pytest(root, tests, tmp / f"{mid}.log")
        finally:
            p.write_text(src)
        log = (tmp / f"{mid}.log").read_text()
        failed = [l.split(" - ")[0].replace("FAILED ", "").split("::")[-1] for l in log.splitlines() if l.startswith(("FAILED", "ERROR"))]
        caught = rc == 1
        ok &= caught
        rows.append((mid, f, what, "caught" if caught else f"SURVIVED (pytest exit {rc})", ", ".join(failed[:3]) + (f" (+{len(failed) - 3})" if len(failed) > 3 else "")))
        print(f"{mid:22s} {'caught' if caught else 'SURVIVED':9s} {time.time() - t0:5.0f} s  {', '.join(failed[:2])}", flush=True)
    n_caught = sum(r[3] == "caught" for r in rows)
    md = [f"# Fault injection result\n", f"{n_caught} of {len(rows)} planted bugs caught by the unit tests "
          f"(baseline on the unmutated copy passed).\n", "| Mutant | File | Planted bug | Result | Failing tests |",
          "|---|---|---|---|---|"]
    md += [f"| `{m}` | `{f}` | {w} | {r} | {t} |" for m, f, w, r, t in rows]
    text = "\n".join(md) + "\n"
    print("\n" + text)
    if a.out:
        a.out.write_text(text)
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
