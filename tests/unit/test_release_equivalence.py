"""8. Release equivalence on the paper's real model and data, and 9. configuration rendering.

8. One real batch (two basins of the paper's training cache, 365 days) goes through
   (a) the release's vendored model source (vendor/swot_ml/src) and
   (b) the original staged source the paper's runs imported
       (CONUS_storage_fork/eval/v35_launch/stage_src_v35),
   each with the trained C5 checkpoint at step 50,000, seed S1 and seed S2. The batches built by
   the two copies, the model outputs and the training loss must be identical, bit for bit.
   The same run also compares the last-day predictions of the two prediction runners (full:
   REMAT_LEVEL 0, seq2seq, last day; low-memory: REMAT_LEVEL 1, heads on the final state only).
   This is the CPU, one-batch form of docs/OPEN_ITEMS.md item 9; it does not re-run the
   prediction scripts or their data loaders over a whole window.
   An open-file audit during dataset construction, sampling and the forward pass checks that
   no gauge file is read.
9. The two configuration templates, rendered with the original project root, have the md5
   sums of the configurations the paper's runs were launched with (e91fd492..., 55881f63...),
   and the checkpoint's own config.json agrees with them on every training setting.

Needs the authors' cluster (checkpoints, cache, stores); skipped elsewhere. Read-only.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import ORIGINAL_ROOT, REPO, RUN_S1, RUN_S2, SRC, STAGED_SRC, TEMPLATES, on_cluster

WORKER = Path(__file__).with_name("_equivalence_worker.py")
GAUGE = re.compile(r"testset|truth_clean|resops|regime\d|gauge|nwis|agency", re.I)
CONFIG_MD5 = {"S1": "e91fd492fd5861fd3430869fc1583599", "S2": "55881f6336c7003399bfcbfc5d417920"}
RUNS = {"S1": RUN_S1, "S2": RUN_S2}


# ------------------------------------------------------------------ 9. configuration rendering
@pytest.mark.parametrize("seed", ["S1", "S2"])
def test_templates_render_to_paper_configs(seed):
    """Check that a template rendered with the original root has the md5 of the paper's config."""
    text = TEMPLATES[seed].read_text().replace("${SWOTGAT_WORK}", ORIGINAL_ROOT)
    assert hashlib.md5(text.encode()).hexdigest() == CONFIG_MD5[seed]
    assert "${" not in text


def test_deploy_render_matches_static_check(monkeypatch, tmp_path):
    """Check that tools/deploy.py's render() (the function users run) gives the same bytes."""
    monkeypatch.setenv("SWOTGAT_WORK", ORIGINAL_ROOT)
    monkeypatch.setenv("SWOTGAT_PATHS_FILE", str(tmp_path / "none.env"))
    for m in [m for m in sys.modules if m.startswith("swotgat")]:
        monkeypatch.delitem(sys.modules, m)
    sys.path.insert(0, str(REPO / "tools"))
    try:
        import importlib
        deploy = importlib.import_module("deploy")
        for seed, p in TEMPLATES.items():
            assert hashlib.md5(deploy.render(p.read_text()).encode()).hexdigest() == CONFIG_MD5[seed]
    finally:
        sys.path.remove(str(REPO / "tools"))
        sys.modules.pop("deploy", None)


@on_cluster
@pytest.mark.parametrize("seed", ["S1", "S2"])
def test_launched_and_checkpoint_configs_agree(seed):
    """Check that the launched configuration and the checkpoint's config.json match the template.

    The checkpoint must also record 12 dense inputs, no sparse source and the area scaling of
    dv_SWOTAE.
    """
    launched = Path(ORIGINAL_ROOT) / "CONUS" / TEMPLATES[seed].name.replace(".template", "")
    rendered = TEMPLATES[seed].read_text().replace("${SWOTGAT_WORK}", ORIGINAL_ROOT)
    assert launched.read_text() == rendered
    t = json.loads(rendered)
    ck = json.loads((RUNS[seed] / "checkpoints/step_050000/config.json").read_text())
    for k, v in t.items():
        if k.startswith("_") or k in ("model_args", "step_kwargs", "features"):
            continue
        cv = ck.get(k)
        if k.endswith("date_range") or k == "rereference_window":
            cv = [str(x)[:10] for x in cv]; v = [str(x)[:10] for x in v]
        assert cv == v, (k, cv, v)
    for k, v in t["model_args"].items():
        assert ck["model_args"][k] == v, k
    assert ck["model_args"]["dense_sizes"] == {"era5": 12} and ck["model_args"]["sparse_sizes"] == {}
    assert ck["model_args"]["area_scaled"][0]["target"] == "dv_SWOTAE"
    for k, v in t["step_kwargs"].items():
        assert ck["step_kwargs"][k] == v, k
    assert ck["features"]["dynamic"]["era5"] == t["features"]["dynamic"]["era5"] + t["extra_dynamic_vars"]
    assert ck["features"]["target"] == t["features"]["target"]


# ------------------------------------------------------------------ 8. equivalence on real data
def _worker(src: Path, ckpt: Path, out: Path, batch: Path | None = None, audit: Path | None = None):
    cmd = [sys.executable, str(WORKER), "--src", str(src), "--ckpt", str(ckpt), "--out", str(out)]
    if batch:
        cmd += ["--batch", str(batch)]
    if audit:
        cmd += ["--audit", str(audit)]
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3000, cwd=out.parent)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    print(r.stdout.strip().splitlines()[-1])
    return dict(np.load(out, allow_pickle=False))


@pytest.fixture(scope="module", params=["S1", "S2"])
def runs(request, tmp_path_factory):
    """Run the worker three times for one seed and return its results.

    The return value is (seed, folder, release, staged, staged on release batch). The last three
    are the worker's saved arrays: the release source and the staged source, each on the batch it
    built itself, and the staged source on the batch the release source built. Skipped without
    the paper's checkpoints and staged source.
    """
    if not (RUNS[request.param] / "checkpoints/step_050000/model.eqx").is_file() or not STAGED_SRC.is_dir():
        pytest.skip("needs the paper's checkpoints, cache and staged source (authors' cluster)")
    d = tmp_path_factory.mktemp(f"equiv_{request.param}")
    ck = RUNS[request.param] / "checkpoints/step_050000"
    rel = _worker(SRC, ck, d / "release.npz", audit=d / "audit_release.json")
    stg = _worker(STAGED_SRC, ck, d / "staged.npz", audit=d / "audit_staged.json")
    stg_on_rel = _worker(STAGED_SRC, ck, d / "staged_on_release_batch.npz", batch=d / "release.npz")
    return request.param, d, rel, stg, stg_on_rel


def _keys(r, prefix):
    return sorted(k for k in r if k.startswith(prefix))


def test_staged_and_release_source_files_identical():
    """Check that every .py file of vendor/swot_ml/src equals the staged copy the runs imported."""
    if not STAGED_SRC.is_dir():
        pytest.skip("needs the staged source (authors' cluster)")
    rel = {p.relative_to(SRC): p.read_bytes() for p in SRC.rglob("*.py")}
    for rp, b in rel.items():
        assert (STAGED_SRC / rp).read_bytes() == b, rp


def test_same_batch_from_both_sources(runs):
    """Check that the batch built from the paper's training cache is the same for both copies."""
    _, _, rel, stg, _ = runs
    for k in _keys(rel, "dyn_") + _keys(rel, "y_") + ["graph_edges", "static", "node_mask", "edge_mask", "graph_idx", "basins", "dates"]:
        np.testing.assert_array_equal(rel[k], stg[k], err_msg=k)
    assert rel["node_mask"].sum() >= 50                         # two real basins


def test_identical_outputs_and_loss(runs):
    """Check that the two source copies give bit-for-bit identical outputs, loss and predictions.

    The same must hold when the staged copy runs on the batch the release copy built.
    """
    seed, _, rel, stg, stg2 = runs
    outs = _keys(rel, "out_")
    assert len(outs) == 6                                       # 2 targets x (mu, sigma, log_pi)
    for k in outs + ["loss"] + _keys(rel, "pred_"):
        np.testing.assert_array_equal(rel[k], stg[k], err_msg=k)
        np.testing.assert_array_equal(rel[k], stg2[k], err_msg=k)
    assert np.all(np.isfinite(rel["out_dv_SWOTAE_mu"])) and np.isfinite(rel["loss"])


def test_full_and_lowmem_runner_give_the_same_prediction(runs):
    """Check that the full and the low-memory runner's model calls predict the same last day.

    The largest absolute difference per target (MCM) is written to runner_maxdiff.json in the
    test's temporary folder.
    """
    seed, d, rel, _, _ = runs
    rep = {}
    for t in ("dv_SWOTAE", "dv_river"):
        a, b = rel[f"pred_full_{t}"], rel[f"pred_lowmem_{t}"]
        rep[t] = float(np.max(np.abs(a - b)))
        np.testing.assert_allclose(b, a, rtol=1e-5, atol=1e-5 * max(1.0, float(np.abs(a).max())))
    (d / "runner_maxdiff.json").write_text(json.dumps(rep))
    print(seed, "max |full - lowmem| (MCM):", rep)


def test_no_gauge_file_opened(runs):
    """Check that the open-file audit saw no gauge file and that no gauge package was imported.

    Data files are the files opened under /work or /nas, Python environments and source files
    aside.
    """
    _, d, _, _, _ = runs
    for name in ("audit_release.json", "audit_staged.json"):
        audit = json.loads((d / name).read_text())
        assert audit["modules"] == [], audit["modules"]          # no gauge package is imported
        # data files: everything under /work or /nas except Python environments and source trees
        # (importlib.metadata scans every installed package's entry_points.txt, e.g. of the
        # `global_gauges` package that happens to be installed in the environment)
        data_files = [p for p in audit["opened"] if p.startswith(("/work", "/nas")) and "/site-packages/" not in p
                      and not p.endswith(".py") and "/.venv/" not in p]
        assert data_files, "audit recorded nothing"
        bad = [p for p in data_files if GAUGE.search(p)]
        assert not bad, bad[:20]
        roots = sorted({"/".join(Path(p).parts[:8]) for p in data_files})
        print(name, len(data_files), "data files opened under", roots)


@on_cluster
@pytest.mark.parametrize("seed", ["S1", "S2"])
def test_checkpoint_parameter_count(seed):
    """Check that the trained checkpoint has 300 static columns and 2,239,256 parameters."""
    ck = json.loads((RUNS[seed] / "checkpoints/step_050000/config.json").read_text())
    assert ck["model_args"]["static_size"] == 300
    code = (f"import sys; sys.path.insert(0, {str(SRC)!r}); import json, equinox as eqx, models; from config import Config; "
            f"cfg = Config.from_file({str(RUNS[seed] / 'checkpoints/step_050000/config.json')!r}); _, t = models.make(cfg); "
            f"m = eqx.tree_deserialise_leaves({str(RUNS[seed] / 'checkpoints/step_050000/model.eqx')!r}, t); "
            f"print('NPARAM', models.count_parameters(m))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600,
                       env=dict(os.environ, JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1"))
    assert r.returncode == 0, r.stderr[-3000:]
    n = int([l for l in r.stdout.splitlines() if l.startswith("NPARAM")][0].split()[1])
    assert n == 2_239_256
