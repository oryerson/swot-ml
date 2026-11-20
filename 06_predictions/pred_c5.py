"""Predict with a trained SWOT-GAT checkpoint: the full sequence-to-sequence runner of the paper (C5).

This is a thin wrapper around run_predict_ext.py, which does the prediction. For the paper it made the
1981-2009 windows (06_predictions/predict.sh with RUNNER=full). pred_v35_lowmem.py is the other runner.

Usage: python pred_c5.py <checkpoint_dir> [arguments of run_predict_ext.py ...]
  for example: pred_c5.py <run>/checkpoints/step_050000 --zarr <store> --nodes nodes.csv --start 1981-01-01
               --end 1985-12-31 --tag x8085
  Environment variables
    IP_CACHE      folder for the prediction cache (required). predict.sh points it to node-local disk and
                  removes it afterwards.
    C5_LM_STORE   optional: a long-memory store that replaces the one named in the checkpoint configuration

What the wrapper does
  1. It imports the model source named in $SWOTGAT_WORK/CONUS_storage_fork/eval/v35_launch/SRC (written by
     tools/deploy.py; vendor/swot_ml/src) before it loads run_predict_ext.py, and asserts that run_predict_ext.py
     then uses that source and not its own default path.
  2. It sets the cache root of run_predict_ext.py to IP_CACHE.
  3. If C5_LM_STORE is set, it wraps Config.from_file so that extra_dynamic_store becomes that store. The
     long-memory store used in training starts in 2009. Windows before that need the 1980-2026 store built by
     03_forcings/e_long_memory/build_lm_c5.py, which holds the same two inputs.
  4. It calls run_predict_ext.main() with the remaining arguments.
"""
from swotgat.paths import WORK
import importlib.util
import os
import sys
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork"); STAGE = Path(open(f'{WORK}/CONUS_storage_fork/eval/v35_launch/SRC').read().strip())   # model source folder, read from v35_launch/SRC
sys.path.insert(0, str(STAGE))
import config, models, data, train, data.cached_basingraphdataset as cbd, evaluate.inference   # noqa: E401,E402,F401
assert str(STAGE) in cbd.__file__, cbd.__file__
spec = importlib.util.spec_from_file_location("run_predict_ext", F/"eval/overnight/run_predict_ext.py"); rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)
assert str(STAGE) in rp.cbd.__file__, f"run_predict_ext picked up {rp.cbd.__file__}"
print(f"swot-ml source: {rp.cbd.__file__} | cache root: {os.environ['IP_CACHE']}", flush=True)
rp.C = Path(os.environ["IP_CACHE"])
sys.argv = ["run_predict_ext.py", str(Path(sys.argv[1]).resolve())] + sys.argv[2:]
LM = os.environ.get("C5_LM_STORE")
if LM:   # the 1980-2026 store of C5's two long-memory inputs replaces the extra_dynamic_store of the checkpoint configuration
    _Base = rp.Config
    class _Cfg:
        @staticmethod
        def from_file(p, *a, **k):
            c = _Base.from_file(p, *a, **k); old = c.extra_dynamic_store; c.extra_dynamic_store = Path(LM)
            assert list(c.extra_dynamic_vars) and str(c.extra_dynamic_store) == LM, (c.extra_dynamic_vars, c.extra_dynamic_store)
            print(f"extra_dynamic_store {old} -> {c.extra_dynamic_store} (inputs {list(c.extra_dynamic_vars)})", flush=True); return c
    rp.Config = _Cfg
rp.main()
