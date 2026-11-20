"""Predict with a trained SWOT-GAT checkpoint using less GPU memory: the low-memory runner.

This is a wrapper around run_predict_ext.py, like pred_c5.py, with two settings that change only how much memory
inference needs. It was written to run 365-day sequences on 48 GB GPUs. For the paper it made the 2010-2020
window and the SWOT Period (06_predictions/predict.sh with RUNNER=lowmem).

Usage: python pred_v35_lowmem.py <checkpoint_dir> [arguments of run_predict_ext.py ...]
  Environment variables
    IP_CACHE          folder for the prediction cache (required)
    REMAT_INFER       default 1 (see below)
    INFER_LAST_ONLY   default 1 (see below)
    NODES_PER_BATCH   optional (see below)
  The long-memory store is always the one named in the checkpoint configuration; this runner has no C5_LM_STORE.
  The script prints the GPU's peak memory at the end.

The memory settings
  REMAT_INFER (default 1) sets models.sparse_fusion_gann.REMAT_LEVEL before the model is first traced, as the
    trainer does from the configuration's remat_level. run_predict_ext.py never sets it, so without this wrapper
    inference runs at level 0. Level 1 only stops the scan over time from stacking the per-day graph traces.
    Prediction never reads those traces (return_weights and supervised_attn are false), so the arithmetic is
    unchanged. In a test with this setting alone the traces turned out not to be the memory cost: the peak was
    the same 7.78 GiB at 3,000 nodes per batch, and 12,000 nodes per batch ran out of memory at 11.2 GiB as
    before.
  INFER_LAST_ONLY (default 1) runs the model with seq2seq switched off, so the output heads are applied to the
    final hidden state only instead of to all 365 days (the model's own non-seq2seq branch). Prediction keeps
    only the last day of each sequence anyway (`_last_step` in evaluate/inference.py), and the final state is
    the hidden state of that last day, so the outputs are the same by construction.
  NODES_PER_BATCH overrides target_nodes_per_batch of the configuration. It must exceed the largest basin (on
    graph version 2.2 the Mississippi, USGS-07374000, with 10,727 nodes), so on its own it can save at most
    about 11%. It was used only for a reference run in testing.

What else the wrapper does
  It imports the model source named in $SWOTGAT_WORK/CONUS_storage_fork/eval/v35_launch/SRC before it loads
  run_predict_ext.py, asserts that run_predict_ext.py uses that source, sets the cache root to IP_CACHE, and
  calls run_predict_ext.main() with the remaining arguments.
"""
from swotgat.paths import WORK
import importlib.util
import os
import sys
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork"); STAGE = Path(open(F/"eval/v35_launch/SRC").read().strip())
sys.path.insert(0, str(STAGE))
import config, models, data, train, data.cached_basingraphdataset as cbd, evaluate.inference   # noqa: E401,E402,F401
import models.sparse_fusion_gann as sfg                                                         # noqa: E402
assert str(STAGE) in cbd.__file__, cbd.__file__
assert str(STAGE) in sfg.__file__, sfg.__file__
spec = importlib.util.spec_from_file_location("run_predict_ext", F/"eval/overnight/run_predict_ext.py"); rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)
assert str(STAGE) in rp.cbd.__file__, f"run_predict_ext picked up {rp.cbd.__file__}"
assert sys.modules["models.sparse_fusion_gann"] is sfg
sfg.REMAT_LEVEL = int(os.environ.get("REMAT_INFER", "1"))
N = os.environ.get("NODES_PER_BATCH")
if N:
    _from_file = rp.Config.from_file

    class _Cfg:                                     # run_predict_ext.py uses Config only through from_file
        @staticmethod
        def from_file(p):
            c = _from_file(p); c.target_nodes_per_batch = int(N); return c
    rp.Config = _Cfg
LAST = os.environ.get("INFER_LAST_ONLY", "1") == "1"
if LAST:
    import equinox as eqx
    _mdi = rp.model_df_iterate

    def _model_df_iterate_last(model, *args, **kw):
        assert model.seq2seq, "expected a seq2seq checkpoint"
        return _mdi(eqx.tree_at(lambda m: m.seq2seq, model, False), *args, **kw)
    rp.model_df_iterate = _model_df_iterate_last
print(f"swot-ml source: {rp.cbd.__file__} | cache root: {os.environ['IP_CACHE']} | REMAT_LEVEL {sfg.REMAT_LEVEL} | "
      f"nodes per batch {N or 'from config'} | heads on last day only {LAST}", flush=True)
rp.C = Path(os.environ["IP_CACHE"])
sys.argv = ["run_predict_ext.py", str(Path(sys.argv[1]).resolve())] + sys.argv[2:]
rp.main()
import jax                                                                                      # noqa: E402
st = jax.devices()[0].memory_stats() or {}
print(f"GPU peak_bytes_in_use {st.get('peak_bytes_in_use', 0)/2**30:.2f} GiB of limit {st.get('bytes_limit', 0)/2**30:.2f} GiB | "
      f"matmul precision {jax.config.jax_default_matmul_precision}", flush=True)
