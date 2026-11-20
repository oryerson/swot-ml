#!/bin/bash
# One SWOT-GAT prediction run: one trained seed, one time window, one node list.
#
#   sbatch $SWOTGAT_SBATCH_GPU -t 12:00:00 06_predictions/predict.sh \
#          <run folder> <tag> <forcing store> <nodes.csv> <start> <end> [<long-memory store>]
#
# Output: $SWOTGAT_WORK/CONUS_storage_fork/predictions/<run>/step_050000_<tag>.parquet
# with one row per node and day (pred_dv_SWOTAE = mean of the Gaussian mixture,
# in million cubic metres) and a .json sidecar. An existing output is not redone.
#
# RUNNER selects the prediction code, as used for the paper:
#   RUNNER=full    pred_c5.py: the model's own sequence-to-sequence pass. Used for
#                  the 1981-2009 windows. The optional 7th argument replaces the
#                  long-memory store named in the checkpoint (needed before 2009,
#                  where the training store has no long-memory inputs).
#   RUNNER=lowmem  pred_v35_lowmem.py: applies the output heads to the last day
#                  only and frees per-day graph traces. Same numbers by
#                  construction, less GPU memory. Used for 2010-2020 and the
#                  SWOT Period.
# Both runners check the inputs before predicting (every graph node of every
# basin is in the store, normalisation statistics equal the training store's)
# and the outputs afterwards (every requested node finite on every day).
#SBATCH -c 12
#SBATCH --mem=110G
#SBATCH --gres=gpu:1
#SBATCH --requeue
set -eo pipefail
: "${SWOTGAT_WORK:?run 'source config/paths.env' before sbatch}"
RUN_DIR=$1; TAG=$2; STORE=$3; NODES=$4; START=$5; END=$6; LM_STORE=${7:-}
RUNNER=${RUNNER:-full}; STEP=step_050000
F=$SWOTGAT_WORK/CONUS_storage_fork; PY=$SWOTGAT_PYTHON; R=$SWOTGAT_REPO
RUN=$(basename "$RUN_DIR"); CKPT=$RUN_DIR/checkpoints/$STEP
test -f "$CKPT/model.eqx" || { echo "FATAL: no checkpoint $CKPT"; exit 1; }
OUT=$F/predictions/$RUN/${STEP}_${TAG}.parquet
if [ -f "$OUT" ] && [ -f "${OUT%.parquet}.json" ]; then echo "exists: $OUT"; exit 0; fi
echo "$(date -u) host $(hostname) run $RUN $STEP tag $TAG runner $RUNNER store $STORE nodes $NODES $START..$END long-memory ${LM_STORE:-from checkpoint}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Node-local cache for the prediction data set (needs about 25 GB), removed at exit.
T=${TMPDIR:-/tmp}; [ "$(df -BG --output=avail "$T" | tail -1 | tr -dc 0-9)" -ge 25 ] || T=$F/eval/c5_paper/tmpcache
CACHE=$T/swotgat_pred_${SLURM_JOB_ID:-$$}; mkdir -p "$CACHE"; trap 'rm -rf "$CACHE"' EXIT

if [ -f /usr/share/lmod/lmod/init/bash ]; then source /usr/share/lmod/lmod/init/bash; module load cuda/12.6 || true; fi
unset JAX_PLATFORMS
"$PY" -c "import jax, sys; d = jax.devices(); print('jax devices', d, flush=True); sys.exit(0 if d[0].platform == 'gpu' else 1)" \
  || { echo "FATAL: JAX sees no GPU"; exit 1; }
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 IP_CACHE=$CACHE
[ -n "${SWOTGAT_CUDA_DIR:-}" ] && export XLA_FLAGS=--xla_gpu_cuda_data_dir=$SWOTGAT_CUDA_DIR

case $RUNNER in
  full)
    export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
    [ -n "$LM_STORE" ] && export C5_LM_STORE=$LM_STORE
    "$R/bin/run" 06_predictions/pred_assert_c5.py pre "$CKPT" "$STORE" "$NODES" "$START" "$END"
    "$R/bin/run" 06_predictions/pred_c5.py "$CKPT" --zarr "$STORE" --nodes "$NODES" --start "$START" --end "$END" --tag "$TAG"
    "$R/bin/run" 06_predictions/pred_assert_c5.py post "$CKPT" "$OUT" "$NODES" "$START" "$END" ;;
  lowmem)
    [ -z "$LM_STORE" ] || { echo "FATAL: RUNNER=lowmem takes the long-memory store from the checkpoint"; exit 1; }
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    "$R/bin/run" 06_predictions/pred_assert_v35.py pre "$CKPT" "$STORE" "$NODES" "$START" "$END"
    "$R/bin/run" 06_predictions/pred_v35_lowmem.py "$CKPT" --zarr "$STORE" --nodes "$NODES" --start "$START" --end "$END" --tag "$TAG"
    "$R/bin/run" 06_predictions/pred_assert_v35.py post "$CKPT" "$OUT" "$NODES" "$START" "$END" ;;
  *) echo "FATAL: RUNNER must be full or lowmem"; exit 1 ;;
esac
echo "$(date -u) done"
