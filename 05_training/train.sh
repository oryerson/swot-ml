#!/bin/bash
# Train, or resume, one SWOT-GAT seed on one GPU.
#
#   sbatch $SWOTGAT_SBATCH_GPU -t 30:00:00 -J swotgat_S1 05_training/train.sh \
#          $SWOTGAT_WORK/CONUS/era5_conus_v35c5_S1_storage.json
#
# The paper model needs 50,000 steps (about 64 GPU-hours on an NVIDIA L40S,
# 48 GB). One job stops itself after 27 hours (max_training_hours in the
# configuration), so submit_training.sh chains four jobs per seed. Each job
# resumes from the newest checkpoint; a run already at step 50,000 exits at once.
#
# Before training it checks that
#   * JAX sees a GPU;
#   * the training cache was built and verified by build_cache_v35.py;
#   * the data stores still have the content the cache was built from.
#
# Settings that matter for reproducibility: PYTHONHASHSEED=0 fixes the order in
# which each basin's graph edges are assembled, so every job and every resume
# sees the same graphs.
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --open-mode=append
set -eo pipefail
: "${SWOTGAT_WORK:?run 'source config/paths.env' before sbatch}"
CFG=$1
STEM=$(basename "$CFG" .json); CFG_DIR=$(dirname "$CFG")
LAUNCH=$SWOTGAT_WORK/CONUS_storage_fork/eval/v35_launch     # v35lib.py, build_cache_v35.py and logs live here
SRC=$SWOTGAT_REPO/vendor/swot_ml/src                        # model source
PY=$SWOTGAT_PYTHON
mkdir -p "$LAUNCH/logs"
echo "$(date -u) host $(hostname) job ${SLURM_JOB_ID:-none} config $CFG model source $SRC"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
test -f "$CFG" || { echo "FATAL: no configuration $CFG (run tools/deploy.py)"; exit 1; }

# Resume from the newest run folder of this configuration that has checkpoints.
RESUME=""
for d in $(ls -dt "$CFG_DIR/${STEM}"_2* 2>/dev/null); do
  [ -n "$(ls "$d/checkpoints" 2>/dev/null)" ] && { RESUME=$d; break; }
done
if [ -n "$RESUME" ] && [ -d "$RESUME/checkpoints/step_050000" ]; then
  echo "already at step 50,000 ($RESUME): nothing to do"; exit 0
fi

if [ -f /usr/share/lmod/lmod/init/bash ]; then
  source /usr/share/lmod/lmod/init/bash; module load cuda/12.6 || echo "note: 'module load cuda/12.6' failed"
fi
unset JAX_PLATFORMS   # a CPU-only setting inherited from the submitting shell would silently train on CPU
"$PY" -c "import jax, sys; d = jax.devices(); print('jax devices', d, flush=True); sys.exit(0 if d[0].platform == 'gpu' else 1)" \
  || { echo "FATAL: JAX sees no GPU"; exit 1; }
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION:-0.95}
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
[ -n "${SWOTGAT_CUDA_DIR:-}" ] && export XLA_FLAGS=--xla_gpu_cuda_data_dir=$SWOTGAT_CUDA_DIR

# The cache for this configuration must exist and be marked verified.
read -r HASH CACHE_DIR < <(cd "$SRC" && "$PY" -c "
from config import Config; import data.cached_basingraphdataset as c
k = Config.from_file('$CFG'); print(c.config_hash(k), k.cache_dir)" | tail -1)
"$PY" -c "import json, sys; sys.exit(0 if json.load(open('$CACHE_DIR/$HASH/V35_BUILT.json')).get('ok') else 1)" \
  || { echo "FATAL: cache $CACHE_DIR/$HASH was not built and verified by build_cache_v35.py"; exit 1; }
# The stores must still hold what the cache was built from (digest of every chunk; about 1 minute).
(cd "$LAUNCH" && "$PY" -c "
import json, sys, v35lib as V
built = json.load(open('$CACHE_DIR/$HASH/V35_BUILT.json'))['digests']; now = V.data_digests(json.load(open('$CFG')))
print('store digests', now, '| cache built from', built); sys.exit(0 if now == built else 1)") \
  || { echo "FATAL: the store content changed since the cache was built; rebuild the cache"; exit 1; }

cd "$SRC"
OUT=$LAUNCH/logs/train_${STEM}_${SLURM_JOB_ID:-local}.out
set +e
if [ -n "$RESUME" ]; then
  echo "RESUMING $RESUME (last checkpoint: $(ls "$RESUME/checkpoints" | sort | tail -1))"
  "$PY" -u run.py continue-train "$RESUME" 2>&1 | tee -a "$OUT"
else
  echo "starting from scratch"
  "$PY" -u run.py train "$CFG" 2>&1 | tee -a "$OUT"
fi
rc=${PIPESTATUS[0]}; set -e
[ "$rc" = 0 ] || { echo "training exited with status $rc"; exit "$rc"; }
echo "$(date -u) job finished"
