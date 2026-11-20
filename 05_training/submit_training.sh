#!/bin/bash
# Submit the full training of the paper model: build and verify the training
# cache (CPU), then train the two seeds, each as four chained 30-hour GPU jobs.
#
#   source config/paths.env && python tools/deploy.py
#   05_training/submit_training.sh            # DRYRUN=1 prints the sbatch lines only
#
# The two configurations differ only in the random seed (0 and 1). The paper
# reports the mean of the two trained models' predictions.
set -eo pipefail
: "${SWOTGAT_WORK:?run 'source config/paths.env' first}"
HERE=$SWOTGAT_REPO/05_training
LAUNCH=$SWOTGAT_WORK/CONUS_storage_fork/eval/v35_launch; LOG=$LAUNCH/logs; mkdir -p "$LOG"
CFG1=$SWOTGAT_WORK/CONUS/era5_conus_v35c5_S1_storage.json
CFG2=$SWOTGAT_WORK/CONUS/era5_conus_v35c5_S2_storage_s1.json
test -f "$CFG1" && test -f "$CFG2" || { echo "FATAL: configurations missing; run tools/deploy.py"; exit 1; }
if [ "${DRYRUN:-0}" = 1 ]; then sb() { echo "[dry run] sbatch $*" >&2; echo "JOB$RANDOM"; }
else sb() { env -u JAX_PLATFORMS sbatch --parsable "$@"; }; fi

# shellcheck disable=SC2086  # the flag variables hold several words on purpose
CACHE=$(sb $SWOTGAT_SBATCH_CPU -c 16 --mem=128G -t 4:00:00 -J swotgat_cache -o "$LOG/%x_%j.log" \
        --export=ALL,PYTHONHASHSEED=0,JAX_PLATFORMS=cpu \
        --wrap "$SWOTGAT_REPO/bin/run 05_training/build_cache_v35.py $CFG1 $CFG2 --min-gb 30")
for cfg in "$CFG1" "$CFG2"; do
  seed=$(basename "$cfg" .json | sed 's/.*_\(S[12]\)_.*/\1/'); dep=afterok:$CACHE
  for segment in 1 2 3 4; do
    # shellcheck disable=SC2086
    job=$(sb $SWOTGAT_SBATCH_GPU -t 30:00:00 -J swotgat_$seed -o "$LOG/%x_%j.log" \
          --export=ALL,PYTHONHASHSEED=0 --dependency=$dep "$HERE/train.sh" "$cfg")
    dep=afterany:$job
  done
  echo "seed $seed: 4 chained jobs, last $job"
done
