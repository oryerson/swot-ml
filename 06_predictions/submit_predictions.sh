#!/bin/bash
# Submit every prediction the paper's scores use, for both seeds:
#
#   tag     window                    forcing store            runner
#   x8085   1981-01-01..1985-12-31    zarr_store_ext1980_v35   full  (+ 1980-2026 long-memory store)
#   x8691   1986-01-01..1991-12-31    "                        "
#   x9297   1992-01-01..1997-12-31    "                        "
#   x9803   1998-01-01..2003-12-31    "                        "
#   x0409   2004-01-01..2009-12-31    "                        "
#   h10     2010-01-01..2020-12-31    "                        lowmem
#   x26     2023-03-01..2026-09-30    zarr_store_vd2026_v35    lowmem   (the SWOT Period)
#
# Each prediction reads 365 days of inputs before its window, so no window
# starts cold. With PLUS=1 the same windows are also predicted at the expanded
# gauge set (tags h10p, x26p, x8085p ...; a sensitivity set, not in the paper figures).
#
#   source config/paths.env && python tools/deploy.py
#   06_predictions/submit_predictions.sh      # DRYRUN=1 prints the sbatch lines only
#
# Needs the 1980-2026 long-memory store first:
#   bin/run 03_forcings/e_long_memory/build_lm_c5.py --workers 8     (CPU, about 1 hour, 128 GB)
#   bin/run 03_forcings/e_long_memory/check_lm_c5.py
set -eo pipefail
: "${SWOTGAT_WORK:?run 'source config/paths.env' first}"
: "${SWOTGAT_RUN_S1:?set SWOTGAT_RUN_S1 and SWOTGAT_RUN_S2 (run folder names) in config/paths.env}"
HERE=$SWOTGAT_REPO/06_predictions; C=$SWOTGAT_WORK/CONUS; F=$SWOTGAT_WORK/CONUS_storage_fork
EXT=$F/eval/v35_inputs/forcing/zarr_store_ext1980_v35
VD=$F/eval/v35_inputs/forcing/zarr_store_vd2026_v35
LM_8026=$F/eval/c5_paper/longmem_v35_c5_8026
NODES_H10=$F/eval/v35_launch/results/C5/nodes_h10.csv      # 527 reservoir nodes + 1,252 river gauge nodes
NODES_X26=$F/eval/v35_launch/results/C5/nodes_x26.csv      # 477 + 1,252
NODES_8109=$F/eval/c5_paper/nodes_r3_union.csv             # 529 reservoir nodes
PLUS_HIND=$F/gauges_expand/testsets/regime3_plus_1980_2020/predict_nodes.csv
PLUS_SWOT=$F/gauges_expand/testsets/regime1x_plus_2023_2026/predict_nodes.csv
LOG=$F/eval/c5_paper/logs; mkdir -p "$LOG"
if [ "${DRYRUN:-0}" = 1 ]; then sb() { echo "[dry run] sbatch $*" >&2; echo "JOB$RANDOM"; }
else sb() { sbatch --parsable "$@"; }; fi
submit() {  # submit <runner> <hours> <seed> <run dir> <tag> <store> <nodes> <start> <end> [<long-memory store>]
  local runner=$1 hours=$2 seed=$3; shift 3
  # shellcheck disable=SC2086  # the flag variable holds several words on purpose
  job=$(sb $SWOTGAT_SBATCH_GPU -t "$hours":00:00 -J "swotgat_pred_${seed}_$2" -o "$LOG/%x_%j.log" \
        --export=ALL,RUNNER="$runner" "$HERE/predict.sh" "$@")
  echo "$seed $2: job $job"
}
for S in S1:$C/$SWOTGAT_RUN_S1 S2:$C/$SWOTGAT_RUN_S2; do
  seed=${S%%:*}; RUN_DIR=${S#*:}
  submit lowmem 12 "$seed" "$RUN_DIR" h10 "$EXT" "$NODES_H10" 2010-01-01 2020-12-31
  submit lowmem 12 "$seed" "$RUN_DIR" x26 "$VD"  "$NODES_X26" 2023-03-01 2026-09-30
  if [ "${PLUS:-0}" = 1 ]; then
    submit full 12 "$seed" "$RUN_DIR" h10p "$EXT" "$PLUS_HIND" 2010-01-01 2020-12-31
    submit full 6  "$seed" "$RUN_DIR" x26p "$VD"  "$PLUS_SWOT" 2023-03-01 2026-09-30
  fi
  for w in x8085:1981-01-01:1985-12-31 x8691:1986-01-01:1991-12-31 x9297:1992-01-01:1997-12-31 \
           x9803:1998-01-01:2003-12-31 x0409:2004-01-01:2009-12-31; do
    IFS=: read -r tag start end <<< "$w"
    submit full 8 "$seed" "$RUN_DIR" "$tag" "$EXT" "$NODES_8109" "$start" "$end" "$LM_8026"
    [ "${PLUS:-0}" = 1 ] && submit full 8 "$seed" "$RUN_DIR" "${tag}p" "$EXT" "$PLUS_HIND" "$start" "$end" "$LM_8026"
  done
done
