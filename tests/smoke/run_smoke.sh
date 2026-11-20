#!/bin/bash
# End-to-end smoke test of the release copy, without training or prediction:
#   1. rebuild the graph v3.5 JSON from the sub-basin table (stage 01);
#   2. render the two model configurations and check the vendored model source;
#   3. re-score the stored SWOT-GAT and rule-model runs against the gauges (stage 07c);
#   4. rebuild the figure tables and the four paper figures (stage 08);
#   5. compare every output with tests/smoke/expected_md5.tsv.
# Runs on one CPU node in about 10 minutes with 48 GB of memory.
#
#   source config/paths.env            # a NEW, separate work tree
#   python tools/deploy.py
#   tests/smoke/link_inputs.sh <existing project root> tests/smoke/inputs_eval.txt
#   sbatch $SWOTGAT_SBATCH_CPU -c 4 --mem=48G -t 1:00:00 tests/smoke/run_smoke.sh
#SBATCH -J swotgat_smoke
set -uo pipefail
: "${SWOTGAT_WORK:?source config/paths.env first}"
R=$SWOTGAT_REPO; E=$SWOTGAT_WORK/CONUS_storage_fork/eval
LOG=$SWOTGAT_WORK/smoke_logs; mkdir -p "$LOG"
FAILED=0
step() {  # step <name> <command...>: run, log, keep going on failure
  local name=$1; shift
  if "$@" > "$LOG/$name.log" 2>&1; then echo "ok    $name"; else echo "FAIL  $name (see $LOG/$name.log)"; FAILED=1; fi
}
echo "smoke test on $(hostname) at $(date -Iseconds); repo $R; work tree $SWOTGAT_WORK"

# decade.py and full.py refuse to overwrite their output folders; say so before anything runs
for d in "$E/overlap50/score/score_j" "$E/grl_figs/intermediate/c5_o50_20261113/score/score_full"; do
  if [ -e "$d" ]; then
    echo "STOP  $d exists already (an earlier smoke run, or this is not a separate work tree)."
    echo "      In a smoke-test work tree, remove that folder and submit again; never remove it in a real work tree."
    exit 2
  fi
done

source "$R/07_evaluation/c_scoring/env.sh"   # the two seeds, the series file, fixed hash seed

# 1. graph
mkdir -p "$SWOTGAT_WORK/smoke_out"
step graph "$R/bin/run" --python DELINEATOR_PYTHON 01_graph/build_conus_graphs_v2.py \
     --subbasins "$E/graph_v3/v35/subbasins_conus_v35.parquet" --out "$SWOTGAT_WORK/smoke_out/graphs_conus_v35.json"

# 2. configurations and model source
step static_checks "$SWOTGAT_PYTHON" "$R/tests/smoke/static_checks.py"

# 3. scoring with the 50%-of-days overlap rule (BASE=1 adds the two baselines)
export BASE=1
step score_2010      "$R/bin/run" 07_evaluation/c_scoring/overlap50/score_2010.py
step score_x26span   "$R/bin/run" 07_evaluation/c_scoring/overlap50/score_x26span.py
step score_decades   "$R/bin/run" 07_evaluation/c_scoring/overlap50/decade.py
step matched_sets    "$R/bin/run" 07_evaluation/c_scoring/overlap50/fig1_nums.py "$E/overlap50/score" "$E/overlap50/fig1_o50.json"
mkdir -p "$E/overlap50/tables"
step paired_tests    "$R/bin/run" 07_evaluation/c_scoring/overlap50/analyze.py
#    full-record windows and the rule-model ensemble (ENS=1)
export BASE=0 ENS=1 HIND=2010
step score_full      "$R/bin/run" 07_evaluation/c_scoring/full_record/full.py
step score_x26_ens   "$R/bin/run" 07_evaluation/c_scoring/full_record/score_x26span.py
step check_score     "$R/bin/run" 07_evaluation/c_scoring/full_record/check_score.py
step pinned_check    "$R/bin/run" 07_evaluation/c_scoring/full_record/pinned_check.py
step picks           "$R/bin/run" 07_evaluation/c_scoring/full_record/picks.py
step picks_d         "$R/bin/run" 07_evaluation/c_scoring/full_record/picks_d.py
step sd_gauge_8120   "$R/bin/run" 07_evaluation/c_scoring/full_record/sd8120.py

# 4. figures
step prepare_kit     "$R/bin/run" 08_figures/prepare_kit.py
for f in fig1_c5d fig2_c5d fig3_c5d fig4_c5d; do step "$f" "$R/bin/run" 08_figures/$f.py; done
if command -v pdflatex > /dev/null; then
  step figures_pdf "$R/bin/run" 08_figures/figs_only_c5d.py "$E/grl_figs/intermediate/c5_o50_20261113/kit_d"
else echo "skip  figures_pdf (no pdflatex on this node)"; fi

# 5. compare
"$SWOTGAT_PYTHON" "$R/tests/smoke/compare.py" "$R/tests/smoke/expected_md5.tsv" | tee "$LOG/compare.log"
[ "${PIPESTATUS[0]}" = 0 ] || FAILED=1
[ $FAILED = 0 ] && echo "SMOKE TEST PASSED" || echo "SMOKE TEST FAILED"
exit $FAILED
