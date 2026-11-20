# Smoke test: last result on the authors' cluster

| | |
|---|---|
| Date | 20 November 2026 (the runs of 16 and 19 November, jobs 1866030 and 2006754, gave the same result) |
| Where | UMass Unity cluster, one CPU node, 4 cores, 48 GB; Slurm job 2035400 |
| Run time | 4 min 40 s |
| Work tree | new and separate from the project's working folder; inputs linked read-only from it |
| Result | **passed**: 20 of 20 steps ran; 32 of 32 outputs reproduce the reference |

What was compared with the files behind the paper (`expected_md5.tsv`):

| Output | Result |
|---|---|
| Graph v3.5 JSON rebuilt from the sub-basin table | byte-identical |
| 10 score tables with the 50%-overlap rule (2010-2020, SWOT Period, 1981-1990, matched sets, paired tests) | byte-identical |
| 8 full-record tables (1981-2020 scores with the rule-model ensemble, pinned-storage check, Figure 1 picks, gauge standard deviations) | byte-identical |
| 5 figure tables written by `08_figures/prepare_kit.py` | byte-identical |
| Figure numbers `num_c5d_fig2.json`, `fig3`, `fig4` | byte-identical |
| Figure numbers `num_c5d_fig1.json` | equal values: 13 of 138 numbers differ in the 16th significant digit (compared with the copy in `tests/smoke/expected/`) |
| Figure images `c5d_fig1.png` … `c5d_fig4.png` | byte-identical |

Also checked in the same run (`static_checks.py`): the two model
configurations render to the md5 sums of the files the paper's runs were
trained from; all 54 files of `vendor/swot_ml/src` match the source the runs
imported; every Python file of the repository compiles.

Not covered by the smoke test: delineation, static attributes, forcing and
target stores, training, prediction, the gauge downloads, the rule-model
simulations, the driver tables and `text_nums_o50.py`. Their scripts are the
ones that ran, with only path names changed (`PROVENANCE.md`). The job scripts
of stages 05-06 were dry-run only: `DRYRUN=1 05_training/submit_training.sh`
prints 1 cache job and 8 training jobs, `DRYRUN=1 06_predictions/submit_predictions.sh`
prints 14 prediction jobs.

A second run in the same work tree stops at once with a message: `decade.py`
and `full.py` do not overwrite their output folders (`score_j`, `score_full`).
Remove those two folders of the smoke-test work tree first.
