# What is not in this repository, and why

The project's working folders hold many more scripts than the paper needs.
This release contains only the lineage of the paper's model (configuration
"C5" on graph v3.5) and of the paper's scores and figures.

## Other model configurations (test cases)

| Left out | What it was |
|---|---|
| C1, C2, C3, C4, C6 on graph v3.5 | 180-day window; with and without long-memory inputs; fine-tuned variants. Configurations, launch scripts, gates and decision scripts (`v35_launch/gate_v35.*`, `launch_v35.sh`, `c4/make_c4.py`, `launch_c4_*.sh`, `launch_c6.sh`, `ft_*.sh`, `finetune_v35.py`, `decide_*.py`, `summary_*.py`, `ip_compare_*.py`, `predgate_*.py`, pre-registration notes) |
| C5w, C5s, C5abs | river water-level target, plain storage target, absolute-storage model |
| A180, A365, "lmdft" and earlier (v7 to v12) | models on earlier graph versions and earlier targets, their figure sets and scorers |
| Loss fine-tunes (`a365_lossft/`) | alternative loss functions |
| Feature screens (`featscreen_v2/`, except the long-memory builder), `perf_attrib/` except the four scripts the figures need | input screening and skill-attribution studies |
| Hold-out and ablation experiments | `build_*holdout*.py`, `build_river_ablation_masks.py`, `make_fb_*`, `make_weekend_*`, `probe_*.py`, `audit_*.py` in `CONUS/preprocess` |
| Soil-moisture and discharge inputs | `extract_soilmoisture_conus.py`, `merge_soil_conus.py`, `merge_gauges_conus.py`: not inputs or targets of the paper model |

## Earlier graph versions

| Left out | Note |
|---|---|
| The August 2026 delineation (`delineate_basin.py`, `submit_delineate_array.sh`, `build_conus_graphs.py`) | Graph v3.5 is a full re-delineation; `delineate_basin_v34.py` and `build_conus_graphs_v2.py` replace them |
| Graph v2, v2.1, v2.2, v2.3 edit chains (`graph_v2/`, `graph_map/v21_build` … `v23_build`) | Their net effect on v3.5 is the 106-row edit list `01_graph/inputs/carry_edits_v34.csv` |
| Graph v3.0 to v3.3 builds, the first failed v3.4 run, the aborted first v3.5 build | superseded |
| Review maps, web pages and reports (`build_review_v34.py`, `web_v3*`, `make_report_*.py`, `graph_audit_v35/`) | review tooling; the rulings themselves are in `01_graph/inputs/` |
| First statics run (`*_r1`), control reproductions | superseded by the second run, which is the one shipped |

## Superseded or intermediate evaluation code

| Left out | Note |
|---|---|
| Scorer generations before the 50%-overlap rule (`c5_paper/fig_c5` scorers, `baselines_c5/score/gate`, `nse_direct/`) and the patch scripts that derived each generation from the previous one (`stage_o50.py`, `stage_score.py`, `stage_nse.py`) | The final scorers are shipped as they ran. One earlier file is kept (`normal_year_source/score_fullrecord.py`) because `bl.py` takes one function from it |
| Figure versions a, b, c and their staging scripts (`stage_kit*.py`); the variant `d_ex7` | Version d is the paper's set. `d_ex7` differs only in `excl_d.py` (it also excludes reservoirs 386 and 567) |
| `ip_build_c5.py` | orchestrator of an earlier figure set; `ip_series.py` is called directly |
| Seeded reruns of the 1980-2020 rule models (`*_g_y1980s`), the ISTARF dead-pool variant (`rule_repro/deadpool`), unrouted and graph-free rule runs, observed-inflow runs | computed for sensitivity checks; not used in the paper's figures |
| `modrom_rf_heldout.py` and the MODROM-tr (transferred-parameter) variant | `score_2010.py` still reads the archived MODROM-tr run and writes its rows, but no figure or number in the paper uses them |
| `swot_vs_gauge` leakage check and its first comparison set | they refer to an earlier model's configurations |
| Checks, diagnostics and plots of intermediate steps (`validate*.py`, `diag*.py`, `check_2026*.py`, `chk_susq.py`, `inspect_inputs.py`, `river_loss_reasons.py`, `mask_diff_v35ra.py`, …) | optional; the gates that the pipeline depends on are included |
| Job wrappers (`*.sh`) | replaced by `bin/submit` and the four job scripts in stages 05-06; the commands they ran are in the README |

## Not code

No data, no trained models, no data stores and no logs are included. Small
tables that record hand decisions (review rulings, edit lists, node lists,
overrides) are included under `*/inputs/` because the results cannot be
reproduced without them; together they are about 2 MB.

## Personal and site-specific information removed

- The contact e-mail address in the client name of seven download scripts.
- Absolute paths of the authors' cluster (replaced by `config/paths.env`), Slurm accounts and partitions.
- Not removed: `vendor/half_v1` names its authors' addresses as published; `vendor/swot_ml/src/train/bayesian_search.py` contains one path of the upstream author's cluster in a string (file kept byte-identical to what ran; the paper does not use it).
- Comments and docstrings still mention the first author by name where they quote design decisions, and refer to the project's internal log. These are harmless but verbose; see `docs/OPEN_ITEMS.md`.
