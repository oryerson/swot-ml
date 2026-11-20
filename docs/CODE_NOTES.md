# Notes from the documentation pass

When the docstrings and comments were rewritten, each script was read against
its old documentation. This page lists what turned up. None of it was changed
in the code: the release reproduces the paper's results with the code as it
ran. Items are possible problems or loose ends for a later clean-up, not known
errors in the paper's numbers.

## Stage 07b rule models, 07d SWOT vs gauge
- Both simulators build each node's upstream list from a Python set of strings, so summation order depends on PYTHONHASHSEED (known; KNOWN_LIMITS 6).
- `grades_sim_v4*.py`: `(A.s0 or 0.6)` treats `--s0 0` as unset for GDROM and MODROM (ISTARF branch is correct).
- A GDROM run past 2025 without `--pdsi` silently repeats the December 2025 drought-index value.
- `validate_engines.py`: the `modrom` and `inflow` steps pair records by id; for non-GRanD ids (>= 10000) MODROM and GDROM number reservoirs differently, so a pairing can be wrong. Its `parser` step executes part of the GDROM dataset's `rule2model.py`.
- `grades_cascade_prep.py`, `grades_prep_2026.py`: if every retry of a state's drought-index download fails, the previous state's text is reused; `grades_prep_2026.py` then stops on an assert, `grades_cascade_prep.py` would write duplicate rows.
- `swot_vs_gauge.py`: does not create its output folder; hard-asserts the set sizes 272, 307 and 556; its first set imports a figure kit of an earlier model configuration and its leakage check reads that configuration's files; it writes a `summary.md` whose text (in code strings) quotes internal notes.
- `grades_sim_v4.py`: argparse help strings carry decision dates and name `unitfix_v1.csv`, which is not in the repository.
- Old documentation that contradicted the code was corrected in `grades_prep.py`, `grades_cascade_prep.py`, `modrom_nongrand_nodes.py`, `validate_engines.py`, `resops_engines.py`, `grades_sim_v4_hindcast.py`.
- `modrom_rf_heldout.py` (makes the MODROM-tr parameters) is not in the release, but `score_2010.py` reads the MODROM-tr run `grades_modromrf3c_g_y1980_pubnorm` and writes its rows (no figure uses them).

## Stage 07a crosswalk, expanded gauge set, g4 test sets
- For Owen to decide: `step6c_audit_labels.py` records that the visual audit of the crosswalk sample was done by an AI assistant (the shipped label tables carry `reviewer="Claude (visual, 2026-09-23)"`); the docstring says so and advises a spot-check.
- For Owen to decide: the expanded-set harvest fetched from services whose robots.txt disallows automated access (CWMS catalog and time series, CDEC, TWDB, RISE). The docstrings keep this fact and tell users to check each service's current terms.
- `build_testsets_g4.py` writes a note containing a personal name into the `g4_note` column of sites.csv (a code string, not changed).
- `step7_build_testsets.py`: `crosswalk_overrides.csv` is optional when applied but hashed unconditionally, so the script fails at the end if the file is missing; two thresholds in `TH4` are unused (the limits are hard-coded).
- `step1b_resops_verdict.py`: a record flagged "duplicate" is graded B with the reason "duplicate partner rejected" without checking the partner.
- `step23_crosswalk.py`: one conflict branch is unreachable. `build_plus.py`: one grade-B branch cannot fire.
- Unused variables or imports in `step0_graph.py`, `step6b_review_batch2.py`, `harvest_expand.py`.
- Old documentation that contradicted the code was corrected in `step6b_review_batch2.py` (its docstring was a copy of step6's), `step6_review.py`, `step5_gauge_overlap.py`, `step1b_resops_verdict.py`, `size_candidates.py`, `build_plus.py`, `build_testsets_g4.py`.
- Inputs whose producing script is not in the release: `crosswalk/pld_wse_graph.parquet` (`extract_pld_wse.py`), `gauges_expand/crosswalk_plus/*` (`make_crosswalk_plus.py`), `crosswalk/legacy_rederive_check.csv`, `harvested_legacy*.csv`, `testsets/regime1x_2023_2026/candidate_additions/`, the staged copy of `gauge_stat_screen.py` under `gauges_expand/stage` (`stage_gss.py`).

## Stage 08 figures
- `kit_h.py` holds drawing functions that the four figures do not use (`thru_panel`, `map_irreg`, `map_r`, `epa_key`, `third20`, ...); `irreg_0020.parquet` is read at import only. `kit_j.py` loads one table it never uses.
- Variable names `N305` (fig1) and `IDS314` (kit_h, fig4) hold 301 reservoirs in the paper's run.
- `figs_only_c5d.py` imports `cmaps_b` without using it; that import is the only reason `cmaps_b.py` is shipped.
- `ipcfg.py` defaults point at an earlier model configuration; always `source 07_evaluation/c_scoring/env.sh` first.
- `grl_common.py` has an unused function `kge_of()`. `grl2.py` and most of `grl_common.py` serve earlier figure sets; the four figures use only `_clean_name`, `ST`, `albers`, `cap_size`, `STATES_JSON`.
- Labels inside code strings that could not be changed: "SWOT–GNN", "held-out backtest", "SWOT era" (grl_common.py, grl2.py; not drawn by the four figures); "HALF-exact vD storage", "overlap50" in the caption text of `mk_captions_d.py`.
- The figure scripts read the environment variable `ERA` (must be unset or "span"); they do not use BASE, ENS, HIND or RULES.

## Stage 07a agency gauge records, exclusions, 1980-2020 test set
- `gauge_stat_screen.py` has no main guard: importing `screen` from it (as `build_regime3.py` does) re-runs the whole screening and rewrites `gauge_stat_flags.parquet` and `gauge_stat_summary.csv`.
- `build_regime3.py`: values <= 0 before 2015 are dropped early, so the later zero-dropout and negative-value rules cannot fire for 1980-2014; the comparison with the 2015-2020 set is written to `version.json` but a mismatch does not stop the script.
- `harvest_2026.py`: two fetchers have hard-coded date ranges that `--start` and `--plus` do not move.
- `build_gauge_exclusions.py`: manually pinned USGS series cached by `manual_override.py` are looked up under a different cache tag, so they get no agency flags (C.J. Strike, Horseshoe).
- `harvest_v2.py`: a cache keyed by `id()` of a DataFrame (fragile; has a fallback). `manual_override.py`: `validate_monthly` ignores its `cap` argument. `merge_truth.py`: a capacity lookup is computed and never used.
- Run order matters and is not recorded in code: `nwis_v2.py` and `rise_by_name.py` overwrite catalogs written by `build_catalogs.py`; `merge_truth.py` expects `harvested.csv` and `harvested_pass2..5.csv` from `HARVEST_TAG=_passN` runs of the harvesters.
- Old documentation that contradicted the code was corrected in `harvest.py`, `harvest_v2.py`, `merge_truth.py`, `qa_truth.py`, `gauge_stat_screen.py`, `build_regime3.py`.
- Inputs with no producing script in the release: `resops_targets_full.csv`, `agency_storage_2023_2025_union.csv`, `harvested_legacy_gated.csv`, `catalogs/twdb.csv`, `agency_cache/cdec_stations.txt`, `gauges2026/extra_candidates.csv`, `extra_manual.csv`, `extra_plus.csv`, `plus_dams.csv`.

## Stages 01-02 graph and static attributes
- `lake_evidence_v35.py`: `sub=it.sub` stores the pandas method instead of the column, so column `sub` of the shipped `lake_evidence_v35.csv` holds a method's text. `make_decisions_v35.py` does not use that column.
- Possible half-cell offset: `merit_uparea_v35.py` corrects for a mosaic that sits half a cell off the tiles' frame; `snap_gauges_v34.py` and `lake_evidence_v35.py` read the same mosaic by coordinate without the correction (about 45 m). The delineator re-snaps gauges, so the effect is probably small. Not verified.
- Gates report but do not stop: `assemble_checks_v35_r2.py` writes the final table even if a check prints FAIL; `install_v35.py` only tests that the three `CHECKS.md` files exist.
- `assemble_checks_v35_r2.py` compares against files the README sequence does not build: the first statics build, the chain without the freeze, a graph v2.3 reference table, the v2.2 and v2.3 tables, the first reach-area table and test-set site tables.
- `build_conus_static.py`, `build_conus_gdw_static.py`, `build_storage_static.py`, `fix_gdw_static.py` read `V35S_*` environment variables at import and fail with KeyError if they are unset.
- `snap_gauges_v34.py --radius-km` accepts only 0.5, 1, 2 or 5. `carry_v2_fixes.py`: the default `--out-dir` is `v32`.
- Text written by code strings still names people and internal notes: the `why` column of `decisions_v35.csv` (also in the shipped inputs), `INSTALL_v35.md` written by `install_v35.py`, `CHECKS.md` written by `assemble_checks_v35_r2.py`.
- Old documentation that contradicted the code was corrected in `build_conus_gdw_static.py`, `build_conus_graphs_v2.py`, `carry_v2_fixes.py`, `delineate_basin_v34.py`, `build_merit_vrts.py`.

## Stage 03 forcing and long-memory inputs
- `flipped_extract_sub_basins.py`: a failed batch is only printed and the script exits 0; the skip-already-processed query is commented out, so a rerun appends everything again; the "Mississippi first" reordering has no effect; defaults (`--start-date 1980-01-01`, 20 workers) differ from the arguments of the paper's run (README).
- Silent failures: `download_inst.py` moves on after 5 failed attempts and exits 0; `build_longmem.py` run directly prints FAILED basins but exits 0 (the v3.5 builders exit 1).
- `check_c3_ties.py` only works when the strict tie test has failed (it crashes on an empty table otherwise).
- `node_map.py` creates the column `area_ratio_vs_template` only when at least one kept id has differing geometry bytes, but `checks_v35.py` reads it unconditionally (harmless in the paper's run).
- `checks_v35.py` writes `checks.json` and three CSVs, not `CHECKS.md`; `write_ready_v35.py` still records `forcing/CHECKS.md` as the gates path.
- `check_longmem_v35.py` and `check_c3_ties.py` compare against the store `features/longmem_vd2026` of an earlier configuration, whose builder is not shipped.
- Text written by code strings still names people and internal notes: the READY markers written by `write_ready_v35.py`, `checks_pred.json` written by `checks_pred_v35.py`.
- Old documentation that contradicted the code was corrected in `write_ready_v35.py`, `checks_v35.py`, `weights_v35.py`, `build_longmem_v35.py`.

## Stage 07c scoring and drivers
- For Owen to check: `text_nums_o50.py` computes its driver numbers with the earlier irregularity measure (`irreg_8120`) and the earlier coverage measure (`cov00`, read from an earlier model configuration's table), not with the harmonic irregularity and standard-deviation coverage that Figure 3 shows. Numbers for those two come from `irr_harm.py` and `fig3c_sd.py` (`irr_harm_numbers.json`, `fig3c_sd_numbers.json`).
- `text_nums_o50.py` loads `per_reservoir_plus.csv` and never uses it. `picks_d.py` loads `picks_pool.csv` and never uses it.
- `analyze.py` does not create its output folder (`overlap50/tables`).
- `ipcfg.py` defaults are for an earlier model; forgetting `env.sh` silently scores other runs where they exist.
- `pa_lib.py` has leftovers no shipped script uses (`RUNS`, `PRED`, `OFFICIAL26`, `RULES_2010`, `check_frozen()`, `centred_score()`).
- `bl.py` has no main block: the SWOT seasonal-cycle fit is built by calling `bl.build_fit()`.
- Driver scripts must run in the order `pa_blocks.py`, `pa_vars.py`, `pa_ops8120.py`.
- Old documentation that contradicted the code was corrected in `score_2010.py`, `score_x26span.py`, `ipx.py`, `ip_series.py`, `analyze.py`, `pa_ops8120.py`, `spot_ens.py`.
- Statements that the baseline analysis and the driver-variable list were fixed before skill was looked at are kept from the old text; the code cannot prove them.

## Stages 04-06 SWOT targets, training helpers, prediction runners
- File-name gaps left by manual steps: `build_storage_splits.py` writes `train_nodes.csv`, `valid_nodes.csv`, `test_nodes.csv`, but `filter_graph_basins.py` reads `train_nodes_all.csv`, `valid_nodes_insample.csv`, `test_nodes_insitu.csv`; `make_t9_mask.py` reads `CONUS/basin_lists/swotqc/drop_river_paper_min_present.parquet`, which `river_build_v2.py` writes to `CONUS_storage_fork/eval/swotext/`. No shipped script renames or copies these.
- `reach_area_v35.py` and `reach_area_v35_fix.py` both write `reach_area_conus_v35.csv` and both refuse to overwrite it: move the first table aside before running the fix. The first computes `cv_area` with ddof=1 (the reason for the fix; the areas are the same).
- `make_t9_mask.py` reads the graph v2 sub-basin table and `make_t9b_mask.py` defaults to graph v2.1: two earlier graph versions for the same mask family.
- `rekey_lists_v35.py` writes a re-keyed mask named `drop_river_paper_min_present_t9b_v35.parquet` in the lists folder; the model's mask of the same name under `CONUS/basin_lists/swotqc/` is the rebuilt `..._t9b_v35ra.parquet`. Same name, different content.
- `run_predict_ext.py`: `--zarr` asserts that a group named `USGS-01034500` exists; its default model source path is not the vendored one (harmless through the two runners, which import the vendored source first and assert it).
- `build_river_paper_mask.py`: observations with no matching RiverSP record skip two rules but can still be dropped by the area, ice or Hampel rules; a missing area fraction counts as failing.
- Text and keys in code strings still name people and internal notes (for example the reasons written to `removed_gauges_v35.csv`).
- Old documentation that contradicted the code was corrected in `build_storage_splits.py`, `river_build_v35.py`, `river_build_v35ra.py`, `pred_c5.py`, `pred_v35_lowmem.py`, `build_cache_v35.py`, `make_t9_mask_v35ra.py`, `make_t9b_mask_v35ra.py`, `dup_targets.py`, `run_predict_ext.py`.

## Further notes from the PEP 257 pass
- `agency_2023_2025/obsfilter.py`: its audit block reads `harvested.csv` and `harvested_pass2.csv` by relative path, so it only works when run from `CONUS_storage_fork/`, and it fails if `harvested_pass2.csv` is missing.
- `build_gauge_exclusions.py:clean` never uses its argument `key`.
- `harvest.py`, `harvest_v2.py`: `fetch_twdb` returns the whole Texas record, not the requested window; the caller cuts it, so results are unaffected.
- `resops_engines.py:arith()` validates each release expression by evaluating it once at Inflow = Storage = 1; an expression that divides by zero there would stop the load. Not checked against the rule files.
- `resops_engines.py:epiweek()`: late-December days that belong to week 1 of the next year are counted from the current year's week 1 and capped at 52, as in mosartwmpy.
- `build_conus_static.py:attrs_for_basin` is annotated as returning a table but returns a pair (attributes, number of polygons).
- `build_conus_gdw_static.py:pld_areas` computes planar area in an equal-area projection (the old text said geodesic); a comment in `main` about "the median of the matched reservoirs" is inexact: the median is taken after area-based imputation.
- `carry_v2_fixes.py:accumulate`: `float(x or 0.0)` does not turn a missing area into 0 (harmless unless the table has missing areas).
- `snap_gauges_v34.py`: two comments name a variable `RADIUS_KM` that does not exist (it is `A.radius_km`).
- `step5_gauge_overlap.py`: `fetch_codwr` and `fetch_hydromet` hard-code 2015-2020 dates, so when `step5b_tournament.py` switches the window to 2023-2025 a Colorado or Hydromet winner can never be swapped in (no wrong numbers result).
- `score_x26span.py` (both copies): the constant `MIN = 180` is assigned and never used. `pa_vars.py`: one value is assigned twice in a row.
- `discover_expand.py:jl` appears to be unused. `step23_crosswalk.py` writes `dam_node_crosswalk.csv` twice.
