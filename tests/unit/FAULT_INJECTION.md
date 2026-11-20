# Fault injection result

32 of 32 planted bugs caught by the unit tests (baseline on the unmutated copy passed).

| Mutant | File | Planted bug | Result | Failing tests |
|---|---|---|---|---|
| `graph-direction` | `vendor/swot_ml/src/models/sparse_fusion_gann.py` | downstream aggregation reads the edges backwards | caught | test_downstream_attends_only_over_upstream_neighbours, test_downstream_masked_edge_contributes_nothing, test_one_step_reaches_exactly_k_hops[edges0-25-12] (+3) |
| `one-hop-short` | `vendor/swot_ml/src/models/sparse_fusion_gann.py` | the graph step has one layer fewer than k_hops | caught | test_one_step_reaches_exactly_k_hops[edges0-25-12], test_one_step_reaches_exactly_k_hops[edges1-40-17], test_one_step_reaches_exactly_k_hops[edges2-40-3] (+1) |
| `carry-pre-gat-h` | `vendor/swot_ml/src/models/sparse_fusion_gann.py` | the next day receives the LSTM's h, not the GAT-revised h | caught | test_lstm_then_gat_and_gat_leaves_cell_state_alone, test_revised_h_carries_to_next_day |
| `gat-writes-c` | `vendor/swot_ml/src/models/sparse_fusion_gann.py` | the graph step also changes the cell state | caught | test_lstm_then_gat_and_gat_leaves_cell_state_alone, test_revised_h_carries_to_next_day |
| `area-after-dropout` | `vendor/swot_ml/src/models/sparse_fusion_gann.py` | area scaling reads the static vector after static dropout | caught | test_area_scaling_uses_static_before_dropout |
| `area-sigma` | `vendor/swot_ml/src/models/sparse_fusion_gann.py` | the mixture spread is not scaled by the lake area | caught | test_area_scaling_formula, test_area_scaling_floor_for_nodes_without_lake_area, test_volume_nll_equals_level_nll_plus_log_area_over_scale (+1) |
| `no-burn-in` | `vendor/swot_ml/src/train/loss.py` | the 90-day burn-in is not excluded from the loss | caught | test_loss_counts_only_observed_days_real_nodes_after_burn_in, test_burn_in_is_90_days_in_c5, test_fallback_pooled_with_primary_per_observation (+2) |
| `loss-padding` | `vendor/swot_ml/src/train/loss.py` | padding nodes count in the loss | caught | test_loss_counts_only_observed_days_real_nodes_after_burn_in, test_fallback_pooled_with_primary_per_observation, test_target_weights (+1) |
| `loss-weights` | `vendor/swot_ml/src/train/loss.py` | target weights not normalised | caught | test_loss_counts_only_observed_days_real_nodes_after_burn_in, test_fallback_pooled_with_primary_per_observation, test_target_weights (+1) |
| `fallback-overrides` | `vendor/swot_ml/src/data/cached_basingraphdataset.py` | the fallback is emitted where the primary target exists too | caught | test_sample_window_masks_and_fallback |
| `median-whole-record` | `vendor/swot_ml/src/data/cached_basingraphdataset.py` | re-referencing median over the whole record instead of the window | caught | test_rereference_uses_only_the_window_and_kept_observations |
| `window-off-by-one` | `vendor/swot_ml/src/data/cached_basingraphdataset.py` | each sample ends one day before its date | caught | test_sample_window_masks_and_fallback |
| `history-targets-kept` | `vendor/swot_ml/src/data/cached_basingraphdataset.py` | targets inside the 365-day input history are not blanked | caught | test_input_history_keeps_inputs_and_blanks_targets |
| `nse-centring` | `07_evaluation/c_scoring/overlap50/score_x26span.py` | the model series is centred on the gauge mean | caught | test_score_closed_forms[overlap50], test_score_matches_definitions_and_identity[overlap50] |
| `nrmse-range` | `07_evaluation/c_scoring/overlap50/score_x26span.py` | nRMSE divided by the model's range | caught | test_score_closed_forms[overlap50], test_score_matches_definitions_and_identity[overlap50] |
| `overlap-40-fullrecord` | `07_evaluation/c_scoring/full_record/score_fullrecord.py` | full-record overlap rule at 40% of the window's days | caught | test_fullrecord_scorer_end_to_end |
| `overlap-40-2010` | `07_evaluation/c_scoring/overlap50/score_2010.py` | 2010-2020 overlap rule at 40% | caught | test_2010_scorer_end_to_end |
| `normal-year-leak` | `07_evaluation/c_scoring/full_record/score_fullrecord.py` | the normal-year baseline sees the scored window | caught | test_normal_year_is_the_same_function_in_every_copy, test_fullrecord_normal_year_uses_only_days_outside_the_window |
| `loyo-leak` | `07_evaluation/c_scoring/overlap50/score_2010.py` | the leave-one-year-out normal year sees its own year | caught | test_2010_normal_year_is_leave_one_year_out |
| `irr-annual-only` | `07_evaluation/c_scoring/drivers/irr_harm.py` | harmonic irregularity without the semiannual harmonic | caught | test_harmonic_irregularity |
| `coverage-denominator` | `08_figures/fig3_c5d.py` | training coverage divided by the 2010-2020 gauge sd | caught | test_training_coverage_definition |
| `release-end-of-day` | `07_evaluation/b_rule_models/resops_engines.py` | GDROM decides the release from end-of-day storage | caught | test_release_decided_from_start_of_day_storage |
| `spill-not-outflow` | `07_evaluation/b_rule_models/resops_engines.py` | spilled water is not counted as outflow (first engine, GDROM) | caught | test_water_balance_every_day_every_engine |
| `istarf-percent` | `07_evaluation/b_rule_models/resops_engines.py` | ISTARF compares a storage fraction with bounds in percent | caught | test_istarf_branches_by_hand, test_istarf_nor_harmonics_and_bounds |
| `modrom-month` | `07_evaluation/b_rule_models/resops_engines.py` | MODROM uses the next month's mode | caught | test_modrom_normalisation_and_calendar_month |
| `gdrom-fallback` | `07_evaluation/b_rule_models/resops_engines.py` | GDROM falls back to the first release line instead of the last | caught | test_gdrom_module_and_release_selection_and_fallbacks |
| `parser-power` | `07_evaluation/b_rule_models/resops_engines.py` | the rule parser accepts more than the documented grammar | caught | test_arith_refuses_anything_else[Inflow ** 2] |
| `acre-foot-rounded` | `07_evaluation/b_rule_models/grades_sim_v4.py` | acre-feet converted with a rounded constant | caught | test_simulator_matches_engines_cascade_units_and_init[gdrom], test_simulator_matches_engines_cascade_units_and_init[modrom_cal], test_v4_equals_hindcast_simulator[gdrom] (+1) |
| `cascade-no-subtract` | `07_evaluation/b_rule_models/grades_sim_v4.py` | upstream natural flow counted twice in the cascade | caught | test_simulator_matches_engines_cascade_units_and_init[gdrom], test_simulator_matches_engines_cascade_units_and_init[modrom_cal], test_v4_equals_hindcast_simulator[gdrom] (+1) |
| `init-last-gauge-day` | `07_evaluation/b_rule_models/grades_sim_v4.py` | start storage taken from the gauge 15 days after the start | caught | test_simulator_matches_engines_cascade_units_and_init[gdrom], test_simulator_matches_engines_cascade_units_and_init[istarf], test_simulator_matches_engines_cascade_units_and_init[modrom_cal] (+3) |
| `area-ratio-ignored` | `07_evaluation/b_rule_models/grades_sim_v4.py` | natural flow not scaled by the node/reach area ratio | caught | test_simulator_matches_engines_cascade_units_and_init[gdrom], test_simulator_matches_engines_cascade_units_and_init[istarf], test_simulator_matches_engines_cascade_units_and_init[modrom_cal] (+3) |
| `resops-fix-ignored` | `07_evaluation/b_rule_models/grades_sim_v4.py` | ResOpsUS correction factor not applied | caught | test_simulator_matches_engines_cascade_units_and_init[istarf], test_v4_equals_hindcast_simulator[istarf] |
