# Unit tests of the model and scoring code

These tests check that each piece of the SWOT-GAT model and of the paper's scorers does what the
paper says. Most run on small synthetic inputs on a CPU in a few minutes. One file
(`test_release_equivalence.py`) also uses the paper's trained checkpoints and training cache; it
runs only where those exist (the authors' cluster) and is skipped elsewhere. No test trains or
predicts over a full window, uses a GPU, or writes outside pytest's temporary folder.

The model is `SparseFusionGRNN` (`vendor/swot_ml/src/models/sparse_fusion_gann.py`, configuration
name `sf_grnn`), built with the paper model C5's settings from
`05_training/configs/era5_conus_v35c5_S1_storage.json.template`: 128 hidden units, 5 graph hops,
4 attention heads, Gaussian-mixture (GMM) head, sequence-to-sequence output, `dv_SWOTAE` scaled by
`lake_area_km2`.

## How to run

```bash
source config/paths.env            # only needed for the cluster tests; sets SWOTGAT_PYTHON
JAX_PLATFORMS=cpu "$SWOTGAT_PYTHON" -m pytest tests/unit -p no:cacheprovider
```

pytest is not part of the model environment (`environment/swot_ml`). Install it into that
environment with `uv pip install --python "$SWOTGAT_PYTHON" -r environment/tests/requirements.txt`,
or into a separate folder put on `PYTHONPATH` (that is how the recorded run below was made:
pytest 8.3.4, pluggy 1.5.0, iniconfig 2.0.0 in a folder of their own).
On a Slurm cluster:

```bash
sbatch $SWOTGAT_SBATCH_CPU -c 8 --mem=48G -t 1:00:00 --wrap \
  "JAX_PLATFORMS=cpu $SWOTGAT_PYTHON -m pytest tests/unit -p no:cacheprovider -rA"
```

The whole suite took 5 min 52 s on 8 CPU cores (Slurm job 1883333); the fault injection took
about 12 minutes more.

`PYTEST_OUTPUT.txt` in this folder is the full output of the recorded run: 103 passed, 4 expected
failures (findings 1, 5 and 9 below). `FAULT_INJECTION.md` records the fault injection: all 32
planted bugs caught.

## What each test checks

### `test_graph_attention.py` — graph attention (paper: bidirectional message passing on the river graph)

Edges run from upstream (`src`) to downstream (`dest`).

| Test | Checks |
|---|---|
| `test_segment_softmax_sums_to_one_per_receiver` | attention weights sum to 1 over each node's incoming edges and equal a per-node softmax, also for large logits |
| `test_downstream_attends_only_over_upstream_neighbours` | exact Jacobian: a node's downstream message depends only on itself (query) and the nodes draining into it; headwaters get exactly 0 |
| `test_downstream_masked_edge_contributes_nothing` | a masked edge gets weight 0 and its sender has no influence |
| `test_downstream_masked_node_output_is_zero` | a masked node's message is 0 |
| `test_downstream_all_incoming_edges_masked_gives_no_message` | **expected failure (finding 1 below)** |
| `test_upstream_receives_only_from_single_downstream_neighbour` | each node's upstream message depends only on itself and its one downstream neighbour, and equals message_net × sigmoid(gate_net) computed by hand; the outlet gets 0 |
| `test_upstream_masked_edge_contributes_nothing` | masked edge → no upstream message |
| `test_isolated_node_receives_no_information` | a node without edges depends on no other node; its per-layer update is the constant `combine_proj` bias (finding 2) |
| `test_padding_nodes_never_reach_real_nodes` | with batches packed by the real `padding_collate_fn`, no padding node influences a real node and packed basins stay separate |
| `test_one_step_reaches_exactly_k_hops` | one daily step (LSTM + 5 GAT layers): perturbing one node's input changes h at exactly the nodes within 5 hops (undirected distance) on a chain and two random trees; c changes only at that node |
| `test_full_model_reach_grows_5_hops_per_day` | whole C5 model on a 31-node chain: a day-0 perturbation changes day-t predictions at exactly the nodes within 5(t+1) hops |

### `test_recurrent_and_fusion.py` — recurrent cell and observation fusion

| Test | Checks |
|---|---|
| `test_lstm_then_gat_and_gat_leaves_cell_state_alone` | the cell's c equals the LSTM's c bit for bit; h equals GAT(LSTM h); changing GAT weights changes h, never c |
| `test_revised_h_carries_to_next_day` | the model equals a hand-written loop over days in which the GAT-revised h is carried to the next day (and differs from a loop that carries the pre-GAT h) |
| `test_remat_levels_do_not_change_the_forward_pass` | memory settings `REMAT_LEVEL` 0-3 (C5 trained at 3) give the same outputs |
| `test_c5_configuration_has_no_observation_source` | with C5's `time_gaps = {"era5": False}`, `set_model_data_args` gives no sparse source and one dense source of 12 inputs |
| `test_c5_model_has_no_sparse_sources_and_never_fuses` | the built model has no sparse source; the fusion trace is empty; the fusion module returns h unchanged when given no source |
| `test_observations_and_targets_cannot_reach_predictions` | adding targets, fallback targets and an extra observation-like input to the batch leaves every output bit-for-bit unchanged; the gradient of the outputs with respect to the targets is 0 |
| `test_c5_parameter_count` | C5 (300 static columns, 12 inputs) has 2,239,256 parameters, as its training logs print (finding 6) |

### `test_area_scaling_and_loss.py` — area scaling and the loss

| Test | Checks |
|---|---|
| `test_area_scaling_formula` | mu' = (A·mu − y_offset)/y_scale, sigma' = A·sigma/y_scale, log_pi unchanged (C5's constants); `dv_river` untouched |
| `test_area_scaling_floor_for_nodes_without_lake_area` | nodes with NaN or zero area get A = 1e-3, so sigma stays positive |
| `test_volume_nll_equals_level_nll_plus_log_area_over_scale` | NLL(volume) = NLL(level) + log(A/y_scale) for 40 random mixtures of 100 components |
| `test_area_scaling_uses_static_before_dropout` | in training mode (static dropout 0.2) the scaling uses the true area, not the dropped-out static vector |
| `test_area_scaled_args_from_dataset_scalers` | the scaling constants come from the dataset's own scalers (log1p-normalised `lake_area_km2`, z-scored `dv_SWOTAE`) |
| `test_gmm_nll_matches_hand_computation` | `gmm_nll` equals a float64 hand computation of the mixture NLL (residuals capped at 20 sd, as coded) |
| `test_loss_counts_only_observed_days_real_nodes_after_burn_in` | the loss equals the hand mean over observed days ≥ 90, real nodes only; changing burn-in days or padding-node values changes nothing |
| `test_burn_in_is_90_days_in_c5` | C5's burn-in is 90; observations on days 0-88 give loss 0; day 90 is the first scored day |
| `test_fallback_pooled_with_primary_per_observation` | `dv_SWOTAE__fb` is pooled with the primary target as (n_p L_p + w n_f L_f)/(n_p + w n_f); w = 1 in C5; w = 0 ignores it |
| `test_target_weights` | weighted mean over targets, Σ w_t L_t / Σ w_t; C5 uses 1 and 1 |
| `test_target_without_observations_still_counts_in_the_weight_total` | documents finding 3 |

### `test_data.py` — data pipeline and leakage

| Test | Checks |
|---|---|
| `test_model_code_has_no_reference_to_gauge_data` | no source file of `data/`, `train/`, `config/`, `models/`, `evaluate/` mentions gauge data (test sets, `truth_clean`, ResOpsUS, NWIS, agency) |
| `test_c5_configuration_names_no_gauge_file` | no path in the C5 configuration points to gauge data |
| `test_rereference_uses_only_the_window_and_kept_observations` | re-referencing to the median uses only 2023-01-01..2026-09-30 and only observations the SWOT QC drop list keeps; the fallback source is re-referenced by its own median |
| `test_input_history_keeps_inputs_and_blanks_targets` | the training slice starts 365 days early; forcing is kept there, targets and fallback are NaN |
| `test_valid_sample_dates` | sample end dates lie in the training range, every 365-day window lies in the cached slice, and a date is a sample exactly when a target or fallback is observed on it |
| `test_sample_window_masks_and_fallback` | the real `__getitem__` on a synthetic cache: the 365 days ending on the sample date, nodes in graph order, `dv_river` masked to the base node list and `dv_SWOTAE` to its own list, fallback only where `dv_SWOTAE` is missing and converted into its z-units |
| `test_c5_sequence_settings` | sequence length 365, input history 365, re-reference window and variables as in the paper |

### `test_scoring.py` — the paper's scorers (`07_evaluation/c_scoring`)

| Test | Checks |
|---|---|
| `test_score_closed_forms` | SWOT Period scorer (both copies): offset → NSE 1, VR 1, r 1; doubled swing → NSE 0, VR 2; mirror → NSE −3, r −1; constant → flagged flat, r = VR = 0 |
| `test_score_matches_definitions_and_identity` | NSE, VR, nRMSE, r equal an independent implementation; NSE = 2·VR·r − VR² |
| `test_score_overlap50_and_centring_over_shared_days` | 50% shared days are scored, one day fewer is not; gauge days without a model value do not enter the means |
| `test_fullrecord_metrics` | full-record `metrics()` (three copies) equals the definitions and the identity |
| `test_fullrecord_scorer_end_to_end` | the unchanged full-record scorer, run on a synthetic work tree: seeds averaged per node and day (both needed), members summed (all needed), reservoirs below 50% shared days (model or gauge) dropped, exactly 50% kept, metrics as defined |
| `test_2010_scorer_end_to_end` | the same for the 2010-2020 scorer, plus a rule model equal to gauge + constant scoring NSE 1 |
| `test_harmonic_irregularity` | `irr_harm`: 0 for a pure annual + semiannual cycle, 0.5 when the residual sd is half the cycle sd, no value with < 5 water years, water years with < 292 days dropped |
| `test_training_coverage_definition` | sd of the gauge over 1981-2020 is the sample sd on the window's days (NaN under 10 days); Figure 3c plots `sd_swot / sd_g8120` |
| `test_normal_year_is_the_same_function_in_every_copy` | the three copies of `normal_year()` (the one `bl.py` executes and the two scorers') are the same text |
| `test_fullrecord_normal_year_uses_only_days_outside_the_window` | the full-record normal-year baseline is unchanged when the scored window's gauge values change, recovers a known seasonal cycle, and needs 3 reference years |
| `test_2010_normal_year_is_leave_one_year_out` | the 2010-2020 normal year of year y is unchanged when only year y changes (and changes for the other years) |

### `test_rule_engines.py` — rule-model engines (`07_evaluation/b_rule_models/resops_engines.py`)

| Test | Checks |
|---|---|
| `test_water_balance_every_day_every_engine` | GDROM, ISTARF and MODROM through a drought and a flood: S_t = S_{t-1} + I_t − outflow_t every day, 0 ≤ S ≤ capacity, executed release = min(decision, S_{t-1} + I_t), spill only when full and counted as outflow |
| `test_release_decided_from_start_of_day_storage` | each engine's release decisions equal those of its release-only mode (realS) fed with the simulated start-of-day storage; realS returns no storage and outflow = decision |
| `test_storage_never_negative_in_drought` | a release far above inflow empties the reservoir to exactly 0, never below |
| `test_arith_equals_python_arithmetic_on_random_expressions` | the GDROM rule parser gives the same value as Python arithmetic on 3,000 random expressions (numbers, Inflow, Storage, + − × ÷, unary minus, max, min) |
| `test_arith_refuses_anything_else` | function calls, attributes, powers, comparisons, conditionals, other variable names and imports are refused |
| `test_gdrom_module_and_release_selection_and_fallbacks` | module chosen by DOY/PDSI conditions, release line chosen by Inflow/Storage conditions; the three documented fallbacks (last module line, first module, last release line) and their counters |
| `test_gdrom_rule_set_with_nan_is_refused` | a rule set with NaN thresholds is refused |
| `test_istarf_branches_by_hand` | above, inside and below the normal operating range, and the clamp at Release_min/Release_max, by hand |
| `test_istarf_nor_harmonics_and_bounds` | the NOR bounds are the clipped weekly harmonics in % of capacity |
| `test_epiweek_known_dates`, `test_epiweek_matches_mmwr_except_year_end_week1` | the MMWR week (capped at 52) against known dates and an independent implementation, 1979-2026 |
| `test_epiweek_year_end_days_belong_to_next_years_week1` | **expected failure (finding 9)** |
| `test_modrom_modes_by_hand_and_continuity` | modes 2 and 3 by hand; mode 3 is continuous at its breakpoint |
| `test_modrom_normalisation_and_calendar_month` | R = R_n(I/inflow_max, S/storage_max) × outflow_max; January uses the first monthly mode |

### `test_rule_simulator.py` — rule-model simulators, run unchanged on a synthetic river

A four-node river with an upstream reservoir and a test-set reservoir, two years of discharge with
a negative value and a 3-day gap, a node/reach area ratio of 2, gauge records in acre-feet
(GDROM's) and MCM (ResOpsUS, with a correction factor). `grades_sim_v4.py` and
`grades_sim_v4_hindcast.py` are run for GDROM, ISTARF and MODROM (`--init gauge`), 9 runs in all.

| Test | Checks |
|---|---|
| `test_simulator_matches_engines_cascade_units_and_init` | the written storage (MCM) equals the engine driven by the documented inputs: natural flow (clipped, gaps ≤ 5 days interpolated, × area ratio), cascade inflow = max(natural_B − natural_A, 0) + outflow of A, acre-feet ↔ m³ with 1233.48184, start storage from the gauge day nearest the start (ResOpsUS × correction factor for ISTARF) |
| `test_gauge_enters_only_as_start_storage` | tripling every gauge value more than 15 days after the start leaves the simulation unchanged |
| `test_v4_equals_hindcast_simulator` | `grades_sim_v4.py` without its three new options gives the same series as the script that made the 1980-2020 runs |
| `test_simulated_storage_within_capacity` | 0 ≤ storage ≤ capacity in every run |

### Fault injection (`fault_injection.py`)

Not a pytest file: it copies the repository to a temporary folder, plants one realistic bug at a
time (32 in all: graph direction, one hop short, the next day getting the pre-GAT state, area
scaling after dropout, no burn-in, padding nodes in the loss, fallback overriding the target, median
over the whole record, sample window off by one day, NSE centring, overlap rule at 40%, the scored
year leaking into its baseline, annual-only irregularity, wrong coverage denominator, release from
end-of-day storage, spill not counted, ISTARF percent vs fraction, MODROM month shifted, rounded
acre-foot constant, cascade double counting, wrong start storage, area ratio or correction factor
ignored, ...) and runs the tests that cover that code. Every planted bug must make a test fail.
`FAULT_INJECTION.md` is the recorded result.

```bash
JAX_PLATFORMS=cpu "$SWOTGAT_PYTHON" tests/unit/fault_injection.py --out tests/unit/FAULT_INJECTION.md
```

### `test_release_equivalence.py` — release equivalence and configurations (cluster only, except the first two)

| Test | Checks |
|---|---|
| `test_templates_render_to_paper_configs` | both templates, rendered with the original root, have md5 e91fd492… and 55881f63… |
| `test_deploy_render_matches_static_check` | `tools/deploy.py`'s `render()` gives the same bytes |
| `test_launched_and_checkpoint_configs_agree` | the launched configs equal the rendered templates; each checkpoint's `config.json` agrees on every setting and records 12 dense inputs and no sparse source |
| `test_staged_and_release_source_files_identical` | every `.py` of `vendor/swot_ml/src` equals the staged source the paper's runs imported |
| `test_same_batch_from_both_sources` | a two-basin, 365-day batch from the paper's training cache is identical when built by either copy |
| `test_identical_outputs_and_loss` | trained checkpoint (step 50,000, seeds S1 and S2): GMM outputs, training loss and the predictions in MCM are bit-for-bit identical between the copies, and when the staged copy runs on the release copy's batch |
| `test_full_and_lowmem_runner_give_the_same_prediction` | the full runner's model call (REMAT 0, seq2seq, last day) and the low-memory runner's (REMAT 1, heads on the final state) give the same last-day predictions (docs/OPEN_ITEMS.md item 9, one batch, CPU) |
| `test_no_gauge_file_opened` | an open-file audit during dataset construction, sampling and the forward pass sees no gauge file among the data files opened, and no gauge package is imported |
| `test_checkpoint_parameter_count` | each trained checkpoint has 300 static columns and 2,239,256 parameters |

## Findings (code against the paper's description)

Each finding is marked in the suite: an expected failure (`xfail`, strict, so the suite says
when the code changes) or a test that documents the behaviour. None of them changes a number of
the paper; the reason is given for each.

1. **Downstream attention when every incoming edge is masked** (`xfail`
   `test_downstream_all_incoming_edges_masked_gives_no_message`). Masked edges get score −1e9. If
   all edges into a real node are masked, `segment_softmax` gives them equal weights and the
   masked senders' values are summed into that node. The data loader never creates this case:
   padding edges are self-loops on the first padding node, which is itself masked
   (`test_padding_nodes_never_reach_real_nodes`). A safe fix would multiply the weights by
   `edge_mask`.
2. **"No message" for a node without neighbours is a constant, not zero.** Such a node gets no
   information from other nodes (exact Jacobian), but each of the 5 layers adds the
   `combine_proj` bias, the same for every node (`test_isolated_node_receives_no_information`).
   Also, a node with a single tributary attends to it with weight 1 whatever its own state:
   attention only weighs tributaries against each other.
3. **Loss normalisation with a target absent from a batch.** The batch loss is
   Σ_t w_t L_t / Σ_t w_t over all configured targets. A target with no observation in the
   batch adds 0 but keeps its weight in the denominator, so that batch's loss is the other
   target's NLL halved (C5: weights 1 and 1)
   (`test_target_without_observations_still_counts_in_the_weight_total`). Only the gradient
   scale of such batches changes. Worth a sentence in the Methods if the loss is written as a
   weighted mean.
4. **The GMM NLL caps each standardised residual at ±20 sd** (`gmm_nll`). It is an
   implementation detail that the Methods may not mention.
5. **Seed averaging in the scorers** (`xfail` `test_fullrecord_scorer_drops_day_missing_in_one_seed`,
   `test_2010_scorer_drops_day_missing_in_one_seed`). `score_fullrecord.py` and
   `score_2010.py` average the seeds with `mean(skipna=False)`. Then they test "every member
   present" with `groupby(date).subbasin.nunique()`, which also counts a member whose seed mean
   is NaN, and sum members with a NaN-skipping sum. On a day when one seed lacks one member node,
   the reservoir's value is the partial sum of the other members; it should be dropped. In the
   paper's runs this never happens: for all 20 prediction files (`h10`, `x26`, `x8085` …
   `x0409` and their `ab`/`p` variants), the two seeds have identical row counts and no null
   prediction (checked 2026-11-16).
6. **Parameter count.** C5 has **2,239,256** parameters (`test_c5_parameter_count`,
   `test_checkpoint_parameter_count`). That is what the training logs print, and it matches
   "about 2.2 million" in the README. The 300 static columns after one-hot encoding contribute
   2,560 parameters each. A model built with the 85 columns before one-hot encoding has
   1,688,856, which is presumably where a different figure in the project notes came from.
7. **Scope of the runner check.** `docs/OPEN_ITEMS.md` item 9 asks for the two prediction runners
   to predict one whole window and be compared. This suite compares their model calls on one
   real batch (identical: maximum difference 0.0 MCM for both seeds and both targets). It does
   not run the two scripts' loaders over a window.
8. **Small notes, not disagreements.** The first day of the 365-day input-history slice is never
   inside a window: the first window starts one day later. The SWOT standard deviation behind
   training coverage (`drivers/fig3c_sd.py`) uses 2023-03-01 to 2026-09-30, while training and
   re-referencing use 2023-01-01 to 2026-09-30. A `global_gauges` package is installed in the
   model environment. Nothing imports it; `importlib.metadata` only reads its `entry_points.txt`
   while scanning installed packages.

9. **ISTARF week at the turn of the year** (`xfail` `test_epiweek_year_end_days_belong_to_next_years_week1`).
   `epiweek()` says week 1 is the week that contains 4 January. But the last 1-3 days of some
   Decembers (for example 2024-12-29..31), which belong to week 1 of the next MMWR year, get
   week 52. On those days the ISTARF harmonics use week 52 instead of week 1. Whether mosartwmpy,
   which the docstring follows, does the same was not checked (it is not installed here).
   The effect is small: a few days per affected year, between neighbouring weeks of a smooth harmonic.

## Limits of these tests

- The open-file audit sees files opened by Python and by the wrapped readers (pandas, pyarrow,
  xarray, zarr, numpy). A C library that opens a file by itself would not be recorded.
- The equivalence test uses one batch of two basins (55 nodes, 365 days, training cache) per seed.
- The rule-model simulator test covers one cascade of two reservoirs with gauge-based start
  storage. It does not cover `--inflow obs`, `--unitfix`, `modrom_rf` (random-forest parameters,
  gauge-free normalisation by bisection), the PDSI state matching, or reservoirs that fail.
- `validate_engines.py` (the engines against the authors' own code and published scores) needs the
  rule datasets and is not wrapped as a test yet.
- Tests that cut a function out of a scoring script (`load_function`, `run_statements`) depend on
  the function's name and on the first characters of the selected statements.
