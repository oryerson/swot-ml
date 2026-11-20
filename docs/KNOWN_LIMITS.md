# Known limits of reproducibility

This page lists every place where running the scripts in order from raw data
will not give byte-identical results, and what is archived to close the gap.
Nothing here affects re-scoring or the figures, which reproduce exactly from
the archived predictions and gauge tables (see the smoke test).

## 1. Graph (stage 01)

- **Earlier graph versions are inputs to three steps.**
  - `snap_gauges_v34.py` compares against the graph v3.3 build and a USGS
    drainage-area download. Its output, the 102 gauge tables, is shipped in
    `01_graph/inputs/gauges_v34/`.
  - `carry_v2_fixes.py` reads the graph v2.3 tables and the edit lists of
    versions 2 to 2.3. Its output edit list is shipped
    (`inputs/carry_edits_v34.csv`, 106 rows), but the script has no mode that
    applies a shipped list, so re-running it needs the archived v2.3 files.
  - `make_decisions_v35.py` encodes review rulings made on an interactive map.
    The rulings are shipped (`inputs/decisions_v35.csv`), as are the reviewed
    items and the evidence table they were made from.
- **The delineation job ran in two parts.** 99 basins ran minutes before a
  retry branch was added to the delineator; 3 basins were rerun with the final
  code. The retry only triggers when a hinted gauge snap fails, which the logs
  show happened in the rerun basins only, so the shipped delineator reproduces
  all 102.
- The arguments used for `build_conus_basin_list.py` and `build_merit_vrts.py`
  were not logged. Defaults reproduce the 102-basin manifest; the mosaics
  cover Pfafstetter regions 73, 74, 75, 77 and 78.

## 2. Static attributes (stage 02)

- `freeze_dam_v35.py` copies the dam attributes of unchanged lakes from the
  graph v2.3 attribute table, so that imputed values do not drift between
  graph versions. Re-running it needs the archived
  `static_conus_storage_fixgdw_v23.parquet`. Without the freeze, imputed
  capacity, depth and degree-of-regulation values change by up to 6% at about
  1,290 lakes.

- `assemble_checks_v35_r2.py` adds the upstream-area column and then compares the
  table with earlier builds: the first statics build of graph v3.5 (made with the
  first graph's reach areas), the graph v2.2 and v2.3 tables, and the test-set
  site tables. These comparison files are archived with the legacy inputs. Its
  checks print PASS or FAIL but do not stop the script.

## 3. Forcing (stage 03)

- **The stores are a chain of copies.** The graph v3.5 stores copy 20,081
  unchanged nodes bit for bit from the earlier stores and re-extract 4,416
  nodes. An exact rebuild therefore runs passes a, b, c and d in order.
  Extracting every node afresh with the pass-b method agrees to about 1e-6
  standard deviations, except at three Susquehanna nodes (USGS-01545500,
  73005469, 7310084002) whose first extraction used an older geometry.
- **Normalisation statistics** were computed once, on the first store
  (September 2026), and copied unchanged into every later store, including the
  statistics of the target variables, which at that time held version-C lake
  targets. Rebuild them by running pass a and `04_swot_targets/a_lakes/merge_res_swot_conus.py`
  before `update_zarr_stats_conus.py`, or take the archived stores. No log
  shows whether `fix_empty_stats.py` was run after the last statistics run.
- **The yearly extraction files of 1980-1999 were deleted** after the store was
  built; re-running `b_extension_1980_1999/extract.py` recreates them.
- **The 2026 ERA5-Land files** were downloaded on 14-15 October 2026. Three
  months came from 6-hourly fields because the daily-statistics service
  stalled. The downloaded files are archived.
- **Long-memory inputs.** The rank transform uses each basin's full column
  set, so values at tied ranks depend on which nodes are in the basin array.
  Build all 32 inputs for all graph nodes, as `build_longmem_v35.py` does, to
  match the archived store exactly.

## 4. SWOT targets (stage 04)

- **Downloads are snapshots.** The lake pull is dated 14 October 2026 (data to
  10 October) and the reach pull 15 October 2026. Hydrocron serves the current
  processing, so a new pull can differ. The three downloaded tables are
  archived.
- 584 of the 5,846 river reaches matched on graph v3.5 were not in the
  Hydrocron reach list (it was built from the first graph). Their series end
  on 27 June 2026.
- At 50 river nodes with several reaches, a tie between equally good
  observations on the same day (57 rows) resolves by input order.
- The version-C lake series with storage curves are unpublished data of a
  collaborating group (see `docs/DATA_SOURCES.md`).

- **A few files were renamed or copied by hand between steps.** For example,
  `filter_graph_basins.py` reads list files under names that
  `build_storage_splits.py` does not write, and `make_t9_mask.py` reads the
  river drop list from a different folder than `river_build_v2.py` writes it
  to. `docs/CODE_NOTES.md` lists the cases found. The archived stage-04
  outputs are the reference.

## 5. Training and prediction (stages 05-06)

- **Training is not bit-reproducible.** Each seed ran as three jobs on
  different GPU hosts, with checkpoints at the 27-hour limits (steps 17,458
  and 34,952 for seed 0; 17,537 and 34,967 for seed 1). The trained models
  are archived.
- **Two prediction runners.** The 2010-2020 and SWOT Period predictions used
  the low-memory runner; 1981-2009 used the full runner. They are equivalent
  by construction (both keep only the last day of each window). That
  equivalence was tested for an earlier model, not for the paper model.
- The job scripts in `05_training` and `06_predictions` are new, cleaner
  versions of the project's launch scripts and have only been dry-run.

## 6. Evaluation (stage 07)

- **Gauge downloads are snapshots**, and the test sets built from them were
  frozen before scoring. Use the archived test sets.
- **Rule-model runs for 1980-2020 were made without a fixed hash seed.** A
  seeded rerun differs at 79 of 510 dams deep in reservoir cascades; medians
  over the scored sets are identical. The archived runs are the ones scored.
  The script and arguments of these runs are documented in the project log,
  not echoed in the job logs.
- **The rule models run on the first graph version**, not on graph v3.5, and
  with routed (cascade) inflow.
- **Driver tables read some inputs from other model configurations.** The
  scripts in `07_evaluation/c_scoring/drivers/` take their list of reservoirs
  from an earlier model's score table (ids only), their network from graph
  v2.2, their gauge truth from a second copy of the 1980-2020 test set
  (`g4_fix/testsets`), and write comparison columns from an earlier model
  next to the ones the paper uses. The columns that reach the paper's Figure 3
  (harmonic irregularity, SWOT and gauge standard deviations) do not depend on
  any model. The legacy inputs are archived; removing the dependence needs a
  rerun (see `docs/OPEN_ITEMS.md`).
- `series_ip.npz`, the series file of the SWOT Period scorer, takes its list
  and order of dams from `eval/grl_figs/series_g.npz`, a file made for an
  earlier figure set. It is archived.
- `swot_vs_gauge.py` (Results 3.1) builds two comparison sets; the paper uses
  the second (`per_reservoir_plus.csv`), filtered to the scored set in
  `text_nums_o50.py`. The first set and the script's leakage check refer to an
  earlier model's files.

## 7. Figures (stage 08)

- Figure images depend on the Matplotlib version and installed fonts. With
  the pinned environment on the authors' cluster the PNG files reproduce byte
  for byte; elsewhere compare `num_c5d_fig*.json`, which hold every number
  drawn.
