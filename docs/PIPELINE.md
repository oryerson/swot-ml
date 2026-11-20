# The pipeline, step by step

This page has the full detail behind the short table in the [README](../README.md): every script of every
stage, what it reads and writes, how long it took, and the commands that were run for the paper.
Set up the repository first (README, "Setup").

## The pipeline at a glance

```
 raw data                               stage                                   main product
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
 MERIT-Hydro, MERIT-Basins,       ┌─ 01_graph ──────────────┐   sub-basins and river-network graphs (graph v3.5):
 SWOT lake database, gauge list ─►│ delineate basins,        │─► 24,381 nodes, 24,281 links, 100 basins,
                                  │ build and correct graphs │   3,453 lake nodes
                                  └──────────┬───────────────┘
 BasinATLAS, Global Dam Watch ───►┌─ 02_static_attributes ───┐   80 static attributes per node
                                  └──────────┬───────────────┘
 ERA5-Land 1980-2026 ────────────►┌─ 03_forcings ────────────┐   10 daily weather inputs per node,
                                  │ + long-memory inputs     │   2 long-memory runoff indices (3 and 12 months)
                                  └──────────┬───────────────┘
 SWOT LakeSP and RiverSP          ┌─ 04_swot_targets ────────┐   daily storage-change targets at lakes (HALF filter,
 version D ──────────────────────►│ lakes, rivers, masks     │─► storage curves) and rivers; node lists; data stores
                                  └──────────┬───────────────┘
                                  ┌─ 05_training ────────────┐   SWOT-GAT, two random seeds, 50,000 steps each
                                  └──────────┬───────────────┘
                                  ┌─ 06_predictions ─────────┐   daily predictions 1981-2020 and the SWOT Period
                                  └──────────┬───────────────┘
 ResOpsUS, agency gauges,         ┌─ 07_evaluation ──────────┐   scores against gauges (NSE, nRMSE, r, VR);
 GRADES-hydroDL inflow, GDROM, ──►│ gauge truth, rule models,│─► rule-model benchmarks (GDROM, ISTARF, MODROM)
 ISTARF and MODROM rules          │ scoring, SWOT vs gauge   │   and their ensemble; drivers of skill
                                  └──────────┬───────────────┘
                                  ┌─ 08_figures ─────────────┐   Figures 1-4 of the paper
                                  └──────────────────────────┘
```

Words used throughout:

- **Graph.** One directed network per river basin. A node is a sub-basin: a lake (reservoir), a river reach with a gauge, or a plain river sub-basin. A link joins a node to the node it drains into.
- **Graph v3.5.** The graph version the paper uses. Scripts and files carry the tags `v34` (the re-delineation) and `v35` (v3.4 plus reviewed corrections).
- **C5.** The internal name of the paper's model configuration: 365-day input window, two long-memory inputs, graph v3.5. File names such as `era5_conus_v35c5_S1_storage.json` and `pred_c5.py` refer to it. `S1` and `S2` are the two random seeds (0 and 1).
- **Paper model.** The mean of the two seeds' predictions, taken per node and day.
- **Metrics.** NSE (Nash-Sutcliffe efficiency), nRMSE (root-mean-square error divided by the gauge's range), r (Pearson correlation) and VR (variability ratio: model standard deviation divided by gauge standard deviation). All four are computed on series from which each series' own mean has been removed, because model and gauge use different reference levels.
- **overlap50.** The rule that a reservoir is scored in a period only if model and gauge overlap on at least 50% of the period's days.

## Repository layout

```
01_graph/ ... 08_figures/   pipeline scripts, one folder per stage, as they ran (only paths changed)
  */inputs/                 small frozen tables: hand decisions and review rulings that are part of the method
vendor/swot_ml/             the model source the paper runs imported (from Fluvial-UMass/swot-ml, with changes)
vendor/graph_delineator/    the basin delineator (from tedlanghorst/graph_delineator, with changes)
vendor/half_v1/             HALF v1.0 lake filter (Wang & Trudel), unmodified
src/swotgat/                two small helpers: paths (the single path configuration) and http (polite downloads)
config/paths.env.example    the one file to edit for your machine
bin/run, bin/submit         run one script; submit one script as a Slurm job
tools/deploy.py             places the scripts where they expect to run (see Setup, step 3)
tools/release_build/        how this release was assembled from the working folders (audit trail)
tests/smoke/                end-to-end check without training
environment/                pinned Python environments
provenance/files.tsv        every file: original location, md5, what changed, where it is deployed
docs/                       data sources, data layout, known limits, code notes, open items
PROVENANCE.md  EXCLUDED.md  LICENSE  CITATION.cff
```

**Why scripts are "deployed".** The project grew in one working folder with code and data side by side. Many scripts find their inputs, their sibling modules and their output folder relative to their own location. Rewriting that would risk changing results, so the scripts are unchanged apart from path names (this is checked file by file; see [PROVENANCE.md](../PROVENANCE.md)). `tools/deploy.py` copies each script into the work tree at the place it originally ran from, and `bin/run` runs it there. You always refer to scripts by their path in this repository.

## Running the stages

Run the steps of each stage in the order listed. Times are the wall-clock times of the original jobs on the UMass Unity cluster. "×102" means a Slurm array with one task per basin. Shell variables used below:

```bash
W=$SWOTGAT_WORK; C=$W/CONUS; E=$W/CONUS_storage_fork/eval
V3=$E/graph_v3; V=$E/v35_inputs; FORC=$V/forcing; LISTS=$V/lists; ST=$V/statics/work
DP="--python DELINEATOR_PYTHON"        # stages 01-02 use the delineator environment
```

### Stage 01: basins and graphs

The delineator splits each basin into sub-basins of at most 500 km², with one node per gauge and one per reservoir of at least 0.5 km² in the SWOT Prior Lake Database, and routes every sub-basin through the reservoir it drains into.

| # | Script | What it does | Reads → writes | Time |
|---|---|---|---|---|
| 1 | `build_conus_basin_list.py` | Picks the 102 basins (outlet gauge in the conterminous US, at least 5 gauges) and writes one gauge table per basin | `$W/graph_delineator/all_matchups.csv` (placed there by `tools/deploy.py` from `vendor/graph_delineator/`) → `$C/basin_lists/basin_manifest.csv`, `basin_lists/gauges/` | min |
| 2 | `build_merit_vrts.py` | Mosaics the MERIT-Hydro flow-direction and flow-accumulation tiles per Pfafstetter region (73, 74, 75, 77, 78) | MERIT-Hydro v1.0.1 → `$C/merit_rasters/` | min |
| 3 | `snap_gauges_v34.py` | Re-snaps gauges whose delineated drainage area was more than 5% off, using MERIT upstream area within 2 km. *Its output is shipped* (`inputs/gauges_v34/`), because the script compares against an earlier graph version | gauge tables, USGS drainage areas → `$V3/gauges_v34/` | 1 min |
| 4 | `delineate_basin_v34.py` | Delineates one basin (`max_area=500`, `min_reservoir_area_km2=0.5`) with `vendor/graph_delineator` | gauge tables, lake polygons, MERIT rasters → `$V3/delineation_v34/<basin>/` | ×102, up to 28 min each |
| 5 | `build_conus_subbasins.py` | Joins the 102 basin results into one table | → `$V3/subbasins_conus_v34base.parquet` | 1 min |
| 6 | `build_conus_graphs_v2.py` | Builds the graphs and refuses any basin with an interior dead end | sub-basin table → graph JSON | 1 min |
| 7 | `carry_v2_fixes.py` | Re-points 106 inflows past lakes, as corrected in graph versions 2 to 2.3. *Its edit list is shipped* (`inputs/carry_edits_v34.csv`); re-deriving it needs the archived v2.3 files | → `$V3/v34/subbasins_conus_v34.parquet`, then step 6 again | 1 min |
| 8 | `lake_evidence_v35.py`, `make_decisions_v35.py` | Encode the review rulings, one row per reviewed item (`inputs/decisions_v35.csv`, 240 rows: 38 lakes to reshape, 4 gauges to pass through, 14 reference overrides, 184 left as they are) | review items → `$V3/v35_work/decisions_v35.csv` | 2 min |
| 9 | `apply_v35.py` | Applies the rulings: 38 lakes reshaped to their MERIT drainage, 4 gauges made pass-through, 13 links re-pointed | → `$V3/v35/subbasins_conus_v35.parquet`, then step 6 → `graphs_conus_v35.json` | 1 min |
| 10 | `install_v35.py` | Copies the finished graph, attributes, lists and training store into `$C/` with md5 checks (run after stage 04) | plan `inputs/install_plan_v35.json.template` | 8 min |

```bash
bin/run $DP 01_graph/build_conus_basin_list.py
bin/run $DP 01_graph/build_merit_vrts.py --pfaf2 73 74 75 77 78
bin/submit cpu --array=0-101%12 -c 4 --mem=48G -t 12:00:00 -- $DP 01_graph/delineate_basin_v34.py \
    --array-index '$SLURM_ARRAY_TASK_ID' --out-dir $V3/delineation_v34 --gauge-dir $V3/gauges_v34
bin/run $DP 01_graph/build_conus_subbasins.py --delineation-dirs $V3/delineation_v34/* --out $V3/subbasins_conus_v34base.parquet
bin/run $DP 01_graph/build_conus_graphs_v2.py --subbasins $V3/subbasins_conus_v34base.parquet --out $V3/graphs_conus_v34base.json
bin/run $DP --cwd $V3 01_graph/carry_v2_fixes.py --base-subs subbasins_conus_v34base.parquet --base-graph graphs_conus_v34base.json --tag v34 --out-dir v34
bin/run $DP 01_graph/build_conus_graphs_v2.py --subbasins $V3/v34/subbasins_conus_v34.parquet --out $V3/v34/graphs_conus_v34.json
bin/run $DP --cwd $V3 01_graph/make_decisions_v35.py
bin/run $DP --cwd $V3 01_graph/apply_v35.py --base-subs v34/subbasins_conus_v34.parquet --base-graph v34/graphs_conus_v34.json --out-dir v35
bin/run $DP 01_graph/build_conus_graphs_v2.py --subbasins $V3/v35/subbasins_conus_v35.parquet --out $V3/v35/graphs_conus_v35.json
```

Expected md5: `subbasins_conus_v35.parquet` 19a764d9…, `graphs_conus_v35.json` c580577d… (the smoke test rebuilds the latter). Two of the 102 basins (USGS-02246500, USGS-02298202) are left out of the graph because their outlet gauge could not be delineated, which leaves 100.

### Stage 02: static attributes

| # | Script | What it does | Time |
|---|---|---|---|
| 1 | `build_conus_static.py` | Joins BasinATLAS level-9 attributes to each sub-basin (columns listed in `inputs/properties.txt`) | 2 min (with 2) |
| 2 | `build_conus_gdw_static.py` | Adds 13 dam columns from Global Dam Watch (capacity, depth, degree of regulation, main use), imputing missing values | |
| 3 | `merit_uparea_v35.py`, `merit_override_v35.py` | MERIT upstream area at each node's routed outlet; 8 documented overrides | 8 min |
| 4 | `build_storage_static.py` | Adds lake area, water area and SWOT reach area (needs `reach_area_conus_v35.csv` from stage 04b) | 1 min (4-8) |
| 5 | `step0_keys.py`, `fix_gdw_static.py` | Matches each lake to its dam by lake-database id, then by polygon overlap (corrects mismatches of the plain spatial join) | |
| 6 | `freeze_dam_v35.py` | Keeps the dam values of unchanged lakes exactly as in the previous graph version, so that imputation drift cannot change them. Needs the archived `static_conus_storage_fixgdw_v23.parquet` | |
| 7 | `apply_g4_overrides.py` | Applies 6 hand-checked dam assignments (`inputs/g4_overrides.csv`) | |
| 8 | `assemble_checks_v35_r2.py` | Adds the MERIT upstream-area column, runs consistency gates, writes the final table | |

The scripts read their file names from environment variables (and fail at import if these are unset); the two sequences below are the ones that produced the paper's table. `build_conus_static.py` looks for the attribute list at the path of the original archive; pass `--props 02_static_attributes/inputs/properties.txt` if `tools/deploy.py` could not write there. The last step also compares the result with earlier builds, which must be in the work tree (see `KNOWN_LIMITS.md`, section 2).

```bash
S=02_static_attributes
export V35S_TAG=v35 V35S_SUBS=$V3/v35/subbasins_conus_v35.parquet V35S_WORK=$ST
bin/run $DP $S/build_conus_static.py --subbasins $V35S_SUBS --out $ST/static_conus_v35.parquet --props $SWOTGAT_REPO/02_static_attributes/inputs/properties.txt
bin/run $DP $S/build_conus_gdw_static.py
bin/submit cpu -c 16 --mem=64G -t 2:00:00 -- $DP $S/merit_uparea_v35.py --workers 16
bin/run $DP $S/merit_override_v35.py
# after stage 04b has written reach_area_conus_v35.csv:
export V35S_TAG=v35r2 V35S_GDW_TAG=v35 V35S_REACH=$FORC/work/river/reach_area_conus_v35.csv
bin/run $DP $S/build_storage_static.py
bin/run $DP $S/fix_gdw_static.py
bin/run $DP $S/apply_g4_overrides.py $ST/static_conus_storage_fixgdw_v35r2.parquet $ST/static_conus_storage_fixgdw_v35r2_g4.parquet \
    --overrides $E/g4_fix/g4_overrides.csv --report $ST/apply_g4_report_v35r2.csv     # chain without the freeze, for comparison
bin/run $DP $S/freeze_dam_v35.py $ST/static_conus_storage_fixgdw_v35r2.parquet $ST/static_conus_storage_fixgdw_v35r2frz.parquet $ST/freeze_dam_report_v35r2.csv
bin/run $DP $S/apply_g4_overrides.py $ST/static_conus_storage_fixgdw_v35r2frz.parquet $ST/static_conus_storage_fixgdw_v35r2frz_g4.parquet \
    --overrides $E/g4_fix/g4_overrides.csv --report $ST/apply_g4_report_v35r2frz.csv
bin/run $DP $S/assemble_checks_v35_r2.py
```

Output: `$V/statics/static_conus_storage_fixgdw_v35_g4_r2.parquet` (md5 a146e7dc…; 24,497 rows; the model reads 80 of its columns).

### Stage 03: ERA5-Land forcing and long-memory inputs

The model reads ten daily ERA5-Land variables per node (surface solar and thermal radiation, runoff, surface runoff, evaporation, precipitation, pressure, dew point, temperature, albedo), averaged over each sub-basin polygon.

The stores were built in four passes, and the paper's stores contain values from all four. The sub-folders keep them apart.

| Folder | Script | What it does | Time |
|---|---|---|---|
| `a_production_2000_2025/` | `flipped_extract_sub_basins.py` | Polygon-weighted means for 2000-2025 on the first graph version (Dask on Slurm, 64 workers) | 10 h |
| | `merge_era5_conus.py` | Writes them into the data store `$C/zarr_store` | |
| | `update_zarr_stats_conus.py`, `fix_empty_stats.py` | Per-basin normalisation statistics. **Every later store copies these statistics unchanged**, so run this once, here | |
| `b_extension_1980_1999/` | `weights.py`, `extract.py`, `build_store.py` | Sparse weight matrix per node; one extraction job per year 1980-1999; new store `zarr_store_ext1980` | 1 min; ×20, 6 min; ×102, 8 min |
| `c_extension_2026/` | `download_2026.py`, `download_inst.py`, `prep_inst.py` | Download January-September 2026 from the Copernicus Climate Data Store (needs your own CDS key) | hours (queue) |
| | `prep_2026.py`, `store_inputs_vd.py`, `build_store_vd2026.py` | Extract 2026 and write the store `zarr_store_vd2026` (2020-2026) | 1 min; ×102, 1 min |
| `d_graph_v35_stores/` | `node_map.py` | Decides per graph-v3.5 node: copy (same id and identical polygon; 20,081 nodes) or extract again (4,416 nodes) | min |
| | `weights_v35.py`, `extract_v35.py`, `prep_2026_v35.py` | Weights and extraction for the nodes to redo | 1 min; ×6, 2 min |
| | `build_store_v35.py` | Training store `zarr_store_vd2026_halfexact_v35` (forcing + the stage-04 targets) | ×102, 1 min |
| | `make_pred_weights.py`, `extract_pred_v35.py`, `build_store_pred_v35.py` | Prediction stores `zarr_store_ext1980_v35` (1980-2025) and `zarr_store_vd2026_v35` (2020-2026) | ×40, 2 min; ×102, 4 min |
| | `checks_v35.py`, `checks_pred_v35.py`, `write_ready_v35.py` | Gates on the finished stores | min |
| `e_long_memory/` | `build_longmem_v35.py` (numerics in `build_longmem.py`) | Long-memory inputs for 2009-2026: runoff averaged over each node and everything upstream of it, summed over 3 and 12 months, then ranked against the same calendar month of 1981-2020 and mapped to a standard normal (`lm_up_ssi3`, `lm_up_ssi12`) | 1 h, 180 GB |
| | `check_longmem_v35.py`, `check_c3_ties.py` | Gates | min |
| | `build_lm_c5.py`, `check_lm_c5.py` | The same two inputs for 1980-2026, needed to predict before 2009; checked bit-identical to the first store on 2009-2026 | 5 min |

The first extraction ran with these arguments (the script's defaults differ):

```bash
bin/run 03_forcings/a_production_2000_2025/flipped_extract_sub_basins.py --basin-file $C/metadata/subbasins_conus.parquet \
    --save-dir $C/datalakes/training --n-workers 64 --batch-size 16 --start-date 2000-01-01 --end-date 2025-12-31
```

`d_graph_v35_stores/make_scripts.py` records how the `*_v35.py` extraction scripts were derived from the `b_` and `c_` scripts (each is the original plus a short list of text substitutions). It is kept for reference and is not run: the derived scripts are shipped. The store builders take the basin index from the Slurm array (`--array=0-101`), for example:

```bash
bin/submit cpu -c 24 --mem=60G -t 1:00:00 -- --cwd $FORC 03_forcings/d_graph_v35_stores/weights_v35.py
bin/submit cpu --array=2020-2025 -c 4 --mem=32G -t 1:00:00 -- --cwd $FORC 03_forcings/d_graph_v35_stores/extract_v35.py '$SLURM_ARRAY_TASK_ID'
bin/submit cpu --array=0-101%34 -c 2 --mem=24G -t 1:00:00 -- --cwd $FORC 03_forcings/d_graph_v35_stores/build_store_v35.py '$SLURM_ARRAY_TASK_ID'
bin/submit cpu -c 16 --mem=180G -t 4:00:00 -- 03_forcings/e_long_memory/build_longmem_v35.py \
    --src1 $FORC/zarr_store_ext1980_v35 --src2 $FORC/zarr_store_vd2026_v35 --workers 8
```

### Stage 04: SWOT training targets

**Lakes (`a_lakes/`).** Targets come from the SWOT LakeSP product, version D, filtered exactly as in the HALF framework (Trudel et al., in review), then converted from water level to storage change.

| # | Script | What it does | Time |
|---|---|---|---|
| 1 | `lake_lists.py` | Lists the graph's lakes and reads the version-C daily series with their storage curves | 2 min |
| 2 | `hc_pull.py` | Downloads LakeSP version D (and C) from the PO.DAAC Hydrocron service; no login needed. The paper's pull is frozen at 10 October 2026 | 1-2 h at 1 request/s |
| 3 | `half_batch.py` | Runs HALF v1.0 (`vendor/half_v1`): strict, lenient and Tukey outlier filters, cross-pass adjustment, daily median | 3 min |
| 4 | `build_vd_half.py` | Storage change from level: SWOT area-elevation curve where available, else the 3D-LAKES curve, else level change × static lake area | 1 min |
| 5 | `release_replay.py`, `compare_replay.py` | Replays the unmodified HALF release script on the stored download, to prove step 3 matches it | ×16, 3 min |
| 6 | `build_halfexact.py` | Applies HALF's coverage rule (record at least 1 year long with no gap over 120 days) → 3,224 lakes, 311,091 lake-days | 1 min |
| 7 | `make_lake_targets_hx.py` | Writes the lake target table for the stores | 1 min |

`merge_res_swot_conus.py` wrote the first, version-C lake targets. None of its values reach the model, but the normalisation statistics computed in stage 03a include them, so it is part of the lineage.

**Rivers (`b_rivers/`).** River targets are SWOT RiverSP version D water-level anomalies times the reach's median SWOT area.

| # | Script | What it does | Time |
|---|---|---|---|
| 1 | `merge_swot_river_conus.py`, `svs_reach_overrides.py` | Matches each river node to SWOT reaches (MERIT-Basins → SWORD table; gauge matches from the SWOT Validation Set override). Keeps observations with `reach_q_b < 2^18` and level uncertainty `wse_u < 0.20 m` | hours |
| 2 | `build_storage_target_conus.py` | Reach area per node: median of the SWOT total area over at least 10 passes | min |
| 3 | `reach_list.py`, `hc_pull_reach.py` | Downloads RiverSP after June 2026 from Hydrocron | 6 min |
| 4 | `river_build_v2.py` | Builds the river series to October 2026 and the list of observations dropped by quality rules (`build_river_paper_mask.py` defines the rules) | 8 min |
| 5 | `river_xwalk_v1.py`, `river_build_v35.py`, `reach_area_v35.py`, `reach_area_v35_fix.py`, `river_build_v35ra.py` | The same on graph v3.5, with reach areas for new node ids | 8 + 6 min |

Run step 5 in the order listed. `reach_area_v35_fix.py` rewrites the table that `reach_area_v35.py` wrote (same areas, corrected spread column) and refuses to overwrite it, so move the first table aside before running the fix.

**Stores, lists and masks.** `c_template_store/` builds the store that stage 03d copies lake targets from (`build_store_vd2026_halfexact.py`; `make_builder.py` records how that script was derived from the stage 03c store builder and is not run) and the first masks of duplicated river targets (`dup_targets.py`, `make_t9_mask.py`, `make_t9b_mask.py`). `d_lists_and_masks/` writes the final node lists and mask on graph v3.5:

| Script | What it does |
|---|---|
| `build_storage_splits.py`, `filter_graph_basins.py` | The test list: river gauges and gauged reservoirs |
| `map_rivers_v35.py`, `rekey_lists_v35.py`, `filter_graph_basins_v35.py` | Re-key the validation and test lists to graph v3.5 ids |
| `build_lists_v35.py`, `mask_rekey_info_v35.py` | Training list = every node with a SWOT target in January 2023 to September 2026, plus the validation and test nodes. **There is no spatial hold-out:** the test is in time (1981-2020), not in space |
| `make_t9_mask_v35ra.py`, `make_t9b_mask_v35ra.py` | Final river mask: quality drops, plus 34 + 9 tributary nodes whose SWOT series is a copy of a larger river's reach |

### Stage 05: training

```bash
python tools/deploy.py                 # writes the two configurations into $C/
05_training/submit_training.sh         # DRYRUN=1 to print the jobs only
```

`submit_training.sh` first builds and verifies the training cache (`build_cache_v35.py`, CPU, 14 min), then trains each seed in four chained 30-hour GPU jobs (`train.sh`; about 64 GPU-hours per seed on an NVIDIA L40S). The configurations are `configs/era5_conus_v35c5_S{1,2}_*.json.template`; they differ only in the seed. Model: 128 hidden units, 5 graph hops, 4 attention heads, Gaussian-mixture output, about 2.2 million parameters; AdamW, learning rate 1e-4 falling to 5e-6; 365-day sequences with a 90-day burn-in.

### Stage 06: predictions

```bash
bin/submit cpu -c 16 --mem=128G -t 8:00:00 -- 03_forcings/e_long_memory/build_lm_c5.py --workers 8
bin/run 03_forcings/e_long_memory/check_lm_c5.py
06_predictions/submit_predictions.sh   # DRYRUN=1 to print the jobs only
```

Seven windows per seed cover 1981-2020 and the SWOT Period (table in `submit_predictions.sh`; 1.5 to 4.5 GPU-hours each). Outputs: `$W/CONUS_storage_fork/predictions/<run>/step_050000_<tag>.parquet`.

### Stage 07: evaluation

**a. Gauge truth (`a_gauge_truth/`).** These scripts download and screen the gauge records and match each dam to its graph node. They were run once, in September and October 2026, and the resulting test sets were frozen before any model was scored. To reproduce the paper, use the frozen test sets from the data archive; agency services revise their records, so a new download will differ.

| Folder / script | What it does |
|---|---|
| `agency_2023_2025/` | Catalogues and downloads 2023-2025 storage from USACE, USGS, NRCS, Reclamation, Texas, California and Colorado services (`build_catalogs.py`, `harvest_v2.py`, …), merges and screens them |
| `crosswalk/` | Dam → graph node, ResOpsUS record quality, gauge → dam, with a confidence tier per link (`step0` … `step7_build_testsets.py`). Only High and Medium tiers are scored |
| `build_gauge_exclusions.py`, `gauge_stat_screen.py` | Removes gauge errors (invalid flags, spikes, negative storage) |
| `build_regime3.py` | The 1980-2020 test set from ResOpsUS |
| `agency_2026/` | Extends the agency records to 2026 (`harvest_2026.py`, `build_regime1x.py`) |
| `span_vd.py` | Each reservoir's SWOT observation span |
| `expanded_set/` | A larger gauge set, used for the SWOT-versus-gauge validation |

All downloads send a generic client name and at most one request per 1.1 s per host, and cache every response.

**b. Rule-model benchmarks (`b_rule_models/`).** Three published reservoir rule models are driven by GRADES-hydroDL simulated inflow and routed down the graph: GDROM, ISTARF and MODROM (calibrated).

| Script | What it does | Time |
|---|---|---|
| `grades_prep.py`, `grades_prep_early.py`, `grades_prep_2026.py` | Download GRADES-hydroDL river discharge and NOAA state drought index (PDSI) | 1-3 h |
| `grades_map.py`, `grades_cascade_prep.py`, `modrom_nongrand_nodes.py` | Match nodes to GRADES river segments; place the rule reservoirs on the graph | min |
| `resops_engines.py` | The three rule engines (checked against the authors' code by `validate_engines.py`) | |
| `grades_sim_v4_hindcast.py` | Runs a rule model from 1980: `<family> --init gauge --start 1980-01-01` for `gdrom`, `istarf`, `modrom_cal` | 4-8 min each |
| `grades_sim_v4.py` | Same simulator with an end date; made the SWOT Period runs (start 2000, to 12 June 2026) | 1-2 min each |

Set `PYTHONHASHSEED=0` for rule runs (`bin/run` does). The 1980-2020 runs in the paper were made without it; a rerun differs slightly at about 79 dams deep in reservoir cascades, with identical medians ([`KNOWN_LIMITS.md`](KNOWN_LIMITS.md)).

**c. Scoring (`c_scoring/`).**

```bash
source 07_evaluation/c_scoring/env.sh          # the two seeds, fixed hash seed
S=07_evaluation/c_scoring
bin/run $S/ip_series.py --out $IP_SERIES --span x26 --hind h10 --p1x x26   # series file of the SWOT Period scorer
BASE=1 bin/run $S/overlap50/score_2010.py      # 2010-2020            → overlap50/score/sites_2010_g.csv
BASE=1 bin/run $S/overlap50/score_x26span.py   # SWOT Period          → sites_2325span_g.csv
BASE=1 bin/run $S/overlap50/decade.py          # 1981-1990 and others → score_j/fullrecord_site_window.csv
bin/run $S/overlap50/fig1_nums.py $E/overlap50/score $E/overlap50/fig1_o50.json   # matched reservoir sets
mkdir -p $E/overlap50/tables && bin/run $S/overlap50/analyze.py   # paired tests → overlap50/tables/
export BASE=0 ENS=1 HIND=2010
bin/run $S/full_record/full.py                 # 1981-2020 + rule-model ensemble → score_full/
bin/run $S/full_record/score_x26span.py        # SWOT Period + ensemble
bin/run $S/full_record/check_score.py          # gate: both scorer sets agree
bin/run $S/full_record/pinned_check.py         # reservoirs where ISTARF storage is pinned full or empty
bin/run $S/full_record/picks.py && bin/run $S/full_record/picks_d.py   # example reservoirs for Figure 1
bin/run $S/drivers/pa_blocks.py && bin/run $S/drivers/pa_vars.py && bin/run $S/drivers/pa_ops8120.py   # in this order
bin/run $S/drivers/fig3c_sd.py && bin/run $S/drivers/irr_harm.py       # training coverage, harmonic irregularity
bin/run $S/drivers/irreg0020_g.py
bin/run $S/full_record/sd8120.py               # gauge standard deviation 1981-2020
bin/run $S/overlap50/text_nums_o50.py          # every number quoted in the Results text
```

All of the above take under 2 minutes each on 4 CPUs.

- The four metrics are computed in `score_fullrecord.py::metrics()` and `score_x26span.py::score()` (and inline in `score_2010.py`); NSE is `1 − Σ(S′−O′)² / ΣO′²` on mean-removed series.
- `bl.py` adds two baselines when `BASE=1`: the SWOT seasonal cycle and a gauge "normal year". Build the first once, before the `BASE=1` scorers: `mkdir -p $E/baselines_c5/score && (cd $E/overlap50/score && BASE=1 $SWOTGAT_PYTHON -c "import bl; bl.build_fit()")` writes `baselines_c5/score/swot_fit.csv`.
- The rule-model ensemble (RULE-ENS, `ENS=1`) is the daily mean of the three rule models at reservoirs that have all three.
- **Harmonic irregularity** (`irr_harm.py`): standard deviation of the gauge's departures from its annual-plus-semiannual fit, divided by the standard deviation of that fit, water years 1981-2020. **Training coverage** (`fig3c_sd.py`, `sd8120.py`): standard deviation of the SWOT targets divided by the standard deviation of the gauge record.

**d. SWOT against gauges (`d_swot_vs_gauge/`).** `swot_vs_gauge.py` compares the SWOT storage targets with gauge storage on SWOT pass days (Results 3.1).

### Stage 08: figures

See the next section.

## Reproducing each paper figure

```bash
source 07_evaluation/c_scoring/env.sh      # the figure scripts need IP_RUNS, IP_STEPS, IP_HIND; leave ERA unset
bin/run 08_figures/prepare_kit.py      # gathers the score tables; drops rule-model rows of the excluded reservoirs
bin/run 08_figures/fig1_c5d.py         # → c5d_fig1.pdf/.png and num_c5d_fig1.json (every number drawn)
bin/run 08_figures/fig2_c5d.py
bin/run 08_figures/fig3_c5d.py
bin/run 08_figures/fig4_c5d.py
bin/run 08_figures/figs_only_c5d.py $E/grl_figs/intermediate/c5_o50_20261113/kit_d    # 4-page PDF (needs pdflatex)
```

| Figure | Script | Shows | Score tables it reads |
|---|---|---|---|
| 1 | `fig1_c5d.py` | Hydrographs at eight reservoirs: SWOT-GAT, the rule models, gauge and SWOT observations; 2010-2020 with the SWOT Period (rows a-d), and 1981-2020 (rows e-h) | predictions, rule runs, gauge truth, `picks.json`, `picks_d.json` |
| 2 | `fig2_c5d.py` | (a) map of SWOT-GAT NSE, 1981-2020; (b, c) NSE distributions of SWOT-GAT and the rule-model ensemble by reservoir purpose, 1981-2020 and the SWOT Period | `score_full_d.csv`, `sites_2325span_ens_d.csv` |
| 3 | `fig3_c5d.py` | (a) map of SWOT-GAT variability ratio; (b) NSE by harmonic-irregularity third; (c) SWOT-GAT variability ratio against training coverage (standard deviation of the SWOT targets ÷ standard deviation of the gauge record, 1981-2020), coloured by NSE; (d) variability-ratio distributions, 1981-2020 and the SWOT Period | `score_full_d.csv`, `sites_2325span_g.csv`, `all_sites_harm.csv`, `sd_g8120.csv` |
| 4 | `fig4_c5d.py` | NSE and variability ratio by reservoir purpose, and each model's share of best scores, 2010-2020 | `sites_2010_g.csv` |

Outputs land in `$E/grl_figs/intermediate/c5_o50_20261113/kit_d/`. The figure set is "version d": the five reservoirs listed in `08_figures/excl_d.py` are removed from every panel that involves the rule models, because the published ISTARF parameters pin their storage at full or empty.
