# Open items before publication

Decisions and checks that remain. Delete this file when they are settled.

## Already decided (16 November 2026)

- Repository: `Fluvial-UMass/swot-gat`.
- Ted Langhorst's code is vendored as exact copies with attribution (`vendor/`).
- Licence: to match upstream.
- Figure set: version d.
- Paper model: the mean of the two seeds, per node and day (settled from the scoring code; the two configurations differ only in the seed).

## A. Blocking

1. **Licence.** Neither `Fluvial-UMass/swot-ml` nor `tedlanghorst/graph_delineator` has a licence file, so there is nothing to match yet. Vendoring their code needs Ted Langhorst's permission and an agreed licence (MIT or BSD-3-Clause are the usual choices for research code). Then replace `LICENSE` and the `license` field of `CITATION.cff`.
2. **Organisation access.** Creating `Fluvial-UMass/swot-gat` needs rights in the organisation.
3. **Data and model archive.** Proposal, to be confirmed (sizes are of the files on the authors' cluster):

   | Part | Contents | Size | Purpose |
   |---|---|---|---|
   | 1. Results | SWOT-GAT predictions (2 seeds × 7 windows), rule-model runs, frozen gauge test sets, the lake record after HALF, score tables, driver tables, figure tables (the files in `tests/smoke/inputs_eval.txt` and the smoke-test outputs) | about 3 GB | reproduce every score and figure; needed by the smoke test |
   | 2. Trained models | `checkpoints/step_050000/{model.eqx, config.json}` of both seeds (plus `opt.eqx`, `trainer_state.json` to resume) | 54 MB | reproduce the predictions |
   | 3. Model inputs | graph and sub-basin tables, static attributes, node lists, mask, training store, two prediction stores, two long-memory stores | about 36 GB (the 1980-2025 prediction store is 22 GB, the long-memory stores 8.6 GB) | retrain, or predict elsewhere |
   | 4. Download snapshots | SWOT lake and reach downloads (Hydrocron, October 2026), ERA5-Land 2026 raw files, agency gauge caches | about 6.5 GB (agency caches 4.5 GB, ERA5-Land 2026 raw files 1.5 GB) | rebuild stages 03-04 and 07a exactly |
   | 5. Legacy inputs | graph v2.3 sub-basin table and graph, the v2 to v2.3 edit lists, `static_conus_storage_fixgdw_v23.parquet`, `series_g.npz`, and the earlier-model files the driver scripts read | 0.3 GB without the four template stores; 42 GB with them (needed only to rebuild the forcing stores bit for bit) | rebuild stages 01-02 and the driver tables exactly |

   Zenodo accepts 50 GB per record by default (more on request), so parts 1, 2 and 5 fit in one record and parts 3 and 4 may need their own records or a request. Before archiving, check redistribution terms for: the version-C lake series and storage curves (unpublished, collaborating group); GRADES-hydroDL discharge inside the rule-model runs; ResOpsUS (CC BY 4.0) inside the test sets; ERA5-Land (Copernicus licence, attribution required) inside the stores.
4. **Push the delineator.** The fork `oryerson/graph_delineator` (branch `Reservoirs`) does not contain the edits the paper's delineation used; a clone of it runs the gauge-only delineator. The exact package is in `vendor/graph_delineator`. Either commit and push those files to the fork, or state in the paper that this repository holds the delineator.

## B. Decisions that change numbers or wording

5. **ISTARF.** Figure set d uses the original ISTARF runs and removes five reservoirs where ISTARF storage is pinned full or empty. The dead-pool variant of ISTARF (computed, slightly better ISTARF scores) is not used. Confirm that the dead-pool variant stays out. (Choosing version d already settles that the two further pinned reservoirs found later, 386 Owyhee and 567 McPhee, stay in.)
6. **Rule-model runs for 1980-2020 were made without a fixed hash seed.** The scored runs are archived as they are. A seeded rerun differs at 79 of 510 dams (medians identical). Either keep the archived runs (recommended; say so in the data statement) or adopt the seeded reruns and rebuild the figures.
7. **Rule models run on the first graph version and with routed inflow**, while SWOT-GAT uses graph v3.5. The Methods should say so.

## C. Checks worth doing before publication

8. **GPU job scripts are untested.** `05_training/train.sh` and `06_predictions/predict.sh` are new, cleaner versions of the launch scripts. A 10-minute training start and one short prediction would confirm them. This needs GPU time.
9. **Equivalence of the two prediction runners for the paper model.** Predict one window with both runners and compare (about 2 GPU-hours each). So far this was tested for an earlier model only.
10. **Driver scripts** (`07_evaluation/c_scoring/drivers/`): they read their reservoir list, network and comparison columns from earlier model configurations (details in `docs/KNOWN_LIMITS.md`, section 6). The paper's quantities do not depend on those models, but a clean version needs the scripts changed to take the list from the paper's score table and the network from graph v3.5, followed by a rerun and a check that Figure 3's numbers are unchanged. The same applies to `ip_series.py` (dam list from `series_g.npz`) and to the first comparison set in `swot_vs_gauge.py`.
11. **Citations marked [confirm]** in `docs/DATA_SOURCES.md`.
12. **Results-text numbers.** `text_nums_o50.py` produces the numbers quoted in the Results. It is included but was not part of the smoke test.

13a. **Notes from the documentation pass** (`docs/CODE_NOTES.md`). Three need a decision or a check:
   - `text_nums_o50.py` computes its driver numbers with the earlier irregularity and coverage measures, not the harmonic irregularity and standard-deviation coverage shown in Figure 3. Check which numbers the Results text quotes.
   - The crosswalk audit labels record an AI assistant as the visual reviewer of the sampled dam-to-node matches (`step6c_audit_labels.py` and the two shipped label tables). Decide how the paper and the repository should describe that review.
   - The expanded gauge harvest fetched from agency services whose robots.txt disallows automated access. The docstrings say so and tell users to check current terms. Decide whether to keep that wording.
13b. **Names inside data and code strings.** The first author's first name, and references to the project's internal log, remain where they are data: the `why` column of `01_graph/inputs/decisions_v35.csv` and `review_v34_items.csv`, and text that scripts write into report files and a few table columns. Removing them means editing code strings and frozen inputs (their md5 sums would change).

## D. Smaller points

13. **Small tables in the repository.** About 2.2 MB of frozen inputs are shipped under `*/inputs/` and `vendor/graph_delineator/all_matchups.csv` (review rulings, edit lists, gauge tables, node lists, map outlines). They are hand decisions or tiny lookups the scripts cannot run without. Say if any should move to the data archive instead.
14. **File and label names** keep the project's internal tags (`v34`, `v35`, `c5`, `c5d`, `v10pm`, `SWOT-GNN` in table labels). Renaming would break the one-to-one link to the files that ran; the README has a glossary instead.
15. **KGE.** No scorer of the paper computes KGE. `validate_engines.py` reports it only because MODROM's published calibration metric is KGE, and the vendored upstream model code has a KGE function the paper does not call.
16. **Code style.** Many scripts put several statements on one line. An automatic formatter would fix that without changing behaviour of most files, but six scripts read other scripts as text and patch them, so a formatting pass needs the smoke test and a careful check of those six. Not done.
17. **Paper details to fill in**: title, authors, DOI in `README.md` and `CITATION.cff`.
