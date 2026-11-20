<h1 align='center'>swot-gat</h1>

Code for *[paper title to be added]* (Ryerson et al., in preparation). SWOT-GAT is a graph-attention LSTM that predicts daily reservoir and river storage change on river-network graphs. It is trained only on storage change observed by the SWOT satellite (July 2023 to September 2026, the "SWOT Period") in 100 river basins of the conterminous United States, then run back to 1981 on ERA5-Land weather and scored against reservoir gauges it never saw. This code includes methods for:
- delineating basins and building the river-network graphs
- static attributes, ERA5-Land forcing and SWOT training targets
- training (two random seeds) and prediction
- evaluation against gauge records and against rule-model benchmarks
- the paper's figures

Every step runs through two small commands, [bin/run](./bin/run) and [bin/submit](./bin/submit). The repository holds code only; data and trained models are archived separately (see [Data files](#data-files)).

> This is a staged release that has not been published. Open decisions (licence, data archive) are listed in [docs/OPEN_ITEMS.md](./docs/OPEN_ITEMS.md).

## Setup
### Environment
The model, data, scoring and figure code use one Python 3.11 environment (JAX, Equinox). You can use [uv](https://docs.astral.sh/uv/getting-started/) to create it from the exact package list of the paper's runs:
```sh
uv venv --python 3.11 /path/to/swot-ml-venv
uv pip install --python /path/to/swot-ml-venv/bin/python -r environment/swot_ml/requirements-frozen.txt
```
JAX with CUDA support is installed from that list, but your system still needs the CUDA drivers. Check that JAX finds your GPU:
```sh
/path/to/swot-ml-venv/bin/python -c "import jax; print(jax.devices())"
```
Scoring and figures run on a CPU. Delineation, the HALF lake filter and the Copernicus downloads use three more environments, described in [environment/README.md](./environment/README.md).

### Configuration file
All machine-specific paths, interpreters and Slurm flags are set in one file:
```sh
cp config/paths.env.example config/paths.env    # then edit it
source config/paths.env                         # in every shell, and before every sbatch
```

### Deploy
The scripts find their inputs and each other relative to where they sit, so they must run from a fixed layout inside your work folder. This command copies each script to its place (it never overwrites a file that differs unless you pass `--force`):
```sh
python tools/deploy.py
```
You still refer to a script by its path in this repository: `bin/run <script>` runs it from its deployed place, and `bin/submit cpu|gpu [sbatch options] -- <script>` does the same as a Slurm job.

### Data files
To reproduce the paper's scores and figures you need about 15 GB from the data archive: the model predictions, the gauge test sets and the rule-model runs listed in [tests/smoke/inputs_eval.txt](./tests/smoke/inputs_eval.txt). To rebuild everything from raw public data, fetch the datasets in [docs/DATA_SOURCES.md](./docs/DATA_SOURCES.md). The layout of the work folder is in [docs/DATA_LAYOUT.md](./docs/DATA_LAYOUT.md):
```
<work_dir>/
├── CONUS/                      graphs, attributes, node lists, data stores, model runs
├── CONUS_storage_fork/
│   ├── predictions/            SWOT-GAT and rule-model predictions
│   ├── testsets/               frozen gauge test sets
│   └── eval/                   everything else: build folders, scores, figures
└── GRADES/                     inflow and rule parameters for the benchmarks
```

## Run the pipeline
Run the stages in order. Each folder holds the scripts of one stage; [docs/PIPELINE.md](./docs/PIPELINE.md) lists every script with its inputs, outputs, run time and the exact commands used for the paper.

| Stage | Makes | Start here |
| --- | ----------- | --- |
| [01_graph](./01_graph) | sub-basins and river-network graphs (24,381 nodes in 100 basins) | `delineate_basin_v34.py`, `apply_v35.py`, `build_conus_graphs_v2.py` |
| [02_static_attributes](./02_static_attributes) | 80 static attributes per node | `build_conus_static.py`, `assemble_checks_v35_r2.py` |
| [03_forcings](./03_forcings) | daily ERA5-Land inputs per node, long-memory runoff indices | `build_store_v35.py`, `build_longmem_v35.py` |
| [04_swot_targets](./04_swot_targets) | SWOT storage-change targets at lakes and rivers, node lists, masks | `build_halfexact.py`, `river_build_v35ra.py`, `build_lists_v35.py` |
| [05_training](./05_training) | the trained model, two seeds | `submit_training.sh` |
| [06_predictions](./06_predictions) | daily predictions, 1981-2020 and the SWOT Period | `submit_predictions.sh` |
| [07_evaluation](./07_evaluation) | gauge test sets, rule-model benchmarks, scores (NSE, nRMSE, r, variability ratio) | `c_scoring/overlap50/score_2010.py` |
| [08_figures](./08_figures) | Figures 1-4 | `fig1_c5d.py` ... `fig4_c5d.py` |

### Train a model
Once the environment, configuration file and data stores are in place:
```sh
05_training/submit_training.sh        # DRYRUN=1 prints the jobs without submitting
```
This checks the training cache, then trains each seed for 50,000 steps as four chained 30-hour GPU jobs (about 64 GPU-hours per seed on a 48 GB GPU). The two configurations in [05_training/configs](./05_training/configs) differ only in the seed. The paper model is the mean of the two seeds' predictions.

### Predict
```sh
06_predictions/submit_predictions.sh  # DRYRUN=1 prints the jobs without submitting
```
Seven time windows per seed cover 1981-2020 and the SWOT Period (1.5 to 4.5 GPU-hours each).

### Score and draw the figures
```sh
source 07_evaluation/c_scoring/env.sh
BASE=1 bin/run 07_evaluation/c_scoring/overlap50/score_2010.py       # 2010-2020
BASE=1 bin/run 07_evaluation/c_scoring/overlap50/score_x26span.py    # SWOT Period
BASE=1 bin/run 07_evaluation/c_scoring/overlap50/decade.py           # 1981-1990
bin/run 08_figures/prepare_kit.py
bin/run 08_figures/fig1_c5d.py                                       # likewise fig2, fig3, fig4
```
These are the main scorers; the full sequence (matched reservoir sets, paired tests, the rule-model ensemble, driver tables) is in [docs/PIPELINE.md](./docs/PIPELINE.md#stage-07-evaluation). A reservoir is scored in a period only if model and gauge overlap on at least 50% of its days. All metrics are computed after removing each series' own mean, because model and gauge use different reference levels.

## Tests
All tests are in [tests](./tests). Neither set trains a model or needs a GPU.

### Smoke test
The smoke test ([tests/smoke](./tests/smoke)) rebuilds the graph file, re-scores the archived predictions, redraws the four figures and compares every output with the paper's (about 6 minutes on one CPU node with 48 GB). Run it in a work folder of its own:
```sh
tests/smoke/link_inputs.sh <folder holding the data archive> tests/smoke/inputs_eval.txt
sbatch $SWOTGAT_SBATCH_CPU -c 4 --mem=48G -t 1:00:00 tests/smoke/run_smoke.sh
```
The last result is in [tests/smoke/RESULT.md](./tests/smoke/RESULT.md).

### Unit tests
The unit tests ([tests/unit](./tests/unit)) check the model, the loss, the data pipeline, the scorers and the rule models on small synthetic inputs: 107 tests, about 8 minutes on 8 CPU cores, no data archive needed. They need `pytest` on top of the model environment:
```sh
uv pip install --python "$SWOTGAT_PYTHON" -r environment/tests/requirements.txt
JAX_PLATFORMS=cpu "$SWOTGAT_PYTHON" -m pytest tests/unit -p no:cacheprovider
```
[tests/unit/README.md](./tests/unit/README.md) lists what each test checks and what the tests found; the last recorded run is in [tests/unit/PYTEST_OUTPUT.txt](./tests/unit/PYTEST_OUTPUT.txt) (103 passed, 4 expected failures).

## What reproduces exactly
- **Scores and figures**, from the archived predictions and gauge tables: identical (checked by the smoke test).
- **Predictions**, from the archived trained models: identical on the same GPU type and library versions.
- **Training**: not bit-identical; the paper's runs were resumed across several GPU hosts.
- **Graph, attributes, forcing and targets**, from raw data: the scripts are the ones that ran, but a few steps reuse small tables from earlier graph versions and some downloads are dated snapshots. [docs/KNOWN_LIMITS.md](./docs/KNOWN_LIMITS.md) lists each case.

## About this code
- The scripts are the ones that produced the paper's results, changed only in their path names and documentation. [PROVENANCE.md](./PROVENANCE.md) explains how that is checked for every file, and [EXCLUDED.md](./EXCLUDED.md) lists what was left out (other model configurations, earlier graph versions).
- File names keep the project's internal tags: `v35` is the paper's graph (version 3.5), `c5` its model configuration, `S1`/`S2` the two seeds. In some tables the labels `SWOT-GNN` and `v10pm` mean SWOT-GAT.
- The model in [vendor/swot_ml](./vendor/swot_ml) and the delineator in [vendor/graph_delineator](./vendor/graph_delineator) are by Ted Langhorst, with the changes listed in each folder's `UPSTREAM.md`. [vendor/half_v1](./vendor/half_v1) is the HALF lake filter of Wang and Trudel, unmodified.
- Docstrings of the pipeline scripts, tools and tests follow [PEP 257](https://peps.python.org/pep-0257/); `python tools/check_docstrings.py <folder>` checks them. The vendored code is kept exactly as it ran.
- Licence: to be decided, see [LICENSE](./LICENSE). To cite this code, see [CITATION.cff](./CITATION.cff).
