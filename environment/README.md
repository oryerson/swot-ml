# Python environments

The paper's results were produced with four environments. Each folder holds
`requirements-frozen.txt`, the exact package versions installed in the
environment that ran (exported with `uv pip freeze`). These lists are the
reference. Create an environment from one with, for example:

```bash
uv venv --python 3.11 /path/to/swot-ml-venv
uv pip install --python /path/to/swot-ml-venv/bin/python -r environment/swot_ml/requirements-frozen.txt
```

Then put the interpreter paths into `config/paths.env`.

| Folder | Python | Used by | Variable in `config/paths.env` |
|---|---|---|---|
| `swot_ml/` | 3.11.15 | data stores, long-memory inputs, training, prediction, scoring, figures (stages 03-08) | `SWOTGAT_PYTHON` |
| `graph_delineator/` | 3.12.6 | graph builds and static attributes (stages 01-02) | `SWOTGAT_DELINEATOR_PYTHON` |
| `half/` | 3.12.3 | HALF lake filter (stage 04a: `half_batch.py`, `build_vd_half.py`, `release_replay.py`) | `SWOTGAT_HALF_PYTHON` |
| `cds/` | 3.12.3 | Copernicus downloads (stage 03c: `download_2026.py`, `download_inst.py`) | `SWOTGAT_CDS_PYTHON` |
| `tests/` | as `swot_ml/` | unit tests (`tests/unit`): `requirements.txt` adds pytest to the model environment | `SWOTGAT_PYTHON` |

Notes

- **Model environment.** JAX 0.5.3 with the CUDA 12 plugin, Equinox 0.13.0, Optax 0.2.5, NumPy 1.26.4. The paper's GPU jobs also loaded the cluster's CUDA 12.6 module and pointed XLA at it (`SWOTGAT_CUDA_DIR`). `swot_ml/pyproject.toml`, `uv.lock` and `.python-version` are the project files of the upstream model repository as they stood in the working copy; they are kept for reference, and `uv.lock` was not checked against the installed set.
- **Delineation.** The 102-basin delineation job itself (`01_graph/delineate_basin_v34.py`) ran under the cluster's system Python 3.12 with pysheds 0.5, shapely 2.1.2, geopandas 1.1.2, rasterio 1.5.0 and numba 0.63.1. The pandas version of that interpreter on the day of the run was not recorded. The other stage 01-02 scripts ran in the `graph_delineator/` environment.
- **Not on PyPI.** The upstream model project lists `global-gauges` (github.com/tedlanghorst/global_gauges) as a dependency. No script in this repository imports it; it was used to download the USGS site table listed in `docs/DATA_SOURCES.md`.
- **Figures.** The four-page figure PDF needs `pdflatex` with the `helvet` package.
