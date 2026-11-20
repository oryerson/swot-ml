# vendor/swot_ml: the model source

`src/` is the exact source tree that every training and prediction run of the
paper imported. It is a copy of the public repository

    https://github.com/Fluvial-UMass/swot-ml   (Ted Langhorst, University of Massachusetts Amherst)
    commit 92a7c0e246ca58b28b7ada7235961c2836755078 (2026-04-16)

plus the changes listed below, which were made for this study. The copy was
frozen in a staging folder before the runs started; `provenance/files.tsv`
records the md5 of every file, and `tests/smoke/static_checks.py` re-checks
them. The staging folder's own test scripts (`test_*.py`) are not included.

**Licence: the upstream repository carries no licence file. Redistribution
here needs the author's permission and an agreed licence (see
`docs/OPEN_ITEMS.md`).**

## Files changed relative to upstream commit 92a7c0e

| File | Change (lines touched) |
|---|---|
| `config/config.py` | New options: fallback targets, per-target node lists and drop masks, area-scaled targets, re-referencing of targets to their median, external long-memory store, burn-in steps, checkpointed time scan (`remat_level`), seeded sampler, cache folder (230) |
| `config/model_args.py` | `static_dropout`, `area_scaled` (6) |
| `data/cached_basingraphdataset.py` | Per-target node lists, fallback target columns, drop masks, external long-memory store, statistics from supervised basins only, cache keys, empty-cache guard (392) |
| `data/cached_basingraphdataloader.py` | Seeded sampler (6) |
| `data/basindatalake.py` | Retry on concurrent Delta Lake commits (16) |
| `data/zarr_store.py` | Time-zone handling on write (6) |
| `evaluate/inference.py` | Mixture mean and standard deviation outputs, last-step output of sequence-to-sequence models, tolerance of all-NaN rows (96) |
| `evaluate/metrics.py` | Masks, guarded metrics (30) |
| `models/__init__.py` | Area-scaling argument (17) |
| `models/sparse_fusion_gann.py` | Block-checkpointed time scan (memory), static dropout, area scaling of outputs (117) |
| `run.py` | Separate validation cache; plotting errors do not stop a run (17) |
| `train/loss.py` | Burn-in mask, fallback-target loss, and optional loss terms that the paper model does not use (407) |
| `train/trainer.py` | Checkpointed-scan switch, validation accumulators, a counter of zeroed NaN gradients (96) |

All other files are unchanged from upstream. `CHANGES_v35.md` describes the
last change (the NaN-gradient counter, logging only).

The paper model is `models/sparse_fusion_gann.py` with `models/layers/`
(`graph_attn.py`, `lstm.py`, `heads.py`, `static_mlp.py`); the repository's
other model classes are imported by `models/__init__.py` but not used.

`README_upstream.md` and `code_overview.drawio.png` are the upstream
repository's own documentation.
