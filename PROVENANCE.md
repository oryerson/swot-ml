# Provenance

This repository was assembled from the project's working folders on the UMass
Unity cluster by copying, never by moving or editing the originals.
`provenance/files.tsv` has one row per file:

| Column | Meaning |
|---|---|
| `release_path` | the file in this repository |
| `original_path` | where it was copied from (`$R` = the project's working root) |
| `original_md5`, `release_md5` | md5 of the original and of the file here |
| `class` | how the two relate (below) |
| `lines_changed` | lines that differ, counting added and removed lines |
| `deploy_to` | where `tools/deploy.py` places the file in the work tree (`ROOT:relative path`) |
| `note` | what was changed by hand, and why |

## Classes

| Class | Meaning | How it is established |
|---|---|---|
| `identical` | byte-for-byte copy | md5 |
| `paths-only` | Python file whose only change is that hard-coded folder names became names imported from `swotgat.paths` | Two automatic checks. (1) `tools/release_build/rewrite_paths.py` rewrites string literals token by token, then parses the original and the rewritten file and requires identical syntax trees once the original folder names are substituted back. (2) `build_release.py provenance` requires the file here to have the same syntax tree as that rewritten file. Comments and docstrings are outside both checks, so they may be reworded freely. |
| `paths-only*` | text file (a configuration template) in which folder names became `${SWOTGAT_...}` placeholders | substituting the original names back gives the original bytes; for the two model configurations the smoke test also checks the md5 sums of the files the paper runs were trained from |
| `edited` | code changed beyond paths | every such change is an exact text replacement listed in `tools/release_build/hand_edits.py`, with its reason copied into `note` |
| `new` | written for this release | — |

`paths-only` therefore means: with `config/paths.env` pointing at the original
folders, the file computes exactly what the original computed.

## Docstrings and comments

After the copy, the module docstrings and comments of the pipeline scripts were
rewritten for readers outside the project: what each script does, what it reads
and writes, how to call it, and the rules it applies. The original text was an
internal lab notebook. No code was changed in that pass. This is checked the
same way as above: a `paths-only` file still has the syntax tree of the
automatic edit of its original, with docstrings and comments set aside.
`lines_changed` therefore counts many changed lines for such files, all of them
documentation. Two side effects are harmless: scripts that pass their docstring
to `argparse` show the new text under `--help`, and text that scripts write
from string literals (report files, some table columns) was not touched.

The docstrings were then brought to [PEP 257](https://peps.python.org/pep-0257/):
every module, public class, public function and public method has a docstring
with a one-line summary, and comments that described a function became its
docstring. Adding a docstring adds a statement to the syntax tree, so the
comparison drops every statement that is only a string before comparing. A few
functions written on one line were re-wrapped to make room for the docstring;
their statements are unchanged. `python tools/check_docstrings.py <folder>`
checks the conventions. The vendored code under `vendor/` was left byte-identical
and is not covered.

Because of this pass, `build_release.py copy --force` would put the original
notebook text back; use `copy` without `--force`.

## What was edited by hand, in summary

| Files | Change | Effect on results |
|---|---|---|
| 8 gauge-harvest scripts, 2 SWOT download scripts, 4 GRADES/drought-index download scripts | downloads go through `swotgat.http`: a generic client name instead of a personal contact string, and at most one request per 1.1 s per host | none on the data; downloads take longer |
| `08_figures/combo_e.py`, `kit_j.py`, `jw_fig_tt_e.py` | removed reads of other model configurations' files that no panel of the paper's figures uses, and two plotting functions of an earlier figure set | none: the smoke test reproduces the figure numbers and images |
| `07_evaluation/c_scoring/full_record/pinned_check.py`, `picks.py` | read two small tables from their place in this release instead of from superseded folders (same content) | none: outputs byte-identical in the smoke test |

## What is new

- `05_training/train.sh`, `submit_training.sh`, `06_predictions/predict.sh`, `submit_predictions.sh`: clean job scripts. They replace the project's launch scripts (`v35_launch/train_v35.sh`, `c4/launch_c5_train.sh`, `c4/pred_c4.sh`, `c5_paper/pred_c5.sh`, `c5_paper/submit_c5.sh`), which also carried the gates and bookkeeping for comparing model configurations. The new scripts keep the same commands, checks, environment variables, time limits and resources. **They have been syntax-checked and dry-run only; they have not been run on a GPU.**
- `08_figures/prepare_kit.py` replaces a chain of four copy-and-patch scripts that built the figure tables for successive figure versions. The smoke test shows that its five tables are byte-identical to the paper's.
- `07_evaluation/c_scoring/env.sh`, `bin/`, `tools/`, `src/swotgat/`, `tests/smoke/`, `config/`, and all documentation.

## The model source and the delineator

`vendor/swot_ml/src` is byte-identical to the source tree the paper's runs
imported (54 Python files; the staging folder's seven `test_*.py` scripts are
left out). `vendor/graph_delineator/src` is byte-identical to the package the
delineation job imported, except that one file had two comment lines edited
after the job. Each folder's `UPSTREAM.md` gives the upstream commit and the
changes made for this study.

## Rebuilding this release

`tools/release_build/` holds the manifest (`manifest.tsv`), the folder names
that were replaced (`roots.tsv`, `relocations.tsv`), the hand edits and the
build script:

```bash
python tools/release_build/build_release.py copy           # originals -> this tree (authors' cluster only; skips existing files)
python tools/release_build/build_release.py edits
python tools/release_build/build_release.py provenance     # rewrites provenance/files.tsv
```
