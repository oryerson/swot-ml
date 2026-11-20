"""Choose which set of rule-model runs the figure tables refer to, through the GRL_TAG environment variable.

  GRL_TAG="_g" (default)  the rule models were started in 2000 from gauge storage
                          (grades_sim_v4.py --init gauge)
  GRL_TAG=""              earlier runs started at 60% of capacity (ISTARF: at the middle of its normal
                          operating range); their tables are kept unchanged

Every file derived from the rule runs carries the tag in its name: series{TAG}.npz,
sites_v2{TAG}.csv ... sites_v5{TAG}.csv, caption_numbers_v2{TAG}.json. tagged("series.npz") returns
"series_g.npz" under the default. Only the rule models differ between the two sets; the rows of the
SWOT model are the same in both.

RUN, ATLAS and SCORES give the run tags and the folder names of the series atlases and score folders of an
earlier model configuration. In these names and in the tables, the labels `v10pm` and `SWOT-GNN` stand for
the SWOT model: SWOT-GAT (the paper's model: mean of two seeds) in the paper's tables.

Notes: the paper's figures reach this module only through grl2.py, which imports TAG and tagged().
Any value of GRL_TAG other than "" or "_g" stops the import with an assertion error.
"""
import os
from pathlib import Path

TAG = os.environ.get("GRL_TAG", "_g")
assert TAG in ("", "_g"), f"GRL_TAG must be '' or '_g', not {TAG!r}"

def tagged(fname):
    """Return the file name with TAG inserted before its extension ("series.npz" -> "series_g.npz")."""
    p = Path(fname)
    return f"{p.stem}{TAG}{p.suffix}"

RUN = {"GDROM": f"gdrom3c{TAG}", "ISTARF": f"istarf3c{TAG}", "MODROM-cal": f"modromcal3c{TAG}",
       "MODROM-tr": f"modromrf3c{TAG}_pubnorm"}
ATLAS = {"v10pm": "storage_atlas_v10pm", **{m: f"storage_atlas_{t}" for m, t in RUN.items()}}
SCORES = {"v10pm": "scores_v10pm_50k_clean", **{m: f"scores_grades_{t}_50k_clean" for m, t in RUN.items()}}
