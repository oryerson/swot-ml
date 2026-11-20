# vendor/graph_delineator: the basin delineator

`src/graph_delineator/` is the package that the graph v3.4 delineation job
imported. Its lineage:

1. https://github.com/tedlanghorst/graph_delineator (Ted Langhorst), the gauge-only delineator.
2. The fork https://github.com/oryerson/graph_delineator, branch `Reservoirs`,
   commit 70e7b087 (2026-03-26), which adds reservoir-aware delineation
   (`delineate_reservoirs_working.py`, `split_catchment_reservoirs.py`,
   `dataclasses_reservoirs.py`).
3. Changes made after that commit and **not yet pushed to the fork**:
   - `__init__.py` imports the reservoir-aware delineator (the committed
     version imports the gauge-only one, so a clone of the fork runs the wrong code);
   - `resolve_raster()` accepts a VRT mosaic or a GeoTIFF;
   - reservoirs are pre-screened against the basin footprint;
   - the pour-point check uses `grid.nearest_cell(..., snap="corner")`, which
     fixes an off-by-one that dropped valid gauges.
4. `patches/delineate_reservoirs_working_v34.patch`: the cumulative patch to
   `delineate_reservoirs_working.py` behind graph v3.4 (six routing rules for
   lakes and a minimum snap area for gauges, with a retry). The file in `src/`
   already contains it; the patch is kept so the change can be read on its own.
   Two comment lines of the file were edited four days after the delineation
   job; the code is the same.

**Licence: neither upstream repository carries a licence file (see `docs/OPEN_ITEMS.md`).**

`all_matchups.csv` is the gauge-to-outlet table the basin list is built from
(tracked in the fork). `Catchment_Split_Rules.txt` and `README_upstream.md`
are the fork's own notes. `delineate.py`, `delineate_reservoirs_withjoins.py`
and `split_catchment.py` are part of the package but are not on the import
path of the paper's delineation.
