"""Give target variables that are entirely NaN in a basin a zero-count normalization entry.

Reads and writes (in place): the `normalization_stats` attribute of every basin group in
`$SWOTGAT_WORK/CONUS/zarr_store`.

Usage: python fix_empty_stats.py        (no arguments; run after update_zarr_stats_conus.py)

Why it is needed
    04_swot_targets/b_rivers/build_storage_target_conus.py creates the river target `dv_river` in
    every basin so that all basins share one schema. In the 43 basins with no SWOT river data the
    variable is entirely NaN. The statistics step (`compute_and_store_stats` in the vendored
    swot_ml package) writes an entry only when a variable has at least one finite value. The
    variable is then present in the dataset but has no statistics, and the training data loader
    (`_load_and_normalize_basin`) raises an error for exactly that case. The loader catches the
    error per basin and only warns, so training would silently run on a partial cache. In the
    paper's run, 40 of the 87 basins loaded for training were in that state.

Method
    For each basin and each target variable (`dv_river`, `dv_SWOTAE`) that exists in the store
    but has no statistics:
    - if the variable holds any finite value, the gap is real; the script reports it and leaves
      it alone (the scan reads 400 nodes at a time and stops at the first finite value);
    - otherwise it writes a zero-count entry.
    A zero-count entry is the additive identity for every field the loader aggregates:
        count, sum, sum_sq, log_sum, log_sum_sq, positive_count are summed -> 0 adds nothing
        min is combined with min()  -> +inf changes nothing
        max is combined with max()  -> -inf changes nothing
    The statistics aggregated over the training basins are therefore bit-identical to leaving
    the basin out, and the loader's check passes.

    The script ends with a read-back check. Zarr attributes are stored as JSON, so it confirms
    that the +inf and -inf values survived the round trip.
"""
from swotgat.paths import WORK
import os, json, warnings
from pathlib import Path
import numpy as np, zarr
warnings.filterwarnings("ignore")
C = Path(f"{WORK}/CONUS"); Z = C / "zarr_store"
TARGETS = ["dv_river", "dv_SWOTAE"]
NEUTRAL = {"count": 0, "sum": 0.0, "sum_sq": 0.0,
           "min": float("inf"), "max": float("-inf"),
           "log_sum": 0.0, "log_sum_sq": 0.0, "positive_count": 0}

fixed = skipped = 0
for b in sorted(os.listdir(Z)):
    if not (Z / b / "zarr.json").is_file(): continue
    g = zarr.open(str(Z / b), mode="r")
    st = dict(g.attrs.get("normalization_stats", {}))
    add = {}
    for v in TARGETS:
        if not (Z / b / v).is_dir() or v in st:
            continue
        gg = zarr.open_group(str(Z / b), mode="r")
        nT, nS = gg[v].shape
        finite = 0
        for i in range(0, nS, 400):
            finite += int(np.isfinite(gg[v][:, i:min(i + 400, nS)]).sum())
            if finite: break
        if finite:
            print(f"  SKIP {b}/{v}: has {finite}+ finite values but no stats -- "
                  f"that is a real stats gap, not an empty variable")
            skipped += 1
            continue
        add[v] = dict(NEUTRAL)
    if add:
        st.update(add)
        zarr.open(str(Z / b), mode="r+").attrs["normalization_stats"] = st
        fixed += len(add)
print(f"neutral entries written: {fixed}   real gaps skipped: {skipped}")

print("\nread-back check (zarr attrs are JSON; infinities must survive):")
bad = 0
for b in sorted(os.listdir(Z)):
    if not (Z / b / "zarr.json").is_file(): continue
    st = zarr.open(str(Z / b), mode="r").attrs.get("normalization_stats", {})
    for v in TARGETS:
        if v in st and st[v]["count"] == 0:
            e = st[v]
            if not (e["min"] == float("inf") and e["max"] == float("-inf")):
                print(f"  BAD {b}/{v}: min={e['min']} max={e['max']}"); bad += 1
print(f"  zero-count entries with corrupted infinities: {bad}")
