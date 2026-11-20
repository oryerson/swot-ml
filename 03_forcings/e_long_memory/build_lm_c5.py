"""Build the two long-memory inputs of the paper's model for the whole record, 1980-2026.

The paper's model (configuration C5) reads two long-memory inputs: lm_up_ssi3 and lm_up_ssi12,
the standardized runoff of each node's upstream area summed over 3 and 12 months. The store
longmem_v35 (build_longmem_v35.py) holds them from 2009-01-01 only. Predicting before 2009 needs
them from 1980 on, and this script writes them for 1980-01-01 to 2026-09-30.

Reads (paths relative to $SWOTGAT_WORK/CONUS_storage_fork; all inputs are only read)
    eval/v35_inputs/forcing/zarr_store_ext1980_v35   forcing (ro_mean) up to 2025-12-31
    eval/v35_inputs/forcing/zarr_store_vd2026_v35    forcing (ro_mean) from 2026-01-01
    eval/graph_v3/v35/graphs_conus_v35.json, subbasins_conus_v35.parquet   graph v3.5 and the area
                                                     weights (through build_longmem_v35.py)
    features/longmem_v35/manifest.json               to assert the same sources and reference
    eval/overnight/longmem/build_longmem.py and eval/featscreen_v2/lmrank/build_longmem_v35.py
                                                     imported as modules (the deployed copies of
                                                     the two scripts of this folder)

Writes
    eval/c5_paper/longmem_v35_c5_8026/<basin>/   one consolidated Zarr group per basin with the two
                                                 variables (date, subbasin), float16
    eval/c5_paper/longmem_v35_c5_8026/manifest.json, build_log.json

Usage
    python build_lm_c5.py [--workers 8] [--basins A,B]
    The exit code is 1 if any basin failed. Run check_lm_c5.py afterwards.

Method
    The per-basin routine follows run_basin of build_longmem_v35.py, with these differences:
    - Only the two inputs are computed, with the functions of build_longmem.py (rolling_sum,
      rank_normal, upstream_matrix, MONTH) and the formula of its basin_features:
          lm_up_ssi<n> = rank_normal(rolling_sum(Mup @ ro, round(n * MONTH)), month,
                                     reference rows that are valid)
      where Mup is the upstream-mean matrix and ro the daily runoff. The fixed 1981-2020
      reference and the replacement of non-finite values by 0 are unchanged.
    - The values are written for 1980-01-01 to 2026-09-30. (The builder reads the forcing up to
      the last written day anyway, and the reference needs 1981-2020.)
    - The output goes to its own folder, never into longmem_v35 or a forcing store.
    The forcing, the graph and the area weights are those of longmem_v35. check_lm_c5.py then
    requires the values of 2009-01-01 to 2026-09-30 to be bit-identical to longmem_v35.

Notes
    There is no forcing before 1980-01-01. The 3-month sum is therefore undefined on the first
    90 days of 1980 and the 12-month sum on the first 364 days. These values are written as 0
    (z = 0, the reference median), as the builder does for every non-finite value. They only
    enter the 365-day input windows of the first predicted year (1981).
"""
from swotgat.paths import WORK
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
F = Path(f"{WORK}/CONUS_storage_fork")
sys.path.insert(0, str(F/"eval/featscreen_v2/lmrank")); sys.path.insert(0, str(F/"eval/overnight/longmem"))
import build_longmem as B                 # noqa: E402  (numerics, read only)
import build_longmem_v35 as LV            # noqa: E402  (graph, areas, md5 pins, read only)
KEEP = {"lm_up_ssi3": 3, "lm_up_ssi12": 12}
OUT = F/"eval/c5_paper/longmem_v35_c5_8026"; W0, W1 = "1980-01-01", "2026-09-30"; VERSION = "longmem_v35_c5_8026"
SRC1 = F/"eval/v35_inputs/forcing/zarr_store_ext1980_v35"; SRC2 = F/"eval/v35_inputs/forcing/zarr_store_vd2026_v35"


def run_basin(basin, dtype="float16"):
    """Build and write the group of one basin for 1980-01-01 to 2026-09-30.

    Reads the runoff of 1980-2025 from the first forcing store and that of 2026 from the second,
    takes its area-weighted upstream mean and computes the two inputs in KEEP (standardized
    3-month and 12-month runoff) with the numerics of build_longmem.py. Sub-basins with a missing
    or non-positive area get the median area of the basin as weight. NaN results are written
    as 0. Returns the log entry of the basin: the number of sub-basins, the NaN counts before
    filling (over all days, and from 1981-01-01 on) and the run time in seconds.
    """
    import xarray as xr, networkx as nx
    t0 = time.time(); G = LV.graph(); vs = ["tp_mean", "e_mean", "ro_mean"]
    d1 = xr.open_zarr(SRC1/basin, consolidated=False)[vs].sel(date=slice(None, "2025-12-31")).transpose("date", "subbasin")
    d2 = xr.open_zarr(SRC2/basin, consolidated=False)[vs].sel(date=slice("2026-01-01", W1)).transpose("date", "subbasin")
    assert list(d1.subbasin.values.astype(str)) == list(d2.subbasin.values.astype(str)), f"{basin}: subbasin order differs"
    ds = xr.concat([d1, d2], dim="date"); dates = pd.DatetimeIndex(ds.date.values); subs = [str(s) for s in ds.subbasin.values]
    assert dates[0] == pd.Timestamp(W0) and dates[-1] == pd.Timestamp(W1), f"{basin}: forcing {dates[0]}..{dates[-1]}"
    assert (np.diff(dates.values).astype("timedelta64[D]").astype(int) == 1).all(), f"{basin}: dates not continuous"
    comp = next(c for c in nx.weakly_connected_components(G) if basin in c); sub = G.subgraph(comp)
    miss = sorted(set(sub.nodes) - set(subs)); assert not miss, f"{basin}: {len(miss)} graph nodes have no forcing column"
    ro = ds["ro_mean"].values.astype(np.float64)
    a = LV.areas(); a = a[a.outlet_id == basin]["area_km2"]
    area = a.reindex(subs).values.astype(np.float64); bad = ~np.isfinite(area) | (area <= 0)
    if bad.any(): area[bad] = np.nanmedian(area[~bad]) if (~bad).any() else 1.0
    Mup = B.upstream_matrix(sub, subs, area)
    month = dates.month.values; ref_rows = (dates >= pd.Timestamp(B.REF[0])) & (dates <= pd.Timestamp(B.REF[1]))
    R = (Mup @ ro.T).T; del ro                                   # upstream-mean runoff, as R of the lm_up_ pass in basin_features
    Fx = {}
    for k, n in KEEP.items():
        w = int(round(n * B.MONTH)); valid = np.arange(len(dates)) >= w - 1
        Fx[k] = B.rank_normal(B.rolling_sum(R, w), month, ref_rows & valid)
    n_nan = {k: int((~np.isfinite(v)).sum()) for k, v in Fx.items()}
    n_nan_after_1980 = {k: int((~np.isfinite(v[dates >= pd.Timestamp("1981-01-01")])).sum()) for k, v in Fx.items()}
    data = {k: (("date", "subbasin"), np.nan_to_num(v, nan=0.0).astype(dtype)) for k, v in Fx.items()}
    xo = xr.Dataset(data, coords={"date": dates.values, "subbasin": ds.subbasin.values})
    xo.attrs = dict(version=VERSION, reference=f"{B.REF[0]}..{B.REF[1]} ERA5-Land forcing (v3.5 store)", forcing=f"1980-2025 {SRC1}; 2026 {SRC2}",
                    transform="per subbasin x calendar month rank->normal, Gringorten (build_longmem.py)", features=list(KEEP),
                    nan_filled_with_zero=json.dumps(n_nan), built=time.strftime("%Y-%m-%d %H:%M"))
    xo.to_zarr(OUT/basin, mode="w", consolidated=True, encoding={k: {"chunks": (365, len(subs))} for k in data})
    return dict(basin=basin, n_subbasins=len(subs), n_nan=n_nan, n_nan_from_1981=n_nan_after_1980, seconds=round(time.time() - t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--basins", default=None); a = ap.parse_args()
    for p, h in LV.MD5.items(): assert LV.md5(p) == h, f"md5 changed: {p}"
    man = json.load(open(F/"features/longmem_v35/manifest.json")); assert man["source"] == [str(SRC1), str(SRC2)], man["source"]
    assert man["reference"] == list(B.REF), (man["reference"], B.REF); assert set(KEEP) <= set(man["features"])
    OUT.mkdir(parents=True, exist_ok=True)
    G = LV.graph(); s = LV.areas(); gn = sorted(G.nodes)
    basins = a.basins.split(",") if a.basins else sorted(set(s.loc[gn, "outlet_id"]))
    sizes = s.groupby("outlet_id").size(); basins = sorted(basins, key=lambda b: -sizes.get(b, 0))
    json.dump(dict(version=VERSION, features=list(KEEP), write=[W0, W1], reference=list(B.REF), dtype="float16", source=[str(SRC1), str(SRC2)],
                   note="C5's two long-memory inputs extended back to 1980 for the 1981-2009 hindcast (c5_paper); 2009-2026 checked bit-identical "
                        "to features/longmem_v35; 1980 values before a full accumulation window are zero-filled"), open(OUT/"manifest.json", "w"), indent=1)
    print(f"{VERSION}: {len(basins)} basins, write {W0}..{W1}", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    log, fails = [], 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_basin, b): b for b in basins}
        for f in as_completed(futs):
            try: r = f.result(); log.append(r); print("done", r["basin"], r["n_subbasins"], r["n_nan"], r["n_nan_from_1981"], r["seconds"], flush=True)
            except Exception as e: fails += 1; log.append(dict(basin=futs[f], error=f"{type(e).__name__}: {e}")); print("FAILED", futs[f], type(e).__name__, e, flush=True)
    json.dump(log, open(OUT/"build_log.json", "w"), indent=1)
    print(f"{len(basins) - fails} built, {fails} failed", flush=True); sys.exit(1 if fails else 0)
