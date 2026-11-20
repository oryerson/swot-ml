"""Build the long-memory inputs of the paper's graph (v3.5) for 2009-2026: the store longmem_v35.

The long-memory inputs are 32 standardized indices of precipitation minus evaporation and of
runoff, accumulated over 3 to 48 months, computed from ERA5-Land forcing at each node ("local",
lm_*) and over each node's upstream area ("upstream", lm_up_*). build_longmem.py defines them and
holds the numerics; this script applies them to graph v3.5. All 32 are written. The paper's model
(configuration C5) reads two of them: lm_up_ssi3 and lm_up_ssi12, standardized runoff of the
upstream area summed over 3 and 12 months. The store was first built for earlier model
configurations (C2 and C3).

Reads (paths relative to $SWOTGAT_WORK; all inputs are only read)
    --src1 DIR   forcing store read up to 2025-12-31; in the paper's run
                 CONUS_storage_fork/eval/v35_inputs/forcing/zarr_store_ext1980_v35
    --src2 DIR   forcing store read from 2026-01-01; in the paper's run
                 CONUS_storage_fork/eval/v35_inputs/forcing/zarr_store_vd2026_v35
                 (give the same store twice if one store holds 1980-2026)
                 Variables used: tp_mean, e_mean, ro_mean.
    CONUS_storage_fork/eval/graph_v3/v35/graphs_conus_v35.json       graph v3.5, for the upstream
                 set of each node (the node and all its ancestors)
    CONUS_storage_fork/eval/graph_v3/v35/subbasins_conus_v35.parquet `area_km2`, the weight of
                 each sub-basin in the upstream mean, and `outlet_id`, its basin
    CONUS_storage_fork/eval/overnight/longmem/build_longmem.py       imported as a module (the
                 deployed copy of build_longmem.py of this folder)

Writes (--out, default CONUS_storage_fork/features/longmem_v35)
    <basin>/          one consolidated Zarr group per basin: 32 variables (date, subbasin), float16,
                      chunks of 365 days x all sub-basins
    manifest.json     version, input names, written dates, reference period, dtype, sources
    build_info.json   graph and sub-basin files with their md5, script paths, counts
    build_log.json    per basin: sizes, NaN counts, seconds, or the error
                      (named build_log_<n>basins.json when --basins is given)

Usage
    python build_longmem_v35.py --src1 DIR --src2 DIR [--out DIR] [--write-start 2009-01-01]
                                [--write-end 2026-09-30] [--workers 8] [--dtype float16]
                                [--basins A,B]
    Basins are processed in parallel, largest first. The exit code is 1 if any basin failed.
    In the paper's run: about 1 hour with 8 workers and 180 GB of memory.

Method
    Same numerics as build_longmem.py: the rank-to-normal transform per sub-basin and calendar
    month against the fixed 1981-2020 forcing reference. The forcing must start on 1980-01-01,
    end on --write-end and be daily without gaps (asserted); values are written from
    --write-start on. Differences from running build_longmem.py itself:
    - the forcing comes from the two graph v3.5 stores;
    - the upstream sets come from the graph v3.5 file;
    - the area weights come from the v3.5 sub-basin table (a missing or non-positive area gets the
      median area of the basin);
    - every graph node of a basin must be a column of its forcing group (asserted);
    - NaN values are counted before they are replaced by 0, over all written days and over the
      check window 2022-01-01 to 2026-09-30, and the counts go to build_log.json.
    The script stops if the md5 of the graph file or of the sub-basin table differs from the
    values pinned in MD5, and it refuses an output folder that is, or is named like, a forcing
    store.

Notes
    To reproduce the archived store exactly, build all 32 inputs for all nodes of a basin in one
    call, as this script does: the rank of values tied with reference values depends on the full
    set of columns of the basin (see check_c3_ties.py).
"""
from swotgat.paths import WORK
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, f"{WORK}/CONUS_storage_fork/eval/overnight/longmem")
import build_longmem as B
F = Path(f"{WORK}/CONUS_storage_fork"); G35 = F/"eval/graph_v3/v35"
GRAPH = G35/"graphs_conus_v35.json"; SUBS = G35/"subbasins_conus_v35.parquet"
MD5 = {GRAPH: "c580577d7581bbfb975c261bea609051", SUBS: "19a764d997a07c870b2edb0688dbf3a8"}
VERSION = "longmem_v35"; CHK = ("2022-01-01", "2026-09-30")

def md5(p):
    """Return the md5 hex digest of the file p."""
    return hashlib.md5(open(p, "rb").read()).hexdigest()
def graph():
    """Return graph v3.5 as a networkx DiGraph with string node ids."""
    import networkx as nx
    G = nx.readwrite.json_graph.node_link_graph(json.load(open(GRAPH)), edges="edges"); return nx.relabel_nodes(G, {v: str(v) for v in G.nodes})
def areas():
    """Return the sub-basin table (columns area_km2 and outlet_id) indexed by string id."""
    s = pd.read_parquet(SUBS, columns=["area_km2", "outlet_id"]); s.index = s.index.astype(str); return s

def run_basin(basin, out, src1, src2, w0, w1, dtype):
    """Build and write the group of one basin for the dates w0..w1, and return its log entry.

    src1 is the forcing store read up to 2025-12-31 and src2 the store read for 2026; the joined
    record must start on 1980-01-01, be continuous and hold every graph node of the basin
    (asserted). out is the output folder and dtype the stored data type. Sub-basins with a
    missing or non-positive area get the median area of the basin as weight. NaN results are
    counted (over all written days and over the check window CHK), then written as 0.
    """
    import xarray as xr, networkx as nx
    t0 = time.time(); G = graph(); vs = ["tp_mean", "e_mean", "ro_mean"]
    d1 = xr.open_zarr(src1/basin, consolidated=False)[vs].sel(date=slice(None, "2025-12-31")).transpose("date", "subbasin")
    d2 = xr.open_zarr(src2/basin, consolidated=False)[vs].sel(date=slice("2026-01-01", w1)).transpose("date", "subbasin")
    assert list(d1.subbasin.values.astype(str)) == list(d2.subbasin.values.astype(str)), f"{basin}: subbasin order differs"
    ds = xr.concat([d1, d2], dim="date")
    dates = pd.DatetimeIndex(ds.date.values); subs = [str(s) for s in ds.subbasin.values]
    assert dates[0] == pd.Timestamp("1980-01-01") and dates[-1] == pd.Timestamp(w1), f"{basin}: forcing {dates[0]}..{dates[-1]}"
    assert (np.diff(dates.values).astype("timedelta64[D]").astype(int) == 1).all(), f"{basin}: dates not continuous"
    comp = next(c for c in nx.weakly_connected_components(G) if basin in c); sub = G.subgraph(comp)
    miss = sorted(set(sub.nodes) - set(subs)); assert not miss, f"{basin}: {len(miss)} graph nodes have no forcing column, e.g. {miss[:5]}"
    tp, e, ro = (ds[v].values.astype(np.float64) for v in vs); pe = tp + e
    a = areas(); a = a[a.outlet_id == basin]["area_km2"]
    area = a.reindex(subs).values.astype(np.float64); bad = ~np.isfinite(area) | (area <= 0)
    if bad.any(): area[bad] = np.nanmedian(area[~bad]) if (~bad).any() else 1.0
    Mup = B.upstream_matrix(sub, subs, area)
    wsel = (dates >= pd.Timestamp(w0)) & (dates <= pd.Timestamp(w1)); dw = dates[wsel]; chk = (dw >= pd.Timestamp(CHK[0])) & (dw <= pd.Timestamp(CHK[1]))
    nan_forcing_cols = int((~np.isfinite(pe[(dates >= pd.Timestamp(CHK[0])) & (dates <= pd.Timestamp(CHK[1]))])).any(axis=0).sum())
    Fx = B.basin_features(pe, ro, dates, Mup, keep=wsel); del tp, e, ro, pe
    n_nan = {k: int((~np.isfinite(v)).sum()) for k, v in Fx.items()}; n_nan_chk = {k: int((~np.isfinite(v[chk])).sum()) for k, v in Fx.items()}
    data = {k: (("date", "subbasin"), np.nan_to_num(v, nan=0.0).astype(dtype)) for k, v in Fx.items()}
    xo = xr.Dataset(data, coords={"date": dw.values, "subbasin": ds.subbasin.values})
    xo.attrs = dict(version=VERSION, reference=f"{B.REF[0]}..{B.REF[1]} ERA5-Land forcing (v3.5 store)", forcing=f"1980-2025 {src1}; 2026 {src2}",
                    pe="tp_mean + e_mean", acc_months=list(B.ACC), leak_months=list(B.LEAK), month_days=B.MONTH,
                    transform="per subbasin x calendar month rank->normal, Gringorten",
                    upstream="area_km2-weighted mean over {node} U ancestors in graphs_conus_v35.json (md5 c580577d)",
                    nan_filled_with_zero=json.dumps(n_nan), built=time.strftime("%Y-%m-%d %H:%M"))
    xo.to_zarr(out/basin, mode="w", consolidated=True, encoding={k: {"chunks": (365, len(subs))} for k in data})
    return dict(basin=basin, n_subbasins=len(subs), n_graph_nodes=sub.number_of_nodes(), mup_nnz=int(Mup.nnz), nan_forcing_cols_chk=nan_forcing_cols,
                n_nan=n_nan, n_nan_chk=n_nan_chk, seconds=round(time.time() - t0))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--src1", required=True); ap.add_argument("--src2", required=True)
    ap.add_argument("--out", default=str(F/"features/longmem_v35")); ap.add_argument("--write-start", default="2009-01-01")
    ap.add_argument("--write-end", default="2026-09-30"); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--dtype", default="float16")
    ap.add_argument("--basins", default=None); a = ap.parse_args()
    for p, h in MD5.items(): assert md5(p) == h, f"md5 changed: {p}"
    src1, src2, out = Path(a.src1), Path(a.src2), Path(a.out)
    for s in (src1, src2): assert "zarr_store" not in out.name and out.resolve() != s.resolve(), "never write into a source store"
    out.mkdir(parents=True, exist_ok=True)
    G = graph(); s = areas(); gn = sorted(G.nodes)
    basins = a.basins.split(",") if a.basins else sorted(set(s.loc[gn, "outlet_id"]))
    sizes = s.groupby("outlet_id").size(); basins = sorted(basins, key=lambda b: -sizes.get(b, 0))
    names = B.feature_names()
    json.dump(dict(version=VERSION, features=names, write=[a.write_start, a.write_end], reference=B.REF, dtype=a.dtype, source=[str(src1), str(src2)]),
              open(out/"manifest.json", "w"), indent=1)
    json.dump(dict(graph=str(GRAPH), graph_md5=MD5[GRAPH], subbasins=str(SUBS), subbasins_md5=MD5[SUBS], area="subbasins_conus_v35 area_km2",
                   builder=str(Path(__file__).resolve()), numerics=str(Path(B.__file__).resolve()), graph_nodes=len(gn), basins=len(basins),
                   check_window=list(CHK), started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), open(out/"build_info.json", "w"), indent=1)
    print(f"{VERSION}: {len(basins)} basins, {len(gn)} graph nodes, write {a.write_start}..{a.write_end}, src {src1} + {src2}", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    log, fails = [], 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_basin, b, out, src1, src2, a.write_start, a.write_end, a.dtype): b for b in basins}
        for f in as_completed(futs):
            try: r = f.result(); log.append(r); print("done", r["basin"], r["n_graph_nodes"], r["nan_forcing_cols_chk"], sum(r["n_nan_chk"].values()), r["seconds"], flush=True)
            except Exception as e: fails += 1; log.append(dict(basin=futs[f], error=f"{type(e).__name__}: {e}")); print("FAILED", futs[f], type(e).__name__, e, flush=True)
    json.dump(log, open(out/("build_log.json" if not a.basins else f"build_log_{len(basins)}basins.json"), "w"), indent=1)
    print(f"{len(basins) - fails} built, {fails} failed", flush=True); sys.exit(1 if fails else 0)
