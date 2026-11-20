"""Long-memory inputs from ERA5-Land forcing: the numerics, and a builder for the first graph.

Role in the release
    The functions of this file (rolling_sum, leaky, rank_normal, month_anomaly, upstream_matrix,
    basin_features) are the numerics of all long-memory inputs. build_longmem_v35.py and
    build_lm_c5.py import them to build the stores of the paper's graph (v3.5). Run as a script,
    this file builds the same inputs on the first graph version (store version "longmem_v1"),
    which the paper's model does not read.

The inputs
    For every sub-basin and day there are 32 inputs: 16 "local" ones (prefix lm_), computed from
    the sub-basin's own forcing, and the same 16 "upstream" ones (prefix lm_up_), computed from
    the area-weighted mean of the forcing over the node and everything upstream of it. The
    upstream set of a node is the node plus all its ancestors in the basin graph. It is a set, so
    a contributor reached by two paths (a bifurcation) is counted once.
      spei{N}  standardized precipitation minus evaporation (P-E) summed over N months.
               P-E = tp_mean + e_mean, because ERA5 evaporation is negative.
      ssi{N}   standardized runoff (ro_mean) summed over N months.        N = 3, 6, 12, 24, 36, 48
      lk{M}    leaky integrator a_t = exp(-1/tau) * a_(t-1) + q_t of the standardized daily runoff
               anomaly q (calendar-month mean and standard deviation of the reference period
               removed), with tau = M months.                             M = 6, 12, 24, 48
    A month is 30.4375 days. Accumulation windows are trailing: they end on the current day.
    The paper's model (configuration C5) reads two of the 32: lm_up_ssi3 and lm_up_ssi12.

Standardization
    Every input is standardized per sub-basin and calendar month by a non-parametric rank-to-normal
    transform. A value gets the rank i it would have among the n reference values of the same
    sub-basin and calendar month (mid-rank for ties), the Gringorten plotting position
    p = (i - 0.44)/(n + 0.12), and z = inverse standard normal CDF of p.
    The reference is fixed: the ERA5-Land forcing of 1981-01-01 to 2020-12-31. The normalization
    therefore uses forcing only: no targets, no gauges, no SWOT data and no training samples.
    Values outside the reference range are clipped to the extreme plotting positions
    (|z| up to about 3.3).
    An N-month sum needs N months of history, so it is valid from 1980-01-01 plus N months, and
    the reference uses valid days only. The leaky integrators start from zero on 1980-01-01, so
    the longest one (48 months) carries spin-up into the first reference years. The SWOT Period
    and the predictions from 2010 on are decades past the spin-up.
    Non-finite results are written as 0 (z = 0 is the reference median); their count per input is
    stored in the group attribute `nan_filled_with_zero`.

Running this file as a script
    python build_longmem.py --out DIR [--write-start 2009-01-01] [--write-end <last source day>]
                            [--basins A,B] [--workers 8] [--dtype float16]
    Reads   $SWOTGAT_WORK/CONUS/zarr_store_ext1980 (tp_mean, e_mean, ro_mean from 1980 on),
            CONUS/metadata/graphs_conus.json (first graph version),
            CONUS/attributes/static_conus_storage.parquet (area_km2, the weights), and, unless
            --basins is given, the basin lists CONUS/basin_lists/basin_holdout/{train,test}.csv.
    Writes  DIR/<basin>/ (one consolidated Zarr group per basin, chunks of 365 days x all
            sub-basins, dates --write-start to --write-end) and DIR/manifest.json.
    The script never writes into the source stores (asserted). It can be re-run: the reference is
    fixed, so after the source store is extended, a run with a later --write-end rewrites each
    basin group with identical values on the old dates. A basin that fails is reported as
    `FAILED` in the log and the script still exits normally.

Notes
    rank_normal ranks all sub-basins of a basin in one pass. For values that are equal or nearly
    equal (within about 1e-11) to a reference value, the rank can therefore depend on which other
    sub-basins are in the basin array. See check_c3_ties.py.
"""
from swotgat.paths import WORK
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

C = Path(f"{WORK}/CONUS")
SRC = C/"zarr_store_ext1980"
ATTR = C/"attributes/static_conus_storage.parquet"
GRAPH = C/"metadata/graphs_conus.json"
REF = ("1981-01-01", "2020-12-31")
MONTH = 30.4375
ACC = (3, 6, 12, 24, 36, 48)
LEAK = (6, 12, 24, 48)
VERSION = "longmem_v1"


def feature_names():
    """Return the names of the 32 inputs: 16 local ones (lm_), then 16 upstream ones (lm_up_)."""
    loc = [f"spei{n}" for n in ACC] + [f"ssi{n}" for n in ACC] + [f"lk{m}" for m in LEAK]
    return [f"lm_{x}" for x in loc] + [f"lm_up_{x}" for x in loc]


# ---- numerics (NumPy and SciPy only) --------------------------------------------------------------------------------------
def rolling_sum(x, w):
    """Compute the trailing w-day sum along axis 0 (the window ends on the current row), in float64.

    The first w-1 rows of the result are NaN.
    """
    cs = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0, dtype=np.float64)])
    out = np.full(x.shape, np.nan)
    out[w - 1:] = cs[w:] - cs[:-w]
    return out


def leaky(q, tau_days):
    """Apply a leaky integrator to q along axis 0, starting from zero (a_(-1) = 0).

    a_t = lam * a_(t-1) + q_t with lam = exp(-1/tau_days).
    """
    from scipy.signal import lfilter
    lam = np.exp(-1.0 / tau_days)
    return lfilter([1.0], [1.0, -lam], q, axis=0)


def rank_normal(X, month, ref_rows):
    """Apply the rank-to-normal transform, per column (sub-basin) and calendar month.

    z = Phi^-1((i - 0.44)/(n + 0.12)), where Phi^-1 is the inverse standard normal CDF, n is the
    number of reference values of that column and month, and i is the rank of the value among
    them (mid-rank: i = (number below + number below or equal)/2 + 0.5). The plotting position is
    clipped to those of rank 1 and rank n.
    X (T, n) float64; month (T,) with values 1..12; ref_rows (T,) bool marks the reference rows.
    Returns float32 (T, n). Values that are not finite in X stay NaN. A column that is not finite
    on every reference row is NaN everywhere.
    Vectorized: the sorted reference block of each column is shifted by a column offset (column
    index x the value span of X), so that one searchsorted call serves all columns. The offset
    limits the resolution of the comparison, so values within about 1e-11 of a reference value
    can rank as ties (see check_c3_ties.py).
    """
    from scipy.special import ndtri
    T, n = X.shape
    Z = np.full((T, n), np.nan, np.float32)
    finite = np.isfinite(X)
    good_cols = finite[ref_rows].all(axis=0) if ref_rows.any() else np.zeros(n, bool)
    lo, hi = np.nanmin(np.where(finite, X, np.nan)), np.nanmax(np.where(finite, X, np.nan))
    span = (hi - lo) + 1.0
    off = (np.arange(n, dtype=np.float64) * span)[None, :]
    for m in range(1, 13):
        rows = np.where(month == m)[0]
        rref = np.where((month == m) & ref_rows)[0]
        nr = len(rref)
        if nr == 0: continue
        R = np.sort(X[rref][:, good_cols], axis=0)
        k = good_cols.sum()
        flat = (R - lo + off[:, :k]).T.ravel()                   # column blocks, each sorted, blocks increasing
        Q = X[rows][:, good_cols]
        qf = (np.where(np.isfinite(Q), Q, lo) - lo + off[:, :k])
        base = (np.arange(k) * nr)[None, :]
        left = np.searchsorted(flat, qf.ravel(), "left").reshape(qf.shape) - base
        right = np.searchsorted(flat, qf.ravel(), "right").reshape(qf.shape) - base
        i = 0.5 * (left + right) + 0.5
        p = np.clip((i - 0.44) / (nr + 0.12), (1 - 0.44) / (nr + 0.12), (nr - 0.44) / (nr + 0.12))
        z = ndtri(p).astype(np.float32)
        z[~np.isfinite(Q)] = np.nan
        Z[np.ix_(rows, np.where(good_cols)[0])] = z
    return Z


def month_anomaly(X, month, ref_rows):
    """Compute the standardized anomaly of each column of X by calendar month.

    The anomaly is (X - mean) / standard deviation, where mean and standard deviation are those
    of the same calendar month over the reference rows. Where the standard deviation is 0 the
    result is 0. month (T,) holds the calendar month of each row; ref_rows (T,) bool marks the
    reference rows.
    """
    out = np.zeros_like(X, dtype=np.float64)
    for m in range(1, 13):
        rows = month == m; r = rows & ref_rows
        mu = np.nanmean(X[r], axis=0); sd = np.nanstd(X[r], axis=0)
        out[rows] = np.where(sd > 0, (X[rows] - mu) / np.where(sd > 0, sd, 1.0), 0.0)
    return out


def upstream_matrix(G, nodes, area):
    """Build the sparse (n, n) matrix that takes the area-weighted upstream mean of every node.

    Row i takes the area-weighted mean over node i and all its ancestors in the directed graph G
    (edges point downstream). Each row sums to 1.
    nodes: node ids in the column order of the data; area: (n,) weights > 0. Ancestors that are
    not in `nodes` are ignored.
    """
    import networkx as nx, scipy.sparse as sp
    idx = {v: i for i, v in enumerate(nodes)}
    rows, cols = [], []
    for j, v in enumerate(nodes):                                # v contributes to itself and every descendant
        rows.append(j); cols.append(j)
        if v in G:
            for d in nx.descendants(G, v):
                if d in idx: rows.append(idx[d]); cols.append(j)
    rows = np.asarray(rows); cols = np.asarray(cols)
    M = sp.csr_matrix((area[cols].astype(np.float64), (rows, cols)), shape=(len(nodes), len(nodes)))
    rs = np.asarray(M.sum(axis=1)).ravel()
    return sp.diags(1.0 / rs) @ M


def basin_features(pe, ro, dates, Mup, ref=REF, keep=None):
    """Compute all 32 inputs of one basin, as a dict {name: float32 (rows kept, n)}.

    pe, ro (T, n) float64: daily P-E and runoff of the n sub-basins; dates: DatetimeIndex of the
    T days; Mup: sparse (n, n) upstream-mean matrix from upstream_matrix; ref: first and last day
    of the reference period. keep (T,) bool: the rows to return. The computation and the
    reference always use the full record.
    """
    sl = (lambda z: z[keep]) if keep is not None else (lambda z: z)
    month = dates.month.values
    ref_rows = (dates >= pd.Timestamp(ref[0])) & (dates <= pd.Timestamp(ref[1]))
    out = {}
    for pre, P, R in (("lm_", pe, ro), ("lm_up_", (Mup @ pe.T).T, (Mup @ ro.T).T)):
        for n in ACC:
            w = int(round(n * MONTH))
            valid = np.arange(len(dates)) >= w - 1
            out[f"{pre}spei{n}"] = sl(rank_normal(rolling_sum(P, w), month, ref_rows & valid))
            out[f"{pre}ssi{n}"] = sl(rank_normal(rolling_sum(R, w), month, ref_rows & valid))
        q = month_anomaly(R, month, ref_rows)
        for m in LEAK:
            out[f"{pre}lk{m}"] = sl(rank_normal(leaky(q, m * MONTH), month, ref_rows))
    return out


# ---- per-basin driver -----------------------------------------------------------------------------------------------------
def _graph():
    """Read the graph file as a networkx DiGraph."""
    import networkx as nx
    return nx.readwrite.json_graph.node_link_graph(json.load(open(GRAPH)), edges="edges")


def run_basin(basin, out, w0, w1, dtype, G=None):
    """Build and write the group of one basin for the dates w0..w1.

    out is the output folder, dtype the stored data type and G the basin graph (read from the
    graph file if not given). Sub-basins with a missing or non-positive area get the median area
    of the basin as weight. NaN results are written as 0 and counted in the group attributes.
    Returns (basin, number of sub-basins, number of non-zeros of the upstream matrix, number of
    sub-basins whose P-E is non-finite on every day, total NaN count before filling, seconds).
    """
    import xarray as xr, networkx as nx
    t0 = time.time()
    G = G or _graph()
    ds = xr.open_zarr(SRC/basin, consolidated=False)[["tp_mean", "e_mean", "ro_mean"]].transpose("date", "subbasin")
    dates = pd.DatetimeIndex(ds.date.values); subs = [str(s) for s in ds.subbasin.values]
    tp, e, ro = (ds[v].values.astype(np.float64) for v in ("tp_mean", "e_mean", "ro_mean"))
    pe = tp + e
    comp = next(c for c in nx.weakly_connected_components(G) if basin in c)
    sub = G.subgraph(comp); sub = nx.relabel_nodes(sub, {v: str(v) for v in sub.nodes})
    a = pd.read_parquet(ATTR, columns=["area_km2"])
    a = a[a.index.get_level_values("basin") == basin].droplevel("basin")["area_km2"]; a.index = a.index.astype(str)
    area = a.reindex(subs).values.astype(np.float64)
    bad = ~np.isfinite(area) | (area <= 0)
    if bad.any(): area[bad] = np.nanmedian(area[~bad]) if (~bad).any() else 1.0
    Mup = upstream_matrix(sub, subs, area)
    nan_cols = int((~np.isfinite(pe)).all(axis=0).sum())
    wsel = (dates >= pd.Timestamp(w0)) & (dates <= pd.Timestamp(w1))
    F = basin_features(pe, ro, dates, Mup, keep=wsel)
    del tp, e, ro, pe
    n_nan = {k: int((~np.isfinite(v)).sum()) for k, v in F.items()}
    data = {k: (("date", "subbasin"), np.nan_to_num(v, nan=0.0).astype(dtype)) for k, v in F.items()}
    xo = xr.Dataset(data, coords={"date": dates[wsel].values, "subbasin": ds.subbasin.values})
    xo.attrs = dict(version=VERSION, reference=f"{REF[0]}..{REF[1]} ERA5-Land forcing (zarr_store_ext1980)",
                    pe="tp_mean + e_mean (ERA5 e < 0 for evaporation)", acc_months=list(ACC), leak_months=list(LEAK),
                    month_days=MONTH, transform="per subbasin x calendar month rank->normal, Gringorten",
                    upstream="area_km2-weighted mean over {node} U ancestors in graphs_conus.json",
                    nan_filled_with_zero=json.dumps(n_nan), built=time.strftime("%Y-%m-%d %H:%M"))
    enc = {k: {"chunks": (365, len(subs))} for k in data}
    xo.to_zarr(out/basin, mode="w", consolidated=True, encoding=enc)
    return basin, len(subs), int(Mup.nnz), nan_cols, sum(n_nan.values()), round(time.time() - t0)


def main():
    """Parse the command line and build the store, one basin per task in a pool of processes.

    Writes manifest.json to the output folder, then the basin groups, starting with the basins
    that have the most sub-basins. A basin that fails is printed as FAILED; the function still
    returns normally.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--write-start", default="2009-01-01")
    ap.add_argument("--write-end", default=None); ap.add_argument("--basins", default=None)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--dtype", default="float16")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    assert not str(out.resolve()).startswith(str(C/"zarr_store")), "never write into the source stores"
    import xarray as xr
    w1 = a.write_end or str(pd.Timestamp(xr.open_zarr(SRC/"USGS-01034500", consolidated=False).date.values[-1]).date())
    basins = a.basins.split(",") if a.basins else sorted(
        set(pd.concat([pd.read_csv(C/f"basin_lists/basin_holdout/{s}.csv", dtype=str) for s in ("train", "test")]).iloc[:, 0]))
    sizes = pd.read_parquet(ATTR, columns=["area_km2"]).reset_index().groupby("basin").size()
    basins = sorted(basins, key=lambda b: -sizes.get(b, 0))
    print(f"{VERSION}: {len(basins)} basins, write {a.write_start}..{w1} as {a.dtype}, features {feature_names()}", flush=True)
    json.dump(dict(version=VERSION, features=feature_names(), write=[a.write_start, w1], reference=REF, dtype=a.dtype,
                   source=str(SRC)), open(out/"manifest.json", "w"), indent=1)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_basin, b, out, a.write_start, w1, a.dtype): b for b in basins}
        for f in as_completed(futs):
            try: print("done", *f.result(), flush=True)
            except Exception as e: print("FAILED", futs[f], type(e).__name__, e, flush=True)


if __name__ == "__main__":
    main()
