"""Tie test for check c3 of check_longmem_v35.py: explain the values that differ by more than 1 ulp.

Background
    Check c3 requires that, at nodes whose polygon and upstream set did not change, the store
    longmem_v35 equals the earlier store longmem_vd2026 within 1 float16 ulp (unit in the last
    place). This strict test cannot hold at tied ranks. build_longmem.rank_normal ranks all
    columns of a basin with one searchsorted call, after adding a column offset j * span to
    column j (span comes from the minimum and maximum over the whole basin). Values that are equal
    to a reference value, or within about 1e-11 of it, can therefore rank as a tie or not,
    depending on which other nodes are in the basin. The basins of the two graph versions do not
    hold the same nodes, so a few such values differ.

Reads (paths relative to $SWOTGAT_WORK)
    CONUS_storage_fork/features/longmem_v35     with check_basins.csv (written by
                                                check_longmem_v35.py) and manifest.json
    CONUS_storage_fork/features/longmem_vd2026  the earlier long-memory store
    --src1 DIR, --src2 DIR                      the graph v3.5 forcing stores the builder was given
    CONUS/zarr_store_ext1980 (to 2025-12-31) and CONUS/zarr_store_vd2026 (2026)   the forcing
                                                stores that longmem_vd2026 was built from
    graph and sub-basin tables of both graph versions

Writes (in CONUS_storage_fork/features/longmem_v35; nothing else is modified)
    check_c3_ties.csv    one row per value beyond 1 ulp
    check_c3_ties.json   summary; `ok` is true if every such value is explained

Usage
    python check_c3_ties.py --src1 DIR --src2 DIR
    Run it after check_longmem_v35.py has written check_basins.csv and reported values beyond
    1 ulp, then run check_longmem_v35.py again. The script expects at least one such value.
    The exit code is 0 only if `ok`.

Method
    The script takes every value beyond 1 ulp, in the basins that check_basins.csv flags, and
    accepts it only if both conditions hold:
    (a) Identical inputs. The forcing of the node (and of every ancestor, for lm_up_* inputs) is
        bitwise equal in the v3.5 stores and in the earlier stores, on all days.
    (b) Bounded by the tie. Take the cluster of reference values within DELTA = 1e-10 of the
        ranked value (same calendar month, 1981-2020, valid days). The cluster must hold at least
        one value other than the day itself, and
            |v35 - vd2026| <= z(highest mid-rank the cluster allows) - z(lowest) + 1 float16 ulp
        with the Gringorten plotting positions of rank_normal (a further 1e-6 is allowed for
        rounding).
    Values within 1 ulp are not examined; for them the strict test stands. The leaky-integrator
    inputs (lk*) are not explained by rank ties: any lk difference fails.
"""
from swotgat.paths import WORK
import argparse, hashlib, json, sys, numpy as np, pandas as pd, xarray as xr, networkx as nx
from pathlib import Path
from scipy.special import ndtri
sys.path.insert(0, str(Path(__file__).resolve().parent)); import build_longmem_v35 as L
B = L.B; C = Path(f"{WORK}/CONUS"); OUT = L.F/"features/longmem_v35"; VD = L.F/"features/longmem_vd2026"; DELTA = 1e-10
ap = argparse.ArgumentParser(); ap.add_argument("--src1", required=True); ap.add_argument("--src2", required=True); a = ap.parse_args()
SRC35, SRCPR = (Path(a.src1), Path(a.src2)), (C/"zarr_store_ext1980", C/"zarr_store_vd2026")
def ulp(x):
    """Return the spacing between |x| and the next larger float16 value, as float32."""
    return np.spacing(np.abs(x).astype(np.float16)).astype(np.float32)
BR = pd.read_csv(OUT/"check_basins.csv"); bad = BR[(BR.unch_beyond_ulp > 0) | (BR.local_beyond_ulp > 0)].basin.tolist()
G35 = L.graph(); G1 = nx.relabel_nodes(B._graph(), lambda v: str(v))
s35 = pd.read_parquet(L.SUBS, columns=["geometry", "area_km2", "outlet_id"]); s35.index = s35.index.astype(str)
s1 = pd.read_parquet(C/"metadata/subbasins_conus.parquet", columns=["geometry", "outlet_id"]); s1.index = s1.index.astype(str)
cm = s35.index.intersection(s1.index); cm = cm[s35.loc[cm, "outlet_id"].values == s1.loc[cm, "outlet_id"].values]
ps = pd.Series(False, index=s35.index); ps.loc[cm] = s35.loc[cm, "geometry"].values == s1.loc[cm, "geometry"].values
NB = s35.loc[list(G35.nodes), "outlet_id"]; vs = ["tp_mean", "e_mean", "ro_mean"]
def forcing(stores, b, ids):
    """Return the 1980-2026 forcing of the nodes ids of basin b, joined from a pair of stores.

    The first store gives the days up to 2025-12-31 and the second store those from 2026-01-01
    to 2026-09-30. The result is a Dataset of tp_mean, e_mean and ro_mean on (date, subbasin).
    """
    d1 = xr.open_zarr(stores[0]/b, consolidated=False)[vs].sel(date=slice(None, "2025-12-31"), subbasin=ids)
    d2 = xr.open_zarr(stores[1]/b, consolidated=False)[vs].sel(date=slice("2026-01-01", "2026-09-30"), subbasin=ids)
    return xr.concat([d1, d2], dim="date").transpose("date", "subbasin")
rows = []
for b in bad:
    nodes = sorted(v for v in NB[NB == b].index if ps.get(v, False)); anc = {v: nx.ancestors(G35, v) for v in nodes}
    unch = {v for v in nodes if v in G1 and anc[v] == nx.ancestors(G1, v) and all(ps.get(u, False) for u in anc[v])}
    x, y = xr.open_zarr(OUT/b, consolidated=True), xr.open_zarr(VD/b, consolidated=True); d = pd.DatetimeIndex(x.date.values)
    diffs = []
    for f in x.data_vars:
        test = sorted(unch) if f.startswith("lm_up_") else nodes
        X = x[f].sel(subbasin=test).values.astype(np.float32); Y = y[f].sel(subbasin=test).values.astype(np.float32)
        r, c = np.where(np.abs(X - Y) > ulp(Y)); diffs += [(test[j], f, i, X[i, j], Y[i, j]) for i, j in zip(r, c)]
    if not diffs: continue
    need = sorted({v for v, f, *_ in diffs} | {u for v, f, *_ in diffs if f.startswith("lm_up_") for u in anc[v]})
    A35, APR = forcing(SRC35, b, need).load(), forcing(SRCPR, b, need).load(); assert (A35.date.values == APR.date.values).all()
    eqc = np.ones(len(need), bool)
    for k in vs: u, w0 = A35[k].values, APR[k].values; eqc &= ((u == w0) | (np.isnan(u) & np.isnan(w0))).all(axis=0)
    eq = dict(zip(need, eqc))
    fd = pd.DatetimeIndex(A35.date.values); col = {s: i for i, s in enumerate(need)}; mon = fd.month.values
    ref = (fd >= pd.Timestamp(B.REF[0])) & (fd <= pd.Timestamp(B.REF[1]))
    PE = (A35.tp_mean.values + A35.e_mean.values).astype(np.float64); RO = A35.ro_mean.values.astype(np.float64)
    ar = s35.loc[need, "area_km2"].values.astype(np.float64); cache = {}
    for v, f, i, zv, zy in diffs:
        up = f.startswith("lm_up_"); kind = f.replace("lm_up_", "").replace("lm_", ""); ids = [v] + (sorted(anc[v]) if up else [])
        rec = dict(basin=b, node=v, input=f, date=d[i], v35=float(zv), vd2026=float(zy), diff=float(zv - zy), inputs_equal=all(eq[s] for s in ids))
        if kind.startswith("lk"): rows.append(dict(rec, tie_ok=False, band=np.nan, cluster=0, why="leaky integrator")); continue
        key = (v, up, kind)
        if key not in cache:
            k = [col[s] for s in ids]; w_ = ar[k]/ar[k].sum()
            ser = (PE if kind.startswith("spei") else RO)[:, k] @ w_
            nm = int(kind[4:] if kind.startswith("spei") else kind[3:]); wd = int(round(nm*B.MONTH))
            cache[key] = (B.rolling_sum(ser[:, None], wd)[:, 0], ref & (np.arange(len(fd)) >= wd - 1))
        Xr, rr = cache[key]; t = fd.get_loc(d[i]); q = Xr[t]; R = Xr[(mon == d[i].month) & rr]; n = len(R)
        lo = int((R < q - DELTA).sum()); hi = int((R <= q + DELTA).sum()); own = 1 if rr[t] else 0
        imin, imax = lo + 0.5 + 0.5*own, hi + 0.5 - 0.5*own
        p = lambda i_: np.clip((i_ - 0.44)/(n + 0.12), (1 - 0.44)/(n + 0.12), (n - 0.44)/(n + 0.12))
        band = float(ndtri(p(imax)) - ndtri(p(imin))); gap = np.abs(R[np.abs(R - q) > 0] - q).min() if (np.abs(R - q) > 0).any() else np.inf
        tol = band + float(ulp(np.float32(max(abs(zv), abs(zy))))) + 1e-6
        rows.append(dict(rec, cluster=hi - lo, exact_ties=int((R == q).sum()) - own, nearest_other=float(gap), band=band,
                         tie_ok=bool(rec["inputs_equal"] and hi - lo - own >= 1 and abs(zv - zy) <= tol), why=""))
    print(b, len(diffs), "differences;", sum(r["tie_ok"] for r in rows if r["basin"] == b), "tie-bounded", flush=True)
T = pd.DataFrame(rows); T.to_csv(OUT/"check_c3_ties.csv", index=False)
fail = T[~T.tie_ok]
md5 = lambda q: hashlib.md5(open(q, 'rb').read()).hexdigest()
S = dict(check_basins_md5=md5(OUT/'check_basins.csv'), manifest_md5=md5(OUT/'manifest.json'), delta=DELTA, values_beyond_1ulp=len(T), inputs_equal=int(T.inputs_equal.sum()), tie_bounded=int(T.tie_ok.sum()), not_explained=len(fail),
         by_input={k: int(v) for k, v in T.input.value_counts().items()}, in_2022_2026=int((T.date >= "2022-01-01").sum()),
         abs_diff_max=float(T["diff"].abs().max()), abs_diff_median=float(T["diff"].abs().median()), ok=bool(len(fail) == 0 and len(T) > 0))
json.dump(S, open(OUT/"check_c3_ties.json", "w"), indent=1); print(json.dumps(S), flush=True)
if len(fail): print(fail.head(30).to_string(index=False), flush=True)
sys.exit(0 if S["ok"] else 1)
