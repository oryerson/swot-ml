"""Check the long-memory store longmem_v35; write READY.json only if every check passes.

The store is compared with longmem_vd2026, the same 32 inputs built earlier on the first graph
version and its forcing stores. Where neither a node's polygon nor its upstream set changed, the
two stores must agree.

Reads (paths relative to $SWOTGAT_WORK)
    CONUS_storage_fork/features/longmem_v35     the store to check, with manifest.json and
                                                build_log.json (build_longmem_v35.py)
    CONUS_storage_fork/features/longmem_vd2026  the earlier long-memory store (first graph version)
    --src1 DIR, --src2 DIR                      the forcing stores the builder was given
    graph v3.5 and its sub-basin table (paths from build_longmem_v35.py);
    CONUS/metadata/graphs_conus.json and CONUS/metadata/subbasins_conus.parquet (first graph
    version)
    CONUS_storage_fork/eval/v35_inputs/forcing/zarr_store_vd2026_halfexact_v35   training store,
                                                for its node sets
    CONUS_storage_fork/features/longmem_v35/check_c3_ties.json   result of check_c3_ties.py, if
                                                present

Writes (in CONUS_storage_fork/features/longmem_v35)
    check_basins.csv   per basin: counts of missing nodes, non-finite values, differences
    check_spot.csv     check c4, per node and input
    READY.json         only if all checks pass: the manifest facts, md5 values and all results

Usage
    python check_longmem_v35.py --src1 DIR --src2 DIR [--jobs TEXT] [--workers 8]
    --jobs is free text copied into READY.json (for example the ids of the build jobs).
    The exit code is 1 if any check fails. If c3 fails on its strict test, run check_c3_ties.py
    with the same --src1 and --src2 and then this script again.

Checks
    c1  Manifest: same keys as the manifest of longmem_vd2026, and the same input names, written
        dates, reference period and dtype.
    c2  Finite inputs over 2022-01-01 to 2026-09-30 at every graph v3.5 node: every graph node is
        a column of its basin group; build_log.json has every basin, without error, with no NaN
        before the zero fill in that window and no forcing column with NaN there; the store has no
        non-finite value there and no (node, input) series that is zero on every day of the
        window (the sign of a filled NaN).
    c3  Unchanged nodes. A node is unchanged if it has the same id, basin and polygon (equal WKB
        bytes) in both graph versions, the same set of ancestors in both graphs, and every
        ancestor's polygon is unchanged too. At these nodes all 32 inputs on every written day
        must equal longmem_vd2026 within float16 rounding: |difference| <= 1 float16 ulp (unit in
        the last place) of the value. The same test is applied to the 16 local inputs at every
        node whose own polygon is unchanged. If any value is beyond 1 ulp, c3 passes only if
        check_c3_ties.py, run on the same check_basins.csv and manifest.json (md5 compared), shows
        that every such value has bitwise-identical forcing and lies within the rank band of its
        tied reference values.
    c4  Spot check of changed nodes: 4 random nodes (seed 0) of each kind (new id; polygon
        changed; upstream set changed), plus the changed node with the largest upstream area. For
        each, lm_spei12, lm_ssi3, lm_lk48 and their lm_up_ versions are recomputed independently
        from the v3.5 forcing and the ancestors in the v3.5 graph (area-weighted mean, then the
        build_longmem.py numerics on a single column). They must equal the store within 1 float16
        ulp.
    c5  Requirements of the training step, first set for an earlier model configuration (C2) that
        read lm_up_lk6: lm_up_lk6 is finite at every graph node on every day from 2022-01-01 to
        2026-09-30, with all days present; the node set of each basin group equals that of the
        training store; the written dates are daily and cover 2009-07-05 to 2026-09-30.
"""
from swotgat.paths import WORK
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr, networkx as nx
sys.path.insert(0, str(Path(__file__).resolve().parent)); import build_longmem_v35 as L
B = L.B; C = Path(f"{WORK}/CONUS"); OUT = L.F/"features/longmem_v35"; VD = L.F/"features/longmem_vd2026"
TR = L.F/"eval/v35_inputs/forcing/zarr_store_vd2026_halfexact_v35"; GATE = ("2009-07-05", "2026-09-30"); GATE_IN = "lm_up_lk6"
ap = argparse.ArgumentParser(); ap.add_argument("--src1", required=True); ap.add_argument("--src2", required=True); ap.add_argument("--jobs", default="")
ap.add_argument("--workers", type=int, default=8); a = ap.parse_args(); SRC1, SRC2 = Path(a.src1), Path(a.src2)
def md5(p):
    """Return the md5 hex digest of the file p."""
    return hashlib.md5(open(p, "rb").read()).hexdigest()
def ulp(x):
    """Return the spacing between |x| and the next larger float16 value (the store is float16)."""
    return np.spacing(np.abs(x).astype(np.float16)).astype(np.float32)
R = {}

m35, mvd = json.load(open(OUT/"manifest.json")), json.load(open(VD/"manifest.json"))
R["c1_manifest"] = dict(ok=bool(set(m35) == set(mvd) and m35["version"] == "longmem_v35" and all(m35[k] == mvd[k] for k in ("features", "write", "reference", "dtype"))),
                        manifest_md5=md5(OUT/"manifest.json"), source=m35["source"])
print("c1", R["c1_manifest"], flush=True)

G35 = L.graph(); G1 = nx.relabel_nodes(B._graph(), lambda v: str(v))
s35 = pd.read_parquet(L.SUBS, columns=["geometry", "area_km2", "uparea_km2", "outlet_id"]); s35.index = s35.index.astype(str)
s1 = pd.read_parquet(C/"metadata/subbasins_conus.parquet", columns=["geometry", "area_km2", "outlet_id"]); s1.index = s1.index.astype(str)
gn = sorted(G35.nodes); NB = s35.loc[gn, "outlet_id"]; basins = sorted(set(NB)); BYB = {b: list(g.index) for b, g in NB.groupby(NB)}
log = {r["basin"]: r for r in json.load(open(OUT/"build_log.json"))}
common = s35.index.intersection(s1.index)
poly_same = pd.Series(False, index=s35.index)
cc = common[(s35.loc[common, "outlet_id"].values == s1.loc[common, "outlet_id"].values)]
poly_same.loc[cc] = s35.loc[cc, "geometry"].values == s1.loc[cc, "geometry"].values
anc35 = {v: nx.ancestors(G35, v) for v in gn}
UNCH = [v for v in gn if poly_same.get(v, False) and v in G1 and anc35[v] == nx.ancestors(G1, v) and all(poly_same.get(u, False) for u in anc35[v])]
LOCAL = [v for v in gn if poly_same.get(v, False)]
print(f"{len(gn)} graph nodes in {len(basins)} basins; polygon unchanged {len(LOCAL)}; polygon + upstream set unchanged {len(UNCH)}", flush=True)

CW = (pd.Timestamp(L.CHK[0]), pd.Timestamp(L.CHK[1])); NAMES = m35["features"]; LOC = [c for c in NAMES if not c.startswith("lm_up_")]
uset, lset = set(UNCH), set(LOCAL)
def basin_checks(b):
    """Compute all counts of checks c2, c3 and c5 for basin b (one row of check_basins.csv)."""
    x = xr.open_zarr(OUT/b, consolidated=True); subs = x.subbasin.values.astype(str); nodes = BYB[b]
    miss = sorted(set(nodes) - set(subs)); col = {s: i for i, s in enumerate(subs)}; gi = [col[v] for v in nodes if v in col]
    d = pd.DatetimeIndex(x.date.values); w = (d >= CW[0]) & (d <= CW[1]); r = dict(basin=b, nodes=len(nodes), missing=len(miss), nonfinite=0, all_zero=0)
    y = xr.open_zarr(VD/b, consolidated=True) if (VD/b).exists() else None
    vcol = {s: i for i, s in enumerate(y.subbasin.values.astype(str))} if y is not None else {}
    if y is not None: assert (y.date.values == x.date.values).all()
    ui = [(col[v], vcol[v]) for v in nodes if v in uset and v in vcol and v in col]; li = [(col[v], vcol[v]) for v in nodes if v in lset and v in vcol and v in col]
    r.update(unch_nodes=len(ui), unch_values=0, unch_beyond_ulp=0, unch_max=0.0, local_nodes=len(li), local_values=0, local_beyond_ulp=0, local_max=0.0)
    tsubs = xr.open_zarr(TR/b, consolidated=False).subbasin.values.astype(str)
    r.update(node_set_eq_training=bool(set(tsubs) == set(subs)), node_order_eq_training=bool(list(tsubs) == list(subs)),
             range_ok=bool(d[0] <= pd.Timestamp(GATE[0]) and d[-1] >= pd.Timestamp(GATE[1]) and (np.diff(d.values).astype("timedelta64[D]").astype(int) == 1).all()),
             window_days=int(w.sum()), gate_nonfinite=int((~np.isfinite(x[GATE_IN].values.astype(np.float32)[w][:, gi])).sum()))
    for v in NAMES:
        X = x[v].values.astype(np.float32)
        Xw = X[w][:, gi]; r["nonfinite"] += int((~np.isfinite(Xw)).sum()); r["all_zero"] += int((Xw == 0).all(axis=0).sum())
        if y is None: continue
        Y = y[v].values.astype(np.float32)
        for key, pairs in (("unch", ui), ("local", li if v in LOC else [])):
            if not pairs: continue
            xa = X[:, [p[0] for p in pairs]]; ya = Y[:, [p[1] for p in pairs]]; dd = np.abs(xa - ya)
            r[f"{key}_values"] += dd.size; r[f"{key}_beyond_ulp"] += int((dd > ulp(ya)).sum()); r[f"{key}_max"] = max(r[f"{key}_max"], float(dd.max()))
    return r
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=a.workers) as ex: BR = pd.DataFrame(list(ex.map(basin_checks, basins)))
BR.to_csv(OUT/"check_basins.csv", index=False)
nan_chk = sum(sum(log[b]["n_nan_chk"].values()) for b in basins if b in log); nan_forc = sum(log[b]["nan_forcing_cols_chk"] for b in basins if b in log)
R["c2_finite"] = dict(ok=bool(len(log) == len(basins) and not any("error" in log[b] for b in basins) and BR.missing.sum() == 0 and BR.nonfinite.sum() == 0
                              and BR.all_zero.sum() == 0 and nan_chk == 0 and nan_forc == 0),
                      window=list(L.CHK), graph_nodes=len(gn), basins=len(basins), missing_nodes=int(BR.missing.sum()), nonfinite=int(BR.nonfinite.sum()),
                      all_zero_series=int(BR.all_zero.sum()), nan_before_fill=int(nan_chk), forcing_nan_cols=int(nan_forc))
strict = bool(BR.unch_beyond_ulp.sum() == 0 and BR.local_beyond_ulp.sum() == 0); TJ = OUT/"check_c3_ties.json"
ties = json.load(open(TJ)) if TJ.exists() else None   # check_c3_ties.py: every value beyond 1 ulp must be explained by tied ranks
ties_ok = bool(ties and ties["ok"] and ties["check_basins_md5"] == md5(OUT/"check_basins.csv") and ties["manifest_md5"] == md5(OUT/"manifest.json"))
R["c3_unchanged"] = dict(ok=bool(BR.unch_nodes.sum() == len(UNCH) and (strict or ties_ok)), strict_1ulp=strict, tie_test=ties,
                         nodes=int(BR.unch_nodes.sum()), values=int(BR.unch_values.sum()),
                         beyond_1ulp=int(BR.unch_beyond_ulp.sum()), max_abs_diff=float(BR.unch_max.max()),
                         local_inputs_at_polygon_unchanged=dict(nodes=int(BR.local_nodes.sum()), values=int(BR.local_values.sum()),
                                                                beyond_1ulp=int(BR.local_beyond_ulp.sum()), max_abs_diff=float(BR.local_max.max())))
print("c2", R["c2_finite"], "\nc3", R["c3_unchanged"], flush=True)

rng = np.random.default_rng(0); new = [v for v in gn if v not in s1.index]; pchg = [v for v in gn if v in s1.index and not poly_same.get(v, False)]
uchg = [v for v in gn if poly_same.get(v, False) and v not in uset]
pick = []
for grp in (new, pchg, uchg):
    pick += list(rng.choice(sorted(grp), size=min(4, len(grp)), replace=False)) if grp else []
chg = [v for v in gn if v not in uset]; pick.append(max(chg, key=lambda v: (float(s35.loc[v, "uparea_km2"]), v))); pick = list(dict.fromkeys(pick))
SPOT = ["spei12", "ssi3", "lk48"]; rows = []
for v in pick:
    b = s35.loc[v, "outlet_id"]; ids = [v] + sorted(anc35[v]); vs = ["tp_mean", "e_mean", "ro_mean"]
    d1 = xr.open_zarr(SRC1/b, consolidated=False)[vs].sel(date=slice(None, "2025-12-31"), subbasin=ids)
    d2 = xr.open_zarr(SRC2/b, consolidated=False)[vs].sel(date=slice("2026-01-01", m35["write"][1]), subbasin=ids)
    ds = xr.concat([d1, d2], dim="date").transpose("date", "subbasin"); dates = pd.DatetimeIndex(ds.date.values)
    pe = (ds.tp_mean.values + ds.e_mean.values).astype(np.float64); ro = ds.ro_mean.values.astype(np.float64)
    ar = s35.loc[ids, "area_km2"].values.astype(np.float64)
    loc_pe, loc_ro = pe[:, :1], ro[:, :1]; up_pe = (pe @ ar / ar.sum())[:, None]; up_ro = (ro @ ar / ar.sum())[:, None]
    mon = dates.month.values; ref = (dates >= pd.Timestamp(B.REF[0])) & (dates <= pd.Timestamp(B.REF[1])); keep = (dates >= pd.Timestamp(m35["write"][0])) & (dates <= pd.Timestamp(m35["write"][1]))
    st = xr.open_zarr(OUT/b, consolidated=True).sel(subbasin=v); vd = xr.open_zarr(VD/b, consolidated=True) if (VD/b).exists() else None
    for pre, P_, R_ in (("lm_", loc_pe, loc_ro), ("lm_up_", up_pe, up_ro)):
        for f in SPOT:
            if f.startswith("spei"): n = int(f[4:]); wd = int(round(n*B.MONTH)); z = B.rank_normal(B.rolling_sum(P_, wd), mon, ref & (np.arange(len(dates)) >= wd - 1))
            elif f.startswith("ssi"): n = int(f[3:]); wd = int(round(n*B.MONTH)); z = B.rank_normal(B.rolling_sum(R_, wd), mon, ref & (np.arange(len(dates)) >= wd - 1))
            else: m = int(f[2:]); z = B.rank_normal(B.leaky(B.month_anomaly(R_, mon, ref), m*B.MONTH), mon, ref)
            z = np.nan_to_num(z[keep, 0], nan=0.0).astype(np.float16).astype(np.float32); s = st[pre + f].values.astype(np.float32); dd = np.abs(z - s)
            old = vd[pre + f].sel(subbasin=v).values.astype(np.float32) if vd is not None and v in set(vd.subbasin.values.astype(str)) else None
            rows.append(dict(node=v, basin=b, kind="new" if v in new else "polygon changed" if v in pchg else "upstream changed", n_ancestors=len(anc35[v]),
                             input=pre + f, max_abs_diff=float(dd.max()), beyond_1ulp=int((dd > ulp(s)).sum()),
                             mean_abs_change_vs_vd2026=float(np.abs(s - old).mean()) if old is not None else np.nan))
SP = pd.DataFrame(rows); SP.to_csv(OUT/"check_spot.csv", index=False)
R["c4_spot"] = dict(ok=bool(len(SP) and SP.beyond_1ulp.sum() == 0), nodes=pick, kinds={k: int(n) for k, n in SP.groupby("kind").node.nunique().items()}, values_checked=int(len(SP)*int(keep.sum())),
                    beyond_1ulp=int(SP.beyond_1ulp.sum()), max_abs_diff=float(SP.max_abs_diff.max()))
print("c4", R["c4_spot"]); print(SP.round(4).to_string(index=False), flush=True)
ndays = (CW[1] - CW[0]).days + 1; tr_groups = sorted(p.name for p in TR.iterdir() if (p/"zarr.json").is_file())
R["c5_c2_gate"] = dict(ok=bool(BR.gate_nonfinite.sum() == 0 and (BR.window_days == ndays).all() and BR.node_set_eq_training.all() and BR.range_ok.all()),
                       input=GATE_IN, window=list(L.CHK), days_expected=ndays, gate_input_nonfinite=int(BR.gate_nonfinite.sum()),
                       training_store=str(TR), basins_node_set_equal=int(BR.node_set_eq_training.sum()), basins_node_order_equal=int(BR.node_order_eq_training.sum()),
                       training_groups_not_built=[g for g in tr_groups if g not in set(basins)], range_required=list(GATE), basins_range_ok=int(BR.range_ok.sum()))
print("c5", R["c5_c2_gate"], flush=True)
ok = all(v["ok"] for v in R.values())
if ok:
    json.dump(dict(ready=True, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), store=str(OUT), manifest_md5=R["c1_manifest"]["manifest_md5"],
                   features=m35["features"], write=m35["write"], reference=m35["reference"], dtype=m35["dtype"], source=m35["source"],
                   graph=str(L.GRAPH), graph_md5=L.MD5[L.GRAPH], subbasins_md5=L.MD5[L.SUBS], build_jobs=a.jobs, checks=R),
              open(OUT/"READY.json", "w"), indent=1)
print("ALL CHECKS PASS -> READY.json" if ok else "CHECKS FAILED: READY.json not written", flush=True); sys.exit(0 if ok else 1)
