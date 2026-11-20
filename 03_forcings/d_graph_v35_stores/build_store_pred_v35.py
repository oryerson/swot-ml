"""Build one basin group of a graph v3.5 prediction store (the stores the model predicts from).

There are two prediction stores. Each is assembled from its "template": the store of the same
period on the earlier node set, which is only read. The method is that of the training store
(build_store_v35.py).

    store     template (read)             output                          dates
    ext1980   CONUS/zarr_store_ext1980    <forcing>/zarr_store_ext1980_v35   1980-01-01 to 2025-12-31
    vd2026    CONUS/zarr_store_vd2026     <forcing>/zarr_store_vd2026_v35    2020-01-01 to 2026-09-30
    (paths relative to $SWOTGAT_WORK; <forcing> = CONUS_storage_fork/eval/v35_inputs/forcing)

Also reads
    <forcing>/work/node_map.csv                    copy or extract per sub-basin (node_map.py)
    <forcing>/work_pred/years/<year>.npy, work_pred/subbasins.csv    extraction 1980-2019
                                                   (extract_pred_v35.py)
    <forcing>/work/years/<year>.npy, work/subbasins.csv              extraction 2020-2025
                                                   (extract_v35.py)
    <forcing>/work/years/2026_raw.npy              extraction from the 2026 downloads
                                                   (prep_2026_v35.py)
    <forcing>/work/river/reach_lookup_v1.csv, reach_lookup_v35.csv   sub-basin to SWOT reach
                                                   crosswalks of the earlier node set and of v3.5
    <forcing>/work/river/reach_area_v35_all_reach_nodes.csv          names, for a new v3.5 node,
                                                   the removed earlier node with the same reach
                                                   set (column `source` = "v1_row_of_gone:<id>")
    <forcing>/work/river/riversp_vD_present_graph_v35ra.parquet      rebuilt river targets
                                                   (vd2026 only)

Usage
    python build_store_pred_v35.py <ext1980|vd2026> <basin_index>
    basin_index (0..101) indexes the sorted basin groups of the template; the script asserts that
    there are 102. Run one job per store and basin. Prints one JSON line of counts.

Layout
    As the template: Zarr format 3, float64 (date, subbasin) arrays in chunks of 365 days x all
    sub-basins, dates stored as int64 days since the first date. The ext1980 groups are not
    consolidated and the vd2026 groups are, as in their templates (asserted). All group attributes
    (among them `normalization_stats`) are copied unchanged from the template group.
    The columns are all sub-basin rows of the basin in the graph v3.5 table, in the order of that
    table. The order is the same in both prediction stores and in the training store.

Method
    ERA5-Land means (10 variables <variable>_mean)
        action "copy" (same id in the template group and identical polygon): the template column,
        bit for bit.
        action "extract": the new extraction over the v3.5 polygon, with the same weights as in
        the training store: 1980-2019 and 2020-2025 from the ERA5-Land archive, January to
        September 2026 from the 2026 downloads.
    Every other variable of the template group (lake targets dv_SWOTAE, dv_staticPLD and their
    dv_e_* companions, wse, wse_u, and the river target dv_river)
        copied by id wherever the id is a column of the template group. For dv_river the script
        asserts that every retained id maps to the same set of SWOT reaches in both crosswalks.
    dv_river at ids that are new in v3.5
        vd2026:  the rebuilt series that the training store also uses (v3.5 reach areas), so that
                 dv_river is identical in vd2026_v35 and in the training store, as it is between
                 their templates.
        ext1980: the template holds an older river record. A new node inherits the template
                 column of the removed earlier node that has the identical reach set, that is, the
                 reach's series exactly as the earlier store had it. Without such a node: NaN.
    Other variables at new ids: NaN (no lake or gauge id is new in v3.5).
"""
from swotgat.paths import WORK
import sys, json, numpy as np, pandas as pd, xarray as xr, zarr
from pathlib import Path
ROOT = Path(f"{WORK}"); C = ROOT/"CONUS"
F = ROOT/"CONUS_storage_fork/eval/v35_inputs/forcing"; W = F/"work"; P = F/"work_pred"
VARS = ["ssrd", "strd", "ro", "sro", "e", "tp", "sp", "d2m", "t2m", "fal"]
STORE = sys.argv[1]; T = C/{"ext1980": "zarr_store_ext1980", "vd2026": "zarr_store_vd2026"}[STORE]; NEW = F/f"zarr_store_{STORE}_v35"
CONS = {"ext1980": False, "vd2026": True}[STORE]; RANGE = {"ext1980": ("1980-01-01", "2025-12-31"), "vd2026": ("2020-01-01", "2026-09-30")}[STORE]
basins = sorted(p.name for p in T.iterdir() if (p/"zarr.json").is_file()); assert len(basins) == 102
b = basins[int(sys.argv[2])]
assert bool(json.load(open(T/b/"zarr.json")).get("consolidated_metadata")) == CONS, "template consolidation state"
NM = pd.read_csv(W/"node_map.csv", dtype={"subbasin": str, "outlet_id": str, "template_group": str}, keep_default_na=False)
nm = NM[NM.outlet_id == b].reset_index(drop=True); subs = nm.subbasin.tolist(); ns = len(subs); assert ns > 0 and len(set(subs)) == ns
src = xr.open_zarr(T/b, consolidated=CONS); tsubs = [str(s) for s in src.subbasin.values]; tpos = {s: j for j, s in enumerate(tsubs)}
dates = pd.DatetimeIndex(src.date.values); assert dates.equals(pd.date_range(*RANGE, freq="D")), "template dates"
copy = (nm.action == "copy").values; assert all(s in tpos for s in nm.subbasin[copy])
jc = np.flatnonzero(copy); je = np.flatnonzero(~copy); jt = np.array([tpos[subs[j]] for j in jc], dtype=int)
EX = np.full((len(dates), len(je), len(VARS)), np.nan)
if len(je):
    S0 = pd.read_csv(W/"subbasins.csv", dtype=str); pos0 = {s: i for i, s in enumerate(S0.subbasin)}; ie0 = np.array([pos0[subs[j]] for j in je])
    S1 = pd.read_csv(P/"subbasins.csv", dtype=str); pos1 = {s: i for i, s in enumerate(S1.subbasin)}
    r = 0
    for y in sorted(set(dates.year)):
        if y <= 2019:
            A = np.load(P/f"years/{y}.npy", mmap_mode="r"); D = pd.DatetimeIndex(pd.to_datetime(np.load(P/f"years/{y}_dates.npy"))).normalize()
            ie = np.array([pos1[subs[j]] for j in je])
        elif y <= 2025:
            A = np.load(W/f"years/{y}.npy", mmap_mode="r"); D = pd.DatetimeIndex(pd.to_datetime(np.load(W/f"years/{y}_dates.npy"))).normalize(); ie = ie0
        else:
            A = np.load(W/"years/2026_raw.npy", mmap_mode="r"); D = pd.DatetimeIndex(pd.to_datetime(np.load(W/"years/2026_raw_dates.npy"))).normalize(); ie = ie0
        want = dates[dates.year == y]; assert D.equals(want), (y, D[0], D[-1], len(D), want[0], want[-1])
        assert dates[r] == D[0]; EX[r:r + len(D)] = A[:, ie, :]; r += len(D)
    assert r == len(dates)
out = xr.Dataset(coords={"date": dates, "subbasin": np.array(subs)})
enc = {"date": {"units": f"days since {RANGE[0]} 00:00:00", "calendar": "proleptic_gregorian", "dtype": "int64"}}
out.to_zarr(NEW/b, mode="w", zarr_format=3, encoding=enc, consolidated=False)
def put(name, arr):
    """Append the float64 array arr (dates x sub-basins) to the new group as the variable `name`.

    The variable is chunked by 365 days x all sub-basins. Stops with an assertion error if arr
    has another shape or dtype.
    """
    assert arr.shape == (len(dates), ns) and arr.dtype == np.float64
    da = xr.DataArray(arr, dims=("date", "subbasin"), coords={"date": dates, "subbasin": out.subbasin})
    xr.Dataset({name: da}).to_zarr(NEW/b, mode="a", zarr_format=3, consolidated=False, encoding={name: {"chunks": (365, ns)}})
log = dict(store=STORE, basin=b, subbasins=ns, copy=int(copy.sum()), extract=int((~copy).sum()))
for k, v in enumerate(VARS):
    tv = src[f"{v}_mean"].transpose("date", "subbasin").values; assert tv.dtype == np.float64
    o = np.full((len(dates), ns), np.nan); o[:, jc] = tv[:, jt]; o[:, je] = EX[:, :, k]; del tv
    put(f"{v}_mean", o); log[f"{v}_nonfinite"] = int((~np.isfinite(o)).sum())
del EX
def xw(p):
    """Read a crosswalk file as {sub-basin: sorted tuple of its SWOT reach ids}."""
    x = pd.read_csv(p, dtype={"subbasin": str}); x["reach_id"] = x.reach_id.astype(float).astype(np.int64)
    return x.groupby("subbasin").reach_id.agg(lambda s: tuple(sorted(set(s)))).to_dict()
X1, X35 = xw(W/"river/reach_lookup_v1.csv"), xw(W/"river/reach_lookup_v35.csv")
RA = pd.read_csv(W/"river/reach_area_v35_all_reach_nodes.csv", dtype={"subbasin": str}); inherit = {r.subbasin: r.source.split(":")[1] for r in RA.itertuples() if r.source.startswith("v1_row_of_gone")}
jid = np.array([j for j, s in enumerate(subs) if s in tpos], dtype=int); tid = np.array([tpos[subs[j]] for j in jid], dtype=int)
others = [v for v in src.data_vars if not (v.endswith("_mean") and v[:-5] in VARS)]
for name in others:
    tv = src[name].transpose("date", "subbasin").values; assert tv.dtype == np.float64
    o = np.full((len(dates), ns), np.nan)
    if name != "dv_river":
        o[:, jid] = tv[:, tid]
    else:
        same = [j for j, s in enumerate(subs) if s in tpos and X1.get(s, ()) == X35.get(s, ())]
        assert len(same) == len(jid), "a retained id with a changed crosswalk"
        o[:, jid] = tv[:, tid]; newj = [j for j in range(ns) if subs[j] not in tpos]
        if STORE == "vd2026" and newj:
            R = pd.read_parquet(W/"river/riversp_vD_present_graph_v35ra.parquet", columns=["subbasin", "date", "dv_river"]); R["subbasin"] = R.subbasin.astype(str)
            R = R[R.subbasin.isin({subs[j] for j in newj})]
            di = dates.get_indexer(pd.to_datetime(R.date)); si = pd.Index(subs).get_indexer(R.subbasin); ok = (di >= 0) & (si >= 0)
            o[di[ok], si[ok]] = R.dv_river.values[ok]
        elif STORE == "ext1980":
            n_inh = 0
            for j in newj:
                g = inherit.get(subs[j])
                if g is not None and g in tpos: o[:, j] = tv[:, tpos[g]]; n_inh += 1
            log["dv_river_new_ids_inherited_from_gone_v1_node"] = n_inh
        log["dv_river_new_ids_with_data"] = int(np.isfinite(o[:, newj]).any(0).sum()) if newj else 0
    put(name, o); log[f"{name}_finite"] = int(np.isfinite(o).sum()); log[f"{name}_template_finite"] = int(np.isfinite(tv).sum()); del tv, o
grp = zarr.open_group(NEW/b, mode="r+")
for k, v in dict(zarr.open_group(T/b, mode="r").attrs).items(): grp.attrs[k] = v
if CONS: zarr.consolidate_metadata(str(NEW/b))
print(json.dumps(log))
