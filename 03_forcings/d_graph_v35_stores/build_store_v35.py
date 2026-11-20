"""Build one basin group of the graph v3.5 training store (forcing and SWOT targets, 2020-2026).

The store is assembled from a "template store": the training store of the same period on the
earlier node set, which is only read. Sub-basins that did not change keep the template values bit
for bit; the others get newly extracted forcing.

Reads (paths relative to $SWOTGAT_WORK; <forcing> = CONUS_storage_fork/eval/v35_inputs/forcing)
    CONUS/zarr_store_vd2026_halfexact/<basin>   the template group
        (04_swot_targets/c_template_store/build_store_vd2026_halfexact.py)
    <forcing>/work/node_map.csv                 copy or extract per sub-basin (node_map.py)
    <forcing>/work/subbasins.csv                sub-basin order of the extraction arrays
    <forcing>/work/years/2020.npy .. 2025.npy   extraction from the ERA5-Land archive (extract_v35.py)
    <forcing>/work/years/2026_raw.npy           extraction from the 2026 downloads (prep_2026_v35.py)
    <forcing>/work/river/reach_lookup_v1.csv, reach_lookup_v35.csv
                                                sub-basin to SWOT reach crosswalks of the earlier
                                                node set and of graph v3.5
    <forcing>/work/river/riversp_vD_present_graph_v35ra.parquet
                                                river targets rebuilt on the v3.5 sub-basins with
                                                the v3.5 reach-area table (river_build_v35ra.py)

Writes
    <forcing>/zarr_store_vd2026_halfexact_v35/<basin>

Usage
    python build_store_v35.py <basin_index>
    basin_index (0..101) indexes the sorted basin groups of the template; the script asserts that
    there are 102. Run one job per basin, for example as a Slurm array. Prints one JSON line of
    counts.

Layout (the same as the template)
    One Zarr format 3 group per basin. Coordinates: `date` (2020-01-01 to 2026-09-30, stored as
    int64 days since 2020-01-01) and `subbasin`. Every variable is a float64 (date, subbasin)
    array in chunks of 365 days x all sub-basins. The group attribute `normalization_stats` is
    copied unchanged from the template group and never recomputed. Metadata is consolidated per
    group.
    The columns are all sub-basin rows of the basin in the graph v3.5 table, in the order of that
    table (the template likewise carries every sub-basin row of its node set).

Method
    ERA5-Land means (10 variables <variable>_mean)
        action "copy" (same id in the template group and identical polygon): the template column,
        bit for bit.
        action "extract": the new extraction over the v3.5 polygon, from the archive for
        2020-2025 and from the 2026 downloads for 2026-01-01 to 2026-09-30. The yearly files must
        cover exactly the expected dates (asserted).
    dv_SWOTAE, dv_staticPLD (lake targets; a lake node's id is its lake id)
        the template column wherever the id is a column of the template group, otherwise NaN.
    dv_river (river target)
        the template column where the id is a template column and the sub-basin maps to the same
        set of SWOT reaches in both crosswalks; otherwise the rebuilt series (NaN where it has no
        data). The rebuilt series uses the graph v3.5 reach areas. For the copied nodes the log
        also reports whether the rebuilt series equals the template column.
"""
from swotgat.paths import WORK
import sys, json, numpy as np, pandas as pd, xarray as xr, zarr
from pathlib import Path
ROOT = Path(f"{WORK}"); T = ROOT/"CONUS/zarr_store_vd2026_halfexact"
F = ROOT/"CONUS_storage_fork/eval/v35_inputs/forcing"; W = F/"work"; NEW = F/"zarr_store_vd2026_halfexact_v35"
VARS = ["ssrd", "strd", "ro", "sro", "e", "tp", "sp", "d2m", "t2m", "fal"]
basins = sorted(p.name for p in T.iterdir() if (p/"zarr.json").is_file()); assert len(basins) == 102
b = basins[int(sys.argv[1])]
NM = pd.read_csv(W/"node_map.csv", dtype={"subbasin": str, "outlet_id": str, "template_group": str}, keep_default_na=False)
nm = NM[NM.outlet_id == b].reset_index(drop=True); subs = nm.subbasin.tolist(); ns = len(subs); assert ns > 0 and len(set(subs)) == ns
src = xr.open_zarr(T/b, consolidated=True); tsubs = [str(s) for s in src.subbasin.values]; tpos = {s: j for j, s in enumerate(tsubs)}
dates = pd.DatetimeIndex(src.date.values); assert dates.equals(pd.date_range("2020-01-01", "2026-09-30", freq="D")), "template dates"
copy = (nm.action == "copy").values; assert all(s in tpos for s in nm.subbasin[copy])
jc = np.flatnonzero(copy); je = np.flatnonzero(~copy); jt = np.array([tpos[subs[j]] for j in jc], dtype=int)
# ---- extracted ERA5-Land for the columns with action "extract": array (dates, n_extract, 10)
EX = np.full((len(dates), len(je), len(VARS)), np.nan)
if len(je):
    SB = pd.read_csv(W/"subbasins.csv", dtype=str); pos = {s: i for i, s in enumerate(SB.subbasin)}; ie = np.array([pos[subs[j]] for j in je])
    r = 0
    for y in range(2020, 2026):
        A = np.load(W/f"years/{y}.npy", mmap_mode="r"); D = pd.DatetimeIndex(pd.to_datetime(np.load(W/f"years/{y}_dates.npy"))).normalize()
        assert D.equals(pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="D")), (y, D[0], len(D))
        assert dates[r] == D[0]; EX[r:r + len(D)] = A[:, ie, :]; r += len(D)
    A26 = np.load(W/"years/2026_raw.npy", mmap_mode="r"); D26 = pd.DatetimeIndex(pd.to_datetime(np.load(W/"years/2026_raw_dates.npy"))).normalize()
    new = dates[r:]; assert D26.equals(new), ("2026 dates", D26[0], D26[-1], new[0], new[-1])
    EX[r:] = A26[:, ie, :]
out = xr.Dataset(coords={"date": dates, "subbasin": np.array(subs)})
enc = {"date": {"units": "days since 2020-01-01 00:00:00", "calendar": "proleptic_gregorian", "dtype": "int64"}}
out.to_zarr(NEW/b, mode="w", zarr_format=3, encoding=enc, consolidated=False)
def put(name, arr):
    """Append the float64 array arr (dates x sub-basins) to the new group as the variable `name`.

    The variable is chunked by 365 days x all sub-basins. Stops with an assertion error if arr
    has another shape or dtype.
    """
    assert arr.shape == (len(dates), ns) and arr.dtype == np.float64
    da = xr.DataArray(arr, dims=("date", "subbasin"), coords={"date": dates, "subbasin": out.subbasin})
    xr.Dataset({name: da}).to_zarr(NEW/b, mode="a", zarr_format=3, consolidated=False, encoding={name: {"chunks": (365, ns)}})
log = dict(basin=b, subbasins=ns, copy=int(copy.sum()), extract=int((~copy).sum()), template_subbasins=len(tsubs))
for k, v in enumerate(VARS):
    tv = src[f"{v}_mean"].transpose("date", "subbasin").values; assert tv.dtype == np.float64
    o = np.full((len(dates), ns), np.nan); o[:, jc] = tv[:, jt]; o[:, je] = EX[:, :, k]
    put(f"{v}_mean", o); log[f"{v}_nonfinite_extract"] = int((~np.isfinite(EX[:, :, k])).sum())
# ---- lake targets: copy by id
jid = np.array([j for j, s in enumerate(subs) if s in tpos], dtype=int); tid = np.array([tpos[subs[j]] for j in jid], dtype=int)
for name in ("dv_SWOTAE", "dv_staticPLD"):
    tv = src[name].transpose("date", "subbasin").values; o = np.full((len(dates), ns), np.nan); o[:, jid] = tv[:, tid]
    put(name, o); log[f"{name}_finite"] = int(np.isfinite(o).sum()); log[f"{name}_template_finite"] = int(np.isfinite(tv).sum())
# ---- dv_river: copy where the crosswalk is identical, else rebuilt
def xw(p):
    """Read a crosswalk file as {sub-basin: sorted tuple of its SWOT reach ids}."""
    x = pd.read_csv(p, dtype={"subbasin": str}); x["reach_id"] = x.reach_id.astype(float).astype(np.int64)
    return x.groupby("subbasin").reach_id.agg(lambda s: tuple(sorted(set(s)))).to_dict()
X1, X35 = xw(W/"river/reach_lookup_v1.csv"), xw(W/"river/reach_lookup_v35.csv")
R = pd.read_parquet(W/"river/riversp_vD_present_graph_v35ra.parquet", columns=["subbasin", "date", "dv_river"]); R["subbasin"] = R.subbasin.astype(str)
R = R[R.subbasin.isin(set(subs))]
g = np.full((len(dates), ns), np.nan)
if len(R):
    di = dates.get_indexer(pd.to_datetime(R.date)); si = pd.Index(subs).get_indexer(R.subbasin); ok = (di >= 0) & (si >= 0)
    assert not pd.DataFrame({"d": di[ok], "s": si[ok]}).duplicated().any(), "duplicate (subbasin, date) in rebuilt river"
    g[di[ok], si[ok]] = R.dv_river.values[ok]
tv = src["dv_river"].transpose("date", "subbasin").values; o = g.copy(); same, agree, differ = 0, 0, []
for j, s in enumerate(subs):
    if s in tpos and X1.get(s, ()) == X35.get(s, ()):
        same += 1; t = tv[:, tpos[s]]
        if np.array_equal(t, g[:, j], equal_nan=True): agree += 1
        else: differ.append(s)
        o[:, j] = t
put("dv_river", o)
log.update(dv_river_rebuilt_nodes_finite=int(np.isfinite(o[:, [j for j, s in enumerate(subs) if not (s in tpos and X1.get(s, ()) == X35.get(s, ()))]]).any(0).sum()), dv_river_same_crosswalk=same, dv_river_rebuild_equals_template=agree, dv_river_rebuild_differs=differ[:20], n_rebuild_differs=len(differ),
           dv_river_nodes_finite=int(np.isfinite(o).any(0).sum()), dv_river_template_nodes_finite=int(np.isfinite(tv).any(0).sum()),
           dv_river_finite=int(np.isfinite(o).sum()), dv_river_template_finite=int(np.isfinite(tv).sum()))
grp = zarr.open_group(NEW/b, mode="r+"); grp.attrs["normalization_stats"] = dict(zarr.open_group(T/b, mode="r").attrs)["normalization_stats"]
zarr.consolidate_metadata(str(NEW/b))
print(json.dumps(log))
