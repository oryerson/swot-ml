"""Prepare three inputs of the 2020-2026 training store (zarr_store_vd2026, build_store_vd2026.py).

"vD" is version D of the SWOT lake and river products. Paths below are relative to $SWOTGAT_WORK;
<swotext> = CONUS_storage_fork/eval/swotext and <era5_2026> = CONUS_storage_fork/eval/overnight/
era5_2026.

Usage: python store_inputs_vd.py        (no arguments)

1. Lake targets: <swotext>/lake_targets_vd2026.parquet
   Reads <swotext>/lakesp_vD_half_graph.parquet: the version-D lake series after the HALF outlier
   filter (Trudel et al.), one row per lake and day, for the lakes that are graph nodes. The node
   id (`subbasin`) of a lake is its lake_id in the SWOT Prior Lake Database (PLD).
   - dv_SWOTAE: storage change from the SWOT area-elevation curve where a row has it, otherwise
     from the 3D-LAKES curve (174 lakes rely on 3D-LAKES in the paper's run).
   - dv_staticPLD: storage change from level change times the static PLD lake area. The model
     falls back to this target where a lake has no dv_SWOTAE.
   - primary_source: which of the two curves gave dv_SWOTAE ("SWOTAE", "3DLAKES" or "none").
   Rows with neither target are dropped.

2. List of lakes with a dv_SWOTAE target: CONUS/basin_lists/fallback/dv_SWOTAE_vd2026_fb.csv
   Reads the list of an earlier model configuration (CONUS/basin_lists/fallback/
   dv_SWOTAE_v8_fb.csv) and appends every graph lake that has data in step 1 and is not yet
   listed. The basin of each added lake comes from CONUS/metadata/subbasins_conus.parquet; a lake
   without a basin there is dropped. CONUS/basin_lists/storage/train_nodes_g.csv is read only to
   print the share of added lakes that lie in training basins.

3. Forcing correction for two Susquehanna sub-basins: <era5_2026>/susq_map.json
   The production store (CONUS/zarr_store) holds forcing for USGS-01545500 and 73005469 that was
   extracted with an older polygon than the one behind the current weights. To keep their series
   from stepping at 2026-01-01, the script fits, for each of the 10 variables, an ordinary least
   squares line  production = a + b * (this pipeline's extraction)  on the daily values of 2025,
   and stores a, b, the correlation r and the residual standard deviation divided by the
   standard deviation of the production series. build_store_vd2026.py applies the line to the
   2026 forcing of the two sub-basins. Reads the 2025 archive files through
   prep_2026.month_means and the basin group CONUS/zarr_store/USGS-01578310.

Notes
    The script imports prep_2026.py as a module from <era5_2026> (the folder where the scripts
    of this step were deployed).
"""
from swotgat.paths import WORK
import sys, json, numpy as np, pandas as pd, geopandas as gpd, xarray as xr
sys.argv = ["x"]; sys.path.insert(0, f"{WORK}/CONUS_storage_fork/eval/overnight/era5_2026")
C = f"{WORK}/CONUS"; S = f"{WORK}/CONUS_storage_fork/eval/swotext"
E = f"{WORK}/CONUS_storage_fork/eval/overnight/era5_2026"
X = pd.read_parquet(f"{S}/lakesp_vD_half_graph.parquet"); X["lake_id"] = X.lake_id.astype(str)
T = pd.DataFrame({"subbasin": X.lake_id, "date": pd.to_datetime(X.date), "dv_SWOTAE": X.dv_SWOTAE.fillna(X.dv_3DLAKES), "dv_staticPLD": X.dv_staticPLD,
                  "primary_source": np.where(X.dv_SWOTAE.notna(), "SWOTAE", np.where(X.dv_3DLAKES.notna(), "3DLAKES", "none"))})
T = T[T.dv_SWOTAE.notna() | T.dv_staticPLD.notna()]; T.to_parquet(f"{S}/lake_targets_vd2026.parquet", index=False)
lk = T.groupby("subbasin").primary_source.agg(lambda s: s.mode().iat[0])
print(f"[1] lake targets: {len(T):,} lake-days, {T.subbasin.nunique()} lakes ({lk.value_counts().to_dict()}), {T.date.min().date()} .. {T.date.max().date()}", flush=True)
subs = gpd.read_parquet(f"{C}/metadata/subbasins_conus.parquet"); subs.index = subs.index.astype(str)
old = pd.read_csv(f"{C}/basin_lists/fallback/dv_SWOTAE_v8_fb.csv", dtype=str)
new = sorted(set(T.subbasin) - set(old.subbasin)); add = pd.DataFrame({"basin": [str(subs.outlet_id.get(s)) for s in new], "subbasin": new})
miss = add[add.basin.isin(["None", "nan"])]; add = add[~add.basin.isin(["None", "nan"])]
L = pd.concat([old, add], ignore_index=True); L.to_csv(f"{C}/basin_lists/fallback/dv_SWOTAE_vd2026_fb.csv", index=False)
tr = pd.read_csv(f"{C}/basin_lists/storage/train_nodes_g.csv", dtype=str)
print(f"[2] dv_SWOTAE list: {len(old)} -> {len(L)} rows (+{len(add)} lakes new with vD data; {len(miss)} without a basin dropped); "
      f"of the added lakes, in the training basins {add.basin.isin(set(tr.basin)).mean():.0%}", flush=True)
import prep_2026 as P
SB = pd.read_csv(f"{P.EXT}/subbasins.csv", dtype=str); pos = {s: i for i, s in enumerate(SB.subbasin)}; two = ["USGS-01545500", "73005469"]
ours = []; ts = []
for m in range(1, 13):
    t, A = P.month_means("nas", 2025, m, None); ours.append(A[:, [pos[s] for s in two], :]); ts.append(t)
ours = np.concatenate(ours); ts = np.concatenate([t.values for t in ts])
z = xr.open_zarr(f"{C}/zarr_store/USGS-01578310", consolidated=True); out = {}
for k, v in enumerate(P.VARS):
    prod = z[f"{v}_mean"].sel(date=slice("2025-01-01", "2025-12-31"), subbasin=two).transpose("date", "subbasin").values
    for j, s in enumerate(two):
        x, y = ours[:, j, k], prod[:, j]; ok = np.isfinite(x) & np.isfinite(y)
        b, a = np.polyfit(x[ok], y[ok], 1); res = y[ok] - (a + b*x[ok])
        out.setdefault(s, {})[v] = dict(a=float(a), b=float(b), r=float(np.corrcoef(x[ok], y[ok])[0, 1]), resid_sd_over_sd=float(res.std()/y[ok].std()))
json.dump(out, open(f"{E}/susq_map.json", "w"), indent=1)
for s in two: print(f"[3] {s}: r {[round(out[s][v]['r'], 4) for v in P.VARS]} | resid/sd {[round(out[s][v]['resid_sd_over_sd'], 3) for v in P.VARS]}", flush=True)
