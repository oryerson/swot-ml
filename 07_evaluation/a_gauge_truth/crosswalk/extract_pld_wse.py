"""Extract the SWOT lake level and storage-change series (LakeSP version C) for every graph node that is a lake.

Writes crosswalk/pld_wse_graph.parquet (lake_id, date, wse, dv_SWOTAE, dv_staticPLD), which the dam-to-node
crosswalk uses to compare SWOT water levels with reservoir pool levels. The input series are unpublished
data of a collaborating group (see docs/DATA_SOURCES.md), so the output is not redistributed.

Usage
  python extract_pld_wse.py        (no arguments; reads the basin_7*_daily_timeseries.csv files
                                    with 8 processes)
"""
from swotgat.paths import NAS_WATER, WORK
import pandas as pd, pyarrow.csv as pc, pyarrow as pa, pyarrow.compute as pcmp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
D = Path(f"{NAS_WATER}/cjgleason/ted/LakeSPvC_ts_for_Ted/daily_ts_from_LakeSPvC")
F = Path(f"{WORK}/CONUS_storage_fork"); C = Path(f"{WORK}/CONUS")
st = pd.read_parquet(C/"attributes/static_conus_storage.parquet", columns=["lake_area_km2"]).reset_index()
IDS = sorted(set(st.subbasin.astype(str)))
COLS = ["lake_id", "date", "wse", "dv_SWOTAE", "dv_staticPLD"]
def one(f):
    """Read one basin file and keep the rows of graph lakes that hold at least one value.

    f is the path of a daily time-series CSV. Only the columns in COLS are read, lake_id and date as
    text. Rows are kept if their lake_id is a graph node (IDS) and wse, dv_SWOTAE or dv_staticPLD is
    not missing. Returns (file name, DataFrame).
    """
    t = pc.read_csv(f, convert_options=pc.ConvertOptions(include_columns=COLS,
                    column_types={"lake_id": pa.string(), "date": pa.string()}))
    t = t.filter(pcmp.is_in(t["lake_id"], value_set=pa.array(IDS)))
    df = t.to_pandas(); df = df[df[["wse", "dv_SWOTAE", "dv_staticPLD"]].notna().any(axis=1)]
    return f.name, df
if __name__ == "__main__":
    files = sorted(D.glob("basin_7*_daily_timeseries.csv"))
    out = []
    with ProcessPoolExecutor(8) as ex:
        for name, df in ex.map(one, files):
            print(name, len(df), df.lake_id.nunique(), flush=True); out.append(df)
    X = pd.concat(out, ignore_index=True); X["date"] = pd.to_datetime(X.date)
    X.to_parquet(F/"crosswalk/pld_wse_graph.parquet", index=False)
    print("TOTAL rows", len(X), "lakes", X.lake_id.nunique(), " of graph nodes", len(IDS))
