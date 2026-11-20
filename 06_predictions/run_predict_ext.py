"""Run a trained model in prediction mode for a list of nodes and a time window (no targets, no loss).

This is the script that does the prediction. For the paper it is not called directly: the wrappers pred_c5.py and
pred_v35_lowmem.py load it, with the model source of this repository imported first, and call main().

Usage
    python run_predict_ext.py <checkpoint_dir> --nodes nodes.csv --start YYYY-MM-DD --end YYYY-MM-DD --tag NAME
                              [--zarr STORE] [--out FILE]
    python run_predict_ext.py <checkpoint_dir> --regime r2|r1 [--nodes ...] [--start ...] [--end ...] [--tag ...]
  <checkpoint_dir>  <run>/checkpoints/step_NNNNNN, with model.eqx and config.json
  --nodes           csv with columns basin, subbasin: the nodes to write
  --start, --end    prediction window
  --tag             name of the run, used in the output file name and the cache folder
  --zarr            read the inputs from this store instead of the one in the configuration (for example the
                    1980-2025 store). It must hold every basin group with identical normalization_stats.
  --out             output file; default
                    $SWOTGAT_WORK/CONUS_storage_fork/predictions/<run>/<checkpoint name>_<tag>.parquet
  --regime          presets from earlier test sets, not used for the paper: r2 = 2015-2020 (ResOpsUS), r1 =
                    2023-2025 (agency gauges); each sets the node list, the window and the tag

What it does
  * The configuration comes from <checkpoint>/config.json, exactly what the weights were trained with. Only the
    test node list, the test window and cache_dir are changed (and zarr_dir with --zarr).
  * The `predict` subset is narrowed to the requested nodes and window. The library default would be the
    training and test nodes over the whole span of all date ranges.
    Normalization is untouched: it comes from train_basin_file and the precomputed statistics of the store,
    which are not modified.
  * The inputs start sequence_length days before --start, so that the first prediction falls on --start.
  * The model's inputs are ERA5-Land forcing and static attributes only. The script asserts that the model has
    no observation inputs, so no SWOT observation can reach a prediction.
  * Each run gets its own cache folder (per tag and run), because the cache hash of the model source ignores
    test_basin_file and test_date_range, and because parallel jobs for the two seeds must not share one cache.

Writes
  A parquet file with columns basin, subbasin, date, pred_dv_SWOTAE, pred_dv_SWOTAE_std, pred_dv_river,
  pred_dv_river_std: predicted storage change in million cubic meters (denormalized), one row per requested
  node and day of the window. Rows of other nodes are not written. A .json file next to it records the
  checkpoint, window, store, row count and run time.

Notes
  With --zarr the script asserts that the store has a group named USGS-01034500, as a simple test that the path
  is a basin store.
  Run directly, the script imports the model source from $SWOTGAT_WORK/swot-ml/src.
"""
from swotgat.paths import WORK
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

REPO = Path(f"{WORK}/swot-ml")
C = Path(f"{WORK}/CONUS")
F = Path(f"{WORK}/CONUS_storage_fork")
sys.path.insert(0, str(REPO/"src"))
import equinox as eqx                                             # noqa: E402
import models                                                     # noqa: E402
from config import Config, DataSubset                             # noqa: E402
import data.cached_basingraphdataset as cbd                       # noqa: E402
from data import CachedBasinGraphDataLoader, CachedBasinGraphDataset, DynamicCacheManager  # noqa: E402
from evaluate.inference import model_df_iterate                   # noqa: E402

REGIMES = {"r2": ("regime2_2015_2020", "2015-01-01", "2020-12-31"),
           "r1": ("regime1_2023_2025", "2023-01-01", "2025-12-31")}


def main():
    """Parse the command line, predict for the requested nodes and window, and write the output."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path, help="<run>/checkpoints/step_NNNNNN")
    ap.add_argument("--regime", choices=REGIMES)
    ap.add_argument("--nodes", type=Path, help="(basin, subbasin) csv; overrides --regime node list")
    ap.add_argument("--start"); ap.add_argument("--end"); ap.add_argument("--tag")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--zarr", type=Path, help="read inputs from this zarr store instead of the config's (e.g. the 1980-extended "
                    "store); it must hold every basin group with identical normalization_stats (verify_store.py)")
    a = ap.parse_args()
    if a.regime:
        sub, s0, s1 = REGIMES[a.regime]
        nodes = a.nodes or F/"testsets"/sub/"predict_nodes.csv"
        start, end, tag = a.start or s0, a.end or s1, a.tag or a.regime
    else:
        if not (a.nodes and a.start and a.end and a.tag): sys.exit("need --regime or --nodes/--start/--end/--tag")
        nodes, start, end, tag = a.nodes, a.start, a.end, a.tag
    ck = a.checkpoint.resolve()
    if not (ck/"model.eqx").is_file(): sys.exit(f"not a checkpoint dir: {ck}")

    cfg = Config.from_file(ck/"config.json")
    seq = int(cfg.sequence_length)
    w0 = (pd.Timestamp(start) - pd.Timedelta(days=seq)).strftime("%Y-%m-%d")
    run_name = ck.parent.parent.name
    cache_root = C/f"_cache_predict_{tag}_{run_name}"      # per run: parallel seed jobs must not race on one cache
    cfg.test_basin_file = Path(nodes).resolve()
    cfg.test_date_range = [pd.Timestamp(w0).to_pydatetime(), pd.Timestamp(end).to_pydatetime()]
    cfg.cache_dir = cache_root
    if a.zarr:
        cfg.zarr_dir = a.zarr.resolve()
        assert (cfg.zarr_dir/"USGS-01034500"/"zarr.json").is_file(), f"not a basin zarr store: {cfg.zarr_dir}"
    if str(cache_root) == str(C/"_cache"): sys.exit("FATAL predict cache would share the training cache")

    want = pd.read_csv(nodes, dtype=str)
    keep = set(zip(want.basin, want.subbasin))
    print(f"checkpoint {ck}\n  nodes {len(keep):,} in {want.basin.nunique()} basins | window {start}..{end} "
          f"(inputs from {w0}, seq {seq}) | cache {cache_root}", flush=True)

    # --- narrow only the predict subset; the train and validate subsets (normalization) are untouched ---
    _slice0, _dict0 = cbd._get_time_slice, cbd._get_basin_subbasin_dict
    def _is_pred(s): return (s == DataSubset.predict) or (str(getattr(s, "value", s)) == "predict")
    def _time_slice(c, s): return slice(*[str(pd.Timestamp(x).date()) for x in c.test_date_range]) if _is_pred(s) else _slice0(c, s)
    def _basin_dict(c, s):
        if _is_pred(s):
            d = {}
            for b, n in keep: d.setdefault(b, set()).add(n)
            return d
        return _dict0(c, s)
    cbd._get_time_slice, cbd._get_basin_subbasin_dict = _time_slice, _basin_dict

    # --- model: weights only (no optimizer, no training dataloader) ---
    with open(ck/"model.eqx", "rb") as f:
        _, template = models.make(cfg)
        model = eqx.tree_deserialise_leaves(f, template)
    # Works for any model with a dv_SWOTAE output head (the paper's model has dv_river and dv_SWOTAE); every head is written.
    # Earlier models that predict lake level (target wse, an anomaly in m) get pred_dv_SWOTAE = pred_wse x PLD lake area
    # (km2 x m = MCM, as dv_staticPLD is built), so that the storage scoring scripts can run; the level is written too.
    level = "wse" in list(model.target) and "dv_SWOTAE" not in list(model.target)
    assert "dv_SWOTAE" in list(model.target) or level, model.target
    if level:
        _st = pd.read_parquet(cfg.attributes_file, columns=["lake_area_km2"]).reset_index()
        lake_km2 = dict(zip(_st.subbasin.astype(str), _st.lake_area_km2.astype(float)))
    assert not getattr(model, "sparse_sources", ()), "model has observation inputs; predict-mode would not be observation-free"

    t0 = time.time()
    cache_dir = DynamicCacheManager(cfg).create_cache(DataSubset.predict)
    ds = CachedBasinGraphDataset(cfg, cache_dir, "predict")
    dl = CachedBasinGraphDataLoader(cfg, ds, infinite_shuffle=False)
    out = a.out or (F/"predictions"/run_name/f"{ck.name}_{tag}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer, n_rows = None, 0
    for df in model_df_iterate(model, dl, quiet=True):
        df = df.reset_index()
        df.columns = ["_".join(c for c in col if c) if isinstance(col, tuple) else col for col in df.columns]
        df = df[[(b, s) in keep for b, s in zip(df.basin.astype(str), df.subbasin.astype(str))]]
        df = df[(df.date >= pd.Timestamp(start)) & (df.date <= pd.Timestamp(end))]
        if df.empty: continue
        df = df[[c for c in df.columns if c in ("basin", "subbasin", "date") or c.startswith("pred_")]]
        df["subbasin"] = df.subbasin.astype(str); df["basin"] = df.basin.astype(str)
        if level:
            a_ = df.subbasin.map(lake_km2).astype(float).values
            df["pred_dv_SWOTAE"] = df["pred_wse"].values * a_
            df["pred_dv_SWOTAE_std"] = df["pred_wse_std"].values * a_
        t = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None: writer = pq.ParquetWriter(out, t.schema)
        writer.write_table(t); n_rows += len(df)
    if writer: writer.close()
    meta = dict(checkpoint=str(ck), regime=tag, window=[start, end], inputs_from=w0, nodes=len(keep), zarr=str(cfg.zarr_dir),
                rows=n_rows, seconds=round(time.time() - t0), cache=str(cache_root))
    json.dump(meta, open(out.with_suffix(".json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
