import copy
import hashlib
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple
import concurrent.futures

import dask.array as da
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from jax.tree import map as jt_map
from jaxtyping import Array
from torch.utils.data import Dataset
from tqdm import tqdm

from config import Config, DataSubset, Features


class GraphBatch(NamedTuple):
    dynamic: dict[str, Array]
    graph_edges: Array
    inflow_indices: Array = None
    inflow_mask: Array = None
    graph_idx: Array = None
    node_mask: Array = None
    edge_mask: Array = None
    static: Array = None
    y: dict[str, Array] = {}

    def __getitem__(self, key):
        warnings.warn(
            f"Batch: dict-style access ('batch[\"{key}\"]') is deprecated. Use attribute access ('batch.{key}') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(self, key)

    def to_jax(self):
        """Convert all numpy arrays to jax arrays, skipping None values."""

        def _to_jax_item(x):
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                return jnp.asarray(x)
            return x

        return jt_map(_to_jax_item, self)


def get_train_ds_stats(cfg: Config, *, dynamic: bool = True, static: bool = True) -> dict:
    print("Calculating training statistics for encoding and normalization...")

    train_basin_dict = _get_basin_subbasin_dict(cfg, "train")
    # A basin pulled in only by a per-target list carries an EMPTY base node set. It is
    # loaded so that one target can be supervised inside it, but it must not inform
    # normalization: aggregating its stored stats would fold a held-out basin's
    # DISCHARGE distribution into the scale factors even though no discharge gradient
    # ever flows from it. Statistics come from base-supervised basins only.
    train_basins = [b for b, subs in train_basin_dict.items() if subs]
    train_subbasins = {x for s in train_basin_dict.values() for x in s}

    static_df = pd.read_parquet(cfg.attributes_file)
    subbasin_mask = static_df.index.get_level_values("subbasin").isin(train_subbasins)
    static_train_df = static_df[subbasin_mask]

    stats = {}
    if static:
        stats["s_encoding"] = _get_static_encoding(
            static_train_df, cfg.static_encoding.model_dump()
        )
        stats["s_scale"] = _calculate_scale_from_data(cfg, static_train_df, stats["s_encoding"])
    if dynamic:
        stats["d_encoding"] = _get_prescribed_encoding(cfg.dynamic_encoding.model_dump())
        stats["d_scale"] = _get_scale_from_precomp_stats(cfg, train_basins, stats["d_encoding"])
        # A trailing mean has its variable's mean and a smaller spread, so the variable's
        # own z-score keeps it on a sensible scale without stats of its own.
        for name, (var, _) in _rolling_mean_specs(cfg).items():
            stats["d_scale"][name] = dict(stats["d_scale"][var])
        if getattr(cfg, "add_doy_features", False):   # already in [-1, 1]
            for name in ("doy_sin", "doy_cos"):
                stats["d_scale"][name] = {"encoded": False, "log_norm": False, "offset": 0.0, "scale": 1.0}
        # External-store inputs are already standardised (see Config.extra_dynamic_store): identity scaling,
        # never statistics of the training window.
        for name in getattr(cfg, "extra_dynamic_vars", None) or []:
            stats["d_scale"][name] = {"encoded": False, "log_norm": False, "offset": 0.0, "scale": 1.0}
        for var, o in (getattr(cfg, "target_scale_override", None) or {}).items():
            stats["d_scale"][var] = {"encoded": False, "log_norm": False,
                                     "offset": float(o["offset"]), "scale": float(o["scale"])}
        # Observation-noise columns are standard deviations in the target's physical units, so
        # they are divided by the target's scale and never shifted.
        for name, t in _obs_noise_columns(cfg).items():
            stats["d_scale"][name] = {"encoded": False, "log_norm": False, "offset": 0.0,
                                      "scale": float(stats["d_scale"][t]["scale"])}

    return stats


def _obs_noise_columns(cfg: Config) -> dict[str, str]:
    """{"<t>__sigma": t, "<t>__fb__sigma": t} for Config.obs_noise."""
    out = {}
    for t, n in (getattr(cfg, "obs_noise", None) or {}).items():
        out[f"{t}__sigma"] = t
        if n.fb_var:
            out[f"{t}__fb__sigma"] = t
    return out


_DROP_MASKS = {}


def _drop_mask_table(path) -> dict:
    """{basin: DataFrame(subbasin, date)} for one drop-list parquet, read once per process."""
    key = str(path)
    if key not in _DROP_MASKS:
        d = pd.read_parquet(path, columns=["basin", "subbasin", "date"])
        d["subbasin"] = d.subbasin.astype(str)
        d["date"] = pd.to_datetime(d.date).values.astype("datetime64[D]")
        _DROP_MASKS[key] = {b: g[["subbasin", "date"]] for b, g in d.groupby("basin")}
    return _DROP_MASKS[key]


def _file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rolling_mean_specs(cfg: Config) -> dict[str, tuple[str, int]]:
    """{"<var>_rm<w>": (var, w)} for Config.rolling_mean_vars x add_rolling_means."""
    if not getattr(cfg, "rolling_mean_vars", None):
        return {}
    return {f"{v}_rm{w}": (v, int(w)) for v in cfg.rolling_mean_vars for w in cfg.add_rolling_means}


def _get_prescribed_encoding(encoding):
    new_cols = {}
    # Handle categorical columns
    if encoding["categorical"]:
        for col, categories in encoding["categorical"].items():
            if not categories:
                raise ValueError(
                    f"Dynamic categorical variable '{col}' must either be excluded or have "
                    "its categories prescribed in the config."
                )
            new_cols[col] = [f"{col}_{cat}" for cat in categories]
    # Handle bitmask columns
    if encoding["bitmask"]:
        for col, bits in encoding["bitmask"].items():
            if not bits:
                raise ValueError(
                    f"Dynamic bitmask variable '{col}' must either be excluded or have "
                    "its bits prescribed in the config."
                )
            new_cols[col] = [f"{col}_bit_{bit}" for bit in bits]

    encoding["encoded_columns"] = new_cols
    return encoding


def _get_scale_from_precomp_stats(cfg: Config, basins: list[str], encoding: dict):
    """
    Calculates mean and std for a subset of basins by aggregating pre-computed stats.
    Uses pre-stored attributes in each basin's Zarr group.
    """
    aggregated_stats = defaultdict(
        lambda: {
            "count": 0,
            "sum": 0.0,
            "sum_sq": 0.0,
            "min": np.inf,
            "max": -np.inf,
            "log_sum": 0.0,
            "log_sum_sq": 0.0,
            "positive_count": 0,
        }
    )

    for basin_id in basins:
        try:
            basin_path = cfg.zarr_dir / basin_id
            z_group = zarr.open(basin_path, mode="r")
            basin_stats = z_group.attrs["normalization_stats"]

            for var, stats in basin_stats.items():
                agg = aggregated_stats[var]
                agg["count"] += stats["count"]
                agg["sum"] += stats["sum"]
                agg["sum_sq"] += stats["sum_sq"]
                agg["min"] = min(agg["min"], stats["min"])
                agg["max"] = max(agg["max"], stats["max"])
                agg["log_sum"] += stats["log_sum"]
                agg["log_sum_sq"] += stats["log_sum_sq"]
                agg["positive_count"] += stats["positive_count"]
        except Exception:
            raise KeyError(f"Failed to load normalization stats for {basin_path}")

    scale = {}
    all_vars = set(aggregated_stats.keys()).union(set(encoding["encoded_columns"]))

    for var in all_vars:
        scale[var] = {"encoded": False, "log_norm": False, "offset": 0.0, "scale": 1.0}

        if var in encoding["encoded_columns"]:
            scale[var]["encoded"] = True
            continue

        stats = aggregated_stats[var]

        if var in cfg.log_norm_cols:
            log_mean = stats["log_sum"] / stats["positive_count"]
            log_var = (stats["log_sum_sq"] / stats["positive_count"]) - log_mean**2
            log_std = np.sqrt(max(log_var, 1e-12))
            scale[var]["log_norm"] = True
            scale[var]["offset"] = log_mean
            scale[var]["scale"] = log_std
        elif var in cfg.no_norm_cols:
            scale[var]["offset"] = 0
            scale[var]["scale"] = 1
        elif var in cfg.range_norm_cols:
            min_val, max_val = stats["min"], stats["max"]
            scale[var]["offset"] = min_val
            scale[var]["scale"] = max_val - min_val if max_val > min_val else 1.0
        else:  # z-score
            mean = stats["sum"] / stats["count"]
            variance = (stats["sum_sq"] / stats["count"]) - (mean**2)
            std = np.sqrt(max(variance, 0))
            scale[var]["offset"] = mean
            scale[var]["scale"] = std if std > 1e-9 else 1.0
    return scale


def _get_static_encoding(df: pd.DataFrame, encoding: dict):
    new_cols = {}
    # Handle categorical columns
    for col, categories in encoding["categorical"].items():
        if not categories:
            categories = df[col].astype("category").unique()
            encoding["categorical"][col] = categories  # Add inferred categories to encoding
        new_cols[col] = [f"{col}_{cat}" for cat in categories]
    # Handle bitmask columns
    for col, bits in encoding["bitmask"].items():
        if not bits:
            bit_data = df[col].astype(int)
            max_val = bit_data.max()
            nbits = int(np.ceil(np.log2(max_val + 1))) if max_val > 0 else 0
            # Check which bits are active across the dataset
            bits = [n for n in range(nbits) if ((bit_data // 2**n) % 2).any()]
            encoding["bitmask"][col] = bits  # Add inferred bits to encoding
        new_cols[col] = [f"{col}_bit_{bit}" for bit in bits]

    encoding["encoded_columns"] = new_cols
    encoding["removed_columns"] = list(new_cols.keys())
    return encoding


def _calculate_scale_from_data(cfg: Config, df: pd.DataFrame, encoding: dict):
    """
    Calculate normalization directly from dataset values.
    """
    encoded_cols = set([c for cl in encoding["encoded_columns"] for c in cl])
    removed_cols = set(encoding["removed_columns"])
    all_cols = set(df.columns).union(encoded_cols) - removed_cols
    scale = {
        k: {"encoded": False, "log_norm": False, "offset": 0.0, "scale": 1.0} for k in all_cols
    }
    for var in all_cols:
        if var in encoded_cols:
            scale[var]["encoded"] = True
            continue
        if var in cfg.log_norm_cols:
            scale[var]["log_norm"] = True
            scale[var]["offset"] = np.nanmean(np.log1p(df[var]))
        elif var in cfg.range_norm_cols:
            min_val = float(df[var].min())
            max_val = float(df[var].max())
            scale[var]["offset"] = min_val
            scale[var]["scale"] = max_val - min_val if max_val > min_val else 1.0
        else:  # z-score
            mean = float(df[var].mean())
            std = float(df[var].std())
            scale[var]["offset"] = mean
            scale[var]["scale"] = std if std > 1e-9 else 1.0
    return scale


# TODO Allow different sets of features, but check what is already in cache and append as needed.
# Each var is normalized separately, so there is no conflict with extra vars.
def config_hash(cfg: Config) -> str:
    """Return a unique hash based on a subset of fields from a Pydantic config object."""
    fields_to_hash = [
        "data_root",
        "zarr_dir",
        "attributes_file",
        "train_basin_file",
        "features",
        "dynamic_encoding",
        "static_encoding",
        "log_norm_cols",
        "no_norm_cols",
        "range_norm_cols",
        "train_date_range",
        "add_rolling_means",
        "clip_feature_range",
        "value_filter",
    ]
    # Extract selected field values into dict
    selected = {f: getattr(cfg, f) for f in fields_to_hash if hasattr(cfg, f)}
    # Per-target lists can pull extra basins into the cached set (see
    # _add_per_target_basins), so they must key the cache. Added only when present so
    # configs that do not use the feature keep the hash they already have.
    if getattr(cfg, "target_basin_files", None):
        selected["target_basin_files"] = cfg.target_basin_files
    # Fallback sources are cached as extra columns and change which dates are indexed.
    if getattr(cfg, "rereference_to_median", None):
        selected["rereference_to_median"] = sorted(cfg.rereference_to_median)
        selected["rereference_window"] = [str(d) for d in cfg.rereference_window]
    # Drop lists are keyed by CONTENT (a path alone would silently reuse a stale cache).
    if getattr(cfg, "target_drop_masks", None):
        selected["target_drop_masks"] = {v: _file_sha256(f) for v, f in cfg.target_drop_masks.items()}
    if getattr(cfg, "target_scale_override", None):
        selected["target_scale_override"] = cfg.target_scale_override
    if getattr(cfg, "obs_noise", None):
        selected["obs_noise"] = {t: n.model_dump() for t, n in cfg.obs_noise.items()}
    if getattr(cfg, "input_history_days", None):
        selected["input_history_days"] = int(cfg.input_history_days)
    if getattr(cfg, "fallback_targets", None):
        selected["fallback_targets"] = {
            t: fb.model_dump() for t, fb in cfg.fallback_targets.items()
        }
    # External feature store: keyed by path, variables AND the store's manifest (version, write range,
    # reference), so a rebuilt or extended store gets a new cache instead of silently reusing a stale one.
    if getattr(cfg, "extra_dynamic_vars", None):
        man = Path(cfg.extra_dynamic_store) / "manifest.json"
        selected["extra_dynamic"] = {
            "store": str(cfg.extra_dynamic_store), "vars": list(cfg.extra_dynamic_vars),
            "manifest": _file_sha256(man) if man.exists() else None,
        }
    # Convert to JSON string for deterministic serialization
    serialized = json.dumps(selected, sort_keys=True, default=str)
    # Compute SHA256 hash
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


class DynamicCacheManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        base = cfg.cache_dir if cfg.cache_dir is not None else cfg.zarr_dir / "_cache4"
        self.cache_dir = base / config_hash(cfg)
        _loc = os.environ.get("SWOTML_LOCAL_CACHE")
        if _loc and (Path(_loc) / config_hash(cfg)).is_dir():
            print(f"SWOTML_LOCAL_CACHE: using the node-local copy {Path(_loc) / config_hash(cfg)} instead of {self.cache_dir}")
            self.cache_dir = Path(_loc) / config_hash(cfg)
        self.train_stats = None
        print(f"Caches will be stored at: {self.cache_dir}")

    def create_cache(
        self, subset: str | DataSubset, overwrite: bool = False, num_workers: int = 16
    ):
        """
        Create cache with parallel processing, combining normalization,
        target identification, and index generation in one pass.
        """
        if isinstance(subset, str):
            subset = DataSubset(subset)

        subset_cache_dir = self.cache_dir / subset.name
        subset_cache_dir.mkdir(exist_ok=True, parents=True)

        # Check for existing index to avoid redundant work
        index_dir = subset_cache_dir / "_indices" / f"sequence={self.cfg.sequence_length}"
        if not overwrite and (index_dir / "sample_list.npy").exists():
            print(f"Indices for {subset.name} already exist. Skipping.")
            return subset_cache_dir

        # Pre-compute normalization stats once
        if self.train_stats is None:
            self.train_stats = get_train_ds_stats(self.cfg, static=False)

        features = copy.deepcopy(self.cfg.features)
        encoding = self.train_stats["d_encoding"]
        scale = self.train_stats["d_scale"]
        dynamic_columns = set([vv for v in features.dynamic.values() for vv in v])
        all_config_columns = dynamic_columns.union(set(features.target))
        all_config_columns |= {fb.source for fb in (self.cfg.fallback_targets or {}).values()}
        all_config_columns |= set(_obs_noise_columns(self.cfg))

        self._process_subset(
            subset=subset,
            subset_cache_dir=subset_cache_dir,
            overwrite=overwrite,
            num_workers=num_workers,
            features=features,
            encoding=encoding,
            scale=scale,
            all_config_columns=all_config_columns,
            index_dir=index_dir,
        )

        print("All caches created and indexed.")

        return subset_cache_dir

    def _process_subset(
        self,
        subset: DataSubset,
        subset_cache_dir: Path,
        overwrite: bool,
        num_workers: int,
        features,
        encoding,
        scale,
        all_config_columns,
        index_dir: Path,
    ):
        """Process a single subset: cache data and generate indices in one pass."""

        basin_subset_dict = _get_basin_subbasin_dict(self.cfg, subset)
        basins = sorted(list(basin_subset_dict.keys()))

        time_slice = _get_time_slice(self.cfg, subset)
        seq_len = np.timedelta64(self.cfg.sequence_length, "D")
        min_valid_date = np.datetime64(time_slice.start) + seq_len
        target_vars = self.cfg.features.target

        globally_found_columns = set()
        all_basin_indices = {}
        written_zarrs = 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    self._process_single_basin_with_indices,
                    basin_id=basin_id,
                    cache_dir=subset_cache_dir,
                    time_slice=time_slice,
                    features=features,
                    encoding=encoding,
                    scale=scale,
                    all_config_columns=all_config_columns,
                    min_valid_date=min_valid_date,
                    target_vars=target_vars,
                    subset=subset,
                    overwrite=overwrite,
                ): basin_id
                for basin_id in basins
            }

            with tqdm(total=len(futures), desc=f"Caching {subset.name}") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    basin_id = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            found_columns, valid_date_ints, was_written = result
                            globally_found_columns.update(found_columns)
                            if valid_date_ints is not None and len(valid_date_ints) > 0:
                                all_basin_indices[basin_id] = valid_date_ints
                            if was_written:
                                written_zarrs += 1
                    except Exception as e:
                        warnings.warn(f"Failed {basin_id}: {e}")
                    pbar.update(1)

        if written_zarrs > 0:
            print(f"Wrote {written_zarrs} new Zarr files.")

        # GUARD: per-basin failures above are caught and warned, so a build in which every
        # basin failed is indistinguishable from a good one - the caller proceeds, the
        # sampler gets an empty index, and __iter__ spins on `while True` with no error
        # anywhere. Refuse to hand back a cache that indexes nothing.
        if not all_basin_indices:
            raise RuntimeError(
                f"Cache build for {subset.name} produced no usable data across "
                f"{len(basins)} basins - every basin failed or yielded no valid dates. "
                f"See the per-basin warnings above for the underlying error."
            )

        self._save_indices(index_dir, basins, all_basin_indices)

    def _process_single_basin_with_indices(
        self,
        basin_id: str,
        cache_dir: Path,
        time_slice,
        features,
        encoding,
        scale,
        all_config_columns,
        min_valid_date,
        target_vars,
        subset: DataSubset,
        overwrite: bool,
    ):
        basin_cache_path = cache_dir / basin_id
        was_written = False

        try:
            # Logic: If Zarr exists and not overwriting, load lazy metadata only
            if basin_cache_path.exists() and not overwrite:
                ds = xr.open_zarr(basin_cache_path, consolidated=True)
            else:
                # Full processing required
                ds = self._load_and_normalize_basin(
                    basin_id, time_slice, features, encoding, scale, all_config_columns
                )
                if ds is None:
                    return None
                ds.to_zarr(basin_cache_path, mode="w", consolidated=True)
                was_written = True

            found_columns = set(ds.data_vars)
            valid_date_ints = self._extract_valid_dates(ds, min_valid_date, target_vars, subset)

            return found_columns, valid_date_ints, was_written

        except Exception as e:
            warnings.warn(f"Error in {basin_id}: {e}")
            return None

    def _save_indices(self, index_dir: Path, basins: list, all_basin_indices: dict):
        index_dir.mkdir(exist_ok=True, parents=True)

        basin_to_int = {b: i for i, b in enumerate(basins)}
        sample_list_ints = []
        basin_index_map = defaultdict(list)
        global_idx = 0

        for basin in basins:
            if basin not in all_basin_indices:
                continue
            b_idx = basin_to_int[basin]
            for d_int in all_basin_indices[basin]:
                sample_list_ints.append([b_idx, d_int])
                basin_index_map[basin].append(global_idx)
                global_idx += 1

        np.save(index_dir / "sample_list.npy", np.array(sample_list_ints, dtype=np.int32))

        with open(index_dir / "basin_lookup.json", "w") as f:
            json.dump(basins, f)

        with open(index_dir / "basin_index_map.json", "w") as f:
            json.dump(basin_index_map, f)

        print(f"✅ Saved {global_idx} indices to {index_dir}")

    def _load_and_normalize_basin(
        self,
        basin_id: str,
        time_slice: slice,
        features: Features,
        encoding: dict,
        scale: dict,
        all_config_columns: set,
    ) -> xr.Dataset:
        # Open with consolidation for faster metadata reads
        ds = xr.open_zarr(
            self.cfg.zarr_dir / basin_id,
            consolidated=True if (self.cfg.zarr_dir / basin_id / ".zmetadata").exists() else False,
        )

        # External feature store: same group name and subbasin coordinates; aligned on (date, subbasin).
        # The requested window must lie inside the store's written range (checked here, so a short store
        # fails loudly at cache build instead of feeding NaN inputs).
        if getattr(self.cfg, "extra_dynamic_vars", None):
            ex = xr.open_zarr(Path(self.cfg.extra_dynamic_store) / basin_id, consolidated=True)
            ex = ex[list(self.cfg.extra_dynamic_vars)].astype("float32")
            ex = ex.assign_coords(subbasin=ex.subbasin.values.astype(ds.subbasin.dtype))
            if not np.array_equal(np.sort(ex.subbasin.values), np.sort(ds.subbasin.values)):
                raise ValueError(f"{basin_id}: extra store subbasins differ from {self.cfg.zarr_dir}")
            t0, t1 = np.datetime64(pd.Timestamp(time_slice.start)), np.datetime64(pd.Timestamp(time_slice.stop))
            if ex.date.values[0] > t0 or ex.date.values[-1] < t1:
                raise ValueError(f"{basin_id}: extra store covers {ex.date.values[0]}..{ex.date.values[-1]}, "
                                 f"window needs {t0}..{t1}; extend it (build_longmem.py --write-start/--write-end)")
            ex = ex.reindex(subbasin=ds.subbasin.values).reindex(date=ds.date.values)
            ds = ds.assign({v: ex[v].transpose(*ds[list(ds.data_vars)[0]].dims) for v in self.cfg.extra_dynamic_vars})

        # SWOT QC drop lists: set the listed observations to NaN on the FULL record, so the
        # re-referencing median below is computed from kept observations only.
        for v, f in (getattr(self.cfg, "target_drop_masks", None) or {}).items():
            if v not in ds:
                continue
            rows = _drop_mask_table(f).get(basin_id)
            if rows is None or rows.empty:
                continue
            subs = pd.Index(ds.subbasin.values.astype(str))
            dates = pd.Index(ds.date.values.astype("datetime64[D]"))
            si, di = subs.get_indexer(rows.subbasin), dates.get_indexer(rows.date)
            k = (si >= 0) & (di >= 0)
            drop = np.zeros((len(dates), len(subs)), dtype=bool)
            drop[di[k], si[k]] = True
            dm = xr.DataArray(drop, coords={"date": ds.date, "subbasin": ds.subbasin}, dims=["date", "subbasin"])
            ds[v] = ds[v].where(~dm)

        # Select time slice
        ds_full = ds
        ds = ds.sel(date=time_slice)

        # Observation-noise columns (physical 1-sigma of each target / fallback observation).
        on = getattr(self.cfg, "obs_noise", None) or {}
        if on:
            raw_static = None
            for t, n in on.items():
                srcs = [(f"{t}__sigma", n.var)] + ([(f"{t}__fb__sigma", n.fb_var)] if n.fb_var else [])
                for name, var in srcs:
                    if var not in ds:
                        continue
                    sig = ds[var].astype("float32")
                    if n.static_multiplier:
                        if raw_static is None:
                            raw_static = pd.read_parquet(self.cfg.attributes_file)
                            raw_static = raw_static[raw_static.index.get_level_values("basin") == basin_id].droplevel("basin")
                            raw_static.index = raw_static.index.astype(str)
                        mult = raw_static[n.static_multiplier].reindex(ds.subbasin.values.astype(str)).values
                        sig = sig * xr.DataArray(mult.astype("float32"), coords={"subbasin": ds.subbasin}, dims=["subbasin"])
                    ds[name] = abs(sig)

        # Re-reference listed variables to each node's median over the SWOT record (see Config).
        rw = slice(*getattr(self.cfg, "rereference_window", [None, None]))
        for v in getattr(self.cfg, "rereference_to_median", None) or []:
            if v in ds_full and v in ds:
                med = ds_full[v].sel(date=rw).load().median("date", skipna=True)
                ds[v] = ds[v] - med

        # v21 input history: the window may start before train_date_range[0]; nothing before that day is a target.
        hist = getattr(self.cfg, "input_history_days", None)
        if hist and pd.Timestamp(time_slice.start) < pd.Timestamp(self.cfg.train_date_range[0]):
            t0 = np.datetime64(pd.Timestamp(self.cfg.train_date_range[0]))
            tvars = set(self.cfg.features.target) | {fb.source for fb in (self.cfg.fallback_targets or {}).values()}
            tvars |= set(_obs_noise_columns(self.cfg))
            for v in tvars:
                if v in ds:
                    ds[v] = ds[v].where(ds.date >= t0)

        # Trailing means are computed from the record BEFORE the window as well, so the
        # first day of the window already sees a full year (or month...) of history.
        specs = {n: vw for n, vw in _rolling_mean_specs(self.cfg).items() if vw[0] in ds_full}
        if specs:
            lead = np.timedelta64(max(w for _, w in specs.values()), "D")
            start = np.datetime64(pd.Timestamp(time_slice.start)) - lead
            hist = ds_full.sel(date=slice(start, time_slice.stop))
            for name, (var, w) in specs.items():
                ds[name] = (
                    hist[var].rolling(date=w, min_periods=max(1, int(0.8 * w))).mean().sel(date=ds.date)
                )

        # Annual cycle, for calendar-driven behaviour (reservoir rule curves).
        if getattr(self.cfg, "add_doy_features", False):
            dyn = [c for cols in features.dynamic.values() for c in cols if c in ds.data_vars]
            template = ds[dyn[0]].fillna(0.0) * 0.0          # (date, subbasin) in the cache's own dim order
            ang = 2 * np.pi * (ds.date.dt.dayofyear - 1) / 365.25
            ds["doy_sin"] = template + np.sin(ang)
            ds["doy_cos"] = template + np.cos(ang)

        # Find available columns
        available_columns = all_config_columns.intersection(set(ds.data_vars))
        ds = ds[list(available_columns)]

        # GUARD: a configured feature with no normalization stats is silently passed
        # through un-normalised, keeps encoding['chunks'] from the source zarr, and then
        # makes the cache write raise - which is caught per-basin as a warning, leaving an
        # empty cache and a trainer that spins forever on an empty sample index. That cost
        # ~20h of idle GPU when swvl1-4_mean were merged without re-running
        # preprocess/update_zarr_stats_conus.py.
        #
        # Checked HERE, not inside _normalize_data: _encode_data below expands categoricals
        # into one-hot columns (cls_cl_smj_*, tec_cl_smj_*, ...) that legitimately have no
        # stats of their own, so a check further down false-positives on ~220 of them.
        missing_stats = sorted(c for c in available_columns if c not in scale)
        if missing_stats:
            raise ValueError(
                f"No normalization stats for configured features {missing_stats}. "
                f"Run preprocess/update_zarr_stats_conus.py after any merge that adds variables."
            )

        # Apply encoding and normalization
        ds = self._clip_data(ds)
        ds, features = self._encode_data(ds, features, encoding)
        ds = _normalize_data(ds, scale)

        # Optimize chunking for the sequence length
        ds = ds.chunk(
            {
                "date": self.cfg.sequence_length,
                "subbasin": -1,  # Keep subbasin dimension in single chunk
            }
        )

        return ds

    def _clip_data(self, ds: xr.Dataset) -> xr.Dataset:
        return ds.assign(
            **{
                feat: ds[feat].where((ds[feat] >= vmin) & (ds[feat] <= vmax))
                for feat, (vmin, vmax) in self.cfg.clip_feature_range.items()
                if feat in ds
            }
        )

    def _encode_data(self, ds: xr.Dataset, features: Features, encoding: dict):
        """Optimized encoding without unnecessary conversions."""

        # Process categorical encodings
        for col, prescribed_cats in encoding["categorical"].items():
            if col not in ds.data_vars:
                continue

            prescribed_cols = [f"{col}_{cat}" for cat in prescribed_cats]

            # Direct numpy operations instead of dask dataframe conversion
            col_data = ds[col].values

            # Create one-hot encoded arrays directly
            encoded_arrays = {}
            for cat_col in prescribed_cols:
                cat_name = cat_col.replace(f"{col}_", "")
                encoded_arrays[cat_col] = xr.DataArray(
                    (col_data == cat_name).astype(float), dims=ds[col].dims, coords=ds[col].coords
                )

            # Update dataset
            ds = ds.drop_vars(col)
            for name, arr in encoded_arrays.items():
                ds[name] = arr

            # Update features
            for src, columns in features.dynamic.items():
                if col in columns:
                    features.dynamic[src].remove(col)
                    features.dynamic[src].extend(prescribed_cols)

        # Process bitmask encodings (keeping your efficient implementation)
        for col, prescribed_bits in encoding["bitmask"].items():
            if col not in ds.data_vars:
                continue

            x = ds[col]
            finite_mask = (
                da.isfinite(x.data) if isinstance(x.data, da.Array) else np.isfinite(x.data)
            )
            x_int = np.where(finite_mask, x.data, 0).astype(int)

            new_vars = {}
            for n in prescribed_bits:
                bit_arr_data = ((x_int // 2**n) % 2).astype(float)
                bit_arr_data = np.where(finite_mask, bit_arr_data, np.nan)

                new_var_name = f"{col}_bit_{n}"
                new_vars[new_var_name] = xr.DataArray(
                    bit_arr_data, dims=ds[col].dims, coords=ds[col].coords
                )

            # Update dataset and features
            encoded_columns = list(new_vars.keys())
            ds = ds.drop_vars(col)
            ds = ds.assign(**new_vars)

            for src, columns in features.dynamic.items():
                if col in columns:
                    features.dynamic[src].remove(col)
                    features.dynamic[src].extend(encoded_columns)

        return ds, features

    def _extract_valid_dates(
        self,
        ds: xr.Dataset,
        min_valid_date: np.datetime64,
        target_vars: list,
        subset: DataSubset,
    ) -> np.ndarray:
        """Extract valid sample dates from an in-memory dataset."""
        # Filter by minimum date
        valid_seq_mask = ds["date"] >= min_valid_date
        ds_filtered = ds.sel(date=valid_seq_mask)

        if subset in [DataSubset.train, DataSubset.test]:
            # Check target availability
            missing = [t for t in target_vars if t not in ds_filtered]
            if missing:
                return np.array([], dtype=np.int32)

            # A fallback source also makes a date usable, but it is optional: a basin without
            # the column is still judged on its primary targets, exactly as before.
            fb_vars = [v for v in _fallback_sources(self.cfg, subset) if v in ds_filtered]
            not_nan_arr = ~np.isnan(ds_filtered[list(target_vars) + fb_vars]).to_array()
            target_mask = not_nan_arr.any(dim=["subbasin", "variable"]).values
            valid_dates = ds_filtered["date"][target_mask].values
        else:
            valid_dates = ds_filtered["date"].values

        if len(valid_dates) == 0:
            return np.array([], dtype=np.int32)

        # Convert to integer days since epoch
        return valid_dates.astype("datetime64[D]").astype(np.int32)

    def _load_existing_indices(
        self,
        basins: set,
        cache_dir: Path,
        min_valid_date,
        target_vars,
        subset,
    ) -> dict:
        """Load index data for already-cached basins."""

        loaded_indices = {}
        for basin in tqdm(basins, desc="Loading existing indices", disable=self.cfg.quiet):
            try:
                ds = xr.open_zarr(cache_dir / basin, consolidated=True)
                valid_date_ints = self._extract_valid_dates(ds, min_valid_date, target_vars, subset)
                if len(valid_date_ints) > 0:
                    loaded_indices[basin] = valid_date_ints
            except Exception:
                continue
        return loaded_indices


def _read_basin_file(fp) -> dict:
    df = pd.read_csv(fp, dtype=str)
    if "basin" not in df.columns or "subbasin" not in df.columns:
        raise ValueError(f"Subbasin file {fp} must contain 'basin' and 'subbasin' columns.")

    return df.groupby("basin")["subbasin"].apply(set).to_dict()


def _active_fallbacks(cfg: Config, subset: DataSubset) -> dict:
    """cfg.fallback_targets restricted to those emitted in this subset."""
    subset = DataSubset(subset)
    return {
        t: fb
        for t, fb in (getattr(cfg, "fallback_targets", None) or {}).items()
        if subset.value in fb.subsets
    }


def _fallback_sources(cfg: Config, subset: DataSubset) -> list[str]:
    return sorted({fb.source for fb in _active_fallbacks(cfg, subset).values()})


def _get_target_subset_dicts(cfg: Config, subset: DataSubset) -> dict:
    """Per-target node lists from cfg.target_basin_files.

    A target named there is supervised on its own node set, so one target can be
    observed at nodes where the others are held out. Applied only to the train and
    validate subsets - the test subset keeps using test_basin_file for every target
    so held-out evaluation stays comparable across runs.
    """
    if not getattr(cfg, "target_basin_files", None):
        return {}
    if subset == DataSubset.validate and getattr(cfg, "valid_basin_file", None):
        return {}          # a dedicated validation list governs every target there
    if subset not in (DataSubset.train, DataSubset.validate):
        return {}
    return {t: _read_basin_file(fp) for t, fp in cfg.target_basin_files.items()}


def _add_per_target_basins(cfg: Config, basin_subset_dict: dict) -> None:
    """Load basins that appear only in a per-target list.

    Such a basin is added with an EMPTY base node set, so every ordinary target is
    masked inside it and only its per-target override supervises anything. That is
    what lets stage be learned inside a held-out basin while the discharge and
    storage of that basin stay hidden - the whole-basin equivalent of the node-level
    override used on the Columbia split.
    """
    if not getattr(cfg, "target_basin_files", None):
        return
    for fp in cfg.target_basin_files.values():
        for basin in _read_basin_file(fp):
            basin_subset_dict.setdefault(basin, set())


def _get_basin_subbasin_dict(cfg: Config, subset: DataSubset):
    read_file = _read_basin_file

    match subset:
        case DataSubset.train:
            basin_subset_dict = read_file(cfg.train_basin_file)
            _add_per_target_basins(cfg, basin_subset_dict)
        case DataSubset.validate:
            # A dedicated validation list makes this subset out-of-sample. It supplies the
            # node set for EVERY target, and per-target overrides are deliberately not
            # applied, so nothing supervised during training leaks into validation.
            if getattr(cfg, "valid_basin_file", None):
                basin_subset_dict = read_file(cfg.valid_basin_file)
            else:
                basin_subset_dict = read_file(cfg.train_basin_file)
                _add_per_target_basins(cfg, basin_subset_dict)
        case DataSubset.test:
            basin_subset_dict = read_file(cfg.test_basin_file)
        case DataSubset.predict:
            train = read_file(cfg.train_basin_file)
            test = read_file(cfg.test_basin_file)
            basin_subset_dict = {
                b: train.get(b, set()) | test.get(b, set()) for b in train.keys() | test.keys()
            }
        case _:
            raise ValueError(f"This data_subset ({subset}) is not implemented.")

    return basin_subset_dict


def _get_time_slice(cfg: Config, subset: DataSubset):
    match subset:
        case DataSubset.train:
            start, end = tuple(cfg.train_date_range)
            if getattr(cfg, "input_history_days", None):
                start = pd.Timestamp(start) - pd.Timedelta(days=int(cfg.input_history_days))
        case DataSubset.validate:
            start, end = tuple(cfg.validate_date_range)
        case DataSubset.test:
            start, end = tuple(cfg.test_date_range)
        case DataSubset.predict:
            ranges = [
                cfg.train_date_range,
                cfg.validate_date_range,
                cfg.test_date_range,
            ]
            start = min(r[0] for r in ranges)
            end = max(r[1] for r in ranges)

        case _:
            raise ValueError(f"This data_subset ({subset}) is not implemented.")
    return slice(start, end)


def _load_basin_graphs(cfg: Config, basins: list[str]) -> dict[str, np.ndarray]:
    """Loads all basin-specific graphs."""
    print("Loading basin graphs...", end="")

    with open(cfg.graph_network_file) as f:
        graph_json = json.load(f)
    G = nx.readwrite.json_graph.node_link_graph(graph_json, edges="edges")

    graphs = {}
    for nodes in nx.weakly_connected_components(G):
        subG = G.subgraph(nodes).copy()

        # Find outlet: node with no outgoing edges
        outlets = [n for n in subG.nodes if subG.out_degree(n) == 0]
        if len(outlets) != 1:
            raise ValueError(f"Expected exactly one outlet per basin, found {outlets}")

        basin_id = outlets[0]
        if basin_id not in basins:
            continue

        # Map subbasin IDs to consecutive indices
        subbasin_map = {node: i for i, node in enumerate(subG.nodes())}

        source_nodes, dest_nodes = [], []
        for source, dest in subG.edges:
            source_nodes.append(subbasin_map[source])
            dest_nodes.append(subbasin_map[dest])
        graphs[basin_id] = np.array([source_nodes, dest_nodes], dtype=np.int32)

    missing_basins = set(basins) - set(graphs.keys())
    if missing_basins:
        raise ValueError(f"Not all basins were found in graph file. {missing_basins=}")

    print("Done!")
    return graphs


def _encode_data(ds: xr.Dataset, feat_group: str, features: Features, encoding: dict):
    assert feat_group in ["dynamic", "static"]

    ds, features = _one_hot_encoding(ds, feat_group, features, encoding["categorical"])
    ds, features = _bitmask_expansion(ds, feat_group, features, encoding["bitmask"])

    return ds, features


def _one_hot_encoding(
    ds: xr.Dataset, feat_group: str, features: Features, cat_enc: dict[str, list[str]]
):
    for col, prescribed_cats in cat_enc.items():
        if col not in ds.data_vars:
            continue

        # Get the underlying dask array
        da_col = ds[col].data

        # Create a dictionary of new variables
        new_vars = {}
        for cat in prescribed_cats:
            # Broadcast comparison (lazy dask operation)
            # Use strict equality; assumes categories match exactly
            if isinstance(cat, str):
                # handle string categories if necessary, though usually encoded as int/float in zarr
                # Ideally, your input Zarr already has these as numerics or strict types
                pass

            mask = da_col == cat

            # Convert boolean mask to float (or int)
            new_var_name = f"{col}_{cat}"
            new_vars[new_var_name] = xr.DataArray(
                mask.astype(np.float32), dims=ds[col].dims, coords=ds[col].coords
            )

        # Merge new variables
        encoded_ds = xr.Dataset(new_vars)
        ds = ds.drop_vars(col)
        ds = xr.merge([ds, encoded_ds], compat="no_conflicts")

        # Update feature tracking (same logic as before)
        encoded_columns = list(encoded_ds.data_vars)
        if feat_group == "dynamic":
            for src, columns in features.dynamic.items():
                if col in columns:
                    features.dynamic[src].remove(col)
                    features.dynamic[src].extend(encoded_columns)
        else:
            features.static.remove(col)
            features.static.extend(encoded_columns)

    return ds, features


def _bitmask_expansion(
    ds: xr.Dataset, feat_group: str, features: Features, bitmask_enc: dict[str, list[int]]
):
    for col, prescribed_bits in bitmask_enc.items():
        if col not in ds.data_vars:
            continue

        # Operate directly on the lazy xarray.DataArray (which is a Dask array)
        x = ds[col]
        finite_mask = da.isfinite(x.data)
        x_int = da.where(finite_mask, x.data, 0).astype(int)

        new_vars = {}
        for n in prescribed_bits:
            # All these operations are lazy Dask operations
            bit_arr_data = ((x_int // 2**n) % 2).astype(float)
            bit_arr_data = da.where(finite_mask, bit_arr_data, np.nan)

            new_var_name = f"{col}_bit_{n}"
            new_vars[new_var_name] = xr.DataArray(
                bit_arr_data, dims=ds[col].dims, coords=ds[col].coords
            )

        encoded_ds = xr.Dataset(new_vars)
        ds = ds.drop_vars(col)
        ds = xr.merge([ds, encoded_ds], compat="no_conflicts")

        # Update the feature dicts (logic remains the same)
        encoded_columns = list(encoded_ds.data_vars)
        # print(f"BME: Removing {col}, adding {encoded_columns}")
        if feat_group == "dynamic":
            # Loop through the sources
            for src, columns in features.dynamic.items():
                if col in columns:
                    features.dynamic[src].remove(col)
                    features.dynamic[src].extend(encoded_columns)
        else:  # static
            features.static.remove(col)
            features.static.extend(encoded_columns)

    return ds, features


def _normalize_data(ds: xr.Dataset, scale=None):
    """
    Normalize the input data using log normalization for specified variables and standard normalization for others.

    Returns:
        ds: the input xarray dataset after normalization
        scale: A dictionary containing the 'offset', 'scale', and 'log_norm' for each variable.
    """
    for var in set(ds.data_vars).intersection(scale.keys()):
        scl = scale[var]
        if scl["encoded"]:
            continue
        elif scl["log_norm"]:
            ds[var] = (np.log1p(ds[var]) - scl["offset"]) / scl["scale"]
        else:
            if scl["scale"] == 0:
                ds[var] = ds[var] - scl["offset"]
            else:
                ds[var] = (ds[var] - scl["offset"]) / scl["scale"]

    return ds


class CachedBasinGraphDataset(Dataset):
    """
    DataLoader class for loading and preprocessing hydrological time series data.
    """

    def __init__(self, cfg: Config, cache_dir: Path, subset: str | DataSubset):
        self.cfg = cfg
        self.cache_dir = cache_dir

        if isinstance(subset, str):
            subset = DataSubset(subset)
        self.data_subset = subset

        self.features = copy.deepcopy(self.cfg.features)
        self.time_slice = _get_time_slice(cfg, subset)

        self.basin_subset_dict = _get_basin_subbasin_dict(cfg, subset)
        self.basins = list(self.basin_subset_dict.keys())
        self.subbasins = {x for s in self.basin_subset_dict.values() for x in s}
        self.target_subset_dict = _get_target_subset_dicts(cfg, subset)
        self._tmask_cache = {}

        train_stats = get_train_ds_stats(cfg)
        self.s_encoding = train_stats["s_encoding"]
        self.s_scale = train_stats["s_scale"]
        self.d_encoding = train_stats["d_encoding"]
        self.d_scale = train_stats["d_scale"]
        self._update_encoded_features()

        self.x_s, self.basin_subbasin_counts = self._load_attributes()
        self.graphs = self._load_basin_graphs()

        self.target = self.features.target
        self.fallbacks = _active_fallbacks(cfg, subset)
        self.basin_lookup, self.sample_list, self.basin_index_map = self._load_indices()

    def _load_indices(self):
        subset_name = self.data_subset.name
        index_dir = self.cache_dir / "_indices" / f"sequence={str(self.cfg.sequence_length)}"

        lookup_path = index_dir / "basin_lookup.json"
        list_path = index_dir / "sample_list.npy"
        map_path = index_dir / "basin_index_map.json"

        if not list_path.exists():
            raise FileNotFoundError(
                f"Index cache not found for {subset_name}. Run DynamicCacheManager."
            )

        print(f"Loading optimized indices for {subset_name}...", end="")

        # 1. Load Basin Lookup (List of strings)
        with open(lookup_path, "r") as f:
            basin_lookup = json.load(f)

        # 2. Load Sample List (Memory Mapped Integer Array)
        # mmap_mode='r' means it stays on disk and loads pages into RAM only when accessed.
        # This reduces initial RAM usage to near zero for this object.
        sample_list = np.load(list_path, mmap_mode="r")

        # 3. Load Index Map (for Sampler)
        with open(map_path, "r") as f:
            basin_index_map = json.load(f)

        print("Done!")
        return basin_lookup, sample_list, basin_index_map

    def __len__(self):
        """
        Returns the number of valid sequences in the dataset.
        """
        return len(self.sample_list)

    def _update_encoded_features(self):
        for old_col, new_cols in self.d_encoding["encoded_columns"].items():
            # Have to comb through the sources
            for source, source_cols in self.features.dynamic.items():
                if old_col in source_cols:
                    self.features.dynamic[source].remove(old_col)
                    self.features.dynamic[source].extend(new_cols)
                    break

    def _load_basin_graphs(self) -> dict[str, np.ndarray]:
        """Loads all basin-specific graphs."""
        print("Loading basin graphs...", end="")

        with open(self.cfg.graph_network_file) as f:
            graph_json = json.load(f)
        G = nx.readwrite.json_graph.node_link_graph(graph_json, edges="edges")

        graphs = {}
        for nodes in nx.weakly_connected_components(G):
            subG = G.subgraph(nodes).copy()

            # Find outlet: node with no outgoing edges
            outlets = [n for n in subG.nodes if subG.out_degree(n) == 0]
            if len(outlets) != 1:
                raise ValueError(f"Expected exactly one outlet per basin, found {outlets}")

            basin_id = outlets[0]
            if basin_id not in self.basins:
                continue

            # Force deterministic sort (alphanumeric)
            ordered_nodes = sorted(list(subG.nodes()))
            subbasin_map = {node: i for i, node in enumerate(ordered_nodes)}

            source_nodes, dest_nodes = [], []
            for source, dest in subG.edges:
                source_nodes.append(subbasin_map[source])
                dest_nodes.append(subbasin_map[dest])

            # Return BOTH the edges and the ordered node list to ensure safety
            graph_entry = {
                "edges": np.array([source_nodes, dest_nodes], dtype=np.int32),
                "nodes": ordered_nodes,
            }

            # Conditional Inflow Calculation
            if self.cfg.model_args.name == "river_lstm":
                num_nodes = len(ordered_nodes)
                inflow_indices = np.full((num_nodes, 2), -1, dtype=np.int32)
                inflow_count = np.zeros(num_nodes, dtype=np.int32)

                for s, d in zip(source_nodes, dest_nodes):
                    slot = inflow_count[d]
                    if slot < 2:
                        inflow_indices[d, slot] = s
                        inflow_count[d] += 1

                graph_entry["inflow_indices"] = inflow_indices
                graph_entry["inflow_mask"] = inflow_indices != -1

            graphs[basin_id] = graph_entry

        missing_basins = set(self.basins) - set(graphs.keys())
        if missing_basins:
            raise ValueError(f"Not all basins were found in graph file. {missing_basins=}")

        print("Done!")
        return graphs

    def _load_attributes(self) -> xr.Dataset:
        """
        Loads the basin attributes from a CSV file.

        Returns:
            xr.Dataset: An xarray dataset of attribute data with basin coordinates.
        """
        df = pd.read_parquet(self.cfg.attributes_file)
        # Trim to basin subset
        df = df[df.index.get_level_values("basin").isin(self.basins)]

        # Get the size of each basin (useful for graph packing later)
        basin_subbasin_counts = df.groupby("basin").apply(len).to_dict()

        df = df.droplevel("basin")

        # Encode and scale the data.
        ds = df.to_xarray()
        ds, self.features = _encode_data(ds, "static", self.features, self.s_encoding)
        ds = _normalize_data(ds, self.s_scale)

        return ds, basin_subbasin_counts

    def __getitem__(self, idx: int) -> dict:
        """Generate one batch of data."""
        basin_idx, date_int = self.sample_list[idx]

        basin = self.basin_lookup[basin_idx]
        subset_subbasins = self.basin_subset_dict[basin]

        # Retrieve the authoritative order from the graph loader
        graph_data = self.graphs[basin]
        expected_order = graph_data["nodes"]
        edges = graph_data["edges"]

        end_date = np.datetime64(int(date_int), "D")
        start_date = end_date - np.timedelta64(self.cfg.sequence_length - 1, "D")

        with xr.open_zarr(self.cache_dir / basin, consolidated=True) as basin_ds:
            date_slice = slice(start_date, end_date)
            ds = basin_ds.sel(date=date_slice)
            if ds.sizes["date"] != self.cfg.sequence_length:
                raise ValueError(f"{basin}: window ending {end_date} has {ds.sizes['date']} days, not "
                                 f"sequence_length {self.cfg.sequence_length} (cache does not cover the window)")

            # Reindex the Zarr data to match the Graph indices
            ds = ds.reindex(subbasin=expected_order)
            # Now this list is guaranteed to match indices 0, 1, 2... of the edge
            subbasins = list(ds.coords["subbasin"].values)

            # NaN padding
            # Identify all dynamic and target columns that are required for this sample.
            all_dynamic_cols = [col for cols in self.features.dynamic.values() for col in cols]
            all_needed_cols = set(all_dynamic_cols + self.target)
            all_needed_cols |= {fb.source for fb in self.fallbacks.values()}
            all_needed_cols |= set(_obs_noise_columns(self.cfg))
            missing_cols = all_needed_cols - set(ds.data_vars)
            # If any columns are missing, create NaN placeholders and assign them to the dataset
            # We do it here, instead of during data loading, because we only need to pad this slice.
            if missing_cols:
                nan_arrays_to_add = {}
                for col in missing_cols:
                    nan_da = xr.DataArray(
                        data=np.full((len(ds.date), len(ds.subbasin)), np.nan, dtype=np.float32),
                        coords={"date": ds.date, "subbasin": ds.subbasin},
                        dims=["date", "subbasin"],
                    )
                    nan_arrays_to_add[col] = nan_da
                ds = ds.assign(**nan_arrays_to_add)

            dynamic = {}
            for source, columns in self.features.dynamic.items():
                source_cols = [c for c in columns if c in ds.data_vars]
                source_arr = (
                    ds[source_cols]
                    .to_array(dim="variable")
                    .transpose("date", "subbasin", "variable")
                    .compute()  # Explicitly compute
                )
                dynamic[source] = np.asarray(source_arr)  # Ensure numpy, not dask

            if not self.cfg.model_args.seq2seq:
                ds = ds.sel(date=[end_date])

            # Build an xarray condition along the "subbasin" coordinate
            subbasin_mask = xr.DataArray(
                [sb in subset_subbasins for sb in ds.subbasin.values],
                coords={"subbasin": ds.subbasin},
                dims=["subbasin"],
            )
            fb_masks = {}   # target -> the node mask its fallback must share
            if not self.target_subset_dict:
                fb_masks = {t: subbasin_mask for t in self.fallbacks}
                masked_ds = ds[self.target].where(subbasin_mask, other=np.nan)
                target_arr = (
                    masked_ds.to_array(dim="variable")
                    .transpose("date", "subbasin", "variable")
                    .compute()
                )
                # Slice out each target's own variable so it keeps a singleton last
                # dim of shape (date, subbasin, 1). Selecting by name (not position)
                # ensures e.g. y["dv_SWOTAE"] holds dv_SWOTAE and not discharge.
                target = {
                    t_name: np.asarray(target_arr.sel(variable=[t_name]))
                    for t_name in self.target
                }
            else:
                # A target with its own node list is masked separately, so it can be
                # supervised where the other targets are held out. Node order is fixed
                # per basin (reindexed to the graph order above), so cache the masks.
                target = {}
                for t_name in self.target:
                    per_target = self.target_subset_dict.get(t_name)
                    if per_target is None:
                        t_mask = subbasin_mask
                    else:
                        key = (basin, t_name)
                        if key not in self._tmask_cache:
                            nodes = per_target.get(basin, set())
                            self._tmask_cache[key] = np.fromiter(
                                (sb in nodes for sb in ds.subbasin.values),
                                dtype=bool,
                                count=ds.sizes["subbasin"],
                            )
                        t_mask = xr.DataArray(
                            self._tmask_cache[key],
                            coords={"subbasin": ds.subbasin},
                            dims=["subbasin"],
                        )
                    fb_masks[t_name] = t_mask
                    masked_t = ds[[t_name]].where(t_mask, other=np.nan)
                    target[t_name] = np.asarray(
                        masked_t.to_array(dim="variable")
                        .transpose("date", "subbasin", "variable")
                        .compute()
                    )

            # Fallback: where the target is NaN on a supervised node, take the source,
            # moved from its own z-score into the TARGET's z-score (physical units are
            # shared, e.g. MCM). Emitted under a separate key so the loss can weight it
            # and nothing that reads y[target] (metrics, inference) ever sees it.
            for t_name, fb in self.fallbacks.items():
                if t_name not in target:
                    continue
                src = ds[[fb.source]].where(fb_masks[t_name], other=np.nan)
                src = np.asarray(
                    src.to_array(dim="variable").transpose("date", "subbasin", "variable").compute(),
                    dtype=np.float32,
                )
                s, p = self.d_scale[fb.source], self.d_scale[t_name]
                src = ((src * s["scale"] + s["offset"]) - p["offset"]) / p["scale"]
                target[f"{t_name}__fb"] = np.where(
                    np.isnan(target[t_name]), src, np.nan
                ).astype(target[t_name].dtype)

            # Observation noise, only where the matching observation is present (z-units of t).
            for name, t_name in _obs_noise_columns(self.cfg).items():
                ref = f"{t_name}__fb" if name.endswith("__fb__sigma") else t_name
                if ref not in target:
                    continue
                sig = np.asarray(
                    ds[[name]].to_array(dim="variable").transpose("date", "subbasin", "variable").compute(),
                    dtype=np.float32,
                )
                target[name] = np.where(np.isnan(target[ref]), np.nan, np.nan_to_num(sig)).astype(target[ref].dtype)

        static = None
        if self.x_s is not None and self.features.static:
            static_ds = self.x_s.sel(subbasin=ds.subbasin)
            static_arr = (
                static_ds[self.features.static]
                .to_array(dim="variable")
                .transpose("subbasin", "variable")
                .compute()
            )
            static = np.asarray(static_arr)

        sample = GraphBatch(
            dynamic=dynamic,
            graph_edges=edges,
            static=static,
            y=target,
            inflow_indices=graph_data.get("inflow_indices"),
            inflow_mask=graph_data.get("inflow_mask"),
        )

        return basin, subbasins, end_date, sample

    def denormalize(self, x: Array, name: str, smearing_cf: float = 1) -> Array:
        """
        Denormalizes a feature or target by its name.

        Args:
            x (Array): Normalized data.
            name (str): Name of the variable to denormalize.

        Returns:
            np.ndarray: Denormalized data.
        """
        offset = self.d_scale[name]["offset"]
        scale = self.d_scale[name]["scale"]
        log_norm = self.d_scale[name]["log_norm"]

        if log_norm:
            log_space = x * scale + offset
            return jnp.expm1(log_space) * smearing_cf

        else:
            return x * scale + offset

    def denormalize_std(self, mu_normalized: Array, sigma_normalized: Array, name: str) -> Array:
        """
        Denormalizes standard deviation. For log-normal, requires both mu and sigma.
        """
        offset = self.d_scale[name]["offset"]
        log_norm = self.d_scale[name]["log_norm"]

        if log_norm:
            # mu and sigma are in log-space (normalized)
            mu_log = mu_normalized + offset
            sigma_log = sigma_normalized

            # Log-normal variance in original space
            # https://en.wikipedia.org/wiki/Log-normal_distribution
            var_original = (jnp.exp(sigma_log**2) - 1) * jnp.exp(2 * mu_log + sigma_log**2)
            return jnp.sqrt(var_original)
        else:
            # Z-score and range-norm case
            scale = self.d_scale[name]["scale"]
            return sigma_normalized * scale

    def denormalize_targets(self, y_normalized: dict[str, Array]) -> dict[str, Array]:
        """
        Denormalizes the target variable(s).
        Returns:
            dict[str, Array]: Denormalized target data.
        """
        y_denorm = {}
        for t_name, t_norm in y_normalized.items():
            y_denorm[t_name] = self.denormalize(t_norm, name=t_name)

        return y_denorm

    def normalize(self, x: Array, name: str) -> Array:
        offset = self.d_scale[name]["offset"]
        scale = self.d_scale[name]["scale"]
        log_norm = self.d_scale[name]["log_norm"]
        encoded = self.d_scale[name]["encoded"]

        if encoded:
            raise ValueError(f"Cannot normalize an encoded column: {name}")

        if log_norm:
            # Resolve the module (np or jnp) from the input x
            xp = jnp if isinstance(x, jnp.ndarray) else np
            return (xp.log1p(x) - offset) / scale
        elif scale != 0:
            return (x - offset) / scale
        else:
            return x - offset
