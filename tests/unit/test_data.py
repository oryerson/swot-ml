"""6. Data pipeline and leakage (vendor/swot_ml/src/data/cached_basingraphdataset.py).

* No gauge file is ever read by the dataset, data loader, model or training code: the source
  has no reference to gauge data, and the configuration names no gauge file.
  (The runtime check, with the paper's real data and an open-file audit, is in
  test_release_equivalence.py.)
* Re-referencing dv_SWOTAE / dv_staticPLD / dv_river to each node's median uses only the
  configured window (2023-01-01 to 2026-09-30) and only observations that the SWOT QC drop
  list keeps.
* input_history_days = 365: the training slice starts 365 days before 2023-01-01, inputs are
  kept there but targets are blanked, and every sample is a 365-day window that ends on a day
  with an observation.
* A sample: the 365 days ending on the sample date, nodes in graph order, targets masked to
  the supervised nodes (dv_SWOTAE to its own node list), and the fallback dv_staticPLD,
  moved into dv_SWOTAE's z-units, only where dv_SWOTAE is missing.
"""
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from conftest import REPO, SRC, c5_template
from config import DataSubset
from data.cached_basingraphdataset import (CachedBasinGraphDataset, DynamicCacheManager, _get_time_slice)

GAUGE = re.compile(r"testset|truth|resops|regime\d|gauge|usgs|nwis|agency", re.I)


# ------------------------------------------------------------------ gauge files are never read
def test_model_code_has_no_reference_to_gauge_data():
    """Check that no source file of the model code mentions gauge data.

    The folders data/, train/, config/, models/ and evaluate/ are searched. river_lstm.py
    (another model, not C5) is skipped: it has the word only in its comments.
    """
    hits = []
    for sub in ("data", "train", "config", "models", "evaluate"):
        for f in sorted((SRC / sub).rglob("*.py")):
            if f.name == "river_lstm.py":            # another model (not C5); "gauge" only in its comments
                continue
            for i, line in enumerate(f.read_text().splitlines(), 1):
                if GAUGE.search(line):
                    hits.append(f"{f.relative_to(SRC)}:{i}: {line.strip()}")
    assert not hits, "\n".join(hits)


def test_c5_configuration_names_no_gauge_file():
    """Check that no path in the C5 configuration points to gauge data."""
    t = c5_template()
    paths = [v for v in _walk(t) if isinstance(v, str) and "${SWOTGAT_WORK}" in v]
    assert len(paths) >= 10
    assert not [p for p in paths if GAUGE.search(p)], paths


def _walk(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from _walk(v)
    else:
        yield x


# ------------------------------------------------------------------ synthetic basin store
DATES = pd.date_range("2021-01-01", "2026-12-31", freq="D")
NODES = ["n1", "n2", "n3"]
WIN = (pd.Timestamp("2023-01-01"), pd.Timestamp("2026-09-30"))


def _store(tmp_path):
    rng = np.random.default_rng(0)
    T, N = len(DATES), len(NODES)
    base = rng.normal(0, 1, (T, N))
    inside = (DATES >= WIN[0]) & (DATES <= WIN[1])
    dv = np.where(inside[:, None], base + np.array([10.0, -3.0, 0.5]), 1e4)   # huge values outside the window
    dv[rng.random((T, N)) < .7] = np.nan
    ds = xr.Dataset({"dv_SWOTAE": (("date", "subbasin"), dv.astype("float32")),
                     "dv_staticPLD": (("date", "subbasin"), (dv * 0.9).astype("float32")),
                     "t2m_mean": (("date", "subbasin"), np.tile(np.arange(T, dtype="float32")[:, None], (1, N)))},
                    coords={"date": DATES, "subbasin": NODES})
    root = tmp_path / "store"; root.mkdir()
    ds.to_zarr(root / "B1", consolidated=True)
    # QC drop list: two dv_SWOTAE observations of n1 inside the window
    keep = np.flatnonzero(inside & ~np.isnan(dv[:, 0]))
    drop_idx = keep[:2]
    pd.DataFrame({"basin": "B1", "subbasin": "n1", "date": DATES[drop_idx]}).to_parquet(tmp_path / "drop.parquet")
    return root, ds, DATES[drop_idx]


def _cfg(root, tmp_path):
    return SimpleNamespace(
        zarr_dir=root, extra_dynamic_vars=None, obs_noise=None, rolling_mean_vars=[], add_doy_features=False,
        clip_feature_range={}, sequence_length=365,
        target_drop_masks={"dv_SWOTAE": tmp_path / "drop.parquet"},
        rereference_to_median=["dv_SWOTAE", "dv_staticPLD"],
        rereference_window=[datetime(2023, 1, 1), datetime(2026, 9, 30)],
        input_history_days=365,
        train_date_range=[datetime(2023, 1, 1), datetime(2026, 9, 30)],
        features=SimpleNamespace(target=["dv_SWOTAE"], dynamic={"era5": ["t2m_mean"]}),
        fallback_targets={"dv_SWOTAE": SimpleNamespace(source="dv_staticPLD", subsets=["train"])},
    )


def _load(tmp_path):
    root, raw, dropped = _store(tmp_path)
    cfg = _cfg(root, tmp_path)
    mgr = object.__new__(DynamicCacheManager); mgr.cfg = cfg
    ts = _get_time_slice(cfg, DataSubset.train)
    unit = {"encoded": False, "log_norm": False, "offset": 0.0, "scale": 1.0}
    out = mgr._load_and_normalize_basin("B1", ts, cfg.features, {"categorical": {}, "bitmask": {}},
                                        {v: dict(unit) for v in ("dv_SWOTAE", "dv_staticPLD", "t2m_mean")},
                                        {"dv_SWOTAE", "dv_staticPLD", "t2m_mean"})
    return cfg, mgr, ts, raw, dropped, out.load()


def test_rereference_uses_only_the_window_and_kept_observations(tmp_path):
    """Check that re-referencing to the median uses only the window and the kept observations.

    The median is taken over 2023-01-01..2026-09-30 after the SWOT QC drop list is applied; the
    fallback source is re-referenced by its own median.
    """
    cfg, mgr, ts, raw, dropped, out = _load(tmp_path)
    r = raw.dv_SWOTAE.to_pandas()
    w = r.loc[WIN[0]:WIN[1]].copy()
    w.loc[dropped, "n1"] = np.nan                                   # the drop list applies before the median
    med = w.median()
    got = out.dv_SWOTAE.to_pandas()[NODES]
    inside = got.loc[WIN[0]:WIN[1]]
    exp = r.loc[WIN[0]:WIN[1]] - med
    exp.loc[dropped, "n1"] = np.nan
    np.testing.assert_allclose(inside.values, exp.values, rtol=1e-6, atol=1e-5)
    # the test can tell: a median over the whole record (with the 1e4 values outside the window) gives other values
    assert not np.allclose(inside.values, (r - r.median()).loc[WIN[0]:WIN[1]].values, equal_nan=True)
    # the fallback source is re-referenced with its own median
    pm = raw.dv_staticPLD.to_pandas().loc[WIN[0]:WIN[1]].median()
    np.testing.assert_allclose(out.dv_staticPLD.to_pandas()[NODES].loc[WIN[0]:WIN[1]].values,
                               (raw.dv_staticPLD.to_pandas().loc[WIN[0]:WIN[1]] - pm).values, rtol=1e-6, atol=1e-5)


def test_input_history_keeps_inputs_and_blanks_targets(tmp_path):
    """Check that the training slice starts 365 days before the training range.

    Forcing is kept in that input history; the target and the fallback source are NaN there.
    """
    cfg, mgr, ts, raw, dropped, out = _load(tmp_path)
    assert pd.Timestamp(ts.start) == pd.Timestamp("2022-01-01") and pd.Timestamp(ts.stop) == WIN[1]
    d = pd.DatetimeIndex(out.date.values)
    assert d[0] == pd.Timestamp("2022-01-01") and d[-1] == WIN[1]
    hist = d < WIN[0]
    assert np.isnan(out.dv_SWOTAE.values[hist]).all() and np.isnan(out.dv_staticPLD.values[hist]).all()
    assert np.isfinite(out.t2m_mean.values[hist]).all()            # forcing is kept as input history


def test_valid_sample_dates(tmp_path):
    """Check which dates are samples.

    Sample end dates lie in the training range, every 365-day window lies in the cached slice, and
    a date is a sample exactly when the target or the fallback is observed on it.
    """
    cfg, mgr, ts, raw, dropped, out = _load(tmp_path)
    ends = np.asarray(mgr._extract_valid_dates(out, np.datetime64(ts.start) + np.timedelta64(365, "D"),
                                               ["dv_SWOTAE"], DataSubset.train)).astype("datetime64[D]")
    ends = pd.DatetimeIndex(ends)
    assert ends.min() >= WIN[0] and ends.max() <= WIN[1]
    # every window of 365 days lies inside the cached slice
    assert (ends - pd.Timedelta(days=364) >= pd.Timestamp(ts.start)).all()
    # a date is a sample iff the primary target or the fallback has an observation on it
    o = out.sel(date=slice(WIN[0], WIN[1]))
    has = (o.dv_SWOTAE.notnull() | o.dv_staticPLD.notnull()).any("subbasin").to_pandas()
    assert set(ends) == set(has.index[has.values])


# ------------------------------------------------------------------ one sample from the cache
def _cache_and_dataset(tmp_path):
    rng = np.random.default_rng(1)
    dates = pd.date_range("2022-01-01", "2023-12-31")
    order = ["n3", "n1", "n2"]                                       # zarr order differs from graph order
    T = len(dates)
    v = lambda: rng.normal(0, 1, (T, 3)).astype("float32")
    sw = v(); sw[rng.random((T, 3)) < .6] = np.nan
    ds = xr.Dataset({"t2m_mean": (("date", "subbasin"), np.tile(np.arange(T, dtype="float32")[:, None], (1, 3))),
                     "dv_river": (("date", "subbasin"), v()), "dv_SWOTAE": (("date", "subbasin"), sw),
                     "dv_staticPLD": (("date", "subbasin"), v())},
                    coords={"date": dates, "subbasin": order})
    (tmp_path / "cache").mkdir()
    ds.to_zarr(tmp_path / "cache" / "B1", consolidated=True)
    x_s = xr.Dataset({"a": ("subbasin", np.array([1., 2., 3.])), "lake_area_km2": ("subbasin", np.array([.1, .2, .3]))},
                     coords={"subbasin": NODES})
    d = object.__new__(CachedBasinGraphDataset)
    end = pd.Timestamp("2023-06-30")
    d.__dict__.update(
        cfg=SimpleNamespace(sequence_length=365, model_args=SimpleNamespace(seq2seq=True, name="sf_grnn"), obs_noise=None),
        cache_dir=tmp_path / "cache", basin_lookup=["B1"],
        sample_list=np.array([[0, np.datetime64(end.date(), "D").astype(np.int32)]]),
        basin_subset_dict={"B1": {"n1", "n2"}},                       # base supervised nodes
        target_subset_dict={"dv_SWOTAE": {"B1": {"n1", "n3"}}},       # dv_SWOTAE's own node list (C5 has one)
        _tmask_cache={},
        graphs={"B1": {"edges": np.array([[0, 1], [1, 2]], np.int32), "nodes": NODES}},
        features=SimpleNamespace(dynamic={"era5": ["t2m_mean"]}, static=["a", "lake_area_km2"]),
        target=["dv_river", "dv_SWOTAE"],
        fallbacks={"dv_SWOTAE": SimpleNamespace(source="dv_staticPLD")},
        d_scale={"dv_SWOTAE": {"offset": 1.0, "scale": 2.0}, "dv_staticPLD": {"offset": -0.5, "scale": 4.0}},
        x_s=x_s,
    )
    return d, ds, end


def test_sample_window_masks_and_fallback(tmp_path):
    """Check one sample of the real __getitem__ on a synthetic cache.

    It must hold the 365 days ending on the sample date, nodes in graph order, dv_river masked to
    the base node list and dv_SWOTAE to its own list, and the fallback only where dv_SWOTAE is
    missing, converted into dv_SWOTAE's z-units.
    """
    d, ds, end = _cache_and_dataset(tmp_path)
    basin, subs, end_date, s = d[0]
    assert basin == "B1" and list(subs) == NODES
    assert s.dynamic["era5"].shape == (365, 3, 1)
    raw = ds.reindex(subbasin=NODES).sel(date=slice(end - pd.Timedelta(days=364), end))
    np.testing.assert_array_equal(s.dynamic["era5"][..., 0], raw.t2m_mean.values)   # the 365 days ending on the sample date
    assert pd.Timestamp(end_date) == end
    river, swot, fb = s.y["dv_river"][..., 0], s.y["dv_SWOTAE"][..., 0], s.y["dv_SWOTAE__fb"][..., 0]
    # dv_river: base list {n1, n2}; dv_SWOTAE: its own list {n1, n3}
    np.testing.assert_array_equal(river[:, :2], raw.dv_river.values[:, :2]); assert np.isnan(river[:, 2]).all()
    assert np.isnan(swot[:, 1]).all()
    np.testing.assert_array_equal(swot[:, [0, 2]], raw.dv_SWOTAE.values[:, [0, 2]])
    # fallback: only where dv_SWOTAE is missing, on dv_SWOTAE's nodes, converted to dv_SWOTAE's z-units
    conv = ((raw.dv_staticPLD.values * 4.0 - 0.5) - 1.0) / 2.0
    want = np.where(np.isnan(raw.dv_SWOTAE.values), conv, np.nan); want[:, 1] = np.nan
    np.testing.assert_allclose(fb, want, rtol=1e-6, atol=1e-6)
    assert not np.any(~np.isnan(fb) & ~np.isnan(swot))             # never both on the same node and day
    np.testing.assert_array_equal(s.static, np.array([[1., .1], [2., .2], [3., .3]]))


def test_c5_sequence_settings():
    """Check C5's sequence settings.

    Sequence length 365, input history 365 days, the re-reference window and variables, and the
    training date range.
    """
    t = c5_template()
    assert t["sequence_length"] == 365 and t["input_history_days"] == 365
    assert t["rereference_window"] == ["2023-01-01T00:00:00", "2026-09-30T00:00:00"]
    assert sorted(t["rereference_to_median"]) == ["dv_SWOTAE", "dv_river", "dv_staticPLD"]
    assert t["train_date_range"] == ["2023-01-01", "2026-09-30"]
