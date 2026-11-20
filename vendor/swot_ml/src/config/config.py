import functools
import json
import warnings
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)
from pydantic.types import DirectoryPath, FilePath

from .add_assimilator import AssimConfig

# Import model argument classes from model_args.py
from .model_args import ModelArgs


class Features(BaseModel):
    dynamic: dict[str, list[str]]
    static: list[str] | None = None
    target: list[str]


class Encoding(BaseModel):
    categorical: dict[str, list[str]] = Field(default_factory=dict)
    bitmask: dict[str, list[int]] = Field(default_factory=dict)


class FallbackTarget(BaseModel):
    """A second, lower-tier source for a target, used only where the target itself is NaN.

    Example: dv_SWOTAE (SWOT storage anomaly from the observed per-pass area) is missing at
    ~2,100 lakes where SWOT still measured the level, and dv_staticPLD (the same level
    anomaly times the static PLD area) exists there. The dataset converts the source into
    the TARGET's normalisation and emits it as y["<target>__fb"]; the loss scores the
    target's head against it with `weight` relative to one primary observation.
    """
    source: str
    weight: float = Field(1.0, ge=0.0)
    # Subsets in which the fallback is emitted. Train only by default, so the logged
    # validation loss stays on the primary target and comparable with runs without it.
    subsets: list[str] = Field(default_factory=lambda: ["train"])


class ObsNoise(BaseModel):
    """Reported observation uncertainty (1 sigma, physical units) for a GMM target.

    The loss scores each observation against the mixture with every component's variance
    inflated by the observation's own variance, sigma_k^2 + sigma_obs^2, so a pass SWOT
    itself calls noisy pulls the prediction less than a clean one (instead of being deleted).
    sigma_obs = var [x static_multiplier], e.g. wse_u_river (m) x reach_area_km2 = MCM, the
    unit of dv_river. fb_var is the same for the target's fallback source.
    Emitted as y["<target>__sigma"] / y["<target>__fb__sigma"] in the target's z-units.
    """
    var: str
    fb_var: str | None = None
    static_multiplier: str | None = None


class StepKwargs(BaseModel):
    loss_name: Literal["mse", "mae", "huber", "nse", "spin_up_nse", "gmm_nll"] = "mse"
    target_weights: dict[str, float] = None
    corr_keys: dict = {}
    # Optional variability terms on the GMM mixture mean (see train/loss.py); empty = off.
    multiscale_keys: dict = {}
    varcal_keys: dict = {}
    # Optional per-observation NLL weights from input-feature values (see train/loss.py); empty = off.
    obs_weight_keys: dict = {}
    # Filled from Config.fallback_targets (target -> weight); not set in the YAML.
    fallback_weights: dict[str, float] = {}
    # seq2seq only: days at the start of each window excluded from the loss (see compute_loss).
    burn_in_steps: int = Field(0, ge=0)
    # gmm_nll only: beta-NLL (Seitzer et al. 2022). Each observation's NLL is weighted by stop_gradient(mixture variance)^beta,
    # normalised to mean 1 over the target's valid observations in the batch. 0 = off (plain NLL); 0.5 = mean gradient ~ 1/sigma.
    beta_nll: float = Field(0.0, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def check_deprecated_loss(cls, values):
        if isinstance(values, dict) and "loss" in values:
            warnings.warn("'loss' is deprecated. Use 'loss_name' instead.", DeprecationWarning)
            # Only override loss_name if it wasn't explicitly passed
            values.setdefault("loss_name", values["loss"])
        return values


class EarlyStopKwargs(BaseModel):
    patience: int = Field(0, gt=0)
    threshold: float = Field(..., gt=0, lt=1.0)


class ValueFilter(BaseModel):
    column: str
    operation: str
    value: Any
    feature_list: list[str] | None = None
    feature_group: str | None = None

    @model_validator(mode="before")
    @classmethod
    def feature_type(cls, values: dict):
        f_list = values.get("feature_list")
        f_group = values.get("feature_group")
        if f_list and f_group:
            raise ValueError("Specify only one of 'feature_list' or 'feature_group', not both.")
        if not f_list and not f_group:
            raise ValueError("Specify exactly one of 'feature_list' or 'feature_group'.")
        return values


class DataSubset(str, Enum):
    train = "train"  # Loss, gradients, and back propagation
    validate = "validate"  # Loss only
    test = "test"  # Error statistics only
    predict = "predict"  # Same as test but for sequences without test data


class Config(BaseModel):
    def to_json(self, path: str | Path, exclude_none: bool = True):
        """Dump the config to a JSON file."""
        cfg_out = self.model_copy(deep=True)
        json_str = cfg_out.model_dump_json(
            indent=4, exclude_none=exclude_none, exclude_unset=False, exclude={"cfg_path"}
        )
        with open(path, "w") as f:
            f.write(json_str)

    @classmethod
    def from_file(cls, file_path: Path | str) -> "Config":
        """
        Load the config from a YAML or JSON file. Automatically detects file type by extension.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        ext = file_path.suffix.lower()
        if ext in [".yml", ".yaml"]:
            with open(file_path, "r") as f:
                cfg_dict = yaml.safe_load(f)
        elif ext == ".json":
            with open(file_path, "r") as f:
                cfg_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file extension: {ext}")

        cfg = cls(**cfg_dict)
        cfg.cfg_path = file_path
        return cfg

    def __str__(self) -> str:
        """
        Provides a nicely formatted JSON string of all model attributes,
        """
        return self.model_dump_json(indent=4)

    # Data paths
    data_root: DirectoryPath
    zarr_dir: DirectoryPath
    cache_dir: Path | None = None
    attributes_file: FilePath
    train_basin_file: FilePath
    test_basin_file: FilePath
    graph_network_file: FilePath
    # Optional per-target node lists, e.g. {"d_wse_river": "basin_lists/.../stage_all.csv"}.
    # A target named here is supervised on ITS OWN node list during the train and validate
    # subsets instead of on train_basin_file, which lets one target be observed at nodes
    # where the others are held out. Targets not named here are unaffected. Ignored for the
    # test subset so held-out evaluation stays comparable across runs.
    target_basin_files: dict[str, Path] | None = None
    # Optional held-out node list for the validate subset. Without it, DataSubset.validate
    # reads train_basin_file, so the logged "Validation Loss" is computed on training
    # nodes - in-sample, and not a generalization estimate. When set, the validate subset
    # uses THIS list for every target and per-target overrides do not apply to it, so
    # validation is a clean out-of-sample signal.
    valid_basin_file: Path | None = None
    # Optional lower-tier sources per target, e.g. {"dv_SWOTAE": {"source": "dv_staticPLD"}}.
    # See FallbackTarget. Absent = off, and existing configs are unchanged.
    fallback_targets: dict[str, FallbackTarget] | None = None
    # Optional GMM targets predicted as a level change times a static area column,
    # e.g. {"dv_SWOTAE": "lake_area_km2"} (km2 x m = MCM). See
    # SparseFusionGRNN._apply_area_scaling. Model-side only: data and caches unchanged.
    area_scaled_targets: dict[str, str] | None = None
    # Dynamic variables that get trailing means over each window in add_rolling_means
    # (days), added as inputs named "<var>_rm<window>" in the variable's own source.
    rolling_mean_vars: list[str] = Field(default_factory=list)
    # Precomputed dynamic inputs from a standalone per-basin zarr store (same group names and subbasin
    # coordinates as zarr_dir), appended to features.dynamic[extra_dynamic_source]. They are used as stored:
    # already standardised, so they get identity scaling and are never renormalised on the training window.
    extra_dynamic_store: Path | None = None
    extra_dynamic_vars: list[str] = Field(default_factory=list)
    extra_dynamic_source: str = "era5"
    # Add day-of-year inputs doy_sin / doy_cos (annual cycle) to the first dynamic source.
    add_doy_features: bool = False
    # Variables re-referenced to each node's own median over `rereference_window` when loaded.
    # Jida's LakeSPvC dv_* series are referenced to each lake's FIRST valid pass (96-98% start at
    # exactly 0), which puts an arbitrary per-lake offset - including that pass's error - into the
    # training target. The window is the SWOT record itself, so runs with different train ranges
    # get identical targets. Model-independent: zarr data and normalisation stats are unchanged.
    rereference_to_median: list[str] = Field(default_factory=list)
    rereference_window: list[datetime] = Field(default_factory=lambda: [datetime(2023, 1, 1), datetime(2025, 12, 31)])
    # v21: forcing history before the train window (days); targets stay NaN before train_date_range[0]. None = off.
    input_history_days: int | None = Field(None, ge=0)
    # v21: seed the packing sampler's shuffle with model_args.seed (reproducible batch order). False = old unseeded order.
    seed_sampler: bool = False
    # v21 memory: 0 = old; 1 = unused per-day graph traces not stacked; 2 = 1 + block checkpoint of the time scan;
    # 3 = 2 + checkpointed embedding/heads. Same outputs and gradients at every level (test_v21_src.py).
    remat_level: int = Field(0, ge=0, le=3)
    # {variable: parquet with columns basin, subbasin, date}: those observations are set to NaN
    # when the basin is loaded, BEFORE re-referencing (SWOT quality-control drop lists built
    # offline, e.g. CONUS/preprocess/build_swot_qc_masks.py). The file's sha256 keys the cache.
    target_drop_masks: dict[str, Path] | None = None
    # {variable: {"offset": o, "scale": s}} replaces that variable's stored z-score statistics,
    # e.g. lake wse used as a target: the stored stats describe ABSOLUTE elevations (sd ~ 100s of
    # m) and would squash the re-referenced anomalies to ~0.
    target_scale_override: dict[str, dict[str, float]] | None = None
    # {target: ObsNoise}: observation-noise-aware GMM loss (see ObsNoise). gmm_nll only.
    obs_noise: dict[str, ObsNoise] | None = None

    # Data processing
    features: Features
    time_gaps: dict
    precomp_scaling: bool = True
    dynamic_encoding: Encoding = Field(default_factory=Encoding)
    static_encoding: Encoding = Field(default_factory=Encoding)
    log_norm_cols: list[str] = Field(default_factory=list)
    no_norm_cols: list[str] = Field(default_factory=list)
    range_norm_cols: list[str] = Field(default_factory=list)
    in_memory: bool = False

    # Data slicing
    train_date_range: list[datetime] = Field(..., min_length=2, max_length=2)
    validate_date_range: list[datetime] = Field(..., min_length=2, max_length=2)
    test_date_range: list[datetime] = Field(..., min_length=2, max_length=2)
    add_rolling_means: list[int] | None = None
    clip_feature_range: dict[str, list[float | None]] = Field(default_factory=dict)
    value_filters: list[ValueFilter] = Field(default_factory=list)
    exclude_target_from_index: list[str] = Field(default_factory=list)

    # DataLoader
    data_subset: DataSubset = DataSubset.train
    target_nodes_per_batch: int
    candidate_pool_size: int
    shuffle: bool = True
    num_workers: int = Field(..., ge=0)
    pin_memory: bool = False
    drop_last: bool = True
    backend: Literal["cpu", "gpu", "tpu"] | None = None
    num_devices: int | None = Field(None, gt=0)

    dataset_type: Literal["graph", "mcfli"] = "graph"
    batch_size: int | None = Field(None, gt=0)
    pad_size: int | None = Field(None, gt=0)

    # Model
    sequence_length: int
    model_args: ModelArgs = Field(discriminator="name")

    # Trainer
    max_training_steps: int = Field(..., gt=0)  # Required
    max_training_hours: int | None = Field(None, gt=0)
    log_every_n_steps: int = Field(..., gt=0)
    log_every_n_minutes: int | None = Field(None, gt=0)
    validate_every_n_steps: int | None = Field(None, gt=0)
    max_validation_steps: int | None = Field(None, gt=0)

    warmup_steps: int = Field(0, ge=0)
    initial_lr: float = Field(..., gt=0, lt=1.0)
    final_lr: float = Field(..., gt=0.0, lt=1.0)
    max_grad_norm: float | None = None
    step_kwargs: StepKwargs
    early_stop_kwargs: EarlyStopKwargs = None
    optimizer_name: Literal["adam", "adamw"] = "adam"
    weight_decay: float = Field(0.01, gt=0.0, lt=1.0)

    # Meta
    parameter_search_grid: dict = Field(default_factory=dict)

    # Outputs
    quiet: bool = True
    log: bool = True

    # Internal (not from YML)
    cfg_path: Path = Path()

    # MODEL / BEFORE
    @model_validator(mode="before")
    @classmethod
    def validate_paths(cls, values: dict):
        def resolve_path(attr_str):
            str_path = values.get(attr_str)
            if str_path:
                path = Path(str_path)
                # If already absolute, use as-is; otherwise resolve
                return path if path.is_absolute() else path.resolve()
            else:
                return None

        abs_paths = [
            "data_root",
            "zarr_dir",
        ]
        for attr in abs_paths:
            values[attr] = resolve_path(attr)

        rel_paths = [
            "attributes_file",
            "train_basin_file",
            "test_basin_file",
            "graph_network_file",
        ]
        for attr in rel_paths:
            values[attr] = values["data_root"] / values[attr]

        vbf = values.get("valid_basin_file")
        if vbf:
            vbf = Path(vbf)
            values["valid_basin_file"] = vbf if vbf.is_absolute() else values["data_root"] / vbf

        tdm = values.get("target_drop_masks")
        if tdm:
            values["target_drop_masks"] = {
                k: (Path(v) if Path(v).is_absolute() else values["data_root"] / v)
                for k, v in tdm.items()
            }

        tbf = values.get("target_basin_files")
        if tbf:
            values["target_basin_files"] = {
                k: (Path(v) if Path(v).is_absolute() else values["data_root"] / v)
                for k, v in tbf.items()
            }

        return values

    # MODEL / AFTER
    @model_validator(mode="after")
    def validate_fallback_targets(self):
        if not self.fallback_targets:
            self.step_kwargs.fallback_weights = {}
            return self
        for t, fb in self.fallback_targets.items():
            if t not in self.features.target:
                raise ValueError(f"fallback_targets: {t!r} is not in features.target")
            if fb.source in self.features.target:
                raise ValueError(f"fallback_targets: source {fb.source!r} is itself a target")
            # The conversion between the two normalisations is linear; a log-normalised
            # anomaly would not survive it (and signed anomalies must not be log-normed).
            bad = {t, fb.source} & set(self.log_norm_cols + self.range_norm_cols + self.no_norm_cols)
            if bad:
                raise ValueError(f"fallback_targets: {sorted(bad)} must be z-score normalised")
            unknown = set(fb.subsets) - {"train", "validate", "test"}
            if unknown:
                raise ValueError(f"fallback_targets: unknown subsets {sorted(unknown)}")
        self.step_kwargs.fallback_weights = {
            t: fb.weight for t, fb in self.fallback_targets.items()
        }
        return self

    @model_validator(mode="after")
    def validate_area_scaled_targets(self):
        for t, col in (self.area_scaled_targets or {}).items():
            if t not in self.features.target:
                raise ValueError(f"area_scaled_targets: {t!r} is not in features.target")
            if col not in (self.features.static or []):
                raise ValueError(f"area_scaled_targets: {col!r} is not in features.static")
            if t in set(self.log_norm_cols + self.range_norm_cols + self.no_norm_cols):
                raise ValueError(f"area_scaled_targets: {t!r} must be z-score normalised")
        return self

    @model_validator(mode="after")
    def validate_rolling_means(self):
        """Append "<var>_rm<w>" inputs next to each var. Idempotent, because a saved
        config.json already carries the expanded feature lists when it is reloaded."""
        if not self.rolling_mean_vars:
            return self
        if not self.add_rolling_means:
            raise ValueError("rolling_mean_vars needs add_rolling_means (window lengths in days)")
        for v in self.rolling_mean_vars:
            src = [s for s, cols in self.features.dynamic.items() if v in cols]
            if len(src) != 1:
                raise ValueError(f"rolling_mean_vars: {v!r} must be in exactly one dynamic source")
            cols = self.features.dynamic[src[0]]
            for w in self.add_rolling_means:
                if f"{v}_rm{w}" not in cols:
                    cols.append(f"{v}_rm{w}")
        return self

    @model_validator(mode="after")
    def validate_extra_dynamic(self):
        """Append the extra-store variables to their dynamic source. Idempotent (a saved config.json
        already carries them)."""
        if not self.extra_dynamic_vars:
            return self
        if self.extra_dynamic_store is None:
            raise ValueError("extra_dynamic_vars needs extra_dynamic_store")
        cols = self.features.dynamic.setdefault(self.extra_dynamic_source, [])
        for v in self.extra_dynamic_vars:
            if v not in cols:
                cols.append(v)
        return self

    @model_validator(mode="after")
    def validate_rereference(self):
        if not self.rereference_to_median:
            return self
        allowed = set(self.features.target) | {fb.source for fb in (self.fallback_targets or {}).values()}
        bad = [v for v in self.rereference_to_median if v not in allowed]
        if bad:
            raise ValueError(f"rereference_to_median: {bad} are not targets or fallback sources")
        if set(self.rereference_to_median) & set(self.log_norm_cols + self.range_norm_cols):
            raise ValueError("rereference_to_median: variables must be z-score normalised")
        return self

    @model_validator(mode="after")
    def validate_qc_options(self):
        for v, f in (self.target_drop_masks or {}).items():
            if not Path(f).is_file():
                raise ValueError(f"target_drop_masks: {f} does not exist")
        for v, o in (self.target_scale_override or {}).items():
            if set(o) != {"offset", "scale"} or not o["scale"] > 0:
                raise ValueError(f"target_scale_override[{v}] needs offset and a positive scale")
            if v in set(self.log_norm_cols + self.range_norm_cols + self.no_norm_cols):
                raise ValueError(f"target_scale_override: {v!r} must be z-score normalised")
        for t, n in (self.obs_noise or {}).items():
            if t not in self.features.target:
                raise ValueError(f"obs_noise: {t!r} is not in features.target")
            if self.step_kwargs.loss_name != "gmm_nll":
                raise ValueError("obs_noise needs step_kwargs.loss_name: gmm_nll")
            if n.fb_var and t not in (self.fallback_targets or {}):
                raise ValueError(f"obs_noise[{t}].fb_var set but {t!r} has no fallback")
        return self

    @model_validator(mode="after")
    def validate_doy_and_burn_in(self):
        if self.add_doy_features:
            first = next(iter(self.features.dynamic))
            for c in ("doy_sin", "doy_cos"):
                if not any(c in cols for cols in self.features.dynamic.values()):
                    self.features.dynamic[first].append(c)
        b = self.step_kwargs.burn_in_steps
        if b:
            if not self.model_args.seq2seq:
                raise ValueError("step_kwargs.burn_in_steps needs model_args.seq2seq (only the last day is scored otherwise)")
            if b >= self.sequence_length:
                raise ValueError(f"burn_in_steps ({b}) must be < sequence_length ({self.sequence_length})")
        return self

    @model_validator(mode="after")
    def validate_dataset_type(self):
        # This should probably be refactored to have pydantic models for each dataset.
        if self.dataset_type == "mcfli":
            required_keys = ["batch_size", "pad_size"]
            missing_keys = [k for k in required_keys if getattr(self, k) is None]
            if any(missing_keys):
                raise ValueError(f"incomplete config for MCFLIDataset. {missing_keys = }")

        return self

    @model_validator(mode="after")
    def validate_log_intervals(self):
        """
        Ensure at least one of log_every_n_steps or log_every_n_minutes is not inf.
        """
        steps_inf = self.log_every_n_steps == np.inf
        minutes_inf = self.log_every_n_minutes == np.inf
        if steps_inf and minutes_inf:
            raise ValueError(
                "At least one of log_every_n_steps or log_every_n_minutes must be set to a finite value."
            )
        return self

    @model_validator(mode="after")
    def validate_nse_requires_seq2seq(self):
        """
        Ensure that if loss_name == 'nse', then model_args.seq2seq == True.
        Only checks models that actually have the seq2seq attribute.
        """
        if self.step_kwargs.loss_name in ["nse", "spin_up_nse"]:
            if hasattr(self.model_args, "seq2seq"):
                if not self.model_args.seq2seq:
                    raise ValueError(
                        "When using loss_name='nse', model_args.seq2seq must be set to True."
                    )
            else:
                raise ValueError(
                    "loss_name='nse' requires a model type with a 'seq2seq' attribute AND seq2seq=True."
                )
        return self

    @model_validator(mode="after")
    def validate_time_gaps_keys(self):
        """
        Ensure that time_gaps dict has all the same keys as features.dynamic dict,
        and that all values in time_gaps are single booleans.
        """
        dynamic_keys = set(self.features.dynamic.keys())
        time_gaps_keys = set(self.time_gaps.keys())
        if dynamic_keys != time_gaps_keys:
            missing = dynamic_keys - time_gaps_keys
            extra = time_gaps_keys - dynamic_keys
            msg = []
            if missing:
                msg.append(f"Missing keys in time_gaps: {sorted(missing)}")
            if extra:
                msg.append(f"Extra keys in time_gaps: {sorted(extra)}")
            raise ValueError("time_gaps keys must match features.dynamic keys. " + "; ".join(msg))
        # Check that all values are single booleans
        for k, v in self.time_gaps.items():
            if not isinstance(v, bool):
                raise TypeError(f"time_gaps['{k}'] must be a boolean, got {type(v)}.")
        return self

    # FIELD / BEFORE
    @field_validator(
        "train_date_range",
        "validate_date_range",
        "test_date_range",
        mode="before",
    )
    def parse_time_slice(cls, v):
        if (not isinstance(v, list)) or (len(v) != 2):
            raise ValueError("date ranges must be a list of two elements (start, end).")
        # Convert to datetime objects
        t_start = datetime.fromisoformat(v[0])
        t_end = datetime.fromisoformat(v[1])
        return [t_start, t_end]

    @field_validator("clip_feature_range", mode="before")
    def process_clip_feature_range(cls, v):
        if not v:
            return {}
        processed_ranges = {}
        for key, range_list in v.items():
            if (not isinstance(range_list, list)) or (len(range_list) != 2):
                raise ValueError(f"Each clip range for '{key}' must have exactly 2 elements.")
            lower = -np.inf if range_list[0] is None else range_list[0]
            upper = np.inf if range_list[1] is None else range_list[1]
            processed_ranges[key] = [lower, upper]
        return processed_ranges

    # FIELD / AFTER
    @field_validator("step_kwargs", mode="after")
    def validate_step_kwargs(cls, v: StepKwargs, info):
        cfg = info.data  # Access the entire config data

        # Convert target_weights dict to list based on features.target order
        if v.target_weights is None:
            v.target_weights = {}

        if isinstance(v.target_weights, dict):
            v.target_weights = {
                target: v.target_weights.get(target, 1) for target in cfg["features"].target
            }
        else:
            raise TypeError(f"target_weights must be a dict or None, got {type(v.target_weights)}.")

        return v

    # CONFIG MANIPULATION
    def add_assimilation(self, assim: AssimConfig):
        """
        Inserts new feature groups, encodings, time gaps, etc. based on the assimilation target
        """
        self.features.dynamic.update(assim.new_features)
        self.dynamic_encoding.categorical.update(assim.categorical_encoding)
        self.dynamic_encoding.bitmask.update(assim.bitmask_encoding)
        for new_group in assim.new_features.keys():
            self.time_gaps[new_group] = True

        self.model_args.assim_sizes.update(assim.model_args)

        self.max_training_steps += assim.additional_steps

        # Hours do not accumulate from saved states.
        self.max_training_hours = assim.additional_hours

    def rgetattr(self, attr: str, *args):
        """
        Recursively get attribute from the instance using dot notation.
        """

        def _getattr(obj, attr):
            return getattr(obj, attr, *args)

        return functools.reduce(_getattr, [self] + attr.split("."))

    def rsetattr(self, attr: str, value: Any):
        """
        Recursively set attribute on the instance using dot notation.
        """
        pre, _, post = attr.rpartition(".")
        return setattr(self.rgetattr(pre) if pre else self, post, value)

    def update_from_grid(self, idx: int) -> "Config":
        param_dict = self.parameter_search_grid
        if param_dict == {}:
            raise AttributeError("param_search_dict is empty. Cannot update from grid.")

        def flatten_param_dict(d: dict, prefix=""):
            for k, v in d.items():
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    yield from flatten_param_dict(v, full_key)
                else:
                    yield full_key, v

        dotted_keys, value_set = zip(*flatten_param_dict(param_dict))

        # Compute the total number of combinations
        sizes = [len(v) for v in value_set]
        n_total = np.prod(sizes)
        if idx < 0 or idx >= n_total:
            raise IndexError(f"Index {idx} is out of range for grid of size {n_total}")

        # Do not change the seed! This ensures deterministic grid shuffling between runs.
        rng = np.random.default_rng(42)
        # Shuffle the grid indices (permutation of size n_total) and select our index
        grid_idx = rng.permutation(n_total)[idx]

        # Convert grid_idx to N-dimensional indices for each parameter
        unravel = np.unravel_index(grid_idx, sizes)
        new_values = [v[i] for v, i in zip(value_set, unravel)]

        for attr, value in zip(dotted_keys, new_values):
            self.rsetattr(attr, value)
