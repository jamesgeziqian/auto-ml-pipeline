"""Configuration loading, defaults, merging, and validation."""

from __future__ import annotations

import codecs
import copy
import csv as csv_module
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


class ConfigError(ValueError):
    """Raised when a pipeline configuration is invalid."""


DEFAULTS: Dict[str, Any] = {
    "project": {
        "name": "auto_ml_run",
        "work_dir": "runs/auto_ml_run",
        "random_seed": 42,
        "log_level": "INFO",
        "heartbeat_seconds": 15,
        "overwrite": False,
        "reuse_from": None,
        "reuse_mode": "link",
    },
    "data": {
        "feature_paths": [],
        "label_path": None,
        "feature_key": "id",
        "label_key": "id",
        "label_column": "target",
        "concat_mode": "column",
        "split_strategy": "random",
        "temporal_split": {
            "time_column": None,
            "date_format": None,
            "timezone": None,
            "oot_start": None,
            "oot_quantile": 0.5,
            "valid_fraction": 0.25,
            "min_positive_count": None,
            "max_positive_count": None,
        },
        "include_label_features": True,
        "drop_columns": [],
        "deduplicate_columns": True,
        "column_block_size": 32,
        "max_source_columns_in_memory": 2000,
        "wide_block_size": 512,
        "wide_row_group_size": 5000,
        "cache_numeric_dtype": "float64",
        "ingest_target_chunk_bytes": 268435456,
        "sample_rows": 2000,
        "max_rows": None,
        "train_fraction": 0.7,
        "valid_fraction": 0.15,
        "test_fraction": 0.15,
        "stratify": True,
        "task": "auto",
        "csv": {
            "sep": ",",
            "encoding": "utf-8",
            "chunksize": 2000,
            "engine": "pandas",
            "arrow_block_size": 67108864,
            "arrow_use_threads": True,
            "quoting": "minimal",
            "quotechar": '"',
            "doublequote": True,
            "escapechar": None,
            "na_values": ["", "NA", "NaN", "nan", "null", "None"],
            "compression": "infer",
        },
    },
    "type_inference": {
        "numeric_threshold": 0.95,
        "datetime_threshold": 0.98,
        "detect_datetime": False,
        "identifier_unique_ratio": 0.98,
        "identifier_min_unique": 100,
    },
    "filtering": {
        "max_nan_ratio": 0.98,
        "stats_sample_rows": None,
        "min_unique": 2,
        "max_unique_ratio": 0.98,
        "min_abs_pearson": 0.0,
        "min_iv": 0.0,
        "max_iv": None,
        "max_psi": 0.5,
        "max_features": 2000,
        "max_pairwise_corr": None,
        "iv_bins": 10,
        "psi_bins": 10,
        "categorical_top_k": 50,
        "temporal_psi": {
            "enabled": False,
            "time_path": None,
            "time_key": None,
            "time_column": None,
            "date_format": None,
            "timezone": None,
            "reference_start": None,
            "reference_end": None,
            "comparison_start": None,
            "comparison_end": None,
            "min_rows_per_period": 100,
        },
        "drop_identifiers": True,
        "keep_columns": [],
        "min_row_coverage": 0.0,
    },
    "preprocessing": {
        "numeric_impute": "median",
        "numeric_constant": 0.0,
        "categorical_impute": "constant",
        "categorical_constant": "__MISSING__",
        "add_missing_indicator": True,
        "categorical_encoding": "native",
        "max_categories": 64,
        "min_category_frequency": 5,
        "clip_numeric": None,
        "clip_quantiles": [0.001, 0.999],
        "max_output_features": 5000,
    },
    "model": {
        "name": "lightgbm",
        "search_space": "search_spaces/lightgbm.yaml",
        "params": {},
        "num_boost_round": 2000,
        "early_stopping_rounds": 80,
        "seed": 42,
        "num_threads": 0,
        "class_weight": None,
        "log_period": 100,
    },
    "tuning": {
        "enabled": True,
        "n_trials": 20,
        "timeout_seconds": None,
        "direction": "auto",
        "metric": "auto",
        "cv": {
            "enabled": False,
            "folds": 5,
            "seed": None,
            "prune_after_folds": 2,
        },
        "sampler": {"n_startup_trials": 10, "seed": 42},
        "pruning": True,
        "n_warmup_steps": 20,
        "refit_on_train_valid": True,
    },
    "metrics": {
        "threshold": "auto",
        "plots": True,
        "top_importance": 30,
    },
    "output": {
        "report_dir": None,
        "save_predictions": True,
        "save_model": True,
    },
}


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _resolve_path(path: Any, base: Path) -> Optional[str]:
    if path is None:
        return None
    expanded = os.path.expanduser(str(path))
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = base / candidate
    return str(candidate.resolve())


def _resolve_input_path(path: Any, config_dir: Path) -> Optional[str]:
    """Resolve input paths against the config directory, then the working directory."""
    if path is None:
        return None
    expanded = os.path.expanduser(str(path))
    candidate = Path(expanded)
    if candidate.is_absolute():
        return str(candidate.resolve())
    from_config = (config_dir / candidate).resolve()
    from_cwd = (Path.cwd() / candidate).resolve()
    if from_config.exists() or not from_cwd.exists():
        return str(from_config)
    return str(from_cwd)


def csv_char(value: Any, default: Optional[str], name: str) -> Optional[str]:
    """Resolve a one-character CSV option, accepting escapes such as ``\\x03``."""

    if value is None:
        return default
    text = str(value)
    if len(text) == 1:
        return text
    decoded = text
    if text.startswith("\\"):
        try:
            decoded = codecs.decode(text, "unicode_escape")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ConfigError(f"{name} must be a single character; got {value!r}") from exc
    if len(decoded) != 1:
        raise ConfigError(f"{name} must be a single character; got {value!r}")
    return decoded


def pandas_csv_options(csv_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Shared pandas CSV keyword arguments for headers, samples, and ingest."""

    quoting = str(csv_cfg.get("quoting", "minimal")).lower()
    if quoting not in {"minimal", "none"}:
        raise ConfigError("data.csv.quoting must be 'minimal' or 'none'.")
    options: Dict[str, Any] = {
        "sep": csv_char(csv_cfg.get("sep", ","), ",", "data.csv.sep"),
        "doublequote": bool(csv_cfg.get("doublequote", True)),
        "escapechar": csv_char(csv_cfg.get("escapechar"), None, "data.csv.escapechar"),
    }
    if quoting == "none":
        options["quoting"] = csv_module.QUOTE_NONE
        options["quotechar"] = None
    else:
        options["quoting"] = csv_module.QUOTE_MINIMAL
        options["quotechar"] = csv_char(csv_cfg.get("quotechar", '"'), '"', "data.csv.quotechar")
    return options


def load_config(path: str | Path, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Load a YAML config, merge defaults, apply dotted overrides, resolve paths."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError("Top-level config must be a mapping.")

    cfg = _deep_merge(DEFAULTS, raw)
    config_dir = config_path.parent
    cfg["_meta"] = {"config_path": str(config_path), "config_dir": str(config_dir)}

    for override in overrides or []:
        if "=" not in override:
            raise ConfigError(f"Override must look like key=value, got: {override}")
        dotted, value_text = override.split("=", 1)
        value = yaml.safe_load(value_text)
        node = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"Override path conflicts with scalar value: {dotted}")
        node[parts[-1]] = value

    project = cfg["project"]
    project["work_dir"] = _resolve_path(project["work_dir"], Path.cwd())
    project["reuse_from"] = _resolve_path(project.get("reuse_from"), Path.cwd())
    output = cfg["output"]
    output["report_dir"] = _resolve_path(output.get("report_dir") or (Path(project["work_dir"]) / "reports"), Path("."))

    data = cfg["data"]
    data["label_path"] = _resolve_input_path(data.get("label_path"), config_dir)
    temporal_psi = cfg["filtering"]["temporal_psi"]
    temporal_psi["time_path"] = _resolve_input_path(temporal_psi.get("time_path"), config_dir)
    expanded_paths: List[str] = []
    for entry in _as_list(data.get("feature_paths")):
        resolved = _resolve_input_path(entry, config_dir)
        if resolved is None:
            continue
        if any(char in resolved for char in "*?["):
            from glob import glob

            matches = sorted(glob(resolved))
            if not matches:
                expanded_entry = os.path.expanduser(str(entry))
                if not Path(expanded_entry).is_absolute():
                    matches = sorted(glob(str((Path.cwd() / expanded_entry).resolve())))
            expanded_paths.extend(matches)
        else:
            expanded_paths.append(resolved)
    data["feature_paths"] = expanded_paths

    model = cfg["model"]
    if model.get("search_space"):
        search_path = Path(os.path.expanduser(str(model["search_space"])))
        candidates = [search_path]
        if not search_path.is_absolute():
            candidates = [config_dir / search_path, Path.cwd() / search_path]
        for candidate in candidates:
            if candidate.exists():
                model["search_space"] = str(candidate.resolve())
                break
        else:
            model["search_space"] = str(candidates[0].resolve())

    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    data = cfg["data"]
    project = cfg["project"]
    if not data["feature_paths"]:
        raise ConfigError("data.feature_paths must contain at least one CSV path or glob.")
    if not data.get("label_path"):
        raise ConfigError("data.label_path is required.")
    label_path = Path(data["label_path"])
    if not label_path.exists():
        raise ConfigError(f"Label file does not exist: {label_path}")
    for feature_path in data["feature_paths"]:
        if not Path(feature_path).exists():
            raise ConfigError(f"Feature file does not exist: {feature_path}")

    if data["concat_mode"] not in {"column", "row"}:
        raise ConfigError("data.concat_mode must be 'column' or 'row'.")
    split_strategy = str(data.get("split_strategy", "random")).lower()
    if split_strategy not in {"random", "temporal"}:
        raise ConfigError("data.split_strategy must be 'random' or 'temporal'.")
    if split_strategy == "temporal":
        temporal_split = data.get("temporal_split") or {}
        if not isinstance(temporal_split, dict):
            raise ConfigError("data.temporal_split must be a mapping.")
        if not temporal_split.get("time_column"):
            raise ConfigError(
                "data.temporal_split.time_column is required when "
                "split_strategy is 'temporal'."
            )
        time_column = str(temporal_split["time_column"])
        if time_column in {str(data["label_column"]), str(data["label_key"])}:
            raise ConfigError(
                "data.temporal_split.time_column must be a separate date column, "
                "not the label or key column."
            )
        valid_fraction = float(temporal_split.get("valid_fraction", 0.25))
        if not 0.0 < valid_fraction < 1.0:
            raise ConfigError(
                "data.temporal_split.valid_fraction must be between 0 and 1."
            )
        if (
            temporal_split.get("oot_start") is None
            and temporal_split.get("oot_quantile") is None
        ):
            raise ConfigError(
                "data.temporal_split requires either oot_start or oot_quantile."
            )
        oot_quantile = temporal_split.get("oot_quantile", 0.5)
        if oot_quantile is not None:
            try:
                oot_quantile = float(oot_quantile)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    "data.temporal_split.oot_quantile must be a number or null."
                ) from exc
            if not 0.0 < oot_quantile < 1.0:
                raise ConfigError(
                    "data.temporal_split.oot_quantile must be strictly between 0 and 1."
                )
        min_positive = temporal_split.get("min_positive_count")
        max_positive = temporal_split.get("max_positive_count")
        try:
            min_positive = None if min_positive is None else int(min_positive)
            max_positive = None if max_positive is None else int(max_positive)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "data.temporal_split positive count limits must be integers "
                "or null."
            ) from exc
        if min_positive is not None and min_positive < 0:
            raise ConfigError(
                "data.temporal_split.min_positive_count must be >= 0 or null."
            )
        if max_positive is not None and max_positive < 0:
            raise ConfigError(
                "data.temporal_split.max_positive_count must be >= 0 or null."
            )
        if (
            min_positive is not None
            and max_positive is not None
            and min_positive > max_positive
        ):
            raise ConfigError(
                "data.temporal_split.min_positive_count must not exceed "
                "data.temporal_split.max_positive_count."
            )
    csv_engine = str(data.get("csv", {}).get("engine", "pandas")).lower()
    if csv_engine not in {"pandas", "arrow"}:
        raise ConfigError("data.csv.engine must be 'pandas' or 'arrow'.")
    csv_chunksize = int(data.get("csv", {}).get("chunksize", 2000))
    if csv_chunksize < 1:
        raise ConfigError("data.csv.chunksize must be positive.")
    pandas_csv_options(data.get("csv", {}))
    if split_strategy == "random":
        fractions = [
            float(data["train_fraction"]),
            float(data["valid_fraction"]),
            float(data["test_fraction"]),
        ]
        if any(value <= 0 for value in fractions):
            raise ConfigError("train/valid/test fractions must all be positive.")
        if abs(sum(fractions) - 1.0) > 1e-6:
            raise ConfigError("data train/valid/test fractions must sum to 1.0.")

    pre = cfg["preprocessing"]
    if pre["numeric_impute"] not in {"median", "mean", "zero", "constant"}:
        raise ConfigError("preprocessing.numeric_impute must be median, mean, zero, or constant.")
    if pre["categorical_impute"] not in {"mode", "constant"}:
        raise ConfigError("preprocessing.categorical_impute must be mode or constant.")
    if pre["categorical_encoding"] not in {"native", "onehot", "ordinal"}:
        raise ConfigError("preprocessing.categorical_encoding must be native, onehot, or ordinal.")
    if cfg["tuning"]["enabled"] and int(cfg["tuning"]["n_trials"]) < 1:
        raise ConfigError("tuning.n_trials must be >= 1 when tuning is enabled.")
    cv = cfg["tuning"].get("cv") or {}
    if not isinstance(cv, dict):
        raise ConfigError("tuning.cv must be a mapping.")
    if cfg["tuning"]["enabled"] and cv.get("enabled", False):
        try:
            folds = int(cv.get("folds", 5))
            prune_after = int(cv.get("prune_after_folds", 2))
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "tuning.cv.folds and tuning.cv.prune_after_folds must be integers."
            ) from exc
        if folds < 2:
            raise ConfigError("tuning.cv.folds must be >= 2.")
        if prune_after < 1 or prune_after > folds:
            raise ConfigError(
                "tuning.cv.prune_after_folds must be between 1 and "
                "tuning.cv.folds."
            )
        cv_seed = cv.get("seed")
        if cv_seed is not None:
            try:
                int(cv_seed)
            except (TypeError, ValueError) as exc:
                raise ConfigError("tuning.cv.seed must be an integer or null.") from exc
    if float(project["heartbeat_seconds"]) <= 0:
        raise ConfigError("project.heartbeat_seconds must be positive.")
    filtering = cfg["filtering"]
    iv_limits: Dict[str, Optional[float]] = {}
    for key in ("min_iv", "max_iv"):
        value = filtering.get(key)
        if value is None:
            iv_limits[key] = None
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"filtering.{key} must be a finite non-negative number or null."
            ) from exc
        if not math.isfinite(parsed) or parsed < 0:
            raise ConfigError(
                f"filtering.{key} must be a finite non-negative number or null."
            )
        iv_limits[key] = parsed
    if (
        iv_limits["min_iv"] is not None
        and iv_limits["max_iv"] is not None
        and iv_limits["min_iv"] > iv_limits["max_iv"]
    ):
        raise ConfigError("filtering.min_iv must not exceed filtering.max_iv.")
    reuse_mode = str(project.get("reuse_mode", "link")).lower()
    if reuse_mode not in {"link", "copy"}:
        raise ConfigError("project.reuse_mode must be 'link' or 'copy'.")
    reuse_from = project.get("reuse_from")
    if reuse_from and not Path(reuse_from).is_dir():
        raise ConfigError(f"project.reuse_from is not a directory: {reuse_from}")
    if reuse_from and Path(reuse_from).resolve() == Path(project["work_dir"]).resolve():
        raise ConfigError(
            "project.reuse_from must point to a different run directory "
            "than project.work_dir."
        )

    temporal_psi = cfg["filtering"]["temporal_psi"]
    if temporal_psi.get("enabled"):
        required = [
            "time_column",
            "reference_start",
            "reference_end",
            "comparison_start",
            "comparison_end",
        ]
        missing = [key for key in required if temporal_psi.get(key) is None or temporal_psi.get(key) == ""]
        if missing:
            raise ConfigError("filtering.temporal_psi is enabled but is missing: " + ", ".join(missing))
        time_path = temporal_psi.get("time_path") or data["label_path"]
        if not Path(time_path).exists():
            raise ConfigError(f"Temporal PSI time file does not exist: {time_path}")
        try:
            min_rows = int(temporal_psi.get("min_rows_per_period", 0))
        except (TypeError, ValueError) as exc:
            raise ConfigError("filtering.temporal_psi.min_rows_per_period must be an integer.") from exc
        if min_rows < 1:
            raise ConfigError("filtering.temporal_psi.min_rows_per_period must be positive.")


def config_to_yaml(cfg: Dict[str, Any]) -> str:
    clean = copy.deepcopy(cfg)
    clean.pop("_meta", None)
    return yaml.safe_dump(clean, sort_keys=False)
