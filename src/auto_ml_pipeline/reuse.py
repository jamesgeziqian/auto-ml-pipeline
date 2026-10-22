"""Fingerprint-checked reuse of data-side artifacts between pipeline runs."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import ConfigError
from .preprocess import MatrixResult
from .stats import FilterResult
from .temporal import TemporalPSIWindows

REUSE_STATE_VERSION = 1
REUSABLE_STAGES = ("cache", "splits", "temporal", "filtering", "preprocessing", "matrix")
_SOURCE_DIR = Path(__file__).resolve().parent

_CACHE_DATA_KEYS = (
    "feature_paths",
    "label_path",
    "feature_key",
    "label_key",
    "label_column",
    "concat_mode",
    "include_label_features",
    "drop_columns",
    "deduplicate_columns",
    "max_rows",
    "max_source_columns_in_memory",
    "wide_block_size",
    "wide_row_group_size",
    "cache_numeric_dtype",
    "ingest_target_chunk_bytes",
    "sample_rows",
    "csv",
)

_ARTIFACT_FILES: Dict[str, Sequence[str]] = {
    "cache": ("cache/manifest.json", "schema/sources.json", "cache/row_coverage.npy"),
    "splits": ("splits.npz",),
    "temporal": ("temporal_psi_windows.npz", "temporal_psi_windows.json"),
    "filtering": ("filter_report.csv", "selected_features.json"),
    "preprocessing": ("preprocess_state.json",),
}


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _input_signatures(paths: Sequence[Any]) -> Dict[str, Any]:
    signatures: Dict[str, Any] = {}
    for raw in paths:
        if not raw:
            continue
        path = Path(raw).expanduser().resolve()
        try:
            stat = path.stat()
        except OSError:
            signatures[str(path)] = None
        else:
            signatures[str(path)] = {
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
    return signatures


def _code_hash(names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        path = _SOURCE_DIR / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _versions(names: Sequence[str]) -> Dict[str, Any]:
    versions: Dict[str, Any] = {}
    for name in names:
        try:
            versions[name] = getattr(importlib.import_module(name), "__version__", None)
        except Exception:
            versions[name] = None
    return versions


def _cache_component(cfg: Dict[str, Any]) -> Dict[str, Any]:
    data = cfg["data"]
    inputs = [data.get("label_path"), *data.get("feature_paths", [])]
    return {
        "artifact_version": REUSE_STATE_VERSION,
        "data": {key: data.get(key) for key in _CACHE_DATA_KEYS},
        "type_inference": cfg.get("type_inference", {}),
        "inputs": _input_signatures(inputs),
        "versions": _versions(("pandas", "numpy", "pyarrow")),
        "code": _code_hash(("cache.py", "schema.py")),
    }


def _split_component(cfg: Dict[str, Any], cache_fingerprint: str) -> Dict[str, Any]:
    data = cfg["data"]
    return {
        "cache": cache_fingerprint,
        "max_rows": data.get("max_rows"),
        "train_fraction": data.get("train_fraction"),
        "valid_fraction": data.get("valid_fraction"),
        "test_fraction": data.get("test_fraction"),
        "stratify": data.get("stratify"),
        "seed": cfg["project"].get("random_seed"),
        "inputs": _input_signatures([data.get("label_path")]),
        "versions": _versions(("sklearn", "numpy")),
        "code": _code_hash(("schema.py",)),
    }


def _temporal_component(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    temporal = cfg["filtering"].get("temporal_psi", {})
    if not temporal.get("enabled"):
        return None
    data = cfg["data"]
    time_path = temporal.get("time_path") or data.get("label_path")
    return {
        "config": temporal,
        "csv": data.get("csv"),
        "inputs": _input_signatures([time_path, data.get("label_path")]),
        "versions": _versions(("pandas", "numpy")),
        "code": _code_hash(("temporal.py",)),
    }


def _filter_component(
    cfg: Dict[str, Any],
    cache_fingerprint: str,
    split_fingerprint: str,
    temporal_fingerprint: Optional[str],
) -> Dict[str, Any]:
    return {
        "cache": cache_fingerprint,
        "splits": split_fingerprint,
        "temporal": temporal_fingerprint,
        "filtering": cfg["filtering"],
        "column_block_size": cfg["data"].get("column_block_size"),
        "versions": _versions(("pandas", "numpy")),
        "code": _code_hash(("stats.py",)),
    }


def _preprocess_component(cfg: Dict[str, Any], filter_fingerprint: str) -> Dict[str, Any]:
    return {
        "filtering": filter_fingerprint,
        "preprocessing": cfg["preprocessing"],
        "column_block_size": cfg["data"].get("column_block_size"),
        "versions": _versions(("pandas", "numpy")),
        "code": _code_hash(("preprocess.py",)),
    }


def _matrix_component(cfg: Dict[str, Any], preprocess_fingerprint: str) -> Dict[str, Any]:
    return {
        "preprocessing": preprocess_fingerprint,
        "column_block_size": cfg["data"].get("column_block_size"),
        "versions": _versions(("numpy", "scipy")),
        "code": _code_hash(("preprocess.py",)),
    }


def stage_fingerprints(cfg: Dict[str, Any]) -> Dict[str, Optional[str]]:
    cache = _digest(_cache_component(cfg))
    splits = _digest(_split_component(cfg, cache))
    temporal_component = _temporal_component(cfg)
    temporal = _digest(temporal_component) if temporal_component is not None else None
    filtering = _digest(_filter_component(cfg, cache, splits, temporal))
    preprocessing = _digest(_preprocess_component(cfg, filtering))
    matrix = _digest(_matrix_component(cfg, preprocessing))
    return {
        "cache": cache,
        "splits": splits,
        "temporal": temporal,
        "filtering": filtering,
        "preprocessing": preprocessing,
        "matrix": matrix,
    }


class ReuseState:
    """Link reusable artifacts from a previous run and record stage completion."""

    def __init__(self, work_dir: str | Path, cfg: Dict[str, Any], logger: Any):
        self.work_dir = Path(work_dir)
        self.cfg = cfg
        self.logger = logger
        self.fingerprints = stage_fingerprints(cfg)
        self.stages: Dict[str, bool] = {stage: False for stage in REUSABLE_STAGES}
        self.reused: Dict[str, bool] = {}
        self.reasons: Dict[str, str] = {}
        self.mode = str(cfg["project"].get("reuse_mode", "link")).lower()
        raw_source = cfg["project"].get("reuse_from")
        self.source = Path(raw_source).expanduser().resolve() if raw_source else None
        self.source_state: Optional[Dict[str, Any]] = None

        if self.source is not None:
            if not self.source.is_dir():
                raise ConfigError(f"project.reuse_from is not a directory: {self.source}")
            if self.source == self.work_dir.resolve():
                raise ConfigError(
                    "project.reuse_from must point to a different run directory "
                    "than project.work_dir."
                )
            self._load_source_state()
            self._evaluate()
            self._materialize()
            for stage, enabled in self.reused.items():
                if enabled:
                    self.stages[stage] = True
            self.logger.info(
                "[reuse] source=%s mode=%s stages=%s",
                self.source,
                self.mode,
                ", ".join(sorted(stage for stage, ok in self.reused.items() if ok)) or "none",
            )
            for stage, reason in self.reasons.items():
                if not self.reused.get(stage) and reason != "disabled":
                    self.logger.info("[reuse] %s will be recomputed: %s", stage, reason)
        self._write_state()

    def _state_path(self, root: Path) -> Path:
        return root / "reuse_state.json"

    def _load_source_state(self) -> None:
        path = self._state_path(self.source)
        if not path.exists():
            raise ConfigError(
                f"Reuse source {self.source} has no reuse_state.json. "
                "Run once with this version to create reusable state, or "
                "remove project.reuse_from."
            )
        with path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        if int(state.get("version", -1)) != REUSE_STATE_VERSION:
            raise ConfigError(
                f"Reuse source {self.source} uses state version "
                f"{state.get('version')!r}; expected {REUSE_STATE_VERSION}."
            )
        self.source_state = state

    def _artifacts_present(self, root: Path, stage: str) -> bool:
        if stage == "matrix":
            meta_path = root / "matrix" / "matrix_meta.json"
            if not meta_path.exists():
                return False
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return False
            matrix_path = root / "matrix" / ("X.npz" if meta.get("sparse") else "X.npy")
            return matrix_path.exists()
        return all((root / relative).exists() for relative in _ARTIFACT_FILES.get(stage, ()))

    def _evaluate(self) -> None:
        assert self.source_state is not None
        source_stages = self.source_state.get("stages", {})
        source_fingerprints = self.source_state.get("fingerprints", {})
        for stage in REUSABLE_STAGES:
            fingerprint = self.fingerprints.get(stage)
            if fingerprint is None:
                self.reasons[stage] = "disabled"
            elif not source_stages.get(stage):
                self.reasons[stage] = "not completed in source run"
            elif source_fingerprints.get(stage) != fingerprint:
                self.reasons[stage] = "config, input, version, or code changed"
            elif not self._artifacts_present(self.source, stage):
                self.reasons[stage] = "artifacts missing"
            else:
                self.reused[stage] = True

    def _link_dir(self, source: Path, target: Path) -> None:
        if target.exists() or target.is_symlink():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.mode == "copy":
            shutil.copytree(source, target)
            return
        try:
            relative = os.path.relpath(source, target.parent)
            os.symlink(relative, target, target_is_directory=True)
        except OSError as exc:
            self.logger.warning(
                "[reuse] symlink failed for %s (%s); copying instead", target, exc
            )
            shutil.copytree(source, target)

    def _link_file(self, source: Path, target: Path) -> None:
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        # Copy the small metadata/split/filter artifacts so a new experiment
        # cannot accidentally mutate its source run by editing them.
        shutil.copy2(source, target)

    def _materialize(self) -> None:
        assert self.source is not None
        actions = []
        if self.reused.get("cache"):
            actions.append(("cache", self._link_dir, self.source / "cache", self.work_dir / "cache"))
            actions.append(
                (
                    "cache",
                    lambda src, dst: shutil.copytree(src, dst) if not dst.exists() else None,
                    self.source / "schema",
                    self.work_dir / "schema",
                )
            )
        if self.reused.get("splits"):
            actions.append(("splits", self._link_file, self.source / "splits.npz", self.work_dir / "splits.npz"))
        if self.reused.get("temporal"):
            for name in _ARTIFACT_FILES["temporal"]:
                actions.append(("temporal", self._link_file, self.source / name, self.work_dir / name))
        if self.reused.get("filtering"):
            for name in _ARTIFACT_FILES["filtering"]:
                actions.append(("filtering", self._link_file, self.source / name, self.work_dir / name))
        if self.reused.get("preprocessing"):
            name = _ARTIFACT_FILES["preprocessing"][0]
            actions.append(("preprocessing", self._link_file, self.source / name, self.work_dir / name))
        if self.reused.get("matrix"):
            actions.append(("matrix", self._link_dir, self.source / "matrix", self.work_dir / "matrix"))

        for stage, operation, source, target in actions:
            if not self.reused.get(stage):
                continue
            try:
                operation(source, target)
            except OSError as exc:
                self.reused[stage] = False
                self.reasons[stage] = f"artifact materialization failed: {exc}"
                self.logger.warning("[reuse] %s cannot be reused: %s", stage, exc)
                self._cleanup_stage(stage)

    def _cleanup_stage(self, stage: str) -> None:
        targets: List[Path] = []
        if stage == "cache":
            targets = [self.work_dir / "cache", self.work_dir / "schema"]
        elif stage == "splits":
            targets = [self.work_dir / "splits.npz"]
        elif stage == "matrix":
            targets = [self.work_dir / "matrix"]
        else:
            targets = [self.work_dir / name for name in _ARTIFACT_FILES.get(stage, ())]
        for target in targets:
            try:
                if target.is_symlink() or target.is_file():
                    target.unlink()
                elif target.is_dir():
                    shutil.rmtree(target)
            except OSError as exc:
                self.logger.warning("[reuse] could not clean %s: %s", target, exc)

    def _write_state(self) -> None:
        payload = {
            "version": REUSE_STATE_VERSION,
            "source": str(self.source) if self.source is not None else None,
            "mode": self.mode,
            "fingerprints": self.fingerprints,
            "stages": self.stages,
            "reasons": dict(self.reasons),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self._state_path(self.work_dir)
        temporary = path.with_name(".reuse_state.json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        os.replace(temporary, path)

    def can_reuse(self, stage: str) -> bool:
        return bool(self.reused.get(stage, False))

    def mark(self, stage: str) -> None:
        if stage not in self.stages:
            return
        self.stages[stage] = True
        self._write_state()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "from": str(self.source) if self.source is not None else None,
            "mode": self.mode,
            "reused_stages": sorted(stage for stage, ok in self.reused.items() if ok),
            "stages": dict(self.stages),
            "reasons": dict(self.reasons),
            "fingerprints": dict(self.fingerprints),
        }

    def load_schema_stats(self) -> Dict[str, Any]:
        payload = json.loads(
            (self.work_dir / "schema" / "sources.json").read_text(encoding="utf-8")
        )
        return dict(payload.get("stats", {}))

    def load_manifest(self) -> Dict[str, Any]:
        return json.loads(
            (self.work_dir / "cache" / "manifest.json").read_text(encoding="utf-8")
        )

    def load_splits(self) -> Dict[str, np.ndarray]:
        with np.load(self.work_dir / "splits.npz") as data:
            return {name: np.asarray(data[name]) for name in data.files}

    def load_filter_result(self) -> FilterResult:
        report = pd.read_csv(self.work_dir / "filter_report.csv")
        selected_payload = json.loads(
            (self.work_dir / "selected_features.json").read_text(encoding="utf-8")
        )
        return FilterResult(
            selected=[str(column) for column in selected_payload.get("selected", [])],
            report=report,
            stats={},
            elapsed_seconds=float(selected_payload.get("elapsed_seconds", 0.0)),
        )

    def load_preprocess_state(self) -> Dict[str, Any]:
        return json.loads(
            (self.work_dir / "preprocess_state.json").read_text(encoding="utf-8")
        )

    def load_matrix(self, y: np.ndarray) -> MatrixResult:
        meta = json.loads(
            (self.work_dir / "matrix" / "matrix_meta.json").read_text(encoding="utf-8")
        )
        sparse = bool(meta.get("sparse"))
        path = self.work_dir / "matrix" / ("X.npz" if sparse else "X.npy")
        if sparse:
            matrix = sp.load_npz(path)
        else:
            # build_matrix writes X.npy as a raw float32 memmap, not a .npy file.
            shape = tuple(int(value) for value in meta.get("shape", ()))
            if len(shape) != 2:
                raise ValueError(f"Invalid matrix shape in {path}: {shape}")
            matrix = np.memmap(path, dtype=np.float32, mode="r", shape=shape)
        return MatrixResult(
            X=matrix,
            y=y,
            feature_names=list(meta.get("feature_names", [])),
            output_specs=list(meta.get("output_specs", [])),
            categorical_indices=list(meta.get("categorical_indices", [])),
            sparse=sparse,
            elapsed_seconds=float(meta.get("elapsed_seconds", 0.0)),
        )

    def load_temporal(self) -> TemporalPSIWindows:
        with np.load(self.work_dir / "temporal_psi_windows.npz") as data:
            reference = np.asarray(data["reference_mask"])
            comparison = np.asarray(data["comparison_mask"])
        metadata = json.loads(
            (self.work_dir / "temporal_psi_windows.json").read_text(encoding="utf-8")
        )
        return TemporalPSIWindows(
            reference_mask=reference,
            comparison_mask=comparison,
            metadata=metadata,
        )
