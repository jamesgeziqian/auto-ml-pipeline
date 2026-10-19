"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .config import ConfigError, load_config
from .logging_utils import configure_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-ml",
        description="Chunked auto machine learning pipeline for wide tabular data.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the pipeline from a YAML config.")
    run_parser.add_argument("--config", required=True, help="Path to pipeline YAML config.")
    run_parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted config override, e.g. --override tuning.n_trials=5",
    )
    run_parser.add_argument("--work-dir", help="Override project.work_dir.")
    run_parser.add_argument("--n-trials", type=int, help="Override tuning.n_trials.")
    run_parser.add_argument("--max-rows", type=int, help="Limit number of loaded label rows.")
    run_parser.add_argument("--heartbeat-seconds", type=float, help="Heartbeat interval.")
    run_parser.add_argument("--log-level", help="DEBUG, INFO, WARNING, ERROR.")
    run_parser.add_argument("--no-tune", action="store_true", help="Disable Optuna tuning.")
    run_parser.add_argument("--no-plots", action="store_true", help="Disable metric plots.")

    data_parser = subparsers.add_parser(
        "generate-data", help="Generate a large synthetic multi-file CSV dataset."
    )
    data_parser.add_argument("--out", required=True, help="Output directory.")
    data_parser.add_argument("--rows", type=int, default=32_500)
    data_parser.add_argument("--cols", type=int, default=500, help="Number of feature columns.")
    data_parser.add_argument("--target-gb", type=float, default=3.5)
    data_parser.add_argument("--parts", type=int, default=4, help="Number of feature CSV files.")
    data_parser.add_argument("--seed", type=int, default=42)
    data_parser.add_argument("--chunk-rows", type=int, default=1000)
    data_parser.add_argument("--pad-len", type=int, default=None)
    data_parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    data_parser.add_argument("--log-level", default="INFO")

    wide_parser = subparsers.add_parser(
        "generate-wide",
        help="Generate identical wide CSVs for row-wise concatenation stress testing.",
    )
    wide_parser.add_argument("--out", required=True)
    wide_parser.add_argument("--rows-per-file", type=int, default=50_000)
    wide_parser.add_argument("--files", type=int, default=4)
    wide_parser.add_argument("--cols", type=int, default=32_000)
    wide_parser.add_argument("--signal-cols", type=int, default=20)
    wide_parser.add_argument("--pattern-pool", type=int, default=256)
    wide_parser.add_argument("--chunk-rows", type=int, default=1000)
    wide_parser.add_argument("--seed", type=int, default=42)
    wide_parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    wide_parser.add_argument("--log-level", default="INFO")
    return parser


def _apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.work_dir:
        cfg["project"]["work_dir"] = str(Path(args.work_dir).expanduser().resolve())
    if args.n_trials is not None:
        cfg["tuning"]["n_trials"] = int(args.n_trials)
    if args.max_rows is not None:
        cfg["data"]["max_rows"] = int(args.max_rows)
    if args.heartbeat_seconds is not None:
        cfg["project"]["heartbeat_seconds"] = float(args.heartbeat_seconds)
    if args.log_level:
        cfg["project"]["log_level"] = args.log_level
    if args.no_tune:
        cfg["tuning"]["enabled"] = False
    if args.no_plots:
        cfg["metrics"]["plots"] = False


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate-data":
        from .synthetic import generate_synthetic_dataset

        logger = configure_logging(args.log_level)
        try:
            info = generate_synthetic_dataset(
                out_dir=args.out,
                rows=args.rows,
                n_features=args.cols,
                target_gb=args.target_gb,
                n_feature_files=args.parts,
                seed=args.seed,
                chunk_rows=args.chunk_rows,
                pad_len=args.pad_len,
                heartbeat_seconds=args.heartbeat_seconds,
                logger=logger,
            )
            print(json.dumps(info, indent=2))
            return 0
        except Exception as exc:
            logger.error("Data generation failed: %s", exc)
            return 1
    if args.command == "generate-wide":
        from .synthetic import generate_wide_rowwise_dataset

        logger = configure_logging(args.log_level)
        try:
            info = generate_wide_rowwise_dataset(
                out_dir=args.out,
                rows_per_file=args.rows_per_file,
                n_files=args.files,
                n_features=args.cols,
                n_signal=args.signal_cols,
                pattern_pool=args.pattern_pool,
                chunk_rows=args.chunk_rows,
                seed=args.seed,
                heartbeat_seconds=args.heartbeat_seconds,
                logger=logger,
            )
            print(json.dumps(info, indent=2))
            return 0
        except Exception as exc:
            logger.error("Wide data generation failed: %s", exc)
            return 1

    try:
        cfg = load_config(args.config, overrides=args.override)
        _apply_cli_overrides(cfg, args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    try:
        from .pipeline import run_pipeline

        summary = run_pipeline(cfg)
    except Exception as exc:
        print(f"Pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    logger = configure_logging(cfg["project"].get("log_level", "INFO"))
    logger.info(
        "[done] status=%s total=%.2fs work_dir=%s",
        summary.get("status"),
        float(summary.get("elapsed_seconds", 0.0)),
        summary.get("work_dir"),
    )
    timings = summary.get("stage_timings", {})
    if timings:
        logger.info("[done] stage timings:")
        for name, payload in timings.items():
            logger.info(
                "[done]   %-22s %8.2fs", name, float(payload["elapsed_seconds"])
            )
    print(json.dumps(
        {
            "status": summary.get("status"),
            "elapsed_seconds": summary.get("elapsed_seconds"),
            "work_dir": summary.get("work_dir"),
            "metrics": (summary.get("metrics") or {}).get("test"),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
