"""Logging, stage timers, and an always-on liveness heartbeat."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional

import psutil


_PEAK_RSS_MB = 0.0


def configure_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("auto_ml_pipeline")
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    return logger


def rss_mb() -> float:
    global _PEAK_RSS_MB
    current = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    _PEAK_RSS_MB = max(_PEAK_RSS_MB, current)
    return current


def peak_rss_mb() -> float:
    return float(_PEAK_RSS_MB)


def format_bytes(num_bytes: float) -> str:
    value = float(num_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


class Heartbeat:
    """Background thread that proves the process is still making progress."""

    def __init__(self, logger: logging.Logger, interval: float = 15.0):
        self.logger = logger
        self.interval = max(1.0, float(interval))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = 0.0
        self._stage = "startup"
        self._detail = ""
        self._lock = threading.Lock()
        self._process = psutil.Process(os.getpid())
        self._process.cpu_percent(None)
        self.beats = 0

    def set_stage(self, stage: str, detail: str = "") -> None:
        with self._lock:
            self._stage = stage
            self._detail = detail

    def detail(self, detail: str) -> None:
        with self._lock:
            self._detail = detail

    def start(self) -> None:
        if self._thread is not None:
            return
        self._started = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, name="heartbeat", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.beats += 1
            with self._lock:
                stage, detail = self._stage, self._detail
            elapsed = time.perf_counter() - self._started
            cpu = self._process.cpu_percent(None)
            text = (
                f"[heartbeat] alive stage={stage} elapsed={elapsed:.1f}s "
                f"rss={rss_mb():.1f}MiB peak={peak_rss_mb():.1f}MiB cpu={cpu:.1f}%"
            )
            if detail:
                text += f" {detail}"
            self.logger.info(text)

    def __enter__(self) -> "Heartbeat":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


@dataclass
class StageTiming:
    name: str
    elapsed_seconds: float
    start_rss_mb: float
    end_rss_mb: float
    details: Dict[str, Any] = field(default_factory=dict)


class StageRecorder:
    def __init__(self, logger: logging.Logger, heartbeat: Heartbeat):
        self.logger = logger
        self.heartbeat = heartbeat
        self.timings: "OrderedDict[str, StageTiming]" = OrderedDict()
        self.run_started = time.perf_counter()

    @contextmanager
    def stage(self, name: str, detail: str = "") -> Iterator[Dict[str, Any]]:
        self.logger.info("[stage] START %s%s", name, f" ({detail})" if detail else "")
        self.heartbeat.set_stage(name, detail)
        start_rss = rss_mb()
        started = time.perf_counter()
        extra: Dict[str, Any] = {}
        try:
            yield extra
        finally:
            elapsed = time.perf_counter() - started
            end_rss = rss_mb()
            self.timings[name] = StageTiming(name, elapsed, start_rss, end_rss, extra)
            self.logger.info(
                "[stage] END   %s elapsed=%.2fs (%.2f min) rss=%.1fMiB",
                name,
                elapsed,
                elapsed / 60.0,
                end_rss,
            )

    def total_seconds(self) -> float:
        return time.perf_counter() - self.run_started

    def as_dict(self) -> Dict[str, Any]:
        return {
            name: {
                "elapsed_seconds": round(timing.elapsed_seconds, 4),
                "elapsed_minutes": round(timing.elapsed_seconds / 60.0, 4),
                "start_rss_mb": round(timing.start_rss_mb, 1),
                "end_rss_mb": round(timing.end_rss_mb, 1),
                "details": timing.details,
            }
            for name, timing in self.timings.items()
        }

    def log_table(self) -> None:
        self.logger.info("[timing] stage summary")
        for name, timing in self.timings.items():
            self.logger.info(
                "[timing] %-28s %8.2fs  %8.2f min  peak_rss=%.1fMiB",
                name,
                timing.elapsed_seconds,
                timing.elapsed_seconds / 60.0,
                max(timing.start_rss_mb, timing.end_rss_mb),
            )
        self.logger.info(
            "[timing] %-28s %8.2fs  %8.2f min",
            "TOTAL",
            self.total_seconds(),
            self.total_seconds() / 60.0,
        )
