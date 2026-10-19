#!/usr/bin/env python
"""Convenience wrapper around `auto-ml generate-data`."""

from auto_ml_pipeline.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["generate-data", *__import__("sys").argv[1:]]))
