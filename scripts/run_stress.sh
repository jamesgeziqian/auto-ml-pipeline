#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

.venv/bin/auto-ml generate-data \
  --out data/stress \
  --rows 32500 \
  --cols 500 \
  --target-gb 3.5 \
  --parts 4 \
  --heartbeat-seconds 10

.venv/bin/auto-ml run \
  --config configs/stress_500x32k.yaml \
  --work-dir runs/stress_500x32k

