#!/usr/bin/env bash
set -euo pipefail
TASK="${1:-Put the blue rectangular prism into the red metal box}"
python scripts/run_pipeline.py --config configs/demo.yaml --task "$TASK" --mode auto
