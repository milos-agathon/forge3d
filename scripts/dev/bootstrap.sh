#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip maturin
maturin develop --release
python scripts/dev/smoke_test.py