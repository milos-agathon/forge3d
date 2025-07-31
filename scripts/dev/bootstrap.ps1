$ErrorActionPreference = "Stop"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip maturin
maturin develop --release
python scripts/dev/smoke_test.py