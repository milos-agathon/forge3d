#!/usr/bin/env python3
import sys
from pathlib import Path

# Prefer in-repo package via shared import shim
from _import_shim import ensure_repo_import
ensure_repo_import()

try:
    from forge3d import run_benchmark
except Exception as e:
    print(f"Failed to import forge3d: {e}")
    print("Make sure the package is installed or run 'maturin develop' first")
    sys.exit(0)

def main():
    try:
        res = run_benchmark("renderer_rgba", iterations=20, width=256, height=256, warmup=3)
        print("Benchmark result:")
        print(res)
    except Exception as e:
        print(f"Benchmark failed (GPU may not be available): {e}")
        sys.exit(0)  # Graceful exit, not an error

if __name__ == "__main__":
    main()
