#!/usr/bin/env python3
import sys
from pathlib import Path

# Add repo root to sys.path for forge3d import
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "python"))

try:
    from forge3d import run_benchmark
except ImportError as e:
    print(f"Failed to import forge3d: {e}")
    print("Make sure the package is installed or run 'maturin develop' first")
    sys.exit(1)

def main():
    try:
        res = run_benchmark("renderer_rgba", repeats=20)
        print("Benchmark result:", res)
    except Exception as e:
        print(f"Benchmark failed (GPU may not be available): {e}")
        sys.exit(0)  # Graceful exit, not an error

if __name__ == "__main__":
    main()