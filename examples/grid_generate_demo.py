#!/usr/bin/env python3
import sys
from pathlib import Path

# Add repo root to sys.path for forge3d import
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "python"))

try:
    import forge3d as f3d
except ImportError as e:
    print(f"Failed to import forge3d: {e}")
    print("Make sure the package is installed or run 'maturin develop' first")
    sys.exit(1)

def main():
    try:
        xy, uv, idx = f3d.grid_generate(32, 32, spacing=(1.0, 1.0), origin="center")
        print("grid:", xy.shape, uv.shape, idx.shape)
    except Exception as e:
        print(f"Grid generation failed: {e}")
        sys.exit(0)  # Graceful exit, not an error

if __name__ == "__main__":
    main()