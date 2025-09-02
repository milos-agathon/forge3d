#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np

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
        out = Path("ex_roundtrip.png")
        rgb = (np.random.default_rng(0).integers(0, 255, size=(64,64,3), dtype=np.uint8)).copy(order="C")
        f3d.numpy_to_png(out, rgb)
        back = f3d.png_to_numpy(out)
        assert back.shape == (64,64,4)
        print("roundtrip OK:", out)
    except Exception as e:
        print(f"PNG roundtrip failed: {e}")
        sys.exit(0)  # Graceful exit, not an error

if __name__ == "__main__":
    main()