#!/usr/bin/env python3
from pathlib import Path
import sys
from _import_shim import ensure_repo_import
ensure_repo_import()

try:
    import forge3d as f3d
except Exception as e:
    print(f"Failed to import forge3d: {e}")
    print("Make sure the package is installed or run 'maturin develop' first")
    sys.exit(0)

def main():
    try:
        r = f3d.Renderer(256, 256)
        r.render_triangle_png(Path("ex_triangle.png"))
        print("triangle.png written")
    except Exception as e:
        print(f"Rendering failed (GPU may not be available): {e}")
        sys.exit(0)  # Graceful exit, not an error

if __name__ == "__main__":
    main()
