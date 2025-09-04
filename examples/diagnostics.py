#!/usr/bin/env python3
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
        print("Adapters:", f3d.enumerate_adapters())
        print("Probe:", f3d.device_probe())
    except Exception as e:
        print(f"GPU diagnostics failed (GPU may not be available): {e}")
        sys.exit(0)  # Graceful exit, not an error

if __name__ == "__main__":
    main()
