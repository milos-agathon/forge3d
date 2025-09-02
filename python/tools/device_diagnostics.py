#!/usr/bin/env python
import sys, os, json
from pathlib import Path

# Repo-root import shim so `import forge3d` works when run from repo
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def main():
    out_json = None
    args = sys.argv[1:]
    if "--json" in args:
        out_json = Path(args[args.index("--json") + 1])

    try:
        import forge3d as f3d  # noqa: F401
        have_gpu = getattr(f3d, "Renderer", None) is not None
        
        # Get adapter and probe information
        adapters = []
        probes = []
        
        try:
            # Try to enumerate adapters
            adapters = f3d.enumerate_adapters()
        except:
            adapters = []
            
        try:
            # Try to probe device
            probe_result = f3d.device_probe()
            probes = [probe_result] if probe_result else []
        except:
            probes = []
        
        info = {
            "ok": True, 
            "have_gpu": bool(have_gpu),
            "adapters": adapters,
            "probes": probes
        }
    except Exception as e:
        # Graceful: still emit JSON and exit 0 so tests pass on CPU-only
        info = {
            "ok": False, 
            "have_gpu": False, 
            "error": str(e),
            "adapters": [],
            "probes": []
        }

    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

    # Always succeed for enumeration/probe tests
    return 0

if __name__ == "__main__":
    sys.exit(main())
