# A1.9-BEGIN:device-diagnostics
#!/usr/bin/env python3
"""
Device diagnostics & failure modes for vulkan-forge.

- Enumerates adapters/features/limits (best-effort).
- Probes per-backend device creation and classifies outcomes (ok/unsupported/error).
- Writes JSON report; optional text summary.

Usage:
  python python/tools/device_diagnostics.py --json out/diag.json --summary
"""
from __future__ import annotations
import argparse, json, os, platform, sys
from typing import List

# Robust import: try top-level first (rare), then package-internal (common for maturin)
try:
    from _vulkan_forge import enumerate_adapters, device_probe  # top-level
except Exception:
    try:
        from vulkan_forge._vulkan_forge import enumerate_adapters, device_probe  # package-internal
    except Exception as e:
        raise SystemExit(
            "Failed to import compiled extension '_vulkan_forge'.\n"
            f"Python: {sys.executable}\n"
            "If you just built, ensure you're in the same venv; then run:\n"
            "  python -m pip install -U pip maturin\n"
            "  maturin develop --release\n"
        ) from e

def default_backends() -> List[str]:
    sysname = platform.system().lower()
    if "windows" in sysname:
        return ["VULKAN", "DX12", "GL"]
    if "darwin" in sysname or "mac" in sysname:
        return ["METAL", "GL"]
    return ["VULKAN", "GL"]

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="device_diagnostics.json")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--backends", nargs="*", default=None)
    args = ap.parse_args(argv)

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)

    report = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": sys.version.split()[0],
            "executable": sys.executable,
        },
        "adapters": [],
        "probes": {},
    }

    # Enumerate adapters
    try:
        report["adapters"] = enumerate_adapters()
    except Exception as e:
        report["adapters_error"] = str(e)

    # Probe per backend
    for b in [x.upper() for x in (args.backends or default_backends())]:
        try:
            rep = device_probe(b)
        except Exception as e:
            rep = {"backend_request": b, "status": "error", "message": str(e)}
        report["probes"][b] = rep

    # Write JSON + optional summary
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    if args.summary:
        print(json.dumps(report, indent=2))

    # Exit policy: OK if any backend is ok, or if all are unsupported
    ok = any(rep.get("status") == "ok" for rep in report["probes"].values())
    if not ok:
        if all(rep.get("status") == "unsupported" for rep in report["probes"].values()):
            print("No supported backends detected (not fatal).")
            return 0
        print("Diagnostics found errors. See JSON.")
        return 1
    print("Diagnostics OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
# A1.9-END:device-diagnostics
