# A1.7-BEGIN:backends-runner
#!/usr/bin/env python3
"""
Cross-backend runner for forge3d.

Spawns a fresh Python subprocess per backend so that wgpu Instance/OnceCell isn't reused.
Collects SHA256 of RAW RGBA bytes, timings, and (optionally) writes PNG artifacts.

Examples:
  python python/tools/backends_runner.py --runs 2 --png
  python python/tools/backends_runner.py --require-same --width 96 --height 96
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, subprocess, sys, time
from dataclasses import dataclass, asdict
from typing import Literal

Backend = Literal["VULKAN", "DX12", "METAL", "GL"]

def default_backends() -> list[Backend]:
    sysname = platform.system().lower()
    if "windows" in sysname:
        return ["VULKAN", "DX12", "GL"]
    if "darwin" in sysname or "mac" in sysname:
        return ["METAL", "GL"]
    # linux / others
    return ["VULKAN", "GL"]

@dataclass
class BackendResult:
    backend: str
    status: Literal["ok", "unsupported", "error"]
    sha256: str | None
    millis: float | None
    message: str | None
    png: str | None

CHILD_SNIPPET = r"""
import os, sys, time, hashlib
from forge3d import Renderer
w = int(os.environ.get("VF_WIDTH", "128"))
h = int(os.environ.get("VF_HEIGHT", "128"))
png = os.environ.get("VF_WRITE_PNG", "0") == "1"
t0 = time.perf_counter()
r = Renderer(w, h)
arr = r.render_triangle_rgba()
dt = (time.perf_counter() - t0) * 1000.0
sha = hashlib.sha256(arr.tobytes()).hexdigest()
if png:
    r.render_triangle_png(os.environ["VF_PNG_PATH"])
print(sha, f"{dt:.3f}")
"""

def run_once(backend: Backend, width: int, height: int, write_png: bool, out_dir: str) -> BackendResult:
    env = os.environ.copy()
    # Ensure our compiled module is used from current venv
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Pass backend selection to the child BEFORE import. (wgpu reads its backend choice at instance creation.)
    # We use a single well-known variable name for our runner; the Rust side reads default Backends::all(),
    # but we spawn a fresh interpreter per backend so wgpu runtime picks an available adapter on that backend.
    env["VF_WIDTH"] = str(width)
    env["VF_HEIGHT"] = str(height)
    if write_png:
        os.makedirs(out_dir, exist_ok=True)
        env["VF_WRITE_PNG"] = "1"
        env["VF_PNG_PATH"] = os.path.join(out_dir, f"triangle_{backend.lower()}.png")
    else:
        env["VF_WRITE_PNG"] = "0"

    # Hint the backend to the platform via wgpu's common env var name; if ignored, the child will still fail fast and we report.
    # (Different hal layers may use different names; we try a couple common ones.)
    env["WGPU_BACKEND"] = backend.lower()
    env["WGPU_BACKENDS"] = backend.lower()

    # Build the child command that performs a single render and prints "<sha> <millis>"
    code = CHILD_SNIPPET
    cmd = [sys.executable, "-c", code]

    t0 = time.perf_counter()
    try:
        out = subprocess.check_output(cmd, env=env, stderr=subprocess.STDOUT, text=True, timeout=120)
        dt = (time.perf_counter() - t0) * 1000.0
        line = out.strip().splitlines()[-1]
        sha, millis_str = line.split()
        return BackendResult(backend, "ok", sha, float(millis_str), None, os.path.basename(env.get("VF_PNG_PATH", "")) or None)
    except subprocess.CalledProcessError as e:
        # If the backend truly isn't supported, treat it as "unsupported" not "error"
        msg = e.output.strip() if isinstance(e.output, str) else str(e)
        lowered = msg.lower()
        if any(k in lowered for k in ["no suitable gpu adapter", "failed to find", "unsupported", "no adapter", "surface not supported"]):
            return BackendResult(backend, "unsupported", None, None, msg, None)
        return BackendResult(backend, "error", None, None, msg, None)
    except Exception as e:
        return BackendResult(backend, "error", None, None, str(e), None)

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="*", default=None, help="Override backend list (e.g. VULKAN DX12 GL METAL)")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--png", action="store_true", help="write per-backend PNG to --out-dir")
    ap.add_argument("--out-dir", default="backends_artifacts")
    ap.add_argument("--require-same", action="store_true", help="require identical RAW bytes across successful backends")
    args = ap.parse_args(argv)

    bk_list: list[Backend] = [b.upper() for b in (args.backends or default_backends())]

    os.makedirs(args.out_dir, exist_ok=True)
    per_backend: dict[str, dict] = {}
    for bk in bk_list:
        # Do multiple runs per backend to ensure within-backend stability
        shas, times = [], []
        last_png = None
        status = "ok"
        message = None
        for i in range(args.runs):
            res = run_once(bk, args.width, args.height, args.png and i == 0, args.out_dir)
            if res.status != "ok":
                status, message = res.status, res.message
                break
            shas.append(res.sha256)
            times.append(res.millis or 0.0)
            last_png = res.png
        if status == "ok":
            status = "ok" if len(set(shas)) == 1 else "error"
            if status == "error":
                message = f"non-deterministic across runs: {shas}"
        per_backend[bk] = {
            "status": status,
            "message": message,
            "runs": args.runs if status == "ok" else 1,
            "sha256": (shas[0] if shas else None),
            "avg_ms": (sum(times) / max(1, len(times))) if times else None,
            "png": last_png,
        }

    # Cross-backend comparison if requested
    ok_hashes = [d["sha256"] for d in per_backend.values() if d["status"] == "ok" and d["sha256"]]
    cross_equal = (len(set(ok_hashes)) == 1) if ok_hashes else False
    report = {
        "width": args.width, "height": args.height, "runs": args.runs,
        "backends": bk_list,
        "require_same": args.require_same,
        "per_backend": per_backend,
        "at_least_one_ok": any(d["status"] == "ok" for d in per_backend.values()),
        "cross_backend_equal": cross_equal,
    }

    rep_path = os.path.join(args.out_dir, "backends_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    if not report["at_least_one_ok"]:
        raise SystemExit("All requested backends failed or unsupported.")
    if args.require_same and not report["cross_backend_equal"]:
        raise SystemExit("Cross-backend hashes differ and --require-same was set.")
    print("Cross-backend check OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
# A1.7-END:backends-runner