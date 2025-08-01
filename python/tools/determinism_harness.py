# A1.6-BEGIN:determinism-harness
#!/usr/bin/env python3
"""
Determinism harness for vulkan-forge.

- Renders the triangle N times and asserts identical raw RGBA bytes each run.
- Optionally runs in multiple processes to shake out shared-state issues.
- Writes a JSON report (hashes, timings) and optional PNG.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from dataclasses import dataclass, asdict

try:
    from vulkan_forge import Renderer
except Exception as e:
    raise SystemExit(f"Failed to import vulkan_forge.Renderer: {e}")

@dataclass
class RunResult:
    sha256: str
    millis: float

def render_bytes(w: int, h: int) -> bytes:
    r = Renderer(w, h)
    arr = r.render_triangle_rgba()
    # NumPy ndarray -> contiguous bytes (already tightly packed as per A1.4)
    return arr.tobytes()

def run_sequential(w: int, h: int, runs: int) -> list[RunResult]:
    out: list[RunResult] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        b = render_bytes(w, h)
        dt = (time.perf_counter() - t0) * 1000.0
        out.append(RunResult(hashlib.sha256(b).hexdigest(), dt))
    return out

def run_multiprocess(w: int, h: int, runs: int, procs: int) -> list[RunResult]:
    import multiprocessing as mp
    def worker(q: "mp.Queue[int]") -> None:
        b = render_bytes(w, h)
        q.put((hashlib.sha256(b).hexdigest(), len(b)))
    results: list[RunResult] = []
    for _ in range(runs):
        procs_list = []
        q: "mp.Queue[tuple[str,int]]" = mp.Queue()
        for _ in range(max(1, procs)):
            p = mp.Process(target=worker, args=(q,))
            p.start()
            procs_list.append(p)
        t0 = time.perf_counter()
        shas = []
        for p in procs_list:
            p.join()
            shas.append(q.get()[0])
        dt = (time.perf_counter() - t0) * 1000.0
        # All children should match each other in the same iteration
        if len(set(shas)) != 1:
            raise AssertionError(f"Non-deterministic across processes: {shas}")
        results.append(RunResult(shas[0], dt))
    return results

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width",  type=int, default=128)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--runs",   type=int, default=5)
    ap.add_argument("--processes", type=int, default=0, help=">0 enables multi-process check")
    ap.add_argument("--png", action="store_true", help="also write PNG to --out-dir")
    ap.add_argument("--out-dir", default="determinism_artifacts")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.processes > 0:
        results = run_multiprocess(args.width, args.height, args.runs, args.processes)
    else:
        results = run_sequential(args.width, args.height, args.runs)

    shas = [r.sha256 for r in results]
    unique = sorted(set(shas))
    report = {
        "width": args.width,
        "height": args.height,
        "runs": args.runs,
        "processes": args.processes,
        "hashes": shas,
        "unique": unique,
        "all_equal": len(unique) == 1,
        "avg_ms": sum(r.millis for r in results) / max(1, len(results)),
    }

    # Optional PNG (uses same pipeline path)
    if args.png:
        try:
            r = Renderer(args.width, args.height)
            r.render_triangle_png(os.path.join(args.out_dir, "triangle.png"))
            report["png"] = "triangle.png"
        except Exception as e:
            report["png_error"] = str(e)

    # Write JSON report
    rep_path = os.path.join(args.out_dir, "determinism_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    if not report["all_equal"]:
        raise SystemExit("Determinism check FAILED: differing hashes")
    print("Determinism check OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
# A1.6-END:determinism-harness