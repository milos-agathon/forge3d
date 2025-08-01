# A1.10-BEGIN:perf-sanity
#!/usr/bin/env python3
"""
Performance sanity harness for vulkan-forge.

Measures:
  - init_ms: Renderer(...) + first render (cold)
  - steady_ms: repeated render_triangle_rgba() timings (warmups excluded)
Outputs:
  - JSON report with stats (mean, median, p95, stdev, min, max), dims, runs, warmups
  - Optional CSV of per-iteration timings

Thresholds:
  - By default, never fails (CI-safe). Prints report only.
  - If VF_ENFORCE_PERF=1:
      * If --baseline is given, fail if steady p95 exceeds baseline p95 by > --regress-pct (default 50%).
      * Else scale a simple budget: base 40ms @ 512x512 → budget_ms = 40 * (W*H)/(512*512)
        and fail if steady p95 > budget_ms * --budget-mult (default 3.0).
"""
from __future__ import annotations
import argparse, csv, json, math, os, statistics as stats, sys, time
from typing import List, Dict, Any

try:
    from vulkan_forge import Renderer
except Exception as e:
    raise SystemExit(f"Failed to import vulkan_forge.Renderer: {e}")

def percentile(values: List[float], p: float) -> float:
    if not values: return float("nan")
    k = (len(values) - 1) * (p / 100.0)
    f = math.floor(k); c = math.ceil(k)
    if f == c: return values[int(k)]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1

def measure(width: int, height: int, runs: int, warmups: int) -> Dict[str, Any]:
    t0 = time.perf_counter()
    r = Renderer(width, height)
    # cold render included in init cost
    r.render_triangle_rgba()
    init_ms = (time.perf_counter() - t0) * 1000.0

    # warmups (not recorded)
    for _ in range(max(0, warmups)):
        r.render_triangle_rgba()

    # steady-state timings
    steady = []
    for _ in range(runs):
        t = time.perf_counter()
        r.render_triangle_rgba()
        steady.append((time.perf_counter() - t) * 1000.0)

    steady_sorted = sorted(steady)
    rep = {
        "width": width, "height": height,
        "runs": runs, "warmups": warmups,
        "init_ms": init_ms,
        "steady": {
            "samples_ms": steady,  # raw list for CSV if desired
            "mean_ms": stats.fmean(steady) if steady else float("nan"),
            "median_ms": stats.median(steady) if steady else float("nan"),
            "p95_ms": percentile(steady_sorted, 95.0) if steady else float("nan"),
            "stdev_ms": stats.pstdev(steady) if len(steady) > 1 else 0.0,
            "min_ms": min(steady) if steady else float("nan"),
            "max_ms": max(steady) if steady else float("nan"),
        },
    }
    return rep

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmups", type=int, default=3)
    ap.add_argument("--json", default="perf_report.json")
    ap.add_argument("--csv", default="")
    ap.add_argument("--baseline", default="")
    ap.add_argument("--regress-pct", type=float, default=50.0, help="Allow this % over baseline p95 before failing (VF_ENFORCE_PERF=1)")
    ap.add_argument("--budget-mult", type=float, default=3.0, help="Multiplier for scaled budget when baseline is not provided (VF_ENFORCE_PERF=1)")
    args = ap.parse_args(argv)

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)

    rep = measure(args.width, args.height, args.runs, args.warmups)

    # Optional CSV
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["iter", "ms"])
            for i, ms in enumerate(rep["steady"]["samples_ms"]):
                w.writerow([i, f"{ms:.3f}"])

    # Write JSON
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))

    # Enforcement (opt-in)
    if os.environ.get("VF_ENFORCE_PERF", "").strip() == "1":
        p95 = float(rep["steady"]["p95_ms"] or "nan")
        if args.baseline:
            try:
                base = load_json(args.baseline)
                base_p95 = float(base["steady"]["p95_ms"])
                limit = base_p95 * (1.0 + args.regress_pct / 100.0)
                if p95 > limit:
                    print(f"FAIL: p95 {p95:.3f}ms > baseline {base_p95:.3f}ms * (1 + {args.regress_pct:.1f}%) = {limit:.3f}ms")
                    return 2
            except Exception as e:
                print(f"WARNING: failed to read baseline '{args.baseline}': {e}")
        else:
            # simple scaled budget: 40ms @ 512x512
            budget = 40.0 * (args.width * args.height) / (512.0 * 512.0)
            limit = budget * args.budget_mult
            if p95 > limit:
                print(f"FAIL: p95 {p95:.3f}ms > scaled budget {limit:.3f}ms (budget {budget:.3f} * mult {args.budget_mult:.2f})")
                return 2

    print("Performance sanity OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
# A1.10-END:perf-sanity