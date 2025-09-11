#!/usr/bin/env python3
"""
LOD Impostors Demo (scaffold)

Simulates a scripted camera sweep and records basic LOD metrics to JSON. This
is a CPU-only scaffold that does not render; it exists to drive audit artifacts.
"""
import argparse
import json
from pathlib import Path
import random


def simulate_triangle_reduction(num_tiles: int = 16*16, base_tris: int = 2048, seed: int = 123) -> float:
    rng = random.Random(seed)
    tiles_lod = []
    for _ in range(num_tiles):
        lod = min(4, max(0, int(abs(rng.gauss(2.0, 1.0)))))
        tiles_lod.append(lod)
    full = num_tiles * base_tris
    lod_sum = 0
    for lod in tiles_lod:
        lod_sum += max(1, base_tris // (4 ** max(0, lod)))
    reduction = max(0.0, float(full - lod_sum) / float(full)) if full else 0.0
    return reduction


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", action="store_true", help="Run a scripted sweep (no-op in scaffold)")
    p.add_argument("--record", type=Path, default=Path("reports/q_lod_metrics.json"))
    args = p.parse_args()

    args.record.parent.mkdir(parents=True, exist_ok=True)

    # Simulated metrics
    if args.sweep:
        series = []
        for step in range(0, 181, 15):  # 0..180 degrees sweep
            tri_reduction = simulate_triangle_reduction(seed=123 + step)
            series.append({
                "step": step,
                "tri_reduction_ratio": tri_reduction,
                "update_budget_ms": 8.0,
            })
        out = {"scene": "scaffold", "series": series}
    else:
        tri_reduction = simulate_triangle_reduction()
        out = {
            "scene": "scaffold",
            "tri_reduction_ratio": tri_reduction,
            "update_budget_ms": 8.0,
            "notes": "Scaffold metrics; replace with real scene when available",
        }

    with open(args.record, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
