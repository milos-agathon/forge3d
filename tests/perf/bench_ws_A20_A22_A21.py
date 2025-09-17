#!/usr/bin/env python3
"""
Headless numeric benches for Workstream A tasks A20, A21, A22.

Outputs a single JSON blob to stdout with the following keys:
- A20: energy_conservation_error, energy_within_2_percent, penumbra_energy_factors
- A21: ao_time_seconds, width, height, ok_time_le_1s
- A22: instances, memory_bytes, ok_mem_le_512_mib
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import sys
import argparse
import numpy as np

# Ensure local package imports without wheel install
REPO_ROOT = Path(__file__).resolve().parents[2]
PY_SRC = REPO_ROOT / "python"
if str(PY_SRC) not in sys.path:
    sys.path.insert(0, str(PY_SRC))

# A20 imports
from forge3d.lighting import AreaLight, AreaLightManager

# A21 imports
from forge3d.ambient_occlusion import AmbientOcclusionRenderer, create_test_ao_scene

# A22 imports
from forge3d.instancing import InstancedGeometry


def bench_a20(radii: list[float] | None = None) -> dict:
    mgr = AreaLightManager()
    # Two diverse lights
    l1 = AreaLight.disc(position=(0, 5, 0), direction=(0, -1, 0), disc_radius=2.0, intensity=15.0, penumbra_radius=1.0)
    l2 = AreaLight.rectangle(position=(3, 4, 0), direction=(-1, -1, 0), width=2.0, height=1.5, intensity=10.0, penumbra_radius=0.8)
    mgr.add_light(l1)
    mgr.add_light(l2)

    initial_energy = mgr.calculate_total_energy()
    mgr.set_energy_target(initial_energy)

    # Modify radii significantly
    l1.set_radius(l1.radius * 2.5)
    l2.set_radius(max(0.1, l2.radius * 0.4))

    error = mgr.normalize_energy()  # fractional error

    # Penumbra/radius energy factor sampling
    if radii is None:
        radii = [0.5, 1.0, 2.0, 3.0]
    factors = []
    for r in radii:
        tmp = AreaLight(radius=float(r), intensity=10.0)
        factors.append(tmp.energy_factor)

    return {
        "energy_conservation_error": float(error),
        "energy_within_2_percent": bool(error < 0.02),
        "penumbra_energy_factors": {str(r): float(f) for r, f in zip(radii, factors)},
    }


ess = None

def bench_a21(width: int = 3840, height: int = 2160, *, samples: int = 16, tile_step: int | None = None) -> dict:
    depth, normals = create_test_ao_scene(width=width, height=height)
    ao = AmbientOcclusionRenderer(radius=1.0, intensity=1.0, samples=int(samples), tile_step=tile_step)
    t0 = time.time()
    _ = ao.render_ao(depth, normals)
    t1 = time.time()
    elapsed = t1 - t0
    return {
        "samples": int(samples),
        "tile_step": (None if tile_step is None else int(tile_step)),
        "ao_time_seconds": float(elapsed),
        "width": width,
        "height": height,
        "ok_time_le_1s": bool(elapsed <= 1.0),
    }


def bench_a22(instances: int = 10000) -> dict:
    geo = InstancedGeometry(max_instances=instances)
    # Fill with identity transforms
    T = np.eye(4, dtype=np.float32)
    for i in range(instances):
        geo.add_instance(T, material_id=0)
    mem = geo.get_memory_usage()
    ok = geo.validate_memory_budget()
    return {
        "instances": instances,
        "memory_bytes": int(mem),
        "ok_mem_le_512_mib": bool(ok),
    }

def parse_list(s: str, typ=float):
    vals = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        vals.append(typ(part))
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless benches for A20/A21/A22")
    parser.add_argument("--mode", choices=["single", "sweep"], default="single")
    # A20
    parser.add_argument("--a20-radii", type=str, default="0.5,1.0,2.0,3.0")
    # A21
    parser.add_argument("--a21-width", type=int, default=3840)
    parser.add_argument("--a21-height", type=int, default=2160)
    parser.add_argument("--a21-samples", type=str, default="16")
    parser.add_argument("--a21-tile-steps", type=str, default="16")
    # A22
    parser.add_argument("--a22-instances", type=int, default=10000)

    args = parser.parse_args()

    # A20
    a20_radii = parse_list(args.a20_radii, typ=float)
    a20 = bench_a20(a20_radii)

    # A22
    a22 = bench_a22(args.a22_instances)

    # A21
    width, height = args.a21_width, args.a21_height
    a21_samples = parse_list(args.a21_samples, typ=int)
    a21_tiles = parse_list(args.a21_tile_steps, typ=int)

    out = {
        "A20": a20,
        "A22": a22,
    }

    if args.mode == "single":
        # Use first values
        s0 = a21_samples[0] if a21_samples else 16
        t0 = a21_tiles[0] if a21_tiles else None
        out["A21"] = bench_a21(width, height, samples=s0, tile_step=t0)
    else:
        results = []
        for s in (a21_samples or [16]):
            for t in (a21_tiles or [None]):
                results.append(bench_a21(width, height, samples=s, tile_step=t))
        out["A21_sweep"] = results

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
