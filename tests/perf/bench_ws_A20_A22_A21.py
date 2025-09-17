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


def bench_a20() -> dict:
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
    radii = [0.5, 1.0, 2.0, 3.0]
    factors = []
    for r in radii:
        tmp = AreaLight(radius=r, intensity=10.0)
        factors.append(tmp.energy_factor)

    return {
        "energy_conservation_error": float(error),
        "energy_within_2_percent": bool(error < 0.02),
        "penumbra_energy_factors": {str(r): float(f) for r, f in zip(radii, factors)},
    }


ess = None

def bench_a21(width: int = 3840, height: int = 2160) -> dict:
    depth, normals = create_test_ao_scene(width=width, height=height)
    ao = AmbientOcclusionRenderer(radius=1.0, intensity=1.0, samples=16)
    t0 = time.time()
    _ = ao.render_ao(depth, normals)
    t1 = time.time()
    elapsed = t1 - t0
    return {
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


def main() -> None:
    out = {
        "A20": bench_a20(),
        "A21": bench_a21(),
        "A22": bench_a22(),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
