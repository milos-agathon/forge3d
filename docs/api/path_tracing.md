<!-- docs/api/path_tracing.md
Minimal API notes for A1 GPU Path Tracer.
This file exists to document usage and limitations of the MVP tracer.
RELEVANT FILES:src/shaders/pt_kernel.wgsl,src/path_tracing/compute.rs,python/forge3d/path_tracing.py,README.md
-->

# A1: GPU Path Tracer (MVP)

The minimal GPU path tracer exposes a single function from the Rust extension and a simple CPU reference in Python.

## GPU Usage

```python
from forge3d import _forge3d as _f

scene = [{"center": (0,0,-3), "radius": 1.0, "albedo": (0.8,0.2,0.2)}]
cam   = {"origin": (0,0,0), "look_at": (0,0,-1), "up": (0,1,0), "fov_y": 45.0, "aspect": 1.0, "exposure": 1.0}
img   = _f._pt_render_gpu(64, 64, scene, cam, seed=123, frames=1)  # (H,W,4) uint8
```

Notes:
- Deterministic per fixed `seed` and `frames=1` for the MVP kernel.
- Requires a compatible GPU adapter; tests skip if unavailable.

## CPU Reference

```python
from forge3d.path_tracing import PathTracer

t = PathTracer(64, 64, seed=2)
t.add_sphere((0.0, 0.0, 0.0), 0.6, {"type": "lambert", "base_color": (0.8, 0.8, 0.8)})
img = t.render_rgba(spp=1)
```

## Limitations

- MVP supports spheres only on GPU; triangles and materials are covered in the CPU reference.
- Future milestones (BVH, BSDFs, wavefront) will expand both GPU and Python APIs.

