<!-- docs/api/svgf.md
Minimal SVGF overview and API notes
RELEVANT FILES:python/forge3d/path_tracing.py,python/forge3d/denoise.py
-->

# SVGF Denoiser

This repository includes a CPU-backed SVGF-like denoiser exposed via the Python API.

## Features

- **Inputs**: albedo, normal, depth, and current-frame radiance
- **Stages**: temporal accumulation (simulated) and A-trous edge-aware filtering
- **Parameters**: `denoiser="svgf"`, `svgf_iters` (default 5)

## Example

```python
from forge3d.path_tracing import PathTracer, make_camera, make_sphere

tr = PathTracer()
scene = [make_sphere(center=(0,0,-3), radius=1.0, albedo=(0.7,0.7,0.7))]
cam = make_camera(origin=(0,0,0), look_at=(0,0,-1), up=(0,1,0), fov_y=45.0, aspect=1.0, exposure=1.0)
img = tr.render_rgba(128,128,scene,cam,seed=3,frames=1,denoiser="svgf", svgf_iters=5)
```

## Notes

- Pure Python/CPU implementation
- For higher quality, consider using external GPU denoisers (OIDN, OptiX)

