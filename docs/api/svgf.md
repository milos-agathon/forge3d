<!-- docs/api/svgf.md
Minimal SVGF overview and API notes (stub documentation)
Exists to outline AOV requirements and parameters for future GPU-backed implementation
RELEVANT FILES:python/forge3d/path_tracing.py,src/denoise/svgf/pipelines.rs,src/shaders/svgf_reproject.wgsl
-->

# SVGF Denoiser (Stub)

This repository includes a minimal, CPU-backed placeholder for an SVGF-like denoiser exposed via the Python API.

- Inputs: albedo, normal, depth (synthetic for now), and current-frame radiance.
- Stages: temporal accumulation (simulated) and A-trous edge-aware filtering on CPU.
- Parameters: `denoiser="svgf"`, `svgf_iters` (default 5).

GPU WGSL shader stubs and Rust scaffolding are present for future wiring:

- `src/shaders/svgf_reproject.wgsl` — temporal reprojection (stub)
- `src/shaders/svgf_variance.wgsl` — moments/variance update (stub)
- `src/shaders/svgf_atrous.wgsl` — edge-aware filter (stub)
- `src/denoise/svgf/*` — Rust module scaffolding

Example (Python):

```
from forge3d.path_tracing import PathTracer, make_camera, make_sphere

tr = PathTracer()
scene = [make_sphere(center=(0,0,-3), radius=1.0, albedo=(0.7,0.7,0.7))]
cam = make_camera(origin=(0,0,0), look_at=(0,0,-1), up=(0,1,0), fov_y=45.0, aspect=1.0, exposure=1.0)
img = tr.render_rgba(128,128,scene,cam,seed=3,frames=1,denoiser="svgf", svgf_iters=5)
```

