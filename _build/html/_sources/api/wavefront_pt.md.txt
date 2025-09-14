// docs/api/wavefront_pt.md
// Wavefront path tracer overview and API usage.
// This file exists to document queue-based PT, compaction, persistent threads, and engine selection.
// RELEVANT FILES:src/path_tracing/wavefront/mod.rs,src/shaders/pt_raygen.wgsl,python/forge3d/path_tracing.py,README.md

# Wavefront Path Tracer (Queue-Based)

This engine decomposes path tracing into stages: raygen → intersect → shade → scatter, coordinated by queues on the GPU.

- Queues live in storage buffers with atomic counters for push/pop.
- Periodic compaction removes terminated rays to keep work dense.
- A persistent-threads loop drains queues until completion.

## Selecting the Engine

Python interface accepts an engine selector for parity checks and experimentation.

```python
from forge3d.path_tracing import render_rgba, TracerEngine

img_wave = render_rgba(128, 128, scene, cam, seed=7, frames=1,
                       use_gpu=True, engine=TracerEngine.WAVEFRONT)
img_mega = render_rgba(128, 128, scene, cam, seed=7, frames=1,
                       use_gpu=True, engine=TracerEngine.MEGAKERNEL)
```

Notes:

- When the GPU backend or wavefront path isn’t available, the call falls back to the deterministic CPU implementation.
- For AOVs, use `render_aovs(..., engine=...)` and `save_aovs()`.

## WGSL Bindings (Overview)

- Bind Group 0: Uniforms (dimensions, frame index, spp, camera params, exposure, RNG seeds)
- Bind Group 1: Scene (accel, materials)
- Bind Group 2: Queues (headers + data for rays/hits/scatters)
- Bind Group 3: Accumulation/Outputs (HDR buffer or storage textures)

See shader files under `src/shaders/pt_*.wgsl` for exact layouts.

