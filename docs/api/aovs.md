// docs/api/aovs.md
// Overview of path tracing AOVs and debug outputs exposed by forge3d.
// This file exists to document formats, ranges, and naming for AOV outputs and their Python API.
// RELEVANT FILES:python/forge3d/path_tracing.py,src/shaders/pt_kernel.wgsl,README.md

# Path Tracing AOVs

This document describes the AOVs (arbitrary output variables) produced by the path tracing APIs.

The Python entry points are `forge3d.path_tracing.render_aovs()` and `forge3d.path_tracing.save_aovs()`.

## Canonical AOVs

- albedo: Base color at first hit. Float32 array, shape (H, W, 3).
- normal: Shading normal (xyz). Float32 array, shape (H, W, 3).
- depth: Linear distance t along the primary ray. Float32 array, shape (H, W). np.nan for miss.
- direct: Direct lighting estimate. Float32 array, shape (H, W, 3).
- indirect: Indirect lighting estimate. Float32 array, shape (H, W, 3).
- emission: Emissive contribution. Float32 array, shape (H, W, 3).
- visibility: Primary hit mask. uint8 array, shape (H, W). 255 hit, 0 miss.

## Notes

- GPU-first design with CPU fallback. In this repository snapshot the CPU path is authoritative and deterministic.
- File I/O: `save_aovs()` writes `.npy` for HDR fields and `.png` for `visibility` when the PNG helper is available; otherwise it falls back to `.npy`.
- Expected GPU formats when implemented in WGSL/wgpu: rgba16float (albedo/normal/direct/indirect/emission), r32float (depth), r8unorm (visibility).

## Example

```python
import forge3d.path_tracing as pt
scene = [{"center": (0.0, 0.0, 0.0), "radius": 0.5, "albedo": (0.8, 0.3, 0.2)}]
aovs = pt.render_aovs(64, 64, scene, {}, seed=123, frames=1)
pt.save_aovs("out/demo", aovs)
```
