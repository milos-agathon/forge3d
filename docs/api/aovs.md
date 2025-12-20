<!-- docs/api/aovs.md -->
<!-- API docs for AOVs (albedo, normal, depth, direct, indirect, emission, visibility). -->
<!-- Exists to document formats, shapes, and Python usage for render_aovs/save_aovs. -->
<!-- RELEVANT FILES:python/forge3d/path_tracing.py,src/path_tracing/aov.rs,src/shaders/pt_kernel.wgsl -->

# AOVs and Debug Outputs

- Names: {"albedo","normal","depth","direct","indirect","emission","visibility"}.

- Shapes and dtypes:
  - albedo/normal/direct/indirect/emission: (H, W, 3) float32.
  - depth: (H, W) float32.
  - visibility: (H, W) uint8.

- Formats (GPU):
  - albedo/normal/direct/indirect/emission: `rgba16float` storage textures.
  - depth: `r32float`.
  - visibility: `r8unorm`.

## EXR Output

HDR AOVs are written as OpenEXR when the native EXR writer is available. The Python
`save_aovs` helper prefers the Rust writer (`forge3d._forge3d.numpy_to_exr`) and
falls back to `imageio` if installed. Visibility remains PNG.

Channel naming follows a `prefix.channel` convention:

- beauty: `beauty.R`, `beauty.G`, `beauty.B`, `beauty.A`.
- albedo/direct/indirect/emission: `prefix.R`, `prefix.G`, `prefix.B`.
- normal: `normal.X`, `normal.Y`, `normal.Z`.
- depth: `depth.Z`.
- scalar AOVs (roughness/metallic/ao/sun_vis/id/mask): `prefix`.

Arrays must be float32 with shape `(H, W)`, `(H, W, 3)`, or `(H, W, 4)`.

## Python Usage

```python
import forge3d.path_tracing as pt

scene, cam = [{"center": (0,0,0), "radius": 0.5, "albedo": (0.8,0.3,0.2)}], {"pos": (0,0,1.5)}
aovs = pt.render_aovs(64, 64, scene, cam, aovs=("albedo","depth","visibility"), seed=7, frames=1, use_gpu=True)

paths = pt.save_aovs(aovs, basename="frame0001", output_dir="out")
```

The CPU implementation is deterministic and mirrors GPU semantics.

