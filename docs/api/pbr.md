# Physically-Based Rendering (PBR) Materials

There is no public `forge3d.pbr` Python module in the current package. PBR
support is spread across a few existing modules instead.

## Current Python surface

- `forge3d.materials.PbrMaterial`
- `forge3d.textures.PbrTexSet`
- `forge3d.textures.build_pbr_textures()`
- `forge3d.textures.load_texture()`
- `forge3d.textures.gltf_mr_channels()`
- `forge3d.path_tracing.PathTracer`
- `forge3d.buildings.BuildingMaterial`

## Example

```python
import numpy as np
from forge3d.materials import PbrMaterial
from forge3d.textures import build_pbr_textures

base = np.full((4, 4, 4), [200, 140, 90, 255], dtype=np.uint8)
normal = np.full((4, 4, 4), [128, 128, 255, 255], dtype=np.uint8)

texset = build_pbr_textures(base_color=base, normal=normal)
material = PbrMaterial(
    base_color_factor=(1.0, 1.0, 1.0, 1.0),
    metallic_factor=0.0,
    roughness_factor=0.6,
).with_textures(texset)

print(material)
```

## Notes

- The deterministic CPU fallback tracer in `forge3d.path_tracing` consumes the
  same `PbrMaterial` container.
- Terrain-specific PBR/POM work lives in `forge3d.terrain_pbr_pom` and viewer
  IPC terrain controls, not in a standalone `forge3d.pbr` module.
