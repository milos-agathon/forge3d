# Normal Mapping

There is no public `forge3d.normalmap` helper module in the current package.

## Current building blocks

- `forge3d.mesh.generate_cube_tbn()`
- `forge3d.mesh.generate_plane_tbn()`
- `forge3d.mesh.validate_tbn_data()`
- `forge3d.materials.PbrMaterial`
- `forge3d.textures.build_pbr_textures(normal=...)`

## Example

```python
import numpy as np
import forge3d.mesh as mesh
from forge3d.materials import PbrMaterial
from forge3d.textures import build_pbr_textures

vertices, indices, tbn_data = mesh.generate_cube_tbn()
print(mesh.validate_tbn_data(tbn_data))

normal = np.full((4, 4, 4), [128, 128, 255, 255], dtype=np.uint8)
texset = build_pbr_textures(normal=normal)
material = PbrMaterial().with_textures(texset)
print(material)
```

## Status

The renderer has tangent-space normal-mapping support and TBN generation, but
the old `forge3d.normalmap.*` utility API described in earlier docs is not part
of the current public Python surface.
