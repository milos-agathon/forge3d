# UV Unwrap Helpers

Forge3D provides minimal UV unwrap helpers focused on planar and spherical projections. These are intended for quick parameterizations for simple assets and previews.

## API (Python)

- `forge3d.geometry.unwrap_planar(mesh, axis=2) -> MeshBuffers`
  - Drops the specified axis (0=X, 1=Y, 2=Z) and projects the mesh AABB onto the remaining axes.
  - Generates UVs in [0, 1].

- `forge3d.geometry.unwrap_spherical(mesh) -> MeshBuffers`
  - Computes UVs by projecting vertices onto a sphere centered at the AABB center using `(phi, theta)`.
  - UVs are in [0, 1].

## Notes

- These helpers do not adjust seams or tangent space; they're intended as quick, deterministic unwraps.
- Use planar unwrap for charts with natural planar projection (e.g., terrain tiles, facades).
- Use spherical unwrap for roughly spherical shapes.

## Example

```python
import numpy as np
from forge3d.geometry import MeshBuffers, unwrap_planar, unwrap_spherical

# Square in XY plane
positions = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float32)
indices   = np.array([[0,1,2],[0,2,3]], dtype=np.uint32)
mesh = MeshBuffers(positions=positions, normals=np.empty((0,3), np.float32), uvs=np.empty((0,2), np.float32), indices=indices)

# Planar unwrap along Z (projected to XY)
mesh_u = unwrap_planar(mesh, axis=2)
print(mesh_u.uvs.shape)  # (4, 2)
```
