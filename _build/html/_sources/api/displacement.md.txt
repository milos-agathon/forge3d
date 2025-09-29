# Displacement (F12)

Apply displacement to meshes using a heightmap or a simple procedural pattern.
Normals are recomputed after displacement.

## API (Python)

- `forge3d.geometry.displace_heightmap(mesh: MeshBuffers, heightmap: np.ndarray, scale: float = 1.0, *, uv_space: bool = False) -> MeshBuffers`
  - Samples a 2D float32 heightmap and displaces along vertex normals.
  - When `uv_space=False`, samples over the mesh XY bounds. When `uv_space=True`, samples using vertex UVs.
  - `scale` multiplies sampled height values.

- `forge3d.geometry.displace_procedural(mesh: MeshBuffers, amplitude: float = 1.0, frequency: float = 1.0) -> MeshBuffers`
  - Applies a sin/cos-based displacement along normals using world-space XY coordinates.

- `forge3d.geometry.generate_tangents(mesh: MeshBuffers) -> np.ndarray`
  - Generates per-vertex tangents as `(N,4)` array `[tx, ty, tz, w]` (w = handedness).

## Example

```python
import numpy as np
from forge3d.geometry import primitive_mesh, displace_heightmap

mesh = primitive_mesh("plane", resolution=(32, 32))
H, W = 128, 128
x = np.linspace(0, np.pi * 2, W, dtype=np.float32)
y = np.linspace(0, np.pi * 2, H, dtype=np.float32)
Y, X = np.meshgrid(y, x, indexing="ij")
hm = (np.sin(X) * np.cos(Y)).astype(np.float32)

mesh2 = displace_heightmap(mesh, hm, scale=0.1)
print(mesh2.vertex_count, mesh2.triangle_count)

# UV-space sampling example (after planar unwrap)
from forge3d.geometry import unwrap_planar, generate_tangents
mesh = unwrap_planar(mesh, axis=2)
mesh_uv = displace_heightmap(mesh, hm, scale=0.1, uv_space=True)
tans = generate_tangents(mesh_uv)
print(tans.shape)  # (N,4)
```
