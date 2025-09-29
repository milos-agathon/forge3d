# Subdivision (F11)

Loop subdivision for triangle meshes with crease and boundary preservation.
Each level splits triangles into four and applies Loop vertex smoothing.

## API (Python)

- `forge3d.geometry.subdivide_mesh(mesh: MeshBuffers, levels: int = 1, *, creases: Optional[np.ndarray] = None, preserve_boundary: bool = True) -> MeshBuffers`

Notes:
- Each level multiplies triangle count by 4.
- Vertex normals are recomputed for smooth shading.
- UVs are linearly interpolated if present; otherwise remain empty.

## Example

```python
import numpy as np
from forge3d.geometry import MeshBuffers, subdivide_mesh

positions = np.array(
    [[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float32
)
indices = np.array([[0,1,2],[0,2,3]], dtype=np.uint32)
uvs = positions[:, :2].copy()
mesh = MeshBuffers(positions=positions, normals=np.empty((0,3), np.float32), uvs=uvs, indices=indices)

refined = subdivide_mesh(mesh, levels=2)

# Add creases along the top edge (2-3) and shared diagonal (0-2): shape (K,2) uint32
creases = np.array([[2,3],[0,2]], dtype=np.uint32)
refined_crease = subdivide_mesh(mesh, levels=2, creases=creases, preserve_boundary=True)
print(refined.vertex_count, refined.triangle_count)
```

### Adaptive Subdivision

`subdivide_adaptive(mesh, *, edge_length_limit=None, curvature_threshold=None, max_levels=3, creases=None, preserve_boundary=True)`

Refines based on geometric thresholds:

- `edge_length_limit`: target maximum edge length in world units; estimated levels via log2 of the ratio.
- `curvature_threshold`: maximum allowed dihedral angle (radians) across edges; exceeding increases levels.
- `max_levels`: cap on total refinement.
- `creases`, `preserve_boundary`: same semantics as uniform subdivision.

Example:

```python
from forge3d.geometry import primitive_mesh, subdivide_adaptive

mesh = primitive_mesh("sphere", rings=8, radial_segments=8)
refined = subdivide_adaptive(
    mesh,
    edge_length_limit=0.05,
    curvature_threshold=0.6,
    max_levels=3,
)
```
