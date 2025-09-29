# Converters: MultipolygonZ → Mesh

Forge3D includes a minimal converter to triangulate simple MultiPolygonZ inputs into meshes.

This is useful for quickly previewing extruded footprints or polygonal datasets without bringing in a full GIS pipeline.

## API (Python)

- `forge3d.converters.multipolygonz_to_mesh(polygons: List[np.ndarray]) -> MeshBuffers`
  - Each polygon must be a `(N, 3)` float32 array representing an XYZ ring.
  - Triangulates each ring using a fan around the first vertex.
  - Computes simple averaged vertex normals.

## Example

```python
import numpy as np
import forge3d.converters as conv

# Two squares
r1 = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float32)
r2 = np.array([[2,0,0],[3,0,0],[3,1,0],[2,1,0]], dtype=np.float32)
mesh = conv.multipolygonz_to_mesh([r1, r2])
print(mesh.vertex_count, mesh.triangle_count)
```

## Notes & Limitations

- The triangulation is a basic fan; does not handle holes or complex polygon topology.
- Normals are averaged across adjacent faces for simple shading.
- For watertight extrusions and robust topology, integrate with extrusion utilities (F1) and dedicated GIS preprocessors.
