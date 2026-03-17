# Polygon Extrusion

The current public polygon extrusion wrapper is
`forge3d.geometry.extrude_polygon()`.

## Public API

```python
import numpy as np
import forge3d as f3d

polygon = np.array(
    [
        [0.0, 1.0],
        [0.8, 0.0],
        [0.0, -1.0],
        [-0.8, 0.0],
    ],
    dtype=np.float32,
)

mesh = f3d.geometry.extrude_polygon(polygon, height=0.5)
print(mesh.positions.shape)
print(mesh.indices.shape)
```

`extrude_polygon()` returns a `forge3d.geometry.MeshBuffers` object containing
positions, normals, UVs, indices, and optional tangents.

## Status

- The wrapped native entry point is `geometry_extrude_polygon_py`.
- Older examples that used `forge3d.extrude_polygon_py(...)` directly are
  outdated.
- The extension also exposes lower-level extrusion helpers, but they are not
  part of the curated top-level Python API.
