# Polygon Extrusion

The `forge3d.extrude_polygon_py` function allows you to create 3D meshes by extruding 2D polygons.

## Usage

The function takes a NumPy array of 2D vertices defining a polygon and an extrusion height. It returns a tuple containing the vertices, indices, normals, and UVs of the resulting 3D mesh.

```python
import numpy as np
import forge3d as f3d

# Define a polygon
polygon = np.array([
    [0.0, 1.0],
    [0.2, 0.2],
    [1.0, 0.2],
    [0.4, -0.2],
    [0.6, -1.0],
    [0.0, -0.6],
    [-0.6, -1.0],
    [-0.4, -0.2],
    [-1.0, 0.2],
    [-0.2, 0.2],
], dtype=np.float32)

# Extrude the polygon
vertices, indices, normals, uvs = f3d.extrude_polygon_py(polygon, height=0.5)

# Add the extruded mesh to a scene
scene = f3d.Scene()
scene.add_mesh(
    vertices=vertices,
    indices=indices,
    normals=normals,
    uvs=uvs,
    color=[0.8, 0.2, 0.2, 1.0],
)
```

## API Reference

`forge3d.extrude_polygon_py(polygon, height)`

**Parameters:**

-   `polygon` (np.ndarray): A NumPy array of shape `(N, 2)` and `dtype=np.float32` representing the 2D vertices of the polygon.
-   `height` (float): The height of the extrusion.

**Returns:**

A tuple `(vertices, indices, normals, uvs)`:
-   `vertices` (np.ndarray): The 3D vertices of the mesh.
-   `indices` (np.ndarray): The indices of the mesh.
-   `normals` (np.ndarray): The normals of the mesh.
-   `uvs` (np.ndarray): The UV coordinates of the mesh.
\n## GPU Extrusion\n\nThe GPU accelerated variant \orge3d.extrude_polygon_gpu_py\ accepts a list of polygons (each shaped \(N, 2\) float32).\nThe function returns flattened arrays for positions (xyz), indices, normals (xyz), and uvs (uv).\n\n`python\nvertices, indices, normals, uvs = f3d.extrude_polygon_gpu_py([polygon], height=0.5)\n`\n\nThe GPU path uses the same tessellation as the CPU reference and produces identical results when an adapter is available.\n
