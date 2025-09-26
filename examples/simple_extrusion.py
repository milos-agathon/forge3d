# examples/simple_extrusion.py

import numpy as np
import forge3d as f3d

def main():
    # Define a simple star-shaped polygon
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

    print(f"vertices: {vertices.shape}, indices: {indices.shape}, normals: {normals.shape}, uvs: {uvs.shape}")
    print("Vertices:")
    print(vertices)
    print("Indices:")
    print(indices)

if __name__ == "__main__":
    main()
