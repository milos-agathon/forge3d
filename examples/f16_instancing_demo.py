# examples/f16_instancing_demo.py
# Demonstrates CPU instancing fallback: duplicating a base mesh with transforms.

from __future__ import annotations

import numpy as np

from forge3d.geometry import primitive_mesh, instance_mesh
from forge3d.io import save_obj


def main() -> None:
    base = primitive_mesh("box")

    # Build a small grid of transforms (row-major 4x4 flattened)
    def translate(x: float, y: float, z: float) -> np.ndarray:
        T = np.eye(4, dtype=np.float32)
        # Row-major 4x4: last column stores translation
        T[0, 3] = float(x)
        T[1, 3] = float(y)
        T[2, 3] = float(z)
        return T.reshape(-1)

    transforms = []
    for gx in range(3):
        for gy in range(2):
            transforms.append(translate(gx * 1.5, gy * 1.0, 0.0))
    trs = np.stack(transforms, axis=0)

    inst = instance_mesh(base, trs)
    print(f"Instanced mesh: V={inst.vertex_count} T={inst.triangle_count}")
    try:
        save_obj(inst, "instanced_boxes.obj")
        print("Saved: instanced_boxes.obj")
    except Exception as e:
        print(f"OBJ export skipped/failed: {e}")


if __name__ == "__main__":
    main()
