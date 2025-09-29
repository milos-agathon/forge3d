# examples/f16_instancing_gpu_demo.py
# Demonstrates GPU instanced mesh rendering (feature-gated).

from __future__ import annotations

import numpy as np

from forge3d.geometry import (
    primitive_mesh,
    gpu_instancing_available,
    instance_mesh_gpu_render,
)


def main() -> None:
    if not gpu_instancing_available():
        print("GPU instancing is not available in this build. Rebuild with Cargo feature 'enable-gpu-instancing'.")
        return

    base = primitive_mesh("box")

    # Build a grid of transforms (row-major 4x4)
    def translate(x: float, y: float, z: float) -> np.ndarray:
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = float(x)
        T[1, 3] = float(y)
        T[2, 3] = float(z)
        return T.reshape(-1)

    transforms = []
    for gx in range(5):
        for gy in range(4):
            transforms.append(translate(gx * 1.5, gy * 1.2, 0.0))
    trs = np.stack(transforms, axis=0)

    rgba = instance_mesh_gpu_render(base, trs, width=640, height=480)
    print(f"Rendered image shape: {rgba.shape}, dtype={rgba.dtype}")

    # Optional: save using Pillow if available
    try:
        from PIL import Image

        Image.fromarray(rgba, mode="RGBA").save("instanced_gpu.png")
        print("Saved: instanced_gpu.png")
    except Exception as e:  # noqa: BLE001
        print(f"Saving via Pillow skipped/failed: {e}")


if __name__ == "__main__":
    main()
