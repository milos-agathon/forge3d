from _import_shim import ensure_repo_import
ensure_repo_import()

try:
    import forge3d as f3d
    from forge3d.geometry import primitive_mesh, gpu_instancing_available
except Exception:
    print("forge3d extension not available; skipping demo.")
    import sys; sys.exit(0)

import numpy as np
from pathlib import Path


def _grid_transforms(nx: int, ny: int, dx: float, dy: float, z: float = 0.0) -> np.ndarray:
    trs = []
    for j in range(ny):
        for i in range(nx):
            T = np.eye(4, dtype=np.float32)
            T[0, 3] = (i - (nx - 1) * 0.5) * dx
            T[1, 3] = (j - (ny - 1) * 0.5) * dy
            T[2, 3] = z
            trs.append(T.reshape(-1))
    return np.stack(trs, axis=0)


def main() -> None:
    if not gpu_instancing_available():
        print("GPU instancing not available. Rebuild with Cargo feature 'enable-gpu-instancing'.")
        return

    # Create a Scene
    W, H = 800, 600
    s = f3d.Scene(W, H, grid=128, colormap="viridis")

    # Build a simple base mesh (box)
    base = primitive_mesh("box")
    positions = np.asarray(base.positions, dtype=np.float32)
    normals = np.asarray(base.normals, dtype=np.float32)
    indices = np.asarray(base.indices, dtype=np.uint32).reshape(-1, 3)

    # Create a grid of instances
    transforms = _grid_transforms(8, 6, dx=1.5, dy=1.2, z=0.0)

    # Add instanced mesh to the Scene
    batch_idx = s.add_instanced_mesh(positions, indices, transforms, normals=normals,
                                     color=(0.85, 0.85, 0.9, 1.0),
                                     light_dir=(0.3, 0.7, 0.2), light_intensity=1.2)
    print("Added instanced batch index:", batch_idx)

    # Set camera and render
    s.set_camera_look_at((6.0, 5.0, 9.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 45.0, 0.1, 100.0)
    out = s.render_rgba()
    print("Rendered RGBA:", out.shape, out.dtype)

    try:
        from PIL import Image
        Image.fromarray(out, mode="RGBA").save(Path("scene_instancing_gpu.png"))
        print("Saved: scene_instancing_gpu.png")
    except Exception as e:  # pillow optional
        print("Skipping save (Pillow not available):", e)


if __name__ == "__main__":
    main()
