# examples/textured_sphere.py
# Minimal example creating a textured material and rendering a small image.
# Exists to demonstrate the Python API and writes to out/ per guidelines.
# RELEVANT FILES:python/forge3d/path_tracing.py,python/forge3d/textures.py,python/forge3d/materials.py

from pathlib import Path
import numpy as np

from forge3d.path_tracing import PathTracer, make_camera
from forge3d.textures import build_pbr_textures
from forge3d.materials import PbrMaterial


def _checker_rgba8(w=128, h=128):
    img = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            v = 255 if ((x // 8 + y // 8) % 2) == 0 else 0
            img[y, x] = (v, v, v, 255)
    return img


def main():
    out_dir = Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)

    base = _checker_rgba8()
    mr = np.zeros((128, 128, 4), dtype=np.uint8)
    mr[..., 1] = 128  # roughness
    mr[..., 2] = 0    # metallic
    nrm = np.zeros((128, 128, 4), dtype=np.uint8)
    nrm[..., 0] = 128
    nrm[..., 1] = 255
    nrm[..., 2] = 128
    nrm[..., 3] = 255

    tex = build_pbr_textures(base_color=base, metallic_roughness=mr, normal=nrm)
    mat = PbrMaterial().with_textures(tex)
    cam = make_camera(
        origin=(0, 0, 2.5), look_at=(0, 0, 0), up=(0, 1, 0), fov_y=45.0, aspect=1.0, exposure=1.0
    )

    img = PathTracer().render_rgba(256, 256, camera=cam, material=mat, use_gpu=False)
    # Save via numpy to a raw RGBA file to avoid extra deps.
    (out_dir / "textured_sphere.rgba").write_bytes(img.tobytes())
    print(str(out_dir / "textured_sphere.rgba"))


if __name__ == "__main__":
    main()

