# tests/test_pbr_textures_gpu.py
# GPU-aware smoke test for textured rendering path.
# Exists to assert API shape and basic behavioral delta with textures while skipping on non-GPU CI.
# RELEVANT FILES:python/forge3d/path_tracing.py,python/forge3d/materials.py,python/forge3d/textures.py

import os
import numpy as np
import pytest

from forge3d.path_tracing import PathTracer, make_camera
from forge3d.textures import build_pbr_textures
from forge3d.materials import PbrMaterial


def _checker_rgba8(w=64, h=64):
    img = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            v = 255 if ((x // 8 + y // 8) % 2) == 0 else 0
            img[y, x] = (v, v, v, 255)
    return img


_HAS_GPU = os.environ.get("FORGE3D_FORCE_GPU", "0") == "1"


@pytest.mark.skipif(not _HAS_GPU, reason="No compatible GPU adapter in this environment")
def test_textured_sphere_normalmap_affects_shading():
    base = _checker_rgba8()
    mr = np.zeros((64, 64, 4), dtype=np.uint8)
    mr[..., 1] = 64  # roughness in G
    mr[..., 2] = 0   # metallic in B
    nrm = np.zeros((64, 64, 4), dtype=np.uint8)
    nrm[..., 0] = 128
    nrm[..., 1] = 255
    nrm[..., 2] = 128
    nrm[..., 3] = 255

    tex = build_pbr_textures(base_color=base, metallic_roughness=mr, normal=nrm, emissive=None)
    mat = PbrMaterial().with_textures(tex)
    cam = make_camera(
        origin=(0, 0, 2.5),
        look_at=(0, 0, 0),
        up=(0, 1, 0),
        fov_y=45.0,
        aspect=1.0,
        exposure=1.0,
    )
    tr = PathTracer()
    img_no = tr.render_rgba(64, 64, scene=None, camera=cam, material=PbrMaterial(), seed=7, frames=1, use_gpu=True)
    img_tx = tr.render_rgba(64, 64, scene=None, camera=cam, material=mat, seed=7, frames=1, use_gpu=True)
    assert img_tx.shape == img_no.shape
    mad = np.abs(img_tx.astype(np.int16) - img_no.astype(np.int16)).mean()
    assert mad > 1.0

