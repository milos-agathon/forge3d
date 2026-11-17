"""The instanced mesh renderer must accept the caller's camera.

Until now `instance_mesh_gpu_render` hard-coded its eye at (3, 2, 3) looking at
the origin (`src/render/instancing.rs`). A mesh composited over a terrain frame
therefore cannot share the terrain's camera: the shell would draw the model
from one viewpoint and the ground from another. The renderer itself already
supports `set_view_proj`; this test pins the Python contract that exposes an
orbit camera (eye/target/up/fov/aspect), with the engine building the matrices.
"""

from __future__ import annotations

import numpy as np
import pytest


def _camera(eye: tuple[float, float, float]) -> dict[str, list[float]]:
    """An orbit camera — the engine builds view/proj with its own helpers."""
    return {"eye": list(eye), "target": [0.0, 0.0, 0.0]}


def test_instance_mesh_gpu_render_accepts_the_callers_camera() -> None:
    import forge3d.geometry as geometry

    if not hasattr(geometry, "instance_mesh_gpu_render"):
        pytest.skip("GPU instancing is not compiled in")
    # The GPU gate skips itself on an adapter that cannot render; this test is
    # about the argument contract, so it still asserts the call is accepted.
    mesh = geometry.primitive_mesh("sphere")
    transforms = np.eye(4, dtype=np.float32).reshape(1, 16)

    near = geometry.instance_mesh_gpu_render(mesh, transforms, 128, 128, camera=_camera((2.0, 1.5, 2.0)))
    far = geometry.instance_mesh_gpu_render(mesh, transforms, 128, 128, camera=_camera((6.0, 1.5, 2.0)))

    assert near.shape == (128, 128, 4) and near.dtype == np.uint8
    assert far.shape == near.shape
    # Two different eye positions must move the mesh in the frame: a renderer
    # that ignores `camera` would produce byte-identical images here.
    assert not np.array_equal(near, far), "camera was ignored by the mesh renderer"


def test_instance_mesh_gpu_render_rejects_a_malformed_camera() -> None:
    import forge3d.geometry as geometry

    if not hasattr(geometry, "instance_mesh_gpu_render"):
        pytest.skip("GPU instancing is not compiled in")
    mesh = geometry.primitive_mesh("sphere")
    transforms = np.eye(4, dtype=np.float32).reshape(1, 16)
    with pytest.raises(ValueError, match="camera"):
        geometry.instance_mesh_gpu_render(mesh, transforms, 64, 64, camera={"eye": [1.0, 2.0]})
