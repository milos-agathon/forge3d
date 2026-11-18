"""Phase 19.2: imported meshes land in the Rendered frame at a known position.

The regression under test is a silent mismatch between the Solid viewport and
the still: a transform that is right in the 12-unit Y-up viewport frame but
wrong in the Z-up metre frame of the MapScene still. A cube placed at the
still's camera aim must appear at the centre of the frame; a model placed
elsewhere must move there. The ablation (remove the mesh from the request)
must return the exact baseline frame.

The affine conversion itself is unit-tested in Rust
(`workspace::tests::mesh_render_matrix_maps_viewport_units_to_the_still_frame`);
this test pins the service that consumes it end to end.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "python"))


def _service():
    import importlib.util

    path = REPO / "studio" / "src-tauri" / "src" / "studio_render.py"
    spec = importlib.util.spec_from_file_location("studio_render_service", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _decode(service, reply) -> np.ndarray:
    import base64
    import io

    from PIL import Image

    return np.asarray(
        Image.open(io.BytesIO(base64.b64decode(reply["png"]))).convert("RGB"), dtype=np.int32
    )


def _mesh_at_aim_placement(renderer, size: int) -> np.ndarray:
    """Row-major matrix placing the glb quad's centre at the camera aim.

    The glb fixture's two quads span x=0..3, y=0..1; its centre is (1.5, 0.5).
    Moving that point to viewport (0, half_relief / s, 0) puts it at render
    (0, 0, 0) through the same affine map the core implements.
    """
    span = max(renderer.span_x, renderer.span_y)
    s = span / 12.0
    low, high = renderer.range
    half_relief = (float(high) - float(low)) * renderer.z_scale / 2.0
    viewport_to_render = np.array(
        [
            [s, 0, 0, 0],
            [0, 0, -s, 0],
            [0, s, 0, -half_relief],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    object_transform = np.eye(4, dtype=np.float32)
    object_transform[0, 3] = -1.5
    object_transform[1, 3] = half_relief / s - 0.5
    return (viewport_to_render @ object_transform).reshape(16).tolist()


def test_imported_mesh_lands_at_the_still_camera_aim(tmp_path):
    from test_gltf_import import _write_material_glb

    service = _service()
    glb = tmp_path / "quad.glb"
    _write_material_glb(glb)

    dem = REPO / "assets" / "tif" / "Mount_Fuji_30m.tif"
    renderer = service.StudioRenderer(str(dem), z_scale=1.0)
    placement = _mesh_at_aim_placement(renderer, 256)
    camera = {
        "phiDeg": 90.0,
        "thetaDeg": 45.0,
        "radius": 24000.0,
        "target": [0.0, 0.0, 0.0],
        "fovDeg": 45.0,
    }
    sun = {"azimuthDeg": 315.0, "elevationDeg": 30.0, "intensity": 3.0, "color": [1.0, 0.97, 0.92]}
    base_request = {
        "width": 256,
        "height": 256,
        "camera": dict(camera),
        "sun": dict(sun),
        "exposure": 1.35,
        "quality": {"shadows": True},
    }

    baseline = _decode(service, renderer.render(dict(base_request)))
    with_mesh = dict(base_request, meshes=[{"path": str(glb), "transform": placement, "node": None}])
    reply = renderer.render(with_mesh)
    image = _decode(service, reply)

    assert reply["meshReport"]["meshesRendered"] == 1, reply["meshReport"]
    diff = np.abs(image - baseline).sum(axis=2)
    changed = diff > 8
    assert int(changed.sum()) > 20, "the imported mesh did not reach the still"

    # The mesh centre coincides with the camera aim, so its pixels must be
    # near the frame centre. This is the test that catches a wrong axis map.
    ys, xs = np.nonzero(changed)
    cx, cy = float(xs.mean()), float(ys.mean())
    assert abs(cx - 128.0) < 25.0 and abs(cy - 128.0) < 25.0, (
        f"mesh centroid ({cx:.1f}, {cy:.1f}) is not at the frame centre"
    )

    # Ablation: without the mesh key, the frame is byte-identical to baseline.
    without = renderer.render(dict(base_request))
    assert np.array_equal(_decode(service, without), baseline)


def test_node_scoped_mesh_is_reported_not_silently_rendered(tmp_path):
    from test_gltf_import import _write_material_glb

    service = _service()
    glb = tmp_path / "quad.glb"
    _write_material_glb(glb)
    renderer = service.StudioRenderer(str(REPO / "assets" / "tif" / "Mount_Fuji_30m.tif"), z_scale=1.0)
    request = {
        "width": 128,
        "height": 128,
        "camera": {"phiDeg": 90.0, "thetaDeg": 45.0, "radius": 24000.0, "target": [0, 0, 0], "fovDeg": 45.0},
        "sun": {"azimuthDeg": 315.0, "elevationDeg": 30.0, "intensity": 3.0, "color": [1.0, 0.97, 0.92]},
        "exposure": 1.35,
        "quality": {"shadows": True},
        "meshes": [{"path": str(glb), "transform": _mesh_at_aim_placement(renderer, 128), "node": "one-node"}],
    }
    reply = renderer.render(request)
    assert reply["meshReport"]["meshesRendered"] == 0
    assert reply["meshReport"]["meshesSkipped"] == ["one-node"]


def test_rendered_viewport_path_shows_the_mesh_at_an_off_aim_transform(tmp_path):
    """The viewport's preview-quality path uses the same service. A mesh moved
    to a different viewport position must project away from the frame centre;
    this proves the placement follows the object transform through the shared
    path rather than accidentally landing at the aim every time."""
    from test_gltf_import import _write_material_glb

    service = _service()
    glb = tmp_path / "quad.glb"
    _write_material_glb(glb)
    renderer = service.StudioRenderer(str(REPO / "assets" / "tif" / "Mount_Fuji_30m.tif"), z_scale=1.0)

    center = _mesh_at_aim_placement(renderer, 256)
    moved = np.asarray(center, dtype=np.float32).reshape(4, 4)
    moved[0, 3] += 3.0 * (max(renderer.span_x, renderer.span_y) / 12.0)  # 3 viewport units east
    request = {
        "width": 256,
        "height": 256,
        # A viewport-style orbit (different from the still's default) with the
        # frame kept at preview quality, as render_frame preview does.
        "camera": {
            "phiDeg": 135.0,
            "thetaDeg": 35.0,
            "radius": 26000.0,
            "target": [0.0, 0.0, 0.0],
            "fovDeg": 50.0,
        },
        "sun": {"azimuthDeg": 315.0, "elevationDeg": 30.0, "intensity": 3.0, "color": [1.0, 0.97, 0.92]},
        "exposure": 1.35,
        "quality": {"shadows": True},
        "meshes": [{"path": str(glb), "transform": moved.reshape(16).tolist(), "node": None}],
    }
    reply = renderer.render(request)
    assert reply["meshReport"]["meshesRendered"] == 1
    image = _decode(service, reply)
    # The terrain-only frame for the same camera is the ablation.
    baseline_request = {key: value for key, value in request.items() if key != "meshes"}
    baseline = _decode(service, renderer.render(baseline_request))
    changed = np.abs(image - baseline).sum(axis=2) > 8
    assert int(changed.sum()) > 20, "the moved mesh did not reach the preview frame"
    ys, xs = np.nonzero(changed)
    assert abs(float(xs.mean()) - 128.0) > 15.0 or abs(float(ys.mean()) - 128.0) > 15.0, (
        "a mesh moved 3 viewport units still projects at the centre"
    )
