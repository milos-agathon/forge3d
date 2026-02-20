"""P2.1 Point Cloud GPU Integration tests.

These tests exercise a real integration slice:
1. Read LAZ fixture metadata/points through native decode.
2. Traverse dataset nodes through Python LOD traversal.
3. Build native PointBuffer objects and produce GPU-ready interleaved buffers.

This is intentionally stronger than a proxy import: it runs end-to-end fixture
decode + traversal + native buffer packing.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from forge3d._native import NATIVE_AVAILABLE, get_native_module

if not NATIVE_AVAILABLE:
    pytest.skip(
        "Point cloud integration tests require the compiled _forge3d extension",
        allow_module_level=True,
    )

_native = get_native_module()


def _fixture_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "assets",
        "lidar",
        "MtStHelens.laz",
    )


class TestPointCloudGpuIntegration:
    def test_fixture_exists(self):
        assert os.path.isfile(_fixture_path())

    def test_laz_decode_to_native_gpu_buffer(self):
        """Decode fixture sample points, then produce native [x,y,z,r,g,b] GPU buffer."""
        point_count, coords, has_rgb = _native.read_laz_points_info(_fixture_path())
        assert point_count > 0
        assert len(coords) == 9  # first 3 points, xyz each

        positions = [float(c) for c in coords]
        colors = [255, 255, 255] * (len(positions) // 3) if has_rgb else None

        pb = _native.PointBuffer(positions, colors)
        gpu = pb.create_gpu_buffer()

        assert isinstance(gpu, np.ndarray)
        assert gpu.dtype == np.float32
        assert gpu.shape == (pb.point_count * 6,)

    def test_viewer_interleaved_layout_stride(self):
        """Viewer interleaving matches PointInstance3D stride: 12 floats / 48 bytes per point."""
        pb = _native.PointBuffer(
            [0.0, 10.0, 0.0, 1.0, 20.0, 2.0],
            [255, 0, 0, 0, 255, 0],
        )
        vbuf = pb.create_viewer_gpu_buffer([0.0, 0.0, 0.0], [10.0, 20.0, 10.0])
        assert isinstance(vbuf, np.ndarray)
        assert vbuf.dtype == np.float32
        assert vbuf.shape == (2 * 12,)
        assert vbuf.nbytes == 2 * 48

    def test_python_lod_traversal_with_fixture(self):
        """LOD traversal over fixture dataset returns at least one visible node."""
        from forge3d import pointcloud as pc

        dataset = pc.open_copc(_fixture_path())
        assert dataset.total_points > 0
        assert dataset.bounds is not None

        cx, cy, cz = dataset.bounds.center()
        camera = (cx, cy + 500.0, cz + 500.0)

        renderer = pc.PointCloudRenderer(point_budget=100_000, max_depth=6)
        visible = renderer.get_visible_nodes(dataset, camera)

        assert len(visible) > 0
        assert all(node.point_count >= 0 for node in visible)
