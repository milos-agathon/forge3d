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
import struct

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


def _write_tiny_copc(path: str) -> None:
    header = bytearray(375)
    header[0:4] = b"LASF"
    header[24] = 1
    header[25] = 4
    header[94:96] = (375).to_bytes(2, "little")
    header[100:104] = (1).to_bytes(4, "little")
    header[104] = 3
    header[105:107] = (34).to_bytes(2, "little")
    header[247:255] = (2).to_bytes(8, "little")
    header[131:139] = struct.pack("<d", 0.01)
    header[139:147] = struct.pack("<d", 0.01)
    header[147:155] = struct.pack("<d", 0.01)
    header[155:163] = struct.pack("<d", 100.0)
    header[163:171] = struct.pack("<d", 200.0)
    header[171:179] = struct.pack("<d", 300.0)
    header[179:187] = struct.pack("<d", 101.0)
    header[187:195] = struct.pack("<d", 100.0)
    header[203:211] = struct.pack("<d", 202.0)
    header[211:219] = struct.pack("<d", 200.0)
    header[227:235] = struct.pack("<d", 303.0)
    header[235:243] = struct.pack("<d", 300.0)

    copc_info = bytearray(72)
    copc_info[0:8] = struct.pack("<d", 100.5)
    copc_info[8:16] = struct.pack("<d", 201.0)
    copc_info[16:24] = struct.pack("<d", 301.5)
    copc_info[24:32] = struct.pack("<d", 4.0)
    copc_info[32:40] = struct.pack("<d", 1.0)
    root_hier_offset = 375 + 54 + len(copc_info)
    chunk_offset = root_hier_offset + 32
    copc_info[40:48] = root_hier_offset.to_bytes(8, "little")
    copc_info[48:56] = (32).to_bytes(8, "little")

    vlr = bytearray(54)
    vlr[2:6] = b"copc"
    vlr[18:20] = (1).to_bytes(2, "little")
    vlr[20:22] = len(copc_info).to_bytes(2, "little")

    hierarchy = bytearray(32)
    hierarchy[16:24] = chunk_offset.to_bytes(8, "little")
    hierarchy[24:28] = (68).to_bytes(4, "little", signed=True)
    hierarchy[28:32] = (2).to_bytes(4, "little", signed=True)

    def record(x: int, y: int, z: int, intensity: int, classification: int, rgb: tuple[int, int, int]) -> bytes:
        data = bytearray(34)
        data[0:4] = x.to_bytes(4, "little", signed=True)
        data[4:8] = y.to_bytes(4, "little", signed=True)
        data[8:12] = z.to_bytes(4, "little", signed=True)
        data[12:14] = intensity.to_bytes(2, "little")
        data[15] = classification
        data[28:30] = rgb[0].to_bytes(2, "little")
        data[30:32] = rgb[1].to_bytes(2, "little")
        data[32:34] = rgb[2].to_bytes(2, "little")
        return bytes(data)

    chunk = record(100, 200, 300, 42, 2, (0xFFFF, 0x8000, 0)) + record(
        110, 210, 310, 7, 6, (0, 0x4000, 0xFFFF)
    )
    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(vlr)
        handle.write(copc_info)
        handle.write(hierarchy)
        handle.write(chunk)


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

    def test_laz_attributes_python_facade(self):
        """Python pointcloud facade exposes classification/intensity samples from native decode."""
        from forge3d import pointcloud as pc

        attrs = pc.read_laz_point_attributes(_fixture_path(), sample_count=3)

        assert attrs["point_count"] > 0
        assert attrs["coords"].shape == (3, 3)
        assert attrs["coords"].dtype == np.float64
        assert attrs["intensities"].shape == (3,)
        assert attrs["intensities"].dtype == np.uint16
        assert attrs["classifications"].shape == (3,)
        assert attrs["classifications"].dtype == np.uint8

    def test_native_copc_node_read_preserves_attributes(self, tmp_path):
        """Native COPC node read returns positions, colors, intensity, and classification."""
        from forge3d import pointcloud as pc

        path = tmp_path / "tiny.copc.laz"
        _write_tiny_copc(str(path))

        native = _native.copc_read_node_points(str(path), 0, 0, 0, 0, None)
        np.testing.assert_allclose(
            np.asarray(native["positions"], dtype=np.float32).reshape((-1, 3)),
            np.array([[101.0, 202.0, 303.0], [101.1, 202.1, 303.1]], dtype=np.float32),
        )
        np.testing.assert_array_equal(native["colors"], np.array([255, 128, 0, 0, 64, 255], dtype=np.uint8))
        np.testing.assert_array_equal(native["intensities"], np.array([42, 7], dtype=np.uint16))
        np.testing.assert_array_equal(native["classifications"], np.array([2, 6], dtype=np.uint8))

        dataset = pc.open_copc(path)
        assert isinstance(dataset, pc.CopcDataset)
        data = dataset.read_points(budget=1)
        assert data.positions.shape == (1, 3)
        np.testing.assert_array_equal(data.classifications, np.array([2], dtype=np.uint8))

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
