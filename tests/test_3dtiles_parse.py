"""P5.1: 3D Tiles parsing tests.

Tests tileset.json parsing, tile tree structure, and bounding volume handling.
"""

import pytest
import json
import struct
import tempfile
from pathlib import Path

import numpy as np
from forge3d.tiles3d import (
    load_tileset,
    Tile,
    TileContent,
    BoundingVolume,
    _parse_bounding_volume,
    _parse_tile,
    decode_b3dm,
    decode_pnts,
    Tiles3dRenderer,
    Tileset,
)
from forge3d._native import NATIVE_AVAILABLE, get_native_module
from forge3d.map_scene import (
    _pick_tiles3d_projected,
    _project_tiles3d_perspective,
    _transform_tiles3d_positions,
)


def _pad4(data: bytes, fill: bytes = b" ") -> bytes:
    return data + fill * ((4 - len(data) % 4) % 4)


def _minimal_b3dm_triangle() -> bytes:
    positions = struct.pack(
        "<9f",
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    )
    indices = struct.pack("<3H", 0, 1, 2)
    bin_chunk = _pad4(positions + indices, b"\x00")
    gltf_json = _pad4(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "buffers": [{"byteLength": len(positions) + len(indices)}],
                "bufferViews": [
                    {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
                    {"buffer": 0, "byteOffset": len(positions), "byteLength": len(indices)},
                ],
                "accessors": [
                    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                    {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
                ],
                "meshes": [
                    {"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}
                ],
            },
            separators=(",", ":"),
        ).encode("utf-8"),
        b" ",
    )
    glb_length = 12 + 8 + len(gltf_json) + 8 + len(bin_chunk)
    glb = b"".join(
        [
            b"glTF",
            struct.pack("<II", 2, glb_length),
            struct.pack("<II", len(gltf_json), 0x4E4F534A),
            gltf_json,
            struct.pack("<II", len(bin_chunk), 0x004E4942),
            bin_chunk,
        ]
    )
    feature_table = b"{}  "
    byte_length = 28 + len(feature_table) + len(glb)
    return b"".join(
        [
            b"b3dm",
            struct.pack("<IIIIII", 1, byte_length, len(feature_table), 0, 0, 0),
            feature_table,
            glb,
        ]
    )


def _minimal_pnts_points() -> bytes:
    positions = struct.pack("<6f", 0.0, 0.0, 0.0, 1.0, 2.0, 3.0)
    colors = bytes([255, 0, 0, 0, 255, 0])
    feature_table = _pad4(
        json.dumps(
            {
                "POINTS_LENGTH": 2,
                "POSITION": {"byteOffset": 0},
                "RGB": {"byteOffset": len(positions)},
            },
            separators=(",", ":"),
        ).encode("utf-8"),
        b" ",
    )
    feature_binary = _pad4(positions + colors, b"\x00")
    byte_length = 28 + len(feature_table) + len(feature_binary)
    return b"".join(
        [
            b"pnts",
            struct.pack("<IIIIII", 1, byte_length, len(feature_table), len(feature_binary), 0, 0),
            feature_table,
            feature_binary,
        ]
    )


def _pnts_with_rtc(center_x: float = 6_378_137.0) -> bytes:
    positions = struct.pack("<6f", 0.0, 0.0, 0.0, 0.001, 0.0, 0.0)
    feature_table = _pad4(
        json.dumps(
            {
                "POINTS_LENGTH": 2,
                "POSITION": {"byteOffset": 0},
                "RTC_CENTER": [center_x, 0.0, 0.0],
            },
            separators=(",", ":"),
        ).encode("utf-8"),
        b" ",
    )
    byte_length = 28 + len(feature_table) + len(positions)
    return b"".join(
        [
            b"pnts",
            struct.pack("<IIIIII", 1, byte_length, len(feature_table), len(positions), 0, 0),
            feature_table,
            positions,
        ]
    )


def _pnts_quantized_with_rtc() -> bytes:
    positions = struct.pack("<6H", 0, 0, 0, 1, 0, 0)
    feature_table = _pad4(
        json.dumps(
            {
                "POINTS_LENGTH": 2,
                "POSITION_QUANTIZED": {"byteOffset": 0},
                "QUANTIZED_VOLUME_OFFSET": [10.0, 0.0, 0.0],
                "QUANTIZED_VOLUME_SCALE": [65.535, 1.0, 1.0],
                "RTC_CENTER": [6_378_137.0, 0.0, 0.0],
            },
            separators=(",", ":"),
        ).encode("utf-8"),
        b" ",
    )
    byte_length = 28 + len(feature_table) + len(positions)
    return b"".join(
        [
            b"pnts",
            struct.pack("<IIIIII", 1, byte_length, len(feature_table), len(positions), 0, 0),
            feature_table,
            positions,
        ]
    )


class TestBoundingVolume:
    """Tests for BoundingVolume parsing and methods."""

    def test_parse_sphere(self):
        data = {"sphere": [1.0, 2.0, 3.0, 10.0]}
        bv = _parse_bounding_volume(data)
        assert bv.volume_type == "sphere"
        assert bv.data == [1.0, 2.0, 3.0, 10.0]

    def test_parse_box(self):
        data = {"box": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]}
        bv = _parse_bounding_volume(data)
        assert bv.volume_type == "box"
        assert len(bv.data) == 12

    def test_parse_region(self):
        data = {"region": [-1.0, -0.5, 1.0, 0.5, 0.0, 100.0]}
        bv = _parse_bounding_volume(data)
        assert bv.volume_type == "region"
        assert len(bv.data) == 6

    def test_sphere_center(self):
        bv = BoundingVolume("sphere", [10.0, 20.0, 30.0, 5.0])
        center = bv.center()
        assert center == (10.0, 20.0, 30.0)

    def test_sphere_radius(self):
        bv = BoundingVolume("sphere", [0.0, 0.0, 0.0, 15.0])
        assert bv.radius() == 15.0


class TestTileParsing:
    """Tests for Tile parsing."""

    def test_parse_minimal_tile(self):
        data = {
            "boundingVolume": {"sphere": [0, 0, 0, 100]},
            "geometricError": 50.0,
        }
        tile = _parse_tile(data)
        assert tile.geometric_error == 50.0
        assert tile.refine == "REPLACE"
        assert tile.content is None
        assert len(tile.children) == 0

    def test_parse_tile_with_content(self):
        data = {
            "boundingVolume": {"sphere": [0, 0, 0, 100]},
            "geometricError": 50.0,
            "content": {"uri": "tile.b3dm"},
        }
        tile = _parse_tile(data)
        assert tile.has_content()
        assert tile.content_uri() == "tile.b3dm"

    def test_parse_tile_with_children(self):
        data = {
            "boundingVolume": {"sphere": [0, 0, 0, 100]},
            "geometricError": 50.0,
            "children": [
                {
                    "boundingVolume": {"sphere": [-50, 0, 0, 50]},
                    "geometricError": 10.0,
                },
                {
                    "boundingVolume": {"sphere": [50, 0, 0, 50]},
                    "geometricError": 10.0,
                },
            ],
        }
        tile = _parse_tile(data)
        assert len(tile.children) == 2
        assert tile.count_tiles() == 3
        assert tile.max_depth() == 2

    def test_parse_tile_refine_add(self):
        data = {
            "boundingVolume": {"sphere": [0, 0, 0, 100]},
            "geometricError": 50.0,
            "refine": "ADD",
        }
        tile = _parse_tile(data)
        assert tile.refine == "ADD"


class TestTilesetParsing:
    """Tests for Tileset parsing."""

    def test_load_minimal_tileset(self):
        tileset_json = {
            "asset": {"version": "1.0"},
            "geometricError": 500.0,
            "root": {
                "boundingVolume": {"sphere": [0, 0, 0, 100]},
                "geometricError": 100.0,
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(tileset_json, f)
            f.flush()
            
            tileset = load_tileset(f.name)
            
            assert tileset.version == "1.0"
            assert tileset.geometric_error == 500.0
            assert tileset.tile_count == 1
            assert tileset.max_depth == 1

    def test_load_tileset_with_hierarchy(self):
        tileset_json = {
            "asset": {"version": "1.1"},
            "geometricError": 1000.0,
            "root": {
                "boundingVolume": {"sphere": [0, 0, 0, 200]},
                "geometricError": 200.0,
                "refine": "REPLACE",
                "children": [
                    {
                        "boundingVolume": {"sphere": [-100, 0, 0, 100]},
                        "geometricError": 50.0,
                        "content": {"uri": "left.b3dm"},
                        "children": [
                            {
                                "boundingVolume": {"sphere": [-100, -50, 0, 50]},
                                "geometricError": 10.0,
                                "content": {"uri": "left_front.b3dm"},
                            },
                            {
                                "boundingVolume": {"sphere": [-100, 50, 0, 50]},
                                "geometricError": 10.0,
                                "content": {"uri": "left_back.b3dm"},
                            },
                        ],
                    },
                    {
                        "boundingVolume": {"sphere": [100, 0, 0, 100]},
                        "geometricError": 50.0,
                        "content": {"uri": "right.b3dm"},
                    },
                ],
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(tileset_json, f)
            f.flush()
            
            tileset = load_tileset(f.name)
            
            assert tileset.version == "1.1"
            assert tileset.tile_count == 5
            assert tileset.max_depth == 3

    def test_resolve_relative_uri(self):
        tileset_json = {
            "asset": {"version": "1.0"},
            "geometricError": 100.0,
            "root": {
                "boundingVolume": {"sphere": [0, 0, 0, 100]},
                "geometricError": 50.0,
                "content": {"uri": "tiles/tile.b3dm"},
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tileset_path = Path(tmpdir) / "tileset.json"
            with open(tileset_path, "w") as f:
                json.dump(tileset_json, f)
            
            tileset = load_tileset(tileset_path)
            resolved = tileset.resolve_uri("tiles/tile.b3dm")
            
            assert resolved == Path(tmpdir) / "tiles/tile.b3dm"

    def test_nested_parent_and_tile_transforms_are_composed_in_f64(self, tmp_path):
        translate = lambda x: [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            x, 0.0, 0.0, 1.0,
        ]
        root = Tile(
            bounding_volume=BoundingVolume("sphere", [0.0, 0.0, 0.0, 100.0]),
            geometric_error=0.0,
            transform=translate(6_378_137.0),
            children=[
                Tile(
                    bounding_volume=BoundingVolume("sphere", [0.0, 0.0, 0.0, 1.0]),
                    geometric_error=0.0,
                    content=TileContent("a.pnts"),
                    transform=translate(10.0),
                ),
                Tile(
                    bounding_volume=BoundingVolume("sphere", [0.0, 0.0, 0.0, 1.0]),
                    geometric_error=0.0,
                    content=TileContent("b.pnts"),
                    transform=translate(20.0),
                ),
            ],
        )
        tileset = Tileset(tmp_path, "1.1", 0.0, root)
        visible = Tiles3dRenderer(sse_threshold=16.0).get_visible_tiles(
            tileset, (6_378_137.0, 0.0, 10.0)
        )
        translations = sorted(float(tile.world_transform[0, 3]) for tile in visible)
        assert translations == [6_378_147.0, 6_378_157.0]


class TestB3dmDecode:
    def test_decode_b3dm_uses_native_geometry_extraction(self):
        decoded = decode_b3dm(_minimal_b3dm_triangle())

        assert decoded["vertex_count"] == 3
        assert decoded["triangle_count"] == 1
        assert decoded["positions"].shape == (9,)
        assert decoded["indices"].tolist() == [0, 1, 2]
        assert decoded["feature_table"] == {}


class TestPntsDecode:
    @pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native extension unavailable")
    def test_decode_pnts_uses_native_point_extraction(self):
        assert hasattr(get_native_module(), "decode_pnts_py")

        decoded = decode_pnts(_minimal_pnts_points())

        assert decoded["point_count"] == 2
        assert decoded["positions"].shape == (2, 3)
        assert decoded["colors"].shape == (2, 3)
        np.testing.assert_array_equal(decoded["colors"][0], np.array([255, 0, 0], dtype=np.uint8))

    @pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native extension unavailable")
    def test_rtc_center_is_applied_in_f64_and_relative_positions_remain_available(self):
        decoded = decode_pnts(_pnts_with_rtc())
        assert decoded["positions"].dtype == np.float64
        assert decoded["relative_positions"].dtype == np.float64
        assert decoded["positions"][0, 0] == 6_378_137.0
        assert abs(decoded["positions"][1, 0] - 6_378_137.001) < 1e-9
        assert tuple(decoded["rtc_center"]) == (6_378_137.0, 0.0, 0.0)
        second = decode_pnts(_pnts_with_rtc(6_378_157.0))
        assert tuple(second["rtc_center"]) == (6_378_157.0, 0.0, 0.0)
        assert second["positions"][0, 0] - decoded["positions"][0, 0] == 20.0

    @pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native extension unavailable")
    def test_quantized_offset_scale_and_rtc_compose_in_f64(self):
        decoded = decode_pnts(_pnts_quantized_with_rtc())
        assert decoded["positions"].dtype == np.float64
        assert decoded["positions"][0, 0] == 6_378_147.0
        assert decoded["positions"][1, 0] - decoded["positions"][0, 0] == pytest.approx(
            0.001, abs=1e-9
        )


def test_shared_tiles3d_projection_culling_and_picking_are_rebase_invariant():
    earth = np.array([6_378_137.0, 2_000_000.0, -1_000_000.0], dtype=np.float64)
    local = np.array(
        [[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [10.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = earth
    world = _transform_tiles3d_positions(local, transform.reshape(16, order="F"))
    camera = {
        "camera_position": (earth + np.array([0.0, 0.0, 50.0])).tolist(),
        "camera_target": earth.tolist(),
        "camera_up": [0.0, 1.0, 0.0],
        "fov_y_deg": 45.0,
    }
    projected = np.asarray(_project_tiles3d_perspective(world, camera, 640, 400))
    assert np.isfinite(projected).all()
    assert projected[0, 0] != projected[1, 0]

    picked = _pick_tiles3d_projected(
        world,
        camera,
        640,
        400,
        tuple(projected[1]),
        tolerance_ndc=1.0e-4,
    )
    assert picked == 1
    assert world[picked, 0] - earth[0] == pytest.approx(0.001, abs=1e-9)

    rigid = np.array([1_500.0, -750.0, 300.0], dtype=np.float64)
    shifted_camera = dict(camera)
    shifted_camera["camera_position"] = (
        np.asarray(camera["camera_position"], dtype=np.float64) + rigid
    ).tolist()
    shifted_camera["camera_target"] = (
        np.asarray(camera["camera_target"], dtype=np.float64) + rigid
    ).tolist()
    shifted = np.asarray(
        _project_tiles3d_perspective(world + rigid, shifted_camera, 640, 400)
    )
    np.testing.assert_allclose(shifted, projected, rtol=0.0, atol=1e-12)
    assert (
        _pick_tiles3d_projected(
            world + rigid,
            shifted_camera,
            640,
            400,
            tuple(shifted[1]),
            tolerance_ndc=1.0e-4,
        )
        == picked
    )

    behind = earth + np.array([0.0, 0.0, 100.0])
    culled = np.asarray(_project_tiles3d_perspective([behind], camera, 640, 400))
    assert np.isnan(culled).all()
    assert _pick_tiles3d_projected([behind], camera, 640, 400, (0.0, 0.0)) is None


class TestTileMetrics:
    """Tests for tile tree metrics."""

    def test_count_single_tile(self):
        tile = Tile(
            bounding_volume=BoundingVolume("sphere", [0, 0, 0, 10]),
            geometric_error=10.0,
        )
        assert tile.count_tiles() == 1

    def test_count_nested_tiles(self):
        child1 = Tile(
            bounding_volume=BoundingVolume("sphere", [-5, 0, 0, 5]),
            geometric_error=5.0,
        )
        child2 = Tile(
            bounding_volume=BoundingVolume("sphere", [5, 0, 0, 5]),
            geometric_error=5.0,
        )
        parent = Tile(
            bounding_volume=BoundingVolume("sphere", [0, 0, 0, 10]),
            geometric_error=10.0,
            children=[child1, child2],
        )
        assert parent.count_tiles() == 3

    def test_max_depth_single(self):
        tile = Tile(
            bounding_volume=BoundingVolume("sphere", [0, 0, 0, 10]),
            geometric_error=10.0,
        )
        assert tile.max_depth() == 1

    def test_max_depth_nested(self):
        grandchild = Tile(
            bounding_volume=BoundingVolume("sphere", [0, 0, 0, 2]),
            geometric_error=1.0,
        )
        child = Tile(
            bounding_volume=BoundingVolume("sphere", [0, 0, 0, 5]),
            geometric_error=5.0,
            children=[grandchild],
        )
        parent = Tile(
            bounding_volume=BoundingVolume("sphere", [0, 0, 0, 10]),
            geometric_error=10.0,
            children=[child],
        )
        assert parent.max_depth() == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
