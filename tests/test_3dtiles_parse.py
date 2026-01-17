"""P5.1: 3D Tiles parsing tests.

Tests tileset.json parsing, tile tree structure, and bounding volume handling.
"""

import pytest
import json
import tempfile
from pathlib import Path

from forge3d.tiles3d import (
    load_tileset,
    Tileset,
    Tile,
    BoundingVolume,
    TileContent,
    _parse_bounding_volume,
    _parse_tile,
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
