"""P5.2: 3D Tiles SSE and traversal tests.

Tests screen-space error computation and LOD traversal behavior.
"""

import pytest
import json
import tempfile
from pathlib import Path

from forge3d.tiles3d import (
    load_tileset,
    Tiles3dRenderer,
    SseParams,
    BoundingVolume,
)


class TestSseComputation:
    """Tests for Screen-Space Error computation."""

    def test_sse_params_default(self):
        params = SseParams()
        assert params.viewport_height == 1080.0
        assert params.fov_y == pytest.approx(0.785398, rel=1e-3)

    def test_sse_factor(self):
        params = SseParams(viewport_height=1080.0, fov_y=0.785398)
        factor = params.sse_factor()
        # SSE factor should be viewport_height / (2 * tan(fov/2))
        assert factor > 0
        assert factor == pytest.approx(1305.0, rel=0.1)

    def test_sse_decreases_with_distance(self):
        renderer = Tiles3dRenderer()
        bv = BoundingVolume("sphere", [0.0, 0.0, 0.0, 10.0])
        
        sse_near = renderer.compute_sse(10.0, bv, (0.0, 0.0, 100.0))
        sse_far = renderer.compute_sse(10.0, bv, (0.0, 0.0, 1000.0))
        
        assert sse_near > sse_far

    def test_sse_increases_with_geometric_error(self):
        renderer = Tiles3dRenderer()
        bv = BoundingVolume("sphere", [0.0, 0.0, 0.0, 10.0])
        camera = (0.0, 0.0, 100.0)
        
        sse_small = renderer.compute_sse(1.0, bv, camera)
        sse_large = renderer.compute_sse(10.0, bv, camera)
        
        assert sse_large > sse_small

    def test_sse_at_zero_distance(self):
        renderer = Tiles3dRenderer()
        bv = BoundingVolume("sphere", [0.0, 0.0, 0.0, 10.0])
        camera = (0.0, 0.0, 0.0)  # At center
        
        sse = renderer.compute_sse(10.0, bv, camera)
        
        assert sse == float("inf")


class TestTraversal:
    """Tests for tileset traversal."""

    @pytest.fixture
    def sample_tileset_path(self):
        """Create a sample tileset for testing."""
        tileset_json = {
            "asset": {"version": "1.0"},
            "geometricError": 500.0,
            "root": {
                "boundingVolume": {"sphere": [0, 0, 0, 100]},
                "geometricError": 100.0,
                "refine": "REPLACE",
                "content": {"uri": "root.b3dm"},
                "children": [
                    {
                        "boundingVolume": {"sphere": [-50, 0, 0, 50]},
                        "geometricError": 20.0,
                        "content": {"uri": "left.b3dm"},
                        "children": [
                            {
                                "boundingVolume": {"sphere": [-50, -25, 0, 25]},
                                "geometricError": 5.0,
                                "content": {"uri": "left_front.b3dm"},
                            },
                            {
                                "boundingVolume": {"sphere": [-50, 25, 0, 25]},
                                "geometricError": 5.0,
                                "content": {"uri": "left_back.b3dm"},
                            },
                        ],
                    },
                    {
                        "boundingVolume": {"sphere": [50, 0, 0, 50]},
                        "geometricError": 20.0,
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
            yield f.name

    def test_visible_tiles_at_distance(self, sample_tileset_path):
        """Test that distant camera sees fewer tiles."""
        tileset = load_tileset(sample_tileset_path)
        renderer = Tiles3dRenderer(sse_threshold=16.0)
        
        # Far camera - should see root only
        far_camera = (0.0, 0.0, 10000.0)
        visible_far = renderer.get_visible_tiles(tileset, far_camera)
        
        # Near camera - should see more tiles
        near_camera = (0.0, 0.0, 50.0)
        visible_near = renderer.get_visible_tiles(tileset, near_camera)
        
        assert len(visible_near) >= len(visible_far)

    def test_sse_threshold_affects_tile_count(self, sample_tileset_path):
        """Test that higher SSE threshold results in fewer tiles."""
        tileset = load_tileset(sample_tileset_path)
        camera = (0.0, 0.0, 200.0)
        
        renderer_low = Tiles3dRenderer(sse_threshold=4.0)
        renderer_high = Tiles3dRenderer(sse_threshold=64.0)
        
        visible_low = renderer_low.get_visible_tiles(tileset, camera)
        visible_high = renderer_high.get_visible_tiles(tileset, camera)
        
        # Lower threshold = more refinement = more tiles
        assert len(visible_low) >= len(visible_high)

    def test_visible_tiles_have_content(self, sample_tileset_path):
        """Test that all visible tiles have content URIs."""
        tileset = load_tileset(sample_tileset_path)
        renderer = Tiles3dRenderer(sse_threshold=16.0)
        camera = (0.0, 0.0, 200.0)
        
        visible = renderer.get_visible_tiles(tileset, camera)
        
        for vt in visible:
            assert vt.tile.has_content()

    def test_max_depth_respected(self, sample_tileset_path):
        """Test that max depth limit is respected."""
        tileset = load_tileset(sample_tileset_path)
        renderer = Tiles3dRenderer(sse_threshold=1.0, max_depth=1)  # Very low SSE, max depth 1
        camera = (0.0, 0.0, 10.0)  # Very close
        
        visible = renderer.get_visible_tiles(tileset, camera)
        
        for vt in visible:
            assert vt.depth <= 1

    def test_sse_monotonicity(self, sample_tileset_path):
        """Test that tile count generally decreases with SSE threshold."""
        tileset = load_tileset(sample_tileset_path)
        camera = (0.0, 0.0, 200.0)
        
        sse_values = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
        counts = []
        
        for sse in sse_values:
            renderer = Tiles3dRenderer(sse_threshold=sse)
            visible = renderer.get_visible_tiles(tileset, camera)
            counts.append(len(visible))
        
        # Check that counts are non-increasing (monotonic decreasing)
        for i in range(len(counts) - 1):
            assert counts[i] >= counts[i + 1], \
                f"Tile count should decrease: SSE {sse_values[i]}={counts[i]}, SSE {sse_values[i+1]}={counts[i+1]}"


class TestCacheStats:
    """Tests for renderer cache statistics."""

    def test_cache_stats_initial(self):
        renderer = Tiles3dRenderer()
        stats = renderer.cache_stats()
        
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["entries"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
