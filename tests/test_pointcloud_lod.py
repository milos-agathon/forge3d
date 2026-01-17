"""P5.5: Point Cloud LOD traversal tests.

Tests octree traversal and point budget enforcement.
"""

import pytest
import numpy as np

from forge3d.pointcloud import (
    OctreeKey,
    OctreeBounds,
    OctreeNode,
    PointCloudRenderer,
    VisibleNode,
)


class TestOctreeKey:
    """Tests for OctreeKey."""

    def test_root_key(self):
        key = OctreeKey.root()
        assert key.depth == 0
        assert key.x == 0
        assert key.y == 0
        assert key.z == 0

    def test_child_key(self):
        root = OctreeKey.root()
        
        # Octant 0 (---) 
        child0 = root.child(0)
        assert child0.depth == 1
        assert child0.x == 0
        assert child0.y == 0
        assert child0.z == 0
        
        # Octant 7 (+++)
        child7 = root.child(7)
        assert child7.depth == 1
        assert child7.x == 1
        assert child7.y == 1
        assert child7.z == 1

    def test_key_to_string(self):
        key = OctreeKey(depth=2, x=3, y=1, z=2)
        assert str(key) == "2-3-1-2"

    def test_key_from_string(self):
        key = OctreeKey.from_str("2-3-1-2")
        assert key is not None
        assert key.depth == 2
        assert key.x == 3
        assert key.y == 1
        assert key.z == 2

    def test_key_from_invalid_string(self):
        assert OctreeKey.from_str("invalid") is None
        assert OctreeKey.from_str("1-2") is None
        assert OctreeKey.from_str("a-b-c-d") is None


class TestOctreeBounds:
    """Tests for OctreeBounds."""

    def test_center(self):
        bounds = OctreeBounds(
            min=(0.0, 0.0, 0.0),
            max=(10.0, 10.0, 10.0),
        )
        center = bounds.center()
        assert center == (5.0, 5.0, 5.0)

    def test_size(self):
        bounds = OctreeBounds(
            min=(0.0, 0.0, 0.0),
            max=(10.0, 20.0, 30.0),
        )
        size = bounds.size()
        assert size == (10.0, 20.0, 30.0)

    def test_radius(self):
        bounds = OctreeBounds(
            min=(0.0, 0.0, 0.0),
            max=(10.0, 10.0, 10.0),
        )
        radius = bounds.radius()
        # Diagonal / 2 = sqrt(300) / 2 ≈ 8.66
        assert radius == pytest.approx(8.66, rel=0.01)


class TestOctreeNode:
    """Tests for OctreeNode."""

    def test_create_node(self):
        key = OctreeKey.root()
        bounds = OctreeBounds((0, 0, 0), (100, 100, 100))
        node = OctreeNode(
            key=key,
            bounds=bounds,
            point_count=1000,
            spacing=1.0,
        )
        assert node.point_count == 1000
        assert node.spacing == 1.0
        assert len(node.children) == 0


class TestPointCloudRenderer:
    """Tests for PointCloudRenderer."""

    def test_default_params(self):
        renderer = PointCloudRenderer()
        assert renderer.point_budget == 5_000_000
        assert renderer.max_depth == 20

    def test_set_point_budget(self):
        renderer = PointCloudRenderer()
        renderer.set_point_budget(1_000_000)
        assert renderer.point_budget == 1_000_000

    def test_set_viewport(self):
        renderer = PointCloudRenderer()
        renderer.set_viewport(720.0, 1.0)
        assert renderer.viewport_height == 720.0
        assert renderer.fov_y == 1.0


class TestMockDataset:
    """Tests with mock dataset."""

    class MockDataset:
        """Mock point cloud dataset for testing."""
        
        def __init__(self, levels: int = 3, points_per_node: int = 10000):
            self.levels = levels
            self.points_per_node = points_per_node
            self._nodes = {}
            self._build_tree(OctreeKey.root(), 0)
        
        def _build_tree(self, key: OctreeKey, depth: int):
            self._nodes[str(key)] = self.points_per_node
            if depth < self.levels:
                for octant in range(8):
                    self._build_tree(key.child(octant), depth + 1)
        
        @property
        def total_points(self) -> int:
            return sum(self._nodes.values())
        
        @property
        def node_count(self) -> int:
            return len(self._nodes)
        
        @property
        def bounds(self) -> OctreeBounds:
            return OctreeBounds((-100, -100, -100), (100, 100, 100))
        
        def root_node(self) -> OctreeNode:
            return OctreeNode(
                key=OctreeKey.root(),
                bounds=self.bounds,
                point_count=self._nodes.get("0-0-0-0", 0),
                spacing=200.0,
            )
        
        def children(self, key: OctreeKey) -> list:
            children = []
            for octant in range(8):
                child_key = key.child(octant)
                if str(child_key) in self._nodes:
                    size = 200.0 / (2 ** child_key.depth)
                    children.append(OctreeNode(
                        key=child_key,
                        bounds=OctreeBounds(
                            (-size/2, -size/2, -size/2),
                            (size/2, size/2, size/2),
                        ),
                        point_count=self._nodes[str(child_key)],
                        spacing=size / 128,
                    ))
            return children

    def test_visible_nodes_respect_budget(self):
        """Test that point budget is respected."""
        dataset = self.MockDataset(levels=3, points_per_node=100000)
        renderer = PointCloudRenderer(point_budget=500000)
        
        camera = (0.0, 0.0, 500.0)
        visible = renderer.get_visible_nodes(dataset, camera)
        
        total_points = sum(n.point_count for n in visible)
        
        # Allow 5% tolerance
        assert total_points <= renderer.point_budget * 1.05

    def test_closer_camera_more_nodes(self):
        """Test that closer camera sees more nodes."""
        dataset = self.MockDataset(levels=3, points_per_node=10000)
        renderer = PointCloudRenderer(point_budget=10_000_000)
        
        near_camera = (0.0, 0.0, 50.0)
        far_camera = (0.0, 0.0, 1000.0)
        
        visible_near = renderer.get_visible_nodes(dataset, near_camera)
        visible_far = renderer.get_visible_nodes(dataset, far_camera)
        
        # Closer camera should generally see more (refined) nodes
        # or at least same number
        assert len(visible_near) >= len(visible_far)

    def test_budget_scaling(self):
        """Test that point count scales with budget."""
        dataset = self.MockDataset(levels=3, points_per_node=50000)
        camera = (0.0, 0.0, 200.0)
        
        budgets = [100000, 500000, 1000000]
        point_counts = []
        
        for budget in budgets:
            renderer = PointCloudRenderer(point_budget=budget)
            visible = renderer.get_visible_nodes(dataset, camera)
            total = sum(n.point_count for n in visible)
            point_counts.append(total)
        
        # Higher budget should result in more points
        for i in range(len(point_counts) - 1):
            assert point_counts[i] <= point_counts[i + 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
