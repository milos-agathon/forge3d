"""P5.5: Point Cloud LOD traversal tests.

Tests octree traversal and point budget enforcement.
"""

import json
import pickle
import struct

import pytest

from forge3d.pointcloud import (
    OctreeKey,
    OctreeBounds,
    OctreeNode,
    PointCloudRenderer,
    CopcDataset,
    EptDataset,
    LazDataset,
    open_pointcloud,
)


ROOT_BOUNDS = OctreeBounds((-8.0, 12.0, -16.0), (8.0, 28.0, 0.0))
HIERARCHY_COUNTS = {
    "0-0-0-0": 20,
    **{str(OctreeKey.root().child(octant)): 10 for octant in range(8)},
    "2-2-0-2": 20,
    "2-3-1-3": 30,
}


def _write_copc_hierarchy(path, counts, spacing=3.25):
    """Write COPC-shaped metadata accepted by CopcDataset, without point payloads."""
    header = bytearray(375)
    header[:4] = b"LASF"
    header[24:26] = bytes((1, 4))
    struct.pack_into("<H", header, 94, len(header))
    struct.pack_into("<I", header, 100, 1)
    struct.pack_into("<BH", header, 104, 3, 34)
    struct.pack_into("<Q", header, 247, sum(c for c in counts.values() if c > 0))
    struct.pack_into("<3d", header, 131, 1.0, 1.0, 1.0)
    struct.pack_into("<6d", header, 179, 8.0, -8.0, 28.0, 12.0, 0.0, -16.0)

    hierarchy = b"".join(
        struct.pack("<4iQ2i", *(int(part) for part in key.split("-")), 0, 0, count)
        for key, count in reversed(list(counts.items()))
    )
    info = bytearray(72)
    struct.pack_into("<5dQQ", info, 0, 0.0, 20.0, -8.0, 8.0, spacing,
                     len(header) + 54 + len(info), len(hierarchy))
    vlr = bytearray(54)
    vlr[2:6] = b"copc"
    struct.pack_into("<HH", vlr, 18, 1, len(info))
    path.write_bytes(header + vlr + info + hierarchy)


def _write_ept_hierarchy(directory, counts, bounds=ROOT_BOUNDS, span=16, paged=False):
    directory.mkdir()
    hierarchy_dir = directory / "ept-hierarchy"
    hierarchy_dir.mkdir()
    path = directory / "ept.json"
    path.write_text(json.dumps({
        "bounds": [*bounds.min, *bounds.max],
        "span": span,
        "points": sum(c for c in counts.values() if c > 0),
        "dataType": "binary",
    }), encoding="utf-8")
    entries = dict(reversed(list(counts.items())))
    if paged:
        page_keys = ("1-1-0-1", "2-2-0-2", "2-3-1-3")
        page = {key: entries.pop(key) for key in page_keys}
        entries["1-1-0-1"] = -1
        (hierarchy_dir / "1-1-0-1.json").write_text(json.dumps(page), encoding="utf-8")
    (hierarchy_dir / "0-0-0-0.json").write_text(json.dumps(entries), encoding="utf-8")
    return path


@pytest.fixture
def equivalent_datasets(tmp_path):
    copc_path = tmp_path / "cloud.copc.laz"
    _write_copc_hierarchy(copc_path, HIERARCHY_COUNTS)
    ept_path = _write_ept_hierarchy(tmp_path / "ept", HIERARCHY_COUNTS, paged=True)
    return open_pointcloud(copc_path), open_pointcloud(ept_path)


class TestFormatHierarchies:
    def test_equivalent_bounds_and_octant_order(self, equivalent_datasets):
        copc, ept = equivalent_datasets
        assert isinstance(copc, CopcDataset)
        assert isinstance(ept, EptDataset)
        assert copc.bounds == ept.bounds == ROOT_BOUNDS
        assert copc.total_points == ept.total_points == 150
        assert copc.node_count == ept.node_count == len(HIERARCHY_COUNTS)
        assert copc.root_node().point_count == ept.root_node().point_count == 20

        for dataset in equivalent_datasets:
            children = dataset.children(OctreeKey.root())
            assert [node.key for node in children] == [
                OctreeKey.root().child(octant) for octant in range(8)
            ]
            for octant, node in enumerate(children):
                minimum = tuple(ROOT_BOUNDS.min[axis] + 8 * ((octant >> axis) & 1)
                                for axis in range(3))
                assert node.bounds == OctreeBounds(minimum, tuple(v + 8 for v in minimum))
                assert node.point_count == 10
            grandchildren = dataset.children(OctreeKey(1, 1, 0, 1))
            assert [str(node.key) for node in grandchildren] == ["2-2-0-2", "2-3-1-3"]
            assert [node.bounds for node in grandchildren] == [
                OctreeBounds((0.0, 12.0, -8.0), (4.0, 16.0, -4.0)),
                OctreeBounds((4.0, 16.0, -4.0), (8.0, 20.0, 0.0)),
            ]
            assert dataset.children(OctreeKey(2, 3, 1, 3)) == []
            assert dataset.children(OctreeKey(4, 15, 15, 15)) == []

    def test_format_specific_spacing(self, equivalent_datasets):
        copc, ept = equivalent_datasets
        assert copc.root_node().spacing == 3.25
        assert ept.root_node().spacing == 1.0
        for dataset, root_spacing in ((copc, 3.25), (ept, 1.0)):
            assert [node.spacing for node in dataset.children(OctreeKey.root())] == [
                root_spacing / 2
            ] * 8
            assert [node.spacing for node in dataset.children(OctreeKey(1, 1, 0, 1))] == [
                root_spacing / 4
            ] * 2

    def test_non_cubic_ept_bounds(self, tmp_path):
        bounds = OctreeBounds((-10.0, 20.0, 100.0), (30.0, 100.0, 260.0))
        path = _write_ept_hierarchy(tmp_path / "ept", HIERARCHY_COUNTS, bounds=bounds)
        dataset = EptDataset(path)
        assert dataset.root_node().spacing == 2.5
        node = dataset.children(OctreeKey(1, 1, 0, 1))[1]
        assert node.bounds == OctreeBounds((20.0, 40.0, 220.0), (30.0, 60.0, 260.0))
        assert node.spacing == 0.625

    def test_hierarchy_loading_rules(self, tmp_path):
        counts = {"0-0-0-0": 20, "1-0-0-0": 10, "1-1-0-0": 0}
        copc_path = tmp_path / "cloud.copc.laz"
        _write_copc_hierarchy(copc_path, counts)
        ept_path = _write_ept_hierarchy(tmp_path / "ept", counts)
        copc, ept = CopcDataset(copc_path), EptDataset(ept_path)
        # COPC retains empty hierarchy nodes so their descendants stay reachable.
        assert copc.node_count == 3
        assert [str(node.key) for node in copc.children(OctreeKey.root())] == [
            "1-0-0-0", "1-1-0-0"]
        assert ept.node_count == 2
        assert [str(node.key) for node in ept.children(OctreeKey.root())] == ["1-0-0-0"]
        # A -1 entry is a page reference, not an ignorable point-count sentinel.
        _write_copc_hierarchy(copc_path, {**counts, "1-0-1-0": -1})
        with pytest.raises(ValueError, match="Invalid COPC hierarchy page extent"):
            CopcDataset(copc_path)

    def test_missing_root_entry_and_empty_hierarchy(self, tmp_path):
        counts = {"1-1-0-0": 10}
        copc_path = tmp_path / "cloud.copc.laz"
        _write_copc_hierarchy(copc_path, counts)
        ept_path = _write_ept_hierarchy(tmp_path / "ept", counts)
        for dataset in (CopcDataset(copc_path), EptDataset(ept_path)):
            assert dataset.root_node().point_count == 0
            assert dataset.root_node().bounds == ROOT_BOUNDS
            assert len(dataset.children(OctreeKey.root())) == 1
        (ept_path.parent / "ept-hierarchy" / "0-0-0-0.json").unlink()
        empty = EptDataset(ept_path)
        assert empty.node_count == 0
        assert empty.root_node().point_count == 0
        assert empty.children(OctreeKey.root()) == []

    @pytest.mark.parametrize("budget,max_depth,viewport,expected,ept_expected", [
        (150, 0, 1080.0, ["0-0-0-0"], ["0-0-0-0"]),
        (30, 1, 1080.0, ["1-0-0-1", "1-1-0-1", "1-0-1-1"],
         ["0-0-0-0", "1-0-0-1"]),
        (150, 1, 1080.0, ["1-0-0-1", "1-1-0-1", "1-0-1-1", "1-1-1-1",
                          "1-0-0-0", "1-1-0-0", "1-0-1-0", "1-1-1-0"],
         ["0-0-0-0", "1-0-0-1", "1-1-0-1", "1-0-1-1", "1-1-1-1",
          "1-0-0-0", "1-1-0-0", "1-0-1-0", "1-1-1-0"]),
        (150, 2, 1080.0, ["1-0-0-1", "1-0-1-1", "1-1-1-1", "1-0-0-0",
                          "1-1-0-0", "1-0-1-0", "1-1-1-0", "2-3-1-3", "2-2-0-2"],
         ["0-0-0-0", "1-0-0-1", "1-1-0-1", "1-0-1-1", "1-1-1-1",
          "1-0-0-0", "1-1-0-0", "1-0-1-0", "1-1-1-0", "2-3-1-3", "2-2-0-2"]),
        (30, 2, 1080.0, ["1-0-0-1", "1-0-1-1", "1-1-1-1"],
         ["0-0-0-0", "1-0-0-1"]),
        (150, 2, 1.0, ["0-0-0-0"], ["0-0-0-0"]),
    ])
    def test_selected_nodes_and_budget_order(self, equivalent_datasets, budget, max_depth,
                                            viewport, expected, ept_expected):
        renderer = PointCloudRenderer(point_budget=budget, max_depth=max_depth,
                                      viewport_height=viewport)
        copc_nodes, ept_nodes = [
            renderer.get_visible_nodes(dataset, (0.0, 20.0, 10.0))
            for dataset in equivalent_datasets
        ]
        assert [str(node.key) for node in copc_nodes] == expected
        assert [str(node.key) for node in ept_nodes] == ept_expected
        common = {str(node.key): node for node in copc_nodes}
        for node in ept_nodes:
            if str(node.key) in common:
                assert node == common[str(node.key)]
        for nodes in (copc_nodes, ept_nodes):
            assert sum(node.point_count for node in nodes) <= budget

    def test_inside_node_obeys_strict_budget(self, equivalent_datasets):
        # Being inside a node must not bypass the point budget.
        renderer = PointCloudRenderer(point_budget=5, max_depth=2)
        for dataset in equivalent_datasets:
            nodes = renderer.get_visible_nodes(dataset, (0.0, 20.0, -8.0))
            assert nodes == []
            assert sum(node.point_count for node in nodes) <= 5

    def test_large_offset_midpoint_bounds(self, tmp_path):
        bounds = OctreeBounds((1e16, -1e16, 0.1), (1e16 + 16, -1e16 + 16, 0.9))
        path = _write_ept_hierarchy(tmp_path / "ept", HIERARCHY_COUNTS, bounds=bounds)
        dataset = EptDataset(path)
        node = dataset.children(OctreeKey(1, 1, 0, 1))[1]
        assert node.bounds == OctreeBounds(
            (1e16 + 12, -1e16 + 4, 0.7), (1e16 + 16, -1e16 + 8, 0.9)
        )

    def test_decoding_capabilities_remain_separate(self, equivalent_datasets):
        copc, ept = equivalent_datasets
        assert callable(copc.read_points)
        assert not hasattr(ept, "read_points")

    def test_plain_las_uses_single_node(self, tmp_path):
        path = tmp_path / "cloud.las"
        _write_copc_hierarchy(path, {"0-0-0-0": 20})
        dataset = open_pointcloud(path)
        assert isinstance(dataset, LazDataset)
        assert dataset.children(OctreeKey.root()) == []
        nodes = PointCloudRenderer().get_visible_nodes(dataset, (0.0, 20.0, 100.0))
        assert [str(node.key) for node in nodes] == ["0-0-0-0"]
        assert nodes[0].point_count == 20

    def test_public_spatial_types_keep_pickle_identity(self):
        node = OctreeNode(OctreeKey(2, 3, 1, 3), ROOT_BOUNDS, 30, 1.0)
        assert pickle.loads(pickle.dumps(node)) == node
        for cls in (OctreeKey, OctreeBounds, OctreeNode):
            assert cls.__module__ == "forge3d.pointcloud"


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

    def test_hierarchy_contract_is_format_independent(self):
        from typing import get_type_hints

        from forge3d.pointcloud import _PointCloudHierarchy

        hints = get_type_hints(PointCloudRenderer.get_visible_nodes)
        assert hints["dataset"] is _PointCloudHierarchy


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
