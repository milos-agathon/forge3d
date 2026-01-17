"""P5: Point Cloud support for forge3d.

This module provides Python bindings for loading and rendering point cloud datasets.
Supports COPC (Cloud Optimized Point Cloud) and EPT (Entwine Point Tile) formats.

Usage:
    from forge3d.pointcloud import open_copc, open_ept, PointCloudRenderer

    # COPC
    dataset = open_copc("path/to/pointcloud.copc.laz")
    print(f"Points: {dataset.total_points}, Nodes: {dataset.node_count}")

    # EPT
    dataset = open_ept("path/to/ept.json")

    # Rendering
    renderer = PointCloudRenderer(point_budget=5_000_000)
    visible = renderer.get_visible_nodes(dataset, camera_position)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


@dataclass
class OctreeKey:
    """Octree node key (Morton code style: D-X-Y-Z)."""
    depth: int
    x: int
    y: int
    z: int

    @staticmethod
    def root() -> "OctreeKey":
        return OctreeKey(0, 0, 0, 0)

    def child(self, octant: int) -> "OctreeKey":
        return OctreeKey(
            depth=self.depth + 1,
            x=(self.x << 1) | (octant & 1),
            y=(self.y << 1) | ((octant >> 1) & 1),
            z=(self.z << 1) | ((octant >> 2) & 1),
        )

    def __str__(self) -> str:
        return f"{self.depth}-{self.x}-{self.y}-{self.z}"

    @staticmethod
    def from_str(s: str) -> Optional["OctreeKey"]:
        parts = s.split("-")
        if len(parts) != 4:
            return None
        try:
            return OctreeKey(int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
        except ValueError:
            return None


@dataclass
class OctreeBounds:
    """Axis-aligned bounding box for octree nodes."""
    min: Tuple[float, float, float]
    max: Tuple[float, float, float]

    def center(self) -> Tuple[float, float, float]:
        return (
            (self.min[0] + self.max[0]) / 2,
            (self.min[1] + self.max[1]) / 2,
            (self.min[2] + self.max[2]) / 2,
        )

    def size(self) -> Tuple[float, float, float]:
        return (
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        )

    def radius(self) -> float:
        s = self.size()
        return np.sqrt(s[0]**2 + s[1]**2 + s[2]**2) / 2


@dataclass
class OctreeNode:
    """Octree node with metadata."""
    key: OctreeKey
    bounds: OctreeBounds
    point_count: int
    spacing: float
    children: List[OctreeKey] = field(default_factory=list)


@dataclass
class PointData:
    """Decoded point data."""
    positions: np.ndarray  # (N, 3) float32
    colors: Optional[np.ndarray] = None  # (N, 3) uint8
    intensities: Optional[np.ndarray] = None  # (N,) uint16

    @property
    def point_count(self) -> int:
        return len(self.positions)


class LazDataset:
    """Simple LAZ/LAS dataset handle (non-hierarchical)."""

    def __init__(self, path: Path):
        self.path = path
        self._header: Dict[str, Any] = {}
        self._root_bounds: Optional[OctreeBounds] = None
        self._load()

    def _load(self):
        """Load LAS header."""
        with open(self.path, "rb") as f:
            magic = f.read(4)
            if magic != b"LASF":
                raise ValueError("Not a valid LAS file")

            # Version: bytes 24-25
            f.seek(24)
            version_major = int.from_bytes(f.read(1), "little")
            version_minor = int.from_bytes(f.read(1), "little")
            self._version = f"{version_major}.{version_minor}"

            # Header size at offset 94
            f.seek(94)
            header_size = int.from_bytes(f.read(2), "little")

            # Point data offset at offset 96
            f.seek(96)
            point_data_offset = int.from_bytes(f.read(4), "little")

            # Point format at offset 104
            f.seek(104)
            point_format = int.from_bytes(f.read(1), "little")
            point_record_length = int.from_bytes(f.read(2), "little")

            # Legacy point count at offset 107 (LAS 1.0-1.3)
            f.seek(107)
            legacy_point_count = int.from_bytes(f.read(4), "little")

            # For LAS 1.4, use 8-byte count at offset 247
            if version_major >= 1 and version_minor >= 4 and header_size >= 375:
                f.seek(247)
                point_count = int.from_bytes(f.read(8), "little")
            else:
                point_count = legacy_point_count

            # Scale factors at offset 131
            f.seek(131)
            scale = [
                np.frombuffer(f.read(8), dtype=np.float64)[0],
                np.frombuffer(f.read(8), dtype=np.float64)[0],
                np.frombuffer(f.read(8), dtype=np.float64)[0],
            ]

            # Offsets at offset 155
            offset = [
                np.frombuffer(f.read(8), dtype=np.float64)[0],
                np.frombuffer(f.read(8), dtype=np.float64)[0],
                np.frombuffer(f.read(8), dtype=np.float64)[0],
            ]

            # Bounds: max_x, min_x, max_y, min_y, max_z, min_z at offset 179
            f.seek(179)
            max_x = np.frombuffer(f.read(8), dtype=np.float64)[0]
            min_x = np.frombuffer(f.read(8), dtype=np.float64)[0]
            max_y = np.frombuffer(f.read(8), dtype=np.float64)[0]
            min_y = np.frombuffer(f.read(8), dtype=np.float64)[0]
            max_z = np.frombuffer(f.read(8), dtype=np.float64)[0]
            min_z = np.frombuffer(f.read(8), dtype=np.float64)[0]

            self._header = {
                "version": self._version,
                "point_count": point_count,
                "point_format": point_format,
                "point_record_length": point_record_length,
                "point_data_offset": point_data_offset,
                "scale": scale,
                "offset": offset,
                "min_bounds": [min_x, min_y, min_z],
                "max_bounds": [max_x, max_y, max_z],
            }

            self._root_bounds = OctreeBounds(
                min=(min_x, min_y, min_z),
                max=(max_x, max_y, max_z),
            )

    @property
    def version(self) -> str:
        return self._version

    @property
    def total_points(self) -> int:
        return self._header.get("point_count", 0)

    @property
    def node_count(self) -> int:
        return 1  # Single node for non-hierarchical

    @property
    def bounds(self) -> Optional[OctreeBounds]:
        return self._root_bounds

    @property
    def is_copc(self) -> bool:
        return False

    def root_node(self) -> OctreeNode:
        bounds = self._root_bounds or OctreeBounds((0, 0, 0), (1, 1, 1))
        return OctreeNode(
            key=OctreeKey.root(),
            bounds=bounds,
            point_count=self.total_points,
            spacing=bounds.size()[0] / 128,
        )

    def children(self, key: OctreeKey) -> List[OctreeNode]:
        return []  # No hierarchy


class CopcDataset:
    """COPC dataset handle."""

    def __init__(self, path: Path):
        self.path = path
        self._header: Dict[str, Any] = {}
        self._info: Dict[str, Any] = {}
        self._hierarchy: Dict[str, Dict] = {}
        self._root_bounds: Optional[OctreeBounds] = None
        self._is_copc = False
        self._load()

    def _load(self):
        """Load COPC header and hierarchy."""
        with open(self.path, "rb") as f:
            magic = f.read(4)
            if magic != b"LASF":
                raise ValueError("Not a valid LAS/COPC file")

            # Version
            f.seek(24)
            version_major = int.from_bytes(f.read(1), "little")
            version_minor = int.from_bytes(f.read(1), "little")

            # Header size
            f.seek(94)
            header_size = int.from_bytes(f.read(2), "little")

            f.seek(104)
            point_format = int.from_bytes(f.read(1), "little")
            point_record_length = int.from_bytes(f.read(2), "little")

            # Legacy point count (LAS 1.0-1.3)
            f.seek(107)
            legacy_point_count = int.from_bytes(f.read(4), "little")

            # For LAS 1.4+, use 8-byte count
            if version_major >= 1 and version_minor >= 4 and header_size >= 375:
                f.seek(247)
                point_count = int.from_bytes(f.read(8), "little")
            else:
                point_count = legacy_point_count

            f.seek(131)
            scale = [
                np.frombuffer(f.read(8), dtype=np.float64)[0],
                np.frombuffer(f.read(8), dtype=np.float64)[0],
                np.frombuffer(f.read(8), dtype=np.float64)[0],
            ]

            offset = [
                np.frombuffer(f.read(8), dtype=np.float64)[0],
                np.frombuffer(f.read(8), dtype=np.float64)[0],
                np.frombuffer(f.read(8), dtype=np.float64)[0],
            ]

            f.seek(179)
            max_x = np.frombuffer(f.read(8), dtype=np.float64)[0]
            min_x = np.frombuffer(f.read(8), dtype=np.float64)[0]
            max_y = np.frombuffer(f.read(8), dtype=np.float64)[0]
            min_y = np.frombuffer(f.read(8), dtype=np.float64)[0]
            max_z = np.frombuffer(f.read(8), dtype=np.float64)[0]
            min_z = np.frombuffer(f.read(8), dtype=np.float64)[0]

            self._header = {
                "point_count": point_count,
                "point_format": point_format,
                "point_record_length": point_record_length,
                "scale": scale,
                "offset": offset,
                "min_bounds": [min_x, min_y, min_z],
                "max_bounds": [max_x, max_y, max_z],
            }

            # Read COPC VLR
            f.seek(375)
            vlr_header = f.read(54)
            user_id = vlr_header[2:18].decode("utf-8").rstrip("\x00")

            if user_id == "copc":
                content_size = int.from_bytes(vlr_header[20:22], "little")
                content = f.read(content_size)

                center = [
                    np.frombuffer(content[0:8], dtype=np.float64)[0],
                    np.frombuffer(content[8:16], dtype=np.float64)[0],
                    np.frombuffer(content[16:24], dtype=np.float64)[0],
                ]
                halfsize = np.frombuffer(content[24:32], dtype=np.float64)[0]

                self._info = {
                    "center": center,
                    "halfsize": halfsize,
                    "spacing": np.frombuffer(content[32:40], dtype=np.float64)[0],
                    "root_hier_offset": int.from_bytes(content[40:48], "little"),
                    "root_hier_size": int.from_bytes(content[48:56], "little"),
                }

                self._root_bounds = OctreeBounds(
                    min=(center[0] - halfsize, center[1] - halfsize, center[2] - halfsize),
                    max=(center[0] + halfsize, center[1] + halfsize, center[2] + halfsize),
                )

                # Load root hierarchy
                f.seek(self._info["root_hier_offset"])
                hier_data = f.read(self._info["root_hier_size"])
                self._parse_hierarchy(hier_data)

    def _parse_hierarchy(self, data: bytes):
        """Parse hierarchy page."""
        entry_size = 32
        count = len(data) // entry_size

        for i in range(count):
            off = i * entry_size
            d = int.from_bytes(data[off:off+4], "little", signed=True)
            x = int.from_bytes(data[off+4:off+8], "little", signed=True)
            y = int.from_bytes(data[off+8:off+12], "little", signed=True)
            z = int.from_bytes(data[off+12:off+16], "little", signed=True)
            file_offset = int.from_bytes(data[off+16:off+24], "little")
            byte_size = int.from_bytes(data[off+24:off+28], "little", signed=True)
            point_count = int.from_bytes(data[off+28:off+32], "little", signed=True)

            if d >= 0 and point_count > 0:
                key = f"{d}-{x}-{y}-{z}"
                self._hierarchy[key] = {
                    "offset": file_offset,
                    "byte_size": byte_size,
                    "point_count": point_count,
                }

    @property
    def total_points(self) -> int:
        return self._header.get("point_count", 0)

    @property
    def node_count(self) -> int:
        return len(self._hierarchy)

    @property
    def bounds(self) -> Optional[OctreeBounds]:
        return self._root_bounds

    def root_node(self) -> OctreeNode:
        """Get root node."""
        key = OctreeKey.root()
        entry = self._hierarchy.get(str(key), {})
        point_count = entry.get("point_count", 0)
        bounds = self._root_bounds or OctreeBounds((0, 0, 0), (1, 1, 1))
        spacing = self._info.get("spacing", 1.0)
        return OctreeNode(key=key, bounds=bounds, point_count=point_count, spacing=spacing)

    def children(self, key: OctreeKey) -> List[OctreeNode]:
        """Get child nodes."""
        children = []
        for octant in range(8):
            child_key = key.child(octant)
            if str(child_key) in self._hierarchy:
                entry = self._hierarchy[str(child_key)]
                bounds = self._bounds_for_key(child_key)
                spacing = self._info.get("spacing", 1.0) / (2 ** child_key.depth)
                children.append(OctreeNode(
                    key=child_key,
                    bounds=bounds,
                    point_count=entry["point_count"],
                    spacing=spacing,
                ))
        return children

    def _bounds_for_key(self, key: OctreeKey) -> OctreeBounds:
        """Compute bounds for a key."""
        if not self._root_bounds:
            return OctreeBounds((0, 0, 0), (1, 1, 1))

        min_pt = list(self._root_bounds.min)
        max_pt = list(self._root_bounds.max)

        for d in range(key.depth):
            shift = key.depth - d - 1
            mid = [(min_pt[i] + max_pt[i]) / 2 for i in range(3)]

            if (key.x >> shift) & 1:
                min_pt[0] = mid[0]
            else:
                max_pt[0] = mid[0]

            if (key.y >> shift) & 1:
                min_pt[1] = mid[1]
            else:
                max_pt[1] = mid[1]

            if (key.z >> shift) & 1:
                min_pt[2] = mid[2]
            else:
                max_pt[2] = mid[2]

        return OctreeBounds(tuple(min_pt), tuple(max_pt))


class EptDataset:
    """EPT dataset handle."""

    def __init__(self, path: Path):
        self.path = path
        self.base_path = path.parent
        self._info: Dict[str, Any] = {}
        self._hierarchy: Dict[str, int] = {}
        self._root_bounds: Optional[OctreeBounds] = None
        self._load()

    def _load(self):
        """Load EPT metadata and hierarchy."""
        with open(self.path) as f:
            self._info = json.load(f)

        bounds = self._info.get("bounds", [0, 0, 0, 1, 1, 1])
        self._root_bounds = OctreeBounds(
            min=(bounds[0], bounds[1], bounds[2]),
            max=(bounds[3], bounds[4], bounds[5]),
        )

        # Load hierarchy
        self._load_hierarchy(OctreeKey.root())

    def _load_hierarchy(self, key: OctreeKey):
        """Load hierarchy JSON for a node."""
        hier_path = self.base_path / "ept-hierarchy" / f"{key}.json"
        if not hier_path.exists():
            return

        with open(hier_path) as f:
            hier = json.load(f)

        for key_str, count in hier.items():
            if count > 0:
                self._hierarchy[key_str] = count
            elif count == -1:
                node_key = OctreeKey.from_str(key_str)
                if node_key:
                    self._load_hierarchy(node_key)

    @property
    def total_points(self) -> int:
        return self._info.get("points", 0)

    @property
    def node_count(self) -> int:
        return len(self._hierarchy)

    @property
    def bounds(self) -> Optional[OctreeBounds]:
        return self._root_bounds

    def root_node(self) -> OctreeNode:
        """Get root node."""
        key = OctreeKey.root()
        point_count = self._hierarchy.get(str(key), 0)
        bounds = self._root_bounds or OctreeBounds((0, 0, 0), (1, 1, 1))
        span = self._info.get("span", 128)
        spacing = bounds.size()[0] / span
        return OctreeNode(key=key, bounds=bounds, point_count=point_count, spacing=spacing)

    def children(self, key: OctreeKey) -> List[OctreeNode]:
        """Get child nodes."""
        children = []
        for octant in range(8):
            child_key = key.child(octant)
            if str(child_key) in self._hierarchy:
                point_count = self._hierarchy[str(child_key)]
                bounds = self._bounds_for_key(child_key)
                span = self._info.get("span", 128)
                spacing = bounds.size()[0] / span
                children.append(OctreeNode(
                    key=child_key,
                    bounds=bounds,
                    point_count=point_count,
                    spacing=spacing,
                ))
        return children

    def _bounds_for_key(self, key: OctreeKey) -> OctreeBounds:
        """Compute bounds for a key."""
        if not self._root_bounds:
            return OctreeBounds((0, 0, 0), (1, 1, 1))

        min_pt = list(self._root_bounds.min)
        max_pt = list(self._root_bounds.max)

        for d in range(key.depth):
            shift = key.depth - d - 1
            mid = [(min_pt[i] + max_pt[i]) / 2 for i in range(3)]

            if (key.x >> shift) & 1:
                min_pt[0] = mid[0]
            else:
                max_pt[0] = mid[0]

            if (key.y >> shift) & 1:
                min_pt[1] = mid[1]
            else:
                max_pt[1] = mid[1]

            if (key.z >> shift) & 1:
                min_pt[2] = mid[2]
            else:
                max_pt[2] = mid[2]

        return OctreeBounds(tuple(min_pt), tuple(max_pt))


def open_laz(path: str | Path) -> LazDataset:
    """Open a LAZ/LAS file (non-hierarchical)."""
    return LazDataset(Path(path))


def open_copc(path: str | Path) -> CopcDataset | LazDataset:
    """Open a COPC dataset. Falls back to LazDataset if not COPC."""
    path = Path(path)
    # Try COPC first
    try:
        dataset = CopcDataset(path)
        if dataset._is_copc:
            return dataset
    except Exception:
        pass
    # Fall back to regular LAZ
    return LazDataset(path)


def open_ept(path: str | Path) -> EptDataset:
    """Open an EPT dataset."""
    return EptDataset(Path(path))


def open_pointcloud(path: str | Path) -> LazDataset | CopcDataset | EptDataset:
    """Auto-detect and open a point cloud file.
    
    Supports: .laz, .las, .copc.laz, ept.json
    """
    path = Path(path)
    name = path.name.lower()
    
    if name == "ept.json":
        return open_ept(path)
    elif name.endswith(".copc.laz"):
        return open_copc(path)
    elif name.endswith(".laz") or name.endswith(".las"):
        return open_laz(path)
    else:
        # Try to auto-detect
        return open_laz(path)


@dataclass
class VisibleNode:
    """Result of traversal for a visible node."""
    key: OctreeKey
    bounds: OctreeBounds
    point_count: int
    priority: float


class PointCloudRenderer:
    """Point cloud renderer with LOD traversal."""

    def __init__(
        self,
        point_budget: int = 5_000_000,
        viewport_height: float = 1080.0,
        fov_y: float = 0.785398,
        max_depth: int = 20,
    ):
        self.point_budget = point_budget
        self.viewport_height = viewport_height
        self.fov_y = fov_y
        self.max_depth = max_depth
        self._cache: Dict[str, PointData] = {}

    def set_point_budget(self, budget: int):
        self.point_budget = budget

    def set_viewport(self, height: float, fov_y: float):
        self.viewport_height = height
        self.fov_y = fov_y

    def get_visible_nodes(
        self,
        dataset: CopcDataset | EptDataset,
        camera_position: Tuple[float, float, float],
    ) -> List[VisibleNode]:
        """Get visible nodes within point budget."""
        root = dataset.root_node()
        result = []
        total_points = 0

        candidates = [(root, self._compute_priority(root.bounds, camera_position))]

        while candidates:
            candidates.sort(key=lambda x: -x[1])  # Sort by priority descending
            node, priority = candidates.pop(0)

            if node.key.depth > self.max_depth:
                continue

            if total_points + node.point_count > self.point_budget:
                if priority < 1.0:
                    continue

            screen_size = self._compute_screen_size(node.bounds, camera_position)
            children = dataset.children(node.key)

            if screen_size > 1.0 and children and node.key.depth < self.max_depth:
                for child in children:
                    child_priority = self._compute_priority(child.bounds, camera_position)
                    candidates.append((child, child_priority))
            else:
                total_points += node.point_count
                result.append(VisibleNode(
                    key=node.key,
                    bounds=node.bounds,
                    point_count=node.point_count,
                    priority=priority,
                ))

                if total_points >= self.point_budget:
                    break

        return result

    def _compute_priority(
        self,
        bounds: OctreeBounds,
        camera_pos: Tuple[float, float, float],
    ) -> float:
        center = bounds.center()
        dx = center[0] - camera_pos[0]
        dy = center[1] - camera_pos[1]
        dz = center[2] - camera_pos[2]
        distance = np.sqrt(dx*dx + dy*dy + dz*dz)
        radius = bounds.radius()

        if distance < radius:
            return float("inf")

        return radius / distance

    def _compute_screen_size(
        self,
        bounds: OctreeBounds,
        camera_pos: Tuple[float, float, float],
    ) -> float:
        center = bounds.center()
        dx = center[0] - camera_pos[0]
        dy = center[1] - camera_pos[1]
        dz = center[2] - camera_pos[2]
        distance = max(np.sqrt(dx*dx + dy*dy + dz*dz), 0.001)
        radius = bounds.radius()

        sse_factor = self.viewport_height / (2.0 * np.tan(self.fov_y / 2.0))
        return (radius / distance) * sse_factor
