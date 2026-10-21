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
from typing import Protocol as _Protocol
import numpy as np

from . import _pointcloud_octree as _octree


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


class _PointCloudHierarchy(_Protocol):
    """Spatial facts required by LOD traversal, without decoding capabilities."""

    def root_node(self) -> OctreeNode:
        ...

    def children(self, key: OctreeKey) -> List[OctreeNode]:
        ...


@dataclass
class PointData:
    """Decoded point data."""
    positions: np.ndarray  # (N, 3) float64 world coordinates
    colors: Optional[np.ndarray] = None  # (N, 3) uint8
    intensities: Optional[np.ndarray] = None  # (N,) uint16
    classifications: Optional[np.ndarray] = None  # (N,) uint8

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
        del key
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
                self._is_copc = True
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

                self._load_hierarchy_pages(f)

    def _load_hierarchy_pages(self, stream):
        """Follow hierarchy page references and reject cycles or truncated pages."""
        pending = [(self._info["root_hier_offset"], self._info["root_hier_size"])]
        seen = set()
        stream.seek(0, 2)
        file_size = stream.tell()
        while pending:
            offset, size = pending.pop()
            if (offset, size) in seen:
                raise ValueError("Duplicate or cyclic COPC hierarchy page")
            seen.add((offset, size))
            if size <= 0 or size % 32 or offset < 0 or offset + size > file_size:
                raise ValueError("Invalid COPC hierarchy page extent")
            stream.seek(offset)
            pending.extend(self._parse_hierarchy(stream.read(size)))

    def _parse_hierarchy(self, data: bytes):
        """Parse one complete hierarchy page and return child page references."""
        import struct
        if len(data) % 32:
            raise ValueError("Truncated COPC hierarchy page")
        pages = []
        for d, x, y, z, offset, size, count in struct.iter_unpack("<iiiiQii", data):
            if d < 0 or count < -1 or size < 0:
                raise ValueError("Invalid COPC hierarchy entry")
            if count == -1:
                pages.append((offset, size))
            else:
                key = f"{d}-{x}-{y}-{z}"
                if key in self._hierarchy:
                    raise ValueError("Duplicate COPC hierarchy node")
                self._hierarchy[key] = {"offset": offset, "byte_size": size, "point_count": count}
        return pages

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
        return _octree.child_nodes(key, self._hierarchy, self._node_for_key)

    def _node_for_key(self, key: OctreeKey) -> OctreeNode:
        entry = self._hierarchy[str(key)]
        return OctreeNode(
            key=key,
            bounds=self._bounds_for_key(key),
            point_count=entry["point_count"],
            spacing=self._info.get("spacing", 1.0) / (2 ** key.depth),
        )

    def _bounds_for_key(self, key: OctreeKey) -> OctreeBounds:
        """Compute bounds for a key."""
        return _octree.bounds_for_key(self._root_bounds, key)

    def read_points(self, key: OctreeKey | None = None, budget: int | None = None) -> PointData:
        """Read decoded points for a COPC node through the native decoder."""
        from ._native import get_native_module

        native = get_native_module()
        if native is None:
            raise RuntimeError("CopcDataset.read_points requires the compiled native extension")
        fn = getattr(native, "copc_read_node_points", None)
        if fn is None:
            raise RuntimeError("native copc_read_node_points is unavailable")

        node_key = key or OctreeKey.root()
        result = fn(
            str(self.path),
            int(node_key.depth),
            int(node_key.x),
            int(node_key.y),
            int(node_key.z),
            None if budget is None else int(budget),
        )
        positions = np.asarray(result["positions"], dtype=np.float64).reshape((-1, 3))
        colors = result.get("colors")
        intensities = result.get("intensities")
        classifications = result.get("classifications")
        return PointData(
            positions=positions,
            colors=None if colors is None else np.asarray(colors, dtype=np.uint8).reshape((-1, 3)),
            intensities=None if intensities is None else np.asarray(intensities, dtype=np.uint16),
            classifications=None if classifications is None else np.asarray(classifications, dtype=np.uint8),
        )


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
        return _octree.child_nodes(key, self._hierarchy, self._node_for_key)

    def _node_for_key(self, key: OctreeKey) -> OctreeNode:
        bounds = self._bounds_for_key(key)
        return OctreeNode(
            key=key,
            bounds=bounds,
            point_count=self._hierarchy[str(key)],
            spacing=bounds.size()[0] / self._info.get("span", 128),
        )

    def _bounds_for_key(self, key: OctreeKey) -> OctreeBounds:
        """Compute bounds for a key."""
        return _octree.bounds_for_key(self._root_bounds, key)


def open_laz(path: str | Path) -> LazDataset:
    """Open a LAZ/LAS file (non-hierarchical)."""
    return LazDataset(Path(path))


def read_laz_point_attributes(path: str | Path, sample_count: int = 3) -> Dict[str, Any]:
    """Read native LAS/LAZ sample attributes including intensity and classification."""
    from ._native import get_native_module

    native = get_native_module()
    if native is None:
        raise RuntimeError("read_laz_point_attributes requires the compiled native extension")
    fn = getattr(native, "read_laz_point_attributes", None)
    if fn is None:
        raise RuntimeError("native read_laz_point_attributes is unavailable")

    result = dict(fn(str(path), int(sample_count)))
    result["coords"] = np.asarray(result["coords"], dtype=np.float64).reshape((-1, 3))
    result["intensities"] = np.asarray(result["intensities"], dtype=np.uint16)
    result["classifications"] = np.asarray(result["classifications"], dtype=np.uint8)
    return result


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
        dataset: _PointCloudHierarchy,
        camera_position: Tuple[float, float, float],
    ) -> List[VisibleNode]:
        """Get visible nodes using only ``root_node()`` and ``children(key)``.

        Any dataset supplying these spatial facts can participate in traversal;
        point decoding remains a separate, format-specific capability.
        """
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
                # Whole nodes are indivisible here; omit any node that does not fit.
                continue

            screen_size = self._compute_screen_size(node.bounds, camera_position)
            children = dataset.children(node.key)

            refining = screen_size > 1.0 and children and node.key.depth < self.max_depth
            if refining:
                for child in children:
                    child_priority = self._compute_priority(child.bounds, camera_position)
                    candidates.append((child, child_priority))
            if not refining or isinstance(dataset, EptDataset):
                if total_points + node.point_count > self.point_budget:
                    continue
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
