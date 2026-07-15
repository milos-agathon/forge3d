"""P5: 3D Tiles support for forge3d.

This module provides Python bindings for loading and rendering OGC 3D Tiles datasets.
Supports tileset.json, b3dm (batched 3D model), and pnts (point cloud) payloads.

Usage:
    from forge3d.tiles3d import load_tileset, Tiles3dRenderer

    tileset = load_tileset("path/to/tileset.json")
    print(f"Tiles: {tileset.tile_count}, Depth: {tileset.max_depth}")

    renderer = Tiles3dRenderer(sse_threshold=16.0)
    visible = renderer.get_visible_tiles(tileset, camera_position)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from ._native import NATIVE_AVAILABLE, get_native_module
from .diagnostics import (
    LayerSummary,
    ValidationReport,
    python_public_3dtiles_incomplete_diagnostic,
)


@dataclass
class BoundingVolume:
    """Bounding volume for a 3D Tile."""
    volume_type: str  # "box", "sphere", or "region"
    data: List[float]

    def center(self) -> Tuple[float, float, float]:
        if self.volume_type == "sphere":
            return (self.data[0], self.data[1], self.data[2])
        elif self.volume_type == "box":
            return (self.data[0], self.data[1], self.data[2])
        elif self.volume_type == "region":
            lon = (self.data[0] + self.data[2]) / 2
            lat = (self.data[1] + self.data[3]) / 2
            h = (self.data[4] + self.data[5]) / 2
            return (lon, lat, h)
        return (0.0, 0.0, 0.0)

    def radius(self) -> float:
        if self.volume_type == "sphere":
            return self.data[3]
        elif self.volume_type == "box":
            x = np.linalg.norm(self.data[3:6])
            y = np.linalg.norm(self.data[6:9])
            z = np.linalg.norm(self.data[9:12])
            return np.sqrt(x*x + y*y + z*z)
        return 1.0


def _transform_bounding_volume(volume: BoundingVolume, matrix: np.ndarray) -> BoundingVolume:
    """Transform a tile bound in f64 for the pure-Python traversal fallback."""
    if volume.volume_type == "box":
        center = matrix @ np.asarray([*volume.data[:3], 1.0], dtype=np.float64)
        axes = []
        for offset in (3, 6, 9):
            axis = matrix @ np.asarray([*volume.data[offset:offset + 3], 0.0], dtype=np.float64)
            axes.extend(axis[:3].tolist())
        return BoundingVolume("box", [*center[:3].tolist(), *axes])
    if volume.volume_type == "sphere":
        center = matrix @ np.asarray([*volume.data[:3], 1.0], dtype=np.float64)
        scale = max(np.linalg.norm(matrix[:3, axis]) for axis in range(3))
        return BoundingVolume("sphere", [*center[:3].tolist(), float(volume.data[3]) * float(scale)])
    return volume


@dataclass
class TileContent:
    """Content description for a tile."""
    uri: str
    bounding_volume: Optional[BoundingVolume] = None


@dataclass
class Tile:
    """A single tile in the 3D Tiles hierarchy."""
    bounding_volume: BoundingVolume
    geometric_error: float
    refine: str = "REPLACE"
    content: Optional[TileContent] = None
    children: List["Tile"] = field(default_factory=list)
    transform: Optional[List[float]] = None

    def has_content(self) -> bool:
        return self.content is not None

    def content_uri(self) -> Optional[str]:
        return self.content.uri if self.content else None

    def count_tiles(self) -> int:
        return 1 + sum(c.count_tiles() for c in self.children)

    def max_depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.max_depth() for c in self.children)


@dataclass
class Tileset:
    """A loaded 3D Tiles tileset."""
    base_path: Path
    version: str
    geometric_error: float
    root: Tile
    properties: Optional[Dict[str, Any]] = None

    @property
    def tile_count(self) -> int:
        return self.root.count_tiles()

    @property
    def max_depth(self) -> int:
        return self.root.max_depth()

    def resolve_uri(self, uri: str) -> Path:
        if uri.startswith("http://") or uri.startswith("https://"):
            return Path(uri)
        return self.base_path / uri


@dataclass
class Tiles3dDataset:
    """Native-backed 3D Tiles dataset wrapper."""

    path: Path
    tileset: Tileset

    @classmethod
    def from_tileset_json(cls, path: str | Path) -> "Tiles3dDataset":
        path = Path(path)
        return cls(path=path, tileset=load_tileset(path))

    def traverse(
        self,
        camera_position: Tuple[float, float, float],
        *,
        sse_threshold: float = 16.0,
        max_depth: int = 32,
    ) -> List[Dict[str, Any]]:
        native = get_native_module() if NATIVE_AVAILABLE else None
        native_traverse = getattr(native, "tiles3d_traverse_py", None) if native else None
        if callable(native_traverse):
            return [
                dict(tile)
                for tile in native_traverse(
                    str(self.path),
                    tuple(float(v) for v in camera_position),
                    float(sse_threshold),
                    int(max_depth),
                )
            ]

        renderer = Tiles3dRenderer(sse_threshold=sse_threshold, max_depth=max_depth)
        return [
            {
                "uri": tile.tile.content_uri(),
                "resolved_path": str(self.tileset.resolve_uri(tile.tile.content_uri() or "")),
                "sse": float(tile.sse),
                "depth": int(tile.depth),
                "world_transform": tile.world_transform.reshape(16, order="F").tolist(),
                "world_bounds_center": list(tile.world_bounds.center()),
                "world_bounds_radius": float(tile.world_bounds.radius()),
            }
            for tile in renderer.get_visible_tiles(self.tileset, camera_position)
        ]


def _parse_bounding_volume(data: Dict) -> BoundingVolume:
    """Parse a bounding volume from JSON."""
    if "sphere" in data:
        return BoundingVolume("sphere", data["sphere"])
    elif "box" in data:
        return BoundingVolume("box", data["box"])
    elif "region" in data:
        return BoundingVolume("region", data["region"])
    raise ValueError(f"Unknown bounding volume type: {data.keys()}")


def _parse_tile(data: Dict) -> Tile:
    """Parse a tile from JSON."""
    bv = _parse_bounding_volume(data["boundingVolume"])
    ge = data.get("geometricError", 0.0)
    refine = data.get("refine", "REPLACE")
    transform = data.get("transform")

    content = None
    if "content" in data:
        c = data["content"]
        content_bv = None
        if "boundingVolume" in c:
            content_bv = _parse_bounding_volume(c["boundingVolume"])
        content = TileContent(uri=c["uri"], bounding_volume=content_bv)

    children = [_parse_tile(c) for c in data.get("children", [])]

    return Tile(
        bounding_volume=bv,
        geometric_error=ge,
        refine=refine,
        content=content,
        children=children,
        transform=transform,
    )


def load_tileset(path: str | Path) -> Tileset:
    """Load a tileset from a tileset.json file.

    Args:
        path: Path to tileset.json

    Returns:
        Loaded Tileset object
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    base_path = path.parent
    version = data["asset"]["version"]
    geometric_error = data.get("geometricError", 0.0)
    root = _parse_tile(data["root"])
    properties = data.get("properties")

    return Tileset(
        base_path=base_path,
        version=version,
        geometric_error=geometric_error,
        root=root,
        properties=properties,
    )


@dataclass
class VisibleTile:
    """Result of traversal for a single tile."""
    tile: Tile
    world_transform: np.ndarray
    world_bounds: BoundingVolume
    sse: float
    depth: int


@dataclass
class SseParams:
    """Parameters for SSE computation."""
    viewport_height: float = 1080.0
    fov_y: float = 0.785398  # 45 degrees

    def sse_factor(self) -> float:
        return self.viewport_height / (2.0 * np.tan(self.fov_y / 2.0))


class Tiles3dRenderer:
    """3D Tiles renderer with SSE-based LOD selection."""

    def __init__(
        self,
        sse_threshold: float = 16.0,
        max_depth: int = 32,
        cache_size_mb: int = 256,
    ):
        self.sse_threshold = sse_threshold
        self.max_depth = max_depth
        self.cache_size_mb = cache_size_mb
        self.sse_params = SseParams()
        self._cache: Dict[str, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def set_viewport(self, height: float, fov_y: float):
        """Set viewport parameters for SSE computation."""
        self.sse_params.viewport_height = height
        self.sse_params.fov_y = fov_y

    def compute_sse(
        self,
        geometric_error: float,
        bounding_volume: BoundingVolume,
        camera_position: Tuple[float, float, float],
    ) -> float:
        """Compute screen-space error for a tile."""
        center = bounding_volume.center()
        dx = center[0] - camera_position[0]
        dy = center[1] - camera_position[1]
        dz = center[2] - camera_position[2]
        distance = np.sqrt(dx*dx + dy*dy + dz*dz)

        if distance < 0.001:
            return float("inf")

        return (geometric_error / distance) * self.sse_params.sse_factor()

    def get_visible_tiles(
        self,
        tileset: Tileset,
        camera_position: Tuple[float, float, float],
    ) -> List[VisibleTile]:
        """Get list of visible tiles for rendering.

        Args:
            tileset: The tileset to traverse
            camera_position: Camera position in world space (x, y, z)

        Returns:
            List of visible tiles with SSE and depth info
        """
        result = []
        self._traverse(
            tileset.root,
            camera_position,
            tileset.root.refine,
            np.eye(4, dtype=np.float64),
            0,
            result,
        )
        return result

    def _traverse(
        self,
        tile: Tile,
        camera_pos: Tuple[float, float, float],
        parent_refine: str,
        parent_transform: np.ndarray,
        depth: int,
        result: List[VisibleTile],
    ):
        if depth > self.max_depth:
            return

        local_transform = (
            np.asarray(tile.transform, dtype=np.float64).reshape((4, 4), order="F")
            if tile.transform is not None
            else np.eye(4, dtype=np.float64)
        )
        world_transform = parent_transform @ local_transform
        world_bounds = _transform_bounding_volume(tile.bounding_volume, world_transform)
        sse = self.compute_sse(tile.geometric_error, world_bounds, camera_pos)
        refine = tile.refine or parent_refine
        should_refine = sse > self.sse_threshold and tile.children

        if should_refine:
            if refine == "REPLACE":
                for child in tile.children:
                    self._traverse(child, camera_pos, refine, world_transform, depth + 1, result)
            else:  # ADD
                if tile.has_content():
                    result.append(VisibleTile(tile=tile, world_transform=world_transform, world_bounds=world_bounds, sse=sse, depth=depth))
                for child in tile.children:
                    self._traverse(child, camera_pos, refine, world_transform, depth + 1, result)
        else:
            if tile.has_content():
                result.append(VisibleTile(tile=tile, world_transform=world_transform, world_bounds=world_bounds, sse=sse, depth=depth))
            elif tile.children:
                for child in tile.children:
                    self._traverse(child, camera_pos, refine, world_transform, depth + 1, result)

    def cache_stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "entries": len(self._cache),
        }


def decode_b3dm(data: bytes) -> Dict[str, Any]:
    """Decode a B3DM file.

    Args:
        data: Raw B3DM file bytes

    Returns:
        Dict with positions, normals, colors, indices
        
    Raises:
        NotImplementedError: glTF mesh extraction is not yet implemented
    """
    native = get_native_module() if NATIVE_AVAILABLE else None
    native_decode = getattr(native, "decode_b3dm_py", None) if native else None
    if callable(native_decode):
        decoded = dict(native_decode(data))
        for key in ("feature_table", "batch_table"):
            value = decoded.get(key)
            if isinstance(value, str) and value:
                decoded[key] = json.loads(value)
        return decoded

    if len(data) < 28:
        raise ValueError("B3DM file too small")

    magic = data[0:4]
    if magic != b"b3dm":
        raise ValueError(f"Invalid B3DM magic: {magic}")

    version = int.from_bytes(data[4:8], "little")
    if version != 1:
        raise ValueError(f"Unsupported B3DM version: {version}")

    # Parse header
    ft_json_len = int.from_bytes(data[12:16], "little")
    ft_bin_len = int.from_bytes(data[16:20], "little")
    bt_json_len = int.from_bytes(data[20:24], "little")
    bt_bin_len = int.from_bytes(data[24:28], "little")

    offset = 28 + ft_json_len + ft_bin_len + bt_json_len + bt_bin_len
    gltf_data = data[offset:]
    
    # Validate that glTF data is present
    if len(gltf_data) < 12:
        raise ValueError(f"B3DM contains no valid glTF data (size={len(gltf_data)})")
    
    # Check for glTF magic (binary glTF starts with "glTF")
    gltf_magic = gltf_data[0:4]
    if gltf_magic != b"glTF":
        raise ValueError(f"Embedded content is not glTF (magic={gltf_magic!r})")
    
    # glTF mesh extraction requires a full glTF parser to extract:
    # - accessor data for POSITION, NORMAL, TEXCOORD attributes
    # - buffer views and buffer data
    # - mesh primitives and indices
    # This is deferred until a glTF parsing library is integrated.
    raise NotImplementedError(
        f"B3DM glTF mesh extraction not implemented. "
        f"Found {len(gltf_data)} bytes of glTF data. "
        f"Use a glTF library (e.g., pygltflib) to extract mesh geometry, "
        f"or use the Rust renderer which has native glTF support."
    )


def decode_pnts(data: bytes) -> Dict[str, Any]:
    """Decode a PNTS file.

    Args:
        data: Raw PNTS file bytes

    Returns:
        Dict with positions, colors, normals
    """
    native = get_native_module() if NATIVE_AVAILABLE else None
    native_decode = getattr(native, "decode_pnts_py", None) if native else None
    if callable(native_decode):
        decoded = dict(native_decode(data))
        point_count = int(decoded.get("point_count", 0))
        decoded["positions"] = np.asarray(decoded["positions"], dtype=np.float64).reshape(point_count, 3)
        if decoded.get("relative_positions") is not None:
            decoded["relative_positions"] = np.asarray(
                decoded["relative_positions"], dtype=np.float64
            ).reshape(point_count, 3)
        else:
            # Wheels predating the explicit dual-coordinate native contract
            # return tile-relative positions only. Preserve compatibility at
            # the Python boundary without ever narrowing the absolute result.
            decoded["relative_positions"] = decoded["positions"].copy()
            if len(data) >= 28 and data[:4] == b"pnts":
                ft_json_len = int.from_bytes(data[12:16], "little")
                feature_table = json.loads(data[28 : 28 + ft_json_len].decode("utf-8"))
                rtc_center = feature_table.get("RTC_CENTER")
                if rtc_center is not None:
                    decoded["rtc_center"] = tuple(float(value) for value in rtc_center)
                    decoded["positions"] = decoded["positions"] + np.asarray(
                        rtc_center, dtype=np.float64
                    )
        if decoded.get("colors") is not None:
            decoded["colors"] = np.asarray(decoded["colors"], dtype=np.uint8).reshape(point_count, 3)
        if decoded.get("colors_rgba") is not None:
            decoded["colors_rgba"] = np.asarray(decoded["colors_rgba"], dtype=np.uint8).reshape(point_count, 4)
        if decoded.get("normals") is not None:
            decoded["normals"] = np.asarray(decoded["normals"], dtype=np.float32).reshape(point_count, 3)
        for key in ("feature_table", "batch_table"):
            value = decoded.get(key)
            if isinstance(value, str) and value:
                decoded[key] = json.loads(value)
        return decoded

    if len(data) < 28:
        raise ValueError("PNTS file too small")

    magic = data[0:4]
    if magic != b"pnts":
        raise ValueError(f"Invalid PNTS magic: {magic}")

    version = int.from_bytes(data[4:8], "little")
    if version != 1:
        raise ValueError(f"Unsupported PNTS version: {version}")

    ft_json_len = int.from_bytes(data[12:16], "little")
    offset = 28
    ft_json = json.loads(data[offset:offset + ft_json_len].decode("utf-8"))
    offset += ft_json_len

    ft_bin_len = int.from_bytes(data[16:20], "little")
    ft_bin = data[offset:offset + ft_bin_len]

    points_length = ft_json.get("POINTS_LENGTH", 0)

    # Extract positions
    positions = np.zeros((points_length, 3), dtype=np.float64)
    if "POSITION" in ft_json:
        pos_offset = ft_json["POSITION"].get("byteOffset", 0)
        positions = np.frombuffer(
            ft_bin[pos_offset:pos_offset + points_length * 12],
            dtype=np.float32
        ).astype(np.float64).reshape(-1, 3)

    relative_positions = positions.copy()
    rtc_center = ft_json.get("RTC_CENTER")
    if rtc_center is not None:
        rtc = np.asarray(rtc_center, dtype=np.float64).reshape(3)
        positions = positions + rtc

    colors = None
    if "RGB" in ft_json:
        rgb_offset = ft_json["RGB"].get("byteOffset", 0)
        colors = np.frombuffer(
            ft_bin[rgb_offset:rgb_offset + points_length * 3],
            dtype=np.uint8
        ).reshape(-1, 3)

    return {
        "positions": positions,
        "relative_positions": relative_positions,
        "rtc_center": rtc_center,
        "colors": colors,
        "normals": None,
        "point_count": points_length,
    }


def validate_tiles3d_support(
    tileset: Tileset,
    *,
    layer_id: str | None = None,
) -> ValidationReport:
    """Validate public Python 3D Tiles support before render preparation."""
    effective_layer_id = layer_id or "tiles3d"
    diag = python_public_3dtiles_incomplete_diagnostic(layer_id=effective_layer_id)
    return ValidationReport(
        diagnostics=[diag],
        layer_summaries=[
            LayerSummary(
                layer_id=effective_layer_id,
                layer_type="tiles3d",
                support_level="underdeveloped",
                diagnostic_codes=[diag.code],
                object_count=tileset.tile_count,
                details={
                    "max_depth": tileset.max_depth,
                    "tile_count": tileset.tile_count,
                    "version": tileset.version,
                },
            )
        ],
        unsupported_features={"tiles3d.public_python_render": "underdeveloped"},
    )
