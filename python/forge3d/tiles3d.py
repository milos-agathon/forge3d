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
            0,
            result,
        )
        return result

    def _traverse(
        self,
        tile: Tile,
        camera_pos: Tuple[float, float, float],
        parent_refine: str,
        depth: int,
        result: List[VisibleTile],
    ):
        if depth > self.max_depth:
            return

        sse = self.compute_sse(tile.geometric_error, tile.bounding_volume, camera_pos)
        refine = tile.refine or parent_refine
        should_refine = sse > self.sse_threshold and tile.children

        if should_refine:
            if refine == "REPLACE":
                for child in tile.children:
                    self._traverse(child, camera_pos, refine, depth + 1, result)
            else:  # ADD
                if tile.has_content():
                    result.append(VisibleTile(tile=tile, sse=sse, depth=depth))
                for child in tile.children:
                    self._traverse(child, camera_pos, refine, depth + 1, result)
        else:
            if tile.has_content():
                result.append(VisibleTile(tile=tile, sse=sse, depth=depth))
            elif tile.children:
                for child in tile.children:
                    self._traverse(child, camera_pos, refine, depth + 1, result)

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
    if len(data) < 28:
        raise ValueError("B3DM file too small")

    magic = data[0:4]
    if magic != b"b3dm":
        raise ValueError(f"Invalid B3DM magic: {magic}")

    version = int.from_bytes(data[4:8], "little")
    if version != 1:
        raise ValueError(f"Unsupported B3DM version: {version}")

    # Parse header
    byte_length = int.from_bytes(data[8:12], "little")
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
    positions = np.zeros((points_length, 3), dtype=np.float32)
    if "POSITION" in ft_json:
        pos_offset = ft_json["POSITION"].get("byteOffset", 0)
        positions = np.frombuffer(
            ft_bin[pos_offset:pos_offset + points_length * 12],
            dtype=np.float32
        ).reshape(-1, 3)

    colors = None
    if "RGB" in ft_json:
        rgb_offset = ft_json["RGB"].get("byteOffset", 0)
        colors = np.frombuffer(
            ft_bin[rgb_offset:rgb_offset + points_length * 3],
            dtype=np.uint8
        ).reshape(-1, 3)

    return {
        "positions": positions,
        "colors": colors,
        "normals": None,
        "point_count": points_length,
    }
