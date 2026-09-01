#!/usr/bin/env python3
"""High-quality daytime Paris/Eiffel still using OSM and official IGN BATI 3D.

The reference is an oblique, stylised city-builder view.  This example keeps
real OSM land-cover geometry and official IGN LoD2.2 building geometry, then
uses an attributed high-detail Eiffel landmark mesh because the streamed IGN
representation does not preserve the tower's visible lattice structure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import numpy as np
from pyproj import Transformer

EIFFEL_LON = 2.2945
EIFFEL_LAT = 48.8584
BUILDING_SOURCE_KIND = "IGN BATI 3D official tiles"
BUILDING_LEVEL = "LoD2.2"
BUILDING_SOURCE_URL = "https://batiment3d.ign.fr/"
EIFFEL_STL_URL = "https://upload.wikimedia.org/wikipedia/commons/4/41/EiffelTower_fixed.stl"
EIFFEL_STL_LICENSE = "CC BY-SA 3.0; Newcandle from Thingiverse"
EIFFEL_STL_SHA256 = "bff2ce1d08609b2d761a225cb7c8c1da782369714011567ede7a5008a1e55adf"

WGS84_TO_ECEF = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
IGN_TILESET_URL = "https://batiment3d.ign.fr/data/tileset.json"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("FORGE3D_PARIS_EIFFEL_DATA_DIR", "D:/forge3d_data/paris_eiffel_day")
)
DECODER_SCRIPT = PROJECT_ROOT / "tools" / "paris_eiffel_day" / "decode_ign_glb.cjs"
REFERENCE_EYE_SCALE = (-1.35, 1.65, 1.15)
REFERENCE_ROTATION_DEG = 30.0
REFERENCE_FOV_DEG = 31.0
REFERENCE_MARGIN_RATIO = 0.045
DEFAULT_RADIUS_M = 400.0
REFERENCE_CONTENT_ZOOM = 1.35

# The daytime reference is intentionally bright and stylised, but the geometry
# below remains sourced from OSM/IGN rather than painted as an illustration.
REFERENCE_SURFACE_RGB = {
    "base": (238, 240, 228, 255),
    "landuse": (210, 226, 178, 255),
    "park": (170, 220, 119, 255),
    "water": (34, 112, 224, 255),
    "road": (244, 232, 207, 255),
    "road_hi": (252, 243, 220, 255),
}
REFERENCE_BUILDING_RGB = (180, 188, 214, 255)
REFERENCE_TOWER_RGB = (153, 89, 35, 255)
REFERENCE_TOWER_DECK_RGB = (67, 143, 171, 255)
REFERENCE_TREE_RGB = (83, 153, 62, 255)
REFERENCE_PITCH_RGB = (92, 176, 78, 255)


@dataclass(frozen=True)
class DecodedOfficialTile:
    """One decoded IGN tile in the service's stored Y-up/ECEF frame."""

    source_url: str
    glb_path: Path
    positions: np.ndarray
    indices: np.ndarray
    metadata: dict
def official_tile_cache_key(url: str) -> str:
    """Return a stable, filesystem-safe key for one official tile URL."""
    return hashlib.sha256(str(url).encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class MeshData:
    """Triangle mesh in local east/up/south render coordinates."""

    positions: np.ndarray
    indices: np.ndarray
    rgba: tuple[int, int, int, int] = (156, 91, 38, 255)
    shadow_alpha: int = 80
    specular: float = 0.08


def enu_basis(lon_deg: float, lat_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return east, north and up unit vectors in Earth-centered coordinates."""
    lon = math.radians(float(lon_deg))
    lat = math.radians(float(lat_deg))
    east = np.asarray([-math.sin(lon), math.cos(lon), 0.0], dtype=np.float64)
    north = np.asarray(
        [-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)],
        dtype=np.float64,
    )
    up = np.asarray(
        [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)],
        dtype=np.float64,
    )
    return east, north, up


def validate_eiffel_center(lon: float, lat: float) -> None:
    """Reject a non-Eiffel anchor instead of mislabeling a generated landmark."""
    requested = np.asarray(WGS84_TO_ECEF.transform(float(lon), float(lat), 0.0), dtype=np.float64)
    target = np.asarray(WGS84_TO_ECEF.transform(EIFFEL_LON, EIFFEL_LAT, 0.0), dtype=np.float64)
    distance_m = float(np.linalg.norm(requested - target))
    if distance_m > 5.0:
        raise ValueError(
            f"Paris Eiffel renderer requires centre ({EIFFEL_LON:.4f}, {EIFFEL_LAT:.4f}); "
            f"requested centre is {distance_m:.1f} m away"
        )


def ign_y_up_to_ecef(points: np.ndarray) -> np.ndarray:
    """Convert IGN BATI 3D's Y-up ECEF storage order to XYZ ECEF."""
    raw = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return np.column_stack((raw[:, 0], -raw[:, 2], raw[:, 1]))


def ign_y_up_to_render(
    points: np.ndarray,
    center_ecef: np.ndarray,
    lon_deg: float,
    lat_deg: float,
    *,
    ground_up: float = 0.0,
) -> np.ndarray:
    """Convert IGN Y-up ECEF storage to local render (east, up, south) metres."""
    ecef = ign_y_up_to_ecef(points)
    local = ecef_to_local_enu(ecef, center_ecef, lon_deg, lat_deg)
    return np.column_stack((local[:, 0], local[:, 2] - float(ground_up), -local[:, 1]))


def ecef_to_local_enu(
    points_ecef: np.ndarray,
    center_ecef: np.ndarray,
    lon_deg: float,
    lat_deg: float,
) -> np.ndarray:
    """Convert ECEF positions to local (east, north, up) metres."""
    points = np.asarray(points_ecef, dtype=np.float64).reshape(-1, 3)
    center = np.asarray(center_ecef, dtype=np.float64).reshape(3)
    east, north, up = enu_basis(lon_deg, lat_deg)
    delta = points - center[None, :]
    return np.column_stack((delta @ east, delta @ north, delta @ up))


def obb_intersects_sphere(
    box: list[float] | tuple[float, ...] | np.ndarray,
    center: np.ndarray,
    radius_m: float,
) -> bool:
    """Exact intersection test for a 3D-Tiles OBB and a sphere.

    The test operates on the orthogonal half-axis vectors used by 3D Tiles
    oriented boxes. It is exact for the service's boxes and avoids fetching
    neighbouring tile branches merely because their circumscribed spheres touch
    the AOI.
    """
    values = np.asarray(box, dtype=np.float64).reshape(-1)
    if values.size != 12:
        return False
    tile_center = values[:3]
    delta = np.asarray(center, dtype=np.float64).reshape(3) - tile_center
    distance_sq = 0.0
    for offset in (3, 6, 9):
        axis = values[offset:offset + 3]
        length = float(np.linalg.norm(axis))
        if length <= 0.0:
            return False
        coordinate = abs(float(np.dot(delta, axis / length)))
        excess = max(coordinate - length, 0.0)
        distance_sq += excess * excess
    return distance_sq <= max(float(radius_m), 0.0) ** 2


def _append_beam(
    vertices: list[np.ndarray],
    triangles: list[tuple[int, int, int]],
    p0: np.ndarray,
    p1: np.ndarray,
    width: float,
    sides: int = 6,
) -> None:
    """Append a closed low-poly cylindrical truss beam."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    direction = p1 - p0
    length = float(np.linalg.norm(direction))
    if length <= 1e-6:
        return
    direction /= length
    helper = np.asarray([0.0, 1.0, 0.0])
    if abs(float(np.dot(direction, helper))) > 0.92:
        helper = np.asarray([1.0, 0.0, 0.0])
    u = np.cross(direction, helper)
    u /= max(float(np.linalg.norm(u)), 1e-12)
    v = np.cross(direction, u)
    v /= max(float(np.linalg.norm(v)), 1e-12)
    start = len(vertices)
    radius = float(width) * 0.5
    for point in (p0, p1):
        for side in range(sides):
            angle = 2.0 * math.pi * side / sides
            vertices.append(point + radius * (math.cos(angle) * u + math.sin(angle) * v))
    for side in range(sides):
        nxt = (side + 1) % sides
        a, b = start + side, start + nxt
        c, d = start + sides + nxt, start + sides + side
        triangles.extend(((a, b, c), (a, c, d)))
    for side in range(1, sides - 1):
        triangles.extend(((start, start + side + 1, start + side),
                          (start + sides, start + sides + side, start + sides + side + 1)))


def _append_box(
    vertices: list[np.ndarray],
    triangles: list[tuple[int, int, int]],
    center: np.ndarray,
    half_x: float,
    half_y: float,
    half_z: float,
) -> None:
    """Append a closed rectangular architectural volume."""
    cx, cy, cz = (float(value) for value in center)
    hx, hy, hz = float(half_x), float(half_y), float(half_z)
    start = len(vertices)
    vertices.extend(
        np.asarray(
            [
                [cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
                [cx + hx, cy - hy, cz + hz], [cx - hx, cy - hy, cz + hz],
                [cx - hx, cy + hy, cz - hz], [cx + hx, cy + hy, cz - hz],
                [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz],
            ],
            dtype=np.float64,
        )
    )
    faces = (
        (0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
        (1, 5, 6, 2), (2, 6, 7, 3), (4, 0, 3, 7),
    )
    for a, b, c, d in faces:
        triangles.extend(((start + a, start + b, start + c), (start + a, start + c, start + d)))


def _append_polyline_beam(
    vertices: list[np.ndarray],
    triangles: list[tuple[int, int, int]],
    points: list[np.ndarray],
    width: float,
    sides: int = 6,
) -> None:
    for first, second in zip(points[:-1], points[1:]):
        _append_beam(vertices, triangles, first, second, width, sides=sides)


def _append_arch(
    vertices: list[np.ndarray],
    triangles: list[tuple[int, int, int]],
    first: np.ndarray,
    second: np.ndarray,
    peak_height: float,
    width: float,
) -> None:
    """Append one broad parabolic entrance arch between two tower legs."""
    points = []
    for step in range(13):
        t = step / 12.0
        point = first * (1.0 - t) + second * t
        point = point.copy()
        point[1] = float(peak_height) * math.sin(math.pi * t) ** 0.82
        points.append(point)
    _append_polyline_beam(vertices, triangles, points, width, sides=7)


def _append_deck_frame(
    vertices: list[np.ndarray],
    triangles: list[tuple[int, int, int]],
    ring: np.ndarray,
    width: float,
) -> None:
    """Append a deck perimeter and its two diagonal support beams."""
    for side in range(4):
        _append_beam(vertices, triangles, ring[side], ring[(side + 1) % 4], width)
    _append_beam(vertices, triangles, ring[0], ring[2], max(1.0, width * 0.62), sides=5)
    _append_beam(vertices, triangles, ring[1], ring[3], max(1.0, width * 0.62), sides=5)


def build_eiffel_tower_mesh() -> MeshData:
    """Build a curved-leg, arched and lattice-like Eiffel Tower landmark mesh."""
    vertices: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    levels = (
        (0.0, 62.5),
        (18.0, 59.0),
        (38.0, 53.0),
        (57.6, 47.0),
        (75.0, 42.0),
        (98.0, 34.0),
        (115.7, 28.0),
        (140.0, 23.0),
        (165.0, 19.0),
        (195.0, 15.0),
        (225.0, 11.0),
        (250.0, 8.0),
        (276.0, 5.0),
        (300.0, 3.0),
        (324.0, 0.8),
    )
    corners: list[np.ndarray] = []
    for height, radius in levels:
        corners.append(
            np.asarray(
                [[sx * radius, height, -sz * radius] for sx, sz in
                 ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))],
                dtype=np.float64,
            )
        )

    # Curved main legs. The measured tower envelope is preserved, but the legs
    # now arc inward through the three architectural levels instead of forming a
    # single straight pyramid.
    for level in range(len(corners) - 1):
        main_width = 4.8 if level < 3 else 3.4 if level < 7 else 2.2 if level < 11 else 1.5
        for corner in range(4):
            _append_beam(vertices, triangles, corners[level][corner], corners[level + 1][corner], main_width)

        # Alternating face braces produce a readable truss rhythm instead of a
        # dense double-X mesh that collapses into visual noise at distance.
        if level % 2 == 0:
            brace_width = max(0.9, main_width * 0.34)
            for corner in range(4):
                nxt = (corner + 1) % 4
                if level % 2 == 0:
                    _append_beam(vertices, triangles, corners[level][corner], corners[level + 1][nxt], brace_width, sides=5)
                else:
                    _append_beam(vertices, triangles, corners[level][nxt], corners[level + 1][corner], brace_width, sides=5)

    # Four signature base arches, one on every face.
    for side in range(4):
        _append_arch(
            vertices,
            triangles,
            corners[0][side],
            corners[0][(side + 1) % 4],
            peak_height=43.0,
            width=4.0,
        )

    # First and second floors plus the upper platform are explicit architectural
    # decks, not just heightfield break lines.
    for level, width in ((3, 3.8), (6, 3.2), (12, 2.0)):
        _append_deck_frame(vertices, triangles, corners[level], width)

    # Foot beams and short inner braces give the four legs a grounded, substantial
    # base when the tower is viewed obliquely.
    for corner in range(4):
        _append_beam(vertices, triangles, corners[0][corner], corners[0][(corner + 1) % 4], 5.8)
        inner = corners[1][corner] * 0.93
        _append_beam(vertices, triangles, corners[0][corner], inner, 3.0, sides=5)

    # Four tapered mast rails, then the central antenna.
    for corner in range(4):
        _append_beam(vertices, triangles, corners[12][corner], corners[13][corner], 1.6, sides=5)
        _append_beam(vertices, triangles, corners[13][corner], corners[14][corner], 1.0, sides=5)
    _append_beam(
        vertices,
        triangles,
        np.asarray([0.0, 276.0, 0.0]),
        np.asarray([0.0, 324.0, 0.0]),
        2.8,
        sides=7,
    )
    _append_beam(
        vertices,
        triangles,
        np.asarray([0.0, 316.0, 0.0]),
        np.asarray([0.0, 324.0, 0.0]),
        0.9,
        sides=5,
    )

    # Clamp the landmark to its documented ground and tip planes; beam sections
    # around sloping endpoints otherwise extend a few centimetres beyond them.
    positions = np.asarray(vertices, dtype=np.float64)
    positions[:, 1] = np.clip(positions[:, 1], 0.0, 324.0)
    positions = positions.astype(np.float32)
    indices = np.asarray(triangles, dtype=np.uint32).reshape(-1, 3)
    return MeshData(positions=positions, indices=indices)


def build_eiffel_deck_mesh() -> MeshData:
    """Build the three visually important observation-deck volumes."""
    vertices: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    for height, half_extent, thickness in ((57.6, 32.0, 1.6), (115.7, 19.0, 1.1), (276.0, 5.5, 0.6)):
        _append_box(
            vertices,
            triangles,
            np.asarray([0.0, height, 0.0]),
            half_extent,
            thickness,
            half_extent,
        )
    return MeshData(
        positions=np.asarray(vertices, dtype=np.float32),
        indices=np.asarray(triangles, dtype=np.uint32).reshape(-1, 3),
        rgba=REFERENCE_TOWER_DECK_RGB,
        shadow_alpha=90,
        specular=0.18,
    )


def parse_binary_stl(data: bytes) -> np.ndarray:
    """Parse a binary STL into ``(triangle, vertex, xyz)`` float32 values."""
    if len(data) < 84:
        raise ValueError("binary STL is shorter than its header")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + triangle_count * 50
    if len(data) < expected_size:
        raise ValueError(
            f"binary STL is truncated: header declares {triangle_count} triangles "
            f"but only {len(data)} bytes are present"
        )
    dtype = np.dtype(
        {
            "names": ["normal", "vertices", "attribute"],
            "formats": [("<f4", (3,)), ("<f4", (3, 3)), "<u2"],
            "offsets": [0, 12, 48],
            "itemsize": 50,
        }
    )
    records = np.frombuffer(data, dtype=dtype, offset=84, count=triangle_count)
    vertices = np.asarray(records["vertices"], dtype=np.float32).copy()
    if not np.isfinite(vertices).all():
        raise ValueError("binary STL contains non-finite vertex coordinates")
    return vertices


def landmark_mesh_from_stl_triangles(triangles: np.ndarray) -> MeshData:
    """Scale a Z-up STL to the measured Eiffel envelope and convert to render axes."""
    triangles = np.asarray(triangles, dtype=np.float32).reshape(-1, 3, 3)
    if triangles.size == 0 or not np.isfinite(triangles).all():
        raise ValueError("Eiffel STL geometry must contain finite triangles")
    flat = triangles.reshape(-1, 3)
    minimum = flat.min(axis=0)
    maximum = flat.max(axis=0)
    span = maximum - minimum
    if float(span[2]) <= 0.0 or float(max(span[0], span[1])) <= 0.0:
        raise ValueError("Eiffel STL has no positive 3D envelope")
    horizontal_scale = 125.0 / float(max(span[0], span[1]))
    vertical_scale = 324.0 / float(span[2])
    local = triangles.astype(np.float64)
    local[:, :, 0] = (local[:, :, 0] - (float(minimum[0]) + float(maximum[0])) * 0.5) * horizontal_scale
    local[:, :, 1] = (local[:, :, 1] - (float(minimum[1]) + float(maximum[1])) * 0.5) * horizontal_scale
    local[:, :, 2] = (local[:, :, 2] - float(minimum[2])) * vertical_scale
    # STL is Z-up; Forge3D's local render coordinates are east/up/south.
    render_triangles = np.stack(
        (local[:, :, 0], local[:, :, 2], -local[:, :, 1]),
        axis=2,
    ).astype(np.float32)
    positions, inverse = np.unique(render_triangles.reshape(-1, 3), axis=0, return_inverse=True)
    indices = inverse.reshape(-1, 3).astype(np.uint32)
    nondegenerate = (
        (indices[:, 0] != indices[:, 1])
        & (indices[:, 1] != indices[:, 2])
        & (indices[:, 0] != indices[:, 2])
    )
    if not np.any(nondegenerate):
        raise ValueError("Eiffel STL contains no non-degenerate triangles")
    return MeshData(
        positions=positions.astype(np.float32),
        indices=indices[nondegenerate],
        rgba=REFERENCE_TOWER_RGB,
        shadow_alpha=125,
        specular=0.16,
    )


def load_eiffel_landmark_mesh(data_root: Path, *, refresh: bool = False) -> MeshData:
    """Load and normalize the attributed Commons landmark mesh."""
    path = Path(data_root) / "raw" / "landmark" / "EiffelTower_fixed.stl"
    payload = _fetch_resource(
        EIFFEL_STL_URL,
        path,
        refresh=refresh,
        allowed_hosts={"upload.wikimedia.org"},
    )
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != EIFFEL_STL_SHA256:
        raise RuntimeError(
            f"Eiffel landmark checksum mismatch: expected {EIFFEL_STL_SHA256}, "
            f"got {actual_sha256}"
        )
    mesh = landmark_mesh_from_stl_triangles(parse_binary_stl(payload))
    print(
        f"[Eiffel] landmark {mesh.positions.shape[0]:,} vertices / "
        f"{mesh.indices.shape[0]:,} triangles; source={EIFFEL_STL_LICENSE}",
        flush=True,
    )
    return mesh


def validate_remote_origin(url: str, allowed_hosts: set[str] | frozenset[str]) -> bool:
    """Allow only HTTPS URLs to an explicitly approved host."""
    parsed = urlparse(str(url))
    hostname = (parsed.hostname or "").lower()
    if (
        parsed.scheme.lower() != "https"
        or hostname not in {str(host).lower() for host in allowed_hosts}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
    ):
        raise ValueError(f"untrusted remote origin: {url}")
    return True


def triangles_intersect_circle(triangles_xy: np.ndarray, radius_m: float) -> np.ndarray:
    """Return a mask for triangles that touch or cross a centred circle."""
    triangles = np.asarray(triangles_xy, dtype=np.float64).reshape(-1, 3, 2)
    radius_sq = max(float(radius_m), 0.0) ** 2
    keep = np.any(np.sum(triangles * triangles, axis=2) <= radius_sq, axis=1)
    for edge in ((0, 1), (1, 2), (2, 0)):
        first = triangles[:, edge[0], :]
        second = triangles[:, edge[1], :]
        delta = second - first
        length_sq = np.sum(delta * delta, axis=1)
        t = np.zeros(length_sq.shape, dtype=np.float64)
        valid = length_sq > 1e-18
        t[valid] = np.clip(-np.sum(first[valid] * delta[valid], axis=1) / length_sq[valid], 0.0, 1.0)
        closest = first + t[:, None] * delta
        keep |= np.sum(closest * closest, axis=1) <= radius_sq
    # A circle containing the triangle's centre can intersect all three edges
    # outside the radius, so also test whether the origin is inside the triangle.
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    cross = np.stack(
        (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
         b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0],
         c[:, 0] * a[:, 1] - c[:, 1] * a[:, 0]),
        axis=1,
    )
    keep |= ((np.all(cross >= 0.0, axis=1) | np.all(cross <= 0.0, axis=1)) & (radius_sq >= 0.0))
    return keep


def _fetch_resource(
    url: str,
    cache_path: Path,
    *,
    refresh: bool = False,
    timeout: float = 180.0,
    allowed_hosts: set[str] | frozenset[str] = frozenset(),
) -> bytes:
    """Fetch one allowlisted URL atomically, reusing a verified local cache."""
    validate_remote_origin(url, allowed_hosts)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.is_file() and not refresh:
        return cache_path.read_bytes()
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request = Request(
                str(url),
                headers={"User-Agent": "forge3d-paris-eiffel-day/1.0"},
            )
            with urlopen(request, timeout=float(timeout)) as response:
                validate_remote_origin(response.geturl(), allowed_hosts)
                payload = response.read()
            temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
            temporary.write_bytes(payload)
            temporary.replace(cache_path)
            return payload
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}: {last_error}") from last_error


def _cached_json(url: str, cache_dir: Path, *, refresh: bool) -> dict:
    path = cache_dir / f"{official_tile_cache_key(url)}.json"
    try:
        return json.loads(
            _fetch_resource(
                url,
                path,
                refresh=refresh,
                allowed_hosts={"batiment3d.ign.fr"},
            ).decode("utf-8")
        )
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        if path.exists():
            path.unlink()
        raise RuntimeError(f"invalid JSON from official IGN tileset {url}: {exc}") from exc


def discover_official_glb_urls(
    lon: float = EIFFEL_LON,
    lat: float = EIFFEL_LAT,
    radius_m: float = DEFAULT_RADIUS_M,
    *,
    cache_dir: Path | None = None,
    refresh: bool = False,
    root_url: str = IGN_TILESET_URL,
) -> list[str]:
    """Discover official IGN GLB tiles whose conservative bounds meet the AOI."""
    cache_dir = (cache_dir or DEFAULT_DATA_ROOT / "cache" / "ign_tilesets").resolve()
    target = np.asarray(WGS84_TO_ECEF.transform(float(lon), float(lat), 0.0), dtype=np.float64)
    validate_remote_origin(root_url, {"batiment3d.ign.fr"})
    visited: set[str] = set()
    found: set[str] = set()

    def visit(url: str) -> None:
        validate_remote_origin(url, {"batiment3d.ign.fr"})
        if url in visited:
            return
        visited.add(url)
        payload = _cached_json(url, cache_dir, refresh=refresh)
        root = payload.get("root", payload)
        children = root.get("children", []) or []
        if not children:
            uri = root.get("content", {}).get("uri")
            if uri and str(uri).lower().endswith(".glb"):
                resolved = urljoin(url, str(uri))
                validate_remote_origin(resolved, {"batiment3d.ign.fr"})
                found.add(resolved)
            return
        for child in children:
            box = child.get("boundingVolume", {}).get("box")
            if box is not None and not obb_intersects_sphere(box, target, radius_m):
                continue
            uri = child.get("content", {}).get("uri")
            if uri:
                resolved = urljoin(url, str(uri))
                validate_remote_origin(resolved, {"batiment3d.ign.fr"})
                if str(uri).lower().endswith(".json"):
                    visit(resolved)
                elif str(uri).lower().endswith(".glb"):
                    found.add(resolved)
            elif child.get("children"):
                # Inline child nodes are legal in 3D Tiles. They inherit this
                # tileset URL as their relative-resolution base.
                inline = dict(child)
                inline_children = inline.pop("children", []) or []
                for inline_child in inline_children:
                    inline_box = inline_child.get("boundingVolume", {}).get("box")
                    if inline_box is not None and not obb_intersects_sphere(
                        inline_box, target, radius_m
                    ):
                        continue
                    inline_uri = inline_child.get("content", {}).get("uri")
                    if not inline_uri:
                        continue
                    resolved = urljoin(url, str(inline_uri))
                    validate_remote_origin(resolved, {"batiment3d.ign.fr"})
                    if str(inline_uri).lower().endswith(".json"):
                        visit(resolved)
                    elif str(inline_uri).lower().endswith(".glb"):
                        found.add(resolved)

    visit(root_url)
    urls = sorted(found)
    if not urls:
        raise RuntimeError(
            f"official IGN BATI 3D hierarchy returned no GLB tiles for "
            f"({lon:.6f}, {lat:.6f}) radius {radius_m:.0f} m"
        )
    print(f"[IGN] discovered {len(urls)} official GLB tile(s) for the Eiffel AOI", flush=True)
    return urls


def download_official_tiles(
    urls: list[str],
    data_root: Path,
    *,
    refresh: bool = False,
) -> list[tuple[str, Path]]:
    """Download official GLB payloads into the D:-drive data root."""
    raw_dir = Path(data_root) / "raw" / "tiles"
    records: list[tuple[str, Path]] = []
    for index, url in enumerate(urls, start=1):
        path = raw_dir / f"{official_tile_cache_key(url)}.glb"
        payload = _fetch_resource(
            url,
            path,
            refresh=refresh,
            allowed_hosts={"batiment3d.ign.fr"},
        )
        if len(payload) < 12 or payload[:4] != b"glTF":
            raise RuntimeError(f"official IGN tile is not a GLB: {url}")
        records.append((url, path))
        print(f"[IGN] tile {index}/{len(urls)} {path.name} {len(payload)/1024:.0f} KiB", flush=True)
    return records


def _meshopt_root(data_root: Path, explicit: str | Path | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_value = os.environ.get("FORGE3D_MESHOPT_ROOT")
    if env_value:
        candidates.append(Path(env_value))
    candidates.extend(
        (
            Path(data_root) / "tools" / "node" / "node_modules" / "meshoptimizer",
            PROJECT_ROOT / "node_modules" / "meshoptimizer",
        )
    )
    for candidate in candidates:
        if (candidate / "meshopt_decoder.cjs").is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise RuntimeError(
        "meshopt decoder package is missing; install meshoptimizer@1.2.0 under "
        f"{Path(data_root) / 'tools' / 'node'} or pass --meshopt-root. Searched: {searched}"
    )


def decode_official_tile(
    source_url: str,
    glb_path: Path,
    data_root: Path,
    *,
    refresh: bool = False,
    meshopt_root: str | Path | None = None,
) -> DecodedOfficialTile:
    """Decode one official meshopt GLB through the dedicated Node adapter."""
    decoded_dir = Path(data_root) / "decoded"
    prefix = decoded_dir / official_tile_cache_key(source_url)
    positions_path = prefix.with_suffix(".positions.f64")
    indices_path = prefix.with_suffix(".indices.u32")
    metadata_path = prefix.with_suffix(".meta.json")
    try:
        source_sha256 = hashlib.sha256(Path(glb_path).read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(f"cannot hash official GLB cache input {glb_path}: {exc}") from exc
    cache_valid = False
    if positions_path.is_file() and indices_path.is_file() and metadata_path.is_file() and not refresh:
        try:
            cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cache_valid = (
                cached_metadata.get("source_url") == source_url
                and cached_metadata.get("source_sha256") == source_sha256
            )
        except (OSError, json.JSONDecodeError):
            cache_valid = False
    if not cache_valid:
        node = shutil.which("node") or "node"
        decoder = DECODER_SCRIPT
        if not decoder.is_file():
            raise RuntimeError(f"missing official GLB decoder adapter: {decoder}")
        package_root = _meshopt_root(Path(data_root), meshopt_root)
        result = subprocess.run(
            [node, str(decoder), str(glb_path), str(prefix), str(package_root)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"meshopt decode failed for {source_url}: {result.stderr.strip() or result.stdout.strip()}"
            )
        if result.stdout.strip():
            print(result.stdout.strip(), flush=True)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["source_url"] = source_url
        metadata["source_sha256"] = source_sha256
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        vertex_count = int(metadata["vertex_count"])
        triangle_count = int(metadata["triangle_count"])
        positions = np.fromfile(positions_path, dtype="<f8").reshape(vertex_count, 3)
        indices = np.fromfile(indices_path, dtype="<u4").reshape(triangle_count, 3)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid decoded official tile cache for {source_url}: {exc}") from exc
    if positions.shape[0] == 0 or indices.shape[0] == 0:
        raise RuntimeError(f"official tile decoded to empty geometry: {source_url}")
    return DecodedOfficialTile(
        source_url=source_url,
        glb_path=Path(glb_path),
        positions=positions,
        indices=indices,
        metadata=metadata,
    )


def load_decoded_official_tiles(
    tile_files: list[tuple[str, Path]],
    data_root: Path,
    *,
    refresh: bool = False,
    meshopt_root: str | Path | None = None,
) -> list[DecodedOfficialTile]:
    """Decode all selected official tiles with per-tile cache reuse."""
    return [
        decode_official_tile(
            source_url,
            path,
            data_root,
            refresh=refresh,
            meshopt_root=meshopt_root,
        )
        for source_url, path in tile_files
    ]


def _compact_mesh(positions: np.ndarray, indices: np.ndarray, **kwargs) -> MeshData:
    used, inverse = np.unique(indices.reshape(-1), return_inverse=True)
    compact_positions = np.asarray(positions, dtype=np.float32)[used]
    compact_indices = inverse.reshape(-1, 3).astype(np.uint32)
    return MeshData(positions=compact_positions, indices=compact_indices, **kwargs)


def merge_official_tiles(
    tiles: list[DecodedOfficialTile],
    lon: float,
    lat: float,
    radius_m: float,
) -> tuple[MeshData, float]:
    """Convert and merge official tiles, clipping triangle coverage to the AOI."""
    if not tiles:
        raise RuntimeError("no official IGN tiles were decoded")
    center_ecef = np.asarray(WGS84_TO_ECEF.transform(float(lon), float(lat), 0.0), dtype=np.float64)
    positions_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    offset = 0
    for tile in tiles:
        render_positions = ign_y_up_to_render(
            tile.positions,
            center_ecef,
            lon,
            lat,
            ground_up=0.0,
        )
        tri_positions = render_positions[tile.indices]
        tri_xy = tri_positions[:, :, (0, 2)]
        keep = triangles_intersect_circle(tri_xy, float(radius_m) * 1.08)
        if not np.any(keep):
            continue
        positions_parts.append(render_positions.astype(np.float64))
        index_parts.append(tile.indices[keep].astype(np.uint32) + np.uint32(offset))
        offset += render_positions.shape[0]
    if not positions_parts or not index_parts:
        raise RuntimeError("official IGN tiles contain no triangles intersecting the requested AOI")
    positions = np.concatenate(positions_parts, axis=0)
    indices = np.concatenate(index_parts, axis=0)
    ground_up = float(np.percentile(positions[:, 1], 1.0))
    positions[:, 1] -= ground_up
    mesh = MeshData(
        positions=positions.astype(np.float32),
        indices=indices,
        rgba=REFERENCE_BUILDING_RGB,
        shadow_alpha=42,
        specular=0.02,
    )
    print(
        f"[IGN] merged {mesh.positions.shape[0]:,} vertices / {mesh.indices.shape[0]:,} triangles; "
        f"ground reference {ground_up:.2f} m; height {mesh.positions[:, 1].min():.1f}.."
        f"{mesh.positions[:, 1].max():.1f} m",
        flush=True,
    )
    return mesh, ground_up


def separate_official_landmark(
    mesh: MeshData,
    *,
    zone_radius_m: float = 155.0,
    min_height_m: float = 120.0,
) -> tuple[MeshData | None, MeshData | None, bool]:
    """Separate high geometry around Eiffel from the ordinary building fabric.

    The complete tower zone is removed from the ordinary mesh intentionally, so
    the thin official IGN tower representation cannot show through the attributed
    landmark mesh used for the visible tower.
    """
    triangle_positions = mesh.positions[mesh.indices]
    centroid = triangle_positions.mean(axis=1)
    radial = np.hypot(centroid[:, 0], centroid[:, 2])
    landmark_mask = radial <= float(zone_radius_m)
    high = triangle_positions[:, :, 1].max(axis=1) >= float(min_height_m)
    official_landmark_mask = landmark_mask & high
    if not np.any(official_landmark_mask):
        return mesh, None, False
    ordinary = None
    if np.any(~landmark_mask):
        ordinary = _compact_mesh(
            mesh.positions,
            mesh.indices[~landmark_mask],
            rgba=REFERENCE_BUILDING_RGB,
            shadow_alpha=mesh.shadow_alpha,
            specular=mesh.specular,
        )
    landmark = _compact_mesh(
        mesh.positions,
        mesh.indices[official_landmark_mask],
        rgba=REFERENCE_TOWER_RGB,
        shadow_alpha=115,
        specular=0.12,
    )
    present = bool(float(landmark.positions[:, 1].max()) >= 250.0 and landmark.indices.shape[0] >= 200)
    print(
        f"[IGN] landmark split triangles={landmark.indices.shape[0]:,} "
        f"height={landmark.positions[:, 1].max():.1f} m present={present}",
        flush=True,
    )
    return ordinary, landmark, present


def combined_osm_query(bbox: tuple[float, float, float, float]) -> str:
    """Build one Overpass request for land cover/infrastructure only."""
    west, south, east, north = (float(value) for value in bbox)
    extent = f"({south:.8f},{west:.8f},{north:.8f},{east:.8f})"
    selectors = (
        f'way["landuse"]{extent};',
        f'way["leisure"="park"]{extent};',
        f'way["leisure"="pitch"]{extent};',
        f'way["natural"~"^(wood|scrub|water)$"]{extent};',
        f'way["waterway"="riverbank"]{extent};',
        f'way["water"~"^(river|canal)$"]{extent};',
        f'relation["natural"="water"]{extent};',
        f'relation["waterway"="riverbank"]{extent};',
        f'relation["water"~"^(river|canal)$"]{extent};',
        f'way["highway"]{extent};',
        f'way["railway"]{extent};',
    )
    return f"[out:json][timeout:90];({''.join(selectors)});out geom qt;"


def fetch_combined_osm_elements(
    bbox: tuple[float, float, float, float],
    cache_path: Path,
    *,
    refresh: bool = False,
) -> list[dict]:
    """Fetch one combined OSM response, with cache and mirror fallback."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.is_file() and not refresh:
        try:
            return list(json.loads(cache_path.read_text(encoding="utf-8")).get("elements", []))
        except (OSError, json.JSONDecodeError):
            cache_path.unlink(missing_ok=True)
    query = combined_osm_query(bbox)
    last_error: Exception | None = None
    endpoints = (
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    )
    for endpoint in endpoints:
        for attempt in range(2):
            try:
                request = Request(
                    endpoint,
                    data=query.encode("utf-8"),
                    headers={
                        "Content-Type": "text/plain; charset=utf-8",
                        "User-Agent": "forge3d-paris-eiffel-day/1.0",
                    },
                )
                with urlopen(request, timeout=180.0) as response:
                    payload = json.load(response)
                cache_path.write_text(json.dumps(payload), encoding="utf-8")
                elements = list(payload.get("elements", []))
                print(f"[OSM] combined response elements={len(elements):,}", flush=True)
                return elements
            except HTTPError as exc:
                last_error = exc
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    wait_seconds = max(1.0, float(retry_after)) if retry_after else 8.0 * (attempt + 1)
                except (TypeError, ValueError):
                    wait_seconds = 8.0 * (attempt + 1)
                if exc.code == 429 and attempt == 0:
                    print(f"[OSM] {endpoint} rate-limited; waiting {wait_seconds:.0f}s", flush=True)
                    time.sleep(wait_seconds)
            except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt == 0:
                    time.sleep(2.0)
    if cache_path.is_file() and not refresh:
        try:
            return list(json.loads(cache_path.read_text(encoding="utf-8")).get("elements", []))
        except (OSError, json.JSONDecodeError):
            pass
    raise RuntimeError(f"combined Overpass request failed: {last_error}") from last_error




def build_osm_reference_surfaces(
    lon: float,
    lat: float,
    radius_m: float,
    data_root: Path,
    *,
    refresh: bool = False,
):
    """Build OSM land-cover surfaces without querying OSM buildings."""
    from _import_shim import ensure_repo_import
    from pyproj import Transformer
    from shapely.geometry import Point, box
    from shapely.ops import transform, unary_union

    ensure_repo_import()
    import osm_city_demo as city

    cache_root = Path(data_root) / "cache" / "osm"
    cache_root.mkdir(parents=True, exist_ok=True)
    epsg = city.utm_epsg_for_lon_lat(float(lon), float(lat))
    to_metric = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    to_wgs84 = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    center = transform(to_metric.transform, Point(float(lon), float(lat)))
    center_xy = tuple(float(value) for value in center.coords[0])
    aoi = center.buffer(float(radius_m), resolution=96)
    bbox = transform(to_wgs84.transform, aoi).bounds
    slug = city.cache_slug(lon, lat, radius_m)

    old_cache = city.CACHE_DIR
    old_rotation = city.SCENE_ROTATION_DEG
    old_overpass_urls = city.OVERPASS_URLS
    city.CACHE_DIR = cache_root
    city.SCENE_ROTATION_DEG = REFERENCE_ROTATION_DEG
    city.OVERPASS_URLS = ("https://overpass-api.de/api/interpreter",)
    try:
        combined_path = cache_root / f"{slug}_land_cover_bundle_v2.json"
        elements = fetch_combined_osm_elements(
            bbox,
            combined_path,
            refresh=bool(refresh),
        )

        def tagged(predicate):
            return [element for element in elements if predicate(dict(element.get("tags", {})))]

        landuse_elements = tagged(lambda tags: "landuse" in tags)
        park_elements = tagged(lambda tags: tags.get("leisure") == "park")
        park_elements += tagged(lambda tags: tags.get("natural") in {"wood", "scrub"})
        water_elements = tagged(lambda tags: tags.get("natural") == "water")
        water_elements += tagged(lambda tags: tags.get("waterway") == "riverbank")
        water_elements += tagged(lambda tags: tags.get("water") in {"river", "canal"})
        road_elements = tagged(lambda tags: "highway" in tags)
        rail_elements = tagged(lambda tags: "railway" in tags)
        pitch_elements = tagged(lambda tags: tags.get("leisure") == "pitch")

        landuse = city.parse_polygon_features(landuse_elements, to_metric, aoi)
        parks = city.parse_polygon_features(park_elements, to_metric, aoi)
        water = city.parse_polygon_features(water_elements, to_metric, aoi)
        roads = city.parse_line_features(road_elements, to_metric, aoi)
        rails = city.parse_line_features(rail_elements, to_metric, aoi)
        pitches = city.parse_polygon_features(pitch_elements, to_metric, aoi)

        landuse_local = [city.localize_feature(feature, center_xy) for feature in landuse]
        park_local = [
            city.localize_feature(feature, center_xy)
            for feature in landuse
            if str(feature["tags"].get("landuse", "")).lower() in city.GREEN_LANDUSE
        ]
        park_local.extend(city.localize_feature(feature, center_xy) for feature in parks)
        water_local = [city.localize_feature(feature, center_xy) for feature in water]
        roads_local = [city.localize_feature(feature, center_xy) for feature in roads]
        rails_local = [city.localize_feature(feature, center_xy) for feature in rails]
        pitch_local = [city.localize_feature(feature, center_xy) for feature in pitches]

        surfaces = []

        def append_surface(geometry, rgba, *, elevation: float, specular: float = 0.0, reflectivity: float = 0.0):
            polygonal = city._extract_polygonal(city._fix_geom(geometry))
            if polygonal is None or polygonal.is_empty:
                return
            surfaces.append(
                city.SurfaceLayer(
                    geometry=polygonal,
                    rgba=rgba,
                    elevation=float(elevation),
                    specular=float(specular),
                    reflectivity=float(reflectivity),
                )
            )

        append_surface(
            box(-float(radius_m), -float(radius_m), float(radius_m), float(radius_m)),
            city.COLORS["base"],
            elevation=0.02,
        )
        if landuse_local:
            append_surface(
                city.merge_surface_geometry(landuse_local, simplify_tolerance=1.4),
                city.COLORS["landuse"],
                elevation=0.5,
            )
        if park_local:
            append_surface(
                city.merge_surface_geometry(park_local, simplify_tolerance=1.2),
                city.COLORS["park"],
                elevation=0.51,
            )
        if water_local:
            append_surface(
                city.merge_surface_geometry(water_local, simplify_tolerance=0.9),
                city.COLORS["water"],
                elevation=-0.5,
                specular=city.WATER_SPECULAR_INTENSITY,
                reflectivity=0.22,
            )
        if roads_local:
            roads_union = unary_union([feature["geometry"] for feature in roads_local])
            append_surface(
                city.simplify_geom(
                    roads_union.buffer(3.0, cap_style=city.BufferCapStyle.round, join_style=city.BufferJoinStyle.round),
                    0.7,
                ),
                city.COLORS["road"],
                elevation=1.0,
            )
            append_surface(
                city.simplify_geom(
                    roads_union.buffer(1.3, cap_style=city.BufferCapStyle.round, join_style=city.BufferJoinStyle.round),
                    0.5,
                ),
                city.COLORS["road_hi"],
                elevation=1.12,
            )
        if rails_local:
            rails_union = unary_union([feature["geometry"] for feature in rails_local])
            append_surface(
                city.simplify_geom(
                    rails_union.buffer(2.0, cap_style=city.BufferCapStyle.round, join_style=city.BufferJoinStyle.round),
                    0.6,
                ),
                city.COLORS["road"],
                elevation=1.0,
            )
        if pitch_local:
            append_surface(
                city.merge_surface_geometry(pitch_local, simplify_tolerance=0.4),
                REFERENCE_PITCH_RGB,
                elevation=1.14,
            )
        print(
            f"[OSM] landuse={len(landuse_local)} parks={len(park_local)} "
            f"water={len(water_local)} roads={len(roads_local)} rails={len(rails_local)} "
            f"pitches={len(pitch_local)}",
            flush=True,
        )
    finally:
        city.CACHE_DIR = old_cache
        city.SCENE_ROTATION_DEG = old_rotation
        city.OVERPASS_URLS = old_overpass_urls

    styled = []
    for surface in surfaces:
        key = tuple(int(value) for value in surface.rgba[:3])
        replacements = {
            tuple(city.COLORS["base"][:3]): REFERENCE_SURFACE_RGB["base"],
            tuple(city.COLORS["landuse"][:3]): REFERENCE_SURFACE_RGB["landuse"],
            tuple(city.COLORS["park"][:3]): REFERENCE_SURFACE_RGB["park"],
            tuple(city.COLORS["water"][:3]): REFERENCE_SURFACE_RGB["water"],
            tuple(city.COLORS["road"][:3]): REFERENCE_SURFACE_RGB["road"],
            tuple(city.COLORS["road_hi"][:3]): REFERENCE_SURFACE_RGB["road_hi"],
        }
        replacement = replacements.get(key)
        styled.append(surface if replacement is None else replace(surface, rgba=replacement))
    print(f"[OSM] reference surfaces={len(styled)}", flush=True)
    return city, surfaces, styled


def _polygon_parts(geometry):
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return [part for part in geometry.geoms if not part.is_empty]
    if geometry.geom_type == "GeometryCollection":
        parts = []
        for part in geometry.geoms:
            parts.extend(_polygon_parts(part))
        return parts
    return []


def sample_tree_points(surfaces, *, spacing_m: float = 27.0) -> np.ndarray:
    """Sample deterministic tree points only inside OSM park polygons."""
    from shapely.geometry import Point

    points: list[tuple[float, float]] = []
    spacing = max(float(spacing_m), 2.0)
    park_rgb = (95, 190, 96)
    for surface_index, surface in enumerate(surfaces):
        if tuple(int(value) for value in surface.rgba[:3]) != park_rgb:
            continue
        for polygon_index, polygon in enumerate(_polygon_parts(surface.geometry)):
            min_x, min_y, max_x, max_y = polygon.bounds
            seed_bytes = hashlib.sha256(f"{surface_index}:{polygon_index}:{polygon.bounds}".encode()).digest()
            phase_x = (int.from_bytes(seed_bytes[:4], "little") / 2**32) * spacing
            phase_y = (int.from_bytes(seed_bytes[4:8], "little") / 2**32) * spacing
            x = min_x + phase_x
            while x <= max_x:
                y = min_y + phase_y
                while y <= max_y:
                    if polygon.contains(Point(x, y)):
                        jitter = hashlib.sha256(f"{x:.2f}:{y:.2f}".encode()).digest()
                        jx = (int.from_bytes(jitter[:2], "little") / 65535.0 - 0.5) * spacing * 0.28
                        jy = (int.from_bytes(jitter[2:4], "little") / 65535.0 - 0.5) * spacing * 0.28
                        candidate = Point(x + jx, y + jy)
                        if polygon.contains(candidate):
                            points.append((float(candidate.x), float(candidate.y)))
                    y += spacing
                x += spacing
    if not points:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def _append_canopy(
    vertices: list[np.ndarray],
    triangles: list[tuple[int, int, int]],
    center: np.ndarray,
    radius: float,
    height: float,
    sides: int = 7,
) -> None:
    """Append a two-ring low-poly tree canopy."""
    center = np.asarray(center, dtype=np.float64)
    start = len(vertices)
    lower_y = center[1] + height * 0.28
    upper_y = center[1] + height * 0.72
    for y, ring_radius in ((lower_y, radius), (upper_y, radius * 0.72)):
        for side in range(sides):
            angle = 2.0 * math.pi * side / sides
            vertices.append(
                np.asarray(
                    [center[0] + ring_radius * math.cos(angle), y,
                     center[2] + ring_radius * math.sin(angle)],
                    dtype=np.float64,
                )
            )
    top = len(vertices)
    vertices.append(np.asarray([center[0], center[1] + height, center[2]], dtype=np.float64))
    for side in range(sides):
        nxt = (side + 1) % sides
        triangles.extend(
            (
                (start + side, start + nxt, start + sides + nxt),
                (start + side, start + sides + nxt, start + sides + side),
                (start + sides + side, start + sides + nxt, top),
            )
        )


def build_tree_scatter_mesh(points: np.ndarray) -> MeshData | None:
    """Build lightweight tree meshes at OSM park sample points."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if points.size == 0:
        return None
    vertices: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    for index, (east, north) in enumerate(points):
        digest = hashlib.sha256(f"tree:{index}:{east:.2f}:{north:.2f}".encode()).digest()
        scale = 0.78 + (digest[0] / 255.0) * 0.42
        base = np.asarray([east, 0.0, -north], dtype=np.float64)
        _append_beam(
            vertices,
            triangles,
            base,
            base + np.asarray([0.0, 2.1 * scale, 0.0]),
            0.48 * scale,
            sides=5,
        )
        _append_canopy(
            vertices,
            triangles,
            base,
            2.0 * scale,
            6.4 * scale,
            sides=7,
        )
    return MeshData(
        positions=np.asarray(vertices, dtype=np.float32),
        indices=np.asarray(triangles, dtype=np.uint32).reshape(-1, 3),
        rgba=REFERENCE_TREE_RGB,
        shadow_alpha=22,
        specular=0.0,
    )


def rotate_render_mesh(mesh: MeshData, degrees: float) -> MeshData:
    """Rotate local east/south render coordinates around the AOI origin."""
    angle = math.radians(float(degrees))
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    positions = np.asarray(mesh.positions, dtype=np.float32).copy()
    x = positions[:, 0].copy()
    z = positions[:, 2].copy()
    # OSM/Shapely rotates (east, north) counter-clockwise. The renderer stores
    # north as -Z, so the equivalent operation in (east, south) is:
    #   x' = x cos(theta) + z sin(theta)
    #   z' = -x sin(theta) + z cos(theta)
    positions[:, 0] = x * cos_a + z * sin_a
    positions[:, 2] = -x * sin_a + z * cos_a
    return replace(mesh, positions=positions)


def build_reference_scene(
    lon: float,
    lat: float,
    radius_m: float,
    data_root: Path,
    *,
    refresh_osm: bool = False,
    refresh_ign: bool = False,
    refresh_decode: bool = False,
    refresh_landmark: bool = False,
    meshopt_root: str | Path | None = None,
    tree_spacing_m: float = 27.0,
    no_trees: bool = False,
):
    """Fetch, decode, and assemble the complete reference scene."""
    validate_eiffel_center(lon, lat)
    city, raw_surfaces, surfaces = build_osm_reference_surfaces(
        lon,
        lat,
        radius_m,
        data_root,
        refresh=refresh_osm,
    )
    tile_urls = discover_official_glb_urls(
        lon,
        lat,
        radius_m,
        cache_dir=Path(data_root) / "cache" / "ign_tilesets",
        refresh=refresh_ign,
    )
    tile_files = download_official_tiles(tile_urls, data_root, refresh=refresh_ign)
    decoded = load_decoded_official_tiles(
        tile_files,
        data_root,
        refresh=bool(refresh_decode or refresh_ign),
        meshopt_root=meshopt_root,
    )
    official, ground_up = merge_official_tiles(decoded, lon, lat, radius_m)
    ordinary, official_tower, official_tower_present = separate_official_landmark(official)
    meshes = []
    if ordinary is not None:
        meshes.append(rotate_render_mesh(ordinary, REFERENCE_ROTATION_DEG))
    # The IGN tile contains the authoritative tower location/height but its
    # streamed representation is a thin slab. Use the attributed high-detail
    # landmark mesh for the visible tower while retaining IGN LoD2.2 for every
    # ordinary building.
    landmark = load_eiffel_landmark_mesh(data_root, refresh=refresh_landmark)
    meshes.append(rotate_render_mesh(landmark, REFERENCE_ROTATION_DEG))
    if not no_trees:
        tree_points = sample_tree_points(raw_surfaces, spacing_m=tree_spacing_m)
        trees = build_tree_scatter_mesh(tree_points)
        tree_count = int(tree_points.shape[0])
        if trees is not None:
            meshes.append(trees)
    else:
        tree_count = 0
    scene = city.SceneLayers(
        surfaces=surfaces,
        meshes=[
            city.MeshLayer(
                positions=mesh.positions,
                indices=mesh.indices,
                rgba=mesh.rgba,
                shadow_alpha=mesh.shadow_alpha,
                specular=mesh.specular,
            )
            for mesh in meshes
        ],
        roof_outlines=[],
        focus_landmarks=[],
        radius=float(radius_m),
    )
    return scene, {
        "tile_urls": tile_urls,
        "tile_count": len(decoded),
        "ground_reference_m": ground_up,
        "official_height_max_m": float(official.positions[:, 1].max()),
        "tower_from_official": False,
        "official_tower_detected": official_tower_present,
        "eiffel_landmark_source": f"{EIFFEL_STL_URL} ({EIFFEL_STL_LICENSE})",
        "tree_count": tree_count,
    }


def apply_reference_day_grade(image):
    """Lift saturation and exposure to match the bright daytime reference."""
    from PIL import Image

    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    luma = array @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    graded = luma[:, :, None] + (array - luma[:, :, None]) * 1.10
    graded *= 1.035
    return Image.fromarray(np.clip(graded, 0.0, 255.0).astype(np.uint8), mode="RGB")


def zoom_reference_frame(image, factor: float = REFERENCE_CONTENT_ZOOM):
    """Scale the active scene into the canvas while preserving the top anchor."""
    from PIL import Image

    factor = float(factor)
    if factor <= 1.0:
        return image.copy()
    width, height = image.size
    crop_width = max(2, int(round(width / factor)))
    crop_height = max(2, int(round(height / factor)))
    left = max(0, (width - crop_width) // 2)
    top = 0
    crop = image.crop((left, top, left + crop_width, top + crop_height))
    resampling = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    return crop.resize((width, height), resample=resampling)


def render_day_still(
    scene,
    width: int,
    height: int,
    *,
    supersample: int = 1,
    frame_index: int = 120,
):
    """Render a fixed, bright daytime frame with the existing city renderer."""
    from _import_shim import ensure_repo_import
    from PIL import Image

    ensure_repo_import()
    import osm_city_daycycle as day

    prepared = day.prepare_scene(
        scene,
        width=int(width),
        height=int(height),
        supersample=max(1, int(supersample)),
        eye_scale=REFERENCE_EYE_SCALE,
        target_height_ratio=0.045,
        fov_deg=REFERENCE_FOV_DEG,
        margin_ratio=REFERENCE_MARGIN_RATIO,
    )
    sun = day.sun_state_for_frame(int(frame_index), 240)
    old_background = day.make_background

    def reference_background(render_width: int, render_height: int, current_sun) -> Image.Image:
        base = day.city.shade_rgba(
            REFERENCE_SURFACE_RGB["base"],
            normal=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            light_dir=current_sun.light_dir,
            view_dir=prepared.view_dir,
            specular=0.0,
        )
        return Image.new("RGBA", (int(render_width), int(render_height)), tuple(base[:3]) + (255,))

    day.make_background = reference_background
    try:
        image = day.render_frame(
            prepared,
            sun,
            frame_index=int(frame_index),
            total_frames=240,
            clock_start_hour=10.0,
            clock_end_hour=16.0,
            show_timer=False,
            shadow_opacity=0.92,
        )
    finally:
        day.make_background = old_background
    graded = apply_reference_day_grade(image)
    return zoom_reference_frame(graded), prepared


def write_source_manifest(path: Path, *, lon: float, lat: float, radius_m: float, metadata: dict) -> None:
    """Write provenance beside the render without claiming an absent licence."""
    payload = {
        "schema_version": 1,
        "scene": "Paris Eiffel daytime reference still",
        "centre": {"lon": float(lon), "lat": float(lat)},
        "radius_m": float(radius_m),
        "land_cover_source": "OpenStreetMap contributors via Overpass",
        "building_source_kind": BUILDING_SOURCE_KIND,
        "building_level": BUILDING_LEVEL,
        "building_source_url": BUILDING_SOURCE_URL,
        "eiffel_landmark_url": EIFFEL_STL_URL,
        "eiffel_landmark_license": EIFFEL_STL_LICENSE,
        "eiffel_landmark_sha256": EIFFEL_STL_SHA256,
        "official_tile_hierarchy": IGN_TILESET_URL,
        "official_tile_axis_conversion": "IGN BATI 3D Y-up storage [X, Z, -Y] -> ECEF XYZ -> local ENU",
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lon", type=float, default=EIFFEL_LON)
    parser.add_argument("--lat", type=float, default=EIFFEL_LAT)
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS_M)
    parser.add_argument("--size", type=int, nargs=2, default=(1600, 1200), metavar=("W", "H"))
    parser.add_argument("--supersample", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATA_ROOT / "out" / "paris_eiffel_day.png",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--meshopt-root", type=Path, default=None)
    parser.add_argument("--refresh-osm", action="store_true")
    parser.add_argument("--refresh-ign", action="store_true")
    parser.add_argument("--refresh-decode", action="store_true")
    parser.add_argument("--refresh-landmark", action="store_true")
    parser.add_argument("--tree-spacing", type=float, default=27.0)
    parser.add_argument("--no-trees", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    print(
        f"[Paris-Eiffel] centre=({args.lon:.6f}, {args.lat:.6f}) "
        f"radius={args.radius:.0f} m size={args.size[0]}x{args.size[1]}",
        flush=True,
    )
    scene, metadata = build_reference_scene(
        float(args.lon),
        float(args.lat),
        float(args.radius),
        data_root,
        refresh_osm=bool(args.refresh_osm),
        refresh_ign=bool(args.refresh_ign),
        refresh_decode=bool(args.refresh_decode),
        refresh_landmark=bool(args.refresh_landmark),
        meshopt_root=args.meshopt_root,
        tree_spacing_m=float(args.tree_spacing),
        no_trees=bool(args.no_trees),
    )
    image, prepared = render_day_still(
        scene,
        int(args.size[0]),
        int(args.size[1]),
        supersample=max(1, int(args.supersample)),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    from PIL.PngImagePlugin import PngInfo

    pnginfo = PngInfo()
    pnginfo.add_text("building_source", BUILDING_SOURCE_KIND)
    pnginfo.add_text("building_level", BUILDING_LEVEL)
    pnginfo.add_text("building_source_url", BUILDING_SOURCE_URL)
    pnginfo.add_text("land_cover_source", "OpenStreetMap contributors via Overpass")
    pnginfo.add_text("camera", f"high-angle oblique; fov={REFERENCE_FOV_DEG:g} deg")
    pnginfo.add_text("scene_rotation_deg", str(REFERENCE_ROTATION_DEG))
    pnginfo.add_text("content_zoom", str(REFERENCE_CONTENT_ZOOM))
    pnginfo.add_text("official_tile_count", str(metadata["tile_count"]))
    pnginfo.add_text("tower_from_official", str(metadata["tower_from_official"]))
    pnginfo.add_text("official_tower_detected", str(metadata["official_tower_detected"]))
    pnginfo.add_text("eiffel_landmark_source", metadata["eiffel_landmark_source"])
    pnginfo.add_text("eiffel_landmark_license", EIFFEL_STL_LICENSE)
    pnginfo.add_text("eiffel_landmark_sha256", EIFFEL_STL_SHA256)
    image.save(output, format="PNG", pnginfo=pnginfo)
    write_source_manifest(
        data_root / "paris_eiffel_day_manifest.json",
        lon=float(args.lon),
        lat=float(args.lat),
        radius_m=float(args.radius),
        metadata={**metadata, "render_width": int(image.width), "render_height": int(image.height)},
    )
    print(
        f"[Paris-Eiffel] prepared triangles={len(prepared.triangles):,} "
        f"surfaces={len(prepared.surfaces):,}",
        flush=True,
    )
    print(f"[Paris-Eiffel] wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
