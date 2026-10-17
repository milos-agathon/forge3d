#!/usr/bin/env python3
"""SPLAT-FUSED sensor + terrain — a cartographic sheet of ONE fused render.

What this makes
---------------
``splat_fused_map.png`` — a map sheet (title, legend, scale bar, north
arrow, camera/render readouts, optional AOV insets) around a single
path-traced image in which three heterogeneous representations live in one
scene:

  * a terrain heightfield  (exact 2.5D patch tracing),
  * a Gaussian-splat cloud (analytic ray / Gaussian intersection),
  * a LiDAR COPC point cloud (oriented surfels),

all occluding and shadowing each other in one ReSTIR pass, with the unified
transmittance ``T = T_splat * T_lidar * T_terrain``.

World frame (important!)
------------------------
The fused tracer is **y-up and the terrain grid is centred on the world
origin**: column ``c`` sits at ``x = (c-(cols-1)/2)*spacing[0]``, row ``r``
at ``z = (r-(rows-1)/2)*spacing[1]``, height along ``y``. Every camera in
this script is built in that frame.

Scenes
------
Default (offline): the committed three-representation survey fixture
(65x65 DEM ridge + 3,500 splats + 5,500 LiDAR returns) with its canonical
camera. ``--dem fuji`` swaps in real terrain fetched from the web (Git LFS
media CDN, cached in ``~/.forge3d/datasets``), tints it with a hypsometric
palette and derives a splat layer from the same DEM so the layers stay
geographically coherent.

Camera control
--------------
  --view hero|canonical|oblique|high|top|low   framing presets
  --azimuth/--elevation/--distance            orbit around --target
  --fov                                        vertical field of view
  --origin/--look-at                           raw camera (overrides all)

The default for ``--dem`` is ``hero``: a low cinematic oblique (az 35,
el 10, fov 55) with sky above the summit, warm side lighting and a
brighter sky. ``--view top`` gives the north-up hypsometric map view.

Run:
    python examples/splat_fused_map.py
    python examples/splat_fused_map.py --dem fuji
    python examples/splat_fused_map.py --dem fuji --view top --aovs
    python examples/splat_fused_map.py --view low --azimuth 120 --elevation 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from _import_shim import ensure_repo_import

ensure_repo_import()

import forge3d as f3d  # noqa: E402
from forge3d import splat  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "splat_fusion"
ASSET_DIR = REPO_ROOT / "assets"
DEFAULT_OUT = REPO_ROOT / "examples" / "out" / "splat_fused_map"

DEM_PRESETS = {
    "fuji": ASSET_DIR / "tif" / "Mount_Fuji_30m.tif",
    "swiss": ASSET_DIR / "tif" / "switzerland_dem.tif",
    "rainier": ASSET_DIR / "tif" / "dem_rainier.tif",
    "luxembourg": ASSET_DIR / "tif" / "luxembourg_dem.tif",
}

LFS_BASE_URL = "https://media.githubusercontent.com/media/milos-agathon/forge3d/main/assets"

SUN_INTENSITY = 2.5
SUN_COLOR = (1.0, 0.97, 0.92)
SH_C0 = 0.28209479177387814  # 3DGS DC spherical-harmonic constant

#: hit_kind -> (legend label, legend swatch RGB)
HIT_LEGEND = {
    0: ("Sky / miss", (46, 52, 62)),
    3: ("Terrain heightfield", (104, 168, 92)),
    4: ("Gaussian splats", (112, 112, 200)),
    5: ("LiDAR returns", (200, 108, 104)),
}

#: view presets as (azimuth deg, elevation deg); "canonical" = fixture camera
VIEW_PRESETS: Dict[str, Optional[Tuple[float, float]]] = {
    "canonical": None,
    "hero": (35.0, 10.0),  # cinematic low oblique with sky above the summit
    "oblique": (35.0, 50.0),
    "high": (30.0, 60.0),
    "top": (0.0, 88.0),
    "low": (35.0, 10.0),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(80).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def _resolve_dem_path(path: Path) -> Path:
    """Local .tif is a Git LFS pointer -> fetch the real binary once."""
    if not _is_lfs_pointer(path):
        return path
    cache_dir = Path.home() / ".forge3d" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / path.name
    if cached.exists() and cached.stat().st_size > 1000:
        print(f"[dem] using cached download: {cached}")
        return cached
    if path.is_relative_to(REPO_ROOT):
        rel = str(path.relative_to(REPO_ROOT)).replace("assets/", "", 1)
        url = f"{LFS_BASE_URL}/{rel}"
    else:
        url = f"{LFS_BASE_URL}/tif/{path.name}"
    print(f"[dem] {path.name} is a Git LFS pointer; downloading real binary...")
    import urllib.request

    try:
        urllib.request.urlretrieve(url, cached)
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            f"could not download {path.name} from {url}: {exc}\n"
            "install git-lfs (`git lfs pull`) or pass --terrain <file.tif>"
        ) from exc
    if cached.stat().st_size < 1000:
        cached.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded file for {path.name} is too small (still a pointer?)")
    print(f"[dem] downloaded {cached.stat().st_size // 1024} KiB -> {cached}")
    return cached


def _load_dem(path: Path) -> Tuple[np.ndarray, Tuple[float, float], float, str, str]:
    """Return (heights f32, cell size per grid cell, metres per cell along x,
    place name, datum label)."""
    path = _resolve_dem_path(path)
    import rasterio  # type: ignore

    with rasterio.open(path) as src:
        band = src.read(1, masked=True)
        data = np.where(band.mask, np.nan, band.astype(np.float32))
        transform = src.transform
        resolution = (abs(transform.a), abs(transform.e))
        nodata = src.nodata
        bounds = src.bounds
    if nodata is not None:
        data = np.where(data == float(nodata), np.nan, data)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError(f"DEM {path} has no finite elevation cells")
    data = np.where(np.isfinite(data), data, float(np.nanmedian(finite)))

    if max(abs(bounds.left), abs(bounds.right)) <= 180.0 and max(
        abs(bounds.top), abs(bounds.bottom)
    ) <= 90.0:
        lat = (bounds.top + bounds.bottom) / 2.0
        metres_per_cell = resolution[0] * 111320.0 * math.cos(math.radians(lat))
        datum = f"{resolution[0]:.5f} deg, ~{metres_per_cell:.1f} m/cell @ {lat:.2f} deg"
    else:
        metres_per_cell = resolution[0]
        datum = f"{resolution[0]:.1f} m/cell"
    name = path.stem.replace("_", " ")
    for suffix in (" DEM", " dem", " 30m", " 12m", " 90m"):
        name = name.replace(suffix, "")
    return data.astype(np.float32), resolution, metres_per_cell, name.strip(), datum


def _hypso_map(heights: np.ndarray) -> np.ndarray:
    """Hypsometric tint (rows, cols, 3) uint8: green lowland -> brown ->
    rock -> snow."""
    lo = float(heights.min())
    hi = float(heights.max())
    t = (heights - lo) / (hi - lo) if hi > lo else np.zeros_like(heights)
    stops = np.array(
        [
            [0.00, 38, 68, 40],   # dense forest: dark, desaturated green
            [0.22, 58, 86, 48],   # upper forest / scrub
            [0.42, 150, 138, 86], # alpine tan
            [0.60, 140, 106, 72],
            [0.74, 108, 92, 92],  # bare rock
            [0.86, 148, 148, 158],
            [0.94, 206, 208, 216],
            [1.00, 252, 252, 255],
        ],
        dtype=np.float64,
    )
    out = np.empty(t.shape + (3,), dtype=np.uint8)
    for ch in range(3):
        out[..., ch] = np.interp(t, stops[:, 0], stops[:, ch + 1]).astype(np.uint8)
    return out


def _derive_splat_cloud(
    heights: np.ndarray,
    spacing: Tuple[float, float],
    tint: np.ndarray,
    target_count: int = 2400,
    lift: float = 0.8,
    opacity: float = 0.45,
) -> Tuple[Any, Dict[str, float]]:
    """Digitize a *scan block* of the DEM into a Gaussian-splat layer.

    Splats are sampled on a step grid inside a circle on the summit flank,
    oriented perpendicular to the local terrain normal and lifted along it,
    so together they form one semi-translucent, violet-tinted quilted skin
    that follows the relief instead of burying into it. Everywhere outside
    the circle the exact heightfield shows untouched — "one sensor block
    fused over the base map" instead of confetti — and both representations
    take hits in roughly the block/terrain area ratio.
    """
    rows, cols = heights.shape
    half_x = (cols - 1) / 2.0 * spacing[0]
    half_z = (rows - 1) / 2.0 * spacing[1]
    centre_x, centre_z = 0.38 * half_x, 0.34 * half_z
    # Clamped so a full-extent DEM still gets a survey-sized block (0.9 km)
    # instead of one scaled to the whole mountain.
    radius = min(0.45 * min(half_x, half_z), 130.0)
    block_area = math.pi * radius * radius
    step = max(2, int(round(math.sqrt(block_area / max(target_count, 1)))))
    rz = np.arange(0, rows, step)
    cx = np.arange(0, cols, step)
    gr, gc = np.meshgrid(rz, cx, indexing="ij")
    x = (gc - (cols - 1) / 2.0) * spacing[0]
    z = (gr - (rows - 1) / 2.0) * spacing[1]
    inside = (x - centre_x) ** 2 + (z - centre_z) ** 2 <= radius**2
    gr, gc = gr[inside], gc[inside]
    x, z = x[inside], z[inside]
    # Terrain normal at each sampled cell (y-up world, 1 unit per cell):
    # n = normalize(-dz/dx, 1, -dz/drow); discs are oriented perpendicular to
    # it so the skin follows steep slopes instead of burying into them, and
    # lifted along it so the block visibly hovers above the base map.
    d_row, d_col = np.gradient(heights)  # [d/dz, d/dx]
    nxv = -d_col[gr, gc]
    nzv = -d_row[gr, gc]
    nyv = np.ones_like(nxv)
    norm = np.sqrt(nxv * nxv + nyv * nyv + nzv * nzv)
    nxv, nyv, nzv = nxv / norm, nyv / norm, nzv / norm
    y = heights[gr, gc]
    positions = np.stack(
        [x + nxv * lift, y + nyv * lift, z + nzv * lift], axis=-1
    ).reshape(-1, 3).astype(np.float32)
    n = positions.shape[0]
    sigma_xz = float(step) * 0.9
    scales = np.empty((n, 3), dtype=np.float32)
    scales[:, 0] = sigma_xz
    scales[:, 1] = 0.35
    scales[:, 2] = sigma_xz
    # Quaternion rotating local +Y (the thin axis) onto the terrain normal:
    # q = normalize(1 + dot(Y, n), cross(Y, n)) with cross(Y, n) = (nz, 0, -nx).
    qw = 1.0 + nyv
    qx = nzv
    qz = -nxv
    qn = np.sqrt(qw * qw + qx * qx + qz * qz)
    rotations = np.stack(
        [qw / qn, qx / qn, np.zeros_like(qw), qz / qn], axis=-1
    ).astype(np.float32)
    opacities = np.full(n, opacity, dtype=np.float32)
    rgb = tint[gr, gc].reshape(-1, 3).astype(np.float32) / 255.0
    # Semi-translucent violet survey-overlay tint — deliberately different
    # from the base map and matching the "Gaussian splats" legend swatch.
    # Only a touch of the ground tint bleeds through, so over green or tan
    # terrain the block still reads as one clean violet layer, not a smudge.
    rgb = 0.10 * rgb + 0.90 * np.array([0.44, 0.42, 0.95], dtype=np.float32)
    sh0 = ((rgb - 0.5) / SH_C0).astype(np.float32)
    cloud = splat.GaussianSplatCloud.from_arrays(positions, scales, rotations, opacities, sh0)
    meta = {
        "count": float(n),
        "sigma_xz": sigma_xz,
        "lift": lift,
        "opacity": opacity,
        "radius": radius,
        "centre": (centre_x, centre_z),
    }
    return cloud, meta


def _derive_lidar_swath(
    heights: np.ndarray,
    spacing: Tuple[float, float],
    target_count: int = 10000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Simulate an airborne LiDAR scan swath over the DEM.

    Flight lines run east-west across a survey strip on the same flank as
    the digitized scan block — the strip sits in front of the block and
    partly underlaps it, so splat and point returns share a region and the
    unified transmittance (T_splat x T_lidar x T_terrain) has somewhere to
    act. Pulse spacing, line spacing and a small perpendicular jitter follow
    a real multi-pulse pattern; return intensity is approximated from the
    local slope (steep ground returns darker and warmer), so the swath reads
    as an intensity image draped on the relief.

    The result is world-space y-up float32 positions plus sRGB byte colours
    — exactly what ``render_fused(point_positions=..., point_colors=...)``
    accepts — so no file is written and the run stays deterministic.
    """
    rows, cols = heights.shape
    half_x = (cols - 1) / 2.0 * spacing[0]
    half_z = (rows - 1) / 2.0 * spacing[1]
    centre_x, centre_z = 0.38 * half_x, 0.34 * half_z
    radius = min(0.45 * min(half_x, half_z), 130.0)
    # Survey strip: laid on the camera-facing flank in front of the block and
    # partly underlapping it, so the violet quilt and the sand-coloured scan
    # interleave at the boundary and all three representations are legible
    # from the hero camera. (The example's hero and top views both look down
    # the +x/+z quadrant, so "front" is a fixed world direction here.)
    x_lo = max(-half_x, centre_x - 0.7 * radius)
    x_hi = min(half_x, centre_x + 1.7 * radius)
    z_lo = max(-half_z, centre_z + 0.4 * radius)
    z_hi = min(half_z, centre_z + 1.4 * radius)
    if x_hi - x_lo < 4.0 or z_hi - z_lo < 4.0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), {"count": 0}
    area = (x_hi - x_lo) * (z_hi - z_lo)
    along = max(1.0, math.sqrt(area / max(target_count, 1)))
    cross = along * 1.45
    rng = np.random.default_rng(seed)
    line_z = np.arange(z_lo + 0.5 * cross, z_hi, cross)
    if line_z.size == 0:
        line_z = np.array([0.5 * (z_lo + z_hi)])
    x_line = np.arange(x_lo, x_hi, along)
    gz, gx = np.meshgrid(line_z, x_line, indexing="ij")
    # Perpendicular pulse jitter (deterministic for a given seed).
    jx = gx + (rng.random(gx.shape) - 0.5) * along * 0.8
    jz = gz + (rng.random(gz.shape) - 0.5) * cross * 0.6
    xw = jx.ravel()
    zw = jz.ravel()
    col_i = np.clip(np.rint(xw / spacing[0] + (cols - 1) / 2.0).astype(int), 0, cols - 1)
    row_i = np.clip(np.rint(zw / spacing[1] + (rows - 1) / 2.0).astype(int), 0, rows - 1)
    yw = heights[row_i, col_i] + 0.05
    positions = np.stack([xw, yw, zw], axis=-1).astype(np.float32)
    # Slope-driven intensity: gentle ground returns pale sand, steep ground
    # warm terracotta — the legend's "LiDAR returns" swatch family.
    d_row, d_col = np.gradient(heights)
    slope = np.hypot(d_row, d_col)[row_i, col_i]
    lo_p, hi_p = (float(v) for v in np.percentile(slope, (5, 95)))
    t = np.clip((slope - lo_p) / max(hi_p - lo_p, 1e-9), 0.0, 1.0).astype(np.float32)
    color_lo = np.array([243.0, 225.0, 196.0], np.float32)
    color_hi = np.array([168.0, 76.0, 66.0], np.float32)
    rgb = color_lo + (color_hi - color_lo) * t[:, None]
    rgb *= (0.94 + 0.12 * rng.random(rgb.shape)).astype(np.float32)
    colors = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    meta = {
        "count": int(positions.shape[0]),
        "flight_lines": int(line_z.size),
        "along_spacing": along,
        "cross_spacing": cross,
        "strip": (x_lo, z_lo, x_hi, z_hi),
    }
    return positions, colors, meta


# ---------------------------------------------------------------------------
# Camera rig (y-up, terrain centred on origin)
# ---------------------------------------------------------------------------


def _orbit_camera(
    target: Sequence[float],
    azimuth_deg: float,
    elevation_deg: float,
    distance: float,
    fov_y_deg: float,
) -> splat.FusedCamera:
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    origin = (
        target[0] + distance * math.cos(el) * math.sin(az),
        target[1] + distance * math.sin(el),
        target[2] + distance * math.cos(el) * math.cos(az),
    )
    return splat.FusedCamera(
        origin=origin, look_at=tuple(target), up=(0.0, 1.0, 0.0), fov_y_deg=fov_y_deg
    )


def _bounds_corners(bounds) -> List[Tuple[float, float, float]]:
    (x0, y0, z0), (x1, y1, z1) = bounds
    return [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]


def _fit_distance(
    bounds,
    target: Sequence[float],
    azimuth_deg: float,
    elevation_deg: float,
    fov_y_deg: float,
    width: int,
    height: int,
    pad: float = 0.08,
) -> float:
    """Smallest orbit distance at which every scene-bounds corner projects
    inside the frame with a ``pad`` margin (a bounding sphere frames flat
    terrain far too loosely)."""
    corners = _bounds_corners(bounds)

    def fits(distance: float) -> bool:
        cam = _orbit_camera(target, azimuth_deg, elevation_deg, distance, fov_y_deg)
        mx, my = width * pad, height * pad
        for corner in corners:
            p = _project(cam, corner, width, height)
            if p is None:
                return False
            if not (mx <= p[0] <= width - mx and my <= p[1] <= height - my):
                return False
        return True

    radius = max(math.dist(corner, target) for corner in corners)
    lo, hi = radius * 0.2, radius * 6.0
    if not fits(hi):  # pathological framing: fall back to the sphere fit
        fov_v = math.radians(fov_y_deg)
        fov_h = 2.0 * math.atan(math.tan(fov_v / 2.0) * (width / height))
        return radius / math.sin(min(fov_v, fov_h) / 2.0) * 1.15
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if fits(mid):
            hi = mid
        else:
            lo = mid
    return hi


def _point_in_quad(point: Sequence[float], quad: Sequence[Sequence[float]]) -> bool:
    sign: Optional[bool] = None
    for i in range(4):
        ax, ay = quad[i]
        bx, by = quad[(i + 1) % 4]
        cross = (bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax)
        if abs(cross) < 1e-9:
            continue
        side = cross > 0.0
        if sign is None:
            sign = side
        elif side != sign:
            return False
    return sign is not None


def _cover_distance(
    bounds,
    target: Sequence[float],
    azimuth_deg: float,
    elevation_deg: float,
    fov_y_deg: float,
    width: int,
    height: int,
    fallback: float,
) -> float:
    """Largest orbit distance at which the projected ground plane still
    covers the WHOLE frame — a full-bleed map instead of a floating island
    in a sea of sky. Falls back to the frame fit when the plane can never
    cover the frame (e.g. very low views)."""
    (x0, _, z0), (x1, _, z1) = bounds
    ground_y = bounds[0][1]
    ring = [
        (x0, ground_y, z0),
        (x1, ground_y, z0),
        (x1, ground_y, z1),
        (x0, ground_y, z1),
    ]
    frame_corners = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]

    def covers(distance: float) -> bool:
        cam = _orbit_camera(target, azimuth_deg, elevation_deg, distance, fov_y_deg)
        projected = [_project(cam, corner, width, height) for corner in ring]
        if any(p is None for p in projected):
            return False
        quad = [(p[0], p[1]) for p in projected if p is not None]
        return all(_point_in_quad(corner, quad) for corner in frame_corners)

    radius = max(math.dist(corner, target) for corner in _bounds_corners(bounds))
    # `covers` is true only on an interval: too far and the quad shrinks
    # inside the frame, too close and ring corners pass behind the camera.
    # Scan down from far to bracket the top of that interval, then bisect.
    previous_false: Optional[float] = None
    found: Optional[float] = None
    distance = radius * 6.0
    while distance > radius * 0.05:
        if covers(distance):
            found = distance
            break
        previous_false = distance
        distance *= 0.85
    if found is None or previous_false is None:
        return fallback
    lo_d, hi_d = found, previous_false
    for _ in range(30):
        mid = (lo_d + hi_d) / 2.0
        if covers(mid):
            lo_d = mid
        else:
            hi_d = mid
    return lo_d


def _hero_distance(
    bounds,
    target: Sequence[float],
    fov_y_deg: float,
    elevation_deg: float,
    margin: float = 0.8,
) -> float:
    """Distance for the `hero` low oblique: the frame's bottom ray must land
    inside the terrain (ground fills the bottom of the frame instead of a void
    under the near edge), while the shot stays wide enough for sky above the
    summit. Solves  d*cos(el) - (ty + d*sin(el)) / tan(el + fov/2) = m*half."""
    (x0, _, z0), (x1, _, z1) = bounds
    half = min((x1 - x0), (z1 - z0)) / 2.0
    el = math.radians(elevation_deg)
    delta = math.radians(elevation_deg + fov_y_deg / 2.0)  # bottom-ray depression
    denom = math.cos(el) - math.sin(el) / math.tan(delta)
    if denom <= 1e-6:
        return _fit_distance(bounds, target, 35.0, elevation_deg, fov_y_deg, 800, 560)
    return (margin * half + target[1] / math.tan(delta)) / denom


def _nice_contour_interval(relief_m: float) -> float:
    for step in (10.0, 20.0, 25.0, 50.0, 100.0, 200.0, 250.0, 500.0, 1000.0, 2000.0):
        if relief_m / step <= 20.0:
            return step
    return 5000.0


def _draw_contours(
    rgba: np.ndarray,
    heights: np.ndarray,
    metres_per_world: float,
    cam: splat.FusedCamera,
    interval_m: Optional[float] = None,
) -> Tuple[np.ndarray, str]:
    """Marching-squares elevation contours projected through the camera.

    Contours are cartographic annotation, not render output: the beauty PNG
    stays untouched and only the sheet's map area receives the overlay, so
    the path-traced image itself remains exactly what the tracer produced.
    Every fifth contour is an index line (darker, thicker). Cells with the
    ambiguous four-crossing marching-squares case are skipped; at DEM
    resolution the gap is invisible.
    """
    from PIL import Image, ImageDraw  # noqa: E402

    rows, cols = heights.shape
    lo, hi = float(heights.min()), float(heights.max())
    relief_m = (hi - lo) * metres_per_world
    step_m = interval_m or _nice_contour_interval(relief_m)
    step_w = step_m / metres_per_world
    index_m = step_m * 5.0
    frame_h, frame_w = rgba.shape[:2]
    overlay = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    hx, hz = (cols - 1) / 2.0, (rows - 1) / 2.0

    h00, h01 = heights[:-1, :-1], heights[:-1, 1:]
    h10, h11 = heights[1:, :-1], heights[1:, 1:]
    first = math.ceil(lo / step_w) * step_w
    levels = [first + i * step_w for i in range(int((hi - first) / step_w) + 1)]
    n_segments = 0
    for level in levels:
        below = heights < level
        e_t = below[:-1, :-1] != below[:-1, 1:]
        e_b = below[1:, :-1] != below[1:, 1:]
        e_l = below[:-1, :-1] != below[1:, :-1]
        e_r = below[:-1, 1:] != below[1:, 1:]
        ncross = e_t.astype(np.uint8) + e_b + e_l + e_r
        sel = ncross == 2
        n_sel = int(sel.sum())
        if n_sel == 0:
            continue
        p0x = np.empty(n_sel)
        p0z = np.empty(n_sel)
        p1x = np.empty(n_sel)
        p1z = np.empty(n_sel)
        filled = np.zeros(n_sel, dtype=bool)
        pos = np.full(e_t.shape, -1, dtype=np.int64)
        pos[sel] = np.arange(n_sel)
        for edge, kind in ((e_t, "t"), (e_b, "b"), (e_l, "l"), (e_r, "r")):
            idx = np.flatnonzero(sel & edge)
            if idx.size == 0:
                continue
            r, c = np.unravel_index(idx, sel.shape)
            if kind == "t":
                t = (level - h00[r, c]) / (h01[r, c] - h00[r, c])
                wx, wz = c + t - hx, np.full(idx.size, r - hz, dtype=np.float64)
            elif kind == "b":
                t = (level - h10[r, c]) / (h11[r, c] - h10[r, c])
                wx, wz = c + t - hx, np.full(idx.size, r + 1.0 - hz, dtype=np.float64)
            elif kind == "l":
                t = (level - h00[r, c]) / (h10[r, c] - h00[r, c])
                wx, wz = np.full(idx.size, c - hx, dtype=np.float64), r + t - hz
            else:
                t = (level - h01[r, c]) / (h11[r, c] - h01[r, c])
                wx, wz = np.full(idx.size, c + 1.0 - hx, dtype=np.float64), r + t - hz
            j = pos[r, c]
            take0 = ~filled[j]
            i0 = j[take0]
            p0x[i0] = wx[take0]
            p0z[i0] = wz[take0]
            filled[i0] = True
            take1 = ~take0
            i1 = j[take1]
            p1x[i1] = wx[take1]
            p1z[i1] = wz[take1]
        ratio = (level * metres_per_world) / index_m
        is_index = abs(ratio - round(ratio)) < 1e-6
        colour = (70, 46, 28, 150) if is_index else (94, 62, 38, 85)
        width = 2 if is_index else 1
        for i in range(n_sel):
            a = _project(cam, (p0x[i], level, p0z[i]), frame_w, frame_h)
            if a is None:
                continue
            b = _project(cam, (p1x[i], level, p1z[i]), frame_w, frame_h)
            if b is None:
                continue
            draw.line([a[0], a[1], b[0], b[1]], fill=colour, width=width)
            n_segments += 1
    base = Image.fromarray(rgba, "RGBA")
    base.alpha_composite(overlay)
    desc = f"every {step_m:.0f} m (index {index_m:.0f} m), {n_segments:,} segments"
    return np.asarray(base), desc


def _project(
    cam: splat.FusedCamera, point: Sequence[float], width: int, height: int
) -> Optional[Tuple[float, float, float]]:
    """Pinhole project -> (px, py, pixels-per-world-unit at that depth)."""
    ox, oy, oz = cam.origin
    fx, fy, fz = (cam.look_at[0] - ox, cam.look_at[1] - oy, cam.look_at[2] - oz)
    fl = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
    fx, fy, fz = fx / fl, fy / fl, fz / fl
    ux, uy, uz = cam.up
    # right = normalize(forward x up)
    rx, ry, rz = fy * uz - fz * uy, fz * ux - fx * uz, fx * uy - fy * ux
    rl = math.sqrt(rx * rx + ry * ry + rz * rz) or 1.0
    rx, ry, rz = rx / rl, ry / rl, rz / rl
    # true up = right x forward
    sx, sy, sz = ry * fz - rz * fy, rz * fx - rx * fz, rx * fy - ry * fx
    vx, vy, vz = point[0] - ox, point[1] - oy, point[2] - oz
    depth = vx * fx + vy * fy + vz * fz
    if depth <= 1e-6:
        return None
    xs = vx * rx + vy * ry + vz * rz
    ys = vx * sx + vy * sy + vz * sz
    tan_half = math.tan(math.radians(cam.fov_y_deg) / 2.0)
    aspect = width / height
    ndc_x = xs / (depth * tan_half * aspect)
    ndc_y = ys / (depth * tan_half)
    px = (ndc_x * 0.5 + 0.5) * width
    py = (1.0 - (ndc_y * 0.5 + 0.5)) * height
    pixels_per_unit = (height / 2.0) / (depth * tan_half)
    return px, py, pixels_per_unit


def _camera_angles(cam: splat.FusedCamera) -> Tuple[float, float, float]:
    dx = cam.origin[0] - cam.look_at[0]
    dy = cam.origin[1] - cam.look_at[1]
    dz = cam.origin[2] - cam.look_at[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
    az = math.degrees(math.atan2(dx, dz)) % 360.0
    el = math.degrees(math.asin(max(-1.0, min(1.0, dy / dist))))
    return az, el, dist


def _nice_length(x: float) -> float:
    if x <= 0.0:
        return 1.0
    exponent = math.floor(math.log10(x))
    best = 10.0**exponent
    for mantissa in (1.0, 2.0, 5.0):
        candidate = mantissa * 10.0**exponent
        if candidate <= x * 1.0001:
            best = max(best, candidate)
    if best > x:
        best = 10.0**exponent
    return best


# ---------------------------------------------------------------------------
# Sheet composition (PIL imported lazily so the CLI parses without Pillow)
# ---------------------------------------------------------------------------


def _load_font(size: int):
    from PIL import ImageFont  # noqa: E402

    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _ramp_lut(stops: Sequence[Tuple[float, Sequence[int]]]) -> np.ndarray:
    xs = np.array([s[0] for s in stops])
    lut = np.empty((256, 3), dtype=np.uint8)
    t = np.linspace(0.0, 1.0, 256)
    for ch in range(3):
        lut[:, ch] = np.interp(t, xs, [s[1][ch] for s in stops]).astype(np.uint8)
    return lut


def _hit_kind_rgb(hit_kind: np.ndarray) -> np.ndarray:
    rgb = np.empty(hit_kind.shape + (3,), dtype=np.uint8)
    rgb[...] = HIT_LEGEND[0][1]
    for kind, (_, color) in HIT_LEGEND.items():
        rgb[hit_kind == kind] = color
    return rgb


def _depth_rgb(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth)
    out = np.zeros(depth.shape + (3,), dtype=np.uint8)
    if not finite.any():
        return out
    lo, hi = np.percentile(depth[finite], [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1.0
    t = np.clip((np.nan_to_num(depth, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
    lut = _ramp_lut(
        [
            (0.00, (16, 18, 46)),
            (0.25, (58, 32, 110)),
            (0.50, (154, 46, 96)),
            (0.75, (232, 122, 60)),
            (1.00, (252, 236, 170)),
        ]
    )
    return lut[(t * 255).astype(np.uint8)]


def _compose_sheet(
    beauty: np.ndarray,
    hit_kind: np.ndarray,
    depth: Optional[np.ndarray],
    transmittance: Optional[np.ndarray],
    camera: splat.FusedCamera,
    meta: Dict[str, Any],
    include_aovs: bool,
):
    from PIL import Image, ImageDraw  # noqa: E402

    paper = (247, 244, 238)
    ink = (24, 28, 33)
    muted = (104, 109, 116)
    rule = (214, 208, 197)
    panel_tone = (240, 236, 228)

    map_h, map_w = beauty.shape[:2]
    margin = 22
    panel_w = 270
    panel_gap = 18
    header_h = 86
    map_y = header_h
    canvas_w = margin + map_w + panel_gap + panel_w + margin

    insets: List[Tuple[str, np.ndarray]] = []
    if include_aovs:
        insets = [("hit kind", _hit_kind_rgb(hit_kind))]
        if depth is not None:
            insets.append(("depth (world units)", _depth_rgb(depth)))
        if transmittance is not None:
            t_rgb = (np.clip(transmittance[..., 0], 0.0, 1.0) * 255).astype(np.uint8)
            insets.append(("sun transmittance T", np.stack([t_rgb] * 3, axis=-1)))

    inset_block = 0
    if insets:
        inset_block = int(map_h * 0.42) + 30
    footer_h = 40
    canvas_h = map_y + map_h + inset_block + footer_h + margin

    canvas = Image.new("RGBA", (canvas_w, canvas_h), paper + (255,))
    draw = ImageDraw.Draw(canvas, "RGBA")

    title_font = _load_font(27)
    sub_font = _load_font(13)
    head_font = _load_font(12)
    body_font = _load_font(12)
    small_font = _load_font(11)

    # ---- header ----------------------------------------------------------
    draw.text((margin, 18), meta["title"], font=title_font, fill=ink)
    draw.text((margin, 54), meta["subtitle"], font=sub_font, fill=muted)
    draw.line((margin, header_h - 10, canvas_w - margin, header_h - 10), fill=rule, width=1)

    # ---- map -------------------------------------------------------------
    map_x = margin
    beauty_img = Image.fromarray(beauty[..., :3], mode="RGB").convert("RGBA")
    canvas.alpha_composite(beauty_img, dest=(map_x, map_y))
    draw.rectangle(
        (map_x - 1, map_y - 1, map_x + map_w, map_y + map_h),
        outline=(38, 42, 48, 220),
        width=2,
    )

    # scale bar (world units at the camera target -> metres)
    az, el, dist = _camera_angles(camera)
    projected = _project(camera, camera.look_at, map_w, map_h)
    if projected is not None:
        ppp = projected[2]
        world_len = _nice_length(140.0 / ppp)
        bar_px = max(40.0, world_len * ppp)
        metres = world_len * meta["meters_per_world"]
        if metres >= 1000.0:
            label = f"{metres / 1000.0:.3g} km"
        else:
            label = f"{metres:.3g} m"
        bx = map_x + 18
        by = map_y + map_h - 30
        chip = (bx - 10, by - 14, bx + bar_px + 58, by + 30)
        draw.rectangle(chip, fill=(250, 248, 244, 205))
        seg = bar_px / 4.0
        for i in range(4):
            fill = (28, 32, 38, 235) if i % 2 == 0 else (250, 248, 244, 235)
            draw.rectangle((bx + i * seg, by, bx + (i + 1) * seg, by + 7), fill=fill)
        draw.rectangle((bx, by, bx + bar_px, by + 7), outline=(28, 32, 38, 235), width=1)
        draw.text((bx - 2, by + 11), "0", font=small_font, fill=ink)
        draw.text((bx + bar_px - 6, by + 11), label, font=small_font, fill=ink)

    # north arrow (world north = -z, projected onto the screen)
    north_probe = _project(
        camera,
        (
            camera.look_at[0],
            camera.look_at[1],
            camera.look_at[2] - max(meta["extent_x"], meta["extent_z"]) * 0.25,
        ),
        map_w,
        map_h,
    )
    centre = _project(camera, camera.look_at, map_w, map_h)
    if north_probe is not None and centre is not None:
        dx = north_probe[0] - centre[0]
        dy = north_probe[1] - centre[1]
        norm = math.hypot(dx, dy)
        if norm > 1e-3:
            dx, dy = dx / norm, dy / norm
            cxp = map_x + map_w - 42
            cyp = map_y + 42
            draw.ellipse(
                (cxp - 24, cyp - 24, cxp + 24, cyp + 24), fill=(250, 248, 244, 205)
            )
            tip = (cxp + 17 * dx, cyp + 17 * dy)
            base = (cxp - 9 * dx, cyp - 9 * dy)
            wing = (-dy, dx)
            w1 = (base[0] + 6 * wing[0], base[1] + 6 * wing[1])
            w2 = (base[0] - 6 * wing[0], base[1] - 6 * wing[1])
            draw.polygon([tip, w1, w2], fill=(28, 32, 38, 240))
            draw.text(
                (cxp + 24 * dx - 4, cyp + 24 * dy - 7),
                "N",
                font=head_font,
                fill=(28, 32, 38, 245),
            )

    # ---- AOV insets ------------------------------------------------------
    if insets:
        iy = map_y + map_h + 14
        gap = 12
        iw = (map_w - gap * (len(insets) - 1)) // len(insets)
        ih = int(map_h * 0.42)
        for idx, (caption, rgb) in enumerate(insets):
            ix = map_x + idx * (iw + gap)
            panel = Image.fromarray(rgb[..., :3], mode="RGB").convert("RGBA")
            if panel.size != (iw, ih):
                panel = panel.resize((iw, ih))
            canvas.alpha_composite(panel, dest=(ix, iy))
            draw.rectangle((ix - 1, iy - 1, ix + iw, iy + ih), outline=(38, 42, 48, 160), width=1)
            draw.text((ix, iy + ih + 6), caption, font=small_font, fill=muted)

    # ---- right panel -----------------------------------------------------
    px = margin + map_w + panel_gap
    py = map_y
    draw.rectangle((px, py, px + panel_w, py + map_h), fill=panel_tone + (255,))

    def wrap(value: str, max_width: int, font) -> List[str]:
        lines: List[str] = []
        current = ""
        for word in value.split():
            trial = f"{current} {word}".strip()
            if not current or draw.textlength(trial, font=font) <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
                if len(lines) == 3:
                    break
        if current and len(lines) < 3:
            lines.append(current)
        return lines or [""]

    def block(y: int, heading: str, rows: List[Tuple[str, str]]) -> int:
        draw.text((px + 16, y), heading, font=head_font, fill=muted)
        y += 21
        key_w = 84
        value_max = panel_w - key_w - 30
        for key, value in rows:
            lines = wrap(value, value_max, body_font)
            draw.text((px + 16, y), key, font=body_font, fill=ink)
            for i, line in enumerate(lines):
                draw.text((px + 16 + key_w, y), line, font=body_font, fill=muted)
                y += 17
        return y + 8

    y = py + 16
    draw.text((px + 16, y), "LAYERS (HIT PIXELS)", font=head_font, fill=muted)
    y += 21
    for kind, count in meta["hits"]:
        label, color = HIT_LEGEND[kind]
        draw.rectangle((px + 16, y + 1, px + 30, y + 15), fill=color + (255,))
        draw.rectangle((px + 16, y + 1, px + 30, y + 15), outline=(38, 42, 48, 120), width=1)
        draw.text((px + 38, y), label, font=body_font, fill=ink)
        draw.text((px + panel_w - 76, y), f"{count:,}", font=body_font, fill=muted)
        y += 19
    y += 8
    y = block(y, "SCENE", meta["scene_rows"])
    y = block(y, "CAMERA", meta["camera_rows"])
    y = block(y, "RENDER", meta["render_rows"])
    # Unified-occlusion note, anchored to the panel bottom. Dropped when the
    # readout blocks would collide with it (the footer states the formula too).
    note_y = py + map_h - 44
    if y <= note_y - 10:
        draw.text((px + 16, note_y), "one integrator, one pass:", font=small_font, fill=muted)
        draw.text(
            (px + 16, py + map_h - 28),
            "T = T_splat x T_lidar x T_terrain",
            font=head_font,
            fill=ink,
        )

    # ---- footer ----------------------------------------------------------
    fy = canvas_h - footer_h + 6
    draw.line((margin, fy - 8, canvas_w - margin, fy - 8), fill=rule, width=1)
    draw.text((margin, fy), meta["footer"], font=small_font, fill=muted)
    return canvas


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render a SPLAT-FUSED sensor + terrain map sheet (y-up world)."
    )
    scene = p.add_argument_group("scene inputs")
    scene.add_argument("--dem", choices=sorted(DEM_PRESETS), default=None,
                       help="bundled real DEM preset (fetched from the web on first run)")
    scene.add_argument("--terrain", default=None, help="path to a GeoTIFF DEM")
    scene.add_argument("--crop-size", type=int, default=0,
                       help="crop real DEMs to this square size from centre (0 = full extent)")
    scene.add_argument("--splats", default=None, help="path to a 3DGS .ply splat cloud")
    scene.add_argument("--splat-count", type=int, default=None,
                       help="splat count for the derived scan block (default 2400; 6000 for hero)")
    scene.add_argument("--splat-lift", type=float, default=0.8,
                       help="height of the derived splat block above the terrain (world units)")
    scene.add_argument("--splat-offset", nargs=3, type=float, default=None,
                       metavar=("X", "Y", "Z"), help="translate a supplied splat cloud (y-up)")
    scene.add_argument("--splat-scale", type=float, default=None,
                       help="uniform scale for a supplied splat cloud")
    scene.add_argument("--pointcloud", default=None, help="path to a COPC/point-store file")
    scene.add_argument(
        "--no-lidar",
        action="store_true",
        help="drop the LiDAR layer (fixture COPC, or the swath derived from the DEM)",
    )

    view = p.add_argument_group("camera (y-up, terrain centred on origin)")
    view.add_argument("--view", choices=sorted(VIEW_PRESETS), default=None,
                      help="framing preset (default: canonical for the fixture, hero for --dem)")
    view.add_argument("--azimuth", type=float, default=None,
                      help="orbit azimuth in degrees (0 = +z, 90 = +x)")
    view.add_argument("--elevation", type=float, default=None,
                      help="orbit elevation in degrees above the horizon")
    view.add_argument("--distance", type=float, default=None,
                      help="orbit distance in world units (default: auto-fit)")
    view.add_argument("--target", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"),
                      help="camera target (default: scene bounds centre)")
    view.add_argument("--fov", type=float, default=None, help="vertical field of view in degrees")
    view.add_argument("--origin", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"),
                      help="raw camera origin (requires --look-at, overrides --view)")
    view.add_argument("--look-at", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"),
                      help="raw camera target (requires --origin)")

    render = p.add_argument_group("render")
    render.add_argument("--width", type=int, default=800, help="map width in pixels")
    render.add_argument("--height", type=int, default=560, help="map height in pixels")
    render.add_argument("--samples", type=int, default=64, help="camera samples per pixel")
    render.add_argument("--seed", type=int, default=7, help="deterministic render seed")
    render.add_argument("--sun-azimuth", type=float, default=None, help="sun azimuth in degrees")
    render.add_argument("--sun-elevation", type=float, default=None, help="sun elevation in degrees")
    render.add_argument("--sun-color", nargs=3, type=float, default=None,
                        metavar=("R", "G", "B"), help="sun colour (linear RGB)")
    render.add_argument("--exposure", type=float, default=1.0)
    render.add_argument("--sky-intensity", type=float, default=None,
                        help="sky/ambient intensity (default 0.35; 0.6 map views; 2.5 hero)")
    render.add_argument("--sky-turbidity", type=float, default=None,
                        help="Hosek-Wilkie turbidity (default 2.5; 4.0 hero)")
    render.add_argument("--sky-ground-albedo", type=float, default=None,
                        help="sky-model ground albedo (default 0.2; 0.5 hero for a hazy horizon band)")
    render.add_argument("--aovs", action="store_true",
                        help="add hit-kind / depth / transmittance inset panels")
    render.add_argument("--no-contours", action="store_true",
                        help="skip the elevation-contour overlay on --dem plan views")
    render.add_argument("--tile", nargs=2, type=int, default=None, metavar=("W", "H"),
                        help="render in seamless tiles of this size (large frames, low memory)")
    render.add_argument("--animate", type=int, default=0, metavar="N",
                        help="also render an N-frame deterministic orbit fly-through GIF")
    render.add_argument("--anim-size", default="640x450",
                        help="fly-through frame size as WxH (default 640x450)")
    render.add_argument("--certificate", action="store_true", help="write a signed render certificate")
    render.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    return p


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if not splat.splat_fusion_available():
        print("error: this wheel was built without the splat-fusion feature.", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_dem = bool(args.dem or args.terrain)
    manifest: Dict[str, Any] = {}
    splat_meta: Dict[str, Any] = {}
    lidar_meta: Optional[Dict[str, Any]] = None
    # Resolved view / lighting mode up front: it also drives the derived
    # splat-block density below, before the camera section runs.
    effective_view = args.view or ("hero" if use_dem else "canonical")
    is_hero = effective_view == "hero"

    # ---- terrain ---------------------------------------------------------
    albedo_map: Optional[np.ndarray] = None
    metres_per_world = 1.0
    if use_dem:
        dem_path = DEM_PRESETS[args.dem] if args.dem else Path(args.terrain)
        if not dem_path.exists():
            print(f"error: DEM not found: {dem_path}", file=sys.stderr)
            return 2
        raw, resolution, metres_per_cell, place, datum = _load_dem(dem_path)
        if args.crop_size > 0 and max(raw.shape) > args.crop_size:
            cs = args.crop_size
            r0 = max(0, (raw.shape[0] - cs) // 2)
            c0 = max(0, (raw.shape[1] - cs) // 2)
            raw = raw[r0 : r0 + cs, c0 : c0 + cs]
        # One world unit = one grid cell = metres_per_cell metres, so
        # elevation (metres) is converted to the SAME units: slopes keep
        # their real steepness. (Squashing elevation to 0..10 instead makes
        # the mountain a 4%-grade pancake and the path-traced relief turns
        # to mush — the texture exists in the DEM, N.L just cannot see it.)
        z_range_m = float(raw.max() - raw.min())
        heights = ((raw - float(raw.min())) / metres_per_cell).astype(np.float32)
        albedo_map = _hypso_map(heights)
        metres_per_world = metres_per_cell
        scene_name = place
        terrain_desc = (
            f"{place} {heights.shape[1]}x{heights.shape[0]} grid, {datum}, "
            f"relief {z_range_m:.0f} m"
        )
        print(
            f"[terrain] {dem_path.name}: {heights.shape[1]}x{heights.shape[0]}, "
            f"relief {z_range_m:.0f} m -> y 0..{heights.max():.0f} cells, {place}"
        )
    else:
        manifest = json.loads((FIXTURE_DIR / "scene.json").read_text())
        heights = np.fromfile(FIXTURE_DIR / manifest["dem_file"], dtype="<f4").reshape(
            manifest["dem_height"], manifest["dem_width"]
        )
        scene_name = "Synthetic survey site"
        terrain_desc = f"fixture ridge {heights.shape[1]}x{heights.shape[0]} @ 1 m grid"
        print(f"[terrain] fixture: {heights.shape[1]}x{heights.shape[0]}, y {heights.min():.2f}..{heights.max():.2f}")

    terrain = splat.FusedTerrain(
        heights=heights,
        spacing=(1.0, 1.0) if use_dem else tuple(manifest["dem_spacing"]),
        exaggeration=1.0,
        albedo=tuple(manifest["terrain_albedo"]) if manifest else (0.55, 0.52, 0.47),
        albedo_map=albedo_map,
        albedo_sampling="bilinear",
    )
    rows, cols = heights.shape
    terrain_bounds = (
        (-(cols - 1) / 2.0, float(heights.min()), -(rows - 1) / 2.0),
        ((cols - 1) / 2.0, float(heights.max()), (rows - 1) / 2.0),
    )

    # ---- splats ----------------------------------------------------------
    if args.splats:
        splat_path = Path(args.splats)
        cloud = splat.load_gaussian_splats(splat_path)
        if args.splat_scale is not None or args.splat_offset is not None:
            cloud = cloud.transformed(
                scale=args.splat_scale if args.splat_scale is not None else 1.0,
                rotation=(1.0, 0.0, 0.0, 0.0),
                translation=tuple(args.splat_offset or (0.0, 0.0, 0.0)),
            )
        splat_desc = f"{splat_path.name} - {cloud.count:,} splats (y-up frame)"
        splat_meta = {"count": cloud.count}
        print(f"[splats] {splat_path.name}: count={cloud.count}, sh_degree={cloud.sh_degree}")
    elif use_dem:
        cloud, splat_meta = _derive_splat_cloud(
            heights,
            (1.0, 1.0),
            albedo_map,
            target_count=args.splat_count or (6000 if is_hero else 2400),
            lift=args.splat_lift,
        )
        splat_desc = (
            f"{int(splat_meta['count']):,} digitized scan block "
            f"(r {splat_meta['radius']:.0f}, sigma {splat_meta['sigma_xz']:.1f}, "
            f"lift {args.splat_lift})"
        )
        print(f"[splats] derived scan block: {splat_desc}")
    else:
        splat_path = FIXTURE_DIR / manifest["splat_file"]
        cloud = splat.load_gaussian_splats(splat_path)
        splat_desc = f"cloud.ply - {cloud.count:,} splats (SH1), canonical placement"
        splat_meta = {"count": cloud.count}
        print(f"[splats] fixture cloud.ply: count={cloud.count}, sh_degree={cloud.sh_degree}")

    # ---- LiDAR -----------------------------------------------------------
    pointcloud = None
    pc_path = Path(args.pointcloud) if args.pointcloud else None
    if pc_path is None and not use_dem and not args.no_lidar:
        pc_path = FIXTURE_DIR / manifest["copc_file"]
    if pc_path is not None:
        origin = (0.0, 0.0, 0.0)
        if manifest and pc_path.resolve() == (FIXTURE_DIR / manifest["copc_file"]).resolve():
            origin = tuple(manifest["copc_origin"])
        try:
            pointcloud = splat.FusedPointCloud(path=pc_path, origin=origin, z_up=True)
            lidar_meta = {"count": int(manifest.get("copc_point_count", 0)) if manifest else 0}
            lidar_desc = f"{pc_path.name} - origin ({origin[0]:.0f}, {origin[1]:.0f}, {origin[2]:.0f})"
            print(f"[pointcloud] {pc_path.name}: origin={origin}")
        except Exception as exc:
            print(f"[pointcloud] failed to load {pc_path.name}: {exc}; continuing without LiDAR")
    if pointcloud is None:
        lidar_desc = None

    # ---- derived LiDAR (DEM runs without an explicit file) ----------------
    point_arrays = None
    if use_dem and pointcloud is None and not args.no_lidar:
        positions, colors, lidar_meta = _derive_lidar_swath(
            heights, (1.0, 1.0), seed=args.seed
        )
        if lidar_meta.get("count", 0) > 0:
            point_arrays = (positions, colors)
            lidar_desc = (
                f"derived scan swath - {lidar_meta['count']:,} pts, "
                f"{lidar_meta['flight_lines']} flight lines (slope-intensity tint)"
            )
            print(f"[pointcloud] derived LiDAR swath: {lidar_desc}")

    # ---- camera ----------------------------------------------------------
    fov = args.fov if args.fov is not None else (manifest.get("fov_y_deg", 45.0) if manifest else 45.0)
    all_bounds = [
        terrain_bounds,
        (tuple(cloud.bounds[0]), tuple(cloud.bounds[1])),
    ]
    mins = [min(b[0][i] for b in all_bounds) for i in range(3)]
    maxs = [max(b[1][i] for b in all_bounds) for i in range(3)]
    scene_bounds = (tuple(mins), tuple(maxs))
    centre = tuple((mins[i] + maxs[i]) / 2.0 for i in range(3))

    orbit_given = any(
        v is not None for v in (args.azimuth, args.elevation, args.distance, args.target)
    )
    view_label = args.view
    if args.origin is not None and args.look_at is not None:
        camera = splat.FusedCamera(
            origin=tuple(args.origin),
            look_at=tuple(args.look_at),
            up=(0.0, 1.0, 0.0),
            fov_y_deg=fov,
        )
        view_label = "custom"
    elif effective_view == "canonical" and not orbit_given and manifest:
        camera = splat.FusedCamera(
            origin=tuple(manifest["cam_origin"]),
            look_at=tuple(manifest["cam_look_at"]),
            up=tuple(manifest["cam_up"]),
            fov_y_deg=fov,
        )
        view_label = "canonical"
    else:
        preset_name = "oblique" if effective_view == "canonical" else effective_view
        preset = VIEW_PRESETS[preset_name]
        if is_hero and args.fov is None:
            fov = 55.0  # wider frame: summit below the top edge, sky above it
        azimuth = args.azimuth if args.azimuth is not None else (preset[0] if preset else 35.0)
        elevation = args.elevation if args.elevation is not None else (preset[1] if preset else 28.0)
        if args.target is not None:
            target = tuple(args.target)
        elif is_hero:
            # Aim slightly above the base: the summit lands ~20% below the
            # top edge instead of crowding it, and foreground compresses.
            target = (centre[0], 0.16 * maxs[1], centre[2])
        else:
            target = centre
        if args.distance is not None:
            distance = args.distance
        elif is_hero:
            distance = _hero_distance(scene_bounds, target, fov, elevation)
        else:
            distance = _fit_distance(
                scene_bounds, target, azimuth, elevation, fov, args.width, args.height
            )
            if use_dem:
                # Map style: crop the terrain into the frame rather than
                # floating it in a sea of sky (falls back when impossible).
                distance = _cover_distance(
                    scene_bounds,
                    target,
                    azimuth,
                    elevation,
                    fov,
                    args.width,
                    args.height,
                    fallback=distance,
                )
        camera = _orbit_camera(target, azimuth, elevation, distance, fov)
        view_label = f"{preset_name} az {azimuth:.0f} el {elevation:.0f}"
    if args.view == "canonical" and not manifest:
        print("[camera] --view canonical needs the fixture scene; falling back to oblique")

    az, el, dist = _camera_angles(camera)
    print(
        f"[camera] {view_label}: origin=({camera.origin[0]:.1f}, {camera.origin[1]:.1f}, "
        f"{camera.origin[2]:.1f}) target=({camera.look_at[0]:.1f}, {camera.look_at[1]:.1f}, "
        f"{camera.look_at[2]:.1f}) fov={camera.fov_y_deg:.0f}"
    )

    # ---- sun -------------------------------------------------------------
    # The hero shot gets warm low side lighting (~90 deg off the camera
    # azimuth) for form-shading on the flanks; map-style views keep the
    # cartographic NW light; the fixture keeps its canonical acceptance sun
    # from scene.json.
    if args.sun_azimuth is not None:
        sun_az = args.sun_azimuth
    elif is_hero:
        sun_az = 125.0
    else:
        # DEM default: cartographic NW light (az 215 = WNW of +x-toward-+z,
        # north = -z), the standard relief-map convention of light from the
        # upper left when north is up.
        sun_az = float(manifest.get("sun_azimuth_deg", 215.0)) if manifest else 215.0
    if args.sun_elevation is not None:
        sun_el = args.sun_elevation
    elif is_hero:
        # Golden hour: a low warm sun reads the relief and warms the Hosek
        # sky into a cream-to-blue gradient instead of a flat pale band.
        sun_el = 8.0
    else:
        sun_el = float(manifest.get("sun_elevation_deg", 45.0)) if manifest else 45.0

    sun_intensity = 6.0 if is_hero else SUN_INTENSITY
    sun_color = (1.0, 0.85, 0.62) if is_hero else SUN_COLOR
    if args.sun_color is not None:
        sun_color = tuple(args.sun_color)

    # ---- render ----------------------------------------------------------
    # Shared lighting kwargs: the still, every fly-through frame and the AOVs
    # all use exactly the same sun/sky configuration.
    sky_i = (
        args.sky_intensity
        if args.sky_intensity is not None
        else (3.0 if is_hero else (0.6 if use_dem else 0.35))
    )
    sky_t = args.sky_turbidity if args.sky_turbidity is not None else (5.0 if is_hero else 2.5)
    sky_ga = (
        args.sky_ground_albedo
        if args.sky_ground_albedo is not None
        else (0.35 if is_hero else 0.2)
    )
    light_kwargs = dict(
        exposure=args.exposure,
        sky_intensity=sky_i,
        sky_turbidity=sky_t,
        sky_ground_albedo=sky_ga,
        sun_azimuth_deg=sun_az,
        sun_elevation_deg=sun_el,
        sun_intensity=sun_intensity,
        sun_color=sun_color,
        # A derived swath gets a pulse-sized radius (0.7 x along-spacing) so
        # neighbouring returns overlap into a scan strip; a file-backed cloud
        # keeps the manifest's radius.
        lidar_radius=(
            float(lidar_meta["along_spacing"] * 0.7)
            if (point_arrays is not None and lidar_meta)
            else float(manifest.get("lidar_radius", 0.2))
        ),
    )
    print(f"[render] {args.width}x{args.height}, {args.samples} spp ...")
    t0 = time.time()
    result = splat.render_fused(
        splats=cloud,
        pointcloud=pointcloud,
        point_positions=None if point_arrays is None else point_arrays[0],
        point_colors=None if point_arrays is None else point_arrays[1],
        terrain=terrain,
        camera=camera,
        width=args.width,
        height=args.height,
        samples=args.samples,
        seed=args.seed,
        return_aovs=True,
        certificate=args.certificate,
        tile=tuple(args.tile) if args.tile else None,
        **light_kwargs,
    )
    render_s = time.time() - t0

    beauty_path = out_dir / "splat_fused_beauty.png"
    f3d.numpy_to_png(beauty_path, result.rgba)

    hits = {
        kind: int((result.hit_kind == kind).sum()) for kind in (0, 3, 4, 5)
    }
    print(f"[render] {render_s:.1f}s -> {beauty_path}")
    print(f"[hits] miss={hits[0]:,} terrain={hits[3]:,} splat={hits[4]:,} lidar={hits[5]:,}")

    # ---- sheet -----------------------------------------------------------
    layers = [k for k in (3, 4, 5) if hits[k] > 0]
    if hits[0] > 0:
        layers.append(0)
    legend_hits = [(k, hits[k]) for k in (3, 4, 5) if hits[k] > 0]
    if hits[0] > 0:
        legend_hits.append((0, hits[0]))

    if use_dem:
        title = f"SPLAT-FUSED - {scene_name}"
        subtitle = (
            f"hypsometric base map + {int(splat_meta.get('count', 0)):,} splats digitized from "
            f"a summit-flank scan block  |  {rows}x{cols} cells  |  seed {args.seed}  |  "
            f"view: {view_label}"
        )
    else:
        title = "SPLAT-FUSED SURVEY SHEET"
        subtitle = (
            f"{scene_name}  |  DEM ridge + {int(splat_meta.get('count', 0)):,} Gaussian splats"
            f"{' + %d LiDAR returns' % lidar_meta['count'] if lidar_meta else ''}"
            f"  |  seed {args.seed}  |  view: {view_label}"
        )

    scene_rows = [
        ("Terrain", terrain_desc),
        ("Splats", splat_desc),
        ("LiDAR", lidar_desc or "not loaded"),
    ]
    # Cartographic annotation for plan views: contours are drawn over the
    # sheet's map copy only — the beauty PNG stays the raw tracer output.
    map_rgba = result.rgba
    if use_dem and not args.no_contours and effective_view in ("top", "high"):
        print("[sheet] drawing elevation contours ...")
        map_rgba, contour_desc = _draw_contours(result.rgba, heights, metres_per_world, camera)
        scene_rows.append(("Contours", contour_desc))
        print(f"[sheet] contours: {contour_desc}")
    if use_dem:
        distance_str = f"{dist:.0f} cells (~{dist * metres_per_world / 1000.0:.2f} km)"
    else:
        distance_str = f"{dist:.0f} m"
    camera_rows = [
        ("View", view_label if len(view_label) <= 26 else view_label[:25] + "..."),
        ("Azimuth", f"{az:.1f} deg"),
        ("Elevation", f"{el:.1f} deg"),
        ("Distance", distance_str),
        ("FOV y", f"{camera.fov_y_deg:.0f} deg"),
        ("Target", f"({camera.look_at[0]:.0f}, {camera.look_at[1]:.0f}, {camera.look_at[2]:.0f})"),
    ]
    render_rows = [
        ("Size", f"{args.width} x {args.height} px"),
        ("Samples", f"{args.samples} spp"),
        ("Seed", f"{args.seed} (deterministic)"),
        ("Sun", f"az {sun_az:.0f} / el {sun_el:.0f}"),
        ("Primitives", f"{int(result.stats.get('logical_primitives', 0)):,}"),
    ]
    meta = {
        "title": title,
        "subtitle": subtitle,
        "hits": legend_hits,
        "scene_rows": scene_rows,
        "camera_rows": camera_rows,
        "render_rows": render_rows,
        "meters_per_world": metres_per_world,
        "extent_x": cols * (1.0 if use_dem else manifest["dem_spacing"][0]),
        "extent_z": rows * (1.0 if use_dem else manifest["dem_spacing"][1]),
        "footer": (
            f"forge3d SPLAT-FUSED  |  one ReSTIR pass: T = T_splat x T_lidar x T_terrain  |  "
            f"{'examples/splat_fused_map.py' if not use_dem else 'real DEM + derived splats'}  |  "
            f"seed {args.seed} reproduces this sheet byte-for-byte"
        ),
    }
    sheet = _compose_sheet(
        map_rgba, result.hit_kind, result.depth, result.transmittance, camera, meta, args.aovs
    )
    sheet_path = out_dir / "splat_fused_map.png"
    sheet.save(sheet_path)
    print(f"[sheet] {sheet_path}")

    # ---- fly-through ------------------------------------------------------
    if args.animate > 0:
        from PIL import Image  # noqa: E402

        try:
            aw, ah = (int(v) for v in args.anim_size.lower().split("x"))
        except ValueError:
            print(f"error: bad --anim-size {args.anim_size!r}, expected WxH", file=sys.stderr)
            return 2
        print(f"[anim] {args.animate} frames orbiting at {aw}x{ah}, {args.samples} spp ...")
        t0 = time.time()
        frames = []
        for i in range(args.animate):
            frame_cam = _orbit_camera(
                camera.look_at, az + 360.0 * i / args.animate, el, dist, camera.fov_y_deg
            )
            frame = splat.render_fused(
                splats=cloud,
                pointcloud=pointcloud,
        point_positions=None if point_arrays is None else point_arrays[0],
        point_colors=None if point_arrays is None else point_arrays[1],
                terrain=terrain,
                camera=frame_cam,
                width=aw,
                height=ah,
                samples=args.samples,
                seed=args.seed,
                return_aovs=False,
                certificate=False,
                **light_kwargs,
            )
            frame_rgba = frame if isinstance(frame, np.ndarray) else frame.rgba
            frames.append(
                Image.fromarray(frame_rgba, "RGBA")
                .convert("RGB")
                .convert("P", palette=Image.ADAPTIVE, colors=256)
            )
            print(f"[anim] frame {i + 1}/{args.animate}  ({time.time() - t0:.0f}s elapsed)")
        gif_path = out_dir / "splat_fused_fly.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=110,
            loop=0,
            optimize=True,
        )
        print(f"[anim] {gif_path} ({gif_path.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s)")

    report = {
        "scene": scene_name,
        "terrain": terrain_desc,
        "splats": splat_desc,
        "lidar": lidar_desc,
        "view": view_label,
        "camera": {
            "origin": list(camera.origin),
            "look_at": list(camera.look_at),
            "up": list(camera.up),
            "fov_y_deg": camera.fov_y_deg,
            "azimuth_deg": az,
            "elevation_deg": el,
            "distance": dist,
        },
        "meters_per_world_unit": metres_per_world,
        "width": args.width,
        "height": args.height,
        "samples": args.samples,
        "seed": args.seed,
        "sun_azimuth_deg": sun_az,
        "sun_elevation_deg": sun_el,
        "render_seconds": round(render_s, 3),
        "hit_counts": {"miss": hits[0], "terrain": hits[3], "splat": hits[4], "lidar": hits[5]},
        "stats": dict(result.stats),
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[report] {report_path}")

    if args.certificate:
        cert = getattr(result, "certificate", None)
        if cert is not None:
            cert_path = out_dir / "certificate.json"
            cert_path.write_text(
                json.dumps(cert, indent=2) if isinstance(cert, (dict, list)) else str(cert)
            )
            print(f"[certificate] {cert_path}")

    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
