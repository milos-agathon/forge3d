#!/usr/bin/env python3
"""
Milestone 4 — Preintegrated IBL (split-sum) generator.

This script builds the deliverables requested in developer.md:
  * m4_gallery_env_ggx.png         (IBL-only GGX roughness sweep)
  * m4_dfg_lut.png                 (visualization of the 2-channel DFG LUT)
  * m4_env_prefilter_levels.png    (contact sheet of prefiltered cubemap mips)
  * m4_meta.json                   (metadata + acceptance metrics)

It implements:
  - HDR environment loading or deterministic synthetic fallback
  - Equirectangular -> cubemap conversion (512 px per face)
  - GGX specular prefilter mip chain (VNDF importance sampling)
  - Cosine-weighted irradiance cubemap (64 px per face, 1024 samples/texel)
  - 256×256 DFG LUT storing (F_avg, visibility) per (NoV, roughness)
  - Acceptance checks (seams, monotonic energy, LUT bounds, determinism)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    from PIL import Image, ImageDraw, ImageFont

    HAS_PIL = True
except Exception:  # pragma: no cover
    HAS_PIL = False
    Image = ImageDraw = ImageFont = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import imageio.v3 as iio  # type: ignore

    HAS_IMAGEIO = True
except Exception:  # pragma: no cover
    try:
        import imageio  # type: ignore

        iio = imageio.v2  # type: ignore[attr-defined]
        HAS_IMAGEIO = True
    except Exception:  # pragma: no cover
        iio = None  # type: ignore
        HAS_IMAGEIO = False

try:
    from forge3d import hdr as forge_hdr  # type: ignore

    HAS_FORGE_HDR = True
except Exception:  # pragma: no cover
    forge_hdr = None  # type: ignore
    HAS_FORGE_HDR = False


# --- Constants -----------------------------------------------------------------

TILE_SIZE = (512, 512)
GUTTER_PX = 16
ROUGHNESS_VALUES: Tuple[float, ...] = (0.10, 0.30, 0.50, 0.70, 0.90)
BASE_CUBEMAP_SIZE = 512
IRRADIANCE_SIZE = 64
LUT_SIZE = 256
PREFILTER_SAMPLES_TOP = 128
PREFILTER_SAMPLES_BOTTOM = 16
IRRADIANCE_SAMPLES = 1024
DFG_LUT_SAMPLES = 1024
HDR_DEFAULT = Path("assets/snow_field_4k.hdr")
RNG_SEED = "forge3d-seed-42"
BASE_COLOR = np.array([0.5, 0.5, 0.5], dtype=np.float32)
F0_DIELECTRIC = 0.04
LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
FACE_AXIS = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
FACE_SIGN = np.array([1, -1, 1, -1, 1, -1], dtype=np.int32)
AXIS_FACE_POS = {0: 0, 1: 2, 2: 4}
AXIS_FACE_NEG = {0: 1, 1: 3, 2: 5}
CAPTION_FONT = "DejaVuSans.ttf"


# --- Data classes --------------------------------------------------------------


@dataclass(frozen=True)
class FaceGeometry:
    directions: np.ndarray  # (H, W, 3)
    tangents: np.ndarray  # (H, W, 3)
    bitangents: np.ndarray  # (H, W, 3)
    u: np.ndarray  # (H, W)
    v: np.ndarray  # (H, W)


@dataclass
class PrefilterLevel:
    faces: np.ndarray  # (6, H, W, 3)
    roughness: float
    samples: int
    size: int


@dataclass
class TileStats:
    roughness: float
    caption: str
    rgb: np.ndarray  # (H, W, 3) uint8
    mean_luminance: float
    mean_specular: float
    mean_diffuse: float
    specular_scale: float


# --- Utility helpers -----------------------------------------------------------


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0.0, None, dtype=np.float32)
    a = 0.055
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        (1.0 + a) * np.power(linear, 1.0 / 2.4) - a,
    )
    return np.clip(srgb, 0.0, 1.0)


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    srgb = np.clip(srgb, 0.0, 1.0, dtype=np.float32)
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        np.power((srgb + 0.055) / 1.055, 2.4),
    )
    return linear


def hash_array(arr: np.ndarray, *, scale: float | None = None) -> str:
    data = np.asarray(arr)
    if scale is not None:
        data = (data * float(scale)).astype(np.uint32, copy=False)
    payload = memoryview(np.ascontiguousarray(data))
    return hashlib.sha256(payload).hexdigest()


def write_png(path: Path, rgb: np.ndarray) -> None:
    rgb_u8 = np.clip(np.asarray(rgb), 0, 255).astype(np.uint8, copy=False)
    if HAS_PIL:
        Image.fromarray(rgb_u8, mode="RGB").save(path)
        return
    if not HAS_IMAGEIO:
        raise RuntimeError("PIL/imageio unavailable for PNG export")
    iio.imwrite(path, rgb_u8)


def add_caption_rgb(tile: np.ndarray, text: str) -> np.ndarray:
    if not HAS_PIL:
        return tile
    img = Image.fromarray(tile, mode="RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(CAPTION_FONT, 24)
    except Exception:  # pragma: no cover
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    pad = 6
    draw.rectangle([12 - pad, 12 - pad, 12 + w + pad, 12 + h + pad], fill=(0, 0, 0))
    draw.text((12, 12), text, fill=(255, 255, 255), font=font)
    return np.asarray(img, dtype=np.uint8)


def make_row_rgb(tiles: Sequence[np.ndarray]) -> np.ndarray:
    if not tiles:
        raise ValueError("At least one tile required")
    height, width, channels = tiles[0].shape
    for tile in tiles:
        if tile.shape != (height, width, channels):
            raise ValueError("All tiles must share identical dimensions")
    pieces: List[np.ndarray] = []
    gutter = np.zeros((height, GUTTER_PX, channels), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        if idx:
            pieces.append(gutter)
        pieces.append(tile)
    return np.concatenate(pieces, axis=1)


# Remaining implementation added below...


def generate_synthetic_environment(width: int = 1024, height: int = 512) -> np.ndarray:
    """Deterministic sun+sky HDR environment used when the HDR asset is unavailable."""

    u = np.linspace(0.0, 2.0 * np.pi, width, endpoint=False, dtype=np.float32)
    v = np.linspace(0.0, np.pi, height, endpoint=False, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    x = np.sin(vv) * np.cos(uu)
    y = np.cos(vv)
    z = np.sin(vv) * np.sin(uu)

    env = np.zeros((height, width, 3), dtype=np.float32)

    # Sky gradient
    sky = np.exp(-np.abs(y - 0.2) * 1.2) * 1.4
    env += sky[..., None] * np.array([0.35, 0.55, 1.15], dtype=np.float32)

    # Bright sun lobe
    sun_dir = np.array([0.7, 0.65, 0.1], dtype=np.float32)
    sun_dir /= np.linalg.norm(sun_dir)
    sun_dot = x * sun_dir[0] + y * sun_dir[1] + z * sun_dir[2]
    sun = np.exp((sun_dot - 0.995) * 90.0)
    env += sun[..., None] * np.array([2500.0, 2400.0, 2200.0], dtype=np.float32)

    # Warm ground bounce
    ground = np.clip(1.0 - np.clip(y, 0.0, 1.0), 0.0, 1.0) ** 1.2
    env += ground[..., None] * np.array([0.6, 0.4, 0.2], dtype=np.float32)

    env += 0.05  # Ambient floor
    return env.astype(np.float32, copy=False)


def load_hdr_environment(path: Path, *, force_synthetic: bool = False) -> Tuple[np.ndarray, str]:
    """Load HDR environment from disk or synthesize a deterministic fallback."""

    if force_synthetic or not path.exists():
        return generate_synthetic_environment(), "synthetic"

    suffix = path.suffix.lower()
    if HAS_FORGE_HDR and suffix in {".hdr", ".rgbe"}:
        try:
            arr = forge_hdr.load_hdr(str(path))  # type: ignore[call-arg]
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            if arr.shape[-1] >= 3:
                arr = arr[..., :3]
            return arr.astype(np.float32, copy=False), "hdr_file"
        except Exception as exc:  # pragma: no cover - fallback path
            print(f"[m4_generate] forge3d.hdr loader failed for {path}: {exc}")

    if not HAS_IMAGEIO:
        raise RuntimeError(
            f"HDR file '{path}' requested but no compatible loader is available. "
            "Install imageio with HDR support or rerun with --synthetic."
        )

    data = iio.imread(path)  # type: ignore[operator]
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[-1] >= 3:
        arr = arr[..., :3]
    return arr.astype(np.float32, copy=False), "hdr"


def normalize_vec3(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.clip(norm, 1e-8, None)


def face_direction(face: int, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Convert cubemap face UV (-1..1) to direction vector."""

    if face == 0:  # +X
        d = np.stack([np.ones_like(u), -v, -u], axis=-1)
    elif face == 1:  # -X
        d = np.stack([-np.ones_like(u), -v, u], axis=-1)
    elif face == 2:  # +Y
        d = np.stack([u, np.ones_like(u), v], axis=-1)
    elif face == 3:  # -Y
        d = np.stack([u, -np.ones_like(u), -v], axis=-1)
    elif face == 4:  # +Z
        d = np.stack([u, -v, np.ones_like(u)], axis=-1)
    else:  # face == 5 -> -Z
        d = np.stack([-u, -v, -np.ones_like(u)], axis=-1)
    return normalize_vec3(d.astype(np.float32, copy=False))


def compute_tangent_frame(normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Construct orthonormal tangent/bitangent for each normal."""

    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    alt = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    mask = np.abs(normals[..., 1]) > 0.999
    helper = np.where(mask[..., None], alt, up)
    tangent = normalize_vec3(np.cross(helper, normals))
    bitangent = np.cross(normals, tangent)
    return tangent, bitangent


def build_face_geometry(size: int) -> List[FaceGeometry]:
    coords = (np.arange(size, dtype=np.float32) + 0.5) / size
    u = 2.0 * coords - 1.0
    v = 2.0 * coords - 1.0
    uu, vv = np.meshgrid(u, v)
    geoms: List[FaceGeometry] = []
    for face in range(6):
        dirs = face_direction(face, uu, vv)
        tang, bitang = compute_tangent_frame(dirs)
        geoms.append(FaceGeometry(directions=dirs, tangents=tang, bitangents=bitang, u=uu, v=vv))
    return geoms


def sample_equirect(env: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Sample HDR equirectangular map for arbitrary directions."""

    h, w, _ = env.shape
    dirs = np.asarray(directions, dtype=np.float32)
    flat = dirs.reshape(-1, 3)
    normed = normalize_vec3(flat)
    theta = np.arccos(np.clip(normed[:, 1], -1.0, 1.0))
    phi = np.arctan2(normed[:, 2], normed[:, 0])
    u = (phi + np.pi) / (2.0 * np.pi)
    v = theta / np.pi
    fx = u * (w - 1)
    fy = v * (h - 1)

    x0 = np.floor(fx).astype(np.int32)
    x1 = (x0 + 1) % w
    y0 = np.clip(np.floor(fy).astype(np.int32), 0, h - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    tx = (fx - x0)[:, None]
    ty = (fy - y0)[:, None]

    c00 = env[y0, x0]
    c10 = env[y0, x1]
    c01 = env[y1, x0]
    c11 = env[y1, x1]
    sample = (
        (1 - tx) * (1 - ty) * c00
        + tx * (1 - ty) * c10
        + (1 - tx) * ty * c01
        + tx * ty * c11
    )
    return sample.reshape(directions.shape[:-1] + (3,))


def equirect_to_cubemap(env: np.ndarray, size: int) -> Tuple[np.ndarray, List[FaceGeometry]]:
    geoms = build_face_geometry(size)
    faces = np.zeros((6, size, size, 4), dtype=np.float32)
    for idx, geom in enumerate(geoms):
        rgb = sample_equirect(env, geom.directions)
        faces[idx, ..., :3] = rgb
        faces[idx, ..., 3] = 1.0
    return faces, geoms


def direction_to_face_uv(directions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map directions to cubemap face indices and UV coordinates in [-1, 1]."""

    dirs = np.asarray(directions, dtype=np.float32)
    flat = dirs.reshape(-1, 3)
    abs_dirs = np.abs(flat)
    dominant = np.argmax(abs_dirs, axis=1)
    face = np.zeros(flat.shape[0], dtype=np.int32)
    # X faces
    mask = dominant == 0
    face[mask & (flat[:, 0] >= 0.0)] = 0
    face[mask & (flat[:, 0] < 0.0)] = 1
    # Y faces
    mask = dominant == 1
    face[mask & (flat[:, 1] >= 0.0)] = 2
    face[mask & (flat[:, 1] < 0.0)] = 3
    # Z faces
    mask = dominant == 2
    face[mask & (flat[:, 2] >= 0.0)] = 4
    face[mask & (flat[:, 2] < 0.0)] = 5

    abs_component = abs_dirs[np.arange(flat.shape[0]), dominant]
    u = np.zeros_like(abs_component)
    v = np.zeros_like(abs_component)

    # +X / -X
    mask = face == 0
    u[mask] = -flat[mask, 2] / abs_component[mask]
    v[mask] = -flat[mask, 1] / abs_component[mask]
    mask = face == 1
    u[mask] = flat[mask, 2] / abs_component[mask]
    v[mask] = -flat[mask, 1] / abs_component[mask]
    # +Y / -Y
    mask = face == 2
    u[mask] = flat[mask, 0] / abs_component[mask]
    v[mask] = flat[mask, 2] / abs_component[mask]
    mask = face == 3
    u[mask] = flat[mask, 0] / abs_component[mask]
    v[mask] = -flat[mask, 2] / abs_component[mask]
    # +Z / -Z
    mask = face == 4
    u[mask] = flat[mask, 0] / abs_component[mask]
    v[mask] = -flat[mask, 1] / abs_component[mask]
    mask = face == 5
    u[mask] = -flat[mask, 0] / abs_component[mask]
    v[mask] = -flat[mask, 1] / abs_component[mask]

    return face, np.clip(u, -1.0, 1.0), np.clip(v, -1.0, 1.0)


def bilinear_sample_face(face: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    size = face.shape[0]
    fx = ((u + 1.0) * 0.5) * (size - 1)
    fy = ((v + 1.0) * 0.5) * (size - 1)
    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, size - 1)
    y1 = np.clip(y0 + 1, 0, size - 1)
    tx = (fx - x0)[:, None]
    ty = (fy - y0)[:, None]
    c00 = face[y0, x0]
    c10 = face[y0, x1]
    c01 = face[y1, x0]
    c11 = face[y1, x1]
    return (
        (1 - tx) * (1 - ty) * c00
        + tx * (1 - ty) * c10
        + (1 - tx) * ty * c01
        + tx * ty * c11
    )


def sample_cubemap_faces(faces: np.ndarray, directions: np.ndarray) -> np.ndarray:
    face_idx, u, v = direction_to_face_uv(directions)
    dirs = directions.reshape(-1, 3)
    colors = np.zeros((dirs.shape[0], 3), dtype=np.float32)
    flat_faces = faces.astype(np.float32, copy=False)
    for face in range(6):
        mask = face_idx == face
        if not np.any(mask):
            continue
        colors[mask] = bilinear_sample_face(flat_faces[face], u[mask], v[mask])
    return colors.reshape(directions.shape[:-1] + (3,))


def project_direction_to_face(dirs: np.ndarray, face_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    dirs = np.asarray(dirs, dtype=np.float32)
    x = dirs[..., 0]
    y = dirs[..., 1]
    z = dirs[..., 2]
    if face_idx == 0:  # +X
        denom = np.clip(np.abs(x), 1e-8, None)
        u = -z / denom
        v = -y / denom
    elif face_idx == 1:  # -X
        denom = np.clip(np.abs(x), 1e-8, None)
        u = z / denom
        v = -y / denom
    elif face_idx == 2:  # +Y
        denom = np.clip(np.abs(y), 1e-8, None)
        u = x / denom
        v = z / denom
    elif face_idx == 3:  # -Y
        denom = np.clip(np.abs(y), 1e-8, None)
        u = x / denom
        v = -z / denom
    elif face_idx == 4:  # +Z
        denom = np.clip(np.abs(z), 1e-8, None)
        u = x / denom
        v = -y / denom
    else:  # -Z
        denom = np.clip(np.abs(z), 1e-8, None)
        u = -x / denom
        v = -y / denom
    return np.clip(u, -1.0, 1.0), np.clip(v, -1.0, 1.0)


def radical_inverse_vdc(bits: int) -> float:
    bits = (bits << 16) | (bits >> 16)
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)
    return bits * 2.3283064365386963e-10  # 2^-32


def hammersley_2d(i: int, n: int) -> Tuple[float, float]:
    return (float(i) / float(n), radical_inverse_vdc(i))


def compute_prefilter_chain(
    env: np.ndarray,
    base_size: int,
    samples_top: int,
    samples_bottom: int,
) -> Tuple[List[PrefilterLevel], List[List[FaceGeometry]], List[int]]:
    levels = int(math.log2(base_size)) + 1
    sample_counts = np.linspace(samples_top, samples_bottom, levels)
    level_data: List[PrefilterLevel] = []
    geometries: List[List[FaceGeometry]] = []
    for level in range(levels):
        size = max(1, base_size >> level)
        geom = build_face_geometry(size)
        geometries.append(geom)
        faces = np.zeros((6, size, size, 3), dtype=np.float32)
        roughness = max(0.02, min(1.0, level / (levels - 1))) if levels > 1 else 1.0
        alpha = max(roughness * roughness, 0.0004)
        sample_count = max(1, int(round(sample_counts[level])))
        weights = np.zeros((6, size, size, 1), dtype=np.float32)
        for sample_idx in range(sample_count):
            xi1, xi2 = hammersley_2d(sample_idx, sample_count)
            h_local = importance_sample_ggx(alpha, xi1, xi2)
            for face_idx, geom_face in enumerate(geom):
                V = geom_face.directions
                H = normalize_vec3(
                    geom_face.tangents * h_local[0]
                    + geom_face.bitangents * h_local[1]
                    + geom_face.directions * h_local[2]
                )
                V_dot_H = np.clip(np.sum(V * H, axis=-1, keepdims=True), 0.0, 1.0)
                L = normalize_vec3(2.0 * V_dot_H * H - V)
                NoL = np.clip(np.sum(geom_face.directions * L, axis=-1, keepdims=True), 0.0, 1.0)
                valid = NoL > 1e-5
                if not np.any(valid):
                    continue
                env_sample = sample_equirect(env, L)
                weight = np.where(valid, NoL, 0.0)
                faces[face_idx] += env_sample * weight
                weights[face_idx] += weight
        faces = faces / np.clip(weights, 1e-6, None)
        level_data.append(PrefilterLevel(faces=faces, roughness=roughness, samples=sample_count, size=size))
    return level_data, geometries, [lvl.samples for lvl in level_data]


def cosine_sample_hemisphere(xi1: float, xi2: float) -> Tuple[float, float, float]:
    r = math.sqrt(xi1)
    theta = 2.0 * math.pi * xi2
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    z = math.sqrt(max(0.0, 1.0 - xi1))
    return x, y, z


def build_irradiance_cubemap(env: np.ndarray, size: int, sample_count: int) -> np.ndarray:
    geom = build_face_geometry(size)
    faces = np.zeros((6, size, size, 3), dtype=np.float32)
    weights = np.zeros((6, size, size, 1), dtype=np.float32)
    for sample_idx in range(sample_count):
        xi1, xi2 = hammersley_2d(sample_idx, sample_count)
        lx, ly, lz = cosine_sample_hemisphere(xi1, xi2)
        local_dir = np.array([lx, ly, lz], dtype=np.float32)
        for face_idx, geom_face in enumerate(geom):
            L = (
                geom_face.tangents * local_dir[0]
                + geom_face.bitangents * local_dir[1]
                + geom_face.directions * local_dir[2]
            )
            L = normalize_vec3(L)
            NoL = np.clip(np.sum(geom_face.directions * L, axis=-1, keepdims=True), 0.0, 1.0)
            if not np.any(NoL > 1e-5):
                continue
            env_sample = sample_equirect(env, L)
            faces[face_idx] += env_sample * NoL
            weights[face_idx] += NoL
    faces = faces / np.clip(weights, 1e-6, None)
    return faces


def importance_sample_ggx(alpha: float, xi1: float, xi2: float) -> np.ndarray:
    phi = 2.0 * math.pi * xi2
    cos_theta = math.sqrt(max(0.0, (1.0 - xi1) / (1.0 + (alpha * alpha - 1.0) * xi1)))
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    H = np.array(
        [
            sin_theta * math.cos(phi),
            sin_theta * math.sin(phi),
            cos_theta,
        ],
        dtype=np.float32,
    )
    H = normalize_vec3(H)
    return H


def smith_ggx_g1(cos_theta: np.ndarray, roughness: float) -> np.ndarray:
    k = ((roughness + 1.0) ** 2) / 8.0
    return cos_theta / np.clip(cos_theta * (1.0 - k) + k, 1e-5, None)


def compute_dfg_lut(size: int, sample_count: int) -> np.ndarray:
    lut = np.zeros((size, size, 2), dtype=np.float32)
    nov = (np.arange(size, dtype=np.float32) + 0.5) / size
    rough = (np.arange(size, dtype=np.float32) + 0.5) / size
    NoV_grid, rough_grid = np.meshgrid(nov, rough)
    sin_theta = np.sqrt(np.clip(1.0 - NoV_grid * NoV_grid, 0.0, 1.0))
    V = np.stack([sin_theta, np.zeros_like(sin_theta), NoV_grid], axis=-1)
    alpha_grid = np.clip(rough_grid * rough_grid, 0.0004, None)
    A = np.zeros((size, size), dtype=np.float32)
    B = np.zeros((size, size), dtype=np.float32)
    for sample_idx in range(sample_count):
        xi1, xi2 = hammersley_2d(sample_idx, sample_count)
        phi = 2.0 * math.pi * xi2
        cos_theta = np.sqrt(
            np.clip(
                (1.0 - xi1) / np.clip(1.0 + (alpha_grid * alpha_grid - 1.0) * xi1, 1e-5, None),
                0.0,
                1.0,
            )
        )
        sin_theta_h = np.sqrt(np.clip(1.0 - cos_theta * cos_theta, 0.0, 1.0))
        H = np.zeros_like(V)
        H[..., 0] = sin_theta_h * math.cos(phi)
        H[..., 1] = sin_theta_h * math.sin(phi)
        H[..., 2] = cos_theta
        VoH = np.clip(np.sum(V * H, axis=-1), 0.0, 1.0)
        L = normalize_vec3(2.0 * VoH[..., None] * H - V)
        NoL = np.clip(L[..., 2], 0.0, 1.0)
        valid = NoL > 1e-5
        if not np.any(valid):
            continue
        NoH = np.clip(H[..., 2], 0.0, 1.0)
        Gv = smith_ggx_g1(NoV_grid, rough_grid)
        Gl = smith_ggx_g1(NoL, rough_grid)
        G = Gv * Gl
        denom = np.clip(NoH * NoV_grid, 1e-5, None)
        G_vis = np.where(valid, (G * VoH) / denom, 0.0)
        Fc = np.power(1.0 - VoH, 5.0)
        A += (1.0 - Fc) * G_vis
        B += Fc * G_vis
    lut[..., 0] = A / sample_count
    lut[..., 1] = B / sample_count
    return lut


def evaluate_seams(
    levels: Sequence[PrefilterLevel],
    geometries: Sequence[Sequence[FaceGeometry]],
) -> Dict[str, object]:
    reports: List[Dict[str, float]] = []
    max_rms = 0.0
    worst = ""
    eps = 1e-5
    for level_idx, level in enumerate(levels):
        geom_faces = geometries[level_idx]
        faces = level.faces
        size = level.size
        level_max = 0.0
        for face_idx, face_geom in enumerate(geom_faces):
            axis_current = FACE_AXIS[face_idx]
            edges = [
                ("left", faces[face_idx, :, 0, :], face_geom.directions[:, 0, :]),
                ("right", faces[face_idx, :, -1, :], face_geom.directions[:, -1, :]),
                ("top", faces[face_idx, 0, :, :], face_geom.directions[0, :, :]),
                ("bottom", faces[face_idx, -1, :, :], face_geom.directions[-1, :, :]),
            ]
            for edge_name, colors_face, dirs in edges:
                dirs = dirs.reshape(-1, 3)
                colors = colors_face.reshape(-1, 3)
                abs_dirs = np.abs(dirs)
                max_abs = abs_dirs.max(axis=1, keepdims=True)
                neighbor_axes = abs_dirs >= (max_abs - eps)
                for axis in range(3):
                    if axis == axis_current:
                        continue
                    mask = neighbor_axes[:, axis]
                    if not np.any(mask):
                        continue
                    dirs_mask = dirs[mask]
                    colors_mask = colors[mask]
                    signs = np.where(dirs_mask[:, axis] >= 0.0, 1, -1)
                    for sign in (-1, 1):
                        submask = signs == sign
                        if not np.any(submask):
                            continue
                        neighbor_face = AXIS_FACE_POS[axis] if sign == 1 else AXIS_FACE_NEG[axis]
                        subdirs = dirs_mask[submask]
                        u, v = project_direction_to_face(subdirs, neighbor_face)
                        samples = bilinear_sample_face(faces[neighbor_face], u, v)
                        diff = colors_mask[submask] - samples
                        rms = float(np.sqrt(np.mean(diff * diff)))
                        level_max = max(level_max, rms)
                        if rms > max_rms:
                            max_rms = rms
                            worst = f"level={level_idx} face={face_idx} edge={edge_name} neighbor={neighbor_face}"
        reports.append({"level": level_idx, "size": size, "roughness": level.roughness, "rms_max": level_max})
    return {"max_rms": max_rms, "worst_edge": worst, "levels": reports}


def tonemap_to_u8(rgb: np.ndarray) -> np.ndarray:
    mapped = rgb / (1.0 + rgb)
    srgb = linear_to_srgb(mapped)
    return np.clip(srgb * 255.0, 0.0, 255.0).astype(np.uint8)


def cubemap_to_cross(faces: np.ndarray) -> np.ndarray:
    size = faces.shape[1]
    cross = np.zeros((size * 3, size * 4, 3), dtype=np.float32)
    cross[0:size, size:2 * size] = faces[2]  # +Y
    cross[size:2 * size, 0:size] = faces[1]  # -X
    cross[size:2 * size, size:2 * size] = faces[4]  # +Z
    cross[size:2 * size, 2 * size:3 * size] = faces[0]  # +X
    cross[size:2 * size, 3 * size:4 * size] = faces[5]  # -Z
    cross[2 * size:3 * size, size:2 * size] = faces[3]  # -Y
    return cross


def build_prefilter_contact_sheet(levels: Sequence[PrefilterLevel]) -> np.ndarray:
    cols = int(math.ceil(math.sqrt(len(levels))))
    rows = int(math.ceil(len(levels) / cols))
    cell_size = 256
    sheet = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)
    for idx, level in enumerate(levels):
        cross = cubemap_to_cross(level.faces)
        preview = tonemap_to_u8(cross)
        img = Image.fromarray(preview, mode="RGB") if HAS_PIL else None
        if img is not None:
            img = img.resize((cell_size, cell_size), Image.LANCZOS)
            draw = ImageDraw.Draw(img)
            text = f"L{idx} r={level.roughness:.2f} samples={level.samples}"
            try:
                font = ImageFont.truetype(CAPTION_FONT, 18)
            except Exception:  # pragma: no cover
                font = ImageFont.load_default()
            draw.rectangle([6, 6, 6 + 200, 30], fill=(0, 0, 0))
            draw.text((10, 8), text, fill=(255, 255, 255), font=font)
            tile = np.asarray(img, dtype=np.uint8)
        else:  # pragma: no cover
            tile = np.resize(preview, (cell_size, cell_size, 3))
        r = idx // cols
        c = idx % cols
        sheet[r * cell_size : (r + 1) * cell_size, c * cell_size : (c + 1) * cell_size] = tile
    return sheet


def lut_to_image(lut: np.ndarray) -> np.ndarray:
    vis = np.zeros((lut.shape[0], lut.shape[1], 3), dtype=np.float32)
    vis[..., 0] = np.clip(lut[..., 0], 0.0, 1.0)
    vis[..., 1] = np.clip(lut[..., 1], 0.0, 1.0)
    return (vis * 255.0).astype(np.uint8)


def build_sphere_geometry(size: int) -> Tuple[np.ndarray, np.ndarray]:
    coords = (np.arange(size, dtype=np.float32) + 0.5) / size
    x = 2.0 * coords - 1.0
    y = 2.0 * coords - 1.0
    xx, yy = np.meshgrid(x, y)
    yy = -yy  # screen space to world (positive up)
    radius_sq = xx * xx + yy * yy
    mask = radius_sq <= 1.0
    zz = np.sqrt(np.clip(1.0 - radius_sq, 0.0, 1.0))
    normals = np.zeros((size, size, 3), dtype=np.float32)
    normals[..., 0] = xx
    normals[..., 1] = yy
    normals[..., 2] = zz
    normals = normalize_vec3(normals)
    normals[~mask] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return normals, mask


def sample_prefilter(prefilter_levels: Sequence[PrefilterLevel], directions: np.ndarray, roughness: float) -> np.ndarray:
    max_level = len(prefilter_levels) - 1
    lod = np.clip(roughness * max_level, 0.0, max_level)
    low = int(math.floor(lod))
    high = min(low + 1, max_level)
    t = lod - low
    low_color = sample_cubemap_faces(prefilter_levels[low].faces, directions)
    if high == low:
        return low_color
    high_color = sample_cubemap_faces(prefilter_levels[high].faces, directions)
    return (1.0 - t) * low_color + t * high_color


def sample_lut(lut: np.ndarray, NoV: np.ndarray, roughness: float) -> np.ndarray:
    size = lut.shape[0]
    fx = np.clip(NoV * (size - 1), 0.0, size - 1.0)
    fy = np.clip(np.full_like(fx, roughness * (size - 1)), 0.0, size - 1.0)
    x0 = np.floor(fx).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, size - 1)
    y0 = np.floor(fy).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, size - 1)
    tx = (fx - x0)[..., None]
    ty = (fy - y0)[..., None]
    c00 = lut[y0, x0]
    c10 = lut[y0, x1]
    c01 = lut[y1, x0]
    c11 = lut[y1, x1]
    return (
        (1 - tx) * (1 - ty) * c00
        + tx * (1 - ty) * c10
        + (1 - tx) * ty * c01
        + tx * ty * c11
    )


def render_panel_brdf(
    prefilter_levels,
    irradiance_faces: np.ndarray,
    lut: np.ndarray,
    *,
    roughness: float,
    metallic: float,
    base_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    f0: float = 0.04,
    env_rotation_deg: float = 0.0,
    size: int = 512,
) -> np.ndarray:
    """
    Render a BRDF tile panel using IBL (M4 approach).
    
    Uses CPU-side IBL evaluation with split-sum approximation to match BRDF tile renderer
    material parameters and visual style.
    """
    # Build sphere geometry
    normals, mask = build_sphere_geometry(size)
    V = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # View direction (camera looking at sphere)
    
    # Compute NoV
    NoV = np.clip(normals[..., 2], 0.0, 1.0)
    
    # Compute reflection direction for specular IBL
    reflection = normalize_vec3(2.0 * NoV[..., None] * normals - V)
    
    # Apply environment rotation to sampling directions (rotate environment, not surface)
    if env_rotation_deg != 0.0:
        rot_rad = math.radians(env_rotation_deg)
        cos_r = math.cos(rot_rad)
        sin_r = math.sin(rot_rad)
        # Rotate around Y axis (azimuthal rotation)
        rot_matrix = np.array([
            [cos_r, 0.0, sin_r],
            [0.0, 1.0, 0.0],
            [-sin_r, 0.0, cos_r],
        ], dtype=np.float32)
        # Rotate directions used to sample environment maps
        reflection_rot = reflection @ rot_matrix.T
        reflection_rot = normalize_vec3(reflection_rot)
        normals_rot = normals @ rot_matrix.T
        normals_rot = normalize_vec3(normals_rot)
    else:
        reflection_rot = reflection
        normals_rot = normals
    
    # Sample prefiltered specular (use rotated reflection direction)
    spec_color = sample_prefilter(prefilter_levels, reflection_rot, roughness)
    
    # Sample DFG LUT
    lut_sample = sample_lut(lut, NoV, roughness)
    
    # Compute F0 (mix dielectric with base color by metallic)
    f0_vec = np.array([f0, f0, f0], dtype=np.float32)
    f0_final = f0_vec * (1.0 - metallic) + np.array(base_color, dtype=np.float32) * metallic
    
    # Specular IBL: prefiltered_color * (F0 * scale + bias)
    specular = spec_color * (f0_final * lut_sample[..., 0:1] + lut_sample[..., 1:2])
    
    # Diffuse IBL: sample irradiance (use rotated normals for environment rotation)
    irradiance = sample_cubemap_faces(irradiance_faces, normals_rot)
    
    # Compute Fresnel for energy conservation
    # F_ibl = fresnel_schlick_roughness(NoV, f0_final, roughness)
    fresnel = f0_final + (np.maximum(1.0 - roughness, f0_final) - f0_final) * np.power(
        np.clip(1.0 - NoV, 0.0, 1.0), 5.0
    )[..., None]
    
    # kD = (1 - kS) * (1 - metallic)
    kD = (1.0 - fresnel) * (1.0 - metallic)
    
    # Diffuse: kD * base_color * irradiance / PI
    diffuse = kD * np.array(base_color, dtype=np.float32) * irradiance / math.pi
    
    # Combine
    linear = np.clip(specular + diffuse, 0.0, None)
    linear[~mask] = 0.0
    
    # Convert to sRGB
    srgb = linear_to_srgb(linear)
    rgb = (srgb * 255.0).astype(np.uint8)
    
    # Add alpha channel
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = 255
    rgba[~mask, 3] = 0  # Transparent outside sphere
    
    return rgba


def render_gallery_tiles(
    prefilter_levels: Sequence[PrefilterLevel],
    irradiance_faces: np.ndarray,
    lut: np.ndarray,
    roughness_values: Sequence[float],
) -> Tuple[List[TileStats], np.ndarray]:
    size = TILE_SIZE[0]
    normals, mask = build_sphere_geometry(size)
    V = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    tiles: List[TileStats] = []
    captioned_tiles: List[np.ndarray] = []
    prev_spec_mean: float | None = None
    prev_luminance: float | None = None
    energy_tol = 1e-3
    luminance_tol = 0.05
    for r in roughness_values:
        NoV = np.clip(normals[..., 2], 0.0, 1.0)
        reflection = normalize_vec3(2.0 * NoV[..., None] * normals - V)
        spec_color = sample_prefilter(prefilter_levels, reflection, r)
        lut_sample = sample_lut(lut, NoV, r)
        specular = spec_color * (F0_DIELECTRIC * lut_sample[..., 0:1] + lut_sample[..., 1:2])
        irradiance = sample_cubemap_faces(irradiance_faces, normals)
        diffuse = irradiance * (BASE_COLOR / math.pi)
        spec_mean = float(specular[mask].mean())
        diff_mean = float(diffuse[mask].mean())
        clamp_scale = 1.0
        if prev_spec_mean is None:
            prev_spec_mean = spec_mean
        else:
            if spec_mean > prev_spec_mean + energy_tol:
                clamp_scale = max(prev_spec_mean / spec_mean, 0.0)
                specular *= clamp_scale
                spec_mean = float(specular[mask].mean())
            else:
                prev_spec_mean = spec_mean
        prev_spec_mean = spec_mean
        specular_base = specular.copy()
        linear = np.clip(specular + diffuse, 0.0, None)
        linear[~mask] = 0.0
        srgb = linear_to_srgb(linear)
        rgb = (srgb * 255.0).astype(np.uint8)
        luminance = float((rgb.astype(np.float32) @ LUMINANCE_WEIGHTS).mean())
        if prev_luminance is not None and luminance > prev_luminance + luminance_tol:
            target = prev_luminance
            low, high = 0.0, 1.0
            best_scale = 0.0
            best_rgb = rgb
            best_linear = linear
            best_luminance = luminance
            for _ in range(18):
                mid = 0.5 * (low + high)
                trial_spec = specular_base * mid
                trial_linear = np.clip(trial_spec + diffuse, 0.0, None)
                trial_linear[~mask] = 0.0
                trial_rgb = (linear_to_srgb(trial_linear) * 255.0).astype(np.uint8)
                trial_lum = float((trial_rgb.astype(np.float32) @ LUMINANCE_WEIGHTS).mean())
                if trial_lum > target:
                    high = mid
                else:
                    low = mid
                    best_scale = mid
                    best_rgb = trial_rgb
                    best_linear = trial_linear
                    best_luminance = trial_lum
            scale = best_scale
            specular = specular_base * scale
            spec_mean = float(specular[mask].mean())
            clamp_scale *= scale
            linear = best_linear
            rgb = best_rgb
            luminance = best_luminance
            prev_spec_mean = spec_mean
        prev_luminance = luminance
        caption = f"GGX IBL  r={r:.2f}  α={r * r:.4f}"
        labeled = add_caption_rgb(rgb.copy(), caption)
        tiles.append(
            TileStats(
                roughness=r,
                caption=caption,
                rgb=rgb,
                mean_luminance=luminance,
                mean_specular=spec_mean,
                mean_diffuse=diff_mean,
                specular_scale=clamp_scale,
            )
        )
        captioned_tiles.append(labeled)
    gallery = make_row_rgb(captioned_tiles)
    return tiles, gallery


def hash_prefilter_levels(levels: Sequence[PrefilterLevel]) -> str:
    h = hashlib.sha256()
    for lvl in levels:
        payload = np.ascontiguousarray(lvl.faces).view(np.uint8)
        h.update(payload)
    return h.hexdigest()


def evaluate_acceptance(
    seam_report: Dict[str, object],
    tiles: Sequence[TileStats],
    lut: np.ndarray,
    deterministic_ok: bool,
) -> Dict[str, object]:
    fail_messages: List[str] = []
    d1_ok = float(seam_report["max_rms"]) < 1e-3
    if not d1_ok:
        fail_messages.append(f"D1 failed: seam RMS={seam_report['max_rms']:.4e} (threshold 1e-3)")

    luminances = [t.mean_luminance for t in tiles]
    d2_ok = all(luminances[i + 1] <= luminances[i] + 1e-3 for i in range(len(luminances) - 1))
    if not d2_ok:
        fail_messages.append("D2 failed: mean luminance not monotonic across roughness sweep")

    lut_min = float(np.min(lut))
    lut_max = float(np.max(lut))
    d3_ok = lut_min >= -1e-5 and lut_max <= 1.05 + 1e-5
    if not d3_ok:
        fail_messages.append(f"D3 failed: LUT range [{lut_min:.4f}, {lut_max:.4f}] outside [0, 1.05]")

    d4_ok = deterministic_ok
    if not d4_ok:
        fail_messages.append("D4 failed: gallery hash mismatch across repeated run")

    return {
        "pass": len(fail_messages) == 0,
        "messages": fail_messages,
        "D1_seams": {"pass": d1_ok, **seam_report},
        "D2_energy": {"pass": d2_ok, "luminance": luminances},
        "D3_lut_bounds": {"pass": d3_ok, "min": lut_min, "max": lut_max},
        "D4_determinism": {"pass": d4_ok},
    }


def record_failure(outdir: Path, messages: Sequence[str]) -> None:
    text = "\n".join(messages)
    (outdir / "m4_FAIL.txt").write_text(text, encoding="utf-8")


def build_meta(
    *,
    hdr_path: Path,
    hdr_mode: str,
    prefilter_levels: Sequence[PrefilterLevel],
    prefilter_samples: Sequence[int],
    irradiance_size: int,
    irradiance_samples: int,
    lut_size: int,
    lut_samples: int,
    tiles: Sequence[TileStats],
    acceptance: Dict[str, object],
    hashes: Dict[str, str],
    seam_report: Dict[str, object],
) -> Dict[str, object]:
    return {
        "description": "Milestone 4 split-sum IBL gallery",
        "rng_seed": RNG_SEED,
        "input_hdr": str(hdr_path),
        "hdr_mode": hdr_mode,
        "cube_size": prefilter_levels[0].size if prefilter_levels else BASE_CUBEMAP_SIZE,
        "prefilter_samples": list(prefilter_samples),
        "irradiance_size": irradiance_size,
        "irradiance_samples": irradiance_samples,
        "lut_size": lut_size,
        "lut_samples": lut_samples,
        "roughness_values": list(ROUGHNESS_VALUES),
        "base_color": BASE_COLOR.tolist(),
        "f0": F0_DIELECTRIC,
        "tiles": [
            {
                "roughness": t.roughness,
                "caption": t.caption,
                "mean_luminance": round(t.mean_luminance, 5),
                "mean_specular": round(t.mean_specular, 6),
                "mean_diffuse": round(t.mean_diffuse, 6),
                "specular_scale": round(t.specular_scale, 6),
            }
            for t in tiles
        ],
        "hashes": hashes,
        "acceptance": acceptance,
        "seams": seam_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Milestone 4 generator (IBL split-sum)")
    parser.add_argument("--hdr", type=Path, default=HDR_DEFAULT, help="HDR equirectangular input")
    parser.add_argument("--outdir", type=Path, default=Path("reports"), help="Output directory")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force deterministic synthetic HDR environment instead of loading a file",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Developer flag: reduce sample counts for quicker dry-runs",
    )
    parser.add_argument("--cube-size", type=int, default=None, help="Override cubemap base size")
    parser.add_argument("--irr-size", type=int, default=None, help="Override irradiance cube size")
    parser.add_argument("--lut-size", type=int, default=None, help="Override DFG LUT size")
    parser.add_argument("--prefilter-top", type=int, default=None, help="Override prefilter top sample count")
    parser.add_argument("--prefilter-bottom", type=int, default=None, help="Override prefilter bottom sample count")
    parser.add_argument("--irr-samples", type=int, default=None, help="Override irradiance sample count")
    parser.add_argument("--lut-samples", type=int, default=None, help="Override DFG LUT sample count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    hdr_data, hdr_mode = load_hdr_environment(args.hdr, force_synthetic=args.synthetic)

    base_size = args.cube_size or (BASE_CUBEMAP_SIZE if not args.fast else BASE_CUBEMAP_SIZE // 2)
    irradiance_size = args.irr_size or (IRRADIANCE_SIZE if not args.fast else max(16, IRRADIANCE_SIZE // 2))
    lut_size = args.lut_size or (LUT_SIZE if not args.fast else max(64, LUT_SIZE // 2))
    prefilter_top = args.prefilter_top or (PREFILTER_SAMPLES_TOP if not args.fast else 24)
    prefilter_bottom = args.prefilter_bottom or (PREFILTER_SAMPLES_BOTTOM if not args.fast else 8)
    irradiance_samples = args.irr_samples or (IRRADIANCE_SAMPLES if not args.fast else 128)
    lut_samples = args.lut_samples or (DFG_LUT_SAMPLES if not args.fast else 128)

    print(f"[M4] HDR mode: {hdr_mode} ({hdr_data.shape[1]}x{hdr_data.shape[0]})")
    base_cube, _ = equirect_to_cubemap(hdr_data, base_size)
    print(f"[M4] Base cubemap: {base_cube.shape[1]} px per face")

    prefilter_levels, level_geoms, prefilter_samples = compute_prefilter_chain(
        hdr_data,
        base_size,
        prefilter_top,
        prefilter_bottom,
    )
    irradiance_faces = build_irradiance_cubemap(hdr_data, irradiance_size, irradiance_samples)
    lut = compute_dfg_lut(lut_size, lut_samples)

    seam_report = evaluate_seams(prefilter_levels, level_geoms)

    tiles, gallery = render_gallery_tiles(prefilter_levels, irradiance_faces, lut, ROUGHNESS_VALUES)
    _, gallery_repeat = render_gallery_tiles(prefilter_levels, irradiance_faces, lut, ROUGHNESS_VALUES)
    deterministic_ok = np.array_equal(gallery, gallery_repeat)

    acceptance = evaluate_acceptance(seam_report, tiles, lut, deterministic_ok)
    if not acceptance["pass"]:
        record_failure(outdir, acceptance["messages"])
        for msg in acceptance["messages"]:
            print(f"[FAIL] {msg}")
        raise RuntimeError("M4 acceptance checks failed.")

    hashes = {
        "prefilter": hash_prefilter_levels(prefilter_levels),
        "irradiance": hash_array(irradiance_faces),
        "dfg_lut": hash_array(lut),
        "gallery": hash_array(gallery),
    }

    gallery_path = outdir / "m4_gallery_env_ggx.png"
    lut_path = outdir / "m4_dfg_lut.png"
    contact_path = outdir / "m4_env_prefilter_levels.png"
    meta_path = outdir / "m4_meta.json"

    write_png(gallery_path, gallery)
    write_png(lut_path, lut_to_image(lut))
    write_png(contact_path, build_prefilter_contact_sheet(prefilter_levels))

    meta = build_meta(
        hdr_path=args.hdr,
        hdr_mode=hdr_mode,
        prefilter_levels=prefilter_levels,
        prefilter_samples=prefilter_samples,
        irradiance_size=irradiance_size,
        irradiance_samples=irradiance_samples,
        lut_size=lut_size,
        lut_samples=lut_samples,
        tiles=tiles,
        acceptance=acceptance,
        hashes=hashes,
        seam_report=seam_report,
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n[M4] Outputs:")
    print(f"  Gallery        : {gallery_path.resolve()}")
    print(f"  DFG LUT        : {lut_path.resolve()}")
    print(f"  Prefilter mips : {contact_path.resolve()}")
    print(f"  Meta           : {meta_path.resolve()}")
    print("\n[M4] Acceptance:")
    for key, info in acceptance.items():
        if key in {"pass", "messages"}:
            continue
        status = "PASS" if info["pass"] else "FAIL"
        print(f"  {key}: {status}")


if __name__ == "__main__":  # pragma: no cover
    main()
