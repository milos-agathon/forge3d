#!/usr/bin/env python3
"""
Poland Population Density -- 3D Spike Map
Style: milos makes maps
Data: WorldPop 2020 UN-adjusted, 1 km resolution

Creates a dramatic 3D population density spike map using forge3d's
interactive viewer with Magma palette on a dark background.

Pipeline:
  1. Load GeoTIFF, clip outliers, write uncompressed TIFF for viewer
  2. Generate Magma RGBA overlay from population data
  3. Launch interactive viewer via direct IPC, configure scene, take 4K snapshot
  4. Post-process: remap brightness to Magma LUT, darken floor/bg, add bloom
  5. Compose final 4K plate with title, legend, and attribution
"""

from __future__ import annotations

import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# -- forge3d imports -------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from forge3d.viewer_ipc import find_viewer_binary, send_ipc
from forge3d.colormaps import get as get_colormap

# -- Paths -----------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
DATA_PATH = REPO / "data" / "pol_pd_2020_1km_UNadj.tif"
OUTPUT_DIR = REPO / "examples" / "out"
CLEAN_DEM = OUTPUT_DIR / "pol_pd_clean_simple.tif"
OVERLAY_PATH = OUTPUT_DIR / "poland_magma_overlay.png"
SNAPSHOT_PATH = OUTPUT_DIR / "poland_spikes_raw.png"
FINAL_PATH = OUTPUT_DIR / "poland_population_spikes_4k.png"

# -- Render settings -------------------------------------------------------
WINDOW_SIZE = 1280
SNAP_SIZE = 4096
CLIP_PERCENTILE = 99.0
BG_COLOR = [0.008, 0.008, 0.012]
TARGET_CRS = "EPSG:3035"
TARGET_RES_METERS = 1000.0

# -- Fonts -----------------------------------------------------------------
FONT_DIR = Path("C:/Windows/Fonts")
FONT_BOLD = str(FONT_DIR / "segoeuib.ttf")
FONT_LIGHT = str(FONT_DIR / "segoeuil.ttf")
FONT_REGULAR = str(FONT_DIR / "segoeui.ttf")


# =====================================================================
# 1.  LOAD & PREPARE DATA
# =====================================================================

def load_population_data() -> tuple[np.ndarray, float, np.ndarray]:
    """Load GeoTIFF, reproject to EPSG:3035, preserve valid-data mask, clip outliers."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    with rasterio.open(DATA_PATH) as src:
        src_crs = src.crs
        src_data = src.read(1).astype(np.float32)
        nodata = src.nodata
        src_valid_mask = src.read_masks(1) > 0

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs,
            TARGET_CRS,
            src.width,
            src.height,
            *src.bounds,
            resolution=TARGET_RES_METERS,
        )

        data = np.zeros((dst_height, dst_width), dtype=np.float32)
        valid_mask_u8 = np.zeros((dst_height, dst_width), dtype=np.uint8)

        reproject(
            source=src_data,
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=nodata,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            dst_nodata=0.0,
            resampling=Resampling.bilinear,
        )
        reproject(
            source=src_valid_mask.astype(np.uint8),
            destination=valid_mask_u8,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

    valid_mask = valid_mask_u8 > 0
    if valid_mask.any():
        ys, xs = np.where(valid_mask)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        data = data[y0:y1, x0:x1]
        valid_mask = valid_mask[y0:y1, x0:x1]

    if nodata is not None:
        data[data == nodata] = 0.0
    data[~np.isfinite(data)] = 0.0
    data[~valid_mask] = 0.0
    data = np.clip(data, 0.0, None)

    valid = data[valid_mask]
    clip_max = float(np.percentile(valid, CLIP_PERCENTILE))
    data_clipped = np.clip(data, 0.0, clip_max)

    print(f"  Source CRS     : {src_crs}")
    print(f"  Target CRS     : {TARGET_CRS}")
    print(f"  Shape          : {data.shape[1]}x{data.shape[0]}")
    print(f"  Valid pixels   : {valid.shape[0]:,}")
    print(f"  Nodata pixels  : {(~valid_mask).sum():,}")
    print(f"  P{CLIP_PERCENTILE} clip    : {clip_max:.0f}")
    return data_clipped, clip_max, valid_mask


def write_clean_tiff(data: np.ndarray) -> Path:
    """Write simple uncompressed float32 TIFF the viewer can read.

    The viewer's built-in TIFF decoder does not support compressed GeoTIFF
    (e.g. horizontal predictor), so we write a raw float32 TIFF via PIL.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(data, dtype=np.float32)).save(
        str(CLEAN_DEM), format="TIFF", compression="raw"
    )
    print(f"  Clean DEM      : {CLEAN_DEM}")
    return CLEAN_DEM


# =====================================================================
# 2.  MAGMA OVERLAY IMAGE
# =====================================================================

def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    return np.where(
        c <= 0.0031308,
        c * 12.92,
        1.055 * np.power(np.maximum(c, 1e-10), 1.0 / 2.4) - 0.055,
    )


def build_magma_lut_srgb() -> np.ndarray:
    """Build a 256-entry sRGB uint8 Magma LUT."""
    magma = get_colormap("forge3d:magma")
    rgba = magma.rgba  # (256, 4) linear float32
    srgb = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        for c in range(3):
            v = rgba[i, c]
            if v <= 0.0031308:
                s = v * 12.92
            else:
                s = 1.055 * (max(v, 1e-10) ** (1.0 / 2.4)) - 0.055
            srgb[i, c] = int(np.clip(s * 255, 0, 255))
    return srgb


def generate_magma_overlay(data: np.ndarray, clip_max: float, valid_mask: np.ndarray) -> Path:
    """Render Magma-colored RGBA overlay from population data."""
    magma = get_colormap("forge3d:magma")
    rgba_lut = magma.rgba

    norm = np.clip(data / max(clip_max, 1e-6), 0.0, 1.0)
    indices = np.clip((norm * 255).astype(np.int32), 0, 255)

    h, w = data.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    for c in range(3):
        channel = linear_to_srgb(rgba_lut[indices, c])
        overlay[:, :, c] = (np.clip(channel, 0.0, 1.0) * 255).astype(np.uint8)
    overlay[:, :, 3] = (
        (np.clip(rgba_lut[indices, 3], 0.0, 1.0) * 255).astype(np.uint8)
        * valid_mask.astype(np.uint8)
    )

    Image.fromarray(overlay).save(str(OVERLAY_PATH))
    print(f"  Overlay saved  : {OVERLAY_PATH}")
    return OVERLAY_PATH


# =====================================================================
# 3.  INTERACTIVE VIEWER RENDER (direct IPC)
# =====================================================================

def render_spike_map(dem_path: Path) -> None:
    """Launch viewer via direct IPC, configure scene, take 4K snapshot."""
    binary = find_viewer_binary()
    print(f"  Viewer binary  : {binary}")

    cmd = [binary, "--ipc-port", "0", "--size", f"{WINDOW_SIZE}x{WINDOW_SIZE}"]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
    )

    ready_pattern = re.compile(r"FORGE3D_VIEWER_READY\s+port=(\d+)")
    port = None
    start = time.time()
    while time.time() - start < 30:
        if process.poll() is not None:
            print("  ERROR: viewer exited early")
            return
        line = process.stdout.readline()
        if line:
            m = ready_pattern.search(line)
            if m:
                port = int(m.group(1))
                break

    if port is None:
        print("  ERROR: timeout waiting for viewer")
        process.terminate()
        return

    print(f"  Viewer ready   : port {port}")

    # Drain stdout/stderr in background threads
    stderr_lines: list[str] = []
    threading.Thread(
        target=lambda: [None for _ in process.stdout], daemon=True
    ).start()

    def drain_stderr():
        for line in process.stderr:
            stderr_lines.append(line.rstrip())

    threading.Thread(target=drain_stderr, daemon=True).start()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", port))
    sock.settimeout(120.0)

    try:
        # Delete old snapshot so file-polling works correctly
        if SNAPSHOT_PATH.exists():
            SNAPSHOT_PATH.unlink()

        # Load terrain
        resp = send_ipc(sock, {"cmd": "load_terrain", "path": str(dem_path)})
        print(f"  Terrain loaded : {resp.get('ok')}")
        time.sleep(6)

        # Scene: dark background, dramatic lighting, camera angle
        send_ipc(sock, {
            "cmd": "set_terrain",
            "phi": 80.0, "theta": 55.0, "radius": 2400.0, "fov": 28.0,
            "zscale": 0.005,
            "sun_azimuth": 155.0, "sun_elevation": 16.0,
            "sun_intensity": 4.0, "ambient": 0.10,
            "background": BG_COLOR,
        })

        # PBR: high-quality shadows and lighting
        send_ipc(sock, {
            "cmd": "set_terrain_pbr",
            "enabled": True,
            "exposure": 1.1,
            "shadow_technique": "pcss",
            "shadow_map_res": 4096,
            "msaa": 4,
            "normal_strength": 0.5,
        })
        time.sleep(3)

        # Magma overlay
        send_ipc(sock, {
            "cmd": "load_overlay",
            "name": "magma",
            "path": str(OVERLAY_PATH.resolve()),
            "opacity": 1.0,
        })
        send_ipc(sock, {"cmd": "set_overlay_solid", "solid": False})
        time.sleep(4)

        # 4K snapshot
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        snap_path = str(SNAPSHOT_PATH.resolve())
        print(f"  Taking {SNAP_SIZE}x{SNAP_SIZE} snapshot...")
        send_ipc(sock, {
            "cmd": "snapshot",
            "path": snap_path,
            "width": SNAP_SIZE,
            "height": SNAP_SIZE,
        })

        # Poll for file
        for i in range(120):
            time.sleep(1)
            p = Path(snap_path)
            if p.exists() and p.stat().st_size > 0:
                print(f"  Snapshot saved : {p} ({p.stat().st_size:,} bytes, {i+1}s)")
                break
        else:
            print("  ERROR: snapshot timed out")
            for line in stderr_lines[-10:]:
                print(f"    stderr: {line}")

    finally:
        try:
            send_ipc(sock, {"cmd": "close"})
        except Exception:
            pass
        sock.close()
        process.terminate()


# =====================================================================
# 4.  MAGMA POST-PROCESSING
# =====================================================================

def _gaussian_blur(img: np.ndarray, radius: int) -> np.ndarray:
    """Box-blur approximation of Gaussian (3 passes)."""
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    for _ in range(3):
        pil = pil.filter(ImageFilter.BoxBlur(radius))
    return np.array(pil).astype(np.float32)


def _local_max(img: np.ndarray, radius: int) -> np.ndarray:
    """Small max-filter used to propagate spike presence onto summit caps."""
    padded = np.pad(img, radius, mode="edge")
    windows = [
        padded[dy:dy + img.shape[0], dx:dx + img.shape[1]]
        for dy in range(radius * 2 + 1)
        for dx in range(radius * 2 + 1)
    ]
    return np.maximum.reduce(windows)


def postprocess_magma(raw: np.ndarray) -> np.ndarray:
    """Remap 3D render brightness to Magma palette, darken floor/bg, add bloom."""
    magma_lut = build_magma_lut_srgb()

    rgb = raw[:, :, :3].astype(np.float32)
    lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]

    # Warmth: overlay-colored surfaces have R >> B even in deep shadow.
    # Gray floor has R ≈ B (neutral). Use this to distinguish the two.
    warmth = rgb[:, :, 0] - rgb[:, :, 2]

    # Background = dark AND neutral gray (not colored by the Magma overlay)
    bg_mask = (lum < 25.0) & (warmth < 8.0)

    max_c = np.max(rgb, axis=2)
    min_c = np.min(rgb, axis=2)
    sat = np.where(max_c > 0, (max_c - min_c) / (max_c + 1e-6), 0.0)
    peak_tint = np.clip((warmth - 6.0) / 18.0, 0.0, 1.0)

    # Gray floor has sat < 0.10, spikes have sat > 0.12.
    # Use saturation to weight the brightness used for Magma mapping:
    # gray pixels -> near-zero norm -> dark Magma; colored pixels -> full norm.
    sat_weight = np.clip(sat / 0.10, 0, 1)
    spike_neighbor_conf = _local_max(sat_weight, radius=2)
    summit_cap_conf = (
        np.clip((lum - 178.0) / 55.0, 0.0, 1.0)
        * np.clip((0.20 - sat) / 0.20, 0.0, 1.0)
        * np.clip((spike_neighbor_conf - 0.42) / 0.36, 0.0, 1.0)
    )
    summit_neighbor_conf = _local_max(summit_cap_conf, radius=3)
    bright_neighbor_lum = _local_max(lum, radius=3)
    sat_mean = (
        np.array(
            Image.fromarray((np.clip(sat_weight, 0.0, 1.0) * 255).astype(np.uint8)).filter(
                ImageFilter.BoxBlur(2)
            ),
            dtype=np.float32,
        )
        / 255.0
    )
    structure_conf = np.maximum(
        np.clip((sat_mean - 0.22) / 0.32, 0.0, 1.0),
        np.clip((summit_neighbor_conf - 0.04) / 0.30, 0.0, 1.0),
    )
    structure_conf = np.maximum(
        structure_conf,
        np.clip((spike_neighbor_conf - 0.82) / 0.18, 0.0, 1.0)
        * np.clip((bright_neighbor_lum - 170.0) / 45.0, 0.0, 1.0)
        * np.clip((lum - 95.0) / 90.0, 0.0, 1.0),
    )
    floor_conf = 1.0 - sat_weight
    floor_conf[bg_mask] = 0.0
    spike_mask = (sat_weight > 0.5) & ~bg_mask

    norm = np.clip(lum / 255.0, 0, 1)
    # Scale norm by saturation weight: gray floor maps to dark Magma.
    # But bright pixels (spike tips washed out by sun) only partially bypass
    # sat_weight when they still carry some warm overlay tint. This keeps
    # neutral/olive-looking caps from jumping into the hottest yellow band.
    bright_blend = np.clip((lum - 160.0) / 70.0, 0, 1)
    bright_override = bright_blend * (0.35 + 0.65 * peak_tint)
    norm_weighted = norm * sat_weight * (1.0 - bright_blend) + norm * bright_override
    norm_weighted = np.clip(norm_weighted, 0, 1)

    # Piecewise Magma mapping
    magma_idx = np.zeros_like(norm_weighted, dtype=np.int32)
    lo = norm_weighted < 0.30
    mid = (norm_weighted >= 0.30) & (norm_weighted < 0.70)
    hi = norm_weighted >= 0.70
    magma_idx[lo] = (norm_weighted[lo] / 0.30 * 100).astype(np.int32)
    magma_idx[mid] = (100 + (norm_weighted[mid] - 0.30) / 0.40 * 100).astype(np.int32)
    magma_idx[hi] = (200 + (norm_weighted[hi] - 0.70) / 0.30 * 55).astype(np.int32)
    # Stay off Magma's yellowest tail; that segment can read olive once the
    # viewer's white summit highlights are compressed back into the plate.
    magma_idx = np.clip(magma_idx, 0, 168)

    # Visible spikes get at least deep purple
    magma_idx = np.where(spike_mask & (lum > 25), np.maximum(magma_idx, 18), magma_idx)

    result = np.zeros((raw.shape[0], raw.shape[1], 3), dtype=np.uint8)
    for c in range(3):
        result[:, :, c] = magma_lut[magma_idx, c]

    # Preserve 3D lighting via luminance modulation
    lum_factor = np.clip(lum / 230.0, 0, 1)
    shading = (0.35 + 0.65 * lum_factor)[:, :, np.newaxis]
    result_f = result.astype(np.float32) * shading

    # Mild saturation boost, disabled for bright pixels to prevent splashes
    mean_c = np.mean(result_f, axis=2, keepdims=True)
    sat_boost = np.where(mean_c > 120, 1.0, 1.15)
    result_f = mean_c + (result_f - mean_c) * sat_boost
    result_f = np.clip(result_f, 0, 255)

    # Darken gray/low-sat pixels (floor AND mesa tops -- both appear gray in
    # the raw render because the overlay color doesn't show on horizontal
    # surfaces under the low-angle sun).  Spike sides have sat > 0.30 and
    # are safely above this ramp.  Bright warm spike tips (desaturated by
    # sunlight) are protected by the luminance+warmth guard below.
    visibility = np.clip((sat - 0.12) / 0.15, 0, 1) ** 1.2
    # Protect genuinely bright spike tips: high lum AND overlay-tinted.
    # The bare gray floor outside Poland peaks at warmth ≈ 10; spike tips
    # with Magma overlay tint start at warmth ≈ 12.
    visibility = np.where((lum > 180) & (warmth > 12),
                          np.maximum(visibility, 1.0), visibility)
    # Desaturated summit caps are bright enough to belong to spikes, but can
    # still fall below the saturation ramp and get turned into background.
    visibility = np.maximum(visibility, 0.58 * summit_cap_conf)
    # Keep shadowed spike faces in dark purple instead of letting low-sat
    # lighting collapse them all the way to the background color.
    visibility_soft = np.maximum(visibility, 0.40 * structure_conf)
    result_f *= visibility_soft[:, :, np.newaxis]
    result_f[bg_mask] = 0.0

    # Summit caps are the fragile case: bright, low-saturation, but still
    # warmed by the overlay. Pull them toward a controlled warm highlight to
    # avoid the olive/yellow splash that can show up on the highest peaks.
    summit_strength = summit_cap_conf * (0.72 + 0.28 * np.clip((lum - 190.0) / 45.0, 0.0, 1.0))
    summit_target = np.empty_like(result_f)
    summit_target[:, :, 0] = 242.0 + 13.0 * np.clip((lum - 190.0) / 55.0, 0.0, 1.0)
    summit_target[:, :, 1] = 94.0 + 18.0 * np.clip((lum - 200.0) / 45.0, 0.0, 1.0)
    summit_target[:, :, 2] = 92.0 + 12.0 * np.clip((lum - 200.0) / 45.0, 0.0, 1.0)
    result_f = (
        result_f * (1.0 - summit_strength[:, :, np.newaxis])
        + summit_target * summit_strength[:, :, np.newaxis]
    )

    # Bloom on bright colored peaks only (conservative to avoid color bleed)
    bright_mask_f = np.clip((lum - 190.0) / 65.0, 0, 1) * np.maximum(visibility, 0.55 * structure_conf)
    bright_mask_f[bg_mask] = 0.0
    glow_source = result_f * bright_mask_f[:, :, np.newaxis]
    glow = _gaussian_blur(glow_source.astype(np.uint8), radius=12)
    glow *= np.maximum(visibility, 0.40 * structure_conf)[:, :, np.newaxis]
    result_f = result_f + glow * 0.15

    result = np.clip(result_f, 0, 255).astype(np.uint8)
    final_bg = bg_mask | ((structure_conf < 0.10) & (visibility < 0.04))
    result[final_bg] = [3, 3, 5]

    # Clamp green channel harder to stay warm -- prevent greenish-yellow spike tips
    fg = ~final_bg
    max_green = (result[fg, 0].astype(np.float32) * 0.42).astype(np.uint8)
    result[fg, 1] = np.minimum(result[fg, 1], max_green)

    # Force any remaining near-black pixels to exact background color
    res_lum = (0.2126 * result[:, :, 0].astype(np.float32)
               + 0.7152 * result[:, :, 1].astype(np.float32)
               + 0.0722 * result[:, :, 2].astype(np.float32))
    result[(res_lum < 6.0) & (structure_conf < 0.08)] = [3, 3, 5]

    # Fill tiny dark holes that survive on bright spike caps after the
    # saturation/visibility shaping. Restrict this to spike-like neighborhoods
    # so the background stays untouched.
    local_mean = np.array(Image.fromarray(result).filter(ImageFilter.BoxBlur(2)), dtype=np.uint8)
    mean_lum = (
        0.2126 * local_mean[:, :, 0].astype(np.float32)
        + 0.7152 * local_mean[:, :, 1].astype(np.float32)
        + 0.0722 * local_mean[:, :, 2].astype(np.float32)
    )
    hole_mask = (
        (res_lum < 24.0)
        & (mean_lum > 78.0)
        & (
            (summit_cap_conf > 0.05)
            | (summit_neighbor_conf > 0.10)
            | (structure_conf > 0.24)
            | (spike_neighbor_conf > 0.44)
        )
    )
    result[hole_mask] = local_mean[hole_mask]
    hole_green_cap = (result[hole_mask, 0].astype(np.float32) * 0.42).astype(np.uint8)
    result[hole_mask, 1] = np.minimum(result[hole_mask, 1], hole_green_cap)

    # Recover residual black crush on spike tops and shadowed spike faces by
    # borrowing nearby spike color, but only inside strong spike structure.
    for _ in range(2):
        res_lum = (
            0.2126 * result[:, :, 0].astype(np.float32)
            + 0.7152 * result[:, :, 1].astype(np.float32)
            + 0.0722 * result[:, :, 2].astype(np.float32)
        )
        summit_fill = np.array(Image.fromarray(result).filter(ImageFilter.BoxBlur(3)), dtype=np.uint8)
        summit_fill_lum = (
            0.2126 * summit_fill[:, :, 0].astype(np.float32)
            + 0.7152 * summit_fill[:, :, 1].astype(np.float32)
            + 0.0722 * summit_fill[:, :, 2].astype(np.float32)
        )
        summit_hole_mask = (
            (res_lum < 20.0)
            & (summit_fill_lum > 26.0)
            & (
                (structure_conf > 0.24)
                | ((spike_neighbor_conf > 0.90) & (warmth > 12.0))
                | (summit_neighbor_conf > 0.45)
            )
            & ((summit_neighbor_conf > 0.08) | (bright_neighbor_lum > 205.0))
        )
        if not summit_hole_mask.any():
            break
        result[summit_hole_mask] = summit_fill[summit_hole_mask]
        summit_green_cap = (result[summit_hole_mask, 0].astype(np.float32) * 0.42).astype(np.uint8)
        result[summit_hole_mask, 1] = np.minimum(result[summit_hole_mask, 1], summit_green_cap)

    # Broad flat caps can still keep a dark stain in the middle even after the
    # small-hole repair. Spread nearby summit confidence across those plateaus
    # and blend only the darkest supported pixels toward the warm summit target.
    plateau_support = (
        np.array(
            Image.fromarray((np.clip(summit_neighbor_conf, 0.0, 1.0) * 255).astype(np.uint8)).filter(
                ImageFilter.BoxBlur(12)
            ),
            dtype=np.float32,
        )
        / 255.0
    )
    res_lum = (
        0.2126 * result[:, :, 0].astype(np.float32)
        + 0.7152 * result[:, :, 1].astype(np.float32)
        + 0.0722 * result[:, :, 2].astype(np.float32)
    )
    plateau_conf = (
        np.clip((plateau_support - 0.08) / 0.12, 0.0, 1.0)
        * np.clip((78.0 - res_lum) / 52.0, 0.0, 1.0)
        * np.clip((lum - 145.0) / 20.0, 0.0, 1.0)
    )
    if np.any(plateau_conf > 0.01):
        alpha = 0.92 * plateau_conf[:, :, np.newaxis]
        result_f = result.astype(np.float32) * (1.0 - alpha) + summit_target * alpha
        result = np.clip(result_f, 0, 255).astype(np.uint8)
        plateau_mask = plateau_conf > 0.01
        plateau_green_cap = (result[plateau_mask, 0].astype(np.float32) * 0.42).astype(np.uint8)
        result[plateau_mask, 1] = np.minimum(result[plateau_mask, 1], plateau_green_cap)

    # Warsaw's broad summit is the only truly flat cap in the render. The raw
    # viewer image still carries a few neutral gray dents there, which can map
    # to a dark splash even after the generic summit repair. Detect the single
    # largest bright flat cap and gently lift only its local dips toward the
    # surrounding plateau color.
    try:
        from scipy.ndimage import binary_dilation, label
    except ImportError:
        pass
    else:
        raw_plateau_mean = np.array(
            Image.fromarray(np.clip(lum, 0.0, 255.0).astype(np.uint8)).filter(ImageFilter.BoxBlur(8)),
            dtype=np.float32,
        )
        largest_plateau_seed = (raw_plateau_mean > 185.0) & (sat < 0.30) & (bright_neighbor_lum > 220.0)
        if largest_plateau_seed.any():
            labels, _ = label(largest_plateau_seed)
            areas = np.bincount(labels.ravel())
            if areas.size > 1:
                areas[0] = 0
                largest_plateau = labels == int(np.argmax(areas))
                largest_plateau = binary_dilation(largest_plateau, iterations=4)

                res_lum = (
                    0.2126 * result[:, :, 0].astype(np.float32)
                    + 0.7152 * result[:, :, 1].astype(np.float32)
                    + 0.0722 * result[:, :, 2].astype(np.float32)
                )
                plateau_local = np.array(Image.fromarray(result).filter(ImageFilter.BoxBlur(5)), dtype=np.float32)
                plateau_local_lum = (
                    0.2126 * plateau_local[:, :, 0]
                    + 0.7152 * plateau_local[:, :, 1]
                    + 0.0722 * plateau_local[:, :, 2]
                )
                plateau_lift = np.array(Image.fromarray(result).filter(ImageFilter.MaxFilter(11)), dtype=np.float32)
                plateau_dip_conf = (
                    largest_plateau.astype(np.float32)
                    * np.clip((plateau_local_lum - res_lum - 10.0) / 18.0, 0.0, 1.0)
                    * np.clip((132.0 - res_lum) / 44.0, 0.0, 1.0)
                )
                if np.any(plateau_dip_conf > 0.01):
                    plateau_target = plateau_local * 0.55 + plateau_lift * 0.45
                    alpha = plateau_dip_conf[:, :, np.newaxis]
                    result_f = result.astype(np.float32) * (1.0 - alpha) + plateau_target * alpha
                    result = np.clip(result_f, 0, 255).astype(np.uint8)
                    plateau_dip_mask = plateau_dip_conf > 0.01
                    plateau_green_cap = (result[plateau_dip_mask, 0].astype(np.float32) * 0.42).astype(np.uint8)
                    result[plateau_dip_mask, 1] = np.minimum(result[plateau_dip_mask, 1], plateau_green_cap)

    return result


# =====================================================================
# 5.  COMPOSE FINAL PLATE
# =====================================================================

def make_magma_legend(bar_w: int = 36, bar_h: int = 700, clip_max: float = 2072.0) -> Image.Image:
    """Vertical Magma gradient bar with tick labels."""
    magma_lut = build_magma_lut_srgb()

    bar = np.zeros((bar_h, bar_w, 4), dtype=np.uint8)
    for y in range(bar_h):
        t = 1.0 - (y / (bar_h - 1))
        idx = int(np.clip(t * 255, 0, 255))
        bar[y, :, :3] = magma_lut[idx]
        bar[y, :, 3] = 255
    bar_img = Image.fromarray(bar)

    font = ImageFont.truetype(FONT_LIGHT, 40)
    title_font = ImageFont.truetype(FONT_REGULAR, 36)

    label_w = 240
    total_w = bar_w + 16 + label_w
    title_h = 72
    total_h = title_h + bar_h + 28

    canvas = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((0, 0), "people / km\u00b2", font=title_font, fill=(190, 190, 195, 255))
    canvas.paste(bar_img, (0, title_h))

    n_ticks = 6
    for i in range(n_ticks):
        t = i / (n_ticks - 1)
        value = t * clip_max
        y_pos = title_h + int((1.0 - t) * (bar_h - 1))
        label = f"{value:,.0f}"
        draw.text((bar_w + 18, y_pos - 18), label, font=font, fill=(190, 190, 195, 255))
        draw.line([(bar_w, y_pos), (bar_w + 10, y_pos)], fill=(190, 190, 195, 180), width=2)

    return canvas


def crop_render_to_content(magma_img: np.ndarray, pad: int = 120) -> Image.Image:
    """Crop the rendered map to its non-background content for tighter framing."""
    bg = np.array([3, 3, 5], dtype=np.int16)
    diff = np.abs(magma_img.astype(np.int16) - bg)
    content_mask = np.any(diff > 2, axis=2)
    if not content_mask.any():
        return Image.fromarray(magma_img)

    ys, xs = np.where(content_mask)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(magma_img.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(magma_img.shape[1], int(xs.max()) + pad + 1)
    return Image.fromarray(magma_img[y0:y1, x0:x1])


def compose_final_plate(magma_img: np.ndarray, clip_max: float) -> None:
    """Compose post-processed render + title + legend into final 4K plate."""
    plate_size = SNAP_SIZE
    canvas = Image.new("RGB", (plate_size, plate_size), (3, 3, 5))
    draw = ImageDraw.Draw(canvas)

    margin_top = 220
    margin_bottom = 120
    margin_left = 90
    margin_right = 90

    map_w = plate_size - margin_left - margin_right
    map_h = plate_size - margin_top - margin_bottom

    render = crop_render_to_content(magma_img, pad=140).resize((map_w, map_h), Image.LANCZOS)
    canvas.paste(render, (margin_left, margin_top))

    # Title
    title_font = ImageFont.truetype(FONT_BOLD, 128)
    subtitle_font = ImageFont.truetype(FONT_LIGHT, 60)
    title_text = "POPULATION DENSITY"
    subtitle_text = "POLAND  \u00b7  2020  \u00b7  1 km resolution"
    title_pos = (margin_left, 42)
    draw.text(title_pos, title_text, font=title_font, fill=(235, 235, 238))
    title_bbox = draw.textbbox(title_pos, title_text, font=title_font)
    subtitle_bbox = draw.textbbox((0, 0), subtitle_text, font=subtitle_font)
    subtitle_h = subtitle_bbox[3] - subtitle_bbox[1]
    title_h = title_bbox[3] - title_bbox[1]
    subtitle_x = title_bbox[2] + 56
    subtitle_y = title_pos[1] + max(0, (title_h - subtitle_h) // 2 + 8)
    draw.text(
        (subtitle_x, subtitle_y),
        subtitle_text,
        font=subtitle_font, fill=(140, 140, 148),
    )

    # Subtle accent line
    draw.line([(margin_left, 258), (margin_left + 760, 258)], fill=(60, 20, 80), width=3)

    # Legend
    legend = make_magma_legend(bar_w=46, bar_h=400, clip_max=clip_max)
    legend_x = margin_left
    legend_y = 316
    canvas.paste(legend, (legend_x, legend_y), legend)

    # Attribution
    attr_font = ImageFont.truetype(FONT_LIGHT, 34)
    draw.text(
        (legend_x + legend.width + 70, plate_size - 84),
        "Data: WorldPop 2020 UN-adjusted  \u00b7  Visualization: forge3d  \u00b7  milos makes maps",
        font=attr_font, fill=(80, 80, 88),
    )

    canvas.save(str(FINAL_PATH), "PNG")
    print(f"  Final plate    : {FINAL_PATH}")


# =====================================================================
# MAIN
# =====================================================================

def main() -> int:
    print("== Loading population data ==")
    data, clip_max, valid_mask = load_population_data()

    print("\n== Writing clean DEM ==")
    dem_path = write_clean_tiff(data)

    print("\n== Generating Magma overlay ==")
    generate_magma_overlay(data, clip_max, valid_mask)

    print("\n== Rendering spike map ==")
    render_spike_map(dem_path)

    if not SNAPSHOT_PATH.exists():
        print("ERROR: No snapshot produced, aborting")
        return 1

    print("\n== Post-processing with Magma palette ==")
    raw = np.array(Image.open(str(SNAPSHOT_PATH)).convert("RGB"))
    magma_img = postprocess_magma(raw)

    print("\n== Composing final 4K plate ==")
    compose_final_plate(magma_img, clip_max)

    print("\n== Done ==")
    print(f"  Output: {FINAL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
