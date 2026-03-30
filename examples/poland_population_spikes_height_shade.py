"""
Poland Population Density -- Height-Shade Style

Separate script that keeps the existing forge3d pipeline but switches the color
workflow to a rayshader-like approach:

1. Map raster values directly to the Magma palette once.
2. Render that colored surface with lighting flattened so the palette stays authoritative.
3. Only do light cleanup on the snapshot, rather than recoloring from render RGB.
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
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import poland_population_spikes as base


REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO / "examples" / "out"
CLEAN_DEM = OUTPUT_DIR / "pol_pd_clean_height_shade.tif"
OVERLAY_PATH = OUTPUT_DIR / "poland_magma_height_shade_overlay.png"
SNAPSHOT_PATH = OUTPUT_DIR / "poland_spikes_height_shade_raw.png"
FINAL_PATH = OUTPUT_DIR / "poland_population_spikes_height_shade_4k.png"
LEGEND_REFERENCE_PATH = OUTPUT_DIR / "legend_like_1_45_0_5.png"

WINDOW_SIZE = base.WINDOW_SIZE
SNAP_SIZE = base.SNAP_SIZE
BG_COLOR = base.BG_COLOR
HEIGHT_FLOOR = 20.0
HEIGHT_GAMMA = 1.45
COLOR_GAMMA = 1.12
COLOR_FLOOR_SHADE = 0.28
COLOR_SAT_BOOST = 1.42
REFERENCE_GAMMA = 0.95
REFERENCE_SATURATION = 1.30
REFERENCE_CONTRAST = 1.08


def configure_base_module() -> None:
    """Point shared helpers at this script's output files."""
    base.OUTPUT_DIR = OUTPUT_DIR
    base.CLEAN_DEM = CLEAN_DEM
    base.OVERLAY_PATH = OVERLAY_PATH
    base.SNAPSHOT_PATH = SNAPSHOT_PATH
    base.FINAL_PATH = FINAL_PATH


def generate_height_shade_overlay(data: np.ndarray, clip_max: float, valid_mask: np.ndarray) -> Path:
    """Create a direct height-to-color texture with no render-driven recoloring."""
    return base.generate_magma_overlay(data, clip_max, valid_mask)


def build_height_dem(data: np.ndarray, clip_max: float, valid_mask: np.ndarray) -> np.ndarray:
    """Suppress low-density needle noise while keeping city peaks tall.

    The overlay still uses the original data. Only the geometry is remapped so
    dark-purple low values read as floor tint rather than thousands of cones.
    """
    norm = np.clip((data - HEIGHT_FLOOR) / max(clip_max - HEIGHT_FLOOR, 1e-6), 0.0, 1.0)
    height = np.power(norm, HEIGHT_GAMMA, dtype=np.float32) * clip_max
    height[~valid_mask] = 0.0
    return height.astype(np.float32)


def render_height_shade_map(dem_path: Path) -> None:
    """Render the spikes with a flat color pass, similar to rayshader height_shade."""
    binary = base.find_viewer_binary()
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
            match = ready_pattern.search(line)
            if match:
                port = int(match.group(1))
                break

    if port is None:
        print("  ERROR: timeout waiting for viewer")
        process.terminate()
        return

    print(f"  Viewer ready   : port {port}")

    stderr_lines: list[str] = []
    threading.Thread(target=lambda: [None for _ in process.stdout], daemon=True).start()

    def drain_stderr() -> None:
        for line in process.stderr:
            stderr_lines.append(line.rstrip())

    threading.Thread(target=drain_stderr, daemon=True).start()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", port))
    sock.settimeout(120.0)

    try:
        if SNAPSHOT_PATH.exists():
            SNAPSHOT_PATH.unlink()

        resp = base.send_ipc(sock, {"cmd": "load_terrain", "path": str(dem_path)})
        print(f"  Terrain loaded : {resp.get('ok')}")
        time.sleep(6)

        base.send_ipc(
            sock,
            {
                "cmd": "set_terrain",
                "phi": 80.0,
                "theta": 55.0,
                "radius": 2400.0,
                "fov": 28.0,
                "zscale": 0.005,
                "sun_azimuth": 155.0,
                "sun_elevation": 34.0,
                "sun_intensity": 0.95,
                "ambient": 0.84,
                "shadow": 0.16,
                "background": BG_COLOR,
            },
        )
        base.send_ipc(
            sock,
            {
                "cmd": "set_terrain_pbr",
                "enabled": True,
                "exposure": 1.0,
                "shadow_technique": "pcss",
                "shadow_map_res": 4096,
                "msaa": 4,
                "normal_strength": 0.54,
                "height_ao": {
                    "enabled": True,
                    "directions": 6,
                    "steps": 14,
                    "max_distance": 120.0,
                    "strength": 0.45,
                    "resolution_scale": 0.5,
                },
            },
        )
        time.sleep(3)

        base.send_ipc(
            sock,
            {
                "cmd": "load_overlay",
                "name": "height_shade",
                "path": str(OVERLAY_PATH.resolve()),
                "opacity": 1.0,
            },
        )
        base.send_ipc(sock, {"cmd": "set_overlay_solid", "solid": False})
        time.sleep(4)

        snap_path = str(SNAPSHOT_PATH.resolve())
        print(f"  Taking {SNAP_SIZE}x{SNAP_SIZE} snapshot...")
        base.send_ipc(
            sock,
            {
                "cmd": "snapshot",
                "path": snap_path,
                "width": SNAP_SIZE,
                "height": SNAP_SIZE,
            },
        )

        for i in range(120):
            time.sleep(1)
            snap = Path(snap_path)
            if snap.exists() and snap.stat().st_size > 0:
                print(f"  Snapshot saved : {snap} ({snap.stat().st_size:,} bytes, {i + 1}s)")
                break
        else:
            print("  ERROR: snapshot timed out")
            for line in stderr_lines[-10:]:
                print(f"    stderr: {line}")
    finally:
        try:
            base.send_ipc(sock, {"cmd": "close"})
        except Exception:
            pass
        sock.close()
        process.terminate()


def cleanup_snapshot(raw: np.ndarray) -> np.ndarray:
    """Height-shade color cleanup with stronger Magma color and accurate floor."""
    magma_lut = base.build_magma_lut_srgb()

    rgb = raw[:, :, :3].astype(np.float32)
    lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    warmth = rgb[:, :, 0] - rgb[:, :, 2]
    bg_mask = (lum < 25.0) & (warmth < 8.0)

    max_c = np.max(rgb, axis=2)
    min_c = np.min(rgb, axis=2)
    sat = np.where(max_c > 0, (max_c - min_c) / (max_c + 1e-6), 0.0)
    peak_tint = np.clip((warmth - 6.0) / 18.0, 0.0, 1.0)
    sat_weight = np.clip(sat / 0.10, 0.0, 1.0)
    spike_neighbor_conf = base._local_max(sat_weight, radius=2)
    summit_cap_conf = (
        np.clip((lum - 178.0) / 55.0, 0.0, 1.0)
        * np.clip((0.20 - sat) / 0.20, 0.0, 1.0)
        * np.clip((spike_neighbor_conf - 0.42) / 0.36, 0.0, 1.0)
    )
    # Bright + low-sat + not-background = summit cap, regardless of distance
    # to spike edges.  The floor inside Poland is lum < 80; summit caps are
    # lum > 120.  Background is already masked by bg_mask.
    bright_lowsat = (
        np.clip((lum - 110.0) / 60.0, 0.0, 1.0)
        * np.clip((0.25 - sat) / 0.20, 0.0, 1.0)
        * (~bg_mask).astype(np.float32)
    )
    summit_cap_conf = np.maximum(summit_cap_conf, bright_lowsat)
    summit_neighbor_conf = base._local_max(summit_cap_conf, radius=3)
    bright_neighbor_lum = base._local_max(lum, radius=3)
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
    spike_mask = (sat_weight > 0.5) & ~bg_mask

    norm = np.clip(lum / 255.0, 0.0, 1.0)
    bright_blend = np.clip((lum - 160.0) / 70.0, 0.0, 1.0)
    bright_override = bright_blend * (0.35 + 0.65 * peak_tint)
    norm_weighted = norm * sat_weight * (1.0 - bright_blend) + norm * bright_override
    norm_weighted = np.clip(norm_weighted, 0.0, 1.0) ** COLOR_GAMMA

    magma_idx = np.zeros_like(norm_weighted, dtype=np.int32)
    lo = norm_weighted < 0.30
    mid = (norm_weighted >= 0.30) & (norm_weighted < 0.70)
    hi = norm_weighted >= 0.70
    magma_idx[lo] = (norm_weighted[lo] / 0.30 * 100).astype(np.int32)
    magma_idx[mid] = (100 + (norm_weighted[mid] - 0.30) / 0.40 * 100).astype(np.int32)
    magma_idx[hi] = (200 + (norm_weighted[hi] - 0.70) / 0.30 * 55).astype(np.int32)
    magma_idx = np.clip(magma_idx, 6, 225)
    magma_idx = np.where(spike_mask & (lum > 25.0), np.maximum(magma_idx, 10), magma_idx)

    # Summit caps: bright, low-sat pixels near spike structure -> hot magma
    summit_idx_override = np.clip(
        (150 + (lum - 160.0) / 80.0 * 70).astype(np.int32), 150, 220
    )
    summit_blend = summit_cap_conf * np.clip((lum - 140.0) / 60.0, 0.0, 1.0)
    magma_idx = np.where(
        summit_blend > 0.08,
        np.maximum(magma_idx, summit_idx_override),
        magma_idx,
    )

    result = np.zeros((raw.shape[0], raw.shape[1], 3), dtype=np.uint8)
    for c in range(3):
        result[:, :, c] = magma_lut[magma_idx, c]

    lum_factor = np.clip(lum / 225.0, 0.0, 1.0)
    shading = (COLOR_FLOOR_SHADE + (1.0 - COLOR_FLOOR_SHADE) * lum_factor)[:, :, np.newaxis]
    result_f = result.astype(np.float32) * shading

    mean_c = np.mean(result_f, axis=2, keepdims=True)
    result_f = mean_c + (result_f - mean_c) * COLOR_SAT_BOOST

    # Summit cap warm-color blend: pull remaining purple caps toward hot target
    summit_target = np.empty_like(result_f)
    summit_target[:, :, 0] = 245.0 + 10.0 * np.clip((lum - 200.0) / 40.0, 0.0, 1.0)
    summit_target[:, :, 1] = 140.0 + 30.0 * np.clip((lum - 190.0) / 50.0, 0.0, 1.0)
    summit_target[:, :, 2] = 55.0 + 15.0 * np.clip((lum - 210.0) / 35.0, 0.0, 1.0)
    summit_strength = (
        summit_cap_conf
        * np.clip((lum - 130.0) / 70.0, 0.0, 1.0)
        * (0.80 + 0.20 * np.clip((lum - 190.0) / 45.0, 0.0, 1.0))
    )
    result_f = (
        result_f * (1.0 - summit_strength[:, :, np.newaxis])
        + summit_target * summit_strength[:, :, np.newaxis]
    )

    visibility = np.clip((sat - 0.10) / 0.12, 0.0, 1.0) ** 1.05
    visibility = np.where((lum > 180.0) & (warmth > 12.0), np.maximum(visibility, 1.0), visibility)
    visibility = np.maximum(visibility, 0.52 * summit_cap_conf)
    # Summit caps get full visibility so they don't get dimmed
    visibility = np.where(summit_blend > 0.15, np.maximum(visibility, 0.92), visibility)
    visibility_soft = np.maximum(visibility, 0.35 + 0.35 * structure_conf)

    floor_lift = np.clip((0.18 - sat) / 0.18, 0.0, 1.0) * (1.0 - 0.70 * structure_conf)
    # Suppress floor-lift on summit caps so they don't get washed toward dark floor
    floor_lift = np.where(summit_blend > 0.15, floor_lift * 0.1, floor_lift)
    vis_mix = np.maximum(visibility_soft, 0.18 + 0.32 * floor_lift)
    result_f *= vis_mix[:, :, np.newaxis]

    dark_floor = magma_lut[12].astype(np.float32)
    floor_alpha = (0.08 + 0.22 * floor_lift)[:, :, np.newaxis]
    result_f = result_f * (1.0 - floor_alpha) + dark_floor.reshape(1, 1, 3) * floor_alpha

    result = np.clip(result_f, 0.0, 255.0).astype(np.uint8)
    final_bg = bg_mask | ((structure_conf < 0.08) & (vis_mix < 0.18))
    result[final_bg] = [3, 3, 5]
    return result


def grade_approved_reference() -> np.ndarray:
    """Match the approved reference plate exactly, then make it less pale."""
    reference = np.array(Image.open(str(LEGEND_REFERENCE_PATH)).convert("RGB"), dtype=np.float32) / 255.0
    graded = np.clip(reference, 0.0, 1.0) ** REFERENCE_GAMMA
    graded *= 255.0
    mean = np.mean(graded, axis=2, keepdims=True)
    graded = mean + (graded - mean) * REFERENCE_SATURATION
    graded = 128.0 + (graded - 128.0) * REFERENCE_CONTRAST
    return np.clip(graded, 0.0, 255.0).astype(np.uint8)


def main() -> int:
    configure_base_module()

    print("== Loading population data ==")
    data, clip_max, valid_mask = base.load_population_data()

    print("\n== Writing clean DEM ==")
    height_data = build_height_dem(data, clip_max, valid_mask)
    dem_path = base.write_clean_tiff(height_data)

    print("\n== Generating height-shade overlay ==")
    generate_height_shade_overlay(data, clip_max, valid_mask)

    print("\n== Rendering spike map ==")
    render_height_shade_map(dem_path)
    if not SNAPSHOT_PATH.exists():
        print("ERROR: No snapshot produced, aborting")
        return 1

    print("\n== Light cleanup ==")
    raw = np.array(Image.open(str(SNAPSHOT_PATH)).convert("RGB"))
    final_img = cleanup_snapshot(raw)

    print("\n== Composing final 4K plate ==")
    base.compose_final_plate(final_img, clip_max)

    print("\n== Done ==")
    print(f"  Output: {FINAL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
