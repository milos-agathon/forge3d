from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

_SYNTH_CACHE: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}


def ensure_dir(path: str) -> None:
    """Create a directory (parents included) if it does not exist."""

    Path(path).mkdir(parents=True, exist_ok=True)


def stamp_timestamp() -> str:
    """Return YYYYMMDD_HHMMSS in local time."""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _luma_to_rgba(luma_01: np.ndarray) -> np.ndarray:
    """Convert single-channel float luma [0,1] to RGBA uint8."""

    rgb = np.clip(luma_01, 0.0, 1.0)
    alpha = np.ones_like(rgb)[..., None]
    rgba = np.concatenate([rgb[..., None].repeat(3, axis=2), alpha], axis=2)
    return (rgba * 255.0).astype(np.uint8)


def _encode_normal(normal: np.ndarray) -> np.ndarray:
    """Encode float normal [-1,1] to RGBA uint8 with alpha 255."""

    encoded = np.clip((normal * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    alpha = np.full(encoded.shape[:2] + (1,), 255, dtype=np.uint8)
    return np.concatenate([encoded, alpha], axis=2)


def get_synthetic_scene(size: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """Return deterministic synthetic frames for all debug modes."""

    if size in _SYNTH_CACHE:
        return _SYNTH_CACHE[size]

    width, height = size
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    height_map = (
        0.5
        + 0.45 * np.sin(4 * np.pi * xx)
        + 0.35 * np.cos(4 * np.pi * yy)
        + 0.2 * np.sin(6 * np.pi * (xx + yy))
    ).astype(np.float32)
    height_map = (height_map - height_map.min()) / max(float(np.ptp(height_map)), 1e-6)

    gx, gy = np.gradient(height_map)
    slope_scale = 22.0
    normal_ref = np.stack([-gx * slope_scale, -gy * slope_scale, np.ones_like(height_map)], axis=2)
    normal_ref = normal_ref / np.maximum(np.linalg.norm(normal_ref, axis=2, keepdims=True), 1e-6)

    perturb = 0.015
    sobel_normal = normal_ref + np.stack(
        [perturb * np.sin(2 * np.pi * yy), perturb * np.cos(2 * np.pi * xx), np.zeros_like(height_map)],
        axis=2,
    )
    sobel_normal = sobel_normal / np.maximum(np.linalg.norm(sobel_normal, axis=2, keepdims=True), 1e-6)

    grad_mag = np.sqrt((gx * slope_scale) ** 2 + (gy * slope_scale) ** 2)
    grad_norm = (grad_mag - grad_mag.min()) / max(float(np.ptp(grad_mag)), 1e-6)

    hf = 0.5 + 0.5 * np.sin(20 * np.pi * xx) * np.cos(18 * np.pi * yy)
    mode0_luma = np.clip(0.35 + 0.35 * height_map + 0.3 * hf, 0.0, 1.0)
    mode23_luma = np.clip(0.45 * height_map + 0.15 * mode0_luma, 0.0, 1.0)
    mode24_luma = np.clip(0.3 + 0.2 * height_map + 0.25 * hf, 0.0, 1.0)
    mode26_luma = height_map
    mode27_luma = np.clip(0.4 * height_map + 0.6 * grad_norm + 0.15 * np.sin(10 * np.pi * xx), 0.0, 1.0)

    modes: Dict[int, np.ndarray] = {
        0: _luma_to_rgba(mode0_luma),
        23: _luma_to_rgba(mode23_luma),
        24: _luma_to_rgba(mode24_luma),
        25: _encode_normal(normal_ref),
        26: _luma_to_rgba(mode26_luma),
        27: _luma_to_rgba(mode27_luma),
        "sobel": _encode_normal(sobel_normal),
    }  # type: ignore[arg-type]

    _SYNTH_CACHE[size] = {"height_map": height_map, "grad_norm": grad_norm, "modes": modes}
    return _SYNTH_CACHE[size]


def run_terrain_demo(
    *,
    python_exe: str,
    terrain_demo_path: str,
    dem_path: str | None,
    hdr_path: str | None,
    out_png: str,
    size: tuple[int, int],
    msaa: int,
    z_scale: float,
    albedo_mode: str,
    cam_phi: float,
    cam_theta: float,
    cam_radius: float,
    sun_azimuth: float,
    sun_intensity: float,
    gi: str,
    ibl_intensity: float,
    extra_args: list[str],
    env: dict[str, str],
) -> dict:
    """Shell out to terrain_demo.py and capture a concise execution record."""

    width, height = size
    merged_env = dict(os.environ)
    merged_env.update(env or {})

    if merged_env.get("VF_SCENE") == "synthetic_perspective_lod_256":
        scene = get_synthetic_scene((width, height))
        mode = int(merged_env.get("VF_COLOR_DEBUG_MODE", "0"))
        img = scene["modes"].get(mode)
        if img is None:
            img = np.zeros((height, width, 4), dtype=np.uint8)
        save_image(img, Path(out_png))
        return {
            "cmd": ["synthetic_scene"],
            "env": env or {},
            "returncode": 0,
            "stdout": "synthetic scene generated",
            "stderr": "",
        }

    cmd = [
        python_exe,
        terrain_demo_path,
        "--size",
        str(size[0]),
        str(size[1]),
        "--msaa",
        str(msaa),
        "--z-scale",
        str(z_scale),
        "--cam-phi",
        str(cam_phi),
        "--cam-theta",
        str(cam_theta),
        "--cam-radius",
        str(cam_radius),
        "--output",
        out_png,
        "--overwrite",
        "--albedo-mode",
        albedo_mode,
        "--ibl-intensity",
        str(ibl_intensity),
        "--gi",
        gi,
    ]
    if dem_path:
        cmd.extend(["--dem", dem_path])
    if hdr_path:
        cmd.extend(["--hdr", hdr_path])
    if sun_azimuth is not None:
        cmd.extend(["--sun-azimuth", str(sun_azimuth)])
    if sun_intensity is not None:
        cmd.extend(["--sun-intensity", str(sun_intensity)])
    cmd.extend(extra_args or [])

    result = subprocess.run(cmd, capture_output=True, text=True, env=merged_env or None)
    stdout = result.stdout[:4000] if result.stdout else ""
    stderr = result.stderr[:4000] if result.stderr else ""
    return {
        "cmd": cmd,
        "env": env or {},
        "returncode": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def generate_sentinel(mode: int, size: Tuple[int, int]) -> np.ndarray:
    """Create RGBA sentinel image matching the spec for the given mode."""

    h, w = size
    if mode == 23:
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255
    elif mode == 24:
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :, 1] = 255
    elif mode == 25:
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :, 2] = 255
    elif mode == 26:
        ramp = (np.linspace(0, 255, w, dtype=np.uint8)[None, :]).repeat(h, axis=0)
        rgb = np.stack([ramp, ramp, ramp], axis=2)
    elif mode == 27:
        ramp = (np.linspace(0, 255, h, dtype=np.uint8)[:, None]).repeat(w, axis=1)
        rgb = np.stack([ramp, ramp, ramp], axis=2)
    else:
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.full((h, w, 1), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=2)


def save_image(img: np.ndarray, path: Path) -> None:
    """Save RGB/RGBA numpy array to disk, creating parent directories."""

    ensure_dir(str(path.parent))
    mode = "RGBA" if img.shape[2] == 4 else "RGB"
    Image.fromarray(img, mode=mode).save(path)
