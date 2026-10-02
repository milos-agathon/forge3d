"""Shared AETHER CPU-diagnostic and golden helpers.

The deterministic spectral oracle from ``_aether_pt_oracle`` supports the CPU
golden and supplemental diagnostics only. The hard closure uses the native
stochastic PROMETHEUS spectral tracer with an explicit black environment.
Display conversion mirrors FILMIC_TERRAIN followed by IEC 61966-2-1 sRGB.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import forge3d as f3d
from _aether_pt_oracle import independent_reference_environment


SUN_ELEVATIONS_DEG = (-5.0, 0.0, 5.0, 10.0, 30.0, 60.0, 89.0)
SUNSET_DISPLAY_ORDER_DEG = tuple(reversed(SUN_ELEVATIONS_DEG))


def lut_environment(elevation_deg: float, width: int = 128, height: int = 64) -> np.ndarray:
    report = f3d.atmosphere_generate_environment(
        int(width), int(height), float(elevation_deg), mode="lut"
    )
    rgb = np.asarray(report["rgb_linear"], dtype=np.float32)
    assert rgb.shape == (height, width, 3)
    assert np.all(np.isfinite(rgb)) and np.all(rgb >= 0.0)
    return rgb


def filmic_terrain_srgb(linear_rgb: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    """Mirror tonemap_filmic_terrain + linear_to_srgb from tonemap_common.wgsl."""

    x = np.maximum(np.asarray(linear_rgb, dtype=np.float64) * float(exposure), 0.0)
    a, b, c, d, e, f, white = 0.22, 0.30, 0.10, 0.20, 0.01, 0.30, 11.2

    def curve(value: np.ndarray | float) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        return (
            (value * (a * value + c * b) + d * e)
            / (value * (a * value + b) + d * f)
            - e / f
        )

    display_linear = np.clip(curve(x) / max(float(curve(white)), 1.0e-6), 0.0, 1.0)
    srgb = np.where(
        display_linear <= 0.0031308,
        12.92 * display_linear,
        1.055 * np.power(display_linear, 1.0 / 2.4) - 0.055,
    )
    return np.clip(srgb, 0.0, 1.0)


def sunset_strip(width: int = 128, height: int = 64, exposure: float = 32.0) -> np.ndarray:
    panels = [
        np.rint(filmic_terrain_srgb(lut_environment(e, width, height), exposure) * 255.0)
        .clip(0, 255)
        .astype(np.uint8)
        for e in SUNSET_DISPLAY_ORDER_DEG
    ]
    return np.concatenate(panels, axis=1)


def horizon_sun_signature(elevation_deg: float) -> np.ndarray:
    """Brightest horizon-adjacent LUT sample, before display transforms."""

    environment = lut_environment(elevation_deg)
    center_y = environment.shape[0] // 2
    center_x = environment.shape[1] // 2
    band = environment[center_y - 2 : center_y + 2, center_x - 5 : center_x + 5]
    luminance = band @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    y, x = np.unravel_index(int(np.argmax(luminance)), luminance.shape)
    return band[y, x]


def saturation(rgb: np.ndarray) -> float:
    value = np.maximum(np.asarray(rgb, dtype=np.float64), 0.0)
    hi = float(np.max(value))
    return 0.0 if hi <= 1.0e-12 else float((hi - np.min(value)) / hi)


def write_constant_hdr(path: Path, value: int = 128) -> None:
    """Write a tiny deterministic Radiance fixture accepted by IBL.from_hdr."""

    payload = bytearray(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 4\n")
    payload.extend(bytes([value, value, value, 128]) * 8)
    path.write_bytes(bytes(payload))


AETHER_PHYSICAL_BACKENDS = ("metal", "vulkan")


def aether_physical_backend() -> str:
    """Backend qualified for the AETHER closure: Metal, or NVIDIA Vulkan.

    NVIDIA Vulkan qualification is the owner decision of 2026-10-01
    (roadmap decision AETHER-NV-01); Metal remains the default.
    """

    backend = os.environ.get("FORGE3D_AETHER_PHYSICAL_BACKEND", "metal").strip().lower()
    if backend not in AETHER_PHYSICAL_BACKENDS:
        raise ValueError(
            f"FORGE3D_AETHER_PHYSICAL_BACKEND must be one of {AETHER_PHYSICAL_BACKENDS}, got {backend!r}"
        )
    return backend


def physical_aether_probe(backend: str | None = None) -> tuple[bool, dict]:
    backend = aether_physical_backend() if backend is None else backend
    probe = dict(f3d.device_probe(backend))
    device_type = str(probe.get("device_type", "")).lower()
    name = str(probe.get("name", "")).lower()
    physical = (
        probe.get("status") == "ok"
        and str(probe.get("backend", "")).lower() == backend
        and not bool(probe.get("software_fallback", False))
        and not any(token in name for token in ("software", "swiftshader", "llvmpipe", "virtual"))
    )
    if backend == "metal":
        physical = physical and device_type in {"integratedgpu", "discretegpu"}
    else:
        physical = (
            physical
            and device_type == "discretegpu"
            and int(probe.get("vendor", 0)) == 0x10DE
            and "nvidia" in name
        )
    return physical, probe


def physical_metal_probe() -> tuple[bool, dict]:
    return physical_aether_probe("metal")


__all__ = [
    "SUN_ELEVATIONS_DEG",
    "SUNSET_DISPLAY_ORDER_DEG",
    "filmic_terrain_srgb",
    "horizon_sun_signature",
    "independent_reference_environment",
    "lut_environment",
    "AETHER_PHYSICAL_BACKENDS",
    "aether_physical_backend",
    "physical_aether_probe",
    "physical_metal_probe",
    "saturation",
    "sunset_strip",
    "write_constant_hdr",
]
