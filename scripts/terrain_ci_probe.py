#!/usr/bin/env python3
"""Check whether the current adapter can run the terrain CI lanes."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

import forge3d as f3d
from forge3d.terrain_params import AovSettings, PomSettings, make_terrain_params_config


SOFTWARE_ADAPTER_TOKENS = (
    "basic render driver",
    "lavapipe",
    "llvmpipe",
    "swiftshader",
    "warp",
)
HARDWARE_DEVICE_TYPES = {"discretegpu", "integratedgpu", "virtualgpu"}


def _write_test_hdr(path: Path, width: int = 8, height: int = 4) -> None:
    with path.open("wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                r = int((x / max(width - 1, 1)) * 255)
                g = int((y / max(height - 1, 1)) * 255)
                handle.write(bytes([r, g, 160, 128]))


def _build_heightmap(size: int = 32) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    hill = 0.55 * np.exp(-((xx + 0.2) ** 2 * 7.5 + (yy - 0.1) ** 2 * 9.0))
    ridge = 0.25 * np.exp(-((xx - 0.35) ** 2 * 24.0 + (yy + 0.15) ** 2 * 16.0))
    slope = 0.20 * (1.0 - yy) + 0.08 * xx
    heightmap = hill + ridge + slope
    heightmap -= heightmap.min()
    heightmap /= max(float(heightmap.max()), 1e-6)
    return heightmap.astype(np.float32)


def _build_overlay():
    cmap = f3d.Colormap1D.from_stops(
        stops=[
            (0.0, "#18391f"),
            (0.40, "#4e7c35"),
            (0.72, "#9a8552"),
            (1.0, "#f3f5f9"),
        ],
        domain=(0.0, 1.0),
    )
    return f3d.OverlayLayer.from_colormap1d(cmap, strength=1.0)


def _adapter_is_ci_safe(probe: dict) -> bool:
    if probe.get("status") != "ok":
        return False
    device_type = str(probe.get("device_type", "")).lower()
    if device_type not in HARDWARE_DEVICE_TYPES:
        return False
    name = str(probe.get("name", "")).lower()
    return not any(token in name for token in SOFTWARE_ADAPTER_TOKENS)


def _build_params(*, with_aov: bool) -> object:
    return f3d.TerrainRenderParams(
        make_terrain_params_config(
            size_px=(96, 64),
            render_scale=1.0,
            terrain_span=2.8,
            msaa_samples=1,
            z_scale=1.35,
            exposure=1.0,
            domain=(0.0, 1.0),
            albedo_mode="colormap",
            colormap_strength=1.0,
            ibl_enabled=True,
            light_azimuth_deg=138.0,
            light_elevation_deg=24.0,
            sun_intensity=2.4,
            cam_radius=4.6,
            cam_phi_deg=138.0,
            cam_theta_deg=58.0,
            fov_y_deg=54.0,
            camera_mode="screen",
            overlays=[_build_overlay()],
            pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
            aov=AovSettings(enabled=with_aov, albedo=True, normal=True, depth=True)
            if with_aov
            else None,
        )
    )


def _smoke_render(mode: str) -> None:
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()
    heightmap = _build_heightmap()
    params = _build_params(with_aov=mode == "terrain-aov")

    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp_hdr:
        hdr_path = Path(tmp_hdr.name)
    with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as tmp_exr:
        exr_path = Path(tmp_exr.name)

    try:
        _write_test_hdr(hdr_path)
        ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
        if mode == "terrain-aov":
            beauty_frame, aov_frame = renderer.render_with_aov(
                material_set=material_set,
                env_maps=ibl,
                params=params,
                heightmap=heightmap,
            )
            aov_frame.save_exr(str(exr_path), beauty_frame)
        else:
            renderer.render_terrain_pbr_pom(
                material_set=material_set,
                env_maps=ibl,
                params=params,
                heightmap=heightmap,
                target=None,
                water_mask=None,
            )
    finally:
        hdr_path.unlink(missing_ok=True)
        exr_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("terrain", "terrain-aov"), required=True)
    args = parser.parse_args()

    backend = os.environ.get("WGPU_BACKEND")
    probe = f3d.device_probe(backend)
    print(f"terrain-ci-probe backend={backend!r} probe={probe}")

    if not _adapter_is_ci_safe(probe):
        print("Skipping terrain CI lane on unsupported or software adapter.")
        return 2

    try:
        _smoke_render(args.mode)
    except Exception as exc:
        print(f"Skipping terrain CI lane after failed terrain smoke render: {exc}")
        return 2

    print("Terrain CI lane is supported on this adapter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
