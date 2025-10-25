# tests/test_terrain_demo.py
# Terrain demo integration test validating synthetic DEM renders
# Exists to guard PBR terrain output and artifact-free PNG saves
# RELEVANT FILES: src/terrain_renderer.rs, src/material_set.rs, src/ibl_wrapper.rs, tools/validate_rows.py
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from forge3d.terrain_params import (
    ClampSettings,
    IblSettings,
    LightSettings,
    LodSettings,
    PomSettings,
    SamplingSettings,
    ShadowSettings,
    TerrainRenderParams as TerrainRenderParamsConfig,
    TriplanarSettings,
)


required_symbols = (
    "TerrainRenderer",
    "TerrainRenderParams",
    "MaterialSet",
    "IBL",
    "Colormap1D",
    "OverlayLayer",
)

if not f3d.has_gpu() or not all(hasattr(f3d, name) for name in required_symbols):
    pytest.skip("Terrain demo requires GPU-backed forge3d module", allow_module_level=True)


def test_terrain_demo_synthetic_render(tmp_path: Path) -> None:
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()

    hdr_path = _create_hdr_fixture(tmp_path)
    try:
        ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    finally:
        hdr_path.unlink(missing_ok=True)

    heightmap = _synthetic_dem(256, 256)
    params_config = _build_params()
    params = f3d.TerrainRenderParams(params_config)

    frame = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=params,
        heightmap=heightmap,
        target=None,
    )

    output_path = tmp_path / "terrain_demo_synthetic.png"
    frame.save(str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    pixels = frame.to_numpy()
    assert pixels.shape == (256, 256, 4)
    assert pixels.dtype == np.uint8

    unique_rgb = _unique_color_count(pixels)
    assert unique_rgb >= 256, f"Expected at least 256 unique colors, found {unique_rgb}"

    luminance = _mean_luminance(pixels)
    assert 0.25 <= luminance <= 0.85, f"Mean luminance {luminance:.3f} outside [0.25, 0.85]"


def _build_params() -> TerrainRenderParamsConfig:
    cmap = f3d.Colormap1D.from_stops(
        stops=[(0.0, "#1e3a5f"), (0.5, "#6ca365"), (1.0, "#f5f1d0")],
        domain=(0.0, 1.0),
    )
    overlay = f3d.OverlayLayer.from_colormap1d(cmap, strength=0.4)

    return TerrainRenderParamsConfig(
        size_px=(256, 256),
        render_scale=1.0,
        msaa_samples=1,
        z_scale=1.0,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=6.0,
        cam_phi_deg=135.0,
        cam_theta_deg=42.0,
        cam_gamma_deg=0.0,
        fov_y_deg=55.0,
        clip=(0.1, 500.0),
        light=LightSettings("Directional", 135.0, 40.0, 3.0, [1.0, 0.97, 0.92]),
        ibl=IblSettings(True, 1.0, 0.0),
        shadows=ShadowSettings(
            True,
            "PCSS",
            1024,
            2,
            500.0,
            1.0,
            0.8,
            0.002,
            0.001,
            0.3,
            1e-4,
            0.5,
            2.0,
            0.9,
        ),
        triplanar=TriplanarSettings(6.0, 4.0, 1.0),
        pom=PomSettings(True, "Occlusion", 0.05, 12, 40, 4, True, True),
        lod=LodSettings(0, 0.0, -0.5),
        sampling=SamplingSettings("Linear", "Linear", "Linear", 8, "Repeat", "Repeat", "Repeat"),
        clamp=ClampSettings((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        overlays=[overlay],
        exposure=1.0,
        gamma=2.2,
        albedo_mode="mix",
        colormap_strength=0.5,
    )


def _synthetic_dem(width: int, height: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    peak = 400.0 * np.exp(-(xx ** 2 + yy ** 2) / 0.18)
    ridges = 120.0 * np.sin(9.0 * np.arctan2(yy, xx)) * np.exp(-(xx ** 2 + yy ** 2) / 0.5)

    np.random.seed(7)
    noise = 35.0 * np.random.randn(height, width).astype(np.float32)

    heightmap = peak + ridges + noise
    heightmap = np.clip(heightmap, 0.0, 1000.0)
    return heightmap.astype(np.float32)


def _unique_color_count(pixels: np.ndarray) -> int:
    rgb = pixels[:, :, :3]
    flat = rgb.reshape(-1, 3)
    return int(np.unique(flat, axis=0).shape[0])


def _mean_luminance(pixels: np.ndarray) -> float:
    rgb = pixels[:, :, :3].astype(np.float32) / 255.0
    rgb_linear = np.power(rgb, 2.2)
    luminance = (
        0.2126 * rgb_linear[:, :, 0]
        + 0.7152 * rgb_linear[:, :, 1]
        + 0.0722 * rgb_linear[:, :, 2]
    )
    return float(np.mean(luminance))


def _create_hdr_fixture(tmp_path: Path) -> Path:
    fd, path_str = tempfile.mkstemp(suffix=".hdr", dir=tmp_path)
    os.close(fd)
    path = Path(path_str)

    width, height = 16, 8
    with path.open("wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode("ascii"))
        for y in range(height):
            for x in range(width):
                r = int(255.0 * (x / max(width - 1, 1)))
                g = int(255.0 * (y / max(height - 1, 1)))
                b = 180
                e = 128
                handle.write(bytes((r, g, b, e)))
    return path
