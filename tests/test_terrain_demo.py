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


# ============================================================================
# P0-09: CLI Integration Smoke Test (CPU-only, no rendering)
# ============================================================================

def test_terrain_demo_build_renderer_config() -> None:
    """Smoke test for _build_renderer_config() parsing CLI flags.
    
    Tests the terrain_demo CLI flag parsing without requiring GPU or rendering.
    Validates that _build_renderer_config() correctly translates argparse flags
    into a normalized RendererConfig.
    """
    # Import terrain_demo functions
    import sys
    from pathlib import Path
    
    # Add examples to path to import terrain_demo
    examples_path = Path(__file__).parent.parent / "examples"
    sys.path.insert(0, str(examples_path))
    
    try:
        from terrain_demo import _build_renderer_config
        import argparse
    finally:
        sys.path.pop(0)
    
    # Create mock argparse.Namespace with various CLI flags
    args = argparse.Namespace(
        light=["type=directional,dir=0.2,0.8,-0.55,intensity=8"],
        exposure=1.5,
        brdf="cooktorrance-ggx",
        shadows="pcf",
        shadow_map_res=2048,
        cascades=3,
        pcss_blocker_radius=None,
        pcss_filter_radius=None,
        shadow_light_size=None,
        shadow_moment_bias=None,
        gi="ibl,ssao",
        sky="hosek-wilkie",
        hdr=None,
        volumetric=None,
        preset=None,
    )
    
    # Build config (no rendering, CPU-only)
    config = _build_renderer_config(args)
    
    # Assert config is a RendererConfig instance
    from forge3d.config import RendererConfig
    assert isinstance(config, RendererConfig)
    
    # Validate config structure
    config.validate()
    
    # Get dict representation for assertions
    config_dict = config.to_dict()
    
    # Assert lighting configuration
    assert len(config_dict["lighting"]["lights"]) == 1
    light = config_dict["lighting"]["lights"][0]
    assert light["type"] == "directional"
    assert light["intensity"] == pytest.approx(8.0)
    assert config_dict["lighting"]["exposure"] == pytest.approx(1.5)
    
    # Assert shading configuration
    assert config_dict["shading"]["brdf"] == "cooktorrance-ggx"
    
    # Assert shadow configuration
    assert config_dict["shadows"]["technique"] == "pcf"
    assert config_dict["shadows"]["map_size"] == 2048
    assert config_dict["shadows"]["cascades"] == 3
    
    # Assert GI configuration
    assert "ibl" in config_dict["gi"]["modes"]
    assert "ssao" in config_dict["gi"]["modes"]
    
    # Assert atmosphere configuration
    assert config_dict["atmosphere"]["sky"] == "hosek-wilkie"


def test_terrain_demo_build_renderer_config_with_preset() -> None:
    """Test _build_renderer_config() with preset override.
    
    Validates that CLI flags correctly override preset values.
    """
    import sys
    from pathlib import Path
    
    examples_path = Path(__file__).parent.parent / "examples"
    sys.path.insert(0, str(examples_path))
    
    try:
        from terrain_demo import _build_renderer_config
        import argparse
    finally:
        sys.path.pop(0)
    
    # Test with preset + overrides
    args = argparse.Namespace(
        light=[],
        exposure=1.0,
        brdf="toon",
        shadows="hard",
        shadow_map_res=None,
        cascades=2,
        pcss_blocker_radius=None,
        pcss_filter_radius=None,
        shadow_light_size=None,
        shadow_moment_bias=None,
        gi=None,
        sky=None,
        hdr=None,
        volumetric=None,
        preset="outdoor_sun",  # Apply preset
    )
    
    # Build config
    config = _build_renderer_config(args)
    config.validate()
    config_dict = config.to_dict()
    
    # Overrides should take precedence over preset
    assert config_dict["shading"]["brdf"] == "toon"
    assert config_dict["shadows"]["technique"] == "hard"
    assert config_dict["shadows"]["cascades"] == 2


def test_terrain_demo_build_renderer_config_minimal() -> None:
    """Test _build_renderer_config() with minimal flags (all defaults).
    
    Validates that default config is created when no flags are provided.
    """
    import sys
    from pathlib import Path
    
    examples_path = Path(__file__).parent.parent / "examples"
    sys.path.insert(0, str(examples_path))
    
    try:
        from terrain_demo import _build_renderer_config
        import argparse
    finally:
        sys.path.pop(0)
    
    # Minimal args (all None/empty/default)
    args = argparse.Namespace(
        light=[],
        exposure=1.0,
        brdf=None,
        shadows=None,
        shadow_map_res=None,
        cascades=None,
        pcss_blocker_radius=None,
        pcss_filter_radius=None,
        shadow_light_size=None,
        shadow_moment_bias=None,
        gi=None,
        sky=None,
        hdr=None,
        volumetric=None,
        preset=None,
    )
    
    # Build config
    config = _build_renderer_config(args)
    config.validate()
    config_dict = config.to_dict()
    
    # Should have defaults
    assert config_dict["shading"]["brdf"] == "cooktorrance-ggx"  # Default BRDF
    assert config_dict["shadows"]["technique"] == "pcf"  # Default shadow technique
    assert config_dict["lighting"]["exposure"] == pytest.approx(1.0)


# ============================================================================
# GPU-dependent tests below (will be skipped in CPU-only CI)
# ============================================================================

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
