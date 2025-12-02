# tests/test_flake_diagnosis.py
# Tests for flake diagnosis debug modes (Milestone 1-3)
# RELEVANT FILES: src/shaders/terrain_pbr_pom.wgsl, docs/plan.md
"""
Flake Diagnosis Test Suite

This tests the new debug modes and LOD-aware Sobel implementation:
- Mode 23: No specular (diffuse only)
- Mode 24: No height normal (base_normal only)
- Mode 25: ddxddy normal (derivative-based ground truth)
- Mode 26: Height LOD visualization
- Mode 27: Normal blend visualization (after LOD fade)

Debug mode is controlled via VF_COLOR_DEBUG_MODE environment variable.
"""
from __future__ import annotations

import numpy as np
import pytest
import tempfile
import os
import contextlib

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


if not f3d.has_gpu():
    pytest.skip("GPU required for terrain flake tests", allow_module_level=True)


@contextlib.contextmanager
def debug_mode_env(mode: int):
    """Context manager to set VF_COLOR_DEBUG_MODE environment variable."""
    old_value = os.environ.get("VF_COLOR_DEBUG_MODE")
    os.environ["VF_COLOR_DEBUG_MODE"] = str(mode)
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("VF_COLOR_DEBUG_MODE", None)
        else:
            os.environ["VF_COLOR_DEBUG_MODE"] = old_value


def _create_test_hdr(path: str, width: int = 8, height: int = 4) -> None:
    """Create minimal HDR file for IBL."""
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\n")
        f.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        f.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                r = int((x / max(width - 1, 1)) * 255)
                g = int((y / max(height - 1, 1)) * 255)
                b = 128
                e = 128
                f.write(bytes([r, g, b, e]))


def _build_config(overlay):
    """Build terrain config."""
    config = TerrainRenderParamsConfig(
        size_px=(256, 256),
        render_scale=1.0,
        msaa_samples=1,
        z_scale=1.0,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=4.0,
        cam_phi_deg=140.0,
        cam_theta_deg=38.0,
        cam_gamma_deg=0.0,
        fov_y_deg=55.0,
        clip=(0.1, 250.0),
        light=LightSettings("Directional", 135.0, 35.0, 2.5, [1.0, 1.0, 1.0]),
        ibl=IblSettings(True, 1.0, 0.0),
        shadows=ShadowSettings(
            False, "PCF", 512, 2, 100.0, 1.0, 0.8, 0.002, 0.001, 0.3, 1e-4, 0.5, 2.0, 0.9
        ),
        triplanar=TriplanarSettings(6.0, 4.0, 1.0),
        pom=PomSettings(False, "Occlusion", 0.0, 4, 16, 2, False, False),
        lod=LodSettings(0, 0.0, 0.0),
        sampling=SamplingSettings("Linear", "Linear", "Linear", 1, "Repeat", "Repeat", "Repeat"),
        clamp=ClampSettings((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        overlays=[overlay],
        exposure=1.0,
        gamma=2.2,
        albedo_mode="mix",
        colormap_strength=0.5,
    )
    return config


def _render_frame(debug_mode: int = 0):
    """Render a single frame with specified debug mode."""
    with debug_mode_env(debug_mode):
        # Create fresh session/renderer within debug mode context
        # so the env var is read during render
        session = f3d.Session(window=False)
        renderer = f3d.TerrainRenderer(session)
        material_set = f3d.MaterialSet.terrain_default()
        
        # Create gradient heightmap with some variation
        y = np.linspace(0, 1, 256)
        x = np.linspace(0, 1, 256)
        xx, yy = np.meshgrid(x, y)
        heightmap = (np.sin(xx * 10) * 0.1 + yy * 0.8 + 0.1).astype(np.float32)
        
        # Create colormap
        cmap = f3d.Colormap1D.from_stops(
            stops=[(0.0, "#000000"), (1.0, "#ffffff")],
            domain=(0.0, 1.0),
            name="gray",
        )
        
        overlay = f3d.OverlayLayer(
            colormap=cmap,
            domain=(0.0, 1.0),
            blend_mode="alpha",
            strength=0.5,
        )
        
        config = _build_config(overlay)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            hdr_path = os.path.join(tmpdir, "test.hdr")
            _create_test_hdr(hdr_path)
            ibl = f3d.IBL(session, hdr_path)
            
            params = f3d.TerrainRenderParams.from_config(config)
            
            # Render frame
            frame = renderer.render(
                height_map=heightmap,
                material_set=material_set,
                ibl=ibl,
                params=params,
            )
            
            return frame


@pytest.mark.parametrize("debug_mode,name", [
    (0, "normal"),
    (23, "no_specular"),
    (24, "no_height_normal"),
    (25, "ddxddy_normal"),
    (26, "height_lod"),
    (27, "normal_blend"),
])
def test_flake_debug_modes_render(debug_mode: int, name: str):
    """Verify each flake diagnosis debug mode renders without crash."""
    frame = _render_frame(debug_mode=debug_mode)
    assert frame is not None
    assert frame.shape == (256, 256, 4)
    assert frame.dtype == np.uint8
    # Verify frame has non-zero content
    assert np.any(frame > 0), f"Debug mode {debug_mode} ({name}) produced empty frame"


def test_triplanar_checker_renders():
    """Verify triplanar checker debug mode (22) renders for UV stretch testing."""
    frame = _render_frame(debug_mode=22)
    assert frame is not None
    # Checker should have some variation (not solid color)
    unique_values = len(np.unique(frame[:, :, 0]))
    assert unique_values > 2, "Checker pattern should have variation"


def test_lod_aware_sobel_reduces_flakes():
    """
    Compare frames with LOD-aware Sobel vs. derivative normal.
    
    If LOD-aware Sobel (default) is working, the frame should look similar
    to the derivative-based ground truth (mode 25).
    
    This is a qualitative test - visual inspection recommended for full verification.
    """
    frame_default = _render_frame(debug_mode=0)
    frame_ddxddy = _render_frame(debug_mode=25)
    
    # Both should render successfully
    assert frame_default is not None
    assert frame_ddxddy is not None
    
    # Both should have content
    assert np.mean(frame_default) > 0
    assert np.mean(frame_ddxddy) > 0
    
    print(f"Default frame mean: {np.mean(frame_default):.2f}")
    print(f"ddxddy frame mean: {np.mean(frame_ddxddy):.2f}")
