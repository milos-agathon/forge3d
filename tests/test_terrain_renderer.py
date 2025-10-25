# tests/test_terrain_renderer.py
# Tests for the TerrainRenderer PyO3 GPU pipeline
# Exists to confirm renderer creation and RGBA frame output at the native layer
# RELEVANT FILES: src/terrain_renderer.rs, src/terrain_render_params.rs, tests/test_terrain_render_params_native.py, src/overlay_layer.rs
from __future__ import annotations

import os
import tempfile

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


if not f3d.has_gpu() or not all(
    hasattr(f3d, name)
    for name in ("TerrainRenderer", "TerrainRenderParams", "OverlayLayer", "MaterialSet", "IBL")
):
    pytest.skip("Terrain renderer scaffolding requires GPU-backed native module", allow_module_level=True)


def _create_test_hdr(path: str, width: int = 8, height: int = 4) -> None:
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
    return TerrainRenderParamsConfig(
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
            True,
            "PCSS",
            1024,
            2,
            250.0,
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


def test_terrain_renderer_produces_rgba_frame():
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    assert "TerrainRenderer" in repr(renderer)

    material_set = f3d.MaterialSet.terrain_default()
    cmap = f3d.Colormap1D.from_stops(
        stops=[(0.0, "#000000"), (1.0, "#ffffff")],
        domain=(0.0, 1.0),
    )
    overlay = f3d.OverlayLayer.from_colormap1d(cmap, strength=0.7)
    config = _build_config(overlay)
    native_params = f3d.TerrainRenderParams(config)

    heightmap = np.linspace(0.0, 1.0, 128 * 128, dtype=np.float32).reshape(128, 128)

    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
        tmp.close()
        _create_test_hdr(tmp.name)
        ibl = f3d.IBL.from_hdr(tmp.name, intensity=1.0)
    os.unlink(tmp.name)

    frame = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=native_params,
        heightmap=heightmap,
        target=None,
    )

    # Frame should be a Frame object, not numpy array
    assert hasattr(frame, 'to_numpy')
    assert hasattr(frame, 'save')

    # Convert to numpy for validation
    arr = frame.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (256, 256, 4)
    assert arr.dtype == np.uint8
    assert arr[..., :3].max() > arr[..., :3].min()
