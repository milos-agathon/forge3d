# tests/test_terrain_demo_integration.py
# End-to-end terrain renderer integration tests
# Exists to validate milestone 6 rendering pipeline from Python surface
# RELEVANT FILES: tests/test_terrain_renderer.py, src/terrain_renderer.rs, examples/terrain_demo.py
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


NEEDED = (
    "TerrainRenderer",
    "TerrainRenderParams",
    "OverlayLayer",
    "MaterialSet",
    "IBL",
    "Colormap1D",
)

if not f3d.has_gpu() or not all(hasattr(f3d, name) for name in NEEDED):
    pytest.skip("Terrain renderer integration requires GPU-backed build", allow_module_level=True)


def _create_hdr(path: str, width: int = 8, height: int = 4) -> None:
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\n")
        f.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        f.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                r = int((x / max(width - 1, 1)) * 255)
                g = int((y / max(height - 1, 1)) * 255)
                b = 200
                e = 128
                f.write(bytes([r, g, b, e]))


def _make_params(overlay) -> TerrainRenderParamsConfig:
    return TerrainRenderParamsConfig(
        size_px=(512, 512),
        render_scale=1.0,
        msaa_samples=1,
        z_scale=1.2,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=6.0,
        cam_phi_deg=150.0,
        cam_theta_deg=45.0,
        cam_gamma_deg=0.0,
        fov_y_deg=60.0,
        clip=(0.1, 500.0),
        light=LightSettings("Directional", 120.0, 30.0, 3.0, [1.0, 0.95, 0.9]),
        ibl=IblSettings(True, 1.2, 5.0),
        shadows=ShadowSettings(
            True,
            "PCSS",
            1024,
            2,
            350.0,
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
        triplanar=TriplanarSettings(5.5, 3.5, 1.0),
        pom=PomSettings(True, "Occlusion", 0.05, 12, 40, 4, True, True),
        lod=LodSettings(0, 0.0, -0.5),
        sampling=SamplingSettings("Linear", "Linear", "Linear", 8, "Repeat", "Repeat", "Repeat"),
        clamp=ClampSettings((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        overlays=[overlay],
        exposure=1.1,
        gamma=2.2,
        albedo_mode="mix",
        colormap_strength=0.6,
    )


def test_full_terrain_rendering_integration():
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)

    rng = np.random.default_rng(1234)
    heightmap = rng.random((512, 512), dtype=np.float32) * 1200.0

    cmap = f3d.Colormap1D.from_stops(
        stops=[(0.0, "#1f77b4"), (600.0, "#2ca02c"), (1200.0, "#ffffff")],
        domain=(0.0, 1200.0),
    )
    overlay = f3d.OverlayLayer.from_colormap1d(cmap, strength=0.85)
    config = _make_params(overlay)
    native_params = f3d.TerrainRenderParams(config)
    material_set = f3d.MaterialSet.terrain_default()

    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
        tmp.close()
        _create_hdr(tmp.name)
        ibl = f3d.IBL.from_hdr(tmp.name, intensity=1.0)
    os.unlink(tmp.name)

    frame = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=native_params,
        target=None,
        heightmap=heightmap,
    )

    assert isinstance(frame, np.ndarray)
    assert frame.shape == (512, 512, 4)
    assert frame.dtype == np.uint8
    assert frame[..., :3].std() > 0.0
    assert frame[..., 3].min() == 255
