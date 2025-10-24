# tests/test_terrain_render_params_native.py
# Tests for the native TerrainRenderParams PyO3 wrapper
# Exists to ensure Python dataclasses bridge correctly into native configuration
# RELEVANT FILES: src/terrain_render_params.rs, python/forge3d/terrain_params.py, src/overlay_layer.rs, tests/test_overlay_layer.py
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


if not f3d.has_gpu() or not hasattr(f3d, "TerrainRenderParams"):
    pytest.skip("TerrainRenderParams native wrapper requires GPU-backed module", allow_module_level=True)


def _build_config(overlays):
    return TerrainRenderParamsConfig(
        size_px=(512, 256),
        render_scale=1.0,
        msaa_samples=4,
        z_scale=1.0,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=10.0,
        cam_phi_deg=135.0,
        cam_theta_deg=45.0,
        cam_gamma_deg=0.0,
        fov_y_deg=55.0,
        clip=(0.1, 1000.0),
        light=LightSettings("Directional", 135.0, 35.0, 3.0, [1.0, 1.0, 1.0]),
        ibl=IblSettings(True, 1.0, 0.0),
        shadows=ShadowSettings(
            True,
            "PCSS",
            1024,
            2,
            1000.0,
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
        overlays=overlays,
        exposure=1.0,
        gamma=2.2,
        albedo_mode="mix",
        colormap_strength=0.6,
    )


def test_native_params_creation():
    cmap = f3d.Colormap1D.from_stops(
        stops=[(0.0, "#000000"), (1.0, "#ffffff")],
        domain=(0.0, 1.0),
    )
    overlay = f3d.OverlayLayer.from_colormap1d(cmap, strength=0.8)
    config = _build_config([overlay])

    native = f3d.TerrainRenderParams(config)
    assert native.size_px == (512, 256)
    assert native.msaa_samples == 4
    assert len(native.overlays) == 1
    assert native.albedo_mode == "mix"


def test_native_params_invalid_overlays():
    config = _build_config(overlays=["invalid"])
    with pytest.raises(ValueError):
        f3d.TerrainRenderParams(config)
