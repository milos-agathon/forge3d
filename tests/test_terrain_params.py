# tests/test_terrain_params.py
# Unit tests for terrain parameter dataclasses
# Ensures validation logic on terrain configuration stays correct
# RELEVANT FILES: python/forge3d/terrain_params.py, python/forge3d/__init__.py, tests/test_session.py, tests/test_colormap1d.py
import pytest

from forge3d.terrain_params import (
    ClampSettings,
    IblSettings,
    LightSettings,
    LodSettings,
    PomSettings,
    SamplingSettings,
    ShadowSettings,
    TerrainRenderParams,
    TriplanarSettings,
)


def _valid_nested() -> dict:
    """Construct a dict of valid nested settings for reuse."""

    return {
        "light": LightSettings("Directional", 135.0, 35.0, 3.0, [1.0, 1.0, 1.0]),
        "ibl": IblSettings(True, 1.0, 0.0),
        "shadows": ShadowSettings(
            True,
            "PCSS",
            2048,
            3,
            4000.0,
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
        "triplanar": TriplanarSettings(6.0, 4.0, 1.0),
        "pom": PomSettings(True, "Occlusion", 0.05, 12, 40, 4, True, True),
        "lod": LodSettings(0, 0.0, -0.5),
        "sampling": SamplingSettings("Linear", "Linear", "Linear", 8, "Repeat", "Repeat", "Repeat"),
        "clamp": ClampSettings((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    }


def test_light_settings_rejects_invalid_type() -> None:
    with pytest.raises(ValueError):
        LightSettings("Invalid", 0.0, 0.0, 1.0, [1.0, 1.0, 1.0])


def test_sampling_settings_rejects_bad_filter() -> None:
    with pytest.raises(ValueError):
        SamplingSettings("Linear", "Bad", "Linear", 1, "Repeat", "Repeat", "Repeat")


def test_terrain_render_params_valid_creation() -> None:
    nested = _valid_nested()
    params = TerrainRenderParams(
        size_px=(128, 128),
        render_scale=1.0,
        msaa_samples=4,
        z_scale=1.5,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=10.0,
        cam_phi_deg=135.0,
        cam_theta_deg=45.0,
        cam_gamma_deg=0.0,
        fov_y_deg=55.0,
        clip=(0.1, 5000.0),
        overlays=[],
        exposure=1.0,
        gamma=2.2,
        albedo_mode="mix",
        colormap_strength=0.5,
        **nested,
    )
    assert params.colormap_strength == pytest.approx(0.5)


def test_terrain_render_params_invalid_size() -> None:
    nested = _valid_nested()
    with pytest.raises(ValueError):
        TerrainRenderParams(
            size_px=(32, 64),
            render_scale=1.0,
            msaa_samples=4,
            z_scale=1.0,
            cam_target=[0.0, 0.0, 0.0],
            cam_radius=10.0,
            cam_phi_deg=0.0,
            cam_theta_deg=0.0,
            cam_gamma_deg=0.0,
            fov_y_deg=45.0,
            clip=(0.1, 100.0),
            overlays=[],
            exposure=1.0,
            gamma=2.2,
            albedo_mode="colormap",
            colormap_strength=0.5,
            **nested,
        )
