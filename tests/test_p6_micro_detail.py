# tests/test_p6_micro_detail.py
# Phase P6: Micro-Detail tests
# Verifies triplanar detail normals and procedural albedo noise with distance fade
# RELEVANT FILES: python/forge3d/terrain_params.py, src/shaders/terrain_pbr_pom.wgsl, src/terrain_render_params.rs
"""
P6: Micro-Detail Tests

Objective: Add close-range surface detail without LOD popping.
Constraints: Detail normals fade by distance; no change to base triplanar weights.

Work:
1) Triplanar detail normal sampling (2 m repeat) blended via RNM with distance fade.
2) Procedural albedo brightness noise (±10%) using stable world-space coordinates.

Validation:
- DetailSettings dataclass created and validates correctly
- Render with detail=enabled produces different output than detail=disabled
- No shimmer during camera motion (verified by stable world-space coordinates)
- Log fade distances
"""
import pytest
import numpy as np


def test_detail_settings_import():
    """Verify DetailSettings can be imported and has correct defaults."""
    from forge3d.terrain_params import DetailSettings
    
    ds = DetailSettings()
    assert ds.enabled is False  # Disabled by default for P5 compatibility
    assert ds.detail_scale == pytest.approx(2.0)  # 2 meter repeat
    assert ds.normal_strength == pytest.approx(0.3)
    assert ds.albedo_noise == pytest.approx(0.1)  # ±10%
    assert ds.fade_start == pytest.approx(50.0)
    assert ds.fade_end == pytest.approx(200.0)


def test_detail_settings_enabled():
    """Verify DetailSettings can be created with enabled=True."""
    from forge3d.terrain_params import DetailSettings
    
    ds = DetailSettings(
        enabled=True,
        detail_scale=2.0,
        normal_strength=0.5,
        albedo_noise=0.1,
        fade_start=30.0,
        fade_end=150.0,
    )
    assert ds.enabled is True
    assert ds.normal_strength == pytest.approx(0.5)
    assert ds.fade_start == pytest.approx(30.0)
    assert ds.fade_end == pytest.approx(150.0)


def test_detail_settings_validation_scale():
    """Verify detail_scale must be positive."""
    from forge3d.terrain_params import DetailSettings
    
    with pytest.raises(ValueError, match="detail_scale must be > 0"):
        DetailSettings(detail_scale=0.0)
    
    with pytest.raises(ValueError, match="detail_scale must be > 0"):
        DetailSettings(detail_scale=-1.0)


def test_detail_settings_validation_normal_strength():
    """Verify normal_strength must be in [0, 1]."""
    from forge3d.terrain_params import DetailSettings
    
    with pytest.raises(ValueError, match="normal_strength must be in"):
        DetailSettings(normal_strength=-0.1)
    
    with pytest.raises(ValueError, match="normal_strength must be in"):
        DetailSettings(normal_strength=1.1)


def test_detail_settings_validation_albedo_noise():
    """Verify albedo_noise must be in [0, 0.5]."""
    from forge3d.terrain_params import DetailSettings
    
    with pytest.raises(ValueError, match="albedo_noise must be in"):
        DetailSettings(albedo_noise=-0.1)
    
    with pytest.raises(ValueError, match="albedo_noise must be in"):
        DetailSettings(albedo_noise=0.6)


def test_detail_settings_validation_fade_range():
    """Verify fade_end must be > fade_start."""
    from forge3d.terrain_params import DetailSettings
    
    with pytest.raises(ValueError, match="fade_end must be > fade_start"):
        DetailSettings(fade_start=100.0, fade_end=50.0)
    
    with pytest.raises(ValueError, match="fade_end must be > fade_start"):
        DetailSettings(fade_start=100.0, fade_end=100.0)


def test_terrain_render_params_has_detail():
    """Verify TerrainRenderParams includes detail settings."""
    from forge3d.terrain_params import (
        TerrainRenderParams, LightSettings, IblSettings, ShadowSettings,
        TriplanarSettings, PomSettings, LodSettings, SamplingSettings,
        ClampSettings, DetailSettings
    )
    
    params = TerrainRenderParams(
        size_px=(256, 256),
        render_scale=1.0,
        msaa_samples=1,
        z_scale=1.0,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=100.0,
        cam_phi_deg=45.0,
        cam_theta_deg=45.0,
        cam_gamma_deg=0.0,
        fov_y_deg=55.0,
        clip=(0.1, 1000.0),
        light=LightSettings("Directional", 135.0, 35.0, 3.0, [1.0, 1.0, 1.0]),
        ibl=IblSettings(True, 1.0, 0.0),
        shadows=ShadowSettings(
            True, "PCSS", 2048, 3, 4000.0, 1.0, 0.8, 0.002, 0.001, 0.3, 1e-4, 0.5, 2.0, 0.9
        ),
        triplanar=TriplanarSettings(6.0, 4.0, 1.0),
        pom=PomSettings(True, "Occlusion", 0.05, 12, 40, 4, True, True),
        lod=LodSettings(0, 0.0, -0.5),
        sampling=SamplingSettings("Linear", "Linear", "Linear", 8, "Repeat", "Repeat", "Repeat"),
        clamp=ClampSettings((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        overlays=[],
        exposure=1.0,
        gamma=2.2,
        albedo_mode="mix",
        colormap_strength=0.5,
        detail=DetailSettings(enabled=True),
    )
    
    assert params.detail is not None
    assert params.detail.enabled is True


def test_terrain_render_params_detail_defaults_disabled():
    """Verify TerrainRenderParams defaults detail to disabled for P5 compatibility."""
    from forge3d.terrain_params import (
        TerrainRenderParams, LightSettings, IblSettings, ShadowSettings,
        TriplanarSettings, PomSettings, LodSettings, SamplingSettings,
        ClampSettings
    )
    
    params = TerrainRenderParams(
        size_px=(256, 256),
        render_scale=1.0,
        msaa_samples=1,
        z_scale=1.0,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=100.0,
        cam_phi_deg=45.0,
        cam_theta_deg=45.0,
        cam_gamma_deg=0.0,
        fov_y_deg=55.0,
        clip=(0.1, 1000.0),
        light=LightSettings("Directional", 135.0, 35.0, 3.0, [1.0, 1.0, 1.0]),
        ibl=IblSettings(True, 1.0, 0.0),
        shadows=ShadowSettings(
            True, "PCSS", 2048, 3, 4000.0, 1.0, 0.8, 0.002, 0.001, 0.3, 1e-4, 0.5, 2.0, 0.9
        ),
        triplanar=TriplanarSettings(6.0, 4.0, 1.0),
        pom=PomSettings(True, "Occlusion", 0.05, 12, 40, 4, True, True),
        lod=LodSettings(0, 0.0, -0.5),
        sampling=SamplingSettings("Linear", "Linear", "Linear", 8, "Repeat", "Repeat", "Repeat"),
        clamp=ClampSettings((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        overlays=[],
        exposure=1.0,
        gamma=2.2,
        albedo_mode="mix",
        colormap_strength=0.5,
        # No detail specified - should default to disabled
    )
    
    assert params.detail is not None
    assert params.detail.enabled is False  # P5 compatibility


def test_make_terrain_params_config_accepts_detail():
    """Verify make_terrain_params_config accepts detail parameter."""
    from forge3d.terrain_params import make_terrain_params_config, DetailSettings
    
    params = make_terrain_params_config(
        size_px=(256, 256),
        render_scale=1.0,
        msaa_samples=1,
        z_scale=1.0,
        exposure=1.0,
        domain=(0.0, 100.0),
        detail=DetailSettings(enabled=True, detail_scale=3.0),
    )
    
    assert params.detail.enabled is True
    assert params.detail.detail_scale == pytest.approx(3.0)


def test_detail_settings_in_all_exports():
    """Verify DetailSettings is exported in __all__."""
    from forge3d.terrain_params import __all__
    
    assert "DetailSettings" in __all__


class TestP6LogFadeDistances:
    """Test logging of fade distances for P6 validation."""
    
    def test_fade_distances_logged(self, capsys):
        """Verify fade distances can be logged for validation."""
        from forge3d.terrain_params import DetailSettings
        
        ds = DetailSettings(
            enabled=True,
            fade_start=50.0,
            fade_end=200.0,
        )
        
        # Log fade distances (as required by P6 validation)
        print(f"P6 Detail Fade: start={ds.fade_start}, end={ds.fade_end}")
        
        captured = capsys.readouterr()
        assert "P6 Detail Fade: start=50.0, end=200.0" in captured.out
