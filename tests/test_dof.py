"""M3: Tests for Depth of Field (DoF) with tilt-shift support.

Tests the DoF settings configuration and validates blur effect behavior.
"""
import math
import numpy as np
import pytest

from forge3d.terrain_params import (
    DofSettings,
    TerrainRenderParams,
    make_terrain_params_config,
)


class TestDofSettings:
    """Tests for DofSettings dataclass."""

    def test_dof_settings_default(self):
        """DofSettings should be disabled by default."""
        settings = DofSettings()
        assert settings.enabled is False
        assert settings.f_stop == 5.6
        assert settings.focus_distance == 100.0
        assert settings.focal_length == 50.0
        assert settings.tilt_pitch == 0.0
        assert settings.tilt_yaw == 0.0
        assert settings.method == "gather"
        assert settings.quality == "medium"
        assert settings.show_coc is False
        assert settings.debug_mode == 0

    def test_dof_settings_enabled(self):
        """DofSettings can be enabled with custom parameters."""
        settings = DofSettings(
            enabled=True,
            f_stop=2.8,
            focus_distance=50.0,
            focal_length=85.0,
        )
        assert settings.enabled is True
        assert settings.f_stop == 2.8
        assert settings.focus_distance == 50.0
        assert settings.focal_length == 85.0

    def test_dof_aperture_property(self):
        """aperture property converts f-stop to aperture value."""
        settings = DofSettings(f_stop=2.8)
        expected_aperture = 1.0 / 2.8
        assert abs(settings.aperture - expected_aperture) < 0.001

        settings = DofSettings(f_stop=5.6)
        expected_aperture = 1.0 / 5.6
        assert abs(settings.aperture - expected_aperture) < 0.001

    def test_dof_tilt_shift_parameters(self):
        """Tilt-shift parameters work correctly."""
        settings = DofSettings(
            enabled=True,
            tilt_pitch=15.0,  # degrees
            tilt_yaw=-10.0,   # degrees
        )
        assert settings.tilt_pitch == 15.0
        assert settings.tilt_yaw == -10.0
        
        # Test radian conversion
        assert abs(settings.tilt_pitch_rad - math.radians(15.0)) < 0.0001
        assert abs(settings.tilt_yaw_rad - math.radians(-10.0)) < 0.0001

    def test_dof_has_tilt_property(self):
        """has_tilt property returns correct values."""
        # No tilt
        settings = DofSettings(tilt_pitch=0.0, tilt_yaw=0.0)
        assert settings.has_tilt is False

        # Small tilt (below threshold)
        settings = DofSettings(tilt_pitch=0.005, tilt_yaw=0.005)
        assert settings.has_tilt is False

        # Pitch tilt only
        settings = DofSettings(tilt_pitch=5.0, tilt_yaw=0.0)
        assert settings.has_tilt is True

        # Yaw tilt only
        settings = DofSettings(tilt_pitch=0.0, tilt_yaw=5.0)
        assert settings.has_tilt is True

        # Both tilts
        settings = DofSettings(tilt_pitch=10.0, tilt_yaw=5.0)
        assert settings.has_tilt is True

    def test_dof_method_validation(self):
        """DofSettings validates method field."""
        # Valid methods
        DofSettings(method="gather")
        DofSettings(method="separable")

        # Invalid method
        with pytest.raises(ValueError, match="method must be one of"):
            DofSettings(method="invalid")

    def test_dof_quality_validation(self):
        """DofSettings validates quality field."""
        # Valid qualities
        DofSettings(quality="low")
        DofSettings(quality="medium")
        DofSettings(quality="high")
        DofSettings(quality="ultra")

        # Invalid quality
        with pytest.raises(ValueError, match="quality must be one of"):
            DofSettings(quality="super")

    def test_dof_f_stop_validation(self):
        """DofSettings validates f_stop > 0."""
        with pytest.raises(ValueError, match="f_stop must be > 0"):
            DofSettings(f_stop=0)
        with pytest.raises(ValueError, match="f_stop must be > 0"):
            DofSettings(f_stop=-1.0)

    def test_dof_focus_distance_validation(self):
        """DofSettings validates focus_distance > 0."""
        with pytest.raises(ValueError, match="focus_distance must be > 0"):
            DofSettings(focus_distance=0)
        with pytest.raises(ValueError, match="focus_distance must be > 0"):
            DofSettings(focus_distance=-10.0)

    def test_dof_focal_length_validation(self):
        """DofSettings validates focal_length > 0."""
        with pytest.raises(ValueError, match="focal_length must be > 0"):
            DofSettings(focal_length=0)
        with pytest.raises(ValueError, match="focal_length must be > 0"):
            DofSettings(focal_length=-50.0)


class TestTerrainRenderParamsWithDof:
    """Tests for TerrainRenderParams with DoF settings."""

    def test_terrain_params_default_dof(self):
        """TerrainRenderParams should have disabled DoF by default."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 100.0),
        )
        assert params.dof is not None
        assert params.dof.enabled is False

    def test_terrain_params_with_dof_enabled(self):
        """TerrainRenderParams can have DoF enabled."""
        dof = DofSettings(
            enabled=True,
            f_stop=2.8,
            focus_distance=200.0,
            focal_length=85.0,
        )
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 100.0),
            dof=dof,
        )
        assert params.dof is not None
        assert params.dof.enabled is True
        assert params.dof.f_stop == 2.8
        assert params.dof.focus_distance == 200.0
        assert params.dof.focal_length == 85.0

    def test_terrain_params_with_tilt_shift(self):
        """TerrainRenderParams can have tilt-shift DoF."""
        dof = DofSettings(
            enabled=True,
            f_stop=4.0,
            focus_distance=100.0,
            tilt_pitch=20.0,
            tilt_yaw=-15.0,
        )
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 100.0),
            dof=dof,
        )
        assert params.dof is not None
        assert params.dof.enabled is True
        assert params.dof.has_tilt is True
        assert params.dof.tilt_pitch == 20.0
        assert params.dof.tilt_yaw == -15.0


class TestDofPhysics:
    """Tests for DoF physics calculations."""

    def test_shallow_dof_with_wide_aperture(self):
        """Wide aperture (low f-stop) should produce more blur."""
        # f/1.4 = very wide aperture = shallow DoF
        wide = DofSettings(f_stop=1.4)
        # f/16 = narrow aperture = deep DoF
        narrow = DofSettings(f_stop=16.0)
        
        # Wide aperture has larger aperture value (more light, more blur)
        assert wide.aperture > narrow.aperture

    def test_tilt_shift_focus_plane_geometry(self):
        """Tilt-shift creates a tilted focus plane."""
        # Pure pitch tilt - focus plane tilts around horizontal axis
        pitch_tilt = DofSettings(tilt_pitch=30.0, tilt_yaw=0.0)
        assert pitch_tilt.has_tilt is True
        assert abs(pitch_tilt.tilt_pitch_rad - math.radians(30.0)) < 0.0001
        assert abs(pitch_tilt.tilt_yaw_rad) < 0.0001

        # Pure yaw tilt - focus plane tilts around vertical axis
        yaw_tilt = DofSettings(tilt_pitch=0.0, tilt_yaw=30.0)
        assert yaw_tilt.has_tilt is True
        assert abs(yaw_tilt.tilt_pitch_rad) < 0.0001
        assert abs(yaw_tilt.tilt_yaw_rad - math.radians(30.0)) < 0.0001
