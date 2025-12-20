"""M4: Tests for Motion Blur (camera shutter accumulation).

Tests the motion blur settings configuration and validates shutter accumulation behavior.
"""
import math
import pytest

from forge3d.terrain_params import (
    MotionBlurSettings,
    TerrainRenderParams,
    make_terrain_params_config,
)


class TestMotionBlurSettings:
    """Tests for MotionBlurSettings dataclass."""

    def test_motion_blur_settings_default(self):
        """MotionBlurSettings should be disabled by default."""
        settings = MotionBlurSettings()
        assert settings.enabled is False
        assert settings.samples == 8
        assert settings.shutter_open == 0.0
        assert settings.shutter_close == 0.5
        assert settings.cam_phi_delta == 0.0
        assert settings.cam_theta_delta == 0.0
        assert settings.cam_radius_delta == 0.0
        assert settings.seed is None

    def test_motion_blur_settings_enabled(self):
        """MotionBlurSettings can be enabled with custom parameters."""
        settings = MotionBlurSettings(
            enabled=True,
            samples=16,
            shutter_open=0.0,
            shutter_close=1.0,
            cam_phi_delta=10.0,
        )
        assert settings.enabled is True
        assert settings.samples == 16
        assert settings.shutter_close == 1.0
        assert settings.cam_phi_delta == 10.0

    def test_motion_blur_shutter_angle_property(self):
        """shutter_angle property calculates correct angle in degrees."""
        # 180° shutter (half frame)
        settings = MotionBlurSettings(shutter_open=0.0, shutter_close=0.5)
        assert abs(settings.shutter_angle - 180.0) < 0.001

        # 360° shutter (full frame)
        settings = MotionBlurSettings(shutter_open=0.0, shutter_close=1.0)
        assert abs(settings.shutter_angle - 360.0) < 0.001

        # 90° shutter (quarter frame)
        settings = MotionBlurSettings(shutter_open=0.0, shutter_close=0.25)
        assert abs(settings.shutter_angle - 90.0) < 0.001

        # Centered shutter
        settings = MotionBlurSettings(shutter_open=0.25, shutter_close=0.75)
        assert abs(settings.shutter_angle - 180.0) < 0.001

    def test_motion_blur_has_camera_motion_property(self):
        """has_camera_motion property returns correct values."""
        # No motion
        settings = MotionBlurSettings()
        assert settings.has_camera_motion is False

        # Small motion (below threshold)
        settings = MotionBlurSettings(cam_phi_delta=0.0005)
        assert settings.has_camera_motion is False

        # Phi motion only
        settings = MotionBlurSettings(cam_phi_delta=5.0)
        assert settings.has_camera_motion is True

        # Theta motion only
        settings = MotionBlurSettings(cam_theta_delta=5.0)
        assert settings.has_camera_motion is True

        # Radius motion only
        settings = MotionBlurSettings(cam_radius_delta=10.0)
        assert settings.has_camera_motion is True

        # Combined motion
        settings = MotionBlurSettings(
            cam_phi_delta=5.0,
            cam_theta_delta=2.0,
            cam_radius_delta=10.0,
        )
        assert settings.has_camera_motion is True

    def test_motion_blur_samples_validation(self):
        """MotionBlurSettings validates samples field."""
        # Valid samples
        MotionBlurSettings(samples=1)
        MotionBlurSettings(samples=16)
        MotionBlurSettings(samples=64)

        # Invalid samples (too low)
        with pytest.raises(ValueError, match="samples must be >= 1"):
            MotionBlurSettings(samples=0)

        # Invalid samples (too high)
        with pytest.raises(ValueError, match="samples must be <= 64"):
            MotionBlurSettings(samples=100)

    def test_motion_blur_shutter_validation(self):
        """MotionBlurSettings validates shutter timing."""
        # Valid shutter timings
        MotionBlurSettings(shutter_open=0.0, shutter_close=0.5)
        MotionBlurSettings(shutter_open=0.25, shutter_close=0.75)
        MotionBlurSettings(shutter_open=0.0, shutter_close=1.0)

        # Invalid shutter_open (out of range)
        with pytest.raises(ValueError, match="shutter_open must be in"):
            MotionBlurSettings(shutter_open=-0.1, shutter_close=0.5)
        with pytest.raises(ValueError, match="shutter_open must be in"):
            MotionBlurSettings(shutter_open=1.1, shutter_close=0.5)

        # Invalid shutter_close (out of range)
        with pytest.raises(ValueError, match="shutter_close must be in"):
            MotionBlurSettings(shutter_open=0.0, shutter_close=-0.1)
        with pytest.raises(ValueError, match="shutter_close must be in"):
            MotionBlurSettings(shutter_open=0.0, shutter_close=1.1)

        # Invalid: close <= open
        with pytest.raises(ValueError, match="shutter_close must be > shutter_open"):
            MotionBlurSettings(shutter_open=0.5, shutter_close=0.5)
        with pytest.raises(ValueError, match="shutter_close must be > shutter_open"):
            MotionBlurSettings(shutter_open=0.7, shutter_close=0.3)

    def test_motion_blur_deterministic_seed(self):
        """MotionBlurSettings accepts deterministic seed."""
        settings = MotionBlurSettings(seed=42)
        assert settings.seed == 42

        settings = MotionBlurSettings(seed=None)
        assert settings.seed is None


class TestTerrainRenderParamsWithMotionBlur:
    """Tests for TerrainRenderParams with motion blur settings."""

    def test_terrain_params_default_motion_blur(self):
        """TerrainRenderParams should have disabled motion blur by default."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 100.0),
        )
        assert params.motion_blur is not None
        assert params.motion_blur.enabled is False

    def test_terrain_params_with_motion_blur_enabled(self):
        """TerrainRenderParams can have motion blur enabled."""
        mb = MotionBlurSettings(
            enabled=True,
            samples=16,
            shutter_open=0.0,
            shutter_close=0.5,
            cam_phi_delta=10.0,
        )
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 100.0),
            motion_blur=mb,
        )
        assert params.motion_blur is not None
        assert params.motion_blur.enabled is True
        assert params.motion_blur.samples == 16
        assert params.motion_blur.cam_phi_delta == 10.0

    def test_terrain_params_with_full_camera_motion(self):
        """TerrainRenderParams with full camera motion blur configuration."""
        mb = MotionBlurSettings(
            enabled=True,
            samples=32,
            shutter_open=0.0,
            shutter_close=1.0,  # 360° shutter
            cam_phi_delta=15.0,     # Pan
            cam_theta_delta=5.0,    # Tilt
            cam_radius_delta=50.0,  # Dolly
            seed=12345,
        )
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 100.0),
            motion_blur=mb,
        )
        assert params.motion_blur is not None
        assert params.motion_blur.enabled is True
        assert params.motion_blur.has_camera_motion is True
        assert abs(params.motion_blur.shutter_angle - 360.0) < 0.001


class TestMotionBlurPhysics:
    """Tests for motion blur physics calculations."""

    def test_shutter_angle_film_standard(self):
        """Test common film shutter angles."""
        # 180° - Standard film/cinema shutter
        mb180 = MotionBlurSettings(shutter_open=0.0, shutter_close=0.5)
        assert abs(mb180.shutter_angle - 180.0) < 0.001

        # 90° - Less blur, sharper action
        mb90 = MotionBlurSettings(shutter_open=0.0, shutter_close=0.25)
        assert abs(mb90.shutter_angle - 90.0) < 0.001

        # 45° - Very little blur, staccato effect
        mb45 = MotionBlurSettings(shutter_open=0.0, shutter_close=0.125)
        assert abs(mb45.shutter_angle - 45.0) < 0.001

        # 270° - More blur, dreamy effect
        mb270 = MotionBlurSettings(shutter_open=0.0, shutter_close=0.75)
        assert abs(mb270.shutter_angle - 270.0) < 0.001

    def test_camera_motion_types(self):
        """Test different camera motion types."""
        # Pan (horizontal rotation)
        pan = MotionBlurSettings(enabled=True, cam_phi_delta=30.0)
        assert pan.has_camera_motion is True
        assert pan.cam_phi_delta == 30.0

        # Tilt (vertical rotation)
        tilt = MotionBlurSettings(enabled=True, cam_theta_delta=15.0)
        assert tilt.has_camera_motion is True
        assert tilt.cam_theta_delta == 15.0

        # Dolly (distance change)
        dolly = MotionBlurSettings(enabled=True, cam_radius_delta=-100.0)
        assert dolly.has_camera_motion is True
        assert dolly.cam_radius_delta == -100.0

    def test_sample_count_quality_tradeoff(self):
        """Higher sample counts should be supported for quality."""
        # Low quality (fast)
        low = MotionBlurSettings(samples=4)
        assert low.samples == 4

        # Medium quality
        medium = MotionBlurSettings(samples=16)
        assert medium.samples == 16

        # High quality
        high = MotionBlurSettings(samples=32)
        assert high.samples == 32

        # Maximum quality
        ultra = MotionBlurSettings(samples=64)
        assert ultra.samples == 64
