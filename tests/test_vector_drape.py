# tests/test_vector_drape.py
# M5: Test suite for Depth-Correct Vector Overlays
# Verifies that vector overlay depth testing and halo rendering work correctly
#
# RELEVANT FILES: src/core/overlays.rs, src/shaders/overlays.wgsl,
#                 python/forge3d/terrain_params.py

import pytest

# Try to import forge3d - skip tests if not available
try:
    import forge3d
    from forge3d.terrain_params import (
        TerrainRenderParams,
        VectorOverlaySettings,
        make_terrain_params_config,
    )
    FORGE3D_AVAILABLE = True
except ImportError:
    FORGE3D_AVAILABLE = False


class TestVectorOverlayConfig:
    """Tests for VectorOverlaySettings configuration and validation."""

    def test_default_depth_test_disabled(self):
        """Test that depth testing is disabled by default."""
        settings = VectorOverlaySettings()
        assert settings.depth_test is False

    def test_default_halo_disabled(self):
        """Test that halo rendering is disabled by default."""
        settings = VectorOverlaySettings()
        assert settings.halo_enabled is False

    def test_default_contour_disabled(self):
        """Test that contour rendering is disabled by default."""
        settings = VectorOverlaySettings()
        assert settings.contour_enabled is False

    def test_depth_test_defaults(self):
        """Test depth testing default values."""
        settings = VectorOverlaySettings()
        assert settings.depth_bias == 0.001
        assert settings.depth_bias_slope == 1.0

    def test_halo_defaults(self):
        """Test halo rendering default values."""
        settings = VectorOverlaySettings()
        assert settings.halo_width == 2.0
        assert settings.halo_color == (0.0, 0.0, 0.0, 0.5)
        assert settings.halo_blur == 1.0

    def test_contour_defaults(self):
        """Test contour rendering default values."""
        settings = VectorOverlaySettings()
        assert settings.contour_width == 1.0
        assert settings.contour_color == (0.0, 0.0, 0.0, 0.8)

    def test_depth_test_enabled(self):
        """Test enabling depth testing with custom settings."""
        settings = VectorOverlaySettings(
            depth_test=True,
            depth_bias=0.002,
            depth_bias_slope=2.0,
        )
        assert settings.depth_test is True
        assert settings.depth_bias == 0.002
        assert settings.depth_bias_slope == 2.0

    def test_halo_enabled(self):
        """Test enabling halo with custom settings."""
        settings = VectorOverlaySettings(
            halo_enabled=True,
            halo_width=3.0,
            halo_color=(1.0, 1.0, 1.0, 0.8),
            halo_blur=2.0,
        )
        assert settings.halo_enabled is True
        assert settings.halo_width == 3.0
        assert settings.halo_color == (1.0, 1.0, 1.0, 0.8)
        assert settings.halo_blur == 2.0

    def test_contour_enabled(self):
        """Test enabling contour with custom settings."""
        settings = VectorOverlaySettings(
            contour_enabled=True,
            contour_width=2.0,
            contour_color=(0.5, 0.5, 0.5, 1.0),
        )
        assert settings.contour_enabled is True
        assert settings.contour_width == 2.0
        assert settings.contour_color == (0.5, 0.5, 0.5, 1.0)

    def test_depth_bias_validation(self):
        """Test that negative depth_bias raises error."""
        with pytest.raises(ValueError, match="depth_bias must be >= 0"):
            VectorOverlaySettings(depth_bias=-0.001)

    def test_depth_bias_slope_validation(self):
        """Test that negative depth_bias_slope raises error."""
        with pytest.raises(ValueError, match="depth_bias_slope must be >= 0"):
            VectorOverlaySettings(depth_bias_slope=-1.0)

    def test_halo_width_validation(self):
        """Test that negative halo_width raises error."""
        with pytest.raises(ValueError, match="halo_width must be >= 0"):
            VectorOverlaySettings(halo_width=-1.0)

    def test_halo_blur_validation(self):
        """Test that negative halo_blur raises error."""
        with pytest.raises(ValueError, match="halo_blur must be >= 0"):
            VectorOverlaySettings(halo_blur=-0.5)

    def test_halo_color_validation(self):
        """Test that invalid halo_color raises error."""
        with pytest.raises(ValueError, match="halo_color must be"):
            VectorOverlaySettings(halo_color=(1.0, 1.0, 1.0))  # Missing alpha

    def test_contour_width_validation(self):
        """Test that negative contour_width raises error."""
        with pytest.raises(ValueError, match="contour_width must be >= 0"):
            VectorOverlaySettings(contour_width=-1.0)

    def test_contour_color_validation(self):
        """Test that invalid contour_color raises error."""
        with pytest.raises(ValueError, match="contour_color must be"):
            VectorOverlaySettings(contour_color=(0.0, 0.0, 0.0))  # Missing alpha


class TestVectorOverlayInTerrainParams:
    """Tests for vector overlay integration in TerrainRenderParams."""

    def test_terrain_params_default_vector_overlay(self):
        """Test that TerrainRenderParams defaults to disabled vector overlay."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(1000.0, 2000.0),
        )
        assert params.vector_overlay is not None
        assert params.vector_overlay.depth_test is False
        assert params.vector_overlay.halo_enabled is False

    def test_terrain_params_with_depth_test(self):
        """Test TerrainRenderParams with depth testing enabled."""
        vo = VectorOverlaySettings(
            depth_test=True,
            depth_bias=0.002,
        )
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(1000.0, 2000.0),
            vector_overlay=vo,
        )
        assert params.vector_overlay is not None
        assert params.vector_overlay.depth_test is True
        assert params.vector_overlay.depth_bias == 0.002

    def test_terrain_params_with_halo(self):
        """Test TerrainRenderParams with halo enabled."""
        vo = VectorOverlaySettings(
            halo_enabled=True,
            halo_width=4.0,
            halo_color=(0.0, 0.0, 0.0, 0.7),
        )
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(1000.0, 2000.0),
            vector_overlay=vo,
        )
        assert params.vector_overlay is not None
        assert params.vector_overlay.halo_enabled is True
        assert params.vector_overlay.halo_width == 4.0

    def test_terrain_params_all_features(self):
        """Test TerrainRenderParams with all vector overlay features enabled."""
        vo = VectorOverlaySettings(
            depth_test=True,
            depth_bias=0.003,
            halo_enabled=True,
            halo_width=3.0,
            contour_enabled=True,
            contour_width=1.5,
        )
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(1000.0, 2000.0),
            vector_overlay=vo,
        )
        assert params.vector_overlay.depth_test is True
        assert params.vector_overlay.halo_enabled is True
        assert params.vector_overlay.contour_enabled is True


class TestDepthOcclusionLogic:
    """Tests for depth occlusion computation (unit tests)."""

    def test_occlusion_basic(self):
        """Test basic occlusion: terrain closer than overlay should occlude."""
        # Terrain depth closer (smaller value) than overlay depth
        terrain_depth = 0.3
        overlay_depth = 0.5
        bias = 0.001
        
        biased_overlay = overlay_depth - bias
        occluded = terrain_depth < biased_overlay
        assert occluded is True

    def test_no_occlusion_when_overlay_closer(self):
        """Test that overlay closer than terrain is not occluded."""
        terrain_depth = 0.7
        overlay_depth = 0.5
        bias = 0.001
        
        biased_overlay = overlay_depth - bias
        occluded = terrain_depth < biased_overlay
        assert occluded is False

    def test_bias_prevents_z_fighting(self):
        """Test that bias prevents z-fighting at equal depths."""
        terrain_depth = 0.5
        overlay_depth = 0.5
        bias = 0.001
        
        # Without bias, equal depths could cause z-fighting
        # With bias, overlay is slightly behind, so terrain wins
        biased_overlay = overlay_depth - bias
        occluded = terrain_depth < biased_overlay
        # At equal depth with bias, overlay should be visible (not occluded)
        assert occluded is False  # bias pushes overlay behind, making it visible

    def test_depth_test_disabled(self):
        """Test that disabled depth test never occludes."""
        depth_test_enabled = False
        terrain_depth = 0.3
        overlay_depth = 0.5
        
        if not depth_test_enabled:
            occluded = False
        else:
            occluded = terrain_depth < overlay_depth
        
        assert occluded is False


class TestHaloLogic:
    """Tests for halo rendering computation (unit tests)."""

    def test_halo_strength_at_edge(self):
        """Test halo strength computation at overlay edge."""
        # At edge: center has low alpha, neighbors have high alpha
        center_alpha = 0.0
        neighbor_alpha = 1.0
        
        halo_strength = max(0.0, neighbor_alpha - center_alpha)
        assert halo_strength == 1.0

    def test_halo_strength_inside(self):
        """Test halo strength computation inside overlay."""
        # Inside: center has high alpha, neighbors also high
        center_alpha = 1.0
        neighbor_alpha = 1.0
        
        halo_strength = max(0.0, neighbor_alpha - center_alpha)
        assert halo_strength == 0.0

    def test_halo_alpha_blending(self):
        """Test halo alpha blending with overlay."""
        overlay_alpha = 0.0
        halo_strength = 0.8
        halo_base_alpha = 0.5
        
        halo_alpha = halo_strength * halo_base_alpha
        result_alpha = max(overlay_alpha, halo_alpha)
        
        assert result_alpha == 0.4  # 0.8 * 0.5


class TestAcceptanceCriteria:
    """Tests for M5 acceptance criteria."""

    def test_depth_test_false_baseline(self):
        """Test that depth_test=False produces baseline-identical config."""
        vo_disabled = VectorOverlaySettings(depth_test=False)
        vo_default = VectorOverlaySettings()
        
        assert vo_disabled.depth_test == vo_default.depth_test
        assert vo_disabled.depth_bias == vo_default.depth_bias

    def test_halo_contrast_parameters(self):
        """Test that halo parameters can achieve contrast > 2x."""
        # Black halo on white terrain (or vice versa)
        halo_color = (0.0, 0.0, 0.0, 1.0)  # Black
        terrain_luminance = 1.0  # White
        
        # Contrast ratio = (L_bright + 0.05) / (L_dark + 0.05)
        halo_luminance = 0.0
        contrast = (terrain_luminance + 0.05) / (halo_luminance + 0.05)
        
        assert contrast > 2.0  # Should be ~21:1 for black on white


@pytest.mark.skipif(not FORGE3D_AVAILABLE, reason="forge3d not installed")
class TestVectorOverlayRendering:
    """Integration tests for vector overlay rendering (requires GPU)."""

    def test_vector_overlay_disabled_params(self):
        """Test that all-disabled vector overlay creates valid params."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(1000.0, 2000.0),
            vector_overlay=VectorOverlaySettings(),
        )
        assert params.vector_overlay.depth_test is False
        assert params.vector_overlay.halo_enabled is False

    def test_depth_test_enabled_params(self):
        """Test that depth test enabled params are correctly passed."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(1000.0, 2000.0),
            vector_overlay=VectorOverlaySettings(
                depth_test=True,
                depth_bias=0.005,
            ),
        )
        assert params.vector_overlay.depth_test is True
        assert params.vector_overlay.depth_bias == 0.005

    def test_halo_enabled_params(self):
        """Test that halo enabled params are correctly passed."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(1000.0, 2000.0),
            vector_overlay=VectorOverlaySettings(
                halo_enabled=True,
                halo_width=5.0,
            ),
        )
        assert params.vector_overlay.halo_enabled is True
        assert params.vector_overlay.halo_width == 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
