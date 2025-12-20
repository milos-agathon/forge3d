"""M1: Tests for AOV (Arbitrary Output Variable) plumbing.

Tests the AOV settings configuration and render output functionality.
"""
import numpy as np
import pytest
from pathlib import Path
import tempfile
import os

from forge3d.terrain_params import (
    AovSettings,
    TerrainRenderParams,
    make_terrain_params_config,
)

try:
    import forge3d as f3d
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False

if not HAS_NATIVE:
    pytest.skip("AOV tests require GPU-backed native module", allow_module_level=True)


def _create_test_hdr(path: str, width: int = 8, height: int = 4) -> None:
    """Create a minimal HDR file for IBL testing."""
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\n")
        f.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        f.write(f"-Y {height} +X {width}\n".encode())
        # Write uncompressed RGBE pixels (gray)
        for _ in range(height):
            for _ in range(width):
                f.write(bytes([128, 128, 128, 128]))


class TestAovSettings:
    """Tests for AovSettings dataclass."""

    def test_aov_settings_default(self):
        """AovSettings should be disabled by default."""
        settings = AovSettings()
        assert settings.enabled is False
        assert settings.albedo is True
        assert settings.normal is True
        assert settings.depth is True
        assert settings.output_dir is None
        assert settings.format == "png"

    def test_aov_settings_enabled(self):
        """AovSettings can be enabled with specific AOVs."""
        settings = AovSettings(enabled=True, albedo=True, normal=False, depth=True)
        assert settings.enabled is True
        assert settings.albedo is True
        assert settings.normal is False
        assert settings.depth is True

    def test_aov_settings_any_enabled_property(self):
        """any_enabled property returns correct values."""
        # Disabled master switch
        settings = AovSettings(enabled=False)
        assert settings.any_enabled is False

        # Enabled but no AOVs selected
        settings = AovSettings(enabled=True, albedo=False, normal=False, depth=False)
        assert settings.any_enabled is False

        # Enabled with at least one AOV
        settings = AovSettings(enabled=True, albedo=True)
        assert settings.any_enabled is True

    def test_aov_settings_format_validation(self):
        """AovSettings should validate format field."""
        # Valid formats
        AovSettings(format="png")
        AovSettings(format="exr")
        AovSettings(format="raw")

        # Invalid format
        with pytest.raises(ValueError, match="format must be one of"):
            AovSettings(format="invalid")

    def test_aov_settings_output_dir(self):
        """AovSettings accepts output_dir."""
        settings = AovSettings(output_dir="/tmp/aov_output")
        assert settings.output_dir == "/tmp/aov_output"


class TestTerrainRenderParamsWithAov:
    """Tests for TerrainRenderParams with AOV settings."""

    def test_terrain_params_default_aov(self):
        """TerrainRenderParams should have disabled AOV by default."""
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 100.0),
        )
        assert params.aov is not None
        assert params.aov.enabled is False

    def test_terrain_params_with_aov_enabled(self):
        """TerrainRenderParams can have AOV enabled."""
        aov = AovSettings(enabled=True, albedo=True, normal=True, depth=True)
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 100.0),
            aov=aov,
        )
        assert params.aov is not None
        assert params.aov.enabled is True
        assert params.aov.albedo is True
        assert params.aov.normal is True
        assert params.aov.depth is True


class TestAovRendering:
    """Tests for AOV rendering functionality."""

    @pytest.fixture
    def simple_heightmap(self):
        """Create a simple test heightmap."""
        return np.random.rand(64, 64).astype(np.float32) * 100.0

    @pytest.fixture
    def renderer_setup(self):
        """Set up renderer, materials, and IBL for tests."""
        session = f3d.Session(window=False)
        renderer = f3d.TerrainRenderer(session)
        material_set = f3d.MaterialSet.terrain_default()
        
        # Create a minimal HDR for IBL
        with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
            tmp.close()
            _create_test_hdr(tmp.name)
            ibl = f3d.IBL.from_hdr(tmp.name, intensity=1.0)
        os.unlink(tmp.name)
        
        return renderer, material_set, ibl

    def test_render_with_aov_returns_tuple(
        self, renderer_setup, simple_heightmap
    ):
        """render_with_aov should return (Frame, AovFrame) tuple."""
        renderer, material_set, env_maps = renderer_setup
        params = make_terrain_params_config(
            size_px=(64, 64),
            render_scale=1.0,
            terrain_span=100.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 100.0),
        )
        
        frame, aov_frame = renderer.render_with_aov(
            material_set, env_maps, params, simple_heightmap
        )
        
        # Check frame is valid
        assert frame is not None
        assert frame.size == (64, 64)
        
        # Check aov_frame is valid
        assert aov_frame is not None
        assert aov_frame.size == (64, 64)
        assert aov_frame.has_albedo is True
        assert aov_frame.has_normal is True
        assert aov_frame.has_depth is True

    def test_aov_frame_save_methods(
        self, renderer_setup, simple_heightmap
    ):
        """AovFrame should have working save methods."""
        renderer, material_set, env_maps = renderer_setup
        params = make_terrain_params_config(
            size_px=(64, 64),
            render_scale=1.0,
            terrain_span=100.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 100.0),
        )
        
        _, aov_frame = renderer.render_with_aov(
            material_set, env_maps, params, simple_heightmap
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test individual save methods
            albedo_path = Path(tmpdir) / "albedo.png"
            normal_path = Path(tmpdir) / "normal.png"
            depth_path = Path(tmpdir) / "depth.png"
            
            aov_frame.save_albedo(str(albedo_path))
            aov_frame.save_normal(str(normal_path))
            aov_frame.save_depth(str(depth_path))
            
            assert albedo_path.exists()
            assert normal_path.exists()
            assert depth_path.exists()

    def test_aov_frame_save_all(
        self, renderer_setup, simple_heightmap
    ):
        """AovFrame.save_all should save all AOVs to directory."""
        renderer, material_set, env_maps = renderer_setup
        params = make_terrain_params_config(
            size_px=(64, 64),
            render_scale=1.0,
            terrain_span=100.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 100.0),
        )
        
        _, aov_frame = renderer.render_with_aov(
            material_set, env_maps, params, simple_heightmap
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            aov_frame.save_all(tmpdir, "test_render")
            
            assert (Path(tmpdir) / "test_render_albedo.png").exists()
            assert (Path(tmpdir) / "test_render_normal.png").exists()
            assert (Path(tmpdir) / "test_render_depth.png").exists()

    def test_aov_output_shapes(
        self, renderer_setup, simple_heightmap
    ):
        """AOV outputs should have correct shapes."""
        renderer, material_set, env_maps = renderer_setup
        params = make_terrain_params_config(
            size_px=(128, 64),  # Non-square to test dimensions
            render_scale=1.0,
            terrain_span=100.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 100.0),
        )
        
        _, aov_frame = renderer.render_with_aov(
            material_set, env_maps, params, simple_heightmap
        )
        
        # Check dimensions match the requested size
        assert aov_frame.size == (128, 64)
