#!/usr/bin/env python3
"""Unit tests for colormap LUT functionality - CPU reference validation."""

import pytest
import numpy as np
import sys
import os

# Always import Renderer (always available)
from forge3d import Renderer

# Conditionally import TerrainSpike (feature-dependent)
try:
    from forge3d import TerrainSpike
    TERRAIN_SPIKE_AVAILABLE = True
except ImportError:
    TERRAIN_SPIKE_AVAILABLE = False


@pytest.mark.skipif(not TERRAIN_SPIKE_AVAILABLE, reason="terrain_spike feature not enabled")
class TestColormapLUT:
    """Test colormap LUT texture functionality."""
    
    def test_terrain_spike_colormap_viridis(self):
        """Test that TerrainSpike can be created with viridis colormap."""
        terrain = TerrainSpike(256, 256, grid=64, colormap="viridis")
        assert terrain is not None
        
    def test_terrain_spike_colormap_magma(self):
        """Test that TerrainSpike can be created with magma colormap.""" 
        terrain = TerrainSpike(256, 256, grid=64, colormap="magma")
        assert terrain is not None
        
    def test_terrain_spike_colormap_terrain(self):
        """Test that TerrainSpike can be created with terrain colormap."""
        terrain = TerrainSpike(256, 256, grid=64, colormap="terrain")
        assert terrain is not None
        
    def test_terrain_spike_default_colormap(self):
        """Test that TerrainSpike defaults to viridis when no colormap specified."""
        terrain = TerrainSpike(256, 256, grid=64)
        assert terrain is not None
        
    def test_terrain_spike_invalid_colormap(self):
        """Test that invalid colormap raises appropriate error."""
        with pytest.raises(Exception) as exc_info:
            TerrainSpike(256, 256, grid=64, colormap="invalid_colormap")
        # The error should mention the invalid colormap
        assert "colormap" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()
        
    def test_terrain_spike_colormap_case_sensitive(self):
        """Test that colormap names are case sensitive."""
        # These should work (lowercase)
        terrain1 = TerrainSpike(256, 256, grid=64, colormap="viridis")
        terrain2 = TerrainSpike(256, 256, grid=64, colormap="magma") 
        terrain3 = TerrainSpike(256, 256, grid=64, colormap="terrain")
        assert all(t is not None for t in [terrain1, terrain2, terrain3])
        
        # These should fail (uppercase)
        with pytest.raises(Exception):
            TerrainSpike(256, 256, grid=64, colormap="VIRIDIS")
        with pytest.raises(Exception):
            TerrainSpike(256, 256, grid=64, colormap="MAGMA")
        with pytest.raises(Exception):
            TerrainSpike(256, 256, grid=64, colormap="TERRAIN")
            
    def test_terrain_spike_render_with_colormap(self):
        """Test that terrain can render PNG with different colormaps."""
        import tempfile
        import os
        
        colormaps = ["viridis", "magma", "terrain"]
        
        for colormap in colormaps:
            terrain = TerrainSpike(128, 128, grid=32, colormap=colormap)
            
            # Create temporary file for output
            with tempfile.NamedTemporaryFile(suffix=f"_{colormap}.png", delete=False) as tmp:
                tmp_path = tmp.name
                
            try:
                # This should not raise an exception
                terrain.render_png(tmp_path)
                
                # Verify file was created and has reasonable size
                assert os.path.exists(tmp_path)
                assert os.path.getsize(tmp_path) > 1000  # Should be a reasonable PNG size
                
            finally:
                # Clean up
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)


@pytest.mark.skipif(TERRAIN_SPIKE_AVAILABLE, reason="Testing fallback when terrain_spike not available")
def test_terrain_spike_not_available():
    """Test graceful handling when terrain_spike feature is not enabled."""
    # When terrain_spike is not available, the TerrainSpike class should not be importable
    with pytest.raises((ImportError, AttributeError)):
        from forge3d import TerrainSpike


def test_main_renderer_colormap_integration():
    """Test that main Renderer class properly handles colormap parameter."""
    
    renderer = Renderer(256, 256)
    
    # Create test heightmap
    heightmap = np.random.rand(64, 64).astype(np.float32)
    spacing = (1.0, 1.0)
    exaggeration = 1.0
    
    # Test valid colormap strings
    valid_colormaps = ["viridis", "magma", "terrain"]
    
    for colormap in valid_colormaps:
        # This should not raise an exception
        renderer.add_terrain(heightmap, spacing, exaggeration, colormap)
        
        # Verify terrain was added
        stats = renderer.terrain_stats()
        assert len(stats) == 4  # min, max, mean, std
        
    # Test invalid colormap
    with pytest.raises(RuntimeError, match="Unknown colormap") as exc_info:
        renderer.add_terrain(heightmap, spacing, exaggeration, "invalid_colormap")


def test_colormap_supported_exposes_names():
    """Test that colormap_supported returns expected list of colormap names."""
    from forge3d import colormap_supported
    assert colormap_supported() == ["viridis","magma","terrain"]


def test_colormap_supported_unconditional():
    import forge3d as f3d
    assert f3d.colormap_supported() == ["viridis","magma","terrain"]


@pytest.mark.skipif(not TERRAIN_SPIKE_AVAILABLE, reason="terrain_spike feature not enabled")
def test_terrain_spike_format_selection():
    """Test that TerrainSpike selects the correct format based on environment."""
    import os
    
    # Test default case: should use sRGB or UNORM format (adapter-dependent)
    terrain = TerrainSpike(128, 128, grid=32, colormap="viridis")
    assert terrain.debug_lut_format() in ("Rgba8UnormSrgb", "Rgba8Unorm")
    
    # Test with env var: should use UNORM format
    old_val = os.environ.get('VF_FORCE_LUT_UNORM')
    try:
        os.environ['VF_FORCE_LUT_UNORM'] = '1'
        terrain_unorm = TerrainSpike(128, 128, grid=32, colormap="viridis")
        assert terrain_unorm.debug_lut_format() == "Rgba8Unorm"
    finally:
        if old_val is None:
            os.environ.pop('VF_FORCE_LUT_UNORM', None)
        else:
            os.environ['VF_FORCE_LUT_UNORM'] = old_val


@pytest.mark.skipif(not TERRAIN_SPIKE_AVAILABLE, reason="terrain_spike feature not enabled")
def test_terrain_spike_with_unorm_fallback():
    """Smoke test for TerrainSpike with UNORM fallback when env var is set."""
    import os
    import tempfile
    
    # Set environment variable to force UNORM fallback
    old_val = os.environ.get('VF_FORCE_LUT_UNORM')
    try:
        os.environ['VF_FORCE_LUT_UNORM'] = '1'
        
        # This should work without shader changes
        terrain = TerrainSpike(128, 128, grid=32, colormap="viridis")
        
        # Smoke test - just verify it can render
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            terrain.render_png(tmp_path)
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 1000  # Should be a reasonable PNG size
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    finally:
        # Restore environment variable
        if old_val is None:
            os.environ.pop('VF_FORCE_LUT_UNORM', None)
        else:
            os.environ['VF_FORCE_LUT_UNORM'] = old_val


if __name__ == "__main__":
    # Run tests directly if executed as script
    pytest.main([__file__, "-v"])