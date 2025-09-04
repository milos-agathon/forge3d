"""Tests for sampler modes matrix and policy utilities."""

import pytest
import numpy as np
import forge3d


class TestSamplerModes:
    """Test sampler mode creation and configuration."""
    
    def test_make_sampler_basic(self):
        """Test basic sampler creation."""
        sampler = forge3d.make_sampler("clamp", "linear", "nearest")
        
        assert sampler["address_mode"] == "clamp"
        assert sampler["mag_filter"] == "linear" 
        assert sampler["min_filter"] == "linear"
        assert sampler["mip_filter"] == "nearest"
        assert sampler["name"] == "clamp_linear_linear_nearest"
    
    def test_make_sampler_defaults(self):
        """Test sampler creation with default parameters."""
        sampler = forge3d.make_sampler("repeat")
        
        assert sampler["address_mode"] == "repeat"
        assert sampler["mag_filter"] == "linear"
        assert sampler["min_filter"] == "linear" 
        assert sampler["mip_filter"] == "linear"
        assert sampler["name"] == "repeat_linear_linear_linear"
    
    def test_make_sampler_pixel_art(self):
        """Test sampler for pixel art (nearest filtering)."""
        sampler = forge3d.make_sampler("clamp", "nearest", "nearest")
        
        assert sampler["address_mode"] == "clamp"
        assert sampler["mag_filter"] == "nearest"
        assert sampler["min_filter"] == "nearest"
        assert sampler["mip_filter"] == "nearest"
    
    def test_make_sampler_invalid_mode(self):
        """Test error handling for invalid address mode."""
        with pytest.raises(ValueError, match="Invalid address mode"):
            forge3d.make_sampler("invalid")
    
    def test_make_sampler_invalid_filter(self):
        """Test error handling for invalid filter."""
        with pytest.raises(ValueError, match="Invalid filter"):
            forge3d.make_sampler("clamp", "invalid")
            
    def test_make_sampler_invalid_mip_filter(self):
        """Test error handling for invalid mip filter."""
        with pytest.raises(ValueError, match="Invalid mip filter"):
            forge3d.make_sampler("clamp", "linear", "invalid")


class TestSamplerModesList:
    """Test sampler modes enumeration."""
    
    def test_list_sampler_modes_count(self):
        """Test that we get the expected number of sampler modes."""
        modes = forge3d.list_sampler_modes()
        
        # 3 address modes × 2 filters × 2 mip filters = 12 combinations
        assert len(modes) == 12
    
    def test_list_sampler_modes_structure(self):
        """Test the structure of returned sampler modes."""
        modes = forge3d.list_sampler_modes()
        
        for mode in modes:
            assert isinstance(mode, dict)
            assert "address_mode" in mode
            assert "mag_filter" in mode
            assert "min_filter" in mode
            assert "mip_filter" in mode
            assert "name" in mode
            assert "description" in mode
            
            # Verify address mode is valid
            assert mode["address_mode"] in ["clamp", "repeat", "mirror"]
            
            # Verify filters are valid  
            assert mode["mag_filter"] in ["linear", "nearest"]
            assert mode["min_filter"] in ["linear", "nearest"]
            assert mode["mip_filter"] in ["linear", "nearest"]
            
            # Verify mag and min filters are the same (our simplified matrix)
            assert mode["mag_filter"] == mode["min_filter"]
    
    def test_list_sampler_modes_names_unique(self):
        """Test that all sampler mode names are unique."""
        modes = forge3d.list_sampler_modes()
        names = [mode["name"] for mode in modes]
        
        assert len(names) == len(set(names)), "Sampler mode names should be unique"
    
    def test_list_sampler_modes_coverage(self):
        """Test that we cover all combinations systematically."""
        modes = forge3d.list_sampler_modes()
        
        # Convert to a set of tuples for easier checking
        mode_tuples = {
            (mode["address_mode"], mode["mag_filter"], mode["mip_filter"])
            for mode in modes
        }
        
        # Expected combinations
        expected = set()
        for addr in ["clamp", "repeat", "mirror"]:
            for filt in ["linear", "nearest"]:
                for mip in ["linear", "nearest"]:
                    expected.add((addr, filt, mip))
        
        assert mode_tuples == expected
    
    def test_specific_sampler_modes_exist(self):
        """Test that specific useful sampler modes exist."""
        modes = forge3d.list_sampler_modes()
        mode_names = {mode["name"] for mode in modes}
        
        # Common useful combinations
        expected_modes = {
            "clamp_linear_linear_linear",    # UI textures
            "repeat_linear_linear_linear",   # Tiled textures
            "clamp_nearest_nearest_nearest", # Pixel art
            "clamp_linear_linear_nearest",   # Height maps
        }
        
        for expected in expected_modes:
            assert expected in mode_names, f"Expected mode '{expected}' not found"


class TestSamplerIntegration:
    """Test integration of sampler modes with the broader system."""
    
    def test_sampler_creation_integration(self):
        """Test creating samplers using the forge3d API."""
        # This tests that the API functions work together
        modes = forge3d.list_sampler_modes()
        
        for mode_info in modes[:3]:  # Test first 3 modes
            # Use list info to create equivalent sampler
            sampler = forge3d.make_sampler(
                mode_info["address_mode"],
                mode_info["mag_filter"], 
                mode_info["mip_filter"]
            )
            
            assert sampler["name"] == mode_info["name"]
    
    def test_sampler_documentation_examples(self):
        """Test the examples from the documentation."""
        # Example 1: Basic usage
        sampler = forge3d.make_sampler("clamp", "linear", "nearest")
        assert sampler["name"] == "clamp_linear_linear_nearest"
        
        # Example 2: Pixel art
        pixel_sampler = forge3d.make_sampler("clamp", "nearest", "nearest")
        assert pixel_sampler["address_mode"] == "clamp"
        assert pixel_sampler["mag_filter"] == "nearest"
        
        # Example 3: Tiled textures
        tile_sampler = forge3d.make_sampler("repeat", "linear", "linear")
        assert tile_sampler["address_mode"] == "repeat"
        assert tile_sampler["mag_filter"] == "linear"


class TestSamplerTextureSampling:
    """Test actual texture sampling behavior (when GPU available)."""
    
    @pytest.fixture
    def test_texture_8x8(self):
        """Create an 8x8 test texture with a known pattern."""
        # Create a gradient pattern that can test address modes
        texture = np.zeros((8, 8, 4), dtype=np.float32)
        
        for y in range(8):
            for x in range(8):
                # Create a pattern that's useful for testing address modes
                r = x / 7.0  # Horizontal gradient
                g = y / 7.0  # Vertical gradient
                b = (x + y) / 14.0  # Diagonal gradient
                a = 1.0
                
                texture[y, x, :] = [r, g, b, a]
        
        return texture
    
    def test_texture_pattern_creation(self, test_texture_8x8):
        """Test that our test texture has the expected pattern."""
        texture = test_texture_8x8
        
        assert texture.shape == (8, 8, 4)
        assert texture.dtype == np.float32
        
        # Check corners
        assert np.allclose(texture[0, 0, :], [0.0, 0.0, 0.0, 1.0])  # Top-left
        assert np.allclose(texture[0, 7, :], [1.0, 0.0, 0.5, 1.0])  # Top-right  
        assert np.allclose(texture[7, 0, :], [0.0, 1.0, 0.5, 1.0])  # Bottom-left
        assert np.allclose(texture[7, 7, :], [1.0, 1.0, 1.0, 1.0])  # Bottom-right
    
    @pytest.mark.skipif(not hasattr(forge3d, 'Renderer'), 
                       reason="GPU renderer not available")
    def test_sampler_mode_matrix_coverage(self):
        """Test that we can create samplers for all modes."""
        # This is a smoke test that would be expanded with actual GPU sampling
        modes = forge3d.list_sampler_modes()
        
        # For now, just verify we can generate all combinations
        assert len(modes) == 12
        
        # In a full implementation, this would:
        # 1. Create a renderer
        # 2. Upload the 8x8 test texture
        # 3. Create samplers for each mode
        # 4. Sample the texture with coordinates that test address modes
        # 5. Verify the sampling results match expected behavior
        
        # Placeholder for future GPU-based testing
        for mode in modes:
            # Verify mode is well-formed
            assert "address_mode" in mode
            assert "mag_filter" in mode  
            assert "mip_filter" in mode
            assert mode["address_mode"] in ["clamp", "repeat", "mirror"]
            assert mode["mag_filter"] in ["linear", "nearest"]
            assert mode["mip_filter"] in ["linear", "nearest"]


class TestSamplerPolicyPatterns:
    """Test common sampler policy patterns."""
    
    def test_ui_texture_pattern(self):
        """Test sampler suitable for UI textures."""
        # UI textures typically want clamping and linear filtering
        ui_sampler = forge3d.make_sampler("clamp", "linear", "nearest")
        
        assert ui_sampler["address_mode"] == "clamp"
        assert ui_sampler["mag_filter"] == "linear"
        assert ui_sampler["mip_filter"] == "nearest"
    
    def test_tiled_texture_pattern(self):
        """Test sampler suitable for tiled textures."""  
        # Tiled textures want repeat addressing and linear filtering
        tiled_sampler = forge3d.make_sampler("repeat", "linear", "linear")
        
        assert tiled_sampler["address_mode"] == "repeat"
        assert tiled_sampler["mag_filter"] == "linear"
        assert tiled_sampler["mip_filter"] == "linear"
    
    def test_pixel_art_pattern(self):
        """Test sampler suitable for pixel art."""
        # Pixel art wants no filtering at all
        pixel_sampler = forge3d.make_sampler("clamp", "nearest", "nearest")
        
        assert pixel_sampler["address_mode"] == "clamp"
        assert pixel_sampler["mag_filter"] == "nearest"
        assert pixel_sampler["min_filter"] == "nearest"
        assert pixel_sampler["mip_filter"] == "nearest"
    
    def test_heightmap_pattern(self):
        """Test sampler suitable for height maps."""
        # Height maps often want clamping with linear mag/min but no mip filtering
        heightmap_sampler = forge3d.make_sampler("clamp", "linear", "nearest")
        
        assert heightmap_sampler["address_mode"] == "clamp"
        assert heightmap_sampler["mag_filter"] == "linear"
        assert heightmap_sampler["mip_filter"] == "nearest"


if __name__ == "__main__":
    pytest.main([__file__])