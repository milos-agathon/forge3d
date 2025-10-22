"""
Integration tests for terrain draping pipeline.

Tests the complete workflow from DEM + land-cover to rendered output.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for forge3d import
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    import forge3d
    from forge3d.terrain import drape_landcover, estimate_memory_usage
    FORGE3D_AVAILABLE = True
except ImportError:
    FORGE3D_AVAILABLE = False


# Mark all tests to skip if forge3d is not available
pytestmark = pytest.mark.skipif(not FORGE3D_AVAILABLE, reason="forge3d not installed")


class TestTerrainDrape:
    """Test suite for terrain draping functionality."""
    
    def test_basic_render(self):
        """Test basic terrain draping with synthetic data."""
        # Create synthetic DEM (512x512 with some hills)
        x = np.linspace(-10, 10, 512)
        y = np.linspace(-10, 10, 512)
        X, Y = np.meshgrid(x, y)
        dem = (100 * np.exp(-(X**2 + Y**2) / 50)).astype(np.float32)
        
        # Create synthetic land-cover (green terrain)
        landcover = np.zeros((512, 512, 4), dtype=np.uint8)
        landcover[:, :, 1] = 180  # Green
        landcover[:, :, 3] = 255  # Alpha
        
        # Render
        result = drape_landcover(
            dem,
            landcover,
            width=640,
            height=480,
            height_scale=1.0,
        )
        
        # Validate output
        assert result.shape == (480, 640, 4)
        assert result.dtype == np.uint8
        
        # Check that we got some non-background pixels
        non_white = np.any(result[:, :, :3] != 255, axis=2)
        assert non_white.sum() > 1000, "Output appears mostly empty"
    
    def test_input_validation(self):
        """Test input validation and error handling."""
        dem = np.random.rand(100, 100).astype(np.float32)
        landcover = np.zeros((100, 100, 4), dtype=np.uint8)
        
        # Test mismatched dimensions
        bad_landcover = np.zeros((50, 50, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="same dimensions"):
            drape_landcover(dem, bad_landcover)
        
        # Test wrong landcover shape (missing alpha channel)
        bad_landcover = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="RGBA"):
            drape_landcover(dem, bad_landcover)
        
        # Test wrong DEM dimensions
        bad_dem = np.random.rand(100, 100, 1).astype(np.float32)
        with pytest.raises(ValueError, match="2D"):
            drape_landcover(bad_dem, landcover)
    
    def test_height_scale(self):
        """Test vertical exaggeration."""
        # Create flat terrain with one peak
        dem = np.zeros((256, 256), dtype=np.float32)
        dem[128, 128] = 100.0  # Single peak
        
        landcover = np.zeros((256, 256, 4), dtype=np.uint8)
        landcover[:, :] = [100, 150, 100, 255]  # Green
        
        # Render with different height scales
        result_1x = drape_landcover(dem, landcover, width=320, height=240, height_scale=1.0)
        result_2x = drape_landcover(dem, landcover, width=320, height=240, height_scale=2.0)
        
        # Results should be different (higher exaggeration changes shading)
        assert not np.array_equal(result_1x, result_2x)
    
    def test_camera_angles(self):
        """Test different camera angles produce different views."""
        dem = np.random.rand(128, 128).astype(np.float32) * 100
        landcover = np.zeros((128, 128, 4), dtype=np.uint8)
        landcover[:, :] = [80, 120, 80, 255]
        
        # Different camera angles
        result_45 = drape_landcover(dem, landcover, width=320, height=240, camera_theta=45.0)
        result_135 = drape_landcover(dem, landcover, width=320, height=240, camera_theta=135.0)
        
        # Should produce different views
        diff = np.abs(result_45.astype(float) - result_135.astype(float)).mean()
        assert diff > 1.0, "Different camera angles should produce different views"
    
    def test_categorical_preservation(self):
        """Test that categorical colors are preserved (nearest sampling)."""
        dem = np.zeros((256, 256), dtype=np.float32)
        
        # Create land-cover with distinct color blocks
        landcover = np.zeros((256, 256, 4), dtype=np.uint8)
        landcover[:128, :128] = [255, 0, 0, 255]    # Red
        landcover[:128, 128:] = [0, 255, 0, 255]    # Green
        landcover[128:, :128] = [0, 0, 255, 255]    # Blue
        landcover[128:, 128:] = [255, 255, 0, 255]  # Yellow
        
        result = drape_landcover(dem, landcover, width=512, height=512)
        
        # Check that primary colors are present (within lighting tolerance)
        result_rgb = result[:, :, :3]
        
        # Find dominant colors (excluding black and white)
        unique_colors = np.unique(result_rgb.reshape(-1, 3), axis=0)
        colored = unique_colors[(unique_colors.sum(axis=1) > 50) & (unique_colors.sum(axis=1) < 700)]
        
        # Should have at least a few distinct colors from our 4 input colors
        assert len(colored) >= 4, f"Expected distinct colors, got {len(colored)}"
    
    def test_memory_estimation(self):
        """Test memory usage estimation."""
        mem_small = estimate_memory_usage((512, 512))
        mem_large = estimate_memory_usage((4096, 4096))
        
        # Larger should use more memory
        assert mem_large['total_mb'] > mem_small['total_mb']
        
        # Check 4k×4k doesn't exceed budget
        assert not mem_large['exceeds_512mb_budget']
        
        # Check 8k×8k does exceed budget
        mem_huge = estimate_memory_usage((8192, 8192))
        assert mem_huge['exceeds_512mb_budget']
    
    def test_nan_handling(self):
        """Test handling of NaN values in DEM."""
        dem = np.random.rand(128, 128).astype(np.float32) * 100
        dem[50:60, 50:60] = np.nan  # Add NaN region
        
        landcover = np.zeros((128, 128, 4), dtype=np.uint8)
        landcover[:, :] = [100, 150, 100, 255]
        
        # Should raise error for NaN
        with pytest.raises(ValueError, match="non-finite"):
            drape_landcover(dem, landcover)
    
    def test_output_size(self):
        """Test that output size matches requested dimensions."""
        dem = np.random.rand(256, 256).astype(np.float32) * 100
        landcover = np.zeros((256, 256, 4), dtype=np.uint8)
        landcover[:, :] = [100, 150, 100, 255]
        
        # Test various output sizes
        for w, h in [(640, 480), (1280, 720), (1920, 1080)]:
            result = drape_landcover(dem, landcover, width=w, height=h)
            assert result.shape == (h, w, 4), f"Expected {(h, w, 4)}, got {result.shape}"


class TestTerrainDrapeReference:
    """Reference image comparison tests."""
    
    @pytest.fixture
    def reference_data(self, tmp_path):
        """Create reference test data."""
        # Synthetic mountain terrain
        x = np.linspace(-5, 5, 256)
        y = np.linspace(-5, 5, 256)
        X, Y = np.meshgrid(x, y)
        
        # Multiple peaks
        dem = (
            200 * np.exp(-((X-2)**2 + (Y-2)**2) / 2) +
            150 * np.exp(-((X+2)**2 + (Y+2)**2) / 3) +
            100 * np.exp(-(X**2 + Y**2) / 4)
        ).astype(np.float32)
        
        # Land-cover: water at low elevation, grass at mid, snow at high
        landcover = np.zeros((256, 256, 4), dtype=np.uint8)
        
        water_mask = dem < 50
        grass_mask = (dem >= 50) & (dem < 150)
        snow_mask = dem >= 150
        
        landcover[water_mask] = [30, 144, 255, 255]   # Blue water
        landcover[grass_mask] = [34, 139, 34, 255]    # Green grass
        landcover[snow_mask] = [240, 248, 255, 255]   # White snow
        
        return dem, landcover, tmp_path
    
    def test_reference_render(self, reference_data):
        """Test rendering against reference data."""
        dem, landcover, tmp_path = reference_data
        
        result = drape_landcover(
            dem,
            landcover,
            width=512,
            height=512,
            height_scale=1.5,
            camera_theta=45.0,
            camera_phi=30.0,
        )
        
        # Save reference (for manual inspection during development)
        output_path = tmp_path / "terrain_reference.png"
        from PIL import Image
        Image.fromarray(result).save(output_path)
        
        # Validate basic properties
        assert result.shape == (512, 512, 4)
        
        # Check we have all three color regions represented
        result_rgb = result[:, :, :3]
        mean_b = result_rgb[:, :, 2].mean()  # Blue channel for water
        mean_g = result_rgb[:, :, 1].mean()  # Green channel for grass
        
        # Should have significant blue and green components
        assert mean_b > 30, "Expected water (blue) in scene"
        assert mean_g > 30, "Expected grass (green) in scene"
        
        print(f"\n✓ Reference render saved to {output_path}")
    
    def test_deterministic_output(self):
        """Test that rendering is deterministic."""
        dem = np.random.RandomState(42).rand(128, 128).astype(np.float32) * 100
        landcover = np.zeros((128, 128, 4), dtype=np.uint8)
        landcover[:, :] = [100, 150, 100, 255]
        
        result1 = drape_landcover(dem, landcover, width=256, height=256)
        result2 = drape_landcover(dem, landcover, width=256, height=256)
        
        # Results should be identical
        np.testing.assert_array_equal(result1, result2)


@pytest.mark.benchmark
class TestTerrainDrapePerformance:
    """Performance benchmarks for terrain draping."""
    
    def test_render_performance_small(self, benchmark):
        """Benchmark small terrain (512x512)."""
        dem = np.random.rand(512, 512).astype(np.float32) * 100
        landcover = np.zeros((512, 512, 4), dtype=np.uint8)
        landcover[:, :] = [100, 150, 100, 255]
        
        def render():
            return drape_landcover(dem, landcover, width=640, height=480)
        
        result = benchmark(render)
        assert result.shape == (480, 640, 4)
    
    def test_render_performance_large(self, benchmark):
        """Benchmark large terrain (2048x2048)."""
        dem = np.random.rand(2048, 2048).astype(np.float32) * 100
        landcover = np.zeros((2048, 2048, 4), dtype=np.uint8)
        landcover[:, :] = [100, 150, 100, 255]
        
        def render():
            return drape_landcover(dem, landcover, width=1920, height=1080)
        
        result = benchmark(render)
        assert result.shape == (1080, 1920, 4)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
