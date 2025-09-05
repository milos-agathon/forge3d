"""
HDR off-screen rendering and tone mapping tests

Tests for HDR rendering configuration, tone mapping operators,
HDR statistics computation, and image quality validation.
"""

import numpy as np
import pytest
import logging

# Skip if HDR functionality not available
try:
    import forge3d.hdr as hdr
    _HDR_AVAILABLE = True
except ImportError:
    _HDR_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _HDR_AVAILABLE,
    reason="HDR module not available"
)

logger = logging.getLogger(__name__)


class TestHdrConfig:
    """Test HDR configuration class."""
    
    def test_create_hdr_config_defaults(self):
        """Test HDR configuration with default parameters."""
        config = hdr.HdrConfig()
        
        assert config.width == 1920
        assert config.height == 1080
        assert config.hdr_format == "rgba16float"
        assert config.tone_mapping == hdr.ToneMappingOperator.REINHARD
        assert config.exposure == 1.0
        assert config.white_point == 4.0
        assert config.gamma == 2.2
    
    def test_create_hdr_config_custom(self):
        """Test HDR configuration with custom parameters."""
        config = hdr.HdrConfig(
            width=512,
            height=512,
            hdr_format="rgba32float",
            tone_mapping=hdr.ToneMappingOperator.ACES,
            exposure=2.0,
            white_point=8.0,
            gamma=2.4
        )
        
        assert config.width == 512
        assert config.height == 512
        assert config.hdr_format == "rgba32float"
        assert config.tone_mapping == hdr.ToneMappingOperator.ACES
        assert config.exposure == 2.0
        assert config.white_point == 8.0
        assert config.gamma == 2.4
    
    def test_invalid_dimensions(self):
        """Test HDR configuration with invalid dimensions."""
        with pytest.raises(ValueError, match="Invalid dimensions"):
            hdr.HdrConfig(width=0, height=100)
        
        with pytest.raises(ValueError, match="Invalid dimensions"):
            hdr.HdrConfig(width=100, height=-1)
    
    def test_invalid_hdr_format(self):
        """Test HDR configuration with invalid format."""
        with pytest.raises(ValueError, match="Unsupported HDR format"):
            hdr.HdrConfig(hdr_format="invalid_format")
    
    def test_invalid_exposure(self):
        """Test HDR configuration with invalid exposure."""
        with pytest.raises(ValueError, match="Exposure must be positive"):
            hdr.HdrConfig(exposure=0.0)
        
        with pytest.raises(ValueError, match="Exposure must be positive"):
            hdr.HdrConfig(exposure=-1.0)
    
    def test_invalid_white_point(self):
        """Test HDR configuration with invalid white point."""
        with pytest.raises(ValueError, match="White point must be positive"):
            hdr.HdrConfig(white_point=0.0)
    
    def test_invalid_gamma(self):
        """Test HDR configuration with invalid gamma."""
        with pytest.raises(ValueError, match="Gamma must be positive"):
            hdr.HdrConfig(gamma=0.0)


class TestHdrRenderer:
    """Test HDR renderer functionality."""
    
    def test_create_hdr_renderer(self):
        """Test HDR renderer creation."""
        config = hdr.HdrConfig(width=64, height=64)
        renderer = hdr.HdrRenderer(config)
        
        assert renderer.config == config
        assert renderer._hdr_data is None
        assert renderer._ldr_data is None
    
    def test_render_hdr_scene(self):
        """Test HDR scene rendering."""
        config = hdr.HdrConfig(width=64, height=64)
        renderer = hdr.HdrRenderer(config)
        
        scene_data = hdr.create_hdr_test_scene(
            width=64,
            height=64,
            sun_intensity=50.0,
            sky_intensity=2.0
        )
        
        hdr_image = renderer.render_hdr_scene(scene_data)
        
        # Check output format
        assert hdr_image.shape == (64, 64, 4)
        assert hdr_image.dtype == np.float32
        
        # Check HDR range (should have values > 1.0 for HDR)
        assert np.max(hdr_image) > 1.0
        assert np.min(hdr_image) >= 0.0
        
        # Check that renderer stored the data
        assert renderer._hdr_data is not None
        np.testing.assert_array_equal(renderer._hdr_data, hdr_image)
    
    def test_apply_tone_mapping_reinhard(self):
        """Test Reinhard tone mapping."""
        config = hdr.HdrConfig(
            width=32, 
            height=32,
            tone_mapping=hdr.ToneMappingOperator.REINHARD
        )
        renderer = hdr.HdrRenderer(config)
        
        # Create test HDR data
        scene_data = hdr.create_hdr_test_scene(width=32, height=32, sun_intensity=10.0)
        hdr_image = renderer.render_hdr_scene(scene_data)
        
        # Apply tone mapping
        ldr_image = renderer.apply_tone_mapping()
        
        # Check output format
        assert ldr_image.shape == (32, 32, 4)
        assert ldr_image.dtype == np.uint8
        
        # Check LDR range
        assert np.max(ldr_image) <= 255
        assert np.min(ldr_image) >= 0
        
        # Check that values are in reasonable range
        luminance = 0.299 * ldr_image[:, :, 0] + 0.587 * ldr_image[:, :, 1] + 0.114 * ldr_image[:, :, 2]
        assert np.mean(luminance) > 10  # Should not be too dark
        assert np.mean(luminance) < 250  # Should not be too bright
    
    def test_tone_mapping_without_hdr_data(self):
        """Test tone mapping without HDR data."""
        config = hdr.HdrConfig(width=32, height=32)
        renderer = hdr.HdrRenderer(config)
        
        with pytest.raises(ValueError, match="No HDR data available"):
            renderer.apply_tone_mapping()
    
    def test_apply_tone_mapping_custom_hdr(self):
        """Test tone mapping with custom HDR data."""
        config = hdr.HdrConfig(width=16, height=16)
        renderer = hdr.HdrRenderer(config)
        
        # Create custom HDR data
        custom_hdr = np.random.rand(16, 16, 4).astype(np.float32) * 5.0  # HDR range [0, 5]
        
        ldr_image = renderer.apply_tone_mapping(custom_hdr)
        
        assert ldr_image.shape == (16, 16, 4)
        assert ldr_image.dtype == np.uint8
    
    def test_hdr_statistics(self):
        """Test HDR statistics computation."""
        config = hdr.HdrConfig(width=32, height=32)
        renderer = hdr.HdrRenderer(config)
        
        scene_data = hdr.create_hdr_test_scene(width=32, height=32, sun_intensity=100.0)
        hdr_image = renderer.render_hdr_scene(scene_data)
        
        stats = renderer.get_hdr_statistics()
        
        # Check all required statistics are present
        required_keys = [
            'min_luminance', 'max_luminance', 'mean_luminance', 'median_luminance',
            'std_luminance', 'dynamic_range', 'pixels_above_1', 'pixels_above_10', 'pixels_above_100'
        ]
        
        for key in required_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))
        
        # Check reasonable values
        assert stats['min_luminance'] >= 0
        assert stats['max_luminance'] >= stats['min_luminance']
        assert stats['mean_luminance'] > 0
        assert stats['dynamic_range'] >= 1.0
        assert stats['pixels_above_1'] >= 0
        assert stats['pixels_above_10'] >= 0
        assert stats['pixels_above_100'] >= 0


class TestToneMappingOperators:
    """Test individual tone mapping operators."""
    
    def test_all_tone_mapping_operators(self):
        """Test all tone mapping operators produce valid output."""
        # Create test HDR data
        hdr_data = np.random.rand(32, 32, 4).astype(np.float32) * 10.0  # HDR range
        
        operators = [
            hdr.ToneMappingOperator.REINHARD,
            hdr.ToneMappingOperator.REINHARD_EXTENDED,
            hdr.ToneMappingOperator.ACES,
            hdr.ToneMappingOperator.UNCHARTED2,
            hdr.ToneMappingOperator.EXPOSURE,
            hdr.ToneMappingOperator.GAMMA,
            hdr.ToneMappingOperator.CLAMP,
        ]
        
        for operator in operators:
            config = hdr.HdrConfig(
                width=32,
                height=32, 
                tone_mapping=operator
            )
            
            renderer = hdr.HdrRenderer(config)
            renderer._hdr_data = hdr_data
            
            ldr_result = renderer.apply_tone_mapping()
            
            # Check valid output
            assert ldr_result.shape == (32, 32, 4)
            assert ldr_result.dtype == np.uint8
            assert np.all(ldr_result >= 0)
            assert np.all(ldr_result <= 255)
            
            print(f"✓ {operator.value} tone mapping: OK")
    
    def test_tone_mapping_operator_differences(self):
        """Test that different tone mapping operators produce different results."""
        # Create test HDR data with high dynamic range
        hdr_data = np.zeros((32, 32, 4), dtype=np.float32)
        hdr_data[:, :, :3] = np.random.rand(32, 32, 3) * 50.0  # Wide HDR range
        hdr_data[:, :, 3] = 1.0
        
        # Test two different operators
        reinhard_config = hdr.HdrConfig(width=32, height=32, tone_mapping=hdr.ToneMappingOperator.REINHARD)
        aces_config = hdr.HdrConfig(width=32, height=32, tone_mapping=hdr.ToneMappingOperator.ACES)
        
        reinhard_renderer = hdr.HdrRenderer(reinhard_config)
        aces_renderer = hdr.HdrRenderer(aces_config)
        
        reinhard_renderer._hdr_data = hdr_data
        aces_renderer._hdr_data = hdr_data
        
        reinhard_result = reinhard_renderer.apply_tone_mapping()
        aces_result = aces_renderer.apply_tone_mapping()
        
        # Results should be different
        assert not np.array_equal(reinhard_result, aces_result)
        
        # Compute difference statistics
        reinhard_mean = np.mean(reinhard_result[:, :, :3])
        aces_mean = np.mean(aces_result[:, :, :3])
        
        difference_percent = abs(aces_mean - reinhard_mean) / max(reinhard_mean, aces_mean) * 100
        
        print(f"Reinhard mean: {reinhard_mean:.2f}")
        print(f"ACES mean: {aces_mean:.2f}")
        print(f"Difference: {difference_percent:.2f}%")
        
        # Should have measurable difference
        assert difference_percent > 1.0, f"Tone mapping operators too similar: {difference_percent:.2f}% difference"
    
    def test_exposure_effects(self):
        """Test that exposure affects tone mapping results."""
        hdr_data = np.ones((16, 16, 4), dtype=np.float32) * 5.0  # Constant HDR value
        
        low_exposure_config = hdr.HdrConfig(width=16, height=16, exposure=0.5)
        high_exposure_config = hdr.HdrConfig(width=16, height=16, exposure=2.0)
        
        low_renderer = hdr.HdrRenderer(low_exposure_config)
        high_renderer = hdr.HdrRenderer(high_exposure_config)
        
        low_renderer._hdr_data = hdr_data
        high_renderer._hdr_data = hdr_data
        
        low_result = low_renderer.apply_tone_mapping()
        high_result = high_renderer.apply_tone_mapping()
        
        # Higher exposure should generally produce brighter results
        low_mean = np.mean(low_result[:, :, :3])
        high_mean = np.mean(high_result[:, :, :3])
        
        print(f"Low exposure (0.5) mean: {low_mean:.2f}")
        print(f"High exposure (2.0) mean: {high_mean:.2f}")
        
        # High exposure should be brighter (with tone mapping, this may not always be true)
        # But they should at least be different
        assert abs(high_mean - low_mean) > 5.0, "Exposure should affect tone mapping results"


class TestAdvancedHdrFunctions:
    """Test advanced HDR utility functions."""
    
    def test_create_hdr_test_scene(self):
        """Test HDR test scene creation."""
        scene_data = hdr.create_hdr_test_scene(
            width=64,
            height=32,
            sun_intensity=100.0,
            sky_intensity=5.0
        )
        
        assert scene_data['width'] == 64
        assert scene_data['height'] == 32
        assert scene_data['sun_intensity'] == 100.0
        assert scene_data['sky_intensity'] == 5.0
        assert 'ground_intensity' in scene_data
        assert 'description' in scene_data
    
    def test_compare_tone_mapping_operators(self):
        """Test tone mapping operator comparison."""
        # Create test HDR data
        hdr_data = np.random.rand(16, 16, 4).astype(np.float32) * 20.0
        
        operators = [
            hdr.ToneMappingOperator.REINHARD,
            hdr.ToneMappingOperator.ACES,
            hdr.ToneMappingOperator.EXPOSURE,
        ]
        
        results = hdr.compare_tone_mapping_operators(hdr_data, operators, exposure=1.5)
        
        # Check results structure
        assert len(results) == len(operators)
        
        for operator in operators:
            op_name = operator.value
            assert op_name in results
            
            result = results[op_name]
            assert 'ldr_data' in result
            assert 'hdr_stats' in result
            assert 'ldr_mean' in result
            assert 'ldr_std' in result
            assert 'contrast_ratio' in result
            
            # Check data types
            assert result['ldr_data'].shape == (16, 16, 4)
            assert isinstance(result['ldr_mean'], float)
            assert isinstance(result['ldr_std'], float)
            assert isinstance(result['contrast_ratio'], float)
    
    def test_advanced_hdr_to_ldr(self):
        """Test advanced HDR to LDR conversion function."""
        # Create test HDR data (3-channel)
        hdr_data = np.random.rand(16, 16, 3).astype(np.float32) * 10.0
        
        # Test with ToneMappingOperator enum
        ldr_result = hdr.advanced_hdr_to_ldr(
            hdr_data, 
            method=hdr.ToneMappingOperator.ACES,
            exposure=2.0,
            white_point=8.0
        )
        
        # Check output
        assert ldr_result.shape == (16, 16, 4)  # Should add alpha channel
        assert ldr_result.dtype == np.float32
        assert np.all(ldr_result >= 0.0)
        assert np.all(ldr_result <= 1.0)  # Should be in [0,1] range
        
        # Test with string method (should fall back to legacy)
        ldr_legacy = hdr.advanced_hdr_to_ldr(
            hdr_data,
            method="reinhard",
            exposure=1.0
        )
        
        assert ldr_legacy.shape == (16, 16, 4)
        assert ldr_legacy.dtype == np.float32
    
    def test_has_hdr_support(self):
        """Test HDR support detection."""
        support = hdr.has_hdr_support()
        
        # Should return boolean
        assert isinstance(support, bool)
        
        # If we got here, HDR module loaded successfully, so should be True
        assert support is True


class TestHdrStatisticsValidation:
    """Test HDR statistics computation and validation."""
    
    def test_hdr_dynamic_range_calculation(self):
        """Test HDR dynamic range calculation."""
        # Create HDR data with known dynamic range
        hdr_data = np.zeros((10, 10, 4), dtype=np.float32)
        
        # Set specific luminance values
        hdr_data[0, 0, :3] = [0.1, 0.1, 0.1]  # Min luminance ≈ 0.1
        hdr_data[5, 5, :3] = [10.0, 10.0, 10.0]  # Max luminance ≈ 10.0
        hdr_data[:, :, 3] = 1.0
        
        config = hdr.HdrConfig(width=10, height=10)
        renderer = hdr.HdrRenderer(config)
        renderer._hdr_data = hdr_data
        
        stats = renderer.get_hdr_statistics()
        
        # Check dynamic range calculation
        expected_dr = 10.0 / 0.1  # Max / Min
        actual_dr = stats['dynamic_range']
        
        print(f"Expected dynamic range: {expected_dr:.2f}")
        print(f"Actual dynamic range: {actual_dr:.2f}")
        
        # Allow some tolerance for floating point calculations
        assert abs(actual_dr - expected_dr) < 1.0, f"Dynamic range calculation error: expected {expected_dr:.2f}, got {actual_dr:.2f}"
    
    def test_hdr_pixel_counting(self):
        """Test HDR pixel counting for different thresholds."""
        # Create HDR data with specific luminance distribution
        hdr_data = np.zeros((10, 10, 4), dtype=np.float32)
        
        # Create pixels with different luminance levels
        num_above_1 = 20
        num_above_10 = 10
        num_above_100 = 5
        
        # Fill in pixels systematically
        pixels = [(i // 10, i % 10) for i in range(100)]
        
        # Set pixels above 1.0
        for i in range(num_above_1):
            y, x = pixels[i]
            hdr_data[y, x, :3] = [2.0, 2.0, 2.0]  # Luminance ≈ 2.0
        
        # Set pixels above 10.0
        for i in range(num_above_10):
            y, x = pixels[i]
            hdr_data[y, x, :3] = [20.0, 20.0, 20.0]  # Luminance ≈ 20.0
        
        # Set pixels above 100.0
        for i in range(num_above_100):
            y, x = pixels[i]
            hdr_data[y, x, :3] = [200.0, 200.0, 200.0]  # Luminance ≈ 200.0
        
        hdr_data[:, :, 3] = 1.0
        
        config = hdr.HdrConfig(width=10, height=10)
        renderer = hdr.HdrRenderer(config)
        renderer._hdr_data = hdr_data
        
        stats = renderer.get_hdr_statistics()
        
        print(f"Pixels above 1: expected {num_above_1}, got {stats['pixels_above_1']}")
        print(f"Pixels above 10: expected {num_above_10}, got {stats['pixels_above_10']}")
        print(f"Pixels above 100: expected {num_above_100}, got {stats['pixels_above_100']}")
        
        # Check pixel counts (allow small tolerance due to luminance calculation)
        assert abs(stats['pixels_above_1'] - num_above_1) <= 2
        assert abs(stats['pixels_above_10'] - num_above_10) <= 2
        assert abs(stats['pixels_above_100'] - num_above_100) <= 2


if __name__ == "__main__":
    # Run with verbose output to capture logs for debugging
    pytest.main([__file__, "-v", "-s"])