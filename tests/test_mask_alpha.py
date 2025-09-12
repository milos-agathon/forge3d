"""
Tests for mask to alpha channel functionality in rasterio adapter.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

def test_synthetic_mask_properties():
    """Test properties of synthetic masks."""
    from forge3d.adapters.rasterio_tiles import create_synthetic_nodata_mask
    
    # Test different nodata fractions
    shape = (50, 50)
    for fraction in [0.0, 0.1, 0.3, 0.5]:
        mask = create_synthetic_nodata_mask(shape, nodata_fraction=fraction, seed=123)
        
        nodata_count = np.sum(mask == 0)
        total_pixels = shape[0] * shape[1]
        actual_fraction = nodata_count / total_pixels
        
        # Allow some tolerance due to random sampling
        tolerance = 0.05 if fraction > 0 else 0.001
        assert abs(actual_fraction - fraction) < tolerance
        
        # Check valid values
        assert np.all((mask == 0) | (mask == 255))


def test_alpha_channel_synthesis():
    """Test alpha channel creation from masks."""
    with patch('forge3d.adapters.rasterio_tiles._HAS_RASTERIO', True):
        from forge3d.adapters.rasterio_tiles import windowed_read_with_alpha
        from rasterio.windows import Window
        
        mock_dataset = Mock()
        
        with patch('forge3d.adapters.rasterio_tiles.windowed_read') as mock_read:
            with patch('forge3d.adapters.rasterio_tiles.extract_masks') as mock_mask:
                # Test case 1: RGB data with partial mask
                rgb_data = np.full((32, 32, 3), 128, dtype=np.uint8)
                mock_read.return_value = rgb_data
                
                # Create mask with some nodata areas
                mask = np.full((32, 32), 255, dtype=np.uint8)
                mask[10:20, 10:20] = 0  # 10x10 nodata region
                mock_mask.return_value = mask
                
                result = windowed_read_with_alpha(
                    mock_dataset, 
                    Window(0, 0, 32, 32),
                    add_alpha=True
                )
                
                assert result.shape == (32, 32, 4)
                assert result.dtype == np.uint8
                
                # Check alpha values
                assert np.all(result[10:20, 10:20, 3] == 0)  # Nodata region transparent
                assert np.all(result[:10, :, 3] == 255)  # Valid region opaque
                assert np.all(result[20:, :, 3] == 255)  # Valid region opaque
                
                # Check color channels are preserved
                np.testing.assert_array_equal(result[:, :, :3], rgb_data)


def test_grayscale_to_rgba_conversion():
    """Test conversion of grayscale data to RGBA."""
    with patch('forge3d.adapters.rasterio_tiles._HAS_RASTERIO', True):
        from forge3d.adapters.rasterio_tiles import windowed_read_with_alpha
        from rasterio.windows import Window
        
        mock_dataset = Mock()
        
        with patch('forge3d.adapters.rasterio_tiles.windowed_read') as mock_read:
            with patch('forge3d.adapters.rasterio_tiles.extract_masks') as mock_mask:
                # Grayscale input data
                gray_data = np.random.randint(0, 255, (16, 16), dtype=np.uint8)
                mock_read.return_value = gray_data
                
                # Full valid mask
                mask = np.full((16, 16), 255, dtype=np.uint8)
                mock_mask.return_value = mask
                
                result = windowed_read_with_alpha(
                    mock_dataset,
                    Window(0, 0, 16, 16),
                    add_alpha=True
                )
                
                assert result.shape == (16, 16, 4)
                
                # Check that all RGB channels are the same (grayscale)
                np.testing.assert_array_equal(result[:, :, 0], gray_data)
                np.testing.assert_array_equal(result[:, :, 1], gray_data)
                np.testing.assert_array_equal(result[:, :, 2], gray_data)
                
                # Check alpha is fully opaque
                assert np.all(result[:, :, 3] == 255)


def test_multiband_mask_combination():
    """Test combining multiple band masks."""
    with patch('forge3d.adapters.rasterio_tiles._HAS_RASTERIO', True):
        from forge3d.adapters.rasterio_tiles import windowed_read_with_alpha
        from rasterio.windows import Window
        
        mock_dataset = Mock()
        
        with patch('forge3d.adapters.rasterio_tiles.windowed_read') as mock_read:
            with patch('forge3d.adapters.rasterio_tiles.extract_masks') as mock_mask:
                # RGB input
                rgb_data = np.random.randint(0, 255, (24, 24, 3), dtype=np.uint8)
                mock_read.return_value = rgb_data
                
                # Multi-band masks with different invalid regions
                masks = np.full((3, 24, 24), 255, dtype=np.uint8)
                masks[0, 0:8, :] = 0    # Band 1 invalid in top region
                masks[1, 8:16, :] = 0   # Band 2 invalid in middle region  
                masks[2, 16:24, :] = 0  # Band 3 invalid in bottom region
                mock_mask.return_value = masks
                
                result = windowed_read_with_alpha(
                    mock_dataset,
                    Window(0, 0, 24, 24),
                    add_alpha=True
                )
                
                assert result.shape == (24, 24, 4)
                
                # Alpha should be 0 where ANY band is invalid (AND combination)
                assert np.all(result[0:8, :, 3] == 0)    # Top invalid
                assert np.all(result[8:16, :, 3] == 0)   # Middle invalid
                assert np.all(result[16:24, :, 3] == 0)  # Bottom invalid


def test_no_alpha_option():
    """Test bypassing alpha channel creation."""
    with patch('forge3d.adapters.rasterio_tiles._HAS_RASTERIO', True):
        from forge3d.adapters.rasterio_tiles import windowed_read_with_alpha
        from rasterio.windows import Window
        
        mock_dataset = Mock()
        
        with patch('forge3d.adapters.rasterio_tiles.windowed_read') as mock_read:
            # RGB input
            rgb_data = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
            mock_read.return_value = rgb_data
            
            result = windowed_read_with_alpha(
                mock_dataset,
                Window(0, 0, 20, 20),
                add_alpha=False
            )
            
            # Should return original RGB data without alpha
            np.testing.assert_array_equal(result, rgb_data)
            assert result.shape == (20, 20, 3)


def test_mask_extraction_fallback():
    """Test fallback behavior when mask extraction fails."""
    with patch('forge3d.adapters.rasterio_tiles._HAS_RASTERIO', True):
        from forge3d.adapters.rasterio_tiles import windowed_read_with_alpha
        from rasterio.windows import Window
        
        mock_dataset = Mock()
        
        with patch('forge3d.adapters.rasterio_tiles.windowed_read') as mock_read:
            with patch('forge3d.adapters.rasterio_tiles.extract_masks') as mock_mask:
                # RGB input
                rgb_data = np.random.randint(0, 255, (12, 12, 3), dtype=np.uint8)
                mock_read.return_value = rgb_data
                
                # Make mask extraction fail
                mock_mask.side_effect = Exception("Mock mask read failure")
                
                with pytest.warns(UserWarning, match="Failed to create alpha channel"):
                    result = windowed_read_with_alpha(
                        mock_dataset,
                        Window(0, 0, 12, 12),
                        add_alpha=True
                    )
                
                assert result.shape == (12, 12, 4)
                # Should default to fully opaque alpha
                assert np.all(result[:, :, 3] == 255)


def test_dtype_conversions():
    """Test data type conversions in alpha processing."""
    with patch('forge3d.adapters.rasterio_tiles._HAS_RASTERIO', True):
        from forge3d.adapters.rasterio_tiles import windowed_read_with_alpha
        from rasterio.windows import Window
        
        mock_dataset = Mock()
        
        with patch('forge3d.adapters.rasterio_tiles.windowed_read') as mock_read:
            with patch('forge3d.adapters.rasterio_tiles.extract_masks') as mock_mask:
                # Float input data
                float_data = np.random.rand(8, 8, 3).astype(np.float32)
                mock_read.return_value = float_data
                
                # Valid mask
                mask = np.full((8, 8), 255, dtype=np.uint8)
                mock_mask.return_value = mask
                
                result = windowed_read_with_alpha(
                    mock_dataset,
                    Window(0, 0, 8, 8),
                    dtype='uint8',
                    add_alpha=True
                )
                
                assert result.shape == (8, 8, 4)
                assert result.dtype == np.uint8
                
                # Check conversion from float [0,1] to uint8 [0,255]
                expected = np.clip(float_data * 255, 0, 255).astype(np.uint8)
                np.testing.assert_array_equal(result[:, :, :3], expected)


if __name__ == "__main__":
    test_synthetic_mask_properties()
    test_alpha_channel_synthesis()
    test_grayscale_to_rgba_conversion()
    test_multiband_mask_combination()
    test_no_alpha_option()
    test_mask_extraction_fallback()
    test_dtype_conversions()
    
    print("Mask/alpha tests passed!")