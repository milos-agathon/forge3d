"""
Tests for forge3d.adapters.rasterio_tiles - Windowed reading and block iteration.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

# Test the module can be imported and handles missing dependencies gracefully
def test_import_without_rasterio():
    """Test that module can be imported even without rasterio."""
    with patch.dict('sys.modules', {'rasterio': None}):
        import forge3d.adapters.rasterio_tiles as rt
        assert not rt.is_rasterio_available()


@pytest.mark.skipif(
    condition=True,  # Skip by default since rasterio is optional
    reason="rasterio not available - install with 'pip install forge3d[raster]'"
)
def test_rasterio_functionality():
    """Test core rasterio functionality when available."""
    try:
        import rasterio
        from rasterio.windows import Window
        import forge3d.adapters.rasterio_tiles as rt
    except ImportError:
        pytest.skip("rasterio not available")
    
    assert rt.is_rasterio_available()
    
    # Create synthetic in-memory dataset for testing
    with patch('forge3d.adapters.rasterio_tiles.rasterio') as mock_rio:
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.width = 1024
        mock_dataset.height = 1024
        mock_dataset.count = 3
        mock_dataset.dtype = np.float32
        mock_dataset.transform = rasterio.transform.from_bounds(0, 0, 10, 10, 1024, 1024)
        mock_dataset.bounds = (0, 0, 10, 10)
        mock_dataset.nodata = None
        mock_dataset.block_shapes = [(256, 256)] * 3
        mock_dataset.overviews.return_value = [2, 4, 8]
        
        # Mock successful read
        test_data = np.random.rand(3, 256, 256).astype(np.float32)
        mock_dataset.read.return_value = test_data
        
        # Test windowed read
        window = Window(0, 0, 256, 256)
        result = rt.windowed_read(mock_dataset, window)
        
        assert result.shape == test_data.shape
        mock_dataset.read.assert_called_once()


def test_rasterio_graceful_degradation():
    """Test that functions raise appropriate errors when rasterio is not available."""
    with patch('forge3d.adapters.rasterio_tiles._HAS_RASTERIO', False):
        import forge3d.adapters.rasterio_tiles as rt
        
        with pytest.raises(ImportError, match="rasterio is required"):
            rt.windowed_read(None, None)
            
        with pytest.raises(ImportError, match="rasterio is required"):
            list(rt.block_iterator(None))


def test_synthetic_mask_creation():
    """Test synthetic mask creation (doesn't require rasterio)."""
    from forge3d.adapters.rasterio_tiles import create_synthetic_nodata_mask
    
    # Test mask creation
    shape = (100, 100)
    mask = create_synthetic_nodata_mask(shape, nodata_fraction=0.1, seed=42)
    
    assert mask.shape == shape
    assert mask.dtype == np.uint8
    assert np.all((mask == 0) | (mask == 255))
    
    # Check approximate nodata fraction
    nodata_pixels = np.sum(mask == 0)
    total_pixels = shape[0] * shape[1]
    nodata_fraction = nodata_pixels / total_pixels
    assert 0.05 < nodata_fraction < 0.15  # Roughly 10% with some tolerance
    
    # Test reproducibility
    mask2 = create_synthetic_nodata_mask(shape, nodata_fraction=0.1, seed=42)
    np.testing.assert_array_equal(mask, mask2)


def test_overview_functions_without_rasterio():
    """Test that overview functions handle missing rasterio gracefully."""
    with patch('forge3d.adapters.rasterio_tiles._HAS_RASTERIO', False):
        from forge3d.adapters.rasterio_tiles import (
            select_overview_level, 
            windowed_read_with_overview,
            calculate_overview_savings
        )
        
        with pytest.raises(ImportError, match="rasterio is required"):
            select_overview_level(None, 10.0)
            
        with pytest.raises(ImportError, match="rasterio is required"):
            windowed_read_with_overview(None, None)
            
        with pytest.raises(ImportError, match="rasterio is required"):
            calculate_overview_savings(None, [])


@pytest.mark.skipif(
    condition=True,  # Skip by default since rasterio is optional  
    reason="rasterio not available - install with 'pip install forge3d[raster]'"
)
def test_overview_selection_logic():
    """Test overview selection logic when rasterio is available."""
    try:
        import rasterio
        from forge3d.adapters.rasterio_tiles import select_overview_level
    except ImportError:
        pytest.skip("rasterio not available")
    
    # Mock dataset with overviews
    mock_dataset = Mock()
    mock_dataset.transform = rasterio.transform.from_bounds(0, 0, 10, 10, 1000, 1000)
    mock_dataset.width = 1000
    mock_dataset.height = 1000
    mock_dataset.overviews.return_value = [2, 4, 8, 16]
    
    # Test selection for different target resolutions
    dataset_res = 0.01  # 10m / 1000px = 0.01 per pixel
    
    # Target resolution similar to full res - should select full res
    overview_idx, overview_info = select_overview_level(mock_dataset, dataset_res)
    assert overview_idx == -1  # Full resolution
    assert overview_info['overview_factor'] == 1
    
    # Target resolution 4x coarser - should select overview level 4x
    overview_idx, overview_info = select_overview_level(mock_dataset, dataset_res * 4)
    assert overview_info['overview_factor'] == 4
    assert overview_info['bytes_reduction'] > 0.9  # Should be about 93.75% reduction


@pytest.mark.skipif(
    condition=True,  # Skip by default since rasterio is optional
    reason="rasterio not available - install with 'pip install forge3d[raster]'"
)
def test_mask_extraction():
    """Test mask extraction functionality."""
    try:
        from forge3d.adapters.rasterio_tiles import extract_masks
        import rasterio
    except ImportError:
        pytest.skip("rasterio not available")
    
    # Mock dataset with masks
    mock_dataset = Mock()
    mock_dataset.dataset_mask.return_value = np.full((100, 100), 255, dtype=np.uint8)
    mock_dataset.count = 1
    mock_dataset.width = 100 
    mock_dataset.height = 100
    
    # Test mask extraction
    mask = extract_masks(mock_dataset)
    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    assert np.all(mask == 255)


def test_alpha_channel_processing():
    """Test RGBA processing logic (using mocked data)."""
    with patch('forge3d.adapters.rasterio_tiles._HAS_RASTERIO', True):
        with patch('forge3d.adapters.rasterio_tiles.windowed_read') as mock_read:
            with patch('forge3d.adapters.rasterio_tiles.extract_masks') as mock_mask:
                from forge3d.adapters.rasterio_tiles import windowed_read_with_alpha
                from rasterio.windows import Window
                
                # Mock RGB data
                rgb_data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                mock_read.return_value = rgb_data
                
                # Mock valid mask
                mask_data = np.full((64, 64), 255, dtype=np.uint8)
                mock_mask.return_value = mask_data
                
                # Test RGBA creation
                result = windowed_read_with_alpha(
                    Mock(), Window(0, 0, 64, 64), add_alpha=True
                )
                
                assert result.shape == (64, 64, 4)
                assert result.dtype == np.uint8
                assert np.all(result[:, :, 3] == 255)  # All opaque


if __name__ == "__main__":
    # Run basic tests that don't require rasterio
    test_import_without_rasterio()
    test_rasterio_graceful_degradation()
    test_synthetic_mask_creation()
    test_overview_functions_without_rasterio()
    test_alpha_channel_processing()
    
    print("Basic rasterio adapter tests passed!")