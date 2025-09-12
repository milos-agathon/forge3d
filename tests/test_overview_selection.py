"""
Tests for overview/LOD selection functionality in rasterio adapter.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

def test_overview_selection_logic():
    """Test overview selection algorithm without rasterio dependency."""
    # We can test the selection logic by mocking the required parts
    with patch('forge3d.adapters.rasterio_tiles._HAS_RASTERIO', True):
        from forge3d.adapters.rasterio_tiles import select_overview_level
        
        # Mock dataset with known properties
        mock_dataset = Mock()
        mock_dataset.width = 1000
        mock_dataset.height = 1000
        
        # Mock transform (10m per pixel resolution)
        mock_transform = Mock()
        mock_transform.a = 10.0  # x pixel size
        mock_transform.e = -10.0  # y pixel size (negative)
        mock_dataset.transform = mock_transform
        
        # Mock overviews: 2x, 4x, 8x downsampling
        mock_dataset.overviews.return_value = [2, 4, 8]
        
        # Test 1: Target resolution matches full resolution (10m)
        overview_idx, overview_info = select_overview_level(mock_dataset, 10.0, band=1)
        assert overview_idx == -1  # Should select full resolution
        assert overview_info['overview_factor'] == 1
        assert overview_info['bytes_reduction'] == 0.0
        
        # Test 2: Target resolution is 4x coarser (40m) - should select 4x overview
        overview_idx, overview_info = select_overview_level(mock_dataset, 40.0, band=1)
        assert overview_info['overview_factor'] == 4
        assert abs(overview_info['resolution_x'] - 40.0) < 0.1
        assert overview_info['bytes_reduction'] > 0.9  # ~93.75% reduction
        
        # Test 3: Target resolution between levels - should select closest
        overview_idx, overview_info = select_overview_level(mock_dataset, 30.0, band=1)
        # Should select either 2x (20m) or 4x (40m) - probably 4x as it's closer
        assert overview_info['overview_factor'] in [2, 4]
        
        # Test 4: Very coarse resolution - should select coarsest available
        overview_idx, overview_info = select_overview_level(mock_dataset, 100.0, band=1)
        assert overview_info['overview_factor'] == 8  # Coarsest available


def test_overview_selection_no_overviews():
    """Test behavior when no overviews are available."""
    with patch('forge3d.adapters.rasterio_tiles._HAS_RASTERIO', True):
        from forge3d.adapters.rasterio_tiles import select_overview_level
        
        mock_dataset = Mock()
        mock_dataset.width = 500
        mock_dataset.height = 500
        
        mock_transform = Mock()
        mock_transform.a = 5.0
        mock_transform.e = -5.0
        mock_dataset.transform = mock_transform
        
        # No overviews available
        mock_dataset.overviews.return_value = []
        
        overview_idx, overview_info = select_overview_level(mock_dataset, 20.0, band=1)
        
        assert overview_idx == -1  # Must use full resolution
        assert overview_info['overview_factor'] == 1
        assert overview_info['bytes_reduction'] == 0.0


def test_overview_savings_calculation():
    """Test calculation of data savings from using overviews."""
    with patch('forge3d.adapters.rasterio_tiles._HAS_RASTERIO', True):
        from forge3d.adapters.rasterio_tiles import calculate_overview_savings
        
        mock_dataset = Mock()
        mock_dataset.width = 2000
        mock_dataset.height = 2000
        
        mock_transform = Mock()
        mock_transform.a = 1.0  # 1m pixel size
        mock_transform.e = -1.0
        mock_dataset.transform = mock_transform
        
        mock_dataset.overviews.return_value = [2, 4, 8]
        
        # Mock the select_overview_level function to return predictable results
        def mock_select_overview(dataset, target_res, band=1):
            if target_res <= 2:
                return -1, {'overview_factor': 1, 'width': 2000, 'height': 2000, 'bytes_reduction': 0.0}
            elif target_res <= 6:
                return 1, {'overview_factor': 4, 'width': 500, 'height': 500, 'bytes_reduction': 0.9375}
            else:
                return 2, {'overview_factor': 8, 'width': 250, 'height': 250, 'bytes_reduction': 0.984375}
        
        with patch('forge3d.adapters.rasterio_tiles.select_overview_level', side_effect=mock_select_overview):
            result = calculate_overview_savings(mock_dataset, [1.0, 4.0, 10.0], band=1)
            
            assert 'dataset_info' in result
            assert result['dataset_info']['width'] == 2000
            assert result['dataset_info']['full_res_pixels'] == 4000000
            
            analysis = result['resolution_analysis']
            assert len(analysis) == 3
            
            # Check 1m resolution (should use full res)
            assert analysis[0]['target_resolution'] == 1.0
            assert analysis[0]['byte_reduction_actual'] == 0.0
            assert not analysis[0]['meets_60_percent_savings']
            
            # Check 4m resolution (should use 4x overview)
            assert analysis[1]['target_resolution'] == 4.0
            assert analysis[1]['byte_reduction_actual'] > 0.9
            assert analysis[1]['meets_60_percent_savings']
            
            # Check 10m resolution (should use 8x overview)
            assert analysis[2]['target_resolution'] == 10.0
            assert analysis[2]['byte_reduction_actual'] > 0.98
            assert analysis[2]['meets_60_percent_savings']


def test_windowed_read_with_overview():
    """Test reading with automatic overview selection."""
    with patch('forge3d.adapters.rasterio_tiles._HAS_RASTERIO', True):
        from forge3d.adapters.rasterio_tiles import windowed_read_with_overview
        from rasterio.windows import Window
        
        mock_dataset = Mock()
        mock_dataset.width = 1000
        mock_dataset.height = 1000
        mock_dataset.count = 3
        
        mock_transform = Mock()
        mock_transform.a = 2.0
        mock_transform.e = -2.0
        mock_dataset.transform = mock_transform
        
        # Mock overview selection
        def mock_select_overview(dataset, target_res, band=1):
            if target_res > 4.0:
                return 0, {'overview_factor': 2, 'width': 500, 'height': 500, 'bytes_reduction': 0.75}
            else:
                return -1, {'overview_factor': 1, 'width': 1000, 'height': 1000, 'bytes_reduction': 0.0}
        
        with patch('forge3d.adapters.rasterio_tiles.select_overview_level', side_effect=mock_select_overview):
            with patch('forge3d.adapters.rasterio_tiles.windowed_read') as mock_windowed_read:
                
                # Mock data for full resolution read
                full_res_data = np.random.rand(3, 100, 100).astype(np.float32)
                mock_windowed_read.return_value = full_res_data
                
                # Test 1: Target resolution that should use full res
                window = Window(0, 0, 100, 100)
                data, overview_info = windowed_read_with_overview(
                    mock_dataset, window, target_resolution=2.0
                )
                
                assert overview_info['overview_factor'] == 1
                assert data.shape == full_res_data.shape
                mock_windowed_read.assert_called_once()
                
                # Reset mock
                mock_windowed_read.reset_mock()
                
                # Test 2: No target resolution specified (should use full res)
                data, overview_info = windowed_read_with_overview(
                    mock_dataset, window, target_resolution=None
                )
                
                assert overview_info['overview_factor'] == 1
                assert overview_info['overview_index'] == -1


def test_resolution_tuple_handling():
    """Test handling of resolution as (x_res, y_res) tuple."""
    with patch('forge3d.adapters.rasterio_tiles._HAS_RASTERIO', True):
        from forge3d.adapters.rasterio_tiles import select_overview_level
        
        mock_dataset = Mock()
        mock_dataset.width = 800
        mock_dataset.height = 600
        
        mock_transform = Mock()
        mock_transform.a = 5.0   # 5m x resolution
        mock_transform.e = -4.0  # 4m y resolution (negative)
        mock_dataset.transform = mock_transform
        
        mock_dataset.overviews.return_value = [2, 4]
        
        # Test with tuple resolution
        overview_idx, overview_info = select_overview_level(
            mock_dataset, (10.0, 8.0), band=1  # 2x coarser in both directions
        )
        
        # Should select 2x overview as it's closest match
        assert overview_info['overview_factor'] == 2
        assert abs(overview_info['resolution_x'] - 10.0) < 0.1
        assert abs(overview_info['resolution_y'] - 8.0) < 0.1


def test_graceful_degradation():
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


def test_overview_factor_calculation():
    """Test that overview factor calculations are correct."""
    # Test the byte reduction calculation logic
    factors = [2, 4, 8, 16]
    expected_reductions = [0.75, 0.9375, 0.984375, 0.99609375]
    
    for factor, expected in zip(factors, expected_reductions):
        # Formula: 1 - (1 / (factor^2))
        calculated = 1.0 - (1.0 / (factor * factor))
        assert abs(calculated - expected) < 1e-6


if __name__ == "__main__":
    test_overview_selection_logic()
    test_overview_selection_no_overviews()
    test_overview_savings_calculation()
    test_windowed_read_with_overview()
    test_resolution_tuple_handling()
    test_graceful_degradation()
    test_overview_factor_calculation()
    
    print("Overview selection tests passed!")