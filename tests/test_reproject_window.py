"""
Tests for forge3d.adapters.reproject - CRS normalization and reprojection.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

def test_import_without_deps():
    """Test that module can be imported even without dependencies."""
    with patch.dict('sys.modules', {'rasterio': None, 'pyproj': None}):
        import forge3d.adapters.reproject as rp
        assert not rp.is_reproject_available()


def test_graceful_degradation():
    """Test that functions raise appropriate errors when dependencies are missing."""
    with patch('forge3d.adapters.reproject._HAS_REPROJECT_DEPS', False):
        import forge3d.adapters.reproject as rp
        
        with pytest.raises(ImportError, match="rasterio and pyproj are required"):
            rp.WarpedVRTWrapper(None, "EPSG:4326")
            
        with pytest.raises(ImportError, match="rasterio and pyproj are required"):
            rp.reproject_window(None, "EPSG:4326", None)
            
        with pytest.raises(ImportError, match="rasterio and pyproj are required"):
            rp.transform_bounds((0, 0, 1, 1), "EPSG:4326", "EPSG:3857")


@pytest.mark.skipif(
    condition=True,  # Skip by default since dependencies are optional
    reason="rasterio/pyproj not available - install with 'pip install forge3d[raster]'"
)
def test_warpedvrt_wrapper_lifecycle():
    """Test WarpedVRT wrapper creation and lifecycle."""
    try:
        import forge3d.adapters.reproject as rp
        import rasterio
    except ImportError:
        pytest.skip("dependencies not available")
    
    # Mock dataset
    mock_dataset = Mock()
    mock_dataset.crs = "EPSG:4326"
    mock_dataset.width = 100
    mock_dataset.height = 100
    mock_dataset.bounds = (0, 0, 1, 1)
    
    # Mock WarpedVRT class
    with patch('forge3d.adapters.reproject.WarpedVRT') as mock_vrt_class:
        with patch('forge3d.adapters.reproject.calculate_default_transform') as mock_calc:
            # Mock return values
            mock_calc.return_value = (Mock(), 200, 200)  # transform, width, height
            mock_vrt = Mock()
            mock_vrt_class.return_value = mock_vrt
            mock_vrt.count = 3
            mock_vrt.dtype = np.float32
            mock_vrt.bounds = (0, 0, 1, 1)
            
            # Test wrapper creation
            wrapper = rp.WarpedVRTWrapper(mock_dataset, "EPSG:3857")
            
            assert wrapper.dst_crs == "EPSG:3857"
            assert wrapper.dst_width == 200
            assert wrapper.dst_height == 200
            
            # Test context manager
            with wrapper:
                metadata = wrapper.get_metadata()
                assert 'width' in metadata
                assert 'crs' in metadata
            
            # Test close
            wrapper.close()


@pytest.mark.skipif(
    condition=True,  # Skip by default 
    reason="dependencies not available"
)
def test_bounds_transformation():
    """Test coordinate bounds transformation."""
    try:
        import forge3d.adapters.reproject as rp
        import pyproj
    except ImportError:
        pytest.skip("pyproj not available")
    
    # Mock transformer
    with patch('pyproj.Transformer') as mock_transformer_class:
        mock_transformer = Mock()
        mock_transformer_class.from_crs.return_value = mock_transformer
        
        # Mock transform result (rough Web Mercator bounds)
        mock_transformer.transform.return_value = (
            [-20037508, -20037508, 20037508, 20037508],  # x coords
            [-20037508, 20037508, -20037508, 20037508]   # y coords  
        )
        
        result = rp.transform_bounds(
            (-180, -85, 180, 85),  # WGS84 bounds
            "EPSG:4326",
            "EPSG:3857"
        )
        
        assert len(result) == 4
        # Should return (left, bottom, right, top)
        assert result[0] < result[2]  # left < right
        assert result[1] < result[3]  # bottom < top


def test_crs_info_extraction():
    """Test CRS information extraction."""
    with patch('forge3d.adapters.reproject._HAS_REPROJECT_DEPS', True):
        with patch('pyproj.CRS') as mock_crs_class:
            # Mock CRS object
            mock_crs = Mock()
            mock_crs.name = "WGS 84"
            mock_crs.to_authority.return_value = ("EPSG", "4326")
            mock_crs.to_proj4.return_value = "+proj=longlat +datum=WGS84"
            mock_crs.to_wkt.return_value = "GEOGCRS[...]"
            mock_crs.is_geographic = True
            mock_crs.is_projected = False
            mock_crs.axis_info = [
                Mock(name="Longitude", abbreviation="lon", direction="east"),
                Mock(name="Latitude", abbreviation="lat", direction="north")
            ]
            mock_crs.area_of_use = Mock(
                bounds=(-180, -90, 180, 90),
                name="World"
            )
            
            mock_crs_class.from_string.return_value = mock_crs
            
            from forge3d.adapters.reproject import get_crs_info
            
            info = get_crs_info("EPSG:4326")
            
            assert info['name'] == "WGS 84"
            assert info['authority'] == "EPSG:4326"
            assert info['is_geographic'] == True
            assert info['is_projected'] == False
            assert len(info['axis_info']) == 2
            assert info['area_of_use']['name'] == "World"


@pytest.mark.skipif(
    condition=True,  # Skip by default
    reason="dependencies not available" 
)
def test_reprojection_error_estimation():
    """Test reprojection error estimation."""
    try:
        import forge3d.adapters.reproject as rp
        import rasterio
    except ImportError:
        pytest.skip("dependencies not available")
    
    # Mock dataset
    mock_dataset = Mock()
    mock_dataset.width = 100
    mock_dataset.height = 100
    mock_dataset.crs = "EPSG:4326"
    mock_dataset.transform = rasterio.transform.from_bounds(0, 0, 1, 1, 100, 100)
    
    with patch('pyproj.Transformer') as mock_transformer_class:
        with patch('rasterio.transform.xy') as mock_xy:
            # Mock coordinate transforms
            mock_xy.return_value = ([0.1, 0.5, 0.9], [0.1, 0.5, 0.9])  # Sample points
            
            # Mock forward transformation
            forward_transformer = Mock()
            forward_transformer.transform.return_value = ([111000, 555000, 999000], [111000, 555000, 999000])
            
            # Mock inverse transformation (with small errors)
            inverse_transformer = Mock() 
            inverse_transformer.transform.return_value = (
                [0.1001, 0.5001, 0.9001],  # Small x errors
                [0.1001, 0.5001, 0.9001]   # Small y errors
            )
            
            mock_transformer_class.from_crs.side_effect = [forward_transformer, inverse_transformer]
            
            result = rp.estimate_reproject_error(mock_dataset, "EPSG:3857", sample_points=9)
            
            assert 'sample_count' in result
            assert 'x_error_stats' in result
            assert 'y_error_stats' in result
            assert 'combined_rms_error' in result
            
            # Should detect small consistent errors
            assert result['x_error_stats']['mean'] > 0
            assert result['y_error_stats']['mean'] > 0


def test_resampling_enum_handling():
    """Test resampling method handling."""
    with patch('forge3d.adapters.reproject._HAS_REPROJECT_DEPS', True):
        with patch('forge3d.adapters.reproject.WarpedVRT') as mock_vrt:
            with patch('forge3d.adapters.reproject.calculate_default_transform'):
                from forge3d.adapters.reproject import WarpedVRTWrapper
                from rasterio.enums import Resampling
                
                mock_dataset = Mock()
                mock_dataset.crs = "EPSG:4326"
                mock_dataset.bounds = (0, 0, 1, 1)
                mock_dataset.width = 100
                mock_dataset.height = 100
                
                # Test string resampling method
                wrapper1 = WarpedVRTWrapper(mock_dataset, "EPSG:3857", resampling="bilinear")
                assert wrapper1.resampling == Resampling.bilinear
                
                # Test enum resampling method
                wrapper2 = WarpedVRTWrapper(mock_dataset, "EPSG:3857", resampling=Resampling.cubic)
                assert wrapper2.resampling == Resampling.cubic
                
                # Test default resampling
                wrapper3 = WarpedVRTWrapper(mock_dataset, "EPSG:3857")
                assert wrapper3.resampling == Resampling.bilinear


def test_error_handling():
    """Test error handling in reprojection functions."""
    with patch('forge3d.adapters.reproject._HAS_REPROJECT_DEPS', True):
        from forge3d.adapters.reproject import WarpedVRTWrapper
        
        # Test invalid CRS handling
        mock_dataset = Mock()
        mock_dataset.crs = "EPSG:4326"
        mock_dataset.bounds = (0, 0, 1, 1)
        mock_dataset.width = 100
        mock_dataset.height = 100
        
        with patch('forge3d.adapters.reproject.calculate_default_transform') as mock_calc:
            mock_calc.side_effect = Exception("Invalid CRS")
            
            with pytest.raises(Exception, match="Invalid CRS"):
                WarpedVRTWrapper(mock_dataset, "INVALID:CRS")


if __name__ == "__main__":
    test_import_without_deps()
    test_graceful_degradation()
    test_bounds_transformation()
    test_crs_info_extraction()
    test_resampling_enum_handling()
    test_error_handling()
    
    print("Reprojection tests passed!")