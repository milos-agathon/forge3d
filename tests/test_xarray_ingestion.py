"""
Tests for forge3d.ingest.xarray_adapter - DataArray ingestion functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

def test_import_without_xarray():
    """Test that module can be imported even without xarray/rioxarray."""
    with patch.dict('sys.modules', {'xarray': None, 'rioxarray': None}):
        import forge3d.ingest.xarray_adapter as xa
        assert not xa.is_xarray_available()


def test_graceful_degradation():
    """Test that functions raise appropriate errors when xarray is not available."""
    with patch('forge3d.ingest.xarray_adapter._HAS_XARRAY', False):
        import forge3d.ingest.xarray_adapter as xa
        
        with pytest.raises(ImportError, match="xarray and rioxarray are required"):
            xa.validate_dataarray(None)
            
        with pytest.raises(ImportError, match="xarray and rioxarray are required"):
            xa.ingest_dataarray(None)


def test_dataarray_validation():
    """Test DataArray validation logic."""
    with patch('forge3d.ingest.xarray_adapter._HAS_XARRAY', True):
        import forge3d.ingest.xarray_adapter as xa
        
        # Mock xarray DataArray
        mock_da = Mock()
        mock_da.dims = ('y', 'x')
        mock_da.shape = (100, 200)
        mock_da.dtype = np.float32
        mock_da.attrs = {'description': 'test data'}
        
        # Mock coordinates
        mock_coords = {}
        mock_coords['y'] = Mock()
        mock_coords['y'].values = np.linspace(45, 44, 100)
        mock_coords['x'] = Mock() 
        mock_coords['x'].values = np.linspace(-120, -119, 200)
        mock_da.coords = mock_coords
        
        # Mock rio accessor
        mock_rio = Mock()
        mock_rio.crs = "EPSG:4326"
        mock_rio.transform.return_value = Mock()
        mock_rio.width = 200
        mock_rio.height = 100
        mock_rio.resolution.return_value = (0.005, 0.01)
        mock_rio.bounds.return_value = (-120, 44, -119, 45)
        mock_rio.nodata = -9999
        mock_da.rio = mock_rio
        
        # Mock values array
        mock_values = Mock()
        mock_values.flags.c_contiguous = True
        mock_da.values = mock_values
        
        # Test validation
        result = xa.validate_dataarray(mock_da)
        
        assert result['dims'] == ('y', 'x')
        assert result['shape'] == (100, 200)
        assert result['dtype'] == np.float32
        assert result['spatial_dims'] == {'y': 'y', 'x': 'x'}
        assert result['has_rio_accessor'] == True
        assert result['rio_info']['crs'] == "EPSG:4326"
        assert result['is_c_contiguous'] == True


def test_dataarray_validation_errors():
    """Test validation error cases."""
    with patch('forge3d.ingest.xarray_adapter._HAS_XARRAY', True):
        import forge3d.ingest.xarray_adapter as xa
        import xarray as xr
        
        # Test non-DataArray input
        with pytest.raises(ValueError, match="Expected xarray.DataArray"):
            xa.validate_dataarray("not a dataarray")
        
        # Test insufficient dimensions
        mock_da = Mock(spec=xr.DataArray)
        mock_da.dims = ('single',)
        
        with pytest.raises(ValueError, match="must have at least 2 dimensions"):
            xa.validate_dataarray(mock_da)


def test_dimension_identification():
    """Test spatial dimension identification."""
    with patch('forge3d.ingest.xarray_adapter._HAS_XARRAY', True):
        import forge3d.ingest.xarray_adapter as xa
        
        # Test various dimension naming conventions
        test_cases = [
            (('lat', 'lon'), {'y': 'lat', 'x': 'lon'}),
            (('latitude', 'longitude'), {'y': 'latitude', 'x': 'longitude'}),
            (('north', 'east'), {'y': 'north', 'x': 'east'}),
            (('y', 'x'), {'y': 'y', 'x': 'x'}),
            (('time', 'y', 'x'), {'y': 'y', 'x': 'x'}),  # fallback to last two
            (('band', 'row', 'col'), {'y': 'row', 'x': 'col'}),  # fallback
        ]
        
        for dims, expected_spatial in test_cases:
            mock_da = Mock()
            mock_da.dims = dims
            mock_da.shape = tuple(50 for _ in dims)
            mock_da.dtype = np.float32
            mock_da.attrs = {}
            mock_da.coords = {dim: Mock() for dim in dims}
            for coord in mock_da.coords.values():
                coord.values = np.arange(50)
            
            # Mock values and rio accessor
            mock_da.values = Mock()
            mock_da.values.flags.c_contiguous = True
            mock_da.rio = Mock()
            mock_da.rio.crs = None
            
            result = xa.validate_dataarray(mock_da)
            assert result['spatial_dims'] == expected_spatial


def test_dataarray_ingestion():
    """Test DataArray ingestion with dimension reordering."""
    with patch('forge3d.ingest.xarray_adapter._HAS_XARRAY', True):
        import forge3d.ingest.xarray_adapter as xa
        
        # Mock DataArray with bands, y, x dimensions
        mock_da = Mock()
        mock_da.dims = ('band', 'y', 'x')
        mock_da.shape = (3, 100, 200)
        mock_da.dtype = np.float32
        mock_da.attrs = {'title': 'test raster'}
        
        # Mock coordinates
        mock_coords = {}
        for dim in mock_da.dims:
            mock_coords[dim] = Mock()
            mock_coords[dim].values = np.arange(mock_da.shape[mock_da.dims.index(dim)])
        mock_da.coords = mock_coords
        
        # Mock data array
        test_data = np.random.rand(3, 100, 200).astype(np.float32)
        
        # Mock transpose method
        mock_transposed = Mock()
        mock_transposed.values = test_data.transpose(0, 1, 2)  # Same order
        mock_da.transpose.return_value = mock_transposed
        
        # Mock rio accessor
        mock_rio = Mock()
        mock_rio.crs = "EPSG:32618"
        mock_rio.transform.return_value = Mock()
        mock_da.rio = mock_rio
        mock_da.values = test_data
        
        # Mock validation function
        def mock_validate(da):
            return {
                'dims': ('band', 'y', 'x'),
                'shape': (3, 100, 200),
                'dtype': np.float32,
                'spatial_dims': {'y': 'y', 'x': 'x'},
                'band_dim': 'band',
                'has_rio_accessor': True,
                'rio_info': {'crs': 'EPSG:32618'},
                'attrs': {'title': 'test raster'},
                'coords': mock_coords,
                'is_c_contiguous': True
            }
        
        with patch('forge3d.ingest.xarray_adapter.validate_dataarray', side_effect=mock_validate):
            data, metadata = xa.ingest_dataarray(mock_da)
            
            assert data.shape == (3, 100, 200)
            assert data.dtype == np.float32
            assert metadata['has_rio_metadata'] == True
            assert metadata['rio_info']['crs'] == 'EPSG:32618'
            assert metadata['is_c_contiguous'] == True


def test_dtype_conversion():
    """Test data type conversion during ingestion."""
    with patch('forge3d.ingest.xarray_adapter._HAS_XARRAY', True):
        import forge3d.ingest.xarray_adapter as xa
        
        # Mock float DataArray
        mock_da = Mock()
        mock_da.dims = ('y', 'x')
        mock_da.shape = (50, 50)
        mock_da.dtype = np.float64
        
        # Mock float data in [0, 1] range
        float_data = np.random.rand(50, 50).astype(np.float64)
        mock_transposed = Mock()
        mock_transposed.values = float_data
        mock_da.transpose.return_value = mock_transposed
        
        # Mock validation
        def mock_validate(da):
            return {
                'dims': ('y', 'x'),
                'spatial_dims': {'y': 'y', 'x': 'x'},
                'band_dim': None,
                'has_rio_accessor': False,
                'rio_info': {},
                'attrs': {},
                'coords': {},
                'is_c_contiguous': True
            }
        
        with patch('forge3d.ingest.xarray_adapter.validate_dataarray', side_effect=mock_validate):
            data, metadata = xa.ingest_dataarray(mock_da, target_dtype=np.uint8)
            
            assert data.dtype == np.uint8
            # Should convert from [0,1] float to [0,255] uint8
            assert np.all(data <= 255)
            assert np.all(data >= 0)


def test_c_contiguous_enforcement():
    """Test C-contiguous array enforcement."""
    with patch('forge3d.ingest.xarray_adapter._HAS_XARRAY', True):
        import forge3d.ingest.xarray_adapter as xa
        
        mock_da = Mock()
        
        # Mock non-contiguous data
        non_contiguous_data = Mock()
        non_contiguous_data.flags.c_contiguous = False
        non_contiguous_data.dtype = np.float32
        
        # Mock contiguous conversion
        contiguous_data = np.random.rand(10, 10).astype(np.float32)
        
        mock_transposed = Mock()
        mock_transposed.values = non_contiguous_data
        mock_da.transpose.return_value = mock_transposed
        
        # Mock validation
        def mock_validate(da):
            return {
                'dims': ('y', 'x'),
                'spatial_dims': {'y': 'y', 'x': 'x'},
                'band_dim': None,
                'has_rio_accessor': False,
                'rio_info': {},
                'attrs': {},
                'coords': {},
                'is_c_contiguous': False
            }
        
        with patch('forge3d.ingest.xarray_adapter.validate_dataarray', side_effect=mock_validate):
            with patch('numpy.ascontiguousarray', return_value=contiguous_data) as mock_ascontiguous:
                data, metadata = xa.ingest_dataarray(mock_da, ensure_c_contiguous=True)
                
                mock_ascontiguous.assert_called_once()
                assert metadata['is_c_contiguous'] == True


def test_synthetic_dataarray_creation():
    """Test synthetic DataArray creation."""
    with patch('forge3d.ingest.xarray_adapter._HAS_XARRAY', True):
        import forge3d.ingest.xarray_adapter as xa
        import xarray as xr
        
        # Mock xarray.DataArray constructor
        mock_da = Mock()
        mock_da.rio = Mock()
        mock_da.rio.write_crs.return_value = mock_da
        mock_da.rio.write_transform.return_value = mock_da
        
        with patch('xarray.DataArray', return_value=mock_da):
            with patch('rasterio.transform.from_bounds') as mock_from_bounds:
                mock_from_bounds.return_value = Mock()
                
                result = xa.create_synthetic_dataarray(
                    shape=(100, 200),
                    crs="EPSG:4326",
                    bounds=(-1, -1, 1, 1),
                    seed=42
                )
                
                assert result == mock_da
                mock_da.rio.write_crs.assert_called_with("EPSG:4326")


def test_raster_info_extraction():
    """Test extracting raster info from DataArray."""
    with patch('forge3d.ingest.xarray_adapter._HAS_XARRAY', True):
        import forge3d.ingest.xarray_adapter as xa
        
        mock_da = Mock()
        mock_da.sizes = {'y': 100, 'x': 200, 'band': 3}
        mock_da.dims = ('band', 'y', 'x')
        mock_da.shape = (3, 100, 200)
        mock_da.dtype = np.float32
        
        # Mock validation
        def mock_validate(da):
            return {
                'spatial_dims': {'y': 'y', 'x': 'x'},
                'has_rio_accessor': True,
                'rio_info': {
                    'crs': 'EPSG:4326',
                    'transform': Mock(),
                    'bounds': (-1, -1, 1, 1)
                }
            }
        
        with patch('forge3d.ingest.xarray_adapter.validate_dataarray', side_effect=mock_validate):
            info = xa.dataarray_to_raster_info(mock_da)
            
            assert info['width'] == 200
            assert info['height'] == 100
            assert info['count'] == 3  # band dimension
            assert info['dtype'] == 'float32'
            assert info['crs'] == 'EPSG:4326'


if __name__ == "__main__":
    test_import_without_xarray()
    test_graceful_degradation()
    test_dataarray_validation()
    test_dataarray_validation_errors()
    test_dimension_identification()
    test_dataarray_ingestion()
    test_dtype_conversion()
    test_c_contiguous_enforcement()
    test_synthetic_dataarray_creation()
    test_raster_info_extraction()
    
    print("xarray ingestion tests passed!")
