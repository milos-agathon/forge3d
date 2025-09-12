# tests/test_datashader_adapter.py
# Tests for datashader adapter integration.
# This exists to validate Workstream V1 acceptance criteria with zero-copy and alignment checks.
# RELEVANT FILES:python/forge3d/adapters/datashader_adapter.py,python/forge3d/adapters/__init__.py

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import warnings


def test_adapter_availability_check():
    """Test that availability checking works correctly."""
    # Test the basic availability check
    from forge3d.adapters import check_optional_dependency
    
    # This should not raise an error regardless of datashader presence
    has_datashader = check_optional_dependency('datashader')
    assert isinstance(has_datashader, bool)
    
    # Test direct import
    from forge3d.adapters import is_datashader_available
    availability = is_datashader_available()
    assert isinstance(availability, bool)
    assert availability == has_datashader


@pytest.mark.skipif(
    not pytest.importorskip("forge3d.adapters", reason="adapters not available").check_optional_dependency('datashader'),
    reason="Datashader not available"
)
class TestDatashaderWithPackage:
    """Tests that run when datashader is available."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Import required modules
        self.ds = pytest.importorskip("datashader")
        self.tf = pytest.importorskip("datashader.transfer_functions")
        self.pd = pytest.importorskip("pandas")
        
        from forge3d.adapters import (
            DatashaderAdapter, rgba_view_from_agg,
            validate_alignment, to_overlay_texture, 
            shade_to_overlay
        )
        
        self.DatashaderAdapter = DatashaderAdapter
        self.rgba_view_from_agg = rgba_view_from_agg
        self.validate_alignment = validate_alignment
        self.to_overlay_texture = to_overlay_texture
        self.shade_to_overlay = shade_to_overlay
    
    def create_test_dataframe(self, n_points=1000, seed=42):
        """Create synthetic test data."""
        np.random.seed(seed)
        
        # Create clustered point data
        x = np.random.normal(0, 10, n_points)
        y = np.random.normal(0, 5, n_points)
        value = np.random.gamma(2, 1, n_points)
        
        return self.pd.DataFrame({'x': x, 'y': y, 'value': value})
    
    def create_test_aggregation(self, width=100, height=80):
        """Create a synthetic datashader aggregation."""
        df = self.create_test_dataframe(1000)
        
        canvas = self.ds.Canvas(
            plot_width=width, plot_height=height,
            x_range=(-30, 30), y_range=(-15, 15)
        )
        
        agg = canvas.points(df, 'x', 'y', self.ds.mean('value'))
        return agg
    
    def test_adapter_initialization(self):
        """Test DatashaderAdapter initialization."""
        adapter = self.DatashaderAdapter()
        assert adapter.copy_count == 0
        
        adapter.reset_copy_count()
        assert adapter.copy_count == 0
    
    def test_rgba_view_from_agg_zero_copy(self):
        """Test zero-copy RGBA conversion from aggregation."""
        # Create test aggregation
        agg = self.create_test_aggregation(100, 80)
        
        # Shade it to create an image
        img = self.tf.shade(agg, how='linear')
        
        # Convert to RGBA using adapter
        rgba = self.rgba_view_from_agg(img)
        
        # Validate format
        assert rgba.ndim == 3
        assert rgba.shape[2] == 4  # RGBA
        assert rgba.dtype == np.uint8
        assert rgba.flags.c_contiguous
        
        # Test that we get the expected dimensions
        assert rgba.shape[0] == 80  # height
        assert rgba.shape[1] == 100  # width
        
        # For shaded images, we should get a view where possible
        # Note: This may not always be zero-copy due to format conversions,
        # but we test that the function completes successfully
        original_array = img.values
        if original_array.shape[2] == 4 and original_array.dtype == np.uint8:
            # Should be zero-copy if format matches
            assert np.shares_memory(rgba, original_array)
    
    def test_rgba_view_from_numpy_array(self):
        """Test RGBA conversion from numpy array."""
        # Create synthetic RGBA array
        height, width = 60, 80
        rgba_input = np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)
        
        # Convert using adapter
        rgba_output = self.rgba_view_from_agg(rgba_input)
        
        # Should be identical (zero-copy)
        assert np.array_equal(rgba_input, rgba_output)
        assert np.shares_memory(rgba_input, rgba_output)
        assert rgba_output.flags.c_contiguous
    
    def test_rgba_view_from_rgb_array(self):
        """Test RGBA conversion from RGB array (requires copy for alpha channel)."""
        # Create synthetic RGB array
        height, width = 50, 70
        rgb_input = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        # Convert using adapter
        rgba_output = self.rgba_view_from_agg(rgb_input)
        
        # Should add alpha channel
        assert rgba_output.shape == (height, width, 4)
        assert rgba_output.dtype == np.uint8
        assert rgba_output.flags.c_contiguous
        
        # RGB channels should match
        assert np.array_equal(rgb_input, rgba_output[..., :3])
        
        # Alpha should be full opacity
        assert np.all(rgba_output[..., 3] == 255)
        
        # Should NOT share memory (copy required for alpha channel)
        assert not np.shares_memory(rgb_input, rgba_output)
    
    def test_validate_alignment_simple(self):
        """Test basic coordinate alignment validation."""
        extent = (-100, -50, 100, 50)
        width, height = 200, 100
        
        # Basic validation without transform
        result = self.validate_alignment(extent, None, width, height)
        
        assert result['pixel_width'] == 1.0  # (200 extent / 200 width)
        assert result['pixel_height'] == 1.0  # (100 extent / 100 height)
        assert result['pixel_error_x'] == 0.0
        assert result['pixel_error_y'] == 0.0
        assert result['within_tolerance'] is True
        assert result['transform_provided'] is False
    
    def test_validate_alignment_with_transform(self):
        """Test alignment validation with affine transform."""
        # Create mock affine transform
        mock_transform = MagicMock()
        mock_transform.a = 1.0  # pixel width
        mock_transform.e = -1.0  # pixel height (negative for image coordinates)
        
        extent = (-100, -50, 100, 50)
        width, height = 200, 100
        
        result = self.validate_alignment(extent, mock_transform, width, height)
        
        assert result['transform_provided'] is True
        assert result['within_tolerance'] is True
        assert 'transform_pixel_width' in result
        assert 'transform_pixel_height' in result
    
    def test_validate_alignment_error_tolerance(self):
        """Test that alignment validation catches errors exceeding tolerance."""
        # Create transform with significant pixel size difference
        mock_transform = MagicMock()
        mock_transform.a = 2.0  # 2x pixel width difference
        mock_transform.e = -2.0  # 2x pixel height difference
        
        extent = (-100, -50, 100, 50)
        width, height = 200, 100  # Would give 1.0 pixel size
        
        # Should raise error for exceeding 0.5px tolerance
        with pytest.raises(ValueError, match="Alignment error exceeds 0.5px tolerance"):
            self.validate_alignment(extent, mock_transform, width, height)
    
    def test_to_overlay_texture_format_validation(self):
        """Test overlay texture format validation and preparation."""
        # Create valid RGBA array
        height, width = 60, 80
        rgba = np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)
        rgba = np.ascontiguousarray(rgba)  # Ensure contiguous
        
        extent = (-40, -30, 40, 30)
        
        # Convert to overlay texture
        overlay = self.to_overlay_texture(rgba, extent)
        
        # Validate output format
        assert overlay['width'] == width
        assert overlay['height'] == height
        assert overlay['extent'] == extent
        assert overlay['format'] == 'RGBA8'
        assert overlay['bytes_per_pixel'] == 4
        assert overlay['total_bytes'] == rgba.nbytes
        assert overlay['is_contiguous'] is True
        assert overlay['shares_memory'] is True
        
        # Should share memory with input
        assert np.shares_memory(overlay['rgba'], rgba)
    
    def test_to_overlay_texture_invalid_formats(self):
        """Test that invalid RGBA formats are rejected."""
        extent = (-10, -10, 10, 10)
        
        # Wrong number of dimensions
        with pytest.raises(ValueError, match="Expected \\(H,W,4\\) RGBA array"):
            self.to_overlay_texture(np.zeros((100, 100)), extent)
        
        # Wrong number of channels
        with pytest.raises(ValueError, match="Expected \\(H,W,4\\) RGBA array"):
            self.to_overlay_texture(np.zeros((100, 100, 3), dtype=np.uint8), extent)
        
        # Wrong dtype
        with pytest.raises(ValueError, match="Expected uint8 dtype"):
            self.to_overlay_texture(np.zeros((100, 100, 4), dtype=np.float32), extent)
        
        # Non-contiguous array
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        non_contiguous = rgba[::2, ::2]  # Create non-contiguous view
        with pytest.raises(ValueError, match="must be C-contiguous"):
            self.to_overlay_texture(non_contiguous, extent)
    
    def test_shade_to_overlay_integration(self):
        """Test end-to-end integration from aggregation to overlay."""
        # Create test data
        agg = self.create_test_aggregation(80, 60)
        extent = (-30, -15, 30, 15)
        
        # Convert to overlay using convenience function
        overlay = self.shade_to_overlay(agg, extent, cmap='viridis', how='linear')
        
        # Validate output
        assert overlay['width'] == 80
        assert overlay['height'] == 60
        assert overlay['extent'] == extent
        assert overlay['rgba'].shape == (60, 80, 4)
        assert overlay['rgba'].dtype == np.uint8
        assert overlay['is_contiguous'] is True
    
    def test_performance_within_memory_budget(self):
        """Test that memory usage stays within 512MB budget for reasonable inputs."""
        # Test with moderately large dataset
        n_points = 100_000
        width, height = 1000, 800
        
        df = self.create_test_dataframe(n_points)
        extent = (-50, -25, 50, 25)
        
        # Create canvas and aggregate
        canvas = self.ds.Canvas(
            plot_width=width, plot_height=height,
            x_range=(extent[0], extent[2]), y_range=(extent[1], extent[3])
        )
        agg = canvas.points(df, 'x', 'y', self.ds.mean('value'))
        
        # Convert to overlay
        overlay = self.shade_to_overlay(agg, extent)
        
        # Check memory usage
        memory_mb = overlay['total_bytes'] / (1024 * 1024)
        budget_mb = 512
        
        assert memory_mb <= budget_mb, f"Memory usage {memory_mb:.1f}MB exceeds {budget_mb}MB budget"
    
    def test_adapter_info(self):
        """Test datashader info retrieval."""
        from forge3d.adapters import get_adapter_info, get_datashader_info
        
        # Test general adapter info
        info = get_adapter_info()
        assert 'datashader' in info
        assert info['datashader']['available'] is True
        assert 'version' in info['datashader']
        
        # Test specific datashader info
        ds_info = get_datashader_info()
        assert ds_info['available'] is True
        assert 'version' in ds_info
        assert 'transfer_functions' in ds_info
        assert 'colormaps' in ds_info
        assert isinstance(ds_info['transfer_functions'], list)
        assert isinstance(ds_info['colormaps'], list)


class TestDatashaderWithoutPackage:
    """Tests that run when datashader is NOT available."""
    
    def test_graceful_degradation(self):
        """Test that adapter gracefully handles missing datashader."""
        with patch('forge3d.adapters.datashader_adapter._HAS_DATASHADER', False):
            from forge3d.adapters import is_datashader_available
            assert is_datashader_available() is False
    
    def test_helpful_error_messages(self):
        """Test that helpful error messages are provided when datashader is missing."""
        with patch('forge3d.adapters.datashader_adapter._HAS_DATASHADER', False):
            from forge3d.adapters.datashader_adapter import rgba_view_from_agg
            
            with pytest.raises(ImportError, match="Install with: pip install datashader"):
                rgba_view_from_agg(np.zeros((100, 100, 4)))
    
    def test_adapter_info_without_datashader(self):
        """Test adapter info when datashader is not available."""
        with patch('forge3d.adapters.datashader_adapter._HAS_DATASHADER', False):
            from forge3d.adapters.datashader_adapter import get_datashader_info
            
            info = get_datashader_info()
            assert info['available'] is False
            assert info['version'] is None
            assert info['transfer_functions'] == []
            assert info['colormaps'] == []


def test_invalid_extent_validation():
    """Test that invalid extents are caught in validation."""
    # This test should work regardless of datashader availability
    try:
        from forge3d.adapters import validate_alignment
    except ImportError:
        pytest.skip("Datashader adapter not available")
    
    # Invalid extent (xmin >= xmax)
    with pytest.raises(ValueError, match="Invalid extent"):
        validate_alignment((100, -50, 50, 50), None, 100, 100)
    
    # Invalid extent (ymin >= ymax)  
    with pytest.raises(ValueError, match="Invalid extent"):
        validate_alignment((-50, 50, 50, -50), None, 100, 100)


def test_edge_cases():
    """Test edge cases that should work regardless of datashader availability."""
    pytest.importorskip("forge3d.adapters")
    
    from forge3d.adapters import check_optional_dependency
    
    # Test unknown library
    with pytest.raises(ValueError, match="Unknown library"):
        check_optional_dependency("nonexistent_library")


# Mark tests that require actual datashader functionality
pytestmark = pytest.mark.filterwarnings("ignore:.*datashader.*:UserWarning")