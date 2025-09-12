"""
Tests for forge3d.ingest.dask_adapter - Dask array ingestion functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

def test_import_without_dask():
    """Test that module can be imported even without dask."""
    with patch.dict('sys.modules', {'dask': None, 'dask.array': None}):
        import forge3d.ingest.dask_adapter as da_mod
        assert not da_mod.is_dask_available()


def test_graceful_degradation():
    """Test that functions raise appropriate errors when dask is not available."""
    with patch('forge3d.ingest.dask_adapter._HAS_DASK', False):
        import forge3d.ingest.dask_adapter as da_mod
        
        with pytest.raises(ImportError, match="dask is required"):
            da_mod.estimate_memory_usage(None)
            
        with pytest.raises(ImportError, match="dask is required"):
            da_mod.plan_chunk_processing(None, (256, 256))


def test_memory_estimation():
    """Test memory usage estimation for dask arrays."""
    with patch('forge3d.ingest.dask_adapter._HAS_DASK', True):
        import forge3d.ingest.dask_adapter as da_mod
        
        # Mock dask array
        mock_da = Mock()
        mock_da.nbytes = 1024 * 1024 * 100  # 100MB total
        mock_da.dtype.itemsize = 4  # float32
        mock_da.shape = (1000, 1000)
        
        # Mock chunks
        mock_chunk1 = Mock()
        mock_chunk1.nbytes = 1024 * 1024 * 25  # 25MB chunk
        mock_chunk2 = Mock()
        mock_chunk2.nbytes = 1024 * 1024 * 25  # 25MB chunk
        
        mock_delayed = Mock()
        mock_delayed.flatten.return_value = [mock_chunk1, mock_chunk2, mock_chunk1, mock_chunk2]
        mock_da.to_delayed.return_value = mock_delayed
        
        result = da_mod.estimate_memory_usage(mock_da, target_tile_size=(256, 256))
        
        assert result['total_array_mb'] == 100.0
        assert result['num_chunks'] == 4
        assert result['max_chunk_mb'] == 25.0
        assert 'tile_memory_mb' in result
        assert 'estimated_peak_mb' in result


def test_chunk_processing_plan():
    """Test chunk processing plan generation."""
    with patch('forge3d.ingest.dask_adapter._HAS_DASK', True):
        import forge3d.ingest.dask_adapter as da_mod
        
        # Mock dask array
        mock_da = Mock()
        mock_da.shape = (512, 1024)  # 2D array
        mock_da.dtype.itemsize = 1  # uint8
        
        # Mock memory estimation
        def mock_memory_estimation(da, tile_size):
            return {
                'estimated_peak_mb': 10.0,  # Well under limit
                'max_chunk_mb': 5.0
            }
        
        with patch('forge3d.ingest.dask_adapter.estimate_memory_usage', side_effect=mock_memory_estimation):
            plan = da_mod.plan_chunk_processing(
                mock_da, 
                target_tile_size=(256, 256),
                memory_limit_mb=100.0
            )
            
            assert plan['target_tile_size'] == (256, 256)
            assert plan['memory_limit_mb'] == 100.0
            assert plan['fits_in_memory'] == True
            assert len(plan['tiles']) > 0
            
            # Should create 2x4 = 8 tiles for 512x1024 array with 256x256 tiles
            expected_tiles = 2 * 4
            assert len(plan['tiles']) == expected_tiles


def test_memory_limit_adjustment():
    """Test automatic tile size adjustment when memory limit is exceeded."""
    with patch('forge3d.ingest.dask_adapter._HAS_DASK', True):
        import forge3d.ingest.dask_adapter as da_mod
        
        mock_da = Mock()
        mock_da.shape = (2048, 2048, 4)  # Large 4-band array
        mock_da.dtype.itemsize = 4  # float32
        
        # Mock memory estimation that exceeds limit
        def mock_memory_estimation(da, tile_size):
            return {
                'estimated_peak_mb': 600.0,  # Exceeds 512MB limit
                'max_chunk_mb': 300.0
            }
        
        with patch('forge3d.ingest.dask_adapter.estimate_memory_usage', side_effect=mock_memory_estimation):
            with pytest.warns(UserWarning, match="Adjusted tile size"):
                plan = da_mod.plan_chunk_processing(
                    mock_da,
                    target_tile_size=(1024, 1024),  # Large tiles
                    memory_limit_mb=512.0
                )
                
                # Should have adjusted tile size down
                adjusted_size = plan['target_tile_size']
                assert adjusted_size[0] < 1024 or adjusted_size[1] < 1024


def test_dask_array_ingestion():
    """Test streaming ingestion of dask arrays."""
    with patch('forge3d.ingest.dask_adapter._HAS_DASK', True):
        import forge3d.ingest.dask_adapter as da_mod
        
        mock_da = Mock()
        mock_da.shape = (256, 256)  # Small 2D array
        
        # Mock processing plan
        mock_plan = {
            'fits_in_memory': True,
            'num_tiles': 2,
            'tiles': [
                {'row_start': 0, 'row_end': 128, 'col_start': 0, 'col_end': 256, 'height': 128, 'width': 256, 'pixels': 128*256},
                {'row_start': 128, 'row_end': 256, 'col_start': 0, 'col_end': 256, 'height': 128, 'width': 256, 'pixels': 128*256}
            ]
        }
        
        with patch('forge3d.ingest.dask_adapter.plan_chunk_processing', return_value=mock_plan):
            # Mock tile data
            tile1_data = np.random.rand(128, 256).astype(np.float32)
            tile2_data = np.random.rand(128, 256).astype(np.float32)
            
            # Mock dask array slicing and computation
            mock_tile1 = Mock()
            mock_tile1.compute.return_value = tile1_data
            mock_tile2 = Mock()
            mock_tile2.compute.return_value = tile2_data
            
            mock_da.__getitem__.side_effect = [mock_tile1, mock_tile2]
            
            # Test ingestion
            tiles_received = []
            for tile_data, tile_info in da_mod.ingest_dask_array(mock_da, target_tile_size=(256, 128)):
                tiles_received.append((tile_data, tile_info))
            
            assert len(tiles_received) == 2
            
            # Check first tile
            data1, info1 = tiles_received[0]
            np.testing.assert_array_equal(data1, tile1_data)
            assert info1['tile_index'] == 0
            assert info1['total_tiles'] == 2
            assert info1['height'] == 128
            assert info1['width'] == 256
            
            # Check second tile
            data2, info2 = tiles_received[1]
            np.testing.assert_array_equal(data2, tile2_data)
            assert info2['tile_index'] == 1
            assert info2['progress'] == 1.0  # 100% complete


def test_progress_callback():
    """Test progress callback functionality."""
    with patch('forge3d.ingest.dask_adapter._HAS_DASK', True):
        import forge3d.ingest.dask_adapter as da_mod
        
        mock_da = Mock()
        mock_da.shape = (100, 100)
        
        # Mock simple plan with one tile
        mock_plan = {
            'fits_in_memory': True,
            'num_tiles': 1,
            'tiles': [{'row_start': 0, 'row_end': 100, 'col_start': 0, 'col_end': 100, 'height': 100, 'width': 100, 'pixels': 10000}]
        }
        
        with patch('forge3d.ingest.dask_adapter.plan_chunk_processing', return_value=mock_plan):
            # Mock data
            tile_data = np.random.rand(100, 100).astype(np.float32)
            mock_tile = Mock()
            mock_tile.compute.return_value = tile_data
            mock_da.__getitem__.return_value = mock_tile
            
            # Test with progress callback
            progress_calls = []
            def progress_callback(current, total, info):
                progress_calls.append((current, total, info['progress']))
            
            list(da_mod.ingest_dask_array(
                mock_da, 
                progress_callback=progress_callback
            ))
            
            assert len(progress_calls) == 1
            assert progress_calls[0] == (1, 1, 1.0)


def test_memory_error_handling():
    """Test memory error handling for arrays that are too large."""
    with patch('forge3d.ingest.dask_adapter._HAS_DASK', True):
        import forge3d.ingest.dask_adapter as da_mod
        
        mock_da = Mock()
        mock_da.shape = (10000, 10000)  # Very large array
        
        # Mock plan that doesn't fit in memory
        mock_plan = {
            'fits_in_memory': False,
            'memory_estimate': {'estimated_peak_mb': 2000.0},  # Way over 512MB limit
            'num_tiles': 100,
            'tiles': []
        }
        
        with patch('forge3d.ingest.dask_adapter.plan_chunk_processing', return_value=mock_plan):
            with pytest.raises(MemoryError, match="Array too large for memory limit"):
                list(da_mod.ingest_dask_array(mock_da, memory_limit_mb=512.0))


def test_streaming_materialization():
    """Test materialization using streaming approach."""
    with patch('forge3d.ingest.dask_adapter._HAS_DASK', True):
        import forge3d.ingest.dask_adapter as da_mod
        
        mock_da = Mock()
        mock_da.shape = (100, 200)
        mock_da.dtype = np.float32
        
        # Mock streaming ingestion
        def mock_ingestion(da, tile_size, memory_limit):
            tile1_data = np.random.rand(50, 200).astype(np.float32)
            tile1_info = {'row_start': 0, 'row_end': 50, 'col_start': 0, 'col_end': 200, 'total_tiles': 2}
            
            tile2_data = np.random.rand(50, 200).astype(np.float32)
            tile2_info = {'row_start': 50, 'row_end': 100, 'col_start': 0, 'col_end': 200, 'total_tiles': 2}
            
            yield tile1_data, tile1_info
            yield tile2_data, tile2_info
        
        with patch('forge3d.ingest.dask_adapter.ingest_dask_array', side_effect=mock_ingestion):
            result = da_mod.materialize_dask_array_streaming(
                mock_da,
                output_shape=(100, 200),
                memory_limit_mb=100.0
            )
            
            assert result.shape == (100, 200)
            assert result.dtype == np.float32


def test_output_array_size_limit():
    """Test that materialization respects output array size limits."""
    with patch('forge3d.ingest.dask_adapter._HAS_DASK', True):
        import forge3d.ingest.dask_adapter as da_mod
        
        mock_da = Mock()
        mock_da.shape = (10000, 10000)  # Would be ~400MB as float32
        mock_da.dtype = np.float32
        
        with pytest.raises(MemoryError, match="Output array.*would exceed memory limit"):
            da_mod.materialize_dask_array_streaming(
                mock_da,
                output_shape=(10000, 10000),
                memory_limit_mb=100.0,  # Too small for output
                dtype=np.float32
            )


def test_rechunk_optimization():
    """Test array rechunking for better processing performance."""
    with patch('forge3d.ingest.dask_adapter._HAS_DASK', True):
        import forge3d.ingest.dask_adapter as da_mod
        
        mock_da = Mock()
        mock_da.shape = (1000, 2000, 3)  # 3-band image
        mock_da.dtype.itemsize = 4  # float32
        mock_da.chunks = ((100, 100, 100, 100, 100, 100, 100, 100, 100, 100),
                          (200, 200, 200, 200, 200, 200, 200, 200, 200, 200),
                          (3,))
        
        # Mock rechunk method
        rechunked_da = Mock()
        mock_da.rechunk.return_value = rechunked_da
        
        result = da_mod.rechunk_for_processing(mock_da, target_chunk_mb=64.0)
        
        assert result == rechunked_da
        mock_da.rechunk.assert_called_once()


def test_synthetic_dask_array_creation():
    """Test creation of synthetic dask arrays."""
    with patch('forge3d.ingest.dask_adapter._HAS_DASK', True):
        import forge3d.ingest.dask_adapter as da_mod
        import dask.array as da
        
        # Mock dask.array functions
        mock_random_array = Mock()
        mock_linspace_y = Mock()
        mock_linspace_x = Mock()
        mock_meshgrid_result = (Mock(), Mock())
        mock_pattern = Mock()
        mock_broadcast_result = Mock()
        
        with patch('dask.array.random.random', return_value=mock_random_array):
            with patch('dask.array.linspace', side_effect=[mock_linspace_y, mock_linspace_x]):
                with patch('dask.array.meshgrid', return_value=mock_meshgrid_result):
                    with patch('dask.array.broadcast_to', return_value=mock_broadcast_result):
                        result = da_mod.create_synthetic_dask_array(
                            shape=(100, 200),
                            chunks=(50, 50),
                            dtype=np.float32,
                            seed=42
                        )
                        
                        # Should create structured synthetic data
                        assert result is not None


def test_3d_array_handling():
    """Test handling of 3D arrays in ingestion."""
    with patch('forge3d.ingest.dask_adapter._HAS_DASK', True):
        import forge3d.ingest.dask_adapter as da_mod
        
        mock_da = Mock()
        mock_da.shape = (4, 100, 200)  # 4-band array
        
        # Mock plan for 3D array
        mock_plan = {
            'fits_in_memory': True,
            'num_tiles': 1,
            'tiles': [{'row_start': 0, 'row_end': 100, 'col_start': 0, 'col_end': 200, 'height': 100, 'width': 200, 'pixels': 20000}]
        }
        
        with patch('forge3d.ingest.dask_adapter.plan_chunk_processing', return_value=mock_plan):
            # Mock 3D tile data
            tile_data = np.random.rand(4, 100, 200).astype(np.float32)
            mock_tile = Mock()
            mock_tile.compute.return_value = tile_data
            mock_da.__getitem__.return_value = mock_tile
            
            tiles = list(da_mod.ingest_dask_array(mock_da))
            
            assert len(tiles) == 1
            data, info = tiles[0]
            assert data.shape == (4, 100, 200)


if __name__ == "__main__":
    test_import_without_dask()
    test_graceful_degradation()
    test_memory_estimation()
    test_chunk_processing_plan()
    test_memory_limit_adjustment()
    test_dask_array_ingestion()
    test_progress_callback()
    test_memory_error_handling()
    test_streaming_materialization()
    test_output_array_size_limit()
    test_rechunk_optimization()
    test_synthetic_dask_array_creation()
    test_3d_array_handling()
    
    print("Dask ingestion tests passed!")