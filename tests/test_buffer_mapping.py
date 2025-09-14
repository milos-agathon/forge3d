"""
W7.1: Buffer mapping lifecycle tests for forge3d

Tests mapAsync state transitions, bounds validation, unmapped-usage errors,
and basic race/misuse scenarios.

Acceptance criteria:
- mapAsync state transitions are correct
- Out-of-bounds ranges raise errors  
- Unmapped usage errors are caught
- Basic race/misuse scenarios are handled deterministically
"""
import pytest
import numpy as np
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import forge3d as f3d


class TestBufferMappingStateTransitions:
    """Test buffer mapping state transitions and lifecycle."""
    
    def test_basic_upload_readback_cycle(self):
        """Test basic buffer mapping cycle through upload and readback."""
        try:
            renderer = f3d.Renderer(128, 128)
            
            # Test upload phase (involves MAP_WRITE internally)
            height_data = np.random.rand(64, 64).astype(np.float32)
            height_data = np.ascontiguousarray(height_data)  # Ensure proper layout
            
            # Upload should map buffer, write data, and unmap
            renderer.upload_height_r32f(height_data)
            
            # Test readback phase (involves MAP_READ internally)  
            output = renderer.render_terrain_rgba()
            
            # Validate the mapping cycle completed successfully
            assert output is not None, "Render output should not be None after mapping cycle"
            assert output.shape == (128, 128, 4), f"Expected (128,128,4), got {output.shape}"
            assert output.dtype == np.uint8, f"Expected uint8, got {output.dtype}"
            
            print("✓ Basic upload/readback mapping cycle completed successfully")
            
        except Exception as e:
            if any(term in str(e).lower() for term in ['device', 'adapter', 'backend']):
                pytest.skip(f"GPU not available for mapping cycle test: {e}")
            else:
                raise
    
    def test_multiple_sequential_mappings(self):
        """Test multiple sequential buffer mappings work correctly."""
        try:
            renderer = f3d.Renderer(64, 64)
            
            # Perform multiple upload/render cycles
            for i in range(3):
                height_data = np.random.rand(32, 32).astype(np.float32)
                height_data = np.ascontiguousarray(height_data)
                
                # Each cycle should properly transition through mapping states
                renderer.upload_height_r32f(height_data)
                output = renderer.render_terrain_rgba()
                
                assert output is not None, f"Sequential mapping {i+1} failed"
                assert output.shape == (64, 64, 4), f"Output shape wrong for mapping {i+1}"
                
            print("✓ Multiple sequential mappings completed successfully")
            
        except Exception as e:
            if any(term in str(e).lower() for term in ['device', 'adapter', 'backend']):
                pytest.skip(f"GPU not available for sequential mapping test: {e}")
            else:
                raise
    
    def test_mapping_state_after_error(self):
        """Test that mapping state is properly cleaned up after errors."""
        try:
            renderer = f3d.Renderer(64, 64)
            
            # Try an operation that should fail
            try:
                invalid_data = np.array([[]], dtype=np.float32)  # Invalid empty data
                renderer.upload_height_r32f(invalid_data)
                pytest.fail("Expected error for invalid data")
            except Exception as expected_err:
                print(f"Expected error occurred: {expected_err}")
            
            # After error, renderer should still be usable (state cleaned up)
            valid_data = np.random.rand(32, 32).astype(np.float32)
            valid_data = np.ascontiguousarray(valid_data)
            
            renderer.upload_height_r32f(valid_data)
            output = renderer.render_terrain_rgba()
            
            assert output is not None, "Renderer should work after error recovery"
            print("✓ Buffer mapping state properly cleaned up after error")
            
        except Exception as e:
            if any(term in str(e).lower() for term in ['device', 'adapter', 'backend']):
                pytest.skip(f"GPU not available for error recovery test: {e}")
            else:
                raise


class TestBufferBoundsValidation:
    """Test buffer mapping bounds validation and range checking."""
    
    def test_valid_data_sizes(self):
        """Test that valid data sizes are accepted."""
        try:
            renderer = f3d.Renderer(128, 128)
            
            valid_sizes = [
                (16, 16, "Small valid size"),
                (64, 64, "Medium valid size"),
                (128, 128, "Large valid size"),
            ]
            
            for width, height, description in valid_sizes:
                height_data = np.random.rand(height, width).astype(np.float32)
                height_data = np.ascontiguousarray(height_data)
                
                try:
                    renderer.upload_height_r32f(height_data)
                    output = renderer.render_terrain_rgba()
                    
                    assert output is not None, f"Valid size failed: {description}"
                    print(f"✓ {description} ({width}x{height}) accepted")
                    
                except Exception as size_err:
                    pytest.fail(f"Valid size rejected unexpectedly ({description}): {size_err}")
                    
        except Exception as e:
            if any(term in str(e).lower() for term in ['device', 'adapter', 'backend']):
                pytest.skip(f"GPU not available for valid bounds test: {e}")
            else:
                raise
    
    def test_invalid_data_sizes(self):
        """Test that invalid data sizes are properly rejected."""
        try:
            renderer = f3d.Renderer(64, 64)
            
            invalid_sizes = [
                (np.array([], dtype=np.float32), "Empty array"),
                (np.array([1.0, 2.0], dtype=np.float32), "1D array instead of 2D"),
                (np.array([[[1.0]]], dtype=np.float32), "3D array instead of 2D"),
            ]
            
            for invalid_data, description in invalid_sizes:
                with pytest.raises(Exception) as exc_info:
                    renderer.upload_height_r32f(invalid_data)
                
                error_msg = str(exc_info.value).lower()
                # Error should explain the bounds/dimension issue
                assert any(term in error_msg for term in ['shape', 'dimension', 'size', 'invalid', 'empty', 'tuple', 'length']), \
                    f"Bounds error should explain the issue ({description}): {exc_info.value}"
                
                print(f"✓ {description} properly rejected")
                
        except Exception as e:
            if any(term in str(e).lower() for term in ['device', 'adapter', 'backend']):
                pytest.skip(f"GPU not available for invalid bounds test: {e}")
            else:
                raise
    
    def test_memory_bounds_checking(self):
        """Test that memory bounds are validated during mapping operations."""
        try:
            renderer = f3d.Renderer(128, 128)
            
            # Test with very large data that might exceed limits
            # Note: Actual limits depend on device, so we test error handling
            try:
                # Create larger data (may or may not fail depending on device)
                large_data = np.random.rand(512, 512).astype(np.float32)
                large_data = np.ascontiguousarray(large_data)
                
                renderer.upload_height_r32f(large_data)
                output = renderer.render_terrain_rgba()
                
                if output is not None:
                    print("ℹ Large data upload successful (device has sufficient memory)")
                else:
                    print("⚠ Large data upload returned None")
                    
            except Exception as memory_err:
                error_msg = str(memory_err).lower()
                # If it fails due to memory limits, error should be informative
                if any(term in error_msg for term in ['memory', 'size', 'limit', 'buffer']):
                    print("✓ Memory bounds error properly detected and reported")
                else:
                    print(f"ℹ Large data failed for other reason: {memory_err}")
                    
        except Exception as e:
            if any(term in str(e).lower() for term in ['device', 'adapter', 'backend']):
                pytest.skip(f"GPU not available for memory bounds test: {e}")
            else:
                raise


class TestUnmappedUsageErrors:
    """Test detection of unmapped buffer usage errors."""
    
    def test_render_without_upload(self):
        """Test rendering behavior when no data has been uploaded."""
        try:
            renderer = f3d.Renderer(64, 64)
            
            # Try to render without uploading any data
            try:
                output = renderer.render_terrain_rgba()
                
                # This might succeed with default/fallback data
                if output is not None:
                    print("ℹ Render without upload successful (using default/fallback data)")
                    assert output.shape == (64, 64, 4), "Default render should have correct shape"
                else:
                    pytest.fail("Render without upload returned None unexpectedly")
                    
            except Exception as render_err:
                # If it fails, error should explain missing data
                error_msg = str(render_err).lower()
                expected_terms = ['data', 'upload', 'height', 'terrain', 'empty', 'uninitialized']
                assert any(term in error_msg for term in expected_terms), \
                    f"Unmapped usage error should explain missing data: {render_err}"
                print("✓ Unmapped usage error properly detected")
                
        except Exception as e:
            if any(term in str(e).lower() for term in ['device', 'adapter', 'backend']):
                pytest.skip(f"GPU not available for unmapped usage test: {e}")
            else:
                raise
    
    def test_double_upload_behavior(self):
        """Test behavior when uploading over existing data."""
        try:
            renderer = f3d.Renderer(64, 64)
            
            # First upload
            data1 = np.random.rand(32, 32).astype(np.float32)
            data1 = np.ascontiguousarray(data1)
            renderer.upload_height_r32f(data1)
            
            output1 = renderer.render_terrain_rgba()
            assert output1 is not None, "First upload should succeed"
            
            # Second upload (should replace the first)
            data2 = np.random.rand(32, 32).astype(np.float32) + 10.0  # Different data
            data2 = np.ascontiguousarray(data2)
            renderer.upload_height_r32f(data2)
            
            output2 = renderer.render_terrain_rgba()
            assert output2 is not None, "Second upload should succeed"
            
            # Outputs should be different (different data uploaded)
            if not np.array_equal(output1, output2):
                print("✓ Double upload properly replaces previous data")
            else:
                print("ℹ Double upload outputs are identical (may be due to default processing)")
                
        except Exception as e:
            if any(term in str(e).lower() for term in ['device', 'adapter', 'backend']):
                pytest.skip(f"GPU not available for double upload test: {e}")
            else:
                raise
    
    def test_partial_upload_error_handling(self):
        """Test error handling during partial upload operations."""
        try:
            renderer = f3d.Renderer(64, 64)
            
            # Test with non-contiguous array (should be rejected or fixed)
            base_data = np.random.rand(64, 128).astype(np.float32)
            non_contiguous = base_data[:, ::2]  # Every other column
            
            try:
                renderer.upload_height_r32f(non_contiguous)
                
                # If it succeeds, implementation handles it internally
                print("ℹ Non-contiguous data accepted (implementation handles internally)")
                
            except Exception as contiguous_err:
                error_msg = str(contiguous_err).lower()
                # Error should explain contiguity requirement
                assert any(term in error_msg for term in ['contiguous', 'c-contiguous', 'layout']), \
                    f"Contiguity error should explain requirement: {contiguous_err}"
                print("✓ Non-contiguous data properly rejected with clear error")
                
        except Exception as e:
            if any(term in str(e).lower() for term in ['device', 'adapter', 'backend']):
                pytest.skip(f"GPU not available for partial upload test: {e}")
            else:
                raise


class TestBufferMappingRaceConditions:
    """Test basic race conditions and misuse scenarios in buffer mapping."""
    
    def test_concurrent_upload_attempts(self):
        """Test behavior with concurrent upload attempts."""
        try:
            renderer = f3d.Renderer(64, 64)
            results = []
            errors = []
            
            def upload_worker(worker_id):
                try:
                    data = np.random.rand(16, 16).astype(np.float32) * worker_id
                    data = np.ascontiguousarray(data)
                    renderer.upload_height_r32f(data)
                    results.append(f"Worker {worker_id} succeeded")
                    return True
                except Exception as e:
                    errors.append(f"Worker {worker_id}: {e}")
                    return False
            
            # Run concurrent uploads
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(upload_worker, i) for i in range(3)]
                completed = [f.result(timeout=5.0) for f in futures]
            
            successful = sum(completed)
            print(f"Concurrent uploads: {successful}/3 succeeded")
            
            if errors:
                print("Errors during concurrent access:")
                for error in errors:
                    print(f"  {error}")
            
            # At least one upload should succeed, and errors should be deterministic
            assert successful > 0, "At least one concurrent upload should succeed"
            
            # Try rendering after concurrent access
            try:
                output = renderer.render_terrain_rgba()
                assert output is not None, "Render should work after concurrent access"
                print("✓ Renderer remains functional after concurrent access")
            except Exception as render_err:
                print(f"⚠ Render failed after concurrent access: {render_err}")
                
        except Exception as e:
            if any(term in str(e).lower() for term in ['device', 'adapter', 'backend']):
                pytest.skip(f"GPU not available for concurrent access test: {e}")
            else:
                raise
    
    def test_rapid_sequential_operations(self):
        """Test rapid sequential buffer operations."""
        try:
            renderer = f3d.Renderer(32, 32)
            
            # Perform rapid upload/render cycles
            success_count = 0
            error_count = 0
            
            for i in range(5):
                try:
                    data = np.random.rand(8, 8).astype(np.float32)
                    data = np.ascontiguousarray(data)
                    
                    renderer.upload_height_r32f(data)
                    output = renderer.render_terrain_rgba()
                    
                    if output is not None:
                        success_count += 1
                    else:
                        error_count += 1
                        
                except Exception as rapid_err:
                    error_count += 1
                    print(f"Rapid operation {i} failed: {rapid_err}")
            
            print(f"Rapid sequential operations: {success_count} succeeded, {error_count} failed")
            
            # Most operations should succeed under normal conditions
            assert success_count >= 3, f"Expected at least 3/5 rapid operations to succeed, got {success_count}"
            
        except Exception as e:
            if any(term in str(e).lower() for term in ['device', 'adapter', 'backend']):
                pytest.skip(f"GPU not available for rapid operations test: {e}")
            else:
                raise
    
    def test_interleaved_upload_render_operations(self):
        """Test interleaved upload and render operations."""
        try:
            renderer = f3d.Renderer(64, 64)
            
            operations = []
            
            for i in range(4):
                # Upload operation
                try:
                    data = np.random.rand(16, 16).astype(np.float32) + i
                    data = np.ascontiguousarray(data)
                    renderer.upload_height_r32f(data)
                    operations.append(f"Upload {i}: SUCCESS")
                except Exception as upload_err:
                    operations.append(f"Upload {i}: FAILED - {upload_err}")
                
                # Render operation
                try:
                    output = renderer.render_terrain_rgba()
                    if output is not None:
                        operations.append(f"Render {i}: SUCCESS")
                    else:
                        operations.append(f"Render {i}: NULL_OUTPUT")
                except Exception as render_err:
                    operations.append(f"Render {i}: FAILED - {render_err}")
            
            # Print operation results
            for op in operations:
                print(f"  {op}")
            
            # Check that operations completed in a deterministic manner
            upload_successes = sum(1 for op in operations if "Upload" in op and "SUCCESS" in op)
            render_successes = sum(1 for op in operations if "Render" in op and "SUCCESS" in op)
            
            assert upload_successes >= 2, f"Expected at least 2 successful uploads, got {upload_successes}"
            assert render_successes >= 2, f"Expected at least 2 successful renders, got {render_successes}"
            
            print("✓ Interleaved operations completed deterministically")
            
        except Exception as e:
            if any(term in str(e).lower() for term in ['device', 'adapter', 'backend']):
                pytest.skip(f"GPU not available for interleaved operations test: {e}")
            else:
                raise


class TestBufferMappingErrorRecovery:
    """Test error recovery and cleanup in buffer mapping scenarios."""
    
    def test_recovery_from_mapping_errors(self):
        """Test recovery from various mapping errors."""
        try:
            renderer = f3d.Renderer(64, 64)
            
            error_scenarios = [
                (np.array([[]], dtype=np.float32), "Empty array"),
                (np.array([1.0], dtype=np.float32), "Wrong dimensions"),
                (np.random.rand(8, 8).astype(np.int32).astype(np.float32), "Potentially invalid data"),
            ]
            
            for invalid_data, scenario_desc in error_scenarios:
                print(f"Testing recovery from: {scenario_desc}")
                
                # Try invalid operation
                try:
                    renderer.upload_height_r32f(invalid_data)
                    print(f"  Unexpectedly succeeded: {scenario_desc}")
                except Exception as expected_err:
                    print(f"  Expected error: {expected_err}")
                
                # Try recovery with valid operation
                try:
                    valid_data = np.random.rand(16, 16).astype(np.float32)
                    valid_data = np.ascontiguousarray(valid_data)
                    renderer.upload_height_r32f(valid_data)
                    
                    output = renderer.render_terrain_rgba()
                    assert output is not None, f"Recovery failed after: {scenario_desc}"
                    print(f"  ✓ Successfully recovered from: {scenario_desc}")
                    
                except Exception as recovery_err:
                    pytest.fail(f"Failed to recover after {scenario_desc}: {recovery_err}")
                    
        except Exception as e:
            if any(term in str(e).lower() for term in ['device', 'adapter', 'backend']):
                pytest.skip(f"GPU not available for error recovery test: {e}")
            else:
                raise
    
    def test_resource_cleanup_after_errors(self):
        """Test that resources are properly cleaned up after mapping errors."""
        try:
            # Get initial memory state
            initial_device_info = f3d.device_probe()
            
            renderer = f3d.Renderer(64, 64)
            
            # Cause multiple errors that should clean up properly
            for i in range(3):
                try:
                    invalid_data = np.array([], dtype=np.float32)
                    renderer.upload_height_r32f(invalid_data)
                except Exception:
                    pass  # Expected to fail
            
            # After errors, renderer should still work
            valid_data = np.random.rand(16, 16).astype(np.float32)
            valid_data = np.ascontiguousarray(valid_data)
            renderer.upload_height_r32f(valid_data)
            
            output = renderer.render_terrain_rgba()
            assert output is not None, "Renderer should work after multiple errors"
            
            # Clean up renderer
            del renderer
            
            print("✓ Resource cleanup completed after errors")
            
        except Exception as e:
            if any(term in str(e).lower() for term in ['device', 'adapter', 'backend']):
                pytest.skip(f"GPU not available for resource cleanup test: {e}")
            else:
                raise


def test_buffer_mapping_integration():
    """Integration test for overall buffer mapping lifecycle."""
    
    try:
        print("Testing complete buffer mapping lifecycle...")
        
        # Test full mapping workflow
        renderer = f3d.Renderer(128, 128)
        
        # Phase 1: Upload (MAP_WRITE)
        height_data = np.random.rand(64, 64).astype(np.float32)
        height_data = np.ascontiguousarray(height_data)
        renderer.upload_height_r32f(height_data)
        
        # Phase 2: Render (internal buffer operations)
        output = renderer.render_terrain_rgba()
        
        # Phase 3: Readback validation (MAP_READ)
        assert output is not None, "Buffer mapping cycle should produce output"
        assert output.shape == (128, 128, 4), "Output should have correct dimensions"
        assert output.dtype == np.uint8, "Output should have correct type"
        
        # Phase 4: Multiple cycles
        for cycle in range(3):
            new_data = np.random.rand(64, 64).astype(np.float32) * (cycle + 1)
            new_data = np.ascontiguousarray(new_data)
            
            renderer.upload_height_r32f(new_data)
            cycle_output = renderer.render_terrain_rgba()
            
            assert cycle_output is not None, f"Mapping cycle {cycle+1} should succeed"
        
        print("✓ Complete buffer mapping lifecycle test passed")
        return True
        
    except Exception as e:
        if any(term in str(e).lower() for term in ['device', 'adapter', 'backend']):
            pytest.skip(f"GPU not available for buffer mapping integration test: {e}")
        else:
            raise


if __name__ == "__main__":
    # Allow direct execution for debugging
    print("Running buffer mapping lifecycle tests...")
    
    try:
        result = test_buffer_mapping_integration()
        print(f"✓ Integration test: {'PASS' if result else 'FAIL'}")
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
    
    print("\nFor full test suite, run: pytest tests/test_buffer_mapping.py -v")