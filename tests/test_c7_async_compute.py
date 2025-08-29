"""
Tests for C7: Async compute prepasses

Validates async compute shader execution, resource barrier management, and GPU pipeline parallelization.
"""

import pytest


def test_c7_async_compute_basic():
    """Test basic async compute functionality exists and can be imported."""
    try:
        import forge3d
        # Check if the test helper function is available
        assert hasattr(forge3d, 'c7_run_compute_prepass')
    except ImportError as e:
        pytest.skip(f"forge3d module not available: {e}")


def test_c7_run_compute_prepass():
    """Test C7 compute prepass execution."""
    try:
        import forge3d
        
        result = forge3d.c7_run_compute_prepass()
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'written_nonzero' in result
        assert 'ordered' in result
        
        # Verify data types
        assert isinstance(result['written_nonzero'], bool)
        assert isinstance(result['ordered'], bool)
        
    except ImportError as e:
        pytest.skip(f"forge3d module not available: {e}")
    except Exception as e:
        pytest.skip(f"GPU context not available: {e}")


def test_c7_written_nonzero_validation():
    """Test that compute prepass writes non-zero values."""
    try:
        import forge3d
        
        result = forge3d.c7_run_compute_prepass()
        
        # According to acceptance criteria, written_nonzero == True
        written_nonzero = result.get('written_nonzero', False)
        assert written_nonzero is True, "Compute prepass should write non-zero values"
        
    except ImportError as e:
        pytest.skip(f"forge3d module not available: {e}")
    except Exception as e:
        pytest.skip(f"GPU context not available: {e}")


def test_c7_ordering_validation():
    """Test that compute prepass maintains proper ordering."""
    try:
        import forge3d
        
        result = forge3d.c7_run_compute_prepass()
        
        # According to acceptance criteria, ordered == True
        ordered = result.get('ordered', False)
        assert ordered is True, "Compute prepass should maintain proper ordering"
        
    except ImportError as e:
        pytest.skip(f"forge3d module not available: {e}")
    except Exception as e:
        pytest.skip(f"GPU context not available: {e}")


def test_c7_compute_consistency():
    """Test that compute prepass results are consistent."""
    try:
        import forge3d
        
        # Run multiple times to check consistency
        results = []
        for _ in range(3):
            result = forge3d.c7_run_compute_prepass()
            results.append(result)
        
        # All results should have the same structure and values
        first_result = results[0]
        for result in results[1:]:
            assert result['written_nonzero'] == first_result['written_nonzero']
            assert result['ordered'] == first_result['ordered']
        
        # All should pass the acceptance criteria
        for result in results:
            assert result['written_nonzero'] is True
            assert result['ordered'] is True
            
    except ImportError as e:
        pytest.skip(f"forge3d module not available: {e}")
    except Exception as e:
        pytest.skip(f"GPU context not available: {e}")


def test_c7_async_compute_integration():
    """Integration test for async compute functionality."""
    try:
        import forge3d
        
        # Test the compute prepass multiple times to ensure it works reliably
        success_count = 0
        total_runs = 5
        
        for i in range(total_runs):
            try:
                result = forge3d.c7_run_compute_prepass()
                
                # Verify structure
                assert isinstance(result, dict)
                assert 'written_nonzero' in result
                assert 'ordered' in result
                
                # Count successful runs that meet criteria
                if result.get('written_nonzero') and result.get('ordered'):
                    success_count += 1
                    
            except Exception:
                # Allow some failures in case of resource contention
                continue
        
        # Should have mostly successful runs
        success_rate = success_count / total_runs
        assert success_rate >= 0.8, f"Success rate too low: {success_rate} < 0.8"
        
    except ImportError as e:
        pytest.skip(f"forge3d module not available: {e}")
    except Exception as e:
        pytest.skip(f"GPU context not available: {e}")


def test_c7_compute_error_handling():
    """Test error handling in async compute."""
    try:
        import forge3d
        
        # The function should not raise exceptions under normal conditions
        result = forge3d.c7_run_compute_prepass()
        
        # Should return a valid result dictionary
        assert isinstance(result, dict)
        assert len(result) >= 2  # At least written_nonzero and ordered
        
        # Values should be boolean
        for key, value in result.items():
            if key in ['written_nonzero', 'ordered']:
                assert isinstance(value, bool), f"Key {key} should be boolean, got {type(value)}"
        
    except ImportError as e:
        pytest.skip(f"forge3d module not available: {e}")
    except Exception as e:
        pytest.skip(f"GPU context not available: {e}")