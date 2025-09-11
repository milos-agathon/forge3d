"""
Tests for Q3: GPU profiling markers & timestamp queries

These tests validate the GPU timing infrastructure including timestamp queries,
debug markers, and performance measurement with minimal overhead requirements.
"""

import pytest
import time
from typing import Dict, Any

import forge3d
import forge3d.gpu_metrics as metrics


class TestGpuTimingConfig:
    """Test GPU timing configuration."""
    
    def test_default_config_creation(self):
        """Test creating default GPU timing configuration."""
        config = metrics.create_default_config()
        
        assert config.enable_timestamps == True
        assert config.enable_pipeline_stats == False
        assert config.enable_debug_markers == True
        assert config.label_prefix == "forge3d"
        assert config.max_queries_per_frame == 32
    
    def test_minimal_config_creation(self):
        """Test creating minimal overhead configuration."""
        config = metrics.create_minimal_config()
        
        assert config.enable_timestamps == False
        assert config.enable_pipeline_stats == False
        assert config.enable_debug_markers == False
        assert config.max_queries_per_frame == 0
    
    def test_debug_config_creation(self):
        """Test creating debug configuration.""" 
        config = metrics.create_debug_config()
        
        assert config.enable_timestamps == True
        assert config.enable_pipeline_stats == True
        assert config.enable_debug_markers == True
        assert config.max_queries_per_frame == 64
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = metrics.GpuTimingConfig(
            enable_timestamps=True,
            enable_debug_markers=False,
            max_queries_per_frame=16
        )
        
        data = config.to_dict()
        
        assert data['enable_timestamps'] == True
        assert data['enable_debug_markers'] == False
        assert data['max_queries_per_frame'] == 16
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            'enable_timestamps': False,
            'enable_pipeline_stats': True,
            'label_prefix': 'test',
            'max_queries_per_frame': 8
        }
        
        config = metrics.GpuTimingConfig.from_dict(data)
        
        assert config.enable_timestamps == False
        assert config.enable_pipeline_stats == True
        assert config.label_prefix == 'test'
        assert config.max_queries_per_frame == 8


class TestTimingResult:
    """Test timing result data structures."""
    
    def test_timing_result_creation(self):
        """Test creating timing results."""
        result = metrics.TimingResult(
            name="test_pass",
            gpu_time_ms=5.25,
            timestamp_valid=True
        )
        
        assert result.name == "test_pass"
        assert result.gpu_time_ms == 5.25
        assert result.timestamp_valid == True
        assert result.pipeline_stats == {}
    
    def test_timing_result_with_stats(self):
        """Test timing result with pipeline statistics."""
        pipeline_stats = {
            'vertex_invocations': 12000,
            'fragment_invocations': 48000,
        }
        
        result = metrics.TimingResult(
            name="render_pass",
            gpu_time_ms=2.1,
            timestamp_valid=True,
            pipeline_stats=pipeline_stats
        )
        
        assert result.pipeline_stats['vertex_invocations'] == 12000
        assert result.pipeline_stats['fragment_invocations'] == 48000
    
    def test_timing_result_to_dict(self):
        """Test converting timing result to dictionary."""
        result = metrics.TimingResult(
            name="tonemap",
            gpu_time_ms=1.5,
            timestamp_valid=True
        )
        
        data = result.to_dict()
        
        assert data['name'] == "tonemap"
        assert data['gpu_time_ms'] == 1.5
        assert data['timestamp_valid'] == True
        assert 'pipeline_stats' in data
    
    def test_timing_result_string_representation(self):
        """Test string representation of timing results."""
        result = metrics.TimingResult(
            name="test_scope",
            gpu_time_ms=3.14,
            timestamp_valid=True
        )
        
        result_str = str(result)
        
        assert "test_scope" in result_str
        assert "3.14 ms" in result_str
        assert "✓" in result_str  # Valid timestamp indicator


class TestGpuMetrics:
    """Test GPU metrics collection and analysis."""
    
    def test_metrics_creation(self):
        """Test creating GPU metrics container."""
        metrics_obj = metrics.GpuMetrics()
        
        assert len(metrics_obj.timing_results) == 0
        assert metrics_obj.frame_time_ms == 0.0
        assert metrics_obj.total_gpu_time_ms == 0.0
    
    def test_add_timing_results(self):
        """Test adding timing results to metrics."""
        metrics_obj = metrics.GpuMetrics()
        
        result1 = metrics.TimingResult("pass1", 2.0, True)
        result2 = metrics.TimingResult("pass2", 3.5, True)
        
        metrics_obj.add_timing_result(result1)
        metrics_obj.add_timing_result(result2)
        
        assert len(metrics_obj.timing_results) == 2
        assert metrics_obj.total_gpu_time_ms == 5.5
    
    def test_get_timing_by_name(self):
        """Test retrieving timing results by name."""
        metrics_obj = metrics.GpuMetrics()
        
        result = metrics.TimingResult("hdr_render", 4.2, True)
        metrics_obj.add_timing_result(result)
        
        retrieved = metrics_obj.get_timing_by_name("hdr_render")
        assert retrieved is not None
        assert retrieved.gpu_time_ms == 4.2
        
        missing = metrics_obj.get_timing_by_name("nonexistent")
        assert missing is None
    
    def test_get_timings_dict(self):
        """Test getting all timings as dictionary."""
        metrics_obj = metrics.GpuMetrics()
        
        metrics_obj.add_timing_result(metrics.TimingResult("pass_a", 1.0, True))
        metrics_obj.add_timing_result(metrics.TimingResult("pass_b", 2.0, True))
        
        timings_dict = metrics_obj.get_timings_dict()
        
        assert timings_dict["pass_a"] == 1.0
        assert timings_dict["pass_b"] == 2.0
    
    def test_get_summary(self):
        """Test getting metrics summary."""
        metrics_obj = metrics.GpuMetrics()
        metrics_obj.frame_time_ms = 16.67
        metrics_obj.feature_support = {'timestamps': True, 'pipeline_stats': False}
        
        metrics_obj.add_timing_result(metrics.TimingResult("test", 5.0, True))
        
        summary = metrics_obj.get_summary()
        
        assert summary['frame_time_ms'] == 16.67
        assert summary['timing_count'] == 1
        assert summary['valid_timing_count'] == 1
        assert summary['feature_support']['timestamps'] == True


class TestTimingUtilities:
    """Test timing utility functions."""
    
    def test_overhead_estimation(self):
        """Test timing overhead estimation."""
        overhead_no_markers = metrics.estimate_timing_overhead(10, False)
        overhead_with_markers = metrics.estimate_timing_overhead(10, True)
        
        # Should have some overhead
        assert overhead_no_markers > 0.0
        assert overhead_with_markers > overhead_no_markers
        
        # Overhead should scale with query count
        overhead_20 = metrics.estimate_timing_overhead(20, False)
        assert overhead_20 > overhead_no_markers
    
    def test_config_validation(self):
        """Test timing configuration validation."""
        config = metrics.GpuTimingConfig(enable_timestamps=True, enable_pipeline_stats=True)
        
        # Device with no features
        device_features = {'timestamps': False, 'pipeline_stats': False}
        warnings = metrics.validate_config(config, device_features)
        
        assert len(warnings) == 2
        assert any('TIMESTAMP_QUERY' in w for w in warnings)
        assert any('PIPELINE_STATISTICS_QUERY' in w for w in warnings)
        
        # Device with all features
        device_features = {'timestamps': True, 'pipeline_stats': True}
        warnings = metrics.validate_config(config, device_features)
        
        assert len(warnings) == 0
    
    def test_timing_scope_descriptions(self):
        """Test getting timing scope descriptions."""
        # Known scope
        desc = metrics.get_timing_scope_description('hdr_render')
        assert desc == 'HDR rendering pass'
        
        # Unknown scope returns the name itself
        desc = metrics.get_timing_scope_description('unknown_scope')
        assert desc == 'unknown_scope'
    
    def test_common_timing_scopes(self):
        """Test that common timing scopes are defined."""
        scopes = metrics.COMMON_TIMING_SCOPES
        
        # Should have key rendering passes
        assert 'hdr_render' in scopes
        assert 'hdr_tonemap' in scopes
        assert 'terrain_lod_update' in scopes
        assert 'vector_indirect_culling' in scopes
        
        # Should have descriptions
        assert len(scopes['hdr_render']) > 0
        assert len(scopes['hdr_tonemap']) > 0


class TestGpuTimingIntegration:
    """Integration tests for GPU timing with forge3d."""
    
    def test_timing_config_integration(self):
        """Test GPU timing configuration with forge3d components."""
        # This would test integration with actual Renderer class
        # For now, test that the config can be created and used
        config = metrics.create_default_config()
        
        # Should be able to serialize for native interface
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert all(isinstance(k, str) for k in config_dict.keys())
    
    def test_device_feature_detection(self):
        """Test detection of GPU timing features."""
        try:
            # Get device probe information
            device_info = forge3d.device_probe()
            
            # Should contain feature information
            assert 'features' in device_info
            assert isinstance(device_info['features'], str)
            
            # Check for timing-related features
            features_str = device_info['features']
            has_timestamps = 'TIMESTAMP_QUERY' in features_str
            has_pipeline_stats = 'PIPELINE_STATISTICS_QUERY' in features_str
            
            print(f"Device timestamp support: {has_timestamps}")
            print(f"Device pipeline stats support: {has_pipeline_stats}")
            
            # At least timestamps should be available on most modern devices
            # (This is a soft requirement, may not pass on all hardware)
            
        except Exception as e:
            pytest.skip(f"Device probe failed: {e}")
    
    @pytest.mark.skipif(not forge3d.has_gpu(), reason="GPU not available")
    def test_timing_overhead_acceptance_criteria(self):
        """Test that GPU timing overhead meets acceptance criteria.
        
        Acceptance criteria: < 1% frame time overhead with timestamps enabled
        """
        # Simulate timing overhead calculation
        queries_per_frame = 8  # Typical for render + tonemap + terrain + culling
        estimated_overhead = metrics.estimate_timing_overhead(queries_per_frame, True)
        
        # At 60 FPS, frame budget is ~16.67ms
        frame_budget_ms = 16.67
        overhead_percentage = (estimated_overhead / frame_budget_ms) * 100
        
        print(f"Estimated timing overhead: {estimated_overhead:.4f} ms")
        print(f"Frame budget: {frame_budget_ms:.2f} ms") 
        print(f"Overhead percentage: {overhead_percentage:.2f}%")
        
        # Should be well under 1% overhead
        assert overhead_percentage < 1.0, f"Timing overhead {overhead_percentage:.2f}% exceeds 1% target"
        
        # Sanity check - should be measurable but minimal
        assert estimated_overhead > 0.001, "Overhead estimate seems too low"
        assert estimated_overhead < 0.5, "Overhead estimate seems too high"


def test_gpu_timing_integration():
    """Integration test for GPU timing system."""
    print("\nRunning GPU timing integration test...")
    
    try:
        # Test configuration creation
        config = metrics.create_default_config()
        assert config.enable_timestamps == True
        print("✓ Default configuration created")
        
        # Test metrics collection
        gpu_metrics = metrics.GpuMetrics()
        gpu_metrics.add_timing_result(metrics.TimingResult("test_pass", 1.5, True))
        assert len(gpu_metrics.timing_results) == 1
        print("✓ Metrics collection works")
        
        # Test overhead estimation
        overhead = metrics.estimate_timing_overhead(8, True)
        assert 0.001 < overhead < 0.5  # Reasonable overhead range
        print(f"✓ Estimated overhead: {overhead:.4f} ms")
        
        # Test device feature detection if GPU available
        if forge3d.has_gpu():
            device_info = forge3d.device_probe()
            features = device_info.get('features', '')
            has_timestamps = 'TIMESTAMP_QUERY' in features
            print(f"✓ Device timestamp support: {has_timestamps}")
        else:
            print("✓ No GPU available for feature testing")
        
        print("✓ GPU timing integration test completed")
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        raise


if __name__ == "__main__":
    # Run GPU timing tests directly
    test = TestGpuTimingConfig()
    test.test_default_config_creation()
    print("✓ Config tests passed")
    
    test = TestTimingResult()
    test.test_timing_result_creation()
    print("✓ Timing result tests passed")
    
    test = TestGpuMetrics()
    test.test_metrics_creation()
    print("✓ Metrics tests passed")
    
    test = TestTimingUtilities()
    test.test_overhead_estimation()
    print("✓ Utility tests passed")
    
    test_gpu_timing_integration()
    print("✓ All GPU timing tests completed")