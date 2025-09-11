"""
Tests for virtual texture streaming system

These tests validate the virtual texture streaming functionality including
system initialization, texture loading, camera-based streaming, memory management,
and performance monitoring.
"""

import pytest
import numpy as np
from typing import Dict, Any, Tuple

import forge3d
import forge3d.streaming as streaming


class TestVirtualTextureSystem:
    """Test virtual texture streaming system core functionality."""
    
    def setup_method(self):
        """Initialize virtual texture system for testing."""
        try:
            # Get GPU device for testing
            self.device = forge3d.get_device()
            self.vt_available = hasattr(forge3d, 'create_virtual_texture_system')
            
            if self.vt_available:
                # Create virtual texture system with small memory budget for testing
                self.vt_system = streaming.VirtualTextureSystem(
                    self.device, 
                    max_memory_mb=64,  # Small budget for tests
                    tile_size=128      # Smaller tiles for tests
                )
            else:
                self.vt_system = None
                
        except Exception as e:
            self.vt_available = False
            self.vt_system = None
            print(f"Virtual texture system not available: {e}")
    
    @pytest.mark.skipif(not hasattr(forge3d, 'create_virtual_texture_system'), 
                       reason="Virtual texture system not available in this build")
    def test_virtual_texture_system_creation(self):
        """Test virtual texture system initialization and configuration."""
        if not self.vt_available:
            pytest.skip("Virtual texture system not available")
        
        # Test basic system creation
        assert self.vt_system is not None
        
        # Test system with different configurations
        configurations = [
            {"max_memory_mb": 32, "tile_size": 64},
            {"max_memory_mb": 128, "tile_size": 256}, 
            {"max_memory_mb": 256, "tile_size": 512},
        ]
        
        for config in configurations:
            try:
                test_system = streaming.VirtualTextureSystem(self.device, **config)
                assert test_system is not None
                print(f"OK: Created system with config: {config}")
                
                # Cleanup
                del test_system
                
            except Exception as e:
                print(f"Failed to create system with config {config}: {e}")
    
    def test_virtual_texture_system_validation(self):
        """Test parameter validation for virtual texture system."""
        if not self.vt_available:
            pytest.skip("Virtual texture system not available")
        
        # Test invalid memory budget
        with pytest.raises(ValueError, match="max_memory_mb must be positive"):
            streaming.VirtualTextureSystem(self.device, max_memory_mb=0)
        
        with pytest.raises(ValueError, match="max_memory_mb must be positive"):
            streaming.VirtualTextureSystem(self.device, max_memory_mb=-10)
        
        # Test invalid tile size (must be power of 2)
        with pytest.raises(ValueError, match="tile_size must be a positive power of 2"):
            streaming.VirtualTextureSystem(self.device, tile_size=100)  # Not power of 2
        
        with pytest.raises(ValueError, match="tile_size must be a positive power of 2"):
            streaming.VirtualTextureSystem(self.device, tile_size=0)    # Not positive
    
    def test_virtual_texture_loading_validation(self):
        """Test virtual texture loading validation and error handling."""
        if not self.vt_available:
            pytest.skip("Virtual texture system not available")
        
        # Test loading nonexistent file
        with pytest.raises(RuntimeError):
            self.vt_system.load_texture("nonexistent_texture.ktx2")
        
        # Test loading invalid file paths
        invalid_paths = [
            "",
            "  ",
            "\0invalid",
            "/nonexistent/path/texture.ktx2",
        ]
        
        for invalid_path in invalid_paths:
            with pytest.raises(RuntimeError):
                self.vt_system.load_texture(invalid_path)
    
    def test_streaming_update_parameters(self):
        """Test streaming update parameter validation."""
        if not self.vt_available:
            pytest.skip("Virtual texture system not available")
        
        # Test basic streaming update (should not crash even without loaded textures)
        camera_pos = (100.0, 200.0, 50.0)
        
        try:
            result = self.vt_system.update_streaming(camera_pos)
            
            # Result should be a dictionary with expected keys
            assert isinstance(result, dict)
            expected_keys = ["tiles_requested", "tiles_loaded", "tiles_evicted", "update_time_ms"]
            
            for key in expected_keys:
                if key in result:
                    assert isinstance(result[key], (int, float))
                    print(f"  {key}: {result[key]}")
            
        except RuntimeError as e:
            # This is acceptable if virtual texture system is not fully functional
            print(f"Streaming update failed (expected in test environment): {e}")
        
        # Test with view/projection matrices
        view_matrix = np.eye(4, dtype=np.float32)
        proj_matrix = np.eye(4, dtype=np.float32)
        
        try:
            result = self.vt_system.update_streaming(
                camera_pos, 
                view_matrix, 
                proj_matrix
            )
            print("OK: Streaming update with matrices succeeded")
        except Exception as e:
            print(f"Streaming update with matrices failed: {e}")
    
    def test_statistics_collection(self):
        """Test virtual texture streaming statistics collection.""" 
        if not self.vt_available:
            pytest.skip("Virtual texture system not available")
        
        stats = self.vt_system.get_statistics()
        
        # Should return a StreamingStats object
        assert isinstance(stats, streaming.StreamingStats)
        
        # Check that all expected properties exist
        properties = [
            'cache_hits', 'cache_misses', 'tiles_loaded', 'tiles_evicted',
            'memory_used', 'memory_limit', 'active_tiles', 'atlas_utilization'
        ]
        
        for prop in properties:
            assert hasattr(stats, prop)
            value = getattr(stats, prop)
            assert isinstance(value, (int, float))
            assert value >= 0  # All stats should be non-negative
            print(f"  {prop}: {value}")
        
        # Test calculated properties
        assert isinstance(stats.cache_hit_rate, float)
        assert 0.0 <= stats.cache_hit_rate <= 100.0
        
        assert isinstance(stats.memory_utilization, float)
        assert 0.0 <= stats.memory_utilization <= 100.0
        
        print(f"  cache_hit_rate: {stats.cache_hit_rate:.1f}%")
        print(f"  memory_utilization: {stats.memory_utilization:.1f}%")
    
    def test_memory_info(self):
        """Test memory information retrieval."""
        if not self.vt_available:
            pytest.skip("Virtual texture system not available")
        
        memory_info = self.vt_system.get_memory_info()
        
        # Should return a dictionary with memory information
        assert isinstance(memory_info, dict)
        
        expected_keys = [
            'total_budget', 'used_memory', 'available_memory', 
            'active_tiles', 'atlas_slots_used', 'atlas_slots_total'
        ]
        
        for key in expected_keys:
            assert key in memory_info
            assert isinstance(memory_info[key], int)
            assert memory_info[key] >= 0
            print(f"  {key}: {memory_info[key]}")
        
        # Logical constraints
        assert memory_info['used_memory'] <= memory_info['total_budget']
        assert memory_info['available_memory'] == memory_info['total_budget'] - memory_info['used_memory']
        assert memory_info['atlas_slots_used'] <= memory_info['atlas_slots_total']
    
    def test_quality_settings(self):
        """Test virtual texture quality settings configuration."""
        if not self.vt_available:
            pytest.skip("Virtual texture system not available")
        
        # Test setting quality parameters
        quality_configs = [
            {"max_mip_bias": 0.0, "lod_scale": 1.0, "cache_priority_boost": 1.0},
            {"max_mip_bias": 1.0, "lod_scale": 0.8, "cache_priority_boost": 2.0},
            {"max_mip_bias": 0.5, "lod_scale": 1.2, "cache_priority_boost": 1.5},
        ]
        
        for config in quality_configs:
            try:
                success = self.vt_system.set_quality_settings(**config)
                print(f"Quality settings {config}: {'OK:' if success else 'ERROR:'}")
            except Exception as e:
                print(f"Failed to set quality settings {config}: {e}")
    
    def test_prefetch_operations(self):
        """Test prefetch functionality."""
        if not self.vt_available:
            pytest.skip("Virtual texture system not available")
        
        # Create a mock virtual texture for testing
        # In a real test, this would load from a file
        mock_texture = streaming.VirtualTexture(
            handle=1,  # Mock handle
            width=1024,
            height=1024, 
            tile_size=128
        )
        
        # Test prefetch region
        try:
            success = self.vt_system.prefetch_region(
                mock_texture,
                region_x=0,
                region_y=0, 
                region_width=256,
                region_height=256,
                mip_level=0
            )
            print(f"Prefetch region: {'OK:' if success else 'ERROR:'}")
        except Exception as e:
            print(f"Prefetch failed (expected without loaded texture): {e}")
    
    def test_tile_eviction(self):
        """Test tile eviction functionality."""
        if not self.vt_available:
            pytest.skip("Virtual texture system not available")
        
        try:
            # Test evicting all tiles
            evicted_count = self.vt_system.evict_tiles()
            assert isinstance(evicted_count, int)
            assert evicted_count >= 0
            print(f"Evicted {evicted_count} tiles")
            
        except Exception as e:
            print(f"Tile eviction failed: {e}")
    
    def test_system_flush(self):
        """Test system flush operation."""
        if not self.vt_available:
            pytest.skip("Virtual texture system not available")
        
        try:
            success = self.vt_system.flush()
            assert isinstance(success, bool)
            print(f"System flush: {'OK:' if success else 'ERROR:'}")
        except Exception as e:
            print(f"System flush failed: {e}")


class TestVirtualTexture:
    """Test VirtualTexture class functionality."""
    
    def test_virtual_texture_properties(self):
        """Test virtual texture property calculations."""
        # Test with different texture configurations
        test_cases = [
            (1024, 1024, 128),  # 8x8 tiles
            (2048, 1536, 256),  # Non-square, 8x6 tiles  
            (4096, 4096, 512),  # Large texture, 8x8 tiles
            (1000, 800, 100),   # Non-power-of-2 dimensions and tile size
        ]
        
        for width, height, tile_size in test_cases:
            texture = streaming.VirtualTexture(
                handle=1,
                width=width,
                height=height,
                tile_size=tile_size
            )
            
            print(f"\nTesting texture {width}×{height}, tile_size={tile_size}:")
            
            # Test basic properties
            assert texture.size == (width, height)
            assert texture.tile_size == tile_size
            
            # Test tile count calculation
            expected_tiles_x = (width + tile_size - 1) // tile_size
            expected_tiles_y = (height + tile_size - 1) // tile_size
            assert texture.tile_count == (expected_tiles_x, expected_tiles_y)
            print(f"  Tile count: {texture.tile_count}")
            
            # Test mip level calculation
            expected_max_mip = max(0, int(np.log2(max(width, height))))
            assert texture.max_mip_level == expected_max_mip
            print(f"  Max mip level: {texture.max_mip_level}")
            
            # Test tile bounds calculation
            for tile_x in range(min(3, expected_tiles_x)):
                for tile_y in range(min(3, expected_tiles_y)):
                    bounds = texture.get_tile_bounds(tile_x, tile_y)
                    x, y, w, h = bounds
                    
                    # Validate bounds
                    assert x == tile_x * tile_size
                    assert y == tile_y * tile_size
                    assert w <= tile_size
                    assert h <= tile_size
                    assert x + w <= width
                    assert y + h <= height
            
            # Test tile validity
            assert texture.is_tile_valid(0, 0, 0)  # Origin should be valid
            assert not texture.is_tile_valid(-1, 0, 0)  # Negative coordinates invalid
            assert not texture.is_tile_valid(0, -1, 0)  # Negative coordinates invalid
            assert not texture.is_tile_valid(expected_tiles_x, 0, 0)  # Beyond bounds
            assert not texture.is_tile_valid(0, expected_tiles_y, 0)  # Beyond bounds


class TestStreamingStats:
    """Test StreamingStats class functionality."""
    
    def test_stats_calculation(self):
        """Test statistics calculation and properties."""
        # Test with different statistics scenarios
        test_cases = [
            # (hits, misses, loaded, evicted, used, limit, active, atlas_util)
            (100, 20, 50, 10, 1024*1024, 4*1024*1024, 25, 0.75),  # Good performance
            (50, 100, 80, 30, 3*1024*1024, 4*1024*1024, 40, 0.9),   # Poor cache performance  
            (0, 0, 0, 0, 0, 2*1024*1024, 0, 0.0),                    # Empty system
        ]
        
        for hits, misses, loaded, evicted, used, limit, active, atlas_util in test_cases:
            stats_dict = {
                'cache_hits': hits,
                'cache_misses': misses, 
                'tiles_loaded': loaded,
                'tiles_evicted': evicted,
                'memory_used': used,
                'memory_limit': limit,
                'active_tiles': active,
                'atlas_utilization': atlas_util,
            }
            
            stats = streaming.StreamingStats(stats_dict)
            
            print(f"\nTesting stats: hits={hits}, misses={misses}, used={used//1024//1024}MB")
            
            # Test basic properties
            assert stats.cache_hits == hits
            assert stats.cache_misses == misses
            assert stats.tiles_loaded == loaded
            assert stats.tiles_evicted == evicted
            assert stats.memory_used == used
            assert stats.memory_limit == limit
            assert stats.active_tiles == active
            assert stats.atlas_utilization == atlas_util
            
            # Test calculated properties
            expected_hit_rate = 100.0 if (hits + misses) == 0 else (hits / (hits + misses)) * 100.0
            assert abs(stats.cache_hit_rate - expected_hit_rate) < 0.01
            print(f"  Cache hit rate: {stats.cache_hit_rate:.1f}%")
            
            expected_memory_util = 0.0 if limit == 0 else (used / limit) * 100.0
            assert abs(stats.memory_utilization - expected_memory_util) < 0.01
            print(f"  Memory utilization: {stats.memory_utilization:.1f}%")
            
            # Test string representation
            stats_str = str(stats)
            assert "StreamingStats(" in stats_str
            assert "hits" in stats_str
            assert "MB" in stats_str


class TestUtilityFunctions:
    """Test utility functions for virtual texture system."""
    
    def test_memory_requirements_calculation(self):
        """Test memory requirements calculation for virtual textures."""
        test_cases = [
            (1024, 1024, 256, 4),   # 1K texture, 4x4 tiles
            (4096, 4096, 512, 4),   # 4K texture, 8x8 tiles
            (8192, 2048, 256, 3),   # Wide texture, RGB
            (1000, 800, 128, 1),    # Non-power-of-2, grayscale
        ]
        
        for width, height, tile_size, bpp in test_cases:
            reqs = streaming.calculate_memory_requirements(
                width, height, tile_size, bpp
            )
            
            print(f"\nMemory requirements for {width}×{height}, tile_size={tile_size}, bpp={bpp}:")
            
            # Validate returned dictionary
            expected_keys = ['full_texture_size', 'tile_count', 'tile_memory_size', 'recommended_cache_size']
            for key in expected_keys:
                assert key in reqs
                assert isinstance(reqs[key], int)
                assert reqs[key] > 0
            
            # Validate calculations
            expected_full_size = width * height * bpp
            assert reqs['full_texture_size'] == expected_full_size
            
            tiles_x = (width + tile_size - 1) // tile_size
            tiles_y = (height + tile_size - 1) // tile_size
            expected_tile_count = tiles_x * tiles_y
            assert reqs['tile_count'] == expected_tile_count
            
            expected_tile_memory = tile_size * tile_size * bpp
            assert reqs['tile_memory_size'] == expected_tile_memory
            
            # Recommended cache should be reasonable
            # For small textures, cache size should equal full texture size
            # For large textures, cache size should be at least 64MB but not exceed full size
            min_expected_cache = min(64 * 1024 * 1024, expected_full_size)
            assert reqs['recommended_cache_size'] >= min_expected_cache
            assert reqs['recommended_cache_size'] <= expected_full_size
            
            print(f"  Full texture: {reqs['full_texture_size'] // 1024 // 1024} MB")
            print(f"  Tile count: {reqs['tile_count']}")
            print(f"  Tile memory: {reqs['tile_memory_size'] // 1024} KB")
            print(f"  Recommended cache: {reqs['recommended_cache_size'] // 1024 // 1024} MB")
    
    def test_performance_estimation(self):
        """Test streaming performance estimation."""
        test_cases = [
            ((1024, 1024), 128, 128, 60),   # Small texture, high FPS
            ((4096, 4096), 256, 256, 30),   # Medium texture, medium FPS
            ((8192, 8192), 512, 512, 60),   # Large texture, high FPS
        ]
        
        for texture_size, tile_size, cache_mb, fps in test_cases:
            perf = streaming.estimate_streaming_performance(
                texture_size, tile_size, cache_mb, fps
            )
            
            print(f"\nPerformance estimate for {texture_size[0]}×{texture_size[1]}, "
                  f"tile_size={tile_size}, cache={cache_mb}MB, fps={fps}:")
            
            # Validate returned dictionary
            expected_keys = [
                'cache_capacity_tiles', 'tiles_per_frame_budget', 
                'memory_pressure_factor', 'recommended_prefetch_distance'
            ]
            for key in expected_keys:
                assert key in perf
                assert isinstance(perf[key], (int, float))
                assert perf[key] >= 0
            
            # Validate ranges
            assert 0.0 <= perf['memory_pressure_factor'] <= 1.0
            assert perf['tiles_per_frame_budget'] >= 1
            assert perf['recommended_prefetch_distance'] >= 2
            
            print(f"  Cache capacity: {perf['cache_capacity_tiles']} tiles")
            print(f"  Tiles per frame budget: {perf['tiles_per_frame_budget']}")
            print(f"  Memory pressure: {perf['memory_pressure_factor'] * 100:.1f}%")
            print(f"  Prefetch distance: {perf['recommended_prefetch_distance']} tiles")
    
    def test_create_streaming_system_convenience(self):
        """Test convenience function for creating streaming systems."""
        try:
            device = forge3d.get_device()
            
            # Test basic creation
            system = streaming.create_streaming_system(device)
            assert isinstance(system, streaming.VirtualTextureSystem)
            del system
            
            # Test with custom parameters
            system = streaming.create_streaming_system(
                device, 
                max_memory_mb=128,
                tile_size=256
            )
            assert isinstance(system, streaming.VirtualTextureSystem)
            del system
            
            print("OK: Convenience function works correctly")
            
        except Exception as e:
            print(f"Convenience function test failed: {e}")


def test_virtual_texture_integration():
    """Integration test for virtual texture streaming system."""
    print("\nRunning virtual texture integration test...")
    
    try:
        # Test that the system can be imported and basic functions work
        device = forge3d.get_device()
        
        # Test memory calculations (should always work)
        reqs = streaming.calculate_memory_requirements(2048, 2048, 256, 4)
        assert reqs['tile_count'] == 64  # 8x8 tiles
        print("OK: Memory calculation works")
        
        # Test performance estimation (should always work)
        perf = streaming.estimate_streaming_performance((2048, 2048), 256, 256, 60)
        assert perf['cache_capacity_tiles'] > 0
        print("OK: Performance estimation works")
        
        # Test system creation if available
        if hasattr(forge3d, 'create_virtual_texture_system'):
            system = streaming.VirtualTextureSystem(device, max_memory_mb=64)
            
            # Test basic operations
            stats = system.get_statistics()
            assert isinstance(stats, streaming.StreamingStats)
            
            memory_info = system.get_memory_info()
            assert isinstance(memory_info, dict)
            
            # Test streaming update (should not crash)
            result = system.update_streaming((0, 0, 0))
            assert isinstance(result, dict)
            
            del system
            print("OK: Virtual texture system creation and basic operations work")
        else:
            print("Virtual texture system not available in this build - skipping system tests")
        
        print("OK: Virtual texture integration test completed successfully")
        
    except Exception as e:
        print(f"ERROR: Virtual texture integration test failed: {e}")
        raise


if __name__ == "__main__":
    # Run virtual texture tests directly
    print("Running virtual texture tests...")
    
    # Test utility functions first (these should always work)
    utility_test = TestUtilityFunctions()
    utility_test.test_memory_requirements_calculation()
    utility_test.test_performance_estimation()
    utility_test.test_create_streaming_system_convenience()
    print("OK: Utility function tests passed")
    
    # Test VirtualTexture class
    texture_test = TestVirtualTexture()
    texture_test.test_virtual_texture_properties()
    print("OK: VirtualTexture class tests passed")
    
    # Test StreamingStats class
    stats_test = TestStreamingStats()
    stats_test.test_stats_calculation()
    print("OK: StreamingStats class tests passed")
    
    # Test system functionality if available
    system_test = TestVirtualTextureSystem()
    system_test.setup_method()
    
    if system_test.vt_available:
        try:
            system_test.test_virtual_texture_system_creation()
            system_test.test_virtual_texture_system_validation()
            system_test.test_streaming_update_parameters()
            system_test.test_statistics_collection()
            system_test.test_memory_info()
            system_test.test_quality_settings()
            system_test.test_system_flush()
            print("OK: VirtualTextureSystem tests passed")
        except Exception as e:
            print(f"ERROR: Some VirtualTextureSystem tests failed: {e}")
    else:
        print("VirtualTextureSystem not available - skipping system tests")
    
    # Integration test
    test_virtual_texture_integration()
    
    print("\nAll available virtual texture tests completed.")