#!/usr/bin/env python3
"""Tests for A19: Scene Cache for HQ"""

import pytest
import numpy as np
import time
from unittest.mock import patch, MagicMock

try:
    import forge3d
    from forge3d.cache import (
        SceneCache, CacheEntry, CacheProfiler,
        get_global_cache, reset_cache, cache_scene_data,
        is_scene_cached, get_cache_stats
    )
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False


@pytest.mark.skipif(not CACHE_AVAILABLE, reason="Scene cache module not available")
class TestSceneCache:
    """Test A19: Scene Cache functionality."""

    def test_cache_entry_creation(self):
        """Test cache entry creation and state tracking."""
        entry = CacheEntry("test_hash")
        assert entry.content_hash == "test_hash"
        assert not entry.is_complete
        assert entry.hit_count == 0

    def test_cache_entry_completion(self):
        """Test cache entry completion tracking."""
        entry = CacheEntry("test_hash")

        # Mark components as cached
        entry.bvh_cached = True
        assert not entry.is_complete  # Still incomplete

        entry.materials_cached = True
        assert not entry.is_complete  # Still incomplete

        entry.textures_cached = True
        assert entry.is_complete  # Now complete

    def test_cache_creation(self):
        """Test scene cache creation."""
        cache = SceneCache(max_entries=8)
        assert cache.max_entries == 8
        assert len(cache.entries) == 0
        assert cache.total_hits == 0
        assert cache.total_misses == 0

    def test_scene_hash_computation(self):
        """Test scene content hash computation."""
        cache = SceneCache()

        # Test with numpy arrays
        bvh_data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        material_data = np.array([[0.8, 0.2, 0.1, 1.0]], dtype=np.float32)
        texture_ids = [100, 200, 300]

        hash1 = cache.compute_scene_hash(bvh_data, material_data, texture_ids)
        hash2 = cache.compute_scene_hash(bvh_data, material_data, texture_ids)

        assert hash1 == hash2  # Same input should produce same hash
        assert len(hash1) == 64  # SHA256 hex digest length

    def test_scene_hash_differences(self):
        """Test that different scene data produces different hashes."""
        cache = SceneCache()

        bvh1 = np.array([[1, 2, 3, 4]], dtype=np.float32)
        bvh2 = np.array([[1, 2, 3, 5]], dtype=np.float32)  # Different

        hash1 = cache.compute_scene_hash(bvh_data=bvh1)
        hash2 = cache.compute_scene_hash(bvh_data=bvh2)

        assert hash1 != hash2

    def test_scene_caching(self):
        """Test scene data caching."""
        cache = SceneCache()

        bvh_data = np.random.rand(10, 4).astype(np.float32)
        material_data = np.random.rand(5, 4).astype(np.float32)
        texture_ids = [1, 2, 3]

        content_hash = cache.compute_scene_hash(bvh_data, material_data, texture_ids)

        # Initially not cached
        assert not cache.is_cached(content_hash)

        # Cache the scene
        success = cache.cache_scene(content_hash, bvh_data, material_data, texture_ids)
        assert success

        # Now should be cached
        assert cache.is_cached(content_hash)

    def test_cache_retrieval(self):
        """Test cached scene retrieval."""
        cache = SceneCache()

        bvh_data = np.random.rand(5, 4).astype(np.float32)
        content_hash = cache.compute_scene_hash(bvh_data=bvh_data)

        # Cache BVH only
        cache.cache_scene(content_hash, bvh_data=bvh_data)

        # Should not be retrievable (incomplete)
        entry = cache.get_cached_scene(content_hash)
        assert entry is None

        # Complete the cache
        cache.cache_scene(content_hash, material_data=np.array([[1, 2, 3, 4]], dtype=np.float32), texture_ids=[1])

        # Now should be retrievable
        entry = cache.get_cached_scene(content_hash)
        assert entry is not None
        assert entry.content_hash == content_hash
        assert entry.hit_count == 1

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = SceneCache(max_entries=2)

        # Cache 3 complete scenes
        for i in range(3):
            bvh = np.array([[i, i+1, i+2, i+3]], dtype=np.float32)
            material = np.array([[i*0.1, i*0.2, i*0.3, 1.0]], dtype=np.float32)
            textures = [i]

            hash_val = cache.compute_scene_hash(bvh, material, textures)
            cache.cache_scene(hash_val, bvh, material, textures)

            # Small delay to ensure different timestamps
            time.sleep(0.01)

        # Should only have 2 entries (oldest evicted)
        assert len(cache.entries) <= 2

    def test_cache_reset(self):
        """Test cache reset functionality."""
        cache = SceneCache()

        # Add some entries
        bvh_data = np.random.rand(3, 4).astype(np.float32)
        content_hash = cache.compute_scene_hash(bvh_data=bvh_data)
        cache.cache_scene(content_hash, bvh_data=bvh_data)

        assert len(cache.entries) > 0

        # Reset cache
        cache.reset_cache()

        assert len(cache.entries) == 0
        assert cache.total_hits == 0
        assert cache.total_misses == 0

    def test_cache_statistics(self):
        """Test cache statistics collection."""
        cache = SceneCache()

        # Initially empty stats
        stats = cache.get_cache_stats()
        assert stats['total_entries'] == 0
        assert stats['complete_entries'] == 0
        assert stats['hit_rate'] == 0.0

        # Add a complete cache entry
        bvh_data = np.random.rand(4, 4).astype(np.float32)
        material_data = np.random.rand(2, 4).astype(np.float32)
        content_hash = cache.compute_scene_hash(bvh_data, material_data, [1])
        cache.cache_scene(content_hash, bvh_data, material_data, [1])

        # Check for cache miss
        cache.get_cached_scene("nonexistent_hash")

        # Check for cache hit
        cache.get_cached_scene(content_hash)

        stats = cache.get_cache_stats()
        assert stats['total_entries'] == 1
        assert stats['complete_entries'] == 1
        assert stats['total_hits'] == 1
        assert stats['total_misses'] == 1
        assert stats['hit_rate'] == 0.5
        assert stats['memory_usage_bytes'] > 0

    def test_max_entries_adjustment(self):
        """Test adjusting maximum cache entries."""
        cache = SceneCache(max_entries=5)

        # Fill cache
        for i in range(7):
            bvh = np.array([[i]], dtype=np.float32)
            hash_val = cache.compute_scene_hash(bvh_data=bvh)
            cache.cache_scene(hash_val, bvh_data=bvh)

        assert len(cache.entries) <= 5

        # Reduce max entries
        cache.set_max_entries(3)
        assert len(cache.entries) <= 3


@pytest.mark.skipif(not CACHE_AVAILABLE, reason="Scene cache module not available")
class TestGlobalCache:
    """Test global cache functionality."""

    def test_global_cache_singleton(self):
        """Test global cache singleton behavior."""
        cache1 = get_global_cache()
        cache2 = get_global_cache()
        assert cache1 is cache2  # Should be same instance

    def test_global_cache_operations(self):
        """Test global cache convenience functions."""
        reset_cache()  # Start fresh

        bvh_data = np.random.rand(3, 4).astype(np.float32)
        material_data = np.random.rand(2, 4).astype(np.float32)

        # Cache scene data
        content_hash = cache_scene_data(bvh_data, material_data, [1, 2])

        # Check if cached
        assert is_scene_cached(content_hash)

        # Get stats
        stats = get_cache_stats()
        assert stats['total_entries'] >= 1

    def test_global_cache_reset(self):
        """Test global cache reset."""
        # Add some data
        bvh_data = np.random.rand(2, 4).astype(np.float32)
        cache_scene_data(bvh_data=bvh_data)

        stats_before = get_cache_stats()
        assert stats_before['total_entries'] > 0

        # Reset
        reset_cache()

        stats_after = get_cache_stats()
        assert stats_after['total_entries'] == 0


@pytest.mark.skipif(not CACHE_AVAILABLE, reason="Scene cache module not available")
class TestCacheProfiler:
    """Test cache performance profiling."""

    def test_profiler_creation(self):
        """Test cache profiler creation."""
        profiler = CacheProfiler()
        assert len(profiler.render_times) == 0
        assert len(profiler.cache_hits) == 0

    def test_profiler_recording(self):
        """Test render time recording."""
        profiler = CacheProfiler()

        # Record some renders
        profiler.record_render(1.0, False)  # Miss
        profiler.record_render(0.7, True)   # Hit
        profiler.record_render(0.6, True)   # Hit

        assert len(profiler.render_times) == 3
        assert len(profiler.cache_hits) == 3
        assert profiler.cache_hits == [False, True, True]

    def test_performance_gain_calculation(self):
        """Test performance gain calculation."""
        profiler = CacheProfiler()

        # Record typical cache behavior
        profiler.record_render(1.0, False)  # Miss: 1.0s
        profiler.record_render(1.1, False)  # Miss: 1.1s
        profiler.record_render(0.7, True)   # Hit: 0.7s
        profiler.record_render(0.8, True)   # Hit: 0.8s

        gain = profiler.get_performance_gain()
        assert gain is not None

        # Average miss: 1.05s, Average hit: 0.75s
        # Gain = (1.05 - 0.75) / 1.05 ≈ 0.286 (28.6%)
        assert 0.25 < gain < 0.35

    def test_performance_target_check(self):
        """Test performance target validation."""
        profiler = CacheProfiler()

        # Record performance that meets 30% target
        profiler.record_render(1.0, False)  # Miss
        profiler.record_render(0.6, True)   # Hit (40% improvement)

        assert profiler.meets_performance_target(0.3)  # Should meet 30%
        assert not profiler.meets_performance_target(0.5)  # Should not meet 50%

    def test_insufficient_data_handling(self):
        """Test profiler behavior with insufficient data."""
        profiler = CacheProfiler()

        # No data
        assert profiler.get_performance_gain() is None
        assert not profiler.meets_performance_target()

        # Only misses
        profiler.record_render(1.0, False)
        assert profiler.get_performance_gain() is None

        # Only hits
        profiler = CacheProfiler()
        profiler.record_render(0.5, True)
        assert profiler.get_performance_gain() is None


@pytest.mark.skipif(not CACHE_AVAILABLE, reason="Scene cache module not available")
class TestCacheValidationErrors:
    """Test cache error handling and validation."""

    def test_invalid_hash_handling(self):
        """Test handling of invalid content hashes."""
        cache = SceneCache()

        # Non-existent hash should return None
        entry = cache.get_cached_scene("nonexistent_hash")
        assert entry is None

        # Should record as miss
        stats = cache.get_cache_stats()
        assert stats['total_misses'] > 0

    def test_cache_with_none_data(self):
        """Test caching with None data (edge case)."""
        cache = SceneCache()

        # Should not crash with None data
        content_hash = cache.compute_scene_hash(None, None, None)
        assert content_hash is not None

        success = cache.cache_scene(content_hash, None, None, None)
        assert success

        # Entry should exist but be incomplete
        assert content_hash in cache.entries
        assert not cache.is_cached(content_hash)

    def test_cache_size_zero(self):
        """Test cache with zero max entries."""
        cache = SceneCache(max_entries=0)

        bvh_data = np.array([[1, 2, 3, 4]], dtype=np.float32)
        content_hash = cache.compute_scene_hash(bvh_data=bvh_data)
        cache.cache_scene(content_hash, bvh_data=bvh_data)

        # Should evict immediately
        assert len(cache.entries) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])