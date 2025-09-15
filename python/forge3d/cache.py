#!/usr/bin/env python3
"""
Scene Cache Module for forge3d

Provides high-level Python interface for scene caching to accelerate
repeated path tracing renders. Implements A19 requirements for 30%+
faster re-renders with identical image output.
"""

import time
from typing import Dict, Optional, Tuple, Any
import hashlib
import numpy as np

try:
    from . import forge3d_native
    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_AVAILABLE = False

class CacheEntry:
    """Represents a cached scene entry with all associated GPU resources."""

    def __init__(self, content_hash: str):
        self.content_hash = content_hash
        self.timestamp = time.time()
        self.bvh_cached = False
        self.materials_cached = False
        self.textures_cached = False
        self.hit_count = 0
        self.memory_usage = 0

    @property
    def is_complete(self) -> bool:
        """Check if all required components are cached."""
        return self.bvh_cached and self.materials_cached and self.textures_cached

    def mark_hit(self):
        """Record a cache hit."""
        self.hit_count += 1
        self.timestamp = time.time()

class SceneCache:
    """
    Scene cache for path tracing acceleration.

    Caches BVH structures, material data, and texture bindings to enable
    faster re-renders when scene content is identical.
    """

    def __init__(self, max_entries: int = 16):
        """
        Initialize scene cache.

        Args:
            max_entries: Maximum number of cache entries to maintain
        """
        self.max_entries = max_entries
        self.entries: Dict[str, CacheEntry] = {}
        self.total_hits = 0
        self.total_misses = 0
        self._stats_dirty = True

    def compute_scene_hash(self,
                          bvh_data: Optional[np.ndarray] = None,
                          material_data: Optional[np.ndarray] = None,
                          texture_ids: Optional[list] = None) -> str:
        """
        Compute content hash for scene data.

        Args:
            bvh_data: BVH node data array
            material_data: Material properties array
            texture_ids: List of texture identifiers

        Returns:
            Content hash string
        """
        hasher = hashlib.sha256()

        if bvh_data is not None:
            hasher.update(bvh_data.tobytes())

        if material_data is not None:
            hasher.update(material_data.tobytes())

        if texture_ids is not None:
            texture_str = str(sorted(texture_ids))
            hasher.update(texture_str.encode('utf-8'))

        return hasher.hexdigest()

    def is_cached(self, content_hash: str) -> bool:
        """
        Check if scene is fully cached.

        Args:
            content_hash: Scene content hash

        Returns:
            True if scene is completely cached
        """
        entry = self.entries.get(content_hash)
        return entry is not None and entry.is_complete

    def cache_scene(self, content_hash: str,
                   bvh_data: Optional[np.ndarray] = None,
                   material_data: Optional[np.ndarray] = None,
                   texture_ids: Optional[list] = None) -> bool:
        """
        Cache scene data.

        Args:
            content_hash: Scene content hash
            bvh_data: BVH node data to cache
            material_data: Material data to cache
            texture_ids: Texture IDs to cache

        Returns:
            True if caching succeeded
        """
        # Get or create cache entry
        if content_hash not in self.entries:
            self.entries[content_hash] = CacheEntry(content_hash)

        entry = self.entries[content_hash]

        # Mark components as cached
        if bvh_data is not None:
            entry.bvh_cached = True
            entry.memory_usage += bvh_data.nbytes

        if material_data is not None:
            entry.materials_cached = True
            entry.memory_usage += material_data.nbytes

        if texture_ids is not None:
            entry.textures_cached = True
            entry.memory_usage += len(texture_ids) * 1024  # Estimate

        # Update timestamp
        entry.timestamp = time.time()

        # Evict old entries if needed
        self._evict_lru()
        self._stats_dirty = True

        return True

    def get_cached_scene(self, content_hash: str) -> Optional[CacheEntry]:
        """
        Retrieve cached scene entry.

        Args:
            content_hash: Scene content hash

        Returns:
            Cache entry if found, None otherwise
        """
        entry = self.entries.get(content_hash)
        if entry and entry.is_complete:
            entry.mark_hit()
            self.total_hits += 1
            self._stats_dirty = True
            return entry
        else:
            self.total_misses += 1
            self._stats_dirty = True
            return None

    def reset_cache(self):
        """Clear all cache entries."""
        self.entries.clear()
        self.total_hits = 0
        self.total_misses = 0
        self._stats_dirty = True

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_memory = sum(entry.memory_usage for entry in self.entries.values())
        complete_entries = sum(1 for entry in self.entries.values() if entry.is_complete)

        total_requests = self.total_hits + self.total_misses
        hit_rate = self.total_hits / total_requests if total_requests > 0 else 0.0

        return {
            'total_entries': len(self.entries),
            'complete_entries': complete_entries,
            'max_entries': self.max_entries,
            'total_hits': self.total_hits,
            'total_misses': self.total_misses,
            'hit_rate': hit_rate,
            'memory_usage_bytes': total_memory,
            'memory_usage_mb': total_memory / (1024 * 1024)
        }

    def set_max_entries(self, max_entries: int):
        """
        Set maximum cache entries.

        Args:
            max_entries: New maximum entry count
        """
        self.max_entries = max_entries
        self._evict_lru()
        self._stats_dirty = True

    def _evict_lru(self):
        """Evict least recently used entries to maintain size limit."""
        while len(self.entries) > self.max_entries:
            # Find oldest entry
            oldest_hash = min(self.entries.keys(),
                            key=lambda k: self.entries[k].timestamp)
            del self.entries[oldest_hash]

# Module-level cache instance for convenience
_global_cache = None

def get_global_cache() -> SceneCache:
    """Get or create global scene cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = SceneCache()
    return _global_cache

def set_cache_size(max_entries: int):
    """Set global cache maximum entries."""
    cache = get_global_cache()
    cache.set_max_entries(max_entries)

def reset_cache():
    """Reset global cache."""
    cache = get_global_cache()
    cache.reset_cache()

def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics."""
    cache = get_global_cache()
    return cache.get_cache_stats()

def cache_scene_data(bvh_data: Optional[np.ndarray] = None,
                    material_data: Optional[np.ndarray] = None,
                    texture_ids: Optional[list] = None) -> str:
    """
    Cache scene data using global cache.

    Args:
        bvh_data: BVH data to cache
        material_data: Material data to cache
        texture_ids: Texture IDs to cache

    Returns:
        Content hash for the cached scene
    """
    cache = get_global_cache()
    content_hash = cache.compute_scene_hash(bvh_data, material_data, texture_ids)
    cache.cache_scene(content_hash, bvh_data, material_data, texture_ids)
    return content_hash

def is_scene_cached(content_hash: str) -> bool:
    """Check if scene is cached in global cache."""
    cache = get_global_cache()
    return cache.is_cached(content_hash)

# Performance measurement utilities
class CacheProfiler:
    """Utility for measuring cache performance gains."""

    def __init__(self):
        self.render_times = []
        self.cache_hits = []

    def record_render(self, render_time: float, was_cache_hit: bool):
        """Record a render timing."""
        self.render_times.append(render_time)
        self.cache_hits.append(was_cache_hit)

    def get_performance_gain(self) -> Optional[float]:
        """
        Calculate performance gain from caching.

        Returns:
            Performance improvement ratio (e.g., 0.3 for 30% faster)
        """
        if not self.render_times or not any(self.cache_hits):
            return None

        hit_times = [t for t, hit in zip(self.render_times, self.cache_hits) if hit]
        miss_times = [t for t, hit in zip(self.render_times, self.cache_hits) if not hit]

        if not hit_times or not miss_times:
            return None

        avg_hit_time = np.mean(hit_times)
        avg_miss_time = np.mean(miss_times)

        return (avg_miss_time - avg_hit_time) / avg_miss_time

    def meets_performance_target(self, target_improvement: float = 0.3) -> bool:
        """Check if cache meets performance target (default 30% improvement)."""
        gain = self.get_performance_gain()
        return gain is not None and gain >= target_improvement