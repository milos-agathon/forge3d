"""
Virtual texture streaming API

This module provides high-level Python access to the virtual texture streaming system,
allowing applications to manage large textures that don't fit in GPU memory through
on-demand tile loading and LRU cache management.

The virtual texture system provides:
- Support for very large textures (up to 16K x 16K) with efficient memory usage
- Automatic tile loading based on camera position and view frustum
- LRU cache management for resident tiles
- GPU feedback buffer for tile access pattern analysis
- Integration with staging rings and memory pools for optimal performance

Usage:
    import forge3d.streaming as streaming
    
    # Initialize virtual texture system
    vt_system = streaming.VirtualTextureSystem(device, max_memory_mb=256)
    
    # Load a virtual texture
    texture = vt_system.load_texture("large_texture.ktx2")
    
    # Update streaming based on camera
    camera_pos = (1000, 2000, 500)
    tiles_loaded = vt_system.update_streaming(camera_pos, view_matrix, proj_matrix)
    
    # Get performance statistics
    stats = vt_system.get_statistics()
    print(f"Cache hit rate: {stats['cache_hit_rate']:.1f}%")
"""

import forge3d
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np


class StreamingStats:
    """Virtual texture streaming performance statistics."""
    
    def __init__(self, stats_dict: Dict[str, Any]):
        """Initialize from native statistics dictionary."""
        self.cache_hits = stats_dict.get('cache_hits', 0)
        self.cache_misses = stats_dict.get('cache_misses', 0)
        self.tiles_loaded = stats_dict.get('tiles_loaded', 0)
        self.tiles_evicted = stats_dict.get('tiles_evicted', 0)
        self.memory_used = stats_dict.get('memory_used', 0)
        self.memory_limit = stats_dict.get('memory_limit', 0)
        self.active_tiles = stats_dict.get('active_tiles', 0)
        self.atlas_utilization = stats_dict.get('atlas_utilization', 0.0)
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate as percentage."""
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 100.0
        return (self.cache_hits / total_requests) * 100.0
    
    @property
    def memory_utilization(self) -> float:
        """Calculate memory utilization as percentage."""
        if self.memory_limit == 0:
            return 0.0
        return (self.memory_used / self.memory_limit) * 100.0
    
    def __str__(self) -> str:
        """Format statistics for display."""
        return (f"StreamingStats(\n"
                f"  Cache: {self.cache_hits} hits, {self.cache_misses} misses "
                f"({self.cache_hit_rate:.1f}% hit rate)\n"
                f"  Tiles: {self.active_tiles} active, {self.tiles_loaded} loaded, "
                f"{self.tiles_evicted} evicted\n"
                f"  Memory: {self.memory_used / 1024 / 1024:.1f} MB / "
                f"{self.memory_limit / 1024 / 1024:.1f} MB "
                f"({self.memory_utilization:.1f}% used)\n"
                f"  Atlas: {self.atlas_utilization * 100:.1f}% utilized\n"
                f")")


class VirtualTexture:
    """Represents a virtual texture that can be streamed on demand."""
    
    def __init__(self, handle: int, width: int, height: int, tile_size: int):
        """Initialize virtual texture wrapper."""
        self._handle = handle
        self._width = width
        self._height = height
        self._tile_size = tile_size
        self._tiles_x = (width + tile_size - 1) // tile_size
        self._tiles_y = (height + tile_size - 1) // tile_size
        self._max_mip = max(0, int(np.log2(max(width, height))))
    
    @property
    def handle(self) -> int:
        """Get native handle for this virtual texture."""
        return self._handle
    
    @property
    def size(self) -> Tuple[int, int]:
        """Get texture dimensions (width, height)."""
        return (self._width, self._height)
    
    @property
    def tile_size(self) -> int:
        """Get tile size in pixels."""
        return self._tile_size
    
    @property
    def tile_count(self) -> Tuple[int, int]:
        """Get number of tiles (x, y)."""
        return (self._tiles_x, self._tiles_y)
    
    @property
    def max_mip_level(self) -> int:
        """Get maximum mip level for this texture."""
        return self._max_mip
    
    def get_tile_bounds(self, tile_x: int, tile_y: int) -> Tuple[int, int, int, int]:
        """Get pixel bounds for a specific tile (x, y, width, height)."""
        x = tile_x * self._tile_size
        y = tile_y * self._tile_size
        w = min(self._tile_size, self._width - x)
        h = min(self._tile_size, self._height - y)
        return (x, y, w, h)
    
    def is_tile_valid(self, tile_x: int, tile_y: int, mip_level: int = 0) -> bool:
        """Check if tile coordinates are valid for this texture."""
        if mip_level < 0 or mip_level > self._max_mip:
            return False
        
        scale = 1 << mip_level
        tiles_x = max(1, self._tiles_x // scale)
        tiles_y = max(1, self._tiles_y // scale)
        
        return 0 <= tile_x < tiles_x and 0 <= tile_y < tiles_y


class VirtualTextureSystem:
    """High-level virtual texture streaming system."""
    
    def __init__(self, device: Any, max_memory_mb: int = 256, tile_size: int = 256):
        """
        Initialize virtual texture system.
        
        Args:
            device: GPU device handle
            max_memory_mb: Maximum memory budget in megabytes
            tile_size: Tile size in pixels (must be power of 2)
        """
        if not hasattr(forge3d, 'create_virtual_texture_system'):
            raise RuntimeError("Virtual texture streaming not available in this build")
        
        # Validate parameters
        if max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        
        if tile_size <= 0 or (tile_size & (tile_size - 1)) != 0:
            raise ValueError("tile_size must be a positive power of 2")
        
        self._device = device
        self._tile_size = tile_size
        self._max_memory = max_memory_mb * 1024 * 1024
        
        # Create native virtual texture system
        try:
            self._handle = forge3d.create_virtual_texture_system(
                device, 
                self._max_memory, 
                tile_size
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create virtual texture system: {e}")
        
        self._textures: Dict[int, VirtualTexture] = {}
        self._next_texture_id = 1
    
    def load_texture(self, path: str, **kwargs) -> VirtualTexture:
        """
        Load a texture for virtual streaming.
        
        Args:
            path: Path to texture file (supports KTX2, PNG, etc.)
            **kwargs: Additional loading options
        
        Returns:
            VirtualTexture instance for streaming
        
        Raises:
            FileNotFoundError: If texture file doesn't exist
            RuntimeError: If texture loading fails
        """
        if not hasattr(forge3d, 'load_virtual_texture'):
            raise RuntimeError("Virtual texture loading not available")
        
        try:
            # Load texture through native interface
            result = forge3d.load_virtual_texture(self._handle, path, **kwargs)
            
            # Extract texture information
            handle = result['handle']
            width = result['width']
            height = result['height']
            
            # Create Python wrapper
            texture = VirtualTexture(handle, width, height, self._tile_size)
            self._textures[handle] = texture
            
            return texture
            
        except Exception as e:
            raise RuntimeError(f"Failed to load virtual texture '{path}': {e}")
    
    def update_streaming(
        self, 
        camera_position: Tuple[float, float, float],
        view_matrix: Optional[np.ndarray] = None,
        projection_matrix: Optional[np.ndarray] = None,
        texture_handle: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Update virtual texture streaming based on camera position.
        
        Args:
            camera_position: Camera position (x, y, z)
            view_matrix: 4x4 view transformation matrix (optional)
            projection_matrix: 4x4 projection matrix (optional)
            texture_handle: Specific texture to update (None for all)
        
        Returns:
            Dictionary with update results:
            - tiles_requested: Number of tiles requested for loading
            - tiles_loaded: Number of tiles actually loaded
            - tiles_evicted: Number of tiles evicted from cache
            - update_time_ms: Time spent updating in milliseconds
        """
        if not hasattr(forge3d, 'update_virtual_texture_streaming'):
            raise RuntimeError("Virtual texture streaming update not available")
        
        try:
            # Prepare matrices (convert to flat arrays if provided)
            view_data = view_matrix.flatten().tolist() if view_matrix is not None else None
            proj_data = projection_matrix.flatten().tolist() if projection_matrix is not None else None
            
            # Update through native interface
            result = forge3d.update_virtual_texture_streaming(
                self._handle,
                camera_position,
                view_data,
                proj_data,
                texture_handle
            )
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to update virtual texture streaming: {e}")
    
    def prefetch_region(
        self,
        texture: VirtualTexture,
        region_x: int, 
        region_y: int,
        region_width: int,
        region_height: int,
        mip_level: int = 0
    ) -> bool:
        """
        Prefetch a specific region of a virtual texture.
        
        Args:
            texture: Virtual texture to prefetch
            region_x: Region X coordinate in pixels
            region_y: Region Y coordinate in pixels  
            region_width: Region width in pixels
            region_height: Region height in pixels
            mip_level: Mip level to prefetch
        
        Returns:
            True if prefetch was successful
        """
        if not hasattr(forge3d, 'prefetch_virtual_texture_region'):
            raise RuntimeError("Virtual texture prefetch not available")
        
        try:
            return forge3d.prefetch_virtual_texture_region(
                self._handle,
                texture.handle,
                region_x,
                region_y, 
                region_width,
                region_height,
                mip_level
            )
        except Exception as e:
            print(f"Warning: Failed to prefetch region: {e}")
            return False
    
    def evict_tiles(self, texture: Optional[VirtualTexture] = None) -> int:
        """
        Manually evict tiles from cache.
        
        Args:
            texture: Specific texture to evict tiles from (None for all)
        
        Returns:
            Number of tiles evicted
        """
        if not hasattr(forge3d, 'evict_virtual_texture_tiles'):
            return 0
        
        try:
            texture_handle = texture.handle if texture else None
            return forge3d.evict_virtual_texture_tiles(self._handle, texture_handle)
        except Exception as e:
            print(f"Warning: Failed to evict tiles: {e}")
            return 0
    
    def get_statistics(self) -> StreamingStats:
        """
        Get virtual texture streaming performance statistics.
        
        Returns:
            StreamingStats with current performance metrics
        """
        if not hasattr(forge3d, 'get_virtual_texture_stats'):
            # Return empty stats if not available
            return StreamingStats({})
        
        try:
            stats_dict = forge3d.get_virtual_texture_stats(self._handle)
            return StreamingStats(stats_dict)
        except Exception as e:
            print(f"Warning: Failed to get streaming statistics: {e}")
            return StreamingStats({})
    
    def get_memory_info(self) -> Dict[str, int]:
        """
        Get memory usage information.
        
        Returns:
            Dictionary with memory information:
            - total_budget: Total memory budget in bytes
            - used_memory: Currently used memory in bytes
            - available_memory: Available memory in bytes
            - active_tiles: Number of active tiles in cache
            - atlas_slots_used: Number of atlas slots in use
            - atlas_slots_total: Total number of atlas slots
        """
        stats = self.get_statistics()
        
        available = max(0, stats.memory_limit - stats.memory_used)
        
        return {
            'total_budget': stats.memory_limit,
            'used_memory': stats.memory_used,
            'available_memory': available,
            'active_tiles': stats.active_tiles,
            'atlas_slots_used': int(stats.atlas_utilization * 100),  # Estimate
            'atlas_slots_total': 100,  # Placeholder - would need actual value
        }
    
    def set_quality_settings(
        self,
        max_mip_bias: float = 0.0,
        lod_scale: float = 1.0,
        cache_priority_boost: float = 1.0
    ) -> bool:
        """
        Configure quality settings for virtual texture streaming.
        
        Args:
            max_mip_bias: Maximum mip level bias for LOD calculations
            lod_scale: Scale factor for level-of-detail calculations  
            cache_priority_boost: Priority boost for recently accessed tiles
        
        Returns:
            True if settings were applied successfully
        """
        if not hasattr(forge3d, 'set_virtual_texture_quality'):
            return False
        
        try:
            return forge3d.set_virtual_texture_quality(
                self._handle,
                max_mip_bias,
                lod_scale, 
                cache_priority_boost
            )
        except Exception as e:
            print(f"Warning: Failed to set quality settings: {e}")
            return False
    
    def flush(self) -> bool:
        """
        Flush any pending streaming operations.
        
        Returns:
            True if flush completed successfully
        """
        if not hasattr(forge3d, 'flush_virtual_texture_system'):
            return True  # No-op if not available
        
        try:
            return forge3d.flush_virtual_texture_system(self._handle)
        except Exception as e:
            print(f"Warning: Failed to flush virtual texture system: {e}")
            return False
    
    def __del__(self):
        """Cleanup virtual texture system."""
        try:
            if hasattr(self, '_handle') and hasattr(forge3d, 'destroy_virtual_texture_system'):
                forge3d.destroy_virtual_texture_system(self._handle)
        except:
            pass  # Ignore cleanup errors


# Convenience functions for common operations

def create_streaming_system(device: Any, **kwargs) -> VirtualTextureSystem:
    """
    Create a virtual texture streaming system with default settings.
    
    Args:
        device: GPU device handle
        **kwargs: Additional options passed to VirtualTextureSystem
    
    Returns:
        Configured VirtualTextureSystem instance
    """
    return VirtualTextureSystem(device, **kwargs)


def calculate_memory_requirements(
    texture_width: int,
    texture_height: int, 
    tile_size: int = 256,
    bytes_per_pixel: int = 4
) -> Dict[str, int]:
    """
    Calculate memory requirements for streaming a virtual texture.
    
    Args:
        texture_width: Texture width in pixels
        texture_height: Texture height in pixels
        tile_size: Tile size in pixels
        bytes_per_pixel: Bytes per pixel (4 for RGBA8)
    
    Returns:
        Dictionary with memory calculations:
        - full_texture_size: Size if loaded entirely
        - tile_count: Total number of tiles
        - tile_memory_size: Memory per tile
        - recommended_cache_size: Recommended cache size
    """
    # Calculate tile dimensions
    tiles_x = (texture_width + tile_size - 1) // tile_size
    tiles_y = (texture_height + tile_size - 1) // tile_size
    total_tiles = tiles_x * tiles_y
    
    # Memory calculations
    full_size = texture_width * texture_height * bytes_per_pixel
    tile_memory = tile_size * tile_size * bytes_per_pixel
    
    # Recommend cache size for ~10% of tiles (minimum 64 MB, maximum full texture size)
    min_cache = 64 * 1024 * 1024  # 64 MB minimum
    target_cache = max(1, total_tiles // 10) * tile_memory  # At least 1 tile
    recommended_cache = min(full_size, max(min_cache, target_cache))
    
    return {
        'full_texture_size': full_size,
        'tile_count': total_tiles,
        'tile_memory_size': tile_memory,
        'recommended_cache_size': recommended_cache,
    }


def estimate_streaming_performance(
    texture_size: Tuple[int, int],
    tile_size: int = 256,
    cache_size_mb: int = 256,
    target_fps: int = 60
) -> Dict[str, float]:
    """
    Estimate streaming performance characteristics.
    
    Args:
        texture_size: Texture dimensions (width, height)
        tile_size: Tile size in pixels
        cache_size_mb: Cache size in megabytes
        target_fps: Target frame rate
    
    Returns:
        Dictionary with performance estimates:
        - cache_capacity_tiles: Number of tiles that fit in cache
        - tiles_per_frame_budget: Tiles loadable per frame at target FPS
        - memory_pressure_factor: Estimated memory pressure (0-1)
        - recommended_prefetch_distance: Tiles to prefetch ahead
    """
    width, height = texture_size
    
    # Calculate tile counts and memory
    tiles_x = (width + tile_size - 1) // tile_size
    tiles_y = (height + tile_size - 1) // tile_size
    
    tile_memory = tile_size * tile_size * 4  # RGBA8
    cache_capacity = (cache_size_mb * 1024 * 1024) // tile_memory
    
    # Performance estimates (simplified)
    frame_budget_ms = 1000.0 / target_fps
    tiles_per_frame = max(1, int(frame_budget_ms / 2.0))  # Assume ~2ms per tile
    
    memory_pressure = min(1.0, (tiles_x * tiles_y) / max(1, cache_capacity))
    prefetch_distance = max(2, int(cache_capacity * 0.1))
    
    return {
        'cache_capacity_tiles': cache_capacity,
        'tiles_per_frame_budget': tiles_per_frame,
        'memory_pressure_factor': memory_pressure,
        'recommended_prefetch_distance': prefetch_distance,
    }