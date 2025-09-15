"""
O1: Memory management and staging ring utilities

This module provides Python interfaces for memory management features including
staging buffer rings and memory pool statistics.
"""

from typing import Dict, Any, Optional

# Try to import native extension, fall back to None if not available
try:
    import forge3d._forge3d as _core
except ImportError:
    _core = None


def staging_stats() -> Dict[str, Any]:
    """
    Get statistics for staging buffer rings.
    
    Returns:
        Dict containing:
        - bytes_in_flight: Total bytes currently in-flight
        - current_ring_index: Index of currently active ring buffer
        - buffer_stalls: Number of buffer stalls encountered
        - ring_count: Total number of ring buffers
        - buffer_size: Size of each ring buffer in bytes
    """
    if _core is not None:
        try:
            return _core.staging_stats()
        except AttributeError:
            pass
    # Return default stats if staging rings not available
    return {
            "bytes_in_flight": 0,
            "current_ring_index": 0,
            "buffer_stalls": 0,
            "ring_count": 3,
            "buffer_size": 1048576,  # 1MB default
        }


def pool_stats() -> Dict[str, Any]:
    """
    Get statistics for GPU memory pools.
    
    Returns:
        Dict containing:
        - total_allocated: Total bytes allocated from pools
        - total_freed: Total bytes freed back to pools
        - fragmentation_ratio: Fragmentation as a ratio (0.0-1.0)
        - active_blocks: Number of currently allocated blocks
        - pool_count: Number of memory pools
        - largest_free_block: Size of largest free block
    """
    if _core is not None:
        try:
            return _core.pool_stats()
        except AttributeError:
            pass
    # Return default stats if memory pools not available
    return {
            "total_allocated": 0,
            "total_freed": 0,
            "fragmentation_ratio": 0.0,
            "active_blocks": 0,
            "pool_count": 18,  # Power-of-two buckets from 64B to 8MB
            "largest_free_block": 0,
        }


def allocate_from_pool(size: int) -> Optional[Dict[str, Any]]:
    """
    Allocate a block from the memory pool system.
    
    Args:
        size: Size to allocate in bytes
        
    Returns:
        Dict with block info or None if allocation failed:
        - id: Unique block ID
        - size: Actual allocated size (rounded up to bucket size)
        - offset: Offset within the pool buffer
        - pool_id: ID of the pool this block came from
    """
    try:
        return _core.allocate_from_pool(size)
    except AttributeError:
        return None


def deallocate_pool_block(block_id: int) -> bool:
    """
    Deallocate a block from the memory pool system.
    
    Args:
        block_id: ID of the block to deallocate
        
    Returns:
        True if deallocation succeeded, False otherwise
    """
    try:
        return _core.deallocate_pool_block(block_id)
    except AttributeError:
        return False


class StagingRingManager:
    """
    High-level interface for managing staging buffer rings from Python.
    """
    
    def __init__(self, ring_count: int = 3, buffer_size: int = 1024 * 1024):
        """
        Initialize staging ring manager.
        
        Args:
            ring_count: Number of buffers in the ring (typically 3)
            buffer_size: Size of each buffer in bytes
        """
        self.ring_count = ring_count
        self.buffer_size = buffer_size
        self._initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize the staging ring system.
        
        Returns:
            True if initialization succeeded, False otherwise
        """
        try:
            _core.init_staging_rings(self.ring_count, self.buffer_size)
            self._initialized = True
            return True
        except AttributeError:
            # Staging rings not available in this build
            return False
    
    def stats(self) -> Dict[str, Any]:
        """Get current staging ring statistics."""
        return staging_stats()
    
    def is_initialized(self) -> bool:
        """Check if staging rings are initialized."""
        return self._initialized


class MemoryPoolManager:
    """
    High-level interface for GPU memory pool management from Python.
    """
    
    def __init__(self):
        """Initialize memory pool manager."""
        self._initialized = False
    
    def initialize(self, pool_sizes: Optional[list] = None) -> bool:
        """
        Initialize GPU memory pools.
        
        Args:
            pool_sizes: List of pool sizes in bytes, or None for defaults
        
        Returns:
            True if initialization succeeded, False otherwise
        """
        if pool_sizes is None:
            # Default power-of-two buckets from 64B to 8MB
            pool_sizes = [64 * (2 ** i) for i in range(17)]  # 64B to 8MB
        
        try:
            _core.init_memory_pools(pool_sizes)
            self._initialized = True
            return True
        except AttributeError:
            # Memory pools not available in this build
            return False
    
    def stats(self) -> Dict[str, Any]:
        """Get current memory pool statistics."""
        return pool_stats()
    
    def defragment(self) -> Dict[str, Any]:
        """
        Trigger memory pool defragmentation.
        
        Returns:
            Dict with defragmentation statistics:
            - blocks_moved: Number of blocks moved
            - bytes_compacted: Bytes of memory compacted
            - time_ms: Time taken for defragmentation in milliseconds
        """
        try:
            return _core.defragment_memory_pools()
        except AttributeError:
            return {
                "blocks_moved": 0,
                "bytes_compacted": 0,
                "time_ms": 0.0
            }
    
    def is_initialized(self) -> bool:
        """Check if memory pools are initialized."""
        return self._initialized


# Global instances for convenience
staging_manager = StagingRingManager()
pool_manager = MemoryPoolManager()


def init_memory_system(
    staging_rings: bool = True,
    memory_pools: bool = True,
    ring_count: int = 3,
    buffer_size: int = 1024 * 1024
) -> Dict[str, bool]:
    """
    Initialize the complete memory management system.
    
    Args:
        staging_rings: Whether to initialize staging rings
        memory_pools: Whether to initialize memory pools
        ring_count: Number of staging ring buffers
        buffer_size: Size of each staging buffer
    
    Returns:
        Dict indicating which systems were successfully initialized
    """
    results = {}
    
    if staging_rings:
        staging_manager.ring_count = ring_count
        staging_manager.buffer_size = buffer_size
        results["staging_rings"] = staging_manager.initialize()
    else:
        results["staging_rings"] = False
    
    if memory_pools:
        results["memory_pools"] = pool_manager.initialize()
    else:
        results["memory_pools"] = False
    
    return results


def memory_report() -> Dict[str, Any]:
    """
    Generate a comprehensive memory usage report.
    
    Returns:
        Dict containing staging and pool statistics
    """
    return {
        "staging": staging_stats(),
        "pools": pool_stats(),
        "system_initialized": {
            "staging_rings": staging_manager.is_initialized(),
            "memory_pools": pool_manager.is_initialized(),
        }
    }