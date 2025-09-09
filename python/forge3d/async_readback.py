"""
Async readback system for improved performance

Provides asynchronous texture readback with optional double-buffering
to improve performance in scenarios with frequent readbacks.
"""

import asyncio
import numpy as np
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import time

# Import the Rust async readback functionality (placeholder - would be actual PyO3 bindings)
# from ._forge3d import AsyncReadbackManager, AsyncReadbackConfig

class AsyncReadbackConfig:
    """Configuration for async readback operations"""
    
    def __init__(self, 
                 double_buffered: bool = True,
                 pre_allocate: bool = True, 
                 max_pending_ops: int = 4):
        """
        Args:
            double_buffered: Enable double-buffering for overlapped readbacks
            pre_allocate: Pre-allocate buffers for better performance
            max_pending_ops: Maximum number of pending readback operations
        """
        self.double_buffered = double_buffered
        self.pre_allocate = pre_allocate
        self.max_pending_ops = max_pending_ops

class AsyncReadbackHandle:
    """Handle for a pending async readback operation"""
    
    def __init__(self, future, expected_size: int):
        self._future = future
        self._expected_size = expected_size
        self._completed = False
        self._result = None
    
    async def wait(self) -> np.ndarray:
        """Wait for the readback to complete and get the result as numpy array"""
        if not self._completed:
            self._result = await self._future
            self._completed = True
        
        # Convert bytes to numpy array (assuming RGBA8 format)
        height = int(self._expected_size ** 0.5 / 4)
        width = height
        return np.frombuffer(self._result, dtype=np.uint8).reshape((height, width, 4))
    
    def try_get(self) -> Optional[np.ndarray]:
        """Try to get the result if available (non-blocking)"""
        if self._completed:
            height = int(self._expected_size ** 0.5 / 4)
            width = height  
            return np.frombuffer(self._result, dtype=np.uint8).reshape((height, width, 4))
        
        if self._future.done():
            self._result = self._future.result()
            self._completed = True
            height = int(self._expected_size ** 0.5 / 4)
            width = height
            return np.frombuffer(self._result, dtype=np.uint8).reshape((height, width, 4))
        
        return None
    
    @property
    def expected_size(self) -> int:
        """Get the expected size of the readback data in bytes"""
        return self._expected_size
    
    @property
    def is_complete(self) -> bool:
        """Check if the readback operation is complete"""
        return self._completed or self._future.done()

class AsyncReadbackManager:
    """Async readback manager with double-buffering support"""
    
    def __init__(self, renderer, config: Optional[AsyncReadbackConfig] = None):
        """
        Args:
            renderer: The main Renderer instance
            config: Configuration for async operations
        """
        self.renderer = renderer
        self.config = config or AsyncReadbackConfig()
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._pending_operations = 0
        self._stats = {
            'total_operations': 0,
            'completed_operations': 0,
            'failed_operations': 0,
            'average_time_ms': 0.0,
            'pending_operations': 0,
        }
    
    async def readback_texture_async(self, width: int, height: int) -> AsyncReadbackHandle:
        """Start an async readback operation"""
        if self._pending_operations >= self.config.max_pending_ops:
            raise RuntimeError(f"Too many pending readback operations "
                             f"({self._pending_operations}/{self.config.max_pending_ops})")
        
        self._pending_operations += 1
        self._stats['pending_operations'] = self._pending_operations
        
        # Create future for the readback operation
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self._executor,
            self._sync_readback_wrapper,
            width,
            height
        )
        
        # Wrap future to update stats when complete
        future.add_done_callback(self._on_readback_complete)
        
        expected_size = width * height * 4  # RGBA8
        return AsyncReadbackHandle(future, expected_size)
    
    def readback_texture_sync(self, width: int, height: int) -> np.ndarray:
        """Synchronous readback (fallback for compatibility)"""
        start_time = time.time()
        
        try:
            # Use the existing synchronous readback from the renderer
            rgba_data = self.renderer.render_triangle_rgba()
            self._update_stats(start_time, success=True)
            return rgba_data
        except Exception as e:
            self._update_stats(start_time, success=False)
            raise
    
    def _sync_readback_wrapper(self, width: int, height: int) -> bytes:
        """Wrapper for synchronous readback to run in executor"""
        start_time = time.time()
        
        try:
            # Call the actual renderer readback method
            rgba_array = self.renderer.render_triangle_rgba()
            self._update_stats(start_time, success=True)
            return rgba_array.tobytes()
        except Exception as e:
            self._update_stats(start_time, success=False)
            raise
    
    def _on_readback_complete(self, future):
        """Callback when readback operation completes"""
        self._pending_operations = max(0, self._pending_operations - 1)
        self._stats['pending_operations'] = self._pending_operations
        self._stats['completed_operations'] += 1
    
    def _update_stats(self, start_time: float, success: bool):
        """Update operation statistics"""
        duration_ms = (time.time() - start_time) * 1000
        
        self._stats['total_operations'] += 1
        if success:
            # Update running average
            total_ops = self._stats['total_operations']
            current_avg = self._stats['average_time_ms']
            self._stats['average_time_ms'] = (current_avg * (total_ops - 1) + duration_ms) / total_ops
        else:
            self._stats['failed_operations'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about readback operations"""
        return {
            **self._stats,
            'double_buffered': self.config.double_buffered,
            'pre_allocate': self.config.pre_allocate,
            'max_pending_ops': self.config.max_pending_ops,
        }
    
    def cleanup(self):
        """Clean up resources"""
        self._executor.shutdown(wait=True)

# Example usage and integration
class AsyncRenderer:
    """Extended renderer with async readback capabilities"""
    
    def __init__(self, width: int, height: int, async_config: Optional[AsyncReadbackConfig] = None):
        # This would use the actual Rust Renderer
        # self._renderer = Renderer(width, height)
        self._width = width
        self._height = height
        self._async_manager = AsyncReadbackManager(self, async_config)
    
    async def render_async(self) -> AsyncReadbackHandle:
        """Render and start async readback"""
        # Submit render commands (synchronous)
        # self._renderer.render_triangle()
        
        # Start async readback
        return await self._async_manager.readback_texture_async(self._width, self._height)
    
    def render_sync(self) -> np.ndarray:
        """Synchronous render and readback"""
        return self._async_manager.readback_texture_sync(self._width, self._height)
    
    def get_readback_stats(self) -> Dict[str, Any]:
        """Get readback performance statistics"""
        return self._async_manager.get_stats()
    
    def cleanup(self):
        """Clean up resources"""
        self._async_manager.cleanup()

# Async context manager for convenient usage
class AsyncReadbackContext:
    """Context manager for async readback operations"""
    
    def __init__(self, renderer, config: Optional[AsyncReadbackConfig] = None):
        self.renderer = renderer
        self.config = config
        self._manager = None
    
    async def __aenter__(self):
        self._manager = AsyncReadbackManager(self.renderer, self.config)
        return self._manager
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._manager:
            self._manager.cleanup()

# Utility functions
async def batch_readback(manager: AsyncReadbackManager, 
                        operations: list, 
                        max_concurrent: int = 4) -> list:
    """Perform multiple readback operations with concurrency control"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_readback(width, height):
        async with semaphore:
            handle = await manager.readback_texture_async(width, height)
            return await handle.wait()
    
    tasks = [limited_readback(w, h) for w, h in operations]
    return await asyncio.gather(*tasks)

def benchmark_readback_performance(renderer, num_operations: int = 100):
    """Benchmark synchronous vs asynchronous readback performance"""
    import time
    
    # Benchmark synchronous readback
    start_time = time.time()
    for _ in range(num_operations):
        renderer.render_triangle_rgba()
    sync_time = time.time() - start_time
    
    # Benchmark asynchronous readback
    async def async_benchmark():
        async with AsyncReadbackContext(renderer) as manager:
            handles = []
            start_time = time.time()
            
            # Submit all operations
            for _ in range(num_operations):
                handle = await manager.readback_texture_async(512, 512)
                handles.append(handle)
            
            # Wait for all to complete
            results = await asyncio.gather(*[h.wait() for h in handles])
            return time.time() - start_time, len(results)
    
    async_time, completed = asyncio.run(async_benchmark())
    
    return {
        'sync_time_s': sync_time,
        'async_time_s': async_time,
        'operations': num_operations,
        'completed': completed,
        'sync_ops_per_sec': num_operations / sync_time,
        'async_ops_per_sec': completed / async_time if async_time > 0 else 0,
        'speedup': sync_time / async_time if async_time > 0 else 0,
    }