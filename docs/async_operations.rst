Async Operations
================

forge3d provides asynchronous and multi-threaded operations for improved performance and responsiveness. This includes async readback systems, multi-threaded command recording, and non-blocking GPU operations.

.. note::
   Async operations require careful memory management and proper resource cleanup to maintain the 512 MiB GPU memory budget.

Async Readback System
----------------------

**Overview**

The async readback system allows non-blocking GPU-to-CPU data transfers with double-buffering and automatic resource management:

.. code-block:: python

    import forge3d as f3d
    import numpy as np
    
    # Create renderer
    renderer = f3d.Renderer(512, 512)
    
    # Standard synchronous readback
    sync_image = renderer.render_triangle_rgba()  # Blocks until complete
    
    # TODO: Async readback (when implemented)
    # async_handle = renderer.render_triangle_rgba_async()
    # async_image = async_handle.get()  # Non-blocking until ready

**Double-Buffered Operations**

Double-buffering prevents pipeline stalls:

.. code-block:: python

    def render_sequence_async(renderer, num_frames: int):
        """Example of async rendering pattern."""
        
        frames = []
        
        for frame_idx in range(num_frames):
            # Render current frame
            current_image = renderer.render_triangle_rgba()
            frames.append(current_image)
            
            # In a full async implementation:
            # - Start rendering frame N+1 while processing frame N
            # - Use double-buffered command buffers
            # - Pipeline GPU and CPU operations
        
        return frames

**Buffer Pooling**

Efficient buffer management for async operations:

.. code-block:: python

    class AsyncBufferPool:
        """Buffer pool for async readback operations."""
        
        def __init__(self, buffer_size: int, pool_size: int = 3):
            self.buffer_size = buffer_size
            self.available_buffers = []
            self.used_buffers = []
            
            # Pre-allocate buffers
            for _ in range(pool_size):
                buffer = np.zeros(buffer_size, dtype=np.uint8)
                self.available_buffers.append(buffer)
        
        def get_buffer(self) -> np.ndarray:
            """Get an available buffer from the pool."""
            if self.available_buffers:
                buffer = self.available_buffers.pop()
                self.used_buffers.append(buffer)
                return buffer
            else:
                # Allocate new buffer if pool is empty
                buffer = np.zeros(self.buffer_size, dtype=np.uint8)
                self.used_buffers.append(buffer)
                return buffer
        
        def return_buffer(self, buffer: np.ndarray):
            """Return a buffer to the pool."""
            if buffer in self.used_buffers:
                self.used_buffers.remove(buffer)
                self.available_buffers.append(buffer)
    
    # Usage example
    pool = AsyncBufferPool(buffer_size=512*512*4, pool_size=3)
    buffer = pool.get_buffer()
    # ... use buffer for readback ...
    pool.return_buffer(buffer)

Multi-threaded Command Recording
--------------------------------

**Thread Pool Pattern**

Use ThreadPoolExecutor for parallel rendering tasks:

.. code-block:: python

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    
    def render_task(task_id: int, width: int, height: int) -> dict:
        """Individual rendering task for thread pool."""
        try:
            start_time = time.perf_counter()
            
            # Create renderer (each thread gets its own)
            renderer = f3d.Renderer(width, height)
            
            # Render content
            image = renderer.render_triangle_rgba()
            
            end_time = time.perf_counter()
            
            return {
                'task_id': task_id,
                'image': image,
                'render_time': end_time - start_time,
                'success': True
            }
            
        except Exception as e:
            return {
                'task_id': task_id,
                'image': None,
                'error': str(e),
                'success': False
            }
    
    def parallel_rendering(num_tasks: int, max_workers: int = 4):
        """Render multiple tasks in parallel."""
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(render_task, i, 256, 256): i 
                for i in range(num_tasks)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task_id = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"Task {task_id}: {'Success' if result['success'] else 'Failed'}")
                except Exception as e:
                    print(f"Task {task_id} generated exception: {e}")
        
        return results
    
    # Run parallel rendering
    results = parallel_rendering(num_tasks=8, max_workers=4)
    successful_results = [r for r in results if r['success']]
    print(f"Completed {len(successful_results)}/{len(results)} tasks")

**Producer-Consumer Pattern**

Queue-based task distribution:

.. code-block:: python

    import threading
    from queue import Queue, Empty
    
    def producer_consumer_rendering(tasks: list, num_workers: int = 3):
        """Render using producer-consumer pattern."""
        
        task_queue = Queue()
        result_queue = Queue()
        
        # Producer: Add tasks to queue
        for task in tasks:
            task_queue.put(task)
        
        # Add sentinel values to signal workers to stop
        for _ in range(num_workers):
            task_queue.put(None)
        
        def worker():
            """Worker function that processes tasks."""
            while True:
                try:
                    task = task_queue.get(timeout=1.0)
                    if task is None:  # Sentinel value
                        break
                    
                    # Process task
                    result = render_task(task['id'], task['width'], task['height'])
                    result_queue.put(result)
                    task_queue.task_done()
                    
                except Empty:
                    break
                except Exception as e:
                    result_queue.put({
                        'task_id': task.get('id', -1),
                        'error': str(e),
                        'success': False
                    })
        
        # Start workers
        workers = []
        for i in range(num_workers):
            worker_thread = threading.Thread(target=worker, name=f"Worker-{i}")
            worker_thread.start()
            workers.append(worker_thread)
        
        # Collect results
        results = []
        for _ in range(len(tasks)):
            try:
                result = result_queue.get(timeout=10.0)
                results.append(result)
            except Empty:
                print("Timeout waiting for result")
                break
        
        # Wait for workers to complete
        for worker_thread in workers:
            worker_thread.join(timeout=2.0)
        
        return results

Async Compute Operations
------------------------

**Prepass Operations**

Implement async compute prepasses for performance optimization:

.. code-block:: python

    class AsyncComputePrepass:
        """Async compute prepass for depth optimization."""
        
        def __init__(self, width: int, height: int):
            self.width = width
            self.height = height
            self.depth_buffer = np.ones((height, width), dtype=np.float32)
            self.object_ids = np.full((height, width), -1, dtype=np.int32)
        
        def compute_depth_prepass_async(self, scene_objects: list) -> dict:
            """Compute depth prepass asynchronously."""
            
            start_time = time.perf_counter()
            
            # Simulate async depth computation
            # In real implementation, this would use GPU compute shaders
            
            objects_processed = 0
            pixels_with_geometry = 0
            
            for obj in scene_objects:
                # Simulate object depth testing
                obj_depth = np.random.uniform(0.1, 1.0)
                obj_pixels = np.random.randint(100, 1000)
                
                # Update depth buffer (simplified)
                mask = np.random.random((self.height, self.width)) < 0.1
                closer_mask = mask & (obj_depth < self.depth_buffer)
                
                self.depth_buffer[closer_mask] = obj_depth
                self.object_ids[closer_mask] = obj['id']
                
                objects_processed += 1
                pixels_with_geometry += np.sum(closer_mask)
            
            prepass_time = time.perf_counter() - start_time
            
            return {
                'depth_buffer': self.depth_buffer,
                'object_ids': self.object_ids,
                'stats': {
                    'objects_processed': objects_processed,
                    'pixels_with_geometry': pixels_with_geometry,
                    'prepass_time': prepass_time,
                    'depth_complexity': np.mean(self.depth_buffer[self.object_ids >= 0])
                }
            }
        
        def apply_early_z_optimization(self, main_pass_objects: list) -> dict:
            """Apply early-Z optimization using prepass results."""
            
            start_time = time.perf_counter()
            
            # Only process objects that passed depth test
            visible_objects = []
            pixels_shaded = 0
            
            for obj in main_pass_objects:
                obj_mask = self.object_ids == obj['id']
                obj_pixel_count = np.sum(obj_mask)
                
                if obj_pixel_count > 0:
                    visible_objects.append(obj)
                    pixels_shaded += obj_pixel_count
            
            main_pass_time = time.perf_counter() - start_time
            
            return {
                'visible_objects': visible_objects,
                'pixels_shaded': pixels_shaded,
                'main_pass_time': main_pass_time,
                'optimization_ratio': len(visible_objects) / len(main_pass_objects) if main_pass_objects else 0
            }

Non-blocking GPU Operations
---------------------------

**Async Texture Upload**

Upload textures without blocking the main thread:

.. code-block:: python

    class AsyncTextureUploader:
        """Async texture upload with memory management."""
        
        def __init__(self, max_memory_mb: int = 256):
            self.max_memory_bytes = max_memory_mb * 1024 * 1024
            self.used_memory = 0
            self.upload_queue = Queue()
            self.uploaded_textures = {}
        
        def calculate_texture_memory(self, width: int, height: int, channels: int = 4) -> int:
            """Calculate texture memory usage."""
            return width * height * channels
        
        def can_upload(self, width: int, height: int, channels: int = 4) -> bool:
            """Check if texture fits in memory budget."""
            required_memory = self.calculate_texture_memory(width, height, channels)
            return (self.used_memory + required_memory) <= self.max_memory_bytes
        
        def queue_texture_upload(self, texture_id: str, texture_data: np.ndarray) -> bool:
            """Queue texture for async upload."""
            height, width = texture_data.shape[:2]
            channels = texture_data.shape[2] if len(texture_data.shape) > 2 else 1
            
            if not self.can_upload(width, height, channels):
                return False
            
            upload_task = {
                'id': texture_id,
                'data': texture_data,
                'memory_size': self.calculate_texture_memory(width, height, channels)
            }
            
            self.upload_queue.put(upload_task)
            return True
        
        def process_upload_queue(self) -> dict:
            """Process queued texture uploads."""
            processed = 0
            failed = 0
            
            while not self.upload_queue.empty():
                try:
                    task = self.upload_queue.get_nowait()
                    
                    # Simulate texture upload
                    time.sleep(0.001)  # Simulate upload time
                    
                    # Track memory usage
                    self.used_memory += task['memory_size']
                    self.uploaded_textures[task['id']] = {
                        'data': task['data'],
                        'memory_size': task['memory_size']
                    }
                    
                    processed += 1
                    
                except Empty:
                    break
                except Exception as e:
                    failed += 1
                    print(f"Upload failed: {e}")
            
            return {
                'processed': processed,
                'failed': failed,
                'memory_used_mb': self.used_memory / (1024 * 1024),
                'memory_utilization': self.used_memory / self.max_memory_bytes
            }

Performance Monitoring
----------------------

**Async Performance Metrics**

Monitor async operation performance:

.. code-block:: python

    class AsyncPerformanceMonitor:
        """Monitor performance of async operations."""
        
        def __init__(self):
            self.metrics = {}
            self.start_times = {}
        
        def start_operation(self, operation_name: str):
            """Start timing an async operation."""
            self.start_times[operation_name] = time.perf_counter()
        
        def end_operation(self, operation_name: str, **metadata):
            """End timing and record metrics."""
            if operation_name in self.start_times:
                duration = time.perf_counter() - self.start_times[operation_name]
                
                if operation_name not in self.metrics:
                    self.metrics[operation_name] = {
                        'count': 0,
                        'total_time': 0.0,
                        'min_time': float('inf'),
                        'max_time': 0.0,
                        'metadata': []
                    }
                
                metric = self.metrics[operation_name]
                metric['count'] += 1
                metric['total_time'] += duration
                metric['min_time'] = min(metric['min_time'], duration)
                metric['max_time'] = max(metric['max_time'], duration)
                metric['metadata'].append(metadata)
                
                del self.start_times[operation_name]
        
        def get_summary(self) -> dict:
            """Get performance summary."""
            summary = {}
            
            for op_name, metric in self.metrics.items():
                if metric['count'] > 0:
                    summary[op_name] = {
                        'count': metric['count'],
                        'avg_time_ms': (metric['total_time'] / metric['count']) * 1000,
                        'min_time_ms': metric['min_time'] * 1000,
                        'max_time_ms': metric['max_time'] * 1000,
                        'total_time_ms': metric['total_time'] * 1000
                    }
            
            return summary
    
    # Usage example
    monitor = AsyncPerformanceMonitor()
    
    monitor.start_operation('async_render')
    # ... perform async rendering ...
    monitor.end_operation('async_render', resolution=(800, 600), objects=25)
    
    summary = monitor.get_summary()
    print(f"Async render average: {summary['async_render']['avg_time_ms']:.1f}ms")

**Threading Performance Analysis**

Analyze threading efficiency:

.. code-block:: python

    def analyze_threading_performance(results: list) -> dict:
        """Analyze performance across different threading approaches."""
        
        analysis = {}
        
        # Find sequential baseline
        sequential = next((r for r in results if 'sequential' in r.get('mode', '')), None)
        baseline_time = sequential['total_time'] if sequential else None
        
        for result in results:
            mode = result.get('mode', 'unknown')
            total_time = result.get('total_time', 0)
            worker_count = result.get('max_workers', result.get('num_workers', 1))
            
            # Calculate performance metrics
            speedup = baseline_time / total_time if baseline_time and total_time > 0 else 0
            efficiency = speedup / worker_count if worker_count > 0 else 0
            
            # Thread utilization
            unique_threads = len(set(t.get('thread_id') for t in result.get('tasks', []) if t.get('thread_id')))
            
            analysis[mode] = {
                'total_time_ms': total_time * 1000,
                'speedup': speedup,
                'efficiency_percent': efficiency * 100,
                'threads_used': unique_threads,
                'theoretical_max_speedup': worker_count,
                'utilization_percent': (unique_threads / worker_count * 100) if worker_count > 0 else 0
            }
        
        return analysis

Memory Management for Async Operations
--------------------------------------

**Resource Cleanup**

Ensure proper cleanup of async resources:

.. code-block:: python

    class AsyncResourceManager:
        """Manage resources for async operations."""
        
        def __init__(self, memory_budget_mb: int = 256):
            self.memory_budget = memory_budget_mb * 1024 * 1024
            self.allocated_resources = {}
            self.current_usage = 0
        
        def allocate_resource(self, resource_id: str, size_bytes: int) -> bool:
            """Allocate resource if within budget."""
            if self.current_usage + size_bytes > self.memory_budget:
                return False
            
            self.allocated_resources[resource_id] = size_bytes
            self.current_usage += size_bytes
            return True
        
        def deallocate_resource(self, resource_id: str) -> bool:
            """Deallocate resource and free memory."""
            if resource_id in self.allocated_resources:
                size_bytes = self.allocated_resources[resource_id]
                del self.allocated_resources[resource_id]
                self.current_usage -= size_bytes
                return True
            return False
        
        def cleanup_all(self):
            """Clean up all allocated resources."""
            self.allocated_resources.clear()
            self.current_usage = 0
        
        def get_memory_stats(self) -> dict:
            """Get current memory usage statistics."""
            return {
                'used_bytes': self.current_usage,
                'used_mb': self.current_usage / (1024 * 1024),
                'budget_mb': self.memory_budget / (1024 * 1024),
                'utilization_percent': (self.current_usage / self.memory_budget) * 100,
                'active_resources': len(self.allocated_resources)
            }
    
    # Usage with context manager
    class AsyncOperation:
        def __init__(self, resource_manager: AsyncResourceManager):
            self.resource_manager = resource_manager
            self.allocated_resources = []
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            # Cleanup resources on exit
            for resource_id in self.allocated_resources:
                self.resource_manager.deallocate_resource(resource_id)
            self.allocated_resources.clear()
        
        def allocate(self, resource_id: str, size_bytes: int) -> bool:
            """Allocate resource and track it."""
            if self.resource_manager.allocate_resource(resource_id, size_bytes):
                self.allocated_resources.append(resource_id)
                return True
            return False

Best Practices
--------------

**Thread Safety**

Ensure thread-safe operations:

.. code-block:: python

    import threading
    
    # Use thread-local storage for per-thread resources
    thread_local_data = threading.local()
    
    def get_thread_renderer():
        """Get renderer for current thread."""
        if not hasattr(thread_local_data, 'renderer'):
            thread_local_data.renderer = f3d.Renderer(256, 256)
        return thread_local_data.renderer
    
    # Use locks for shared resources
    resource_lock = threading.Lock()
    shared_counter = 0
    
    def thread_safe_increment():
        global shared_counter
        with resource_lock:
            shared_counter += 1

**Error Handling**

Robust error handling for async operations:

.. code-block:: python

    def safe_async_operation(task_func, *args, **kwargs):
        """Wrapper for safe async operation execution."""
        try:
            result = task_func(*args, **kwargs)
            return {'success': True, 'result': result}
        except MemoryError as e:
            return {'success': False, 'error': 'memory_error', 'message': str(e)}
        except Exception as e:
            return {'success': False, 'error': 'general_error', 'message': str(e)}

**Memory Budget Compliance**

Always respect the 512 MiB memory budget:

.. code-block:: python

    def check_memory_before_async_operation(operation_size_mb: float) -> bool:
        """Check if async operation fits within memory budget."""
        MAX_MEMORY_MB = 512
        
        # Estimate current usage (simplified)
        import psutil
        process = psutil.Process()
        current_mb = process.memory_info().rss / (1024 * 1024)
        
        if current_mb + operation_size_mb > MAX_MEMORY_MB:
            print(f"⚠ Operation would exceed memory budget: {current_mb + operation_size_mb:.1f} MB")
            return False
        
        return True

Example Applications
--------------------

**Batch Rendering Pipeline**

.. code-block:: python

    def create_batch_rendering_pipeline(tasks: list, max_workers: int = 4):
        """Create async batch rendering pipeline."""
        
        # Setup resource management
        resource_manager = AsyncResourceManager(memory_budget_mb=256)
        monitor = AsyncPerformanceMonitor()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = []
            for task in tasks:
                if check_memory_before_async_operation(task.get('memory_mb', 10)):
                    future = executor.submit(render_task, task['id'], task['width'], task['height'])
                    futures.append((future, task))
            
            # Collect results with monitoring
            results = []
            for future, task in futures:
                monitor.start_operation(f"task_{task['id']}")
                try:
                    result = future.result(timeout=30.0)
                    monitor.end_operation(f"task_{task['id']}", **task)
                    results.append(result)
                except Exception as e:
                    print(f"Task {task['id']} failed: {e}")
        
        # Cleanup and report
        resource_manager.cleanup_all()
        performance_summary = monitor.get_summary()
        
        return results, performance_summary

Troubleshooting
---------------

**Common Async Issues**

1. **Memory Leaks in Async Operations**
   - Always use context managers or explicit cleanup
   - Monitor memory usage regularly
   - Implement resource tracking

2. **Threading Deadlocks**
   - Use timeout parameters on blocking operations
   - Avoid nested locking
   - Use thread-safe data structures

3. **Performance Degradation**
   - Don't over-parallelize (optimal worker count ≈ CPU cores)
   - Consider I/O bound vs CPU bound operations
   - Monitor thread utilization

**Memory Budget Violations**

.. code-block:: python

    # Always check memory before large async operations
    def safe_large_operation(data_size_mb: float):
        if not check_memory_before_async_operation(data_size_mb):
            # Use streaming or tiling approach
            return process_in_chunks(data_size_mb)
        else:
            return process_all_at_once()

See the comprehensive async examples:

- ``examples/multithreaded_command_recording.py`` - Threading patterns and performance
- ``examples/async_compute_prepass.py`` - GPU compute optimization
- ``examples/large_texture_upload_policies.py`` - Memory management strategies