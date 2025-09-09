#!/usr/bin/env python3
"""
Advanced Example 8: Multi-threaded Command Recording

Demonstrates multi-threaded command buffer recording and parallel GPU workload submission.
Shows thread pool utilization for rendering pipeline parallelization and performance optimization.
"""

import numpy as np
import sys
import os
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty

sys.path.insert(0, str(Path(__file__).parent.parent))

class RenderTask:
    """Represents a single rendering task that can be executed in parallel."""
    
    def __init__(self, task_id: int, task_type: str, width: int, height: int, params: dict):
        self.task_id = task_id
        self.task_type = task_type
        self.width = width
        self.height = height
        self.params = params
        self.result = None
        self.execution_time = 0.0
        self.thread_id = None


def create_terrain_task(task_id: int, size: int = 128) -> RenderTask:
    """Create a terrain rendering task."""
    # Generate unique terrain for this task
    np.random.seed(task_id + 1000)  # Unique but reproducible
    
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)
    
    # Procedural terrain with task-specific variation
    terrain = (
        0.5 * np.sin(X * 2 + task_id * 0.1) * np.cos(Y * 1.5 + task_id * 0.2) +
        0.3 * np.sin(X * 4 + task_id * 0.3) * np.cos(Y * 3 + task_id * 0.4) +
        0.2 * np.random.random((size, size))
    )
    
    # Normalize to [0, 1]
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    
    return RenderTask(
        task_id=task_id,
        task_type="terrain",
        width=256,
        height=256,
        params={
            'terrain_data': terrain.astype(np.float32),
            'spacing': 2.0,
            'exaggeration': 5.0,
        }
    )


def create_triangle_task(task_id: int) -> RenderTask:
    """Create a triangle rendering task."""
    return RenderTask(
        task_id=task_id,
        task_type="triangle",
        width=128,
        height=128,
        params={}
    )


def create_vector_task(task_id: int) -> RenderTask:
    """Create a vector graphics rendering task."""
    np.random.seed(task_id + 2000)
    
    # Generate random vector graphics elements
    n_points = 20 + (task_id % 30)
    points = np.random.rand(n_points, 2) * 200
    colors = np.random.rand(n_points, 4)
    colors[:, 3] = 0.7  # Set alpha
    sizes = np.random.rand(n_points) * 10 + 2
    
    return RenderTask(
        task_id=task_id,
        task_type="vector",
        width=200,
        height=200,
        params={
            'points': points.astype(np.float32),
            'colors': colors.astype(np.float32),
            'sizes': sizes.astype(np.float32),
        }
    )


def execute_render_task(task: RenderTask) -> RenderTask:
    """Execute a single render task in a worker thread."""
    start_time = time.perf_counter()
    task.thread_id = threading.get_ident()
    
    try:
        import forge3d as f3d
        
        if task.task_type == "terrain":
            # Create scene and render terrain
            scene = f3d.Scene(task.width, task.height)
            scene.set_height_data(
                task.params['terrain_data'],
                spacing=task.params['spacing'],
                exaggeration=task.params['exaggeration']
            )
            
            # Set camera for good view
            terrain_size = task.params['terrain_data'].shape[0]
            scene.set_camera(
                position=(terrain_size * 1.2, terrain_size * 0.5, terrain_size * 1.2),
                target=(terrain_size * 0.5, 0.0, terrain_size * 0.5),
                up=(0.0, 1.0, 0.0)
            )
            
            task.result = scene.render_terrain_rgba()
            
        elif task.task_type == "triangle":
            # Simple triangle rendering
            renderer = f3d.Renderer(task.width, task.height)
            task.result = renderer.render_triangle_rgba()
            
        elif task.task_type == "vector":
            # Vector graphics rendering
            f3d.clear_vectors_py()
            
            # Add points
            points_array = task.params['points'].reshape(1, -1, 2)
            f3d.add_points_py(
                points_array,
                colors=task.params['colors'].reshape(-1, 4),
                sizes=task.params['sizes']
            )
            
            # Render with vector overlay
            renderer = f3d.Renderer(task.width, task.height)
            task.result = renderer.render_triangle_rgba()
            
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")
            
    except Exception as e:
        print(f"Task {task.task_id} ({task.task_type}) failed: {e}")
        # Create fallback result
        task.result = np.zeros((task.height, task.width, 4), dtype=np.uint8)
    
    end_time = time.perf_counter()
    task.execution_time = end_time - start_time
    
    return task


def run_sequential_baseline(tasks: list) -> dict:
    """Run all tasks sequentially to establish baseline performance."""
    print("Running sequential baseline...")
    
    start_time = time.perf_counter()
    results = []
    
    for task in tasks:
        result_task = execute_render_task(task)
        results.append(result_task)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    return {
        'tasks': results,
        'total_time': total_time,
        'task_times': [t.execution_time for t in results],
        'avg_task_time': np.mean([t.execution_time for t in results]),
        'execution_mode': 'sequential'
    }


def run_multithreaded_parallel(tasks: list, max_workers: int = 4) -> dict:
    """Run tasks in parallel using ThreadPoolExecutor."""
    print(f"Running parallel with {max_workers} threads...")
    
    start_time = time.perf_counter()
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(execute_render_task, task): task for task in tasks}
        
        # Collect results as they complete
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result_task = future.result()
                results.append(result_task)
            except Exception as e:
                print(f"Task {task.task_id} generated an exception: {e}")
                task.result = np.zeros((task.height, task.width, 4), dtype=np.uint8)
                results.append(task)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Sort results by task_id to maintain order
    results.sort(key=lambda x: x.task_id)
    
    return {
        'tasks': results,
        'total_time': total_time,
        'task_times': [t.execution_time for t in results],
        'avg_task_time': np.mean([t.execution_time for t in results]),
        'execution_mode': f'parallel_{max_workers}_threads',
        'max_workers': max_workers
    }


def run_producer_consumer_pattern(tasks: list, num_workers: int = 3) -> dict:
    """Run tasks using producer-consumer pattern with work queue."""
    print(f"Running producer-consumer with {num_workers} workers...")
    
    task_queue = Queue()
    result_queue = Queue()
    
    # Producer: Add all tasks to queue
    for task in tasks:
        task_queue.put(task)
    
    # Add sentinel values to signal workers to stop
    for _ in range(num_workers):
        task_queue.put(None)
    
    def worker():
        """Worker function that processes tasks from queue."""
        while True:
            try:
                task = task_queue.get(timeout=1.0)
                if task is None:  # Sentinel value
                    break
                
                result_task = execute_render_task(task)
                result_queue.put(result_task)
                task_queue.task_done()
                
            except Empty:
                break
            except Exception as e:
                print(f"Worker exception: {e}")
                break
    
    # Start workers
    start_time = time.perf_counter()
    workers = []
    for i in range(num_workers):
        worker_thread = threading.Thread(target=worker, name=f"Worker-{i}")
        worker_thread.start()
        workers.append(worker_thread)
    
    # Collect results
    results = []
    for _ in range(len(tasks)):
        try:
            result_task = result_queue.get(timeout=30.0)  # Generous timeout
            results.append(result_task)
        except Empty:
            print("Timeout waiting for result")
            break
    
    # Wait for all workers to complete
    for worker_thread in workers:
        worker_thread.join(timeout=5.0)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Sort results by task_id
    results.sort(key=lambda x: x.task_id)
    
    return {
        'tasks': results,
        'total_time': total_time,
        'task_times': [t.execution_time for t in results],
        'avg_task_time': np.mean([t.execution_time for t in results]) if results else 0,
        'execution_mode': f'producer_consumer_{num_workers}_workers',
        'num_workers': num_workers
    }


def analyze_threading_performance(results: list) -> dict:
    """Analyze performance across different threading approaches."""
    
    analysis = {}
    
    # Find sequential baseline
    sequential = next((r for r in results if r['execution_mode'] == 'sequential'), None)
    if not sequential:
        return {"error": "No sequential baseline found"}
    
    baseline_time = sequential['total_time']
    
    for result in results:
        mode = result['execution_mode']
        total_time = result['total_time']
        
        # Calculate performance metrics
        speedup = baseline_time / total_time if total_time > 0 else 0
        efficiency = speedup / result.get('max_workers', result.get('num_workers', 1))
        
        # Thread utilization analysis
        thread_ids = set()
        for task in result['tasks']:
            if hasattr(task, 'thread_id') and task.thread_id:
                thread_ids.add(task.thread_id)
        
        analysis[mode] = {
            'total_time': total_time,
            'speedup': speedup,
            'efficiency': efficiency,
            'unique_threads_used': len(thread_ids),
            'avg_task_time': result['avg_task_time'],
            'task_time_std': np.std(result['task_times']) if result['task_times'] else 0,
        }
    
    return analysis


def create_performance_visualization(results: list, out_dir: Path):
    """Create visualizations of threading performance."""
    try:
        import forge3d as f3d
        
        # Create a simple bar chart visualization using colored rectangles
        modes = [r['execution_mode'] for r in results]
        times = [r['total_time'] for r in results]
        
        if not times:
            return
        
        # Normalize times for visualization
        max_time = max(times)
        bar_height = 30
        bar_spacing = 40
        chart_width = 400
        chart_height = len(modes) * bar_spacing + 50
        
        # Create chart image
        chart = np.ones((chart_height, chart_width, 4), dtype=np.uint8) * 255
        chart[:, :, 3] = 255  # Alpha
        
        colors = [
            (255, 100, 100),  # Red
            (100, 255, 100),  # Green  
            (100, 100, 255),  # Blue
            (255, 255, 100),  # Yellow
            (255, 100, 255),  # Magenta
        ]
        
        for i, (mode, time_val) in enumerate(zip(modes, times)):
            y_pos = i * bar_spacing + 20
            bar_width = int((time_val / max_time) * (chart_width - 100))
            color = colors[i % len(colors)]
            
            # Draw bar
            chart[y_pos:y_pos+bar_height, 20:20+bar_width, :3] = color
        
        # Save chart
        chart_path = out_dir / "threading_performance_chart.png"
        f3d.numpy_to_png(str(chart_path), chart)
        print(f"Saved performance chart: {chart_path}")
        
        return str(chart_path)
        
    except Exception as e:
        print(f"Performance visualization failed: {e}")
        return None


def main():
    """Main example execution."""
    print("Multi-threaded Command Recording")
    print("================================")
    
    out_dir = Path(__file__).parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    
    try:
        import forge3d as f3d
        
        # Create diverse set of rendering tasks
        print("Creating rendering tasks...")
        tasks = []
        
        # Add terrain tasks
        for i in range(5):
            tasks.append(create_terrain_task(i))
        
        # Add triangle tasks  
        for i in range(3):
            tasks.append(create_triangle_task(i + 100))
        
        # Add vector tasks
        for i in range(4):
            tasks.append(create_vector_task(i + 200))
        
        print(f"Created {len(tasks)} tasks:")
        task_counts = {}
        for task in tasks:
            task_counts[task.task_type] = task_counts.get(task.task_type, 0) + 1
        
        for task_type, count in task_counts.items():
            print(f"  {task_type}: {count} tasks")
        
        # Run different threading approaches
        all_results = []
        
        # 1. Sequential baseline
        sequential_result = run_sequential_baseline(tasks.copy())
        all_results.append(sequential_result)
        
        # 2. ThreadPoolExecutor with 2 threads
        parallel_2_result = run_multithreaded_parallel(tasks.copy(), max_workers=2)
        all_results.append(parallel_2_result)
        
        # 3. ThreadPoolExecutor with 4 threads
        parallel_4_result = run_multithreaded_parallel(tasks.copy(), max_workers=4)
        all_results.append(parallel_4_result)
        
        # 4. Producer-consumer pattern
        producer_consumer_result = run_producer_consumer_pattern(tasks.copy(), num_workers=3)
        all_results.append(producer_consumer_result)
        
        # Analyze performance
        print("\nAnalyzing threading performance...")
        performance_analysis = analyze_threading_performance(all_results)
        
        # Save sample outputs from each approach
        saved_paths = {}
        for i, result in enumerate(all_results):
            if result['tasks']:
                # Save first few task results from each approach
                for j, task in enumerate(result['tasks'][:3]):  # First 3 tasks
                    if task.result is not None:
                        filename = f"threading_{result['execution_mode']}_task_{task.task_id}_{task.task_type}.png"
                        path = out_dir / filename
                        f3d.numpy_to_png(str(path), task.result)
                        
                        if result['execution_mode'] not in saved_paths:
                            saved_paths[result['execution_mode']] = []
                        saved_paths[result['execution_mode']].append(str(path))
        
        # Create performance visualization
        chart_path = create_performance_visualization(all_results, out_dir)
        if chart_path:
            saved_paths['performance_chart'] = chart_path
        
        # Generate comprehensive metrics
        metrics = {
            'task_configuration': {
                'total_tasks': len(tasks),
                'task_types': task_counts,
                'threading_approaches': len(all_results),
            },
            'performance_results': {},
            'threading_analysis': performance_analysis,
            'system_info': {
                'python_threading': True,
                'max_workers_tested': 4,
                'patterns_tested': ['sequential', 'threadpool', 'producer_consumer'],
            },
            'outputs': saved_paths,
        }
        
        # Add detailed results for each approach
        for result in all_results:
            mode = result['execution_mode']
            metrics['performance_results'][mode] = {
                'total_time': result['total_time'],
                'avg_task_time': result['avg_task_time'],
                'min_task_time': min(result['task_times']) if result['task_times'] else 0,
                'max_task_time': max(result['task_times']) if result['task_times'] else 0,
                'tasks_completed': len([t for t in result['tasks'] if t.result is not None]),
            }
        
        # Print performance summary
        print("\nThreading Performance Results:")
        print(f"{'Mode':<25} {'Time(s)':<10} {'Speedup':<10} {'Efficiency':<12} {'Threads':<8}")
        print("-" * 70)
        
        for mode, analysis in performance_analysis.items():
            print(f"{mode:<25} {analysis['total_time']:<10.2f} {analysis['speedup']:<10.1f}x "
                  f"{analysis['efficiency']:<12.1f}% {analysis['unique_threads_used']:<8}")
        
        # Save metrics
        import json
        metrics_path = out_dir / "threading_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics: {metrics_path}")
        
        print("\nExample completed successfully!")
        return 0
        
    except ImportError as e:
        print(f"forge3d not available: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())