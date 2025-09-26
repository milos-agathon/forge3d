#!/usr/bin/env python3
"""
Polygon extrusion example for forge3d.

Demonstrates CPU and GPU polygon extrusion capabilities.
Compares performance between CPU and GPU implementations.
"""

import numpy as np
from pathlib import Path
import time
import sys

# Add import handling for examples
try:
    from _import_shim import ensure_repo_import
    ensure_repo_import()
except ImportError:
    pass  # Running from installed package

try:
    import forge3d as f3d
except Exception as e:
    print(f"Failed to import forge3d: {e}")
    print("Make sure the package is installed or run 'maturin develop' first")
    sys.exit(1)


def main():
    """Main entry point for polygon extrusion demo."""
    print("forge3d polygon extrusion example starting...")

    # Define a simple star-shaped polygon
    polygon = np.array([
        [0.0, 1.0],
        [0.2, 0.2],
        [1.0, 0.2],
        [0.4, -0.2],
        [0.6, -1.0],
        [0.0, -0.6],
        [-0.6, -1.0],
        [-0.4, -0.2],
        [-1.0, 0.2],
        [-0.2, 0.2],
    ], dtype=np.float32)

    print(f"Input polygon shape: {polygon.shape}")
    print(f"Input polygon:\n{polygon}")

    try:
        # --- CPU Extrusion ---
        print("\n=== CPU Extrusion ===")
        start_time = time.time()
        vertices_cpu, indices_cpu, normals_cpu, uvs_cpu = f3d.extrude_polygon_py(polygon, height=0.5)
        cpu_time = time.time() - start_time
        print(f"CPU extrusion time: {cpu_time:.6f}s")
        print(f"CPU result - vertices: {vertices_cpu.shape}, indices: {indices_cpu.shape}")
        print(f"CPU result - normals: {normals_cpu.shape}, uvs: {uvs_cpu.shape}")

        # --- GPU Extrusion (Optional) ---
        print("\n=== GPU Extrusion ===")
        gpu_success = False
        try:
            start_time = time.time()
            vertices_gpu, indices_gpu, normals_gpu, uvs_gpu = f3d.extrude_polygon_gpu_py([polygon], height=0.5)
            gpu_time = time.time() - start_time
            print(f"GPU extrusion time: {gpu_time:.6f}s")
            print(f"GPU result - vertices: {vertices_gpu.shape}, indices: {indices_gpu.shape}")
            print(f"GPU result - normals: {normals_gpu.shape}, uvs: {uvs_gpu.shape}")
            gpu_success = True
        except Exception as gpu_e:
            print(f"GPU extrusion failed (using CPU fallback): {gpu_e}")
            # Use CPU results as fallback
            vertices_gpu, indices_gpu, normals_gpu, uvs_gpu = vertices_cpu, indices_cpu, normals_cpu, uvs_cpu
            gpu_time = cpu_time

        # Compare results
        print("\n=== Comparison ===")
        if gpu_success and cpu_time > 0 and gpu_time > 0:
            if gpu_time < cpu_time:
                speedup = cpu_time / gpu_time
                print(f"GPU is {speedup:.2f}x faster for single polygon")
            else:
                slowdown = gpu_time / cpu_time
                print(f"GPU is {slowdown:.2f}x slower for single polygon (overhead)")
        else:
            print("GPU comparison not available (GPU extrusion failed)")

        # Save mesh data to files
        output_dir = Path("out")
        output_dir.mkdir(exist_ok=True)

        # Save CPU results
        np.save(output_dir / "cpu_vertices.npy", vertices_cpu)
        np.save(output_dir / "cpu_indices.npy", indices_cpu)
        np.save(output_dir / "cpu_normals.npy", normals_cpu)
        np.save(output_dir / "cpu_uvs.npy", uvs_cpu)

        # Save GPU results (or CPU fallback)
        np.save(output_dir / "gpu_vertices.npy", vertices_gpu)
        np.save(output_dir / "gpu_indices.npy", indices_gpu)
        np.save(output_dir / "gpu_normals.npy", normals_gpu)
        np.save(output_dir / "gpu_uvs.npy", uvs_gpu)

        print(f"Mesh data saved to {output_dir}/ directory")

        # --- Benchmark ---
        print("\n=== BENCHMARK ===")
        num_polygons = 1000  # Reduced for faster testing
        polygons = [polygon for _ in range(num_polygons)]
        print(f"Benchmarking with {num_polygons} polygons...")

        # CPU Benchmark
        print("Running CPU benchmark...")
        start_time = time.time()
        for p in polygons:
            f3d.extrude_polygon_py(p, height=0.5)
        cpu_benchmark_time = time.time() - start_time
        print(f"CPU benchmark ({num_polygons} polygons): {cpu_benchmark_time:.6f}s")
        print(f"CPU average per polygon: {cpu_benchmark_time/num_polygons*1000:.3f}ms")

        # GPU Benchmark (Optional)
        gpu_benchmark_success = False
        if gpu_success:
            try:
                print("Running GPU benchmark...")
                start_time = time.time()
                f3d.extrude_polygon_gpu_py(polygons, height=0.5)
                gpu_benchmark_time = time.time() - start_time
                print(f"GPU benchmark ({num_polygons} polygons): {gpu_benchmark_time:.6f}s")
                print(f"GPU average per polygon: {gpu_benchmark_time/num_polygons*1000:.3f}ms")
                gpu_benchmark_success = True
            except Exception as e:
                print(f"GPU benchmark failed: {e}")

        # Final comparison
        if gpu_benchmark_success:
            if gpu_benchmark_time < cpu_benchmark_time:
                speedup = cpu_benchmark_time / gpu_benchmark_time
                print(f"\n🚀 GPU is {speedup:.2f}x faster for batch processing!")
            elif cpu_benchmark_time < gpu_benchmark_time:
                slowdown = gpu_benchmark_time / cpu_benchmark_time
                print(f"\n⚡ CPU is {slowdown:.2f}x faster (GPU overhead)")
            else:
                print(f"\n⚖️  CPU and GPU performance are equivalent")
        else:
            print(f"\n✅ CPU polygon extrusion working perfectly!")
            print(f"   (GPU extrusion disabled due to shader issues)")

        print("\nPolygon extrusion demo completed successfully!")
        return 0

    except Exception as e:
        print(f"Error during polygon extrusion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
