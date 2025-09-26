"""
# examples/polygon_extrusion.py
"""
import numpy as np
from pathlib import Path
import time
import forge3d as f3d

def main():
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

    # --- CPU Extrusion ---
    start_time = time.time()
    vertices_cpu, indices_cpu, normals_cpu, uvs_cpu = f3d.extrude_polygon_py(polygon, height=0.5)
    cpu_time = time.time() - start_time
    print(f"CPU extrusion time: {cpu_time:.6f}s")

    # --- GPU Extrusion ---
    start_time = time.time()
    vertices_gpu, indices_gpu, normals_gpu, uvs_gpu = f3d.extrude_polygon_gpu_py([polygon], height=0.5)
    gpu_time = time.time() - start_time
    print(f"GPU extrusion time: {gpu_time:.6f}s")

    # Create a scene
    scene = f3d.Scene()
    scene.add_mesh(
        vertices=vertices_gpu,
        indices=indices_gpu,
        normals=normals_gpu,
        uvs=uvs_gpu,
        color=[0.8, 0.2, 0.2, 1.0],
    )

    # Set up camera
    scene.set_camera(
        position=(3, 3, 3),
        target=(0, 0, 0),
        up=(0, 1, 0),
    )
    
    # Set up lighting
    scene.set_light(
        direction=(1, 1, 1),
        intensity=1.0,
    )

    # Render the scene
    output_dir = Path("out")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "polygon_extrusion.png"
    scene.render_png(output_path)
    print(f"Saved image to {output_path}")

    # --- Benchmark ---
    print("\n--- BENCHMARK ---")
    num_polygons = 10000
    polygons = [polygon for _ in range(num_polygons)]

    # CPU Benchmark
    start_time = time.time()
    for p in polygons:
        f3d.extrude_polygon_py(p, height=0.5)
    cpu_benchmark_time = time.time() - start_time
    print(f"CPU benchmark ({num_polygons} polygons): {cpu_benchmark_time:.6f}s")

    # GPU Benchmark
    start_time = time.time()
    f3d.extrude_polygon_gpu_py(polygons, height=0.5)
    gpu_benchmark_time = time.time() - start_time
    print(f"GPU benchmark ({num_polygons} polygons): {gpu_benchmark_time:.6f}s")

    if gpu_benchmark_time < cpu_benchmark_time:
        speedup = cpu_benchmark_time / gpu_benchmark_time
        print(f"GPU is {speedup:.2f}x faster")
    else:
        print("CPU is faster or equal to GPU")

if __name__ == "__main__":
    main()
