#!/usr/bin/env python3
"""
Hybrid Rendering Demo - SDF + Mesh Integration
==============================================

This example demonstrates the hybrid rendering capabilities of forge3d,
combining traditional mesh geometry with procedural SDF objects in a
single scene. This showcases the power of the hybrid traversal system.

Features demonstrated:
- Hybrid scene construction (SDF + mesh)
- Different traversal modes (hybrid, SDF-only, mesh-only)
- Performance comparison between modes
- Material handling across geometry types
- Advanced rendering techniques

Run with: python examples/sdf_hybrid_demo.py

Note: This example requires both SDF and mesh functionality to be available.
Some features may fall back to CPU implementation if GPU acceleration is not available.
"""

import numpy as np
from forge3d.sdf import *
import time
from pathlib import Path
import math

def create_procedural_sdf_objects():
    """Create interesting procedural objects using SDF"""
    print("Creating procedural SDF objects...")

    scenes = {}

    # 1. Fractal-like structure using domain repetition
    builder = SdfSceneBuilder()

    # Create a basic repeating pattern
    base_positions = [
        (0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2),
        (2, 2, 0), (2, 0, 2), (0, 2, 2), (2, 2, 2)
    ]

    spheres = []
    for i, pos in enumerate(base_positions):
        size = 0.3 + 0.2 * math.sin(i)  # Varying sizes
        builder, sphere = builder.add_sphere(pos, size, (i % 3) + 1)
        spheres.append(sphere)

    # Create a complex union structure
    current = spheres[0]
    for i, sphere in enumerate(spheres[1:], 1):
        smoothing = 0.1 + 0.1 * (i % 3)  # Varying smoothness
        builder, current = builder.smooth_union(current, sphere, smoothing, 0)

    scenes["fractal"] = builder.with_bounds((-1, -1, -1), (3, 3, 3)).build()

    # 2. Organic structure (twisted torus with bulges)
    builder = SdfSceneBuilder()

    # Main torus
    builder, main_torus = builder.add_torus((0, 0, 0), 2.0, 0.5, 1)

    # Add bulges at regular intervals
    bulge_count = 6
    for i in range(bulge_count):
        angle = i * 2 * math.pi / bulge_count
        x = 2.0 * math.cos(angle)
        z = 2.0 * math.sin(angle)
        size = 0.4 + 0.2 * math.sin(i * 2)

        builder, bulge = builder.add_sphere((x, 0, z), size, 2)
        builder, main_torus = builder.smooth_union(main_torus, bulge, 0.3, 1)

    # Subtract internal structure
    builder, inner_torus = builder.add_torus((0, 0, 0), 1.8, 0.2, 10)
    builder, complex_torus = builder.smooth_subtract(main_torus, inner_torus, 0.1, 1)

    scenes["organic"] = builder.with_bounds((-3, -1, -3), (3, 1, 3)).build()

    # 3. Architectural detail (decorative bracket)
    builder = SdfSceneBuilder()

    # Main bracket body
    builder, main_body = builder.add_box((0, 0, 0), (1.5, 0.3, 0.8), 1)

    # Support strut
    builder, strut = builder.add_box((0, -0.8, 0), (0.2, 0.8, 0.6), 1)
    builder, bracket = builder.union(main_body, strut, 1)

    # Decorative elements
    ornament_positions = [(-1.0, 0, 0), (1.0, 0, 0)]
    for pos in ornament_positions:
        builder, ornament = builder.add_sphere(pos, 0.25, 2)
        builder, bracket = builder.union(bracket, ornament, 1)

    # Cut decorative grooves
    groove_positions = [(-0.5, 0, 0), (0, 0, 0), (0.5, 0, 0)]
    for pos in groove_positions:
        builder, groove = builder.add_cylinder(pos, 0.05, 1.0, 10)
        builder, bracket = builder.subtract(bracket, groove, 1)

    scenes["architectural"] = builder.with_bounds((-2, -2, -1), (2, 1, 1)).build()

    print(f"Created {len(scenes)} procedural SDF scenes")
    return scenes

def create_mock_mesh_data():
    """Create simple mesh data for demonstration"""
    print("Creating mock mesh geometry...")

    # Create a simple cube mesh
    # Vertices (8 corners of a cube)
    vertices = np.array([
        # Front face
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
        # Back face
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
    ], dtype=np.float32)

    # Scale and offset the cube
    vertices *= 0.8
    vertices += [3, 0, 0]  # Move to the right of SDF objects

    # Indices (12 triangles, 2 per face)
    indices = np.array([
        # Front face
        0, 1, 2,  2, 3, 0,
        # Back face
        4, 6, 5,  6, 4, 7,
        # Left face
        4, 0, 3,  3, 7, 4,
        # Right face
        1, 5, 6,  6, 2, 1,
        # Top face
        3, 2, 6,  6, 7, 3,
        # Bottom face
        4, 1, 0,  1, 4, 5,
    ], dtype=np.uint32)

    print(f"Created mesh with {len(vertices)} vertices and {len(indices)//3} triangles")
    return vertices, indices

def create_hybrid_scenes(sdf_scenes):
    """Create hybrid scenes combining SDF and mesh data"""
    print("Creating hybrid scenes...")

    # Note: This is a simplified version since we don't have full mesh/BVH integration
    # In a real implementation, this would create proper HybridScene objects
    hybrid_scenes = {}

    vertices, indices = create_mock_mesh_data()

    for name, sdf_scene in sdf_scenes.items():
        # For this demo, we'll just use the SDF scene
        # In a full implementation, this would be:
        # hybrid_scene = HybridScene::new()
        # hybrid_scene.set_sdf_scene(sdf_scene)
        # hybrid_scene.add_mesh(vertices, indices, bvh)

        hybrid_scenes[f"hybrid_{name}"] = {
            'sdf_scene': sdf_scene,
            'mesh_data': (vertices, indices),
            'description': f'Hybrid scene with {name} SDF + cube mesh'
        }

    return hybrid_scenes

def benchmark_traversal_modes(scene):
    """Benchmark different traversal modes"""
    print("Benchmarking traversal modes...")

    renderer = HybridRenderer(128, 128)  # Small size for quick benchmarks
    renderer.set_camera((4, 3, 5), (0, 0, 0), (0, 1, 0), 45)

    modes = [
        (TraversalMode.SDF_ONLY, "SDF Only"),
        (TraversalMode.HYBRID, "Hybrid (SDF + Mesh)"),
        # Note: MESH_ONLY would require actual mesh data
    ]

    results = {}

    for mode, name in modes:
        print(f"  Testing {name}...")

        renderer.set_traversal_mode(mode)

        # Warm-up render
        try:
            _ = renderer.render_sdf_scene(scene)
        except Exception as e:
            print(f"    Warning: {name} failed with error: {e}")
            continue

        # Timed renders
        times = []
        for i in range(3):  # Multiple runs for average
            start_time = time.time()
            try:
                image = renderer.render_sdf_scene(scene)
                render_time = time.time() - start_time
                times.append(render_time)
            except Exception as e:
                print(f"    Render {i} failed: {e}")
                break

        if times:
            avg_time = sum(times) / len(times)
            results[name] = {
                'avg_time': avg_time,
                'times': times,
                'image': image if 'image' in locals() else None
            }
            print(f"    Average time: {avg_time:.3f}s (runs: {[f'{t:.3f}' for t in times]})")
        else:
            print(f"    Failed to complete benchmark for {name}")

    return results

def render_quality_comparison(scenes):
    """Render scenes with different quality settings"""
    print("Rendering quality comparison...")

    quality_settings = [
        (64, 64, "Low", {"early_exit_distance": 0.05, "shadow_softness": 2.0}),
        (128, 128, "Medium", {"early_exit_distance": 0.02, "shadow_softness": 4.0}),
        (256, 256, "High", {"early_exit_distance": 0.01, "shadow_softness": 8.0}),
    ]

    results = {}

    for scene_name, scene in scenes.items():
        if scene_name == "fractal":  # Just test one scene for time
            scene_results = {}

            for width, height, quality, params in quality_settings:
                print(f"  Rendering {scene_name} at {quality} quality ({width}x{height})...")

                renderer = HybridRenderer(width, height)
                renderer.set_camera((4, 3, 5), (1, 1, 1), (0, 1, 0), 45)
                renderer.set_traversal_mode(TraversalMode.SDF_ONLY)
                renderer.set_performance_params(**params)

                start_time = time.time()
                try:
                    image = renderer.render_sdf_scene(scene)
                    render_time = time.time() - start_time

                    scene_results[quality] = {
                        'image': image,
                        'render_time': render_time,
                        'resolution': (width, height),
                        'params': params
                    }

                    print(f"    Completed in {render_time:.2f}s")

                except Exception as e:
                    print(f"    Failed: {e}")

            results[scene_name] = scene_results

    return results

def analyze_sdf_complexity():
    """Analyze the computational complexity of different SDF constructions"""
    print("Analyzing SDF computational complexity...")

    # Create scenes with increasing complexity
    complexity_tests = []

    # Simple single primitive
    builder = SdfSceneBuilder()
    builder, sphere = builder.add_sphere((0, 0, 0), 1.0, 1)
    complexity_tests.append(("Single sphere", builder.build()))

    # Union of primitives
    for count in [2, 4, 8]:
        builder = SdfSceneBuilder()
        current = None

        for i in range(count):
            angle = i * 2 * math.pi / count
            pos = (1.5 * math.cos(angle), 0, 1.5 * math.sin(angle))
            builder, sphere = builder.add_sphere(pos, 0.8, 1)

            if current is None:
                current = sphere
            else:
                builder, current = builder.union(current, sphere, 0)

        complexity_tests.append((f"Union of {count} spheres", builder.build()))

    # Nested operations
    builder = SdfSceneBuilder()
    builder, outer = builder.add_sphere((0, 0, 0), 2.0, 1)
    builder, middle = builder.add_sphere((0, 0, 0), 1.5, 2)
    builder, inner = builder.add_box((0, 0, 0), (1.0, 1.0, 1.0), 3)

    builder, intersect1 = builder.intersect(middle, inner, 0)
    builder, final = builder.subtract(outer, intersect1, 0)

    complexity_tests.append(("Nested operations", final))

    # Benchmark each scene
    results = {}
    for name, scene in complexity_tests:
        print(f"  Testing {name}...")

        # Time multiple evaluations
        start_time = time.time()
        evaluation_count = 10000

        for i in range(evaluation_count):
            x = (i % 100 - 50) * 0.02
            y = ((i // 100) % 100 - 50) * 0.02
            z = 0
            scene.evaluate((x, y, z))

        eval_time = time.time() - start_time

        results[name] = {
            'primitives': scene.primitive_count(),
            'operations': scene.operation_count(),
            'total_time': eval_time,
            'avg_time_us': eval_time * 1000000 / evaluation_count
        }

        print(f"    Primitives: {scene.primitive_count()}, Operations: {scene.operation_count()}")
        print(f"    {evaluation_count} evaluations: {eval_time:.3f}s ({eval_time * 1000000 / evaluation_count:.1f} μs/eval)")

    return results

def save_demonstration_results(render_results, benchmark_results):
    """Save all demonstration results"""
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    print(f"Saving demonstration results to {output_dir}/...")

    # Save rendered images
    image_count = 0
    for scene_name, quality_results in render_results.items():
        for quality, data in quality_results.items():
            if 'image' in data and data['image'] is not None:
                filename = output_dir / f'hybrid_demo_{scene_name}_{quality.lower()}.png'

                try:
                    from PIL import Image
                    pil_image = Image.fromarray(data['image'])
                    pil_image.save(filename)
                    print(f"  Saved {scene_name} ({quality}) to {filename}")
                    image_count += 1
                except ImportError:
                    np.save(filename.with_suffix('.npy'), data['image'])
                    print(f"  Saved {scene_name} ({quality}) to {filename.with_suffix('.npy')}")
                    image_count += 1

    # Save benchmark results as text
    if benchmark_results:
        benchmark_file = output_dir / 'hybrid_benchmark_results.txt'
        with open(benchmark_file, 'w') as f:
            f.write("Hybrid Rendering Benchmark Results\n")
            f.write("==================================\n\n")

            for mode, data in benchmark_results.items():
                f.write(f"{mode}:\n")
                f.write(f"  Average time: {data['avg_time']:.3f}s\n")
                f.write(f"  Individual runs: {', '.join(f'{t:.3f}s' for t in data['times'])}\n")
                f.write("\n")

        print(f"  Saved benchmark results to {benchmark_file}")

    print(f"  Total files saved: {image_count + (1 if benchmark_results else 0)}")

def main():
    """Main demonstration function"""
    print("=== Hybrid Rendering Demo ===")
    print("This example demonstrates hybrid SDF + mesh rendering capabilities.")

    try:
        # Create procedural SDF objects
        sdf_scenes = create_procedural_sdf_objects()

        # Create hybrid scenes (simplified for this demo)
        hybrid_scenes = create_hybrid_scenes(sdf_scenes)
        print(f"Created {len(hybrid_scenes)} hybrid scene configurations")

        # Benchmark traversal modes
        if sdf_scenes:
            sample_scene = list(sdf_scenes.values())[0]
            benchmark_results = benchmark_traversal_modes(sample_scene)
        else:
            benchmark_results = {}

        # Quality comparison renders
        render_results = render_quality_comparison(sdf_scenes)

        # Complexity analysis
        complexity_results = analyze_sdf_complexity()

        # Save all results
        save_demonstration_results(render_results, benchmark_results)

        # Print summary
        print("\n=== Demo Summary ===")
        print(f"Procedural scenes created: {len(sdf_scenes)}")
        print(f"Hybrid configurations: {len(hybrid_scenes)}")
        print(f"Benchmark modes tested: {len(benchmark_results)}")
        print(f"Quality levels tested: {sum(len(r) for r in render_results.values())}")

        if benchmark_results:
            fastest_mode = min(benchmark_results.items(), key=lambda x: x[1]['avg_time'])
            print(f"Fastest rendering mode: {fastest_mode[0]} ({fastest_mode[1]['avg_time']:.3f}s avg)")

        print("\n=== Demo completed successfully! ===")
        print("Check the output/ directory for:")
        print("- Rendered images at different quality levels")
        print("- Benchmark results comparing traversal modes")
        print("- Performance analysis data")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nThis demo requires the forge3d SDF and hybrid rendering modules.")
        print("Some features may not be available if GPU acceleration is not supported.")
        return 1

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())