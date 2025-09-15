#!/usr/bin/env python3
"""
Advanced CSG Example - Complex Constructive Solid Geometry
==========================================================

This example demonstrates advanced CSG (Constructive Solid Geometry) operations
using the forge3d SDF module. It creates complex shapes by combining primitives
with various boolean operations including smooth blending.

Features demonstrated:
- Complex CSG tree construction
- Smooth vs hard boolean operations
- Material ID management
- Advanced scene composition
- Performance comparison

Run with: python examples/sdf_csg_advanced.py
"""

import numpy as np
from forge3d.sdf import *
import time
from pathlib import Path

def create_csg_bracket():
    """Create a mechanical bracket using CSG operations"""
    print("Creating mechanical bracket with CSG operations...")

    builder = SdfSceneBuilder()

    # Main body: rounded rectangular base
    builder, main_body = builder.add_box((0, 0, 0), (2.0, 0.5, 1.0), 1)

    # Mounting holes
    hole_positions = [(-1.5, 0, 0), (1.5, 0, 0)]
    holes = []

    for i, pos in enumerate(hole_positions):
        builder, hole = builder.add_cylinder(pos, 0.3, 1.0, 10)  # Material 10 for holes
        holes.append(hole)

    # Subtract holes from main body
    current_body = main_body
    for hole in holes:
        builder, current_body = builder.subtract(current_body, hole, 1)

    # Add mounting tabs
    tab_positions = [(-1.5, 0.8, 0), (1.5, 0.8, 0)]
    tabs = []

    for i, pos in enumerate(tab_positions):
        builder, tab = builder.add_box(pos, (0.4, 0.3, 0.8), 2)  # Material 2 for tabs
        tabs.append(tab)

    # Union tabs to main body
    for tab in tabs:
        builder, current_body = builder.union(current_body, tab, 1)

    # Add central reinforcement
    builder, reinforcement = builder.add_cylinder((0, 0.3, 0), 0.4, 0.6, 3)
    builder, final_bracket = builder.union(current_body, reinforcement, 1)

    scene = builder.with_bounds((-3, -1, -2), (3, 2, 2)).build()
    print(f"Bracket created with {scene.primitive_count()} primitives and {scene.operation_count()} operations")

    return scene

def create_organic_blob():
    """Create an organic-looking blob using smooth CSG operations"""
    print("Creating organic blob with smooth CSG operations...")

    builder = SdfSceneBuilder()

    # Create multiple overlapping spheres with smooth union
    sphere_data = [
        ((0, 0, 0), 1.0, 1),
        ((1.2, 0.5, 0.3), 0.8, 1),
        ((-0.8, 0.8, -0.2), 0.9, 1),
        ((0.3, -1.0, 0.8), 0.7, 1),
        ((-0.5, -0.3, -1.1), 0.6, 1),
    ]

    spheres = []
    for pos, radius, material in sphere_data:
        builder, sphere = builder.add_sphere(pos, radius, material)
        spheres.append(sphere)

    # Smoothly union all spheres
    current = spheres[0]
    smoothing_amount = 0.4  # Large smoothing for organic look

    for sphere in spheres[1:]:
        builder, current = builder.smooth_union(current, sphere, smoothing_amount, 1)

    # Add some detail by subtracting smaller spheres
    detail_spheres = [
        ((0.5, 0.5, 0.5), 0.3, 10),
        ((-0.3, 0.7, -0.4), 0.25, 10),
        ((0.8, -0.2, 0.3), 0.2, 10),
    ]

    for pos, radius, material in detail_spheres:
        builder, detail = builder.add_sphere(pos, radius, material)
        builder, current = builder.smooth_subtract(current, detail, 0.1, 1)

    scene = builder.with_bounds((-3, -3, -3), (3, 3, 3)).build()
    print(f"Organic blob created with {scene.primitive_count()} primitives and {scene.operation_count()} operations")

    return scene

def create_architectural_element():
    """Create an architectural element (decorative column capital)"""
    print("Creating architectural column capital...")

    builder = SdfSceneBuilder()

    # Base cylinder (column shaft)
    builder, column = builder.add_cylinder((0, -2, 0), 0.8, 2.0, 1)

    # Capital base (wider cylinder)
    builder, capital_base = builder.add_cylinder((0, 0.2, 0), 1.2, 0.6, 2)

    # Decorative torus rings
    ring_heights = [0.1, 0.3, 0.5]
    rings = []

    for height in ring_heights:
        builder, ring = builder.add_torus((0, height, 0), 1.1, 0.1, 3)
        rings.append(ring)

    # Union column and capital
    builder, structure = builder.union(column, capital_base, 1)

    # Add rings
    for ring in rings:
        builder, structure = builder.union(structure, ring, 1)

    # Add decorative spheres at cardinal points
    sphere_positions = [
        (1.3, 0.4, 0), (-1.3, 0.4, 0),
        (0, 0.4, 1.3), (0, 0.4, -1.3)
    ]

    for pos in sphere_positions:
        builder, sphere = builder.add_sphere(pos, 0.2, 4)
        builder, structure = builder.union(structure, sphere, 1)

    # Subtract fluting (decorative grooves)
    flute_count = 8
    for i in range(flute_count):
        angle = i * 2 * np.pi / flute_count
        x = 0.85 * np.cos(angle)
        z = 0.85 * np.sin(angle)

        # Create a thin box for the flute
        builder, flute = builder.add_box((x, -1, z), (0.08, 1.8, 0.15), 10)
        builder, structure = builder.subtract(structure, flute, 1)

    scene = builder.with_bounds((-2, -3, -2), (2, 1, 2)).build()
    print(f"Column capital created with {scene.primitive_count()} primitives and {scene.operation_count()} operations")

    return scene

def compare_smooth_vs_hard_operations():
    """Compare smooth vs hard CSG operations"""
    print("\nComparing smooth vs hard CSG operations:")

    # Create two intersecting spheres
    base_builder = SdfSceneBuilder()
    base_builder, sphere1 = base_builder.add_sphere((-0.4, 0, 0), 0.8, 1)
    base_builder, sphere2 = base_builder.add_sphere((0.4, 0, 0), 0.8, 2)

    operations = [
        ("Hard Union", lambda b, s1, s2: b.union(s1, s2, 0)),
        ("Smooth Union (0.1)", lambda b, s1, s2: b.smooth_union(s1, s2, 0.1, 0)),
        ("Smooth Union (0.3)", lambda b, s1, s2: b.smooth_union(s1, s2, 0.3, 0)),
        ("Smooth Union (0.5)", lambda b, s1, s2: b.smooth_union(s1, s2, 0.5, 0)),
        ("Hard Subtraction", lambda b, s1, s2: b.subtract(s1, s2, 0)),
        ("Smooth Subtraction (0.2)", lambda b, s1, s2: b.smooth_subtract(s1, s2, 0.2, 0)),
    ]

    results = {}

    for op_name, op_func in operations:
        # Create fresh builder for each operation
        builder = SdfSceneBuilder()
        builder, s1 = builder.add_sphere((-0.4, 0, 0), 0.8, 1)
        builder, s2 = builder.add_sphere((0.4, 0, 0), 0.8, 2)

        builder, result = op_func(builder, s1, s2)
        scene = builder.build()

        # Sample the result at the intersection point
        intersection_distance, _ = scene.evaluate((0, 0, 0))
        results[op_name] = intersection_distance

        print(f"  {op_name}: distance at center = {intersection_distance:.3f}")

    return results

def analyze_csg_tree_complexity():
    """Analyze the complexity of different CSG tree structures"""
    print("\nAnalyzing CSG tree complexity:")

    # Create scenes with different tree structures
    scenes = {}

    # Linear tree (chain of unions)
    builder = SdfSceneBuilder()
    current = None
    for i in range(5):
        builder, sphere = builder.add_sphere((i * 1.5, 0, 0), 0.8, i + 1)
        if current is None:
            current = sphere
        else:
            builder, current = builder.union(current, sphere, 0)

    scenes["Linear (5 spheres)"] = builder.build()

    # Balanced tree
    builder = SdfSceneBuilder()
    spheres = []
    for i in range(4):
        angle = i * np.pi / 2
        pos = (1.5 * np.cos(angle), 0, 1.5 * np.sin(angle))
        builder, sphere = builder.add_sphere(pos, 0.8, i + 1)
        spheres.append(sphere)

    # Create balanced binary tree
    builder, union1 = builder.union(spheres[0], spheres[1], 0)
    builder, union2 = builder.union(spheres[2], spheres[3], 0)
    builder, final = builder.union(union1, union2, 0)

    scenes["Balanced (4 spheres)"] = builder.build()

    # Complex nested operations
    builder = SdfSceneBuilder()
    builder, outer = builder.add_sphere((0, 0, 0), 1.5, 1)
    builder, inner1 = builder.add_sphere((0, 0, 0), 1.0, 2)
    builder, inner2 = builder.add_box((0, 0, 0), (0.8, 0.8, 0.8), 3)

    builder, intersection = builder.intersect(inner1, inner2, 0)
    builder, complex_shape = builder.subtract(outer, intersection, 0)

    scenes["Complex nested"] = builder.build()

    # Analyze each scene
    for name, scene in scenes.items():
        print(f"  {name}:")
        print(f"    Primitives: {scene.primitive_count()}")
        print(f"    Operations: {scene.operation_count()}")
        print(f"    Total nodes: {scene.primitive_count() + scene.operation_count()}")

        # Time evaluation
        start_time = time.time()
        for _ in range(1000):
            scene.evaluate((0, 0, 0))
        eval_time = time.time() - start_time

        print(f"    1000 evaluations: {eval_time * 1000:.1f} ms")

def render_comparison_scenes():
    """Render multiple scenes for comparison"""
    print("\nRendering comparison scenes...")

    scenes = {
        "bracket": create_csg_bracket(),
        "organic": create_organic_blob(),
        "architectural": create_architectural_element(),
    }

    renderer = HybridRenderer(256, 256)  # Smaller for faster rendering
    renderer.set_traversal_mode(TraversalMode.SDF_ONLY)

    results = {}

    for name, scene in scenes.items():
        print(f"  Rendering {name}...")

        # Set camera based on scene type
        if name == "bracket":
            renderer.set_camera((3, 2, 4), (0, 0, 0), (0, 1, 0), 45)
        elif name == "organic":
            renderer.set_camera((3, 3, 3), (0, 0, 0), (0, 1, 0), 45)
        elif name == "architectural":
            renderer.set_camera((2, 0, 4), (0, -1, 0), (0, 1, 0), 45)

        start_time = time.time()
        image = renderer.render_sdf_scene(scene)
        render_time = time.time() - start_time

        results[name] = {
            'image': image,
            'render_time': render_time,
            'primitives': scene.primitive_count(),
            'operations': scene.operation_count()
        }

        print(f"    Rendered in {render_time:.2f}s ({scene.primitive_count()} prims, {scene.operation_count()} ops)")

    return results

def save_results(results):
    """Save rendered results"""
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    print(f"\nSaving results to {output_dir}/...")

    for name, data in results.items():
        filename = output_dir / f'sdf_csg_{name}.png'

        # Try to save with PIL
        try:
            from PIL import Image
            pil_image = Image.fromarray(data['image'])
            pil_image.save(filename)
            print(f"  Saved {name} to {filename}")
        except ImportError:
            # Fallback to numpy
            np.save(filename.with_suffix('.npy'), data['image'])
            print(f"  Saved {name} to {filename.with_suffix('.npy')} (numpy format)")

def create_animation_frames():
    """Create frames for a simple animation"""
    print("\nCreating animation frames...")

    # Create animated smooth union
    frames = []
    frame_count = 8

    for i in range(frame_count):
        t = i / frame_count
        smoothing = 0.1 + 0.4 * (0.5 + 0.5 * np.sin(t * 2 * np.pi))  # Animated smoothing

        builder = SdfSceneBuilder()
        builder, s1 = builder.add_sphere((-0.5, 0, 0), 0.8, 1)
        builder, s2 = builder.add_sphere((0.5, 0, 0), 0.8, 2)
        builder, result = builder.smooth_union(s1, s2, smoothing, 0)

        scene = builder.build()

        # Evaluate at key points to show the animation effect
        center_dist, _ = scene.evaluate((0, 0, 0))
        frames.append((smoothing, center_dist))

    print("  Animation frames (smoothing -> center distance):")
    for i, (smoothing, distance) in enumerate(frames):
        print(f"    Frame {i}: smoothing={smoothing:.3f}, center_distance={distance:.3f}")

def main():
    """Main example function"""
    print("=== Advanced CSG Example ===")
    print("This example demonstrates advanced CSG operations and complex scene construction.")

    try:
        # Demonstrate different types of CSG constructions
        create_csg_bracket()
        create_organic_blob()
        create_architectural_element()

        # Compare operation types
        compare_smooth_vs_hard_operations()

        # Analyze complexity
        analyze_csg_tree_complexity()

        # Create animation frames
        create_animation_frames()

        # Render and save results
        results = render_comparison_scenes()
        save_results(results)

        # Print summary
        print("\n=== Summary ===")
        total_render_time = sum(r['render_time'] for r in results.values())
        print(f"Total rendering time: {total_render_time:.2f} seconds")
        print(f"Average complexity: {sum(r['primitives'] + r['operations'] for r in results.values()) / len(results):.1f} nodes per scene")

        print("\n=== Example completed successfully! ===")
        print("Check the output/ directory for rendered images showing different CSG operations.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("This example requires the forge3d SDF module to be properly installed.")
        print("Make sure to run 'maturin develop --release' in the project directory.")
        return 1

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())