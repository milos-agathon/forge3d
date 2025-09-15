#!/usr/bin/env python3
"""
Basic SDF Example - Simple Primitives
=====================================

This example demonstrates the creation and rendering of basic SDF primitives
using the forge3d SDF module. It creates a scene with different geometric
shapes and renders them using CPU fallback.

Features demonstrated:
- Creating basic SDF primitives (sphere, box, cylinder)
- Building scenes with multiple objects
- CPU-based raymarching rendering
- Saving output images

Run with: python examples/sdf_basic.py
"""

import numpy as np
from forge3d.sdf import *
from pathlib import Path
import time

def create_basic_primitives_scene():
    """Create a scene with basic SDF primitives"""
    print("Creating scene with basic SDF primitives...")

    builder = SdfSceneBuilder()

    # Add a red sphere on the left
    builder, sphere = builder.add_sphere((-2, 0, 0), 1.0, 1)
    print(f"Added sphere at index {sphere}")

    # Add a green box in the center
    builder, box = builder.add_box((0, 0, 0), (0.8, 0.8, 0.8), 2)
    print(f"Added box at index {box}")

    # Add a blue cylinder on the right
    builder, cylinder = builder.add_cylinder((2, 0, 0), 0.8, 1.5, 3)
    print(f"Added cylinder at index {cylinder}")

    # Union all primitives together
    builder, union1 = builder.union(sphere, box, 0)
    builder, final_union = builder.union(union1, cylinder, 0)

    # Set scene bounds for optimization
    scene = builder.with_bounds((-4, -2, -2), (4, 2, 2)).build()

    print(f"Scene created with {scene.primitive_count()} primitives and {scene.operation_count()} operations")
    return scene

def test_sdf_evaluation(scene):
    """Test SDF evaluation at various points"""
    print("\nTesting SDF evaluation at sample points:")

    test_points = [
        (-2, 0, 0),  # Sphere center
        (0, 0, 0),   # Box center
        (2, 0, 0),   # Cylinder center
        (0, 0, 3),   # Outside all objects
        (-1, 0, 0),  # Between sphere and box
    ]

    for point in test_points:
        distance, material = scene.evaluate(point)
        status = "inside" if distance < 0 else "outside"
        print(f"  Point {point}: distance={distance:.3f}, material={material}, {status}")

def render_scene(scene, width=512, height=512):
    """Render the SDF scene"""
    print(f"\nRendering scene at {width}x{height}...")

    # Create renderer
    renderer = HybridRenderer(width, height)

    # Position camera to view all objects
    renderer.set_camera(
        origin=(0, 2, 8),     # Camera position
        target=(0, 0, 0),     # Look at origin
        up=(0, 1, 0),         # Up vector
        fov_degrees=45        # Field of view
    )

    # Use SDF-only rendering for this example
    renderer.set_traversal_mode(TraversalMode.SDF_ONLY)

    # Render (will use CPU fallback)
    start_time = time.time()
    image = renderer.render_sdf_scene(scene)
    render_time = time.time() - start_time

    print(f"Rendering completed in {render_time:.2f} seconds")
    print(f"Output image shape: {image.shape}, dtype: {image.dtype}")

    return image

def analyze_image(image):
    """Analyze the rendered image"""
    print("\nImage analysis:")

    # Calculate basic statistics
    non_zero_pixels = np.any(image[:, :, :3] != 0, axis=2)
    colored_pixels = np.sum(non_zero_pixels)
    total_pixels = image.shape[0] * image.shape[1]

    print(f"  Total pixels: {total_pixels}")
    print(f"  Non-background pixels: {colored_pixels}")
    print(f"  Background percentage: {100 * (1 - colored_pixels / total_pixels):.1f}%")

    # Find unique colors (simplified)
    unique_colors = np.unique(image.reshape(-1, 4), axis=0)
    print(f"  Unique colors: {len(unique_colors)}")

    # Check for material-based coloring
    red_pixels = np.sum(image[:, :, 0] > image[:, :, 1])  # More red than green
    green_pixels = np.sum(image[:, :, 1] > image[:, :, 0])  # More green than red
    blue_pixels = np.sum(image[:, :, 2] > image[:, :, 1])   # More blue than green

    print(f"  Approximate red pixels: {red_pixels}")
    print(f"  Approximate green pixels: {green_pixels}")
    print(f"  Approximate blue pixels: {blue_pixels}")

def save_image_with_fallback(image, filepath):
    """Save image with fallback methods"""
    filepath = Path(filepath)
    filepath.parent.mkdir(exist_ok=True)

    # Try PIL first (most common)
    try:
        from PIL import Image
        pil_image = Image.fromarray(image)
        pil_image.save(filepath)
        print(f"Image saved to {filepath} using PIL")
        return True
    except ImportError:
        pass

    # Try matplotlib
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.title('SDF Basic Primitives')
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Image saved to {filepath} using matplotlib")
        return True
    except ImportError:
        pass

    # Fallback: save as numpy array
    np.save(filepath.with_suffix('.npy'), image)
    print(f"Image saved to {filepath.with_suffix('.npy')} as numpy array")
    return True

def demonstrate_csg_operations():
    """Demonstrate different CSG operations"""
    print("\nDemonstrating CSG operations:")

    operations = [
        ("Union", lambda b, s1, s2: b.union(s1, s2, 0)),
        ("Intersection", lambda b, s1, s2: b.intersect(s1, s2, 0)),
        ("Subtraction", lambda b, s1, s2: b.subtract(s1, s2, 0)),
        ("Smooth Union", lambda b, s1, s2: b.smooth_union(s1, s2, 0.3, 0)),
    ]

    for op_name, op_func in operations:
        print(f"\n  {op_name}:")

        # Create two overlapping spheres
        builder = SdfSceneBuilder()
        builder, sphere1 = builder.add_sphere((-0.3, 0, 0), 0.8, 1)
        builder, sphere2 = builder.add_sphere((0.3, 0, 0), 0.8, 2)

        # Apply operation
        builder, result = op_func(builder, sphere1, sphere2)
        scene = builder.build()

        # Test key points
        test_points = [(-0.6, 0, 0), (0, 0, 0), (0.6, 0, 0)]  # Left, center, right
        for point in test_points:
            distance, material = scene.evaluate(point)
            status = "inside" if distance < 0 else "outside"
            print(f"    Point {point}: {status} (d={distance:.3f})")

def performance_test():
    """Simple performance test"""
    print("\nPerformance test:")

    # Create a moderately complex scene
    builder = SdfSceneBuilder()

    # Add multiple primitives
    primitives = []
    for i in range(5):
        angle = i * 2 * np.pi / 5
        x = 2 * np.cos(angle)
        z = 2 * np.sin(angle)
        builder, prim = builder.add_sphere((x, 0, z), 0.8, i + 1)
        primitives.append(prim)

    # Union them all
    current = primitives[0]
    for prim in primitives[1:]:
        builder, current = builder.union(current, prim, 0)

    scene = builder.build()

    # Time evaluation at many points
    start_time = time.time()
    num_evaluations = 10000

    for i in range(num_evaluations):
        x = (i % 100 - 50) * 0.1
        y = ((i // 100) % 100 - 50) * 0.1
        z = 0
        scene.evaluate((x, y, z))

    eval_time = time.time() - start_time
    print(f"  {num_evaluations} evaluations in {eval_time:.3f} seconds")
    print(f"  Average: {eval_time * 1000000 / num_evaluations:.1f} μs per evaluation")

def main():
    """Main example function"""
    print("=== SDF Basic Example ===")
    print("This example demonstrates basic SDF primitive creation and rendering.")

    try:
        # Create and test scene
        scene = create_basic_primitives_scene()
        test_sdf_evaluation(scene)

        # Render scene
        image = render_scene(scene, width=256, height=256)  # Smaller for faster CPU rendering
        analyze_image(image)

        # Save result
        save_image_with_fallback(image, 'output/sdf_basic_primitives.png')

        # Demonstrate additional features
        demonstrate_csg_operations()
        performance_test()

        print("\n=== Example completed successfully! ===")
        print("Check the output/ directory for rendered images.")

    except Exception as e:
        print(f"Error: {e}")
        print("This example requires the forge3d SDF module to be properly installed.")
        print("Make sure to run 'maturin develop --release' in the project directory.")
        return 1

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())