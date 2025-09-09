#!/usr/bin/env python3
"""
Advanced Example 9: Async Compute Prepass

Demonstrates asynchronous compute shader prepass for depth buffer optimization,
early-Z culling, and GPU pipeline parallelization techniques.
"""

import numpy as np
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def generate_complex_scene_geometry(num_objects: int = 50) -> dict:
    """Generate complex scene with many overlapping objects for depth testing."""
    
    np.random.seed(42)  # Reproducible
    
    objects = []
    
    for i in range(num_objects):
        # Random object properties
        obj_type = np.random.choice(['sphere', 'cube', 'plane'])
        position = np.random.uniform(-10, 10, 3)
        scale = np.random.uniform(0.5, 3.0)
        rotation = np.random.uniform(0, 2*np.pi, 3)
        
        # Material properties
        color = np.random.uniform(0.2, 1.0, 3)
        metallic = np.random.uniform(0.0, 1.0)
        roughness = np.random.uniform(0.1, 1.0)
        
        objects.append({
            'id': i,
            'type': obj_type,
            'position': position,
            'scale': scale,
            'rotation': rotation,
            'color': color,
            'metallic': metallic,
            'roughness': roughness,
        })
    
    return {
        'objects': objects,
        'camera_position': np.array([0, 0, 15]),
        'camera_target': np.array([0, 0, 0]),
        'light_position': np.array([5, 10, 5]),
    }


def simulate_depth_prepass(scene: dict, width: int, height: int) -> dict:
    """Simulate depth prepass computation for the scene."""
    
    print("Simulating depth prepass...")
    start_time = time.perf_counter()
    
    # Create depth buffer
    depth_buffer = np.ones((height, width), dtype=np.float32)  # Far plane = 1.0
    object_ids = np.full((height, width), -1, dtype=np.int32)  # No object = -1
    
    # Simple camera projection
    camera_pos = scene['camera_position']
    camera_target = scene['camera_target']
    
    # View matrix approximation
    forward = camera_target - camera_pos
    forward = forward / np.linalg.norm(forward)
    
    # Simple perspective projection parameters
    fov = 60.0 * np.pi / 180.0
    aspect = width / height
    near_plane = 0.1
    far_plane = 100.0
    
    # Create screen space coordinate grid
    x_screen = np.linspace(-1, 1, width)
    y_screen = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x_screen, y_screen)
    
    # Convert to world space directions (simplified)
    tan_half_fov = np.tan(fov * 0.5)
    world_x = X * tan_half_fov * aspect
    world_y = Y * tan_half_fov
    world_z = np.ones_like(X)
    
    # Ray directions from camera
    ray_dirs = np.stack([world_x, world_y, world_z], axis=2)
    ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=2, keepdims=True)
    
    objects_tested = 0
    objects_rendered = 0
    
    # Test each object for depth
    for obj in scene['objects']:
        objects_tested += 1
        
        # Simple sphere intersection test (all objects treated as spheres for simplicity)
        obj_pos = obj['position']
        obj_radius = obj['scale']
        
        # Vector from camera to object center
        to_object = obj_pos - camera_pos
        
        # Project object center onto camera forward direction
        depth_center = np.dot(to_object, forward)
        
        # Skip objects behind camera or too far
        if depth_center < near_plane or depth_center > far_plane:
            continue
        
        # Simple sphere-ray intersection for each pixel
        cam_to_center = obj_pos - camera_pos.reshape(1, 1, 3)
        
        # Calculate intersection with sphere
        a = np.sum(ray_dirs ** 2, axis=2)  # Should be 1.0 for normalized rays
        b = 2 * np.sum(ray_dirs * cam_to_center, axis=2)
        c = np.sum(cam_to_center ** 2, axis=2) - obj_radius ** 2
        
        discriminant = b ** 2 - 4 * a * c
        hit_mask = discriminant >= 0
        
        if not np.any(hit_mask):
            continue
        
        # Calculate intersection depth
        sqrt_disc = np.sqrt(np.maximum(discriminant, 0))
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        
        # Use nearest intersection
        t = np.where(t1 > 0, t1, t2)
        
        # Convert to normalized depth [0, 1]
        world_depth = t
        normalized_depth = (world_depth - near_plane) / (far_plane - near_plane)
        normalized_depth = np.clip(normalized_depth, 0, 1)
        
        # Update depth buffer (early Z test)
        closer_mask = hit_mask & (normalized_depth < depth_buffer) & (t > 0)
        
        if np.any(closer_mask):
            objects_rendered += 1
            depth_buffer[closer_mask] = normalized_depth[closer_mask]
            object_ids[closer_mask] = obj['id']
    
    prepass_time = time.perf_counter() - start_time
    
    return {
        'depth_buffer': depth_buffer,
        'object_ids': object_ids,
        'stats': {
            'objects_tested': objects_tested,
            'objects_rendered': objects_rendered,
            'culled_objects': objects_tested - objects_rendered,
            'prepass_time': prepass_time,
            'pixels_with_geometry': np.sum(object_ids >= 0),
            'depth_complexity': np.mean(depth_buffer[object_ids >= 0]) if np.any(object_ids >= 0) else 0,
        }
    }


def simulate_main_pass_with_prepass(scene: dict, prepass_result: dict, width: int, height: int) -> dict:
    """Simulate main rendering pass using prepass results for early-Z optimization."""
    
    print("Simulating optimized main pass...")
    start_time = time.perf_counter()
    
    depth_buffer = prepass_result['depth_buffer']
    object_ids = prepass_result['object_ids']
    
    # Create final color buffer
    color_buffer = np.zeros((height, width, 3), dtype=np.float32)
    
    # Simple lighting calculation for visible pixels only
    light_pos = scene['light_position']
    camera_pos = scene['camera_position']
    
    objects_processed = 0
    pixels_shaded = 0
    
    # Only process objects that passed the depth test
    visible_object_ids = np.unique(object_ids[object_ids >= 0])
    
    for obj_id in visible_object_ids:
        obj = scene['objects'][obj_id]
        objects_processed += 1
        
        # Find pixels belonging to this object
        obj_mask = object_ids == obj_id
        pixel_count = np.sum(obj_mask)
        pixels_shaded += pixel_count
        
        if pixel_count == 0:
            continue
        
        # Simple Lambertian shading
        obj_color = obj['color']
        
        # Approximate surface normal (pointing towards camera)
        surface_normal = np.array([0, 0, 1])  # Simplified
        
        # Light direction
        light_dir = light_pos - obj['position']
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        # Diffuse lighting
        diffuse = max(0, np.dot(surface_normal, light_dir))
        
        # Apply material properties
        metallic_factor = 1.0 - obj['metallic'] * 0.5  # Simplified metallic response
        roughness_factor = 1.0 - obj['roughness'] * 0.3
        
        final_color = obj_color * diffuse * metallic_factor * roughness_factor
        
        # Set color for this object's pixels
        color_buffer[obj_mask] = final_color
    
    main_pass_time = time.perf_counter() - start_time
    
    return {
        'color_buffer': color_buffer,
        'stats': {
            'objects_processed': objects_processed,
            'pixels_shaded': pixels_shaded,
            'main_pass_time': main_pass_time,
            'shading_efficiency': pixels_shaded / (width * height) if width * height > 0 else 0,
        }
    }


def simulate_main_pass_without_prepass(scene: dict, width: int, height: int) -> dict:
    """Simulate main rendering pass without prepass optimization (baseline)."""
    
    print("Simulating baseline main pass (no prepass)...")
    start_time = time.perf_counter()
    
    # This would process all objects and do full shading for all pixels
    # Simulated by processing all objects regardless of depth efficiency
    
    color_buffer = np.zeros((height, width, 3), dtype=np.float32)
    
    objects_processed = len(scene['objects'])
    pixels_shaded = width * height * len(scene['objects'])  # Overdraw simulation
    
    # Simulate expensive per-pixel operations for all objects
    for obj in scene['objects']:
        # Simulate expensive shading calculation
        time.sleep(0.001)  # Simulate computation time
    
    main_pass_time = time.perf_counter() - start_time
    
    return {
        'color_buffer': color_buffer,
        'stats': {
            'objects_processed': objects_processed,
            'pixels_shaded': pixels_shaded,
            'main_pass_time': main_pass_time,
            'shading_efficiency': 1.0,  # No optimization
        }
    }


def analyze_prepass_benefits(prepass_result: dict, optimized_result: dict, baseline_result: dict) -> dict:
    """Analyze the performance benefits of the depth prepass."""
    
    prepass_stats = prepass_result['stats']
    optimized_stats = optimized_result['stats']
    baseline_stats = baseline_result['stats']
    
    total_optimized_time = prepass_stats['prepass_time'] + optimized_stats['main_pass_time']
    baseline_time = baseline_stats['main_pass_time']
    
    analysis = {
        'timing_comparison': {
            'prepass_time': prepass_stats['prepass_time'],
            'optimized_main_pass': optimized_stats['main_pass_time'],
            'total_optimized': total_optimized_time,
            'baseline_main_pass': baseline_time,
            'speedup': baseline_time / total_optimized_time if total_optimized_time > 0 else 0,
            'time_saved': baseline_time - total_optimized_time,
        },
        'culling_efficiency': {
            'objects_culled': prepass_stats['culled_objects'],
            'culling_rate': prepass_stats['culled_objects'] / prepass_stats['objects_tested'] if prepass_stats['objects_tested'] > 0 else 0,
            'objects_rendered': prepass_stats['objects_rendered'],
        },
        'pixel_efficiency': {
            'pixels_with_geometry': prepass_stats['pixels_with_geometry'],
            'total_pixels': prepass_result['depth_buffer'].size,
            'pixel_utilization': prepass_stats['pixels_with_geometry'] / prepass_result['depth_buffer'].size,
            'overdraw_reduction': 1.0 - (optimized_stats['pixels_shaded'] / baseline_stats['pixels_shaded']) if baseline_stats['pixels_shaded'] > 0 else 0,
        },
        'depth_complexity': {
            'average_depth': prepass_stats['depth_complexity'],
            'depth_range_utilized': 'early_z_effective' if prepass_stats['depth_complexity'] < 0.8 else 'far_plane_dominant',
        }
    }
    
    return analysis


def visualize_depth_buffer(depth_buffer: np.ndarray) -> np.ndarray:
    """Convert depth buffer to visualizable image."""
    
    # Normalize depth values to [0, 1] and invert (near = white, far = black)
    depth_vis = 1.0 - depth_buffer
    
    # Apply gamma for better visibility
    depth_vis = np.power(depth_vis, 0.5)
    
    # Convert to 8-bit RGB
    depth_rgb = np.stack([depth_vis, depth_vis, depth_vis], axis=2)
    depth_rgb = (depth_rgb * 255).astype(np.uint8)
    
    # Add alpha channel
    depth_rgba = np.zeros((*depth_rgb.shape[:2], 4), dtype=np.uint8)
    depth_rgba[:, :, :3] = depth_rgb
    depth_rgba[:, :, 3] = 255
    
    return depth_rgba


def visualize_object_ids(object_ids: np.ndarray, num_objects: int) -> np.ndarray:
    """Convert object ID buffer to color-coded visualization."""
    
    height, width = object_ids.shape
    colors = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Create color map for objects
    if num_objects > 0:
        color_map = np.random.RandomState(42).rand(num_objects, 3)  # Reproducible colors
        
        for obj_id in range(num_objects):
            mask = object_ids == obj_id
            if np.any(mask):
                colors[mask, :3] = (color_map[obj_id] * 255).astype(np.uint8)
                colors[mask, 3] = 255
    
    # Background (no object) remains black with alpha 255
    colors[object_ids < 0, 3] = 255
    
    return colors


def main():
    """Main example execution."""
    print("Async Compute Prepass")
    print("====================")
    
    out_dir = Path(__file__).parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    
    try:
        import forge3d as f3d
        
        # Configuration
        render_width, render_height = 512, 384
        num_objects = 30
        
        print(f"Generating complex scene ({num_objects} objects)...")
        scene = generate_complex_scene_geometry(num_objects)
        
        print("Running depth prepass simulation...")
        prepass_result = simulate_depth_prepass(scene, render_width, render_height)
        
        print("Running optimized main pass...")
        optimized_result = simulate_main_pass_with_prepass(scene, prepass_result, render_width, render_height)
        
        print("Running baseline main pass...")
        baseline_result = simulate_main_pass_without_prepass(scene, render_width, render_height)
        
        print("Analyzing prepass benefits...")
        analysis = analyze_prepass_benefits(prepass_result, optimized_result, baseline_result)
        
        # Create visualizations
        print("Creating visualizations...")
        
        # Depth buffer visualization
        depth_vis = visualize_depth_buffer(prepass_result['depth_buffer'])
        depth_path = out_dir / "prepass_depth_buffer.png"
        f3d.numpy_to_png(str(depth_path), depth_vis)
        
        # Object ID visualization
        object_id_vis = visualize_object_ids(prepass_result['object_ids'], num_objects)
        object_id_path = out_dir / "prepass_object_ids.png"
        f3d.numpy_to_png(str(object_id_path), object_id_vis)
        
        # Color buffer visualization (if available)
        if optimized_result['color_buffer'] is not None:
            color_buffer = optimized_result['color_buffer']
            # Apply tone mapping and gamma correction
            color_buffer = np.power(np.clip(color_buffer, 0, 1), 1/2.2)
            color_vis = (color_buffer * 255).astype(np.uint8)
            
            # Add alpha channel
            color_rgba = np.zeros((*color_vis.shape[:2], 4), dtype=np.uint8)
            color_rgba[:, :, :3] = color_vis
            color_rgba[:, :, 3] = 255
            
            color_path = out_dir / "prepass_final_render.png"
            f3d.numpy_to_png(str(color_path), color_rgba)
        
        # Create comparison visualization
        try:
            # Side-by-side: depth buffer and object IDs
            comparison = np.zeros((render_height, render_width * 2, 4), dtype=np.uint8)
            comparison[:, :render_width] = depth_vis
            comparison[:, render_width:] = object_id_vis
            
            comp_path = out_dir / "prepass_comparison.png"
            f3d.numpy_to_png(str(comp_path), comparison)
            
        except Exception as e:
            print(f"Comparison creation failed: {e}")
        
        saved_paths = {
            'depth_buffer': str(depth_path),
            'object_ids': str(object_id_path),
        }
        
        if 'color_path' in locals():
            saved_paths['final_render'] = str(color_path)
        if 'comp_path' in locals():
            saved_paths['comparison'] = str(comp_path)
        
        # Generate comprehensive metrics
        metrics = {
            'scene_configuration': {
                'render_size': [render_width, render_height],
                'num_objects': num_objects,
                'scene_bounds': {
                    'camera_position': scene['camera_position'].tolist(),
                    'camera_target': scene['camera_target'].tolist(),
                    'light_position': scene['light_position'].tolist(),
                },
            },
            'prepass_results': prepass_result['stats'],
            'performance_analysis': analysis,
            'optimization_benefits': {
                'speedup_factor': analysis['timing_comparison']['speedup'],
                'culling_effectiveness': analysis['culling_efficiency']['culling_rate'],
                'overdraw_reduction': analysis['pixel_efficiency']['overdraw_reduction'],
                'pixel_utilization': analysis['pixel_efficiency']['pixel_utilization'],
            },
            'compute_prepass_techniques': [
                'early_z_testing',
                'depth_buffer_precomputation', 
                'object_culling',
                'overdraw_reduction',
                'pixel_quad_optimization'
            ],
            'outputs': saved_paths,
        }
        
        # Print performance summary
        print("\nAsync Compute Prepass Results:")
        print(f"  Objects in scene: {num_objects}")
        print(f"  Objects culled: {prepass_result['stats']['culled_objects']} ({analysis['culling_efficiency']['culling_rate']:.1%})")
        print(f"  Pixels with geometry: {prepass_result['stats']['pixels_with_geometry']:,} ({analysis['pixel_efficiency']['pixel_utilization']:.1%})")
        print(f"  Prepass time: {prepass_result['stats']['prepass_time']:.3f}s")
        print(f"  Optimized main pass: {optimized_result['stats']['main_pass_time']:.3f}s") 
        print(f"  Baseline main pass: {baseline_result['stats']['main_pass_time']:.3f}s")
        print(f"  Total speedup: {analysis['timing_comparison']['speedup']:.1f}x")
        print(f"  Overdraw reduction: {analysis['pixel_efficiency']['overdraw_reduction']:.1%}")
        
        # Save metrics
        import json
        metrics_path = out_dir / "prepass_metrics.json"
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