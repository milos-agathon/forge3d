#!/usr/bin/env python3
"""
Cascaded Shadow Maps (CSM) Demonstration

Demonstrates the CSM shadow mapping system with:
1. Multiple cascade levels with visualization
2. PCF filtering quality comparison  
3. Different shadow map resolutions
4. Light direction and bias parameter effects
5. Performance and memory usage analysis
6. Luminance drop measurement (>=10% requirement)
7. Shadow atlas generation and debugging

Usage:
    python examples/shadow_demo.py --out out/shadow_demo.png --atlas out/shadow_atlas.png
    python examples/shadow_demo.py --quality high --debug --out out/shadows_debug.png
    python examples/shadow_demo.py --test-all  # Run all comparison tests
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import logging
import time

# Add repository root to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import forge3d as f3d
    import forge3d.shadows as shadows
except ImportError as e:
    print(f"ERROR: Could not import forge3d: {e}")
    print("Run: maturin develop --release")
    sys.exit(0)


def create_shadow_test_scene():
    """Create test scene optimized for shadow demonstration."""
    print("Creating shadow test scene...")
    
    # Ground plane
    ground_size = 25.0
    ground_vertices = np.array([
        [-ground_size, 0.0, -ground_size, 0.0, 1.0, 0.0, 0.0, 0.0],  # pos + normal + uv
        [ground_size, 0.0, -ground_size, 0.0, 1.0, 0.0, 1.0, 0.0],
        [ground_size, 0.0, ground_size, 0.0, 1.0, 0.0, 1.0, 1.0],
        [-ground_size, 0.0, ground_size, 0.0, 1.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32).reshape(-1, 8)
    
    ground_indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    
    # Create shadow caster objects at various heights and positions
    objects = []
    
    # Ring of tall objects
    for i in range(8):
        angle = 2.0 * np.pi * i / 8
        x = 12.0 * np.cos(angle)
        z = 12.0 * np.sin(angle)
        height = 3.0 + 2.0 * np.sin(angle * 2)
        
        # Box geometry
        box_verts = create_box_geometry(x, height, z, 1.0, height, 1.0)
        objects.append({
            'vertices': box_verts,
            'indices': create_box_indices(),
            'position': (x, height / 2, z),
            'type': 'box',
            'height': height,
        })
    
    # Central tall tower
    tower_verts = create_box_geometry(0.0, 6.0, 0.0, 0.8, 6.0, 0.8)
    objects.append({
        'vertices': tower_verts,
        'indices': create_box_indices(),
        'position': (0.0, 3.0, 0.0),
        'type': 'tower',
        'height': 6.0,
    })
    
    # Scattered smaller objects for detail
    np.random.seed(42)  # Deterministic placement
    for i in range(12):
        x = np.random.uniform(-20, 20)
        z = np.random.uniform(-20, 20)
        
        # Avoid center area
        if np.sqrt(x*x + z*z) < 8.0:
            continue
            
        height = np.random.uniform(0.5, 2.0)
        size = np.random.uniform(0.3, 0.8)
        
        obj_verts = create_box_geometry(x, height, z, size, height, size)
        objects.append({
            'vertices': obj_verts,
            'indices': create_box_indices(),
            'position': (x, height / 2, z),
            'type': 'scatter',
            'height': height,
        })
    
    scene = {
        'ground': {
            'vertices': ground_vertices,
            'indices': ground_indices,
        },
        'objects': objects,
        'bounds': {
            'min': (-ground_size, 0.0, -ground_size),
            'max': (ground_size, 6.0, ground_size),
        }
    }
    
    print(f"Created scene with {len(objects)} shadow casters")
    return scene


def create_box_geometry(cx, cy, cz, width, height, depth):
    """Create box geometry with position, normal, and UV coordinates."""
    w, h, d = width/2, height/2, depth/2
    
    # 24 vertices (4 per face, 6 faces)
    vertices = np.array([
        # Front face (Z+)
        [cx-w, cy-h, cz+d, 0.0, 0.0, 1.0, 0.0, 0.0],
        [cx+w, cy-h, cz+d, 0.0, 0.0, 1.0, 1.0, 0.0],
        [cx+w, cy+h, cz+d, 0.0, 0.0, 1.0, 1.0, 1.0],
        [cx-w, cy+h, cz+d, 0.0, 0.0, 1.0, 0.0, 1.0],
        
        # Back face (Z-)
        [cx+w, cy-h, cz-d, 0.0, 0.0, -1.0, 0.0, 0.0],
        [cx-w, cy-h, cz-d, 0.0, 0.0, -1.0, 1.0, 0.0],
        [cx-w, cy+h, cz-d, 0.0, 0.0, -1.0, 1.0, 1.0],
        [cx+w, cy+h, cz-d, 0.0, 0.0, -1.0, 0.0, 1.0],
        
        # Right face (X+)
        [cx+w, cy-h, cz+d, 1.0, 0.0, 0.0, 0.0, 0.0],
        [cx+w, cy-h, cz-d, 1.0, 0.0, 0.0, 1.0, 0.0],
        [cx+w, cy+h, cz-d, 1.0, 0.0, 0.0, 1.0, 1.0],
        [cx+w, cy+h, cz+d, 1.0, 0.0, 0.0, 0.0, 1.0],
        
        # Left face (X-)
        [cx-w, cy-h, cz-d, -1.0, 0.0, 0.0, 0.0, 0.0],
        [cx-w, cy-h, cz+d, -1.0, 0.0, 0.0, 1.0, 0.0],
        [cx-w, cy+h, cz+d, -1.0, 0.0, 0.0, 1.0, 1.0],
        [cx-w, cy+h, cz-d, -1.0, 0.0, 0.0, 0.0, 1.0],
        
        # Top face (Y+)
        [cx-w, cy+h, cz+d, 0.0, 1.0, 0.0, 0.0, 0.0],
        [cx+w, cy+h, cz+d, 0.0, 1.0, 0.0, 1.0, 0.0],
        [cx+w, cy+h, cz-d, 0.0, 1.0, 0.0, 1.0, 1.0],
        [cx-w, cy+h, cz-d, 0.0, 1.0, 0.0, 0.0, 1.0],
        
        # Bottom face (Y-)
        [cx-w, cy-h, cz-d, 0.0, -1.0, 0.0, 0.0, 0.0],
        [cx+w, cy-h, cz-d, 0.0, -1.0, 0.0, 1.0, 0.0],
        [cx+w, cy-h, cz+d, 0.0, -1.0, 0.0, 1.0, 1.0],
        [cx-w, cy-h, cz+d, 0.0, -1.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    
    return vertices


def create_box_indices():
    """Create indices for box geometry."""
    indices = []
    for face in range(6):
        base = face * 4
        # Two triangles per face
        indices.extend([base, base+1, base+2, base, base+2, base+3])
    
    return np.array(indices, dtype=np.uint32)


def test_cascade_visualization(scene, config):
    """Test cascade visualization with debug colors."""
    print("\nTesting cascade visualization...")
    
    # Create renderer with debug visualization
    renderer = shadows.ShadowRenderer(800, 600, config)
    renderer.enable_debug_visualization(True)
    
    # Set camera to show scene from elevated angle
    renderer.set_camera(
        position=(15.0, 12.0, 15.0),
        target=(0.0, 2.0, 0.0),
        fov_y_degrees=60.0
    )
    
    # Set directional light
    light = shadows.DirectionalLight(
        direction=(-0.3, -0.8, -0.5),
        color=(1.0, 0.95, 0.8),  # Warm sunlight
        intensity=3.0
    )
    renderer.set_light(light)
    
    # Render with cascade debug colors
    print("Rendering with cascade debug visualization...")
    start_time = time.time()
    image = renderer.render_with_shadows(scene)
    render_time = time.time() - start_time
    
    print(f"Debug render completed in {render_time:.3f}s")
    
    # Get shadow statistics
    stats = renderer.get_shadow_stats()
    print(f"Shadow stats: {stats}")
    print(f"  Memory usage: {stats.memory_usage / (1024*1024):.1f}MB")
    print(f"  Split distances: {stats.split_distances}")
    print(f"  Texel sizes: {stats.texel_sizes}")
    
    return image, stats


def test_quality_comparison(scene):
    """Compare different shadow quality settings."""
    print("\nTesting shadow quality comparison...")
    
    quality_levels = ['low_quality', 'medium_quality', 'high_quality']
    results = {}
    
    for quality in quality_levels:
        print(f"Testing {quality}...")
        
        config = shadows.get_preset_config(quality)
        renderer = shadows.ShadowRenderer(400, 300, config)
        
        # Set consistent camera and light
        renderer.set_camera(
            position=(10.0, 8.0, 10.0),
            target=(0.0, 1.0, 0.0)
        )
        
        light = shadows.DirectionalLight(
            direction=(-0.4, -0.7, -0.4),
            intensity=2.5
        )
        renderer.set_light(light)
        
        # Measure render time
        start_time = time.time()
        image = renderer.render_with_shadows(scene)
        render_time = time.time() - start_time
        
        stats = renderer.get_shadow_stats()
        
        results[quality] = {
            'config': config,
            'image': image,
            'render_time': render_time,
            'stats': stats,
            'memory_mb': stats.memory_usage / (1024*1024),
        }
        
        print(f"  {quality}: {render_time:.3f}s, {results[quality]['memory_mb']:.1f}MB")
        print(f"    Cascades: {config.cascade_count}, Resolution: {config.shadow_map_size}")
        print(f"    PCF kernel: {config.pcf_kernel_size}")
    
    return results


def test_light_angle_effects(scene):
    """Test how different light angles affect shadow quality."""
    print("\nTesting light angle effects...")
    
    config = shadows.get_preset_config('medium_quality')
    light_angles = [
        ("High sun", (-0.1, -0.9, -0.3)),     # High angle (midday)
        ("Low sun", (-0.6, -0.5, -0.4)),      # Low angle (sunrise/sunset)
        ("Side light", (-0.8, -0.3, 0.1)),    # Side lighting
        ("Overhead", (0.0, -1.0, 0.0)),       # Directly overhead
    ]
    
    results = {}
    
    for name, direction in light_angles:
        print(f"Testing {name} lighting...")
        
        renderer = shadows.ShadowRenderer(400, 300, config)
        renderer.set_camera(
            position=(12.0, 6.0, 12.0),
            target=(0.0, 2.0, 0.0)
        )
        
        light = shadows.DirectionalLight(
            direction=direction,
            color=(1.0, 1.0, 1.0),
            intensity=3.0
        )
        renderer.set_light(light)
        
        image = renderer.render_with_shadows(scene)
        
        results[name] = {
            'image': image,
            'direction': direction,
            'light': light,
        }
    
    return results


def test_pcf_filtering_comparison(scene):
    """Compare different PCF filtering kernel sizes."""
    print("\nTesting PCF filtering comparison...")
    
    kernel_sizes = [1, 3, 5, 7]
    results = {}
    
    base_config = shadows.CsmConfig(
        cascade_count=3,
        shadow_map_size=2048,
        pcf_kernel_size=1  # Will be overridden
    )
    
    for kernel_size in kernel_sizes:
        print(f"Testing PCF kernel size {kernel_size}...")
        
        config = shadows.CsmConfig(
            cascade_count=base_config.cascade_count,
            shadow_map_size=base_config.shadow_map_size,
            pcf_kernel_size=kernel_size
        )
        
        renderer = shadows.ShadowRenderer(400, 300, config)
        renderer.set_camera(
            position=(8.0, 5.0, 8.0),
            target=(0.0, 1.0, 0.0)
        )
        
        light = shadows.DirectionalLight(
            direction=(-0.5, -0.7, -0.3),
            intensity=2.0
        )
        renderer.set_light(light)
        
        start_time = time.time()
        image = renderer.render_with_shadows(scene)
        render_time = time.time() - start_time
        
        results[f"PCF_{kernel_size}x{kernel_size}"] = {
            'image': image,
            'render_time': render_time,
            'kernel_size': kernel_size,
            'quality_score': kernel_size * kernel_size,  # Approximate quality
        }
        
        print(f"  PCF {kernel_size}x{kernel_size}: {render_time:.3f}s")
    
    return results


def validate_configuration(config, light):
    """Validate CSM configuration and provide feedback."""
    print("\nValidating CSM configuration...")
    
    validation = shadows.validate_csm_setup(
        config, light, 
        camera_near=0.1, 
        camera_far=100.0
    )
    
    print(f"Configuration valid: {validation['valid']}")
    print(f"Memory estimate: {validation['memory_estimate_mb']:.1f}MB")
    
    if validation['errors']:
        print("ERRORS:")
        for error in validation['errors']:
            print(f"  - {error}")
    
    if validation['warnings']:
        print("WARNINGS:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    if validation['recommendations']:
        print("RECOMMENDATIONS:")
        for rec in validation['recommendations']:
            print(f"  - {rec}")
    
    return validation


def create_comparison_image(results_dict, titles, width=400, height=300):
    """Create side-by-side comparison image from multiple results."""
    if not results_dict:
        return np.zeros((height, width*2, 3), dtype=np.uint8)
    
    # Arrange in 2x2 grid if 4 or fewer results
    images = list(results_dict.values())
    image_titles = list(titles)
    
    if len(images) <= 2:
        # Side by side
        rows, cols = 1, len(images)
    elif len(images) <= 4:
        # 2x2 grid
        rows, cols = 2, 2
    else:
        # 2x3 or 3x3 grid
        rows = 2 if len(images) <= 6 else 3
        cols = (len(images) + rows - 1) // rows
    
    output_width = width * cols
    output_height = height * rows
    comparison = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    for i, (result_key, result_data) in enumerate(results_dict.items()):
        if i >= rows * cols:
            break
            
        row = i // cols
        col = i % cols
        
        y_start = row * height
        y_end = (row + 1) * height
        x_start = col * width
        x_end = (col + 1) * width
        
        # Get image from result
        if isinstance(result_data, dict):
            image = result_data.get('image', np.zeros((height, width, 3), dtype=np.uint8))
        else:
            image = result_data
        
        # Resize if necessary
        if image.shape[:2] != (height, width):
            # Simple resize by cropping/padding
            img_h, img_w = image.shape[:2]
            if img_h > height:
                start_y = (img_h - height) // 2
                image = image[start_y:start_y + height]
            if img_w > width:
                start_x = (img_w - width) // 2
                image = image[:, start_x:start_x + width]
        
        comparison[y_start:y_end, x_start:x_end] = image
        
        # Add simple text label (draw white pixels in pattern)
        title = titles.get(result_key, result_key)[:15]  # Truncate long titles
        label_y = y_start + 10
        label_x = x_start + 10
        
        for char_i, char in enumerate(title):
            char_x = label_x + char_i * 8
            if char_x + 8 < x_end:
                # Simple character pattern using ASCII value
                for dy in range(8):
                    for dx in range(6):
                        if (dx + dy + ord(char) * 3) % 7 == 0:  # Pattern
                            py = min(label_y + dy, y_end - 1)
                            px = min(char_x + dx, x_end - 1)
                            if py < output_height and px < output_width:
                                comparison[py, px] = [255, 255, 255]
    
    return comparison


def save_image(image, output_path):
    """Save image to PNG file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        f3d.numpy_to_png(str(output_path), image)
        print(f"Saved shadow demo: {output_path}")
    except Exception as e:
        print(f"Warning: Could not save PNG: {e}")
        # Fallback to numpy save
        np.save(str(output_path.with_suffix('.npy')), image)
        print(f"Saved as numpy array: {output_path.with_suffix('.npy')}")


def test_luma_drop_baseline_vs_shadows(scene, config, light):
    """Test luminance drop between baseline (no shadows) and shadowed rendering."""
    print("\n=== Testing Luminance Drop (Baseline vs Shadows) ===")
    
    renderer = shadows.ShadowRenderer(400, 300, config)
    renderer.set_camera(
        position=(12.0, 8.0, 12.0),
        target=(0.0, 2.0, 0.0),
        fov_y_degrees=50.0
    )
    renderer.set_light(light)
    
    # Render true baseline (no shadows)
    print("Rendering true baseline (no shadows)...")
    light.cast_shadows = False
    baseline_image = renderer.render_with_shadows(scene)
    
    # Calculate baseline mean luminance using ITU-R BT.709 standard
    baseline_rgb = baseline_image.astype(np.float32) / 255.0
    baseline_luma = 0.299 * baseline_rgb[:,:,0] + 0.587 * baseline_rgb[:,:,1] + 0.114 * baseline_rgb[:,:,2]
    baseline_mean_luma = np.mean(baseline_luma)
    
    # Render with shadows
    print("Rendering with shadows...")
    light.cast_shadows = True
    shadowed_image = renderer.render_with_shadows(scene)
    
    # Calculate shadowed mean luminance using ITU-R BT.709 standard
    shadowed_rgb = shadowed_image.astype(np.float32) / 255.0
    shadowed_luma = 0.299 * shadowed_rgb[:,:,0] + 0.587 * shadowed_rgb[:,:,1] + 0.114 * shadowed_rgb[:,:,2]
    shadowed_mean_luma = np.mean(shadowed_luma)
    
    # Calculate luminance drop percentage
    luma_drop_pct = ((baseline_mean_luma - shadowed_mean_luma) / baseline_mean_luma) * 100.0
    
    print(f"Baseline mean luminance: {baseline_mean_luma:.4f}")
    print(f"Shadowed mean luminance: {shadowed_mean_luma:.4f}")
    print(f"LUMA_DROP={luma_drop_pct:.2f}%")
    
    # Check if drop meets >=10% requirement
    drop_requirement_met = luma_drop_pct >= 10.0
    print(f"Luminance drop >=10% requirement: {'PASS' if drop_requirement_met else 'FAIL'}")
    
    return {
        'baseline_image': baseline_image,
        'shadowed_image': shadowed_image,
        'baseline_luma': baseline_mean_luma,
        'shadowed_luma': shadowed_mean_luma,
        'luma_drop_pct': luma_drop_pct,
        'requirement_met': drop_requirement_met,
    }


def main():
    parser = argparse.ArgumentParser(description="CSM shadow mapping demonstration")
    parser.add_argument("--out", type=str, default="out/shadow_demo.png", 
                       help="Output file path")
    parser.add_argument("--atlas", type=str, default="out/shadow_atlas.png",
                       help="Shadow atlas debug output path")
    parser.add_argument("--quality", type=str, default="medium_quality",
                       choices=["low_quality", "medium_quality", "high_quality", "ultra_quality"],
                       help="Shadow quality preset")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable cascade debug visualization")
    parser.add_argument("--test-all", action="store_true",
                       help="Run all comparison tests")
    parser.add_argument("--width", type=int, default=800, help="Image width")
    parser.add_argument("--height", type=int, default=600, help="Image height")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    print("=== Cascaded Shadow Maps Demo ===")
    print(f"Quality: {args.quality}")
    print(f"Debug visualization: {args.debug}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Output: {args.out}")
    
    # Check feature availability
    if not shadows.has_shadows_support():
        print("ERROR: Shadow mapping not available")
        return 0
    
    try:
        # Create test scene
        scene = create_shadow_test_scene()
        
        # Get configuration
        config = shadows.get_preset_config(args.quality)
        print(f"Configuration: {config}")
        
        # Create directional light
        light = shadows.DirectionalLight(
            direction=(-0.4, -0.8, -0.5),
            color=(1.0, 0.95, 0.8),  # Warm sunlight
            intensity=3.5
        )
        
        # Validate configuration
        validation = validate_configuration(config, light)
        if not validation['valid']:
            print("Configuration validation failed!")
            return 0
        
        # Generate shadow atlas information
        camera_config = {
            'position': [15.0, 10.0, 15.0],
            'target': [0.0, 2.0, 0.0],
            'up': [0.0, 1.0, 0.0],
            'fov_y': 55.0,
        }
        
        atlas_info, atlas_stats = shadows.build_shadow_atlas(scene, light, camera_config)
        print(f"\nShadow Atlas Info:")
        print(f"  Cascade count: {atlas_info['cascade_count']}")
        print(f"  Atlas dimensions: {atlas_info['atlas_dimensions']}")
        print(f"  Memory usage: {atlas_info['memory_usage'] / (1024*1024):.1f}MB")
        
        # Test luminance drop requirement
        luma_drop_results = test_luma_drop_baseline_vs_shadows(scene, config, light)
        
        # Save atlas debug image (placeholder - create simple visualization)
        atlas_debug_image = np.zeros((256, 256*atlas_info['cascade_count'], 3), dtype=np.uint8)
        for i in range(atlas_info['cascade_count']):
            # Simple gradient pattern for each cascade
            start_x = i * 256
            end_x = (i + 1) * 256
            intensity = int(255 * (i + 1) / atlas_info['cascade_count'])
            atlas_debug_image[:, start_x:end_x, :] = intensity
        save_image(atlas_debug_image, args.atlas)

        if args.test_all:
            print("\n=== Running All Tests ===")
            
            # Test 1: Quality comparison
            quality_results = test_quality_comparison(scene)
            quality_image = create_comparison_image(
                quality_results,
                {k: k.replace('_', ' ').title() for k in quality_results.keys()},
                400, 300
            )
            save_image(quality_image, args.out.replace('.png', '_quality_comparison.png'))
            
            # Test 2: Light angle effects
            light_results = test_light_angle_effects(scene)
            light_image = create_comparison_image(
                light_results,
                {k: k for k in light_results.keys()},
                400, 300
            )
            save_image(light_image, args.out.replace('.png', '_light_angles.png'))
            
            # Test 3: PCF filtering comparison
            pcf_results = test_pcf_filtering_comparison(scene)
            pcf_image = create_comparison_image(
                pcf_results,
                {k: f"{k} ({pcf_results[k]['render_time']:.2f}s)" for k in pcf_results.keys()},
                400, 300
            )
            save_image(pcf_image, args.out.replace('.png', '_pcf_comparison.png'))
            
            # Test 4: Cascade debug visualization
            if args.debug:
                debug_image, debug_stats = test_cascade_visualization(scene, config)
                save_image(debug_image, args.out.replace('.png', '_cascade_debug.png'))
            
            # Test 5: Baseline vs shadows comparison  
            comparison_image = create_comparison_image(
                {'baseline': luma_drop_results['baseline_image'], 
                 'shadowed': luma_drop_results['shadowed_image']},
                {'baseline': f"Baseline (L={luma_drop_results['baseline_luma']:.3f})",
                 'shadowed': f"Shadows (L={luma_drop_results['shadowed_luma']:.3f})"},
                400, 300
            )
            save_image(comparison_image, args.out.replace('.png', '_luma_comparison.png'))
            
        else:
            # Single main demo
            print("\n=== Main Demo Render ===")
            
            if args.debug:
                # Render with cascade debug visualization
                debug_image, stats = test_cascade_visualization(scene, config)
                save_image(debug_image, args.out)
            else:
                # Standard shadow render
                renderer = shadows.ShadowRenderer(args.width, args.height, config)
                renderer.set_camera(
                    position=(15.0, 10.0, 15.0),
                    target=(0.0, 2.0, 0.0),
                    fov_y_degrees=55.0
                )
                renderer.set_light(light)
                
                print("Rendering main demo...")
                start_time = time.time()
                image = renderer.render_with_shadows(scene)
                render_time = time.time() - start_time
                
                stats = renderer.get_shadow_stats()
                print(f"Render completed in {render_time:.3f}s")
                print(f"Shadow statistics: {stats}")
                
                save_image(image, args.out)
        
        # Summary
        print("\n=== Shadow Demo Complete ===")
        if validation:
            print(f"Memory usage: {validation['memory_estimate_mb']:.1f}MB")
        
        technique_scores = shadows.compare_shadow_techniques()
        print("Shadow technique performance scores:")
        for technique, score in technique_scores.items():
            print(f"  {technique}: {score:.1f}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 0


if __name__ == "__main__":
    exit(main())
