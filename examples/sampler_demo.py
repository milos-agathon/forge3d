#!/usr/bin/env python3
"""
Sampler configuration demo for forge3d

Demonstrates the sampler modes matrix and policy-based sampler creation,
showing different filtering and addressing mode combinations.
"""

import numpy as np
import forge3d as f3d
import os


def demo_sampler_modes_listing():
    """Demo listing available sampler modes"""
    print("=== Available Sampler Modes ===")
    
    try:
        modes = f3d.list_sampler_modes()
        print(f"Found {len(modes)} sampler mode combinations:")
        print()
        
        # Group by address mode for better readability
        from collections import defaultdict
        by_address = defaultdict(list)
        
        for mode in modes:
            by_address[mode['address']].append(mode)
        
        for address_mode in sorted(by_address.keys()):
            print(f"Address Mode: {address_mode}")
            print("-" * 40)
            
            for mode in by_address[address_mode]:
                print(f"  {mode['name']:20} | Filter: {mode['filter']:6} | Mip: {mode['mip_filter']}")
            print()
            
    except Exception as e:
        print(f"ERROR: Failed to list sampler modes: {e}")


def create_test_pattern():
    """Create a test pattern for demonstrating sampler behavior"""
    size = 64
    pattern = np.zeros((size, size, 4), dtype=np.uint8)
    
    # Create a pattern with sharp edges and gradients
    # Top-left: solid red
    pattern[0:size//2, 0:size//2] = [255, 0, 0, 255]
    
    # Top-right: solid green  
    pattern[0:size//2, size//2:size] = [0, 255, 0, 255]
    
    # Bottom-left: solid blue
    pattern[size//2:size, 0:size//2] = [0, 0, 255, 255]
    
    # Bottom-right: white with black diagonal stripes
    pattern[size//2:size, size//2:size] = [255, 255, 255, 255]
    for i in range(size//2, size):
        for j in range(size//2, size):
            if (i + j) % 8 < 4:
                pattern[i, j] = [0, 0, 0, 255]
    
    return pattern


def demo_basic_sampler_creation():
    """Demo basic sampler creation with different modes"""
    print("=== Basic Sampler Creation ===")
    
    try:
        # Test a few key sampler modes
        test_modes = [
            'clamp_nearest_nearest',
            'clamp_linear_linear', 
            'repeat_linear_linear',
            'mirror_nearest_linear'
        ]
        
        print("Testing sampler creation:")
        for mode in test_modes:
            try:
                # Create sampler (this tests the backend integration)
                sampler_info = f3d.make_sampler(mode)
                print(f"  ✓ {mode:20} -> {sampler_info}")
                
            except Exception as e:
                print(f"  ✗ {mode:20} -> ERROR: {e}")
        
    except Exception as e:
        print(f"ERROR: Basic sampler creation failed: {e}")


def demo_sampler_visual_comparison():
    """Create visual comparison of different sampler modes using terrain rendering"""
    print("\n=== Sampler Visual Comparison ===")
    
    try:
        # Create a small heightmap with sharp features for sampling demonstration
        size = 32
        heights = np.zeros((size, size), dtype=np.float32)
        
        # Create a pyramid-like heightmap
        center = size // 2
        for i in range(size):
            for j in range(size):
                dist_from_center = max(abs(i - center), abs(j - center))
                heights[i, j] = max(0, (center - dist_from_center)) / center * 0.5
        
        print(f"Created test heightmap: {heights.shape}, range: [{heights.min():.3f}, {heights.max():.3f}]")
        
        # Test different sampler modes with terrain rendering
        test_modes = ['clamp_nearest_nearest', 'clamp_linear_linear', 'repeat_linear_linear']
        
        output_dir = "sampler_demo_output"
        os.makedirs(output_dir, exist_ok=True)
        
        for mode in test_modes:
            try:
                print(f"Testing {mode}...")
                
                # Create renderer
                r = f3d.Renderer(256, 256)
                
                # Upload height data
                r.upload_height_r32f(heights)
                
                # Note: The sampler mode affects internal terrain rendering
                # For this demo, we're showing that different modes can be created
                sampler_info = f3d.make_sampler(mode)
                
                # Render terrain (internal sampler usage)
                rgba = r.render_terrain_rgba()
                
                # Save result
                output_file = f"{output_dir}/terrain_{mode}.png"
                f3d.numpy_to_png(output_file, rgba)
                
                print(f"  ✓ Saved {output_file}")
                
            except Exception as e:
                print(f"  ✗ {mode} failed: {e}")
        
        print(f"Visual comparison images saved to {output_dir}/")
        
    except Exception as e:
        print(f"ERROR: Visual comparison failed: {e}")


def demo_address_mode_behavior():
    """Demonstrate different address mode behaviors"""
    print("\n=== Address Mode Behavior Demo ===")
    
    address_modes = {
        'clamp': 'Clamps UV coordinates to [0,1] - edges repeat',
        'repeat': 'Wraps UV coordinates - pattern tiles',
        'mirror': 'Mirrors UV coordinates - creates symmetric tiling'
    }
    
    print("Address Mode Behaviors:")
    for mode, description in address_modes.items():
        print(f"  {mode:8}: {description}")
    
    print("\nSampler modes by address behavior:")
    try:
        modes = f3d.list_sampler_modes()
        
        for addr_mode in ['clamp', 'repeat', 'mirror']:
            matching = [m['name'] for m in modes if m['address'] == addr_mode]
            print(f"  {addr_mode:8}: {len(matching):2d} combinations -> {', '.join(matching[:3])}{'...' if len(matching) > 3 else ''}")
            
    except Exception as e:
        print(f"ERROR: Address mode demo failed: {e}")


def demo_filter_mode_comparison():
    """Demonstrate different filter mode behaviors"""
    print("\n=== Filter Mode Comparison ===")
    
    filter_info = {
        'nearest': {
            'description': 'No interpolation - sharp, pixelated look',
            'best_for': 'Pixel art, exact sampling, performance'
        },
        'linear': {
            'description': 'Bilinear interpolation - smooth blending',
            'best_for': 'Photos, smooth textures, anti-aliasing'
        }
    }
    
    print("Filter Mode Characteristics:")
    for filter_type, info in filter_info.items():
        print(f"  {filter_type:8}:")
        print(f"    Description: {info['description']}")
        print(f"    Best for:    {info['best_for']}")
        print()
    
    # Show available combinations
    try:
        modes = f3d.list_sampler_modes()
        
        nearest_modes = [m for m in modes if m['filter'] == 'nearest']
        linear_modes = [m for m in modes if m['filter'] == 'linear']
        
        print(f"Available combinations:")
        print(f"  Nearest filter: {len(nearest_modes):2d} modes")
        print(f"  Linear filter:  {len(linear_modes):2d} modes")
        
        # Show some examples
        if nearest_modes:
            print(f"  Nearest examples: {', '.join(m['name'] for m in nearest_modes[:3])}")
        if linear_modes:
            print(f"  Linear examples:  {', '.join(m['name'] for m in linear_modes[:3])}")
            
    except Exception as e:
        print(f"ERROR: Filter mode comparison failed: {e}")


def demo_mipmap_filter_effects():
    """Demonstrate mipmap filtering effects"""
    print("\n=== Mipmap Filter Effects ===")
    
    mip_filter_info = {
        'nearest': 'Sharp transitions between mip levels - may show level boundaries',
        'linear': 'Smooth transitions between mip levels - trilinear filtering'
    }
    
    print("Mipmap Filter Behaviors:")
    for filter_type, description in mip_filter_info.items():
        print(f"  {filter_type:8}: {description}")
    print()
    
    # Count combinations by mip filter
    try:
        modes = f3d.list_sampler_modes()
        
        mip_nearest = [m for m in modes if m['mip_filter'] == 'nearest']
        mip_linear = [m for m in modes if m['mip_filter'] == 'linear']
        
        print("Mipmap filtering distribution:")
        print(f"  Nearest mip filter: {len(mip_nearest):2d} combinations")
        print(f"  Linear mip filter:  {len(mip_linear):2d} combinations")
        
        # Show performance vs quality trade-offs
        print("\nPerformance vs Quality:")
        print("  nearest_nearest_nearest -> Fastest, most pixelated")
        print("  linear_linear_linear    -> Slowest, smoothest")
        print("  linear_linear_nearest   -> Good compromise")
        
    except Exception as e:
        print(f"ERROR: Mipmap filter demo failed: {e}")


def demo_sampler_recommendations():
    """Provide recommendations for different use cases"""
    print("\n=== Sampler Recommendations ===")
    
    recommendations = {
        "Terrain/Heightmaps": {
            "recommended": "clamp_linear_linear",
            "reason": "Smooth terrain, no tiling artifacts at edges"
        },
        "Repeating Textures": {
            "recommended": "repeat_linear_linear", 
            "reason": "Seamless tiling with smooth filtering"
        },
        "UI/HUD Elements": {
            "recommended": "clamp_nearest_nearest",
            "reason": "Sharp pixels, exact sampling, no blurring"
        },
        "Pixel Art": {
            "recommended": "clamp_nearest_nearest",
            "reason": "Preserves sharp pixel boundaries"
        },
        "Photo Textures": {
            "recommended": "clamp_linear_linear",
            "reason": "Smooth blending, natural appearance"
        },
        "Symmetric Patterns": {
            "recommended": "mirror_linear_linear",
            "reason": "Creates symmetric tiling effects"
        }
    }
    
    print("Use Case Recommendations:")
    for use_case, info in recommendations.items():
        print(f"  {use_case}:")
        print(f"    Mode:   {info['recommended']}")
        print(f"    Reason: {info['reason']}")
        print()


def main():
    """Run all sampler configuration demos"""
    print("forge3d Sampler Configuration Demo")
    print("==================================")
    
    try:
        demo_sampler_modes_listing()
        demo_basic_sampler_creation()
        demo_sampler_visual_comparison()
        demo_address_mode_behavior()
        demo_filter_mode_comparison() 
        demo_mipmap_filter_effects()
        demo_sampler_recommendations()
        
        print("=== Demo Complete ===")
        print("Check sampler_demo_output/ for visual comparison images.")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()