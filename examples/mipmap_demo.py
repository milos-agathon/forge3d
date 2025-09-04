#!/usr/bin/env python3
"""
Mipmap generation demo for forge3d

Demonstrates CPU mipmap generation with different methods and settings,
and shows how to save and analyze mipmap pyramids.
"""

import numpy as np
import forge3d as f3d
import forge3d.texture as tex
import os


def create_test_texture(width=512, height=512):
    """Create a test texture with high-frequency details for mipmap testing"""
    x = np.linspace(0, 10 * np.pi, width)
    y = np.linspace(0, 10 * np.pi, height)
    X, Y = np.meshgrid(x, y)
    
    # Create a pattern with high frequency details that will benefit from mipmapping
    pattern = (
        np.sin(X) * np.cos(Y) * 0.5 +  # Low frequency base
        np.sin(X * 4) * np.cos(Y * 4) * 0.3 +  # Medium frequency
        np.sin(X * 16) * np.cos(Y * 16) * 0.2  # High frequency detail
    )
    
    # Convert to RGBA (0-1 range)
    pattern = (pattern + 1.0) * 0.5  # Normalize to [0, 1]
    
    rgba = np.zeros((height, width, 4), dtype=np.float32)
    rgba[:, :, 0] = pattern  # Red channel
    rgba[:, :, 1] = pattern * 0.8  # Green channel  
    rgba[:, :, 2] = pattern * 0.6  # Blue channel
    rgba[:, :, 3] = 1.0  # Alpha channel
    
    return rgba


def demo_basic_mipmap_generation():
    """Demo basic mipmap generation with different methods"""
    print("=== Basic Mipmap Generation Demo ===")
    
    # Create test texture
    texture_data = create_test_texture(256, 256)
    print(f"Created test texture: {texture_data.shape}, range: [{texture_data.min():.3f}, {texture_data.max():.3f}]")
    
    # Test different mipmap generation methods
    methods = ['box', 'linear', 'cubic']
    
    for method in methods:
        print(f"\n--- Testing {method} method ---")
        
        try:
            # Generate mipmaps
            pyramid = tex.generate_mipmaps(
                texture_data,
                method=method,
                max_levels=None,  # Generate full pyramid
                gamma_aware=True
            )
            
            print(f"Generated {len(pyramid)} mipmap levels using {method}")
            
            # Show level information
            for i, level in enumerate(pyramid):
                print(f"  Level {i}: {level.shape[1]}x{level.shape[0]} pixels")
            
            # Save the base level and a few mip levels as examples
            output_dir = "mipmap_demo_output"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save base level
            base_u8 = (pyramid[0] * 255).astype(np.uint8)
            f3d.numpy_to_png(f"{output_dir}/base_{method}.png", base_u8)
            
            # Save mip level 2 (4x smaller)
            if len(pyramid) > 2:
                mip2_u8 = (pyramid[2] * 255).astype(np.uint8)
                f3d.numpy_to_png(f"{output_dir}/mip2_{method}.png", mip2_u8)
            
            # Save mip level 4 (16x smaller) if available
            if len(pyramid) > 4:
                mip4_u8 = (pyramid[4] * 255).astype(np.uint8)
                f3d.numpy_to_png(f"{output_dir}/mip4_{method}.png", mip4_u8)
            
            print(f"  Saved mipmap examples to {output_dir}/")
            
        except Exception as e:
            print(f"  ERROR: {method} method failed: {e}")


def demo_gamma_aware_mipmaps():
    """Demo the difference between gamma-aware and linear mipmapping"""
    print("\n=== Gamma-Aware vs Linear Mipmapping Demo ===")
    
    # Create a high-contrast test pattern
    checkerboard_size = 128
    checker = np.zeros((checkerboard_size, checkerboard_size, 4), dtype=np.float32)
    
    # Create checkerboard pattern (white and black squares)
    square_size = 8
    for i in range(0, checkerboard_size, square_size * 2):
        for j in range(0, checkerboard_size, square_size * 2):
            # White squares
            checker[i:i+square_size, j:j+square_size] = [1.0, 1.0, 1.0, 1.0]
            checker[i+square_size:i+2*square_size, j+square_size:j+2*square_size] = [1.0, 1.0, 1.0, 1.0]
    
    print(f"Created checkerboard test pattern: {checker.shape}")
    
    # Generate mipmaps with and without gamma awareness
    try:
        linear_pyramid = tex.generate_mipmaps(checker, method='box', gamma_aware=False)
        gamma_pyramid = tex.generate_mipmaps(checker, method='box', gamma_aware=True)
        
        print(f"Generated {len(linear_pyramid)} levels each for linear and gamma-aware")
        
        # Save comparison at mip level 3 (8x smaller)
        if len(linear_pyramid) > 3 and len(gamma_pyramid) > 3:
            output_dir = "mipmap_demo_output"
            os.makedirs(output_dir, exist_ok=True)
            
            linear_u8 = (linear_pyramid[3] * 255).astype(np.uint8)
            gamma_u8 = (gamma_pyramid[3] * 255).astype(np.uint8)
            
            f3d.numpy_to_png(f"{output_dir}/checkerboard_linear.png", linear_u8)
            f3d.numpy_to_png(f"{output_dir}/checkerboard_gamma.png", gamma_u8)
            
            # Calculate and show the difference
            avg_linear = np.mean(linear_pyramid[3][:, :, 0])
            avg_gamma = np.mean(gamma_pyramid[3][:, :, 0])
            
            print(f"  Linear mipmap average: {avg_linear:.3f}")
            print(f"  Gamma-aware average: {avg_gamma:.3f}")
            print(f"  Gamma-aware should be brighter (closer to 0.5 for 50% checker)")
            print(f"  Saved comparison images to {output_dir}/")
            
    except Exception as e:
        print(f"  ERROR: Gamma comparison failed: {e}")


def demo_mipmap_memory_estimation():
    """Demo mipmap memory usage estimation"""
    print("\n=== Mipmap Memory Estimation Demo ===")
    
    texture_sizes = [(64, 64), (256, 256), (512, 512), (1024, 1024), (2048, 2048)]
    
    print("Texture Size -> Full Pyramid Memory")
    print("-----------------------------------")
    
    for width, height in texture_sizes:
        try:
            # Estimate memory for full mipmap pyramid
            memory_mb = tex.estimate_mipmap_memory(width, height, channels=4, dtype='float32')
            
            # Calculate number of levels
            max_dimension = max(width, height)
            num_levels = int(np.floor(np.log2(max_dimension))) + 1
            
            print(f"{width:4d}x{height:<4d} -> {memory_mb:6.1f} MB ({num_levels:2d} levels)")
            
        except Exception as e:
            print(f"{width:4d}x{height:<4d} -> ERROR: {e}")


def demo_texture_pyramid_analysis():
    """Demo analysis of generated texture pyramids"""
    print("\n=== Texture Pyramid Analysis Demo ===")
    
    # Create gradient texture for analysis
    gradient = np.zeros((128, 128, 4), dtype=np.float32)
    
    # Create horizontal gradient
    for i in range(128):
        gradient[:, i, 0] = i / 127.0  # Red gradient
        gradient[:, i, 1] = 0.5  # Constant green
        gradient[:, i, 2] = 1.0 - (i / 127.0)  # Inverse blue gradient
        gradient[:, i, 3] = 1.0  # Alpha
    
    print("Created gradient test texture")
    
    try:
        # Generate mipmap pyramid
        pyramid = tex.generate_mipmaps(gradient, method='box', max_levels=6)
        
        print(f"Generated pyramid with {len(pyramid)} levels")
        print("\nPyramid Analysis:")
        print("Level | Size      | Avg Red | Avg Blue | Memory")
        print("------|-----------|---------|----------|--------")
        
        total_memory = 0
        for i, level in enumerate(pyramid):
            height, width, channels = level.shape
            avg_red = np.mean(level[:, :, 0])
            avg_blue = np.mean(level[:, :, 2])
            
            # Memory in bytes (float32 = 4 bytes per component)
            level_memory = height * width * channels * 4
            total_memory += level_memory
            
            print(f"{i:5d} | {width:4d}x{height:<4d} | {avg_red:7.3f} | {avg_blue:8.3f} | {level_memory:6d}")
        
        print(f"Total pyramid memory: {total_memory} bytes ({total_memory/1024/1024:.2f} MB)")
        
        # The red channel should converge to 0.5 (average of 0-1 gradient)
        # The blue channel should also converge to 0.5 (average of 1-0 gradient)
        final_red = np.mean(pyramid[-1][:, :, 0])
        final_blue = np.mean(pyramid[-1][:, :, 2])
        
        print(f"\nConvergence check:")
        print(f"  Final red average: {final_red:.3f} (should be ~0.5)")
        print(f"  Final blue average: {final_blue:.3f} (should be ~0.5)")
        
        if abs(final_red - 0.5) < 0.05 and abs(final_blue - 0.5) < 0.05:
            print("  ✓ Gradient converged correctly")
        else:
            print("  ⚠ Gradient convergence may be off")
            
    except Exception as e:
        print(f"  ERROR: Pyramid analysis failed: {e}")


def main():
    """Run all mipmap generation demos"""
    print("forge3d Mipmap Generation Demo")
    print("==============================")
    
    try:
        demo_basic_mipmap_generation()
        demo_gamma_aware_mipmaps()
        demo_mipmap_memory_estimation()
        demo_texture_pyramid_analysis()
        
        print("\n=== Demo Complete ===")
        print("Generated images saved to mipmap_demo_output/")
        print("Check the output files to see the visual differences between methods.")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()