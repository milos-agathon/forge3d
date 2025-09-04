#!/usr/bin/env python3
"""
HDR image processing demo for forge3d

Demonstrates HDR (Radiance) image loading, tone mapping, and processing
capabilities using the forge3d.hdr module.
"""

from _import_shim import ensure_repo_import
ensure_repo_import()

import numpy as np
import os
import forge3d as f3d
import forge3d.hdr as hdr


def create_synthetic_hdr():
    """Create a synthetic HDR image for demonstration"""
    width, height = 256, 128
    
    # Create HDR image with wide dynamic range
    hdr_image = np.zeros((height, width, 3), dtype=np.float32)
    
    # Create a gradient from very dark to very bright
    for x in range(width):
        # Exponential brightness curve from 0.001 to 100.0
        brightness = 0.001 * (100.0 / 0.001) ** (x / (width - 1))
        hdr_image[:, x, :] = brightness
    
    # Add some color variation
    for y in range(height):
        color_factor = y / (height - 1)
        hdr_image[y, :, 0] *= (1.0 + color_factor)  # More red at top
        hdr_image[y, :, 2] *= (2.0 - color_factor)  # More blue at bottom
    
    return hdr_image


def save_test_hdr(filename, hdr_data):
    """Save synthetic HDR data as a simple test file (placeholder)"""
    # Note: This is a placeholder - actual HDR saving would require
    # implementing the Radiance HDR format writer
    print(f"[NOTE] HDR saving not implemented - would save {hdr_data.shape} to {filename}")
    return filename


def demo_hdr_tone_mapping():
    """Demonstrate different tone mapping methods"""
    print("=== HDR Tone Mapping Demo ===")
    
    # Create synthetic HDR image
    hdr_image = create_synthetic_hdr()
    print(f"Created synthetic HDR image: {hdr_image.shape}")
    print(f"HDR range: [{hdr_image.min():.4f}, {hdr_image.max():.4f}]")
    
    # Test different tone mapping methods
    tone_map_methods = {
        'reinhard': 'Reinhard operator - good for natural images',
        'gamma': 'Simple gamma correction - preserves bright areas',
        'clamp': 'Simple clamping - harsh cutoff',
        'aces': 'ACES filmic - cinematic look'
    }
    
    output_dir = "hdr_demo_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nTesting tone mapping methods:")
    for method, description in tone_map_methods.items():
        try:
            # Apply tone mapping
            ldr_image = hdr.hdr_to_ldr(hdr_image, method=method, exposure=1.0)
            
            print(f"  {method:8}: {description}")
            print(f"           LDR range: [{ldr_image.min():.3f}, {ldr_image.max():.3f}]")
            
            # Convert to uint8 and save
            ldr_u8 = (ldr_image * 255).astype(np.uint8)
            output_file = f"{output_dir}/tonemap_{method}.png"
            f3d.numpy_to_png(output_file, ldr_u8)
            print(f"           Saved: {output_file}")
            
        except Exception as e:
            print(f"  {method:8}: ERROR - {e}")
        print()


def demo_exposure_control():
    """Demonstrate exposure control in tone mapping"""
    print("=== Exposure Control Demo ===")
    
    hdr_image = create_synthetic_hdr()
    
    # Test different exposure values
    exposures = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    output_dir = "hdr_demo_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Testing different exposure values with Reinhard tone mapping:")
    for exposure in exposures:
        try:
            ldr_image = hdr.hdr_to_ldr(hdr_image, method='reinhard', exposure=exposure)
            
            # Calculate brightness statistics
            avg_brightness = np.mean(ldr_image)
            max_brightness = np.max(ldr_image)
            
            print(f"  Exposure {exposure:3.1f}: avg={avg_brightness:.3f}, max={max_brightness:.3f}")
            
            # Save result
            ldr_u8 = (ldr_image * 255).astype(np.uint8)
            output_file = f"{output_dir}/exposure_{exposure:.1f}.png"
            f3d.numpy_to_png(output_file, ldr_u8)
            
        except Exception as e:
            print(f"  Exposure {exposure:3.1f}: ERROR - {e}")


def demo_hdr_info_analysis():
    """Demonstrate HDR image information analysis"""
    print("\n=== HDR Image Analysis Demo ===")
    
    # Create different types of HDR images for analysis
    test_images = {
        'gradient': create_synthetic_hdr(),
        'high_contrast': create_high_contrast_hdr(),
        'natural_range': create_natural_range_hdr()
    }
    
    print("Analyzing different HDR image types:")
    for name, img in test_images.items():
        print(f"\n{name.upper()} IMAGE:")
        print(f"  Shape: {img.shape}")
        print(f"  Range: [{img.min():.6f}, {img.max():.6f}]")
        print(f"  Dynamic range: {img.max() / (img.min() + 1e-10):.0f}:1")
        print(f"  Mean brightness: {img.mean():.6f}")
        print(f"  Std deviation: {img.std():.6f}")
        
        # Analyze channel statistics
        for c, channel_name in enumerate(['Red', 'Green', 'Blue']):
            channel_data = img[:, :, c]
            print(f"  {channel_name:5} channel: [{channel_data.min():.4f}, {channel_data.max():.4f}]")


def create_high_contrast_hdr():
    """Create HDR image with extreme contrast"""
    width, height = 128, 64
    hdr_image = np.zeros((height, width, 3), dtype=np.float32)
    
    # Very dark regions
    hdr_image[:height//2, :, :] = 0.001
    
    # Very bright regions
    hdr_image[height//2:, :, :] = 50.0
    
    # Add some mid-tones in the center
    center_y = height // 2
    hdr_image[center_y-2:center_y+2, :, :] = 1.0
    
    return hdr_image


def create_natural_range_hdr():
    """Create HDR image with natural-looking dynamic range"""
    width, height = 128, 64
    
    # Create a simple scene: sky, horizon, and ground
    hdr_image = np.zeros((height, width, 3), dtype=np.float32)
    
    # Sky (bright)
    sky_height = height // 3
    hdr_image[:sky_height, :, :] = [10.0, 15.0, 20.0]  # Bright blue sky
    
    # Horizon (medium)
    horizon_height = height // 6
    hdr_image[sky_height:sky_height+horizon_height, :, :] = [2.0, 1.5, 1.0]  # Sunset colors
    
    # Ground (dark)
    hdr_image[sky_height+horizon_height:, :, :] = [0.1, 0.05, 0.02]  # Dark ground
    
    return hdr_image


def demo_hdr_workflow():
    """Demonstrate a complete HDR processing workflow"""
    print("\n=== Complete HDR Workflow Demo ===")
    
    # Step 1: Create/Load HDR image
    print("Step 1: Creating HDR image...")
    hdr_image = create_natural_range_hdr()
    print(f"  HDR image: {hdr_image.shape}, range: [{hdr_image.min():.3f}, {hdr_image.max():.3f}]")
    
    # Step 2: Analyze HDR properties
    print("\nStep 2: Analyzing HDR properties...")
    dynamic_range = hdr_image.max() / (hdr_image.min() + 1e-10)
    print(f"  Dynamic range: {dynamic_range:.0f}:1")
    
    # Step 3: Choose appropriate tone mapping
    print("\nStep 3: Selecting tone mapping method...")
    if dynamic_range > 100:
        recommended_method = 'aces'
        recommended_exposure = 0.5
    elif dynamic_range > 10:
        recommended_method = 'reinhard'
        recommended_exposure = 1.0
    else:
        recommended_method = 'gamma'
        recommended_exposure = 1.5
    
    print(f"  Recommended: {recommended_method} with exposure {recommended_exposure}")
    
    # Step 4: Apply tone mapping
    print("\nStep 4: Applying tone mapping...")
    ldr_result = hdr.hdr_to_ldr(hdr_image, method=recommended_method, exposure=recommended_exposure)
    print(f"  LDR result: range [{ldr_result.min():.3f}, {ldr_result.max():.3f}]")
    
    # Step 5: Save final result
    print("\nStep 5: Saving final result...")
    output_dir = "hdr_demo_output"
    os.makedirs(output_dir, exist_ok=True)
    
    ldr_u8 = (ldr_result * 255).astype(np.uint8)
    final_output = f"{output_dir}/workflow_result.png"
    f3d.numpy_to_png(final_output, ldr_u8)
    print(f"  Saved: {final_output}")
    
    print("\nWorkflow complete!")


def demo_hdr_file_info():
    """Demonstrate HDR file information utilities"""
    print("\n=== HDR File Information Demo ===")
    
    # Since we don't have actual HDR files, simulate the info that would be available
    simulated_files = {
        'environment.hdr': {'width': 2048, 'height': 1024, 'size_mb': 24.0},
        'interior.hdr': {'width': 1024, 'height': 512, 'size_mb': 6.0},
        'sunset.hdr': {'width': 4096, 'height': 2048, 'size_mb': 96.0}
    }
    
    print("Simulated HDR file information:")
    print("Filename          | Resolution  | Memory  | Notes")
    print("------------------|-------------|---------|------------------------")
    
    for filename, info in simulated_files.items():
        resolution = f"{info['width']}x{info['height']}"
        memory = f"{info['size_mb']:.1f} MB"
        
        # Add usage notes based on size
        if info['size_mb'] > 50:
            notes = "Large - consider downsampling"
        elif info['size_mb'] > 20:
            notes = "Medium - good for environments"
        else:
            notes = "Small - good for testing"
        
        print(f"{filename:17} | {resolution:11} | {memory:7} | {notes}")
    
    print("\nNote: HDR file loading from disk would use hdr.load_hdr(filename)")


def main():
    """Run all HDR processing demos"""
    print("forge3d HDR Processing Demo")
    print("===========================")
    
    try:
        demo_hdr_tone_mapping()
        demo_exposure_control()
        demo_hdr_info_analysis()
        demo_hdr_workflow()
        demo_hdr_file_info()
        
        print("\n=== Demo Complete ===")
        print("Generated images saved to hdr_demo_output/")
        print("Compare the different tone mapping methods and exposure settings.")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
