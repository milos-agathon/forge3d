#!/usr/bin/env python3
"""
HDR Off-Screen Rendering and Tone Mapping Demo

Demonstrates HDR off-screen rendering with various tone mapping operators by:
1. Creating synthetic HDR scenes with high dynamic range
2. Applying different tone mapping operators (Reinhard, ACES, Uncharted2, etc.)
3. Comparing tone mapping results with statistics and visualizations
4. Saving HDR and LDR output images

Usage:
    python examples/hdr_tone_mapping.py --headless --out out/hdr_comparison.png
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import logging

# Add repository root to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import forge3d as f3d
    import forge3d.hdr as hdr
except ImportError as e:
    print(f"ERROR: Could not import forge3d: {e}")
    print("Run: maturin develop --release")
    sys.exit(1)


def create_complex_hdr_scene(width=512, height=512):
    """Create a complex HDR test scene with various lighting conditions."""
    print("Creating complex HDR test scene...")
    
    scene_data = hdr.create_hdr_test_scene(
        width=width,
        height=height,
        sun_intensity=100.0,  # Very bright sun
        sky_intensity=5.0     # Bright sky
    )
    
    config = hdr.HdrConfig(
        width=width,
        height=height,
        tone_mapping=hdr.ToneMappingOperator.REINHARD  # Initial setting
    )
    
    renderer = hdr.HdrRenderer(config)
    hdr_image = renderer.render_hdr_scene(scene_data)
    
    # Get HDR statistics
    hdr_stats = renderer.get_hdr_statistics()
    print(f"HDR Scene Statistics:")
    print(f"  Dynamic Range: {hdr_stats['dynamic_range']:.2f}")
    print(f"  Min Luminance: {hdr_stats['min_luminance']:.6f}")
    print(f"  Max Luminance: {hdr_stats['max_luminance']:.2f}")
    print(f"  Mean Luminance: {hdr_stats['mean_luminance']:.4f}")
    print(f"  Pixels > 1 cd/m²: {hdr_stats['pixels_above_1']}")
    print(f"  Pixels > 10 cd/m²: {hdr_stats['pixels_above_10']}")
    print(f"  Pixels > 100 cd/m²: {hdr_stats['pixels_above_100']}")
    
    return hdr_image, hdr_stats


def test_tone_mapping_operators(hdr_image, exposure=1.0):
    """Test all available tone mapping operators."""
    print(f"\nTesting tone mapping operators with exposure {exposure}...")
    
    # Test all available operators
    operators = [
        hdr.ToneMappingOperator.REINHARD,
        hdr.ToneMappingOperator.REINHARD_EXTENDED,
        hdr.ToneMappingOperator.ACES,
        hdr.ToneMappingOperator.UNCHARTED2,
        hdr.ToneMappingOperator.EXPOSURE,
        hdr.ToneMappingOperator.GAMMA,
        hdr.ToneMappingOperator.CLAMP,
    ]
    
    results = hdr.compare_tone_mapping_operators(hdr_image, operators, exposure)
    
    print("Tone Mapping Operator Comparison:")
    print(f"{'Operator':<20} {'Mean LDR':<12} {'Contrast':<12} {'HDR Dynamic Range':<18}")
    print("-" * 70)
    
    for op_name, result in results.items():
        hdr_dr = result['hdr_stats']['dynamic_range']
        print(f"{op_name:<20} {result['ldr_mean']:<12.4f} {result['contrast_ratio']:<12.2f} {hdr_dr:<18.2f}")
    
    return results


def create_tone_mapping_comparison(results, output_path):
    """Create a side-by-side comparison of tone mapping results."""
    print(f"\nCreating tone mapping comparison image...")
    
    # Get image dimensions from first result
    first_result = list(results.values())[0]
    height, width = first_result['ldr_data'].shape[:2]
    
    # Create comparison grid
    num_operators = len(results)
    cols = min(3, num_operators)  # Max 3 columns
    rows = (num_operators + cols - 1) // cols
    
    comparison_width = width * cols
    comparison_height = height * rows
    comparison_image = np.zeros((comparison_height, comparison_width, 3), dtype=np.uint8)
    
    # Fill comparison grid
    for i, (op_name, result) in enumerate(results.items()):
        row = i // cols
        col = i % cols
        
        y_start = row * height
        y_end = (row + 1) * height
        x_start = col * width
        x_end = (col + 1) * width
        
        ldr_data = result['ldr_data'][:, :, :3]  # Remove alpha channel
        comparison_image[y_start:y_end, x_start:x_end] = ldr_data
    
    # Save comparison image
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        f3d.numpy_to_png(str(output_path), comparison_image)
        print(f"Saved tone mapping comparison: {output_path}")
    except Exception as e:
        print(f"Warning: Could not save PNG: {e}")
        # Fallback to numpy save
        np.save(str(output_path.with_suffix('.npy')), comparison_image)
        print(f"Saved as numpy array: {output_path.with_suffix('.npy')}")
    
    return comparison_image


def test_exposure_effects(hdr_image, operator=hdr.ToneMappingOperator.ACES):
    """Test the effect of different exposure values."""
    print(f"\nTesting exposure effects with {operator.value} tone mapping...")
    
    exposure_values = [0.25, 0.5, 1.0, 2.0, 4.0]
    results = []
    
    height, width = hdr_image.shape[:2]
    
    print(f"{'Exposure':<10} {'Mean LDR':<12} {'Std LDR':<12} {'Contrast':<12}")
    print("-" * 50)
    
    for exposure in exposure_values:
        config = hdr.HdrConfig(
            width=width,
            height=height,
            tone_mapping=operator,
            exposure=exposure
        )
        
        renderer = hdr.HdrRenderer(config)
        renderer._hdr_data = hdr_image
        
        ldr_data = renderer.apply_tone_mapping()
        
        # Compute statistics
        ldr_luminance = 0.299 * ldr_data[:, :, 0] + 0.587 * ldr_data[:, :, 1] + 0.114 * ldr_data[:, :, 2]
        mean_ldr = float(np.mean(ldr_luminance))
        std_ldr = float(np.std(ldr_luminance))
        contrast = float(np.max(ldr_luminance) / max(np.min(ldr_luminance), 1))
        
        print(f"{exposure:<10.2f} {mean_ldr:<12.4f} {std_ldr:<12.4f} {contrast:<12.2f}")
        
        results.append({
            'exposure': exposure,
            'ldr_data': ldr_data,
            'mean_ldr': mean_ldr,
            'std_ldr': std_ldr,
            'contrast': contrast
        })
    
    return results


def save_hdr_analysis(hdr_image, hdr_stats, output_base):
    """Save HDR analysis data and visualizations."""
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving HDR analysis data...")
    
    # Save HDR statistics as JSON-like text file
    stats_file = output_base.with_suffix('.stats.txt')
    with open(stats_file, 'w') as f:
        f.write("HDR Scene Analysis\n")
        f.write("==================\n\n")
        for key, value in hdr_stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Saved HDR statistics: {stats_file}")
    
    # Create false color visualization
    luminance = 0.299 * hdr_image[:, :, 0] + 0.587 * hdr_image[:, :, 1] + 0.114 * hdr_image[:, :, 2]
    
    # Map luminance to false colors for HDR visualization
    false_color = np.zeros_like(hdr_image[:, :, :3])
    
    # Color mapping for different HDR ranges
    mask_dark = luminance < 0.1
    mask_normal = (luminance >= 0.1) & (luminance < 1.0)
    mask_bright = (luminance >= 1.0) & (luminance < 10.0)
    mask_very_bright = (luminance >= 10.0) & (luminance < 100.0)
    mask_extreme = luminance >= 100.0
    
    false_color[mask_dark] = [0.0, 0.0, 1.0]      # Blue for very dark
    false_color[mask_normal] = [0.0, 1.0, 0.0]    # Green for normal
    false_color[mask_bright] = [1.0, 1.0, 0.0]    # Yellow for bright
    false_color[mask_very_bright] = [1.0, 0.5, 0.0]  # Orange for very bright
    false_color[mask_extreme] = [1.0, 0.0, 0.0]   # Red for extreme
    
    # Save false color image
    false_color_u8 = (false_color * 255).astype(np.uint8)
    false_color_path = output_base.with_suffix('.false_color.png')
    
    try:
        f3d.numpy_to_png(str(false_color_path), false_color_u8)
        print(f"Saved false color HDR visualization: {false_color_path}")
    except Exception as e:
        print(f"Warning: Could not save false color image: {e}")


def main():
    parser = argparse.ArgumentParser(description="HDR tone mapping demonstration")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--out", type=str, default="out/hdr_comparison.png", help="Output file path")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--exposure", type=float, default=1.0, help="Exposure value")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    print("=== HDR Tone Mapping Demo ===")
    print(f"Mode: {'headless' if args.headless else 'interactive'}")
    print(f"Output: {args.out}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Exposure: {args.exposure}")
    
    # Check feature availability
    if not hdr.has_hdr_support():
        print("ERROR: HDR rendering module not available")
        return 1
    
    try:
        # Step 1: Create HDR test scene
        print("\n1. Creating HDR test scene...")
        hdr_image, hdr_stats = create_complex_hdr_scene(args.width, args.height)
        
        # Step 2: Test tone mapping operators
        print("\n2. Testing tone mapping operators...")
        tone_mapping_results = test_tone_mapping_operators(hdr_image, args.exposure)
        
        # Step 3: Create comparison visualization
        print("\n3. Creating tone mapping comparison...")
        comparison_image = create_tone_mapping_comparison(tone_mapping_results, args.out)
        
        # Step 4: Test exposure effects
        print("\n4. Testing exposure effects...")
        exposure_results = test_exposure_effects(hdr_image, hdr.ToneMappingOperator.ACES)
        
        # Step 5: Save HDR analysis
        print("\n5. Saving HDR analysis...")
        output_base = Path(args.out).with_suffix('')
        save_hdr_analysis(hdr_image, hdr_stats, output_base)
        
        # Step 6: Save individual tone mapping results
        print("\n6. Saving individual tone mapping results...")
        for op_name, result in tone_mapping_results.items():
            output_path = output_base.with_suffix(f'.{op_name}.png')
            try:
                f3d.numpy_to_png(str(output_path), result['ldr_data'][:, :, :3])
                print(f"Saved {op_name}: {output_path}")
            except Exception as e:
                print(f"Warning: Could not save {op_name}: {e}")
        
        # Summary
        print("\n=== HDR Tone Mapping Demo Complete ===")
        print(f"Results:")
        print(f"  HDR scene: {args.width}x{args.height}, DR={hdr_stats['dynamic_range']:.1f}")
        print(f"  Tone mapping operators tested: {len(tone_mapping_results)}")
        print(f"  Main comparison: {args.out}")
        print(f"  Individual results: {output_base}_*.png")
        print(f"  HDR analysis: {output_base}.stats.txt")
        print(f"  False color viz: {output_base}.false_color.png")
        
        # Validate tone mapping effects
        reinhard_result = tone_mapping_results.get('reinhard', {})
        aces_result = tone_mapping_results.get('aces', {})
        
        if reinhard_result and aces_result:
            reinhard_mean = reinhard_result['ldr_mean']
            aces_mean = aces_result['ldr_mean']
            difference = abs(aces_mean - reinhard_mean) / max(reinhard_mean, aces_mean) * 100
            
            print(f"\nTone mapping validation:")
            print(f"  Reinhard mean LDR: {reinhard_mean:.4f}")
            print(f"  ACES mean LDR: {aces_mean:.4f}")
            print(f"  Difference: {difference:.2f}%")
            
            if difference > 5.0:
                print("  PASS: Tone mapping operators produce different results")
            else:
                print("  INFO: Tone mapping operators produce similar results")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())