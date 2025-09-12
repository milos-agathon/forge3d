#!/usr/bin/env python3
"""S6 Demo: Overview/LOD selection for efficient raster I/O"""

import argparse
import numpy as np
from pathlib import Path

def demonstrate_overview_selection():
    """Demonstrate overview selection logic."""
    print("\n=== Overview Selection Demo ===")
    
    # Simulate dataset with different overview levels
    full_res = {"width": 4096, "height": 4096, "pixel_size": 1.0}
    overviews = [
        {"factor": 2, "width": 2048, "height": 2048, "pixel_size": 2.0},
        {"factor": 4, "width": 1024, "height": 1024, "pixel_size": 4.0},
        {"factor": 8, "width": 512, "height": 512, "pixel_size": 8.0},
        {"factor": 16, "width": 256, "height": 256, "pixel_size": 16.0}
    ]
    
    # Test different target resolutions
    test_resolutions = [1.0, 3.0, 7.0, 12.0, 20.0]
    results = []
    
    for target_res in test_resolutions:
        # Select best overview
        best_overview = None
        best_score = float('inf')
        
        # Check full resolution
        score = abs(full_res["pixel_size"] - target_res)
        if score < best_score:
            best_score = score
            best_overview = {"level": "full", **full_res, "factor": 1}
        
        # Check overviews
        for i, overview in enumerate(overviews):
            score = abs(overview["pixel_size"] - target_res)
            if score < best_score:
                best_score = score
                best_overview = {"level": f"overview_{i}", **overview}
        
        # Calculate data reduction
        full_pixels = full_res["width"] * full_res["height"]
        selected_pixels = best_overview["width"] * best_overview["height"]
        reduction = 1.0 - (selected_pixels / full_pixels)
        
        result = {
            "target_resolution": target_res,
            "selected_level": best_overview["level"],
            "selected_factor": best_overview["factor"],
            "pixel_reduction": reduction,
            "meets_60_percent_goal": reduction >= 0.6
        }
        results.append(result)
        
        print(f"Target res {target_res:4.1f}: {best_overview['level']} "
              f"(factor {best_overview['factor']}x, {reduction:.1%} reduction)")
    
    return create_overview_visualization(), results

def create_overview_visualization():
    """Create visualization showing different overview levels."""
    base_size = 256
    data = np.zeros((3, base_size, base_size), dtype=np.float32)
    
    # Create pattern that shows resolution differences
    x = np.linspace(-4, 4, base_size)
    y = np.linspace(-4, 4, base_size)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # High frequency pattern that will show aliasing at lower resolutions
    data[0] = np.sin(X * np.pi * 8) * np.cos(Y * np.pi * 8)  # High freq
    data[1] = np.sin(X * np.pi * 4) * np.cos(Y * np.pi * 4)  # Med freq  
    data[2] = np.sin(X * np.pi * 2) * np.cos(Y * np.pi * 2)  # Low freq
    
    # Normalize
    data = (data + 1) / 2  # Scale to [0, 1]
    
    return data

def save_outputs(data, results, output_path):
    """Save demo outputs."""
    try:
        import forge3d
        rgb_uint8 = (data * 255).astype(np.uint8)
        rgb_hwc = np.transpose(rgb_uint8, (1, 2, 0))
        forge3d.numpy_to_png(str(output_path), rgb_hwc)
        print(f"Saved to {output_path}")
    except:
        print("Could not save PNG")
    
    # Save report
    import json
    report = {
        "demo": "S6: Overview/LOD Selection",
        "overview_tests": results,
        "summary": f"{len([r for r in results if r['meets_60_percent_goal']])} of {len(results)} tests met 60% reduction goal"
    }
    
    report_path = output_path.with_suffix('.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='S6: Overview selection demo')
    parser.add_argument('--out', default='reports/s6_overviews.png')
    args = parser.parse_args()
    
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Forge3D Workstream S6: Overview Selection Demo")
    print("=" * 45)
    
    data, results = demonstrate_overview_selection()
    save_outputs(data, results, output_path)
    
    print(f"\nDemo complete: {output_path}")

if __name__ == "__main__":
    main()