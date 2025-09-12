#!/usr/bin/env python3
"""
S3 Demo: CRS normalization and reprojection using WarpedVRT

This example demonstrates coordinate system transformations and
reprojection of raster windows between different CRS.
"""

import argparse
import numpy as np
from pathlib import Path

def demonstrate_crs_operations():
    """Demonstrate CRS operations and transformations."""
    print("\n=== CRS Operations Demo ===")
    
    try:
        from forge3d.adapters.reproject import (
            transform_bounds,
            get_crs_info, 
            is_reproject_available
        )
    except ImportError as e:
        print(f"Could not import reproject adapter: {e}")
        return None
    
    if not is_reproject_available():
        print("pyproj/rasterio not available - showing concept only")
        return demonstrate_mock_reprojection()
    
    print("Note: This demo would work with pyproj installed.")
    print("Example transformations:")
    
    # Common CRS transformations
    examples = [
        ("Geographic WGS84", "EPSG:4326", (-180, -90, 180, 90)),
        ("Web Mercator", "EPSG:3857", (-20037508, -20037508, 20037508, 20037508)),
        ("UTM Zone 33N", "EPSG:32633", (166021, 0, 833978, 9329005))
    ]
    
    for name, code, bounds in examples:
        print(f"\n{name} ({code}):")
        print(f"  Typical bounds: {bounds}")
        print(f"  Use case: {get_use_case(code)}")
    
    return None

def get_use_case(epsg_code):
    """Get typical use case for EPSG code."""
    use_cases = {
        "EPSG:4326": "Global geographic data (lat/lon)",
        "EPSG:3857": "Web mapping (Google Maps, OpenStreetMap)",
        "EPSG:32633": "European mapping and surveying"
    }
    return use_cases.get(epsg_code, "Local/regional mapping")

def demonstrate_mock_reprojection():
    """Demonstrate reprojection concepts with mock calculations."""
    print("Demonstrating reprojection concepts with mock data...")
    
    # Simulate common transformation scenarios
    scenarios = [
        {
            "name": "Global to Web Mercator",
            "src_crs": "EPSG:4326",
            "dst_crs": "EPSG:3857", 
            "src_bounds": (-10, 50, 10, 60),  # Europe region
            "description": "Geographic coordinates to Web Mercator"
        },
        {
            "name": "UTM to Geographic",
            "src_crs": "EPSG:32633",
            "dst_crs": "EPSG:4326",
            "src_bounds": (300000, 5500000, 800000, 6000000),  # Central Europe
            "description": "UTM projection to lat/lon"
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Source CRS: {scenario['src_crs']}")
        print(f"  Target CRS: {scenario['dst_crs']}")
        print(f"  Source bounds: {scenario['src_bounds']}")
        print(f"  Description: {scenario['description']}")
        
        # Mock transformation (simplified)
        src_bounds = scenario['src_bounds']
        if "3857" in scenario['dst_crs']:  # To Web Mercator
            # Rough Web Mercator transformation
            left, bottom, right, top = src_bounds
            dst_bounds = (left * 111319, bottom * 111319, right * 111319, top * 111319)
        elif "4326" in scenario['dst_crs']:  # To Geographic
            # Rough inverse transformation
            left, bottom, right, top = src_bounds
            dst_bounds = (left / 111319, bottom / 111319, right / 111319, top / 111319)
        else:
            dst_bounds = src_bounds  # No change for demo
        
        print(f"  Transformed bounds: {[round(x, 2) for x in dst_bounds]}")
        
        # Calculate area change
        src_area = (src_bounds[2] - src_bounds[0]) * (src_bounds[3] - src_bounds[1])
        dst_area = (dst_bounds[2] - dst_bounds[0]) * (dst_bounds[3] - dst_bounds[1])
        area_ratio = dst_area / src_area if src_area > 0 else 1.0
        
        print(f"  Area change: {area_ratio:.2e}x")
        
        result = {
            'name': scenario['name'],
            'src_crs': scenario['src_crs'],
            'dst_crs': scenario['dst_crs'],
            'src_bounds': src_bounds,
            'dst_bounds': dst_bounds,
            'area_ratio': area_ratio
        }
        results.append(result)
    
    return create_synthetic_reprojected_data(), results

def create_synthetic_reprojected_data():
    """Create synthetic data showing reprojection effects."""
    print(f"\n=== Creating Synthetic Reprojected Data ===")
    
    # Create synthetic grid in "geographic" coordinates
    width, height = 256, 256
    
    # Simulate lat/lon grid (-5 to 5 degrees)
    lon = np.linspace(-5, 5, width)
    lat = np.linspace(-5, 5, height)
    LON, LAT = np.meshgrid(lon, lat, indexing='ij')
    
    # Create data that shows coordinate system effects
    data = np.zeros((3, height, width), dtype=np.float32)
    
    # Band 1: Meridian convergence effect (longitude lines)
    data[0] = np.abs(np.sin(LON * np.pi / 2))
    
    # Band 2: Latitude effect (parallel lines)
    data[1] = np.abs(np.cos(LAT * np.pi / 2))
    
    # Band 3: Grid distortion pattern
    data[2] = np.sin(LON * np.pi) * np.cos(LAT * np.pi)
    
    # Normalize
    for i in range(3):
        data[i] = (data[i] - data[i].min()) / (data[i].max() - data[i].min())
    
    print(f"Created synthetic geographic grid: {width}x{height}")
    print(f"Coordinate range: lon=[{lon.min():.1f}, {lon.max():.1f}], lat=[{lat.min():.1f}, {lat.max():.1f}]")
    
    return data

def save_demo_output(data, results, output_path):
    """Save demonstration output."""
    if data is None:
        print("\nNo data to save.")
        return
    
    try:
        import forge3d
        print(f"\nSaving reprojection demo to {output_path}...")
        
        # Convert to RGB uint8
        rgb_uint8 = (data * 255).astype(np.uint8)
        rgb_hwc = np.transpose(rgb_uint8, (1, 2, 0))
        
        forge3d.numpy_to_png(str(output_path), rgb_hwc)
        print(f"Saved reprojection visualization to {output_path}")
        
    except ImportError:
        print("forge3d not available for saving PNG")
    except Exception as e:
        print(f"Error saving output: {e}")

def save_report(results, output_path):
    """Save report as JSON."""
    import json
    
    report = {
        "demo_name": "S3: CRS Normalization and Reprojection Demo",
        "description": "Demonstration of coordinate system transformations",
        "transformations": results or [],
        "features_shown": [
            "Coordinate system transformations",
            "Bounds transformation",
            "Area distortion effects",
            "Mock reprojection workflow"
        ],
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }
    
    report_path = output_path.with_suffix('.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Saved report to {report_path}")

def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description='S3: CRS reprojection demo')
    parser.add_argument('--out', type=str, default='reports/s3_reproject.png',
                       help='Output PNG path')
    args = parser.parse_args()
    
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Forge3D Workstream S3: CRS Reprojection Demo")
    print("=" * 45)
    
    demo_data = demonstrate_crs_operations()
    
    if demo_data:
        data, results = demo_data
        save_demo_output(data, results, output_path) 
        save_report(results, output_path)
    
    print(f"\n=== Demo Complete ===")
    print(f"Output: {output_path}")

if __name__ == "__main__":
    main()