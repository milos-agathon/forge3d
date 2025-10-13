#!/usr/bin/env python3
"""
Demo: Raytracing from Interactive 3D Viewer Camera

This example demonstrates how to:
1. Get camera state from the running 3D viewer
2. Use that camera state to raytrace a high-quality image
3. Compare rasterization vs raytracing from the same viewpoint

Usage:
    Terminal 1: python examples/geopandas_demo.py --viewer-3d --water
    Terminal 2: python examples/raytrace_from_viewer.py --dem assets/Gore_Range_Albers_1m.tif
"""

import argparse
import sys
from pathlib import Path
import numpy as np

try:
    import forge3d as f3d
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
    import forge3d as f3d


def raytrace_from_viewer(dem_path: Path, output: str = "raytraced_view.png", width: int = 1920, height: int = 1080):
    """
    Raytrace the terrain from the current viewer camera position.
    
    Args:
        dem_path: Path to DEM file
        output: Output PNG file path
        width: Image width
        height: Image height
    """
    print("=" * 60)
    print("Raytracing from Viewer Camera")
    print("=" * 60)
    
    # Get current camera state from viewer
    print("\n1. Getting camera state from viewer...")
    try:
        cam = f3d.get_camera()
    except Exception as e:
        print(f"Error: Could not get camera state. Is the viewer running?")
        print(f"   {e}")
        return
    
    print(f"   Camera position: [{cam['eye'][0]:.1f}, {cam['eye'][1]:.1f}, {cam['eye'][2]:.1f}]")
    print(f"   Looking at: [{cam['target'][0]:.1f}, {cam['target'][1]:.1f}, {cam['target'][2]:.1f}]")
    print(f"   Distance: {cam['distance']:.1f}")
    print(f"   Angles: theta={cam['theta']:.1f}°, phi={cam['phi']:.1f}°")
    print(f"   FOV: {cam['fov']:.1f}°")
    
    # Load DEM data
    print(f"\n2. Loading DEM: {dem_path}")
    try:
        import rasterio
        with rasterio.open(dem_path) as src:
            dem = src.read(1)
            transform = src.transform
            print(f"   DEM shape: {dem.shape}")
            print(f"   Resolution: {abs(transform.a):.2f} x {abs(transform.e):.2f} meters")
    except ImportError:
        print("Error: rasterio not installed. Install with: pip install rasterio")
        return
    except Exception as e:
        print(f"Error loading DEM: {e}")
        return
    
    # TODO: Call raytrace function with camera parameters
    print(f"\n3. Raytracing {width}x{height} image...")
    print("   Note: Full raytracing integration pending")
    print("   This would call:")
    print(f"     f3d.raytrace(dem, eye={cam['eye']}, target={cam['target']},")
    print(f"                  fov={cam['fov']}, width={width}, height={height})")
    
    # For now, take a snapshot from the viewer
    print(f"\n4. Taking snapshot from viewer instead...")
    f3d.snapshot(output, width=width, height=height)
    print(f"   Saved: {output}")
    
    print("\n✓ Complete!")
    print(f"\nTo compare rasterization vs raytracing:")
    print("  1. Current implementation uses GPU rasterization (fast)")
    print("  2. Full raytracing would provide:")
    print("     - Physically accurate lighting")
    print("     - Shadows and ambient occlusion")
    print("     - Reflections on water surfaces")
    print("     - Higher quality but slower rendering")


def demo_camera_control_and_raytrace(dem_path: Path):
    """
    Demo: Set camera angles and raytrace multiple views.
    """
    print("=" * 60)
    print("Multi-View Raytracing Demo")
    print("=" * 60)
    
    views = [
        {"name": "North View", "distance": 2000, "theta": 0, "phi": 30},
        {"name": "East View", "distance": 2000, "theta": 90, "phi": 30},
        {"name": "South View", "distance": 2000, "theta": 180, "phi": 30},
        {"name": "West View", "distance": 2000, "theta": 270, "phi": 30},
        {"name": "Top View", "distance": 3000, "theta": 0, "phi": 85},
    ]
    
    for i, view in enumerate(views):
        print(f"\n{i+1}. {view['name']}")
        print(f"   Setting camera: distance={view['distance']}, theta={view['theta']}°, phi={view['phi']}°")
        
        f3d.set_camera(distance=view['distance'], theta=view['theta'], phi=view['phi'])
        
        import time
        time.sleep(1)  # Wait for camera to update
        
        output_name = f"view_{i+1}_{view['name'].lower().replace(' ', '_')}.png"
        raytrace_from_viewer(dem_path, output=output_name, width=1280, height=720)
    
    print("\n✓ All views rendered!")
    print("  Generated: view_1_north_view.png through view_5_top_view.png")


def main():
    parser = argparse.ArgumentParser(description="Raytrace from Viewer Camera")
    parser.add_argument("--dem", type=Path, help="DEM file path")
    parser.add_argument("--output", default="raytraced_view.png", help="Output PNG file")
    parser.add_argument("--width", type=int, default=1920, help="Image width")
    parser.add_argument("--height", type=int, default=1080, help="Image height")
    parser.add_argument("--multi-view", action="store_true", help="Render multiple views")
    
    args = parser.parse_args()
    
    if not args.dem:
        parser.print_help()
        print("\nExample:")
        print("  Terminal 1: python examples/geopandas_demo.py --viewer-3d --water")
        print("  Terminal 2: python examples/raytrace_from_viewer.py --dem assets/Gore_Range_Albers_1m.tif")
        return
    
    if args.multi_view:
        demo_camera_control_and_raytrace(args.dem)
    else:
        raytrace_from_viewer(args.dem, output=args.output, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
