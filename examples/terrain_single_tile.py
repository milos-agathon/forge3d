#!/usr/bin/env python3
"""
Single-tile terrain rendering example for forge3d.

Creates a procedural heightmap and renders it as a shaded terrain image.
Designed for new developers to verify installation in <10 minutes.
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import numpy as np

# Add repo root to sys.path for forge3d import
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "python"))

try:
    import forge3d as f3d
except ImportError as e:
    print(f"Failed to import forge3d: {e}")
    print("Make sure the package is installed or run 'maturin develop' first")
    sys.exit(1)


def create_procedural_heightmap(size: int) -> np.ndarray:
    """Create a procedural heightmap using sine/cosine waves."""
    x = np.linspace(-2, 2, size, dtype=np.float32)
    y = np.linspace(-2, 2, size, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    
    # Create interesting terrain with multiple frequency components
    Z = (0.3 * np.sin(1.5 * X) * np.cos(1.2 * Y) +
         0.2 * np.sin(3.0 * X + 1.0) +
         0.15 * np.cos(2.5 * Y - 0.5) +
         0.1 * np.sin(4.0 * X) * np.sin(4.0 * Y))
    
    return np.ascontiguousarray(Z, dtype=np.float32)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Render a single terrain tile with procedural heightmap"
    )
    parser.add_argument(
        "--size", 
        type=int, 
        default=128,
        help="Heightmap resolution (default: 128)"
    )
    parser.add_argument(
        "--colormap", 
        type=str, 
        default="viridis",
        help="Colormap name (default: viridis)"
    )
    parser.add_argument(
        "--out", 
        type=str, 
        default="terrain_single_tile.png",
        help="Output PNG file (default: terrain_single_tile.png)"
    )
    parser.add_argument(
        "--render-size",
        type=int,
        default=512,
        help="Output image size (default: 512x512)"
    )
    
    args = parser.parse_args()
    
    print(f"forge3d terrain example starting...")
    print(f"Heightmap size: {args.size}x{args.size}")
    print(f"Colormap: {args.colormap}")
    print(f"Output: {args.out}")
    
    start_time = time.time()
    
    try:
        # Create procedural heightmap
        print("Creating procedural heightmap...")
        heightmap = create_procedural_heightmap(args.size)
        
        # Create scene
        print("Initializing renderer...")
        scene = f3d.Scene(
            width=args.render_size,
            height=args.render_size,
            grid=args.size,
            colormap=args.colormap
        )
        
        # Set terrain data
        print("Uploading terrain data...")
        scene.set_height_from_r32f(heightmap)
        
        # Set camera to show the terrain nicely
        print("Setting up camera...")
        scene.set_camera_look_at(
            eye=(3.0, 2.5, 3.0),       # Camera position
            target=(0.0, 0.0, 0.0),   # Look at center
            up=(0.0, 1.0, 0.0),       # Up vector
            fovy_deg=45.0,            # Field of view
            znear=0.1,                # Near clipping plane
            zfar=100.0                # Far clipping plane
        )
        
        # Render to file
        print(f"Rendering to {args.out}...")
        scene.render_png(Path(args.out))
        
        # Also get the RGBA data to verify dimensions
        rgba = scene.render_rgba()
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"Success! Rendered {rgba.shape[1]}x{rgba.shape[0]} image in {elapsed:.2f} seconds")
        print(f"Output saved to: {Path(args.out).absolute()}")
        print(f"Image shows a procedural terrain with {args.colormap} colormap")
        
        return 0
        
    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Error after {elapsed:.2f} seconds: {e}")
        return 1


if __name__ == "__main__":
    exit(main())