#!/usr/bin/env python3
"""
Advanced Example 2: Contour Overlay Visualization

Demonstrates contour line generation and overlay rendering on terrain.
Shows vector graphics integration with terrain data for topographic visualization.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path for development  
sys.path.insert(0, str(Path(__file__).parent.parent))

def generate_test_terrain(size: int = 256) -> np.ndarray:
    """Generate test terrain with interesting contour features."""
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)
    
    # Create terrain with multiple peaks and valleys
    terrain = np.zeros_like(X)
    
    # Main mountain
    terrain += 2.0 * np.exp(-(X**2 + Y**2))
    
    # Secondary peak
    terrain += 1.2 * np.exp(-((X-1.5)**2 + (Y+1.0)**2))
    
    # Valley
    terrain += 0.8 * np.exp(-((X+1.0)**2 + (Y-1.2)**2))
    
    # Ridge line
    terrain += 0.6 * np.exp(-(0.1*X + 0.3*Y)**2)
    
    # Add some noise for texture
    noise = 0.1 * np.random.random(terrain.shape)
    terrain += noise
    
    return terrain.astype(np.float32)


def generate_contour_lines(terrain: np.ndarray, levels: np.ndarray) -> list:
    """Generate contour lines from terrain elevation data."""
    try:
        from scipy.ndimage import find_contours
        import matplotlib.path as mpath
        
        contours = []
        height, width = terrain.shape
        
        for level in levels:
            try:
                # Find contours at this elevation level
                contour_paths = find_contours(terrain, level)
                
                for path in contour_paths:
                    if len(path) < 3:  # Skip degenerate contours
                        continue
                    
                    # Convert contour coordinates to world space
                    # Scale from array indices to world coordinates
                    world_coords = []
                    for point in path:
                        # Convert from array coordinates to normalized [0,1] space
                        x = point[1] / width   # column -> x
                        y = point[0] / height  # row -> y  
                        world_coords.append([x, y])
                    
                    contours.append({
                        'level': level,
                        'coords': np.array(world_coords, dtype=np.float32),
                        'is_major': level % 0.5 == 0,  # Major contours every 0.5 units
                    })
                    
            except Exception as e:
                print(f"Warning: Failed to generate contour at level {level}: {e}")
                continue
        
        return contours
        
    except ImportError:
        print("scipy not available, generating simplified contour lines")
        return generate_simplified_contours(terrain, levels)


def generate_simplified_contours(terrain: np.ndarray, levels: np.ndarray) -> list:
    """Simplified contour generation without scipy."""
    contours = []
    height, width = terrain.shape
    
    # Simple grid-based contour approximation
    for level in levels:
        coords = []
        
        # Find approximate contour points by checking elevation differences
        for i in range(1, height-1):
            for j in range(1, width-1):
                # Check if this point crosses the contour level
                neighbors = [
                    terrain[i-1, j], terrain[i+1, j],
                    terrain[i, j-1], terrain[i, j+1]
                ]
                
                if (terrain[i, j] <= level <= max(neighbors) or 
                    terrain[i, j] >= level >= min(neighbors)):
                    # This point is near the contour
                    x = j / width
                    y = i / height
                    coords.append([x, y])
        
        if len(coords) > 5:  # Only keep contours with enough points
            contours.append({
                'level': level,
                'coords': np.array(coords, dtype=np.float32),
                'is_major': level % 0.5 == 0,
            })
    
    return contours


def render_contour_overlay(terrain: np.ndarray, contours: list, width: int = 800, height: int = 600):
    """Render terrain with contour line overlay."""
    try:
        import forge3d as f3d
        
        # Create scene with terrain
        scene = f3d.Scene(width, height)
        scene.set_height_data(terrain, spacing=1.0, exaggeration=3.0)
        
        # Set camera for good overview
        scene.set_camera(
            position=(terrain.shape[1] * 0.8, terrain.shape[0] * 0.4, terrain.shape[1] * 0.8),
            target=(terrain.shape[1] * 0.5, 0.0, terrain.shape[0] * 0.5),
            up=(0.0, 1.0, 0.0)
        )
        
        # Render base terrain
        terrain_image = scene.render_terrain_rgba()
        
        # Add contour lines as vector graphics
        try:
            # Clear any existing vectors
            f3d.clear_vectors_py()
            
            # Add contour lines with different styles
            for contour in contours:
                if len(contour['coords']) < 2:
                    continue
                
                # Scale coordinates to render size
                scaled_coords = contour['coords'] * [width, height]
                
                # Style based on contour type
                if contour['is_major']:
                    # Major contours: thicker, darker
                    color = np.array([[0.2, 0.2, 0.8, 0.9]], dtype=np.float32)  # Blue
                    width_px = 2.5
                else:
                    # Minor contours: thinner, lighter
                    color = np.array([[0.4, 0.4, 0.9, 0.7]], dtype=np.float32)  # Light blue
                    width_px = 1.0
                
                # Add contour as polyline
                try:
                    line_coords = scaled_coords.reshape(1, -1, 2)
                    widths = np.array([width_px], dtype=np.float32)
                    
                    f3d.add_lines_py(line_coords, colors=color, widths=widths)
                    
                except Exception as e:
                    print(f"Warning: Failed to add contour line: {e}")
                    continue
            
            # Render scene with vector overlay
            overlay_image = scene.render_terrain_rgba()  # This should include vectors
            
            return overlay_image, terrain_image
            
        except Exception as e:
            print(f"Vector overlay failed: {e}")
            return terrain_image, terrain_image
        
    except Exception as e:
        print(f"Rendering failed: {e}")
        # Return dummy images
        return (np.zeros((height, width, 4), dtype=np.uint8),
                np.zeros((height, width, 4), dtype=np.uint8))


def main():
    """Main example execution."""
    print("Contour Overlay Demonstration")
    print("============================")
    
    # Create output directory
    out_dir = Path(__file__).parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    
    try:
        import forge3d as f3d
        
        # Configuration
        terrain_size = 256
        render_width, render_height = 800, 600
        
        print("Generating test terrain...")
        terrain = generate_test_terrain(terrain_size)
        
        # Define contour levels
        min_elev, max_elev = terrain.min(), terrain.max()
        print(f"Terrain elevation range: {min_elev:.2f} to {max_elev:.2f}")
        
        # Generate contour levels every 0.1 units
        contour_levels = np.arange(
            np.ceil(min_elev * 10) / 10,  # Round up to nearest 0.1
            np.floor(max_elev * 10) / 10,  # Round down to nearest 0.1
            0.1
        )
        
        print(f"Generating {len(contour_levels)} contour levels...")
        contours = generate_contour_lines(terrain, contour_levels)
        print(f"Generated {len(contours)} contour lines")
        
        # Count major vs minor contours
        major_count = sum(1 for c in contours if c['is_major'])
        minor_count = len(contours) - major_count
        print(f"  {major_count} major contours, {minor_count} minor contours")
        
        print("Rendering terrain with contour overlay...")
        overlay_image, terrain_image = render_contour_overlay(
            terrain, contours, render_width, render_height
        )
        
        # Save outputs
        overlay_path = out_dir / "contour_overlay.png"
        f3d.numpy_to_png(str(overlay_path), overlay_image)
        print(f"Saved contour overlay: {overlay_path}")
        
        terrain_path = out_dir / "terrain_base.png"  
        f3d.numpy_to_png(str(terrain_path), terrain_image)
        print(f"Saved base terrain: {terrain_path}")
        
        # Save elevation data as visualization
        terrain_vis = ((terrain - min_elev) / (max_elev - min_elev) * 255).astype(np.uint8)
        elevation_path = out_dir / "elevation_data.png"
        f3d.numpy_to_png(str(elevation_path), terrain_vis)
        print(f"Saved elevation data: {elevation_path}")
        
        # Generate metrics
        metrics = {
            'terrain_size': terrain_size,
            'render_size': [render_width, render_height],
            'elevation_range': [float(min_elev), float(max_elev)],
            'contour_levels': len(contour_levels),
            'generated_contours': len(contours),
            'major_contours': major_count,
            'minor_contours': minor_count,
            'contour_interval': 0.1,
            'outputs': {
                'overlay': str(overlay_path),
                'terrain': str(terrain_path),
                'elevation': str(elevation_path),
            }
        }
        
        print("\nContour Analysis:")
        for key, value in metrics.items():
            if key != 'outputs':
                print(f"  {key}: {value}")
        
        # Save metrics
        import json
        metrics_path = out_dir / "contour_metrics.json"
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