#!/usr/bin/env python3
"""
Advanced Example 6: Normal Mapping on Terrain

Demonstrates surface normal calculation and normal mapping techniques on terrain.
Shows height-based normal generation and surface detail enhancement.
"""

import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def generate_detailed_terrain(size: int = 256) -> np.ndarray:
    """Generate terrain with fine surface detail suitable for normal mapping."""
    x = np.linspace(-4, 4, size)
    y = np.linspace(-4, 4, size)
    X, Y = np.meshgrid(x, y)
    
    # Base terrain with large-scale features
    terrain = np.zeros_like(X)
    
    # Large mountains
    terrain += 1.2 * np.exp(-((X-1)**2 + (Y-1)**2) / 2)
    terrain += 0.8 * np.exp(-((X+1.5)**2 + (Y+0.5)**2) / 3)
    
    # Medium hills and ridges
    terrain += 0.6 * np.sin(X * 2.5) * np.cos(Y * 1.8)
    terrain += 0.4 * np.sin(X * 3.2 + 1.5) * np.cos(Y * 2.7 + 0.8)
    
    # Fine-scale surface detail for normal mapping
    detail_scale = 16
    terrain += 0.15 * np.sin(X * detail_scale) * np.cos(Y * detail_scale)
    terrain += 0.10 * np.sin(X * detail_scale * 1.3 + 2.1) * np.cos(Y * detail_scale * 0.9 + 1.2)
    terrain += 0.08 * np.sin(X * detail_scale * 2.1) * np.cos(Y * detail_scale * 1.8)
    
    # Add micro-detail noise
    np.random.seed(42)  # Reproducible
    terrain += 0.05 * np.random.random(terrain.shape)
    
    # Ensure positive elevations
    terrain = terrain - terrain.min() + 0.1
    
    return terrain.astype(np.float32)


def calculate_surface_normals(heightfield: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    """Calculate surface normals from heightfield data using finite differences."""
    height, width = heightfield.shape
    normals = np.zeros((height, width, 3), dtype=np.float32)
    
    # Calculate gradients using Sobel operators for better noise resistance
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.0
    
    # Pad heightfield for convolution
    padded = np.pad(heightfield, 1, mode='edge')
    
    # Calculate gradients
    grad_x = np.zeros_like(heightfield)
    grad_y = np.zeros_like(heightfield)
    
    for i in range(height):
        for j in range(width):
            # Apply Sobel operators
            patch = padded[i:i+3, j:j+3]
            grad_x[i, j] = np.sum(patch * sobel_x)
            grad_y[i, j] = np.sum(patch * sobel_y)
    
    # Scale gradients by spacing
    grad_x /= spacing
    grad_y /= spacing
    
    # Calculate normal vectors
    # Normal = (-dz/dx, -dz/dy, 1) normalized
    normals[:, :, 0] = -grad_x
    normals[:, :, 1] = -grad_y  
    normals[:, :, 2] = 1.0
    
    # Normalize to unit vectors
    magnitudes = np.sqrt(np.sum(normals**2, axis=2, keepdims=True))
    normals = normals / (magnitudes + 1e-8)  # Avoid division by zero
    
    return normals


def create_normal_map_texture(normals: np.ndarray) -> np.ndarray:
    """Convert surface normals to normal map texture (tangent space)."""
    # Convert normals from [-1, 1] to [0, 255] for texture storage
    # Normal maps store: R=X, G=Y, B=Z components
    normal_texture = np.zeros((*normals.shape[:2], 4), dtype=np.uint8)
    
    # Map [-1, 1] to [0, 255]
    normal_texture[:, :, 0] = np.clip((normals[:, :, 0] + 1.0) * 127.5, 0, 255)  # R = X
    normal_texture[:, :, 1] = np.clip((normals[:, :, 1] + 1.0) * 127.5, 0, 255)  # G = Y
    normal_texture[:, :, 2] = np.clip((normals[:, :, 2] + 1.0) * 127.5, 0, 255)  # B = Z
    normal_texture[:, :, 3] = 255  # Alpha = 1.0
    
    return normal_texture


def apply_normal_mapping_lighting(heightfield: np.ndarray, normals: np.ndarray, 
                                light_dir: tuple = (-0.5, -0.7, -0.5)) -> np.ndarray:
    """Apply simple lighting using calculated normals for demonstration."""
    # Normalize light direction
    light_dir = np.array(light_dir, dtype=np.float32)
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    # Calculate dot product for diffuse lighting
    dot_product = np.sum(normals * light_dir.reshape(1, 1, 3), axis=2)
    
    # Clamp to [0, 1] for valid lighting
    lighting = np.clip(dot_product, 0.0, 1.0)
    
    # Apply ambient + diffuse lighting
    ambient = 0.3
    diffuse_strength = 0.7
    final_lighting = ambient + diffuse_strength * lighting
    
    # Apply lighting to height-based coloring
    height_norm = (heightfield - heightfield.min()) / (heightfield.max() - heightfield.min())
    
    # Create RGB image with lighting applied
    lit_terrain = np.zeros((*heightfield.shape, 3), dtype=np.float32)
    lit_terrain[:, :, 0] = height_norm * 0.8 + 0.2  # Reddish rocks
    lit_terrain[:, :, 1] = height_norm * 0.6 + 0.3  # Greenish vegetation 
    lit_terrain[:, :, 2] = height_norm * 0.4 + 0.2  # Bluish highlights
    
    # Apply lighting
    lit_terrain = lit_terrain * final_lighting[:, :, np.newaxis]
    
    # Convert to 8-bit
    lit_terrain = np.clip(lit_terrain * 255, 0, 255).astype(np.uint8)
    
    # Add alpha channel
    result = np.zeros((*heightfield.shape, 4), dtype=np.uint8)
    result[:, :, :3] = lit_terrain
    result[:, :, 3] = 255
    
    return result


def create_comparison_renders(heightfield: np.ndarray, normals: np.ndarray) -> dict:
    """Create comparison renders showing normal mapping effects."""
    renders = {}
    
    # 1. Raw heightfield visualization
    height_norm = (heightfield - heightfield.min()) / (heightfield.max() - heightfield.min())
    raw_vis = (height_norm * 255).astype(np.uint8)
    renders['heightfield'] = raw_vis
    
    # 2. Normal map visualization
    normal_texture = create_normal_map_texture(normals)
    renders['normal_map'] = normal_texture
    
    # 3. Lighting without normal mapping (flat shading)
    flat_lighting = apply_normal_mapping_lighting(heightfield, 
                                                np.stack([np.zeros_like(heightfield),
                                                         np.zeros_like(heightfield), 
                                                         np.ones_like(heightfield)], axis=2))
    renders['flat_lit'] = flat_lighting
    
    # 4. Lighting with normal mapping
    normal_lit = apply_normal_mapping_lighting(heightfield, normals)
    renders['normal_lit'] = normal_lit
    
    # 5. Normal vectors as RGB (for debugging)
    normal_debug = create_normal_map_texture(normals)
    renders['normal_debug'] = normal_debug
    
    return renders


def analyze_surface_properties(heightfield: np.ndarray, normals: np.ndarray) -> dict:
    """Analyze surface properties from heightfield and normals."""
    
    # Calculate slopes (in degrees)
    slopes = np.arccos(np.clip(normals[:, :, 2], 0, 1)) * 180.0 / np.pi
    
    # Calculate surface roughness (standard deviation of slopes in local neighborhoods)
    from scipy.ndimage import uniform_filter
    
    try:
        # 5x5 neighborhood roughness
        slope_mean = uniform_filter(slopes, size=5)
        slope_sq_mean = uniform_filter(slopes**2, size=5)
        roughness = np.sqrt(np.maximum(0, slope_sq_mean - slope_mean**2))
    except ImportError:
        # Fallback without scipy
        roughness = np.std(slopes) * np.ones_like(slopes)
    
    # Calculate curvature approximation
    grad_x = np.gradient(heightfield, axis=1)
    grad_y = np.gradient(heightfield, axis=0)
    grad_xx = np.gradient(grad_x, axis=1)
    grad_yy = np.gradient(grad_y, axis=0)
    grad_xy = np.gradient(grad_x, axis=0)
    
    # Mean curvature approximation
    curvature = 0.5 * (grad_xx + grad_yy)
    
    analysis = {
        'elevation_stats': {
            'min': float(heightfield.min()),
            'max': float(heightfield.max()),
            'mean': float(heightfield.mean()),
            'std': float(heightfield.std()),
        },
        'slope_stats': {
            'min_degrees': float(slopes.min()),
            'max_degrees': float(slopes.max()),
            'mean_degrees': float(slopes.mean()),
            'std_degrees': float(slopes.std()),
        },
        'roughness_stats': {
            'min': float(roughness.min()),
            'max': float(roughness.max()),
            'mean': float(roughness.mean()),
        },
        'curvature_stats': {
            'min': float(curvature.min()),
            'max': float(curvature.max()),
            'mean': float(curvature.mean()),
        }
    }
    
    return analysis


def main():
    """Main example execution."""
    print("Normal Mapping on Terrain")
    print("========================")
    
    out_dir = Path(__file__).parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    
    try:
        import forge3d as f3d
        
        # Configuration
        terrain_size = 512
        spacing = 2.0  # Units between height samples
        
        print(f"Generating detailed terrain ({terrain_size}x{terrain_size})...")
        heightfield = generate_detailed_terrain(terrain_size)
        
        print("Calculating surface normals...")
        normals = calculate_surface_normals(heightfield, spacing)
        
        print("Creating comparison renders...")
        renders = create_comparison_renders(heightfield, normals)
        
        # Save all renders
        saved_paths = {}
        for name, image in renders.items():
            path = out_dir / f"normal_mapping_{name}.png"
            f3d.numpy_to_png(str(path), image)
            saved_paths[name] = str(path)
            print(f"Saved {name}: {path}")
        
        # Create side-by-side comparison
        try:
            flat_lit = renders['flat_lit']
            normal_lit = renders['normal_lit']
            height, width = flat_lit.shape[:2]
            
            comparison = np.zeros((height, width * 2, 4), dtype=np.uint8)
            comparison[:, :width] = flat_lit
            comparison[:, width:] = normal_lit
            
            comp_path = out_dir / "normal_mapping_comparison.png"
            f3d.numpy_to_png(str(comp_path), comparison)
            saved_paths['comparison'] = str(comp_path)
            print(f"Saved comparison: {comp_path}")
            
        except Exception as e:
            print(f"Comparison creation failed: {e}")
        
        # Surface analysis
        print("Analyzing surface properties...")
        surface_analysis = analyze_surface_properties(heightfield, normals)
        
        # Try 3D terrain rendering if available
        try:
            scene = f3d.Scene(800, 600)
            scene.set_height_data(heightfield, spacing=spacing, exaggeration=10.0)
            
            # Position camera for good view
            scene.set_camera(
                position=(terrain_size * spacing * 0.8, terrain_size * spacing * 0.4, terrain_size * spacing * 0.8),
                target=(terrain_size * spacing * 0.5, heightfield.mean() * 10.0, terrain_size * spacing * 0.5),
                up=(0.0, 1.0, 0.0)
            )
            
            terrain_3d = scene.render_terrain_rgba()
            terrain_3d_path = out_dir / "normal_mapping_terrain_3d.png"
            f3d.numpy_to_png(str(terrain_3d_path), terrain_3d)
            saved_paths['terrain_3d'] = str(terrain_3d_path)
            print(f"Saved 3D terrain: {terrain_3d_path}")
            
        except Exception as e:
            print(f"3D terrain rendering failed: {e}")
        
        # Generate comprehensive metrics
        metrics = {
            'terrain_size': terrain_size,
            'spacing': spacing,
            'surface_analysis': surface_analysis,
            'normal_calculation_method': 'sobel_operators',
            'lighting_model': 'lambert_diffuse',
            'outputs': saved_paths,
        }
        
        print("\nSurface Analysis Results:")
        print(f"  Elevation range: {surface_analysis['elevation_stats']['min']:.2f} to {surface_analysis['elevation_stats']['max']:.2f}")
        print(f"  Mean slope: {surface_analysis['slope_stats']['mean_degrees']:.1f}° (max: {surface_analysis['slope_stats']['max_degrees']:.1f}°)")
        print(f"  Surface roughness: {surface_analysis['roughness_stats']['mean']:.2f}")
        print(f"  Normal mapping renders: {len(renders)} variations")
        
        # Save metrics
        import json
        metrics_path = out_dir / "normal_mapping_metrics.json"
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