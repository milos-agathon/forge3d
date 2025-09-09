#!/usr/bin/env python3
"""
Advanced Example 3: HDR Tone Mapping Comparison

Demonstrates different HDR tone mapping operators and their visual effects.
Compares ACES, Reinhard, and other tone mapping algorithms on high dynamic range content.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

def generate_hdr_test_scene(width: int, height: int) -> np.ndarray:
    """Generate HDR test scene with wide luminance range."""
    
    # Create coordinate grid
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Distance from center
    R = np.sqrt(X**2 + Y**2)
    
    # Create HDR scene with multiple light sources
    scene = np.zeros((height, width, 3), dtype=np.float32)
    
    # Very bright sun disk
    sun_mask = R < 0.1
    scene[sun_mask] = [100.0, 95.0, 80.0]  # Very bright yellow sun
    
    # Bright sky around sun
    sky_mask = (R >= 0.1) & (R < 0.3)
    sky_brightness = np.exp(-(R[sky_mask] - 0.1) * 10) * 15.0
    scene[sky_mask] = np.column_stack([
        sky_brightness * 0.8,  # R
        sky_brightness * 0.9,  # G  
        sky_brightness * 1.0,  # B
    ])
    
    # Medium brightness clouds
    cloud_pattern = np.sin(X * 8) * np.cos(Y * 6) + np.sin(X * 12) * np.cos(Y * 4)
    cloud_mask = (cloud_pattern > 0.3) & (R > 0.3)
    scene[cloud_mask] = [2.0, 2.2, 2.5]  # Bright clouds
    
    # Darker sky background
    background_mask = (R > 0.3) & (~cloud_mask)
    scene[background_mask] = [0.3, 0.5, 0.8]  # Blue sky
    
    # Add some bright specular highlights
    highlight_centers = [(0.5, -0.6), (-0.7, 0.4), (0.3, 0.8)]
    for cx, cy in highlight_centers:
        highlight_R = np.sqrt((X - cx)**2 + (Y - cy)**2)
        highlight_mask = highlight_R < 0.05
        scene[highlight_mask] = [50.0, 45.0, 40.0]  # Bright specular
        
        # Softer glow around highlights
        glow_mask = (highlight_R >= 0.05) & (highlight_R < 0.15)
        glow_strength = np.exp(-(highlight_R[glow_mask] - 0.05) * 20) * 8.0
        scene[glow_mask] = np.column_stack([
            glow_strength * 0.9,
            glow_strength * 0.8,
            glow_strength * 0.7,
        ])
    
    # Add some darker shadow regions
    shadow_pattern = np.sin(X * 3 + 1.5) * np.cos(Y * 4 + 0.8)
    shadow_mask = (shadow_pattern < -0.5) & (R > 0.4)
    scene[shadow_mask] *= 0.1  # Very dark shadows
    
    return scene


def apply_aces_tonemap(hdr_image: np.ndarray) -> np.ndarray:
    """Apply ACES tone mapping (Academy Color Encoding System)."""
    
    def aces_fitted(x):
        # ACES fitted curve parameters
        a = 2.51
        b = 0.03  
        c = 2.43
        d = 0.59
        e = 0.14
        
        return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)
    
    # Convert to linear space and apply tone mapping
    linear = np.power(np.clip(hdr_image / 100.0, 0.0, 1000.0), 1.0)  # Scale HDR range
    tonemapped = aces_fitted(linear)
    
    # Apply gamma correction
    gamma_corrected = np.power(tonemapped, 1.0 / 2.2)
    
    return (gamma_corrected * 255.0).astype(np.uint8)


def apply_reinhard_tonemap(hdr_image: np.ndarray, white_point: float = 4.0) -> np.ndarray:
    """Apply Reinhard tone mapping."""
    
    # Scale HDR values
    scaled = hdr_image / 20.0
    
    # Reinhard tone mapping formula
    numerator = scaled * (1.0 + scaled / (white_point * white_point))
    denominator = 1.0 + scaled
    tonemapped = numerator / denominator
    
    # Apply gamma correction
    gamma_corrected = np.power(np.clip(tonemapped, 0.0, 1.0), 1.0 / 2.2)
    
    return (gamma_corrected * 255.0).astype(np.uint8)


def apply_exposure_tonemap(hdr_image: np.ndarray, exposure: float = -2.0) -> np.ndarray:
    """Apply simple exposure-based tone mapping."""
    
    # Apply exposure adjustment
    exposed = hdr_image * np.power(2.0, exposure)
    
    # Simple gamma correction with clipping
    gamma_corrected = np.power(np.clip(exposed / 10.0, 0.0, 1.0), 1.0 / 2.2)
    
    return (gamma_corrected * 255.0).astype(np.uint8)


def apply_filmic_tonemap(hdr_image: np.ndarray) -> np.ndarray:
    """Apply filmic tone mapping curve."""
    
    def filmic_curve(x):
        # Filmic tone mapping parameters
        shoulder_strength = 0.22
        linear_strength = 0.30
        linear_angle = 0.10
        toe_strength = 0.20
        toe_numerator = 0.01
        toe_denominator = 0.30
        
        # Simplified filmic curve
        x = np.clip(x, 0.0, 100.0)
        return ((x * (shoulder_strength * x + linear_angle * linear_strength) + 
                toe_strength * toe_numerator) / 
               (x * (shoulder_strength * x + linear_strength) + toe_strength * toe_denominator)) - toe_numerator / toe_denominator
    
    # Normalize and apply curve
    normalized = hdr_image / 25.0
    tonemapped = filmic_curve(normalized)
    
    # Apply gamma
    gamma_corrected = np.power(np.clip(tonemapped, 0.0, 1.0), 1.0 / 2.2)
    
    return (gamma_corrected * 255.0).astype(np.uint8)


def create_comparison_grid(hdr_scene: np.ndarray) -> np.ndarray:
    """Create comparison grid of different tone mapping operators."""
    
    # Apply different tone mapping operators
    print("Applying tone mapping operators...")
    
    aces_result = apply_aces_tonemap(hdr_scene)
    reinhard_result = apply_reinhard_tonemap(hdr_scene)
    exposure_result = apply_exposure_tonemap(hdr_scene)
    filmic_result = apply_filmic_tonemap(hdr_scene)
    
    # Create 2x2 grid
    h, w = hdr_scene.shape[:2]
    grid = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    
    # Top row
    grid[:h, :w] = aces_result
    grid[:h, w:] = reinhard_result
    
    # Bottom row
    grid[h:, :w] = exposure_result
    grid[h:, w:] = filmic_result
    
    return grid


def add_labels_to_grid(grid: np.ndarray) -> np.ndarray:
    """Add text labels to identify each tone mapping method."""
    # This would add text labels in a real implementation
    # For now, we'll just return the grid as-is
    return grid


def main():
    """Main example execution."""
    print("HDR Tone Mapping Comparison")
    print("==========================")
    
    # Create output directory
    out_dir = Path(__file__).parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    
    try:
        import forge3d as f3d
        
        # Configuration
        scene_width, scene_height = 512, 384
        
        print("Generating HDR test scene...")
        hdr_scene = generate_hdr_test_scene(scene_width, scene_height)
        
        # Analyze HDR content
        min_lum = np.min(hdr_scene)
        max_lum = np.max(hdr_scene)
        mean_lum = np.mean(hdr_scene)
        dynamic_range = max_lum / max(min_lum, 0.001)  # Avoid division by zero
        
        print(f"HDR Scene Analysis:")
        print(f"  Luminance range: {min_lum:.3f} to {max_lum:.1f} cd/m²")
        print(f"  Mean luminance: {mean_lum:.2f} cd/m²")
        print(f"  Dynamic range: {dynamic_range:.1f}:1")
        
        # Create tone mapping comparison
        print("Creating tone mapping comparison...")
        comparison_grid = create_comparison_grid(hdr_scene)
        
        # Add labels (simplified)
        labeled_grid = add_labels_to_grid(comparison_grid)
        
        # Save comparison grid
        grid_path = out_dir / "hdr_tonemap_comparison.png"
        f3d.numpy_to_png(str(grid_path), labeled_grid)
        print(f"Saved comparison grid: {grid_path}")
        
        # Save individual tone mapped results
        operators = {
            'aces': apply_aces_tonemap(hdr_scene),
            'reinhard': apply_reinhard_tonemap(hdr_scene),
            'exposure': apply_exposure_tonemap(hdr_scene),
            'filmic': apply_filmic_tonemap(hdr_scene),
        }
        
        individual_paths = {}
        for name, result in operators.items():
            path = out_dir / f"tonemap_{name}.png"
            f3d.numpy_to_png(str(path), result)
            individual_paths[name] = str(path)
            print(f"Saved {name} tone mapping: {path}")
        
        # Create HDR visualization (false color)
        hdr_vis = np.log10(np.clip(hdr_scene + 0.001, 0.001, 1000.0))  # Log scale
        hdr_vis_norm = (hdr_vis - hdr_vis.min()) / (hdr_vis.max() - hdr_vis.min())
        hdr_vis_rgb = (hdr_vis_norm * 255).astype(np.uint8)
        
        hdr_vis_path = out_dir / "hdr_scene_log_visualization.png"
        f3d.numpy_to_png(str(hdr_vis_path), hdr_vis_rgb)
        print(f"Saved HDR visualization: {hdr_vis_path}")
        
        # Generate metrics
        metrics = {
            'scene_size': [scene_width, scene_height],
            'hdr_analysis': {
                'min_luminance': float(min_lum),
                'max_luminance': float(max_lum),
                'mean_luminance': float(mean_lum),
                'dynamic_range_ratio': float(dynamic_range),
            },
            'tone_operators': list(operators.keys()),
            'outputs': {
                'comparison_grid': str(grid_path),
                'hdr_visualization': str(hdr_vis_path),
                **individual_paths,
            }
        }
        
        print("\nTone Mapping Results:")
        for key, value in metrics.items():
            if key != 'outputs':
                print(f"  {key}: {value}")
        
        # Save metrics
        import json
        metrics_path = out_dir / "hdr_tonemap_metrics.json"
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