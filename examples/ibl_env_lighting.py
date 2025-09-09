#!/usr/bin/env python3
"""
Advanced Example 7: IBL Environment Lighting

Demonstrates image-based lighting (IBL) with environment maps for realistic lighting.
Shows environment map loading, irradiance calculation, and spherical harmonics approximation.
"""

import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def generate_procedural_hdri(size: int = 256) -> np.ndarray:
    """Generate procedural HDRI environment map for IBL demonstration."""
    
    # Create spherical coordinates
    width, height = size * 2, size  # 2:1 aspect ratio for equirectangular
    
    # Generate coordinate grids
    u = np.linspace(0, 2 * np.pi, width)  # longitude: 0 to 2π
    v = np.linspace(0, np.pi, height)     # latitude: 0 to π
    U, V = np.meshgrid(u, v)
    
    # Convert to Cartesian coordinates
    x = np.sin(V) * np.cos(U)
    y = np.cos(V)
    z = np.sin(V) * np.sin(U)
    
    # Create HDR environment
    hdr_env = np.zeros((height, width, 3), dtype=np.float32)
    
    # Sky gradient (bright at horizon, darker at zenith)
    sky_intensity = np.exp(-np.abs(y) * 2.0) * 2.0
    sky_color = np.stack([
        sky_intensity * 0.4,  # R: warm horizon
        sky_intensity * 0.6,  # G: sky blue
        sky_intensity * 1.0,  # B: sky blue
    ], axis=2)
    
    # Add bright sun
    sun_dir = np.array([0.5, 0.7, 0.5])  # Sun direction
    sun_dir = sun_dir / np.linalg.norm(sun_dir)
    
    # Calculate dot product with sun direction
    sun_dot = x * sun_dir[0] + y * sun_dir[1] + z * sun_dir[2]
    sun_intensity = np.exp((sun_dot - 0.98) * 50.0)  # Sharp sun disk
    sun_intensity = np.clip(sun_intensity, 0, 1000)  # Very bright sun
    
    sun_color = np.stack([
        sun_intensity * 1000.0,  # Bright yellow sun
        sun_intensity * 950.0,
        sun_intensity * 800.0,
    ], axis=2)
    
    # Add some clouds
    cloud_noise = np.sin(U * 4 + 1.5) * np.cos(V * 3 + 0.8) + \
                  np.sin(U * 7 + 2.1) * np.cos(V * 5 + 1.2)
    cloud_mask = (cloud_noise > 0.3) & (np.abs(y) < 0.8)  # Clouds in sky region
    cloud_intensity = np.exp(-np.abs(y) * 1.5) * 5.0
    
    cloud_color = np.zeros_like(sky_color)
    cloud_color[cloud_mask] = cloud_intensity[cloud_mask, np.newaxis] * np.array([0.9, 0.9, 0.95])
    
    # Ground reflection (simple)
    ground_mask = y < -0.1
    ground_intensity = np.exp(y[ground_mask] * 3.0) * 0.5
    ground_color = np.zeros_like(sky_color)
    ground_color[ground_mask] = ground_intensity[:, np.newaxis] * np.array([0.3, 0.4, 0.2])
    
    # Combine all components
    hdr_env = sky_color + sun_color + cloud_color + ground_color
    
    # Add some ambient light
    hdr_env += 0.1
    
    return hdr_env


def calculate_spherical_harmonics_coefficients(env_map: np.ndarray, num_bands: int = 3) -> np.ndarray:
    """Calculate spherical harmonics coefficients from environment map."""
    
    height, width = env_map.shape[:2]
    
    # Spherical coordinates for each pixel
    u = np.linspace(0, 2 * np.pi, width, endpoint=False)
    v = np.linspace(0, np.pi, height, endpoint=False)
    U, V = np.meshgrid(u, v)
    
    # Convert to Cartesian
    x = np.sin(V) * np.cos(U)
    y = np.cos(V)  
    z = np.sin(V) * np.sin(U)
    
    # Weight by solid angle (sin(theta) for equirectangular)
    weights = np.sin(V)
    
    # Calculate SH basis functions (up to 3rd order, 9 coefficients)
    sh_basis = []
    
    if num_bands >= 1:
        # Band 0 (l=0)
        sh_basis.append(0.282095 * np.ones_like(x))  # Y_0_0
    
    if num_bands >= 2:
        # Band 1 (l=1)
        sh_basis.append(0.488603 * y)           # Y_1_-1
        sh_basis.append(0.488603 * z)           # Y_1_0
        sh_basis.append(0.488603 * x)           # Y_1_1
    
    if num_bands >= 3:
        # Band 2 (l=2)
        sh_basis.append(1.092548 * x * y)       # Y_2_-2
        sh_basis.append(1.092548 * y * z)       # Y_2_-1
        sh_basis.append(0.315392 * (3*z*z - 1)) # Y_2_0
        sh_basis.append(1.092548 * x * z)       # Y_2_1
        sh_basis.append(0.546274 * (x*x - y*y)) # Y_2_2
    
    # Calculate coefficients for each RGB channel
    num_coeffs = len(sh_basis)
    sh_coeffs = np.zeros((num_coeffs, 3), dtype=np.float32)
    
    total_weight = np.sum(weights)
    
    for i, basis in enumerate(sh_basis):
        for channel in range(3):
            # Weighted integral over the sphere
            integral = np.sum(env_map[:, :, channel] * basis * weights)
            sh_coeffs[i, channel] = integral * 4.0 * np.pi / total_weight
    
    return sh_coeffs


def render_diffuse_from_sh(sh_coeffs: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Render diffuse lighting from spherical harmonics coefficients."""
    
    height, width = normals.shape[:2]
    result = np.zeros((height, width, 3), dtype=np.float32)
    
    # Extract normal components
    nx, ny, nz = normals[:, :, 0], normals[:, :, 1], normals[:, :, 2]
    
    # Evaluate SH basis functions at normal directions
    num_coeffs = sh_coeffs.shape[0]
    
    if num_coeffs >= 1:
        # Band 0
        result += sh_coeffs[0] * 0.282095
    
    if num_coeffs >= 4:
        # Band 1  
        result += sh_coeffs[1] * (0.488603 * ny[:, :, np.newaxis])
        result += sh_coeffs[2] * (0.488603 * nz[:, :, np.newaxis])
        result += sh_coeffs[3] * (0.488603 * nx[:, :, np.newaxis])
    
    if num_coeffs >= 9:
        # Band 2
        result += sh_coeffs[4] * (1.092548 * nx[:, :, np.newaxis] * ny[:, :, np.newaxis])
        result += sh_coeffs[5] * (1.092548 * ny[:, :, np.newaxis] * nz[:, :, np.newaxis])
        result += sh_coeffs[6] * (0.315392 * (3*nz[:, :, np.newaxis]**2 - 1))
        result += sh_coeffs[7] * (1.092548 * nx[:, :, np.newaxis] * nz[:, :, np.newaxis])
        result += sh_coeffs[8] * (0.546274 * (nx[:, :, np.newaxis]**2 - ny[:, :, np.newaxis]**2))
    
    return np.clip(result, 0, None)


def create_test_geometry() -> tuple:
    """Create test geometry with known normals for IBL demonstration."""
    
    # Create a sphere's worth of normals for testing
    size = 256
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Create sphere mask
    R = np.sqrt(X**2 + Y**2)
    sphere_mask = R <= 1.0
    
    # Calculate Z coordinate for sphere
    Z = np.zeros_like(X)
    Z[sphere_mask] = np.sqrt(1.0 - R[sphere_mask]**2)
    
    # Create normal vectors (same as positions for unit sphere)
    normals = np.zeros((size, size, 3), dtype=np.float32)
    normals[:, :, 0] = X
    normals[:, :, 1] = Y
    normals[:, :, 2] = Z
    
    # Create additional test geometries
    # Plane with constant upward normal
    plane_normals = np.zeros((size, size, 3), dtype=np.float32)
    plane_normals[:, :, 2] = 1.0  # All pointing up
    
    return normals, plane_normals, sphere_mask


def tone_map_for_display(hdr_image: np.ndarray, exposure: float = 0.0) -> np.ndarray:
    """Simple tone mapping for HDR display."""
    
    # Apply exposure
    exposed = hdr_image * (2.0 ** exposure)
    
    # Simple Reinhard tone mapping
    tonemapped = exposed / (1.0 + exposed)
    
    # Gamma correction
    gamma_corrected = np.power(np.clip(tonemapped, 0, 1), 1.0 / 2.2)
    
    # Convert to 8-bit
    result = (gamma_corrected * 255).astype(np.uint8)
    
    # Add alpha channel if needed
    if result.shape[2] == 3:
        alpha = 255 * np.ones((*result.shape[:2], 1), dtype=np.uint8)
        result = np.concatenate([result, alpha], axis=2)
    
    return result


def main():
    """Main example execution."""
    print("IBL Environment Lighting")
    print("=======================")
    
    out_dir = Path(__file__).parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    
    try:
        import forge3d as f3d
        
        print("Generating procedural HDRI environment...")
        hdri_env = generate_procedural_hdri(256)
        
        print("Calculating spherical harmonics coefficients...")
        sh_coeffs = calculate_spherical_harmonics_coefficients(hdri_env, num_bands=3)
        
        print("Creating test geometry...")
        sphere_normals, plane_normals, sphere_mask = create_test_geometry()
        
        print("Rendering diffuse lighting from environment...")
        
        # Render sphere with IBL
        sphere_diffuse = render_diffuse_from_sh(sh_coeffs, sphere_normals)
        sphere_diffuse[~sphere_mask] = 0  # Mask out non-sphere pixels
        
        # Render plane with IBL  
        plane_diffuse = render_diffuse_from_sh(sh_coeffs, plane_normals)
        
        # Tone map results for display
        print("Tone mapping for display...")
        
        # Environment map
        env_display = tone_map_for_display(hdri_env, exposure=-2.0)
        
        # Sphere render
        sphere_display = tone_map_for_display(sphere_diffuse, exposure=0.0)
        
        # Plane render
        plane_display = tone_map_for_display(plane_diffuse, exposure=0.0)
        
        # Save all outputs
        saved_paths = {}
        
        env_path = out_dir / "ibl_environment_map.png"
        f3d.numpy_to_png(str(env_path), env_display)
        saved_paths['environment'] = str(env_path)
        print(f"Saved environment map: {env_path}")
        
        sphere_path = out_dir / "ibl_sphere_diffuse.png"
        f3d.numpy_to_png(str(sphere_path), sphere_display)
        saved_paths['sphere'] = str(sphere_path)
        print(f"Saved sphere diffuse: {sphere_path}")
        
        plane_path = out_dir / "ibl_plane_diffuse.png"
        f3d.numpy_to_png(str(plane_path), plane_display)
        saved_paths['plane'] = str(plane_path)
        print(f"Saved plane diffuse: {plane_path}")
        
        # Create comparison showing environment and lit objects
        try:
            env_h, env_w = env_display.shape[:2]
            sphere_h, sphere_w = sphere_display.shape[:2]
            
            # Resize environment map to match sphere height
            if env_h != sphere_h:
                env_resized = np.zeros((sphere_h, env_w, 4), dtype=np.uint8)
                scale_factor = sphere_h / env_h
                for i in range(sphere_h):
                    src_i = int(i / scale_factor)
                    if src_i < env_h:
                        env_resized[i] = env_display[src_i]
            else:
                env_resized = env_display
            
            # Create side-by-side comparison
            comparison = np.zeros((sphere_h, env_w + sphere_w, 4), dtype=np.uint8)
            comparison[:, :env_w] = env_resized
            comparison[:, env_w:] = sphere_display
            
            comp_path = out_dir / "ibl_environment_sphere_comparison.png"
            f3d.numpy_to_png(str(comp_path), comparison)
            saved_paths['comparison'] = str(comp_path)
            print(f"Saved comparison: {comp_path}")
            
        except Exception as e:
            print(f"Comparison creation failed: {e}")
        
        # Visualize SH coefficients
        try:
            # Create visualization of SH coefficients  
            coeff_vis = np.zeros((3, sh_coeffs.shape[0], 4), dtype=np.uint8)
            
            # Normalize coefficients for visualization
            max_coeff = np.abs(sh_coeffs).max()
            if max_coeff > 0:
                normalized_coeffs = sh_coeffs / max_coeff * 0.5 + 0.5
                
                for channel in range(3):
                    for coeff in range(sh_coeffs.shape[0]):
                        coeff_vis[channel, coeff, channel] = int(normalized_coeffs[coeff, channel] * 255)
                        coeff_vis[channel, coeff, 3] = 255
            
            sh_path = out_dir / "ibl_sh_coefficients.png"
            f3d.numpy_to_png(str(sh_path), coeff_vis)
            saved_paths['sh_coefficients'] = str(sh_path)
            print(f"Saved SH coefficients: {sh_path}")
            
        except Exception as e:
            print(f"SH visualization failed: {e}")
        
        # Analyze lighting characteristics
        env_stats = {
            'luminance_range': {
                'min': float(hdri_env.min()),
                'max': float(hdri_env.max()),
                'mean': float(hdri_env.mean()),
                'std': float(hdri_env.std()),
            },
            'dynamic_range': float(hdri_env.max() / max(hdri_env.min(), 0.001)),
        }
        
        # Analyze SH coefficients
        sh_analysis = {
            'num_bands': 3,
            'num_coefficients': sh_coeffs.shape[0],
            'coefficient_magnitudes': {
                'max': float(np.abs(sh_coeffs).max()),
                'mean': float(np.abs(sh_coeffs).mean()),
                'l0_magnitude': float(np.abs(sh_coeffs[0]).mean()) if sh_coeffs.shape[0] > 0 else 0,
                'l1_magnitude': float(np.abs(sh_coeffs[1:4]).mean()) if sh_coeffs.shape[0] > 3 else 0,
                'l2_magnitude': float(np.abs(sh_coeffs[4:9]).mean()) if sh_coeffs.shape[0] > 8 else 0,
            }
        }
        
        # Generate comprehensive metrics
        metrics = {
            'environment_size': hdri_env.shape[:2],
            'geometry_size': sphere_normals.shape[:2],
            'environment_stats': env_stats,
            'spherical_harmonics': sh_analysis,
            'tone_mapping': {
                'method': 'reinhard',
                'gamma': 2.2,
                'environment_exposure': -2.0,
                'object_exposure': 0.0,
            },
            'outputs': saved_paths,
        }
        
        print("\nIBL Analysis Results:")
        print(f"  Environment dynamic range: {env_stats['dynamic_range']:.1f}:1")
        print(f"  Mean environment luminance: {env_stats['luminance_range']['mean']:.2f}")
        print(f"  SH coefficients: {sh_coeffs.shape[0]} ({sh_analysis['num_bands']} bands)")
        print(f"  L0 (ambient): {sh_analysis['coefficient_magnitudes']['l0_magnitude']:.3f}")
        print(f"  L1 (directional): {sh_analysis['coefficient_magnitudes']['l1_magnitude']:.3f}")
        print(f"  L2 (quadratic): {sh_analysis['coefficient_magnitudes']['l2_magnitude']:.3f}")
        
        # Save metrics
        import json
        metrics_path = out_dir / "ibl_metrics.json"
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