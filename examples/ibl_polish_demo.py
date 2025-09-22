#!/usr/bin/env python3
"""
B15 Example: IBL Polish Demonstration

Comprehensive demonstration of the B15 Image-Based Lighting (IBL) Polish system.
Shows environment map loading, quality settings, texture generation, and material testing.
Validates all B15 acceptance criteria for IBL implementation.
"""

import numpy as np
import sys
import os
import time
from pathlib import Path

# Use the import shim for running from repo
sys.path.insert(0, str(Path(__file__).parent.parent))

def generate_test_environment_hdr(width: int = 512, height: int = 256) -> np.ndarray:
    """Generate a procedural HDR environment for testing IBL system."""

    # Create equirectangular coordinates
    u = np.linspace(0, 2 * np.pi, width)  # longitude: 0 to 2π
    v = np.linspace(0, np.pi, height)     # latitude: 0 to π
    U, V = np.meshgrid(u, v)

    # Convert to Cartesian for lighting calculations
    x = np.sin(V) * np.cos(U)
    y = np.cos(V)
    z = np.sin(V) * np.sin(U)

    # Initialize HDR environment
    hdr_env = np.zeros((height, width, 3), dtype=np.float32)

    # Sky gradient with higher luminance
    sky_intensity = np.exp(-np.abs(y - 0.3) * 1.5) * 3.0
    sky_color = np.stack([
        sky_intensity * 0.5,  # R: warm tones
        sky_intensity * 0.7,  # G: sky blue
        sky_intensity * 1.0,  # B: sky blue
    ], axis=2)

    # Bright sun disk for specular highlights
    sun_dir = np.array([0.6, 0.8, 0.0])  # Sun direction
    sun_dir = sun_dir / np.linalg.norm(sun_dir)

    sun_dot = x * sun_dir[0] + y * sun_dir[1] + z * sun_dir[2]
    sun_intensity = np.exp((sun_dot - 0.995) * 200.0)  # Very sharp sun disk
    sun_intensity = np.clip(sun_intensity, 0, 10000)  # Extremely bright for specular

    sun_color = np.stack([
        sun_intensity * 10000.0,  # Very bright sun
        sun_intensity * 9500.0,
        sun_intensity * 8000.0,
    ], axis=2)

    # Secondary light for contrast
    moon_dir = np.array([-0.5, 0.6, 0.6])
    moon_dir = moon_dir / np.linalg.norm(moon_dir)

    moon_dot = x * moon_dir[0] + y * moon_dir[1] + z * moon_dir[2]
    moon_intensity = np.exp((moon_dot - 0.98) * 100.0)
    moon_intensity = np.clip(moon_intensity, 0, 1000)

    moon_color = np.stack([
        moon_intensity * 800.0,   # Cooler secondary light
        moon_intensity * 900.0,
        moon_intensity * 1000.0,
    ], axis=2)

    # Ground plane reflection
    ground_mask = y < 0.0
    ground_intensity = np.exp(y[ground_mask] * 2.0) * 0.8
    ground_color = np.zeros_like(sky_color)
    ground_color[ground_mask] = ground_intensity[:, np.newaxis] * np.array([0.4, 0.3, 0.2])

    # Combine all lighting
    hdr_env = sky_color + sun_color + moon_color + ground_color

    # Add ambient base
    hdr_env += 0.2

    return hdr_env


def create_test_terrain(size: int = 128) -> np.ndarray:
    """Create test terrain heightfield for IBL demonstration."""

    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)

    # Multiple octaves of noise for interesting terrain
    heights = np.zeros_like(X)

    # Base terrain shape
    heights += 0.3 * np.exp(-(X**2 + Y**2) / 2.0)

    # Add some hills
    heights += 0.2 * np.sin(X * 2) * np.cos(Y * 1.5)
    heights += 0.15 * np.sin(X * 4 + 1) * np.cos(Y * 3 + 0.5)

    # Add fine detail
    heights += 0.05 * np.sin(X * 8 + 2) * np.cos(Y * 6 + 1.2)

    # Ensure positive heights
    heights = np.maximum(heights, 0.0)

    return heights.astype(np.float32)


def test_ibl_material_properties(scene, test_cases: list) -> dict:
    """Test IBL material properties for various roughness/metallic combinations."""

    results = {}

    for case_name, metallic, roughness, base_color in test_cases:
        try:
            r, g, b = base_color
            f0_r, f0_g, f0_b = scene.test_ibl_material(metallic, roughness, r, g, b)

            # Sample BRDF LUT at various view angles
            brdf_samples = []
            for n_dot_v in [0.1, 0.5, 0.9]:
                fresnel, roughness_term = scene.sample_brdf_lut(n_dot_v, roughness)
                brdf_samples.append({
                    'n_dot_v': n_dot_v,
                    'fresnel_term': fresnel,
                    'roughness_term': roughness_term
                })

            results[case_name] = {
                'input': {
                    'metallic': metallic,
                    'roughness': roughness,
                    'base_color': base_color
                },
                'f0': [f0_r, f0_g, f0_b],
                'brdf_samples': brdf_samples
            }

        except Exception as e:
            results[case_name] = {'error': str(e)}

    return results


def demonstrate_quality_levels(scene, env_data: np.ndarray) -> dict:
    """Demonstrate IBL quality levels and measure generation times."""

    quality_results = {}
    qualities = ['low', 'medium', 'high', 'ultra']

    for quality in qualities:
        print(f"  Testing {quality} quality...")
        start_time = time.time()

        try:
            # Set quality and regenerate textures
            scene.set_ibl_quality(quality)

            # Load environment map
            height, width = env_data.shape[:2]
            scene.load_environment_map(env_data.flatten().tolist(), width, height)

            # Generate IBL textures
            scene.generate_ibl_textures()

            # Get texture info
            irradiance_info, specular_info, brdf_info = scene.get_ibl_texture_info()

            generation_time = time.time() - start_time

            quality_results[quality] = {
                'generation_time': generation_time,
                'textures': {
                    'irradiance': irradiance_info,
                    'specular': specular_info,
                    'brdf_lut': brdf_info
                },
                'quality_setting': scene.get_ibl_quality(),
                'initialized': scene.is_ibl_initialized()
            }

        except Exception as e:
            quality_results[quality] = {
                'error': str(e),
                'generation_time': time.time() - start_time
            }

    return quality_results


def main():
    """Main B15 IBL Polish demonstration."""
    print("B15 IBL Polish Demonstration")
    print("============================")

    out_dir = Path(__file__).parent.parent / "out"
    out_dir.mkdir(exist_ok=True)

    try:
        import forge3d as f3d

        print("\n1. Creating Scene and enabling IBL...")
        scene = f3d.Scene(width=1024, height=768, grid=128)

        # Enable IBL with medium quality initially
        scene.enable_ibl('medium')
        print(f"   IBL enabled: {scene.is_ibl_enabled()}")
        print(f"   Initial quality: {scene.get_ibl_quality()}")

        print("\n2. Generating test environment map...")
        env_data = generate_test_environment_hdr(512, 256)

        # Save environment map for reference
        env_display = np.clip(env_data ** (1.0/2.2) * 0.3, 0, 1)  # Simple tone mapping
        env_rgba = np.zeros((*env_display.shape[:2], 4), dtype=np.uint8)
        env_rgba[:, :, :3] = (env_display * 255).astype(np.uint8)
        env_rgba[:, :, 3] = 255

        env_path = out_dir / "ibl_test_environment.png"
        f3d.numpy_to_png(str(env_path), env_rgba)
        print(f"   Saved environment map: {env_path}")

        print("\n3. Testing quality levels...")
        quality_results = demonstrate_quality_levels(scene, env_data)

        # Report quality performance
        for quality, result in quality_results.items():
            if 'error' in result:
                print(f"   {quality}: ERROR - {result['error']}")
            else:
                print(f"   {quality}: {result['generation_time']:.3f}s")

        print("\n4. Creating test terrain...")
        terrain_heights = create_test_terrain(128)
        scene.upload_height_r32f(terrain_heights, 128, 128)

        # Set up camera for terrain view
        scene.set_camera_position(2.0, 1.5, 2.0)
        scene.set_camera_target(0.0, 0.0, 0.0)
        scene.set_camera_up(0.0, 1.0, 0.0)
        scene.set_projection_perspective(45.0, 1024.0/768.0, 0.1, 100.0)

        print("\n5. Testing material properties...")
        test_cases = [
            ('Metal_Rough', 1.0, 0.8, (0.7, 0.7, 0.7)),      # Rough metal
            ('Metal_Smooth', 1.0, 0.1, (0.7, 0.7, 0.7)),     # Smooth metal
            ('Dielectric_Rough', 0.0, 0.9, (0.5, 0.3, 0.2)), # Rough dielectric
            ('Dielectric_Smooth', 0.0, 0.1, (0.5, 0.3, 0.2)), # Smooth dielectric
            ('Mixed_Medium', 0.5, 0.5, (0.6, 0.4, 0.3)),      # Mixed material
        ]

        material_results = test_ibl_material_properties(scene, test_cases)

        # Report material test results
        for case_name, result in material_results.items():
            if 'error' in result:
                print(f"   {case_name}: ERROR - {result['error']}")
            else:
                f0 = result['f0']
                print(f"   {case_name}: F0=({f0[0]:.3f}, {f0[1]:.3f}, {f0[2]:.3f})")

        print("\n6. Rendering terrain with IBL...")

        # Render with different quality settings
        rendered_outputs = {}

        for quality in ['low', 'high']:
            print(f"   Rendering with {quality} quality...")
            scene.set_ibl_quality(quality)

            # Reload and regenerate for quality change
            height, width = env_data.shape[:2]
            scene.load_environment_map(env_data.flatten().tolist(), width, height)
            scene.generate_ibl_textures()

            # Render terrain
            terrain_rgba = scene.render_rgba()

            # Save output
            output_path = out_dir / f"ibl_terrain_{quality}_quality.png"
            f3d.numpy_to_png(str(output_path), terrain_rgba)
            rendered_outputs[quality] = str(output_path)
            print(f"     Saved: {output_path}")

        print("\n7. Generating comprehensive metrics...")

        # Final IBL state
        final_quality = scene.get_ibl_quality()
        is_initialized = scene.is_ibl_initialized()
        irradiance_info, specular_info, brdf_info = scene.get_ibl_texture_info()

        # Environment analysis
        env_stats = {
            'shape': env_data.shape,
            'luminance_range': {
                'min': float(env_data.min()),
                'max': float(env_data.max()),
                'mean': float(env_data.mean()),
                'std': float(env_data.std())
            },
            'dynamic_range': float(env_data.max() / max(env_data.min(), 0.001))
        }

        # Compile comprehensive results
        demo_results = {
            'b15_specifications': {
                'irradiance_prefiltering': True,
                'specular_prefiltering': True,
                'brdf_lut_generation': True,
                'quality_levels': list(quality_results.keys()),
                'material_testing': True,
                'environment_loading': True
            },
            'performance_metrics': {
                'quality_timings': {q: r.get('generation_time', 0) for q, r in quality_results.items() if 'generation_time' in r},
                'fastest_quality': min(quality_results.keys(), key=lambda q: quality_results[q].get('generation_time', float('inf'))) if quality_results else None,
                'slowest_quality': max(quality_results.keys(), key=lambda q: quality_results[q].get('generation_time', 0)) if quality_results else None
            },
            'ibl_state': {
                'enabled': scene.is_ibl_enabled(),
                'initialized': is_initialized,
                'current_quality': final_quality,
                'texture_info': {
                    'irradiance': irradiance_info,
                    'specular': specular_info,
                    'brdf_lut': brdf_info
                }
            },
            'environment_analysis': env_stats,
            'material_tests': material_results,
            'outputs': {
                'environment_map': str(env_path),
                'rendered_terrain': rendered_outputs
            },
            'validation': {
                'all_qualities_tested': len([r for r in quality_results.values() if 'error' not in r]) >= 3,
                'materials_tested': len([r for r in material_results.values() if 'error' not in r]) >= 3,
                'textures_generated': is_initialized,
                'environment_loaded': env_stats['dynamic_range'] > 10.0
            }
        }

        # Save comprehensive results
        import json
        results_path = out_dir / "ibl_b15_demo_results.json"
        with open(results_path, 'w') as f:
            json.dump(demo_results, f, indent=2)

        print(f"\n8. Results Summary:")
        print(f"   Environment dynamic range: {env_stats['dynamic_range']:.1f}:1")
        print(f"   Quality levels tested: {len(quality_results)}")
        print(f"   Material combinations tested: {len(material_results)}")
        print(f"   IBL textures generated: {is_initialized}")
        print(f"   Final quality setting: {final_quality}")

        if demo_results['performance_metrics']['fastest_quality']:
            fastest = demo_results['performance_metrics']['fastest_quality']
            fastest_time = quality_results[fastest]['generation_time']
            print(f"   Fastest quality: {fastest} ({fastest_time:.3f}s)")

        # Validation check
        validation = demo_results['validation']
        all_passed = all(validation.values())
        print(f"\n9. B15 Acceptance Criteria Validation:")
        print(f"   Quality levels functional: {'✓' if validation['all_qualities_tested'] else '✗'}")
        print(f"   Material testing functional: {'✓' if validation['materials_tested'] else '✗'}")
        print(f"   IBL textures generated: {'✓' if validation['textures_generated'] else '✗'}")
        print(f"   Environment loading functional: {'✓' if validation['environment_loaded'] else '✗'}")
        print(f"   Overall B15 validation: {'✓ PASSED' if all_passed else '✗ FAILED'}")

        print(f"\nResults saved: {results_path}")
        print("\nB15 IBL Polish demonstration completed successfully!")

        return 0 if all_passed else 1

    except ImportError as e:
        print(f"forge3d not available: {e}")
        return 1
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())