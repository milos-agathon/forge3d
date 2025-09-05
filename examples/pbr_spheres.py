#!/usr/bin/env python3
"""
PBR Materials Demonstration

Demonstrates Physically-Based Rendering materials with:
1. Various material types (metals, dielectrics, emissive)
2. Material property validation and statistics
3. BRDF evaluation and lighting calculations
4. Texture sampling and material rendering
5. Visual comparison of different material properties

Usage:
    python examples/pbr_materials.py --headless --out out/pbr_materials.png
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
    import forge3d.pbr as pbr
except ImportError as e:
    print(f"ERROR: Could not import forge3d: {e}")
    print("Run: maturin develop --release")
    sys.exit(1)


def create_pbr_material_gallery():
    """Create a gallery of different PBR materials."""
    print("Creating PBR material gallery...")
    
    materials = pbr.create_test_materials()
    
    print(f"Created {len(materials)} test materials:")
    for name, material in materials.items():
        validation = pbr.validate_pbr_material(material)
        stats = validation['statistics']
        
        print(f"  {name:15s}: metallic={material.metallic:.2f}, roughness={material.roughness:.2f}")
        print(f"    {'':17s} {'metallic' if stats['is_metallic'] else 'dielectric'}, "
              f"{'rough' if stats['is_rough'] else 'smooth'}, "
              f"{'emissive' if stats['is_emissive'] else 'non-emissive'}")
        
        if not validation['valid']:
            print(f"    {'':17s} ERRORS: {validation['errors']}")
        if validation['warnings']:
            print(f"    {'':17s} WARNINGS: {validation['warnings']}")
    
    return materials


def test_brdf_evaluation(materials):
    """Test BRDF evaluation for different materials."""
    print("\nTesting BRDF evaluation...")
    
    renderer = pbr.PbrRenderer()
    
    # Set up lighting
    lighting = pbr.PbrLighting(
        light_direction=(0.0, -1.0, 0.3),
        light_color=(1.0, 1.0, 1.0),
        light_intensity=3.0,
        camera_position=(0.0, 0.0, 5.0)
    )
    renderer.set_lighting(lighting)
    
    # Test vectors
    light_dir = np.array([0.0, -1.0, 0.3])
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    view_dir = np.array([0.0, 0.0, 1.0])
    normal = np.array([0.0, 1.0, 0.0])
    
    print(f"{'Material':<15} {'BRDF Result':<25} {'Luminance':<10}")
    print("-" * 55)
    
    brdf_results = {}
    for name, material in materials.items():
        brdf = renderer.evaluate_brdf(material, light_dir, view_dir, normal)
        luminance = 0.299 * brdf[0] + 0.587 * brdf[1] + 0.114 * brdf[2]
        
        brdf_str = f"({brdf[0]:.3f}, {brdf[1]:.3f}, {brdf[2]:.3f})"
        print(f"{name:<15} {brdf_str:<25} {luminance:<10.4f}")
        
        brdf_results[name] = {
            'brdf': brdf,
            'luminance': luminance
        }
    
    return brdf_results


def test_material_textures():
    """Test PBR material textures."""
    print("\nTesting PBR material textures...")
    
    textures = pbr.create_test_textures()
    
    print(f"Created {len(textures)} test textures:")
    for name, texture in textures.items():
        print(f"  {name:20s}: shape={texture.shape}, dtype={texture.dtype}")
        print(f"    {'':22s} range=[{np.min(texture)}, {np.max(texture)}]")
    
    # Create material with textures
    material = pbr.PbrMaterial(
        base_color=(0.8, 0.8, 0.8, 1.0),
        metallic=0.5,
        roughness=0.5
    )
    
    # Set textures
    material.set_base_color_texture(textures['checker_base_color'])
    material.set_metallic_roughness_texture(textures['metallic_roughness'])
    material.set_normal_texture(textures['normal'])
    
    print(f"\nMaterial with textures:")
    print(f"  Texture flags: {material.texture_flags} (binary: {bin(material.texture_flags)})")
    print(f"  Has base color: {bool(material.texture_flags & 1)}")
    print(f"  Has metallic-roughness: {bool(material.texture_flags & 2)}")
    print(f"  Has normal map: {bool(material.texture_flags & 4)}")
    
    return material, textures


def render_material_spheres(materials, output_path, width=800, height=600):
    """Render spheres with different PBR materials."""
    print(f"\nRendering PBR material spheres ({width}x{height})...")
    
    # Create synthetic render of material spheres
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Arrange materials in a grid
    material_names = list(materials.keys())
    cols = min(4, len(material_names))
    rows = (len(material_names) + cols - 1) // cols
    
    sphere_width = width // cols
    sphere_height = height // rows
    
    renderer = pbr.PbrRenderer()
    lighting = pbr.PbrLighting(
        light_direction=(0.0, -1.0, 0.5),
        light_color=(1.0, 1.0, 1.0),
        light_intensity=2.0,
        camera_position=(0.0, 0.0, 10.0)
    )
    renderer.set_lighting(lighting)
    
    # Render each material as a sphere
    for i, (name, material) in enumerate(materials.items()):
        row = i // cols
        col = i % cols
        
        x_start = col * sphere_width
        x_end = (col + 1) * sphere_width
        y_start = row * sphere_height
        y_end = (row + 1) * sphere_height
        
        center_x = x_start + sphere_width // 2
        center_y = y_start + sphere_height // 2
        radius = min(sphere_width, sphere_height) // 3
        
        # Render sphere with material
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                dx = x - center_x
                dy = y - center_y
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist <= radius:
                    # Calculate sphere surface normal
                    dz = np.sqrt(max(0, radius*radius - dx*dx - dy*dy))
                    normal = np.array([dx, dy, dz]) / radius
                    
                    # Calculate view and light directions
                    view_dir = np.array([0, 0, 1])  # Looking down Z
                    light_dir = np.array([0.3, -0.7, 0.6])  # Angled light
                    light_dir = light_dir / np.linalg.norm(light_dir)
                    
                    # Evaluate BRDF
                    try:
                        brdf = renderer.evaluate_brdf(material, light_dir, view_dir, normal)
                        color = np.clip(brdf * 255, 0, 255).astype(np.uint8)
                        image[y, x] = color
                    except Exception:
                        # Fallback color
                        base_color = np.array(material.base_color[:3]) * 128
                        image[y, x] = base_color.astype(np.uint8)
                else:
                    # Background
                    image[y, x] = [32, 32, 32]
    
    # Add text labels (simple)
    for i, name in enumerate(material_names):
        row = i // cols
        col = i % cols
        
        center_x = col * sphere_width + sphere_width // 2
        label_y = (row + 1) * sphere_height - 20
        
        # Simple text rendering by drawing white pixels in a pattern
        text_width = min(len(name) * 6, sphere_width - 20)
        text_start_x = center_x - text_width // 2
        
        for char_i, char in enumerate(name[:10]):  # Limit character count
            char_x = text_start_x + char_i * 6
            if 0 <= char_x < width - 8 and 0 <= label_y < height - 8:
                # Draw simple character pattern
                for dy in range(8):
                    for dx in range(6):
                        if (dx + dy + ord(char)) % 3 == 0:  # Simple pattern
                            px = char_x + dx
                            py = label_y + dy
                            if 0 <= px < width and 0 <= py < height:
                                image[py, px] = [255, 255, 255]
    
    # Save image
    save_image(image, output_path)
    return image


def save_image(image, output_path):
    """Save image to PNG file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        f3d.numpy_to_png(str(output_path), image)
        print(f"Saved PBR materials demo: {output_path}")
    except Exception as e:
        print(f"Warning: Could not save PNG: {e}")
        # Fallback to numpy save
        np.save(str(output_path.with_suffix('.npy')), image)
        print(f"Saved as numpy array: {output_path.with_suffix('.npy')}")


def test_material_validation():
    """Test material validation functionality."""
    print("\nTesting material validation...")
    
    # Test valid material
    valid_material = pbr.PbrMaterial(
        base_color=(0.8, 0.2, 0.2, 1.0),
        metallic=0.0,
        roughness=0.7
    )
    
    validation = pbr.validate_pbr_material(valid_material)
    print(f"Valid material test: {'PASS' if validation['valid'] else 'FAIL'}")
    if validation['warnings']:
        print(f"  Warnings: {validation['warnings']}")
    
    # Test invalid material
    invalid_material = pbr.PbrMaterial(
        base_color=(1.2, -0.1, 0.5, 1.0),  # Invalid range
        metallic=1.5,  # Out of range
        roughness=0.01  # Below minimum
    )
    
    validation = pbr.validate_pbr_material(invalid_material)
    print(f"Invalid material test: {'PASS' if not validation['valid'] else 'FAIL'}")
    print(f"  Errors detected: {len(validation['errors'])}")
    for error in validation['errors']:
        print(f"    - {error}")
    print(f"  Warnings detected: {len(validation['warnings'])}")
    for warning in validation['warnings']:
        print(f"    - {warning}")
    
    return validation['valid'], len(validation['errors']), len(validation['warnings'])


def compare_metallic_vs_dielectric():
    """Compare metallic vs dielectric materials."""
    print("\nComparing metallic vs dielectric materials...")
    
    # Create materials with same base color but different metallic values
    base_color = (0.7, 0.4, 0.3, 1.0)  # Copper-like color
    roughness = 0.2
    
    dielectric = pbr.PbrMaterial(
        base_color=base_color,
        metallic=0.0,
        roughness=roughness
    )
    
    metallic = pbr.PbrMaterial(
        base_color=base_color,
        metallic=1.0,
        roughness=roughness
    )
    
    # Test BRDF response
    renderer = pbr.PbrRenderer()
    light_dir = np.array([0.0, -1.0, 0.5])
    light_dir = light_dir / np.linalg.norm(light_dir)
    view_dir = np.array([0.0, 0.0, 1.0])
    normal = np.array([0.0, 1.0, 0.0])
    
    dielectric_brdf = renderer.evaluate_brdf(dielectric, light_dir, view_dir, normal)
    metallic_brdf = renderer.evaluate_brdf(metallic, light_dir, view_dir, normal)
    
    print(f"Dielectric BRDF: ({dielectric_brdf[0]:.3f}, {dielectric_brdf[1]:.3f}, {dielectric_brdf[2]:.3f})")
    print(f"Metallic BRDF:   ({metallic_brdf[0]:.3f}, {metallic_brdf[1]:.3f}, {metallic_brdf[2]:.3f})")
    
    # Calculate difference
    brdf_diff = np.linalg.norm(metallic_brdf - dielectric_brdf)
    print(f"BRDF difference magnitude: {brdf_diff:.4f}")
    
    # Validate that metallics and dielectrics behave differently
    return brdf_diff > 0.1  # Should have significant difference


def main():
    parser = argparse.ArgumentParser(description="PBR materials demonstration")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--out", type=str, default="out/pbr_materials.png", help="Output file path")
    parser.add_argument("--width", type=int, default=800, help="Image width")
    parser.add_argument("--height", type=int, default=600, help="Image height")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    print("=== PBR Materials Demo ===")
    print(f"Mode: {'headless' if args.headless else 'interactive'}")
    print(f"Output: {args.out}")
    print(f"Resolution: {args.width}x{args.height}")
    
    # Check feature availability
    if not pbr.has_pbr_support():
        print("ERROR: PBR materials module not available")
        return 1
    
    try:
        # Step 1: Create material gallery
        print("\n1. Creating PBR material gallery...")
        materials = create_pbr_material_gallery()
        
        # Step 2: Test BRDF evaluation
        print("\n2. Testing BRDF evaluation...")
        brdf_results = test_brdf_evaluation(materials)
        
        # Step 3: Test material textures
        print("\n3. Testing material textures...")
        textured_material, textures = test_material_textures()
        
        # Step 4: Test material validation
        print("\n4. Testing material validation...")
        valid, error_count, warning_count = test_material_validation()
        
        # Step 5: Compare metallic vs dielectric
        print("\n5. Comparing metallic vs dielectric...")
        metallic_different = compare_metallic_vs_dielectric()
        
        # Step 6: Render material spheres
        print("\n6. Rendering PBR material spheres...")
        image = render_material_spheres(materials, args.out, args.width, args.height)
        
        # Summary
        print("\n=== PBR Materials Demo Complete ===")
        print(f"Results:")
        print(f"  Materials created: {len(materials)}")
        print(f"  BRDF evaluations: {len(brdf_results)}")
        print(f"  Textures created: {len(textures)}")
        print(f"  Validation errors detected: {error_count}")
        print(f"  Validation warnings detected: {warning_count}")
        print(f"  Metallic vs dielectric difference: {'DETECTED' if metallic_different else 'NOT DETECTED'}")
        print(f"  Output image: {args.out}")
        
        # Validate critical functionality
        tests_passed = 0
        total_tests = 5
        
        if len(materials) >= 7:  # Should create expected number of materials
            tests_passed += 1
            print("  PASS: Material creation")
        else:
            print(f"  FAIL: Material creation ({len(materials)} < 7)")
        
        if len(brdf_results) == len(materials):  # Should evaluate all materials
            tests_passed += 1
            print("  PASS: BRDF evaluation")
        else:
            print("  FAIL: BRDF evaluation")
        
        if len(textures) >= 3:  # Should create expected textures
            tests_passed += 1
            print("  PASS: Texture creation")
        else:
            print("  FAIL: Texture creation")
        
        if error_count > 0:  # Should detect validation errors
            tests_passed += 1
            print("  PASS: Validation error detection")
        else:
            print("  FAIL: Validation error detection")
        
        if metallic_different:  # Should differentiate materials
            tests_passed += 1
            print("  PASS: Metallic vs dielectric differentiation")
        else:
            print("  FAIL: Metallic vs dielectric differentiation")
        
        print(f"\nOverall test result: {tests_passed}/{total_tests} tests passed")
        
        return 0 if tests_passed == total_tests else 1
        
    except Exception as e:
        print(f"ERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())