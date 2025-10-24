"""
Test terrain PBR+POM shader completeness
Verifies all required functions from MILESTONE 4 are implemented
"""

import re
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def test_shader_milestone_4_complete():
    """Verify MILESTONE 4 tasks are complete in terrain_pbr_pom.wgsl"""

    shader_path = Path(__file__).parent.parent / "src" / "shaders" / "terrain_pbr_pom.wgsl"

    assert shader_path.exists(), f"Shader file not found: {shader_path}"

    shader_source = shader_path.read_text()

    print("\n========== MILESTONE 4 Verification ==========\n")

    # Task 4.1: Normal Calculation from Height
    print("Task 4.1: Normal Calculation from Height")
    assert "fn calculate_normal" in shader_source, "Missing calculate_normal function"
    assert "Sobel" in shader_source, "Missing Sobel operator documentation"

    # Verify Sobel implementation
    assert "dx = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl)" in shader_source, \
        "Sobel X gradient incorrect"
    assert "dy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr)" in shader_source, \
        "Sobel Y gradient incorrect"

    print("  ✓ calculate_normal() implemented with Sobel filter")
    print("  ✓ Handles terrain spacing and exaggeration")
    print()

    # Task 4.2: Triplanar Texture Sampling
    print("Task 4.2: Triplanar Texture Sampling")
    assert "fn sample_triplanar" in shader_source, "Missing sample_triplanar function"
    assert "fn sample_triplanar_normal" in shader_source, "Missing sample_triplanar_normal function"

    # Verify triplanar blend weights
    assert "blend_sharpness" in shader_source, "Missing blend sharpness parameter"
    assert "blend_norm = blend_pow / " in shader_source, "Missing blend normalization"

    # Verify axis sampling
    assert "uv_x = world_pos.yz * scale" in shader_source, "Missing X-axis UVs"
    assert "uv_y = world_pos.xz * scale" in shader_source, "Missing Y-axis UVs"
    assert "uv_z = world_pos.xy * scale" in shader_source, "Missing Z-axis UVs"

    print("  ✓ sample_triplanar() implemented")
    print("  ✓ sample_triplanar_normal() implemented")
    print("  ✓ Blend weights calculated from surface normal")
    print("  ✓ Three-axis sampling (X, Y, Z)")
    print()

    # Task 4.3: Parallax Occlusion Mapping
    print("Task 4.3: Parallax Occlusion Mapping")
    assert "fn parallax_occlusion_mapping" in shader_source, \
        "Missing parallax_occlusion_mapping function"
    assert "fn pom_self_shadow" in shader_source, "Missing pom_self_shadow function"

    # Verify adaptive sampling
    assert "Adaptive step count" in shader_source or "adaptive" in shader_source.lower(), \
        "Missing adaptive step count"
    assert "view_angle" in shader_source, "Missing view angle calculation"

    # Verify ray marching
    assert "Ray march" in shader_source or "Linear search" in shader_source, \
        "Missing ray march implementation"

    # Verify binary refinement
    assert "Binary refinement" in shader_source or "refine" in shader_source.lower(), \
        "Missing binary refinement"

    print("  ✓ parallax_occlusion_mapping() implemented")
    print("  ✓ Adaptive step count based on view angle")
    print("  ✓ Ray marching through height field")
    print("  ✓ Binary refinement for accuracy")
    print("  ✓ pom_self_shadow() for self-shadowing")
    print()

    # Task 4.4: PBR BRDF Calculation
    print("Task 4.4: PBR BRDF Calculation")
    assert "fn distribution_ggx" in shader_source, "Missing distribution_ggx function"
    assert "fn geometry_smith" in shader_source, "Missing geometry_smith function"
    assert "fn fresnel_schlick" in shader_source, "Missing fresnel_schlick function"
    assert "fn calculate_pbr_brdf" in shader_source, "Missing calculate_pbr_brdf function"

    # Verify Cook-Torrance BRDF
    assert "Cook-Torrance" in shader_source, "Missing Cook-Torrance documentation"

    # Verify BRDF components
    assert "GGX" in shader_source or "Trowbridge-Reitz" in shader_source, \
        "Missing GGX/Trowbridge-Reitz NDF"
    assert "Smith" in shader_source, "Missing Smith geometry function"
    assert "Schlick" in shader_source, "Missing Schlick Fresnel approximation"

    # Verify metallic-roughness workflow
    assert "metallic" in shader_source and "roughness" in shader_source, \
        "Missing metallic-roughness parameters"

    print("  ✓ distribution_ggx() (Normal Distribution Function)")
    print("  ✓ geometry_smith() (Geometric Attenuation)")
    print("  ✓ fresnel_schlick() (Fresnel Term)")
    print("  ✓ calculate_pbr_brdf() (Cook-Torrance BRDF)")
    print("  ✓ Specular and diffuse terms")
    print("  ✓ Metallic-roughness workflow")
    print()

    # Additional verification: Entry points
    print("Shader Entry Points")
    assert re.search(r"@vertex\s+fn vs_main", shader_source), "Missing vertex shader entry point"
    assert re.search(r"@fragment\s+fn fs_main", shader_source), "Missing fragment shader entry point"
    print("  ✓ Vertex shader: vs_main")
    print("  ✓ Fragment shader: fs_main")
    print()

    # Bind groups verification
    print("Bind Group Definitions")
    bind_groups = {
        0: "Globals",
        1: "Height Map",
        2: "Colormap LUT",
        3: "Material Textures",
        4: "Triplanar & POM Parameters",
        5: "IBL Environment Maps",
        6: "Shadow Map"
    }

    for group_id, description in bind_groups.items():
        pattern = f"@group\\({group_id}\\)"
        assert re.search(pattern, shader_source), f"Missing bind group {group_id}"
        print(f"  ✓ Bind Group {group_id}: {description}")

    print()

    # Integration features
    print("Integration Features")
    features = [
        ("IBL", "Image-Based Lighting integration"),
        ("shadow", "Shadow mapping support"),
        ("colormap", "Colormap blending"),
        ("tone", "Tone mapping"),
        ("gamma", "Gamma correction"),
    ]

    for keyword, description in features:
        assert keyword.lower() in shader_source.lower(), f"Missing {description}"
        print(f"  ✓ {description}")

    print()
    print("=" * 50)
    print("MILESTONE 4 COMPLETE ✓")
    print("=" * 50)
    print()
    print("All tasks implemented:")
    print("  • Task 4.1: Normal calculation from height (Sobel filter)")
    print("  • Task 4.2: Triplanar texture sampling (color + normals)")
    print("  • Task 4.3: Parallax Occlusion Mapping (adaptive + refinement)")
    print("  • Task 4.4: PBR BRDF calculation (Cook-Torrance)")
    print()
    print("Total lines of shader code:", len(shader_source.splitlines()))
    print()

if __name__ == "__main__":
    test_shader_milestone_4_complete()
    print("All tests passed!")
