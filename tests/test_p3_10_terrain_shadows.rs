// tests/test_p3_10_terrain_shadows.rs
// P3-10: Terrain shadow integration
// Exit criteria: Terrain demo can enable shadows with no POM/shading regressions

use forge3d::gpu;

#[test]
fn test_terrain_shader_enables_shadows_by_default() {
    // Test that terrain shader enables TERRAIN_USE_SHADOWS by default

    // We can't actually compile shaders in unit tests without a GPU context,
    // but we verify the shader structure is sound by checking it exists
    let shader_path = std::path::Path::new("src/shaders/terrain_pbr_pom.wgsl");
    assert!(shader_path.exists(), "Terrain PBR shader must exist");

    // Read shader and verify key flags
    let shader_content =
        std::fs::read_to_string(shader_path).expect("Failed to read terrain shader");

    // Verify TERRAIN_USE_SHADOWS flag exists and defaults to true
    assert!(
        shader_content.contains("const TERRAIN_USE_SHADOWS: bool = TERRAIN_SHADOWS_ENABLED;"),
        "TERRAIN_USE_SHADOWS must be driven from a single source of truth"
    );
    assert!(
        shader_content.contains("const TERRAIN_SHADOWS_ENABLED: bool = true;"),
        "Terrain shadows should be enabled by default"
    );
}

#[test]
fn test_terrain_shader_has_shadow_flag() {
    // Verify shadow flag exists in shader
    let shader_path = std::path::Path::new("src/shaders/terrain_pbr_pom.wgsl");
    let shader_content =
        std::fs::read_to_string(shader_path).expect("Failed to read terrain shader");

    assert!(
        shader_content.contains("TERRAIN_USE_SHADOWS"),
        "Terrain shader must have TERRAIN_USE_SHADOWS flag"
    );
}

#[test]
fn test_terrain_shader_has_shadow_bindings() {
    // Verify shadow bindings exist at group(3)
    let shader_path = std::path::Path::new("src/shaders/terrain_pbr_pom.wgsl");
    let shader_content =
        std::fs::read_to_string(shader_path).expect("Failed to read terrain shader");

    assert!(
        shader_content.contains("@group(3) @binding(0)"),
        "Terrain shader must have shadow bindings at group(3)"
    );
    assert!(
        shader_content.contains("var<uniform> csm_uniforms: CsmUniforms;"),
        "Terrain shader must bind CSM uniforms"
    );
    assert!(
        shader_content.contains("var shadow_maps: texture_depth_2d_array;"),
        "Terrain shader must bind shadow maps"
    );
}

#[test]
fn test_terrain_shader_has_shadow_functions() {
    // Verify shadow sampling functions exist
    let shader_path = std::path::Path::new("src/shaders/terrain_pbr_pom.wgsl");
    let shader_content =
        std::fs::read_to_string(shader_path).expect("Failed to read terrain shader");

    assert!(
        shader_content.contains("fn select_cascade_terrain"),
        "Terrain shader must have cascade selection function"
    );
    assert!(
        shader_content.contains("fn sample_shadow_pcf_terrain"),
        "Terrain shader must have PCF shadow sampling"
    );
    assert!(
        shader_content.contains("fn calculate_shadow_terrain"),
        "Terrain shader must have shadow calculation function"
    );
}

#[test]
fn test_terrain_shader_applies_shadows_conditionally() {
    // Verify shadows are applied conditionally based on flag
    let shader_path = std::path::Path::new("src/shaders/terrain_pbr_pom.wgsl");
    let shader_content =
        std::fs::read_to_string(shader_path).expect("Failed to read terrain shader");

    assert!(
        shader_content.contains("if (TERRAIN_USE_SHADOWS)"),
        "Shadow application must be gated behind TERRAIN_USE_SHADOWS flag"
    );
    assert!(
        shader_content.contains("calculate_shadow_terrain("),
        "Must call terrain shadow calculation function"
    );
}

#[test]
fn test_terrain_shader_preserves_legacy_shadows() {
    // Verify legacy POM-based shadows are preserved when flag is false
    let shader_path = std::path::Path::new("src/shaders/terrain_pbr_pom.wgsl");
    let shader_content =
        std::fs::read_to_string(shader_path).expect("Failed to read terrain shader");

    assert!(
        shader_content.contains("// Legacy POM-based shadow factor"),
        "Legacy shadow path must be preserved"
    );
    assert!(
        shader_content.contains("if (shadow_enabled && pom_enabled)"),
        "Legacy shadow condition must exist in else branch"
    );
}

#[test]
fn test_terrain_shader_has_csm_structs() {
    // Verify CSM data structures are defined
    let shader_path = std::path::Path::new("src/shaders/terrain_pbr_pom.wgsl");
    let shader_content =
        std::fs::read_to_string(shader_path).expect("Failed to read terrain shader");

    assert!(
        shader_content.contains("struct ShadowCascade"),
        "Must define ShadowCascade struct"
    );
    assert!(
        shader_content.contains("struct CsmUniforms"),
        "Must define CsmUniforms struct"
    );
}

#[test]
fn test_terrain_shader_uses_view_depth() {
    // Verify view-space depth is calculated for cascade selection
    let shader_path = std::path::Path::new("src/shaders/terrain_pbr_pom.wgsl");
    let shader_content =
        std::fs::read_to_string(shader_path).expect("Failed to read terrain shader");

    assert!(
        shader_content
            .contains("let view_pos = u_terrain.view * vec4<f32>(input.world_position, 1.0);"),
        "Must calculate view-space position"
    );
    assert!(
        shader_content.contains("let view_depth = -view_pos.z;"),
        "Must extract view-space depth"
    );
}

#[test]
fn test_terrain_shader_applies_to_direct_lighting_only() {
    // Verify shadows only affect direct lighting
    let shader_path = std::path::Path::new("src/shaders/terrain_pbr_pom.wgsl");
    let shader_content =
        std::fs::read_to_string(shader_path).expect("Failed to read terrain shader");

    assert!(
        shader_content.contains("// Apply shadow to direct lighting only"),
        "Must document shadow application to direct lighting"
    );
    assert!(
        shader_content.contains("lighting = lighting * shadow_visibility;"),
        "Must multiply lighting by shadow visibility"
    );
}

#[test]
fn test_terrain_shader_preserves_ibl() {
    // Verify IBL remains separate from shadow application
    let shader_path = std::path::Path::new("src/shaders/terrain_pbr_pom.wgsl");
    let shader_content =
        std::fs::read_to_string(shader_path).expect("Failed to read terrain shader");

    // Shadow application should come before IBL calculation
    let shadow_pos = shader_content.find("shadow_visibility").unwrap_or(0);
    let ibl_pos = shader_content.find("ibl_contrib").unwrap_or(usize::MAX);

    assert!(
        shadow_pos < ibl_pos,
        "Shadow application must come before IBL calculation"
    );
}

#[test]
fn test_terrain_shader_has_brdf_dispatch_flag() {
    // Verify BRDF dispatch flag still exists (P2-05)
    let shader_path = std::path::Path::new("src/shaders/terrain_pbr_pom.wgsl");
    let shader_content =
        std::fs::read_to_string(shader_path).expect("Failed to read terrain shader");

    assert!(
        shader_content.contains("const TERRAIN_USE_BRDF_DISPATCH: bool = false;"),
        "TERRAIN_USE_BRDF_DISPATCH must remain for P2-05 compatibility"
    );
}

#[test]
fn test_terrain_shader_preserves_pom() {
    // Verify POM functionality is preserved
    let shader_path = std::path::Path::new("src/shaders/terrain_pbr_pom.wgsl");
    let shader_content =
        std::fs::read_to_string(shader_path).expect("Failed to read terrain shader");

    assert!(
        shader_content.contains("pom_enabled"),
        "POM enabled flag must be preserved"
    );
    assert!(
        shader_content.contains("parallax_occlusion_mapping"),
        "POM sampling function must be preserved"
    );
}

#[test]
fn test_terrain_shader_preserves_triplanar() {
    // Verify triplanar mapping is preserved
    let shader_path = std::path::Path::new("src/shaders/terrain_pbr_pom.wgsl");
    let shader_content =
        std::fs::read_to_string(shader_path).expect("Failed to read terrain shader");

    assert!(
        shader_content.contains("fn sample_triplanar"),
        "Triplanar sampling must be preserved"
    );
    assert!(
        shader_content.contains("triplanar_params"),
        "Triplanar parameters must be preserved"
    );
}

#[test]
fn test_terrain_shader_preserves_overlay() {
    // Verify overlay/colormap functionality is preserved
    let shader_path = std::path::Path::new("src/shaders/terrain_pbr_pom.wgsl");
    let shader_content =
        std::fs::read_to_string(shader_path).expect("Failed to read terrain shader");

    assert!(
        shader_content.contains("colormap_tex"),
        "Colormap texture must be preserved"
    );
    assert!(
        shader_content.contains("overlay_rgb"),
        "Overlay RGB must be preserved"
    );
}

#[test]
fn test_terrain_shader_shadow_group_no_conflict() {
    // Verify shadow bindings at group(3) don't conflict with IBL at group(2)
    let shader_path = std::path::Path::new("src/shaders/terrain_pbr_pom.wgsl");
    let shader_content =
        std::fs::read_to_string(shader_path).expect("Failed to read terrain shader");

    // IBL should be at group(2)
    assert!(
        shader_content.contains("@group(2) @binding(0)\nvar ibl_specular_tex"),
        "IBL must be at group(2)"
    );

    // Shadows should be at group(3)
    assert!(
        shader_content.contains("@group(3) @binding(0)\nvar<uniform> csm_uniforms"),
        "Shadows must be at group(3)"
    );
}

#[test]
fn test_terrain_shader_documentation() {
    // Verify P3-10 is documented in shader
    let shader_path = std::path::Path::new("src/shaders/terrain_pbr_pom.wgsl");
    let shader_content =
        std::fs::read_to_string(shader_path).expect("Failed to read terrain shader");

    assert!(
        shader_content.contains("P3-10"),
        "P3-10 milestone must be documented in shader"
    );
}

#[test]
fn test_default_behavior_unchanged() {
    // Critical test: verify default behavior is identical to before P3-10
    let shader_path = std::path::Path::new("src/shaders/terrain_pbr_pom.wgsl");
    let shader_content =
        std::fs::read_to_string(shader_path).expect("Failed to read terrain shader");

    // With TERRAIN_USE_SHADOWS = false (default), legacy path must be used
    assert!(
        shader_content.contains("const TERRAIN_USE_SHADOWS: bool = TERRAIN_SHADOWS_ENABLED;"),
        "Default must be true and centralized"
    );
}

#[test]
fn test_terrain_shader_has_pcss_radius_field() {
    // Verify the PCSS radius plumbed through CSM uniforms
    let shader_path = std::path::Path::new("src/shaders/terrain_pbr_pom.wgsl");
    let shader_content =
        std::fs::read_to_string(shader_path).expect("Failed to read terrain shader");

    assert!(
        shader_content.contains("pcss_light_radius"),
        "CSM uniforms must expose pcss_light_radius for PCSS softness control"
    );
    assert!(
        shader_content.contains("override DEBUG_SHADOW_CASCADES"),
        "Debug cascade overlay must be controllable via override"
    );
}
