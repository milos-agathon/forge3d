// Advanced lighting models for high-quality terrain rendering
// Includes Lambert, Phong, and Blinn-Phong reflection models

// ============================================================================
// Structures
// ============================================================================

struct Light {
    direction: vec3<f32>,    // Light direction (pointing toward light)
    color: vec3<f32>,        // Light color
    intensity: f32,          // Light intensity multiplier
}

struct Material {
    diffuse: vec3<f32>,      // Diffuse albedo (base color)
    specular: vec3<f32>,     // Specular color/strength
    shininess: f32,          // Phong/Blinn-Phong exponent (1-256)
}

// ============================================================================
// Lambert Diffuse (Basic)
// ============================================================================

// Simple Lambertian diffuse reflection
// Returns: diffuse color contribution
fn lambert_diffuse(
    normal: vec3<f32>,
    light_dir: vec3<f32>,
    material: Material,
    light: Light
) -> vec3<f32> {
    let n_dot_l = max(dot(normalize(normal), normalize(light_dir)), 0.0);
    return material.diffuse * light.color * light.intensity * n_dot_l;
}

// ============================================================================
// Phong Reflection Model
// ============================================================================

// Classic Phong lighting with view-dependent specular highlights
// Uses reflected light vector for specular term
// Returns: diffuse + specular color contribution
fn phong_lighting(
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    light_dir: vec3<f32>,
    material: Material,
    light: Light
) -> vec3<f32> {
    let N = normalize(normal);
    let L = normalize(light_dir);
    let V = normalize(view_dir);
    
    // Diffuse term (Lambertian)
    let n_dot_l = max(dot(N, L), 0.0);
    let diffuse = material.diffuse * n_dot_l;
    
    // Specular term (reflected ray)
    let R = reflect(-L, N);
    let r_dot_v = max(dot(R, V), 0.0);
    let spec_factor = pow(r_dot_v, material.shininess);
    let specular = material.specular * spec_factor;
    
    // Combine with light properties
    return (diffuse + specular) * light.color * light.intensity;
}

// ============================================================================
// Blinn-Phong Reflection Model
// ============================================================================

// Blinn-Phong lighting using half-vector
// More efficient than Phong, produces wider/softer highlights
// Returns: diffuse + specular color contribution
fn blinn_phong_lighting(
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    light_dir: vec3<f32>,
    material: Material,
    light: Light
) -> vec3<f32> {
    let N = normalize(normal);
    let L = normalize(light_dir);
    let V = normalize(view_dir);
    
    // Diffuse term (Lambertian)
    let n_dot_l = max(dot(N, L), 0.0);
    let diffuse = material.diffuse * n_dot_l;
    
    // Specular term (half-vector)
    let H = normalize(L + V);
    let n_dot_h = max(dot(N, H), 0.0);
    let spec_factor = pow(n_dot_h, material.shininess);
    let specular = material.specular * spec_factor;
    
    // Combine with light properties
    return (diffuse + specular) * light.color * light.intensity;
}

// ============================================================================
// Hue-Preserving Variant (for categorical landcover)
// ============================================================================

// Modified Blinn-Phong that preserves categorical color hues
// Adds specular as overlay rather than additive blend
// Useful for landcover where base color must remain recognizable
fn blinn_phong_hue_preserving(
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    light_dir: vec3<f32>,
    base_color: vec3<f32>,
    specular_strength: f32,
    shininess: f32,
    light: Light
) -> vec3<f32> {
    let N = normalize(normal);
    let L = normalize(light_dir);
    let V = normalize(view_dir);
    
    // Diffuse term
    let n_dot_l = max(dot(N, L), 0.0);
    let diffuse = base_color * n_dot_l * light.intensity;
    
    // Specular term
    let H = normalize(L + V);
    let n_dot_h = max(dot(N, H), 0.0);
    let spec_factor = pow(n_dot_h, shininess);
    
    // Mix specular as overlay to preserve hue
    let specular_add = vec3<f32>(spec_factor) * specular_strength * light.intensity;
    
    // Combine: base color with controlled specular overlay
    return (diffuse + specular_add) * light.color;
}

// ============================================================================
// Utility Functions
// ============================================================================

// Create default material from base color
fn material_from_color(base_color: vec3<f32>, specular_strength: f32, shininess: f32) -> Material {
    return Material(
        base_color,                              // diffuse = base color
        vec3<f32>(specular_strength),           // specular = grayscale strength
        shininess
    );
}

// Create default directional light
fn directional_light(direction: vec3<f32>, intensity: f32) -> Light {
    return Light(
        normalize(direction),
        vec3<f32>(1.0, 1.0, 1.0),  // white light
        intensity
    );
}
