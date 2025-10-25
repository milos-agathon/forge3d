// src/shaders/lighting.wgsl
// P0 Lighting System: Lights, BRDFs, Shadows, and IBL
// Matches Rust types in src/lighting/types.rs

const PI: f32 = 3.14159265359;

// Light types (matches Rust LightType enum)
const LIGHT_DIRECTIONAL: u32 = 0u;
const LIGHT_POINT: u32 = 1u;
const LIGHT_SPOT: u32 = 2u;
const LIGHT_ENVIRONMENT: u32 = 3u;

// BRDF models (matches Rust BrdfModel enum)
const BRDF_LAMBERT: u32 = 0u;
const BRDF_COOK_TORRANCE_GGX: u32 = 1u;

// Shadow techniques (matches Rust ShadowTechnique enum)
const SHADOW_HARD: u32 = 0u;
const SHADOW_PCF: u32 = 1u;

// GI techniques (matches Rust GiTechnique enum)
const GI_NONE: u32 = 0u;
const GI_IBL: u32 = 1u;

// Uniform buffer structs (must match Rust repr(C) layouts exactly)

struct Light {
    kind: u32,
    intensity: f32,
    color: vec3<f32>,
    _pad0: f32,

    dir_or_pos: vec3<f32>,
    range: f32,

    spot_inner_deg: f32,
    spot_outer_deg: f32,
    _pad1: vec2<f32>,
};

struct MaterialShading {
    brdf: u32,
    roughness: f32,
    metallic: f32,
    ior: f32,
};

struct ShadowSettings {
    tech: u32,
    map_res: u32,
    bias: f32,
    normal_bias: f32,

    softness: f32,
    _pad: vec3<f32>,
};

struct GiSettings {
    tech: u32,
    ibl_intensity: f32,
    ibl_rotation_deg: f32,
    _pad: f32,
};

struct Atmosphere {
    fog_density: f32,
    exposure: f32,
    sky_model: u32,
    _pad: f32,
};

// Bind group declarations (to be included where needed)
// @group(0) @binding(1) var<uniform> uLight: Light;
// @group(0) @binding(2) var<uniform> uMat: MaterialShading;
// @group(0) @binding(3) var<uniform> uShad: ShadowSettings;
// @group(0) @binding(4) var<uniform> uGI: GiSettings;
// @group(0) @binding(5) var<uniform> uAtmo: Atmosphere;

// ====================
// BRDF Functions (P0)
// ====================

/// Lambert diffuse BRDF
fn brdf_lambert(albedo: vec3<f32>) -> vec3<f32> {
    return albedo / PI;
}

/// Trowbridge-Reitz / GGX Normal Distribution Function
fn distribution_ggx(n: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h = max(dot(n, h), 0.0);
    let n_dot_h2 = n_dot_h * n_dot_h;

    let denom = n_dot_h2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

/// Smith's Geometry function with GGX
fn geometry_smith_ggx(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l = max(dot(n, l), 0.0);
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;

    let ggx1 = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let ggx2 = n_dot_l / (n_dot_l * (1.0 - k) + k);

    return ggx1 * ggx2;
}

/// Fresnel-Schlick approximation
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - cos_theta, 5.0);
}

/// Cook-Torrance GGX BRDF
fn brdf_cook_torrance_ggx(
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    ior: f32,
) -> vec3<f32> {
    let h = normalize(v + l);

    // Calculate F0 (base reflectance)
    let dielectric_f0 = ((ior - 1.0) / (ior + 1.0)) * ((ior - 1.0) / (ior + 1.0));
    let f0 = mix(vec3<f32>(dielectric_f0), albedo, metallic);

    // Cook-Torrance microfacet model
    let D = distribution_ggx(n, h, roughness);
    let G = geometry_smith_ggx(n, v, l, roughness);
    let F = fresnel_schlick(max(dot(h, v), 0.0), f0);

    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 0.0);

    // Specular term
    let numerator = D * G * F;
    let denominator = 4.0 * n_dot_v * n_dot_l + 0.0001;
    let specular = numerator / denominator;

    // Diffuse term (energy conserving)
    let kS = F;
    let kD = (vec3<f32>(1.0) - kS) * (1.0 - metallic);
    let diffuse = kD * albedo / PI;

    return diffuse + specular;
}

/// Evaluate BRDF based on material settings
fn eval_brdf(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    albedo: vec3<f32>,
    mat: MaterialShading,
) -> vec3<f32> {
    if (mat.brdf == BRDF_LAMBERT) {
        return brdf_lambert(albedo);
    } else {
        // BRDF_COOK_TORRANCE_GGX
        return brdf_cook_torrance_ggx(albedo, mat.metallic, mat.roughness, n, v, l, mat.ior);
    }
}

// ======================
// Light Functions (P0)
// ======================

/// Calculate light direction and attenuation for a light
/// Returns: (light_direction, attenuation)
fn get_light_vector(light: Light, world_pos: vec3<f32>) -> vec2<f32> {
    var light_dir: vec3<f32>;
    var attenuation: f32 = 1.0;

    if (light.kind == LIGHT_DIRECTIONAL) {
        // Directional light - constant direction
        light_dir = normalize(light.dir_or_pos);
        attenuation = 1.0;
    } else if (light.kind == LIGHT_POINT) {
        // Point light - inverse square falloff
        let to_light = light.dir_or_pos - world_pos;
        let distance = length(to_light);
        light_dir = to_light / distance;

        // Inverse square with range clamping
        let dist_ratio = min(distance / light.range, 1.0);
        attenuation = 1.0 / (distance * distance + 1.0);
        attenuation *= 1.0 - (dist_ratio * dist_ratio);
    } else if (light.kind == LIGHT_SPOT) {
        // Spot light - inverse square + cone attenuation
        let to_light = light.dir_or_pos - world_pos;
        let distance = length(to_light);
        light_dir = to_light / distance;

        // Distance attenuation
        let dist_ratio = min(distance / light.range, 1.0);
        attenuation = 1.0 / (distance * distance + 1.0);
        attenuation *= 1.0 - (dist_ratio * dist_ratio);

        // Cone attenuation (smoothstep between inner and outer)
        // TODO P0-S1: Implement spot light direction and cone angles
        // For now, assume no cone attenuation
    } else {
        // Environment light - no direct lighting
        light_dir = vec3<f32>(0.0, 1.0, 0.0);
        attenuation = 0.0;
    }

    return vec2<f32>(length(vec3<f32>(light_dir.x, 0.0, light_dir.z)), attenuation);
}

/// Evaluate direct lighting from a light source
fn eval_direct_light(
    light: Light,
    mat: MaterialShading,
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    albedo: vec3<f32>,
    shadow: f32,
) -> vec3<f32> {
    if (light.kind == LIGHT_ENVIRONMENT) {
        // Environment lights don't contribute to direct lighting
        return vec3<f32>(0.0);
    }

    let light_vec = get_light_vector(light, world_pos);
    let light_dir = normalize(vec3<f32>(sign(light_vec.x), 1.0, sign(light_vec.x))); // Simplified for now
    let attenuation = light_vec.y;

    let n_dot_l = max(dot(normal, light_dir), 0.0);

    if (n_dot_l <= 0.0) {
        return vec3<f32>(0.0);
    }

    // Evaluate BRDF
    let brdf = eval_brdf(normal, view_dir, light_dir, albedo, mat);

    // Combine: light_color * intensity * attenuation * brdf * n_dot_l * shadow
    let radiance = light.color * light.intensity * attenuation * shadow;

    return brdf * radiance * n_dot_l;
}

// ===================
// Shadow Functions (P0)
// ===================

// TODO P0-S3: Implement shadow_vis function
// For now, return 1.0 (no shadows)
fn shadow_vis(world_pos: vec3<f32>, normal: vec3<f32>, shadow_settings: ShadowSettings) -> f32 {
    // if (shadow_settings.tech == SHADOW_HARD) {
    //     return hard_shadow(world_pos, shadow_settings.bias, shadow_settings.normal_bias);
    // } else if (shadow_settings.tech == SHADOW_PCF) {
    //     return pcf_shadow(world_pos, shadow_settings.bias, shadow_settings.normal_bias, shadow_settings.softness);
    // }
    return 1.0; // No shadows yet
}

// ==============
// IBL Functions (P0)
// ==============

// TODO P0-S0: Implement IBL evaluation
// For now, return ambient
fn eval_ibl(
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    albedo: vec3<f32>,
    mat: MaterialShading,
    gi: GiSettings,
) -> vec3<f32> {
    if (gi.tech != GI_IBL) {
        return vec3<f32>(0.0);
    }

    // Simple ambient approximation for P0
    // TODO: Sample irradiance and specular cubemaps
    let ambient = albedo * gi.ibl_intensity * 0.03;

    return ambient;
}

// ====================
// Main Lighting Function
// ====================

/// Calculate final lit color for a surface point
fn calculate_lighting(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    albedo: vec3<f32>,
    light: Light,
    mat: MaterialShading,
    shadow_settings: ShadowSettings,
    gi: GiSettings,
    atmo: Atmosphere,
) -> vec3<f32> {
    // Calculate shadow visibility
    let shadow = shadow_vis(world_pos, normal, shadow_settings);

    // Direct lighting
    let direct = eval_direct_light(light, mat, world_pos, normal, view_dir, albedo, shadow);

    // Indirect lighting (IBL)
    let indirect = eval_ibl(normal, view_dir, albedo, mat, gi);

    // Combine
    var final_color = direct + indirect;

    // Apply exposure
    final_color *= atmo.exposure;

    // TODO P0-S4: Apply fog based on atmo.fog_density

    return final_color;
}
