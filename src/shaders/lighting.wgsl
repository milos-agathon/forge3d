// src/shaders/lighting.wgsl
// P0 Lighting System: Lights, BRDFs, Shadows, and IBL
// Matches Rust types in src/lighting/types.rs
//
// Bind Groups and Layouts (when used by unified lighting pipelines):
// - @group(0): Included from lights.wgsl
//   - @binding(3): storage<array<LightGPU>> - Light array (from lights.wgsl)
//   - @binding(4): uniform<LightMetadata> - Light count and frame metadata (from lights.wgsl)
//   - @binding(5): uniform<vec4<f32>> - Environment lighting parameters (from lights.wgsl)
// - @group(2): IBL environment textures (unified lighting)
//   - @binding(0): texture_cube<f32> - IBL specular environment
//   - @binding(1): texture_cube<f32> - IBL irradiance cube
//   - @binding(2): sampler - IBL environment sampler
//   - @binding(3): texture_2d<f32> - IBL BRDF LUT
//
// Note: This module defines ShadingParamsGPU and BRDF constants, then includes brdf/dispatch.wgsl.
// Individual pipelines (mesh PBR, terrain) use different group layouts and bind ShadingParamsGPU at their own locations.
// Mesh PBR: @group(0) @binding(2) for ShadingParamsGPU
// Terrain: @group(0) @binding(5) for TerrainShadingUniforms (terrain-specific, P2-05 will bridge to ShadingParamsGPU)

#include "lights.wgsl"

// BRDF models (matches Rust BrdfModel enum)
const BRDF_LAMBERT: u32 = 0u;
const BRDF_PHONG: u32 = 1u;
const BRDF_BLINN_PHONG: u32 = 2u;
const BRDF_OREN_NAYAR: u32 = 3u;
const BRDF_COOK_TORRANCE_GGX: u32 = 4u;
const BRDF_COOK_TORRANCE_BECKMANN: u32 = 5u;
const BRDF_DISNEY_PRINCIPLED: u32 = 6u;
const BRDF_ASHIKHMIN_SHIRLEY: u32 = 7u;
const BRDF_WARD: u32 = 8u;
const BRDF_TOON: u32 = 9u;
const BRDF_MINNAERT: u32 = 10u;
const BRDF_SUBSURFACE: u32 = 11u;
const BRDF_HAIR: u32 = 12u;

// Shadow techniques (matches Rust ShadowTechnique enum)
const SHADOW_HARD: u32 = 0u;
const SHADOW_PCF: u32 = 1u;

// GI techniques (matches Rust GiTechnique enum)
const GI_NONE: u32 = 0u;
const GI_IBL: u32 = 1u;

struct ShadingParamsGPU {
    brdf: u32,
    metallic: f32,
    roughness: f32,
    sheen: f32,
    clearcoat: f32,
    subsurface: f32,
    anisotropy: f32,
    _pad: f32,
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

#include "brdf/dispatch.wgsl"

@group(2) @binding(0)
var ibl_env_specular: texture_cube<f32>;

@group(2) @binding(1)
var ibl_env_irradiance: texture_cube<f32>;

@group(2) @binding(2)
var ibl_env_sampler: sampler;

@group(2) @binding(3)
var ibl_brdf_lut: texture_2d<f32>;

// ======================
// Light Functions (P0)
// ======================

/// Light direction and attenuation result
struct LightResult {
    direction: vec3<f32>,
    attenuation: f32,
};

/// Calculate light direction and attenuation for a light
fn get_light_contribution(light: LightGPU, world_pos: vec3<f32>) -> LightResult {
    var result: LightResult;

    if (light.type_ == LIGHT_DIRECTIONAL) {
        // Directional light - constant direction, no falloff
        result.direction = normalize(-light.dir_ws);
        result.attenuation = 1.0;
    } else if (light.type_ == LIGHT_POINT || light.type_ == LIGHT_AREA_SPHERE) {
        // Point light - inverse square falloff with range
        let to_light = light.pos_ws - world_pos;
        let distance = length(to_light);
        result.direction = to_light / max(distance, 0.0001);

        // Inverse square attenuation with smooth range cutoff
        let range = max(light.range, 0.0001);
        let dist_ratio = clamp(distance / range, 0.0, 1.0);
        let range_attenuation = 1.0 - (dist_ratio * dist_ratio);
        result.attenuation = range_attenuation / max(distance * distance, 0.0001);
    } else if (light.type_ == LIGHT_SPOT) {
        // Spot light - point light with cone attenuation
        let to_light = light.pos_ws - world_pos;
        let distance = length(to_light);
        result.direction = to_light / max(distance, 0.0001);

        // Distance attenuation (same as point light)
        let range = max(light.range, 0.0001);
        let dist_ratio = clamp(distance / range, 0.0, 1.0);
        let range_attenuation = 1.0 - (dist_ratio * dist_ratio);
        result.attenuation = range_attenuation / max(distance * distance, 0.0001);

        // Cone attenuation (smoothstep between inner and outer angles)
        let spot_dir = normalize(light.dir_ws);
        let cos_angle = dot(spot_dir, -result.direction);
        let inner_cos = light.cone_cos.x;
        let outer_cos = light.cone_cos.y;

        // Smooth transition between inner and outer cone
        let cone_attenuation = smoothstep(outer_cos, inner_cos, cos_angle);
        result.attenuation *= cone_attenuation;
    } else if (light.type_ == LIGHT_AREA_RECT || light.type_ == LIGHT_AREA_DISK) {
        result.direction = normalize(-light.dir_ws);
        result.attenuation = 1.0;
    } else {
        // Environment light - no direct lighting
        result.direction = vec3<f32>(0.0, 1.0, 0.0);
        result.attenuation = 0.0;
    }

    return result;
}

/// Evaluate direct lighting from a light source
fn eval_direct_light(
    light: LightGPU,
    mat: ShadingParamsGPU,
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    albedo: vec3<f32>,
    shadow: f32,
) -> vec3<f32> {
    if (light.type_ == LIGHT_ENVIRONMENT) {
        // Environment lights don't contribute to direct lighting
        return vec3<f32>(0.0);
    }

    // Get light contribution (direction and attenuation)
    let light_contrib = get_light_contribution(light, world_pos);
    let light_dir = light_contrib.direction;
    let attenuation = light_contrib.attenuation;

    // Early out if light doesn't reach this point
    if (attenuation <= 0.0001) {
        return vec3<f32>(0.0);
    }

    // Calculate N·L
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

// Bind group for shadow mapping (to be declared where needed)
// @group(1) @binding(0) var<uniform> uShadowMatrix: mat4x4<f32>;
// @group(1) @binding(1) var tShadowMap: texture_depth_2d;
// @group(1) @binding(2) var sShadowMap: sampler_comparison;  // or sampler for PCF

/// Transform world position to light clip space and apply bias
fn world_to_shadow_space(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    shadow_matrix: mat4x4<f32>,
    bias: f32,
    normal_bias: f32,
) -> vec3<f32> {
    // Apply normal bias by offsetting position along normal
    let biased_pos = world_pos + normal * normal_bias;

    // Transform to light clip space
    let light_clip = shadow_matrix * vec4<f32>(biased_pos, 1.0);

    // Perspective divide
    var light_ndc = light_clip.xyz / light_clip.w;

    // Transform from NDC [-1,1] to texture coords [0,1]
    light_ndc.x = light_ndc.x * 0.5 + 0.5;
    light_ndc.y = -light_ndc.y * 0.5 + 0.5;  // Flip Y for texture coords

    // Apply depth bias
    light_ndc.z = light_ndc.z - bias;

    return light_ndc;
}

/// Hard shadow sampling (single sample with hardware comparison)
fn hard_shadow(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    shadow_matrix: mat4x4<f32>,
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    bias: f32,
    normal_bias: f32,
) -> f32 {
    let shadow_coord = world_to_shadow_space(world_pos, normal, shadow_matrix, bias, normal_bias);

    // Check if in shadow map bounds
    if (shadow_coord.x < 0.0 || shadow_coord.x > 1.0 ||
        shadow_coord.y < 0.0 || shadow_coord.y > 1.0 ||
        shadow_coord.z < 0.0 || shadow_coord.z > 1.0) {
        return 1.0;  // Outside shadow map = lit
    }

    // Hardware comparison sample (returns 0.0 if in shadow, 1.0 if lit)
    return textureSampleCompare(shadow_map, shadow_sampler, shadow_coord.xy, shadow_coord.z);
}

/// PCF (Percentage Closer Filtering) shadow sampling
fn pcf_shadow(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    shadow_matrix: mat4x4<f32>,
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler,
    bias: f32,
    normal_bias: f32,
    softness: f32,
) -> f32 {
    let shadow_coord = world_to_shadow_space(world_pos, normal, shadow_matrix, bias, normal_bias);

    // Check if in shadow map bounds
    if (shadow_coord.x < 0.0 || shadow_coord.x > 1.0 ||
        shadow_coord.y < 0.0 || shadow_coord.y > 1.0 ||
        shadow_coord.z < 0.0 || shadow_coord.z > 1.0) {
        return 1.0;  // Outside shadow map = lit
    }

    // Get shadow map dimensions
    let shadow_map_size = vec2<f32>(textureDimensions(shadow_map));
    let texel_size = 1.0 / shadow_map_size;

    // PCF kernel size based on softness (1.0 = 3x3, 2.0 = 5x5)
    let kernel_radius = i32(softness);
    let kernel_size = kernel_radius * 2 + 1;

    var shadow_sum = 0.0;
    var sample_count = 0.0;

    // Sample in a grid pattern
    for (var y = -kernel_radius; y <= kernel_radius; y = y + 1) {
        for (var x = -kernel_radius; x <= kernel_radius; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            let sample_coord = shadow_coord.xy + offset;

            // Sample depth from shadow map
            let shadow_depth = textureSample(shadow_map, shadow_sampler, sample_coord);

            // Compare with current depth
            if (shadow_coord.z <= shadow_depth) {
                shadow_sum = shadow_sum + 1.0;
            }

            sample_count = sample_count + 1.0;
        }
    }

    return shadow_sum / sample_count;
}

/// Main shadow visibility function (dispatches to hard or PCF)
fn shadow_vis(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    shadow_matrix: mat4x4<f32>,
    shadow_map: texture_depth_2d,
    shadow_sampler_cmp: sampler_comparison,
    shadow_sampler: sampler,
    shadow_settings: ShadowSettings,
) -> f32 {
    if (shadow_settings.tech == SHADOW_HARD) {
        return hard_shadow(
            world_pos, normal, shadow_matrix, shadow_map, shadow_sampler_cmp,
            shadow_settings.bias, shadow_settings.normal_bias
        );
    } else if (shadow_settings.tech == SHADOW_PCF) {
        return pcf_shadow(
            world_pos, normal, shadow_matrix, shadow_map, shadow_sampler,
            shadow_settings.bias, shadow_settings.normal_bias, shadow_settings.softness
        );
    }
    return 1.0;  // No shadows
}

fn rotate_y(vec3_value: vec3<f32>, angle_rad: f32) -> vec3<f32> {
    let s = sin(angle_rad);
    let c = cos(angle_rad);
    return vec3<f32>(
        vec3_value.x * c - vec3_value.z * s,
        vec3_value.y,
        vec3_value.x * s + vec3_value.z * c,
    );
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn eval_ibl(
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    albedo: vec3<f32>,
    mat: ShadingParamsGPU,
    gi: GiSettings,
) -> vec3<f32> {
    if (gi.tech != GI_IBL) {
        return vec3<f32>(0.0);
    }

    let rotation = gi.ibl_rotation_deg * (3.14159265 / 180.0);
    let rotated_normal = rotate_y(normalize(normal), rotation);
    let rotated_view = rotate_y(normalize(view_dir), rotation);
    let n_dot_v = max(dot(rotated_normal, rotated_view), 0.0);

    let diffuse_radiance = textureSample(ibl_env_irradiance, ibl_env_sampler, rotated_normal).rgb;

    let reflection_dir = reflect(-rotated_view, rotated_normal);
    let mip_levels = max(f32(textureNumLevels(ibl_env_specular)), 1.0);
    let lod = clamp(mat.roughness * (mip_levels - 1.0), 0.0, mip_levels - 1.0);
    let specular_radiance =
        textureSampleLevel(ibl_env_specular, ibl_env_sampler, reflection_dir, lod).rgb;

    let brdf = textureSample(ibl_brdf_lut, ibl_env_sampler, vec2<f32>(n_dot_v, mat.roughness)).rg;

    let clamped_albedo = clamp(albedo, vec3<f32>(0.0), vec3<f32>(1.0));
    let metallic = clamp(mat.metallic, 0.0, 1.0);
    let f0 = mix(vec3<f32>(0.04), clamped_albedo, metallic);

    let fresnel = fresnel_schlick(n_dot_v, f0);
    let k_s = fresnel;
    let k_d = (vec3<f32>(1.0) - k_s) * (1.0 - metallic);

    let diffuse = diffuse_radiance * clamped_albedo;
    let specular = specular_radiance * (fresnel * brdf.x + brdf.y);

    return (k_d * diffuse + specular) * gi.ibl_intensity;
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
    light: LightGPU,
    mat: ShadingParamsGPU,
    shadow_settings: ShadowSettings,
    shadow_matrix: mat4x4<f32>,
    shadow_map: texture_depth_2d,
    shadow_sampler_cmp: sampler_comparison,
    shadow_sampler: sampler,
    gi: GiSettings,
    atmo: Atmosphere,
) -> vec3<f32> {
    // Calculate shadow visibility
    let shadow = shadow_vis(
        world_pos, normal, shadow_matrix, shadow_map,
        shadow_sampler_cmp, shadow_sampler, shadow_settings
    );

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
