// src/shaders/lighting_ibl.wgsl
// Shared IBL evaluator for split-sum approximation
// Provides unified eval_ibl function used by all PBR shading paths
// RELEVANT FILES: src/shaders/pbr.wgsl, src/shaders/terrain_pbr_pom.wgsl, src/core/ibl.rs
//
// Global constraints (P4 spec):
// - @group(2) @binding(0) envSpecular : texture_cube<f32>
// - @group(2) @binding(1) envIrradiance : texture_cube<f32>
// - @group(2) @binding(2) envSampler  : sampler (filtering + clamp-to-edge)
// - @group(2) @binding(3) brdfLUT     : texture_2d<f32>
//
// Implementation: Lambert irradiance + GGX prefilter × BRDF LUT (split-sum)
// No random sampling at runtime (all precomputed)
// Note: PI and saturate are expected to be defined by including files (e.g., via lighting.wgsl -> brdf/common.wgsl)

// Unified IBL bindings (group(2) as per spec)
@group(2) @binding(0) var envSpecular: texture_cube<f32>;
@group(2) @binding(1) var envIrradiance: texture_cube<f32>;
@group(2) @binding(2) var envSampler: sampler;
@group(2) @binding(3) var brdfLUT: texture_2d<f32>;

/// Fresnel-Schlick approximation with roughness.
/// TERRA-DETERMINATA: operand order pinned; the pow(., 5.0) transcendental is
/// routed through det_pow so the lowering choice is centralized.
fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let gloss_max = max(vec3<f32>(1.0 - roughness), f0);
    let delta = gloss_max - f0;
    let fresnel_term = det_pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
    let scaled = delta * fresnel_term;
    return f0 + scaled;
}

/// Evaluate IBL contribution using split-sum approximation
/// 
/// Parameters:
/// - n: surface normal (world space)
/// - v: view direction (world space, normalized)
/// - base_color: material base color (albedo)
/// - metallic: material metallic factor [0,1]
/// - roughness: material roughness [0,1]
/// - f0: Fresnel reflectance at normal incidence
///
/// Returns: IBL contribution (diffuse + specular) in linear space
fn eval_ibl(
    n: vec3<f32>,
    v: vec3<f32>,
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    f0: vec3<f32>,
) -> vec3<f32> {
    // TERRA-DETERMINATA determinism contract for this function:
    //  - every dot/reflect reduction goes through det_* (fixed reduction tree)
    //  - every mul-add chain is spelled out in pinned left-to-right order
    //  - there is no sample loop: the split-sum is exactly three texture
    //    lookups in fixed program order, each with an EXPLICIT LOD
    //    (textureSampleLevel), so mip selection is never driver-chosen.
    //  - no conditional reorders any evaluation based on runtime values.

    // Clamp inputs for numeric safety
    let n_dot_v = saturate(det_dot3(n, v));
    let roughness_clamped = saturate(roughness);

    // Calculate reflection direction (pinned reflect: i - 2*dot(n,i)*n)
    let reflection_dir = det_reflect3(-v, n);

    // Fresnel term for IBL
    let F_ibl = fresnel_schlick_roughness(n_dot_v, f0, roughness_clamped);

    // Diffuse IBL (Lambertian)
    // kD = (1 - kS) * (1 - metallic)
    let kS_ibl = F_ibl;
    let one_minus_ks = vec3<f32>(1.0) - kS_ibl;
    let kD_ibl = one_minus_ks * (1.0 - metallic);

    // Sample irradiance cubemap (explicit LOD 0)
    let irradiance = textureSampleLevel(envIrradiance, envSampler, n, 0.0).rgb;
    let diffuse_albedo = kD_ibl * base_color;
    let diffuse_ibl = diffuse_albedo * irradiance;

    // Specular IBL (split-sum approximation)
    // Map roughness to mip level: mip = roughness² * (mipCount-1)
    // For prefiltered environment, we use roughness directly to select mip
    let roughness_sq = roughness_clamped * roughness_clamped;
    let mip_level = roughness_sq * 9.0; // Assume 10 mips (0-9)

    // Sample prefiltered specular cubemap (explicit LOD)
    let prefiltered_color = textureSampleLevel(envSpecular, envSampler, reflection_dir, mip_level).rgb;

    // Sample BRDF LUT (explicit LOD 0, fixed operand order)
    let brdf_lut_uv = vec2<f32>(n_dot_v, roughness_clamped);
    let brdf_lut = textureSampleLevel(brdfLUT, envSampler, brdf_lut_uv, 0.0).rg;

    // Split-sum: prefiltered_color * (F0 * scale + bias), scale/bias pinned
    let split_sum = det_fma3(F_ibl, vec3<f32>(brdf_lut.x), vec3<f32>(brdf_lut.y));
    let specular_ibl = prefiltered_color * split_sum;

    // Combine diffuse and specular (single pinned add)
    return diffuse_ibl + specular_ibl;
}

