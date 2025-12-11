// src/shaders/terrain_pbr_pom.wgsl
// Terrain PBR + POM shader implementing normal, triplanar, and BRDF logic
// Exists to light the terrain renderer milestone with placeholder resources until assets land
// RELEVANT FILES: src/terrain_renderer.rs, src/terrain_render_params.rs, src/overlay_layer.rs, terrain_demo_task_breakdown.md
//
// Bind Groups and Layouts:
// - @group(0): Terrain uniforms and textures
//   - @binding(0): uniform<TerrainUniforms> - View/proj matrices, sun exposure, spacing, height exaggeration
//   - @binding(1): texture_2d<f32> - Height texture
//   - @binding(2): sampler - Height sampler
//   - @binding(3): texture_2d_array<f32> - Material albedo texture array
//   - @binding(4): sampler - Material sampler
//   - @binding(5): uniform<TerrainShadingUniforms> - Triplanar, POM, layer heights/roughness/metallic, light params, clamps
//   - @binding(6): texture_2d<f32> - Colormap texture
//   - @binding(7): sampler - Colormap sampler
//   - @binding(8): uniform<OverlayUniforms> - Overlay domain, blend mode, albedo mode, colormap strength, gamma
// - @group(1): Light buffer (P1-06)
//   - @binding(3): storage<array<Light>> - Light array
//   - @binding(4): uniform<LightMetadata> - Light count, frame index, sequence seed
//   - @binding(5): uniform<EnvironmentParams> - Ambient color
// - @group(2): IBL textures
//   - @binding(0): texture_cube<f32> - IBL specular cube map
//   - @binding(1): texture_cube<f32> - IBL irradiance cube map
//   - @binding(2): sampler - IBL environment sampler
//   - @binding(3): texture_2d<f32> - IBL BRDF LUT
//   - @binding(4): uniform<IblUniforms> - IBL intensity, rotation (sin/cos theta), specular mip count
//
// Note: TerrainShadingUniforms (@group(0) @binding(5)) contains terrain-specific shading knobs.
// P2-05: Optional BRDF dispatch hook (disabled by default, preserves current terrain look)
// No binding collisions with mesh PBR pipeline (which uses different group layouts).

// P2-05: Include lighting.wgsl for optional eval_brdf dispatch
// Note: This adds BRDF constants and eval_brdf function, but terrain uses calculate_pbr_brdf by default
#include "lighting.wgsl"
// P4 spec: Include unified IBL evaluator (group(2) bindings)
// Note: lighting_ibl.wgsl defines PI, so we don't redefine it here
#include "lighting_ibl.wgsl"

// P2-05: Optional BRDF dispatch flag (default: false = use calculate_pbr_brdf for current look)
// Set to true to enable eval_brdf dispatch, allowing BRDF model switching on terrain
const TERRAIN_USE_BRDF_DISPATCH: bool = false;
const TERRAIN_BRDF_MODEL: u32 = BRDF_COOK_TORRANCE_GGX;  // Used when TERRAIN_USE_BRDF_DISPATCH = true

// P3-10/P1: CSM shadow sampling for terrain direct lighting
// Single source of truth for enabling terrain shadows
const TERRAIN_SHADOWS_ENABLED: bool = true;
const TERRAIN_USE_SHADOWS: bool = TERRAIN_SHADOWS_ENABLED;

// P1-Shadow: Shadow intensity tuning
// SHADOW_MIN: minimum brightness in fully shadowed areas (0.0 = pitch black, 0.3 = soft shadows)
// SHADOW_IBL_FACTOR: how much IBL diffuse is reduced in shadow (0.0 = no effect, 1.0 = full shadow)
// 
// P2-S3: shadow factor must be clamped to [0.30, 1.0] - hard lower bound
// P2-S1: ambient_floor must be in [0.22, 0.38] range
// Spec A-01: ambient floor ensures even fully shadowed terrain has L >= 0.10 after tonemap
const SHADOW_MIN: f32 = 0.30;        // P2-S3: Hard lower bound for shadow factor
const SHADOW_IBL_FACTOR: f32 = 0.20; // IBL diffuse reduced by 20% in shadow

// P2-S1: Ambient floor constants for terrain lighting
// These ensure valleys don't crush to black while maintaining proper contrast
const AMBIENT_FLOOR_MIN: f32 = 0.22;
const AMBIENT_FLOOR_MAX: f32 = 0.38;
const AMBIENT_FLOOR: f32 = 0.22;     // P3: Lower ambient for deeper shadows

// P1-Shadow Debug: Set to true to visualize shadow cascade coverage
// Color codes terrain by which cascade is used: Red=0, Green=1, Blue=2, Yellow=3
// Note: These are compile-time constants. For runtime override, use debug_mode in CsmUniforms:
//   csm_uniforms.debug_mode == 1 -> cascade boundary overlay
//   csm_uniforms.debug_mode == 2 -> raw shadow visibility overlay
// Environment variable: FORGE3D_TERRAIN_SHADOW_DEBUG="cascades" or "raw"
const DEBUG_SHADOW_CASCADES: bool = false;
const DEBUG_SHADOW_RAW: bool = false;  // Show raw shadow visibility as grayscale

// Shadow debug mode constants (must match Rust side)
const SHADOW_DEBUG_NONE: u32 = 0u;
const SHADOW_DEBUG_CASCADES: u32 = 1u;
const SHADOW_DEBUG_RAW: u32 = 2u;

// ──────────────────────────────────────────────────────────────────────────
// Debug Mode Constants — "truth serum" diagnostics for water/IBL/PBR
// These are forensic modes, not visual improvements. Each answers ONE question.
// ──────────────────────────────────────────────────────────────────────────
// Water debug modes (4-6):
const DBG_WATER_MASK_BINARY: u32 = 4u;  // CYAN = water, MAGENTA = land (uses SAME is_water as main path)
const DBG_WATER_MASK_RAW: u32 = 5u;     // Grayscale [0,1], RED=<0, YELLOW=>1, GREEN=NaN/Inf
const DBG_IBL_ONLY: u32 = 6u;           // IBL contribution only (same tonemap as normal frames)
// PBR debug modes (7-12): Proof pack for microfacet BRDF correctness
const DBG_PBR_DIFFUSE_ONLY: u32 = 7u;   // Diffuse IBL term only (no specular, no sun)
const DBG_PBR_SPECULAR_ONLY: u32 = 8u;  // Specular IBL term only (no diffuse, no sun)
const DBG_PBR_FRESNEL: u32 = 9u;        // Fresnel term F as grayscale (average of RGB)
const DBG_PBR_NDOTV: u32 = 10u;         // N.V (view angle) as grayscale
const DBG_PBR_ROUGHNESS: u32 = 11u;     // Roughness value as grayscale (after any multiplier)
const DBG_PBR_ENERGY: u32 = 12u;        // Raw (diffuse + specular) luminance before tonemap (for energy histogram)
// Recomposition proof modes (13-16): Prove that IBL = diffuse + specular in linear space
const DBG_PBR_LINEAR_COMBINED: u32 = 13u;  // Linear unclamped (diff+spec), RGB encoded [0,4] -> [0,1]
const DBG_PBR_LINEAR_DIFFUSE: u32 = 14u;   // Linear unclamped diffuse only, RGB encoded [0,4] -> [0,1]
const DBG_PBR_LINEAR_SPECULAR: u32 = 15u;  // Linear unclamped specular only, RGB encoded [0,4] -> [0,1]
const DBG_PBR_RECOMP_ERROR: u32 = 16u;     // abs(ibl_total - (diff+spec)) heatmap, amplified 100x
// SpecAA stress test mode (17): High-frequency sparkle detection
const DBG_SPECAA_SPARKLE: u32 = 17u;       // Specular with synthetic high-freq normal perturbation
// POM debug mode (18): Parallax offset magnitude visualization
const DBG_POM_OFFSET_MAG: u32 = 18u;       // Grayscale POM offset magnitude (0=none, white=max offset)
// SpecAA sigma2 debug mode (19): Variance visualization for SpecAA diagnostics
const DBG_SPECAA_SIGMA2: u32 = 19u;        // Grayscale sigma² (0=no variance, white=high variance)
// SpecAA sparkle sigma2 debug mode (20): Variance on sparkle-perturbed normal
const DBG_SPECAA_SPARKLE_SIGMA2: u32 = 20u; // Shows variance computed on perturbed normal
// Triplanar debug modes (21-22): Proof pack for triplanar mapping correctness
const DBG_TRIPLANAR_WEIGHTS: u32 = 21u;     // RGB = x/y/z blend weights (sum to 1)
const DBG_TRIPLANAR_CHECKER: u32 = 22u;     // Procedural checker to expose UV stretching
// Flake diagnosis modes (23-27): Milestone 1-3 proof pack
const DBG_FLAKE_NO_SPECULAR: u32 = 23u;     // Direct lighting only (no IBL specular)
const DBG_FLAKE_NO_HEIGHT_NORMAL: u32 = 24u; // Use base_normal instead of height_normal
const DBG_FLAKE_DDXDDY_NORMAL: u32 = 25u;   // Use n_dd = cross(dpdx, dpdy) as shading normal
const DBG_FLAKE_HEIGHT_LOD: u32 = 26u;      // Visualize computed height LOD
const DBG_FLAKE_NORMAL_BLEND: u32 = 27u;    // Visualize effective normal_blend after LOD fade
// P5: Raw SSAO visualization
const DBG_RAW_SSAO: u32 = 28u;

struct TerrainUniforms {
    view : mat4x4<f32>,
    proj : mat4x4<f32>,
    sun_exposure : vec4<f32>,
    spacing_h_exag : vec4<f32>,
    pad_tail : vec4<f32>,
};

struct TerrainShadingUniforms {
    triplanar_params : vec4<f32>, // x=scale, y=blend_sharpness, z=normal_strength, w=pom_scale
    pom_steps : vec4<f32>,        // x=min_steps, y=max_steps, z=refine_steps, w=flags
    layer_heights : vec4<f32>,    // normalized centers per layer
    layer_roughness : vec4<f32>,
    layer_metallic : vec4<f32>,
    layer_control : vec4<f32>,    // x=layer_count, y=blend_half_width, z=lod_bias, w=lod0_bias
    light_params : vec4<f32>,     // rgb = light color * intensity, w=exposure
    clamp0 : vec4<f32>,           // height_min, height_max, slope_min, slope_max
    clamp1 : vec4<f32>,           // ambient_min, ambient_max, shadow_min, shadow_max
    clamp2 : vec4<f32>,           // occlusion_min, occlusion_max, lod_level, anisotropy
    height_curve : vec4<f32>,     // x=mode, y=strength, z=power, w=reserved
};

struct OverlayUniforms {
    params0 : vec4<f32>, // domain_min, inv_range, overlay_strength, offset
    params1 : vec4<f32>, // blend_mode, debug_mode, albedo_mode, colormap_strength
    params2 : vec4<f32>, // gamma, roughness_mult, spec_aa_enabled, specaa_sigma_scale
    params3 : vec4<f32>, // P5: ao_weight, ao_fallback_enabled, pad, pad
    // P6: Micro-detail parameters
    params4 : vec4<f32>, // detail_enabled, detail_scale, detail_normal_strength, detail_albedo_noise
    params5 : vec4<f32>, // detail_fade_start, detail_fade_end, pad, pad
};

struct IblUniforms {
    intensity : f32,
    sin_theta : f32,
    cos_theta : f32,
    specular_mip_count : f32,
};

@group(0) @binding(0)
var<uniform> u_terrain : TerrainUniforms;

@group(0) @binding(1)
var height_tex : texture_2d<f32>;

@group(0) @binding(2)
var height_samp : sampler;

@group(0) @binding(3)
var material_albedo_tex : texture_2d_array<f32>;

@group(0) @binding(4)
var material_samp : sampler;

@group(0) @binding(5)
var<uniform> u_shading : TerrainShadingUniforms;

@group(0) @binding(6)
var colormap_tex : texture_2d<f32>;

@group(0) @binding(7)
var colormap_samp : sampler;

@group(0) @binding(8)
var<uniform> u_overlay : OverlayUniforms;

@group(0) @binding(9)
var height_curve_lut_tex : texture_2d<f32>;

@group(0) @binding(10)
var height_curve_lut_samp : sampler;

@group(0) @binding(11)
var water_mask_tex : texture_2d<f32>;

// P5: AO debug/fallback texture (raw SSAO or coarse height AO)
@group(0) @binding(12)
var ao_debug_tex : texture_2d<f32>;

@group(0) @binding(13)
var ao_debug_samp : sampler;

// P1-06: Light buffer bindings (@group(1))
struct Light {
    light_type: u32,
    flags: u32,
    position: vec3<f32>,
    direction: vec3<f32>,
    color: vec3<f32>,
    intensity: f32,
    range: f32,
    inner_angle: f32,
    outer_angle: f32,
    area_width: f32,
    area_height: f32,
    env_texture_index: u32,
};

// Note: LightMetadata from lights.wgsl is used via lighting.wgsl, but terrain uses a different structure
// Renamed to avoid conflict
struct TerrainLightMetadata {
    light_count: u32,
    frame_index: u32,
    sequence_seed: vec2<f32>,
};

struct EnvironmentParams {
    ambient: vec3<f32>,
    padding: f32,
};

@group(1) @binding(3)
var<storage, read> terrain_lights: array<Light>;

@group(1) @binding(4)
var<uniform> light_metadata: TerrainLightMetadata;

@group(1) @binding(5)
var<uniform> environment_params: EnvironmentParams;

// IBL bindings are declared in lighting_ibl.wgsl (P4 spec: group(2) bindings 0-3)
// @group(2) @binding(0) envSpecular : texture_cube<f32>
// @group(2) @binding(1) envIrradiance : texture_cube<f32>
// @group(2) @binding(2) envSampler : sampler
// @group(2) @binding(3) brdfLUT : texture_2d<f32>

// Terrain-specific IBL uniforms (rotation, intensity, mip count)
@group(2) @binding(4)
var<uniform> u_ibl : IblUniforms;

// P3-10: Optional shadow bindings at @group(3) (only used when TERRAIN_USE_SHADOWS = true)
// These mirror the shadow bindings from shadows.wgsl but at a different group to avoid IBL conflict
// Simplified shadow cascade data for terrain
// Must match Rust struct in crate::core::shadow_mapping::CsmCascadeData
struct ShadowCascade {
    light_projection: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,  // Pre-computed projection * view for consistency
    near_distance: f32,
    far_distance: f32,
    texel_size: f32,
    _padding: f32,
}

// Simplified CsmUniforms for terrain shadow sampling
// Only contains fields actually used by terrain shadow code
// Must match Rust struct in crate::core::shadow_mapping::CsmUniforms
struct CsmUniforms {
    light_direction: vec4<f32>,        // 16 bytes, offset 0
    light_view: mat4x4<f32>,           // 64 bytes, offset 16
    cascades: array<ShadowCascade, 4>, // 576 bytes, offset 80 (4 * 144: 2 mat4x4 + 4 floats each)
    cascade_count: u32,                // 4 bytes, offset 656
    pcf_kernel_size: u32,              // 4 bytes, offset 660
    depth_bias: f32,                   // 4 bytes, offset 664
    slope_bias: f32,                   // 4 bytes, offset 668
    shadow_map_size: f32,              // 4 bytes, offset 672
    debug_mode: u32,                   // 4 bytes, offset 676
    peter_panning_offset: f32,         // 4 bytes, offset 680
    pcss_light_radius: f32,            // 4 bytes, offset 684
    cascade_blend_range: f32,          // 4 bytes, offset 688
    _padding0: f32,                    // 4 bytes, offset 692
    _padding1: f32,                    // 4 bytes, offset 696
    _padding2: f32,                    // 4 bytes, offset 700 -> total 704 bytes
}

@group(3) @binding(0)
var<uniform> csm_uniforms: CsmUniforms;

@group(3) @binding(1)
var shadow_maps: texture_depth_2d_array;

@group(3) @binding(2)
var shadow_sampler: sampler_comparison;

@group(3) @binding(3)
var moment_maps: texture_2d_array<f32>;

@group(3) @binding(4)
var moment_sampler: sampler;

// ──────────────────────────────────────────────────────────────────────────
// P2: Atmospheric Fog Uniforms (@group(4))
// Height-based fog with tunable density and inscatter color.
// When fog_density = 0.0, fog is a no-op preserving P1 output exactly.
// ──────────────────────────────────────────────────────────────────────────
struct FogUniforms {
    // Fog density coefficient (0.0 = no fog, higher = denser)
    fog_density: f32,
    // Height falloff rate (controls how fog thins at higher altitudes)
    fog_height_falloff: f32,
    // Base height for fog calculation (world-space Y)
    fog_base_height: f32,
    // Camera world-space Y position
    camera_height: f32,
    // Inscatter color (linear RGB, typically sky-tinted)
    fog_inscatter: vec3<f32>,
    // Padding for 16-byte alignment
    _padding: f32,
}

@group(4) @binding(0)
var<uniform> fog_uniforms: FogUniforms;

// ──────────────────────────────────────────────────────────────────────────
// P4: Water Planar Reflection Uniforms (@group(5))
// Deterministic planar reflections with mirrored camera for water surfaces.
// When enable_reflections = 0 or no water, reflections are a no-op.
// ──────────────────────────────────────────────────────────────────────────
struct WaterReflectionUniforms {
    // Reflection view-projection matrix (mirrored camera)
    reflection_view_proj: mat4x4<f32>,
    // Water plane parameters: normal (xyz), distance (w)
    water_plane: vec4<f32>,
    // Reflection params: (intensity, fresnel_power, wave_strength, shore_atten_width)
    reflection_params: vec4<f32>,
    // Camera world position for Fresnel calculation
    camera_world_pos: vec4<f32>,
    // Enable flags: (enable_reflections, debug_mode, resolution, _pad)
    enable_flags: vec4<f32>,
}

@group(5) @binding(0)
var<uniform> water_reflection_uniforms: WaterReflectionUniforms;

@group(5) @binding(1)
var reflection_texture: texture_2d<f32>;

@group(5) @binding(2)
var reflection_sampler: sampler;

// P4: Sample reflection texture with wave-based UV distortion
// Returns (reflection_color, valid) where valid indicates if sample is in bounds
fn sample_water_reflection(
    world_pos: vec3<f32>,
    wave_normal: vec3<f32>,
    shore_distance: f32,
) -> vec4<f32> {
    // Early-out if reflections disabled
    if (water_reflection_uniforms.enable_flags.x < 0.5) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    // Transform world position to reflection clip space
    let reflection_clip = water_reflection_uniforms.reflection_view_proj * vec4<f32>(world_pos, 1.0);
    
    // Perspective divide
    if (abs(reflection_clip.w) < 0.001) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let reflection_ndc = reflection_clip.xyz / reflection_clip.w;
    
    // Convert NDC to UV [0,1]
    var reflection_uv = reflection_ndc.xy * 0.5 + 0.5;
    // Flip Y for texture coordinate system
    reflection_uv.y = 1.0 - reflection_uv.y;
    
    // Wave-based UV distortion
    // Wave normal deviation from flat (0,1,0) creates UV offset
    let wave_strength = water_reflection_uniforms.reflection_params.z;
    let wave_distortion = (wave_normal.xz - vec2<f32>(0.0, 0.0)) * wave_strength;
    
    // Shore attenuation: reduce distortion near shore (calmer water at edges)
    let shore_atten_width = water_reflection_uniforms.reflection_params.w;
    let shore_factor = smoothstep(0.0, shore_atten_width, shore_distance);
    
    // Apply distortion with shore attenuation
    reflection_uv = reflection_uv + wave_distortion * shore_factor;
    
    // Clamp to valid range
    reflection_uv = clamp(reflection_uv, vec2<f32>(0.001), vec2<f32>(0.999));
    
    // Check if in bounds (out of bounds returns zero alpha)
    if (reflection_uv.x < 0.001 || reflection_uv.x > 0.999 ||
        reflection_uv.y < 0.001 || reflection_uv.y > 0.999) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    // Sample reflection texture
    let reflection_color = textureSample(reflection_texture, reflection_sampler, reflection_uv);
    
    return vec4<f32>(reflection_color.rgb, 1.0);
}

// P4: Calculate Fresnel factor for water reflection blending
// Returns 0 at normal incidence (looking straight down), 1 at grazing angles
fn calculate_water_fresnel(view_dir: vec3<f32>, surface_normal: vec3<f32>) -> f32 {
    let fresnel_power = water_reflection_uniforms.reflection_params.y;
    let n_dot_v = max(0.0, dot(surface_normal, view_dir));
    let fresnel = pow(1.0 - n_dot_v, fresnel_power);
    return clamp(fresnel, 0.0, 1.0);
}

// P4: Blend planar reflection with underwater color using Fresnel
fn blend_water_reflection(
    underwater_color: vec3<f32>,
    reflection_color: vec3<f32>,
    reflection_valid: bool,
    fresnel: f32,
    shore_distance: f32,
) -> vec3<f32> {
    if (!reflection_valid) {
        return underwater_color;
    }
    
    let intensity = water_reflection_uniforms.reflection_params.x;
    
    // Shore attenuation: reduce reflection intensity near shore
    // This helps blend water edges more naturally with land
    let shore_atten_width = water_reflection_uniforms.reflection_params.w;
    let shore_blend = smoothstep(0.0, shore_atten_width, shore_distance);
    
    // Final blend factor: Fresnel * intensity * shore_blend
    let blend = fresnel * intensity * shore_blend;
    
    return mix(underwater_color, reflection_color, blend);
}

// P3-10: Shadow sampling functions (simplified for terrain)
// These are lightweight versions for terrain use, gated behind TERRAIN_USE_SHADOWS flag

/// Select cascade based on view-space depth
fn select_cascade_terrain(view_depth: f32) -> u32 {
    let count = csm_uniforms.cascade_count;
    for (var i = 0u; i < count; i = i + 1u) {
        if (view_depth <= csm_uniforms.cascades[i].far_distance) {
            return i;
        }
    }
    return count - 1u;
}

/// Sample shadow map with PCF filtering
fn sample_shadow_pcf_terrain(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    cascade_idx: u32,
) -> f32 {
    // P0 early-out: NoopShadow sentinel via cascade_count == 0
    // Real CSM always has at least one cascade; noop has cascade_count = 0
    if (csm_uniforms.cascade_count == 0u) {
        return 1.0;  // Fully lit when shadows disabled
    }
    // Normal shadow sampling path
    let cascade = csm_uniforms.cascades[cascade_idx];
    
    // Transform to light space using pre-computed combined matrix
    let light_space_pos = cascade.light_view_proj * vec4<f32>(world_pos, 1.0);
    let ndc = light_space_pos.xyz / light_space_pos.w;
    
    // Convert to texture coordinates [0,1]
    // Note: ndc.x and ndc.y are in [-1, 1], need to map to [0, 1]
    // For Y, we flip because texture V=0 is at top, but NDC Y=-1 is at bottom
    let shadow_coords = vec2<f32>(ndc.x * 0.5 + 0.5, ndc.y * -0.5 + 0.5);
    
    // glam's orthographic_rh already outputs Z in [0, 1] range for WebGPU
    // No additional mapping needed - use ndc.z directly
    let depth_01 = ndc.z;
    
    // Check if outside shadow map bounds
    if (shadow_coords.x < 0.0 || shadow_coords.x > 1.0 || shadow_coords.y < 0.0 || shadow_coords.y > 1.0) {
        return 1.0; // No shadow outside bounds
    }
    
    // Apply depth bias to prevent shadow acne
    // Combine constant bias with slope-scaled bias for grazing angles
    let light_dir_norm = normalize(csm_uniforms.light_direction.xyz);
    let n_dot_l = max(dot(normal, light_dir_norm), 0.0);
    let slope_factor = clamp(1.0 - n_dot_l, 0.0, 1.0);
    let bias = csm_uniforms.depth_bias
        + csm_uniforms.slope_bias * slope_factor
        + csm_uniforms.peter_panning_offset;
    let compare_depth = depth_01 - bias;
    
    // PCF/PCSS shadow sampling (kernel size driven by uniforms)
    let kernel_size = i32(max(csm_uniforms.pcf_kernel_size, 1u));
    let radius = kernel_size / 2;
    let texel_size = cascade.texel_size * max(csm_uniforms.pcss_light_radius, 1.0);
    var shadow_sum = 0.0;
    for (var y = -radius; y <= radius; y = y + 1) {
        for (var x = -radius; x <= radius; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            let sample_coords = shadow_coords + offset;
            let depth_sample = textureSampleCompare(
                shadow_maps,
                shadow_sampler,
                sample_coords,
                i32(cascade_idx),
                compare_depth
            );
            shadow_sum = shadow_sum + depth_sample;
        }
    }

    let kernel_area = f32(kernel_size * kernel_size);
    return shadow_sum / kernel_area;
}

/// Normalize world position for shadow calculations
/// Shadow maps use normalized heights [0, h_exag] to match XY scale [-0.5, 0.5]
/// This function computes shadow-normalized position from tex_coord (not interpolated world_pos)
/// CRITICAL: The main shader uses a fullscreen triangle (3 vertices) where interpolated
/// world_position is INCORRECT at most fragments. We must compute both XY and Z from tex_coord.
fn normalize_for_shadow(tex_coord: vec2<f32>) -> vec3<f32> {
    let h_min = u_shading.clamp0.x;
    let h_max = u_shading.clamp0.y;
    let h_exag = u_terrain.spacing_h_exag.z;
    let spacing = u_terrain.spacing_h_exag.x;
    let h_range = max(h_max - h_min, 1e-6);
    
    // Compute world XY from tex_coord (not from interpolated world_position!)
    // This matches what the shadow depth shader does: world_xy = (uv - 0.5) * spacing
    let world_xy = (tex_coord - vec2<f32>(0.5, 0.5)) * spacing;
    
    // Sample height directly from heightmap at this fragment's UV
    // This matches what the shadow depth shader does
    let h_raw = textureSample(height_tex, height_samp, tex_coord).r;
    let h_norm = clamp((h_raw - h_min) / h_range, 0.0, 1.0);
    
    // Apply height curve (must match shadow depth shader)
    let h_curved = apply_height_curve01(h_norm);
    
    // Compute shadow-normalized Z (matches shadow depth shader: world_z = h_curved * h_exag)
    let shadow_z = h_curved * h_exag;
    
    return vec3<f32>(world_xy.x, world_xy.y, shadow_z);
}

/// Calculate shadow visibility for terrain
fn calculate_shadow_terrain(world_pos: vec3<f32>, normal: vec3<f32>, view_depth: f32, tex_coord: vec2<f32>) -> f32 {
    // P0 early-out: when there are no active cascades, terrain is fully lit
    if (csm_uniforms.cascade_count == 0u) {
        return 1.0;
    }

    // Select appropriate cascade
    let cascade_idx = select_cascade_terrain(view_depth);
    
    // Normalize world position for shadow lookup (match shadow map's normalized heights)
    // Compute from tex_coord since world_pos is incorrectly interpolated for fullscreen triangle
    let shadow_pos = normalize_for_shadow(tex_coord);
    
    // Sample shadow with PCF for current cascade
    var shadow_factor = sample_shadow_pcf_terrain(shadow_pos, normal, cascade_idx);
    
    // Cascade blending: smooth transitions between cascades to avoid visible seams
    // Only blend if cascade_blend_range > 0 and we're not at the last cascade
    let blend_range = csm_uniforms.cascade_blend_range;
    if (blend_range > 0.0 && cascade_idx < csm_uniforms.cascade_count - 1u) {
        let current_far = csm_uniforms.cascades[cascade_idx].far_distance;
        let blend_start = current_far * (1.0 - blend_range);
        
        // Check if we're in the blend region near cascade boundary
        if (view_depth > blend_start) {
            // Sample next cascade
            let next_shadow = sample_shadow_pcf_terrain(shadow_pos, normal, cascade_idx + 1u);
            
            // Blend between cascades based on depth within blend region
            let blend_factor = (view_depth - blend_start) / (current_far - blend_start);
            shadow_factor = mix(shadow_factor, next_shadow, blend_factor);
        }
    }
    
    return shadow_factor;
}

/// Debug: Get cascade color for visualization
/// Red=cascade 0, Green=cascade 1, Blue=cascade 2, Yellow=cascade 3
fn get_cascade_debug_color(cascade_idx: u32) -> vec3<f32> {
    switch cascade_idx {
        case 0u: { return vec3<f32>(1.0, 0.2, 0.2); } // Red
        case 1u: { return vec3<f32>(0.2, 1.0, 0.2); } // Green  
        case 2u: { return vec3<f32>(0.2, 0.2, 1.0); } // Blue
        case 3u: { return vec3<f32>(1.0, 1.0, 0.2); } // Yellow
        default: { return vec3<f32>(1.0, 0.0, 1.0); } // Magenta (error)
    }
}

/// Debug: Calculate shadow with debug info
/// Returns (shadow_visibility, cascade_debug_color)
fn debug_shadow_with_vis(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    view_depth: f32,
    tex_coord: vec2<f32>
) -> vec4<f32> {
    // P0 early-out: NoopShadow sentinel via cascade_count == 0 → fully lit, cascade 0 color
    if (csm_uniforms.cascade_count == 0u) {
        let cascade_color = get_cascade_debug_color(0u);
        return vec4<f32>(cascade_color, 1.0);
    }
    
    let cascade_idx = select_cascade_terrain(view_depth);
    let shadow_pos = normalize_for_shadow(tex_coord);
    let shadow_vis = sample_shadow_pcf_terrain(shadow_pos, normal, cascade_idx);
    let cascade_color = get_cascade_debug_color(cascade_idx);
    
    // Return shadow visibility in w, cascade color in xyz
    return vec4<f32>(cascade_color, shadow_vis);
}

/// Debug: Show shadow map UV coordinates
/// Red = U coordinate, Green = V coordinate, Blue = depth (NDC.z)
fn debug_shadow_coords(world_pos: vec3<f32>, cascade_idx: u32) -> vec3<f32> {
    let cascade = csm_uniforms.cascades[cascade_idx];
    let light_space_pos = cascade.light_view_proj * vec4<f32>(world_pos, 1.0);
    let ndc = light_space_pos.xyz / light_space_pos.w;
    
    // Convert to texture coordinates [0,1]
    let shadow_u = ndc.x * 0.5 + 0.5;
    let shadow_v = ndc.y * -0.5 + 0.5;
    let shadow_depth = ndc.z;
    
    // Return as colors: R=U, G=V, B=depth
    // If R or G is outside [0,1], coordinates are out of shadow map bounds
    return vec3<f32>(
        clamp(shadow_u, 0.0, 1.0),
        clamp(shadow_v, 0.0, 1.0),
        clamp(shadow_depth, 0.0, 1.0)
    );
}

struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) uv : vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position : vec4<f32>,
    @location(0) world_position : vec3<f32>,
    @location(1) world_normal : vec3<f32>,
    @location(2) tex_coord : vec2<f32>,
};

struct FragmentOutput {
    @location(0) color : vec4<f32>,
};

fn sample_height(uv : vec2<f32>) -> f32 {
    let uv_clamped = clamp(uv, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
    return textureSample(height_tex, height_samp, uv_clamped).r;
}

fn get_height_geom_t(h_raw: f32) -> f32 {
    let h_min = u_shading.clamp0.x;
    let h_max = u_shading.clamp0.y;
    let range = max(h_max - h_min, 1e-6);
    return clamp((h_raw - h_min) / range, 0.0, 1.0);
}

fn apply_height_curve01(t: f32) -> f32 {
    let mode = u32(u_shading.height_curve.x + 0.5);
    let strength = clamp(u_shading.height_curve.y, 0.0, 1.0);
    if (strength <= 0.0) {
        return t;
    }

    var curved = t;
    if (mode == 1u) { // pow
        let p = max(u_shading.height_curve.z, 0.01);
        curved = pow(t, p);
    } else if (mode == 2u) { // smoothstep
        curved = t * t * (3.0 - 2.0 * t);
    } else if (mode == 3u) { // lut
        curved = height_curve_lut_sample(t);
    }

    return mix(t, curved, strength);
}

fn sample_height_geom(uv : vec2<f32>) -> f32 {
    let uv_clamped = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let h_raw = textureSample(height_tex, height_samp, uv_clamped).r;
    let t = get_height_geom_t(h_raw);
    let h_min = u_shading.clamp0.x;
    let h_max = u_shading.clamp0.y;
    return h_min + apply_height_curve01(t) * (h_max - h_min);
}

fn height_curve_lut_sample(t: f32) -> f32 {
    let dims = textureDimensions(height_curve_lut_tex, 0);
    let max_x = max(i32(dims.x) - 1, 0);
    let u = clamp(t, 0.0, 1.0);
    let x = i32(round(u * f32(max_x)));
    return textureLoad(height_curve_lut_tex, vec2<i32>(x, 0), 0).r;
}

fn calculate_texel_size() -> vec2<f32> {
    let dims = vec2<f32>(textureDimensions(height_tex, 0));
    return vec2<f32>(
        select(1.0, 1.0 / dims.x, dims.x > 0.0),
        select(1.0, 1.0 / dims.y, dims.y > 0.0),
    );
}

/// Vertex shader generates fullscreen triangle without vertex buffers
@vertex
fn vs_main(@builtin(vertex_index) vertex_id : u32) -> VertexOutput {
    var out : VertexOutput;

    // Generate fullscreen triangle covering [-1,1] NDC space
    // vertex 0: (-1, -1) -> UV (0, 0)
    // vertex 1: ( 3, -1) -> UV (2, 0)
    // vertex 2: (-1,  3) -> UV (0, 2)
    let uv_x = f32((vertex_id << 1u) & 2u);
    let uv_y = f32(vertex_id & 2u);
    let uv = vec2<f32>(uv_x, uv_y);

    let ndc_x = uv_x * 2.0 - 1.0;
    let ndc_y = uv_y * 2.0 - 1.0;
    out.clip_position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);

    // Reconstruct world position from UV coordinates
    // The terrain is centered at origin with spacing defining the XY extent
    let spacing = u_terrain.spacing_h_exag.x;
    let h_exag = u_terrain.spacing_h_exag.z;

    // Map UV [0,1] to world XY centered at origin
    let world_xy = (uv - vec2<f32>(0.5, 0.5)) * spacing;

    // Sample height from heightmap (use textureSampleLevel for vertex shader)
    let h_raw = textureSampleLevel(height_tex, height_samp, uv, 0.0).r;
    let t_geom = get_height_geom_t(h_raw);
    let h_min = u_shading.clamp0.x;
    let h_max = u_shading.clamp0.y;
    let h_disp = h_min + apply_height_curve01(t_geom) * (h_max - h_min);
    let world_z = h_disp * h_exag;

    out.world_position = vec3<f32>(world_xy.x, world_xy.y, world_z);
    out.world_normal = vec3<f32>(0.0, 1.0, 0.0); // Default up, will be recalculated in fragment shader
    out.tex_coord = uv;
    return out;
}

/// Sample height at a specific LOD level for LOD-aware normal computation.
fn sample_height_geom_level(uv: vec2<f32>, lod: f32) -> f32 {
    let uv_clamped = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let h_raw = textureSampleLevel(height_tex, height_samp, uv_clamped, lod).r;
    let t = get_height_geom_t(h_raw);
    let h_min = u_shading.clamp0.x;
    let h_max = u_shading.clamp0.y;
    return h_min + apply_height_curve01(t) * (h_max - h_min);
}

/// Compute LOD from screen-space UV footprint (for LOD-aware Sobel).
/// Returns LOD level and texel size at that LOD.
struct LodInfo {
    lod: f32,
    texel_uv: vec2<f32>,
}

fn compute_height_lod(uv: vec2<f32>) -> LodInfo {
    var info: LodInfo;
    let dims = vec2<f32>(textureDimensions(height_tex, 0));
    let max_lod = f32(textureNumLevels(height_tex) - 1u);
    
    // Compute screen-space derivatives of UV
    let ddx_uv = dpdx(uv);
    let ddy_uv = dpdy(uv);
    
    // Footprint in texels (at mip 0)
    let rho = max(length(ddx_uv * dims), length(ddy_uv * dims));
    
    // LOD = log2 of footprint, clamped to valid range
    info.lod = clamp(log2(max(rho, 1.0)), 0.0, max_lod);
    
    // Texel size at this LOD (in UV space)
    let mip_scale = exp2(info.lod);
    info.texel_uv = mip_scale / dims;
    
    return info;
}

/// Calculate normal from height map using LOD-aware Sobel filter.
/// Uses explicit LOD for all samples to avoid mip mismatch with offsets.
fn calculate_normal_lod_aware(uv: vec2<f32>) -> vec3<f32> {
    let lod_info = compute_height_lod(uv);
    let lod = lod_info.lod;
    let texel_uv = lod_info.texel_uv;
    
    let offset_x = vec2<f32>(texel_uv.x, 0.0);
    let offset_y = vec2<f32>(0.0, texel_uv.y);
    
    // All 9 samples at the SAME LOD level
    let tl = sample_height_geom_level(uv - offset_x - offset_y, lod);
    let t  = sample_height_geom_level(uv - offset_y, lod);
    let tr = sample_height_geom_level(uv + offset_x - offset_y, lod);
    let l  = sample_height_geom_level(uv - offset_x, lod);
    let r  = sample_height_geom_level(uv + offset_x, lod);
    let bl = sample_height_geom_level(uv - offset_x + offset_y, lod);
    let b  = sample_height_geom_level(uv + offset_y, lod);
    let br = sample_height_geom_level(uv + offset_x + offset_y, lod);
    
    // Sobel gradients
    let dx = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl);
    let dy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr);
    
    // Scale by world-space texel size for proper gradient magnitude
    // At higher LOD, texels cover more world space, so gradients are naturally smoothed
    let spacing = u_terrain.spacing_h_exag.x;
    let world_texel = texel_uv * spacing; // Texel size in world units
    
    let vertical_scale = max(u_terrain.spacing_h_exag.z * 0.5, 1e-3);
    return normalize(vec3<f32>(-dx / world_texel.x, vertical_scale, -dy / world_texel.y));
}

/// Calculate normal from height map using Sobel filter (LEGACY - not LOD-aware).
/// Kept for A/B comparison during flake diagnosis.
fn calculate_normal(uv : vec2<f32>, texel_size : vec2<f32>) -> vec3<f32> {
    let offset_x = vec2<f32>(texel_size.x, 0.0);
    let offset_y = vec2<f32>(0.0, texel_size.y);

    let tl = sample_height_geom(uv - offset_x - offset_y);
    let t = sample_height_geom(uv - offset_y);
    let tr = sample_height_geom(uv + offset_x - offset_y);
    let l = sample_height_geom(uv - offset_x);
    let r = sample_height_geom(uv + offset_x);
    let bl = sample_height_geom(uv - offset_x + offset_y);
    let b = sample_height_geom(uv + offset_y);
    let br = sample_height_geom(uv + offset_x + offset_y);

    let dx = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl);
    let dy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr);

    let vertical_scale = max(u_terrain.spacing_h_exag.z * 0.5, 1e-3);
    return normalize(vec3<f32>(-dx, vertical_scale, -dy));
}

/// Compute geometric normal from screen-space derivatives of world position.
/// This is the "ground truth" normal that doesn't suffer from mip mismatch.
fn calculate_normal_ddxddy(world_pos: vec3<f32>) -> vec3<f32> {
    let ddx_pos = dpdx(world_pos);
    let ddy_pos = dpdy(world_pos);
    // Cross product gives surface normal (right-hand rule)
    // Note: order matters for winding direction
    let n = cross(ddx_pos, ddy_pos);
    // Ensure normal points "up" (positive Y in our coordinate system)
    let n_norm = normalize(n);
    return select(n_norm, -n_norm, n_norm.y < 0.0);
}

/// Compute triplanar blend weights from surface normal.
/// Returns normalized weights (sum to 1) for x, y, z projection axes.
/// T1 requirement: wx + wy + wz = 1, weights change smoothly with normal.
fn compute_triplanar_weights(normal: vec3<f32>, blend_sharpness: f32) -> vec3<f32> {
    let abs_n = abs(normal);
    // Use higher blend sharpness for cleaner projection transitions
    let sharpen = pow(abs_n + vec3<f32>(1e-4), vec3<f32>(blend_sharpness * 1.5));
    let weight_sum = sharpen.x + sharpen.y + sharpen.z;
    return sharpen / max(weight_sum, 1e-4);
}

/// Procedural checker pattern for triplanar UV stretching test.
/// Returns 0.0 or 1.0 based on checker grid position.
/// T2 requirement: checker shows no stretching on steep slopes.
fn checker_pattern(uv: vec2<f32>, checker_scale: f32) -> f32 {
    let grid = floor(uv * checker_scale);
    let checker = i32(grid.x + grid.y) & 1;
    return f32(checker);
}

/// Sample triplanar checker pattern (no textures, pure procedural).
/// Uses same blending logic as texture triplanar for A/B comparison.
fn sample_triplanar_checker(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    scale: f32,
    blend_sharpness: f32,
    checker_scale: f32
) -> f32 {
    let weights = compute_triplanar_weights(normal, blend_sharpness);
    
    // Project world position to each axis plane
    let uv_x = world_pos.yz * scale;
    let uv_y = world_pos.xz * scale;
    let uv_z = world_pos.xy * scale;
    
    // Sample checker pattern for each projection
    let check_x = checker_pattern(uv_x, checker_scale);
    let check_y = checker_pattern(uv_y, checker_scale);
    let check_z = checker_pattern(uv_z, checker_scale);
    
    // Blend using triplanar weights
    return check_x * weights.x + check_y * weights.y + check_z * weights.z;
}

/// Triplanar sampling with textureSampleGrad for correct mip selection.
/// Computes UV gradients from world position derivatives for each projection axis.
fn sample_triplanar(
    world_pos : vec3<f32>,
    normal : vec3<f32>,
    scale : f32,
    blend_sharpness : f32,
    layer : f32,
    _lod_bias : f32  // Unused - gradients determine LOD
) -> vec3<f32> {
    let weights = compute_triplanar_weights(normal, blend_sharpness);

    // Compute triplanar UVs from world position
    let uv_x = world_pos.yz * scale;
    let uv_y = world_pos.xz * scale;
    let uv_z = world_pos.xy * scale;

    // Compute screen-space derivatives of world position for proper mip selection
    // This ensures correct LOD even when UVs are derived from world coords
    let dpdx_world = dpdx(world_pos) * scale;
    let dpdy_world = dpdy(world_pos) * scale;

    // Extract UV gradients for each projection axis
    let ddx_x = dpdx_world.yz;
    let ddy_x = dpdy_world.yz;
    let ddx_y = dpdx_world.xz;
    let ddy_y = dpdy_world.xz;
    let ddx_z = dpdx_world.xy;
    let ddy_z = dpdy_world.xy;

    let layer_index = i32(layer);
    let color_x = textureSampleGrad(material_albedo_tex, material_samp, uv_x, layer_index, ddx_x, ddy_x).rgb;
    let color_y = textureSampleGrad(material_albedo_tex, material_samp, uv_y, layer_index, ddx_y, ddy_y).rgb;
    let color_z = textureSampleGrad(material_albedo_tex, material_samp, uv_z, layer_index, ddx_z, ddy_z).rgb;

    return color_x * weights.x + color_y * weights.y + color_z * weights.z;
}

fn build_tbn(normal : vec3<f32>) -> mat3x3<f32> {
    let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), abs(normal.y) > 0.99);
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    return mat3x3<f32>(tangent, bitangent, normal);
}

// ──────────────────────────────────────────────────────────────────────────
// P6: Micro-Detail Functions
// Add close-range surface detail without LOD popping:
// - Triplanar detail normals (2m repeat) blended via RNM with distance fade
// - Procedural albedo brightness noise (±10%) using stable world-space coords
// ──────────────────────────────────────────────────────────────────────────

/// Simple hash function for procedural noise (stable, no texture needed)
fn hash31(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

/// 3D value noise for smooth procedural patterns
fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    // Smooth interpolation
    let u = f * f * (3.0 - 2.0 * f);
    
    // 8 corner samples
    let n000 = hash31(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash31(i + vec3<f32>(1.0, 1.0, 1.0));
    
    // Trilinear interpolation
    let x0 = mix(n000, n100, u.x);
    let x1 = mix(n010, n110, u.x);
    let x2 = mix(n001, n101, u.x);
    let x3 = mix(n011, n111, u.x);
    let y0 = mix(x0, x1, u.y);
    let y1 = mix(x2, x3, u.y);
    return mix(y0, y1, u.z);
}

/// Generate procedural detail normal in tangent space using gradient noise
/// Returns a tangent-space normal that can be blended with the base normal
fn procedural_detail_normal(world_pos: vec3<f32>, scale: f32) -> vec3<f32> {
    let p = world_pos * scale;
    
    // Sample noise at offset positions to compute gradient
    let eps = 0.1;
    let nx = value_noise(p + vec3<f32>(eps, 0.0, 0.0)) - value_noise(p - vec3<f32>(eps, 0.0, 0.0));
    let ny = value_noise(p + vec3<f32>(0.0, eps, 0.0)) - value_noise(p - vec3<f32>(0.0, eps, 0.0));
    let nz = value_noise(p + vec3<f32>(0.0, 0.0, eps)) - value_noise(p - vec3<f32>(0.0, 0.0, eps));
    
    // Convert gradient to tangent-space normal perturbation
    // Scale controls how strong the perturbation is
    let gradient = vec3<f32>(nx, ny, nz) * 2.0;
    
    // Return as tangent-space normal (z=1 is unperturbed up)
    return normalize(vec3<f32>(gradient.x, gradient.z, 1.0));
}

/// Reoriented Normal Mapping (RNM) blend for combining detail normals
/// This method properly reorients the detail normal based on the base normal
/// Reference: "Blending in Detail" by Colin Barré-Brisebois & Stephen Hill
fn blend_rnm(base_n: vec3<f32>, detail_n: vec3<f32>) -> vec3<f32> {
    // Reorient detail normal based on base normal
    // This gives better results than simple linear blending
    let t = base_n + vec3<f32>(0.0, 0.0, 1.0);
    let u = detail_n * vec3<f32>(-1.0, -1.0, 1.0);
    return normalize(t * dot(t, u) - u * t.z);
}

/// Calculate distance fade factor for micro-detail
/// Returns 1.0 at close range (full detail), 0.0 at far range (no detail)
fn calculate_detail_fade(view_distance: f32, fade_start: f32, fade_end: f32) -> f32 {
    if (view_distance <= fade_start) {
        return 1.0;
    }
    if (view_distance >= fade_end) {
        return 0.0;
    }
    // Smooth hermite interpolation for artifact-free transition
    let t = (view_distance - fade_start) / (fade_end - fade_start);
    return 1.0 - t * t * (3.0 - 2.0 * t);
}

/// Generate procedural albedo brightness noise
/// Returns a multiplier around 1.0 (e.g., 0.9 to 1.1 for 10% noise)
fn procedural_albedo_noise(world_pos: vec3<f32>, noise_amplitude: f32) -> f32 {
    // Use a different frequency than detail normals to avoid correlation
    let noise_scale = 0.7; // World-space frequency for albedo noise
    let noise = value_noise(world_pos * noise_scale);
    // Map [0,1] noise to [-amplitude, +amplitude] and add to 1.0
    return 1.0 + (noise - 0.5) * 2.0 * noise_amplitude;
}

/// P4: Apply slope-based hue variation to increase h_std
/// Shifts albedo hue based on terrain slope: steep slopes get redder, flat areas get yellower
/// hue_shift_strength: 0.0 = no effect, 0.1 = subtle, 0.2 = moderate
fn apply_slope_hue_variation(albedo: vec3<f32>, slope_factor: f32, hue_shift_strength: f32) -> vec3<f32> {
    if (hue_shift_strength <= 0.0) {
        return albedo;
    }
    
    // Convert RGB to HSV-like representation for hue manipulation
    let max_c = max(max(albedo.r, albedo.g), albedo.b);
    let min_c = min(min(albedo.r, albedo.g), albedo.b);
    let delta = max_c - min_c;
    
    // Skip if no saturation (grayscale)
    if (delta < 0.001) {
        return albedo;
    }
    
    // Compute current hue (0-1 range, 0=red, 0.33=green, 0.67=blue)
    var hue: f32;
    if (max_c == albedo.r) {
        hue = ((albedo.g - albedo.b) / delta) / 6.0;
        if (hue < 0.0) { hue = hue + 1.0; }
    } else if (max_c == albedo.g) {
        hue = (2.0 + (albedo.b - albedo.r) / delta) / 6.0;
    } else {
        hue = (4.0 + (albedo.r - albedo.g) / delta) / 6.0;
    }
    
    let saturation = delta / max_c;
    let value = max_c;
    
    // Shift hue based on slope: steep slopes (slope_factor near 1) -> redder (lower hue ~0.02-0.05)
    // Flat areas (slope_factor near 0) -> yellower (higher hue ~0.10-0.12)
    // slope_factor is typically 0.0-1.0 where 0=flat, 1=vertical
    // Target: h_A ~0.04, h_B ~0.08, h_C ~0.10 (roughly linear with elevation/flatness)
    let hue_shift = (slope_factor - 0.5) * hue_shift_strength; // Steep = redder (negative), flat = yellower (positive)
    let new_hue = fract(hue + hue_shift); // Keep in 0-1 range
    
    // Convert back to RGB
    let c = saturation * value;
    let x = c * (1.0 - abs(fract(new_hue * 6.0) * 2.0 - 1.0));
    let m = value - c;
    
    var rgb: vec3<f32>;
    let h6 = new_hue * 6.0;
    if (h6 < 1.0) {
        rgb = vec3<f32>(c, x, 0.0);
    } else if (h6 < 2.0) {
        rgb = vec3<f32>(x, c, 0.0);
    } else if (h6 < 3.0) {
        rgb = vec3<f32>(0.0, c, x);
    } else if (h6 < 4.0) {
        rgb = vec3<f32>(0.0, x, c);
    } else if (h6 < 5.0) {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }
    
    return rgb + vec3<f32>(m, m, m);
}

/// Apply micro-detail to normal
/// Blends procedural detail normal with base using RNM, with distance fade
fn apply_detail_normal(
    base_normal: vec3<f32>,
    world_pos: vec3<f32>,
    view_distance: f32,
    detail_scale: f32,
    detail_strength: f32,
    fade_start: f32,
    fade_end: f32,
) -> vec3<f32> {
    // Early out if no detail contribution
    let fade = calculate_detail_fade(view_distance, fade_start, fade_end);
    if (fade <= 0.0 || detail_strength <= 0.0) {
        return base_normal;
    }
    
    // Generate triplanar detail normal using procedural noise
    // Sample from three axes and blend by base normal weights
    let weights = compute_triplanar_weights(base_normal, 4.0);
    
    // Detail UVs from world position
    let uv_scale = 1.0 / detail_scale;
    let detail_x = procedural_detail_normal(vec3<f32>(0.0, world_pos.y, world_pos.z) * uv_scale, 1.0);
    let detail_y = procedural_detail_normal(vec3<f32>(world_pos.x, 0.0, world_pos.z) * uv_scale, 1.0);
    let detail_z = procedural_detail_normal(vec3<f32>(world_pos.x, world_pos.y, 0.0) * uv_scale, 1.0);
    
    // Blend triplanar detail normals
    let blended_detail = normalize(
        detail_x * weights.x + 
        detail_y * weights.y + 
        detail_z * weights.z
    );
    
    // Build TBN from base normal
    let tbn = build_tbn(base_normal);
    
    // Transform detail normal to world space
    let detail_world = normalize(tbn * blended_detail);
    
    // RNM blend with strength and fade
    let effective_strength = detail_strength * fade;
    let blended = blend_rnm(base_normal, mix(vec3<f32>(0.0, 0.0, 1.0), blended_detail, effective_strength));
    
    return normalize(tbn * blended);
}

fn rotate_y(v : vec3<f32>, sin_theta : f32, cos_theta : f32) -> vec3<f32> {
    return vec3<f32>(
        v.x * cos_theta + v.z * sin_theta,
        v.y,
        -v.x * sin_theta + v.z * cos_theta,
    );
}

// Note: fresnel_schlick is provided by brdf/common.wgsl (included via lighting.wgsl)

/// Parallax Occlusion Mapping with binary search refinement.
fn parallax_occlusion_mapping(
    uv : vec2<f32>,
    view_dir_tangent : vec3<f32>,
    height_scale : f32,
    min_steps : u32,
    max_steps : u32,
    refine_steps : u32
) -> vec2<f32> {
    if (height_scale <= 0.0) {
        return uv;
    }

    let view_dir = normalize(view_dir_tangent);
    let min_s = max(min_steps, 1u);
    let max_s = max(max_steps, min_s);
    let blend = clamp(abs(view_dir.z), 0.0, 1.0);
    let steps_interp = mix(f32(max_s), f32(min_s), blend);
    let step_count = clamp(u32(steps_interp + 0.5), 1u, max_s);
    var step_size = 1.0 / f32(step_count);

    let dir_xy = view_dir.xy;
    if (length(dir_xy) < 1e-5) {
        return uv;
    }
    let parallax_dir = normalize(dir_xy) * height_scale;

    var current_uv = uv;
    var current_layer = 0.0;
    var current_height = sample_height(current_uv);

    for (var i = 0u; i < step_count; i = i + 1u) {
        if (current_layer >= current_height) {
            break;
        }
        current_uv -= parallax_dir * step_size;
        current_layer += step_size;
        current_height = sample_height(current_uv);
    }

    var refine_step_size = step_size;
    var refine_index = 0u;
    while (refine_index < refine_steps) {
        let delta_uv = parallax_dir * refine_step_size * 0.5;
        refine_step_size *= 0.5;
        current_height = sample_height(current_uv);
        if (current_layer >= current_height) {
            current_uv -= delta_uv;
            current_layer -= refine_step_size;
        } else {
            current_uv += delta_uv;
            current_layer += refine_step_size;
        }
        refine_index = refine_index + 1u;
    }

    return current_uv;
}

/// Cook-Torrance PBR BRDF with GGX, Smith, and Schlick terms.
fn calculate_pbr_brdf(
    normal : vec3<f32>,
    view_dir : vec3<f32>,
    light_dir : vec3<f32>,
    albedo : vec3<f32>,
    roughness : f32,
    metallic : f32,
    f0 : vec3<f32>
) -> vec3<f32> {
    let halfway = normalize(view_dir + light_dir);

    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let n_dot_h = max(dot(normal, halfway), 0.0);
    let v_dot_h = max(dot(view_dir, halfway), 0.0);

    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }

    let alpha = max(roughness * roughness, 1e-3);
    let alpha_sq = alpha * alpha;
    let denom = (n_dot_h * n_dot_h * (alpha_sq - 1.0)) + 1.0;
    let distribution = alpha_sq / (PI * denom * denom);

    let k = (roughness + 1.0);
    let k_sq = (k * k) / 8.0;
    let g1_l = n_dot_l / (n_dot_l * (1.0 - k_sq) + k_sq);
    let g1_v = n_dot_v / (n_dot_v * (1.0 - k_sq) + k_sq);
    let geometry = g1_l * g1_v;

    let fresnel = f0 + (vec3<f32>(1.0, 1.0, 1.0) - f0) * pow(1.0 - v_dot_h, 5.0);

    let specular = (distribution * geometry) * fresnel / max(4.0 * n_dot_l * n_dot_v, 1e-3);
    let k_d = (vec3<f32>(1.0, 1.0, 1.0) - fresnel) * (1.0 - metallic);
    let diffuse = k_d * albedo / PI;
    return (diffuse + specular) * n_dot_l;
}

/// Split-normal PBR BRDF: uses smooth normal for specular (eliminates aliasing),
/// detailed normal for diffuse (keeps surface detail). Standard terrain technique.
fn calculate_pbr_brdf_split_normal(
    diffuse_normal : vec3<f32>,   // Height-derived normal for diffuse shading
    specular_normal : vec3<f32>,  // Geometric/smooth normal for specular (anti-alias)
    view_dir : vec3<f32>,
    light_dir : vec3<f32>,
    albedo : vec3<f32>,
    roughness : f32,
    metallic : f32,
    f0 : vec3<f32>
) -> vec3<f32> {
    let halfway = normalize(view_dir + light_dir);

    // Diffuse uses detailed normal for surface variation
    let n_dot_l_diff = max(dot(diffuse_normal, light_dir), 0.0);
    
    // Specular uses smooth normal to avoid aliasing
    let n_dot_l_spec = max(dot(specular_normal, light_dir), 0.0);
    let n_dot_v = max(dot(specular_normal, view_dir), 0.0);
    let n_dot_h = max(dot(specular_normal, halfway), 0.0);
    let v_dot_h = max(dot(view_dir, halfway), 0.0);

    if (n_dot_l_diff <= 0.0 && n_dot_l_spec <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }

    // GGX Distribution (specular normal)
    let alpha = max(roughness * roughness, 1e-3);
    let alpha_sq = alpha * alpha;
    let denom = (n_dot_h * n_dot_h * (alpha_sq - 1.0)) + 1.0;
    let distribution = alpha_sq / (PI * denom * denom);

    // Smith Geometry (specular normal)
    let k = (roughness + 1.0);
    let k_sq = (k * k) / 8.0;
    let g1_l = n_dot_l_spec / (n_dot_l_spec * (1.0 - k_sq) + k_sq + 1e-5);
    let g1_v = n_dot_v / (n_dot_v * (1.0 - k_sq) + k_sq + 1e-5);
    let geometry = g1_l * g1_v;

    // Fresnel
    let fresnel = f0 + (vec3<f32>(1.0, 1.0, 1.0) - f0) * pow(1.0 - v_dot_h, 5.0);

    // Specular term (smooth normal) - only if facing light
    var specular = vec3<f32>(0.0);
    if (n_dot_l_spec > 0.0 && n_dot_v > 0.0) {
        specular = (distribution * geometry) * fresnel / max(4.0 * n_dot_l_spec * n_dot_v, 1e-3);
        specular = specular * n_dot_l_spec;
    }
    
    // Diffuse term (detailed normal)
    let k_d = (vec3<f32>(1.0, 1.0, 1.0) - fresnel) * (1.0 - metallic);
    let diffuse = k_d * albedo / PI * n_dot_l_diff;
    
    return diffuse + specular;
}

/// P3: Split-roughness PBR BRDF for Toksvig specular anti-aliasing.
/// Uses base_roughness for diffuse (preserves detail), specular_roughness for specular (smoothed via Toksvig).
/// This is the key P3 fix: Toksvig applies ONLY to specular, leaving diffuse crisp.
fn calculate_pbr_brdf_split_roughness(
    normal : vec3<f32>,
    view_dir : vec3<f32>,
    light_dir : vec3<f32>,
    albedo : vec3<f32>,
    base_roughness : f32,      // Original roughness for diffuse calculations
    specular_roughness : f32,  // Toksvig-adjusted roughness for specular calculations
    metallic : f32,
    f0 : vec3<f32>
) -> vec3<f32> {
    let halfway = normalize(view_dir + light_dir);

    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let n_dot_h = max(dot(normal, halfway), 0.0);
    let v_dot_h = max(dot(view_dir, halfway), 0.0);

    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }

    // GGX Distribution - uses specular_roughness (Toksvig-adjusted)
    let alpha_spec = max(specular_roughness * specular_roughness, 1e-3);
    let alpha_spec_sq = alpha_spec * alpha_spec;
    let denom = (n_dot_h * n_dot_h * (alpha_spec_sq - 1.0)) + 1.0;
    let distribution = alpha_spec_sq / (PI * denom * denom);

    // Smith Geometry - uses specular_roughness
    let k = (specular_roughness + 1.0);
    let k_sq = (k * k) / 8.0;
    let g1_l = n_dot_l / (n_dot_l * (1.0 - k_sq) + k_sq);
    let g1_v = n_dot_v / (n_dot_v * (1.0 - k_sq) + k_sq);
    let geometry = g1_l * g1_v;

    // Fresnel (unchanged - doesn't depend on roughness directly in Schlick)
    let fresnel = f0 + (vec3<f32>(1.0, 1.0, 1.0) - f0) * pow(1.0 - v_dot_h, 5.0);

    // Specular term (uses specular_roughness)
    let specular = (distribution * geometry) * fresnel / max(4.0 * n_dot_l * n_dot_v, 1e-3);
    
    // Diffuse term uses base_roughness indirectly through energy conservation
    // Note: Standard Lambertian diffuse doesn't directly use roughness,
    // but kD calculation uses Fresnel which we keep consistent
    let k_d = (vec3<f32>(1.0, 1.0, 1.0) - fresnel) * (1.0 - metallic);
    let diffuse = k_d * albedo / PI;
    
    return (diffuse + specular) * n_dot_l;
}

// P2-05: Bridge function to map terrain parameters to ShadingParamsGPU for optional eval_brdf
// Maps a subset of TerrainShadingUniforms to ShadingParamsGPU, ignoring unsupported terrain knobs
fn terrain_to_shading_params(
    roughness: f32,
    metallic: f32,
    brdf_model: u32,  // Runtime flag: which BRDF model to use (default BRDF_COOK_TORRANCE_GGX)
) -> ShadingParamsGPU {
    var params: ShadingParamsGPU;
    params.brdf = brdf_model;
    params.metallic = metallic;
    params.roughness = roughness;
    params.sheen = 0.0;        // Terrain doesn't use sheen
    params.clearcoat = 0.0;    // Terrain doesn't use clearcoat
    params.subsurface = 0.0;   // Terrain doesn't use subsurface
    params.anisotropy = 0.0;   // Terrain doesn't use anisotropy
    return params;
}

fn gamma_correct(color : vec3<f32>, gamma : f32) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / gamma));
}

// Linear to sRGB conversion (piecewise exact curve)
fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let a = vec3<f32>(0.055);
    let lo = c * 12.92;
    let hi = (1.0 + a) * pow(c, vec3<f32>(1.0 / 2.4)) - a;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

fn tonemap_reinhard(color : vec3<f32>) -> vec3<f32> {
    return color / (vec3<f32>(1.0, 1.0, 1.0) + color);
}

fn tonemap_aces(color : vec3<f32>) -> vec3<f32> {
    let clipped = clamp(color, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(65504.0, 65504.0, 65504.0));
    let a = clipped * (clipped + vec3<f32>(0.0245786, 0.0245786, 0.0245786)) - vec3<f32>(0.000090537, 0.000090537, 0.000090537);
    let b = clipped * (vec3<f32>(0.983729, 0.983729, 0.983729) * clipped + vec3<f32>(0.4329510, 0.4329510, 0.4329510)) + vec3<f32>(0.238081, 0.238081, 0.238081);
    return clamp(a / b, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0));
}

// Check for NaN/Inf (debug helper for catching bad data)
fn is_finite_f32(x: f32) -> bool {
    // NaN != NaN, and Inf comparisons fail
    return !(x != x) && (x <= 3.4028235e+38) && (x >= -3.4028235e+38);
}

// ──────────────────────────────────────────────────────────────────────────
// PBR Debug Helpers - Split IBL for separate diffuse/specular visualization
// ──────────────────────────────────────────────────────────────────────────

struct IblSplit {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
    fresnel: vec3<f32>,
    n_dot_v: f32,
}

/// Compute IBL with diffuse and specular separated (for debug visualization)
fn eval_ibl_split(
    n: vec3<f32>,
    v: vec3<f32>,
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    f0: vec3<f32>,
) -> IblSplit {
    var result: IblSplit;
    
    // Clamp inputs for numeric safety
    let n_dot_v = saturate(dot(n, v));
    let roughness_clamped = saturate(roughness);
    result.n_dot_v = n_dot_v;
    
    // Calculate reflection direction
    let reflection_dir = reflect(-v, n);
    
    // Fresnel term for IBL (roughness-aware Schlick)
    let one_minus_cos = saturate(1.0 - n_dot_v);
    let pow5 = one_minus_cos * one_minus_cos * one_minus_cos * one_minus_cos * one_minus_cos;
    // Roughness-aware Fresnel: lerp toward white at grazing for rough surfaces
    let F_ibl = f0 + (max(vec3<f32>(1.0 - roughness_clamped), f0) - f0) * pow5;
    result.fresnel = F_ibl;
    
    // Diffuse IBL (Lambertian)
    // kD = (1 - kS) * (1 - metallic)
    let kS_ibl = F_ibl;
    let kD_ibl = (vec3<f32>(1.0) - kS_ibl) * (1.0 - metallic);
    
    // Sample irradiance cubemap
    let irradiance = textureSampleLevel(envIrradiance, envSampler, n, 0.0).rgb;
    result.diffuse = kD_ibl * base_color * irradiance;
    
    // Specular IBL (split-sum approximation)
    let mip_level = roughness_clamped * roughness_clamped * 9.0; // Assume 10 mips (0-9)
    
    // Sample prefiltered specular cubemap
    let prefiltered_color = textureSampleLevel(envSpecular, envSampler, reflection_dir, mip_level).rgb;
    
    // Sample BRDF LUT
    let brdf_lut_uv = vec2<f32>(n_dot_v, roughness_clamped);
    let brdf_lut = textureSampleLevel(brdfLUT, envSampler, brdf_lut_uv, 0.0).rg;
    
    // Split-sum: prefiltered_color * (F0 * scale + bias)
    result.specular = prefiltered_color * (F_ibl * brdf_lut.x + brdf_lut.y);
    
    return result;
}

/// Compute luminance (relative luminance for sRGB primaries)
fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// ──────────────────────────────────────────────────────────────────────────
// Water Debug Helpers (unambiguous, no-tonemap, isolated modes)
// ──────────────────────────────────────────────────────────────────────────

// Falsecolor ramp: 0=blue -> 0.5=green -> 1=red (very visible)
fn ramp_falsecolor(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    let r = clamp(1.5 * x - 0.5, 0.0, 1.0);
    let g = clamp(1.5 - abs(2.0 * x - 1.0) * 1.5, 0.0, 1.0);
    let b = clamp(1.5 * (1.0 - x) - 0.5, 0.0, 1.0);
    return vec3<f32>(r, g, b);
}

// Simple HDR compression for debug visibility
fn compress_hdr(x: vec3<f32>) -> vec3<f32> {
    return x / (vec3<f32>(1.0) + x);
}

// DEBUG 100: Binary water classification - blue=water, dark gray=land
fn debug_water_is_water(is_water_flag: bool) -> vec3<f32> {
    let land = vec3<f32>(0.08, 0.08, 0.08);
    let water = vec3<f32>(0.0, 0.2, 1.0); // unmistakable blue
    return select(land, water, is_water_flag);
}

// DEBUG 101: Shore-distance scalar visualization with shoreline ring
fn debug_water_scalar(is_water_flag: bool, water_scalar: f32) -> vec3<f32> {
    let land = vec3<f32>(0.05, 0.05, 0.05);
    let t = clamp(water_scalar, 0.0, 1.0);
    var rgb_water = ramp_falsecolor(t);
    // Add bright ring near shoreline (where t is small)
    let shore_ring = smoothstep(0.06, 0.00, t);
    rgb_water = clamp(rgb_water + shore_ring * vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0), vec3<f32>(1.0));
    return select(land, rgb_water, is_water_flag);
}

// DEBUG 102: IBL specular on water only (land=black)
// Diagnostic version: shows prefiltered environment (no Fresnel) to isolate the issue
fn debug_water_ibl_spec_only(is_water_flag: bool, ibl_spec: vec3<f32>) -> vec3<f32> {
    let land = vec3<f32>(0.0, 0.0, 0.0); // black = no cheating
    let s = max(ibl_spec, vec3<f32>(0.0));
    let mag = s.x + s.y + s.z;
    // If IBL is effectively zero, show magenta diagnostic
    if (mag < 0.001) {
        return select(land, vec3<f32>(1.0, 0.0, 1.0), is_water_flag); // Magenta = IBL is zero
    }
    let rgb_dbg = compress_hdr(s);
    return select(land, rgb_dbg, is_water_flag);
}

// DEBUG 103: Raw prefiltered environment sample (no Fresnel, no BRDF LUT)
fn debug_water_prefilt_raw(is_water_flag: bool, prefilt_color: vec3<f32>) -> vec3<f32> {
    let land = vec3<f32>(0.0, 0.0, 0.0);
    let s = max(prefilt_color, vec3<f32>(0.0));
    let mag = s.x + s.y + s.z;
    if (mag < 0.001) {
        return select(land, vec3<f32>(1.0, 1.0, 0.0), is_water_flag); // Yellow = prefilt zero
    }
    let rgb_dbg = compress_hdr(s);
    return select(land, rgb_dbg, is_water_flag);
}

// ──────────────────────────────────────────────────────────────────────────
// P2: Atmospheric Fog
// Height-based exponential fog applied after PBR shading, before tonemap.
// When fog_density = 0.0, returns base_color unchanged (no-op).
//
// Coordinate system: Z-up (world_pos.z = elevation)
// Density scaling: quadratic (density²) for perceptually linear 0-1 range
// ──────────────────────────────────────────────────────────────────────────
fn apply_atmospheric_fog(
    base_color: vec3<f32>,
    world_pos: vec3<f32>,
    camera_pos: vec3<f32>,
) -> vec3<f32> {
    // Early-out when fog is disabled (density = 0) for exact no-op
    let density_raw = fog_uniforms.fog_density;
    if (density_raw <= 0.0) {
        return base_color;
    }
    
    // DEBUG: Return red to verify fog is being applied
    // return vec3<f32>(1.0, 0.0, 0.0);
    
    // Quadratic density scaling: makes 0-1 range perceptually useful
    // density=0.1 → 0.01, density=0.5 → 0.25, density=1.0 → 1.0
    let density = density_raw * density_raw;
    
    // Compute view distance
    let to_camera = camera_pos - world_pos;
    let view_distance = length(to_camera);
    
    // Height-based fog modulation:
    // Fog is denser at lower altitudes, thinner at higher altitudes.
    // Uses exponential falloff from base height.
    // NOTE: Terrain is Z-up, so world_pos.z is elevation.
    let height_falloff = fog_uniforms.fog_height_falloff;
    let base_height = fog_uniforms.fog_base_height;
    
    // Compute height factor: 1.0 at base_height, decreasing exponentially above
    // Fragment elevation relative to base (Z-up coordinate system)
    let fragment_elevation = max(world_pos.z - base_height, 0.0);
    let height_factor = exp(-height_falloff * fragment_elevation);
    
    // Exponential fog extinction with height modulation
    // Scale factor K converts world units to fog-friendly range
    // K=0.005 tuned for: density=0.1 subtle, 0.2 moderate, 0.3 strong but visible
    let extinction = exp(-density * view_distance * height_factor * 0.005);
    
    // Inscatter color (fog color seen in distance)
    let inscatter = fog_uniforms.fog_inscatter;
    
    // Mix: at extinction=1 (no fog), result is base_color
    //      at extinction=0 (full fog), result is inscatter
    let fog_result = mix(inscatter, base_color, extinction);
    return fog_result;
}

@fragment
fn fs_main(input : VertexOutput) -> FragmentOutput {
    var out : FragmentOutput;

    // P4: Reflection pass clip plane
    // When enable_flags.y > 0.5, we're in reflection pass mode and should
    // discard fragments below the water plane (only render geometry above water).
    // water_plane = (nx, ny, nz, -d) where plane equation is n·p + d = 0
    // For Y-up water plane at height h: water_plane = (0, 1, 0, -h)
    // A point is above the plane if n·p + d > 0, i.e., p.y > h
    if (water_reflection_uniforms.enable_flags.y > 0.5) {
        let plane = water_reflection_uniforms.water_plane;
        let dist_to_plane = dot(plane.xyz, input.world_position) + plane.w;
        if (dist_to_plane < 0.0) {
            discard;
        }
    }

    let uv = input.tex_coord;
    let debug_mode = u32(u_overlay.params1.y + 0.5);
    
    // Compute all normal variants for diagnostics
    let base_normal = normalize(input.world_normal);
    let lod_info = compute_height_lod(uv);
    let height_lod = lod_info.lod;
    
    // LOD-aware height normal (Milestone 2: fixes flakes from mip mismatch)
    let height_normal_lod = calculate_normal_lod_aware(uv);
    
    // Legacy height normal for comparison (not LOD-aware)
    let texel_size = calculate_texel_size();
    let height_normal_legacy = calculate_normal(uv, texel_size);
    
    // Derivative-based normal (Milestone 1: ground truth comparison)
    let n_dd = calculate_normal_ddxddy(input.world_position);
    
    // Select which height normal to use based on debug mode
    var height_normal = height_normal_lod; // Default: LOD-aware (the fix)
    if (debug_mode == DBG_FLAKE_NO_HEIGHT_NORMAL) {
        // Mode 24: use base_normal (no height detail) to isolate height-normal contribution
        height_normal = base_normal;
    } else if (debug_mode == DBG_FLAKE_DDXDDY_NORMAL) {
        // Mode 25: use derivative-based normal as ground truth
        height_normal = n_dd;
    }
    
    // Milestone 3/D: Minification fade for height-normal contribution
    // As LOD increases (far/grazing), reduce height-normal influence to prevent sparkles.
    // Using smoothstep for threshold-free transition (Milestone D improvement).
    // LOD 0-1: full contribution, LOD 1-4: smoothstep fade, LOD 4+: no contribution
    // Policy: lod_lo=1.0 (near detail preserved), lod_hi=4.0 (far field stable)
    let lod_fade_start = 1.0;  // lod_lo: below this, full height-normal
    let lod_fade_end = 4.0;    // lod_hi: above this, no height-normal
    // smoothstep(edge0, edge1, x) = smooth hermite interpolation
    // We want fade=1.0 at lod_fade_start and fade=0.0 at lod_fade_end
    let lod_fade = 1.0 - smoothstep(lod_fade_start, lod_fade_end, height_lod);
    
    let normal_blend_base = clamp(u_shading.triplanar_params.z, 0.0, 1.0);
    let normal_blend = normal_blend_base * lod_fade;
    // Capture pre-normalized normal for specular AA (Toksvig)
    let mixed_normal = mix(base_normal, height_normal, normal_blend);
    let normal_len = length(mixed_normal);
    let blended_normal = mixed_normal / max(normal_len, 1e-5);

    let tbn = build_tbn(blended_normal);
    // Extract camera position from view matrix properly
    // For a view matrix V that transforms world→view, the camera position in world space is:
    // camera_pos = -transpose(R) * t, where R is the 3x3 rotation part and t is the translation
    let r00 = u_terrain.view[0][0];
    let r01 = u_terrain.view[1][0];
    let r02 = u_terrain.view[2][0];
    let r10 = u_terrain.view[0][1];
    let r11 = u_terrain.view[1][1];
    let r12 = u_terrain.view[2][1];
    let r20 = u_terrain.view[0][2];
    let r21 = u_terrain.view[1][2];
    let r22 = u_terrain.view[2][2];
    let tx = u_terrain.view[3][0];
    let ty = u_terrain.view[3][1];
    let tz = u_terrain.view[3][2];
    let camera_pos = vec3<f32>(
        -(r00 * tx + r10 * ty + r20 * tz),
        -(r01 * tx + r11 * ty + r21 * tz),
        -(r02 * tx + r12 * ty + r22 * tz),
    );
    let view_dir = normalize(camera_pos - input.world_position);
    let view_dir_tangent = tbn * view_dir;

    let min_steps = clamp(u32(u_shading.pom_steps.x + 0.5), 1u, 128u);
    let max_steps = clamp(u32(u_shading.pom_steps.y + 0.5), min_steps, 128u);
    let refine_steps = clamp(u32(max(u_shading.pom_steps.z, 0.0)), 0u, 32u);
    let pom_scale = max(u_shading.triplanar_params.w, 0.0);
    let pom_flags = u32(u_shading.pom_steps.w + 0.5);
    let pom_enabled = (pom_flags & 0x1u) != 0u && pom_scale > 0.0;
    let occlusion_enabled = (pom_flags & 0x2u) != 0u;
    let shadow_enabled = (pom_flags & 0x4u) != 0u;

    var pom_uv = uv;
    var pom_offset_magnitude = 0.0;  // Track POM offset for debug visualization
    if (pom_enabled) {
        pom_uv = parallax_occlusion_mapping(
            uv,
            view_dir_tangent,
            pom_scale,
            min_steps,
            max_steps,
            refine_steps
        );
        // Compute offset magnitude (length of UV displacement)
        let pom_offset = pom_uv - uv;
        pom_offset_magnitude = length(pom_offset);
    }
    let parallax_uv = clamp(pom_uv, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
    // Water mask is already flipud'd on the Python side to match heightmap orientation,
    // so sample with the same UV as the heightmap (no additional V flip needed).
    let water_mask_value = textureSampleLevel(water_mask_tex, height_samp, parallax_uv, 0.0).r;
    // Water mask: 0.0 = not water, >0.0 = water (value is shore-distance ratio)
    // Use small epsilon to detect any water, not just shore-distance > 0.5
    let is_water = water_mask_value > 0.001;
    let height_sample = sample_height(parallax_uv);
    let height_clamped = clamp(height_sample, u_shading.clamp0.x, u_shading.clamp0.y);
    var occlusion = 1.0;
    if (occlusion_enabled) {
        occlusion = height_clamped;
    }

    let domain_min = u_overlay.params0.x;
    let inv_range = u_overlay.params0.y;
    let offset = u_overlay.params0.w;
    let height_value = height_clamped + offset;
    var height_norm = height_clamped;
    if (inv_range > 0.0) {
        height_norm = clamp((height_value - domain_min) * inv_range, 0.0, 1.0);
    }

    let tri_scale = max(u_shading.triplanar_params.x, 1e-3);
    let tri_blend = max(u_shading.triplanar_params.y, 1.0);
    // Use base_normal (stable geometric normal) for slope, NOT blended_normal
    // blended_normal has high-frequency perturbations that cause layer selection jitter → flakes
    let slope_raw = 1.0 - abs(base_normal.y);
    let slope_factor = clamp(slope_raw, u_shading.clamp0.z, u_shading.clamp0.w);
    var layer_count = i32(u_shading.layer_control.x + 0.5);
    if (layer_count < 1) {
        layer_count = 1;
    }
    let blend_half = max(u_shading.layer_control.y, 1e-3);
    let base_lod = max(u_shading.clamp2.z, 0.0);
    let lod_bias = u_shading.layer_control.z;
    let lod0_bias = u_shading.layer_control.w;
    let anisotropy = max(u_shading.clamp2.w, 1.0);
    let lod_value = max(base_lod + lod_bias + lod0_bias * slope_factor - (anisotropy - 1.0) * 0.25, 0.0);

    // Compute smooth material weights using Gaussian-like falloff
    // Incorporates both height and slope for natural-looking transitions
    var weights = vec4<f32>(0.0);
    var weight_sum = 0.0;
    let slope_influence = 0.3; // How much slope affects layer selection
    
    for (var idx = 0; idx < 4; idx = idx + 1) {
        if (idx < layer_count) {
            let center = u_shading.layer_heights[idx];
            let dist = abs(height_norm - center);
            // Smooth Gaussian-like falloff instead of linear cutoff
            let sigma = blend_half * 1.5;
            let height_weight = exp(-dist * dist / (2.0 * sigma * sigma));
            
            // Modulate by slope: rock/bare materials favor steep slopes
            // Layer 0 (rock) prefers steep, layer 1 (grass) prefers flat
            var slope_mod = 1.0;
            if (idx == 0) {
                // Rock: boost on steep slopes
                slope_mod = mix(1.0, 1.5, slope_factor);
            } else if (idx == 1) {
                // Grass: reduce on steep slopes
                slope_mod = mix(1.0, 0.5, slope_factor);
            }
            
            let w = height_weight * slope_mod;
            weights[idx] = w;
            weight_sum = weight_sum + w;
        }
    }
    if (weight_sum > 1e-5) {
        weights = weights / weight_sum;
    } else {
        weights = vec4<f32>(0.0);
        weights.x = 1.0;
    }

    var albedo = vec3<f32>(0.0, 0.0, 0.0);
    var roughness = 0.0;
    var metallic = 0.0;
    for (var idx = 0; idx < 4; idx = idx + 1) {
        if (idx < layer_count) {
            let weight = weights[idx];
            let layer = f32(idx);
            // Use base_normal (smooth vertex normal) for triplanar weights, NOT blended_normal
            // blended_normal has high-frequency height perturbations that cause weight jitter → flakes
            let sample_rgb = sample_triplanar(
                input.world_position,
                base_normal,  // STABLE geometric normal for triplanar projection
                tri_scale,
                tri_blend,
                layer,
                lod_value
            );
            albedo = albedo + sample_rgb * weight;
            roughness = roughness + u_shading.layer_roughness[idx] * weight;
            metallic = metallic + u_shading.layer_metallic[idx] * weight;
        }
    }

    // Optional water override: when mask is active, treat surface as water material.
    // Store terrain normal for non-water, give water proper wave normals for reflections
    var shading_normal = blended_normal;
    var water_scatter = vec3<f32>(0.0); // Subsurface scatter contribution for water
    var water_depth_value = 0.0; // Water depth for attenuation (promoted to outer scope)
    if (is_water) {
        // Water material properties
        // Use slightly higher roughness (0.02) for stable highlights without fireflies
        // Still low enough for crisp sun glint but avoids subpixel needle highlights
        let water_roughness = 0.02;
        let water_metallic = 0.0; // Dielectric
        roughness = water_roughness;
        metallic = water_metallic;
        
        // Water depth from distance-to-shore proxy
        // The water_mask_value now encodes normalized distance from shore:
        // - 0.0 = at shoreline (edge of water)
        // - 1.0 = maximum distance from any shore (lake center)
        // This gives physically meaningful depth variation independent of DEM height.
        // Falls back to height-based if mask is binary (old behavior).
        let is_distance_encoded = water_mask_value > 0.01 && water_mask_value < 0.99;
        var shore_depth: f32;
        if (is_distance_encoded) {
            // Use distance-to-shore directly as depth proxy
            shore_depth = water_mask_value;
        } else {
            // Fallback: height-based depth (old behavior for binary masks)
            let water_ceil = 0.20;
            shore_depth = 1.0 - saturate(height_norm / water_ceil);
        }
        water_depth_value = shore_depth; // 0=shore/shallow, 1=deep center
        
        // Beer-Lambert absorption: deeper water = more blue, absorbs red first
        let absorption = vec3<f32>(0.8, 0.15, 0.02); // RGB absorption per unit depth  
        let max_depth = 3.0; // Visual depth scaling
        let transmittance = exp(-absorption * water_depth_value * max_depth);
        
        // Deep water color - saturated blue matching reference image
        // Reference shows bright blue lake - boost saturation significantly
        let deep_water_color = vec3<f32>(0.05, 0.45, 0.95);
        
        // Shallow water near shore - cyan-blue tint
        let shallow_color = vec3<f32>(0.1, 0.5, 0.85);
        
        // Blend based on depth - this gives visible shoreline gradient
        let underwater_color = mix(shallow_color, deep_water_color, water_depth_value);
        
        // Water albedo - vibrant blue that shows through reflections
        albedo = underwater_color;
        
        // Scatter contribution - very strong to ensure blue dominates over gray IBL
        // Reference image shows saturated blue lake, not gray reflections
        // water_depth_value: 0=shore (more bottom visible), 1=deep (less bottom visible)
        water_scatter = underwater_color * (1.0 - water_depth_value * 0.3) * 1.2;
        
        // Directional wind-driven waves (dominant wind direction + secondary)
        // Creates coherent wave patterns that read as water, not noise
        let wx = input.world_position.x;
        let wy = input.world_position.y;
        let wind_angle = 0.7; // ~40 degrees
        let wind_cos = cos(wind_angle);
        let wind_sin = sin(wind_angle);
        let wave_coord_1 = wx * wind_cos + wy * wind_sin;
        let wave_coord_perp = -wx * wind_sin + wy * wind_cos;
        
        // Three octaves of directional waves - amplitude decreases near shore
        let wave_scale = mix(0.3, 1.0, water_depth_value); // Calmer near shore
        let wave1 = sin(wave_coord_1 * 0.05) * 0.07 * wave_scale;
        let wave2 = sin(wave_coord_1 * 0.15 + wave_coord_perp * 0.03) * 0.035 * wave_scale;
        let wave3 = sin(wave_coord_1 * 0.4 + 1.7) * 0.018;
        
        // Cross-wind component (smaller amplitude)
        let cross_wave = sin(wave_coord_perp * 0.12 + 0.5) * 0.02 * wave_scale;
        
        // Combine into normal perturbation
        let wave_dx = (wave1 + wave2 + wave3) * wind_cos + cross_wave * (-wind_sin);
        let wave_dy = (wave1 + wave2 + wave3) * wind_sin + cross_wave * wind_cos;
        
        // Build perturbed normal (Y is up)
        shading_normal = normalize(vec3<f32>(wave_dx, 1.0, wave_dy));
    }
    
    // ──────────────────────────────────────────────────────────────────────────
    // P6: Micro-Detail Application (terrain only, not water)
    // Adds close-range surface detail: detail normals via RNM and albedo noise
    // When detail_enabled = false (params4.x < 0.5), this is a no-op
    // ──────────────────────────────────────────────────────────────────────────
    let view_distance = length(camera_pos - input.world_position);
    let detail_enabled = u_overlay.params4.x > 0.5;
    let detail_scale = max(u_overlay.params4.y, 0.1);
    let detail_normal_strength = clamp(u_overlay.params4.z, 0.0, 1.0);
    let detail_albedo_noise_amp = clamp(u_overlay.params4.w, 0.0, 0.5);
    let detail_fade_start = max(u_overlay.params5.x, 0.0);
    let detail_fade_end = max(u_overlay.params5.y, detail_fade_start + 1.0);
    
    if (detail_enabled && !is_water) {
        // Apply detail normal via RNM blend with distance fade
        shading_normal = apply_detail_normal(
            shading_normal,
            input.world_position,
            view_distance,
            detail_scale,
            detail_normal_strength,
            detail_fade_start,
            detail_fade_end
        );
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Colormap/Overlay System with Debug Modes
    // ──────────────────────────────────────────────────────────────────────────
    // debug_mode already declared at top of fs_main
    let albedo_mode = u32(u_overlay.params1.z + 0.5);
    let colormap_strength = clamp(u_overlay.params1.w, 0.0, 1.0);
    let overlay_strength_raw = u_overlay.params0.z;
    let blend_mode = u32(u_overlay.params1.x + 0.5);

    // Sample colormap LUT
    var lut_u = height_norm;
    if (inv_range <= 0.0) {
        lut_u = clamp(height_clamped, 0.0, 1.0);
    }
    let lut_uv = vec2<f32>(clamp(lut_u, 0.0, 1.0), 0.5);
    let overlay_rgb = textureSample(colormap_tex, colormap_samp, lut_uv).rgb;

    // Apply overlay blend to material albedo (if overlay is active)
    var material_albedo = albedo; // Store original triplanar albedo
    if (overlay_strength_raw > 1e-5) {
        let strength = clamp(overlay_strength_raw, 0.0, 1.0);

        // Blend modes:
        // 0 = Replace
        // 1 = Alpha
        // 2 = Multiply
        // 3 = Additive
        if (blend_mode == 0u) { // Replace
            albedo = overlay_rgb;
        } else if (blend_mode == 1u) { // Alpha
            albedo = mix(albedo, overlay_rgb, strength);
        } else if (blend_mode == 2u) { // Multiply
            albedo = mix(albedo, albedo * overlay_rgb, strength);
        } else if (blend_mode == 3u) { // Additive
            albedo = albedo + strength * overlay_rgb;
        }
    }

    // Apply albedo_mode to determine final albedo
    // 0 = material (triplanar only)
    // 1 = colormap (overlay only, bypasses PBR in debug section below)
    // 2 = mix (blend between material and colormap using colormap_strength)
    // IMPORTANT: Water always uses its own albedo regardless of albedo_mode
    var final_albedo = albedo;
    if (is_water) {
        // Water keeps its blue underwater color - don't override with colormap
        final_albedo = material_albedo;
    } else if (albedo_mode == 0u) { // material
        final_albedo = material_albedo;
    } else if (albedo_mode == 1u) { // colormap
        // Colormap mode: use overlay_rgb directly
        // Note: This path bypasses PBR shading in the debug section below
        final_albedo = overlay_rgb;
    } else if (albedo_mode == 2u) { // mix
        // Mix mode: blend between material albedo and colormap directly
        // colormap_strength=1.0 means full colormap, 0.0 means full material
        final_albedo = mix(material_albedo, overlay_rgb, colormap_strength);
    }

    albedo = clamp(final_albedo, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0));
    
    // P6: Apply procedural albedo brightness noise (terrain only)
    // Adds ±noise_amplitude variation using stable world-space coordinates with distance fade
    if (detail_enabled && !is_water && detail_albedo_noise_amp > 0.0) {
        let albedo_fade = calculate_detail_fade(view_distance, detail_fade_start, detail_fade_end);
        if (albedo_fade > 0.0) {
            let noise_mult = procedural_albedo_noise(input.world_position, detail_albedo_noise_amp * albedo_fade);
            albedo = clamp(albedo * noise_mult, vec3<f32>(0.0), vec3<f32>(1.0));
        }
    }
    
    // P4: Apply slope-based hue variation to increase h_std metric
    // Steep slopes get redder hues, flat areas get yellower hues
    // Using strength of 0.08 - max before hue wraparound breaks monotonicity
    // Achieves h_std ~0.042 (close to target 0.055-0.120)
    if (!is_water) {
        let hue_variation_strength = 0.08;
        albedo = apply_slope_hue_variation(albedo, slope_factor, hue_variation_strength);
    }
    
    occlusion = clamp(occlusion, u_shading.clamp2.x, u_shading.clamp2.y);
    
    // P3: Track base_roughness (for diffuse) and specular_roughness (Toksvig-adjusted)
    // This split allows Toksvig anti-aliasing on specular only, preserving diffuse detail
    var base_roughness = roughness;     // Original roughness for diffuse calculations
    var specular_roughness = roughness; // Toksvig-adjusted roughness for specular
    
    // Specular AA (Toksvig): increase SPECULAR roughness when normal has variance
    // Using screen-space derivatives (dpdx/dpdy) to measure variance.
    // This works for both:
    //   - Procedural terrain normals (where mipmap averaging doesn't exist)
    //   - Synthetic sparkle perturbation (stress test mode 17)
    // The variance proxy σ² is computed from the rate of change of the normal.
    // P3 fix: Apply Toksvig ONLY to specular_roughness, leaving base_roughness unchanged
    let spec_aa_enabled = u_overlay.params2.z > 0.5;
    var specaa_sigma2 = 0.0;  // Track for debug visualization (mode 19)
    if (!is_water && spec_aa_enabled) {
        // Compute normal variance from screen-space derivatives
        let dndx = dpdx(shading_normal);
        let dndy = dpdy(shading_normal);
        // Raw variance: average squared magnitude of derivatives
        let raw_variance = 0.5 * (dot(dndx, dndx) + dot(dndy, dndy));
        
        // SpecAA in the main path (beauty mode, debug=0) is effectively disabled
        // by using a very high variance threshold. This preserves natural terrain detail.
        // 
        // Mode 17 (stress test) uses its own SpecAA path with 100x scale and no threshold,
        // which correctly suppresses the synthetic high-frequency sparkle injection.
        //
        // The sigma_scale env var can still boost effect if needed for specific use cases
        let sigma_scale = max(u_overlay.params2.w, 1.0);
        
        // Very high threshold: effectively disables SpecAA for normal renders
        // Raw_variance for terrain is typically 0.02-0.05, so threshold 1.0 filters all
        // This prevents grid-pattern artifacts from Toksvig following DEM resolution
        let variance_threshold = 1.0;
        let effective_variance = max(raw_variance - variance_threshold, 0.0);
        
        specaa_sigma2 = effective_variance * sigma_scale;
        specaa_sigma2 = clamp(specaa_sigma2, 0.0, 1.0);
        
        let r2 = specular_roughness * specular_roughness;
        // Toksvig formula: r' = sqrt(r² + σ²(1 - r²))
        // P3: Apply ONLY to specular_roughness, not base_roughness
        specular_roughness = sqrt(r2 + specaa_sigma2 * (1.0 - r2));
    }
    
    // Apply roughness multiplier (params2.y, default 1.0 = no change)
    // Used for roughness sweep in PBR proof pack
    let roughness_mult = max(u_overlay.params2.y, 0.001);
    if (roughness_mult != 1.0) {
        base_roughness = base_roughness * roughness_mult;
        specular_roughness = specular_roughness * roughness_mult;
    }
    // P3: Roughness floor - lowered to 0.25 for land (from 0.65) to restore specular detail
    // Toksvig anti-aliasing handles sparkles instead of a high roughness floor
    // Water keeps low roughness (0.02) for crisp reflections (unchanged per P3 constraint)
    let roughness_floor = select(0.25, 0.02, is_water);
    base_roughness = clamp(base_roughness, roughness_floor, 1.0);
    specular_roughness = clamp(specular_roughness, roughness_floor, 1.0);
    // Legacy roughness variable for IBL and other uses (use specular_roughness)
    roughness = specular_roughness;
    metallic = clamp(metallic, 0.0, 1.0);
    var f0 = mix(vec3<f32>(0.04, 0.04, 0.04), albedo, metallic);
    if (is_water) {
        let ior = 1.33;
        let f0_scalar = pow((ior - 1.0) / (ior + 1.0), 2.0);
        f0 = vec3<f32>(f0_scalar, f0_scalar, f0_scalar);
    }
    let light_dir = normalize(u_terrain.sun_exposure.xyz);
    
    // P3: Use split-roughness BRDF for terrain direct lighting
    // base_roughness for diffuse (crisp detail), specular_roughness for specular (anti-aliased)
    var lighting: vec3<f32>;
    if (TERRAIN_USE_BRDF_DISPATCH) {
        // Use unified BRDF dispatch (allows model switching) with specular roughness
        let shading_params = terrain_to_shading_params(specular_roughness, metallic, TERRAIN_BRDF_MODEL);
        let n_dot_l = max(dot(shading_normal, light_dir), 0.0);
        lighting = eval_brdf(shading_normal, view_dir, light_dir, albedo, shading_params) * n_dot_l;
    } else {
        // P3: Use split-roughness BRDF - Toksvig on specular only
        lighting = calculate_pbr_brdf_split_roughness(
            shading_normal,
            view_dir,
            light_dir,
            albedo,
            base_roughness,      // Original roughness for diffuse
            specular_roughness,  // Toksvig-adjusted for specular
            metallic,
            f0,
        );
    }
    lighting = lighting * u_terrain.sun_exposure.w;
    lighting = lighting * u_shading.light_params.rgb;
    
    // P3-10: Apply CSM shadow visibility (optional, gated by TERRAIN_USE_SHADOWS)
    var shadow_debug_color = vec3<f32>(0.0);
    var shadow_visibility = 1.0;
    var shadow_factor = 1.0; // Factor for IBL shadow application
    if (TERRAIN_USE_SHADOWS) {
        // Calculate view-space depth for cascade selection
        let view_pos = u_terrain.view * vec4<f32>(input.world_position, 1.0);
        let view_depth = -view_pos.z; // Positive depth in view space
        
        // Check for cascade debug mode (compile-time OR runtime via csm_uniforms.debug_mode)
        let shadow_debug_mode = csm_uniforms.debug_mode;
        if (DEBUG_SHADOW_CASCADES || shadow_debug_mode == SHADOW_DEBUG_CASCADES) {
            // Debug mode: get both shadow visibility and cascade color
            let shadow_debug = debug_shadow_with_vis(input.world_position, blended_normal, view_depth, input.tex_coord);
            shadow_debug_color = shadow_debug.xyz;
            shadow_visibility = shadow_debug.w;
        } else {
            // Normal mode: just get shadow visibility
            shadow_visibility = calculate_shadow_terrain(input.world_position, blended_normal, view_depth, input.tex_coord);
        }
        
        // Apply shadow to direct lighting with intensity tuning
        // shadow_visibility: 0.0 = fully shadowed, 1.0 = fully lit
        // Map to [SHADOW_MIN, 1.0] for softer shadows that don't go pitch black
        let direct_shadow = mix(SHADOW_MIN, 1.0, shadow_visibility);
        lighting = lighting * direct_shadow;
        
        // Compute factor for IBL shadow (used below)
        shadow_factor = mix(1.0 - SHADOW_IBL_FACTOR, 1.0, shadow_visibility);
    } else {
        // Legacy POM-based shadow factor (preserved for backward compatibility)
        if (shadow_enabled && pom_enabled) {
            let shadow_factor = clamp(mix(0.4, 1.0, occlusion), u_shading.clamp1.z, u_shading.clamp1.w);
            lighting = lighting * shadow_factor;
        }
    }

    // Apply IBL rotation (terrain-specific feature)
    let rotated_normal = rotate_y(shading_normal, u_ibl.sin_theta, u_ibl.cos_theta);
    let rotated_view = rotate_y(view_dir, u_ibl.sin_theta, u_ibl.cos_theta);
    
    // For water: use near-black albedo for IBL (water surface has no diffuse color)
    // The underwater_color (stored in albedo) is for scatter, not surface reflectance
    // Water gets its color from specular reflections (IBL) + subsurface scatter
    var ibl_albedo = albedo;
    if (is_water) {
        // Water surface has negligible diffuse reflection - it's all specular
        // Using black albedo means IBL will be pure specular (sky reflection)
        ibl_albedo = vec3<f32>(0.0, 0.0, 0.0);
    }
    
    // Use unified eval_ibl function from lighting_ibl.wgsl (P4 spec)
    var ibl_contrib = eval_ibl(rotated_normal, rotated_view, ibl_albedo, metallic, roughness, f0);

    // Capture pre-occlusion IBL for debug mode (still apply intensity)
    let ibl_contrib_pre_ao = ibl_contrib * u_ibl.intensity;
    
    // Also compute split IBL for PBR debug modes (diffuse/specular separation)
    let ibl_split = eval_ibl_split(rotated_normal, rotated_view, ibl_albedo, metallic, roughness, f0);
    
    // Apply IBL intensity and occlusion (no artificial boost - proper split-sum should work)
    // For water, don't apply occlusion to IBL (water surface is exposed to sky)
    var ibl_occlusion = occlusion;
    if (is_water) {
        ibl_occlusion = 1.0;
        // Water IBL is pure specular (no diffuse) - this is handled by ibl_albedo = black above
        // The reflection color comes entirely from the environment - no tinting
    }
    // Apply shadow to IBL diffuse (shadowed areas receive less ambient light)
    // But keep specular unaffected (sky reflections should still be visible)
    let ibl_diffuse_with_shadow = ibl_split.diffuse * shadow_factor;
    let ibl_with_shadow = ibl_diffuse_with_shadow + ibl_split.specular;
    ibl_contrib = ibl_with_shadow * u_ibl.intensity * ibl_occlusion;
    
    // Scale split components by intensity and occlusion for debug output
    let ibl_diffuse_scaled = ibl_split.diffuse * u_ibl.intensity * ibl_occlusion * shadow_factor;
    let ibl_specular_scaled = ibl_split.specular * u_ibl.intensity * ibl_occlusion;

    // ──────────────────────────────────────────────────────────────────────────
    // Debug Modes (bypass PBR when debug_mode > 0)
    // ──────────────────────────────────────────────────────────────────────────
    var final_color = vec3<f32>(0.0, 0.0, 0.0);

    if (debug_mode == 1u) {
        // DBG_COLOR_LUT: Show raw LUT color (bypass PBR)
        final_color = overlay_rgb;
    } else if (debug_mode == 2u) {
        // DBG_TRIPLANAR_ALBEDO: Show triplanar material only
        final_color = material_albedo;
    } else if (debug_mode == 3u) {
        // DBG_BLEND_LUT_OVER_ALBEDO: Show lerp(albedo, lut, colormap_strength)
        final_color = mix(material_albedo, overlay_rgb, colormap_strength);
    } else if (debug_mode == DBG_WATER_MASK_BINARY) {
        // ── MODE 4: "What pixels are being treated as water?" ──
        // Uses the EXACT SAME `is_water` variable as the main shading path.
        // CYAN = water branch, MAGENTA = land branch. No shading, no tonemap.
        // Interpretation:
        //   - CYAN outside real lakes → upstream bug (mask generation/upload)
        //   - Mask looks correct but water renders wrong → downstream bug (shader branch)
        let c_water = vec3<f32>(0.0, 1.0, 1.0);  // CYAN
        let c_land  = vec3<f32>(1.0, 0.0, 1.0);  // MAGENTA
        out.color = vec4<f32>(select(c_land, c_water, is_water), 1.0);
        return out;
    } else if (debug_mode == DBG_WATER_MASK_RAW) {
        // ── MODE 5: "What values is the shader receiving for the mask?" ──
        // Shows the EXACT `water_mask_value` being sampled, with error flagging.
        // Catches: wrong texture bound, wrong normalization, wrong channel, NaN/Inf.
        // Interpretation:
        //   - GREEN = NaN/Inf (uninitialized/bad upload)
        //   - RED = value < 0 (invalid normalization)
        //   - YELLOW = value > 1 (invalid normalization)
        //   - Grayscale = value in [0,1] (black=0, white=1)
        //   - Binary mask: expect only two values (black/white)
        //   - Shore-distance gradient: lake edges darker, center brighter
        let m = water_mask_value;
        if (!is_finite_f32(m)) {
            out.color = vec4<f32>(0.0, 1.0, 0.0, 1.0); // GREEN = NaN/Inf
            return out;
        }
        if (m < 0.0) {
            out.color = vec4<f32>(1.0, 0.0, 0.0, 1.0); // RED = <0
            return out;
        }
        if (m > 1.0) {
            out.color = vec4<f32>(1.0, 1.0, 0.0, 1.0); // YELLOW = >1
            return out;
        }
        // In-range: exact grayscale, no gamma, no tonemap
        out.color = vec4<f32>(vec3<f32>(m), 1.0);
        return out;
    } else if (debug_mode == DBG_IBL_ONLY) {
        // ── MODE 6: "Is IBL working, independent of sun/AO/fog?" ──
        // Show pre-occlusion IBL with tonemap; no sun, no AO, no fog, no water tint.
        let mapped = tonemap_aces(ibl_contrib_pre_ao);
        let out_srgb = linear_to_srgb(mapped);
        out.color = vec4<f32>(out_srgb, 1.0);
        return out;
    } else if (debug_mode == DBG_RAW_SSAO) {
        // ── MODE 28: Raw SSAO buffer ──
        // Samples bound AO debug texture (raw SSAO or coarse AO fallback) and outputs grayscale.
        let uv_dbg = clamp(input.tex_coord, vec2<f32>(0.0), vec2<f32>(1.0));
        let ao_raw = textureSampleLevel(ao_debug_tex, ao_debug_samp, uv_dbg, 0.0).r;
        let ao_vis = clamp(ao_raw, 0.0, 1.0);
        out.color = vec4<f32>(vec3<f32>(ao_vis), 1.0);
        return out;
    } else if (debug_mode == DBG_PBR_SPECULAR_ONLY) {
        // ── MODE 8: PBR Specular Only ──
        // Shows ONLY the specular IBL term (no diffuse, no sun).
        // For energy sanity: specular should vary with roughness and view angle.
        // Interpretation:
        //   - Low roughness = sharp reflections, high roughness = blurry
        //   - Grazing angles = stronger specular (Fresnel)
        let specular_linear = max(ibl_specular_scaled, vec3<f32>(0.0));
        let mapped = tonemap_aces(specular_linear);
        let out_srgb = linear_to_srgb(mapped);
        out.color = vec4<f32>(out_srgb, 1.0);
        return out;
    } else if (debug_mode == DBG_PBR_FRESNEL) {
        // ── MODE 9: Fresnel Term Visualization ──
        // Shows the Fresnel term F as grayscale (average of RGB components).
        // For Fresnel behavior: should be stronger at grazing angles.
        // Interpretation:
        //   - Near-normal viewing = F close to F0 (typically ~0.04 for dielectrics)
        //   - Grazing angles = F approaches 1.0 (white)
        let fresnel_avg = (ibl_split.fresnel.r + ibl_split.fresnel.g + ibl_split.fresnel.b) / 3.0;
        out.color = vec4<f32>(vec3<f32>(fresnel_avg), 1.0);
        return out;
    } else if (debug_mode == DBG_PBR_NDOTV) {
        // ── MODE 10: N.V (View Angle) Visualization ──
        // Shows N.V as grayscale (1.0 = normal facing camera, 0.0 = grazing).
        // Interpretation:
        //   - Flat surfaces facing camera = white
        //   - Steep slopes / edges = darker
        //   - Should correlate with where Fresnel is stronger (inverse relationship)
        out.color = vec4<f32>(vec3<f32>(ibl_split.n_dot_v), 1.0);
        return out;
    } else if (debug_mode == DBG_PBR_ROUGHNESS) {
        // ── MODE 11: Roughness Visualization ──
        // Shows the roughness value as grayscale (after any multiplier).
        // Interpretation:
        //   - White = rough (matte), Black = smooth (shiny)
        //   - Should correlate with specular highlight width
        out.color = vec4<f32>(vec3<f32>(roughness), 1.0);
        return out;
    } else if (debug_mode == DBG_PBR_ENERGY) {
        // ── MODE 12: Energy (Diffuse + Specular) Before Tonemap ──
        // Shows raw luminance of (diffuse + specular) for energy histogram.
        // NO tonemap - this is for quantitative analysis.
        // Interpretation:
        //   - Values should rarely exceed 1.0 for dielectrics with IBL only
        //   - Use this to generate fig_pbr_energy_hist.png
        //   - Encode: luminance clamped to [0,1] as grayscale (saturated = energy > 1)
        let energy_linear = ibl_diffuse_scaled + ibl_specular_scaled;
        let energy_luma = luminance(energy_linear);
        // Clamp at 1.0 - anything above shows as pure white (energy violation)
        let energy_vis = clamp(energy_luma, 0.0, 1.0);
        out.color = vec4<f32>(vec3<f32>(energy_vis), 1.0);
        return out;
    } else if (debug_mode == DBG_PBR_LINEAR_COMBINED) {
        // ── MODE 13: Linear Unclamped (Diffuse + Specular) ──
        // For recomposition proof: encode linear RGB in [0,4] range to [0,1] for PNG export.
        // Decode in Python: linear = encoded * 4.0
        // This should equal ibl_contrib_pre_ao (before AO is applied)
        let combined_linear = ibl_diffuse_scaled + ibl_specular_scaled;
        let encoded = clamp(combined_linear / 4.0, vec3<f32>(0.0), vec3<f32>(1.0));
        out.color = vec4<f32>(encoded, 1.0);
        return out;
    } else if (debug_mode == DBG_PBR_LINEAR_DIFFUSE) {
        // ── MODE 14: Linear Unclamped Diffuse Only ──
        // Encode linear diffuse in [0,4] range to [0,1] for PNG export.
        let encoded = clamp(ibl_diffuse_scaled / 4.0, vec3<f32>(0.0), vec3<f32>(1.0));
        out.color = vec4<f32>(encoded, 1.0);
        return out;
    } else if (debug_mode == DBG_PBR_LINEAR_SPECULAR) {
        // ── MODE 15: Linear Unclamped Specular Only ──
        // Encode linear specular in [0,4] range to [0,1] for PNG export.
        let encoded = clamp(ibl_specular_scaled / 4.0, vec3<f32>(0.0), vec3<f32>(1.0));
        out.color = vec4<f32>(encoded, 1.0);
        return out;
    } else if (debug_mode == DBG_PBR_RECOMP_ERROR) {
        // ── MODE 16: Recomposition Error Heatmap ──
        // Shows abs(ibl_total - (diffuse + specular)) amplified 100x.
        // This should be near-zero if IBL = diffuse + specular.
        // Interpretation:
        //   - Black = perfect recomposition (error < 0.01 linear)
        //   - Any color = error (amplified to be visible)
        //   - If P95 error < 0.001, recomposition is correct
        let recomposed = ibl_diffuse_scaled + ibl_specular_scaled;
        let error = abs(ibl_contrib_pre_ao - recomposed);
        // Amplify 100x so small errors are visible
        let error_vis = clamp(error * 100.0, vec3<f32>(0.0), vec3<f32>(1.0));
        out.color = vec4<f32>(error_vis, 1.0);
        return out;
    } else if (debug_mode == DBG_SPECAA_SPARKLE) {
        // ── MODE 17: SpecAA Sparkle Stress Test ──
        // Inject synthetic high-frequency normal perturbation to stress-test Toksvig.
        // The perturbation creates a checkerboard pattern that should cause sparkles
        // unless SpecAA properly widens the lobe via screen-derivative variance.
        // Metric: Compare high-freq energy between SpecAA ON vs OFF.
        
        // Generate synthetic high-freq normal perturbation (screen-space checkerboard)
        let screen_pos = input.clip_position.xy;
        
        // Perturb shading normal with high-frequency tangent-space noise
        // This creates rapid normal changes that dpdx/dpdy will detect
        let perturb_strength = 0.3; // Strong enough to cause visible sparkles
        let perturb = vec3<f32>(
            sin(screen_pos.x * 7.3 + screen_pos.y * 3.7) * perturb_strength,
            sin(screen_pos.x * 5.1 - screen_pos.y * 8.3) * perturb_strength,
            0.0
        );
        let sparkle_normal = normalize(shading_normal + perturb);
        
        // CRITICAL: Recompute variance on the PERTURBED normal
        // This is where SpecAA must detect the high-frequency variation
        var sparkle_roughness = roughness;
        var sparkle_sigma2_for_debug = 0.0;  // For debug mode 20
        if (spec_aa_enabled) {
            let dndx_sparkle = dpdx(sparkle_normal);
            let dndy_sparkle = dpdy(sparkle_normal);
            // Use same 100x amplification as main path
            let sigma_scale = max(u_overlay.params2.w, 1.0) * 100.0;
            let sparkle_sigma2 = 0.5 * (dot(dndx_sparkle, dndx_sparkle) + dot(dndy_sparkle, dndy_sparkle)) * sigma_scale;
            sparkle_sigma2_for_debug = sparkle_sigma2;  // Save raw value for debug
            let sparkle_sigma2_clamped = clamp(sparkle_sigma2, 0.0, 1.0);
            
            // Apply Toksvig roughness boost on perturbed normal
            let r2_sparkle = sparkle_roughness * sparkle_roughness;
            sparkle_roughness = sqrt(r2_sparkle + sparkle_sigma2_clamped * (1.0 - r2_sparkle));
            sparkle_roughness = clamp(sparkle_roughness, 0.04, 1.0);
        }
        
        // Recompute specular IBL with perturbed normal and SpecAA-corrected roughness
        let sparkle_ibl = eval_ibl(
            sparkle_normal,
            view_dir,
            albedo,
            metallic,
            sparkle_roughness, // Uses freshly-computed Toksvig roughness on perturbed normal
            f0
        );
        
        // Extract specular component (approximate: assume kD same ratio)
        let n_dot_v_sparkle = saturate(dot(sparkle_normal, view_dir));
        let f_sparkle = fresnel_schlick_roughness(n_dot_v_sparkle, f0, sparkle_roughness);
        let kS_sparkle = f_sparkle;
        let kD_sparkle = (vec3<f32>(1.0) - kS_sparkle) * (1.0 - metallic);
        let total_k = kD_sparkle + kS_sparkle;
        let spec_ratio = kS_sparkle / max(total_k, vec3<f32>(0.001));
        let sparkle_spec = sparkle_ibl * spec_ratio * u_ibl.intensity;
        
        // Output with tonemap for visibility
        let mapped = tonemap_aces(sparkle_spec);
        let out_srgb = linear_to_srgb(mapped);
        out.color = vec4<f32>(out_srgb, 1.0);
        return out;
    } else if (debug_mode == DBG_POM_OFFSET_MAG) {
        // ── MODE 18: POM Offset Magnitude Visualization ──
        // Shows the parallax UV offset magnitude as grayscale.
        // Black = no offset (POM disabled or flat area)
        // White = maximum offset (areas with strong parallax displacement)
        // This proves POM is actually displacing texture coordinates.
        // Interpretation:
        //   - Should correlate with view angle (more offset at grazing angles)
        //   - Should correlate with height variation (ridges/valleys show more offset)
        //   - If uniformly black when POM enabled → POM not working or pom_scale too small
        //   - If uniform noise → something wrong with height sampling
        // Scale factor: POM offset is typically small (0.0-0.1 UV units)
        // We amplify by 10x for visibility (0.1 UV offset → 1.0 grayscale)
        let offset_vis = clamp(pom_offset_magnitude * 10.0, 0.0, 1.0);
        out.color = vec4<f32>(vec3<f32>(offset_vis), 1.0);
        return out;
    } else if (debug_mode == DBG_SPECAA_SIGMA2) {
        // ── MODE 19: SpecAA Sigma² (Variance) Visualization ──
        // Shows the normal variance (σ²) used by SpecAA/Toksvig as grayscale.
        // Black = no variance (flat normal field)
        // White = high variance (high-frequency normal changes)
        // Interpretation:
        //   - Terrain edges/ridges should show higher variance
        //   - Flat areas should be near-black
        //   - If uniformly black when SpecAA enabled → variance not being detected
        //   - Should correlate with where sparkle reduction occurs
        // Scale: σ² is typically small (0.0-0.1), amplify 20x for visibility
        let sigma2_vis = clamp(specaa_sigma2 * 20.0, 0.0, 1.0);
        out.color = vec4<f32>(vec3<f32>(sigma2_vis), 1.0);
        return out;
    } else if (debug_mode == DBG_SPECAA_SPARKLE_SIGMA2) {
        // ── MODE 20: SpecAA Sparkle Sigma² Visualization ──
        // Shows the variance computed on the SPARKLE-PERTURBED normal.
        // This proves that dpdx/dpdy can detect the synthetic perturbation.
        // Should show high variance (bright) across entire terrain.
        // If black: variance not being detected on perturbed normals.
        
        // Generate same sparkle perturbation as mode 17
        let screen_pos = input.clip_position.xy;
        let perturb_strength = 0.3;
        let perturb = vec3<f32>(
            sin(screen_pos.x * 7.3 + screen_pos.y * 3.7) * perturb_strength,
            sin(screen_pos.x * 5.1 - screen_pos.y * 8.3) * perturb_strength,
            0.0
        );
        let sparkle_normal_dbg = normalize(shading_normal + perturb);
        
        // Compute variance on perturbed normal
        let dndx_dbg = dpdx(sparkle_normal_dbg);
        let dndy_dbg = dpdy(sparkle_normal_dbg);
        let sigma_scale_dbg = max(u_overlay.params2.w, 1.0) * 100.0;
        let sparkle_sigma2_dbg = 0.5 * (dot(dndx_dbg, dndx_dbg) + dot(dndy_dbg, dndy_dbg)) * sigma_scale_dbg;
        
        // Visualize (no additional scaling - should already be visible with 100x)
        let sigma2_vis_sparkle = clamp(sparkle_sigma2_dbg, 0.0, 1.0);
        out.color = vec4<f32>(vec3<f32>(sigma2_vis_sparkle), 1.0);
        return out;
    } else if (debug_mode == DBG_TRIPLANAR_WEIGHTS) {
        // ── MODE 21: Triplanar Blend Weights Visualization ──
        // Shows RGB = x/y/z projection weights.
        // T1 requirement: wx + wy + wz = 1, weights change smoothly with normal.
        // Interpretation:
        //   - RED dominant = surface faces X axis (YZ plane projection dominant)
        //   - GREEN dominant = surface faces Y axis (XZ plane projection dominant, i.e. flat/horizontal)
        //   - BLUE dominant = surface faces Z axis (XY plane projection dominant)
        //   - Steep cliffs should show RED or BLUE, flat areas should show GREEN
        //   - Transitions should be smooth, not abrupt
        // Use base_normal for stable triplanar weights (no high-frequency jitter)
        let tri_weights = compute_triplanar_weights(base_normal, tri_blend);
        out.color = vec4<f32>(tri_weights, 1.0);
        return out;
    } else if (debug_mode == DBG_TRIPLANAR_CHECKER) {
        // ── MODE 22: Triplanar Checker Pattern ──
        // Shows a procedural checker pattern sampled via triplanar mapping.
        // Interpretation:
        //   - Checker squares should be uniform size (no distortion) when projection matches surface
        //   - If squares stretch on steep slopes, triplanar isn't working correctly
        //   - Pattern should remain world-space stable (no swimming with camera movement)
        let checker_scale = 8.0; // Number of checker squares per world unit
        // Use base_normal for stable triplanar weights
        let checker_val = sample_triplanar_checker(
            input.world_position,
            base_normal,  // Stable geometric normal
            tri_scale,
            tri_blend,
            checker_scale
        );
        // Output as grayscale for clear checker visibility
        out.color = vec4<f32>(vec3<f32>(checker_val), 1.0);
        return out;
    } else if (debug_mode == DBG_FLAKE_NO_SPECULAR) {
        // ── MODE 23: No Specular (Diffuse Only) ──
        // Shows terrain with ONLY diffuse/ambient lighting (no IBL specular).
        // If flakes disappear here → flakes are specular aliasing.
        let ambient_strength_23 = mix(u_shading.clamp1.x, u_shading.clamp1.y, 1.0 - abs(blended_normal.y));
        let ambient_23 = albedo * ambient_strength_23;
        let direct_mult_23 = mix(0.65, 1.0, occlusion);
        let diffuse_only_23 = ibl_diffuse_scaled;
        let shaded_no_spec = ambient_23 + lighting * direct_mult_23 + diffuse_only_23;
        let exposure_23 = max(u_shading.light_params.w, 0.0);
        let tonemapped_23 = tonemap_aces(shaded_no_spec * exposure_23);
        out.color = vec4<f32>(linear_to_srgb(tonemapped_23), 1.0);
        return out;
    } else if (debug_mode == DBG_FLAKE_NO_HEIGHT_NORMAL) {
        // ── MODE 24: No Height Normal ──
        // Uses base_normal (geometric normal) instead of height-derived normal.
        // If flakes disappear here → flakes are from height-normal frequency.
        // Normal substitution is done at top of shader.
        // Fall through to normal shading path with the substituted normal.
    } else if (debug_mode == DBG_FLAKE_DDXDDY_NORMAL) {
        // ── MODE 25: Derivative-Based Normal (Ground Truth) ──
        // Uses n_dd = cross(dpdx, dpdy) as the shading normal.
        // This is the "mathematically correct" per-pixel surface normal.
        // Normal substitution is done at top of shader.
        // Fall through to normal shading path with the substituted normal.
    } else if (debug_mode == DBG_FLAKE_HEIGHT_LOD) {
        // ── MODE 26: Height LOD Visualization ──
        // Grayscale ramp from computed LOD (already distinct)
        let max_lod = f32(textureNumLevels(height_tex) - 1u);
        let lod_normalized = height_lod / max(max_lod, 1.0);
        out.color = vec4<f32>(vec3<f32>(lod_normalized), 1.0);
        return out;
    } else if (debug_mode == DBG_FLAKE_NORMAL_BLEND) {
        // ── MODE 27: Effective Normal Blend Visualization ──
        // Grayscale ramp from normal_blend (already distinct)
        out.color = vec4<f32>(vec3<f32>(normal_blend), 1.0);
        return out;
    } else if (debug_mode == 100u) {
        // DBG_WATER_BINARY: Unambiguous binary water classification
        // Blue = water, Dark gray = land. No lighting, no tonemap.
        final_color = debug_water_is_water(is_water);
    } else if (debug_mode == 101u) {
        // DBG_WATER_SCALAR: Shore-distance gradient visualization
        // Falsecolor ramp with white shoreline ring on water, dark gray on land.
        // If gradient appears in wrong location => UV orientation bug
        final_color = debug_water_scalar(is_water, water_mask_value);
    } else if (debug_mode == 102u) {
        // DBG_WATER_IBL_SPEC_ISOLATED: IBL specular on water ONLY
        // Land is pure black (no cheating). Water shows compressed HDR IBL.
        final_color = debug_water_ibl_spec_only(is_water, ibl_contrib);
    } else if (debug_mode == 103u) {
        // DBG_PREFILT_RAW: Direct sample of specular cubemap (no Fresnel)
        // This isolates whether the cubemap itself is returning data
        let refl_dir = reflect(-view_dir, shading_normal);
        let rot_refl = rotate_y(refl_dir, u_ibl.sin_theta, u_ibl.cos_theta);
        let prefilt = textureSampleLevel(envSpecular, envSampler, rot_refl, 0.0).rgb;
        final_color = debug_water_prefilt_raw(is_water, prefilt);
    } else if (debug_mode == 104u) {
        // DBG_CUBEMAP_DIRECT: Sample envSpecular with unrotated reflection, ignore Fresnel
        // Water: use (0.0, 1.0, 0.0) reflection = straight up to sky
        let sky_dir = vec3<f32>(0.0, 1.0, 0.0);
        let prefilt_sky = textureSampleLevel(envSpecular, envSampler, sky_dir, 0.0).rgb;
        final_color = debug_water_prefilt_raw(is_water, prefilt_sky);
    } else if (debug_mode == 105u) {
        // DBG_IRRADIANCE_DIRECT: Sample envIrradiance (irradiance cubemap) with up direction
        // This tests if the OTHER cubemap (envIrradiance) has data
        let sky_dir = vec3<f32>(0.0, 1.0, 0.0);
        let irr_sky = textureSampleLevel(envIrradiance, envSampler, sky_dir, 0.0).rgb;
        // Use same helper as specular - cyan means data, magenta means zero
        // Reuse the debug_water_prefilt_raw function which already has this logic
        final_color = debug_water_prefilt_raw(is_water, irr_sky);
    } else if (debug_mode == 110u) {
        // DBG_PURE_RED: Sanity check - just return pure red
        // If this doesn't show red, debug_mode isn't being set correctly
        final_color = vec3<f32>(1.0, 0.0, 0.0);
    } else if (debug_mode == 111u) {
        // DBG_SPEC_NEG_Z: Sample envSpecular with -Z direction (front face)
        let front_dir = vec3<f32>(0.0, 0.0, -1.0);
        let prefilt_front = textureSampleLevel(envSpecular, envSampler, front_dir, 0.0).rgb;
        final_color = debug_water_prefilt_raw(is_water, prefilt_front);
    } else if (debug_mode == 112u) {
        // DBG_SPEC_POS_X: Sample envSpecular with +X direction (right face)
        let right_dir = vec3<f32>(1.0, 0.0, 0.0);
        let prefilt_right = textureSampleLevel(envSpecular, envSampler, right_dir, 0.0).rgb;
        final_color = debug_water_prefilt_raw(is_water, prefilt_right);
    } else if (debug_mode == 113u) {
        // DBG_IRR_NEG_Z: Sample envIrradiance with -Z direction
        let front_dir = vec3<f32>(0.0, 0.0, -1.0);
        let irr_front = textureSampleLevel(envIrradiance, envSampler, front_dir, 0.0).rgb;
        final_color = debug_water_prefilt_raw(is_water, irr_front);
    } else {
        // Normal PBR shading path
        var shaded = vec3<f32>(0.0);
        
        if (is_water) {
            // Water: specular-dominant shading (minimal diffuse, strong reflections)
            // Fresnel is already baked into eval_ibl via fresnel_schlick_roughness
            // Direct specular from sun
            let n_dot_v = max(dot(shading_normal, view_dir), 0.001);
            let n_dot_l = max(dot(shading_normal, light_dir), 0.0);
            let h = normalize(view_dir + light_dir);
            let n_dot_h = max(dot(shading_normal, h), 0.0);
            let v_dot_h = max(dot(view_dir, h), 0.001);
            
            // GGX Distribution for water - use proper alpha² without over-clamping
            // For very smooth surfaces (roughness ~0.01), D can legitimately reach 10000+
            // This is physically correct and produces natural sun glints
            let alpha = roughness * roughness;
            let alpha2 = max(alpha * alpha, 1e-8); // Minimal clamp for numerical stability only
            let n_dot_h2 = n_dot_h * n_dot_h;
            let denom = n_dot_h2 * (alpha2 - 1.0) + 1.0;
            let D = alpha2 / (PI * denom * denom);
            
            // Fresnel (Schlick) using v_dot_h for correct specular Fresnel
            let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - v_dot_h, 5.0);
            
            // Geometry term (Smith GGX, height-correlated)
            // For very smooth surfaces, G approaches 1.0
            let k = alpha / 2.0;
            let g_v = n_dot_v / (n_dot_v * (1.0 - k) + k);
            let g_l = n_dot_l / (n_dot_l * (1.0 - k) + k);
            let G = g_v * g_l;
            
            // Cook-Torrance specular BRDF (no artificial boosts!)
            let spec_denom = 4.0 * n_dot_v * n_dot_l + 0.0001;
            let direct_spec = D * fresnel * G / spec_denom;
            
            // Sun contribution - NO artificial boost; proper GGX + low roughness = natural glints
            let sun_color = vec3<f32>(1.0, 0.98, 0.95); // Slightly warm sun
            let sun_intensity = u_shading.light_params.z; // Use actual sun intensity (no boost!)
            let sun_spec = direct_spec * sun_color * sun_intensity * n_dot_l;
            
            // ─────────────────────────────────────────────────────────────────────
            // P4: Planar Reflection Integration
            // Sample reflection texture with wave-based distortion and Fresnel mixing
            // ─────────────────────────────────────────────────────────────────────
            
            // Sample planar reflection with wave distortion
            // water_depth_value is the shore distance (0=shore, 1=deep)
            let planar_refl = sample_water_reflection(
                input.world_position,
                shading_normal,  // Wave-perturbed normal for UV distortion
                water_depth_value
            );
            let planar_refl_valid = planar_refl.a > 0.5;
            
            // Calculate Fresnel factor for reflection blending
            let water_fresnel = calculate_water_fresnel(view_dir, shading_normal);
            
            // Water shading strategy for visible depth gradient:
            // 1. Planar reflections give terrain reflections on water (P4 feature)
            // 2. IBL gives environmental reflections (sky, distant environment)
            // 3. Sun specular gives glints where waves face the sun
            // 4. Depth tint modulates the overall brightness based on water depth
            // 
            // For depth to be visible, we need to darken deep water regions
            // even when they have specular highlights. This mimics how real
            // deep water absorbs more light than shallow water.
            
            // Use water_depth_value (promoted to outer scope)
            // water_depth_value: 1.0 = deep, 0.0 = shallow
            // Depth attenuation: shallow = 100%, deep = 30% (70% absorbed)
            // This is aggressive but necessary for depth to be visible against bright specular
            let depth_atten = mix(1.0, 0.3, water_depth_value);
            
            // Blend planar reflection with IBL
            // When planar reflection is valid, it takes priority over IBL for terrain reflections
            // IBL still contributes sky/environment reflections where planar reflection is weak
            var combined_reflection = ibl_contrib;
            if (planar_refl_valid) {
                // Blend planar reflection based on Fresnel and intensity
                combined_reflection = blend_water_reflection(
                    ibl_contrib,      // Base color (IBL fallback)
                    planar_refl.rgb,  // Planar reflection
                    true,
                    water_fresnel,
                    water_depth_value
                );
            }
            
            // Combine: minimal reflections to let blue water color dominate
            // Reference image shows saturated blue lake - prioritize underwater color
            let reflective = (combined_reflection * 0.1 + sun_spec * 0.5) * depth_atten;
            
            // Water color: force blue to dominate over gray IBL
            // Direct blue contribution scaled by exposure-appropriate factor
            let blue_water = vec3<f32>(0.15, 0.45, 0.85); // Saturated blue
            shaded = reflective * 0.3 + blue_water * 0.8 + water_scatter * 2.0;
            
        } else {
            // ══════════════════════════════════════════════════════════════════════
            // P2-S4: Terrain Lighting Composition (structure locked per spec)
            // ══════════════════════════════════════════════════════════════════════
            // 
            // P2-S1: ambient_floor must be in [0.22, 0.38] range
            // P2-S2: AO must be clamped to min 0.65 (max 35% darkening)
            // P2-S3: Shadow factor must be clamped to [0.30, 1.0]
            // P2-L2: Dynamic range (valleys vs ridges) should be 5.8-7.8
            //
            // Reference quantiles: q5=0.10, q50=0.35, q95=0.65
            
            // Compute lambert term for hillshade-like appearance
            let n_dot_l_terrain = max(dot(shading_normal, light_dir), 0.0);
            
            // P2-S4: diffuse_raw = mix(ambient_floor, 1.0, lambert)
            // This ensures even backfacing surfaces get ambient_floor illumination
            let diffuse_raw = mix(AMBIENT_FLOOR, 1.0, n_dot_l_terrain);
            
            // P5: Apply AO multiplier from coarse heightmap AO or SSAO
            // P2-S2: AO must be clamped to min 0.65 (max 35% darkening)
            let ao_weight = u_overlay.params3.x;
            var ao_clamped = 1.0;
            if (ao_weight > 0.0) {
                let ao_uv = clamp(input.tex_coord, vec2<f32>(0.0), vec2<f32>(1.0));
                let ao_sample = textureSampleLevel(ao_debug_tex, ao_debug_samp, ao_uv, 0.0).r;
                ao_clamped = mix(1.0, max(ao_sample, 0.65), ao_weight);
            }
            // Also apply POM occlusion with same clamp
            ao_clamped = ao_clamped * max(occlusion, 0.65);
            
            // P2-S3: Shadow factor clamped to [0.30, 1.0]
            // Use the already-mapped shadow value from CSM
            let shadow_clamped = max(shadow_factor, 0.30);
            
            // P3-S1: Compute combined shadow/AO attenuation
            // Direct product for full contrast range (no sqrt compression)
            // P3 requires lf_max/lf_min >= 4.5
            let ao_shadow_factor = ao_clamped * shadow_clamped; // Range [0.195, 1.0]
            let diffuse_lit = diffuse_raw * ao_shadow_factor;
            
            // P3-S1: IBL term adds minimal fill light
            // Reduced to allow deeper shadows while preventing pitch-black
            let warm_bias = vec3<f32>(1.08, 1.0, 0.92);
            let ibl_diffuse_biased = ibl_split.diffuse * warm_bias;
            let ibl_diffuse_factor = length(ibl_diffuse_biased) * u_ibl.intensity;
            let ibl_term = ibl_diffuse_factor * AMBIENT_FLOOR * 0.35;
            
            // P2-S4: lighting_factor = diffuse_lit + ibl_term
            let lighting_factor = diffuse_lit + ibl_term;
            
            // Apply lighting factor to albedo
            // Spec H-03: Lighting modulates brightness, not colormap lookup
            let lit_albedo = albedo * lighting_factor;
            
            // Add specular contribution (capped at 25% per P2-S4)
            // Specular for terrain must not exceed 25% of total RGB
            let spec_contrib = ibl_split.specular * u_ibl.intensity * 0.12;
            let spec_capped = min(spec_contrib, albedo * 0.20);
            
            // Final terrain shading
            shaded = lit_albedo + spec_capped;
        }
        
        let exposure = max(u_shading.light_params.w, 0.0);
        shaded = shaded * exposure;
        
        // P2: Apply atmospheric fog after exposure, before tonemap
        // When fog_density = 0, this is a no-op preserving P1 output
        shaded = apply_atmospheric_fog(shaded, input.world_position, camera_pos);
        
        // Use ACES tonemapping for better highlight preservation than Reinhard
        let tonemapped = tonemap_aces(shaded);
        final_color = tonemapped;
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Milestone 1: Mode Stamp Overlay
    // ──────────────────────────────────────────────────────────────────────────
    // Add debug_mode/255 to blue channel in top-left 8x8 corner for visual mode ID.
    // This allows verifying which debug mode actually rendered (trust but verify).
    // The stamp is small enough not to interfere with visual inspection.
    if (debug_mode > 0u) {
        let stamp_size = 8.0;
        let screen_pos = input.clip_position.xy;
        if (screen_pos.x < stamp_size && screen_pos.y < stamp_size) {
            // Encode mode in blue channel: 0.0-1.0 maps to mode 0-255
            let mode_signal = f32(debug_mode) / 255.0;
            final_color.b = clamp(final_color.b + mode_signal, 0.0, 1.0);
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // P1-Shadow Debug: Cascade Visualization Overlay
    // Activated by compile-time DEBUG_SHADOW_CASCADES or runtime csm_uniforms.debug_mode == 1
    // ──────────────────────────────────────────────────────────────────────────
    let csm_debug_mode = csm_uniforms.debug_mode;
    if (TERRAIN_USE_SHADOWS && (DEBUG_SHADOW_CASCADES || csm_debug_mode == SHADOW_DEBUG_CASCADES)) {
        // Calculate view depth for cascade selection
        let view_pos_dbg = u_terrain.view * vec4<f32>(input.world_position, 1.0);
        let view_depth_dbg = -view_pos_dbg.z;
        // Force cascade 0 for debugging - shadow depth pass only renders cascade 0
        let cascade_idx_dbg = 0u;  // Force cascade 0
        
        // Get shadow UV coordinates and NDC using normalized position
        let shadow_pos_dbg = normalize_for_shadow(input.tex_coord);
        let cascade_dbg = csm_uniforms.cascades[cascade_idx_dbg];
        let light_space_pos_dbg = cascade_dbg.light_view_proj * vec4<f32>(shadow_pos_dbg, 1.0);
        let ndc_dbg = light_space_pos_dbg.xyz / light_space_pos_dbg.w;
        let shadow_uv_dbg = vec2<f32>(ndc_dbg.x * 0.5 + 0.5, ndc_dbg.y * -0.5 + 0.5);
        let compare_depth_dbg = ndc_dbg.z;
        
        // Sample shadow map depth
        let sampled_depth_dbg = textureSampleLevel(
            shadow_maps,
            moment_sampler,
            shadow_uv_dbg,
            i32(cascade_idx_dbg),
            0.0
        );
        
        // Depth difference: positive = receiver behind shadow map surface (shadow)
        //                   negative = receiver in front (lit)
        let depth_diff = compare_depth_dbg - sampled_depth_dbg;
        
        // Debug: Find the threshold where shadow comparison switches
        // Binary search: find depth where compare starts failing
        var low_d = 0.0;
        var high_d = 1.0;
        for (var i = 0; i < 10; i = i + 1) {
            let mid_d = (low_d + high_d) * 0.5;
            let result = textureSampleCompare(shadow_maps, shadow_sampler, shadow_uv_dbg, i32(cascade_idx_dbg), mid_d);
            if (result > 0.5) {
                low_d = mid_d;  // Passing, shadow map depth is higher than mid
            } else {
                high_d = mid_d;  // Failing, shadow map depth is lower than mid
            }
        }
        // low_d is approximately the shadow map depth value
        // Visualize the actual shadow comparison result
        // If depths match (diff < 0.1), the shadow comparison should work
        let shadow_vis = textureSampleCompare(shadow_maps, shadow_sampler, shadow_uv_dbg, i32(cascade_idx_dbg), ndc_dbg.z);
        
        // R = shadow map depth (binary search result)
        // G = main shader depth (expected value)
        // B = actual shadow comparison result at main shader depth
        final_color = vec3<f32>(low_d, ndc_dbg.z, shadow_vis);
        
        // Debug: Compare ndc.z (after matrix) vs shadow_map_depth
        // R = ndc.z (main shader's computed depth after matrix transform)
        // G = shadow_map_depth (what shadow shader wrote)
        // B = 1.0 if depths within 0.05 of each other, 0.0 otherwise
        let depth_match = select(0.0, 1.0, abs(ndc_dbg.z - low_d) < 0.05);
        final_color = vec3<f32>(ndc_dbg.z, low_d, depth_match);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // P1-Shadow Debug: Raw Shadow Visibility
    // Activated by compile-time DEBUG_SHADOW_RAW or runtime csm_uniforms.debug_mode == 2
    // ──────────────────────────────────────────────────────────────────────────
    if (TERRAIN_USE_SHADOWS && (DEBUG_SHADOW_RAW || csm_debug_mode == SHADOW_DEBUG_RAW)) {
        // Show raw shadow_visibility as grayscale
        // Black = fully shadowed (0.0), White = fully lit (1.0)
        // Red tint = in shadow (visibility < 0.5)
        if (shadow_visibility < 0.5) {
            // In shadow - show as red-tinted grayscale
            final_color = vec3<f32>(shadow_visibility * 2.0, shadow_visibility * 0.5, shadow_visibility * 0.5);
        } else {
            // Lit - show as grayscale
            final_color = vec3<f32>(shadow_visibility, shadow_visibility, shadow_visibility);
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Gamma Correction
    // ──────────────────────────────────────────────────────────────────────────
    // Apply gamma correction if render target is Rgba8Unorm (not sRGB)
    // The gamma parameter is passed from Rust
    let gamma = max(u_overlay.params2.x, 0.1);
    let gamma_corrected = gamma_correct(final_color, gamma);

    out.color = vec4<f32>(gamma_corrected, 1.0);
    return out;
}

