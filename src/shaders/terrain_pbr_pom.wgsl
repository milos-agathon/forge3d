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

// P3-10: Optional shadow sampling flag (default: false to avoid regressions)
// Set to true to enable CSM shadow visibility on terrain direct lighting
const TERRAIN_USE_SHADOWS: bool = false;

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
    params2 : vec4<f32>, // gamma, pad, pad, pad
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
// Shadow cascade data (matches CsmUniforms from shadows.wgsl)
struct ShadowCascade {
    light_projection: mat4x4<f32>,
    near_distance: f32,
    far_distance: f32,
    texel_size: f32,
    _padding: f32,
}

struct CsmUniforms {
    light_direction: vec4<f32>,
    light_view: mat4x4<f32>,
    cascades: array<ShadowCascade, 4>,
    cascade_count: u32,
    pcf_kernel_size: u32,
    technique: u32,
    debug_mode: u32,
    depth_bias: f32,
    slope_bias: f32,
    peter_panning_offset: f32,
    cascade_blend_range: f32,
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
    let cascade = csm_uniforms.cascades[cascade_idx];
    
    // Transform to light space
    let light_space_pos = cascade.light_projection * csm_uniforms.light_view * vec4<f32>(world_pos, 1.0);
    let ndc = light_space_pos.xyz / light_space_pos.w;
    
    // Convert to texture coordinates [0,1]
    let shadow_coords = vec2<f32>(ndc.x * 0.5 + 0.5, ndc.y * -0.5 + 0.5);
    
    // Check if outside shadow map bounds
    if (shadow_coords.x < 0.0 || shadow_coords.x > 1.0 || shadow_coords.y < 0.0 || shadow_coords.y > 1.0) {
        return 1.0; // No shadow outside bounds
    }
    
    // Apply depth bias
    let light_dir_norm = normalize(csm_uniforms.light_direction.xyz);
    let n_dot_l = max(dot(normal, light_dir_norm), 0.0);
    let bias = max(csm_uniforms.depth_bias * (1.0 - n_dot_l), csm_uniforms.peter_panning_offset);
    let compare_depth = ndc.z - bias;
    
    // Simple PCF 3x3
    let texel_size = cascade.texel_size;
    var shadow_sum = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
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
    
    return shadow_sum / 9.0;
}

/// Calculate shadow visibility for terrain
fn calculate_shadow_terrain(world_pos: vec3<f32>, normal: vec3<f32>, view_depth: f32) -> f32 {
    // Select appropriate cascade
    let cascade_idx = select_cascade_terrain(view_depth);
    
    // Sample shadow with PCF
    return sample_shadow_pcf_terrain(world_pos, normal, cascade_idx);
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

/// Calculate normal from height map using Sobel filter populated with parallax scale.
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

/// Triplanar sampling blends projections along the dominant normal axes.
fn sample_triplanar(
    world_pos : vec3<f32>,
    normal : vec3<f32>,
    scale : f32,
    blend_sharpness : f32,
    layer : f32,
    lod_level : f32
) -> vec3<f32> {
    let abs_n = abs(normal);
    let sharpen = pow(abs_n + vec3<f32>(1e-4), vec3<f32>(blend_sharpness));
    let weight_sum = sharpen.x + sharpen.y + sharpen.z;
    let weights = sharpen / max(weight_sum, 1e-4);

    let uv_x = fract(world_pos.yz * scale);
    let uv_y = fract(world_pos.xz * scale);
    let uv_z = fract(world_pos.xy * scale);

    let layer_index = i32(layer);
    let color_x = textureSampleLevel(material_albedo_tex, material_samp, uv_x, layer_index, lod_level).rgb;
    let color_y = textureSampleLevel(material_albedo_tex, material_samp, uv_y, layer_index, lod_level).rgb;
    let color_z = textureSampleLevel(material_albedo_tex, material_samp, uv_z, layer_index, lod_level).rgb;

    return color_x * weights.x + color_y * weights.y + color_z * weights.z;
}

fn build_tbn(normal : vec3<f32>) -> mat3x3<f32> {
    let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), abs(normal.y) > 0.99);
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    return mat3x3<f32>(tangent, bitangent, normal);
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

fn tonemap_reinhard(color : vec3<f32>) -> vec3<f32> {
    return color / (vec3<f32>(1.0, 1.0, 1.0) + color);
}

fn tonemap_aces(color : vec3<f32>) -> vec3<f32> {
    let clipped = clamp(color, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(65504.0, 65504.0, 65504.0));
    let a = clipped * (clipped + vec3<f32>(0.0245786, 0.0245786, 0.0245786)) - vec3<f32>(0.000090537, 0.000090537, 0.000090537);
    let b = clipped * (vec3<f32>(0.983729, 0.983729, 0.983729) * clipped + vec3<f32>(0.4329510, 0.4329510, 0.4329510)) + vec3<f32>(0.238081, 0.238081, 0.238081);
    return clamp(a / b, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0));
}

@fragment
fn fs_main(input : VertexOutput) -> FragmentOutput {
    var out : FragmentOutput;

    let texel_size = calculate_texel_size();
    let uv = input.tex_coord;
    let height_normal = calculate_normal(uv, texel_size);
    let base_normal = normalize(input.world_normal);
    let normal_blend = clamp(u_shading.triplanar_params.z, 0.0, 1.0);
    let blended_normal = normalize(mix(base_normal, height_normal, normal_blend));

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
    if (pom_enabled) {
        pom_uv = parallax_occlusion_mapping(
            uv,
            view_dir_tangent,
            pom_scale,
            min_steps,
            max_steps,
            refine_steps
        );
    }
    let parallax_uv = clamp(pom_uv, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
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
    let slope_raw = 1.0 - abs(blended_normal.y);
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

    var weights = vec4<f32>(0.0);
    var weight_sum = 0.0;
    for (var idx = 0; idx < 4; idx = idx + 1) {
        if (idx < layer_count) {
            let center = u_shading.layer_heights[idx];
            let w = max(0.0, 1.0 - abs(height_norm - center) / blend_half);
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
            let sample_rgb = sample_triplanar(
                input.world_position,
                blended_normal,
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

    // ──────────────────────────────────────────────────────────────────────────
    // Colormap/Overlay System with Debug Modes
    // ──────────────────────────────────────────────────────────────────────────
    let debug_mode = u32(u_overlay.params1.y + 0.5);
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
    // 1 = colormap (overlay only)
    // 2 = mix (blend between material and overlay using colormap_strength)
    var final_albedo = albedo;
    if (albedo_mode == 0u) { // material
        final_albedo = material_albedo;
    } else if (albedo_mode == 1u) { // colormap
        final_albedo = overlay_rgb;
    } else if (albedo_mode == 2u) { // mix
        final_albedo = mix(material_albedo, albedo, colormap_strength);
    }

    albedo = clamp(final_albedo, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0));
    occlusion = clamp(occlusion, u_shading.clamp2.x, u_shading.clamp2.y);
    roughness = clamp(roughness, 0.04, 1.0);
    metallic = clamp(metallic, 0.0, 1.0);
    let f0 = mix(vec3<f32>(0.04, 0.04, 0.04), albedo, metallic);
    let light_dir = normalize(u_terrain.sun_exposure.xyz);
    
    // P2-05: Optional BRDF dispatch (flag-controlled)
    var lighting: vec3<f32>;
    if (TERRAIN_USE_BRDF_DISPATCH) {
        // Use unified BRDF dispatch (allows model switching)
        let shading_params = terrain_to_shading_params(roughness, metallic, TERRAIN_BRDF_MODEL);
        let n_dot_l = max(dot(blended_normal, light_dir), 0.0);
        lighting = eval_brdf(blended_normal, view_dir, light_dir, albedo, shading_params) * n_dot_l;
    } else {
        // Use original terrain-specific calculate_pbr_brdf (default, preserves current look)
        lighting = calculate_pbr_brdf(
            blended_normal,
            view_dir,
            light_dir,
            albedo,
            roughness,
            metallic,
            f0,
        );
    }
    lighting = lighting * u_terrain.sun_exposure.w;
    lighting = lighting * u_shading.light_params.rgb;
    
    // P3-10: Apply CSM shadow visibility (optional, gated by TERRAIN_USE_SHADOWS)
    if (TERRAIN_USE_SHADOWS) {
        // Calculate view-space depth for cascade selection
        let view_pos = u_terrain.view * vec4<f32>(input.world_position, 1.0);
        let view_depth = -view_pos.z; // Positive depth in view space
        
        // Sample shadow visibility (0.0 = full shadow, 1.0 = no shadow)
        let shadow_visibility = calculate_shadow_terrain(input.world_position, blended_normal, view_depth);
        
        // Apply shadow to direct lighting only
        lighting = lighting * shadow_visibility;
    } else {
        // Legacy POM-based shadow factor (preserved for backward compatibility)
        if (shadow_enabled && pom_enabled) {
            let shadow_factor = clamp(mix(0.4, 1.0, occlusion), u_shading.clamp1.z, u_shading.clamp1.w);
            lighting = lighting * shadow_factor;
        }
    }

    // Apply IBL rotation (terrain-specific feature)
    let rotated_normal = rotate_y(blended_normal, u_ibl.sin_theta, u_ibl.cos_theta);
    let rotated_view = rotate_y(view_dir, u_ibl.sin_theta, u_ibl.cos_theta);
    
    // Use unified eval_ibl function from lighting_ibl.wgsl (P4 spec)
    var ibl_contrib = eval_ibl(rotated_normal, rotated_view, albedo, metallic, roughness, f0);
    ibl_contrib = ibl_contrib * (u_ibl.intensity * occlusion);

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
    } else {
        // Normal PBR shading path
        let ambient_strength = mix(u_shading.clamp1.x, u_shading.clamp1.y, slope_factor);
        let ambient = albedo * ambient_strength;
        let direct = lighting * mix(0.65, 1.0, occlusion);
        let exposure = max(u_shading.light_params.w, 0.0);
        let shaded = (ambient + direct + ibl_contrib) * exposure;
        let tonemapped = tonemap_reinhard(shaded);
        final_color = tonemapped;
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

