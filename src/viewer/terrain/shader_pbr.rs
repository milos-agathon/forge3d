// src/viewer/terrain/shader_pbr.rs
// Enhanced PBR-like shader for terrain rendering with better lighting

pub const TERRAIN_PBR_SHADER: &str = r#"
// Terrain PBR Viewer Shader
// Enhanced lighting with Blinn-Phong, soft shadows, and improved materials

struct Uniforms {
    view_proj: mat4x4<f32>,
    sun_dir: vec4<f32>,
    terrain_params: vec4<f32>,  // min_h, h_range, terrain_width, z_scale
    lighting: vec4<f32>,        // sun_intensity, ambient, shadow_intensity, water_level
    background: vec4<f32>,      // r, g, b, _
    water_color: vec4<f32>,     // r, g, b, _
    // PBR params
    pbr_params: vec4<f32>,      // exposure, normal_strength, ibl_intensity, _
    camera_pos: vec4<f32>,      // camera world position
    // Lens effects: vignette_strength, vignette_radius, vignette_softness, _
    lens_params: vec4<f32>,
    // Screen dimensions for UV calculation
    screen_dims: vec4<f32>,     // width, height, _, _
    // Overlay params: enabled (>0.5), opacity, blend_mode (0=normal, 1=multiply, 2=overlay), solid (>0.5)
    overlay_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var heightmap: texture_2d<f32>;
@group(0) @binding(2) var height_sampler: sampler;
@group(0) @binding(3) var height_ao_tex: texture_2d<f32>;
@group(0) @binding(4) var sun_vis_tex: texture_2d<f32>;
// Overlay texture and sampler for lit draped overlays
@group(0) @binding(5) var overlay_tex: texture_2d<f32>;
@group(0) @binding(6) var overlay_sampler: sampler;

// P6.2: CSM shadow bindings
@group(0) @binding(7) var shadow_maps: texture_depth_2d_array;
@group(0) @binding(8) var shadow_sampler: sampler_comparison;
@group(0) @binding(9) var moment_maps: texture_2d_array<f32>;
@group(0) @binding(10) var moment_sampler: sampler;
@group(0) @binding(11) var<storage, read> csm_uniforms: CsmUniforms;

// Shadow cascade data (matches Rust CsmCascadeData: 144 bytes)
struct ShadowCascade {
    light_projection: mat4x4<f32>,  // 64 bytes
    light_view_proj: mat4x4<f32>,   // 64 bytes
    near_distance: f32,             // 4 bytes
    far_distance: f32,              // 4 bytes
    texel_size: f32,                // 4 bytes
    _padding: f32,                  // 4 bytes
}

// CSM uniforms (matches Rust CsmUniforms layout)
struct CsmUniforms {
    light_direction: vec4<f32>,
    light_view: mat4x4<f32>,
    cascades: array<ShadowCascade, 4>,
    cascade_count: u32,
    pcf_kernel_size: u32,
    depth_bias: f32,
    slope_bias: f32,
    shadow_map_size: f32,
    debug_mode: u32,
    evsm_positive_exp: f32,
    evsm_negative_exp: f32,
    peter_panning_offset: f32,
    enable_unclipped_depth: u32,
    depth_clip_factor: f32,
    technique: u32,
    technique_flags: u32,
    _pad1a: f32,
    _pad1b: f32,
    _pad1c: f32,
    technique_params: vec4<f32>,
    technique_reserved: vec4<f32>,
    cascade_blend_range: f32,
    _pad2a: f32,
    _pad2b: f32,
    _pad2c: f32,
    _pad2d: array<vec4<f32>, 6>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) raw_height: f32,
};

@vertex
fn vs_main(@location(0) pos: vec2<f32>, @location(1) uv: vec2<f32>) -> VertexOutput {
    let dims = textureDimensions(heightmap);
    let max_texel = vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1);
    let texel = clamp(
        vec2<i32>(i32(uv.x * f32(dims.x)), i32(uv.y * f32(dims.y))),
        vec2<i32>(0, 0),
        max_texel
    );
    let h = textureLoad(heightmap, texel, 0).r;
    
    let min_h = u.terrain_params.x;
    let terrain_width = u.terrain_params.z;
    let z_scale = u.terrain_params.w;
    
    let world_y = (h - min_h) * z_scale;
    
    let world_x = uv.x * terrain_width;
    let world_z = uv.y * terrain_width;
    
    var out: VertexOutput;
    out.world_pos = vec3<f32>(world_x, world_y, world_z);
    out.position = u.view_proj * vec4<f32>(out.world_pos, 1.0);
    out.uv = uv;
    out.raw_height = h;
    return out;
}

// Improved normal calculation from height gradient
fn compute_normal(world_pos: vec3<f32>) -> vec3<f32> {
    let dx = dpdx(world_pos);
    let dy = dpdy(world_pos);
    var n = normalize(cross(dy, dx));
    // Amplify normal detail based on pbr_params.y (normal_strength)
    let strength = u.pbr_params.y;
    n.x *= strength;
    n.z *= strength;
    return normalize(n);
}

// Blinn-Phong specular
fn blinn_phong_specular(normal: vec3<f32>, light_dir: vec3<f32>, view_dir: vec3<f32>, shininess: f32) -> f32 {
    let half_dir = normalize(light_dir + view_dir);
    let spec = pow(max(dot(normal, half_dir), 0.0), shininess);
    return spec;
}

// Soft shadow approximation based on normal and light direction
fn soft_shadow_factor(normal: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    let ndotl = dot(normal, light_dir);
    // Smooth transition from lit to shadow
    let shadow = smoothstep(-0.1, 0.3, ndotl);
    return shadow;
}

// Height-based material (PBR-like with Imhof palette and roughness variation)
fn get_material(h_norm: f32, slope: f32) -> vec4<f32> {
    // albedo.rgb, roughness
    var albedo: vec3<f32>;
    var roughness: f32;
    
    // Imhof palette: green valleys -> brown slopes -> white peaks
    if h_norm < 0.3 {
        // Green valleys
        albedo = mix(vec3<f32>(0.2, 0.5, 0.2), vec3<f32>(0.4, 0.6, 0.3), h_norm / 0.3);
        roughness = 0.8;
    } else if h_norm < 0.7 {
        // Brown slopes
        albedo = mix(vec3<f32>(0.4, 0.6, 0.3), vec3<f32>(0.5, 0.4, 0.3), (h_norm - 0.3) / 0.4);
        roughness = 0.7;
    } else {
        // White peaks
        albedo = mix(vec3<f32>(0.5, 0.4, 0.3), vec3<f32>(0.95, 0.95, 0.95), (h_norm - 0.7) / 0.3);
        roughness = 0.4;
    }
    
    // Add slope-based rock exposure (darkens steep areas)
    if slope > 0.5 && h_norm < 0.8 {
        let rock_blend = smoothstep(0.5, 0.75, slope);
        albedo = mix(albedo, vec3<f32>(0.35, 0.32, 0.28), rock_blend * 0.4);
        roughness = mix(roughness, 0.65, rock_blend);
    }
    
    return vec4<f32>(albedo, roughness);
}

// ACES tonemapping
fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

// Sky gradient for ambient
fn sky_ambient(normal: vec3<f32>) -> vec3<f32> {
    let sky_color = vec3<f32>(0.4, 0.6, 0.9);
    let ground_color = vec3<f32>(0.15, 0.1, 0.08);
    let sky_factor = normal.y * 0.5 + 0.5;
    return mix(ground_color, sky_color, sky_factor);
}

// ─────────────────────────────────────────────────────────────────────────────
// P6.2: CSM Shadow Sampling Functions
// ─────────────────────────────────────────────────────────────────────────────

// Select cascade based on view depth
fn select_cascade(view_depth: f32) -> u32 {
    let count = csm_uniforms.cascade_count;
    for (var i = 0u; i < count; i = i + 1u) {
        if (view_depth <= csm_uniforms.cascades[i].far_distance) {
            return i;
        }
    }
    return count - 1u;
}

// VSM: Chebyshev upper bound
fn sample_shadow_vsm(shadow_coords: vec2<f32>, receiver_depth: f32, cascade_idx: u32, moment_bias: f32) -> f32 {
    let moments = textureSample(moment_maps, moment_sampler, shadow_coords, i32(cascade_idx));
    let mean = moments.r;
    let mean_sq = moments.g;
    
    // Receiver is in front of mean - fully lit
    if (receiver_depth <= mean) {
        return 1.0;
    }
    
    // Variance with bias to prevent light bleeding
    let variance = max(mean_sq - mean * mean, moment_bias);
    let d = receiver_depth - mean;
    let p_max = variance / (variance + d * d);
    
    return clamp(p_max, 0.0, 1.0);
}

// EVSM: Exponential variance shadow maps
fn sample_shadow_evsm(shadow_coords: vec2<f32>, receiver_depth: f32, cascade_idx: u32, moment_bias: f32) -> f32 {
    let moments = textureSample(moment_maps, moment_sampler, shadow_coords, i32(cascade_idx));
    
    let c_pos = csm_uniforms.evsm_positive_exp;
    let c_neg = csm_uniforms.evsm_negative_exp;
    
    // Warp receiver depth
    let warp_depth_pos = exp(c_pos * receiver_depth);
    let warp_depth_neg = exp(-c_neg * receiver_depth);
    
    // Positive warp moments
    let pos_mean = moments.r;
    let pos_mean_sq = moments.g;
    let pos_variance = max(pos_mean_sq - pos_mean * pos_mean, moment_bias);
    let pos_d = warp_depth_pos - pos_mean;
    let p_max_pos = pos_variance / (pos_variance + pos_d * pos_d);
    
    // Negative warp moments
    let neg_mean = moments.b;
    let neg_mean_sq = moments.a;
    let neg_variance = max(neg_mean_sq - neg_mean * neg_mean, moment_bias);
    let neg_d = warp_depth_neg - neg_mean;
    let p_max_neg = neg_variance / (neg_variance + neg_d * neg_d);
    
    // Combine: take minimum (most shadow)
    var shadow = min(p_max_pos, p_max_neg);
    if (warp_depth_pos <= pos_mean) { shadow = max(shadow, 1.0); }
    if (warp_depth_neg <= neg_mean) { shadow = max(shadow, 1.0); }
    
    return clamp(shadow, 0.0, 1.0);
}

// MSM: Moment shadow maps (4 moments)
fn sample_shadow_msm(shadow_coords: vec2<f32>, receiver_depth: f32, cascade_idx: u32, moment_bias: f32) -> f32 {
    let moments = textureSample(moment_maps, moment_sampler, shadow_coords, i32(cascade_idx));
    
    let z = receiver_depth;
    let b1 = moments.r;
    let b2 = moments.g;
    let b3 = moments.b;
    let b4 = moments.a;
    
    // Hamburger 4MSM approximation
    let det = b2 * b4 - b3 * b3;
    if (abs(det) < 0.00001) {
        return select(1.0, 0.0, z > b1);
    }
    
    let c1 = (b4 * b1 - b3 * b2) / det;
    let c2 = (b2 * b2 - b1 * b3) / det;
    
    // Quadratic: z^2 + c1*z + c2 = 0
    let discriminant = c1 * c1 - 4.0 * c2;
    if (discriminant < 0.0) {
        return select(1.0, 0.0, z > b1);
    }
    
    let sqrt_d = sqrt(discriminant);
    let z1 = (-c1 - sqrt_d) * 0.5;
    let z2 = (-c1 + sqrt_d) * 0.5;
    
    if (z <= z1) { return 1.0; }
    if (z <= z2) { return 0.5; }
    return 0.0;
}

// Main CSM shadow sampling with technique dispatch
fn sample_csm_shadow(world_pos: vec3<f32>, normal: vec3<f32>, cascade_idx: u32) -> f32 {
    // Early out: no cascades means fully lit
    if (csm_uniforms.cascade_count == 0u) {
        return 1.0;
    }
    
    let cascade = csm_uniforms.cascades[cascade_idx];
    
    // Transform to light space
    let light_space_pos = cascade.light_view_proj * vec4<f32>(world_pos, 1.0);
    let ndc = light_space_pos.xyz / light_space_pos.w;
    
    // Convert to shadow map UV [0,1] - glam orthographic_rh uses [-1,1] for all axes
    let shadow_coords = ndc.xy * 0.5 + 0.5;
    let depth_01 = ndc.z * 0.5 + 0.5;  // Convert from [-1,1] to [0,1]
    
    // Out of bounds check
    if (shadow_coords.x < 0.0 || shadow_coords.x > 1.0 ||
        shadow_coords.y < 0.0 || shadow_coords.y > 1.0 ||
        depth_01 < 0.0 || depth_01 > 1.0) {
        return 1.0;
    }
    
    // Compute bias
    let light_dir_norm = normalize(csm_uniforms.light_direction.xyz);
    let n_dot_l = max(dot(normal, light_dir_norm), 0.0);
    let slope_factor = clamp(1.0 - n_dot_l, 0.0, 1.0);
    let bias = csm_uniforms.depth_bias + csm_uniforms.slope_bias * slope_factor + csm_uniforms.peter_panning_offset;
    let compare_depth = depth_01 - bias;
    
    let technique = csm_uniforms.technique;
    let moment_bias = csm_uniforms.technique_params.z;
    let light_size = csm_uniforms.technique_params.w;
    
    // HARD shadows (technique=0)
    if (technique == 0u) {
        return textureSampleCompare(shadow_maps, shadow_sampler, shadow_coords, i32(cascade_idx), compare_depth);
    }
    
    // PCF (technique=1): 3x3 kernel
    if (technique == 1u) {
        let texel_size = 1.0 / max(csm_uniforms.shadow_map_size, 1.0);
        var shadow_sum = 0.0;
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
                shadow_sum += textureSampleCompare(shadow_maps, shadow_sampler, shadow_coords + offset, i32(cascade_idx), compare_depth);
            }
        }
        return shadow_sum / 9.0;
    }
    
    // PCSS (technique=2): larger kernel with light size scaling
    if (technique == 2u) {
        let filter_scale = max(light_size, 1.0);
        let texel_size = (1.0 / max(csm_uniforms.shadow_map_size, 1.0)) * filter_scale;
        var shadow_sum = 0.0;
        for (var y = -2; y <= 2; y = y + 1) {
            for (var x = -2; x <= 2; x = x + 1) {
                let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
                shadow_sum += textureSampleCompare(shadow_maps, shadow_sampler, shadow_coords + offset, i32(cascade_idx), compare_depth);
            }
        }
        return shadow_sum / 25.0;
    }
    
    // VSM (technique=3)
    if (technique == 3u) {
        return sample_shadow_vsm(shadow_coords, compare_depth, cascade_idx, moment_bias);
    }
    
    // EVSM (technique=4)
    if (technique == 4u) {
        return sample_shadow_evsm(shadow_coords, compare_depth, cascade_idx, moment_bias);
    }
    
    // MSM (technique=5)
    if (technique == 5u) {
        return sample_shadow_msm(shadow_coords, compare_depth, cascade_idx, moment_bias);
    }
    
    // Fallback: PCF
    let texel_size = 1.0 / max(csm_uniforms.shadow_map_size, 1.0);
    var shadow_sum = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            shadow_sum += textureSampleCompare(shadow_maps, shadow_sampler, shadow_coords + offset, i32(cascade_idx), compare_depth);
        }
    }
    return shadow_sum / 9.0;
}

// Calculate CSM shadow visibility
fn calculate_csm_shadow(world_pos: vec3<f32>, normal: vec3<f32>, view_depth: f32) -> f32 {
    if (csm_uniforms.cascade_count == 0u) {
        return 1.0;
    }
    
    let cascade_idx = select_cascade(view_depth);
    return sample_csm_shadow(world_pos, normal, cascade_idx);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sun_intensity = u.lighting.x;
    let ambient_strength = u.lighting.y;
    let shadow_strength = u.lighting.z;
    let water_level = u.lighting.w;
    let exposure = u.pbr_params.x;
    let ibl_intensity = u.pbr_params.z;
    
    // P6.2: Debug mode 33 - Shadow technique visualization
    if csm_uniforms.debug_mode == 33u {
        let tech = csm_uniforms.technique;
        var tech_color = vec3<f32>(1.0, 0.0, 1.0); // Magenta = unknown
        if tech == 0u {
            tech_color = vec3<f32>(1.0, 0.0, 0.0); // Red = HARD
        } else if tech == 1u {
            tech_color = vec3<f32>(0.0, 1.0, 0.0); // Green = PCF
        } else if tech == 2u {
            tech_color = vec3<f32>(0.0, 0.0, 1.0); // Blue = PCSS
        } else if tech == 3u {
            tech_color = vec3<f32>(0.0, 1.0, 1.0); // Cyan = VSM
        } else if tech == 4u {
            tech_color = vec3<f32>(1.0, 1.0, 0.0); // Yellow = EVSM
        } else if tech == 5u {
            tech_color = vec3<f32>(1.0, 0.0, 1.0); // Magenta = MSM
        }
        return vec4<f32>(tech_color, 1.0);
    }
    
    // Debug mode 34 - Show cascade_count status (Green=has cascades, Red=no cascades)
    if csm_uniforms.debug_mode == 34u {
        if csm_uniforms.cascade_count > 0u {
            return vec4<f32>(0.0, 1.0, 0.0, 1.0); // Green = cascades available
        } else {
            return vec4<f32>(1.0, 0.0, 0.0, 1.0); // Red = no cascades
        }
    }
    
    // Debug mode 35 - Visualize raw shadow value (grayscale: black=shadow, white=lit)
    if csm_uniforms.debug_mode == 35u {
        let normal = compute_normal(in.world_pos);
        let view_depth = max(length(u.camera_pos.xyz - in.world_pos), 0.1);
        let shadow_val = calculate_csm_shadow(in.world_pos, normal, view_depth);
        return vec4<f32>(shadow_val, shadow_val, shadow_val, 1.0);
    }
    
    // Debug mode 36 - Check if shadow passes ran (technique_reserved[0] == 1.0)
    if csm_uniforms.debug_mode == 36u {
        if csm_uniforms.technique_reserved.x > 0.5 {
            return vec4<f32>(0.0, 1.0, 0.0, 1.0); // Green = shadow passes ran
        } else {
            return vec4<f32>(1.0, 0.0, 0.0, 1.0); // Red = shadow passes did NOT run
        }
    }
    
    // Debug mode 37 - Visualize light-space coordinates
    // R=shadow_uv.x, G=shadow_uv.y, B=receiver_depth
    if csm_uniforms.debug_mode == 37u {
        let view_depth = max(length(u.camera_pos.xyz - in.world_pos), 0.1);
        let cascade_idx = select_cascade(view_depth);
        let cascade = csm_uniforms.cascades[cascade_idx];
        
        // Transform to light space
        let light_space_pos = cascade.light_view_proj * vec4<f32>(in.world_pos, 1.0);
        let ndc = light_space_pos.xyz / light_space_pos.w;
        let shadow_uv = ndc.xy * 0.5 + 0.5;
        let receiver_depth = ndc.z * 0.5 + 0.5;
        
        // Visualize light-space coords: R=u, G=v, B=depth
        return vec4<f32>(shadow_uv.x, shadow_uv.y, receiver_depth, 1.0);
    }
    
    // Debug mode 38 - Sample shadow map and show result with depth info
    // R=shadow_result (0=shadow, 1=lit), G=receiver_depth, B=cascade_idx/4
    if csm_uniforms.debug_mode == 38u {
        let view_depth = max(length(u.camera_pos.xyz - in.world_pos), 0.1);
        let cascade_idx = select_cascade(view_depth);
        let cascade = csm_uniforms.cascades[cascade_idx];
        
        let light_space_pos = cascade.light_view_proj * vec4<f32>(in.world_pos, 1.0);
        let ndc = light_space_pos.xyz / light_space_pos.w;
        let shadow_uv = ndc.xy * 0.5 + 0.5;
        let receiver_depth = ndc.z * 0.5 + 0.5;
        
        // Sample shadow map using comparison sampler (returns 0 or 1)
        let shadow_result = textureSampleCompare(shadow_maps, shadow_sampler, shadow_uv, i32(cascade_idx), receiver_depth);
        
        return vec4<f32>(shadow_result, receiver_depth, f32(cascade_idx) / 4.0, 1.0);
    }
    
    // Check if below water level
    let is_water = in.raw_height < water_level;
    
    // Normalized height for material selection
    let h_norm = clamp((in.raw_height - u.terrain_params.x) / max(u.terrain_params.y, 1.0), 0.0, 1.0);
    
    // Compute surface normal
    let normal = compute_normal(in.world_pos);
    
    // Slope for material blending (0 = flat, 1 = vertical)
    let slope = 1.0 - abs(normal.y);
    
    // Get material properties
    let material = get_material(h_norm, slope);
    var albedo = material.rgb;
    let roughness = material.a;
    
    // Water handling
    if is_water {
        albedo = u.water_color.rgb;
    }
    
    // === OVERLAY BLENDING (before lighting, so overlays are lit) ===
    let overlay_enabled = u.overlay_params.x > 0.5;
    let overlay_opacity = u.overlay_params.y;
    let overlay_blend_mode = u.overlay_params.z;
    let solid_surface = u.overlay_params.w > 0.5;
    
    // Sample overlay texture for solid check and blending
    let overlay = textureSample(overlay_tex, overlay_sampler, in.uv);
    
    // When solid=false and overlay is enabled, discard fragments where overlay alpha is near 0
    // This hides the base surface outside the region of interest (like rayshader solid=FALSE)
    // Use low threshold (0.01) - the overlay image should have alpha=1.0 for valid areas
    // and alpha=0.0 for areas to hide. Only truly transparent pixels should be discarded.
    if overlay_enabled && !solid_surface && overlay.a < 0.01 {
        discard;
    }
    
    if overlay_enabled && overlay_opacity > 0.001 {
        let blend_alpha = overlay.a * overlay_opacity;
        
        if blend_alpha > 0.01 {
            // Apply blend mode (overlay already sampled above for solid check)
            if overlay_blend_mode < 0.5 {
                // Normal blend: fully replace albedo with overlay when alpha is high
                // For land cover overlays, we want the categorical colors to show through
                albedo = mix(albedo, overlay.rgb, blend_alpha);
            } else if overlay_blend_mode < 1.5 {
                // Multiply blend
                let multiplied = albedo * overlay.rgb;
                albedo = mix(albedo, multiplied, blend_alpha);
            } else {
                // Overlay blend (Photoshop-style)
                let base = albedo;
                var blended: vec3<f32>;
                // Overlay formula: 2*base*blend if base < 0.5, else 1 - 2*(1-base)*(1-blend)
                blended.r = select(1.0 - 2.0 * (1.0 - base.r) * (1.0 - overlay.r), 2.0 * base.r * overlay.r, base.r < 0.5);
                blended.g = select(1.0 - 2.0 * (1.0 - base.g) * (1.0 - overlay.g), 2.0 * base.g * overlay.g, base.g < 0.5);
                blended.b = select(1.0 - 2.0 * (1.0 - base.b) * (1.0 - overlay.b), 2.0 * base.b * overlay.b, base.b < 0.5);
                albedo = mix(albedo, blended, blend_alpha);
            }
        }
    }
    // === END OVERLAY BLENDING ===
    
    // Sun lighting
    let sun_dir = normalize(u.sun_dir.xyz);
    let ndotl = max(dot(normal, sun_dir), 0.0);
    
    // Sample heightfield AO (use textureLoad for R32Float)
    let ao_dims = textureDimensions(height_ao_tex, 0);
    let ao_pixel = vec2<i32>(in.uv * vec2<f32>(ao_dims));
    let ao_clamped = clamp(ao_pixel, vec2<i32>(0), vec2<i32>(ao_dims) - vec2<i32>(1));
    let height_ao = textureLoad(height_ao_tex, ao_clamped, 0).r;
    
    // Sample sun visibility (use textureLoad for R32Float)
    let sv_dims = textureDimensions(sun_vis_tex, 0);
    let sv_pixel = vec2<i32>(in.uv * vec2<f32>(sv_dims));
    let sv_clamped = clamp(sv_pixel, vec2<i32>(0), vec2<i32>(sv_dims) - vec2<i32>(1));
    let sun_vis = textureLoad(sun_vis_tex, sv_clamped, 0).r;
    
    // P6.2: CSM shadow sampling with technique dispatch
    // Calculate view depth for cascade selection (approximate from world position)
    let view_pos = u.view_proj * vec4<f32>(in.world_pos, 1.0);
    let view_depth = max(length(u.camera_pos.xyz - in.world_pos), 0.1);
    
    // Sample CSM shadow using technique-specific sampling (HARD, PCF, PCSS, VSM, EVSM, MSM)
    let csm_shadow = calculate_csm_shadow(in.world_pos, normal, view_depth);
    
    // Fallback soft shadow for when CSM has no cascades
    let soft_shadow = soft_shadow_factor(normal, sun_dir);
    
    // Use CSM shadow if cascades available, otherwise fallback to soft shadow
    let shadow = select(soft_shadow, csm_shadow, csm_uniforms.cascade_count > 0u);
    
    // Combine shadow with sun visibility (multiplicative)
    let shadow_term = mix(1.0, shadow * sun_vis, shadow_strength);
    
    // View direction (approximate from camera_pos or use fragment position)
    let view_dir = normalize(u.camera_pos.xyz - in.world_pos);
    
    // Diffuse lighting
    let sun_color = vec3<f32>(1.0, 0.95, 0.85); // Warm sunlight
    let diffuse = albedo * sun_color * ndotl * sun_intensity * shadow_term;
    
    // Specular (Blinn-Phong)
    let shininess = mix(8.0, 64.0, 1.0 - roughness);
    var specular = blinn_phong_specular(normal, sun_dir, view_dir, shininess);
    specular *= sun_intensity * shadow_term * (1.0 - roughness) * 0.3;
    
    // Water specular boost
    if is_water {
        specular = blinn_phong_specular(vec3<f32>(0.0, 1.0, 0.0), sun_dir, view_dir, 128.0);
        specular *= sun_intensity * 0.8;
    }
    
    // Ambient/IBL approximation - modulated by heightfield AO
    let ambient_color = sky_ambient(normal) * albedo * ambient_strength * ibl_intensity * height_ao;
    
    // Combine lighting
    var color = diffuse + vec3<f32>(specular) + ambient_color;
    
    // Apply exposure and tonemapping
    color = color * exposure;
    color = aces_tonemap(color);
    
    // Gamma correction (linear to sRGB)
    color = pow(color, vec3<f32>(1.0 / 2.2));
    
    // Apply vignette (lens effect)
    let vignette_strength = u.lens_params.x;
    let vignette_radius = u.lens_params.y;
    let vignette_softness = u.lens_params.z;
    if vignette_strength > 0.001 {
        // Calculate screen UV from fragment position
        let screen_uv = in.position.xy / u.screen_dims.xy;
        // Distance from center (0.5, 0.5)
        let center_dist = length(screen_uv - vec2<f32>(0.5));
        // Vignette falloff
        let vignette = 1.0 - smoothstep(vignette_radius - vignette_softness, vignette_radius + vignette_softness, center_dist * 2.0);
        // Apply vignette darkening
        color = color * mix(1.0, vignette, vignette_strength);
    }
    
    return vec4<f32>(color, 1.0);
}
"#;
