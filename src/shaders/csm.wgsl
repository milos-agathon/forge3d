// Cascaded Shadow Maps - Main Implementation
// Complete CSM pipeline with 3-4 cascades, PCF/EVSM kernels, and peter-panning prevention
// RELEVANT FILES: src/pipeline/pbr.rs, python/forge3d/lighting.py, tests/test_b4_csm.py

// Shadow cascade data structure
struct ShadowCascade {
    light_projection: mat4x4<f32>,  // Light-space projection matrix for this cascade
    near_distance: f32,             // Near plane distance in view space
    far_distance: f32,              // Far plane distance in view space
    texel_size: f32,                // Texel size in world space for this cascade
    _padding: f32,
}

// CSM configuration and state
struct CsmUniforms {
    light_direction: vec4<f32>,     // Directional light direction (world space)
    light_view: mat4x4<f32>,        // Light view matrix
    cascades: array<ShadowCascade, 4>, // Up to 4 shadow cascades
    cascade_count: u32,             // Number of active cascades (3-4)
    pcf_kernel_size: u32,           // PCF kernel size: 1=none, 3=3x3, 5=5x5, 7=poisson
    depth_bias: f32,                // Base depth bias to prevent shadow acne
    slope_bias: f32,                // Slope-scaled bias factor
    shadow_map_size: f32,           // Shadow map resolution (e.g., 2048.0)
    debug_mode: u32,                // Debug visualization: 0=off, 1=cascades, 2=overdraw
    evsm_positive_exp: f32,         // EVSM positive exponent
    evsm_negative_exp: f32,         // EVSM negative exponent
    peter_panning_offset: f32,      // Offset to prevent peter-panning artifacts
}

// Bind groups for CSM resources
@group(2) @binding(0) var<uniform> csm_uniforms: CsmUniforms;
@group(2) @binding(1) var shadow_maps: texture_depth_2d_array;  // Shadow depth maps
@group(2) @binding(2) var shadow_sampler: sampler_comparison;    // PCF comparison sampler
@group(2) @binding(3) var evsm_maps: texture_2d_array<f32>;     // EVSM moment maps (optional)

// Constants for shadow filtering
const PI: f32 = 3.14159265359;
const SHADOW_EPSILON: f32 = 0.00001;
const MAX_SHADOW_DISTANCE: f32 = 1000.0;

// Transform world position to light space for specific cascade
fn world_to_light_space(world_pos: vec3<f32>, cascade_idx: u32) -> vec4<f32> {
    let light_space_pos = csm_uniforms.cascades[cascade_idx].light_projection *
                         csm_uniforms.light_view *
                         vec4<f32>(world_pos, 1.0);
    return light_space_pos;
}

// Select appropriate cascade based on view space depth
fn select_cascade(view_depth: f32) -> u32 {
    var cascade_idx = csm_uniforms.cascade_count - 1u;

    // Find the first cascade that contains this depth
    for (var i = 0u; i < csm_uniforms.cascade_count; i++) {
        if (view_depth <= csm_uniforms.cascades[i].far_distance) {
            cascade_idx = i;
            break;
        }
    }

    return min(cascade_idx, csm_uniforms.cascade_count - 1u);
}

// Calculate slope-scaled depth bias to prevent shadow acne
fn calculate_depth_bias(world_normal: vec3<f32>, cascade_idx: u32) -> f32 {
    let light_dir = normalize(-csm_uniforms.light_direction.xyz);
    let n_dot_l = max(dot(world_normal, light_dir), 0.001);

    // Slope-scaled bias: bias increases as surface becomes more parallel to light
    let slope_scale = sqrt(1.0 - n_dot_l * n_dot_l) / n_dot_l;
    let texel_size = csm_uniforms.cascades[cascade_idx].texel_size;

    return csm_uniforms.depth_bias + csm_uniforms.slope_bias * slope_scale * texel_size;
}

// Basic shadow test (single sample, no filtering)
fn sample_shadow_basic(light_space_pos: vec4<f32>, cascade_idx: u32, bias: f32) -> f32 {
    // Perspective divide and convert to texture coordinates [0,1]
    let proj_coords = light_space_pos.xyz / light_space_pos.w;
    let shadow_coords = proj_coords * 0.5 + 0.5;

    // Check bounds - return unshadowed if outside
    if (shadow_coords.x < 0.0 || shadow_coords.x > 1.0 ||
        shadow_coords.y < 0.0 || shadow_coords.y > 1.0 ||
        shadow_coords.z < 0.0 || shadow_coords.z > 1.0) {
        return 1.0;
    }

    // Apply bias and peter-panning offset
    let test_depth = shadow_coords.z - bias - csm_uniforms.peter_panning_offset;

    // Single depth comparison
    return textureSampleCompare(shadow_maps, shadow_sampler,
                               shadow_coords.xy, cascade_idx, test_depth);
}

// PCF (Percentage-Closer Filtering) implementation
fn sample_shadow_pcf(light_space_pos: vec4<f32>, cascade_idx: u32, bias: f32) -> f32 {
    let proj_coords = light_space_pos.xyz / light_space_pos.w;
    let shadow_coords = proj_coords * 0.5 + 0.5;

    // Bounds check
    if (shadow_coords.x < 0.0 || shadow_coords.x > 1.0 ||
        shadow_coords.y < 0.0 || shadow_coords.y > 1.0 ||
        shadow_coords.z < 0.0 || shadow_coords.z > 1.0) {
        return 1.0;
    }

    let test_depth = shadow_coords.z - bias - csm_uniforms.peter_panning_offset;

    // PCF kernel configuration
    let kernel_size = i32(csm_uniforms.pcf_kernel_size);
    let half_kernel = kernel_size / 2;
    let texel_size = 1.0 / csm_uniforms.shadow_map_size;

    // Accumulate shadow samples from PCF kernel
    var shadow_factor = 0.0;
    var sample_count = 0.0;

    for (var x = -half_kernel; x <= half_kernel; x++) {
        for (var y = -half_kernel; y <= half_kernel; y++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            let sample_coords = shadow_coords.xy + offset;

            // Individual sample bounds check
            if (sample_coords.x >= 0.0 && sample_coords.x <= 1.0 &&
                sample_coords.y >= 0.0 && sample_coords.y <= 1.0) {

                shadow_factor += textureSampleCompare(shadow_maps, shadow_sampler,
                                                    sample_coords, cascade_idx, test_depth);
                sample_count += 1.0;
            }
        }
    }

    // Normalize by actual sample count (handles edge cases)
    return select(0.0, shadow_factor / sample_count, sample_count > 0.0);
}

// Optimized Poisson disk PCF for higher quality
fn sample_shadow_poisson(light_space_pos: vec4<f32>, cascade_idx: u32, bias: f32) -> f32 {
    let proj_coords = light_space_pos.xyz / light_space_pos.w;
    let shadow_coords = proj_coords * 0.5 + 0.5;

    if (shadow_coords.x < 0.0 || shadow_coords.x > 1.0 ||
        shadow_coords.y < 0.0 || shadow_coords.y > 1.0 ||
        shadow_coords.z < 0.0 || shadow_coords.z > 1.0) {
        return 1.0;
    }

    let test_depth = shadow_coords.z - bias - csm_uniforms.peter_panning_offset;

    // Optimized 16-sample Poisson disk pattern
    let poisson_samples = array<vec2<f32>, 16>(
        vec2<f32>(-0.94201624, -0.39906216), vec2<f32>(0.94558609, -0.76890725),
        vec2<f32>(-0.094184101, -0.92938870), vec2<f32>(0.34495938, 0.29387760),
        vec2<f32>(-0.91588581, 0.45771432), vec2<f32>(-0.81544232, -0.87912464),
        vec2<f32>(-0.38277543, 0.27676845), vec2<f32>(0.97484398, 0.75648379),
        vec2<f32>(0.44323325, -0.97511554), vec2<f32>(0.53742981, -0.47373420),
        vec2<f32>(-0.26496911, -0.41893023), vec2<f32>(0.79197514, 0.19090188),
        vec2<f32>(-0.24188840, 0.99706507), vec2<f32>(-0.81409955, 0.91437590),
        vec2<f32>(0.19984126, 0.78641367), vec2<f32>(0.14383161, -0.14100790)
    );

    let filter_radius = f32(csm_uniforms.pcf_kernel_size) / csm_uniforms.shadow_map_size;
    var shadow_factor = 0.0;

    for (var i = 0; i < 16; i++) {
        let offset = poisson_samples[i] * filter_radius;
        let sample_coords = shadow_coords.xy + offset;

        if (sample_coords.x >= 0.0 && sample_coords.x <= 1.0 &&
            sample_coords.y >= 0.0 && sample_coords.y <= 1.0) {
            shadow_factor += textureSampleCompare(shadow_maps, shadow_sampler,
                                                sample_coords, cascade_idx, test_depth);
        } else {
            shadow_factor += 1.0; // Outside bounds treated as lit
        }
    }

    return shadow_factor / 16.0;
}

// EVSM (Exponential Variance Shadow Maps) implementation
fn sample_shadow_evsm(light_space_pos: vec4<f32>, cascade_idx: u32) -> f32 {
    let proj_coords = light_space_pos.xyz / light_space_pos.w;
    let shadow_coords = proj_coords * 0.5 + 0.5;

    if (shadow_coords.x < 0.0 || shadow_coords.x > 1.0 ||
        shadow_coords.y < 0.0 || shadow_coords.y > 1.0 ||
        shadow_coords.z < 0.0 || shadow_coords.z > 1.0) {
        return 1.0;
    }

    // Sample EVSM moments (M1, M2 in RG channels for positive; BA for negative)
    let moments = textureSample(evsm_maps, shadow_sampler, shadow_coords.xy, cascade_idx);

    let fragment_depth = shadow_coords.z;

    // Positive EVSM
    let pos_exp = csm_uniforms.evsm_positive_exp;
    let pos_depth = exp(pos_exp * fragment_depth);
    let pos_m1 = moments.r;
    let pos_m2 = moments.g;

    // Chebyshev's inequality for variance shadow mapping
    var pos_shadow = 1.0;
    if (fragment_depth > pos_m1) {
        let variance = pos_m2 - pos_m1 * pos_m1;
        let d = fragment_depth - pos_m1;
        pos_shadow = variance / (variance + d * d);
    }

    // Negative EVSM for light bleeding reduction
    let neg_exp = csm_uniforms.evsm_negative_exp;
    let neg_depth = exp(-neg_exp * fragment_depth);
    let neg_m1 = moments.b;
    let neg_m2 = moments.a;

    var neg_shadow = 1.0;
    if (fragment_depth < neg_m1) {
        let variance = neg_m2 - neg_m1 * neg_m1;
        let d = neg_m1 - fragment_depth;
        neg_shadow = variance / (variance + d * d);
    }

    // Combine positive and negative results
    return min(pos_shadow, neg_shadow);
}

// Main shadow calculation entry point
fn calculate_shadow(world_pos: vec3<f32>, view_depth: f32, world_normal: vec3<f32>) -> f32 {
    // Early exit if no shadow cascades
    if (csm_uniforms.cascade_count == 0u) {
        return 1.0;
    }

    // Early exit for extremely distant geometry
    if (view_depth > MAX_SHADOW_DISTANCE) {
        return 1.0;
    }

    // Select appropriate cascade based on view depth
    let cascade_idx = select_cascade(view_depth);

    // Transform to light space for selected cascade
    let light_space_pos = world_to_light_space(world_pos, cascade_idx);

    // Calculate dynamic depth bias
    let bias = calculate_depth_bias(world_normal, cascade_idx);

    // Apply shadow filtering based on configuration
    var shadow_factor: f32;

    if (csm_uniforms.pcf_kernel_size <= 1u) {
        // No filtering - single sample
        shadow_factor = sample_shadow_basic(light_space_pos, cascade_idx, bias);
    } else if (csm_uniforms.pcf_kernel_size <= 5u) {
        // Standard PCF filtering
        shadow_factor = sample_shadow_pcf(light_space_pos, cascade_idx, bias);
    } else {
        // High-quality Poisson disk PCF
        shadow_factor = sample_shadow_poisson(light_space_pos, cascade_idx, bias);
    }

    return clamp(shadow_factor, 0.0, 1.0);
}

// Calculate EVSM shadow (alternative high-quality path)
fn calculate_shadow_evsm(world_pos: vec3<f32>, view_depth: f32) -> f32 {
    if (csm_uniforms.cascade_count == 0u) {
        return 1.0;
    }

    let cascade_idx = select_cascade(view_depth);
    let light_space_pos = world_to_light_space(world_pos, cascade_idx);

    return sample_shadow_evsm(light_space_pos, cascade_idx);
}

// Debug visualization utilities
fn get_cascade_debug_color(cascade_idx: u32) -> vec3<f32> {
    switch (cascade_idx) {
        case 0u: { return vec3<f32>(1.0, 0.2, 0.2); } // Red - closest cascade
        case 1u: { return vec3<f32>(0.2, 1.0, 0.2); } // Green - medium cascade
        case 2u: { return vec3<f32>(0.2, 0.2, 1.0); } // Blue - far cascade
        case 3u: { return vec3<f32>(1.0, 1.0, 0.2); } // Yellow - farthest cascade
        default: { return vec3<f32>(1.0, 0.2, 1.0); } // Magenta - error case
    }
}

// Apply cascade debug visualization overlay
fn apply_cascade_debug(base_color: vec3<f32>, world_pos: vec3<f32>, view_depth: f32) -> vec3<f32> {
    if (csm_uniforms.debug_mode == 0u) {
        return base_color;
    }

    let cascade_idx = select_cascade(view_depth);
    let debug_color = get_cascade_debug_color(cascade_idx);

    // Blend base color with cascade debug color
    let debug_intensity = select(0.0, 0.4, csm_uniforms.debug_mode == 1u);
    return mix(base_color, debug_color, debug_intensity);
}

// Shadow overdraw debug visualization
fn apply_overdraw_debug(base_color: vec3<f32>, shadow_factor: f32) -> vec3<f32> {
    if (csm_uniforms.debug_mode != 2u) {
        return base_color;
    }

    // Visualize shadow intensity: red = fully shadowed, green = fully lit
    let debug_color = mix(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), shadow_factor);
    return mix(base_color, debug_color, 0.5);
}

// Cascade transition smoothing (reduces cascade boundary artifacts)
fn smooth_cascade_transition(world_pos: vec3<f32>, view_depth: f32, world_normal: vec3<f32>) -> f32 {
    let current_cascade = select_cascade(view_depth);

    // Check if we're near a cascade boundary
    if (current_cascade > 0u && current_cascade < csm_uniforms.cascade_count - 1u) {
        let current_far = csm_uniforms.cascades[current_cascade].far_distance;
        let next_far = csm_uniforms.cascades[current_cascade + 1u].far_distance;

        // Transition zone is 10% of the cascade range
        let transition_zone = (next_far - current_far) * 0.1;
        let boundary_distance = abs(view_depth - current_far);

        if (boundary_distance < transition_zone) {
            // Sample both cascades and blend
            let current_shadow = calculate_shadow(world_pos, view_depth, world_normal);

            // Sample next cascade
            let next_light_space = world_to_light_space(world_pos, current_cascade + 1u);
            let next_bias = calculate_depth_bias(world_normal, current_cascade + 1u);
            let next_shadow = sample_shadow_basic(next_light_space, current_cascade + 1u, next_bias);

            // Blend based on distance to boundary
            let blend_factor = boundary_distance / transition_zone;
            return mix(next_shadow, current_shadow, blend_factor);
        }
    }

    // No transition needed
    return calculate_shadow(world_pos, view_depth, world_normal);
}