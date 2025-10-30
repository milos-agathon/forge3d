// shaders/shadows.wgsl
// Unified cascaded shadow sampling supporting hard, PCF, PCSS, and moment-based techniques
// Exists to share a single evaluation path for all shadow modes consumed by Forge3D pipelines
// RELEVANT FILES: src/shadows/csm.rs, src/shadows/manager.rs, src/pipeline/pbr.rs, python/forge3d/lighting.py

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
    technique_params: vec4<f32>,
    technique_reserved: vec4<f32>,
}

@group(3) @binding(0) var<uniform> csm_uniforms: CsmUniforms;
@group(3) @binding(1) var shadow_depth_array: texture_depth_2d_array;
@group(3) @binding(2) var shadow_compare_sampler: sampler_comparison;
@group(3) @binding(3) var shadow_moment_array: texture_2d_array<f32>;
@group(3) @binding(4) var shadow_moment_sampler: sampler;

const PI: f32 = 3.141592653589793;
const MOMENT_BIAS_EPS: f32 = 1e-5;

fn select_cascade(view_depth: f32) -> u32 {
    var cascade_idx = csm_uniforms.cascade_count - 1u;

    for (var i = 0u; i < csm_uniforms.cascade_count; i = i + 1u) {
        if (view_depth <= csm_uniforms.cascades[i].far_distance) {
            cascade_idx = i;
            break;
        }
    }

    return cascade_idx;
}

fn world_to_light_space(world_pos: vec3<f32>, cascade_idx: u32) -> vec4<f32> {
    return csm_uniforms.cascades[cascade_idx].light_projection *
           csm_uniforms.light_view *
           vec4<f32>(world_pos, 1.0);
}

fn project_shadow_coords(light_clip: vec4<f32>) -> vec3<f32> {
    let proj = light_clip.xyz / light_clip.w;
    return proj * 0.5 + vec3<f32>(0.5, 0.5, 0.5);
}

fn calculate_depth_bias(world_normal: vec3<f32>, cascade_idx: u32) -> f32 {
    let light_dir = normalize(-csm_uniforms.light_direction.xyz);
    let n_dot_l = max(dot(world_normal, light_dir), 0.001);
    let slope_scale = sqrt(1.0 - n_dot_l * n_dot_l) / n_dot_l;
    let texel_size = csm_uniforms.cascades[cascade_idx].texel_size;
    return csm_uniforms.depth_bias +
           csm_uniforms.slope_bias * slope_scale * texel_size +
           csm_uniforms.peter_panning_offset;
}

fn texture_in_bounds(coords: vec3<f32>) -> bool {
    return coords.x >= 0.0 && coords.x <= 1.0 &&
           coords.y >= 0.0 && coords.y <= 1.0 &&
           coords.z >= 0.0 && coords.z <= 1.0;
}

fn sample_shadow_hard(coords: vec3<f32>, cascade_idx: u32, bias: f32) -> f32 {
    return textureSampleCompare(
        shadow_depth_array,
        shadow_compare_sampler,
        coords.xy,
        cascade_idx,
        coords.z - bias
    );
}

fn pcf_kernel_extent() -> i32 {
    let kernel = i32(csm_uniforms.pcf_kernel_size);
    return max(kernel, 1) / 2;
}

fn sample_shadow_pcf(coords: vec3<f32>, cascade_idx: u32, bias: f32, kernel_scale: f32) -> f32 {
    let kernel_half = pcf_kernel_extent();
    if kernel_half == 0 {
        return sample_shadow_hard(coords, cascade_idx, bias);
    }

    let texel = kernel_scale / csm_uniforms.shadow_map_size;
    var occlusion = 0.0;
    var taps = 0.0;

    for (var x = -kernel_half; x <= kernel_half; x = x + 1) {
        for (var y = -kernel_half; y <= kernel_half; y = y + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            let sample_coords = coords.xy + offset;
            if (sample_coords.x < 0.0 || sample_coords.x > 1.0 ||
                sample_coords.y < 0.0 || sample_coords.y > 1.0) {
                occlusion = occlusion + 1.0;
            } else {
                occlusion = occlusion + textureSampleCompare(
                    shadow_depth_array,
                    shadow_compare_sampler,
                    sample_coords,
                    cascade_idx,
                    coords.z - bias
                );
            }
            taps = taps + 1.0;
        }
    }

    return occlusion / max(taps, 1.0);
}

fn poisson_disk_sample(coords: vec3<f32>, cascade_idx: u32, bias: f32, radius: f32) -> f32 {
    let disk = array<vec2<f32>, 16>(
        vec2<f32>(-0.94201624, -0.39906216),
        vec2<f32>(0.94558609, -0.76890725),
        vec2<f32>(-0.094184101, -0.92938870),
        vec2<f32>(0.34495938, 0.29387760),
        vec2<f32>(-0.91588581, 0.45771432),
        vec2<f32>(-0.81544232, -0.87912464),
        vec2<f32>(-0.38277543, 0.27676845),
        vec2<f32>(0.97484398, 0.75648379),
        vec2<f32>(0.44323325, -0.97511554),
        vec2<f32>(0.53742981, -0.47373420),
        vec2<f32>(-0.26496911, -0.41893023),
        vec2<f32>(0.79197514, 0.19090188),
        vec2<f32>(-0.24188840, 0.99706507),
        vec2<f32>(-0.81409955, 0.91437590),
        vec2<f32>(0.19984126, 0.78641367),
        vec2<f32>(0.14383161, -0.14100790)
    );

    let texel = radius / csm_uniforms.shadow_map_size;
    var occlusion = 0.0;

    for (var i = 0u; i < 16u; i = i + 1u) {
        let sample_coords = coords.xy + disk[i] * texel;
        if (sample_coords.x < 0.0 || sample_coords.x > 1.0 ||
            sample_coords.y < 0.0 || sample_coords.y > 1.0) {
            occlusion = occlusion + 1.0;
        } else {
            occlusion = occlusion + textureSampleCompare(
                shadow_depth_array,
                shadow_compare_sampler,
                sample_coords,
                cascade_idx,
                coords.z - bias
            );
        }
    }

    return occlusion / 16.0;
}

fn blocker_search(coords: vec3<f32>, cascade_idx: u32, bias: f32, search_radius: f32) -> vec2<f32> {
    let kernel_half = 3;
    let texel = search_radius / csm_uniforms.shadow_map_size;
    var blockers = 0.0;
    var count = 0.0;

    for (var x = -kernel_half; x <= kernel_half; x = x + 1) {
        for (var y = -kernel_half; y <= kernel_half; y = y + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            let sample_coords = coords.xy + offset;
            if (sample_coords.x < 0.0 || sample_coords.x > 1.0 ||
                sample_coords.y < 0.0 || sample_coords.y > 1.0) {
                continue;
            }

            let depth = textureLoad(shadow_depth_array, vec2<i32>(
                i32(sample_coords.x * csm_uniforms.shadow_map_size),
                i32(sample_coords.y * csm_uniforms.shadow_map_size)
            ), cascade_idx).x;

            if (depth < coords.z - bias) {
                blockers = blockers + depth;
                count = count + 1.0;
            }
        }
    }

    return vec2<f32>(blockers, count);
}

fn sample_shadow_pcss(coords: vec3<f32>, cascade_idx: u32, bias: f32) -> f32 {
    let blocker_radius = csm_uniforms.technique_params.x;
    let filter_radius = csm_uniforms.technique_params.y;
    let light_radius = max(csm_uniforms.technique_params.w, 0.0001);

    let search = blocker_search(coords, cascade_idx, bias, blocker_radius);
    if (search.y < 1.0) {
        // No blockers -> fully lit
        return 1.0;
    }

    let avg_blocker = search.x / search.y;
    let receiver = coords.z;
    let penumbra = clamp((receiver - avg_blocker) / avg_blocker, 0.0, 1.0) * light_radius;
    let kernel_scale = filter_radius + penumbra;

    if (kernel_scale <= 0.001) {
        return sample_shadow_hard(coords, cascade_idx, bias);
    }

    if (csm_uniforms.pcf_kernel_size >= 7u) {
        return poisson_disk_sample(coords, cascade_idx, bias, kernel_scale);
    }

    return sample_shadow_pcf(coords, cascade_idx, bias, kernel_scale);
}

fn sample_shadow_vsm(coords: vec3<f32>, cascade_idx: u32, bias: f32) -> f32 {
    let sample_coords = coords.xy;
    let moments = textureSample(shadow_moment_array, shadow_moment_sampler, sample_coords, cascade_idx).xy;
    let moment_bias = max(csm_uniforms.technique_params.z, MOMENT_BIAS_EPS);

    var p = coords.z - bias - moment_bias;
    let mean = moments.x;
    let variance = max(moments.y - mean * mean, moment_bias);
    let d = p - mean;
    let chebyshev = variance / (variance + d * d);
    let lit = p <= mean ? 1.0 : chebyshev;
    return clamp(lit, 0.0, 1.0);
}

fn warp_evsm(value: f32, positive_exp: f32, negative_exp: f32) -> vec2<f32> {
    return vec2<f32>(
        exp(positive_exp * value),
        -exp(-negative_exp * value)
    );
}

fn sample_shadow_evsm(coords: vec3<f32>, cascade_idx: u32, bias: f32) -> f32 {
    let sample_coords = coords.xy;
    let moments = textureSample(shadow_moment_array, shadow_moment_sampler, sample_coords, cascade_idx);
    let positive_exp = csm_uniforms.evsm_positive_exp;
    let negative_exp = csm_uniforms.evsm_negative_exp;

    let warp = warp_evsm(coords.z - bias, positive_exp, negative_exp);
    let pos_moments = vec2<f32>(moments.x, moments.y);
    let neg_moments = vec2<f32>(moments.z, moments.w);

    let pos_variance = max(pos_moments.y - pos_moments.x * pos_moments.x, MOMENT_BIAS_EPS);
    let neg_variance = max(neg_moments.y - neg_moments.x * neg_moments.x, MOMENT_BIAS_EPS);

    var lit = 1.0;
    let warp_pos = warp.x;
    if (warp_pos > pos_moments.x) {
        let d = warp_pos - pos_moments.x;
        lit = min(lit, pos_variance / (pos_variance + d * d));
    }

    let warp_neg = warp.y;
    if (warp_neg < neg_moments.x) {
        let d = neg_moments.x - warp_neg;
        lit = min(lit, neg_variance / (neg_variance + d * d));
    }

    return clamp(lit, 0.0, 1.0);
}

fn sample_shadow_msm(coords: vec3<f32>, cascade_idx: u32, bias: f32) -> f32 {
    let sample_coords = coords.xy;
    let moments = textureSample(shadow_moment_array, shadow_moment_sampler, sample_coords, cascade_idx);

    let c0 = moments.x;
    let c1 = moments.y;
    let c2 = moments.z;
    let c3 = moments.w;

    let dist = coords.z - bias;
    let z1 = c0;
    let z2 = c1 / (c0 + MOMENT_BIAS_EPS);

    if (dist <= z1) {
        return 1.0;
    }

    // Polynomial reconstruction (MSM 4-moment)
    let d1 = dist - z1;
    let d2 = dist - z2;
    let numerator = d1 * d2;
    let denom = d1 * d1 + d2 * d2 + MOMENT_BIAS_EPS;

    let result = 1.0 - numerator / denom;
    return clamp(result, 0.0, 1.0);
}

fn cascade_fade(view_depth: f32, cascade_idx: u32, shadow_value: f32) -> f32 {
    if (cascade_idx + 1u >= csm_uniforms.cascade_count) {
        return shadow_value;
    }

    let near = csm_uniforms.cascades[cascade_idx].near_distance;
    let far = csm_uniforms.cascades[cascade_idx].far_distance;
    let next_far = csm_uniforms.cascades[cascade_idx + 1u].far_distance;
    let fade_start = far - (far - near) * 0.05;
    let fade_end = min(next_far, far + (next_far - far) * 0.05);

    if (view_depth <= fade_start || view_depth >= fade_end) {
        return shadow_value;
    }

    let weight = clamp((fade_end - view_depth) / (fade_end - fade_start), 0.0, 1.0);
    return mix(shadow_value, 1.0, pow(weight, 3.0));
}

fn debug_cascade_tint(cascade_idx: u32) -> vec3<f32> {
    switch (cascade_idx) {
        case 0u: { return vec3<f32>(1.0, 0.2, 0.2); }
        case 1u: { return vec3<f32>(0.2, 1.0, 0.2); }
        case 2u: { return vec3<f32>(0.2, 0.2, 1.0); }
        case 3u: { return vec3<f32>(1.0, 1.0, 0.2); }
        default: { return vec3<f32>(1.0, 0.2, 1.0); }
    }
}

fn apply_debug_overlay(color: vec3<f32>, cascade_idx: u32, shadow_value: f32) -> vec3<f32> {
    if (csm_uniforms.debug_mode == 0u) {
        return color;
    }

    if (csm_uniforms.debug_mode == 1u) {
        let tint = debug_cascade_tint(cascade_idx);
        return mix(color, tint, 0.35);
    }

    let debug_color = mix(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), shadow_value);
    return mix(color, debug_color, 0.5);
}

fn evaluate_shadow_factor(world_pos: vec3<f32>, view_depth: f32, world_normal: vec3<f32>) -> f32 {
    if (csm_uniforms.cascade_count == 0u) {
        return 1.0;
    }

    let cascade_idx = select_cascade(view_depth);
    let light_clip = world_to_light_space(world_pos, cascade_idx);
    let coords = project_shadow_coords(light_clip);

    if (!texture_in_bounds(coords)) {
        return 1.0;
    }

    let bias = calculate_depth_bias(world_normal, cascade_idx);

    let technique = csm_uniforms.technique;
    var shadow = 1.0;

    switch (technique) {
        case 0u: { // Hard
            shadow = sample_shadow_hard(coords, cascade_idx, bias);
        }
        case 1u: { // PCF
            shadow = sample_shadow_pcf(coords, cascade_idx, bias, 1.0);
        }
        case 2u: { // PCSS
            shadow = sample_shadow_pcss(coords, cascade_idx, bias);
        }
        case 3u: { // VSM
            shadow = sample_shadow_vsm(coords, cascade_idx, bias);
        }
        case 4u: { // EVSM
            shadow = sample_shadow_evsm(coords, cascade_idx, bias);
        }
        case 5u: { // MSM
            shadow = sample_shadow_msm(coords, cascade_idx, bias);
        }
        case 6u: { // CSM alias (default to PCF)
            shadow = sample_shadow_pcf(coords, cascade_idx, bias, 1.0);
        }
        default: {
            shadow = sample_shadow_pcf(coords, cascade_idx, bias, 1.0);
        }
    }

    return cascade_fade(view_depth, cascade_idx, shadow);
}

fn evaluate_shadow_color(
    base_color: vec3<f32>,
    world_pos: vec3<f32>,
    view_depth: f32,
    world_normal: vec3<f32>
) -> vec3<f32> {
    let cascade_idx = select_cascade(view_depth);
    let factor = evaluate_shadow_factor(world_pos, view_depth, world_normal);
    return apply_debug_overlay(base_color * factor, cascade_idx, factor);
}
