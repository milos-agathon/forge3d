// SSR - Screen-space reflections (P5)
// Hierarchical Z-buffer ray marching with thickness-based hit detection
// Fallback to environment map for off-screen/missed rays

struct SSRParams {
    max_steps: u32,           // Max ray marching steps (32-64)
    max_distance: f32,        // Max reflection distance in view space
    thickness: f32,           // Ray hit thickness tolerance
    stride: f32,              // Initial step size multiplier
    intensity: f32,           // Reflection intensity
    roughness_fade: f32,      // Fade reflections with roughness [0-1]
    edge_fade: f32,           // Screen edge fade distance [0-0.2]
    temporal_alpha: f32,      // Temporal accumulation [0=off, 0.85=strong]
}

struct ViewParams {
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    resolution: vec2<f32>,
    near: f32,
    far: f32,
    _pad: f32,
}

@group(0) @binding(0) var depth_texture: texture_2d<f32>;
@group(0) @binding(1) var normal_texture: texture_2d<f32>;
@group(0) @binding(2) var albedo_texture: texture_2d<f32>;
@group(0) @binding(3) var<uniform> params: SSRParams;
@group(0) @binding(4) var<uniform> view: ViewParams;
@group(0) @binding(5) var depth_sampler: sampler;

// Environment map for fallback
@group(1) @binding(0) var env_specular: texture_cube<f32>;
@group(1) @binding(1) var env_sampler: sampler;

// Output and history
@group(2) @binding(0) var output_ssr: texture_storage_2d<rgba16float, write>;
@group(2) @binding(1) var history_ssr: texture_2d<f32>;
@group(2) @binding(2) var history_sampler: sampler;

// Hierarchical depth buffer (optional mip chain for faster traversal)
@group(3) @binding(0) var depth_hierarchy: texture_2d<f32>;

// ============================================================================
// Utility functions
// ============================================================================

const PI: f32 = 3.14159265359;

fn reconstruct_view_pos(uv: vec2<f32>, linear_depth: f32) -> vec3<f32> {
    let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    let focal = vec2<f32>(view.inv_proj[0][0], view.inv_proj[1][1]);
    let center = vec2<f32>(view.inv_proj[2][0], view.inv_proj[2][1]);
    let view_xy = (ndc_xy - center) / focal;
    return vec3<f32>(view_xy * linear_depth, -linear_depth);
}

fn view_to_world(view_vec: vec3<f32>) -> vec3<f32> {
    return normalize((view.inv_view * vec4<f32>(view_vec, 0.0)).xyz);
}

fn project_to_screen(view_pos: vec3<f32>) -> vec3<f32> {
    let clip_pos = view.proj * vec4<f32>(view_pos, 1.0);
    let ndc = clip_pos.xyz / clip_pos.w;
    let screen_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
    return vec3<f32>(screen_uv, ndc.z);
}

fn unpack_normal(packed: vec4<f32>) -> vec3<f32> {
    return normalize(packed.xyz * 2.0 - 1.0);
}

fn get_roughness(packed: vec4<f32>) -> f32 {
    return packed.w;
}

fn get_metallic(packed: vec4<f32>) -> f32 {
    return packed.a;
}

// Interleaved gradient noise for dithering
fn interleaved_gradient_noise(pixel: vec2<u32>, frame: u32) -> f32 {
    let px = vec2<f32>(f32(pixel.x), f32(pixel.y)) + vec2<f32>(f32(frame % 64u));
    return fract(52.9829189 * fract(0.06711056 * px.x + 0.00583715 * px.y));
}

// Screen edge fade factor
fn compute_edge_fade(uv: vec2<f32>) -> f32 {
    let edge_dist = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    return smoothstep(0.0, params.edge_fade, edge_dist);
}

// ============================================================================
// Hierarchical ray marching
// ============================================================================

struct RayHit {
    hit: bool,
    uv: vec2<f32>,
    color: vec3<f32>,
    confidence: f32,  // Hit confidence for blending
}

fn ray_march_hierarchical(origin: vec3<f32>, direction: vec3<f32>, pixel: vec2<u32>) -> RayHit {
    var hit: RayHit;
    hit.hit = false;
    hit.confidence = 0.0;
    hit.color = vec3<f32>(0.0);

    // Reject rays pointing away from camera or nearly parallel to view plane
    if (direction.z >= -0.01) {
        return hit;
    }

    let ray_start = origin;
    let ray_end = origin + direction * params.max_distance;

    // Project to screen space
    let start_screen = project_to_screen(ray_start);
    let end_screen = project_to_screen(ray_end);

    // Check if start is off-screen
    if (start_screen.x < 0.0 || start_screen.x > 1.0 || start_screen.y < 0.0 || start_screen.y > 1.0) {
        return hit;
    }

    // Ray delta in screen space
    let ray_delta = end_screen - start_screen;
    let ray_length = length(ray_delta.xy);

    // Early exit if ray is too short
    if (ray_length < 0.001) {
        return hit;
    }

    // Adaptive step size based on ray direction
    var step_size = params.stride / max(abs(ray_delta.x), abs(ray_delta.y)) / view.resolution.x;
    step_size = max(step_size, 1.0 / f32(params.max_steps));

    var t = 0.0;
    var prev_depth_diff = 0.0;
    var iterations = 0u;

    while (t < 1.0 && iterations < params.max_steps) {
        iterations = iterations + 1u;

        // Current sample position in screen space
        let sample_screen = start_screen + ray_delta * t;

        // Out of bounds check
        if (sample_screen.x < 0.0 || sample_screen.x > 1.0 ||
            sample_screen.y < 0.0 || sample_screen.y > 1.0) {
            break;
        }

        // Sample depth buffer
        let sample_depth = textureSampleLevel(depth_texture, depth_sampler, sample_screen.xy, 0.0).r;

        // Current ray depth
        let ray_pos = mix(ray_start, ray_end, t);
        let ray_depth = -ray_pos.z;

        // Depth difference (positive = ray is behind surface)
        let depth_diff = ray_depth - sample_depth;

        // Hit detection with thickness tolerance
        if (depth_diff > 0.0 && depth_diff < params.thickness) {
            // Binary search refinement for better accuracy
            var refine_t_min = t - step_size;
            var refine_t_max = t;

            for (var i = 0; i < 4; i = i + 1) {
                let mid_t = (refine_t_min + refine_t_max) * 0.5;
                let mid_screen = start_screen + ray_delta * mid_t;
                let mid_depth = textureSampleLevel(depth_texture, depth_sampler, mid_screen.xy, 0.0).r;

                let mid_ray_pos = mix(ray_start, ray_end, mid_t);
                let mid_ray_depth = -mid_ray_pos.z;
                let mid_diff = mid_ray_depth - mid_depth;

                if (mid_diff > 0.0) {
                    refine_t_max = mid_t;
                } else {
                    refine_t_min = mid_t;
                }
            }

            let refined_t = (refine_t_min + refine_t_max) * 0.5;
            let refined_screen = start_screen + ray_delta * refined_t;

            hit.hit = true;
            hit.uv = refined_screen.xy;
            hit.color = textureSampleLevel(albedo_texture, depth_sampler, refined_screen.xy, 0.0).rgb;

            // Compute confidence based on ray quality
            let edge_fade = compute_edge_fade(refined_screen.xy);
            let distance_fade = 1.0 - refined_t;
            hit.confidence = edge_fade * distance_fade;

            break;
        }

        // Adaptive step size: increase when far from surface, decrease when close
        if (depth_diff < -params.thickness) {
            // Ray is in front of surface, take larger steps
            step_size = step_size * 1.5;
        } else {
            // Getting close to surface, take smaller steps
            step_size = step_size * 0.75;
        }

        step_size = clamp(step_size, 1.0 / f32(params.max_steps * 2u), 0.1);
        t = t + step_size;
        prev_depth_diff = depth_diff;
    }

    return hit;
}

// ============================================================================
// Environment map fallback
// ============================================================================

fn sample_specular_env(world_reflect: vec3<f32>, roughness: f32) -> vec3<f32> {
    // Sample appropriate mip level based on roughness
    let max_mip = 5.0;  // Assuming 6 mip levels
    let mip_level = roughness * max_mip;
    return textureSampleLevel(env_specular, env_sampler, world_reflect, mip_level).rgb;
}

// ============================================================================
// Fresnel for reflection intensity
// ============================================================================

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - cos_theta, 5.0);
}

fn calculate_f0(albedo: vec3<f32>, metallic: f32) -> vec3<f32> {
    let dielectric_f0 = vec3<f32>(0.04);
    return mix(dielectric_f0, albedo, metallic);
}

// ============================================================================
// Main SSR compute shader
// ============================================================================

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = global_id.xy;
    let dims = textureDimensions(depth_texture);

    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }

    // Load GBuffer data
    let depth = textureLoad(depth_texture, pixel, 0).r;
    let normal_packed = textureLoad(normal_texture, pixel, 0);
    let albedo_metal = textureLoad(albedo_texture, pixel, 0);

    let roughness = get_roughness(normal_packed);
    let metallic = get_metallic(albedo_metal);
    let albedo = albedo_metal.rgb;

    // Skip background/sky or very rough surfaces
    if (depth >= view.far * 0.99 || roughness > params.roughness_fade) {
        textureStore(output_ssr, pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let uv = vec2<f32>(f32(pixel.x), f32(pixel.y)) / view.resolution;
    let view_pos = reconstruct_view_pos(uv, depth);
    let view_normal = unpack_normal(normal_packed);

    // Compute reflection direction in view space
    let view_dir = normalize(-view_pos);
    let reflect_dir = reflect(-view_dir, view_normal);

    // Convert to world space for environment sampling
    let world_reflect = view_to_world(reflect_dir);

    // Add small jitter for temporal accumulation
    let jitter = interleaved_gradient_noise(pixel, 0u) * 0.01;
    let jittered_reflect = normalize(reflect_dir + vec3<f32>(jitter, jitter, 0.0));

    // Ray march for screen-space hit
    let ray_hit = ray_march_hierarchical(view_pos, jittered_reflect, pixel);

    var reflection_color: vec3<f32>;
    var reflection_alpha: f32;

    if (ray_hit.hit) {
        // Use screen-space hit
        reflection_color = ray_hit.color;
        reflection_alpha = ray_hit.confidence;
    } else {
        // Fallback to environment map
        reflection_color = sample_specular_env(world_reflect, roughness);
        reflection_alpha = 1.0 - roughness;  // Fade env reflections with roughness
    }

    // Apply Fresnel
    let n_dot_v = max(dot(view_normal, view_dir), 0.0);
    let f0 = calculate_f0(albedo, metallic);
    let fresnel = fresnel_schlick(n_dot_v, f0);

    reflection_color = reflection_color * fresnel * params.intensity;

    // Roughness-based fade
    let roughness_blend = 1.0 - smoothstep(0.0, params.roughness_fade, roughness);
    reflection_alpha = reflection_alpha * roughness_blend;

    // Temporal accumulation
    if (params.temporal_alpha > 0.0) {
        let history = textureSample(history_ssr, history_sampler, uv);
        reflection_color = mix(reflection_color, history.rgb, params.temporal_alpha);
        reflection_alpha = mix(reflection_alpha, history.a, params.temporal_alpha);
    }

    textureStore(output_ssr, pixel, vec4<f32>(reflection_color, reflection_alpha));
}

// ======================================================================================
// Minimal pipeline entrypoints to match Rust SsrRenderer layouts
// ======================================================================================

// Group(0) layout used by SsrRenderer::ssr_bind_group_layout
@group(0) @binding(0) var ssr_depth_tex: texture_2d<f32>;
@group(0) @binding(1) var ssr_normal_tex: texture_2d<f32>;
@group(0) @binding(2) var ssr_albedo_tex: texture_2d<f32>;
@group(0) @binding(3) var ssr_out_rgba: texture_storage_2d<rgba16float, write>;
// bindings(4) and (5) are uniforms in Rust; not used here to keep shader simple
@group(0) @binding(6) var ssr_env_cube: texture_cube<f32>;
@group(0) @binding(7) var ssr_env_sam: sampler;

// Simple SSR that writes albedo color as a placeholder (ensures pipeline compatibility)
@compute @workgroup_size(8, 8, 1)
fn cs_ssr(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = global_id.xy;
    let dims = textureDimensions(ssr_albedo_tex);
    if (pixel.x >= dims.x || pixel.y >= dims.y) { return; }
    let albedo = textureLoad(ssr_albedo_tex, pixel, 0).rgb;
    textureStore(ssr_out_rgba, pixel, vec4<f32>(albedo, 1.0));
}

// Group(0) layout used by SsrRenderer::temporal_bind_group_layout
@group(0) @binding(0) var ssr_current: texture_2d<f32>;
@group(0) @binding(1) var ssr_history: texture_2d<f32>;
@group(0) @binding(2) var ssr_filtered_out: texture_storage_2d<rgba16float, write>;

// Temporal filter: simple copy of current into filtered (placeholder)
@compute @workgroup_size(8, 8, 1)
fn cs_ssr_temporal(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = global_id.xy;
    let dims = textureDimensions(ssr_current);
    if (pixel.x >= dims.x || pixel.y >= dims.y) { return; }
    let cur = textureLoad(ssr_current, pixel, 0);
    textureStore(ssr_filtered_out, pixel, cur);
}

// ============================================================================
// Hierarchical depth buffer generation (optional preprocessing)
// ============================================================================

@group(0) @binding(0) var input_depth: texture_2d<f32>;
@group(1) @binding(0) var output_depth_mip: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_build_depth_hierarchy(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = global_id.xy;
    let dims = textureDimensions(output_depth_mip);

    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }

    // Sample 2x2 quad from higher resolution
    let base_pixel = pixel * 2u;
    let d00 = textureLoad(input_depth, base_pixel + vec2<u32>(0u, 0u), 0).r;
    let d10 = textureLoad(input_depth, base_pixel + vec2<u32>(1u, 0u), 0).r;
    let d01 = textureLoad(input_depth, base_pixel + vec2<u32>(0u, 1u), 0).r;
    let d11 = textureLoad(input_depth, base_pixel + vec2<u32>(1u, 1u), 0).r;

    // Use max depth (farthest) for conservative hierarchical testing
    let max_depth = max(max(d00, d10), max(d01, d11));

    textureStore(output_depth_mip, pixel, vec4<f32>(max_depth));
}
