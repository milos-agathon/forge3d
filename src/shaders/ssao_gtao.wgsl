// SSAO/GTAO - Screen-space ambient occlusion (P5)
// Supports both SSAO (hemisphere sampling) and GTAO (horizon-based)

struct SSAOParams {
    radius: f32,              // World-space occlusion radius
    intensity: f32,           // AO intensity multiplier [0-2]
    bias: f32,                // Depth bias to prevent self-occlusion
    sample_count: u32,        // Number of samples per pixel
    spiral_turns: f32,        // Spiral pattern parameter
    technique: u32,           // 0=SSAO, 1=GTAO
    blur_radius: u32,         // Bilateral blur radius
    temporal_alpha: f32,      // Temporal accumulation factor [0=off, 0.95=strong]
}

struct ViewParams {
    inv_proj: mat4x4<f32>,
    proj: mat4x4<f32>,
    resolution: vec2<f32>,
    near: f32,
    far: f32,
}

@group(0) @binding(0) var depth_texture: texture_2d<f32>;
@group(0) @binding(1) var normal_texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: SSAOParams;
@group(0) @binding(3) var<uniform> view: ViewParams;
@group(0) @binding(4) var noise_texture: texture_2d<f32>;
@group(0) @binding(5) var noise_sampler: sampler;

// Output texture
@group(1) @binding(0) var output_ao: texture_storage_2d<r16float, write>;

// Temporal history (optional)
@group(2) @binding(0) var history_ao: texture_2d<f32>;
@group(2) @binding(1) var history_sampler: sampler;

// ============================================================================
// Utility functions
// ============================================================================

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

fn reconstruct_view_pos(uv: vec2<f32>, linear_depth: f32) -> vec3<f32> {
    let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    let focal = vec2<f32>(view.inv_proj[0][0], view.inv_proj[1][1]);
    let center = vec2<f32>(view.inv_proj[2][0], view.inv_proj[2][1]);
    let view_xy = (ndc_xy - center) / focal;
    return vec3<f32>(view_xy * linear_depth, -linear_depth);
}

fn unpack_normal(packed: vec4<f32>) -> vec3<f32> {
    return normalize(packed.xyz * 2.0 - 1.0);
}

// Interleaved gradient noise for temporal dithering
fn interleaved_gradient_noise(pixel: vec2<u32>, frame: u32) -> f32 {
    let px = vec2<f32>(f32(pixel.x), f32(pixel.y)) + vec2<f32>(f32(frame % 64u));
    return fract(52.9829189 * fract(0.06711056 * px.x + 0.00583715 * px.y));
}

// ============================================================================
// SSAO - Hemisphere sampling
// ============================================================================

fn compute_ssao(pixel: vec2<u32>, center_pos: vec3<f32>, center_normal: vec3<f32>) -> f32 {
    let uv = vec2<f32>(f32(pixel.x), f32(pixel.y)) / view.resolution;

    // Random rotation
    let noise_uv = uv * view.resolution / 4.0;  // 4x4 noise tile
    let noise = textureSample(noise_texture, noise_sampler, noise_uv).xy * 2.0 - 1.0;

    // Construct TBN matrix (tangent space)
    let tangent = normalize(vec3<f32>(noise.x, noise.y, 0.0) - center_normal * dot(vec3<f32>(noise.x, noise.y, 0.0), center_normal));
    let bitangent = cross(center_normal, tangent);
    let tbn = mat3x3<f32>(tangent, bitangent, center_normal);

    var occlusion = 0.0;
    let sample_count_f = f32(params.sample_count);

    for (var i = 0u; i < params.sample_count; i = i + 1u) {
        // Spiral sampling pattern
        let fi = f32(i);
        let alpha = (fi + 0.5) / sample_count_f;
        let angle = alpha * params.spiral_turns * TWO_PI;
        let radius_scale = alpha;

        // Hemisphere sample
        let h = sqrt(1.0 - alpha);  // Height in hemisphere
        let r = sqrt(alpha);
        let sample_dir = vec3<f32>(cos(angle) * r, sin(angle) * r, h);

        // Transform to view space
        let sample_offset = tbn * sample_dir * params.radius;
        let sample_pos = center_pos + sample_offset;

        // Project to screen space
        let sample_clip = view.proj * vec4<f32>(sample_pos, 1.0);
        var sample_uv = sample_clip.xy / sample_clip.w;
        sample_uv = sample_uv * 0.5 + 0.5;
        sample_uv.y = 1.0 - sample_uv.y;

        // Sample depth at offset position
        if (sample_uv.x >= 0.0 && sample_uv.x <= 1.0 && sample_uv.y >= 0.0 && sample_uv.y <= 1.0) {
            let sample_depth = textureLoad(depth_texture, vec2<u32>(sample_uv * view.resolution), 0).r;
            let sample_view_pos = reconstruct_view_pos(sample_uv, sample_depth);

            // Range check
            let range_check = smoothstep(0.0, 1.0, params.radius / abs(center_pos.z - sample_view_pos.z));

            // Occlusion test with bias
            let delta_z = sample_view_pos.z - sample_pos.z;
            if (delta_z >= params.bias) {
                occlusion = occlusion + range_check;
            }
        }
    }

    occlusion = occlusion / sample_count_f;
    return 1.0 - clamp(occlusion * params.intensity, 0.0, 1.0);
}

// ============================================================================
// GTAO - Ground-truth ambient occlusion (horizon-based)
// ============================================================================

fn compute_gtao(pixel: vec2<u32>, center_pos: vec3<f32>, center_normal: vec3<f32>) -> f32 {
    let uv = vec2<f32>(f32(pixel.x), f32(pixel.y)) / view.resolution;

    // Random rotation per pixel
    let noise = interleaved_gradient_noise(pixel, 0u);
    let angle_offset = noise * TWO_PI;

    var visibility = 0.0;
    let direction_count = max(params.sample_count / 4u, 2u);  // Fewer directions, more samples per direction
    let steps_per_direction = 4u;

    for (var d = 0u; d < direction_count; d = d + 1u) {
        let angle = (f32(d) / f32(direction_count)) * PI + angle_offset;
        let direction = vec2<f32>(cos(angle), sin(angle));

        var horizon_cos = -1.0;

        // March along direction to find horizon
        for (var s = 1u; s <= steps_per_direction; s = s + 1u) {
            let step_size = params.radius * f32(s) / f32(steps_per_direction);
            let sample_uv = uv + direction * step_size / view.resolution;

            if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
                continue;
            }

            let sample_depth = textureLoad(depth_texture, vec2<u32>(sample_uv * view.resolution), 0).r;
            let sample_pos = reconstruct_view_pos(sample_uv, sample_depth);

            let horizon_vec = sample_pos - center_pos;
            let horizon_len = length(horizon_vec);

            // Compute horizon angle
            let h_cos = dot(horizon_vec / horizon_len, center_normal);

            // Attenuation based on distance
            let attenuation = 1.0 - clamp(horizon_len / params.radius, 0.0, 1.0);

            horizon_cos = max(horizon_cos, h_cos * attenuation);
        }

        // Integrate visibility
        let horizon_angle = acos(clamp(horizon_cos, -1.0, 1.0));
        visibility = visibility + sin(horizon_angle) - horizon_angle * cos(horizon_angle) + 0.5 * PI;
    }

    visibility = visibility / (f32(direction_count) * PI);
    return clamp(1.0 - visibility * params.intensity, 0.0, 1.0);
}

// ============================================================================
// Main compute shader
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

    // Skip background/sky
    if (depth >= view.far * 0.99) {
        textureStore(output_ao, pixel, vec4<f32>(1.0));
        return;
    }

    let uv = vec2<f32>(f32(pixel.x), f32(pixel.y)) / view.resolution;
    let view_pos = reconstruct_view_pos(uv, depth);
    let view_normal = unpack_normal(normal_packed);

    // Compute AO based on technique
    var ao: f32;
    if (params.technique == 0u) {
        ao = compute_ssao(pixel, view_pos, view_normal);
    } else {
        ao = compute_gtao(pixel, view_pos, view_normal);
    }

    // Optional temporal accumulation
    if (params.temporal_alpha > 0.0) {
        let history = textureSample(history_ao, history_sampler, uv).r;
        ao = mix(ao, history, params.temporal_alpha);
    }

    textureStore(output_ao, pixel, vec4<f32>(ao));
}

// ============================================================================
// Bilateral blur for denoising
// ============================================================================

@group(0) @binding(0) var input_ao: texture_2d<f32>;
@group(1) @binding(0) var output_blurred: texture_storage_2d<r16float, write>;

fn bilateral_weight(center_depth: f32, center_normal: vec3<f32>,
                    sample_depth: f32, sample_normal: vec3<f32>,
                    spatial_sigma: f32, depth_sigma: f32) -> f32 {
    let depth_diff = abs(center_depth - sample_depth);
    let normal_diff = 1.0 - dot(center_normal, sample_normal);

    let depth_weight = exp(-depth_diff * depth_diff / (2.0 * depth_sigma * depth_sigma));
    let normal_weight = exp(-normal_diff * normal_diff / 0.1);

    return depth_weight * normal_weight;
}

@compute @workgroup_size(8, 8, 1)
fn cs_blur(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = global_id.xy;
    let dims = textureDimensions(input_ao);

    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }

    let center_depth = textureLoad(depth_texture, pixel, 0).r;
    let center_normal = unpack_normal(textureLoad(normal_texture, pixel, 0));
    let center_ao = textureLoad(input_ao, pixel, 0).r;

    var ao_sum = center_ao;
    var weight_sum = 1.0;

    let radius = i32(params.blur_radius);

    for (var dy = -radius; dy <= radius; dy = dy + 1) {
        for (var dx = -radius; dx <= radius; dx = dx + 1) {
            if (dx == 0 && dy == 0) {
                continue;
            }

            let sample_pixel = vec2<i32>(pixel) + vec2<i32>(dx, dy);
            if (sample_pixel.x < 0 || sample_pixel.x >= i32(dims.x) ||
                sample_pixel.y < 0 || sample_pixel.y >= i32(dims.y)) {
                continue;
            }

            let sample_depth = textureLoad(depth_texture, vec2<u32>(sample_pixel), 0).r;
            let sample_normal = unpack_normal(textureLoad(normal_texture, vec2<u32>(sample_pixel), 0));
            let sample_ao = textureLoad(input_ao, vec2<u32>(sample_pixel), 0).r;

            let weight = bilateral_weight(center_depth, center_normal, sample_depth, sample_normal, 2.0, 0.5);

            ao_sum = ao_sum + sample_ao * weight;
            weight_sum = weight_sum + weight;
        }
    }

    let blurred_ao = ao_sum / weight_sum;
    textureStore(output_blurred, pixel, vec4<f32>(blurred_ao));
}
