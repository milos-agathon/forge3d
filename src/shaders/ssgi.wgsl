// SSGI - Screen-space global illumination (P5)
// Half-res ray march in depth buffer with fallback to diffuse IBL and temporal accumulation

struct SSGIParams {
    ray_steps: u32,           // Max ray marching steps (16-32)
    ray_radius: f32,          // Max ray distance in world space
    ray_thickness: f32,       // Ray hit thickness tolerance
    intensity: f32,           // GI intensity multiplier
    temporal_alpha: f32,      // Temporal accumulation [0=off, 0.9=strong]
    use_half_res: u32,        // 1=half resolution for performance
    ibl_fallback: f32,        // IBL contribution when ray misses [0-1]
    _pad: f32,
}

struct ViewParams {
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    resolution: vec2<f32>,
    near: f32,
    far: f32,
}

@group(0) @binding(0) var depth_texture: texture_2d<f32>;
@group(0) @binding(1) var normal_texture: texture_2d<f32>;
@group(0) @binding(2) var albedo_texture: texture_2d<f32>;
@group(0) @binding(3) var<uniform> params: SSGIParams;
@group(0) @binding(4) var<uniform> view: ViewParams;
@group(0) @binding(5) var depth_sampler: sampler;

// IBL for fallback
@group(1) @binding(0) var env_irradiance: texture_cube<f32>;
@group(1) @binding(1) var env_sampler: sampler;

// Output and history
@group(2) @binding(0) var output_gi: texture_storage_2d<rgba16float, write>;
@group(2) @binding(1) var history_gi: texture_2d<f32>;
@group(2) @binding(2) var history_sampler: sampler;

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

fn view_to_world(view_pos: vec3<f32>) -> vec3<f32> {
    let world_pos = view.inv_view * vec4<f32>(view_pos, 1.0);
    return world_pos.xyz;
}

fn world_to_view(world_pos: vec3<f32>) -> vec3<f32> {
    let view_pos = view.view * vec4<f32>(world_pos, 1.0);
    return view_pos.xyz;
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

// Interleaved gradient noise for ray dithering
fn interleaved_gradient_noise(pixel: vec2<u32>, frame: u32) -> f32 {
    let px = vec2<f32>(f32(pixel.x), f32(pixel.y)) + vec2<f32>(f32(frame % 64u));
    return fract(52.9829189 * fract(0.06711056 * px.x + 0.00583715 * px.y));
}

// Cosine-weighted hemisphere sampling
fn sample_cosine_hemisphere(u1: f32, u2: f32) -> vec3<f32> {
    let r = sqrt(u1);
    let theta = TWO_PI * u2;
    let x = r * cos(theta);
    let y = r * sin(theta);
    let z = sqrt(max(0.0, 1.0 - u1));
    return vec3<f32>(x, y, z);
}

// Construct TBN matrix from normal
fn build_tbn(normal: vec3<f32>) -> mat3x3<f32> {
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(normal.y) > 0.999) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    return mat3x3<f32>(tangent, bitangent, normal);
}

// ============================================================================
// Ray marching against depth buffer
// ============================================================================

struct RayHit {
    hit: bool,
    uv: vec2<f32>,
    color: vec3<f32>,
}

fn ray_march(origin: vec3<f32>, direction: vec3<f32>, pixel: vec2<u32>) -> RayHit {
    var hit: RayHit;
    hit.hit = false;
    hit.color = vec3<f32>(0.0);

    // Start position slightly offset from surface
    let start_pos = origin + direction * 0.1;
    let end_pos = origin + direction * params.ray_radius;

    // Project to screen space
    let start_screen = project_to_screen(start_pos);
    let end_screen = project_to_screen(end_pos);

    // Check if ray is off-screen
    if (start_screen.x < 0.0 || start_screen.x > 1.0 || start_screen.y < 0.0 || start_screen.y > 1.0) {
        return hit;
    }

    // Ray step in screen space
    let ray_delta = end_screen - start_screen;
    let step_count = f32(params.ray_steps);

    var prev_depth_diff = 0.0;

    for (var i = 1u; i <= params.ray_steps; i = i + 1u) {
        let t = f32(i) / step_count;
        let sample_screen = start_screen + ray_delta * t;

        // Out of bounds check
        if (sample_screen.x < 0.0 || sample_screen.x > 1.0 ||
            sample_screen.y < 0.0 || sample_screen.y > 1.0) {
            break;
        }

        // Sample depth at current position
        let sample_depth = textureSampleLevel(depth_texture, depth_sampler, sample_screen.xy, 0.0).r;

        // Interpolate ray depth
        let ray_pos = mix(start_pos, end_pos, t);
        let ray_depth = -ray_pos.z;

        // Check for intersection
        let depth_diff = ray_depth - sample_depth;

        // Hit detection with thickness tolerance
        if (depth_diff > 0.0 && depth_diff < params.ray_thickness) {
            // Refine hit with linear search between prev and current
            if (i > 1u) {
                let refined_t = t - (depth_diff / (depth_diff - prev_depth_diff)) / step_count;
                let refined_screen = start_screen + ray_delta * refined_t;

                hit.hit = true;
                hit.uv = refined_screen.xy;
                hit.color = textureLoad(albedo_texture, vec2<u32>(refined_screen.xy * view.resolution), 0).rgb;
            } else {
                hit.hit = true;
                hit.uv = sample_screen.xy;
                hit.color = textureLoad(albedo_texture, vec2<u32>(sample_screen.xy * view.resolution), 0).rgb;
            }
            break;
        }

        prev_depth_diff = depth_diff;
    }

    return hit;
}

// ============================================================================
// IBL fallback for missed rays
// ============================================================================

fn sample_diffuse_ibl(world_normal: vec3<f32>) -> vec3<f32> {
    return textureSample(env_irradiance, env_sampler, world_normal).rgb;
}

// ============================================================================
// Main SSGI compute shader
// ============================================================================

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var pixel = global_id.xy;

    // Handle half-resolution mode
    if (params.use_half_res == 1u) {
        pixel = pixel * 2u;
    }

    let dims = textureDimensions(depth_texture);
    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }

    // Load GBuffer data
    let depth = textureLoad(depth_texture, pixel, 0).r;
    let normal_packed = textureLoad(normal_texture, pixel, 0);
    let albedo = textureLoad(albedo_texture, pixel, 0).rgb;

    // Skip background/sky
    if (depth >= view.far * 0.99) {
        textureStore(output_gi, global_id.xy, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let uv = vec2<f32>(f32(pixel.x), f32(pixel.y)) / view.resolution;
    let view_pos = reconstruct_view_pos(uv, depth);
    let view_normal = unpack_normal(normal_packed);
    let world_normal = normalize((view.inv_view * vec4<f32>(view_normal, 0.0)).xyz);

    // Build tangent space for hemisphere sampling
    let tbn = build_tbn(view_normal);

    // Generate random numbers for this pixel
    let noise1 = interleaved_gradient_noise(pixel, 0u);
    let noise2 = interleaved_gradient_noise(pixel, 1u);

    // Sample one ray per pixel (accumulate over time)
    let tangent_dir = sample_cosine_hemisphere(noise1, noise2);
    let ray_dir = normalize(tbn * tangent_dir);

    // Ray march to find indirect illumination
    let ray_hit = ray_march(view_pos, ray_dir, pixel);

    var gi_color: vec3<f32>;

    if (ray_hit.hit) {
        // Use hit color modulated by surface albedo
        gi_color = ray_hit.color * albedo;
    } else {
        // Fallback to diffuse IBL
        gi_color = sample_diffuse_ibl(world_normal) * albedo * params.ibl_fallback;
    }

    // Apply intensity
    gi_color = gi_color * params.intensity;

    // Temporal accumulation to reduce noise
    if (params.temporal_alpha > 0.0) {
        let history = textureSample(history_gi, history_sampler, uv).rgb;
        gi_color = mix(gi_color, history, params.temporal_alpha);
    }

    textureStore(output_gi, global_id.xy, vec4<f32>(gi_color, 1.0));
}

// ============================================================================
// Upsampling pass for half-res mode
// ============================================================================

@group(0) @binding(0) var half_res_gi: texture_2d<f32>;
@group(0) @binding(1) var half_res_sampler: sampler;
@group(1) @binding(0) var full_res_output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_upsample(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = global_id.xy;
    let dims = textureDimensions(full_res_output);

    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }

    let uv = (vec2<f32>(pixel) + 0.5) / vec2<f32>(dims);

    // Bilateral upsampling using depth and normal
    let center_depth = textureLoad(depth_texture, pixel, 0).r;
    let center_normal = unpack_normal(textureLoad(normal_texture, pixel, 0));

    var gi_sum = vec3<f32>(0.0);
    var weight_sum = 0.0;

    // 3x3 kernel for bilateral upsampling
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let offset = vec2<f32>(f32(dx), f32(dy)) / vec2<f32>(dims);
            let sample_uv = uv + offset;

            let sample_pixel = vec2<u32>(sample_uv * view.resolution);
            if (sample_pixel.x >= dims.x || sample_pixel.y >= dims.y) {
                continue;
            }

            let sample_depth = textureLoad(depth_texture, sample_pixel, 0).r;
            let sample_normal = unpack_normal(textureLoad(normal_texture, sample_pixel, 0));
            let sample_gi = textureSample(half_res_gi, half_res_sampler, sample_uv).rgb;

            // Bilateral weights
            let depth_diff = abs(center_depth - sample_depth);
            let normal_diff = 1.0 - dot(center_normal, sample_normal);

            let weight = exp(-depth_diff / 0.5) * exp(-normal_diff / 0.1);

            gi_sum = gi_sum + sample_gi * weight;
            weight_sum = weight_sum + weight;
        }
    }

    let upsampled_gi = gi_sum / max(weight_sum, 0.001);
    textureStore(full_res_output, pixel, vec4<f32>(upsampled_gi, 1.0));
}
