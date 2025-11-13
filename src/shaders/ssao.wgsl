// SSAO - Screen-Space Ambient Occlusion (P5.1)
// Hemisphere sampling with bilateral blur and optional temporal accumulation

struct SsaoSettings {
    radius: f32,
    intensity: f32,
    bias: f32,
    num_samples: u32,
    technique: u32,  // 0=SSAO, 1=GTAO
    frame_index: u32,
    inv_resolution: vec2<f32>,
    // proj_scale = 0.5 * height / tan(fov/2) = 0.5 * height * P[1][1]
    proj_scale: f32,
    ao_min: f32,     // minimum AO clamp (default 0.35)
}

struct CameraParams {
    view_matrix: mat4x4<f32>,
    inv_view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    inv_proj_matrix: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
}

// SSAO compute pass
@group(0) @binding(0) var depth_texture: texture_2d<f32>;
@group(0) @binding(1) var hzb_texture: texture_2d<f32>;
@group(0) @binding(2) var normal_texture: texture_2d<f32>;
@group(0) @binding(3) var noise_texture: texture_2d<f32>;
@group(0) @binding(4) var tex_sampler: sampler;
@group(0) @binding(5) var output_ao: texture_storage_2d<r32float, write>;
@group(0) @binding(6) var<uniform> settings: SsaoSettings;
@group(0) @binding(7) var<uniform> camera: CameraParams;

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

// Reconstruct view-space position from linear depth (depth = -view_z > 0)
fn reconstruct_view_pos_linear(uv: vec2<f32>, linear_depth: f32) -> vec3<f32> {
    // UV [0,1] -> NDC [-1,1] with Y up
    let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    // Use projection matrix diagonal to back-project NDC to view at given linear depth
    let p00 = camera.proj_matrix[0][0];
    let p11 = camera.proj_matrix[1][1];
    let view_xy = vec2<f32>(ndc_xy.x / p00, ndc_xy.y / p11) * linear_depth;
    return vec3<f32>(view_xy, -linear_depth);
}

// Unpack normal from texture
fn unpack_normal(packed: vec4<f32>) -> vec3<f32> {
    // GBuffer stores normals directly in view space in [-1, 1]
    return normalize(packed.xyz);
}

// Interleaved gradient noise for temporal dithering
fn ign(pixel: vec2<u32>) -> f32 {
    let px = vec2<f32>(f32(pixel.x), f32(pixel.y));
    return fract(52.9829189 * fract(0.06711056 * px.x + 0.00583715 * px.y));
}

// SSAO - Hemisphere sampling
fn compute_ssao(pixel: vec2<u32>, uv: vec2<f32>, view_pos: vec3<f32>, view_normal: vec3<f32>) -> f32 {
    // Frame-indexed interleaved gradient noise
    let noise_angle = ign(pixel + vec2<u32>(settings.frame_index, settings.frame_index * 1664525u)) * TWO_PI;
    
    // Tangent-space basis with robust up selection
    let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), abs(view_normal.y) < 0.99);
    let tangent = normalize(cross(up, view_normal));
    let bitangent = cross(view_normal, tangent);
    let tbn = mat3x3<f32>(tangent, bitangent, view_normal);
    
    var occlusion = 0.0;
    let num_samples_f = f32(settings.num_samples);
    
    for (var i = 0u; i < settings.num_samples; i++) {
        let fi = f32(i);
        let alpha = (fi + 0.5) / num_samples_f;
        let angle = alpha * 4.0 * TWO_PI + noise_angle;
        
        // Hemisphere sample
        let h = sqrt(1.0 - alpha);
        let r = sqrt(alpha);
        let sample_dir = vec3<f32>(cos(angle) * r, sin(angle) * r, h);
        
        // Transform to view space
        let sample_offset = tbn * sample_dir * settings.radius;
        let sample_pos = view_pos + sample_offset;
        
        // Project to screen space
        let sample_clip = camera.proj_matrix * vec4<f32>(sample_pos, 1.0);
        var sample_uv = sample_clip.xy / sample_clip.w;
        sample_uv = sample_uv * 0.5 + 0.5;
        sample_uv.y = 1.0 - sample_uv.y;
        
        // Bounds check
        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
            continue;
        }
        
        let dims = textureDimensions(depth_texture);
        let sample_pixel = vec2<u32>(sample_uv * vec2<f32>(dims));
        let sample_depth = textureLoad(depth_texture, sample_pixel, 0).r;
        let sample_view_pos = reconstruct_view_pos_linear(sample_uv, sample_depth);
        
        // Range attenuation
        let delta = sample_view_pos - view_pos;
        let dist = length(delta);
        let range_check = smoothstep(0.0, 1.0, settings.radius / max(dist, 1e-4));
        // Per-pixel bias based on pixel size in view
        // pixel_size_in_view = z / proj_scale, with z = linear depth = -view.z
        let z_lin = -view_pos.z;
        let pixel_size_in_view = z_lin / max(settings.proj_scale, 1e-4);
        let bias = settings.bias + 0.5 * pixel_size_in_view;
        // Occlusion if the sampled surface is closer than the test position along view direction
        let sample_z_lin = -sample_view_pos.z;
        if (sample_z_lin + bias < z_lin) {
            occlusion += range_check;
        }
    }
    
    occlusion = occlusion / num_samples_f;
    let ao_raw = 1.0 - occlusion;
    // P5.1: clamp to [ao_min, 1.0] to prevent fully black objects
    return clamp(ao_raw, settings.ao_min, 1.0);
}

// GTAO - Ground-truth ambient occlusion (horizon-based)
fn compute_gtao(pixel: vec2<u32>, uv: vec2<f32>, view_pos: vec3<f32>, view_normal: vec3<f32>) -> f32 {
    let noise = ign(pixel + vec2<u32>(settings.frame_index * 747796405u, settings.frame_index * 2891336453u));
    let angle_offset = noise * TWO_PI;
    
    var visibility = 0.0;
    let direction_count = max(settings.num_samples / 4u, 2u);
    let steps_per_direction = 4u;
    
    for (var d = 0u; d < direction_count; d++) {
        let angle = (f32(d) / f32(direction_count)) * PI + angle_offset;
        let direction = vec2<f32>(cos(angle), sin(angle));
        
        var horizon_cos = -1.0;
        
        // March along direction
        for (var s = 1u; s <= steps_per_direction; s++) {
            // Screen-space step in UV units using r_screen = radius * (proj_scale / z)
            let z_lin = -view_pos.z;
            let r_screen = settings.radius * (settings.proj_scale / max(z_lin, 1e-4));
            let step_px = r_screen * (f32(s) / f32(steps_per_direction));
            let sample_uv = uv + direction * (step_px * settings.inv_resolution);
            
            if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
                continue;
            }
            
            let dims = textureDimensions(depth_texture);
            let sample_pixel = vec2<u32>(sample_uv * vec2<f32>(dims));
            let sample_depth = textureLoad(depth_texture, sample_pixel, 0).r;
            let sample_pos = reconstruct_view_pos_linear(sample_uv, sample_depth);
            
            let horizon_vec = sample_pos - view_pos;
            let horizon_len = length(horizon_vec);
            let h_cos = dot(horizon_vec / horizon_len, view_normal);
            
            // Attenuation
            let attenuation = 1.0 - clamp(horizon_len / settings.radius, 0.0, 1.0);
            horizon_cos = max(horizon_cos, h_cos * attenuation);
        }
        
        let horizon_angle = acos(clamp(horizon_cos, -1.0, 1.0));
        visibility += sin(horizon_angle) - horizon_angle * cos(horizon_angle) + 0.5 * PI;
    }
    
    visibility = visibility / (f32(direction_count) * PI);
    let ao_raw = 1.0 - visibility;
    // P5.1: clamp to [ao_min, 1.0] to prevent fully black objects
    return clamp(ao_raw, settings.ao_min, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn cs_ssao(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    let dims = textureDimensions(depth_texture);
    
    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }
    
    let uv = (vec2<f32>(pixel) + 0.5) / vec2<f32>(dims);
    let depth = textureLoad(depth_texture, pixel, 0).r;
    let normal_packed = textureLoad(normal_texture, pixel, 0);
    
    // Background check: our GBuffer depth channel stores linear view depth and clears to 0.0 for background.
    // Treat near-zero as no geometry and return AO=1.0 (no occlusion).
    if (depth <= 1e-6) {
        textureStore(output_ao, pixel, vec4<f32>(1.0));
        return;
    }
    
    // Depth texture stores linear depth (view-space), reconstruct full view position
    let view_pos = reconstruct_view_pos_linear(uv, depth);
    let view_normal = unpack_normal(normal_packed);
    
    var ao: f32;
    if (settings.technique == 0u) {
        ao = compute_ssao(pixel, uv, view_pos, view_normal);
    } else {
        ao = compute_gtao(pixel, uv, view_pos, view_normal);
    }
    
    textureStore(output_ao, pixel, vec4<f32>(ao));
}

// ============================================================================
// Bilateral blur (depth + normal aware)
// ============================================================================

@group(0) @binding(0) var input_ao: texture_2d<f32>;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
@group(0) @binding(2) var output_blurred: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> blur_settings: SsaoSettings;

fn bilateral_weight(center_depth: f32, sample_depth: f32) -> f32 {
    let depth_diff = abs(center_depth - sample_depth);
    let depth_sigma = 0.02;
    return exp(-depth_diff * depth_diff / (2.0 * depth_sigma * depth_sigma));
}

@compute @workgroup_size(8, 8, 1)
fn cs_ssao_blur(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    let dims = textureDimensions(input_ao);
    
    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }
    
    let center_depth = textureLoad(depth_tex, pixel, 0).r;
    let center_ao = textureLoad(input_ao, pixel, 0).r;
    
    var ao_sum = center_ao;
    var weight_sum = 1.0;
    
    let radius = 2i;
    
    for (var dy = -radius; dy <= radius; dy++) {
        for (var dx = -radius; dx <= radius; dx++) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            
            let sample_pixel = vec2<i32>(pixel) + vec2<i32>(dx, dy);
            if (sample_pixel.x < 0 || sample_pixel.x >= i32(dims.x) ||
                sample_pixel.y < 0 || sample_pixel.y >= i32(dims.y)) {
                continue;
            }
            
            let sample_depth = textureLoad(depth_tex, vec2<u32>(sample_pixel), 0).r;
            let sample_ao = textureLoad(input_ao, vec2<u32>(sample_pixel), 0).r;
            
            let weight = bilateral_weight(center_depth, sample_depth);
            
            ao_sum += sample_ao * weight;
            weight_sum += weight;
        }
    }
    
    let blurred_ao = ao_sum / weight_sum;
    textureStore(output_blurred, pixel, vec4<f32>(blurred_ao));
}

// ============================================================================
// Composite: diffuse-only AO application
// ============================================================================

@group(0) @binding(0) var color_input: texture_2d<f32>;
@group(0) @binding(1) var composite_output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var ao_input: texture_2d<f32>;
@group(0) @binding(3) var<uniform> composite_params: vec4<f32>;  // x=ao_intensity

@compute @workgroup_size(8, 8, 1)
fn cs_ssao_composite(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    let dims = textureDimensions(color_input);
    
    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }
    
    let color = textureLoad(color_input, pixel, 0);
    let ao = textureLoad(ao_input, pixel, 0).r;
    
    let ao_intensity = composite_params.x;
    // Apply in linear: diffuse *= mix(1.0, ao, ao_intensity)
    let ao_factor = mix(1.0, ao, ao_intensity);
    let result = vec4<f32>(color.rgb * ao_factor, color.a);
    textureStore(composite_output, pixel, result);
}
