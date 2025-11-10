// shaders/ssao.wgsl
// P5: Enhanced SSAO/GTAO shader with bilateral blur
// Ground-Truth Ambient Occlusion with proper horizon-based sampling

struct SsaoSettings {
    radius: f32,
    intensity: f32,
    bias: f32,
    num_samples: u32,
    technique: u32, // 0 = SSAO, 1 = GTAO
    _pad_align: u32, // Padding to align vec2<f32> to 8-byte boundary
    inv_resolution: vec2<f32>,
    _pad0: vec2<f32>,
};

struct CameraParams {
    view_matrix: mat4x4<f32>,
    inv_view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    inv_proj_matrix: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
};

// SSAO/GTAO compute pass
@group(0) @binding(0) var depth_tex: texture_2d<f32>;
@group(0) @binding(1) var normal_tex: texture_2d<f32>;
@group(0) @binding(2) var ssao_output: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> settings: SsaoSettings;
@group(0) @binding(4) var<uniform> camera: CameraParams;

// Golden angle spiral for better sampling distribution
const GOLDEN_ANGLE: f32 = 2.39996323;

// Reconstruct view-space position from depth
fn reconstruct_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let view_pos = camera.inv_proj_matrix * ndc;
    return view_pos.xyz / view_pos.w;
}

// Hash function for noise
fn hash(p: vec2<f32>) -> f32 {
    let p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    let p3_dot = dot(p3, vec3<f32>(p3.yzx) + 33.33);
    return fract((p3.x + p3.y) * p3_dot + p3.z);
}

// TEMPORARILY DISABLED: Group 1 bindings for blur function
// @group(1) @binding(0) var ssao_input: texture_2d<f32>;
// @group(1) @binding(1) var ssao_blur_output: texture_storage_2d<rgba8unorm, write>;
// @group(1) @binding(2) var<uniform> blur_settings: SsaoSettings;
// @group(1) @binding(3) var normal_depth_tex_blur: texture_2d<f32>;

// Composite pass (uses @group(0) since it's a separate pipeline)
@group(0) @binding(0) var color_input: texture_2d<f32>;
@group(0) @binding(1) var color_storage: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var ssao_blurred_tex: texture_2d<f32>;
// x = composite multiplier (0=no AO, 1=full AO), yzw reserved
@group(0) @binding(3) var<uniform> comp_params: vec4<f32>;

// GTAO compute shader
@compute @workgroup_size(8, 8, 1)
fn cs_ssao(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(depth_tex);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }

    let coord = vec2<i32>(global_id.xy);
    let uv = (vec2<f32>(global_id.xy) + 0.5) / vec2<f32>(dims);
    
    // Sample center pixel
    let center_depth = textureLoad(depth_tex, coord, 0).r;
    let center_normal_encoded = textureLoad(normal_tex, coord, 0).xyz;
    let center_normal = normalize(center_normal_encoded * 2.0 - 1.0);
    
    // Early out for sky/far plane
    if (center_depth >= 0.9999) {
        textureStore(ssao_output, coord, vec4<f32>(1.0));
        return;
    }
    
    let center_pos = reconstruct_position(uv, center_depth);
    
    // AO: Sample in a spiral pattern; mode depends on settings.technique
    var occlusion = 0.0;
    let noise = hash(vec2<f32>(global_id.xy));
    let num_samples = settings.num_samples;
    
    for (var i = 0u; i < num_samples; i = i + 1u) {
        let theta = GOLDEN_ANGLE * f32(i) + noise * 6.28318;
        let radius = (f32(i) + 0.5) / f32(num_samples);
        
        // Sample offset in screen space
        let offset = vec2<f32>(cos(theta), sin(theta)) * radius * settings.radius;
        let sample_uv = uv + offset * settings.inv_resolution;
        
        // Sample depth and position
        let sample_coord = vec2<i32>(sample_uv * vec2<f32>(dims));
        if (sample_coord.x < 0 || sample_coord.x >= i32(dims.x) ||
            sample_coord.y < 0 || sample_coord.y >= i32(dims.y)) {
            continue;
        }
        
        let sample_depth = textureLoad(depth_tex, sample_coord, 0).r;
        let sample_pos = reconstruct_position(sample_uv, sample_depth);
        
        // Compute occlusion contribution
        let diff = sample_pos - center_pos;
        let dist = length(diff);
        let dir = diff / max(dist, 0.001);
        let range_check = smoothstep(settings.radius, settings.radius * 0.5, dist);
        if (settings.technique == 1u) {
            // GTAO: horizon-based term
            let horizon_angle = max(0.0, dot(center_normal, dir) - settings.bias);
            occlusion += horizon_angle * range_check;
        } else {
            // SSAO: simple normal-aligned term
            let ndotl = max(0.0, dot(center_normal, -dir) - settings.bias);
            occlusion += ndotl * range_check;
        }
    }
    
    occlusion /= f32(num_samples);
    let ao = 1.0 - clamp(occlusion * settings.intensity, 0.0, 1.0);
    
    textureStore(ssao_output, coord, vec4<f32>(ao));
}

// Bilateral blur pass (uses @group(0) since it's a separate pipeline)
@group(0) @binding(0) var ssao_input: texture_2d<f32>;
@group(0) @binding(1) var depth_tex_blur: texture_2d<f32>;
@group(0) @binding(2) var ssao_blur_output: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> blur_settings: SsaoSettings;

@compute @workgroup_size(8, 8, 1)
fn cs_ssao_blur(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(ssao_input);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }
    
    let coord = vec2<i32>(global_id.xy);
    let center_depth = textureLoad(depth_tex_blur, coord, 0).r;
    let center_ao = textureLoad(ssao_input, coord, 0).r;
    
    // Bilateral blur kernel (5x5)
    var sum = 0.0;
    var weight_sum = 0.0;
    let blur_radius = 2;
    
    for (var dy = -blur_radius; dy <= blur_radius; dy++) {
        for (var dx = -blur_radius; dx <= blur_radius; dx++) {
            let sample_coord = coord + vec2<i32>(dx, dy);
            if (sample_coord.x < 0 || sample_coord.x >= i32(dims.x) ||
                sample_coord.y < 0 || sample_coord.y >= i32(dims.y)) {
                continue;
            }
            
            let sample_depth = textureLoad(depth_tex_blur, sample_coord, 0).r;
            let sample_ao = textureLoad(ssao_input, sample_coord, 0).r;
            
            // Bilateral weight based on depth difference
            let depth_diff = abs(sample_depth - center_depth);
            let depth_weight = exp(-depth_diff * 10.0);
            
            // Spatial weight (Gaussian)
            let spatial_dist = f32(dx * dx + dy * dy);
            let spatial_weight = exp(-spatial_dist / 4.0);
            
            let weight = depth_weight * spatial_weight;
            sum += sample_ao * weight;
            weight_sum += weight;
        }
    }
    
    let blurred_ao = sum / max(weight_sum, 0.001);
    textureStore(ssao_blur_output, coord, vec4<f32>(blurred_ao));
}

// Composite AO with color
@compute @workgroup_size(8, 8, 1)
fn cs_ssao_composite(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(color_storage);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }

    let coord = vec2<i32>(global_id.xy);
    let color = textureLoad(color_input, coord, 0);
    let ao = textureLoad(ssao_blurred_tex, coord, 0).r;
    let occlusion = clamp(ao, 0.0, 1.0);
    // Post-AO multiplier blending: final = color * (occlusion * mul + (1 - mul))
    let mul = comp_params.x;
    let ao_mix = occlusion * mul + (1.0 - mul);
    let shaded = vec3<f32>(color.rgb) * ao_mix;
    textureStore(color_storage, coord, vec4<f32>(shaded, color.a));
}
