// shaders/ssgi.wgsl
// P5: Screen-Space Global Illumination
// Half-res ray marching in depth buffer with fallback to diffuse IBL

struct SsgiSettings {
    radius: f32,
    intensity: f32,
    num_steps: u32,
    step_size: f32,
    inv_resolution: vec2<f32>,
    temporal_alpha: f32,
    use_half_res: u32,
    // Upsample controls
    upsample_depth_sigma: f32,
    upsample_normal_sigma: f32,
    use_edge_aware: u32,
    _pad1: u32,
};

struct CameraParams {
    view_matrix: mat4x4<f32>,
    inv_view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    inv_proj_matrix: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
};

// SSGI compute pass
@group(0) @binding(0) var depth_tex: texture_2d<f32>;
@group(0) @binding(1) var normal_tex: texture_2d<f32>;
@group(0) @binding(2) var color_tex: texture_2d<f32>;
@group(0) @binding(3) var ssgi_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> settings: SsgiSettings;
@group(0) @binding(5) var<uniform> camera: CameraParams;
@group(0) @binding(6) var ibl_texture: texture_cube<f32>;
@group(0) @binding(7) var ibl_sampler: sampler;

// Reconstruct view-space position from depth
fn reconstruct_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let view_pos = camera.inv_proj_matrix * ndc;
    return view_pos.xyz / view_pos.w;
}

// Project view-space position to screen UV
fn project_to_screen(view_pos: vec3<f32>) -> vec3<f32> {
    let clip_pos = camera.proj_matrix * vec4<f32>(view_pos, 1.0);
    let ndc = clip_pos.xyz / clip_pos.w;
    let uv = ndc.xy * 0.5 + 0.5;
    return vec3<f32>(uv, ndc.z);
}

// Hash function for noise
fn hash(p: vec2<f32>) -> f32 {
    let p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    let p3_dot = dot(p3, vec3<f32>(p3.yzx) + 33.33);
    return fract((p3.x + p3.y) * p3_dot + p3.z);
}

// Generate hemisphere sample direction
fn cosine_hemisphere_sample(u1: f32, u2: f32, normal: vec3<f32>) -> vec3<f32> {
    let r = sqrt(u1);
    let theta = 2.0 * 3.14159265 * u2;
    let x = r * cos(theta);
    let y = r * sin(theta);
    let z = sqrt(max(0.0, 1.0 - u1));
    
    // Build tangent space
    let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.z) > 0.999);
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    
    return normalize(tangent * x + bitangent * y + normal * z);
}

// Ray march in screen space
fn screen_space_ray_march(origin: vec3<f32>, direction: vec3<f32>, dims: vec2<u32>) -> vec3<f32> {
    var ray_pos = origin;
    let step = direction * settings.step_size;
    
    for (var i = 0u; i < settings.num_steps; i = i + 1u) {
        ray_pos += step;
        
        // Project to screen
        let screen = project_to_screen(ray_pos);
        
        // Check if off-screen
        if (screen.x < 0.0 || screen.x > 1.0 || screen.y < 0.0 || screen.y > 1.0) {
            return vec3<f32>(0.0);
        }
        
        // Sample depth
        let sample_coord = vec2<i32>(screen.xy * vec2<f32>(dims));
        let sample_depth = textureLoad(depth_tex, sample_coord, 0).r;
        
        // Check for intersection
        if (screen.z > sample_depth) {
            // Hit something - sample color
            let hit_color = textureLoad(color_tex, sample_coord, 0).rgb;
            return hit_color;
        }
    }
    
    // No hit - sample IBL
    // Convert view-space direction to world-space using inverse view matrix
    let world_dir = (camera.inv_view_matrix * vec4<f32>(direction, 0.0)).xyz;
    return textureSampleLevel(ibl_texture, ibl_sampler, world_dir, 0.0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn cs_ssgi(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Full-resolution dimensions from the GBuffer depth
    let dims_full = textureDimensions(depth_tex);
    // Output dimensions (half or full)
    var dims_out = dims_full;
    if (settings.use_half_res != 0u) {
        dims_out = dims_full / vec2<u32>(2u);
    }
    if (global_id.x >= dims_out.x || global_id.y >= dims_out.y) {
        return;
    }

    let coord_out = vec2<i32>(global_id.xy);
    let uv = (vec2<f32>(global_id.xy) + 0.5) / vec2<f32>(dims_out);
    let coord_full = vec2<i32>(uv * vec2<f32>(dims_full));
    
    // Sample center pixel
    let center_depth = textureLoad(depth_tex, coord_full, 0).r;
    let center_normal_encoded = textureLoad(normal_tex, coord_full, 0).xyz;
    let center_normal = normalize(center_normal_encoded * 2.0 - 1.0);
    
    // Early out for sky/far plane
    if (center_depth >= 0.9999) {
        textureStore(ssgi_output, coord_out, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    
    let center_pos = reconstruct_position(uv, center_depth);
    
    // Sample indirect lighting
    var indirect_light = vec3<f32>(0.0);
    let num_samples = 4u; // Reduced samples for performance
    let noise = hash(vec2<f32>(global_id.xy));
    
    for (var i = 0u; i < num_samples; i = i + 1u) {
        let u1 = fract(f32(i) / f32(num_samples) + noise);
        let u2 = fract(f32(i) * 0.618034 + noise * 0.5);
        
        let sample_dir = cosine_hemisphere_sample(u1, u2, center_normal);
        let sample_color = screen_space_ray_march(center_pos, sample_dir, dims_full);
        
        indirect_light += sample_color;
    }
    
    indirect_light /= f32(num_samples);
    indirect_light *= settings.intensity;
    
    textureStore(ssgi_output, coord_out, vec4<f32>(indirect_light, 1.0));
}

// Temporal accumulation pass
@group(1) @binding(0) var ssgi_current: texture_2d<f32>;
@group(1) @binding(1) var ssgi_history: texture_2d<f32>;
@group(1) @binding(2) var ssgi_accumulated: texture_storage_2d<rgba16float, write>;
@group(1) @binding(3) var<uniform> temporal_settings: SsgiSettings;

@compute @workgroup_size(8, 8, 1)
fn cs_ssgi_temporal(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(ssgi_current);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }
    
    let coord = vec2<i32>(global_id.xy);
    let current = textureLoad(ssgi_current, coord, 0).rgb;
    let history = textureLoad(ssgi_history, coord, 0).rgb;
    
    // Temporal blend (alpha from settings)
    let alpha = clamp(temporal_settings.temporal_alpha, 0.0, 1.0);
    let accumulated = mix(history, current, alpha);
    
    textureStore(ssgi_accumulated, coord, vec4<f32>(accumulated, 1.0));
}

// Upsample pass: rescale SSGI filtered output to full resolution with simple edge-aware filtering
@group(2) @binding(0) var ssgi_src: texture_2d<f32>;
@group(2) @binding(1) var ssgi_upscaled: texture_storage_2d<rgba16float, write>;
@group(2) @binding(2) var up_sampler: sampler;
// Use full-resolution depth only to query dimensions (future: edge-aware upsampling)
@group(2) @binding(3) var depth_tex_full: texture_2d<f32>;
@group(2) @binding(4) var normal_tex_full: texture_2d<f32>;
@group(2) @binding(5) var<uniform> upsample_settings: SsgiSettings;

@compute @workgroup_size(8, 8, 1)
fn cs_ssgi_upsample(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims_full = textureDimensions(depth_tex_full);
    if (global_id.x >= dims_full.x || global_id.y >= dims_full.y) {
        return;
    }

    let uv = (vec2<f32>(global_id.xy) + 0.5) / vec2<f32>(dims_full);

    // Center depth/normal (full-res)
    let depth_center = textureLoad(depth_tex_full, vec2<i32>(global_id.xy), 0).r;
    var n_center = textureLoad(normal_tex_full, vec2<i32>(global_id.xy), 0).xyz * 2.0 - 1.0;
    n_center = normalize(n_center);

    if (upsample_settings.use_edge_aware == 0u) {
        // Simple bilinear upsample
        let col = textureSampleLevel(ssgi_src, up_sampler, uv, 0.0).rgb;
        textureStore(ssgi_upscaled, vec2<i32>(global_id.xy), vec4<f32>(col, 1.0));
        return;
    }

    // Source dims (half or full)
    let dims_src = textureDimensions(ssgi_src);
    let coord_src = uv * vec2<f32>(dims_src) - 0.5;
    let base = floor(coord_src);
    let frac = fract(coord_src);

    // 4-tap manual bilinear with bilateral weights
    var sum = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var dy = 0; dy <= 1; dy = dy + 1) {
        for (var dx = 0; dx <= 1; dx = dx + 1) {
            let offset = vec2<f32>(f32(dx), f32(dy));
            let w_bilinear = (select(1.0 - frac.x, frac.x, dx == 1) * select(1.0 - frac.y, frac.y, dy == 1));
            let uv_tap = (base + offset + 0.5) / vec2<f32>(dims_src);

            // Sample GI
            let c = textureSampleLevel(ssgi_src, up_sampler, uv_tap, 0.0).rgb;

            // Edge-aware weights from full-res depth/normal
            let p_tap = vec2<i32>(uv_tap * vec2<f32>(dims_full));
            let depth_tap = textureLoad(depth_tex_full, p_tap, 0).r;
            var n_tap = textureLoad(normal_tex_full, p_tap, 0).xyz * 2.0 - 1.0;
            n_tap = normalize(n_tap);

            let dz = abs(depth_tap - depth_center);
            let sigma = max(upsample_settings.upsample_depth_sigma, 1e-6);
            let w_depth = exp(-dz / sigma);
            let w_normal = pow(max(dot(n_center, n_tap), 0.0), upsample_settings.upsample_normal_sigma);
            let w = w_bilinear * w_depth * w_normal;
            sum += c * w;
            wsum += w;
        }
    }
    let col = select(sum / wsum, vec3<f32>(0.0), wsum <= 1e-5);
    textureStore(ssgi_upscaled, vec2<i32>(global_id.xy), vec4<f32>(col, 1.0));
}
