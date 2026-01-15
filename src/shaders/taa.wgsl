// P1.3: Temporal Anti-Aliasing (TAA) resolve shader
//
// Uses reprojection with motion vectors (P1.1) and neighborhood clamping
// to blend current frame with history for anti-aliasing.

struct TaaSettings {
    resolution: vec2<f32>,
    jitter_offset: vec2<f32>,
    history_weight: f32,
    clamp_gamma: f32,
    motion_scale: f32,
    frame_index: u32,
};

@group(0) @binding(0) var current_color: texture_2d<f32>;
@group(0) @binding(1) var history_color: texture_2d<f32>;
@group(0) @binding(2) var velocity_tex: texture_2d<f32>;
@group(0) @binding(3) var depth_tex: texture_2d<f32>;
@group(0) @binding(4) var linear_sampler: sampler;
@group(0) @binding(5) var<uniform> settings: TaaSettings;
@group(0) @binding(6) var output_tex: texture_storage_2d<rgba16float, write>;

// Convert RGB to YCoCg color space for better clamping
fn rgb_to_ycocg(rgb: vec3<f32>) -> vec3<f32> {
    let y  = 0.25 * rgb.r + 0.5 * rgb.g + 0.25 * rgb.b;
    let co = 0.5 * rgb.r - 0.5 * rgb.b;
    let cg = -0.25 * rgb.r + 0.5 * rgb.g - 0.25 * rgb.b;
    return vec3<f32>(y, co, cg);
}

// Convert YCoCg back to RGB
fn ycocg_to_rgb(ycocg: vec3<f32>) -> vec3<f32> {
    let y = ycocg.x;
    let co = ycocg.y;
    let cg = ycocg.z;
    let r = y + co - cg;
    let g = y + cg;
    let b = y - co - cg;
    return vec3<f32>(r, g, b);
}

// Sample 3x3 neighborhood and compute min/max in YCoCg space
fn compute_neighborhood_aabb(uv: vec2<f32>, texel_size: vec2<f32>) -> array<vec3<f32>, 2> {
    var min_color = vec3<f32>(1e10);
    var max_color = vec3<f32>(-1e10);
    
    // Sample 3x3 neighborhood
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            let sample_uv = uv + offset;
            let rgb = textureSampleLevel(current_color, linear_sampler, sample_uv, 0.0).rgb;
            let ycocg = rgb_to_ycocg(rgb);
            min_color = min(min_color, ycocg);
            max_color = max(max_color, ycocg);
        }
    }
    
    return array<vec3<f32>, 2>(min_color, max_color);
}

// Variance-based neighborhood clamp (more robust than simple min/max)
fn compute_neighborhood_variance(uv: vec2<f32>, texel_size: vec2<f32>, gamma: f32) -> array<vec3<f32>, 2> {
    var m1 = vec3<f32>(0.0); // Mean
    var m2 = vec3<f32>(0.0); // Mean of squares
    
    // Sample 3x3 neighborhood
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            let sample_uv = uv + offset;
            let rgb = textureSampleLevel(current_color, linear_sampler, sample_uv, 0.0).rgb;
            let ycocg = rgb_to_ycocg(rgb);
            m1 += ycocg;
            m2 += ycocg * ycocg;
        }
    }
    
    m1 /= 9.0;
    m2 /= 9.0;
    
    let sigma = sqrt(max(m2 - m1 * m1, vec3<f32>(0.0)));
    let min_color = m1 - gamma * sigma;
    let max_color = m1 + gamma * sigma;
    
    return array<vec3<f32>, 2>(min_color, max_color);
}

// Clamp color to AABB
fn clip_aabb(aabb_min: vec3<f32>, aabb_max: vec3<f32>, color: vec3<f32>, center: vec3<f32>) -> vec3<f32> {
    let p_clip = 0.5 * (aabb_max + aabb_min);
    let e_clip = 0.5 * (aabb_max - aabb_min) + vec3<f32>(0.0001);
    
    let v_clip = color - p_clip;
    let v_unit = v_clip / e_clip;
    let a_unit = abs(v_unit);
    let ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));
    
    if (ma_unit > 1.0) {
        return p_clip + v_clip / ma_unit;
    } else {
        return color;
    }
}

// Unjitter UV to get the actual screen position
fn unjitter_uv(uv: vec2<f32>, jitter: vec2<f32>, resolution: vec2<f32>) -> vec2<f32> {
    // Jitter is in pixel units [-0.5, 0.5], convert to UV space
    let jitter_uv = jitter / resolution;
    return uv - jitter_uv;
}

@compute @workgroup_size(8, 8, 1)
fn taa_resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = vec2<i32>(gid.xy);
    let dims = vec2<i32>(textureDimensions(current_color));
    
    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }
    
    let texel_size = 1.0 / settings.resolution;
    let uv = (vec2<f32>(pixel) + 0.5) / settings.resolution;
    
    // Unjitter UV for current frame sampling
    let unjittered_uv = unjitter_uv(uv, settings.jitter_offset, settings.resolution);
    
    // Sample current color (unjittered)
    let current = textureSampleLevel(current_color, linear_sampler, unjittered_uv, 0.0);
    
    // Sample velocity for reprojection
    let velocity = textureLoad(velocity_tex, pixel, 0).rg;
    
    // Reproject to previous frame UV
    // Velocity is in screen space [-1, 1] scaled by 0.5, so multiply by 2
    let history_uv = uv - velocity * 2.0;
    
    // Check if history UV is valid (within screen bounds)
    let history_valid = history_uv.x >= 0.0 && history_uv.x <= 1.0 && 
                        history_uv.y >= 0.0 && history_uv.y <= 1.0;
    
    // Sample history color
    var history = textureSampleLevel(history_color, linear_sampler, history_uv, 0.0);
    
    // Compute neighborhood bounds using variance-based clamp (better quality)
    let aabb = compute_neighborhood_variance(unjittered_uv, texel_size, settings.clamp_gamma);
    let aabb_min = aabb[0];
    let aabb_max = aabb[1];
    
    // Convert colors to YCoCg for clamping
    let current_ycocg = rgb_to_ycocg(current.rgb);
    var history_ycocg = rgb_to_ycocg(history.rgb);
    
    // Clamp history to neighborhood bounds
    history_ycocg = clip_aabb(aabb_min, aabb_max, history_ycocg, current_ycocg);
    
    // Convert back to RGB
    let clamped_history = ycocg_to_rgb(history_ycocg);
    
    // Compute motion-based blend factor adjustment
    let motion_length = length(velocity) * settings.motion_scale;
    let motion_factor = saturate(1.0 - motion_length);
    
    // Adjust history weight based on motion and validity
    var blend_weight = settings.history_weight;
    if (!history_valid) {
        blend_weight = 0.0; // No history available, use current only
    } else {
        // Reduce history weight with motion
        blend_weight *= motion_factor;
    }
    
    // Depth-based rejection: if depth changed significantly, reduce history weight
    let current_depth = textureLoad(depth_tex, pixel, 0).r;
    // Note: more sophisticated depth rejection would require storing previous depth
    
    // Blend current and clamped history
    let result_rgb = mix(current.rgb, clamped_history, blend_weight);
    
    // Write output (also serves as history for next frame)
    textureStore(output_tex, pixel, vec4<f32>(result_rgb, current.a));
}
