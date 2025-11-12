// Temporal accumulation for AO (P5.1)
// Reprojection with neighborhood clamping to reduce ghosting

struct TemporalParams {
    temporal_alpha: f32,  // Blend factor [0=no history, 1=full history]
    _pad: vec3<f32>,
}

@group(0) @binding(0) var current_ao: texture_2d<f32>;
@group(0) @binding(1) var history_ao: texture_2d<f32>;
@group(0) @binding(2) var output_ao: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> params: TemporalParams;

// Compute neighborhood min/max for clamping
fn compute_neighborhood_clamp(pixel: vec2<u32>) -> vec2<f32> {
    let dims = textureDimensions(current_ao);
    
    var min_ao = 1.0;
    var max_ao = 0.0;
    
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let sample_pixel = vec2<i32>(pixel) + vec2<i32>(dx, dy);
            if (sample_pixel.x < 0 || sample_pixel.x >= i32(dims.x) ||
                sample_pixel.y < 0 || sample_pixel.y >= i32(dims.y)) {
                continue;
            }
            
            let ao = textureLoad(current_ao, vec2<u32>(sample_pixel), 0).r;
            min_ao = min(min_ao, ao);
            max_ao = max(max_ao, ao);
        }
    }
    
    return vec2<f32>(min_ao, max_ao);
}

@compute @workgroup_size(8, 8, 1)
fn cs_resolve_temporal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    let dims = textureDimensions(current_ao);
    
    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }
    
    let current = textureLoad(current_ao, pixel, 0).r;
    
    // Simple temporal without motion vectors: load history at same pixel
    let history = textureLoad(history_ao, vec2<u32>(pixel), 0).r;
    
    // Neighborhood clamping to reduce ghosting
    let clamp_range = compute_neighborhood_clamp(pixel);
    let clamped_history = clamp(history, clamp_range.x, clamp_range.y);
    
    // Exponential blend
    let alpha = params.temporal_alpha;
    let result = mix(current, clamped_history, alpha);
    
    textureStore(output_ao, pixel, vec4<f32>(result));
}

// Fallback: simple box temporal without motion vectors
@compute @workgroup_size(8, 8, 1)
fn cs_resolve_box_temporal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    let dims = textureDimensions(current_ao);
    
    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }
    
    let current = textureLoad(current_ao, pixel, 0).r;
    let history = textureLoad(history_ao, pixel, 0).r;
    
    let alpha = params.temporal_alpha;
    let result = mix(current, history, alpha);
    
    textureStore(output_ao, pixel, vec4<f32>(result));
}
