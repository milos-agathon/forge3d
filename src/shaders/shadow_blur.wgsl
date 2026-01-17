// shaders/shadow_blur.wgsl
// P0.2/M3: Separable Gaussian blur for VSM/EVSM/MSM moment maps
// Applied after moment generation to produce soft shadow edges

@group(0) @binding(0) var input_texture: texture_2d_array<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: BlurParams;

struct BlurParams {
    direction: vec2<f32>,   // (1,0) for horizontal, (0,1) for vertical
    kernel_radius: u32,     // Blur kernel radius (typically 2-4)
    cascade_count: u32,
    texture_size: u32,
    _padding: vec3<u32>,
}

// Gaussian weights for radius 4 (9-tap kernel)
// sigma = radius / 2 for good quality
const GAUSSIAN_WEIGHTS_9: array<f32, 5> = array<f32, 5>(
    0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216
);

// Gaussian weights for radius 2 (5-tap kernel)
const GAUSSIAN_WEIGHTS_5: array<f32, 3> = array<f32, 3>(
    0.375, 0.25, 0.0625
);

fn get_gaussian_weight(offset: u32, radius: u32) -> f32 {
    if (radius <= 2u) {
        if (offset < 3u) {
            return GAUSSIAN_WEIGHTS_5[offset];
        }
        return 0.0;
    } else {
        if (offset < 5u) {
            return GAUSSIAN_WEIGHTS_9[offset];
        }
        return 0.0;
    }
}

@compute @workgroup_size(8, 8, 1)
fn cs_blur(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = global_id.xy;
    let cascade_idx = global_id.z;
    
    // Bounds check
    if (coords.x >= params.texture_size || 
        coords.y >= params.texture_size ||
        cascade_idx >= params.cascade_count) {
        return;
    }
    
    let radius = params.kernel_radius;
    let dir = params.direction;
    
    // Accumulate weighted samples
    var result = vec4<f32>(0.0);
    var total_weight = 0.0;
    
    // Center sample
    let center_weight = get_gaussian_weight(0u, radius);
    result += textureLoad(input_texture, coords, cascade_idx, 0) * center_weight;
    total_weight += center_weight;
    
    // Symmetric samples
    for (var i = 1u; i <= radius; i++) {
        let weight = get_gaussian_weight(i, radius);
        let offset = vec2<i32>(i32(f32(i) * dir.x), i32(f32(i) * dir.y));
        
        // Positive offset
        let pos_coords = vec2<i32>(coords) + offset;
        if (pos_coords.x >= 0 && pos_coords.x < i32(params.texture_size) &&
            pos_coords.y >= 0 && pos_coords.y < i32(params.texture_size)) {
            result += textureLoad(input_texture, vec2<u32>(pos_coords), cascade_idx, 0) * weight;
            total_weight += weight;
        }
        
        // Negative offset
        let neg_coords = vec2<i32>(coords) - offset;
        if (neg_coords.x >= 0 && neg_coords.x < i32(params.texture_size) &&
            neg_coords.y >= 0 && neg_coords.y < i32(params.texture_size)) {
            result += textureLoad(input_texture, vec2<u32>(neg_coords), cascade_idx, 0) * weight;
            total_weight += weight;
        }
    }
    
    // Normalize and write output
    if (total_weight > 0.0) {
        result /= total_weight;
    }
    
    textureStore(output_texture, coords, cascade_idx, result);
}
