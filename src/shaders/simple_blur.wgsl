// Simple blur post-processing effect
// Q1: Post-processing compute pipeline

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var input_sampler: sampler;

const BLUR_RADIUS: i32 = 2;
const WORKGROUP_SIZE: u32 = 16;

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(input_texture);
    let coord = global_id.xy;
    
    // Early exit for out-of-bounds pixels
    if (coord.x >= dimensions.x || coord.y >= dimensions.y) {
        return;
    }
    
    let texel_size = 1.0 / vec2<f32>(dimensions);
    let center_uv = (vec2<f32>(coord) + 0.5) * texel_size;
    
    var color_sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    
    // Simple box blur
    for (var y = -BLUR_RADIUS; y <= BLUR_RADIUS; y++) {
        for (var x = -BLUR_RADIUS; x <= BLUR_RADIUS; x++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            let sample_uv = center_uv + offset;
            
            // Sample with clamp-to-edge addressing
            let sample_color = textureSampleLevel(input_texture, input_sampler, sample_uv, 0.0);
            
            // Simple uniform weighting
            let weight = 1.0;
            color_sum += sample_color.rgb * weight;
            weight_sum += weight;
        }
    }
    
    // Normalize and write output
    let final_color = color_sum / weight_sum;
    textureStore(output_texture, coord, vec4<f32>(final_color, 1.0));
}