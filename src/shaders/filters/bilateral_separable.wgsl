// Bilateral separable filter for SSAO denoising (P5.1)
// Edge-aware blur using depth and normal information

struct BlurSettings {
    blur_radius: u32,
    depth_sigma: f32,
    normal_sigma: f32,
    _pad: u32,
}

// Horizontal pass
@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var depth_texture: texture_2d<f32>;
@group(0) @binding(2) var normal_texture: texture_2d<f32>;
@group(0) @binding(3) var output_texture: texture_storage_2d<r32float, write>;
@group(0) @binding(4) var<uniform> settings: BlurSettings;

fn unpack_normal(packed: vec4<f32>) -> vec3<f32> {
    // Our GBuffer stores view-space normals directly in [-1,1]
    return normalize(packed.xyz);
}

fn compute_weight(center_depth: f32, center_normal: vec3<f32>,
                  sample_depth: f32, sample_normal: vec3<f32>,
                  spatial_distance: f32) -> f32 {
    // Depth similarity
    let depth_diff = abs(center_depth - sample_depth);
    let depth_weight = exp(-depth_diff * depth_diff / (2.0 * settings.depth_sigma * settings.depth_sigma));
    
    // Normal similarity
    let normal_diff = 1.0 - max(0.0, dot(center_normal, sample_normal));
    let normal_weight = exp(-normal_diff * normal_diff / (2.0 * settings.normal_sigma * settings.normal_sigma));
    
    // Spatial Gaussian
    let spatial_weight = exp(-spatial_distance * spatial_distance / (2.0 * f32(settings.blur_radius) * f32(settings.blur_radius)));
    
    return depth_weight * normal_weight * spatial_weight;
}

@compute @workgroup_size(8, 8, 1)
fn cs_blur_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    let dims = textureDimensions(input_texture);
    
    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }
    
    let center_depth = textureLoad(depth_texture, pixel, 0).r;
    let center_normal = unpack_normal(textureLoad(normal_texture, pixel, 0));
    let center_value = textureLoad(input_texture, pixel, 0).r;
    
    var sum = center_value;
    var weight_sum = 1.0;
    
    let radius = i32(settings.blur_radius);
    
    for (var dx = -radius; dx <= radius; dx++) {
        if (dx == 0) {
            continue;
        }
        
        let sample_pixel = vec2<i32>(pixel) + vec2<i32>(dx, 0);
        if (sample_pixel.x < 0 || sample_pixel.x >= i32(dims.x)) {
            continue;
        }
        
        let sample_depth = textureLoad(depth_texture, vec2<u32>(sample_pixel), 0).r;
        let sample_normal = unpack_normal(textureLoad(normal_texture, vec2<u32>(sample_pixel), 0));
        let sample_value = textureLoad(input_texture, vec2<u32>(sample_pixel), 0).r;
        
        let weight = compute_weight(center_depth, center_normal, 
                                   sample_depth, sample_normal, 
                                   f32(abs(dx)));
        
        sum += sample_value * weight;
        weight_sum += weight;
    }
    
    let result = sum / weight_sum;
    textureStore(output_texture, pixel, vec4<f32>(result));
}

@compute @workgroup_size(8, 8, 1)
fn cs_blur_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    let dims = textureDimensions(input_texture);
    
    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }
    
    let center_depth = textureLoad(depth_texture, pixel, 0).r;
    let center_normal = unpack_normal(textureLoad(normal_texture, pixel, 0));
    let center_value = textureLoad(input_texture, pixel, 0).r;
    
    var sum = center_value;
    var weight_sum = 1.0;
    
    let radius = i32(settings.blur_radius);
    
    for (var dy = -radius; dy <= radius; dy++) {
        if (dy == 0) {
            continue;
        }
        
        let sample_pixel = vec2<i32>(pixel) + vec2<i32>(0, dy);
        if (sample_pixel.y < 0 || sample_pixel.y >= i32(dims.y)) {
            continue;
        }
        
        let sample_depth = textureLoad(depth_texture, vec2<u32>(sample_pixel), 0).r;
        let sample_normal = unpack_normal(textureLoad(normal_texture, vec2<u32>(sample_pixel), 0));
        let sample_value = textureLoad(input_texture, vec2<u32>(sample_pixel), 0).r;
        
        let weight = compute_weight(center_depth, center_normal, 
                                   sample_depth, sample_normal, 
                                   f32(abs(dy)));
        
        sum += sample_value * weight;
        weight_sum += weight;
    }
    
    let result = sum / weight_sum;
    textureStore(output_texture, pixel, vec4<f32>(result));
}
