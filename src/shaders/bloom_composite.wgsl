// M2: Bloom composite pass
// Adds bloom (bright blur) to original image with configurable intensity

@group(0) @binding(0)
var original_texture: texture_2d<f32>;

@group(0) @binding(1)
var bloom_texture: texture_2d<f32>;

@group(0) @binding(2)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(3)
var<uniform> uniforms: BloomCompositeUniforms;

struct BloomCompositeUniforms {
    intensity: f32,
    _pad: vec3<f32>,
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(original_texture);
    let coord = global_id.xy;
    
    if coord.x >= dimensions.x || coord.y >= dimensions.y {
        return;
    }
    
    // Sample original and bloom textures
    let original = textureLoad(original_texture, coord, 0);
    let bloom = textureLoad(bloom_texture, coord, 0);
    
    // Additive blend: original + bloom * intensity
    let result = original.rgb + bloom.rgb * uniforms.intensity;
    
    // Preserve alpha from original
    textureStore(output_texture, coord, vec4<f32>(result, original.a));
}
