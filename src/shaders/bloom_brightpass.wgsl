// Q5: Bloom bright-pass extraction
// Extracts pixels above brightness threshold for bloom effect

@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(2)
var<uniform> uniforms: BloomBrightPassUniforms;

struct BloomBrightPassUniforms {
    threshold: f32,
    softness: f32,
    _pad: vec2<f32>,
}

// Luminance calculation (Rec. 709)
fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Soft threshold function for smoother bloom transition
fn soft_threshold(luma: f32, threshold: f32, softness: f32) -> f32 {
    let knee = threshold * softness;
    if luma < threshold - knee {
        return 0.0;
    } else if luma < threshold + knee {
        let t = (luma - threshold + knee) / (2.0 * knee);
        return t * t;
    } else {
        return 1.0;
    }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(input_texture);
    let coord = global_id.xy;
    
    if coord.x >= dimensions.x || coord.y >= dimensions.y {
        return;
    }
    
    // Sample input color
    let input_color = textureLoad(input_texture, coord, 0);
    let rgb = input_color.rgb;
    
    // Calculate luminance
    let luma = luminance(rgb);
    
    // Apply soft threshold
    let brightness_factor = soft_threshold(luma, uniforms.threshold, uniforms.softness);
    
    // Extract bright regions
    let bright_color = rgb * brightness_factor;
    
    // Store result with alpha preserved
    textureStore(output_texture, coord, vec4<f32>(bright_color, input_color.a));
}