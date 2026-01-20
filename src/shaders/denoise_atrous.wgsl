
// src/shaders/denoise_atrous.wgsl
// Edge-avoiding A-Trous Wavelet Transform filter for denoising

struct DenoiseUniforms {
    width: f32,
    height: f32,
    step_width: f32,
    sigma_color: f32,
    sigma_normal: f32, // Unused if no normal texture
    sigma_depth: f32,  // Unused if no depth texture
    padding: vec2<f32>,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var depth_tex: texture_depth_2d; // Depth for edge stopping
@group(0) @binding(3) var<uniform> uniforms: DenoiseUniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = vec2<i32>(i32(uniforms.width), i32(uniforms.height));
    let coords = vec2<i32>(global_id.xy);

    if (coords.x >= dims.x || coords.y >= dims.y) {
        return;
    }

    let center_color = textureLoad(input_tex, coords, 0);
    // Determine depth at center
    let center_depth = textureLoad(depth_tex, coords, 0);

    var sum = vec4<f32>(0.0);
    var weight_sum = 0.0;
    
    let kernel = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);
    let step = i32(uniforms.step_width);
    
    // 5x5 loop
    for (var i = -2; i <= 2; i = i + 1) {
        for (var j = -2; j <= 2; j = j + 1) {
            let offset = vec2<i32>(i, j) * step;
            let sample_coords = coords + offset;
            
            if (sample_coords.x >= 0 && sample_coords.y >= 0 && 
                sample_coords.x < dims.x && sample_coords.y < dims.y) {
                
                let sample_color = textureLoad(input_tex, sample_coords, 0);
                let sample_depth = textureLoad(depth_tex, sample_coords, 0);
                
                // Color weight
                let color_dist = distance(center_color.rgb, sample_color.rgb);
                let w_color = exp(-(color_dist * color_dist) / (uniforms.sigma_color * uniforms.sigma_color + 1e-5));
                
                // Depth weight
                let d_diff = abs(center_depth - sample_depth);
                let w_depth = exp(-(d_diff * d_diff) / (uniforms.sigma_depth * uniforms.sigma_depth + 1e-5));

                // Kernel weight
                let w_kernel = kernel[i+2] * kernel[j+2];
                
                let w = w_kernel * w_color * w_depth;
                
                sum = sum + sample_color * w;
                weight_sum = weight_sum + w;
            }
        }
    }
    
    let final_color = sum / max(weight_sum, 1e-5);
    textureStore(output_tex, coords, final_color);
}
