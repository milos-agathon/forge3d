//! GPU mipmap downsampling compute shader
//!
//! Downsamples a source texture to create the next mip level using box filtering.
//! Supports both linear and gamma-aware filtering modes.

struct MipmapUniforms {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    gamma_aware: u32,  // 0 = false, 1 = true
    gamma: f32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> uniforms: MipmapUniforms;
@group(0) @binding(1) var src_texture: texture_2d<f32>;
@group(0) @binding(2) var src_sampler: sampler;
@group(0) @binding(3) var dst_texture: texture_storage_2d<rgba32float, write>;

// sRGB <-> Linear conversion functions
fn srgb_to_linear(srgb: f32) -> f32 {
    if (srgb <= 0.04045) {
        return srgb / 12.92;
    } else {
        return pow((srgb + 0.055) / 1.055, 2.4);
    }
}

fn linear_to_srgb(linear: f32) -> f32 {
    if (linear <= 0.0031308) {
        return 12.92 * linear;
    } else {
        return 1.055 * pow(linear, 1.0 / 2.4) - 0.055;
    }
}

fn srgb_to_linear_vec3(srgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        srgb_to_linear(srgb.x),
        srgb_to_linear(srgb.y),
        srgb_to_linear(srgb.z)
    );
}

fn linear_to_srgb_vec3(linear: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        linear_to_srgb(linear.x),
        linear_to_srgb(linear.y),
        linear_to_srgb(linear.z)
    );
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dst_coord = global_id.xy;
    
    // Check if we're within bounds
    if (dst_coord.x >= uniforms.dst_width || dst_coord.y >= uniforms.dst_height) {
        return;
    }
    
    // Calculate the source region that maps to this destination pixel
    let x_ratio = f32(uniforms.src_width) / f32(uniforms.dst_width);
    let y_ratio = f32(uniforms.src_height) / f32(uniforms.dst_height);
    
    let src_x_start = u32(f32(dst_coord.x) * x_ratio);
    let src_y_start = u32(f32(dst_coord.y) * y_ratio);
    let src_x_end = min(u32(ceil(f32(dst_coord.x + 1u) * x_ratio)), uniforms.src_width);
    let src_y_end = min(u32(ceil(f32(dst_coord.y + 1u) * y_ratio)), uniforms.src_height);
    
    var rgba_sum = vec4<f32>(0.0);
    var sample_count = 0.0;
    
    // Box filter: accumulate samples in the source region
    for (var src_y = src_y_start; src_y < src_y_end; src_y++) {
        for (var src_x = src_x_start; src_x < src_x_end; src_x++) {
            let sample = textureLoad(src_texture, vec2<u32>(src_x, src_y), 0);
            
            var processed_sample = sample;
            
            // Apply gamma correction if enabled
            if (uniforms.gamma_aware != 0u) {
                // Convert RGB from sRGB to linear (leave alpha unchanged)
                processed_sample = vec4<f32>(
                    srgb_to_linear_vec3(sample.rgb),
                    sample.a
                );
            }
            
            rgba_sum += processed_sample;
            sample_count += 1.0;
        }
    }
    
    // Average the accumulated samples
    if (sample_count > 0.0) {
        rgba_sum /= sample_count;
        
        // Convert back from linear to sRGB if gamma correction was applied
        if (uniforms.gamma_aware != 0u) {
            rgba_sum = vec4<f32>(
                linear_to_srgb_vec3(rgba_sum.rgb),
                rgba_sum.a
            );
        }
    }
    
    // Store the downsampled result
    textureStore(dst_texture, dst_coord, rgba_sum);
}

// Alternative entry point for simple linear filtering (no gamma correction)
@compute @workgroup_size(8, 8, 1)  
fn cs_linear(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dst_coord = global_id.xy;
    
    if (dst_coord.x >= uniforms.dst_width || dst_coord.y >= uniforms.dst_height) {
        return;
    }
    
    // Simple 2x2 box filter using hardware filtering
    let src_coord = (vec2<f32>(dst_coord) + 0.5) * vec2<f32>(
        f32(uniforms.src_width) / f32(uniforms.dst_width),
        f32(uniforms.src_height) / f32(uniforms.dst_height)
    ) - 0.5;
    
    let result = textureSampleLevel(src_texture, src_sampler, src_coord / vec2<f32>(f32(uniforms.src_width), f32(uniforms.src_height)), 0.0);
    textureStore(dst_texture, dst_coord, result);
}

// Entry point for high-quality filtering with custom weights
@compute @workgroup_size(8, 8, 1)
fn cs_weighted(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dst_coord = global_id.xy;
    
    if (dst_coord.x >= uniforms.dst_width || dst_coord.y >= uniforms.dst_height) {
        return;
    }
    
    // Calculate fractional source coordinates
    let src_coord_f = (vec2<f32>(dst_coord) + 0.5) * vec2<f32>(
        f32(uniforms.src_width) / f32(uniforms.dst_width),
        f32(uniforms.src_height) / f32(uniforms.dst_height)
    ) - 0.5;
    
    let src_coord_base = vec2<i32>(floor(src_coord_f));
    let frac = fract(src_coord_f);
    
    // Bilinear interpolation weights
    let w00 = (1.0 - frac.x) * (1.0 - frac.y);
    let w10 = frac.x * (1.0 - frac.y);
    let w01 = (1.0 - frac.x) * frac.y;
    let w11 = frac.x * frac.y;
    
    var result = vec4<f32>(0.0);
    var total_weight = 0.0;
    
    // Sample the 2x2 neighborhood
    for (var dy = 0; dy < 2; dy++) {
        for (var dx = 0; dx < 2; dx++) {
            let sample_coord = src_coord_base + vec2<i32>(dx, dy);
            
            // Clamp to texture bounds
            let clamped_coord = vec2<u32>(
                clamp(sample_coord.x, 0, i32(uniforms.src_width - 1u)),
                clamp(sample_coord.y, 0, i32(uniforms.src_height - 1u))
            );
            
            let sample = textureLoad(src_texture, clamped_coord, 0);
            
            let weight = select(
                select(w00, w10, dx == 1),
                select(w01, w11, dx == 1),
                dy == 1
            );
            
            var processed_sample = sample;
            if (uniforms.gamma_aware != 0u) {
                processed_sample = vec4<f32>(
                    srgb_to_linear_vec3(sample.rgb),
                    sample.a
                );
            }
            
            result += processed_sample * weight;
            total_weight += weight;
        }
    }
    
    if (total_weight > 0.0) {
        result /= total_weight;
        
        if (uniforms.gamma_aware != 0u) {
            result = vec4<f32>(
                linear_to_srgb_vec3(result.rgb),
                result.a
            );
        }
    }
    
    textureStore(dst_texture, dst_coord, result);
}