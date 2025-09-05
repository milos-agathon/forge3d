//! Tone mapping shaders for HDR to LDR conversion
//! 
//! Provides various tone mapping operators including Reinhard, ACES, 
//! Uncharted 2, and exposure-based tone mapping with gamma correction.

struct ToneMappingUniforms {
    exposure: f32,
    white_point: f32,
    gamma: f32,
    operator_index: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: ToneMappingUniforms;
@group(0) @binding(1) var hdr_texture: texture_2d<f32>;
@group(0) @binding(2) var hdr_sampler: sampler;

// Fullscreen triangle vertex shader
@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;
    
    // Generate fullscreen triangle without vertex buffer
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    
    output.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    output.uv = vec2<f32>(x, 1.0 - y);  // Flip Y for texture coordinates
    
    return output;
}

// Reinhard tone mapping: color / (color + 1)
fn reinhard_tonemap(color: vec3<f32>) -> vec3<f32> {
    return color / (color + vec3<f32>(1.0));
}

// Extended Reinhard with white point: color * (1 + color/white²) / (1 + color)
fn reinhard_extended_tonemap(color: vec3<f32>, white_point: f32) -> vec3<f32> {
    let white_sq = white_point * white_point;
    return color * (vec3<f32>(1.0) + color / white_sq) / (vec3<f32>(1.0) + color);
}

// ACES filmic tone mapping approximation
fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    
    return saturate((color * (color * a + b)) / (color * (color * c + d) + e));
}

// Uncharted 2 filmic tone mapping
fn uncharted2_tonemap_partial(x: vec3<f32>) -> vec3<f32> {
    let a = 0.15;
    let b = 0.50;
    let c = 0.10;
    let d = 0.20;
    let e = 0.02;
    let f = 0.30;
    
    return ((x * (x * a + c * b) + d * e) / (x * (x * a + b) + d * f)) - e / f;
}

fn uncharted2_tonemap(color: vec3<f32>, white_point: f32) -> vec3<f32> {
    let curr = uncharted2_tonemap_partial(color);
    let white_scale = 1.0 / uncharted2_tonemap_partial(vec3<f32>(white_point));
    return curr * white_scale;
}

// Simple exposure-based tone mapping
fn exposure_tonemap(color: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(1.0) - exp(-color);
}

// Apply gamma correction
fn apply_gamma(color: vec3<f32>, gamma: f32) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / gamma));
}

// Fragment shader for tone mapping
@fragment
fn fs_tonemap(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample HDR texture
    let hdr_color = textureSample(hdr_texture, hdr_sampler, input.uv);
    
    // Apply exposure
    let exposed_color = hdr_color.rgb * uniforms.exposure;
    
    // Apply tone mapping based on operator index
    var tone_mapped: vec3<f32>;
    
    if (uniforms.operator_index == 0u) {
        // Reinhard
        tone_mapped = reinhard_tonemap(exposed_color);
    } else if (uniforms.operator_index == 1u) {
        // Reinhard Extended
        tone_mapped = reinhard_extended_tonemap(exposed_color, uniforms.white_point);
    } else if (uniforms.operator_index == 2u) {
        // ACES
        tone_mapped = aces_tonemap(exposed_color);
    } else if (uniforms.operator_index == 3u) {
        // Uncharted 2
        tone_mapped = uncharted2_tonemap(exposed_color, uniforms.white_point);
    } else if (uniforms.operator_index == 4u) {
        // Exposure
        tone_mapped = exposure_tonemap(exposed_color);
    } else {
        // Default to Reinhard
        tone_mapped = reinhard_tonemap(exposed_color);
    }
    
    // Apply gamma correction
    let gamma_corrected = apply_gamma(tone_mapped, uniforms.gamma);
    
    return vec4<f32>(gamma_corrected, hdr_color.a);
}

// Debug fragment shader showing HDR values directly (without tone mapping)
@fragment
fn fs_hdr_debug(input: VertexOutput) -> @location(0) vec4<f32> {
    let hdr_color = textureSample(hdr_texture, hdr_sampler, input.uv);
    
    // Simple linear mapping for debug visualization
    let debug_color = hdr_color.rgb * uniforms.exposure * 0.1; // Scale down for visibility
    
    return vec4<f32>(saturate(debug_color), hdr_color.a);
}

// Luminance calculation fragment shader
@fragment  
fn fs_luminance(input: VertexOutput) -> @location(0) vec4<f32> {
    let hdr_color = textureSample(hdr_texture, hdr_sampler, input.uv);
    
    // Calculate luminance using standard weights
    let luminance = 0.299 * hdr_color.r + 0.587 * hdr_color.g + 0.114 * hdr_color.b;
    
    // Output grayscale luminance
    return vec4<f32>(vec3<f32>(luminance), 1.0);
}

// False color visualization for HDR values
@fragment
fn fs_false_color(input: VertexOutput) -> @location(0) vec4<f32> {
    let hdr_color = textureSample(hdr_texture, hdr_sampler, input.uv);
    let luminance = 0.299 * hdr_color.r + 0.587 * hdr_color.g + 0.114 * hdr_color.b;
    
    // False color mapping for different HDR ranges
    var false_color: vec3<f32>;
    
    if (luminance < 0.1) {
        // Very dark - blue
        false_color = vec3<f32>(0.0, 0.0, 1.0);
    } else if (luminance < 1.0) {
        // Normal range - green
        false_color = vec3<f32>(0.0, 1.0, 0.0);
    } else if (luminance < 4.0) {
        // Bright - yellow
        false_color = vec3<f32>(1.0, 1.0, 0.0);
    } else if (luminance < 16.0) {
        // Very bright - red
        false_color = vec3<f32>(1.0, 0.0, 0.0);
    } else {
        // Extremely bright - magenta
        false_color = vec3<f32>(1.0, 0.0, 1.0);
    }
    
    return vec4<f32>(false_color, 1.0);
}

// Histogram computation for HDR analysis (simplified)
@fragment
fn fs_histogram_bin(input: VertexOutput) -> @location(0) vec4<f32> {
    let hdr_color = textureSample(hdr_texture, hdr_sampler, input.uv);
    let luminance = 0.299 * hdr_color.r + 0.587 * hdr_color.g + 0.114 * hdr_color.b;
    
    // Map luminance to histogram bin (simplified)
    let log_lum = log2(max(luminance, 0.0001));
    let bin_index = (log_lum + 10.0) / 20.0; // Map [-10, 10] log range to [0, 1]
    
    // Output bin index as color for further processing
    return vec4<f32>(saturate(bin_index), 0.0, 0.0, 1.0);
}