//! Tonemap post-process shader: HDR linear → tonemap → sRGB
//! 
//! Full-screen triangle approach with exposure control for converting
//! HDR linear color space to sRGB output with tone mapping.

struct TonemapUniforms {
    exposure: f32,
    _pad0: f32,
    _pad1: f32, 
    _pad2: f32,
}

@group(0) @binding(0) var hdr_texture: texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: TonemapUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Vertex shader - generates full-screen triangle
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Full-screen triangle technique
    // Vertex 0: (-1, -1) -> UV (0, 1)
    // Vertex 1: (-1,  3) -> UV (0, -1) 
    // Vertex 2: ( 3, -1) -> UV (2, 1)
    let uv = vec2<f32>(
        f32((vertex_index << 1u) & 2u),
        f32(vertex_index & 2u)
    );
    let pos = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    
    return VertexOutput(
        vec4<f32>(pos.x, -pos.y, pos.z, pos.w), // Flip Y for correct orientation
        uv
    );
}

// ACES tone mapping function
fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Linear to sRGB conversion
fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    return select(
        pow(color, vec3<f32>(1.0 / 2.4)) * 1.055 - 0.055,
        color * 12.92,
        color <= vec3<f32>(0.0031308)
    );
}

// Fragment shader - tonemap and convert to sRGB
@fragment  
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample HDR input
    let hdr_color = textureSample(hdr_texture, hdr_sampler, input.uv).rgb;
    
    // Apply exposure
    let exposed_color = hdr_color * uniforms.exposure;
    
    // Apply ACES tone mapping
    let tonemapped_color = aces_tonemap(exposed_color);
    
    // Convert to sRGB
    let srgb_color = linear_to_srgb(tonemapped_color);
    
    return vec4<f32>(srgb_color, 1.0);
}