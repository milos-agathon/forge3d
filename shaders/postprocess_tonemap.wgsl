//! Tonemap post-process shader: HDR linear → tonemap → sRGB
//! 
//! Full-screen triangle approach with exposure control for converting
//! HDR linear color space to sRGB output with tone mapping.

struct TonemapUniforms {
    exposure: f32,
    white_point: f32,
    gamma: f32,
    operator_index: u32, // 0=Reinhard, 1=ReinhardExtended, 2=ACES, 3=Uncharted2, 4=Exposure
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

// Reinhard tone mapping function
fn reinhard_tonemap(color: vec3<f32>) -> vec3<f32> {
    return color / (color + vec3<f32>(1.0));
}

// Extended Reinhard tone mapping with white point
fn reinhard_extended_tonemap(color: vec3<f32>, white_point: f32) -> vec3<f32> {
    let white_sq = white_point * white_point;
    return color * (vec3<f32>(1.0) + color / white_sq) / (vec3<f32>(1.0) + color);
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

// Uncharted 2 tone mapping
fn uncharted2_tonemap_partial(x: vec3<f32>) -> vec3<f32> {
    let a = 0.15;
    let b = 0.50;
    let c = 0.10;
    let d = 0.20;
    let e = 0.02;
    let f = 0.30;
    return ((x * (x * a + vec3<f32>(c * b)) + vec3<f32>(d * e)) / (x * (x * a + b) + vec3<f32>(d * f))) - vec3<f32>(e / f);
}

fn uncharted2_tonemap(color: vec3<f32>, white_point: f32) -> vec3<f32> {
    let curr = uncharted2_tonemap_partial(color);
    let white_scale = vec3<f32>(1.0) / uncharted2_tonemap_partial(vec3<f32>(white_point));
    return curr * white_scale;
}

// Exposure tone mapping
fn exposure_tonemap(color: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(1.0) - exp(-color);
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
    
    // Apply tone mapping based on operator selection
    var tonemapped_color: vec3<f32>;
    switch uniforms.operator_index {
        case 0u: { // Reinhard
            tonemapped_color = reinhard_tonemap(exposed_color);
        }
        case 1u: { // ReinhardExtended  
            tonemapped_color = reinhard_extended_tonemap(exposed_color, uniforms.white_point);
        }
        case 2u: { // ACES
            tonemapped_color = aces_tonemap(exposed_color);
        }
        case 3u: { // Uncharted2
            tonemapped_color = uncharted2_tonemap(exposed_color, uniforms.white_point);
        }
        case 4u: { // Exposure
            tonemapped_color = exposure_tonemap(exposed_color);
        }
        default: { // Default to Reinhard
            tonemapped_color = reinhard_tonemap(exposed_color);
        }
    }
    
    // Apply gamma correction (output linear→gamma corrected)
    let gamma_corrected = pow(clamp(tonemapped_color, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / uniforms.gamma));
    
    return vec4<f32>(gamma_corrected, 1.0);
}