//! Tonemap post-process shader: HDR linear → tonemap → sRGB
//! 
//! Full-screen triangle approach with exposure control for converting
//! HDR linear color space to sRGB output with tone mapping.
//! M6: Extended with 3D LUT support and white balance (temperature/tint).

struct TonemapUniforms {
    exposure: f32,
    white_point: f32,
    gamma: f32,
    operator_index: u32, // 0=Reinhard, 1=ReinhardExtended, 2=ACES, 3=Uncharted2, 4=Exposure
    // M6: LUT and white balance parameters
    lut_enabled: u32,        // 0=disabled, 1=enabled
    lut_strength: f32,       // LUT blend strength 0-1
    lut_size: f32,           // LUT dimension (e.g., 32 for 32x32x32)
    white_balance_enabled: u32, // 0=disabled, 1=enabled
    temperature: f32,        // Color temperature in Kelvin (2000-12000)
    tint: f32,               // Green-magenta tint (-1 to 1)
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var hdr_texture: texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: TonemapUniforms;
// M6: 3D LUT texture for color grading
@group(0) @binding(3) var lut_texture: texture_3d<f32>;
@group(0) @binding(4) var lut_sampler: sampler;

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

// M6: White balance adjustment using temperature and tint
// Based on Bradford chromatic adaptation
fn apply_white_balance(color: vec3<f32>, temperature: f32, tint: f32) -> vec3<f32> {
    // Convert temperature to approximate RGB multipliers
    // Based on Kelvin to RGB approximation (simplified Planckian locus)
    let temp_normalized = (temperature - 6500.0) / 5500.0; // Normalize around D65
    
    // Temperature: negative = warmer (more red), positive = cooler (more blue)
    var r_mult = 1.0;
    var b_mult = 1.0;
    if (temp_normalized < 0.0) {
        // Warmer: boost red, reduce blue
        r_mult = 1.0 - temp_normalized * 0.3;
        b_mult = 1.0 + temp_normalized * 0.3;
    } else {
        // Cooler: reduce red, boost blue
        r_mult = 1.0 - temp_normalized * 0.3;
        b_mult = 1.0 + temp_normalized * 0.3;
    }
    
    // Tint: negative = more green, positive = more magenta
    var g_mult = 1.0 - tint * 0.2;
    
    return color * vec3<f32>(r_mult, g_mult, b_mult);
}

// M6: Sample 3D LUT with trilinear interpolation
fn sample_lut(color: vec3<f32>, lut_size: f32) -> vec3<f32> {
    // Clamp input to valid range
    let clamped = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    
    // Scale to LUT coordinates (0.5/size to 1-0.5/size for proper texel centers)
    let half_texel = 0.5 / lut_size;
    let scale = (lut_size - 1.0) / lut_size;
    let lut_coord = clamped * scale + half_texel;
    
    // Sample with trilinear filtering
    return textureSampleLevel(lut_texture, lut_sampler, lut_coord, 0.0).rgb;
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
    var color = textureSample(hdr_texture, hdr_sampler, input.uv).rgb;
    
    // M6: Apply white balance before exposure (in linear space)
    if (uniforms.white_balance_enabled > 0u) {
        color = apply_white_balance(color, uniforms.temperature, uniforms.tint);
    }
    
    // Apply exposure
    let exposed_color = color * uniforms.exposure;
    
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
    
    // M6: Apply 3D LUT after tonemapping (before gamma)
    if (uniforms.lut_enabled > 0u && uniforms.lut_size > 0.0) {
        let lut_color = sample_lut(tonemapped_color, uniforms.lut_size);
        tonemapped_color = mix(tonemapped_color, lut_color, uniforms.lut_strength);
    }
    
    // Apply gamma correction (output linear→gamma corrected)
    let gamma_corrected = pow(clamp(tonemapped_color, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / uniforms.gamma));
    
    return vec4<f32>(gamma_corrected, 1.0);
}