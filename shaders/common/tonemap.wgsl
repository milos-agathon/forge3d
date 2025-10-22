// Tone mapping functions for HDR to LDR conversion
// Provides ACES, Reinhard, and other tone mapping operators

// ============================================================================
// ACES Filmic Tone Mapping
// ============================================================================

// ACES approximation by Stephen Hill (@self_shadow)
// https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    
    let x = color;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// ============================================================================
// Reinhard Tone Mapping
// ============================================================================

// Simple Reinhard tone mapping
fn reinhard_tonemap(color: vec3<f32>) -> vec3<f32> {
    return color / (vec3<f32>(1.0) + color);
}

// Luminance-based Reinhard
fn reinhard_luminance_tonemap(color: vec3<f32>, white_point: f32) -> vec3<f32> {
    let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let numerator = luma * (1.0 + luma / (white_point * white_point));
    let denominator = 1.0 + luma;
    let scale = numerator / denominator / luma;
    return color * scale;
}

// ============================================================================
// Uncharted 2 Filmic Tone Mapping
// ============================================================================

fn uncharted2_tonemap_partial(x: vec3<f32>) -> vec3<f32> {
    let A = 0.15;
    let B = 0.50;
    let C = 0.10;
    let D = 0.20;
    let E = 0.02;
    let F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

fn uncharted2_tonemap(color: vec3<f32>) -> vec3<f32> {
    let exposure_bias = 2.0;
    let curr = uncharted2_tonemap_partial(color * exposure_bias);
    let W = vec3<f32>(11.2);
    let white_scale = vec3<f32>(1.0) / uncharted2_tonemap_partial(W);
    return curr * white_scale;
}

// ============================================================================
// Gamma Correction
// ============================================================================

// Apply gamma correction (typically 2.2 for sRGB displays)
fn gamma_correct(color: vec3<f32>, gamma: f32) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / gamma));
}

// Linear to sRGB (accurate sRGB transfer function)
fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    let cutoff = 0.0031308;
    let a = 0.055;
    
    var result: vec3<f32>;
    for (var i = 0; i < 3; i++) {
        if (color[i] <= cutoff) {
            result[i] = 12.92 * color[i];
        } else {
            result[i] = (1.0 + a) * pow(color[i], 1.0 / 2.4) - a;
        }
    }
    return result;
}

// sRGB to Linear
fn srgb_to_linear(color: vec3<f32>) -> vec3<f32> {
    let cutoff = 0.04045;
    let a = 0.055;
    
    var result: vec3<f32>;
    for (var i = 0; i < 3; i++) {
        if (color[i] <= cutoff) {
            result[i] = color[i] / 12.92;
        } else {
            result[i] = pow((color[i] + a) / (1.0 + a), 2.4);
        }
    }
    return result;
}

// ============================================================================
// Exposure and Contrast
// ============================================================================

// Apply exposure adjustment (stops)
fn apply_exposure(color: vec3<f32>, exposure: f32) -> vec3<f32> {
    return color * pow(2.0, exposure);
}

// Apply contrast adjustment
fn apply_contrast(color: vec3<f32>, contrast: f32) -> vec3<f32> {
    return (color - 0.5) * contrast + 0.5;
}

// ============================================================================
// Combined Tone Mapping Pipeline
// ============================================================================

// Apply full tone mapping pipeline
// tonemap_mode: 0=none, 1=aces, 2=reinhard, 3=uncharted2
fn apply_tonemap(
    color: vec3<f32>,
    tonemap_mode: u32,
    gamma: f32,
    exposure: f32,
) -> vec3<f32> {
    var result = color;
    
    // Apply exposure
    result = apply_exposure(result, exposure);
    
    // Apply tone mapping
    if (tonemap_mode == 1u) {
        result = aces_tonemap(result);
    } else if (tonemap_mode == 2u) {
        result = reinhard_tonemap(result);
    } else if (tonemap_mode == 3u) {
        result = uncharted2_tonemap(result);
    }
    // else: no tone mapping (mode 0)
    
    // Apply gamma correction
    result = gamma_correct(result, gamma);
    
    // Clamp to [0, 1]
    return clamp(result, vec3<f32>(0.0), vec3<f32>(1.0));
}
