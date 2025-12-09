// shaders/tone_map.wgsl
// Tone mapping curves for Workstream B PBR post-processing.
// Exists to share curve math between GPU pass and CPU reference.
// RELEVANT FILES:src/pipeline/pbr.rs,python/forge3d/pbr.py,tests/test_b2_tonemap.py,examples/pbr_spheres.py

const HABLE_WHITE_POINT : f32 = 11.2;

fn curve_reinhard(value: f32) -> f32 {
    return value / (1.0 + value);
}

fn curve_aces(value: f32) -> f32 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    let numerator = value * (a * value + b);
    let denominator = value * (c * value + d) + e;
    if (abs(denominator) < 1e-6) {
        return 0.0;
    }
    return clamp(numerator / denominator, 0.0, 1.0);
}

fn curve_hable(value: f32) -> f32 {
    let a = 0.15;
    let b = 0.50;
    let c = 0.10;
    let d = 0.20;
    let e = 0.02;
    let f = 0.30;
    let numerator = value * (value * a + c * b) + d * e;
    let denominator = value * (value * a + b) + d * f;
    if (abs(denominator) < 1e-6) {
        return 0.0;
    }
    let tone = numerator / denominator - e / f;
    let white_num = HABLE_WHITE_POINT * (HABLE_WHITE_POINT * a + c * b) + d * e;
    let white_den = HABLE_WHITE_POINT * (HABLE_WHITE_POINT * a + b) + d * f;
    let white = select(white_num / white_den - e / f, 1.0, abs(white_den) < 1e-6);
    if (abs(white) < 1e-6) {
        return 0.0;
    }
    return clamp(tone / white, 0.0, 1.0);
}

fn tone_map_unit(value: f32, mode: u32) -> f32 {
    switch(mode) {
        case 0u: { return curve_aces(value); }
        case 1u: { return curve_reinhard(value); }
        case 2u: { return curve_hable(value); }
        default: { return curve_reinhard(value); }
    }
}

fn tone_map_color(color: vec3<f32>, mode: u32, exposure: f32) -> vec3<f32> {
    let exposed = max(color * exposure, vec3<f32>(0.0));
    return vec3<f32>(
        tone_map_unit(exposed.x, mode),
        tone_map_unit(exposed.y, mode),
        tone_map_unit(exposed.z, mode)
    );
}

fn sample_unit_curve(mode: u32, value: f32) -> f32 {
    return tone_map_unit(max(value, 0.0), mode);
}

struct ToneMapUniforms {
    exposure : f32,
    mode : u32,
    padding : vec2<f32>,
};

@group(0) @binding(0) var<uniform> tone_map_uniforms : ToneMapUniforms;

fn tone_map_with_uniforms(color: vec3<f32>) -> vec3<f32> {
    return tone_map_color(color, tone_map_uniforms.mode, tone_map_uniforms.exposure);
}
