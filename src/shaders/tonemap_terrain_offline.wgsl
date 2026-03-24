struct OfflineTonemapUniforms {
    width: u32,
    height: u32,
    operator_index: u32,
    _pad0: u32,
    white_point: f32,
    gamma: f32,
    _pad1: vec2<f32>,
}

@group(0) @binding(0) var hdr_input: texture_2d<f32>;
@group(0) @binding(1) var ldr_output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> uniforms: OfflineTonemapUniforms;

fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    let a = vec3<f32>(0.055);
    let lo = color * 12.92;
    let hi = (1.0 + a) * pow(color, vec3<f32>(1.0 / 2.4)) - a;
    return select(hi, lo, color <= vec3<f32>(0.0031308));
}

fn reinhard_tonemap(color: vec3<f32>) -> vec3<f32> {
    return color / (vec3<f32>(1.0) + color);
}

fn reinhard_extended_tonemap(color: vec3<f32>, white_point: f32) -> vec3<f32> {
    let white_sq = white_point * white_point;
    return color * (vec3<f32>(1.0) + color / white_sq) / (vec3<f32>(1.0) + color);
}

fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn uncharted2_tonemap_partial(x: vec3<f32>) -> vec3<f32> {
    let a = 0.15;
    let b = 0.50;
    let c = 0.10;
    let d = 0.20;
    let e = 0.02;
    let f = 0.30;
    return ((x * (x * a + vec3<f32>(c * b)) + vec3<f32>(d * e)) /
        (x * (x * a + b) + vec3<f32>(d * f))) - vec3<f32>(e / f);
}

fn uncharted2_tonemap(color: vec3<f32>, white_point: f32) -> vec3<f32> {
    let curr = uncharted2_tonemap_partial(color);
    let white_scale = vec3<f32>(1.0) / uncharted2_tonemap_partial(vec3<f32>(white_point));
    return curr * white_scale;
}

fn exposure_tonemap(color: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(1.0) - exp(-color);
}

fn tonemap_filmic_terrain(color: vec3<f32>) -> vec3<f32> {
    let A = 0.22;
    let B = 0.30;
    let C = 0.10;
    let D = 0.20;
    let E = 0.01;
    let F = 0.30;
    let W = 11.2;
    let x = max(color, vec3<f32>(0.0));
    let curve = ((x * (A * x + vec3<f32>(C * B)) + vec3<f32>(D * E)) /
        (x * (A * x + vec3<f32>(B)) + vec3<f32>(D * F))) - vec3<f32>(E / F);
    let white_curve = ((W * (A * W + C * B) + D * E) / (W * (A * W + B) + D * F)) - E / F;
    return clamp(curve / white_curve, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn apply_operator(color: vec3<f32>) -> vec3<f32> {
    switch uniforms.operator_index {
        case 0u: {
            return reinhard_tonemap(color);
        }
        case 1u: {
            return reinhard_extended_tonemap(color, uniforms.white_point);
        }
        case 2u: {
            return aces_tonemap(color);
        }
        case 3u: {
            return uncharted2_tonemap(color, uniforms.white_point);
        }
        case 4u: {
            return exposure_tonemap(color);
        }
        default: {
            return tonemap_filmic_terrain(color);
        }
    }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) {
        return;
    }

    let coords = vec2<i32>(gid.xy);
    let hdr = textureLoad(hdr_input, coords, 0);
    let mapped = apply_operator(hdr.rgb);
    let encoded = linear_to_srgb(clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0)));
    textureStore(ldr_output, coords, vec4<f32>(encoded, clamp(hdr.a, 0.0, 1.0)));
}
