// Shared tone-mapping operators for HDR linear Rec.709 RGB.
// Loaded by Rust shader assembly; raw WGSL #include is not a preprocessor.

const TONEMAP_OPERATOR_REINHARD: u32 = 0u;
const TONEMAP_OPERATOR_REINHARD_EXTENDED: u32 = 1u;
const TONEMAP_OPERATOR_ACES: u32 = 2u;
const TONEMAP_OPERATOR_UNCHARTED2: u32 = 3u;
const TONEMAP_OPERATOR_EXPOSURE: u32 = 4u;
const TONEMAP_OPERATOR_FILMIC_TERRAIN: u32 = 5u;

fn tonemap_reinhard(color: vec3<f32>) -> vec3<f32> {
    return color / (vec3<f32>(1.0) + color);
}

fn tonemap_reinhard_extended(color: vec3<f32>, white_point: f32) -> vec3<f32> {
    let white_sq = max(white_point * white_point, 1.0e-6);
    return color * (vec3<f32>(1.0) + color / white_sq) / (vec3<f32>(1.0) + color);
}

fn tonemap_aces(color: vec3<f32>) -> vec3<f32> {
    let clipped = max(color, vec3<f32>(0.0));
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp(
        (clipped * (clipped * a + b)) / (clipped * (clipped * c + d) + e),
        vec3<f32>(0.0),
        vec3<f32>(1.0)
    );
}

fn tonemap_uncharted2_partial(x: vec3<f32>) -> vec3<f32> {
    let a = 0.15;
    let b = 0.50;
    let c = 0.10;
    let d = 0.20;
    let e = 0.02;
    let f = 0.30;
    return ((x * (x * a + vec3<f32>(c * b)) + vec3<f32>(d * e)) /
        (x * (x * a + b) + vec3<f32>(d * f))) - vec3<f32>(e / f);
}

fn tonemap_uncharted2(color: vec3<f32>, white_point: f32) -> vec3<f32> {
    let curr = tonemap_uncharted2_partial(max(color, vec3<f32>(0.0)));
    let white_scale = vec3<f32>(1.0) / max(
        tonemap_uncharted2_partial(vec3<f32>(max(white_point, 1.0e-3))),
        vec3<f32>(1.0e-6)
    );
    return clamp(curr * white_scale, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn tonemap_exposure(color: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(1.0) - exp(-max(color, vec3<f32>(0.0)));
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
    return clamp(curve / max(white_curve, 1.0e-6), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn tonemap_apply_operator(color: vec3<f32>, operator_index: u32, white_point: f32) -> vec3<f32> {
    switch operator_index {
        case TONEMAP_OPERATOR_REINHARD: {
            return tonemap_reinhard(color);
        }
        case TONEMAP_OPERATOR_REINHARD_EXTENDED: {
            return tonemap_reinhard_extended(color, white_point);
        }
        case TONEMAP_OPERATOR_ACES: {
            return tonemap_aces(color);
        }
        case TONEMAP_OPERATOR_UNCHARTED2: {
            return tonemap_uncharted2(color, white_point);
        }
        case TONEMAP_OPERATOR_EXPOSURE: {
            return tonemap_exposure(color);
        }
        case TONEMAP_OPERATOR_FILMIC_TERRAIN: {
            return tonemap_filmic_terrain(color);
        }
        default: {
            return tonemap_reinhard(color);
        }
    }
}

fn gamma_correct(color: vec3<f32>, gamma: f32) -> vec3<f32> {
    return pow(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / max(gamma, 0.1)));
}

fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    let clamped = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    let a = vec3<f32>(0.055);
    let lo = clamped * 12.92;
    let hi = (vec3<f32>(1.0) + a) * pow(clamped, vec3<f32>(1.0 / 2.4)) - a;
    return select(hi, lo, clamped <= vec3<f32>(0.0031308));
}
