// Shared tone-mapping operators for HDR linear Rec.709 RGB.
// Loaded by Rust shader assembly; raw WGSL #include is not a preprocessor.
//
// TERRA-DETERMINATA: this file is the LAST shared terrain math before
// write-out, so a single ULP of divergence here poisons the output hash.
// Every polynomial is spelled out in pinned left-to-right mul/add order (no
// contraction freedom) and every pow/exp transcendental routes through the
// det_* helpers in includes/determinism.wgsl, which the Rust shader assembly
// concatenates ahead of this file at every include site.

const TONEMAP_OPERATOR_REINHARD: u32 = 0u;
const TONEMAP_OPERATOR_REINHARD_EXTENDED: u32 = 1u;
const TONEMAP_OPERATOR_ACES: u32 = 2u;
const TONEMAP_OPERATOR_UNCHARTED2: u32 = 3u;
const TONEMAP_OPERATOR_EXPOSURE: u32 = 4u;
const TONEMAP_OPERATOR_FILMIC_TERRAIN: u32 = 5u;

fn tonemap_reinhard(color: vec3<f32>) -> vec3<f32> {
    let denom = vec3<f32>(1.0) + color;
    return color / denom;
}

fn tonemap_reinhard_extended(color: vec3<f32>, white_point: f32) -> vec3<f32> {
    let white_sq = max(white_point * white_point, 1.0e-6);
    let ratio = color / white_sq;
    let num_inner = vec3<f32>(1.0) + ratio;
    let num = color * num_inner;
    let denom = vec3<f32>(1.0) + color;
    return num / denom;
}

fn tonemap_aces(color: vec3<f32>) -> vec3<f32> {
    let clipped = max(color, vec3<f32>(0.0));
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    // Pinned Horner steps: x*(a*x + b) / (x*(c*x + d) + e)
    let num_inner = det_fma3(clipped, vec3<f32>(a), vec3<f32>(b));
    let num = clipped * num_inner;
    let den_inner = det_fma3(clipped, vec3<f32>(c), vec3<f32>(d));
    let den = det_fma3(clipped, den_inner, vec3<f32>(e));
    return clamp(num / den, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn tonemap_uncharted2_partial(x: vec3<f32>) -> vec3<f32> {
    let a = 0.15;
    let b = 0.50;
    let c = 0.10;
    let d = 0.20;
    let e = 0.02;
    let f = 0.30;
    // Pinned: ((x*(a*x + c*b) + d*e) / (x*(a*x + b) + d*f)) - e/f
    let ax = x * a;
    let num_inner = ax + vec3<f32>(c * b);
    let num = det_fma3(x, num_inner, vec3<f32>(d * e));
    let den_inner = ax + vec3<f32>(b);
    let den = det_fma3(x, den_inner, vec3<f32>(d * f));
    let ratio = num / den;
    return ratio - vec3<f32>(e / f);
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
    let attenuation = det_exp3(-max(color, vec3<f32>(0.0)));
    return vec3<f32>(1.0) - attenuation;
}

// Highest-priority determinism site: shared terrain resolve operator.
// Hable-style curve with every mul/add step pinned; the white-point
// normalization is scalar constant math but pinned the same way so all
// vendors fold it identically.
fn tonemap_filmic_terrain(color: vec3<f32>) -> vec3<f32> {
    let A = 0.22;
    let B = 0.30;
    let C = 0.10;
    let D = 0.20;
    let E = 0.01;
    let F = 0.30;
    let W = 11.2;
    let x = max(color, vec3<f32>(0.0));
    // curve = ((x*(A*x + C*B) + D*E) / (x*(A*x + B) + D*F)) - E/F
    let ax = x * A;
    let num_inner = ax + vec3<f32>(C * B);
    let num = det_fma3(x, num_inner, vec3<f32>(D * E));
    let den_inner = ax + vec3<f32>(B);
    let den = det_fma3(x, den_inner, vec3<f32>(D * F));
    let ratio = num / den;
    let curve = ratio - vec3<f32>(E / F);
    // white_curve = ((W*(A*W + C*B) + D*E) / (W*(A*W + B) + D*F)) - E/F
    let aw = A * W;
    let wnum_inner = aw + C * B;
    let wnum_prod = W * wnum_inner;
    let wnum = wnum_prod + D * E;
    let wden_inner = aw + B;
    let wden_prod = W * wden_inner;
    let wden = wden_prod + D * F;
    let wratio = wnum / wden;
    let white_curve = wratio - E / F;
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
    let clamped = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    let inv_gamma = 1.0 / max(gamma, 0.1);
    return det_pow3(clamped, vec3<f32>(inv_gamma));
}

fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    let clamped = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    let a = vec3<f32>(0.055);
    let lo = clamped * 12.92;
    // hi = (1 + a) * pow(clamped, 1/2.4) - a, pinned step by step
    let powed = det_pow3(clamped, vec3<f32>(1.0 / 2.4));
    let scaled = (vec3<f32>(1.0) + a) * powed;
    let hi = scaled - a;
    return select(hi, lo, clamped <= vec3<f32>(0.0031308));
}
