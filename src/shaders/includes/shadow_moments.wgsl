// Moment-shadow visibility: 1.0 = lit, 0.0 = shadowed.
const MSM_FINITE_LIMIT: f32 = 1e20;
const MSM_MATRIX_EPSILON: f32 = 1e-7;
const MSM_DETERMINANT_EPSILON: f32 = 1e-12;
const EVSM_MINIMUM_VARIANCE: f32 = 0.000375;
const EVSM_MAX_EXPONENT_RGBA16F: f32 = 9.0;
// Maximum normalized-depth span assigned to a mixed-distribution penumbra.
const EVSM_MAX_DEPTH_TRANSITION: f32 = 0.03;
// Span three sigmas, starting two sigmas toward the occluder from the mean.
const EVSM_EDGE_SIGMA_SCALE: f32 = 3.0;
const EVSM_OCCLUDER_SIGMA_SCALE: f32 = 2.0;

fn chebyshev_upper_bound_visibility(mean: f32, variance: f32, receiver: f32) -> f32 {
    if (receiver <= mean) {
        return 1.0;
    }
    let delta = receiver - mean;
    return variance / (variance + delta * delta);
}

fn evsm_minimum_variance(
    warped_receiver: vec2<f32>,
    exponents: vec2<f32>
) -> vec2<f32> {
    let depth_scale =
        EVSM_MINIMUM_VARIANCE * exponents * abs(warped_receiver);
    return depth_scale * depth_scale;
}

fn evsm_visibility_from_moments(
    moments: vec4<f32>,
    positive_receiver: f32,
    negative_receiver: f32,
    variance_floor: vec2<f32>
) -> f32 {
    let positive_variance =
        max(moments.g - moments.r * moments.r, variance_floor.x);
    let negative_variance =
        max(moments.a - moments.b * moments.b, variance_floor.y);
    let positive_visibility =
        chebyshev_upper_bound_visibility(moments.r, positive_variance, positive_receiver);
    let negative_visibility =
        chebyshev_upper_bound_visibility(moments.b, negative_variance, negative_receiver);
    return min(positive_visibility, negative_visibility);
}

fn evsm_moment_leak_control(
    positive_moments: vec2<f32>,
    positive_receiver: f32,
    positive_exponent: f32,
    minimum_variance: f32
) -> f32 {
    if (
        !(positive_exponent > 0.0)
        || !(positive_exponent <= EVSM_MAX_EXPONENT_RGBA16F)
        || positive_moments.x != positive_moments.x
        || positive_moments.y != positive_moments.y
        || positive_receiver != positive_receiver
    ) {
        return 1.0;
    }

    let minimum_warp = exp(-positive_exponent);
    let positive_mean = clamp(positive_moments.x, minimum_warp, 1.0);
    let receiver = clamp(positive_receiver, minimum_warp, 1.0);

    let mean_depth =
        1.0 + log(positive_mean) / positive_exponent;
    let receiver_depth =
        1.0 + log(receiver) / positive_exponent;
    let variance = max(
        positive_moments.y - positive_mean * positive_mean,
        minimum_variance
    );
    let warp_derivative =
        max(positive_exponent * positive_mean, 0.000001);
    let edge_depth_sigma =
        sqrt(max(variance - minimum_variance, 0.0)) / warp_derivative;
    let floor_depth_sigma = sqrt(minimum_variance) / warp_derivative;
    if (edge_depth_sigma <= floor_depth_sigma) {
        return select(0.0, 1.0, receiver_depth <= mean_depth);
    }
    let transition_width = min(
        EVSM_EDGE_SIGMA_SCALE * edge_depth_sigma,
        EVSM_MAX_DEPTH_TRANSITION
    );
    let support_offset = min(
        EVSM_OCCLUDER_SIGMA_SCALE * edge_depth_sigma,
        EVSM_MAX_DEPTH_TRANSITION
    );
    let occluder_depth = mean_depth - support_offset;
    if (receiver_depth <= occluder_depth) {
        return 1.0;
    }
    return 1.0 - smoothstep(
        0.0,
        transition_width,
        receiver_depth - occluder_depth
    );
}

fn msm_visibility_from_moments(
    moments: vec4<f32>,
    receiver_depth: f32,
    moment_bias: f32
) -> f32 {
    if (!all(abs(moments) <= vec4<f32>(MSM_FINITE_LIMIT)) ||
        !(abs(receiver_depth) <= MSM_FINITE_LIMIT) ||
        !(abs(moment_bias) <= MSM_FINITE_LIMIT)) {
        return 1.0;
    }
    let receiver = clamp(receiver_depth, 0.0, 1.0);
    let raw_moments = clamp(moments, vec4<f32>(0.0), vec4<f32>(1.0));
    if (receiver <= raw_moments.x) {
        return 1.0;
    }

    let variance_floor = max(moment_bias, 0.000001);
    let variance = max(
        raw_moments.y - raw_moments.x * raw_moments.x,
        variance_floor
    );
    let fallback = clamp(
        chebyshev_upper_bound_visibility(raw_moments.x, variance, receiver),
        0.0,
        1.0
    );

    // Hamburger 4MSM. A small bias towards a valid moment vector compensates
    // Rgba16Float quantization before the Hankel-system reconstruction.
    let quantization_bias = clamp(max(moment_bias, 0.00003), 0.0, 0.01);
    let b = mix(raw_moments, vec4<f32>(0.5), quantization_bias);
    let d22 = b.y - b.x * b.x;
    let l32_d22 = b.z - b.x * b.y;
    let d33_d22 =
        (b.w - b.y * b.y) * d22 - l32_d22 * l32_d22;
    if (!(d22 > MSM_MATRIX_EPSILON) ||
        !(d33_d22 > MSM_DETERMINANT_EPSILON)) {
        return fallback;
    }

    let l32 = l32_d22 / d22;
    var coefficients = vec3<f32>(1.0, receiver, receiver * receiver);
    coefficients.y -= b.x;
    coefficients.z -= b.y + l32 * coefficients.y;
    coefficients.y /= d22;
    coefficients.z *= d22 / d33_d22;
    coefficients.y -= l32 * coefficients.z;
    coefficients.x -= dot(coefficients.yz, b.xy);
    if (!(abs(coefficients.z) > MSM_MATRIX_EPSILON) ||
        !all(abs(coefficients) <= vec3<f32>(MSM_FINITE_LIMIT))) {
        return fallback;
    }

    let p = coefficients.y / coefficients.z;
    let q = coefficients.x / coefficients.z;
    let discriminant = p * p * 0.25 - q;
    if (!(discriminant >= 0.0) || !(discriminant <= MSM_FINITE_LIMIT)) {
        return fallback;
    }

    let root_radius = sqrt(discriminant);
    let root1 = -p * 0.5 - root_radius;
    let root2 = -p * 0.5 + root_radius;
    var branch = vec4<f32>(0.0);
    if (root2 < receiver) {
        branch = vec4<f32>(root1, receiver, 1.0, 1.0);
    } else if (root1 < receiver) {
        branch = vec4<f32>(receiver, root1, 0.0, 1.0);
    }
    if (branch.w == 0.0) {
        return 1.0;
    }

    let denominator = (root2 - branch.y) * (receiver - root1);
    if (!(abs(denominator) > MSM_MATRIX_EPSILON) ||
        !(abs(denominator) <= MSM_FINITE_LIMIT)) {
        return fallback;
    }
    let quotient =
        (branch.x * root2 - b.x * (branch.x + root2) + b.y) /
        denominator;
    let visibility = 1.0 - clamp(branch.z + branch.w * quotient, 0.0, 1.0);
    if (!(visibility >= 0.0) || !(visibility <= 1.0)) {
        return fallback;
    }
    return visibility;
}
