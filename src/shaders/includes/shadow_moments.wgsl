// Moment-shadow visibility: 1.0 = lit, 0.0 = shadowed.
fn chebyshev_upper_bound_visibility(mean: f32, variance: f32, receiver: f32) -> f32 {
    if (receiver <= mean) {
        return 1.0;
    }
    let delta = receiver - mean;
    return variance / (variance + delta * delta);
}

fn evsm_visibility_from_moments(
    moments: vec4<f32>,
    positive_receiver: f32,
    negative_receiver: f32,
    variance_floor: f32
) -> f32 {
    let positive_variance =
        max(moments.g - moments.r * moments.r, variance_floor);
    let negative_variance =
        max(moments.a - moments.b * moments.b, variance_floor);
    let positive_visibility =
        chebyshev_upper_bound_visibility(moments.r, positive_variance, positive_receiver);
    let negative_visibility =
        chebyshev_upper_bound_visibility(moments.b, negative_variance, negative_receiver);
    return min(positive_visibility, negative_visibility);
}
