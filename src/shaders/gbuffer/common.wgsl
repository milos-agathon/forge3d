// shaders/gbuffer/common.wgsl
// P5.0: helper functions for GBuffer encode/decode and depth utilities.
// Keeps behavior identical for now; used by later milestones.

fn decode_view_normal_rgb(n: vec3<f32>) -> vec3<f32> {
    // Input encoded in [0,1]; map back to [-1,1] and renormalize for safety
    return normalize(n * 2.0 - vec3<f32>(1.0));
}

fn encode_view_normal_rgb(n: vec3<f32>) -> vec3<f32> {
    // Map [-1,1] to [0,1]
    return n * 0.5 + vec3<f32>(0.5);
}

// Linearize NDC depth (0..1) to view-space Z distance using near/far.
// Assumes standard perspective projection with depth in [0,1].
fn linearize_depth(ndc_z: f32, near: f32, far: f32) -> f32 {
    // Avoid division by zero if far==near
    let nf = max(1e-6, far - near);
    // Equivalent to: z_view = near * far / (far - ndc_z * (far - near))
    // This yields positive distances increasing with depth.
    return (near * far) / max(1e-6, (far - ndc_z * nf));
}
