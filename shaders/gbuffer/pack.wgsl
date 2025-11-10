// shaders/gbuffer/pack.wgsl
// P5.0: packing stubs for GBuffer attributes.
// For now, normals are stored directly as RGB in [0,1] using encode/decode helpers.
// Later milestones may switch to oct-encoding or spherical mapping.

// Re-export encode/decode as the current pack/unpack API
fn pack_normal(n_view: vec3<f32>) -> vec3<f32> {
    return encode_view_normal_rgb(normalize(n_view));
}

fn unpack_normal(rgb: vec3<f32>) -> vec3<f32> {
    return decode_view_normal_rgb(rgb);
}
