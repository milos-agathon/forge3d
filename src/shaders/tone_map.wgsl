// shaders/tone_map.wgsl
// Tone mapping curves for Workstream B PBR post-processing.
// Exists to share curve math between GPU pass and CPU reference.
// RELEVANT FILES:src/pipeline/pbr.rs,python/forge3d/pbr.py,tests/test_b2_tonemap.py,examples/pbr_spheres.py

fn tone_map_unit(value: f32, mode: u32) -> f32 {
    let sample = vec3<f32>(max(value, 0.0));
    switch(mode) {
        case TONEMAP_OPERATOR_ACES: { return tonemap_aces(sample).x; }
        case TONEMAP_OPERATOR_REINHARD: { return tonemap_reinhard(sample).x; }
        case TONEMAP_OPERATOR_UNCHARTED2: { return tonemap_uncharted2(sample, 11.2).x; }
        case TONEMAP_OPERATOR_FILMIC_TERRAIN: { return tonemap_filmic_terrain(sample).x; }
        default: { return tonemap_reinhard(sample).x; }
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
