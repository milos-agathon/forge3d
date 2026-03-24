struct OfflineResolveUniforms {
    width: u32,
    height: u32,
    sample_count: u32,
    renormalize_normals: u32,
}

@group(0) @binding(0) var accumulation: texture_2d<f32>;
@group(0) @binding(1) var resolved: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> uniforms: OfflineResolveUniforms;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) {
        return;
    }

    let coords = vec2<i32>(gid.xy);
    var value = textureLoad(accumulation, coords, 0) / f32(max(uniforms.sample_count, 1u));
    if (uniforms.renormalize_normals > 0u) {
        let normal = value.xyz;
        let len_sq = dot(normal, normal);
        if (len_sq > 1e-12) {
            value = vec4<f32>(normalize(normal), value.w);
        }
    }

    textureStore(resolved, coords, value);
}
