// TV12: Resolve accumulated HDR buffer by dividing by sample count.
//
// Two modes:
//   mode 0 (beauty/albedo): divide by N, write to Rgba16Float output.
//   mode 1 (normal): divide by N, then renormalize the vector.
//
// Input is Rgba32Float (full-precision accumulation sum).
// Output is Rgba16Float (sufficient for HDR handoff and OIDN).

struct ResolveParams {
    width: u32,
    height: u32,
    sample_count: u32,
    mode: u32,  // 0 = beauty/albedo (divide only), 1 = normal (divide + renormalize)
}

@group(0) @binding(0) var accumulated: texture_2d<f32>;
@group(0) @binding(1) var resolved: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: ResolveParams;

@compute @workgroup_size(8, 8)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let accum = textureLoad(accumulated, coords, 0);
    let n = f32(max(params.sample_count, 1u));
    var result = accum / n;

    if (params.mode == 1u) {
        // Normal: renormalize after averaging (averaged unit vectors lose magnitude)
        let len = length(result.xyz);
        if (len > 1e-6) {
            result = vec4<f32>(result.xyz / len, result.w);
        }
    }

    textureStore(resolved, coords, result);
}
