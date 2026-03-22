// TV12: Quarter-resolution luminance extraction for convergence metrics.
//
// Reads the accumulated HDR buffer, divides by sample count, averages 4x4 pixel
// blocks, computes luminance, writes to a quarter-res R32Float output.
// The CPU side then reads this back to compute per-tile temporal convergence.

struct LuminanceParams {
    src_width: u32,
    src_height: u32,
    sample_count: u32,
    _pad: u32,
}

@group(0) @binding(0) var accumulated: texture_2d<f32>;
@group(0) @binding(1) var luminance_out: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: LuminanceParams;

@compute @workgroup_size(8, 8)
fn extract_luminance(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_w = (params.src_width + 3u) / 4u;
    let dst_h = (params.src_height + 3u) / 4u;
    if (gid.x >= dst_w || gid.y >= dst_h) {
        return;
    }

    let n = f32(max(params.sample_count, 1u));
    var sum_lum: f32 = 0.0;
    var count: f32 = 0.0;

    // Average over 4x4 source pixel block
    for (var dy: u32 = 0u; dy < 4u; dy++) {
        for (var dx: u32 = 0u; dx < 4u; dx++) {
            let sx = gid.x * 4u + dx;
            let sy = gid.y * 4u + dy;
            if (sx < params.src_width && sy < params.src_height) {
                let accum = textureLoad(accumulated, vec2<i32>(i32(sx), i32(sy)), 0);
                let avg = accum / n;
                // ITU-R BT.709 luminance
                sum_lum += 0.2126 * avg.r + 0.7152 * avg.g + 0.0722 * avg.b;
                count += 1.0;
            }
        }
    }

    let mean_lum = sum_lum / max(count, 1.0);
    textureStore(luminance_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(mean_lum, 0.0, 0.0, 0.0));
}
