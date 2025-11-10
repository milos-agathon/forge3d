// shaders/hzb_build.wgsl
// P5.0: Build a hierarchical Z (min-depth) pyramid from a depth attachment.
// Two entry points with separate bind sets:
//  - group(0): cs_copy samples the real depth texture with a sampler and writes HZB level 0 (r32float)
//  - group(1): cs_downsample min-reduces from HZB level N to level N+1 (both r32float)

// Copy from depth attachment -> HZB level 0
@group(0) @binding(0) var depth_tex : texture_depth_2d;
@group(0) @binding(1) var dst_lvl0  : texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_copy(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(depth_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let d = textureLoad(depth_tex, vec2<i32>(gid.xy), 0);
    textureStore(dst_lvl0, vec2<i32>(gid.xy), vec4<f32>(d, 0.0, 0.0, 1.0));
}

// Downsample HZB level N -> N+1 (min/max over 2x2 depending on depth convention)
struct HzbParams {
    reversed_z: u32,  // 1 = reversed-Z (use max), 0 = regular-Z (use min)
};
@group(0) @binding(0) var prev_lvl : texture_2d<f32>;
@group(0) @binding(1) var next_lvl : texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: HzbParams;

@compute @workgroup_size(8, 8, 1)
fn cs_downsample(@builtin(global_invocation_id) gid : vec3<u32>) {
    let src_dims = textureDimensions(prev_lvl);
    let dst_w = max(1u, (src_dims.x + 1u) / 2u);
    let dst_h = max(1u, (src_dims.y + 1u) / 2u);
    if (gid.x >= dst_w || gid.y >= dst_h) { return; }

    let base_x = i32(gid.x * 2u);
    let base_y = i32(gid.y * 2u);

    // Initialize to far plane for min, near plane for max
    var m = select(1e9, 0.0, params.reversed_z != 0u);
    for (var dy : i32 = 0; dy < 2; dy = dy + 1) {
        for (var dx : i32 = 0; dx < 2; dx = dx + 1) {
            let sx = base_x + dx;
            let sy = base_y + dy;
            if (sx < i32(src_dims.x) && sy < i32(src_dims.y)) {
                let v = textureLoad(prev_lvl, vec2<i32>(sx, sy), 0).r;
                // Reversed-Z: use max (farther geometry), Regular-Z: use min (nearer geometry)
                m = select(min(m, v), max(m, v), params.reversed_z != 0u);
            }
        }
    }
    textureStore(next_lvl, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(m, 0.0, 0.0, 1.0));
}
