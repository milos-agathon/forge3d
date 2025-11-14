// src/shaders/ssgi/trace.wgsl
// P5.2: Half-resolution SSGI view-space tracing using HZB

struct SsgiSettings {
    radius: f32,
    intensity: f32,
    num_steps: u32,
    step_size: f32,
    inv_resolution: vec2<f32>,
    temporal_alpha: f32,
    use_half_res: u32,
    upsample_depth_sigma: f32,
    upsample_normal_exp: f32,
    use_edge_aware: u32,
    _pad1: u32,
};

struct CameraParams {
    view_matrix: mat4x4<f32>,
    inv_view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    inv_proj_matrix: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
};

@group(0) @binding(0) var tDepth: texture_2d<f32>;
@group(0) @binding(1) var tNormal: texture_2d<f32>;
@group(0) @binding(2) var tHzb: texture_2d<f32>;
@group(0) @binding(3) var outHit: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> uSsgi: SsgiSettings;
@group(0) @binding(5) var<uniform> uCam: CameraParams;

fn unpack_normal(packed: vec4<f32>) -> vec3<f32> {
    return normalize(packed.xyz * 2.0 - 1.0);
}

fn project_to_screen(view_pos: vec3<f32>, proj: mat4x4<f32>) -> vec3<f32> {
    let clip = proj * vec4<f32>(view_pos, 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return vec3<f32>(uv, ndc.z);
}

// Simple hierarchical depth test using HZB base mip (can be extended to mip selection)
fn hzb_occluded(uv: vec2<f32>, ray_depth: f32) -> bool {
    // Clamp UV
    let uv_c = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let texel = uv_c * vec2<f32>(textureDimensions(tHzb));
    let d = textureLoad(tHzb, vec2<u32>(texel), 0).r;
    // Standard-Z assumption: stored depth ~ linear depth in same space as tDepth
    // Consider occlusion when ray is behind the HZB depth (greater)
    return ray_depth > d;
}

@compute @workgroup_size(8, 8, 1)
fn cs_trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Compute target output pixel in half/full resolution
    let full_dims = textureDimensions(tDepth);
    var out_xy = gid.xy;
    var pixel = gid.xy;
    if (uSsgi.use_half_res == 1u) {
        // Dispatch grid should be half dims; map to full-res texel grid by *2
        pixel = gid.xy * 2u;
    }

    if (pixel.x >= full_dims.x || pixel.y >= full_dims.y) {
        return;
    }

    let depth = textureLoad(tDepth, pixel, 0).r;
    // Skip sky/background
    if (depth <= 0.0) {
        textureStore(outHit, out_xy, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let packed_n = textureLoad(tNormal, pixel, 0);
    let n_vs = unpack_normal(packed_n);

    // View-space position reconstruction from depth and NDC
    let uv = (vec2<f32>(vec2<u32>(pixel)) + vec2<f32>(0.5)) * uSsgi.inv_resolution;
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let pos_vs = (uCam.inv_proj_matrix * ndc).xyz;

    // Bent normal approximation: bias along normal
    let dir_vs = normalize(n_vs);

    // Ray marching in view space
    let step_count = max(uSsgi.num_steps, 1u);
    let max_dist = uSsgi.radius;
    let step_len = max(uSsgi.step_size, max_dist / f32(step_count));

    var hit_mask = 0.0;
    var hit_uv = uv;
    var accum_dist = 0.0;

    for (var i: u32 = 0u; i < step_count; i = i + 1u) {
        accum_dist = f32(i + 1u) * step_len;
        if (accum_dist > max_dist) { break; }
        let p_vs = pos_vs + dir_vs * accum_dist;
        let scr = project_to_screen(p_vs, uCam.proj_matrix);
        let suv = scr.xy;
        // Out of bounds
        if (suv.x <= 0.0 || suv.x >= 1.0 || suv.y <= 0.0 || suv.y >= 1.0) { break; }
        // Hierarchical test (base mip)
        let ray_d = -p_vs.z;
        if (hzb_occluded(suv, ray_d)) {
            hit_mask = 1.0;
            hit_uv = suv;
            break;
        }
    }

    // Pack out: (uv.x, uv.y, ray_depth, mask)
    textureStore(outHit, out_xy, vec4<f32>(hit_uv, accum_dist, hit_mask));
}
