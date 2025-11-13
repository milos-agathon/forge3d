// shaders/ao/gtao.wgsl
// P5.1 GTAO compute shader (Ground-Truth Ambient Occlusion)
// Bindings (strict):
//  @group(0) @binding(0) texture_2d<f32> depth
//  @group(0) @binding(1) texture_2d<f32> hzb  // mip chain, sampled with textureLoad(..., mip)
//  @group(0) @binding(2) texture_2d<f32> normals
//  @group(0) @binding(3) sampler linearClamp
//  @group(0) @binding(4) storage_texture_2d<rgba16float, write> aoRaw
//  @group(0) @binding(5) var<uniform> params: GtaoParams
//  @group(0) @binding(6) var<uniform> camera: CameraParams

struct GtaoParams {
    radius: f32,
    bias: f32,
    samples: u32,
    directions: u32,
    temporal_alpha: f32, // unused in P5.1 (0.0)
    inv_resolution: vec2<f32>,
    proj_scale: f32,
    hzb_mips: u32,
    seed: u32, // must be 1337u for strict determinism
    _pad: u32,
};

struct CameraParams {
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
};

@group(0) @binding(0) var depth_tex: texture_2d<f32>;
@group(0) @binding(1) var hzb_tex: texture_2d<f32>;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var lin_sampler: sampler;
@group(0) @binding(4) var ao_raw: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<uniform> params: GtaoParams;
@group(0) @binding(6) var<uniform> camera: CameraParams;

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

fn reconstruct_view_pos_linear(uv: vec2<f32>, linear_depth: f32) -> vec3<f32> {
    // UV [0,1] -> NDC [-1,1] with Y up
    let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    // Use inv_proj to get view XY per unit Z
    let focal = vec2<f32>(camera.inv_proj[0][0], camera.inv_proj[1][1]);
    let center = vec2<f32>(camera.inv_proj[2][0], camera.inv_proj[2][1]);
    let view_xy = (ndc_xy - center) / focal;
    return vec3<f32>(view_xy * linear_depth, -linear_depth);
}

fn pack_normal(packed: vec4<f32>) -> vec3<f32> {
    // normals are assumed to be stored in view space in [-1,1]
    return normalize(packed.xyz);
}

// Deterministic interleaved gradient with fixed seed (frame-constant)
fn ign(pixel: vec2<u32>, salt: u32) -> f32 {
    let p = pixel ^ vec2<u32>(params.seed, params.seed ^ salt);
    let f = vec2<f32>(f32(p.x), f32(p.y));
    return fract(52.9829189 * fract(0.06711056 * f.x + 0.00583715 * f.y));
}

// Choose HZB mip based on step in pixels, clamped to [0, hzb_mips-1]
fn hzb_mip_for_step(step_px: f32) -> i32 {
    let l = log2(max(step_px, 1.0));
    let m = i32(floor(l));
    let max_m = i32(max(1u, params.hzb_mips) - 1u);
    return clamp(m, 0, max_m);
}

@compute @workgroup_size(8, 8, 1)
fn cs_gtao(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    let dims = textureDimensions(depth_tex);
    if (pixel.x >= dims.x || pixel.y >= dims.y) { return; }

    let uv = (vec2<f32>(pixel) + 0.5) / vec2<f32>(dims);

    let depth = textureLoad(depth_tex, pixel, 0);
    if (depth >= 0.9999) {
        textureStore(ao_raw, pixel, vec4<f32>(1.0, 1.0, 1.0, 1.0));
        return;
    }

    let normal_packed = textureLoad(normal_tex, pixel, 0);
    let n = pack_normal(normal_packed);
    let p_vs = reconstruct_view_pos_linear(uv, depth);

    // Horizon-based AO
    let dir_count = max(params.directions, 2u);
    let steps_per_dir = max(params.samples / max(dir_count, 1u), 2u);
    let angle_jitter = ign(pixel, 747796405u) * TWO_PI;

    var visibility = 0.0;

    for (var d = 0u; d < dir_count; d++) {
        let angle = (f32(d) / f32(dir_count)) * PI + angle_jitter;
        let dir2 = vec2<f32>(cos(angle), sin(angle));
        var horizon_cos = -1.0;

        for (var s = 1u; s <= steps_per_dir; s++) {
            let z_lin = -p_vs.z;
            let r_screen = params.radius * (params.proj_scale / max(z_lin, 1e-4));
            let step_px = r_screen * (f32(s) / f32(steps_per_dir));
            let sample_uv = uv + dir2 * (step_px * params.inv_resolution);

            if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) { continue; }

            let hzb_dims = textureDimensions(hzb_tex);
            let mip = hzb_mip_for_step(step_px);
            let spix = vec2<u32>(sample_uv * vec2<f32>(hzb_dims));
            let min_depth = textureLoad(hzb_tex, spix, mip).r;

            // Reconstruct sample position at min depth
            let spos_vs = reconstruct_view_pos_linear(sample_uv, min_depth);
            let h = spos_vs - p_vs;
            let hlen = length(h);
            let hcos = dot(h / max(hlen, 1e-4), n);
            let attenuation = 1.0 - clamp(hlen / params.radius, 0.0, 1.0);
            horizon_cos = max(horizon_cos, hcos * attenuation);
        }

        let horizon_angle = acos(clamp(horizon_cos, -1.0, 1.0));
        visibility += sin(horizon_angle) - horizon_angle * cos(horizon_angle) + 0.5 * PI;
    }

    visibility = visibility / (f32(dir_count) * PI);
    let ao = clamp(1.0 - visibility, 0.0, 1.0);
    textureStore(ao_raw, pixel, vec4<f32>(ao, ao, ao, 1.0));
}
