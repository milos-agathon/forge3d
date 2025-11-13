// shaders/ao/ssao.wgsl
// P5.1 SSAO compute shader (hemisphere sampling)
// Bindings (strict):
//  @group(0) @binding(0) texture_depth_2d depth
//  @group(0) @binding(1) texture_2d<f32> hzb   // present but not required for SSAO
//  @group(0) @binding(2) texture_2d<f32> normals
//  @group(0) @binding(3) sampler linearClamp
//  @group(0) @binding(4) storage_texture_2d<rgba16float, write> aoRaw
//  @group(0) @binding(5) var<uniform> params: SsaoParams
//  @group(0) @binding(6) var<uniform> camera: CameraParams

struct SsaoParams {
    radius: f32,
    bias: f32,
    samples: u32,
    temporal_alpha: f32, // 0.0 in P5.1
    inv_resolution: vec2<f32>,
    proj_scale: f32,
    seed: u32, // 1337u for determinism
    _pad: u32,
};

struct CameraParams {
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
};

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var hzb_tex: texture_2d<f32>;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var lin_sampler: sampler;
@group(0) @binding(4) var ao_raw: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<uniform> params: SsaoParams;
@group(0) @binding(6) var<uniform> camera: CameraParams;

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

fn reconstruct_view_pos_from_depth(uv: vec2<f32>, depth01: f32) -> vec3<f32> {
    // Convert to NDC and unproject using inv_proj
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth01 * 2.0 - 1.0, 1.0);
    let v = camera.inv_proj * ndc;
    return (v.xyz / v.w);
}

fn unpack_normal(packed: vec4<f32>) -> vec3<f32> { return normalize(packed.xyz); }

fn ign(pixel: vec2<u32>, salt: u32) -> f32 {
    let p = pixel ^ vec2<u32>(params.seed, params.seed ^ salt);
    let f = vec2<f32>(f32(p.x), f32(p.y));
    return fract(52.9829189 * fract(0.06711056 * f.x + 0.00583715 * f.y));
}

@compute @workgroup_size(8, 8, 1)
fn cs_ssao(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    let dims = textureDimensions(depth_tex);
    if (pixel.x >= dims.x || pixel.y >= dims.y) { return; }

    let uv = (vec2<f32>(pixel) + 0.5) / vec2<f32>(dims);
    let depth = textureLoad(depth_tex, pixel, 0);
    if (depth >= 0.9999) { textureStore(ao_raw, pixel, vec4<f32>(1.0)); return; }

    let nrm = unpack_normal(textureLoad(normal_tex, pixel, 0));
    let p_vs = reconstruct_view_pos_from_depth(uv, depth);

    // Tangent basis
    let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), abs(nrm.y) < 0.99);
    let t = normalize(cross(up, nrm));
    let b = cross(nrm, t);
    let tbn = mat3x3<f32>(t, b, nrm);

    let nsamp = max(params.samples, 8u);
    let jitter = ign(pixel, 2891336453u) * TWO_PI;
    var occ = 0.0;

    for (var i = 0u; i < nsamp; i++) {
        let fi = f32(i);
        let a = (fi + 0.5) / f32(nsamp);
        let ang = a * TWO_PI * 2.0 + jitter;
        let h = sqrt(1.0 - a);
        let r = sqrt(a);
        let dir_ts = vec3<f32>(cos(ang) * r, sin(ang) * r, h);
        let dir_vs = tbn * dir_ts;

        // Screen-space step size based on projected radius
        let z_lin = -p_vs.z;
        let r_screen = params.radius * (params.proj_scale / max(z_lin, 1e-4));
        let step_px = r_screen;
        let suv = uv + normalize(dir_vs.xy) * (step_px * params.inv_resolution);
        if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) { continue; }

        let sd = textureLoad(depth_tex, vec2<u32>(suv * vec2<f32>(dims)), 0);
        let sp = reconstruct_view_pos_from_depth(suv, sd);
        let dz = (-sp.z) - z_lin;
        let bias = max(params.bias, 0.001);
        let range = smoothstep(0.0, 1.0, params.radius / max(length(sp - p_vs), 1e-4));
        if (dz < -bias) { occ += range; }
    }

    let ao = clamp(1.0 - occ / f32(nsamp), 0.0, 1.0);
    textureStore(ao_raw, pixel, vec4<f32>(ao));
}
