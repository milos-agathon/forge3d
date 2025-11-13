// shaders/ao/bilateral_separable.wgsl
// P5.1 Separable bilateral blur for AO (horizontal and vertical variants)
// Bindings (kept simple for example):
//  @group(0) @binding(0) texture_2d<f32> ao_in
//  @group(0) @binding(1) texture_2d<f32> depth
//  @group(0) @binding(2) texture_2d<f32> normals
//  @group(0) @binding(3) sampler linearClamp
//  @group(0) @binding(4) storage_texture_2d<rgba16float, write> ao_out
//  @group(0) @binding(5) var<uniform> direction: vec2<f32> // (1,0) or (0,1)

@group(0) @binding(0) var ao_in: texture_2d<f32>;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var lin_sampler: sampler;
@group(0) @binding(4) var ao_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<uniform> direction: vec2<f32>;

fn unpack_normal(p: vec4<f32>) -> vec3<f32> { return normalize(p.xyz); }

const KERNEL_RADIUS: i32 = 3; // width 7
const SIGMA_DEPTH: f32 = 1.5;
const SIGMA_NORMAL: f32 = 32.0; // cosine power base

fn bilateral_weight(center_depth: f32, sample_depth: f32, center_n: vec3<f32>, sample_n: vec3<f32>, dist: f32) -> f32 {
    let dd = abs(center_depth - sample_depth);
    let wd = exp(-(dd * dd) / (2.0 * SIGMA_DEPTH * SIGMA_DEPTH));
    let ndot = max(dot(center_n, sample_n), 0.0);
    let wn = pow(ndot, SIGMA_NORMAL);
    let ws = exp(-(dist * dist) / (2.0 * 3.0 * 3.0));
    return wd * wn * ws;
}

@compute @workgroup_size(8, 8, 1)
fn cs_blur(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    let dims = textureDimensions(ao_in);
    if (pixel.x >= dims.x || pixel.y >= dims.y) { return; }

    let center_depth = textureLoad(depth_tex, pixel, 0);
    let center_n = unpack_normal(textureLoad(normal_tex, pixel, 0));
    let center_ao = textureLoad(ao_in, pixel, 0).r;

    var sum = center_ao;
    var wsum = 1.0;

    for (var i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++) {
        if (i == 0) { continue; }
        let offs = vec2<i32>(i32(direction.x) * i, i32(direction.y) * i);
        let sp = vec2<i32>(pixel) + offs;
        if (sp.x < 0 || sp.y < 0 || sp.x >= i32(dims.x) || sp.y >= i32(dims.y)) { continue; }
        let sdepth = textureLoad(depth_tex, vec2<u32>(sp), 0);
        let sn = unpack_normal(textureLoad(normal_tex, vec2<u32>(sp), 0));
        let sao = textureLoad(ao_in, vec2<u32>(sp), 0).r;
        let w = bilateral_weight(center_depth, sdepth, center_n, sn, f32(abs(i)));
        sum += sao * w;
        wsum += w;
    }

    let out_ao = sum / max(wsum, 1e-4);
    textureStore(ao_out, pixel, vec4<f32>(out_ao, out_ao, out_ao, 1.0));
}
