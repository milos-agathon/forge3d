// viewer_pbr_scene_present.wgsl
// Fullscreen present of the resolved PBR-scene HDR buffer into the viewer
// frame output. Tonemap = the shared adjudication operator: Reinhard with
// pre-exposure, x / (1 + x) with x = rgb * exposure, then sRGB encode
// (piecewise IEC 61966-2-1) — skipped when the target format is *Srgb
// because the hardware then applies the same OETF on write.
// RELEVANT FILES: src/viewer/pbr_scene/mod.rs, src/core/tonemap.rs

struct PresentParams {
    exposure: f32,
    manual_srgb: u32,
    pad0: f32,
    pad1: f32,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: PresentParams;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var out: VsOut;
    let xy = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    out.pos = vec4<f32>(xy * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
    return out;
}

fn srgb_encode(c: f32) -> f32 {
    let cc = clamp(c, 0.0, 1.0);
    if (cc <= 0.0031308) {
        return 12.92 * cc;
    }
    return 1.055 * pow(cc, 1.0 / 2.4) - 0.055;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let hdr = textureLoad(src_tex, vec2<i32>(in.pos.xy), 0);
    let x = max(hdr.rgb, vec3<f32>(0.0)) * params.exposure;
    var ldr = x / (vec3<f32>(1.0) + x);
    if (params.manual_srgb != 0u) {
        ldr = vec3<f32>(srgb_encode(ldr.x), srgb_encode(ldr.y), srgb_encode(ldr.z));
    }
    return vec4<f32>(ldr, 1.0);
}
