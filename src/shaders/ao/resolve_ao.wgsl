// shaders/ao/resolve_ao.wgsl
// Diffuse-only AO resolve and composition
// Inputs:
//  @group(0) @binding(0) texture_2d<f32> diffuse_in
//  @group(0) @binding(1) texture_2d<f32> specular_in
//  @group(0) @binding(2) texture_2d<f32> ao_blur
//  @group(0) @binding(3) storage_texture_2d<rgba8unorm, write> out_rgba
//  @group(0) @binding(4) var<uniform> params: ResolveParams

struct ResolveParams {
    intensity: f32; // AO intensity multiplier
    ssao_mul: f32;  // set to 1.0 to enable, 0.0 to bypass
    _pad: vec2<f32>;
};

@group(0) @binding(0) var t_diffuse: texture_2d<f32>;
@group(0) @binding(1) var t_spec: texture_2d<f32>;
@group(0) @binding(2) var t_ao: texture_2d<f32>;
@group(0) @binding(3) var out_img: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<uniform> cfg: ResolveParams;

@compute @workgroup_size(8, 8, 1)
fn cs_resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(t_diffuse);
    let p = gid.xy;
    if (p.x >= dims.x || p.y >= dims.y) { return; }

    let diff = textureLoad(t_diffuse, p, 0);
    let spec = textureLoad(t_spec, p, 0);
    let ao = textureLoad(t_ao, p, 0).r;

    // diffuse *= mix(1.0, 1.0 - ao * intensity, ssao_mul)
    let ao_factor = mix(1.0, 1.0 - ao * cfg.intensity, cfg.ssao_mul);
    let out_rgb = diff.rgb * ao_factor + spec.rgb;
    textureStore(out_img, p, vec4<f32>(out_rgb, 1.0));
}
