// src/shaders/gi/debug.wgsl
// P5.3/P5.4: GI debug packing pass
// Packs AO, SSGI, and SSR contributions into RGB channels for visualization.

@group(0) @binding(0) var ao_tex: texture_2d<f32>;
@group(0) @binding(1) var ssgi_tex: texture_2d<f32>;
@group(0) @binding(2) var ssr_tex: texture_2d<f32>;
@group(0) @binding(3) var debug_out: texture_storage_2d<rgba16float, write>;

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(8, 8, 1)
fn cs_gi_debug(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(debug_out);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let coord = vec2<i32>(gid.xy);

    // AO channel (R): raw AO scalar in [0,1]
    var ao = textureLoad(ao_tex, coord, 0).r;
    ao = clamp(ao, 0.0, 1.0);

    // SSGI channel (G): tonemapped luminance of diffuse GI
    let ssgi_rgb = textureLoad(ssgi_tex, coord, 0).rgb;
    let ssgi_luma = luminance(ssgi_rgb);
    let ssgi_dbg = clamp(ssgi_luma, 0.0, 4.0) / 4.0;

    // SSR channel (B): tonemapped luminance of specular GI, masked by hit alpha
    let ssr_sample = textureLoad(ssr_tex, coord, 0);
    let ssr_luma = luminance(ssr_sample.rgb) * ssr_sample.a;
    let ssr_dbg = clamp(ssr_luma, 0.0, 4.0) / 4.0;

    textureStore(debug_out, coord, vec4<f32>(ao, ssgi_dbg, ssr_dbg, 1.0));
}
