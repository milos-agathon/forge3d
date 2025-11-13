// AO composite kernel: multiplies material buffer by resolved AO

@group(0) @binding(0) var color_input: texture_2d<f32>;
@group(0) @binding(1) var composite_output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var ao_input: texture_2d<f32>;
@group(0) @binding(3) var<uniform> composite_params: vec4<f32>; // x = multiplier

@compute @workgroup_size(8, 8, 1)
fn cs_ssao_composite(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    let dims = textureDimensions(color_input);
    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }

    let color = textureLoad(color_input, pixel, 0);
    let ao = textureLoad(ao_input, pixel, 0).r;
    let mul = composite_params.x;
    let occlusion = clamp(ao, 0.0, 1.0);
    let ao_mix = occlusion * mul + (1.0 - mul);
    let shaded = vec4<f32>(color.rgb * ao_mix, color.a);
    textureStore(composite_output, pixel, shaded);
}
