// SSGI composite kernel: adds SSGI diffuse radiance to the material/color buffer.
// material_input   : viewer material buffer (RGBA8) promoted to float; rgb = base color.
// ssgi_input       : Rgba16Float GI radiance (rgb) from SSGI, alpha unused.
// composite_output : Rgba8Unorm material+GI buffer used by P5 harness for visualization.

@group(0) @binding(0) var material_input: texture_2d<f32>;
@group(0) @binding(1) var composite_output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var ssgi_input: texture_2d<f32>;
@group(0) @binding(3) var<uniform> composite_params: vec4<f32>; // x = intensity multiplier

@compute @workgroup_size(8, 8, 1)
fn cs_ssgi_composite(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    let dims = textureDimensions(material_input);
    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }

    let material = textureLoad(material_input, pixel, 0);
    let ssgi_radiance = textureLoad(ssgi_input, pixel, 0).rgb;
    let intensity = composite_params.x;
    
    // Add SSGI radiance to material color (SSGI is already in linear space)
    let final_color = material.rgb + ssgi_radiance * intensity;
    
    // Clamp to prevent overflow (before tone mapping)
    let clamped = clamp(final_color, vec3<f32>(0.0), vec3<f32>(10.0));
    
    textureStore(composite_output, pixel, vec4<f32>(clamped, material.a));
}

