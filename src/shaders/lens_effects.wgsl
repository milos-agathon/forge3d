// M5: Lens effects post-processing shader
// Applies barrel/pincushion distortion, chromatic aberration, and vignetting
// Applied after tonemapping, before final output

struct LensEffectsUniforms {
    enabled: u32,
    distortion: f32,           // Barrel (+) / pincushion (-) distortion
    chromatic_aberration: f32, // Lateral CA strength
    vignette_strength: f32,    // Corner darkening intensity
    vignette_radius: f32,      // Start radius (0-1)
    vignette_softness: f32,    // Falloff softness
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<uniform> params: LensEffectsUniforms;
@group(0) @binding(1) var input_texture: texture_2d<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var texture_sampler: sampler;

// Apply barrel/pincushion distortion to UV coordinates
// distortion > 0: barrel (edges curve inward)
// distortion < 0: pincushion (edges curve outward)
fn apply_distortion(uv: vec2<f32>, strength: f32) -> vec2<f32> {
    let centered = uv - 0.5;
    let r2 = dot(centered, centered);
    let r4 = r2 * r2;
    
    // Brown-Conrady distortion model (simplified)
    let factor = 1.0 + strength * r2 + strength * 0.5 * r4;
    
    return centered * factor + 0.5;
}

// Sample with chromatic aberration (RGB channel separation)
fn sample_with_ca(uv: vec2<f32>, ca_strength: f32) -> vec3<f32> {
    let centered = uv - 0.5;
    let dist = length(centered);
    
    // Offset scales for each channel (red shifts out, blue shifts in)
    let r_scale = 1.0 + ca_strength * dist;
    let g_scale = 1.0;
    let b_scale = 1.0 - ca_strength * dist;
    
    let uv_r = centered * r_scale + 0.5;
    let uv_g = centered * g_scale + 0.5;
    let uv_b = centered * b_scale + 0.5;
    
    // Sample each channel with its offset UV
    let r = textureSampleLevel(input_texture, texture_sampler, uv_r, 0.0).r;
    let g = textureSampleLevel(input_texture, texture_sampler, uv_g, 0.0).g;
    let b = textureSampleLevel(input_texture, texture_sampler, uv_b, 0.0).b;
    
    return vec3<f32>(r, g, b);
}

// Calculate vignette falloff
fn calculate_vignette(uv: vec2<f32>) -> f32 {
    let centered = uv - 0.5;
    let dist = length(centered) * 2.0; // Normalize to [0, sqrt(2)]
    
    // Smooth falloff from vignette_radius to edge
    let vignette = smoothstep(params.vignette_radius, 
                               params.vignette_radius + params.vignette_softness, 
                               dist);
    
    return 1.0 - vignette * params.vignette_strength;
}

@compute @workgroup_size(8, 8)
fn cs_lens_effects(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let coords = vec2<i32>(i32(x), i32(y));
    var uv = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5) / vec2<f32>(f32(params.width), f32(params.height));
    
    // If lens effects disabled, just copy through
    if (params.enabled == 0u) {
        let color = textureLoad(input_texture, coords, 0);
        textureStore(output_texture, coords, color);
        return;
    }
    
    // Apply distortion to UV coordinates
    var sample_uv = uv;
    if (abs(params.distortion) > 0.001) {
        sample_uv = apply_distortion(uv, params.distortion);
    }
    
    // Check if distorted UV is out of bounds
    if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
        textureStore(output_texture, coords, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    
    // Sample color with optional chromatic aberration
    var color: vec3<f32>;
    if (abs(params.chromatic_aberration) > 0.001) {
        color = sample_with_ca(sample_uv, params.chromatic_aberration);
    } else {
        color = textureSampleLevel(input_texture, texture_sampler, sample_uv, 0.0).rgb;
    }
    
    // Apply vignette
    if (params.vignette_strength > 0.001) {
        let vignette = calculate_vignette(uv);
        color = color * vignette;
    }
    
    textureStore(output_texture, coords, vec4<f32>(color, 1.0));
}
