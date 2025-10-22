// Shadow mapping utilities with Percentage-Closer Filtering (PCF)
// Provides soft shadows with configurable kernel radius

/// Sample shadow map with PCF (Percentage-Closer Filtering)
/// Returns shadow factor: 0.0 = fully shadowed, 1.0 = fully lit
fn sample_shadow_pcf(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    shadow_pos_ndc: vec3<f32>,
    shadow_bias: f32,
    pcf_radius: f32,
) -> f32 {
    // Convert NDC [-1, 1] to texture coordinates [0, 1]
    let shadow_uv = vec2<f32>(
        shadow_pos_ndc.x * 0.5 + 0.5,
        shadow_pos_ndc.y * -0.5 + 0.5  // Flip Y for texture space
    );
    
    // Apply depth bias to prevent shadow acne
    let biased_depth = shadow_pos_ndc.z - shadow_bias;
    
    // Early out if outside shadow map bounds
    if (shadow_uv.x < 0.0 || shadow_uv.x > 1.0 || shadow_uv.y < 0.0 || shadow_uv.y > 1.0) {
        return 1.0;  // No shadow outside frustum
    }
    
    // Early out if behind light (shouldn't happen with proper frustum)
    if (biased_depth < 0.0 || biased_depth > 1.0) {
        return 1.0;
    }
    
    // PCF: Sample multiple points in a kernel around the current position
    let shadow_map_size = textureDimensions(shadow_map);
    let texel_size = 1.0 / vec2<f32>(f32(shadow_map_size.x), f32(shadow_map_size.y));
    
    var shadow_sum = 0.0;
    var sample_count = 0.0;
    
    // Adaptive kernel size based on pcf_radius (typically 1.0 to 5.0)
    let kernel_steps = i32(ceil(pcf_radius));
    
    for (var y = -kernel_steps; y <= kernel_steps; y++) {
        for (var x = -kernel_steps; x <= kernel_steps; x++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size * pcf_radius;
            let sample_uv = shadow_uv + offset;
            
            // Sample with hardware comparison
            let shadow_sample = textureSampleCompare(
                shadow_map,
                shadow_sampler,
                sample_uv,
                biased_depth
            );
            
            shadow_sum += shadow_sample;
            sample_count += 1.0;
        }
    }
    
    return shadow_sum / sample_count;
}

/// Optimized 2x2 PCF for soft shadows (faster than full kernel)
fn sample_shadow_pcf_2x2(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    shadow_pos_ndc: vec3<f32>,
    shadow_bias: f32,
) -> f32 {
    let shadow_uv = vec2<f32>(
        shadow_pos_ndc.x * 0.5 + 0.5,
        shadow_pos_ndc.y * -0.5 + 0.5
    );
    
    let biased_depth = shadow_pos_ndc.z - shadow_bias;
    
    if (shadow_uv.x < 0.0 || shadow_uv.x > 1.0 || shadow_uv.y < 0.0 || shadow_uv.y > 1.0) {
        return 1.0;
    }
    
    if (biased_depth < 0.0 || biased_depth > 1.0) {
        return 1.0;
    }
    
    let shadow_map_size = textureDimensions(shadow_map);
    let texel_size = 1.0 / vec2<f32>(f32(shadow_map_size.x), f32(shadow_map_size.y));
    
    // 2x2 tap pattern
    let s0 = textureSampleCompare(shadow_map, shadow_sampler, shadow_uv + vec2<f32>(-0.5, -0.5) * texel_size, biased_depth);
    let s1 = textureSampleCompare(shadow_map, shadow_sampler, shadow_uv + vec2<f32>(0.5, -0.5) * texel_size, biased_depth);
    let s2 = textureSampleCompare(shadow_map, shadow_sampler, shadow_uv + vec2<f32>(-0.5, 0.5) * texel_size, biased_depth);
    let s3 = textureSampleCompare(shadow_map, shadow_sampler, shadow_uv + vec2<f32>(0.5, 0.5) * texel_size, biased_depth);
    
    return (s0 + s1 + s2 + s3) * 0.25;
}

/// Hard shadow (single sample, no PCF) - fastest option
fn sample_shadow_hard(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    shadow_pos_ndc: vec3<f32>,
    shadow_bias: f32,
) -> f32 {
    let shadow_uv = vec2<f32>(
        shadow_pos_ndc.x * 0.5 + 0.5,
        shadow_pos_ndc.y * -0.5 + 0.5
    );
    
    let biased_depth = shadow_pos_ndc.z - shadow_bias;
    
    if (shadow_uv.x < 0.0 || shadow_uv.x > 1.0 || shadow_uv.y < 0.0 || shadow_uv.y > 1.0) {
        return 1.0;
    }
    
    if (biased_depth < 0.0 || biased_depth > 1.0) {
        return 1.0;
    }
    
    return textureSampleCompare(shadow_map, shadow_sampler, shadow_uv, biased_depth);
}
