// HDRI Environment Lighting Utilities
// Equirectangular (lat-long) environment map sampling with rotation support

/// Sample an equirectangular environment map using a world-space direction.
/// 
/// Parameters:
/// - dir: Normalized world-space direction vector
/// - rotation_rad: Y-axis rotation in radians (0 = no rotation)
/// - tex: The HDRI texture (equirectangular format)
/// - samp: Linear sampler for the texture
///
/// Returns: RGB color from the environment map
fn sample_environment(
    dir: vec3<f32>,
    rotation_rad: f32,
    tex: texture_2d<f32>,
    samp: sampler
) -> vec3<f32> {
    // Apply Y-axis rotation to direction
    let cos_rot = cos(rotation_rad);
    let sin_rot = sin(rotation_rad);
    
    let rotated_dir = vec3<f32>(
        dir.x * cos_rot - dir.z * sin_rot,
        dir.y,
        dir.x * sin_rot + dir.z * cos_rot
    );
    
    // Convert direction to equirectangular UV coordinates
    // u = atan2(x, z) / (2π) + 0.5
    // v = asin(y) / π + 0.5
    let u = atan2(rotated_dir.x, rotated_dir.z) / (2.0 * 3.14159265359) + 0.5;
    let v = asin(clamp(rotated_dir.y, -1.0, 1.0)) / 3.14159265359 + 0.5;
    
    let uv = vec2<f32>(u, v);

    let dims_u32 = textureDimensions(tex);
    let dims = vec2<f32>(dims_u32);

    // Wrap horizontally and clamp vertically to avoid sampling outside texture bounds
    let u_wrapped = fract(uv.x);
    let v_clamped = clamp(uv.y, 0.0, 1.0 - (1.0 / max(dims.y, 1.0)));
    let tex_coords = vec2<i32>(vec2<f32>(u_wrapped, v_clamped) * dims);
    
    return textureLoad(tex, tex_coords, 0).rgb;
}

/// Sample environment with normal vector (for diffuse IBL)
/// Uses the surface normal to sample the environment map
fn sample_environment_diffuse(
    normal: vec3<f32>,
    rotation_rad: f32,
    tex: texture_2d<f32>,
    samp: sampler
) -> vec3<f32> {
    return sample_environment(normal, rotation_rad, tex, samp);
}

/// Sample environment with reflection vector (for specular IBL)
/// Uses the reflected view direction for glossy reflections
fn sample_environment_specular(
    reflection: vec3<f32>,
    rotation_rad: f32,
    tex: texture_2d<f32>,
    samp: sampler
) -> vec3<f32> {
    return sample_environment(reflection, rotation_rad, tex, samp);
}

/// Compute luminance of an RGB color (Rec. 709 coefficients)
fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

/// Tonemap HDR color to LDR (simple Reinhard)
fn tonemap_reinhard(hdr: vec3<f32>) -> vec3<f32> {
    return hdr / (vec3<f32>(1.0) + hdr);
}
