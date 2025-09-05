//! Normal mapping utilities for tangent-space normal processing
//! 
//! Provides functions for transforming normals from tangent space to world space
//! and applying normal maps to surface geometry.

/// Transform a tangent-space normal to world space using TBN matrix
/// 
/// @param tangent_normal: Normal from normal map texture (range [-1,1])
/// @param tangent: World-space tangent vector (unit length)
/// @param bitangent: World-space bitangent vector (unit length) 
/// @param normal: World-space surface normal (unit length)
/// @return: Transformed world-space normal (unit length)
fn apply_normal_map(
    tangent_normal: vec3<f32>,
    tangent: vec3<f32>,
    bitangent: vec3<f32>, 
    normal: vec3<f32>
) -> vec3<f32> {
    // Construct TBN matrix (tangent -> world space transform)
    let tbn = mat3x3<f32>(
        tangent,
        bitangent,
        normal
    );
    
    // Transform tangent-space normal to world space
    let world_normal = tbn * tangent_normal;
    
    // Normalize to ensure unit length (handle any numerical error)
    return normalize(world_normal);
}

/// Decode normal from texture sample and convert to tangent space
/// 
/// @param normal_sample: RGBA sample from normal map texture
/// @param strength: Normal map intensity/strength multiplier (default 1.0)
/// @return: Tangent-space normal vector (range [-1,1])
fn decode_normal_map(normal_sample: vec4<f32>, strength: f32) -> vec3<f32> {
    // Decode from [0,1] texture range to [-1,1] normal range
    var tangent_normal = normal_sample.rgb * 2.0 - 1.0;
    
    // Apply strength/intensity scaling
    tangent_normal.xy *= strength;
    
    // Ensure Z component maintains unit length constraint
    // For well-formed normal maps, Z should be positive
    let xy_len_sq = dot(tangent_normal.xy, tangent_normal.xy);
    tangent_normal.z = sqrt(max(0.0, 1.0 - xy_len_sq));
    
    return tangent_normal;
}

/// Combined normal mapping: decode texture and transform to world space
/// 
/// @param normal_texture: Sampled normal map texture value
/// @param strength: Normal map strength/intensity
/// @param tangent: World-space tangent vector
/// @param bitangent: World-space bitangent vector  
/// @param surface_normal: World-space surface normal
/// @return: Final world-space normal with normal mapping applied
fn sample_normal_map(
    normal_texture: vec4<f32>,
    strength: f32,
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
    surface_normal: vec3<f32>
) -> vec3<f32> {
    let tangent_normal = decode_normal_map(normal_texture, strength);
    return apply_normal_map(tangent_normal, tangent, bitangent, surface_normal);
}

/// Compute normal matrix from model matrix for correct normal transformation
/// 
/// @param model_matrix: 4x4 model transformation matrix
/// @return: 3x3 normal transformation matrix (inverse-transpose of upper-left 3x3)
fn compute_normal_matrix(model_matrix: mat4x4<f32>) -> mat3x3<f32> {
    // Extract upper-left 3x3 portion of model matrix
    let model3x3 = mat3x3<f32>(
        model_matrix[0].xyz,
        model_matrix[1].xyz,
        model_matrix[2].xyz
    );
    
    // Compute inverse-transpose for correct normal transformation
    // For orthogonal matrices (uniform scaling), this simplifies to normalization
    return transpose(model3x3); // Simplified - assumes orthogonal transforms
}

/// Perturb normal using detail normal map (for layered normal mapping)
/// 
/// @param base_normal: Base world-space normal
/// @param detail_normal: Detail tangent-space normal from secondary texture
/// @param detail_strength: Strength of detail normal perturbation
/// @param tangent: World-space tangent vector
/// @param bitangent: World-space bitangent vector
/// @return: Combined world-space normal with detail perturbation
fn apply_detail_normal(
    base_normal: vec3<f32>,
    detail_normal: vec3<f32>, 
    detail_strength: f32,
    tangent: vec3<f32>,
    bitangent: vec3<f32>
) -> vec3<f32> {
    // Scale detail normal by strength
    var scaled_detail = detail_normal;
    scaled_detail.xy *= detail_strength;
    
    // Renormalize to maintain unit length
    let xy_len_sq = dot(scaled_detail.xy, scaled_detail.xy);
    scaled_detail.z = sqrt(max(0.0, 1.0 - xy_len_sq));
    
    // Transform detail normal to world space
    let tbn = mat3x3<f32>(tangent, bitangent, base_normal);
    let world_detail = tbn * scaled_detail;
    
    // Blend with base normal (whiteout blending)
    let blended = vec3<f32>(
        base_normal.x + world_detail.x,
        base_normal.y + world_detail.y, 
        base_normal.z * world_detail.z
    );
    
    return normalize(blended);
}

/// Validate TBN matrix for normal mapping correctness
/// 
/// @param tangent: Tangent vector
/// @param bitangent: Bitangent vector
/// @param normal: Normal vector
/// @return: True if TBN is valid for normal mapping
fn validate_tbn(tangent: vec3<f32>, bitangent: vec3<f32>, normal: vec3<f32>) -> bool {
    // Check unit length (within tolerance)
    let t_len = length(tangent);
    let b_len = length(bitangent);
    let n_len = length(normal);
    
    if (abs(t_len - 1.0) > 1e-3 || abs(b_len - 1.0) > 1e-3 || abs(n_len - 1.0) > 1e-3) {
        return false;
    }
    
    // Check orthogonality (tangent ⊥ normal)
    if (abs(dot(tangent, normal)) > 1e-3) {
        return false;
    }
    
    // Check determinant (should be ±1 for proper handedness)
    let cross_tb = cross(tangent, bitangent);
    let det = dot(cross_tb, normal);
    if (abs(abs(det) - 1.0) > 1e-2) {
        return false;
    }
    
    return true;
}