//! PBR (Physically-Based Rendering) shaders
//! 
//! Implements metallic-roughness workflow with support for:
//! - Base color, metallic, roughness, normal, occlusion, emissive textures
//! - Direct and indirect lighting (IBL)
//! - Standard PBR BRDF (GGX + Lambertian)

struct Uniforms {
    model_matrix: mat4x4<f32>,
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
}

struct PbrMaterial {
    base_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    normal_scale: f32,
    occlusion_strength: f32,
    emissive: vec3<f32>,
    alpha_cutoff: f32,
    texture_flags: u32,
    // padding aligns to 16 bytes
}

struct PbrLighting {
    light_direction: vec3<f32>,
    // padding: f32,
    light_color: vec3<f32>,
    light_intensity: f32,
    camera_position: vec3<f32>,
    // padding: f32,
    ibl_intensity: f32,
    ibl_rotation: f32,
    exposure: f32,
    gamma: f32,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) world_normal: vec3<f32>,
    @location(3) world_tangent: vec3<f32>,
    @location(4) world_bitangent: vec3<f32>,
}

// Uniforms and textures
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> lighting: PbrLighting;

@group(1) @binding(0) var<uniform> material: PbrMaterial;
@group(1) @binding(1) var base_color_texture: texture_2d<f32>;
@group(1) @binding(2) var metallic_roughness_texture: texture_2d<f32>;
@group(1) @binding(3) var normal_texture: texture_2d<f32>;
@group(1) @binding(4) var occlusion_texture: texture_2d<f32>;
@group(1) @binding(5) var emissive_texture: texture_2d<f32>;
@group(1) @binding(6) var material_sampler: sampler;

// Optional IBL textures
@group(2) @binding(0) var irradiance_texture: texture_2d<f32>;
@group(2) @binding(1) var irradiance_sampler: sampler;
@group(2) @binding(2) var prefilter_texture: texture_2d<f32>;
@group(2) @binding(3) var prefilter_sampler: sampler;
@group(2) @binding(4) var brdf_lut_texture: texture_2d<f32>;
@group(2) @binding(5) var brdf_lut_sampler: sampler;

// Texture flags
const FLAG_BASE_COLOR: u32 = 1u;
const FLAG_METALLIC_ROUGHNESS: u32 = 2u;
const FLAG_NORMAL: u32 = 4u;
const FLAG_OCCLUSION: u32 = 8u;
const FLAG_EMISSIVE: u32 = 16u;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    
    // Transform position to world space then to clip space
    let world_position = uniforms.model_matrix * vec4<f32>(input.position, 1.0);
    output.world_position = world_position.xyz;
    output.clip_position = uniforms.projection_matrix * uniforms.view_matrix * world_position;
    
    // Pass through UV coordinates
    output.uv = input.uv;
    
    // Transform TBN vectors to world space
    let normal_mat = mat3x3<f32>(
        uniforms.normal_matrix[0].xyz,
        uniforms.normal_matrix[1].xyz,
        uniforms.normal_matrix[2].xyz
    );
    
    output.world_normal = normalize(normal_mat * input.normal);
    output.world_tangent = normalize(normal_mat * input.tangent);
    output.world_bitangent = normalize(normal_mat * input.bitangent);
    
    return output;
}

// Sample normal map and transform to world space
fn sample_normal_map(uv: vec2<f32>, tbn: mat3x3<f32>) -> vec3<f32> {
    if (material.texture_flags & FLAG_NORMAL) != 0u {
        let normal_sample = textureSample(normal_texture, material_sampler, uv);
        
        // Decode normal from texture (range [0,1] to [-1,1])
        var tangent_normal = normal_sample.xyz * 2.0 - 1.0;
        
        // Apply normal scale
        tangent_normal.xy = tangent_normal.xy * material.normal_scale;
        
        // Normalize to ensure unit length
        tangent_normal = normalize(tangent_normal);
        
        // Transform to world space
        return normalize(tbn * tangent_normal);
    } else {
        // Use vertex normal
        return normalize(tbn[2]);
    }
}

// Distribution function (GGX/Trowbridge-Reitz)
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let n_dot_h2 = n_dot_h * n_dot_h;
    
    let num = alpha2;
    let denom = 3.14159265359 * pow(n_dot_h2 * (alpha2 - 1.0) + 1.0, 2.0);
    
    return num / max(denom, 1e-6);
}

// Geometry function (Smith model)
fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let k = pow(roughness + 1.0, 2.0) / 8.0;
    
    let ggx1 = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let ggx2 = n_dot_l / (n_dot_l * (1.0 - k) + k);
    
    return ggx1 * ggx2;
}

// Fresnel-Schlick approximation
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Fresnel-Schlick with roughness for IBL
fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Convert direction to equirectangular UV coordinates
fn direction_to_uv(direction: vec3<f32>) -> vec2<f32> {
    let phi = atan2(direction.z, direction.x);
    let theta = acos(direction.y);
    
    let u = (phi / (2.0 * 3.14159265359) + 0.5) % 1.0;
    let v = theta / 3.14159265359;
    
    return vec2<f32>(u, v);
}

// Sample IBL irradiance
fn sample_irradiance(normal: vec3<f32>) -> vec3<f32> {
    let uv = direction_to_uv(normal);
    return textureSample(irradiance_texture, irradiance_sampler, uv).rgb * lighting.ibl_intensity;
}

// Sample IBL prefiltered environment map
fn sample_prefilter(reflection: vec3<f32>, roughness: f32) -> vec3<f32> {
    let uv = direction_to_uv(reflection);
    let mip_level = roughness * 6.0; // Assume 7 mip levels (0-6)
    return textureSampleLevel(prefilter_texture, prefilter_sampler, uv, mip_level).rgb * lighting.ibl_intensity;
}

// Sample BRDF LUT
fn sample_brdf_lut(n_dot_v: f32, roughness: f32) -> vec2<f32> {
    let uv = vec2<f32>(n_dot_v, roughness);
    return textureSample(brdf_lut_texture, brdf_lut_sampler, uv).rg;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Construct TBN matrix
    let T = normalize(input.world_tangent);
    let B = normalize(input.world_bitangent);
    let N = normalize(input.world_normal);
    let TBN = mat3x3<f32>(T, B, N);
    
    // Sample normal map and get world normal
    let world_normal = sample_normal_map(input.uv, TBN);
    
    // Sample material textures
    var base_color = material.base_color;
    if (material.texture_flags & FLAG_BASE_COLOR) != 0u {
        base_color = base_color * textureSample(base_color_texture, material_sampler, input.uv);
    }
    
    var metallic = material.metallic;
    var roughness = material.roughness;
    if (material.texture_flags & FLAG_METALLIC_ROUGHNESS) != 0u {
        let mr_sample = textureSample(metallic_roughness_texture, material_sampler, input.uv);
        metallic = metallic * mr_sample.b; // Blue channel = metallic
        roughness = roughness * mr_sample.g; // Green channel = roughness
    }
    
    // Clamp roughness to avoid singularities
    roughness = clamp(roughness, 0.04, 1.0);
    
    var occlusion = 1.0;
    if (material.texture_flags & FLAG_OCCLUSION) != 0u {
        occlusion = mix(1.0, textureSample(occlusion_texture, material_sampler, input.uv).r, material.occlusion_strength);
    }
    
    var emissive = material.emissive;
    if (material.texture_flags & FLAG_EMISSIVE) != 0u {
        emissive = emissive * textureSample(emissive_texture, material_sampler, input.uv).rgb;
    }
    
    // Alpha testing
    if base_color.a < material.alpha_cutoff {
        discard;
    }
    
    // Calculate lighting vectors
    let view_dir = normalize(lighting.camera_position - input.world_position);
    let light_dir = normalize(-lighting.light_direction);
    let half_dir = normalize(light_dir + view_dir);
    let reflection_dir = reflect(-view_dir, world_normal);
    
    // Calculate dot products
    let n_dot_v = max(dot(world_normal, view_dir), 0.0);
    let n_dot_l = max(dot(world_normal, light_dir), 0.0);
    let n_dot_h = max(dot(world_normal, half_dir), 0.0);
    let v_dot_h = max(dot(view_dir, half_dir), 0.0);
    
    // Calculate F0 (surface reflection at zero incidence)
    let dielectric_f0 = vec3<f32>(0.04);
    let f0 = mix(dielectric_f0, base_color.rgb, metallic);
    
    // DIRECT LIGHTING
    var direct_lighting = vec3<f32>(0.0);
    
    if n_dot_l > 0.0 {
        // Calculate BRDF terms
        let D = distribution_ggx(n_dot_h, roughness);
        let G = geometry_smith(n_dot_v, n_dot_l, roughness);
        let F = fresnel_schlick(v_dot_h, f0);
        
        // Cook-Torrance specular BRDF
        let specular = (D * G * F) / max(4.0 * n_dot_v * n_dot_l, 1e-6);
        
        // Diffuse BRDF (Lambertian)
        let kS = F;
        let kD = (vec3<f32>(1.0) - kS) * (1.0 - metallic);
        let diffuse = kD * base_color.rgb / 3.14159265359;
        
        // Add direct lighting contribution
        let radiance = lighting.light_color * lighting.light_intensity;
        direct_lighting = (diffuse + specular) * radiance * n_dot_l;
    }
    
    // INDIRECT LIGHTING (IBL)
    var indirect_lighting = vec3<f32>(0.0);
    
    // Check if we have IBL textures available (simplified check)
    let has_ibl = lighting.ibl_intensity > 0.0;
    
    if has_ibl {
        // Diffuse IBL
        let irradiance = sample_irradiance(world_normal);
        let F_ibl = fresnel_schlick_roughness(n_dot_v, f0, roughness);
        let kS_ibl = F_ibl;
        let kD_ibl = (vec3<f32>(1.0) - kS_ibl) * (1.0 - metallic);
        let diffuse_ibl = kD_ibl * base_color.rgb * irradiance;
        
        // Specular IBL
        let prefiltered_color = sample_prefilter(reflection_dir, roughness);
        let brdf_lut = sample_brdf_lut(n_dot_v, roughness);
        let specular_ibl = prefiltered_color * (F_ibl * brdf_lut.x + brdf_lut.y);
        
        indirect_lighting = diffuse_ibl + specular_ibl;
    } else {
        // Simple ambient lighting fallback
        let ambient = vec3<f32>(0.03) * base_color.rgb;
        indirect_lighting = ambient;
    }
    
    // Apply ambient occlusion
    indirect_lighting = indirect_lighting * occlusion;
    
    // Combine direct and indirect lighting
    var color = direct_lighting + indirect_lighting + emissive;
    
    // Tone mapping and gamma correction
    color = color * lighting.exposure;
    
    // Simple Reinhard tone mapping
    color = color / (color + vec3<f32>(1.0));
    
    // Gamma correction
    color = pow(color, vec3<f32>(1.0 / lighting.gamma));
    
    return vec4<f32>(color, base_color.a);
}

// Simplified PBR fragment shader without IBL (for fallback)
@fragment
fn fs_pbr_simple(input: VertexOutput) -> @location(0) vec4<f32> {
    // Construct TBN matrix
    let T = normalize(input.world_tangent);
    let B = normalize(input.world_bitangent);
    let N = normalize(input.world_normal);
    let TBN = mat3x3<f32>(T, B, N);
    
    // Sample normal map
    let world_normal = sample_normal_map(input.uv, TBN);
    
    // Sample material properties
    var base_color = material.base_color;
    if (material.texture_flags & FLAG_BASE_COLOR) != 0u {
        base_color = base_color * textureSample(base_color_texture, material_sampler, input.uv);
    }
    
    var metallic = material.metallic;
    var roughness = clamp(material.roughness, 0.04, 1.0);
    
    if (material.texture_flags & FLAG_METALLIC_ROUGHNESS) != 0u {
        let mr_sample = textureSample(metallic_roughness_texture, material_sampler, input.uv);
        metallic = metallic * mr_sample.b;
        roughness = clamp(roughness * mr_sample.g, 0.04, 1.0);
    }
    
    var emissive = material.emissive;
    if (material.texture_flags & FLAG_EMISSIVE) != 0u {
        emissive = emissive * textureSample(emissive_texture, material_sampler, input.uv).rgb;
    }
    
    // Alpha testing
    if base_color.a < material.alpha_cutoff {
        discard;
    }
    
    // Lighting calculation
    let view_dir = normalize(lighting.camera_position - input.world_position);
    let light_dir = normalize(-lighting.light_direction);
    let half_dir = normalize(light_dir + view_dir);
    
    let n_dot_v = max(dot(world_normal, view_dir), 0.0);
    let n_dot_l = max(dot(world_normal, light_dir), 0.0);
    let n_dot_h = max(dot(world_normal, half_dir), 0.0);
    let v_dot_h = max(dot(view_dir, half_dir), 0.0);
    
    // Calculate F0
    let f0 = mix(vec3<f32>(0.04), base_color.rgb, metallic);
    
    // BRDF calculation
    var color = vec3<f32>(0.0);
    
    if n_dot_l > 0.0 {
        let D = distribution_ggx(n_dot_h, roughness);
        let G = geometry_smith(n_dot_v, n_dot_l, roughness);
        let F = fresnel_schlick(v_dot_h, f0);
        
        let specular = (D * G * F) / max(4.0 * n_dot_v * n_dot_l, 1e-6);
        
        let kS = F;
        let kD = (vec3<f32>(1.0) - kS) * (1.0 - metallic);
        let diffuse = kD * base_color.rgb / 3.14159265359;
        
        let radiance = lighting.light_color * lighting.light_intensity;
        color = (diffuse + specular) * radiance * n_dot_l;
    }
    
    // Simple ambient
    color = color + vec3<f32>(0.03) * base_color.rgb + emissive;
    
    // Tone mapping
    color = color * lighting.exposure;
    color = color / (color + vec3<f32>(1.0));
    color = pow(color, vec3<f32>(1.0 / lighting.gamma));
    
    return vec4<f32>(color, base_color.a);
}

// Debug fragment shaders for development
@fragment
fn fs_debug_normals(input: VertexOutput) -> @location(0) vec4<f32> {
    let T = normalize(input.world_tangent);
    let B = normalize(input.world_bitangent);
    let N = normalize(input.world_normal);
    let TBN = mat3x3<f32>(T, B, N);
    
    let world_normal = sample_normal_map(input.uv, TBN);
    return vec4<f32>(world_normal * 0.5 + 0.5, 1.0);
}

@fragment
fn fs_debug_metallic_roughness(input: VertexOutput) -> @location(0) vec4<f32> {
    var metallic = material.metallic;
    var roughness = material.roughness;
    
    if (material.texture_flags & FLAG_METALLIC_ROUGHNESS) != 0u {
        let mr_sample = textureSample(metallic_roughness_texture, material_sampler, input.uv);
        metallic = metallic * mr_sample.b;
        roughness = roughness * mr_sample.g;
    }
    
    return vec4<f32>(roughness, metallic, 0.0, 1.0);
}

@fragment
fn fs_debug_base_color(input: VertexOutput) -> @location(0) vec4<f32> {
    var base_color = material.base_color;
    if (material.texture_flags & FLAG_BASE_COLOR) != 0u {
        base_color = base_color * textureSample(base_color_texture, material_sampler, input.uv);
    }
    
    return vec4<f32>(base_color.rgb, 1.0);
}