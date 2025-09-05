//! Environment mapping and image-based lighting (IBL) shaders
//! 
//! Provides diffuse/specular sampling with roughness-based mip LOD for 
//! physically-based rendering with environment maps and cubemap textures.

struct Uniforms {
    model_matrix: mat4x4<f32>,
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    camera_position: vec4<f32>,
    light_direction: vec4<f32>,
    roughness: f32,
    metallic: f32,
    base_color: vec4<f32>,
    padding: f32,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) world_normal: vec3<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var env_texture: texture_2d<f32>;
@group(1) @binding(1) var env_sampler: sampler;
@group(1) @binding(2) var irradiance_texture: texture_2d<f32>;
@group(1) @binding(3) var irradiance_sampler: sampler;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    
    let world_position = uniforms.model_matrix * vec4<f32>(input.position, 1.0);
    output.world_position = world_position.xyz;
    output.clip_position = uniforms.projection_matrix * uniforms.view_matrix * world_position;
    output.uv = input.uv;
    
    // Transform normal to world space
    let normal_mat = mat3x3<f32>(
        uniforms.normal_matrix[0].xyz,
        uniforms.normal_matrix[1].xyz,
        uniforms.normal_matrix[2].xyz
    );
    output.world_normal = normalize(normal_mat * input.normal);
    
    return output;
}

// Convert 3D direction to equirectangular UV coordinates
fn direction_to_uv(direction: vec3<f32>) -> vec2<f32> {
    let phi = atan2(direction.z, direction.x);
    let theta = acos(direction.y);
    
    let u = (phi / (2.0 * 3.14159265359) + 0.5) % 1.0;
    let v = theta / 3.14159265359;
    
    return vec2<f32>(u, v);
}

// Sample environment map with bilinear filtering
fn sample_environment(direction: vec3<f32>) -> vec3<f32> {
    let uv = direction_to_uv(direction);
    let sample = textureSample(env_texture, env_sampler, uv);
    return sample.rgb;
}

// Sample environment map with specified mip level for roughness
fn sample_environment_lod(direction: vec3<f32>, mip_level: f32) -> vec3<f32> {
    let uv = direction_to_uv(direction);
    let sample = textureSampleLevel(env_texture, env_sampler, uv, mip_level);
    return sample.rgb;
}

// Sample irradiance map for diffuse lighting
fn sample_irradiance(normal: vec3<f32>) -> vec3<f32> {
    let uv = direction_to_uv(normal);
    let sample = textureSample(irradiance_texture, irradiance_sampler, uv);
    return sample.rgb;
}

// Fresnel-Schlick approximation
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// GGX distribution function
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let n_dot_h2 = n_dot_h * n_dot_h;
    
    let num = alpha2;
    var denom = (n_dot_h2 * (alpha2 - 1.0) + 1.0);
    denom = 3.14159265359 * denom * denom;
    
    return num / denom;
}

// Smith geometry function
fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    
    let ggx1 = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let ggx2 = n_dot_l / (n_dot_l * (1.0 - k) + k);
    
    return ggx1 * ggx2;
}

// Calculate reflection direction
fn reflect_direction(incident: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return incident - 2.0 * dot(incident, normal) * normal;
}

// Calculate mip level based on roughness for environment sampling
fn calculate_mip_level(roughness: f32, max_mip: f32) -> f32 {
    // Map roughness to mip level - higher roughness = higher mip (more blur)
    return roughness * max_mip;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let world_normal = normalize(input.world_normal);
    let view_direction = normalize(uniforms.camera_position.xyz - input.world_position);
    let reflection_direction = reflect_direction(-view_direction, world_normal);
    
    // Material properties
    let base_color = uniforms.base_color.rgb;
    let roughness = clamp(uniforms.roughness, 0.04, 1.0);
    let metallic = clamp(uniforms.metallic, 0.0, 1.0);
    
    // Calculate F0 (surface reflection at zero incidence)
    let dielectric_f0 = vec3<f32>(0.04);
    let f0 = mix(dielectric_f0, base_color, metallic);
    
    // Diffuse IBL contribution
    let irradiance = sample_irradiance(world_normal);
    let diffuse_color = base_color * (1.0 - metallic);
    let diffuse_ibl = diffuse_color * irradiance;
    
    // Specular IBL contribution
    let max_mip_level = 8.0; // Should match environment texture mip levels
    let mip_level = calculate_mip_level(roughness, max_mip_level);
    let prefilteredColor = sample_environment_lod(reflection_direction, mip_level);
    
    // Fresnel for IBL
    let n_dot_v = max(dot(world_normal, view_direction), 0.0);
    let fresnel_ibl = fresnel_schlick(n_dot_v, f0);
    
    // Combine diffuse and specular IBL
    let kS = fresnel_ibl;
    let kD = vec3<f32>(1.0) - kS;
    let kD_final = kD * (1.0 - metallic);
    
    // Simple BRDF approximation for IBL
    let specular_ibl = prefilteredColor * fresnel_ibl;
    let final_diffuse = kD_final * diffuse_ibl;
    
    // Add some ambient to prevent pure black
    let ambient = vec3<f32>(0.03) * base_color;
    
    let color = ambient + final_diffuse + specular_ibl;
    
    // Simple tone mapping
    let tone_mapped = color / (color + vec3<f32>(1.0));
    
    // Gamma correction
    let gamma_corrected = pow(tone_mapped, vec3<f32>(1.0 / 2.2));
    
    return vec4<f32>(gamma_corrected, uniforms.base_color.a);
}

// Skybox vertex shader
@vertex  
fn vs_skybox(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    
    // Remove translation from view matrix for skybox
    var view_no_translation = uniforms.view_matrix;
    view_no_translation[3][0] = 0.0;
    view_no_translation[3][1] = 0.0; 
    view_no_translation[3][2] = 0.0;
    
    let world_position = vec4<f32>(input.position, 1.0);
    output.world_position = input.position; // Use local position as direction
    output.clip_position = uniforms.projection_matrix * view_no_translation * world_position;
    output.clip_position = output.clip_position.xyww; // Ensure skybox renders at far plane
    output.uv = input.uv;
    output.world_normal = input.normal;
    
    return output;
}

// Skybox fragment shader
@fragment
fn fs_skybox(input: VertexOutput) -> @location(0) vec4<f32> {
    let direction = normalize(input.world_position);
    let color = sample_environment(direction);
    
    // Simple tone mapping for HDR environment
    let tone_mapped = color / (color + vec3<f32>(1.0));
    let gamma_corrected = pow(tone_mapped, vec3<f32>(1.0 / 2.2));
    
    return vec4<f32>(gamma_corrected, 1.0);
}

// Environment map convolution compute shader for preprocessing
@compute @workgroup_size(8, 8, 1)
fn cs_convolve_irradiance(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // This would be used for runtime convolution if needed
    // For now, we assume irradiance maps are precomputed on CPU
}