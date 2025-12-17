// src/viewer/terrain/shader_pbr.rs
// Enhanced PBR-like shader for terrain rendering with better lighting

pub const TERRAIN_PBR_SHADER: &str = r#"
// Terrain PBR Viewer Shader
// Enhanced lighting with Blinn-Phong, soft shadows, and improved materials

struct Uniforms {
    view_proj: mat4x4<f32>,
    sun_dir: vec4<f32>,
    terrain_params: vec4<f32>,  // min_h, h_range, terrain_width, z_scale
    lighting: vec4<f32>,        // sun_intensity, ambient, shadow_intensity, water_level
    background: vec4<f32>,      // r, g, b, _
    water_color: vec4<f32>,     // r, g, b, _
    // PBR params
    pbr_params: vec4<f32>,      // exposure, normal_strength, ibl_intensity, _
    camera_pos: vec4<f32>,      // camera world position
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var heightmap: texture_2d<f32>;
@group(0) @binding(2) var height_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) raw_height: f32,
};

@vertex
fn vs_main(@location(0) pos: vec2<f32>, @location(1) uv: vec2<f32>) -> VertexOutput {
    let dims = textureDimensions(heightmap);
    let max_texel = vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1);
    let texel = clamp(
        vec2<i32>(i32(uv.x * f32(dims.x)), i32(uv.y * f32(dims.y))),
        vec2<i32>(0, 0),
        max_texel
    );
    let h = textureLoad(heightmap, texel, 0).r;
    
    let min_h = u.terrain_params.x;
    let terrain_width = u.terrain_params.z;
    let z_scale = u.terrain_params.w;
    
    let world_y = (h - min_h) * z_scale;
    
    let world_x = uv.x * terrain_width;
    let world_z = uv.y * terrain_width;
    
    var out: VertexOutput;
    out.world_pos = vec3<f32>(world_x, world_y, world_z);
    out.position = u.view_proj * vec4<f32>(out.world_pos, 1.0);
    out.uv = uv;
    out.raw_height = h;
    return out;
}

// Improved normal calculation from height gradient
fn compute_normal(world_pos: vec3<f32>) -> vec3<f32> {
    let dx = dpdx(world_pos);
    let dy = dpdy(world_pos);
    var n = normalize(cross(dy, dx));
    // Amplify normal detail based on pbr_params.y (normal_strength)
    let strength = u.pbr_params.y;
    n.x *= strength;
    n.z *= strength;
    return normalize(n);
}

// Blinn-Phong specular
fn blinn_phong_specular(normal: vec3<f32>, light_dir: vec3<f32>, view_dir: vec3<f32>, shininess: f32) -> f32 {
    let half_dir = normalize(light_dir + view_dir);
    let spec = pow(max(dot(normal, half_dir), 0.0), shininess);
    return spec;
}

// Soft shadow approximation based on normal and light direction
fn soft_shadow_factor(normal: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    let ndotl = dot(normal, light_dir);
    // Smooth transition from lit to shadow
    let shadow = smoothstep(-0.1, 0.3, ndotl);
    return shadow;
}

// Height-based material (PBR-like with roughness variation)
fn get_material(h_norm: f32, slope: f32) -> vec4<f32> {
    // albedo.rgb, roughness
    var albedo: vec3<f32>;
    var roughness: f32;
    
    // Vegetation (low elevation, gentle slopes)
    if h_norm < 0.25 && slope < 0.5 {
        albedo = vec3<f32>(0.15, 0.35, 0.12); // Dark green grass
        roughness = 0.85;
    }
    // Grass to rock transition
    else if h_norm < 0.4 {
        let t = (h_norm - 0.25) / 0.15;
        albedo = mix(vec3<f32>(0.2, 0.4, 0.15), vec3<f32>(0.4, 0.35, 0.3), t);
        roughness = mix(0.8, 0.7, t);
    }
    // Rocky terrain (mid elevation or steep slopes)
    else if h_norm < 0.7 || slope > 0.6 {
        albedo = vec3<f32>(0.45, 0.4, 0.35); // Gray-brown rock
        roughness = 0.6;
    }
    // Snow/ice (high elevation)
    else {
        let snow_factor = smoothstep(0.7, 0.85, h_norm);
        albedo = mix(vec3<f32>(0.5, 0.45, 0.4), vec3<f32>(0.95, 0.95, 0.98), snow_factor);
        roughness = mix(0.5, 0.3, snow_factor); // Snow is slightly glossy
    }
    
    // Add slope-based rock exposure
    if slope > 0.4 && h_norm < 0.8 {
        let rock_blend = smoothstep(0.4, 0.7, slope);
        albedo = mix(albedo, vec3<f32>(0.4, 0.38, 0.35), rock_blend * 0.6);
        roughness = mix(roughness, 0.65, rock_blend);
    }
    
    return vec4<f32>(albedo, roughness);
}

// ACES tonemapping
fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

// Sky gradient for ambient
fn sky_ambient(normal: vec3<f32>) -> vec3<f32> {
    let sky_color = vec3<f32>(0.4, 0.6, 0.9);
    let ground_color = vec3<f32>(0.15, 0.1, 0.08);
    let sky_factor = normal.y * 0.5 + 0.5;
    return mix(ground_color, sky_color, sky_factor);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sun_intensity = u.lighting.x;
    let ambient_strength = u.lighting.y;
    let shadow_strength = u.lighting.z;
    let water_level = u.lighting.w;
    let exposure = u.pbr_params.x;
    let ibl_intensity = u.pbr_params.z;
    
    // Check if below water level
    let is_water = in.raw_height < water_level;
    
    // Normalized height for material selection
    let h_norm = clamp((in.raw_height - u.terrain_params.x) / max(u.terrain_params.y, 1.0), 0.0, 1.0);
    
    // Compute surface normal
    let normal = compute_normal(in.world_pos);
    
    // Slope for material blending (0 = flat, 1 = vertical)
    let slope = 1.0 - abs(normal.y);
    
    // Get material properties
    let material = get_material(h_norm, slope);
    var albedo = material.rgb;
    let roughness = material.a;
    
    // Water handling
    if is_water {
        albedo = u.water_color.rgb;
    }
    
    // Sun lighting
    let sun_dir = normalize(u.sun_dir.xyz);
    let ndotl = max(dot(normal, sun_dir), 0.0);
    
    // Soft shadow
    let shadow = soft_shadow_factor(normal, sun_dir);
    let shadow_term = mix(1.0, shadow, shadow_strength);
    
    // View direction (approximate from camera_pos or use fragment position)
    let view_dir = normalize(u.camera_pos.xyz - in.world_pos);
    
    // Diffuse lighting
    let sun_color = vec3<f32>(1.0, 0.95, 0.85); // Warm sunlight
    let diffuse = albedo * sun_color * ndotl * sun_intensity * shadow_term;
    
    // Specular (Blinn-Phong)
    let shininess = mix(8.0, 64.0, 1.0 - roughness);
    var specular = blinn_phong_specular(normal, sun_dir, view_dir, shininess);
    specular *= sun_intensity * shadow_term * (1.0 - roughness) * 0.3;
    
    // Water specular boost
    if is_water {
        specular = blinn_phong_specular(vec3<f32>(0.0, 1.0, 0.0), sun_dir, view_dir, 128.0);
        specular *= sun_intensity * 0.8;
    }
    
    // Ambient/IBL approximation
    let ambient_color = sky_ambient(normal) * albedo * ambient_strength * ibl_intensity;
    
    // Combine lighting
    var color = diffuse + vec3<f32>(specular) + ambient_color;
    
    // Apply exposure and tonemapping
    color = color * exposure;
    color = aces_tonemap(color);
    
    // Gamma correction (linear to sRGB)
    color = pow(color, vec3<f32>(1.0 / 2.2));
    
    return vec4<f32>(color, 1.0);
}
"#;
