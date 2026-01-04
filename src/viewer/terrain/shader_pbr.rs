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
    // Lens effects: vignette_strength, vignette_radius, vignette_softness, _
    lens_params: vec4<f32>,
    // Screen dimensions for UV calculation
    screen_dims: vec4<f32>,     // width, height, _, _
    // Overlay params: enabled (>0.5), opacity, blend_mode (0=normal, 1=multiply, 2=overlay), solid (>0.5)
    overlay_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var heightmap: texture_2d<f32>;
@group(0) @binding(2) var height_sampler: sampler;
@group(0) @binding(3) var height_ao_tex: texture_2d<f32>;
@group(0) @binding(4) var sun_vis_tex: texture_2d<f32>;
// Overlay texture and sampler for lit draped overlays
@group(0) @binding(5) var overlay_tex: texture_2d<f32>;
@group(0) @binding(6) var overlay_sampler: sampler;

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

// Height-based material (PBR-like with Imhof palette and roughness variation)
fn get_material(h_norm: f32, slope: f32) -> vec4<f32> {
    // albedo.rgb, roughness
    var albedo: vec3<f32>;
    var roughness: f32;
    
    // Imhof palette: green valleys -> brown slopes -> white peaks
    if h_norm < 0.3 {
        // Green valleys
        albedo = mix(vec3<f32>(0.2, 0.5, 0.2), vec3<f32>(0.4, 0.6, 0.3), h_norm / 0.3);
        roughness = 0.8;
    } else if h_norm < 0.7 {
        // Brown slopes
        albedo = mix(vec3<f32>(0.4, 0.6, 0.3), vec3<f32>(0.5, 0.4, 0.3), (h_norm - 0.3) / 0.4);
        roughness = 0.7;
    } else {
        // White peaks
        albedo = mix(vec3<f32>(0.5, 0.4, 0.3), vec3<f32>(0.95, 0.95, 0.95), (h_norm - 0.7) / 0.3);
        roughness = 0.4;
    }
    
    // Add slope-based rock exposure (darkens steep areas)
    if slope > 0.5 && h_norm < 0.8 {
        let rock_blend = smoothstep(0.5, 0.75, slope);
        albedo = mix(albedo, vec3<f32>(0.35, 0.32, 0.28), rock_blend * 0.4);
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
    
    // === OVERLAY BLENDING (before lighting, so overlays are lit) ===
    let overlay_enabled = u.overlay_params.x > 0.5;
    let overlay_opacity = u.overlay_params.y;
    let overlay_blend_mode = u.overlay_params.z;
    let solid_surface = u.overlay_params.w > 0.5;
    
    // Sample overlay texture for solid check and blending
    let overlay = textureSample(overlay_tex, overlay_sampler, in.uv);
    
    // When solid=false and overlay is enabled, discard fragments where overlay alpha is near 0
    // This hides the base surface outside the region of interest (like rayshader solid=FALSE)
    // Use low threshold (0.01) - the overlay image should have alpha=1.0 for valid areas
    // and alpha=0.0 for areas to hide. Only truly transparent pixels should be discarded.
    if overlay_enabled && !solid_surface && overlay.a < 0.01 {
        discard;
    }
    
    if overlay_enabled && overlay_opacity > 0.001 {
        let blend_alpha = overlay.a * overlay_opacity;
        
        if blend_alpha > 0.01 {
            // Apply blend mode (overlay already sampled above for solid check)
            if overlay_blend_mode < 0.5 {
                // Normal blend: fully replace albedo with overlay when alpha is high
                // For land cover overlays, we want the categorical colors to show through
                albedo = mix(albedo, overlay.rgb, blend_alpha);
            } else if overlay_blend_mode < 1.5 {
                // Multiply blend
                let multiplied = albedo * overlay.rgb;
                albedo = mix(albedo, multiplied, blend_alpha);
            } else {
                // Overlay blend (Photoshop-style)
                let base = albedo;
                var blended: vec3<f32>;
                // Overlay formula: 2*base*blend if base < 0.5, else 1 - 2*(1-base)*(1-blend)
                blended.r = select(1.0 - 2.0 * (1.0 - base.r) * (1.0 - overlay.r), 2.0 * base.r * overlay.r, base.r < 0.5);
                blended.g = select(1.0 - 2.0 * (1.0 - base.g) * (1.0 - overlay.g), 2.0 * base.g * overlay.g, base.g < 0.5);
                blended.b = select(1.0 - 2.0 * (1.0 - base.b) * (1.0 - overlay.b), 2.0 * base.b * overlay.b, base.b < 0.5);
                albedo = mix(albedo, blended, blend_alpha);
            }
        }
    }
    // === END OVERLAY BLENDING ===
    
    // Sun lighting
    let sun_dir = normalize(u.sun_dir.xyz);
    let ndotl = max(dot(normal, sun_dir), 0.0);
    
    // Sample heightfield AO (use textureLoad for R32Float)
    let ao_dims = textureDimensions(height_ao_tex, 0);
    let ao_pixel = vec2<i32>(in.uv * vec2<f32>(ao_dims));
    let ao_clamped = clamp(ao_pixel, vec2<i32>(0), vec2<i32>(ao_dims) - vec2<i32>(1));
    let height_ao = textureLoad(height_ao_tex, ao_clamped, 0).r;
    
    // Sample sun visibility (use textureLoad for R32Float)
    let sv_dims = textureDimensions(sun_vis_tex, 0);
    let sv_pixel = vec2<i32>(in.uv * vec2<f32>(sv_dims));
    let sv_clamped = clamp(sv_pixel, vec2<i32>(0), vec2<i32>(sv_dims) - vec2<i32>(1));
    let sun_vis = textureLoad(sun_vis_tex, sv_clamped, 0).r;
    
    // Soft shadow
    let shadow = soft_shadow_factor(normal, sun_dir);
    // Combine soft shadow with sun visibility (multiplicative)
    let shadow_term = mix(1.0, shadow * sun_vis, shadow_strength);
    
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
    
    // Ambient/IBL approximation - modulated by heightfield AO
    let ambient_color = sky_ambient(normal) * albedo * ambient_strength * ibl_intensity * height_ao;
    
    // Combine lighting
    var color = diffuse + vec3<f32>(specular) + ambient_color;
    
    // Apply exposure and tonemapping
    color = color * exposure;
    color = aces_tonemap(color);
    
    // Gamma correction (linear to sRGB)
    color = pow(color, vec3<f32>(1.0 / 2.2));
    
    // Apply vignette (lens effect)
    let vignette_strength = u.lens_params.x;
    let vignette_radius = u.lens_params.y;
    let vignette_softness = u.lens_params.z;
    if vignette_strength > 0.001 {
        // Calculate screen UV from fragment position
        let screen_uv = in.position.xy / u.screen_dims.xy;
        // Distance from center (0.5, 0.5)
        let center_dist = length(screen_uv - vec2<f32>(0.5));
        // Vignette falloff
        let vignette = 1.0 - smoothstep(vignette_radius - vignette_softness, vignette_radius + vignette_softness, center_dist * 2.0);
        // Apply vignette darkening
        color = color * mix(1.0, vignette, vignette_strength);
    }
    
    return vec4<f32>(color, 1.0);
}
"#;
