// src/viewer/terrain/shader.rs
// WGSL shader for terrain rendering

pub const TERRAIN_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    sun_dir: vec4<f32>,
    terrain_params: vec4<f32>,  // min_h, h_range, terrain_width, z_scale
    lighting: vec4<f32>,        // sun_intensity, ambient, shadow_intensity, water_level
    background: vec4<f32>,      // r, g, b, _
    water_color: vec4<f32>,     // r, g, b, _
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
    let dims = vec2<f32>(textureDimensions(heightmap));
    let max_texel = vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1);
    let texel = clamp(
        vec2<i32>(i32(uv.x * f32(dims.x)), i32(uv.y * f32(dims.y))),
        vec2<i32>(0, 0),
        max_texel
    );
    let h = textureLoad(heightmap, texel, 0).r;
    
    let min_h = u.terrain_params.x;
    let h_range = u.terrain_params.y;
    let terrain_width = u.terrain_params.z;
    let z_scale = u.terrain_params.w;
    
    // Normalize height to 0-1 range, base at Y=0
    let h_normalized = (h - min_h) / max(h_range, 1.0);
    // Minimal vertical scale (0.001) creates very thin relief surface
    let world_y = h_normalized * terrain_width * z_scale * 0.001;
    
    let world_x = uv.x * terrain_width;
    let world_z = uv.y * terrain_width;
    
    var out: VertexOutput;
    out.world_pos = vec3<f32>(world_x, world_y, world_z);
    out.position = u.view_proj * vec4<f32>(out.world_pos, 1.0);
    out.uv = uv;
    out.raw_height = h;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sun_intensity = u.lighting.x;
    let ambient = u.lighting.y;
    let shadow_strength = u.lighting.z;
    let water_level = u.lighting.w;
    
    // Check if below water level
    let is_water = in.raw_height < water_level;
    
    // Simple height-based coloring with sun shading
    let h_norm = clamp((in.raw_height - u.terrain_params.x) / max(u.terrain_params.y, 1.0), 0.0, 1.0);
    
    // Terrain colormap (green valleys, brown slopes, white peaks)
    var color: vec3<f32>;
    if is_water {
        color = u.water_color.rgb;
    } else if h_norm < 0.3 {
        color = mix(vec3<f32>(0.2, 0.5, 0.2), vec3<f32>(0.4, 0.6, 0.3), h_norm / 0.3);
    } else if h_norm < 0.7 {
        color = mix(vec3<f32>(0.4, 0.6, 0.3), vec3<f32>(0.5, 0.4, 0.3), (h_norm - 0.3) / 0.4);
    } else {
        color = mix(vec3<f32>(0.5, 0.4, 0.3), vec3<f32>(0.95, 0.95, 0.95), (h_norm - 0.7) / 0.3);
    }
    
    // Approximate normal from height gradient (finite differences via dFdx/dFdy)
    let dx = dpdx(in.world_pos);
    let dy = dpdy(in.world_pos);
    let normal = normalize(cross(dy, dx));
    
    // Diffuse lighting with shadow
    let sun_dir = normalize(u.sun_dir.xyz);
    let ndotl = max(dot(normal, sun_dir), 0.0);
    
    // Shadow darkening for faces away from sun
    let shadow = mix(1.0, 1.0 - shadow_strength, 1.0 - ndotl);
    
    // Final lighting
    let diffuse = ndotl * sun_intensity;
    let lit = ambient + (1.0 - ambient) * diffuse * shadow;
    
    // Water gets specular highlight
    var final_color = color * lit;
    if is_water {
        let view_dir = normalize(-in.world_pos);
        let reflect_dir = reflect(-sun_dir, vec3<f32>(0.0, 1.0, 0.0));
        let spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0);
        final_color = final_color + vec3<f32>(spec * sun_intensity * 0.5);
    }
    
    return vec4<f32>(final_color, 1.0);
}
"#;
