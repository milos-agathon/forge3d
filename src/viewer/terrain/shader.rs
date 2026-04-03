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

fn terrain_depth_from_dims(dims: vec2<f32>) -> f32 {
    return u.terrain_params.z * dims.y / max(dims.x, 1.0);
}

fn height_to_world_y(h: f32) -> f32 {
    let min_h = u.terrain_params.x;
    let h_range = u.terrain_params.y;
    let terrain_width = u.terrain_params.z;
    let z_scale = u.terrain_params.w;
    let h_normalized = (h - min_h) / max(h_range, 1.0);
    return h_normalized * terrain_width * z_scale * 0.001;
}

fn compute_heightmap_normal(uv: vec2<f32>) -> vec3<f32> {
    let dims_u = textureDimensions(heightmap);
    let dims = vec2<f32>(f32(dims_u.x), f32(dims_u.y));
    let terrain_depth = terrain_depth_from_dims(dims);
    let max_texel = vec2<i32>(i32(dims_u.x) - 1, i32(dims_u.y) - 1);
    let texel = clamp(
        vec2<i32>(
            i32(uv.x * max(f32(dims_u.x) - 1.0, 0.0)),
            i32(uv.y * max(f32(dims_u.y) - 1.0, 0.0)),
        ),
        vec2<i32>(0, 0),
        max_texel
    );
    let left = vec2<i32>(max(texel.x - 1, 0), texel.y);
    let right = vec2<i32>(min(texel.x + 1, max_texel.x), texel.y);
    let down = vec2<i32>(texel.x, max(texel.y - 1, 0));
    let up = vec2<i32>(texel.x, min(texel.y + 1, max_texel.y));
    let step_x = u.terrain_params.z / max(f32(dims_u.x) - 1.0, 1.0);
    let step_z = terrain_depth / max(f32(dims_u.y) - 1.0, 1.0);
    let h_l = height_to_world_y(textureLoad(heightmap, left, 0).r);
    let h_r = height_to_world_y(textureLoad(heightmap, right, 0).r);
    let h_d = height_to_world_y(textureLoad(heightmap, down, 0).r);
    let h_u = height_to_world_y(textureLoad(heightmap, up, 0).r);
    let tangent_x = vec3<f32>(2.0 * step_x, h_r - h_l, 0.0);
    let tangent_z = vec3<f32>(0.0, h_u - h_d, 2.0 * step_z);
    return normalize(cross(tangent_z, tangent_x));
}

@vertex
fn vs_main(@location(0) pos: vec2<f32>, @location(1) uv: vec2<f32>) -> VertexOutput {
    let dims = vec2<f32>(textureDimensions(heightmap));
    let terrain_depth = terrain_depth_from_dims(dims);
    let max_texel = vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1);
    let texel = clamp(
        vec2<i32>(i32(uv.x * f32(dims.x)), i32(uv.y * f32(dims.y))),
        vec2<i32>(0, 0),
        max_texel
    );
    let h = textureLoad(heightmap, texel, 0).r;
    let terrain_width = u.terrain_params.z;
    let world_y = height_to_world_y(h);

    let world_x = uv.x * terrain_width;
    let world_z = uv.y * terrain_depth;
    
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
    
    let normal = compute_heightmap_normal(in.uv);
    
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
