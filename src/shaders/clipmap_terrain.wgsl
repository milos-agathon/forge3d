// P2.1/M5: Clipmap terrain shader with geo-morphing support.
//
// This shader renders nested-ring clipmap terrain with smooth LOD transitions
// via vertex geo-morphing. Height values are blended between fine and coarse
// LOD levels based on the morph_weight attribute.
//
// Vertex Attributes:
//   @location(0) position: vec2<f32>   - XZ world position
//   @location(1) uv: vec2<f32>         - Heightmap UV [0,1]
//   @location(2) morph_data: vec2<f32> - x=morph_weight, y=ring_index
//
// Morph weight interpretation:
//   0.0 = use fine LOD height
//   1.0 = use coarse LOD height (fully morphed)
//   <0  = skirt vertex (apply depth offset)

struct ClipmapUniforms {
    view_proj: mat4x4<f32>,
    terrain_params: vec4<f32>,  // x=min_h, y=h_range, z=terrain_width, w=z_scale
    camera_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    lighting: vec4<f32>,        // x=sun_intensity, y=ambient, z=shadow_intensity, w=_
    skirt_params: vec4<f32>,    // x=skirt_depth, y=morph_range, z=_, w=_
}

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) morph_data: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: ClipmapUniforms;
@group(0) @binding(1) var height_tex: texture_2d<f32>;
@group(0) @binding(2) var height_samp: sampler;

// Snap UV to coarser grid for morphing
fn snap_to_coarser_grid(uv: vec2<f32>, ring_index: u32) -> vec2<f32> {
    let lod_scale = f32(1u << ring_index);
    let tex_size = vec2<f32>(textureDimensions(height_tex, 0));
    let coarse_texel_size = lod_scale / tex_size;
    return floor(uv / coarse_texel_size) * coarse_texel_size;
}

// Sample height with optional LOD bias
fn sample_height(uv: vec2<f32>, lod: f32) -> f32 {
    return textureSampleLevel(height_tex, height_samp, uv, lod).r;
}

// Calculate terrain normal from heightmap using central differences
fn calculate_normal(uv: vec2<f32>, texel_size: vec2<f32>) -> vec3<f32> {
    let h_l = sample_height(uv - vec2<f32>(texel_size.x, 0.0), 0.0);
    let h_r = sample_height(uv + vec2<f32>(texel_size.x, 0.0), 0.0);
    let h_d = sample_height(uv - vec2<f32>(0.0, texel_size.y), 0.0);
    let h_u = sample_height(uv + vec2<f32>(0.0, texel_size.y), 0.0);
    
    let z_scale = uniforms.terrain_params.w;
    let terrain_width = uniforms.terrain_params.z;
    let cell_size = terrain_width / f32(textureDimensions(height_tex, 0).x);
    
    let dx = (h_r - h_l) * z_scale / (2.0 * cell_size);
    let dz = (h_u - h_d) * z_scale / (2.0 * cell_size);
    
    return normalize(vec3<f32>(-dx, 1.0, -dz));
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    let morph_weight = in.morph_data.x;
    let ring_index = u32(in.morph_data.y);
    let is_skirt = morph_weight < 0.0;
    
    // Sample height at current LOD
    let height_fine = sample_height(in.uv, 0.0);
    
    // For morphing, sample at coarser LOD and blend
    var height: f32;
    if is_skirt {
        // Skirt vertices use fine height but get depth offset
        height = height_fine;
    } else if morph_weight > 0.001 {
        // Blend between fine and coarse height for geo-morphing
        let coarse_uv = snap_to_coarser_grid(in.uv, ring_index + 1u);
        let height_coarse = sample_height(coarse_uv, f32(ring_index + 1u));
        height = mix(height_fine, height_coarse, morph_weight);
    } else {
        height = height_fine;
    }
    
    // Transform height to world space
    let min_h = uniforms.terrain_params.x;
    let h_range = uniforms.terrain_params.y;
    let z_scale = uniforms.terrain_params.w;
    let world_y = (height - min_h) / max(h_range, 0.001) * h_range * z_scale;
    
    // Apply skirt depth offset
    var final_y = world_y;
    if is_skirt {
        final_y -= uniforms.skirt_params.x;
    }
    
    let world_pos = vec3<f32>(in.position.x, final_y, in.position.y);
    out.world_pos = world_pos;
    out.clip_position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = in.uv;
    
    // Calculate normal
    let tex_size = vec2<f32>(textureDimensions(height_tex, 0));
    let texel_size = 1.0 / tex_size;
    out.normal = calculate_normal(in.uv, texel_size);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sun_dir = normalize(uniforms.sun_dir.xyz);
    let normal = normalize(in.normal);
    
    // Simple Lambert shading
    let n_dot_l = max(dot(normal, sun_dir), 0.0);
    let sun_intensity = uniforms.lighting.x;
    let ambient = uniforms.lighting.y;
    
    let diffuse = n_dot_l * sun_intensity;
    let lighting = diffuse + ambient;
    
    // Height-based coloring
    let min_h = uniforms.terrain_params.x;
    let h_range = uniforms.terrain_params.y;
    let z_scale = uniforms.terrain_params.w;
    let normalized_height = (in.world_pos.y / z_scale) / max(h_range, 0.001);
    
    // Simple terrain colormap
    let low_color = vec3<f32>(0.2, 0.4, 0.1);   // Green valleys
    let mid_color = vec3<f32>(0.5, 0.4, 0.3);   // Brown slopes
    let high_color = vec3<f32>(0.9, 0.9, 0.95); // Snow peaks
    
    var base_color: vec3<f32>;
    if normalized_height < 0.3 {
        base_color = mix(low_color, mid_color, normalized_height / 0.3);
    } else if normalized_height < 0.7 {
        base_color = mix(mid_color, high_color, (normalized_height - 0.3) / 0.4);
    } else {
        base_color = high_color;
    }
    
    let final_color = base_color * lighting;
    return vec4<f32>(final_color, 1.0);
}
