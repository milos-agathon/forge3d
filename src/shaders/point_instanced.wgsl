//! H11,H20: Instanced point shader with texture atlas support
//! GPU-based point sprite rendering with anti-aliasing and rotation

struct PointUniform {
    transform: mat4x4<f32>,
    viewport_size: vec2<f32>,
    pixel_scale: f32,
    debug_mode: u32,
    atlas_size: vec2<f32>,
    enable_clip_w_scaling: u32,
    depth_range: vec2<f32>,
    shape_mode: u32,        // H2/H6: point shape/material mode (0=circle,4=texture,5=sphere impostor)
    lod_threshold: f32,     // H2: pixel-size threshold for LOD simplification
}

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) size: f32,
    @location(2) color: vec4<f32>,
    @location(3) rotation: f32,
    @location(4) uv_offset: vec2<f32>,  // H21: texture atlas support
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec2<f32>,
    @location(1) point_center: vec2<f32>,
    @location(2) local_pos: vec2<f32>,    // Local position within point [-1, 1]
    @location(3) color: vec4<f32>,
    @location(4) size: f32,
    @location(5) uv: vec2<f32>,           // H21: texture coordinates
    @location(6) instance_id: u32,        // H5: instance id for picking
}

@group(0) @binding(0)
var<uniform> uniforms: PointUniform;

// Pick uniform (used only in fs_pick)
struct PickUniform {
    pick_id: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// Bind at a separate slot to keep normal pipeline layout unchanged
@group(0) @binding(3)
var<uniform> pick_uniform: PickUniform;

// H21: Optional texture atlas bindings
@group(0) @binding(1)
var atlas_texture: texture_2d<f32>;

@group(0) @binding(2)
var atlas_sampler: sampler;

@vertex
fn vs_main(
    vertex: VertexInput,
    @builtin(vertex_index) vertex_id: u32,
    @builtin(instance_index) instance_id: u32,
) -> VertexOutput {
    var out: VertexOutput;
    
    let point_center = vertex.position;
    let point_size_pixels = vertex.size;
    
    // Convert point size from pixels to world units
    let point_size_world = point_size_pixels / uniforms.pixel_scale;
    let half_size = point_size_world * 0.5;
    
    // Generate quad vertices (triangle strip)
    // vertex_id: 0=top-left, 1=bottom-left, 2=top-right, 3=bottom-right
    let x_offset = select(-half_size, half_size, (vertex_id & 2u) != 0u);
    let y_offset = select(-half_size, half_size, (vertex_id & 1u) != 0u);
    
    // Apply rotation if needed (H21)
    var local_offset = vec2<f32>(x_offset, y_offset);
    if (abs(vertex.rotation) > 0.001) {
        let cos_rot = cos(vertex.rotation);
        let sin_rot = sin(vertex.rotation);
        local_offset = vec2<f32>(
            local_offset.x * cos_rot - local_offset.y * sin_rot,
            local_offset.x * sin_rot + local_offset.y * cos_rot
        );
    }
    
    let world_pos = point_center + local_offset;
    
    // Transform to clip space
    out.clip_position = uniforms.transform * vec4<f32>(world_pos, 0.0, 1.0);
    out.world_pos = world_pos;
    out.point_center = point_center;
    out.color = vertex.color;
    out.size = point_size_pixels;
    
    // Local coordinates for shape rendering [-1, 1]
    out.local_pos = vec2<f32>(x_offset, y_offset) / half_size;
    
    // UV coordinates for texture atlas (H21)
    let u = select(0.0, 1.0, (vertex_id & 2u) != 0u);
    let v = select(0.0, 1.0, (vertex_id & 1u) != 0u);
    out.uv = vertex.uv_offset + vec2<f32>(u, v) * 0.1; // 0.1 = 1/10 for 10x10 atlas
    out.instance_id = instance_id;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = in.color;
    
    // Shape/material selection with LOD fallback
    // 5 = sphere impostor; if in.size < lod_threshold, fall back to circle
    if (uniforms.shape_mode == 5u && in.size >= uniforms.lod_threshold) {
        // Compute sphere shading only inside circle
        let r2 = dot(in.local_pos, in.local_pos);
        if (r2 > 1.0) { discard; }
        let z = sqrt(max(0.0, 1.0 - r2));
        let n = normalize(vec3<f32>(in.local_pos, z));
        // Fixed light direction
        let L = normalize(vec3<f32>(0.5, 0.5, 1.0));
        let diff = max(0.0, dot(n, L));
        let lit = color.rgb * (0.3 + 0.7 * diff);
        color = vec4<f32>(lit, in.color.a);
    } else if (uniforms.atlas_size.x > 1.0 && uniforms.shape_mode == 4u) {
        // Sample from texture atlas  
        let atlas_color = textureSample(atlas_texture, atlas_sampler, in.uv);
        // Blend atlas color with vertex color
        color = color * atlas_color;
    } else {
        // Standard circle rendering without atlas
        let dist = length(in.local_pos);
        let edge_softness = 2.0 / in.size;
        let alpha = 1.0 - smoothstep(1.0 - edge_softness, 1.0, dist);
        color.a *= alpha;
    }
    
    // H20: Debug mode coloring
    if ((uniforms.debug_mode & 1u) != 0u) {
        // Show bounding box
        let box_dist = max(abs(in.local_pos.x), abs(in.local_pos.y));
        if (box_dist > 0.9) {
            color = vec4<f32>(1.0, 0.0, 0.0, 1.0); // Red border
        }
    }
    if ((uniforms.debug_mode & 4u) != 0u) {
        // Color by depth
        let depth = in.clip_position.z / in.clip_position.w;
        color = vec4<f32>(depth, 0.5, 1.0 - depth, color.a);
    }
    
    // Early discard for fully transparent pixels
    if (color.a < 0.01) {
        discard;
    }
    
    // Apply sRGB gamma correction
    color = vec4<f32>(
        pow(color.r, 1.0 / 2.2),
        pow(color.g, 1.0 / 2.2),
        pow(color.b, 1.0 / 2.2),
        color.a
    );
    
    return color;
}

// Fragment entry for picking to an R32Uint target
@fragment
fn fs_pick(in: VertexOutput) -> @location(0) u32 {
    return pick_uniform.pick_id + in.instance_id;
}

// OIT output for points
struct OITOutput {
    @location(0) accum: vec4<f32>,
    @location(1) reveal: f32,
}

@fragment
fn fs_oit(in: VertexOutput) -> OITOutput {
    // Base alpha based on shape selection
    var alpha = 1.0;
    if (uniforms.shape_mode == 5u) {
        // Sphere impostor mask
        let r2 = dot(in.local_pos, in.local_pos);
        if (r2 > 1.0) { alpha = 0.0; }
    } else {
        let dist = length(in.local_pos);
        let edge_softness = 2.0 / in.size;
        alpha = 1.0 - smoothstep(1.0 - edge_softness, 1.0, dist);
    }

    var out: OITOutput;
    out.accum = vec4<f32>(in.color.rgb * alpha, alpha);
    out.reveal = 1.0 - alpha;
    return out;
}

// Alternative fragment shader for different point shapes (future use)
@fragment
fn fs_shape_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var alpha = 1.0;
    let edge_softness = 2.0 / in.size;
    
    // Square shape
    let square_dist = max(abs(in.local_pos.x), abs(in.local_pos.y));
    alpha = 1.0 - smoothstep(1.0 - edge_softness, 1.0, square_dist);
    
    // Diamond shape (Manhattan distance)
    // let diamond_dist = abs(in.local_pos.x) + abs(in.local_pos.y);
    // alpha = 1.0 - smoothstep(1.0 - edge_softness, 1.0, diamond_dist);
    
    var color = in.color;
    color.a *= alpha;
    
    if (color.a < 0.01) {
        discard;
    }
    
    // Apply sRGB gamma correction
    color = vec4<f32>(
        pow(color.r, 1.0 / 2.2),
        pow(color.g, 1.0 / 2.2),
        pow(color.b, 1.0 / 2.2),
        color.a
    );
    
    return color;
}
