//! H8,H9: Anti-aliased line shader with GPU expansion
//! Instanced line segment rendering with smooth edges

struct LineUniform {
    transform: mat4x4<f32>,
    stroke_color: vec4<f32>,  // Default color (overridden per-instance)
    stroke_width: f32,        // Default width
    viewport_size: vec2<f32>, // For anti-aliasing calculations
    miter_limit: f32,         // H9: Miter limit for joins
    cap_style: u32,           // H9: LineCap style (0=butt, 1=round, 2=square)
    join_style: u32,          // H9: LineJoin style (0=miter, 1=bevel, 2=round)
    _pad0: f32,
    _pad1: f32,
}

struct VertexInput {
    @location(0) start_pos: vec2<f32>,
    @location(1) end_pos: vec2<f32>,
    @location(2) width: f32,
    @location(3) color: vec4<f32>,
    @location(4) miter_limit: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec2<f32>,
    @location(1) line_pos: vec2<f32>,     // Position along line segment
    @location(2) distance: f32,          // Distance from line center
    @location(3) width: f32,             // Line width for AA
    @location(4) color: vec4<f32>,       // Per-instance color
    @location(5) instance_id: u32,       // H5: instance id for picking
}

@group(0) @binding(0)
var<uniform> uniforms: LineUniform;

// Pick uniform (used only in fs_pick)
struct PickUniform {
    pick_id: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// Bind at a dedicated slot for picking pipeline
@group(0) @binding(3)
var<uniform> pick_uniform: PickUniform;

@vertex
fn vs_main(
    vertex: VertexInput,
    @builtin(vertex_index) vertex_id: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    
    // Calculate line direction and normal
    let line_vec = vertex.end_pos - vertex.start_pos;
    let line_length = length(line_vec);
    
    // Handle degenerate lines
    if (line_length < 0.001) {
        out.clip_position = vec4<f32>(0.0);
        return out;
    }
    
    let line_dir = line_vec / line_length;
    let line_normal = vec2<f32>(-line_dir.y, line_dir.x);
    
    // Expand line segment to quad (triangle strip)
    // vertex_id: 0=start-top, 1=start-bottom, 2=end-top, 3=end-bottom
    let half_width = vertex.width * 0.5;
    let is_end = (vertex_id & 2u) != 0u;
    let is_top = (vertex_id & 1u) == 0u;
    
    // Calculate quad vertices with anti-aliasing margin
    let aa_margin = 1.0; // 1 pixel margin for AA
    let expanded_width = half_width + aa_margin;
    
    let base_pos = select(vertex.start_pos, vertex.end_pos, is_end);
    let normal_offset = select(-expanded_width, expanded_width, is_top);
    let world_pos = base_pos + line_normal * normal_offset;
    
    // Transform to clip space
    out.clip_position = uniforms.transform * vec4<f32>(world_pos, 0.0, 1.0);
    out.world_pos = world_pos;
    
    // Calculate line-local coordinates for fragment shader
    let t = select(0.0, 1.0, is_end);
    out.line_pos = vec2<f32>(t * line_length, normal_offset);
    out.distance = abs(normal_offset);
    out.width = half_width;
    out.color = vertex.color;
    out.instance_id = instance_index;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // H9: Enhanced anti-aliased line rendering with caps and joins
    let edge_softness = 1.0; // Softness in pixels
    
    // Distance from line center to edge
    let distance_from_center = abs(in.distance);
    let line_half_width = in.width;
    
    var alpha = 1.0;
    
    // H9: Apply cap style at line ends
    let line_length = in.line_pos.x;
    let is_at_start = line_length < edge_softness;
    let is_at_end = line_length > (length(in.line_pos) - edge_softness);
    
    if (is_at_start || is_at_end) {
        // Apply cap style
        if (uniforms.cap_style == 0u) {
            // Butt cap - no extension
            alpha = select(1.0, 0.0, is_at_start && line_length < 0.0);
            alpha = select(alpha, 0.0, is_at_end && line_length > length(in.line_pos));
        } else if (uniforms.cap_style == 1u) {
            // Round cap - circular extension
            let cap_center = select(vec2<f32>(0.0), vec2<f32>(length(in.line_pos)), is_at_end);
            let cap_distance = length(in.line_pos - cap_center);
            alpha = 1.0 - smoothstep(line_half_width - edge_softness, line_half_width, cap_distance);
        } else if (uniforms.cap_style == 2u) {
            // Square cap - rectangular extension
            let extension = line_half_width;
            let extended_start = -extension;
            let extended_end = length(in.line_pos) + extension;
            alpha = select(1.0, 0.0, line_length < extended_start || line_length > extended_end);
        }
    }
    
    // H9: Apply join style for line connections
    // Join logic applies when multiple line segments meet at vertices
    // Note: Join rendering would typically require additional geometry data
    // For now, apply miter limiting to sharp angles
    if (uniforms.join_style == 0u) {
        // Miter join - limit sharp angles based on miter_limit
        let max_miter_distance = line_half_width * uniforms.miter_limit;
        if (distance_from_center > max_miter_distance) {
            alpha *= 1.0 - smoothstep(max_miter_distance - edge_softness, max_miter_distance, distance_from_center);
        }
    } else if (uniforms.join_style == 1u) {
        // Bevel join - cut off sharp angles
        let bevel_threshold = line_half_width * 1.5;
        if (distance_from_center > bevel_threshold) {
            alpha = 0.0;
        }
    }
    // Round join (2u) uses natural circular falloff, no additional processing needed
    
    // Calculate base alpha for anti-aliasing
    let edge_distance = distance_from_center - line_half_width;
    let base_alpha = 1.0 - smoothstep(0.0, edge_softness, edge_distance);
    
    // Combine cap/join alpha with base alpha
    alpha *= base_alpha;
    
    // Apply alpha to color
    var color = in.color;
    color.a *= alpha;
    
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

// OIT output type: accumulation color (rgba: rgb accum, a weight) and revealage (single channel)
struct OITOutput {
    @location(0) accum: vec4<f32>,
    @location(1) reveal: f32,
}

// Fragment entry for weighted OIT MRT
@fragment
fn fs_oit(in: VertexOutput) -> OITOutput {
    // Recompute alpha similarly to fs_main but keep in linear space and without gamma correction
    let edge_softness = 1.0;
    let distance_from_center = abs(in.distance);
    let line_half_width = in.width;

    var alpha = 1.0;
    // Base AA alpha
    let edge_distance = distance_from_center - line_half_width;
    let base_alpha = 1.0 - smoothstep(0.0, edge_softness, edge_distance);
    alpha *= base_alpha;

    // Compose outputs
    let color = in.color;
    var out: OITOutput;
    out.accum = vec4<f32>(color.rgb * alpha, alpha);
    out.reveal = 1.0 - alpha;
    return out;
}