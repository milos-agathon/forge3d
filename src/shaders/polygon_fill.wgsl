//! H5: Polygon fill shader with proper sRGB output
//! Supports filled polygons with holes via tessellation

struct PolygonUniform {
    transform: mat4x4<f32>,
    fill_color: vec4<f32>,
    stroke_color: vec4<f32>,
    stroke_width: f32,
    _pad0: f32,
    _pad1: f32, 
    _pad2: f32,
}

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: PolygonUniform;

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Transform to clip space
    let world_pos = vertex.position;
    out.clip_position = uniforms.transform * vec4<f32>(world_pos, 0.0, 1.0);
    out.world_pos = world_pos;
    out.uv = vertex.uv;
    
    return out;
}

@fragment  
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Basic filled polygon rendering
    // UV coordinates can be used for texture mapping in future versions
    
    // Write linear color; the Rgba8UnormSrgb target handles sRGB encoding.
    return uniforms.fill_color;
}