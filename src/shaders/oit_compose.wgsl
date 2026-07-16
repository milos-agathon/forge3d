//! H16: Order Independent Transparency composition shader
//! Final compositing of weighted OIT accumulation buffers

@group(0) @binding(0)
var color_accumulation: texture_2d<f32>;

@group(0) @binding(1)
var reveal_accumulation: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Generate fullscreen triangle
    // Vertex 0: (-1, -1)
    // Vertex 1: ( 3, -1)  
    // Vertex 2: (-1,  3)
    let x = f32((vertex_index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vertex_index & 2u) * 2.0 - 1.0;
    
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x, -y) * 0.5 + 0.5;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Resolve one accumulation texel into the corresponding output pixel.
    // These attachments have identical extents; filtering is both unnecessary
    // and harmful here (Metal may return poison values for the float resolve
    // sample). Fragment position is already in framebuffer pixel coordinates.
    let pixel = vec2<i32>(in.clip_position.xy);
    let accum_color = textureLoad(color_accumulation, pixel, 0);
    let reveal = textureLoad(reveal_accumulation, pixel, 0).r;
    
    // Weighted OIT resolve
    // accum_color.rgb contains weighted color sum
    // accum_color.a contains weight sum  
    // reveal contains product of (1 - alpha_i)
    
    // Avoid division by zero
    let epsilon = 1e-5;
    
    // Calculate final color
    var final_color: vec3<f32>;
    if (accum_color.a > epsilon) {
        // Normalize weighted color by total weight
        final_color = accum_color.rgb / accum_color.a;
    } else {
        // No transparent fragments
        final_color = vec3<f32>(0.0);
    }
    
    // Calculate final alpha
    // reveal is product of (1 - alpha_i) for all fragments
    // So final alpha = 1 - reveal
    let final_alpha = 1.0 - reveal;
    
    return vec4<f32>(final_color, final_alpha);
}
