
struct Uniforms {
    view_proj: mat4x4<f32>,
    viewport_size: vec2<f32>,
    point_size: f32,
    color_mode: u32,  // 0=elevation, 1=rgb, 2=intensity
    has_rgb: u32,     // 1 if file has RGB data
    has_intensity: u32, // 1 if file has meaningful intensity
    _pad: vec2<u32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) elevation_norm: f32,
    @location(2) rgb: vec3<f32>,
    @location(3) intensity: f32,
    @location(4) size: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
}

// Elevation colormap: blue -> green -> yellow -> brown -> white
fn elevation_color(t: f32) -> vec3<f32> {
    // 5-stop terrain colormap
    let c0 = vec3<f32>(0.2, 0.4, 0.6);  // Low: blue-gray
    let c1 = vec3<f32>(0.3, 0.5, 0.2);  // Green
    let c2 = vec3<f32>(0.6, 0.6, 0.3);  // Yellow-green
    let c3 = vec3<f32>(0.5, 0.4, 0.3);  // Brown
    let c4 = vec3<f32>(0.9, 0.9, 0.9);  // High: white
    
    let t_scaled = clamp(t, 0.0, 1.0) * 4.0;
    
    if t_scaled < 1.0 {
        return mix(c0, c1, t_scaled);
    } else if t_scaled < 2.0 {
        return mix(c1, c2, t_scaled - 1.0);
    } else if t_scaled < 3.0 {
        return mix(c2, c3, t_scaled - 2.0);
    } else {
        return mix(c3, c4, t_scaled - 3.0);
    }
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    instance: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    
    // Quad vertices for point sprite (use switch for runtime indexing)
    var offset: vec2<f32>;
    switch vertex_index {
        case 0u: { offset = vec2<f32>(-1.0, -1.0); }
        case 1u: { offset = vec2<f32>(1.0, -1.0); }
        case 2u: { offset = vec2<f32>(-1.0, 1.0); }
        case 3u: { offset = vec2<f32>(1.0, 1.0); }
        default: { offset = vec2<f32>(0.0, 0.0); }
    }
    out.uv = offset * 0.5 + 0.5;
    
    // Transform point to clip space
    let clip_pos = uniforms.view_proj * vec4<f32>(instance.position, 1.0);
    
    // Calculate point size in clip space
    let size_px = uniforms.point_size * instance.size;
    let size_clip = vec2<f32>(
        size_px / uniforms.viewport_size.x * 2.0,
        size_px / uniforms.viewport_size.y * 2.0,
    );
    
    // Apply quad offset
    out.clip_position = clip_pos + vec4<f32>(offset * size_clip * clip_pos.w, 0.0, 0.0);
    
    // Compute color based on mode (with fallback to grayscale elevation if data unavailable)
    var color: vec3<f32>;
    switch uniforms.color_mode {
        case 1u: {
            // RGB mode - use stored RGB, fallback to tinted elevation if no RGB data
            if uniforms.has_rgb == 1u {
                color = instance.rgb;
            } else {
                // Fallback to reddish elevation to indicate missing RGB
                color = elevation_color(instance.elevation_norm) * vec3<f32>(1.0, 0.5, 0.5);
            }
        }
        case 2u: {
            // Intensity mode - grayscale, fallback to tinted elevation if no intensity data
            if uniforms.has_intensity == 1u {
                color = vec3<f32>(instance.intensity);
            } else {
                // Fallback to green-tinted elevation to indicate missing Intensity
                color = elevation_color(instance.elevation_norm) * vec3<f32>(0.5, 1.0, 0.5);
            }
        }
        default: {
            // Elevation mode (0 or fallback)
            color = elevation_color(instance.elevation_norm);
        }
    }
    out.color = vec4<f32>(color, 1.0);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Circular point with soft edge
    let uv = in.uv * 2.0 - 1.0;
    let dist = length(uv);
    
    if dist > 1.0 {
        discard;
    }
    
    // Soft edge
    let alpha = 1.0 - smoothstep(0.7, 1.0, dist);
    
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
