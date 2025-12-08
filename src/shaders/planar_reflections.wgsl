// Placeholder planar reflection shader.
// Matches PlanarReflectionUniforms in src/core/reflections.rs to keep layouts compatible.

struct ReflectionPlane {
    plane_equation: vec4<f32>,
    reflection_matrix: mat4x4<f32>,
    reflection_view: mat4x4<f32>,
    reflection_projection: mat4x4<f32>,
    plane_center: vec4<f32>,
    plane_size: vec4<f32>,
};

struct PlanarReflectionUniforms {
    reflection_plane: ReflectionPlane,
    enable_reflections: u32,      // 0=disabled, 1=enabled, 2=reflection pass
    reflection_intensity: f32,
    fresnel_power: f32,
    blur_kernel_size: u32,
    max_blur_radius: f32,
    reflection_resolution: f32,
    distance_fade_start: f32,
    distance_fade_end: f32,
    debug_mode: u32,
    camera_position: vec4<f32>,
    padding: vec4<f32>,
    padding_tail: vec3<f32>,
};

@group(0) @binding(0) var<uniform> reflection_uniforms : PlanarReflectionUniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    // Fullscreen triangle
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = positions[idx];

    var out: VsOut;
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = p * 0.5 + 0.5;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Simple visualization: encode mode in color; keep opaque output to avoid NaNs.
    let mode = reflection_uniforms.enable_reflections;
    if (mode == 0u) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    if (mode == 2u) {
        return vec4<f32>(0.0, 0.2, 0.8, 1.0);
    }
    return vec4<f32>(0.1, 0.1, 0.1, 1.0);
}
