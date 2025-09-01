// Split bind groups shader - each object has its own bind group

struct Transform {
    matrix: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> transform: Transform;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return transform.matrix * vec4<f32>(position, 1.0);
}

@fragment  
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.5, 0.8, 0.3, 1.0); // Green
}