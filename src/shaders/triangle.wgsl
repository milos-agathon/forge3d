struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(1) color: vec3<f32>,
}

@vertex
fn vs_main(
    @location(0) pos: vec2<f32>,
    @location(1) color: vec3<f32>,
) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(pos, 0.0, 1.0);
    output.color = color;
    return output;
}

struct FSInput {
    @location(1) color: vec3<f32>,
}

@fragment
fn fs_main(input: FSInput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}