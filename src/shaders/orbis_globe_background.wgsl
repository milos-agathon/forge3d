struct OrbisGlobeUniforms {
    inverse_view_projection: mat4x4<f32>,
    sphere: vec4<f32>,
    base_color: vec4<f32>,
    rim_color: vec4<f32>,
};
@group(0) @binding(0) var<uniform> globe: OrbisGlobeUniforms;

struct FullscreenOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> FullscreenOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    );
    var out: FullscreenOutput;
    out.position = vec4<f32>(positions[index], 0.0, 1.0);
    out.ndc = positions[index];
    return out;
}

@fragment
fn fs_main(input: FullscreenOutput) -> @location(0) vec4<f32> {
    let near_h = globe.inverse_view_projection * vec4<f32>(input.ndc, 0.0, 1.0);
    let far_h = globe.inverse_view_projection * vec4<f32>(input.ndc, 1.0, 1.0);
    let near_point = near_h.xyz / near_h.w;
    let far_point = far_h.xyz / far_h.w;
    let ray = normalize(far_point - near_point);
    let center = globe.sphere.xyz;
    let radius = globe.sphere.w;
    let oc = -center;
    let half_b = dot(oc, ray);
    let c = dot(oc, oc) - radius * radius;
    let discriminant = half_b * half_b - c;
    if (discriminant < 0.0) { discard; }
    var t = -half_b - sqrt(discriminant);
    if (t <= 0.0) {
        t = -half_b + sqrt(discriminant);
    }
    if (t <= 0.0) { discard; }
    let normal = normalize(ray * t - center);
    let light = normalize(vec3<f32>(-0.35, 0.2, 0.92));
    let diffuse = 0.32 + 0.68 * max(dot(normal, light), 0.0);
    let rim_factor = pow(1.0 - max(dot(normal, -ray), 0.0), 3.0) * 0.35;
    return vec4<f32>(globe.base_color.rgb * diffuse + globe.rim_color.rgb * rim_factor, 1.0);
}
