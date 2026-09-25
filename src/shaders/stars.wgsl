// SIDERA celestial sprites rendered over the computed daylight/twilight sky.
// Each instance is an observation-frame direction, V-band relative flux, and
// an angular radius. Moon shading uses the Sun vector for the bright limb.

struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    eye_position: vec3<f32>,
    viewport_height: f32,
}
@group(0) @binding(0) var<uniform> camera: Camera;

struct Instance {
    @location(0) direction_radius: vec4<f32>,
    @location(1) color_flux: vec4<f32>,
    @location(2) sun_kind: vec4<f32>,
}

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color_flux: vec4<f32>,
    @location(2) sun_view: vec3<f32>,
    @location(3) kind: f32,
}

@vertex
fn vs_celestial(@builtin(vertex_index) vertex: u32, instance: Instance) -> VertexOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
    );
    let uv = corners[vertex];
    let view_direction = camera.view * vec4<f32>(instance.direction_radius.xyz, 0.0);
    let clip = camera.proj * vec4<f32>(view_direction.xyz, 1.0);
    var output: VertexOut;
    output.uv = uv;
    output.color_flux = instance.color_flux;
    output.sun_view = normalize((camera.view * vec4<f32>(instance.sun_kind.xyz, 0.0)).xyz);
    output.kind = instance.sun_kind.w;
    if (clip.w <= 0.0) {
        output.position = vec4<f32>(3.0, 3.0, 1.0, 1.0);
        return output;
    }
    let angular_y = instance.direction_radius.w * camera.proj[1][1];
    let radius_y = max(angular_y, 2.0 / camera.viewport_height);
    let ratio = camera.proj[0][0] / camera.proj[1][1];
    let ndc = clip.xy / clip.w + uv * vec2<f32>(radius_y * ratio, radius_y);
    output.position = vec4<f32>(ndc * clip.w, clip.z, clip.w);
    return output;
}

@fragment
fn fs_celestial(input: VertexOut) -> @location(0) vec4<f32> {
    let r2 = clamp(dot(input.uv, input.uv), 0.0, 2.0);
    if (r2 > 1.0) { discard; }
    if (input.kind > 1.5) {
        let normal = vec3<f32>(input.uv, sqrt(max(0.0, 1.0 - r2)));
        let lit = clamp(dot(normal, input.sun_view), 0.0, 1.0);
        let maria = clamp(0.88 + 0.06 * input.uv.x * input.uv.y
                          + 0.06 * (input.uv.x * input.uv.x - input.uv.y * input.uv.y),
                          0.75, 1.0);
        let edge = 1.0 - smoothstep(0.94, 1.0, sqrt(r2));
        return vec4<f32>(input.color_flux.rgb * lit * maria, edge * input.color_flux.w);
    }
    // Point-source flux is square-root encoded for an 8-bit display. The
    // stored flux itself retains the photometric 10^(-0.4 V) relation.
    let energy = min(1.0, 1.2 * sqrt(max(0.0, input.color_flux.w)));
    let profile = exp(-3.0 * r2);
    return vec4<f32>(input.color_flux.rgb * energy, profile);
}
