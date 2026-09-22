struct OrbisGlobeUniforms {
    inverse_view_projection: mat4x4<f32>,
    sphere: vec4<f32>,
    base_color: vec4<f32>,
    rim_color: vec4<f32>,
    // xyz = direction towards the sun in render space (the terrain light),
    // w = 1 when a sky texture already owns the pixels outside the Earth.
    sun_direction: vec4<f32>,
    // Clear colour the atmosphere halo composites over when there is no sky.
    space_color: vec4<f32>,
};
@group(0) @binding(0) var<uniform> globe: OrbisGlobeUniforms;

struct FullscreenOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

// Scale height of the visible atmosphere shell, as a fraction of the radius.
const ATMOSPHERE_THICKNESS: f32 = 0.012;

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
    let sun = normalize(globe.sun_direction.xyz);
    let oc = -center;
    let half_b = dot(oc, ray);
    let c = dot(oc, oc) - radius * radius;
    let discriminant = half_b * half_b - c;

    var t = -1.0;
    if (discriminant >= 0.0) {
        t = -half_b - sqrt(discriminant);
        if (t <= 0.0) {
            t = -half_b + sqrt(discriminant);
        }
    }
    if (t > 0.0) {
        // Earth surface outside the loaded DEM: lit by the same sun as the
        // terrain, with a thin scattering rim towards the limb.
        let normal = normalize(ray * t - center);
        let sun_cos = dot(normal, sun);
        let diffuse = 0.22 + 0.78 * max(sun_cos, 0.0);
        let rim = pow(1.0 - max(dot(normal, -ray), 0.0), 3.0) * 0.45;
        let day = smoothstep(-0.15, 0.25, sun_cos);
        return vec4<f32>(globe.base_color.rgb * diffuse + globe.rim_color.rgb * rim * (0.35 + 0.65 * day), 1.0);
    }
    if (globe.sun_direction.w > 0.5) {
        discard;
    }
    // Ray misses the Earth: the atmosphere halo just above the limb. It is
    // seen from outside the shell only; inside it (flight altitudes) the sky
    // keeps the flat clear colour, faded so the descent never pops.
    let camera_height = (length(oc) - radius) / (radius * ATMOSPHERE_THICKNESS);
    let outside = smoothstep(1.0, 1.5, camera_height);
    let closest = oc - ray * half_b;
    let height = (length(closest) - radius) / (radius * ATMOSPHERE_THICKNESS);
    if (half_b > 0.0 || height > 1.0 || outside <= 0.0) {
        return vec4<f32>(globe.space_color.rgb, 1.0);
    }
    let glow = pow(clamp(1.0 - height, 0.0, 1.0), 3.0) * outside;
    let limb_normal = normalize(closest);
    let day = smoothstep(-0.2, 0.3, dot(limb_normal, sun));
    let halo = globe.rim_color.rgb * glow * (0.25 + 0.75 * day) * 1.4;
    return vec4<f32>(globe.space_color.rgb + halo, 1.0);
}
