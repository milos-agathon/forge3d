// Shared SSAO/GTAO data and helpers
// This chunk is concatenated ahead of the specific kernel entry points so both
// techniques see identical bindings and utility functions.

struct SsaoSettings {
    radius: f32,
    intensity: f32,
    bias: f32,
    num_samples: u32,
    technique: u32,
    frame_index: u32,
    inv_resolution: vec2<f32>,
    proj_scale: f32,
    ao_min: f32,
};

struct CameraParams {
    view_matrix: mat4x4<f32>,
    inv_view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    inv_proj_matrix: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
};

@group(0) @binding(0) var depth_texture: texture_2d<f32>;
@group(0) @binding(1) var hzb_texture: texture_2d<f32>;
@group(0) @binding(2) var normal_texture: texture_2d<f32>;
@group(0) @binding(3) var noise_texture: texture_2d<f32>;
@group(0) @binding(4) var tex_sampler: sampler;
@group(0) @binding(5) var output_ao: texture_storage_2d<r32float, write>;
@group(0) @binding(6) var<uniform> settings: SsaoSettings;
@group(0) @binding(7) var<uniform> camera: CameraParams;

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

fn reconstruct_view_pos_linear(uv: vec2<f32>, linear_depth: f32) -> vec3<f32> {
    let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    let p00 = camera.proj_matrix[0][0];
    let p11 = camera.proj_matrix[1][1];
    let view_xy = vec2<f32>(ndc_xy.x / p00, ndc_xy.y / p11) * linear_depth;
    return vec3<f32>(view_xy, -linear_depth);
}

fn unpack_normal(packed: vec4<f32>) -> vec3<f32> {
    return normalize(packed.xyz * 2.0 - vec3<f32>(1.0));
}

fn ign(pixel: vec2<u32>) -> f32 {
    let px = vec2<f32>(f32(pixel.x), f32(pixel.y));
    return fract(52.9829189 * fract(0.06711056 * px.x + 0.00583715 * px.y));
}

fn compute_ssao(pixel: vec2<u32>, uv: vec2<f32>, view_pos: vec3<f32>, view_normal: vec3<f32>) -> f32 {
    let noise_angle = ign(pixel + vec2<u32>(settings.frame_index, settings.frame_index * 1664525u)) * TWO_PI;

    let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), abs(view_normal.y) < 0.99);
    let tangent = normalize(cross(up, view_normal));
    let bitangent = cross(view_normal, tangent);
    let tbn = mat3x3<f32>(tangent, bitangent, view_normal);

    var occlusion = 0.0;
    let num_samples_f = f32(settings.num_samples);

    for (var i = 0u; i < settings.num_samples; i++) {
        let fi = f32(i);
        let alpha = (fi + 0.5) / num_samples_f;
        let angle = alpha * 4.0 * TWO_PI + noise_angle;

        let h = sqrt(1.0 - alpha);
        let r = sqrt(alpha);
        let sample_dir = vec3<f32>(cos(angle) * r, sin(angle) * r, h);

        let sample_offset = tbn * sample_dir * settings.radius;
        let sample_pos = view_pos + sample_offset;

        let sample_clip = camera.proj_matrix * vec4<f32>(sample_pos, 1.0);
        var sample_uv = sample_clip.xy / sample_clip.w;
        sample_uv = sample_uv * 0.5 + 0.5;
        sample_uv.y = 1.0 - sample_uv.y;

        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
            continue;
        }

        let dims = textureDimensions(depth_texture);
        let sample_pixel = vec2<u32>(sample_uv * vec2<f32>(dims));
        let sample_depth = textureLoad(depth_texture, sample_pixel, 0).r;
        let sample_view_pos = reconstruct_view_pos_linear(sample_uv, sample_depth);

        let delta = sample_view_pos - view_pos;
        let dist = length(delta);
        let range_check = smoothstep(0.0, 1.0, settings.radius / max(dist, 1e-4));
        let z_lin = -view_pos.z;
        let pixel_size_in_view = z_lin / max(settings.proj_scale, 1e-4);
        let bias = settings.bias + 0.5 * pixel_size_in_view;
        let sample_z_lin = -sample_view_pos.z;
        let dz = sample_z_lin - z_lin;
        let n_dir = dot(view_normal, normalize(delta));
        let falloff = exp(-dist * 2.0) * smoothstep(bias, settings.radius, dist);
        let contribution = max(0.0, n_dir - bias) * falloff;
        if (dz < -bias) {
            occlusion += contribution * range_check;
        }
    }

    let raw = 1.0 - clamp(occlusion / max(num_samples_f, 1.0) * settings.intensity, 0.0, 1.0);
    return clamp(raw, settings.ao_min, 1.0);
}

fn compute_gtao(pixel: vec2<u32>, uv: vec2<f32>, view_pos: vec3<f32>, view_normal: vec3<f32>) -> f32 {
    let noise = ign(pixel + vec2<u32>(settings.frame_index * 747796405u, settings.frame_index * 2891336453u));
    let angle_offset = noise * TWO_PI;

    var visibility = 0.0;
    let direction_count = max(settings.num_samples / 4u, 2u);
    let steps_per_direction = 4u;

    for (var d = 0u; d < direction_count; d++) {
        let angle = (f32(d) / f32(direction_count)) * PI + angle_offset;
        let direction = vec2<f32>(cos(angle), sin(angle));

        var horizon_cos = -1.0;

        for (var s = 1u; s <= steps_per_direction; s++) {
            let z_lin = -view_pos.z;
            let r_screen = settings.radius * (settings.proj_scale / max(z_lin, 1e-4));
            let step_px = r_screen * (f32(s) / f32(steps_per_direction));
            let sample_uv = uv + direction * (step_px * settings.inv_resolution);

            if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
                continue;
            }

            let dims = textureDimensions(depth_texture);
            let sample_pixel = vec2<u32>(sample_uv * vec2<f32>(dims));
            let sample_depth = textureLoad(depth_texture, sample_pixel, 0).r;
            let sample_pos = reconstruct_view_pos_linear(sample_uv, sample_depth);

            let horizon_vec = sample_pos - view_pos;
            let horizon_len = length(horizon_vec);
            let h_cos = dot(horizon_vec / max(horizon_len, 1e-4), view_normal);

            let attenuation = 1.0 - clamp(horizon_len / settings.radius, 0.0, 1.0);
            horizon_cos = max(horizon_cos, h_cos * attenuation);
        }

        let horizon_angle = acos(clamp(horizon_cos, -1.0, 1.0));
        visibility += sin(horizon_angle) - horizon_angle * cos(horizon_angle) + 0.5 * PI;
    }

    visibility = visibility / (f32(direction_count) * PI);
    let ao_raw = 1.0 - clamp(visibility * settings.intensity, 0.0, 1.0);
    return clamp(ao_raw, settings.ao_min, 1.0);
}
