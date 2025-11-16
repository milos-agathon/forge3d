// src/shaders/ssr/trace.wgsl
// Screen-space reflection tracing: outputs hit UV + metadata per pixel

struct SsrSettings {
    max_steps: u32,
    thickness: f32,
    max_distance: f32,
    intensity: f32,
    inv_resolution: vec2<f32>,
    _pad: vec2<f32>,
}

struct CameraParams {
    view_matrix: mat4x4<f32>,
    inv_view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    inv_proj_matrix: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
}

@group(0) @binding(0) var depth_texture: texture_2d<f32>;
@group(0) @binding(1) var normal_texture: texture_2d<f32>;
@group(0) @binding(2) var hit_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> settings: SsrSettings;
@group(0) @binding(4) var<uniform> camera: CameraParams;

fn decode_normal(encoded: vec4<f32>) -> vec3<f32> {
    return normalize(encoded.xyz * 2.0 - 1.0);
}

fn reconstruct_view_position(uv: vec2<f32>, linear_depth: f32) -> vec3<f32> {
    let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    let focal = vec2<f32>(camera.inv_proj_matrix[0][0], camera.inv_proj_matrix[1][1]);
    let center = vec2<f32>(camera.inv_proj_matrix[2][0], camera.inv_proj_matrix[2][1]);
    let view_xy = (ndc_xy - center) / focal;
    return vec3<f32>(view_xy * linear_depth, -linear_depth);
}

fn project_to_screen(view_pos: vec3<f32>) -> vec3<f32> {
    let clip = camera.proj_matrix * vec4<f32>(view_pos, 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return vec3<f32>(uv, ndc.z);
}

@compute @workgroup_size(8, 8, 1)
fn cs_trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    let dims = textureDimensions(depth_texture);
    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }
    let dims_f = vec2<f32>(f32(dims.x), f32(dims.y));

    let depth = textureLoad(depth_texture, pixel, 0).r;
    if (depth <= 0.0) {
        textureStore(hit_output, pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) * settings.inv_resolution;
    let normal_sample = textureLoad(normal_texture, pixel, 0);
    let normal_vs = decode_normal(normal_sample);
    let view_pos = reconstruct_view_position(uv, depth);
    let view_dir = normalize(-view_pos);
    let reflect_dir = normalize(reflect(-view_dir, normal_vs));

    var hit_uv = uv;
    var hit_mask = 0.0;
    var steps_norm = 0.0;
    let max_steps = max(settings.max_steps, 1u);
    let step_len = settings.max_distance / f32(max_steps);
    var traveled = step_len;

    for (var i: u32 = 0u; i < max_steps; i = i + 1u) {
        let sample_vs = view_pos + reflect_dir * traveled;
        if (sample_vs.z >= -settings.thickness) {
            break;
        }

        let projected = project_to_screen(sample_vs);
        if (projected.x < 0.0 || projected.x > 1.0 || projected.y < 0.0 || projected.y > 1.0) {
            break;
        }

        let coord_f = clamp(projected.xy, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0))
            * (dims_f - vec2<f32>(1.0, 1.0));
        let texel = vec2<u32>(u32(coord_f.x), u32(coord_f.y));
        let scene_depth = textureLoad(depth_texture, texel, 0).r;
        if (scene_depth <= 0.0) {
            traveled = traveled + step_len;
            continue;
        }

        let ray_depth = -sample_vs.z;
        if (abs(ray_depth - scene_depth) <= settings.thickness) {
            hit_uv = projected.xy;
            hit_mask = 1.0;
            steps_norm = f32(i + 1u) / f32(max_steps);
            break;
        }

        traveled = traveled + step_len;
        if (traveled > settings.max_distance) {
            break;
        }
    }

    textureStore(hit_output, pixel, vec4<f32>(hit_uv, steps_norm, hit_mask));
}
