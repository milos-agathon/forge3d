// src/shaders/ssr/fallback_env.wgsl
// Fill SSR misses with environment reflections and output final specular buffer

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

struct SsrCounters {
    rays: atomic<u32>,
    hits: atomic<u32>,
    total_steps: atomic<u32>,
    misses: atomic<u32>,
    miss_ibl_samples: atomic<u32>,
}

@group(0) @binding(0) var ssr_spec: texture_2d<f32>;
@group(0) @binding(1) var hit_texture: texture_2d<f32>;
@group(0) @binding(2) var depth_texture: texture_2d<f32>;
@group(0) @binding(3) var normal_texture: texture_2d<f32>;
@group(0) @binding(4) var env_cube: texture_cube<f32>;
@group(0) @binding(5) var env_sampler: sampler;
@group(0) @binding(6) var final_ssr: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var<uniform> settings: SsrSettings;
@group(0) @binding(8) var<uniform> camera: CameraParams;
@group(0) @binding(9) var<storage, read_write> counters: SsrCounters;

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

fn to_world(dir_vs: vec3<f32>) -> vec3<f32> {
    return normalize((camera.inv_view_matrix * vec4<f32>(dir_vs, 0.0)).xyz);
}

@compute @workgroup_size(8, 8, 1)
fn cs_fallback(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    let dims = textureDimensions(hit_texture);
    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }

    let hit = textureLoad(hit_texture, pixel, 0);
    let hit_mask = step(0.5, hit.w);
    let ssr_value = textureLoad(ssr_spec, pixel, 0).rgb;

    if (hit_mask > 0.5) {
        textureStore(final_ssr, pixel, vec4<f32>(ssr_value, 1.0));
        return;
    }

    let depth = textureLoad(depth_texture, pixel, 0).r;

    // Reconstruct view ray from screen-space UV. For depth <= 0 (sky/background), we
    // approximate a ray through this pixel using a fixed linear depth; for valid
    // geometry we use the true linear depth.
    let uv = (vec2<f32>(f32(pixel.x), f32(pixel.y)) + vec2<f32>(0.5, 0.5)) * settings.inv_resolution;

    var normal_vs: vec3<f32>;
    var view_dir: vec3<f32>;
    if (depth <= 0.0) {
        // Approximate a forward view ray from the camera through this pixel.
        let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
        let focal = vec2<f32>(camera.inv_proj_matrix[0][0], camera.inv_proj_matrix[1][1]);
        let center = vec2<f32>(camera.inv_proj_matrix[2][0], camera.inv_proj_matrix[2][1]);
        let view_xy = (ndc_xy - center) / focal;
        let ray_vs = vec3<f32>(view_xy, -1.0);
        view_dir = normalize(ray_vs);
        // For sky/background, treat the "surface" as facing the camera so that
        // the reflection direction roughly matches the primary view ray.
        normal_vs = -view_dir;
    } else {
        normal_vs = decode_normal(textureLoad(normal_texture, pixel, 0));
        let view_pos = reconstruct_view_position(uv, depth);
        view_dir = normalize(-view_pos);
    }

    let reflect_vs = normalize(reflect(-view_dir, normal_vs));
    let reflect_ws = to_world(reflect_vs);

    let roughness = clamp(textureLoad(normal_texture, pixel, 0).w, 0.0, 1.0);
    let max_mips = max(f32(textureNumLevels(env_cube)) - 1.0, 0.0);
    let mip = roughness * max_mips;
    let env_color = textureSampleLevel(env_cube, env_sampler, reflect_ws, mip).rgb;
    let shaded = env_color * settings.intensity;
    let min_floor = vec3<f32>(2.0 / 255.0);
    let safe_color = max(shaded, min_floor);
    // For miss pixels, we keep a non-black env color in the SSR buffer (for metrics),
    // but encode alpha = 0 so the composite pass can avoid double-counting IBL.
    textureStore(final_ssr, pixel, vec4<f32>(safe_color, 0.0));
    atomicAdd(&counters.miss_ibl_samples, 1u);
}
