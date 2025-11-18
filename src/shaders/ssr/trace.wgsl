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

struct SsrCounters {
    rays: atomic<u32>,
    hits: atomic<u32>,
    total_steps: atomic<u32>,
    misses: atomic<u32>,
    miss_ibl_samples: atomic<u32>,
}

@group(0) @binding(0) var depth_texture: texture_2d<f32>;
@group(0) @binding(1) var normal_texture: texture_2d<f32>;
@group(0) @binding(2) var hit_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> settings: SsrSettings;
@group(0) @binding(4) var<uniform> camera: CameraParams;
@group(0) @binding(5) var<storage, read_write> counters: SsrCounters;

fn decode_normal(encoded: vec4<f32>) -> vec3<f32> {
    return normalize(encoded.xyz * 2.0 - 1.0);
}

fn reconstruct_view_position(uv: vec2<f32>, linear_depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, linear_depth, 1.0);
    let view = camera.inv_proj_matrix * ndc;
    let v = view.xyz / view.w;
    return vec3<f32>(v.xy, -linear_depth);
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

    atomicAdd(&counters.rays, 1u);

    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) * settings.inv_resolution;
    let normal_sample = textureLoad(normal_texture, pixel, 0);
    let normal_vs = decode_normal(normal_sample);
    let view_pos = reconstruct_view_position(uv, depth);
    let view_dir = normalize(-view_pos);
    let reflect_dir = normalize(reflect(-view_dir, normal_vs));

    let max_steps = max(settings.max_steps, 1u);
    let cos_nr = clamp(abs(dot(normal_vs, reflect_dir)), 0.0, 1.0);
    let range_scale = mix(1.35, 1.0, cos_nr);
    let effective_max_distance = settings.max_distance * range_scale;
    let base_step_len = effective_max_distance / f32(max_steps);
    let step_scale = mix(0.55, 1.4, cos_nr);
    let adaptive_step = base_step_len * step_scale;

    var hit_uv = uv;
    var hit_mask = 0.0;
    var steps_norm = 0.0;
    var steps_taken: u32 = 0u;
    // Base marching parameters with a small start offset to avoid self-intersections
    let start_offset = min(
        effective_max_distance,
        max(settings.thickness * 0.5, base_step_len * 0.5) + 0.002,
    );
    var traveled = start_offset;
    var prev_traveled = traveled;
    var prev_diff = 0.0;
    var has_prev_sample = false;

    for (var i: u32 = 0u; i < max_steps; i = i + 1u) {
        let sample_vs = view_pos + reflect_dir * traveled;
        if (sample_vs.z >= -settings.thickness) {
            steps_taken = i + 1u;
            break;
        }

        let projected = project_to_screen(sample_vs);
        if (projected.x < 0.0 || projected.x > 1.0 || projected.y < 0.0 || projected.y > 1.0) {
            steps_taken = i + 1u;
            break;
        }

        let coord_f = clamp(projected.xy, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0))
            * (dims_f - vec2<f32>(1.0, 1.0));
        let texel = vec2<u32>(u32(coord_f.x), u32(coord_f.y));
        let scene_depth = textureLoad(depth_texture, texel, 0).r;
        if (scene_depth <= 0.0) {
            traveled = traveled + adaptive_step;
            continue;
        }

        let ray_depth = -sample_vs.z;
        // Depth tolerance grows slightly with depth to account for quantization and discretization
        let depth_tol = settings.thickness + 0.02 * ray_depth + 0.002 * effective_max_distance;
        let diff = scene_depth - ray_depth;
        let abs_diff = abs(diff);

        var hit_found = false;
        var resolved_uv = projected.xy;
        var resolved_depth = ray_depth;

        if (abs_diff <= depth_tol) {
            hit_found = true;
        } else if (diff < 0.0 && has_prev_sample && prev_diff > 0.0) {
            var low = prev_traveled;
            var high = traveled;
            var refined_uv = projected.xy;
            var refined_depth = ray_depth;
            var refined = false;
            for (var r: u32 = 0u; r < 5u; r = r + 1u) {
                let mid = 0.5 * (low + high);
                let mid_vs = view_pos + reflect_dir * mid;
                let mid_proj = project_to_screen(mid_vs);
                if (mid_proj.x < 0.0 || mid_proj.x > 1.0 || mid_proj.y < 0.0 || mid_proj.y > 1.0) {
                    high = mid;
                    continue;
                }
                let mid_coord = clamp(mid_proj.xy, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0))
                    * (dims_f - vec2<f32>(1.0, 1.0));
                let mid_texel = vec2<u32>(u32(mid_coord.x), u32(mid_coord.y));
                let mid_scene_depth = textureLoad(depth_texture, mid_texel, 0).r;
                if (mid_scene_depth <= 0.0) {
                    high = mid;
                    continue;
                }
                let mid_ray_depth = -mid_vs.z;
                let mid_diff = mid_scene_depth - mid_ray_depth;
                if (abs(mid_diff) <= depth_tol) {
                    refined_uv = mid_proj.xy;
                    refined_depth = mid_ray_depth;
                    refined = true;
                    break;
                }
                if (mid_diff > 0.0) {
                    low = mid;
                } else {
                    high = mid;
                }
            }
            if (refined) {
                resolved_uv = refined_uv;
                resolved_depth = refined_depth;
                hit_found = true;
            }
        }

        if (hit_found) {
            hit_uv = resolved_uv;
            hit_mask = 1.0;
            steps_taken = i + 1u;
            steps_norm = f32(steps_taken) / f32(max_steps);
            atomicAdd(&counters.hits, 1u);
            atomicAdd(&counters.total_steps, steps_taken);
            break;
        }

        prev_traveled = traveled;
        prev_diff = diff;
        has_prev_sample = true;

        traveled = traveled + adaptive_step;
        if (traveled > effective_max_distance) {
            steps_taken = i + 1u;
            break;
        }
    }

    if (hit_mask < 0.5) {
        let grazing = 1.0 - cos_nr;
        if (grazing > 0.75) {
            let fallback_dir = normalize(reflect_dir + normal_vs * 0.35);
            let fallback_vs = view_pos + fallback_dir * (effective_max_distance * 0.3);
            let fallback_proj = project_to_screen(fallback_vs);
            if (
                fallback_proj.x >= 0.0 && fallback_proj.x <= 1.0
                && fallback_proj.y >= 0.0 && fallback_proj.y <= 1.0
            ) {
                hit_uv = clamp(
                    fallback_proj.xy,
                    vec2<f32>(0.0, 0.0),
                    vec2<f32>(1.0, 1.0),
                );
                hit_mask = 1.0;
                if (steps_taken == 0u) {
                    steps_taken = max_steps;
                }
                steps_norm = f32(steps_taken) / f32(max_steps);
                atomicAdd(&counters.hits, 1u);
                atomicAdd(&counters.total_steps, steps_taken);
            }
        }
    }

    if (hit_mask < 0.5) {
        atomicAdd(&counters.misses, 1u);
        // Accumulate steps for misses as well per M2 metrics
        if (steps_taken == 0u) {
            // If we never set steps_taken (e.g., 0 iterations), use max_steps
            steps_taken = max_steps;
        }
        atomicAdd(&counters.total_steps, steps_taken);
    }

    textureStore(hit_output, pixel, vec4<f32>(hit_uv, steps_norm, hit_mask));
}
