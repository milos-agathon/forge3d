// TV5: Local irradiance probe bindings and helpers.

struct ProbeGridUniforms {
    // xy = world origin, z = height offset, w = enabled
    grid_origin: vec4<f32>,
    // x = spacing_x, y = spacing_y, z = dims_x, w = dims_y
    grid_params: vec4<f32>,
    // x = fallback blend distance, y = probe count, zw = pad
    blend_params: vec4<f32>,
}

struct GpuProbeData {
    sh_r_01: vec4<f32>,
    sh_r_23: vec4<f32>,
    sh_r_4: vec4<f32>,
    sh_g_01: vec4<f32>,
    sh_g_23: vec4<f32>,
    sh_g_4: vec4<f32>,
    sh_b_01: vec4<f32>,
    sh_b_23: vec4<f32>,
    sh_b_4: vec4<f32>,
}

struct ProbeIrradianceResult {
    irradiance: vec3<f32>,
    weight: f32,
}

@group(6) @binding(1)
var<uniform> probe_grid: ProbeGridUniforms;

@group(6) @binding(2)
var<storage, read> probe_data: array<GpuProbeData>;

fn evaluate_sh_l2(n: vec3<f32>, probe: GpuProbeData) -> vec3<f32> {
    let Y00 = 0.282095;
    let Y1m1 = 0.488603 * n.y;
    let Y10 = 0.488603 * n.z;
    let Y11 = 0.488603 * n.x;
    let Y2m2 = 1.092548 * n.x * n.y;
    let Y2m1 = 1.092548 * n.y * n.z;
    let Y20 = 0.315392 * (3.0 * n.z * n.z - 1.0);
    let Y21 = 1.092548 * n.x * n.z;
    let Y22 = 0.546274 * (n.x * n.x - n.y * n.y);

    let basis_01 = vec4<f32>(Y00, Y1m1, Y10, Y11);
    let basis_23 = vec4<f32>(Y2m2, Y2m1, Y20, Y21);

    var result: vec3<f32>;
    result.r = dot(probe.sh_r_01, basis_01) + dot(probe.sh_r_23, basis_23) + probe.sh_r_4.x * Y22;
    result.g = dot(probe.sh_g_01, basis_01) + dot(probe.sh_g_23, basis_23) + probe.sh_g_4.x * Y22;
    result.b = dot(probe.sh_b_01, basis_01) + dot(probe.sh_b_23, basis_23) + probe.sh_b_4.x * Y22;
    return max(result, vec3<f32>(0.0));
}

fn sample_probe_irradiance(world_pos: vec3<f32>, normal: vec3<f32>) -> ProbeIrradianceResult {
    var result: ProbeIrradianceResult;
    result.irradiance = vec3<f32>(0.0);
    result.weight = 0.0;

    if (probe_grid.grid_origin.w < 0.5) {
        return result;
    }

    let dims = vec2<u32>(u32(probe_grid.grid_params.z), u32(probe_grid.grid_params.w));
    if (dims.x == 0u || dims.y == 0u) {
        return result;
    }

    let spacing = max(probe_grid.grid_params.xy, vec2<f32>(1e-6, 1e-6));
    let grid_uv = (world_pos.xy - probe_grid.grid_origin.xy) / spacing;
    let grid_extent = vec2<f32>(f32(dims.x - 1u), f32(dims.y - 1u));

    var i0 = vec2<u32>(0u, 0u);
    var frac = vec2<f32>(0.0, 0.0);
    if (dims.x > 1u) {
        let clamped_x = clamp(grid_uv.x, 0.0, grid_extent.x);
        let base_x = min(u32(floor(clamped_x)), dims.x - 1u);
        i0.x = base_x;
        if (base_x < dims.x - 1u) {
            frac.x = fract(clamped_x);
        }
    }
    if (dims.y > 1u) {
        let clamped_y = clamp(grid_uv.y, 0.0, grid_extent.y);
        let base_y = min(u32(floor(clamped_y)), dims.y - 1u);
        i0.y = base_y;
        if (base_y < dims.y - 1u) {
            frac.y = fract(clamped_y);
        }
    }

    let i1 = vec2<u32>(min(i0.x + 1u, dims.x - 1u), min(i0.y + 1u, dims.y - 1u));
    let idx00 = i0.y * dims.x + i0.x;
    let idx10 = i0.y * dims.x + i1.x;
    let idx01 = i1.y * dims.x + i0.x;
    let idx11 = i1.y * dims.x + i1.x;

    let sh00 = evaluate_sh_l2(normal, probe_data[idx00]);
    let sh10 = evaluate_sh_l2(normal, probe_data[idx10]);
    let sh01 = evaluate_sh_l2(normal, probe_data[idx01]);
    let sh11 = evaluate_sh_l2(normal, probe_data[idx11]);
    let bilinear = mix(mix(sh00, sh10, frac.x), mix(sh01, sh11, frac.x), frac.y);

    let fallback_dist = max(probe_grid.blend_params.x, 1e-6);
    var dist_x = fallback_dist + 1.0;
    var dist_y = fallback_dist + 1.0;
    if (dims.x > 1u) {
        dist_x = min(grid_uv.x, grid_extent.x - grid_uv.x) * spacing.x;
    }
    if (dims.y > 1u) {
        dist_y = min(grid_uv.y, grid_extent.y - grid_uv.y) * spacing.y;
    }
    let dist_to_edge = min(dist_x, dist_y);

    result.irradiance = bilinear;
    result.weight = clamp(dist_to_edge / fallback_dist, 0.0, 1.0);
    return result;
}
