// TV5: Local irradiance and reflection probe bindings and helpers.

struct ProbeGridUniforms {
    // xy = world origin, z = height offset, w = enabled
    grid_origin: vec4<f32>,
    // x = spacing_x, y = spacing_y, z = dims_x, w = dims_y
    grid_params: vec4<f32>,
    // x = fallback blend distance, y = probe count, z = feature strength, w = pad
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

struct GpuReflectionProbeData {
    pos_x: vec4<f32>,
    neg_x: vec4<f32>,
    pos_y: vec4<f32>,
    neg_y: vec4<f32>,
    pos_z: vec4<f32>,
    neg_z: vec4<f32>,
    average: vec4<f32>,
}

struct ProbeIrradianceResult {
    irradiance: vec3<f32>,
    weight: f32,
}

struct ReflectionProbeResult {
    prefiltered_color: vec3<f32>,
    weight: f32,
}

struct ProbeGridBlend {
    idx00: u32,
    idx10: u32,
    idx01: u32,
    idx11: u32,
    frac: vec2<f32>,
    weight: f32,
    valid: f32,
}

@group(6) @binding(1)
var<uniform> probe_grid: ProbeGridUniforms;

@group(6) @binding(2)
var<storage, read> probe_data: array<GpuProbeData>;

@group(6) @binding(3)
var<uniform> reflection_probe_grid: ProbeGridUniforms;

@group(6) @binding(4)
var<storage, read> reflection_probe_data: array<GpuReflectionProbeData>;

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

fn compute_probe_grid_blend(grid: ProbeGridUniforms, world_pos: vec3<f32>) -> ProbeGridBlend {
    var blend: ProbeGridBlend;
    blend.idx00 = 0u;
    blend.idx10 = 0u;
    blend.idx01 = 0u;
    blend.idx11 = 0u;
    blend.frac = vec2<f32>(0.0, 0.0);
    blend.weight = 0.0;
    blend.valid = 0.0;

    if (grid.grid_origin.w < 0.5) {
        return blend;
    }

    let dims = vec2<u32>(u32(grid.grid_params.z), u32(grid.grid_params.w));
    if (dims.x == 0u || dims.y == 0u) {
        return blend;
    }

    let spacing = max(grid.grid_params.xy, vec2<f32>(1e-6, 1e-6));
    let grid_uv = (world_pos.xy - grid.grid_origin.xy) / spacing;
    let grid_extent = vec2<f32>(f32(dims.x - 1u), f32(dims.y - 1u));

    var i0 = vec2<u32>(0u, 0u);
    if (dims.x > 1u) {
        let clamped_x = clamp(grid_uv.x, 0.0, grid_extent.x);
        let base_x = min(u32(floor(clamped_x)), dims.x - 1u);
        i0.x = base_x;
        if (base_x < dims.x - 1u) {
            blend.frac.x = fract(clamped_x);
        }
    }
    if (dims.y > 1u) {
        let clamped_y = clamp(grid_uv.y, 0.0, grid_extent.y);
        let base_y = min(u32(floor(clamped_y)), dims.y - 1u);
        i0.y = base_y;
        if (base_y < dims.y - 1u) {
            blend.frac.y = fract(clamped_y);
        }
    }

    let i1 = vec2<u32>(min(i0.x + 1u, dims.x - 1u), min(i0.y + 1u, dims.y - 1u));
    blend.idx00 = i0.y * dims.x + i0.x;
    blend.idx10 = i0.y * dims.x + i1.x;
    blend.idx01 = i1.y * dims.x + i0.x;
    blend.idx11 = i1.y * dims.x + i1.x;

    let fallback_dist = max(grid.blend_params.x, 1e-6);
    var dist_x = fallback_dist + 1.0;
    var dist_y = fallback_dist + 1.0;
    if (dims.x > 1u) {
        dist_x = min(grid_uv.x, grid_extent.x - grid_uv.x) * spacing.x;
    }
    if (dims.y > 1u) {
        dist_y = min(grid_uv.y, grid_extent.y - grid_uv.y) * spacing.y;
    }
    let dist_to_edge = min(dist_x, dist_y);
    blend.weight = clamp(dist_to_edge / fallback_dist, 0.0, 1.0);
    blend.valid = 1.0;
    return blend;
}

fn sample_probe_irradiance(world_pos: vec3<f32>, normal: vec3<f32>) -> ProbeIrradianceResult {
    var result: ProbeIrradianceResult;
    result.irradiance = vec3<f32>(0.0);
    result.weight = 0.0;

    let blend = compute_probe_grid_blend(probe_grid, world_pos);
    if (blend.valid < 0.5) {
        return result;
    }

    let sh00 = evaluate_sh_l2(normal, probe_data[blend.idx00]);
    let sh10 = evaluate_sh_l2(normal, probe_data[blend.idx10]);
    let sh01 = evaluate_sh_l2(normal, probe_data[blend.idx01]);
    let sh11 = evaluate_sh_l2(normal, probe_data[blend.idx11]);
    result.irradiance = mix(
        mix(sh00, sh10, blend.frac.x),
        mix(sh01, sh11, blend.frac.x),
        blend.frac.y,
    );
    result.weight = blend.weight;
    return result;
}

fn sample_reflection_probe_direction(
    direction: vec3<f32>,
    roughness: f32,
    probe: GpuReflectionProbeData,
) -> vec3<f32> {
    let d = normalize(direction);
    let sharpness = mix(8.0, 1.0, clamp(roughness, 0.0, 1.0));
    let w_pos_x = pow(max(d.x, 0.0), sharpness);
    let w_neg_x = pow(max(-d.x, 0.0), sharpness);
    let w_pos_y = pow(max(d.y, 0.0), sharpness);
    let w_neg_y = pow(max(-d.y, 0.0), sharpness);
    let w_pos_z = pow(max(d.z, 0.0), sharpness);
    let w_neg_z = pow(max(-d.z, 0.0), sharpness);
    let total = max(w_pos_x + w_neg_x + w_pos_y + w_neg_y + w_pos_z + w_neg_z, 1e-5);
    let directional =
        probe.pos_x.rgb * w_pos_x +
        probe.neg_x.rgb * w_neg_x +
        probe.pos_y.rgb * w_pos_y +
        probe.neg_y.rgb * w_neg_y +
        probe.pos_z.rgb * w_pos_z +
        probe.neg_z.rgb * w_neg_z;
    let blur = clamp(roughness * roughness, 0.0, 1.0);
    return mix(directional / total, probe.average.rgb, blur);
}

fn sample_reflection_probe(
    world_pos: vec3<f32>,
    reflection_dir: vec3<f32>,
    roughness: f32,
) -> ReflectionProbeResult {
    var result: ReflectionProbeResult;
    result.prefiltered_color = vec3<f32>(0.0);
    result.weight = 0.0;

    let blend = compute_probe_grid_blend(reflection_probe_grid, world_pos);
    if (blend.valid < 0.5) {
        return result;
    }

    let r00 = sample_reflection_probe_direction(reflection_dir, roughness, reflection_probe_data[blend.idx00]);
    let r10 = sample_reflection_probe_direction(reflection_dir, roughness, reflection_probe_data[blend.idx10]);
    let r01 = sample_reflection_probe_direction(reflection_dir, roughness, reflection_probe_data[blend.idx01]);
    let r11 = sample_reflection_probe_direction(reflection_dir, roughness, reflection_probe_data[blend.idx11]);
    result.prefiltered_color = mix(
        mix(r00, r10, blend.frac.x),
        mix(r01, r11, blend.frac.x),
        blend.frac.y,
    );
    result.weight = blend.weight * clamp(reflection_probe_grid.blend_params.z, 0.0, 1.0);
    return result;
}
