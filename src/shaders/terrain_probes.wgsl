// TV24: Local irradiance + local reflection probes for terrain scenes.

struct ProbeGridUniforms {
    // xy = world origin, z = height offset, w = enabled
    grid_origin: vec4<f32>,
    // x = spacing_x, y = spacing_y, z = dims_x, w = dims_y
    grid_params: vec4<f32>,
    // xy = edge blend distance, z = feature strength, w = probe count
    blend_params: vec4<f32>,
}

struct ReflectionProbeGridUniforms {
    // xy = world origin, z = height offset, w = enabled
    grid_origin: vec4<f32>,
    // x = spacing_x, y = spacing_y, z = dims_x, w = dims_y
    grid_params: vec4<f32>,
    // xy = edge blend distance, z = feature strength, w = probe count
    blend_params: vec4<f32>,
    // xyz = scene bounds min, w = cubemap face resolution
    scene_bounds_min: vec4<f32>,
    // xyz = scene bounds max, w = mip count
    scene_bounds_max: vec4<f32>,
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
var<uniform> reflection_probe_grid: ReflectionProbeGridUniforms;

@group(6) @binding(4)
var reflection_probe_tex: texture_2d_array<f32>;

@group(6) @binding(5)
var reflection_probe_samp: sampler;

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

fn compute_probe_grid_blend(
    grid_origin: vec4<f32>,
    grid_params: vec4<f32>,
    blend_params: vec4<f32>,
    world_pos: vec3<f32>,
) -> ProbeGridBlend {
    var blend: ProbeGridBlend;
    blend.idx00 = 0u;
    blend.idx10 = 0u;
    blend.idx01 = 0u;
    blend.idx11 = 0u;
    blend.frac = vec2<f32>(0.0);
    blend.weight = 0.0;
    blend.valid = 0.0;

    if (grid_origin.w < 0.5) {
        return blend;
    }

    let dims = vec2<u32>(u32(grid_params.z), u32(grid_params.w));
    if (dims.x == 0u || dims.y == 0u) {
        return blend;
    }

    let spacing = max(grid_params.xy, vec2<f32>(1e-6));
    let grid_uv = (world_pos.xy - grid_origin.xy) / spacing;
    let grid_extent = vec2<f32>(f32(dims.x - 1u), f32(dims.y - 1u));

    var i0 = vec2<u32>(0u);
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

    let blend_dist = max(blend_params.xy, vec2<f32>(1e-6));
    var weight_x = 1.0;
    var weight_y = 1.0;
    if (dims.x > 1u) {
        let dist_x = min(grid_uv.x, grid_extent.x - grid_uv.x) * spacing.x;
        weight_x = clamp(dist_x / blend_dist.x, 0.0, 1.0);
    }
    if (dims.y > 1u) {
        let dist_y = min(grid_uv.y, grid_extent.y - grid_uv.y) * spacing.y;
        weight_y = clamp(dist_y / blend_dist.y, 0.0, 1.0);
    }

    blend.weight = min(weight_x, weight_y);
    blend.valid = 1.0;
    return blend;
}

fn sample_probe_irradiance(world_pos: vec3<f32>, normal: vec3<f32>) -> ProbeIrradianceResult {
    var result: ProbeIrradianceResult;
    result.irradiance = vec3<f32>(0.0);
    result.weight = 0.0;

    let blend = compute_probe_grid_blend(
        probe_grid.grid_origin,
        probe_grid.grid_params,
        probe_grid.blend_params,
        world_pos,
    );
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
    result.weight = clamp(blend.weight * probe_grid.blend_params.z, 0.0, 1.0);
    return result;
}

fn reflection_probe_center(index: u32) -> vec3<f32> {
    let dims = vec2<u32>(
        max(u32(reflection_probe_grid.grid_params.z), 1u),
        max(u32(reflection_probe_grid.grid_params.w), 1u),
    );
    let probe_x = index % dims.x;
    let probe_y = index / dims.x;
    let center_xy =
        reflection_probe_grid.grid_origin.xy +
        vec2<f32>(f32(probe_x), f32(probe_y)) * reflection_probe_grid.grid_params.xy;
    let terrain_span = max(u_terrain.spacing_h_exag.x, 1e-6);
    let terrain_uv = clamp(center_xy / terrain_span + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
    let center_z = sample_height_geom(terrain_uv) * u_terrain.spacing_h_exag.z + reflection_probe_grid.grid_origin.z;
    return vec3<f32>(center_xy, center_z);
}

fn reflection_probe_cell_extents() -> vec3<f32> {
    let dims = vec2<u32>(
        max(u32(reflection_probe_grid.grid_params.z), 1u),
        max(u32(reflection_probe_grid.grid_params.w), 1u),
    );
    let total_bounds = max(
        reflection_probe_grid.scene_bounds_max.xyz - reflection_probe_grid.scene_bounds_min.xyz,
        vec3<f32>(1e-3),
    );
    let half_x = select(total_bounds.x * 0.5, max(reflection_probe_grid.grid_params.x * 0.5, 1e-3), dims.x > 1u);
    let half_y = select(total_bounds.y * 0.5, max(reflection_probe_grid.grid_params.y * 0.5, 1e-3), dims.y > 1u);
    let half_z = max(total_bounds.z * 0.5, 1e-3);
    return vec3<f32>(half_x, half_y, half_z);
}

fn reflection_probe_box_project(
    world_pos: vec3<f32>,
    reflection_dir: vec3<f32>,
    probe_center: vec3<f32>,
) -> vec3<f32> {
    let dir = normalize(reflection_dir);
    let half_extents = reflection_probe_cell_extents();
    let box_min = vec3<f32>(
        probe_center.xy - half_extents.xy,
        reflection_probe_grid.scene_bounds_min.z,
    );
    let box_max = vec3<f32>(
        probe_center.xy + half_extents.xy,
        reflection_probe_grid.scene_bounds_max.z,
    );

    let tx = select(
        select(1.0e9, (box_min.x - world_pos.x) / dir.x, dir.x < -1.0e-4),
        (box_max.x - world_pos.x) / dir.x,
        dir.x > 1.0e-4,
    );
    let ty = select(
        select(1.0e9, (box_min.y - world_pos.y) / dir.y, dir.y < -1.0e-4),
        (box_max.y - world_pos.y) / dir.y,
        dir.y > 1.0e-4,
    );
    let tz = select(
        select(1.0e9, (box_min.z - world_pos.z) / dir.z, dir.z < -1.0e-4),
        (box_max.z - world_pos.z) / dir.z,
        dir.z > 1.0e-4,
    );

    let travel = max(min(tx, min(ty, tz)), 0.0);
    let hit_pos = world_pos + dir * travel;
    let corrected = hit_pos - probe_center;
    let corrected_len2 = dot(corrected, corrected);
    if (corrected_len2 > 1.0e-8) {
        return normalize(corrected);
    }
    return dir;
}

fn sample_reflection_probe_array(
    probe_index: u32,
    direction: vec3<f32>,
    roughness: f32,
) -> vec3<f32> {
    let dir = normalize(direction);
    let abs_dir = abs(dir);
    var face = 0u;
    var uv = vec2<f32>(0.5, 0.5);

    if (abs_dir.x >= abs_dir.y && abs_dir.x >= abs_dir.z) {
        let ma = max(abs_dir.x, 1.0e-6);
        if (dir.x > 0.0) {
            face = 0u;
            uv = 0.5 * (vec2<f32>(-dir.z, -dir.y) / ma + vec2<f32>(1.0));
        } else {
            face = 1u;
            uv = 0.5 * (vec2<f32>(dir.z, -dir.y) / ma + vec2<f32>(1.0));
        }
    } else if (abs_dir.y >= abs_dir.x && abs_dir.y >= abs_dir.z) {
        let ma = max(abs_dir.y, 1.0e-6);
        if (dir.y > 0.0) {
            face = 2u;
            uv = 0.5 * (vec2<f32>(dir.x, dir.z) / ma + vec2<f32>(1.0));
        } else {
            face = 3u;
            uv = 0.5 * (vec2<f32>(dir.x, -dir.z) / ma + vec2<f32>(1.0));
        }
    } else {
        let ma = max(abs_dir.z, 1.0e-6);
        if (dir.z > 0.0) {
            face = 4u;
            uv = 0.5 * (vec2<f32>(dir.x, -dir.y) / ma + vec2<f32>(1.0));
        } else {
            face = 5u;
            uv = 0.5 * (vec2<f32>(-dir.x, -dir.y) / ma + vec2<f32>(1.0));
        }
    }

    let mip_count = max(reflection_probe_grid.scene_bounds_max.w, 1.0);
    let max_mip = max(mip_count - 1.0, 0.0);
    let mip_level = clamp(roughness * roughness * max_mip, 0.0, max_mip);
    let layer_index = i32(probe_index * 6u + face);
    return textureSampleLevel(
        reflection_probe_tex,
        reflection_probe_samp,
        clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)),
        layer_index,
        mip_level,
    ).rgb;
}

fn sample_reflection_probe_at_index(
    probe_index: u32,
    world_pos: vec3<f32>,
    reflection_dir: vec3<f32>,
    roughness: f32,
) -> vec3<f32> {
    let probe_center = reflection_probe_center(probe_index);
    let corrected_dir = reflection_probe_box_project(world_pos, reflection_dir, probe_center);
    return sample_reflection_probe_array(probe_index, corrected_dir, roughness);
}

fn sample_reflection_probe(
    world_pos: vec3<f32>,
    reflection_dir: vec3<f32>,
    roughness: f32,
) -> ReflectionProbeResult {
    var result: ReflectionProbeResult;
    result.prefiltered_color = vec3<f32>(0.0);
    result.weight = 0.0;

    let blend = compute_probe_grid_blend(
        reflection_probe_grid.grid_origin,
        reflection_probe_grid.grid_params,
        reflection_probe_grid.blend_params,
        world_pos,
    );
    if (blend.valid < 0.5) {
        return result;
    }
    result.weight = clamp(blend.weight * reflection_probe_grid.blend_params.z, 0.0, 1.0);
    if (result.weight <= 0.0) {
        return result;
    }

    let s00 = sample_reflection_probe_at_index(blend.idx00, world_pos, reflection_dir, roughness);
    let s10 = sample_reflection_probe_at_index(blend.idx10, world_pos, reflection_dir, roughness);
    let s01 = sample_reflection_probe_at_index(blend.idx01, world_pos, reflection_dir, roughness);
    let s11 = sample_reflection_probe_at_index(blend.idx11, world_pos, reflection_dir, roughness);
    result.prefiltered_color = mix(
        mix(s00, s10, blend.frac.x),
        mix(s01, s11, blend.frac.x),
        blend.frac.y,
    );
    return result;
}

fn sample_reflection_probe_weight(world_pos: vec3<f32>) -> f32 {
    let blend = compute_probe_grid_blend(
        reflection_probe_grid.grid_origin,
        reflection_probe_grid.grid_params,
        reflection_probe_grid.blend_params,
        world_pos,
    );
    if (blend.valid < 0.5) {
        return 0.0;
    }
    return clamp(blend.weight * reflection_probe_grid.blend_params.z, 0.0, 1.0);
}
