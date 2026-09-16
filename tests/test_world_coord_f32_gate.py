"""Fail-closed inventory for every Rust narrowing primitive.

The inventory is deliberately name-agnostic: every ``as f32``, ``.as_vec*()``,
and f64/DVec-to-f32/Vec helper is occurrence-locked across production Rust.
Separate positive contracts prove that each viewer world-position route calls
the active Anchor inside the producing function.
"""

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SANCTIONED = "src/camera/anchor.rs"
SANCTIONED_DD_SPLITS = {
    (
        "src/core/dd.rs",
        "from_f64",
        "as_f32",
        1,
        "let hi = value as f32",
    ),
    (
        "src/core/dd.rs",
        "from_f64",
        "as_f32",
        2,
        "let lo = (value - hi as f64) as f32",
    ),
}

# Updated only after reviewing the complete inventory printed by a failure.
# The digest includes (file, function, operation, ordinal, normalized statement).
# The db06b15f freeze below is the reviewed baseline the TERMINUS reader,
# ANAMNESIS, and HELIOS records describe; LATER_REVIEWED_TRANSITIONS chains it
# to the current EXPECTED_CONVERSION_COUNT/SHA256.
REVIEWED_BASELINE_COUNT = 1545
# The previous 1438-site freeze already covered the reviewed ANAMNESIS,
# TESSELLA, and first SIDERA transitions described below. The d8313007 base
# source actually contained 1446 sites because SIDERA's later adversarial
# closure added twelve u32 viewport-dimension/reciprocal casts and one
# normalized celestial unit-direction conversion without refreshing this
# constant. SUBSTRATIA adds one u32 feedback-origin telemetry counter converted
# for Python float stats. It also moves four existing tile-index-to-normalized-
# UV casts from finish_frame to ingest_shader_feedback; that changes their
# occurrence ownership, but not the count. The physical TESSELLA picking audit
# also moved two existing screen-coordinate conversions from pixel corners to
# raster pixel centres; their count is unchanged. Its visibility CPU oracle
# adds seven raster-only conversions: pixel centres (2), normalized coarse
# texel steps (2), a bounded LUT index (1), and viewport projection (2). All
# reviewed primitives are render dimensions, normalized directions/UVs,
# bounded raster indices, or telemetry. None stores absolute world coordinates
# or bypasses the camera Anchor. The subsequent physical-terrain closure
# consolidated repeated clipmap ring coordinate construction, reducing the
# reviewed inventory without weakening the Anchor boundary. HELIOS (below)
# adds the curvature-aware GPU viewshed, terrain-to-sun shadow mask, and
# closed-form shadow-tip analysis plus earth-curvature traversal in the hybrid
# terrain reference; see REVIEWED_HELIOS_INVENTORY_TRANSITION for the exact
# reviewed additions. CARTOGRAPHER-PRIME then adds four f64-to-f32 conversions
# for a validated finite silhouette bounding box in labels/optimal.rs; these
# remain bounded screen-space coordinates and do not cross the Anchor boundary.
REVIEWED_BASELINE_SHA256 = "b60331341dbeeb3c24a16fe52b92f1c51f18d8dffcf51d7e4132ba79d35ee065"

# Current freeze: the baseline plus every LATER_REVIEWED_TRANSITIONS entry.
EXPECTED_CONVERSION_COUNT = 1643
EXPECTED_CONVERSION_SHA256 = "6449dccf48eff56b5af8605d52f07a6469226eedbca796aea7887ddcca895d77"

# The reviewed TERMINUS reader transition remains locked below. COMPENDIUM adds
# four integer-to-f32 reconstruction conversions in predict.rs; those are
# included in the current count and digest above without weakening the reader
# transition assertion.
REVIEWED_INVENTORY_TRANSITION = {
    "current_count": REVIEWED_BASELINE_COUNT,
    "removed": (
        "src/terrain/cog/cog_reader.rs",
        "decode_heights",
        "as_f32",
        1,
        "heights.push(f64::from_le_bytes(bytes) as f32)",
    ),
    "added": (
        "src/terrain/cog/cog_reader.rs",
        "decode_heights",
        "as_f32",
        1,
        "heights.push(f64::from_le_bytes(read_le_bytes8(data, i * 8)) as f32)",
    ),
}

# ANAMNESIS retained the exact five adjudication conversions and moved them
# into the incremental implementation beneath the compatibility wrapper. This
# transition records that function-only ownership change without relaxing the
# occurrence count or any normalized conversion statement.
REVIEWED_ANAMNESIS_INVENTORY_TRANSITION = {
    # Re-based on main at the merge: the pre-transition tree is now main rather
    # than this branch's original base, so the count and digest are main's.
    "base_count": 1545,
    "base_digest": "9850587e94805c6d45e321cc54f5ea40dc54e6efa7facbcc45f17b00925283d4",
    "result_digest": REVIEWED_BASELINE_SHA256,
    "path": "src/offscreen/adjudication_raster.rs",
    "removed_function": "render_raster_reference",
    "added_function": "render_raster_reference_incremental",
    "statements": (
        "let aspect = width as f32 / height as f32",
        "let aspect = width as f32 / height as f32",
        "u.misc = [desc.plane_half_extent, i as f32, 1.0, 0.0]",
        "let o = (k as f32 + 0.5) / SSAA as f32 - 0.5",
        "let o = (k as f32 + 0.5) / SSAA as f32 - 0.5",
    ),
}

# HELIOS adds the curvature- and refraction-aware GPU viewshed, the direct
# terrain-to-sun shadow mask, and the closed-form curved-Earth shadow-tip
# analysis in src/terrain/analysis/viewshed.rs and src/py_functions/geodesy.rs,
# plus earth-curvature traversal uniforms in the hybrid terrain reference
# (src/path_tracing/hybrid_compute/terrain_heightfield.rs and render_terrain.rs).
# The 39 reviewed additions are render-space narrowing conversions: grid
# distances/azimuths, pixel-centre offsets, bounded observer coordinates,
# normalized latitude/longitude radians, curvature radius coefficients, and
# runtime-contract telemetry. The 3 removals are moved/consolidated module
# constants (origin_x/origin_z) and a duplicated check_range occurrence; none
# relaxes the Anchor world-coordinate boundary. None of the additions stores
# absolute world coordinates or bypasses the camera Anchor.
REVIEWED_HELIOS_INVENTORY_TRANSITION = {
    # Re-based on main at the merge: the pre-transition tree is now main rather
    # than this branch's original base, so the count and digest are main's.
    "base_count": 1545,
    "base_digest": "9850587e94805c6d45e321cc54f5ea40dc54e6efa7facbcc45f17b00925283d4",
    "result_digest": REVIEWED_BASELINE_SHA256,
    "added_sites": (
        (
            "src/py_functions/geodesy.rs",
            "terrain_grid_heights",
            "as_f32",
            1,
            "positions_m.push([ (distance_m * azimuth.sin()) as f32, (distance_m * azimuth.cos()) as f32, ])",
        ),
        (
            "src/py_functions/geodesy.rs",
            "terrain_shadow_mask",
            "as_f32",
            1,
            "geodetic_positions_and_sun.push([ latitude.to_radians() as f32, longitude.to_radians() as f32, solar.azimuth_deg.to_radians() as f32, launch_elevation_deg.to_radians() as f32, ])",
        ),
        (
            "src/terrain/analysis/viewshed.rs",
            "<module>",
            "as_f32",
            1,
            "Ok([ inv_meridional as f32, inv_prime_vertical as f32, one_minus_k as f32, f32::from(!matches!(options.earth_model, EarthModel::Flat)), ])",
        ),
        (
            "src/path_tracing/hybrid_compute/terrain_heightfield.rs",
            "<module>",
            "as_f32",
            1,
            "(0.5 / effective_radius) as f32",
        ),
        (
            "src/path_tracing/hybrid_compute/render_terrain.rs",
            "record_runtime_contract",
            "as_f32",
            12,
            "check( , &[earth_curvature.enabled as f32], 0.0, 1.0, )",
        ),
    ),
    "removed_sites": (
        (
            "src/path_tracing/hybrid_compute/render_terrain.rs",
            "record_runtime_contract",
            "as_f32",
            12,
            "observed.check_range( , , None, m_min as f32, m_max as f32, 0.0, 512.0, )",
        ),
        (
            "src/path_tracing/hybrid_compute/terrain_heightfield.rs",
            "<module>",
            "as_f32",
            1,
            "let origin_x = -0.5 * (self.width as f32 - 1.0) * spacing_x",
        ),
        (
            "src/path_tracing/hybrid_compute/terrain_heightfield.rs",
            "<module>",
            "as_f32",
            2,
            "let origin_z = -0.5 * (self.height as f32 - 1.0) * spacing_z",
        ),
    ),
}

# PROMETHEUS (cb161cc4 convergence estimator / traversal metrics / GPU SDF /
# real-DEM goldens) plus TERRA v2 (7adf5ce8) moved the reviewed tree from the
# db06b15f freeze (1545 sites) to ae20a1e3 (1608). Reviewed additions:
# adjudication plane-irradiance bake over the bounded fixture plane UV extent
# (texel centres, Fibonacci ray fractions, occlusion/bounce weights, bake
# texel indices), perspective aspect and SSAA tent weights in the unified
# adjudication `render`, Welford estimator variance, per-pixel frame means,
# and runtime-contract telemetry that the rename record_runtime_contract ->
# observe_runtime_contract re-owns and extends with exact-value checks. The
# removals are those re-owned rows, the ANAMNESIS adjudication conversions
# rewritten into `render`, base-uniform aspect rows, render_terrain_reference
# -> render_terrain_reference_inner re-ownership, and TERRA v2 dropping one
# IR-constant f64 narrowing in verify/ir/expr.rs. Every site is a render
# dimension, count, normalized/bounded fixture coordinate, estimator
# statistic, or telemetry value; none stores absolute world coordinates or
# bypasses the camera Anchor.
REVIEWED_PROMETHEUS_INVENTORY_TRANSITION = {
    "base_count": REVIEWED_BASELINE_COUNT,
    "base_digest": REVIEWED_BASELINE_SHA256,
    "result_count": 1608,
    "result_digest": "e571b46cc53851e4189b7b8fc406495224f58f4005e859f9bd3688dd13ca8bd8",
    "added_sites": (
        ('src/offscreen/adjudication_raster.rs', 'bake_plane_indirect', 'as_f32', 1, 'let p = Vec3::new( PLANE_UV_MIN + (i as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, 0.0, PLANE_UV_MIN + (j as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, )'),
        ('src/offscreen/adjudication_raster.rs', 'bake_plane_indirect', 'as_f32', 2, 'let p = Vec3::new( PLANE_UV_MIN + (i as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, 0.0, PLANE_UV_MIN + (j as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, )'),
        ('src/offscreen/adjudication_raster.rs', 'bake_plane_indirect', 'as_f32', 3, 'let p = Vec3::new( PLANE_UV_MIN + (i as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, 0.0, PLANE_UV_MIN + (j as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, )'),
        ('src/offscreen/adjudication_raster.rs', 'bake_plane_indirect', 'as_f32', 4, 'let p = Vec3::new( PLANE_UV_MIN + (i as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, 0.0, PLANE_UV_MIN + (j as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, )'),
        ('src/offscreen/adjudication_raster.rs', 'bake_plane_indirect', 'as_f32', 5, 'let ct = 1.0 - (k as f32 + 0.5) / BAKE_RAYS as f32'),
        ('src/offscreen/adjudication_raster.rs', 'bake_plane_indirect', 'as_f32', 6, 'let ct = 1.0 - (k as f32 + 0.5) / BAKE_RAYS as f32'),
        ('src/offscreen/adjudication_raster.rs', 'bake_plane_indirect', 'as_f32', 7, 'let phi = k as f32 * 2.399_963_2'),
        ('src/offscreen/adjudication_raster.rs', 'bake_plane_indirect', 'as_f32', 8, 'let occ = 2.0 * w_sky / BAKE_RAYS as f32'),
        ('src/offscreen/adjudication_raster.rs', 'bake_plane_indirect', 'as_f32', 9, 'let v_sun = ray_hit_sphere(desc, p + Vec3::Y * 1e-3, light, None).is_none() as i32 as f32'),
        ('src/offscreen/adjudication_raster.rs', 'bake_plane_indirect', 'as_f32', 10, 'let u = ((p.x - PLANE_UV_MIN) / PLANE_UV_EXTENT * BAKE_RES as f32) .clamp(0.0, BAKE_RES as f32 - 1.0) as usize'),
        ('src/offscreen/adjudication_raster.rs', 'bake_plane_indirect', 'as_f32', 11, 'let u = ((p.x - PLANE_UV_MIN) / PLANE_UV_EXTENT * BAKE_RES as f32) .clamp(0.0, BAKE_RES as f32 - 1.0) as usize'),
        ('src/offscreen/adjudication_raster.rs', 'bake_plane_indirect', 'as_f32', 12, 'let v = ((p.z - PLANE_UV_MIN) / PLANE_UV_EXTENT * BAKE_RES as f32) .clamp(0.0, BAKE_RES as f32 - 1.0) as usize'),
        ('src/offscreen/adjudication_raster.rs', 'bake_plane_indirect', 'as_f32', 13, 'let v = ((p.z - PLANE_UV_MIN) / PLANE_UV_EXTENT * BAKE_RES as f32) .clamp(0.0, BAKE_RES as f32 - 1.0) as usize'),
        ('src/offscreen/adjudication_raster.rs', 'bake_plane_indirect', 'as_f32', 14, 'let p = Vec3::new( PLANE_UV_MIN + (i as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, 0.0, PLANE_UV_MIN + (j as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, )'),
        ('src/offscreen/adjudication_raster.rs', 'bake_plane_indirect', 'as_f32', 15, 'let p = Vec3::new( PLANE_UV_MIN + (i as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, 0.0, PLANE_UV_MIN + (j as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, )'),
        ('src/offscreen/adjudication_raster.rs', 'bake_plane_indirect', 'as_f32', 16, 'let p = Vec3::new( PLANE_UV_MIN + (i as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, 0.0, PLANE_UV_MIN + (j as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, )'),
        ('src/offscreen/adjudication_raster.rs', 'bake_plane_indirect', 'as_f32', 17, 'let p = Vec3::new( PLANE_UV_MIN + (i as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, 0.0, PLANE_UV_MIN + (j as f32 + 0.5) / BAKE_RES as f32 * PLANE_UV_EXTENT, )'),
        ('src/offscreen/adjudication_raster.rs', 'bake_plane_indirect', 'as_f32', 18, 'let ct = 1.0 - (k as f32 + 0.5) / BAKE_RAYS as f32'),
        ('src/offscreen/adjudication_raster.rs', 'bake_plane_indirect', 'as_f32', 19, 'let ct = 1.0 - (k as f32 + 0.5) / BAKE_RAYS as f32'),
        ('src/offscreen/adjudication_raster.rs', 'bake_plane_indirect', 'as_f32', 20, 'let phi = k as f32 * 2.399_963_2'),
        ('src/offscreen/adjudication_raster.rs', 'bake_plane_indirect', 'as_f32', 21, '} } let em = a_plane * e_bounce * (2.0 / BAKE_RAYS as f32)'),
        ('src/offscreen/adjudication_raster.rs', 'render', 'as_f32', 1, 'let projection = Mat4::perspective_rh(desc.fov_y_rad(), width as f32 / height as f32, near, far)'),
        ('src/offscreen/adjudication_raster.rs', 'render', 'as_f32', 2, 'let projection = Mat4::perspective_rh(desc.fov_y_rad(), width as f32 / height as f32, near, far)'),
        ('src/offscreen/adjudication_raster.rs', 'render', 'as_f32', 3, 'let weights: Vec<f32> = (0..SSAA) .map(|k| 1.0 - (((k as f32 + 0.5) / SSAA as f32 - 0.5).abs() / 0.5)) .collect()'),
        ('src/offscreen/adjudication_raster.rs', 'render', 'as_f32', 4, 'let weights: Vec<f32> = (0..SSAA) .map(|k| 1.0 - (((k as f32 + 0.5) / SSAA as f32 - 0.5).abs() / 0.5)) .collect()'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'estimator_variance', 'as_f32', 1, '} Ok((m2 as f64 / (frames as f64 * (frames - 1) as f64)) as f32)'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 1, 'check( , &[base.width as f32], 8.0, 8.0)'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 2, 'check( , &[base.height as f32], 8.0, 8.0)'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 3, 'check( , &[0.0, frames.saturating_sub(1) as f32], 0.0, 3.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 4, 'check( , &[ 0.0, (desc.width.max(desc.height).div_ceil(8) * 8 - 1) as f32, ], 0.0, 7.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 5, 'check( , &[hybrid.traversal_mode as f32], 3.0, 3.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 6, 'check( , &[hybrid.mesh_vertex_count as f32], 0.0, 0.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 7, 'check( , &[hybrid.mesh_index_count as f32], 0.0, 0.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 8, 'check( , &[lighting.shadows_enabled as f32], 1.0, 1.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 9, 'check( , &terrain.dims.map(|value| value as f32), 3.0, 4.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 10, 'check( , &terrain.mips.map(|value| value as f32), 0.0, 3.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 11, 'check( , &terrain.extra.map(|value| value as f32), 0.0, 32.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 12, 'check( , &[earth_curvature.enabled as f32], 0.0, 1.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 13, 'exact( , &[base.width as f32], &[desc.width as f32])'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 14, 'exact( , &[base.width as f32], &[desc.width as f32])'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 15, 'exact( , &[base.height as f32], &[desc.height as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 16, 'exact( , &[base.height as f32], &[desc.height as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 17, 'exact( , &[base.cam_aspect], &[desc.width as f32 / desc.height as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 18, 'exact( , &[base.cam_aspect], &[desc.width as f32 / desc.height as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 19, 'exact( , &[base.frame_index as f32], &[frames.saturating_sub(1) as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 20, 'exact( , &[base.frame_index as f32], &[frames.saturating_sub(1) as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 21, 'exact( , &[hybrid.mesh_vertex_count as f32], &[vertices as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 22, 'exact( , &[hybrid.mesh_vertex_count as f32], &[vertices as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 23, 'exact( , &[hybrid.mesh_index_count as f32], &[indices as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 24, 'exact( , &[hybrid.mesh_index_count as f32], &[indices as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 25, 'exact( , &[hybrid.traversal_mode as f32], &[if desc.mesh.is_some() || desc.sdf_scene.is_some() { TraversalMode::Hybrid'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 26, 'TraversalMode::TerrainOnly } as u32 as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 27, 'exact( , &[sdf_prims as f32], &[sdf_prims as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 28, 'exact( , &[sdf_prims as f32], &[sdf_prims as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 29, 'exact( , &[sdf_nodes as f32], &[sdf_nodes as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 30, 'exact( , &[sdf_nodes as f32], &[sdf_nodes as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 31, 'exact( , &[lighting.shadows_enabled as f32], &[1.0], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 32, 'exact( , &terrain.origin_spacing, &[ -0.5 * (desc.dem_width as f32 - 1.0) * desc.spacing.0, -0.5 * (desc.dem_height as f32 - 1.0) * desc.spacing.1, desc.spacing.0, desc.spacing.1, ], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 33, 'exact( , &terrain.origin_spacing, &[ -0.5 * (desc.dem_width as f32 - 1.0) * desc.spacing.0, -0.5 * (desc.dem_height as f32 - 1.0) * desc.spacing.1, desc.spacing.0, desc.spacing.1, ], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 34, 'exact( , &terrain.dims.map(|v| v as f32), &[ desc.dem_width as f32, desc.dem_height as f32, cw as f32, ch as f32, ], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 35, 'exact( , &terrain.dims.map(|v| v as f32), &[ desc.dem_width as f32, desc.dem_height as f32, cw as f32, ch as f32, ], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 36, 'exact( , &terrain.dims.map(|v| v as f32), &[ desc.dem_width as f32, desc.dem_height as f32, cw as f32, ch as f32, ], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 37, 'exact( , &terrain.dims.map(|v| v as f32), &[ desc.dem_width as f32, desc.dem_height as f32, cw as f32, ch as f32, ], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 38, 'exact( , &terrain.dims.map(|v| v as f32), &[ desc.dem_width as f32, desc.dem_height as f32, cw as f32, ch as f32, ], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 39, 'exact( , &terrain.mips.map(|v| v as f32), &[mip_count as f32, 1.0, env_w as f32, env_h as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 40, 'exact( , &terrain.mips.map(|v| v as f32), &[mip_count as f32, 1.0, env_w as f32, env_h as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 41, 'exact( , &terrain.mips.map(|v| v as f32), &[mip_count as f32, 1.0, env_w as f32, env_h as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 42, 'exact( , &terrain.mips.map(|v| v as f32), &[mip_count as f32, 1.0, env_w as f32, env_h as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 43, 'exact( , &terrain.extra.map(|v| v as f32), &[desc.spp as f32, WELFORD_WINDOW as f32, 0.0, 0.0], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 44, 'exact( , &terrain.extra.map(|v| v as f32), &[desc.spp as f32, WELFORD_WINDOW as f32, 0.0, 0.0], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 45, 'exact( , &terrain.extra.map(|v| v as f32), &[desc.spp as f32, WELFORD_WINDOW as f32, 0.0, 0.0], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 46, 'check( , &accum_frames, frames as f32, frames as f32, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 47, 'check( , &accum_frames, frames as f32, frames as f32, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 48, '} observed.check_range( , , None, frames as f32, frames as f32, desc.min_frames.max(2) as f32, desc.max_frames as f32, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 49, '} observed.check_range( , , None, frames as f32, frames as f32, desc.min_frames.max(2) as f32, desc.max_frames as f32, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 50, '} observed.check_range( , , None, frames as f32, frames as f32, desc.min_frames.max(2) as f32, desc.max_frames as f32, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 51, '} observed.check_range( , , None, frames as f32, frames as f32, desc.min_frames.max(2) as f32, desc.max_frames as f32, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 52, 'observed.check_range( , , None, u32::from(convergence_valid) as f32, u32::from(convergence_valid) as f32, 1.0, 1.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 53, 'observed.check_range( , , None, u32::from(convergence_valid) as f32, u32::from(convergence_valid) as f32, 1.0, 1.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 54, 'observed.check_range( , , None, u32::from(finite_samples) as f32, u32::from(finite_samples) as f32, 1.0, 1.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 55, 'observed.check_range( , , None, u32::from(finite_samples) as f32, u32::from(finite_samples) as f32, 1.0, 1.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 56, 'for (name, values, lo, hi) in [ ( , valid .iter() .map(|r| r.sample.light_type as f32) .collect::<Vec<_>>(), 1.0, 1.0, ), ( , valid .iter() .map(|r| r.sample.light_index as f32) .collect::<Vec<_>>(), 0.0, 0.0, ), ( , valid.iter().map(|r| r.target_pdf).collect::<Vec<_>>(), 1.0, 1.0, ), ] { if !values.is_empty() { let (actual_lo, actual_hi) = finite_min_max(values.into_iter())'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 57, 'for (name, values, lo, hi) in [ ( , valid .iter() .map(|r| r.sample.light_type as f32) .collect::<Vec<_>>(), 1.0, 1.0, ), ( , valid .iter() .map(|r| r.sample.light_index as f32) .collect::<Vec<_>>(), 0.0, 0.0, ), ( , valid.iter().map(|r| r.target_pdf).collect::<Vec<_>>(), 1.0, 1.0, ), ] { if !values.is_empty() { let (actual_lo, actual_hi) = finite_min_max(values.into_iter())'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 58, 'observed.check_range( , , None, m_min as f32, m_max as f32, 0.0, 512.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 59, 'observed.check_range( , , None, m_min as f32, m_max as f32, 0.0, 512.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'render_terrain_reference_inner', 'as_f32', 1, 'width, height, frame_index: 0, aov_flags: 0xFF, cam_origin: desc.cam_origin, cam_fov_y: desc.fov_y_deg.to_radians(), cam_right: right.into(), cam_aspect: width as f32 / height as f32, cam_up: up.into(), cam_exposure: exposure, cam_forward: forward.into(), seed_hi: desc.seed, seed_lo: desc .seed .wrapping_mul(0x9E37_79B9) .wrapping_add(0x85EB_CA6B), _pad_end: [0'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'render_terrain_reference_inner', 'as_f32', 2, 'width, height, frame_index: 0, aov_flags: 0xFF, cam_origin: desc.cam_origin, cam_fov_y: desc.fov_y_deg.to_radians(), cam_right: right.into(), cam_aspect: width as f32 / height as f32, cam_up: up.into(), cam_exposure: exposure, cam_forward: forward.into(), seed_hi: desc.seed, seed_lo: desc .seed .wrapping_mul(0x9E37_79B9) .wrapping_add(0x85EB_CA6B), _pad_end: [0'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'render_terrain_reference_inner', 'as_f32', 3, 'let v = px[c] / frames as f32'),
    ),
    "removed_sites": (
        ('src/offscreen/adjudication_raster.rs', 'base_uniforms', 'as_f32', 1, 'cam_origin_fovy: [origin.x, origin.y, origin.z, desc.fov_y_rad()], cam_right_aspect: [right.x, right.y, right.z, aspect], cam_up_w: [up.x, up.y, up.z, rw as f32], cam_forward_h: [forward.x, forward.y, forward.z, rh as f32], sun_dir_intensity: [sun.x, sun.y, sun.z, desc.sun_intensity], sun_color_pad: [desc.sun_color[0], desc.sun_color[1], desc.sun_color[2], 0.0], environment: { let e = desc.environment_raw()'),
        ('src/offscreen/adjudication_raster.rs', 'base_uniforms', 'as_f32', 2, 'cam_origin_fovy: [origin.x, origin.y, origin.z, desc.fov_y_rad()], cam_right_aspect: [right.x, right.y, right.z, aspect], cam_up_w: [up.x, up.y, up.z, rw as f32], cam_forward_h: [forward.x, forward.y, forward.z, rh as f32], sun_dir_intensity: [sun.x, sun.y, sun.z, desc.sun_intensity], sun_color_pad: [desc.sun_color[0], desc.sun_color[1], desc.sun_color[2], 0.0], environment: { let e = desc.environment_raw()'),
        ('src/offscreen/adjudication_raster.rs', 'render_raster_reference_incremental', 'as_f32', 1, 'let aspect = width as f32 / height as f32'),
        ('src/offscreen/adjudication_raster.rs', 'render_raster_reference_incremental', 'as_f32', 2, 'let aspect = width as f32 / height as f32'),
        ('src/offscreen/adjudication_raster.rs', 'render_raster_reference_incremental', 'as_f32', 3, 'u.misc = [desc.plane_half_extent, i as f32, 1.0, 0.0]'),
        ('src/offscreen/adjudication_raster.rs', 'render_raster_reference_incremental', 'as_f32', 4, 'let o = (k as f32 + 0.5) / SSAA as f32 - 0.5'),
        ('src/offscreen/adjudication_raster.rs', 'render_raster_reference_incremental', 'as_f32', 5, 'let o = (k as f32 + 0.5) / SSAA as f32 - 0.5'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'record_runtime_contract', 'as_f32', 1, 'check( , &[base.width as f32], 8.0, 8.0)'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'record_runtime_contract', 'as_f32', 2, 'check( , &[base.height as f32], 8.0, 8.0)'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'record_runtime_contract', 'as_f32', 3, 'check( , &[0.0, frames.saturating_sub(1) as f32], 0.0, 3.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'record_runtime_contract', 'as_f32', 4, 'check( , &[ 0.0, (desc.width.max(desc.height).div_ceil(8) * 8 - 1) as f32, ], 0.0, 7.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'record_runtime_contract', 'as_f32', 5, 'check( , &[hybrid.traversal_mode as f32], 3.0, 3.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'record_runtime_contract', 'as_f32', 6, 'check( , &[hybrid.mesh_vertex_count as f32], 0.0, 0.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'record_runtime_contract', 'as_f32', 7, 'check( , &[hybrid.mesh_index_count as f32], 0.0, 0.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'record_runtime_contract', 'as_f32', 8, 'check( , &[lighting.shadows_enabled as f32], 1.0, 1.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'record_runtime_contract', 'as_f32', 9, 'check( , &terrain.dims.map(|value| value as f32), 3.0, 4.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'record_runtime_contract', 'as_f32', 10, 'check( , &terrain.mips.map(|value| value as f32), 0.0, 3.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'record_runtime_contract', 'as_f32', 11, 'check( , &terrain.extra.map(|value| value as f32), 0.0, 32.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'record_runtime_contract', 'as_f32', 12, 'check( , &[earth_curvature.enabled as f32], 0.0, 1.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'record_runtime_contract', 'as_f32', 13, 'observed.check_range( , , None, m_min as f32, m_max as f32, 0.0, 512.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'record_runtime_contract', 'as_f32', 14, 'observed.check_range( , , None, m_min as f32, m_max as f32, 0.0, 512.0, )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'render_terrain_reference', 'as_f32', 1, 'width, height, frame_index: 0, aov_flags: 0xFF, cam_origin: desc.cam_origin, cam_fov_y: desc.fov_y_deg.to_radians(), cam_right: right.into(), cam_aspect: width as f32 / height as f32, cam_up: up.into(), cam_exposure: exposure, cam_forward: forward.into(), seed_hi: desc.seed, seed_lo: desc.seed ^ 0x85EB_CA6B, _pad_end: [0'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'render_terrain_reference', 'as_f32', 2, 'width, height, frame_index: 0, aov_flags: 0xFF, cam_origin: desc.cam_origin, cam_fov_y: desc.fov_y_deg.to_radians(), cam_right: right.into(), cam_aspect: width as f32 / height as f32, cam_up: up.into(), cam_exposure: exposure, cam_forward: forward.into(), seed_hi: desc.seed, seed_lo: desc.seed ^ 0x85EB_CA6B, _pad_end: [0'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'render_terrain_reference', 'as_f32', 3, 'let n = n_window as f32'),
        ('src/verify/ir/expr.rs', 'eval_expr', 'as_f32', 1, 'Value::Float(Interval::constant(*value as f32))'),
    ),
}

# DIFFERENTIA (f06a026a) moves ae20a1e3 (1608) to da2fd15c (1643). Reviewed
# additions: inverse-solver loss and mean-radiance scalars, replicate ordinal/frame
# reciprocals, Adam cosine-decay and tail-average fractions, log-albedo
# means, pixel-centre bilinear resample/adjoint offsets over render and
# albedo-map texel grids, frame-uniform camera aspect, and albedo-map
# dimensions packed into the inverse parameter block. The six removals are
# runtime-contract rows rewritten to carry the terrain flag bits and the
# spectral reservoir channel. None stores absolute world coordinates or
# bypasses the camera Anchor.
REVIEWED_DIFFERENTIA_INVENTORY_TRANSITION = {
    "base_count": REVIEWED_PROMETHEUS_INVENTORY_TRANSITION["result_count"],
    "base_digest": REVIEWED_PROMETHEUS_INVENTORY_TRANSITION["result_digest"],
    "result_count": 1643,
    "result_digest": "6449dccf48eff56b5af8605d52f07a6469226eedbca796aea7887ddcca895d77",
    "added_sites": (
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 39, 'exact( , &terrain.mips.map(|v| v as f32), &[mip_count as f32, terrain_flags, env_w as f32, env_h as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 40, 'exact( , &terrain.mips.map(|v| v as f32), &[mip_count as f32, terrain_flags, env_w as f32, env_h as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 41, 'exact( , &terrain.mips.map(|v| v as f32), &[mip_count as f32, terrain_flags, env_w as f32, env_h as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 42, 'exact( , &terrain.mips.map(|v| v as f32), &[mip_count as f32, terrain_flags, env_w as f32, env_h as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 56, 'for (name, values, lo, hi) in [ ( , valid .iter() .map(|r| r.sample.light_type as f32) .collect::<Vec<_>>(), 1.0, 1.0, ), ( , valid .iter() .map(|r| r.sample.light_index as f32) .collect::<Vec<_>>(), 0.0, light_index_hi, ), ( , valid.iter().map(|r| r.target_pdf).collect::<Vec<_>>(), target_pdf_bounds.0, target_pdf_bounds.1, ), ] { if !values.is_empty() { let (actual_lo, actual_hi) = finite_min_max(values.into_iter())'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 57, 'for (name, values, lo, hi) in [ ( , valid .iter() .map(|r| r.sample.light_type as f32) .collect::<Vec<_>>(), 1.0, 1.0, ), ( , valid .iter() .map(|r| r.sample.light_index as f32) .collect::<Vec<_>>(), 0.0, light_index_hi, ), ( , valid.iter().map(|r| r.target_pdf).collect::<Vec<_>>(), target_pdf_bounds.0, target_pdf_bounds.1, ), ] { if !values.is_empty() { let (actual_lo, actual_hi) = finite_min_max(values.into_iter())'),
        ('src/path_tracing/inverse/mod.rs', '<module>', 'as_f32', 1, 'let inv_r = 1.0 / reps as f32'),
        ('src/path_tracing/inverse/mod.rs', '<module>', 'as_f32', 2, 'Ok((mean_loss as f32, d_albedo, scalars))'),
        ('src/path_tracing/inverse/mod.rs', '<module>', 'as_f32', 3, 'let frames = scene.frames.max(1) as f32'),
        ('src/path_tracing/inverse/mod.rs', '<module>', 'as_f32', 4, 'Ok(( rgba, [ (mean[0] / n) as f32, (mean[1] / n) as f32, (mean[2] / n) as f32, ], ))'),
        ('src/path_tracing/inverse/mod.rs', '<module>', 'as_f32', 5, 'Ok(( rgba, [ (mean[0] / n) as f32, (mean[1] / n) as f32, (mean[2] / n) as f32, ], ))'),
        ('src/path_tracing/inverse/mod.rs', '<module>', 'as_f32', 6, 'Ok(( rgba, [ (mean[0] / n) as f32, (mean[1] / n) as f32, (mean[2] / n) as f32, ], ))'),
        ('src/path_tracing/inverse/mod.rs', 'downsample3_adjoint', 'as_f32', 1, 'let tz = (iz as f32 + 0.5) * sh as f32 / dh as f32 - 0.5'),
        ('src/path_tracing/inverse/mod.rs', 'downsample3_adjoint', 'as_f32', 2, 'let tz = (iz as f32 + 0.5) * sh as f32 / dh as f32 - 0.5'),
        ('src/path_tracing/inverse/mod.rs', 'downsample3_adjoint', 'as_f32', 3, 'let tz = (iz as f32 + 0.5) * sh as f32 / dh as f32 - 0.5'),
        ('src/path_tracing/inverse/mod.rs', 'downsample3_adjoint', 'as_f32', 4, 'let fz = (tz - z0 as f32).clamp(0.0, 1.0)'),
        ('src/path_tracing/inverse/mod.rs', 'downsample3_adjoint', 'as_f32', 5, 'let tx = (ix as f32 + 0.5) * sw as f32 / dw as f32 - 0.5'),
        ('src/path_tracing/inverse/mod.rs', 'downsample3_adjoint', 'as_f32', 6, 'let tx = (ix as f32 + 0.5) * sw as f32 / dw as f32 - 0.5'),
        ('src/path_tracing/inverse/mod.rs', 'downsample3_adjoint', 'as_f32', 7, 'let tx = (ix as f32 + 0.5) * sw as f32 / dw as f32 - 0.5'),
        ('src/path_tracing/inverse/mod.rs', 'downsample3_adjoint', 'as_f32', 8, 'let fx = (tx - x0 as f32).clamp(0.0, 1.0)'),
        ('src/path_tracing/inverse/mod.rs', 'frame_uniforms', 'as_f32', 1, '0 }, cam_origin: desc.cam_origin, cam_fov_y: desc.fov_y_deg.to_radians(), cam_right: right.into(), cam_aspect: desc.width as f32 / desc.height as f32, cam_up: up.into(), cam_exposure: desc.exposure, cam_forward: forward.into(), seed_hi: seed, seed_lo: seed.wrapping_mul(0x9E37_79B9).wrapping_add(0x85EB_CA6B), _pad_end: [0'),
        ('src/path_tracing/inverse/mod.rs', 'frame_uniforms', 'as_f32', 2, '0 }, cam_origin: desc.cam_origin, cam_fov_y: desc.fov_y_deg.to_radians(), cam_right: right.into(), cam_aspect: desc.width as f32 / desc.height as f32, cam_up: up.into(), cam_exposure: desc.exposure, cam_forward: forward.into(), seed_hi: seed, seed_lo: seed.wrapping_mul(0x9E37_79B9).wrapping_add(0x85EB_CA6B), _pad_end: [0'),
        ('src/path_tracing/inverse/mod.rs', 'solve', 'as_f32', 1, 'let tail_from = ((desc.iters as f32) * 0.7) as u32'),
        ('src/path_tracing/inverse/mod.rs', 'solve', 'as_f32', 2, 'let m0 = log_alb.iter().copied().sum::<f32>() / log_alb.len() as f32'),
        ('src/path_tracing/inverse/mod.rs', 'solve', 'as_f32', 3, '} let cos_decay = 0.5 * (1.0 + (std::f32::consts::PI * it as f32 / desc.iters.max(1) as f32).cos())'),
        ('src/path_tracing/inverse/mod.rs', 'solve', 'as_f32', 4, '} let cos_decay = 0.5 * (1.0 + (std::f32::consts::PI * it as f32 / desc.iters.max(1) as f32).cos())'),
        ('src/path_tracing/inverse/mod.rs', 'solve', 'as_f32', 5, '*v = (m[i % 3] / nt) as f32'),
        ('src/path_tracing/inverse/mod.rs', 'solve', 'as_f32', 6, 'let inv_n = 1.0 / tail_n as f32'),
        ('src/path_tracing/inverse/mod.rs', 'upload_inv_params', 'as_f32', 1, 'gpu_params.rsv0[0] = 1.0 / reps.max(1) as f32'),
        ('src/path_tracing/inverse/mod.rs', 'upload_inv_params', 'as_f32', 2, 'gpu_params.rsv0[1] = rep_ord as f32'),
        ('src/path_tracing/inverse/mod.rs', 'upsample3', 'as_f32', 1, 'let tz = (iz as f32 + 0.5) * sh as f32 / dh as f32 - 0.5'),
        ('src/path_tracing/inverse/mod.rs', 'upsample3', 'as_f32', 2, 'let tz = (iz as f32 + 0.5) * sh as f32 / dh as f32 - 0.5'),
        ('src/path_tracing/inverse/mod.rs', 'upsample3', 'as_f32', 3, 'let tz = (iz as f32 + 0.5) * sh as f32 / dh as f32 - 0.5'),
        ('src/path_tracing/inverse/mod.rs', 'upsample3', 'as_f32', 4, 'let fz = (tz - z0 as f32).clamp(0.0, 1.0)'),
        ('src/path_tracing/inverse/mod.rs', 'upsample3', 'as_f32', 5, 'let tx = (ix as f32 + 0.5) * sw as f32 / dw as f32 - 0.5'),
        ('src/path_tracing/inverse/mod.rs', 'upsample3', 'as_f32', 6, 'let tx = (ix as f32 + 0.5) * sw as f32 / dw as f32 - 0.5'),
        ('src/path_tracing/inverse/mod.rs', 'upsample3', 'as_f32', 7, 'let tx = (ix as f32 + 0.5) * sw as f32 / dw as f32 - 0.5'),
        ('src/path_tracing/inverse/mod.rs', 'upsample3', 'as_f32', 8, 'let fx = (tx - x0 as f32).clamp(0.0, 1.0)'),
        ('src/path_tracing/inverse/params.rs', '<module>', 'as_f32', 1, 'dims: [ alb_dims.0, alb_dims.1, alb_dims.0 * alb_dims.1, frames.max(1), ], ctrl: [flags, frames.max(1).min(tile_size.max(1)), 0, 0], sunv: [sun_color[0], sun_color[1], sun_color[2], sun_intensity], fl: [ LOSS_STRUCT_W, SCORE_CLAMP, 1.0 / (pixel_count.max(1) as f32), 1.0 / ((frames.max(1) as f32) * (spp.max(1) as f32)), ], rsv0: [0.0'),
        ('src/path_tracing/inverse/params.rs', '<module>', 'as_f32', 2, 'dims: [ alb_dims.0, alb_dims.1, alb_dims.0 * alb_dims.1, frames.max(1), ], ctrl: [flags, frames.max(1).min(tile_size.max(1)), 0, 0], sunv: [sun_color[0], sun_color[1], sun_color[2], sun_intensity], fl: [ LOSS_STRUCT_W, SCORE_CLAMP, 1.0 / (pixel_count.max(1) as f32), 1.0 / ((frames.max(1) as f32) * (spp.max(1) as f32)), ], rsv0: [0.0'),
        ('src/path_tracing/inverse/params.rs', '<module>', 'as_f32', 3, 'dims: [ alb_dims.0, alb_dims.1, alb_dims.0 * alb_dims.1, frames.max(1), ], ctrl: [flags, frames.max(1).min(tile_size.max(1)), 0, 0], sunv: [sun_color[0], sun_color[1], sun_color[2], sun_intensity], fl: [ LOSS_STRUCT_W, SCORE_CLAMP, 1.0 / (pixel_count.max(1) as f32), 1.0 / ((frames.max(1) as f32) * (spp.max(1) as f32)), ], rsv0: [0.0'),
    ),
    "removed_sites": (
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 39, 'exact( , &terrain.mips.map(|v| v as f32), &[mip_count as f32, 1.0, env_w as f32, env_h as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 40, 'exact( , &terrain.mips.map(|v| v as f32), &[mip_count as f32, 1.0, env_w as f32, env_h as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 41, 'exact( , &terrain.mips.map(|v| v as f32), &[mip_count as f32, 1.0, env_w as f32, env_h as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 42, 'exact( , &terrain.mips.map(|v| v as f32), &[mip_count as f32, 1.0, env_w as f32, env_h as f32], )'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 56, 'for (name, values, lo, hi) in [ ( , valid .iter() .map(|r| r.sample.light_type as f32) .collect::<Vec<_>>(), 1.0, 1.0, ), ( , valid .iter() .map(|r| r.sample.light_index as f32) .collect::<Vec<_>>(), 0.0, 0.0, ), ( , valid.iter().map(|r| r.target_pdf).collect::<Vec<_>>(), 1.0, 1.0, ), ] { if !values.is_empty() { let (actual_lo, actual_hi) = finite_min_max(values.into_iter())'),
        ('src/path_tracing/hybrid_compute/render_terrain.rs', 'observe_runtime_contract', 'as_f32', 57, 'for (name, values, lo, hi) in [ ( , valid .iter() .map(|r| r.sample.light_type as f32) .collect::<Vec<_>>(), 1.0, 1.0, ), ( , valid .iter() .map(|r| r.sample.light_index as f32) .collect::<Vec<_>>(), 0.0, 0.0, ), ( , valid.iter().map(|r| r.target_pdf).collect::<Vec<_>>(), 1.0, 1.0, ), ] { if !values.is_empty() { let (actual_lo, actual_hi) = finite_min_max(values.into_iter())'),
    ),
}

# Transitions reviewed after the db06b15f freeze, in merge order. Earlier
# records (TERMINUS reader, ANAMNESIS, HELIOS) describe that frozen tree; a
# site they recorded may only disappear or reappear through one of these.
LATER_REVIEWED_TRANSITIONS = (
    REVIEWED_PROMETHEUS_INVENTORY_TRANSITION,
    REVIEWED_DIFFERENTIA_INVENTORY_TRANSITION,
)


def _strip_comments_and_strings(text: str) -> str:
    pattern = re.compile(
        r"//[^\n]*|/\*.*?\*/|r#*\".*?\"#*|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
        re.S,
    )
    return pattern.sub(lambda match: " " * len(match.group(0)), text)


def _remove_cfg_test_modules(text: str) -> str:
    text = _strip_comments_and_strings(text)
    marker = re.compile(r"#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]\s*mod\s+\w+\s*\{")
    while match := marker.search(text):
        depth = 1
        cursor = match.end()
        while cursor < len(text) and depth:
            depth += (text[cursor] == "{") - (text[cursor] == "}")
            cursor += 1
        text = text[: match.start()] + " " * (cursor - match.start()) + text[cursor:]
    return text


def _function_spans(text: str):
    spans = []
    for match in re.finditer(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)[^;{]*\{", text, re.S):
        depth = 1
        cursor = match.end()
        while cursor < len(text) and depth:
            depth += (text[cursor] == "{") - (text[cursor] == "}")
            cursor += 1
        spans.append((match.start(), cursor, match.group(1)))
    return spans


def _function_name(spans, position: int) -> str:
    return next(
        (name for start, end, name in spans if start <= position < end),
        "<module>",
    )


def _statement(text: str, position: int) -> str:
    start = max(text.rfind(";", 0, position), text.rfind("{", 0, position)) + 1
    end_candidates = [candidate for candidate in (text.find(";", position), text.find("}", position)) if candidate >= 0]
    end = min(end_candidates, default=min(len(text), position + 160))
    return re.sub(r"\s+", " ", text[start:end]).strip()


CONVERSION_PATTERNS = {
    "as_f32": re.compile(r"\bas\s+f32\b"),
    "as_vec": re.compile(r"\.\s*as_vec[234]\s*\(\s*\)"),
    "f64_helper": re.compile(
        r"\bfn\s+[A-Za-z_][A-Za-z0-9_]*\s*(?:<[^>{}]*>)?\s*\([^)]*(?:f64|DVec[234]|DMat[234])[^)]*\)\s*"
        r"(?:->\s*(?:f32|Vec[234]|\[\s*f32|Vec\s*<\s*f32))",
        re.S,
    ),
}


def _conversion_inventory_text(rel: str, raw: str):
    text = _remove_cfg_test_modules(raw)
    spans = _function_spans(text)
    matches = []
    for operation, pattern in CONVERSION_PATTERNS.items():
        matches.extend((match.start(), operation) for match in pattern.finditer(text))
    counters = {}
    sites = []
    for position, operation in sorted(matches):
        function = _function_name(spans, position)
        key = (function, operation)
        counters[key] = counters.get(key, 0) + 1
        sites.append(
            (
                rel,
                function,
                operation,
                counters[key],
                _statement(text, position),
            )
        )
    return sites


def _complete_conversion_inventory():
    sites = []
    for path in sorted((ROOT / "src").rglob("*.rs")):
        rel = path.relative_to(ROOT).as_posix()
        sites.extend(_conversion_inventory_text(rel, path.read_text(encoding="utf-8")))
    return sites


def conversion_inventory():
    """Reviewed narrowing inventory, excluding the exact lossless DD split."""
    return [site for site in _complete_conversion_inventory() if site not in SANCTIONED_DD_SPLITS]


def _inventory_digest(sites) -> str:
    payload = "\n".join("\t".join(map(str, site)) for site in sites)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _function_body(rel: str, function: str) -> str:
    text = _remove_cfg_test_modules(_read(rel))
    for start, end, name in _function_spans(text):
        if name == function:
            return text[start:end]
    raise AssertionError(f"missing function {rel}::{function}")


def test_exact_production_conversion_inventory_is_frozen():
    sites = conversion_inventory()
    digest = _inventory_digest(sites)
    assert (len(sites), digest) == (
        EXPECTED_CONVERSION_COUNT,
        EXPECTED_CONVERSION_SHA256,
    ), f"conversion inventory changed: count={len(sites)} sha256={digest}\n" + "\n".join(
        repr(site) for site in sites
    )


def test_all_required_rejecting_probes_change_the_inventory():
    probes = [
        "v as f32",
        "coords[0] as f32",
        "position.as_vec3()",
        "coords.map(|v| v as f32)",
        "point.to_array().map(|v| v as f32)",
        "fn narrow(v: f64) -> f32 { v as f32 }",
        "macro_rules! narrow { ($v:expr) => { $v as f32 } }",
        "fn bad(v: f64, origin: f64) -> f32 { v as f32 - origin as f32 }",
    ]
    for probe in probes:
        assert _conversion_inventory_text("probe.rs", probe), f"scanner missed {probe}"


def test_dd_encode_has_exactly_the_two_reviewed_split_casts():
    actual = {
        site
        for site in _complete_conversion_inventory()
        if site[0] == "src/core/dd.rs" and site[1] == "from_f64"
    }
    assert actual == SANCTIONED_DD_SPLITS


def _removed_later(site) -> bool:
    return any(site in transition["removed_sites"] for transition in LATER_REVIEWED_TRANSITIONS)


def _added_later(site) -> bool:
    return any(site in transition["added_sites"] for transition in LATER_REVIEWED_TRANSITIONS)


def _assert_baseline_site_live(sites, site):
    """A site recorded against the baseline is present unless a later
    reviewed transition explicitly removed it."""
    assert site in sites or _removed_later(site), f"unreviewed disappearance: {site}"


def _assert_baseline_site_gone(sites, site):
    """A site removed before the baseline stays absent unless a later reviewed
    transition explicitly re-added it."""
    assert site not in sites or _added_later(site), f"unreviewed reappearance: {site}"


def test_reviewed_checked_reader_inventory_transition_is_exact():
    sites = conversion_inventory()
    transition = REVIEWED_INVENTORY_TRANSITION
    assert transition["current_count"] == REVIEWED_BASELINE_COUNT
    assert len(sites) == EXPECTED_CONVERSION_COUNT
    assert _inventory_digest(sites) == EXPECTED_CONVERSION_SHA256
    _assert_baseline_site_live(sites, transition["added"])
    _assert_baseline_site_gone(sites, transition["removed"])


def test_reviewed_anamnesis_function_ownership_transition_is_exact():
    sites = conversion_inventory()
    transition = REVIEWED_ANAMNESIS_INVENTORY_TRANSITION
    assert transition["base_count"] == REVIEWED_BASELINE_COUNT
    assert transition["result_digest"] == REVIEWED_BASELINE_SHA256
    assert _inventory_digest(sites) == EXPECTED_CONVERSION_SHA256
    for ordinal, statement in enumerate(transition["statements"], start=1):
        removed = (
            transition["path"],
            transition["removed_function"],
            "as_f32",
            ordinal,
            statement,
        )
        added = (
            transition["path"],
            transition["added_function"],
            "as_f32",
            ordinal,
            statement,
        )
        _assert_baseline_site_gone(sites, removed)
        _assert_baseline_site_live(sites, added)


def test_reviewed_helios_inventory_transition_is_exact():
    sites = conversion_inventory()
    transition = REVIEWED_HELIOS_INVENTORY_TRANSITION
    assert transition["base_count"] == REVIEWED_BASELINE_COUNT
    assert transition["result_digest"] == REVIEWED_BASELINE_SHA256
    assert _inventory_digest(sites) == EXPECTED_CONVERSION_SHA256
    for added in transition["added_sites"]:
        _assert_baseline_site_live(sites, added)
    for removed in transition["removed_sites"]:
        _assert_baseline_site_gone(sites, removed)


def test_later_reviewed_transitions_chain_exactly_to_the_current_freeze():
    previous_count = REVIEWED_BASELINE_COUNT
    previous_digest = REVIEWED_BASELINE_SHA256
    for transition in LATER_REVIEWED_TRANSITIONS:
        added = transition["added_sites"]
        removed = transition["removed_sites"]
        assert transition["base_count"] == previous_count
        assert transition["base_digest"] == previous_digest
        assert len(set(added)) == len(added) and len(set(removed)) == len(removed)
        assert not set(added) & set(removed)
        assert previous_count + len(added) - len(removed) == transition["result_count"]
        previous_count = transition["result_count"]
        previous_digest = transition["result_digest"]
    assert (previous_count, previous_digest) == (
        EXPECTED_CONVERSION_COUNT,
        EXPECTED_CONVERSION_SHA256,
    )


def test_last_reviewed_transition_sites_match_the_current_tree():
    sites = set(conversion_inventory())
    last = LATER_REVIEWED_TRANSITIONS[-1]
    for site in last["added_sites"]:
        assert site in sites, f"reviewed addition missing: {site}"
    for site in last["removed_sites"]:
        assert site not in sites, f"reviewed removal still present: {site}"


def test_anchor_narrow_is_the_only_world_conversion_implementation():
    anchor = _remove_cfg_test_modules(_read(SANCTIONED))
    narrow = _function_body(SANCTIONED, "narrow")
    assert len(re.findall(r"\bas\s+f32\b", narrow)) == 1
    assert "value as f32" in re.sub(r"\s+", " ", narrow)
    assert anchor.count("Self::narrow(") == 6

    position = _function_body(SANCTIONED, "to_render_vec3")
    assert "p - self.origin" in re.sub(r"\s+", " ", position)
    assert position.count("Self::narrow(") == 3
    direction = _function_body(SANCTIONED, "direction_to_render")
    assert direction.count("Self::narrow(") == 3


def test_anchor_dd_split_is_a_named_non_narrowing_crossing():
    body = re.sub(r"\s+", " ", _function_body(SANCTIONED, "to_dd"))
    assert "DDVec3::from_dvec3(p)" in body
    assert "Self::narrow(" not in body
    assert " as f32" not in body


def test_each_viewer_world_route_calls_its_active_anchor_in_the_same_function():
    routes = {
        ("src/viewer/viewer_types.rs", "view"): "self.anchor.view_look_at(",
        ("src/viewer/viewer_types.rs", "render_eye"): "self.anchor.to_render_vec3(",
        ("src/viewer/render/main_loop/frame_anchor.rs", "anchored_object_model"): "frame.anchor.model_offset(",
        ("src/viewer/pointcloud/state.rs", "packed_point"): "anchor.to_render_vec3(",
        ("src/viewer/terrain/vector_overlay.rs", "repack_source_vertices"): "anchor.to_render_vec3(",
        ("src/labels/mod.rs", "update_with_camera_anchored"): "anchor.to_render_vec3(",
        ("src/viewer/terrain/render/screen/setup.rs", "build_screen_render_state"): "frame.anchor.to_render_vec3(",
        ("src/viewer/terrain/render/offscreen/setup.rs", "build_snapshot_render_state"): "frame.anchor.to_render_vec3(",
        ("src/viewer/input/viewer_input.rs", "pick_at_screen"): ".to_world_from_render_f64(",
    }
    for (rel, function), required_call in routes.items():
        body = re.sub(r"\s+", "", _function_body(rel, function))
        normalized_call = re.sub(r"\s+", "", required_call)
        assert normalized_call in body, f"{rel}::{function} lacks {required_call}"


def test_cityjson_and_viewer_absolute_storage_types_are_explicitly_f64():
    assert "pub positions: Vec<f64>" in _read("src/import/cityjson/types.rs")
    assert "pub(crate) object_translation: glam::DVec3" in _read("src/viewer/viewer_struct.rs")
    assert "pub world_pos: DVec3" in _read("src/labels/types.rs")
    assert "pub position: DVec3" in _read("src/viewer/pointcloud/types.rs")


def test_public_camera_helper_preserves_earth_scale_offset():
    import numpy as np
    from forge3d import _forge3d

    local = np.asarray(
        _forge3d.camera_look_at((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    )
    earth = np.asarray(
        _forge3d.camera_look_at(
            (6_378_137.0, 2_000.0, -3_000.0),
            (6_378_147.0, 2_000.0, -3_000.0),
            (0.0, 1.0, 0.0),
        )
    )
    np.testing.assert_allclose(earth, local, rtol=0.0, atol=1e-6)
