# SPLAT-FUSED Limits Remediation: Evidence

Machine: local NVIDIA RTX 3070, Vulkan, Windows 11. Worktree `C:/tmp/splat-fused`,
base `b255a990`. Commands are the plan's "Shared commands".

## Task 1: COPC header bounds (L7)

Step 2 failure, observed before the fix:

```
test pointcloud::copc::header_bounds_tests::fixture_header_bounds_enclose_exactly_the_decoded_points ... FAILED
test pointcloud::copc::header_bounds_tests::las14_bounds_are_read_from_their_specified_offsets ... FAILED
min axis 1: 1207.591275215149 vs 3999983.565
  left: [11.0, 2.0, 0.0]
 right: [11.0, 22.0, 33.0]
test result: FAILED. 0 passed; 2 failed
```

After the fix:

```
UNIT header_bounds_tests
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 1629 filtered out
grep -rn "min_bounds\|max_bounds" src/pointcloud  ->  src/pointcloud/copc.rs, src/pointcloud/copc_decode.rs
GPU test_splat_fusion_occlusion
  splat shadow on terrain        IoU 0.9235
  LiDAR shadow on terrain        IoU 0.9928
  terrain shadow on splat        IoU 0.9730
  terrain shadow on LiDAR        IoU 0.9537
  terrain shadow on splat/LiDAR  IoU 0.9618
test result: ok. 1 passed; 0 failed
```

## Task 2: Python self-shadow bias and `self_bias` AOV (L6)

Step 2 failure, observed before the change:

```
E   TypeError: render_fused() got an unexpected keyword argument 'splat_self_bias_sigmas'
1 failed
```

**Deviation from the plan (test tolerance).** The `self_bias` AOV is the fourth
channel of the `Rgba16Float` emission AOV (shared with the base kernel), so the
plan's `rtol=1e-6` on the LiDAR values cannot hold: measured `0.219970703125`
for `0.22` and `0.65966796875` for `0.66`. The second value shows the driver
rounds f32 to f16 toward zero, which WGSL permits, so even the
nearest-f16 expectation is not portable. The LiDAR checks use one half-float
ulp, `rtol = 2**-10` (still separates 1x from 3x the radius by 200%). The splat
ratio check keeps `rtol=1e-5`: scaling by 0.25 is exact in f16 under either
rounding. The terrain check (`== 0.0`) is unchanged.

After the change:

```
PY tests/test_splat_api.py tests/test_api_contracts.py
184 passed in 9.22s   (FORGE3D_SPLAT_FUSION_REQUIRED_GPU=1, no skips)

inspect.signature(splat.render_fused):
(*, splats=None, pointcloud=None, terrain=None, camera, samples=64, samples_per_frame=None,
 width=512, height=512, seed=0, exposure=1.0, sun_azimuth_deg=135.0, sun_elevation_deg=45.0,
 sun_intensity=2.5, sun_color=(1.0, 0.97, 0.92), sky_turbidity=2.5, sky_ground_albedo=0.2,
 sky_intensity=0.35, kappa=4.0, lidar_radius=0.2, lidar_opacity=1.0, ibl_occlusion_distance=12.0,
 sun_angular_radius_deg=0.2665, splat_self_bias_sigmas=2.0, lidar_self_bias_radii=1.0,
 brdf='lambert', roughness=0.6, metallic=0.0, fog_density=0.0, fog_height_falloff=0.0,
 page_capacity=4096, splat_page_size=None, splat_slots=96, point_slots=96, policy='exact',
 return_aovs=False, certificate=False, cache=None)
```

`splat.pyi` has the two parameters after `sun_angular_radius_deg` in both
`render_fused` overloads and `self_bias: np.ndarray` after
`reservoir_visibility`, matching `splat.py`.

## Task 3: `GaussianSplatCloud.transformed` (L9)

Step 2 failures, observed before the change:

```
error[E0599]: no method named `transformed` found for struct `splat::GaussianSplatCloud`
E   AttributeError: 'forge3d._forge3d.GaussianSplatCloud' object has no attribute 'transformed'
```

After the change:

```
UNIT transformed_copies
test result: ok. 1 passed; 0 failed   (splat::tests::transformed_copies_and_leaves_the_source_untouched)
PY tests/test_splat_api.py tests/test_api_contracts.py
185 passed in 8.80s   (includes test_transformed_applies_a_similarity_without_mutating_the_source, CPU-only, not skipped)
```

## Task 6: conservative f16 min-max pyramid (L2)

Step 3 failure, observed before the change:

```
error[E0425]: cannot find function `f16_floor` in this scope
error[E0425]: cannot find function `f16_ceil` in this scope
```

**Deviation from the plan (test arithmetic).** The plan's adjacency check
`hi.to_bits().wrapping_sub(lo.to_bits()) & 0x7fff` is wrong for negative
values: f16 bits are sign-magnitude, so a negative value's floor has the larger
bit pattern and the subtraction wraps to `0x7fff` for adjacent floats. Observed:
`-8331.562: not adjacent` with floor −8336 and ceil −8328, which are adjacent
half floats. The test now maps bits onto a signed total order before
subtracting; the threshold (≤ 1 step, or straddling zero) is unchanged.

**Implementation note (f32 gate).** `TerrainAlbedoMap<'a>` lives in its own
module `src/path_tracing/hybrid_compute/terrain_albedo.rs`. Inside
`terrain_heightfield.rs`, its lifetime quotes made the f32 gate's
comment/string stripper (which treats `'…'` as a char literal) blank out an
unbalanced `{`, so the file's `#[cfg(test)] mod tests` block was no longer
removed and 8 pre-existing test-only casts entered the inventory (1828 → 1836).
With the enum moved and used there as `TerrainAlbedoMap` (elided lifetime), the
inventory is unchanged at 1828 and `tests/test_world_coord_f32_gate.py` passes
unedited.

After the change:

```
UNIT f16_rounding
test result: ok. 1 passed; 0 failed
GPU test_splat_fusion_occlusion (full file, 8 tests)
test fused_fixture_matches_golden_image ... fused golden drift: SSIM 1.000000, mean abs 0.0000   (no regeneration)
test terrain_f16_minmax_changes_no_output_bit ... ok   (rgba, hit_kind, radiance, depth, transmittance, normal bit-identical)
test terrain_f16_minmax_halves_the_pyramid_bytes ... 1250x2500 DEM: min-max pyramid f32 89478488 B, f16 44739244 B; terrain total f32 101978520 B, f16 57239276 B
  splat shadow on terrain        IoU 0.9235
  LiDAR shadow on terrain        IoU 0.9928
  terrain shadow on splat        IoU 0.9730
  terrain shadow on LiDAR        IoU 0.9537
  terrain shadow on splat/LiDAR  IoU 0.9618
test result: ok. 8 passed; 0 failed
PY tests/test_hybrid_terrain_pt.py tests/test_world_coord_f32_gate.py tests/test_splat_api.py
64 passed in 41.76s
```

Lauterbrunnen-shaped DEM: the pyramid drops from 89.5 MB to 44.7 MB (exactly
half), and terrain total from 102.0 MB to 57.2 MB.

## Task 5: terrain colour map, 8-bit sRGB (L3)

Step 3 failures, observed before the change:

```
error[E0609]: no field `albedo_map` on type `&mut FusedTerrainDesc`
error[E0609]: no field `albedo_sampling` on type `&mut FusedTerrainDesc`
E   TypeError: FusedTerrain.__init__() got an unexpected keyword argument 'albedo_map'
```

After the change:

```
UNIT srgb8_albedo   (terrain_heightfield::tests::srgb8_albedo_map_costs_four_bytes_per_cell)
test result: ok. 1 passed; 0 failed     -> albedo_bytes == 4 * 37 * 23; a short map is rejected
GPU terrain_albedo_map
test terrain_albedo_map_drives_the_albedo_aov ... albedo map: 4244 mapped and 10352 masked terrain pixels within +-1 sRGB code
GPU test_splat_fusion_occlusion (full file)
test fused_fixture_matches_golden_image ... fused golden drift: SSIM 1.000000, mean abs 0.0000   (no regeneration)
  five IoU cases unchanged: 0.9235 / 0.9928 / 0.9730 / 0.9537 / 0.9618
test result: ok. 9 passed; 0 failed
PY tests/test_splat_api.py tests/test_hybrid_terrain_pt.py tests/test_world_coord_f32_gate.py tests/test_api_contracts.py
230 passed in 45.55s   (no skips; includes test_terrain_albedo_map_is_validated and
                        test_terrain_albedo_map_sets_the_terrain_albedo)
```

The Python validation test also covers a 2-channel map and an unknown
`albedo_sampling` (both `ValueError`), in addition to the plan's non-uint8
(`TypeError`) and row-count mismatch (`ValueError`) cases.

## Task 4: pack small COPC nodes into shared pages (L10) [changes default image]

Step 2 failure, observed before the change:

```
3000 points in 65 non-empty nodes -> 65 pages (floor 6)
panicked: pages 65, floor 6
```

(The test counts non-empty nodes; empty nodes never become pages.)

After the change:

```
UNIT splat::   (37 passed; 1 ignored = regenerate_committed_fixture)
test splat::stream::tests::small_copc_nodes_are_packed_without_losing_or_duplicating_points ...
  3000 points in 65 non-empty nodes -> 7 pages (floor 6)   ok
GPU test_splat_fusion_occlusion (full file)
test fused_fixture_matches_golden_image ... fused golden drift: SSIM 0.999965, mean abs 0.0132
  -> above the 0.995 / 2.0 gate: NO golden regeneration was needed for this task.
test fused_transmittance_matches_brute_force_cpu_oracle ... ok (existing threshold)
  splat shadow on terrain        IoU 0.9235
  LiDAR shadow on terrain        IoU 0.9928
  terrain shadow on splat        IoU 0.9730
  terrain shadow on LiDAR        IoU 0.9537
  terrain shadow on splat/LiDAR  IoU 0.9618
test result: ok. 9 passed; 0 failed
```

Lauterbrunnen demo (DoD 3). `process_lidar.py` was changed from `depth=3` to
`max_points_per_node=2600` and rerun at 3.3 m (originals backed up to
`D:/forge3d_data/splat_fused_demos/lauterbrunnen/backup_pre_t4/`):

```
lauterbrunnen 3.3 m: 3007905 points, 12959 nodes, 1228 pages at capacity 2700;
2*ceil(points/2700) = 2230; ok=True        (was 12,959 pages)
```

## Task 7: smooth terrain shading normals (L4) [changes default image]

Step 0: `D:/forge3d/cache/tmp/splat_limits_pre_t7.npy` saved from the post-Task-4 build.

Step 4 failure, observed before the change:

```
error[E0425]: cannot find function `terrain_smooth_normal` in this scope
error[E0425]: cannot find function `terrain_patch_normal` in this scope
```

**Deviation from the plan (continuity test scope).** On the plan's DEM
(white noise in [0, 300) m at 6 m spacing, slopes up to 50:1) the normative
fallback ("a smooth normal facing away from the patch falls back to the patch
normal") fires on 2187 of 5952 edge samples (37 %), and there the shading
normal is the discontinuous patch normal by design. The plan's
"≤ 1e-6 on every edge" therefore contradicts its own WGSL. The test keeps
the plan's DEM, threshold and sensitivity check. It asserts continuity on
every edge sample where neither side falls back (3765 samples), and bounds
the fallback share to at most half of the samples on this worst case. The
WGSL and CPU mirror are exactly the plan's normative code.

```
UNIT smooth_terrain fused_kernel_tests   (8 passed)
smooth_terrain_normal_is_continuous_across_cell_edges:
  max normal jump across cell edges: smooth 5.699694e-7 over 3765 of 5952 edge samples
  (2187 use the back-facing fallback), patch 1.9990692e0
smooth_terrain_normal_tracks_the_analytic_surface:
  mean angular error vs analytic normal: smooth 0.4828 deg, patch 1.4207 deg, ratio 0.340   (gate <= 0.7)
fused_kernel_overrides_the_terrain_shading_normal ... ok
default_hybrid_kernel_is_untouched_by_the_fused_specialization ... ok

GPU test_splat_fusion_occlusion (full file, 11 tests)
gpu_smooth_terrain_normal_matches_cpu_mirror ... 17451 terrain pixels, worst GPU-CPU angle 5.9802e-4 rad   (gate 1e-3, 100 %)
smooth_terrain_normals_add_no_self_shadow_acne ... 0 of 17451 terrain pixels newly terrain-shadowed (0.0000%)   (gate <= 0.1 %)
fused_transmittance_matches_brute_force_cpu_oracle ... worst |GPU - CPU| = 4.81e-4 (existing threshold)
  splat shadow on terrain        IoU 0.9235
  LiDAR shadow on terrain        IoU 0.9928
  terrain shadow on splat        IoU 0.9730
  terrain shadow on LiDAR        IoU 0.9537
  terrain shadow on splat/LiDAR  IoU 0.9618
```

Golden regeneration (this task): before regeneration
`fused golden drift: SSIM 0.989848, mean abs 0.6121` (below 0.995). Regenerated
with `FORGE3D_UPDATE_SPLAT_FUSION_GOLDENS=1`, then twice without it:

```
run1: fused golden drift: SSIM 1.000000, mean abs 0.0000 -> ok
run2: fused golden drift: SSIM 1.000000, mean abs 0.0000 -> ok
```

DoD 6 and Python:

```
terrain_smooth_normals=False equals pre-task image: True
default (smooth) differs from pre-task image: True
PY tests/test_splat_fusion_occlusion.py tests/test_splat_api.py tests/test_api_contracts.py tests/test_world_coord_f32_gate.py
209 passed in 35.77s (no skips)
```

`terrain_smooth_normals` is exposed on `render_fused` only. The hard-proxy
reference traces a triangle mesh whose shadows do not depend on shading
normals, so a kwarg on `render_fused_reference` would have no effect.

## Task 8: LiDAR surfels, oriented discs (L5) [changes default image]

Step 0: `D:/forge3d/cache/tmp/splat_limits_pre_t8.npy` saved from the post-Task-7 build.

Step 3 failure, observed before the change:

```
error[E0425]: cannot find function `estimate_point_surfels` / `oct_encode` / `oct_decode` / `sym_eigen3` / `disc_coverage`
error[E0425]: cannot find value `SURFEL_SPHERE`
```

**Deviations from the plan.**

1. *Octahedral sentinel.* The plan's `oct_encode` maps `(0, 0, -1)`, and any
   direction within one quantisation step of it, to `0xFFFF_FFFF ==
   SURFEL_SPHERE`. All four octahedron corners encode the −z pole, so the
   encoder substitutes the (−1, −1) corner (`0`) for the sentinel. The plan's
   own test, which checks `(0, 0, ±1)`, requires this.
2. *Round-trip measurement.* `acos(a·b)` of an f32-normalised decode reports
   ~0.03° for identical directions. The test measures `atan2(|a×b|, a·b)`;
   the 0.02° threshold is unchanged (measured 0.00355°).
3. *Dense-plane sensitivity.* With the default one-radius self-bias, the
   exactly flat plane does not self-shadow as spheres either (T_lidar 1.0000 at
   10° and 22°; also 1.0 for r = 1.2, 1.5, 2.0 with 0–0.2 m height noise). The
   test therefore runs without bias for both modes. Surfels must hold
   T_lidar ≥ 0.98 at 5°, 10°, 22° and 40°, and the sphere sensitivity check
   runs at 5°. Unbiased spheres measure 0.9665 at 22°, 0.8984 at 10° and
   0.7803 at 5°; the plan's "≤ 0.90 at 22°" does not occur on this plane, and
   10° is too close to the gate to be robust.
4. *Reference.* `render_fused_reference` takes `lidar_surfels` (default
   True). Surfel points get a two-sided 16-gon of the half-coverage radius
   times `sqrt(2π / (16 sin(2π/16)))`, from the same per-page estimate.

```
UNIT surfel disc_coverage   (7 passed)
oct_round_trip_error_is_below_0_02_degrees ... worst error 0.00355 deg
sym_eigen3_decomposes_symmetric_matrices ... ok (1000 SPD matrices, 1e-9)
flat_grid_becomes_horizontal_surfels ... 1156 of 1156 interior points are horizontal surfels
vertical_wall_becomes_vertical_surfels ... 1156 of 1156 interior points are vertical surfels
isotropic_cloud_stays_spheres ... 0 of 2000 points are surfels
estimator_is_deterministic ... ok
disc_coverage_closed_forms ... grazing ray over a coplanar grid: discs T = 1, sphelets T = 0.0017892685
size_of::<PointGpu>() == 32: asserted by fusion::tests (pool sizing unchanged)

GPU test_splat_fusion_occlusion (full file, 13 tests)
dense_lidar_plane_does_not_shadow_itself:
  dense plane, surfels, sun 5 deg:  mean T_lidar 1.0000 over 26570 px
  dense plane, surfels, sun 10 deg: mean T_lidar 1.0000 over 26570 px
  dense plane, surfels, sun 22 deg: mean T_lidar 1.0000 over 26570 px
  dense plane, surfels, sun 40 deg: mean T_lidar 1.0000 over 26570 px
  dense plane, spheres, sun 5 deg:  mean T_lidar 0.7803 over 27960 px
lidar_wall_still_casts_its_shadow ... 1584 of 1584 ground pixels have T_lidar <= 0.05
fused_transmittance_matches_brute_force_cpu_oracle ... worst |GPU - CPU| = 4.81e-4 (existing 4e-3 threshold);
  oracle now evaluates disc_coverage for the 2359 of 5500 fixture returns that are surfels
fused_fixture_matches_golden_image ... SSIM 0.998936, mean abs 0.0931  -> NO golden regeneration needed
  splat shadow on terrain        IoU 0.9235
  LiDAR shadow on terrain        IoU 0.9928
  terrain shadow on splat        IoU 0.9730
  terrain shadow on LiDAR        IoU 0.9701   (was 0.9537)
  terrain shadow on splat/LiDAR  IoU 0.9713   (was 0.9618)
test result: ok. 13 passed; 0 failed

DoD 7
lidar_surfels=False equals pre-task image: True
LiDAR pixels 400 ; transmittance AOV differs on LiDAR pixels: True
PY tests/test_splat_fusion_occlusion.py tests/test_splat_api.py tests/test_api_contracts.py tests/test_world_coord_f32_gate.py
209 passed in 37.17s (no skips)
```

**Finding on the real data (L5 root cause).** On the Lauterbrunnen cloud
(3.3 m, `trees_buildings.copc.laz`: vegetation and buildings, no ground
returns), at sun 22°, `lidar_radius=2.3`, 640x360:

```
surfels=False sun 22.0: 105738 LiDAR px, mean T_lidar 0.5328, mean T_total 0.2971
surfels=True  sun 22.0: 104840 LiDAR px, mean T_lidar 0.5315, mean T_total 0.2973
```

Surfels do not change the demo's LiDAR darkness. That darkness comes from
canopy volume (trees occluding each other), not the planar sphere artefact
the plan diagnosed. Surfels fix planar returns (ground, roofs, walls), as the
synthetic tests show. L5 as reported on this scene stays open and needs a
different model (e.g. a canopy-density or per-point opacity calibration).
This needs an owner decision.

## Task 9: seamless tiles and AOV opt-out (L1) [changes default image]

Owner decision A (always seamless) is implemented: every fused render runs
with `camera_flags = 1`. The fused kernel's main pass takes the reservoir's
sun-disc sample in seamless mode (anchored edit "seamless reservoir sun
direction"; the default kernel is unchanged).

Step 4 failure, observed before the change:

```
error[E0425]: cannot find function `default_tile` in this scope
error[E0425]: cannot find function `fused_tiles` in this scope
```

(The kernel-source test was compiled into the same, failing lib test binary.)

After the change:

```
UNIT tiling_tests fused_kernel_tests   (9 passed)
fused_tiles_cover_every_pixel_exactly_once ... ok   (300x200/128x96, 1920x1080/1024, 7x5/7x5)
default_tile_is_single_up_to_one_megapixel ... ok
fused_kernel_uses_the_reservoir_sun_sample_in_seamless_mode ... ok
default_hybrid_kernel_is_untouched_by_the_fused_specialization ... ok

GPU test_splat_fusion_occlusion (full file, 16 tests)
tiled_fused_render_equals_the_monolithic_render_bit_for_bit:
  tile (150, 100): 4 tiles, bit-identical to the monolithic 300x200 render
  tile (128, 96): 9 tiles, bit-identical to the monolithic 300x200 render
  (rgba, hit_kind, radiance, albedo, normal, position, direct, transmittance, depth,
   sun_cosine, self_bias, reservoir_visibility, variance bits, reservoir_valid_count)
fused_peak_memory_does_not_grow_with_output_resolution:
  960x540   tile None          aovs true:  1 tiles, render peak 285728080 B (272.5 MiB)
  1920x1080 tile (960, 540)    aovs true:  4 tiles, render peak 285728080 B (272.5 MiB)
  960x540   tile None          aovs false: 1 tiles, render peak 260844928 B (248.8 MiB)
  3840x2160 tile (960, 540)    aovs false: 16 tiles, render peak 260844928 B (248.8 MiB)
aov_opt_out_leaves_the_beauty_untouched ... ok (rgba identical; AOV vectors empty)
fused_fixture_matches_golden_image ... SSIM 0.997972, mean abs 0.3167 -> NO golden regeneration needed
  splat shadow on terrain        IoU 0.9235
  LiDAR shadow on terrain        IoU 0.9928
  terrain shadow on splat        IoU 0.9730
  terrain shadow on LiDAR        IoU 0.9701
  terrain shadow on splat/LiDAR  IoU 0.9713
test result: ok. 16 passed; 0 failed

PY tests/test_splat_fusion_occlusion.py tests/test_splat_api.py tests/test_api_contracts.py
   tests/test_render_certificate_contract.py tests/test_world_coord_f32_gate.py
1920x1080 fused render: 4 tiles, peak 471.0 MiB      (test_render_fused_1080p_fits_the_budget)
fused golden drift: SSIM 0.997556, mean abs 0.3167
227 passed in 37.78s (no skips; includes test_render_fused_rejects_invalid_tiles)
```

**Deviations and notes.**

- *AOV saving bound.* Opting out frees exactly 48 B/px: 24,883,152 B at
  960x540. The output layout must still bind seven 1x1 placeholder AOV
  targets (48 B), so the plan's `base_lean + 48*960*540 <= base` misses by
  those 48 bytes. The test asserts `48 * (960*540 - 1)`; the per-pixel figure
  is unchanged.
- *Readback ordering (needed for 1080p within budget).* The first
  implementation peaked at 577 MB for 1080p (default 1024x1024 tiles), because
  readback staging coexisted with every tile buffer. Bind groups,
  `reservoir_curr`/`reservoir_out`, then `reservoir_prev`, the moments, the
  accumulation and the beauty texture are now released as each is read back,
  so the tile peak is the render footprint. Measured 1080p peak: 471.0 MiB.
- *Variance and the f32 gate.* Tiles return their raw second-moment maximum.
  `render_fused_view` performs the existing
  `variance = variance.max((f64::from(s.y) / (n * (n - 1.0))) as f32)`,
  so the frozen `as f32` inventory (1828) is unchanged. `FusedViewResources`
  owns its resources, so `render_fused.rs` gains no lifetime syntax (which
  the gate's string stripper misreads as char literals).
- *Task 5 follow-up found during this task.* With an albedo map, the base
  kernel also raises its spectral-ReSTIR flag, whose candidate path the
  fused specialization does not route through `shadow_transmittance`. The
  fused driver now clears that flag (`terrain_uniforms.mips[1] &= !4`) and
  keeps the map. `terrain_albedo_map_drives_the_albedo_aov` still passes.
- With `aovs=false`, `reservoir_valid_count` is 0 and the
  "no valid reservoirs" integrity check is skipped: no reservoirs are read
  back. Python `render_fused(return_aovs=False)` uses this lean path.

## Task 10: Python `render_fused_sequence` (L8)

Step 2 failure, observed before the change:

```
E   AttributeError: module 'forge3d.splat' has no attribute 'FusedView'
```

After the change:

```
PY tests/test_splat_api.py tests/test_api_contracts.py tests/test_render_certificate_contract.py
sequence paging loads (cumulative): [4, 4, 4]
206 passed in 11.63s (no skips)
python -c "import forge3d; forge3d.render_fused_sequence, forge3d.FusedView"
  -> forge3d.splat <class 'forge3d.splat.FusedView'>
```

`test_render_fused_sequence_matches_independent_renders_and_reuses_pages`:
each of the three results equals the independent `render_fused` of that
view (rgba and radiance bits); the repeated third view adds zero page loads;
the sequence peak is <= 512 MiB; and an unknown view key raises
`ValueError("unknown view key ...")`. The native function builds the scene
and the fused kernel once and calls `HybridPathTracer::render_fused_sequence`.
It accepts `certificate=`/`cache=` (swept by the certificate contract) and
shares its parameter assembly with `render_fused` (`fusion_params`).

## Task 11: final gate, docs, demo evidence

Docs: `docs/splat-fused.md` gains "Options" (tile, albedo_map/albedo_sampling,
terrain_smooth_normals, lidar_surfels, self-bias kwargs,
`GaussianSplatCloud.transformed`, `render_fused_sequence`) and "Measured".
"Limits" is reduced to the plan's out-of-scope list (GPU SH degrees 2-3,
pow2 min-max padding, 2.5D cliffs, baked capture lighting), plus the
storage-buffer requirement and the open L5 finding. CHANGELOG has one entry
per fixed limit.

Step 2, full gate (all exit 0):

```
cargo fmt --all -- --check                                   exit=0
cargo forge3d-clippy                                         exit=0 (0 warnings)
cargo forge3d-clippy-acceptance                              exit=0 (0 warnings)
cargo test --release --workspace --features $F -- --test-threads=1 --skip gpu_extrusion --skip brdf_tile
  12 result blocks: passed 1693, failed 0, ignored 9          exit=0
FORGE3D_SPLAT_FUSION_REQUIRED_GPU=1 cargo test --release --test test_splat_fusion_occlusion
  splat shadow on terrain        IoU 0.9235
  LiDAR shadow on terrain        IoU 0.9928
  terrain shadow on splat        IoU 0.9730
  terrain shadow on LiDAR        IoU 0.9701
  terrain shadow on splat/LiDAR  IoU 0.9713
  logical primitives: 1007625000 | peak resident pages: 61 of 246010
  memory: render peak total 131862480 B (125.8 MiB), host-visible 8.8 MiB, pool 100.5 MiB, limit 536870912 B
  fused golden drift: SSIM 0.997972, mean abs 0.3167
  test result: ok. 16 passed; 0 failed                        exit=0
BUILD_PYD                                                     build=0
PY tests/test_splat_fusion_occlusion.py tests/test_splat_api.py tests/test_api_contracts.py
   tests/test_render_certificate_contract.py tests/test_no_silent_degradation.py
   tests/test_world_coord_f32_gate.py tests/test_hybrid_terrain_pt.py
  266 passed in 56.02s
  scripts/assert_junit_zero_skips.py -> required lane clean: {'tests': 266, 'failures': 0, 'errors': 0, 'skipped': 0}
```

Step 3, demo evidence (`D:/forge3d/cache/tmp/splat_limits_demo_evidence.py`,
frames and `evidence.json` in `D:/forge3d_data/splat_fused_demos/limits_evidence/`):

```
mt_st_helens  1920x1080, 64 spp, default tiling: 4 tiles, 9.7 s, peak 496271068 B (473.3 MiB),
              terrain 4916308 B (f16 pyramid), frames 16, restarts 1
lauterbrunnen 1920x1080, 64 spp, tile 960x540, terrain_6m.npy + procedural colour map
              (meadow < 30 deg, rock >= 30 deg, snow > 2600 m; caption "procedural (slope/elevation),
              not imagery"), lidar_surfels=True, terrain_smooth_normals=True:
              4 tiles, 39.6 s, peak 489793428 B (467.1 MiB), terrain 70263532 B (67.0 MiB, f16 pyramid),
              1232 pages, 1030 resident at peak, restarts 19
```

Lauterbrunnen needs `point_slots=1240` with the 3.3 m cloud (3.0M points,
1,228 point pages). The demo script's 870 slots were sized for the earlier
4 m cloud (1.96M points), and the exact policy reports a budget error at
870. With 1,240 slots the 1024x1024 default tile would exceed 512 MiB, so
the frame uses `tile=(960, 540)`.

### Golden regenerations

| Task | Before | Action |
|---|---|---|
| T4 | SSIM 0.999965 | none needed |
| T7 | SSIM 0.989848, mean abs 0.6121 | regenerated; 1.000000 twice after |
| T8 | SSIM 0.998936 | none needed |
| T9 | SSIM 0.997972 (Rust), 0.997556 (Python) | none needed |

### Files outside the plan's list

`src/path_tracing/hybrid_compute/terrain_albedo.rs` (new). It holds
`TerrainAlbedoMap<'a>`, kept out of `terrain_heightfield.rs` for the f32
gate (see Task 6).

## Review follow-up (2026-10-03)

Changes after the owner's review of the uncommitted work:

1. **Finding 1, terrain copied per view.** `FusedRenderDesc.terrain` is now
   `Option<Arc<FusedTerrainDesc>>`, and the binding builds one `Arc` per
   scene, so sequence views share it. `render_fused_sequence` builds the
   `TerrainPtScene` (heights, f16 min-max pyramid, colour map) once. Each
   view swaps in its sky through the new `TerrainPtScene::set_environment`;
   the environment upload is factored into `upload_environment`. The
   same-terrain check compares every field (`FusedTerrainDesc: PartialEq`,
   `Arc::ptr_eq` fast path), because the scene is built from the first view.
   New GPU test `sequence_shares_one_terrain_and_swaps_the_sky_per_view`:
   a 3-view sun/sky sweep sharing one `Arc` equals the standalone renders
   bit for bit (rgba, radiance), and a view whose terrain differs only in
   its colour map is rejected. GPU suite: 17 passed, golden SSIM 0.997972,
   IoUs unchanged.
   Lauterbrunnen terrain-only, 6 views at 480x270, 4 spp: sequence 1.1 s vs
   1.4 s for 6 standalone renders, images identical.
2. **Finding 2, L5 wording.** Reproduced on the Mount St. Helens crater
   (ground returns, default self-bias, `lidar_radius=2.6`, 960x540,
   16 spp), surfels off -> on:

   ```
   demo camera, compass az 160:  sun 22: T_lidar 0.773 -> 0.805, shadowed 23% -> 21%, LiDAR radiance +8%
                                 sun 10: T_lidar 0.934 -> 0.953, shadowed  7% ->  5%, LiDAR radiance +2%
   video camera, compass az 130: sun 22: T_lidar 0.782 -> 0.822, shadowed 23% -> 19%, LiDAR radiance +25%
                                 sun 10: T_lidar 0.929 -> 0.947, shadowed  7% ->  6%, LiDAR radiance +17%
   video camera, compass az 160: sun 22: T_lidar 0.788 -> 0.819, shadowed 22% -> 20%, LiDAR radiance +15%
                                 sun 10: T_lidar 0.912 -> 0.937, shadowed  9% ->  7%, LiDAR radiance +13%
   video camera, compass az 190: sun 22: T_lidar 0.705 -> 0.753, shadowed 31% -> 27%, LiDAR radiance +15%
                                 sun 10: T_lidar 0.895 -> 0.917, shadowed 11% ->  9%, LiDAR radiance +11%
   ```

   Every configuration improves, but by less than the owner's reported
   0.686 -> 0.832 at 22 degrees (settings for that run not recorded here).
   The docs and CHANGELOG now say surfels fix ground/roof LiDAR and that
   canopy darkness remains.
3. **Finding 4.** One line in `docs/splat-fused.md`: surfel normals are
   per page and can change with `page_capacity`.
4. **Finding 5.** The double-encoded em dash in the "no valid reservoirs"
   error is now ASCII. A scan of every changed text file finds no other
   mojibake.

Findings 3 (per-tile `begin_view` to shrink the pool further) and 6 (stats
dropped without AOVs) are not changed.
