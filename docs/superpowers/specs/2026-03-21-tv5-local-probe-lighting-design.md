# TV5 — Local Probe Lighting for Terrain Scenes

**Date:** 2026-03-21
**Epic:** TV5 from `docs/plans/2026-03-16-terrain-viz-epics.md`
**Scope:** TV5.1 (diffuse irradiance probes), TV5.2 (placement, invalidation, memory), TV5.3 (reflection probes)
**Bake strategy:** Heightfield-analytical for TV5.1-5.2; architecture shaped for hybrid (SceneCapture) upgrade in TV5.3

---

## 1. Problem

The terrain renderer currently uses a single global IBL cubemap for all indirect diffuse and specular lighting. This means a valley floor receives the same irradiance as an exposed ridge, and an area beneath an overpass is lit identically to open terrain. The `GiMode::IrradianceProbes` enum variant exists in `src/render/params/gi.rs` (line 11) but has no runtime implementation.

**Value proposition:** Terrain-native diffuse GI that makes valleys darker and ridges brighter, using the heightfield itself as the occlusion source.

---

## 2. Data Model

### 2.1 Topology

```rust
/// Grid descriptor — where probes live, shared by diffuse and future reflection probes.
pub struct ProbeGridDesc {
    pub origin: [f32; 2],          // World XY of grid minimum corner
    pub spacing: [f32; 2],         // World-space distance between adjacent probes
    pub dims: [u32; 2],            // (cols, rows) — grid dimensions
    pub height_offset: f32,        // Meters above terrain surface per probe
    pub influence_radius: f32,     // World-space radius for per-probe falloff (unused in v1 bilinear; reserved)
}
```

### 2.2 Placement

```rust
/// Resolved world-space positions — derived from grid + terrain, reusable across probe types.
pub struct ProbePlacement {
    pub grid: ProbeGridDesc,
    pub positions_ws: Vec<[f32; 3]>,  // len == dims[0] * dims[1], invariant enforced at construction
}

impl ProbePlacement {
    pub fn new(grid: ProbeGridDesc, positions_ws: Vec<[f32; 3]>) -> Self {
        assert_eq!(positions_ws.len(), (grid.dims[0] * grid.dims[1]) as usize,
            "positions_ws.len() must equal dims[0] * dims[1]");
        Self { grid, positions_ws }
    }
}
```

Placement is computed once from grid topology + heightfield, then reused by both the diffuse baker and any future reflection-probe baker.

### 2.3 Irradiance payload

```rust
/// Canonical CPU-side SH L2 coefficients: 9 basis functions x RGB.
pub struct SHL2 {
    pub coeffs: [[f32; 3]; 9],
}

/// Baked diffuse irradiance — one SHL2 per placed probe.
pub struct ProbeIrradianceSet {
    pub probes: Vec<SHL2>,  // len == placement.positions_ws.len()
}
```

### 2.4 Baker trait

```rust
/// Abstract bake backend — topology + placement in, irradiance out.
/// Returns Result to support fallible backends (e.g. future SceneCaptureBaker).
pub trait ProbeBaker {
    fn bake(&self, placement: &ProbePlacement) -> Result<ProbeIrradianceSet, ProbeError>;
}
```

The trait takes `&ProbePlacement` (resolved positions), not raw heightfield storage. Each baker implementation carries its own scene context:

```rust
/// TV5.1 baker — heightfield-analytical, carries its own scene data.
pub struct HeightfieldAnalyticalBaker {
    pub heightfield: Vec<f32>,
    pub height_dims: (u32, u32),
    pub terrain_span: [f32; 2],   // World-space extent of the heightfield
    pub sky_color: [f32; 3],
    pub sky_intensity: f32,
    pub ray_count: u32,           // Hemisphere samples per probe (default: 64)
    pub max_trace_distance: f32,  // Max ray-march distance in heightfield units
}
```

A future `SceneCaptureBaker` would hold GPU device + renderer references instead of heightfield data, render cubemaps from each probe position, then project to SH — producing the same `ProbeIrradianceSet` output.

---

## 3. GPU Layout

### 3.1 Bind group strategy: extend @group(6)

The terrain pipeline is hardcoded to groups 0–6 in `pipeline_cache.rs` (line 137), and all draw paths explicitly bind groups 0–6 in `draw/execute.rs` (line 159) and `aov.rs` (line 275). Group 6 currently holds a single `MaterialLayerUniforms` uniform at binding 0. Adding probe bindings to group 6 avoids adding a new group index to every pipeline layout, draw call, and AOV path.

**Extended @group(6) layout:**
- binding 0: `uniform<MaterialLayerUniforms>` — existing, unchanged
- binding 1: `uniform<ProbeGridUniforms>` — probe grid metadata (48 bytes)
- binding 2: `storage<array<GpuProbeData>, read>` — probe SH data (read-only SSBO)

### 3.2 GPU structs (WGSL)

All structs are explicitly vec4-aligned. No ambiguity between Rust and WGSL packing.

```wgsl
struct ProbeGridUniforms {
    // xy = world origin of grid, z = height_offset, w = enabled (1.0 = on, 0.0 = off)
    grid_origin:   vec4<f32>,
    // x = spacing_x, y = spacing_y, z = f32(dims_x), w = f32(dims_y)
    grid_params:   vec4<f32>,
    // x = fallback_blend_distance (world-space meters), y = f32(probe_count), zw = pad
    blend_params:  vec4<f32>,
};
// 3 vec4s = 48 bytes

struct GpuProbeData {
    sh_r_01: vec4<f32>,   // R: L0_0, L1_-1, L1_0, L1_1
    sh_r_23: vec4<f32>,   // R: L2_-2, L2_-1, L2_0, L2_1
    sh_r_4:  vec4<f32>,   // R: L2_2, pad, pad, pad
    sh_g_01: vec4<f32>,   // G: L0_0, L1_-1, L1_0, L1_1
    sh_g_23: vec4<f32>,   // G: L2_-2, L2_-1, L2_0, L2_1
    sh_g_4:  vec4<f32>,   // G: L2_2, pad, pad, pad
    sh_b_01: vec4<f32>,   // B: L0_0, L1_-1, L1_0, L1_1
    sh_b_23: vec4<f32>,   // B: L2_-2, L2_-1, L2_0, L2_1
    sh_b_4:  vec4<f32>,   // B: L2_2, pad, pad, pad
};
// 9 vec4s = 144 bytes per probe
```

**Design note:** `position_radius` is omitted from `GpuProbeData` because v1 sampling is purely grid/bilinear in XY. Per-probe positions are only needed on the CPU side for placement; the shader reconstructs grid positions from `ProbeGridUniforms`. This avoids carrying unused dynamic-looking fields in the GPU payload.

### 3.3 Rust-side mirror types

```rust
#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ProbeGridUniformsGpu {
    pub grid_origin: [f32; 4],   // xy=origin, z=height_offset, w=enabled
    pub grid_params: [f32; 4],   // x=spacing_x, y=spacing_y, z=dims_x, w=dims_y
    pub blend_params: [f32; 4],  // x=fallback_blend_distance, y=probe_count, zw=pad
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuProbeData {
    pub sh_r_01: [f32; 4],
    pub sh_r_23: [f32; 4],
    pub sh_r_4:  [f32; 4],
    pub sh_g_01: [f32; 4],
    pub sh_g_23: [f32; 4],
    pub sh_g_4:  [f32; 4],
    pub sh_b_01: [f32; 4],
    pub sh_b_23: [f32; 4],
    pub sh_b_4:  [f32; 4],
}
// 9 * 16 = 144 bytes, static_assert via Pod derive
```

`ProbeIrradianceSet::upload()` repacks `SHL2::coeffs` ([[f32; 3]; 9]) into channel-major `GpuProbeData` layout for the SSBO.

### 3.4 Fallback bind group

When probes are disabled, bind a 1-element SSBO with zeroed `GpuProbeData` and `enabled = 0.0` in the uniform. Same pattern as `sky_fallback_texture` and `height_ao_fallback_view` — zero-cost bind group, shader early-outs. No new group index, no pipeline layout change.

---

## 4. Shader Integration

### 4.1 New include: `terrain_probes.wgsl`

Included via `#include "terrain_probes.wgsl"` in `terrain_pbr_pom.wgsl`. The codebase's `preprocess_terrain_shader()` in `pipeline_cache.rs` manually concatenates all shader includes in dependency order (not via a runtime #include system), so that function must be updated to load and concatenate this file. Contains bindings, SH evaluation, and grid sampling.

**Coordinate convention:** Probes are at grid vertices, not cell centers. Probe index `(i, j)` corresponds to world position `origin + spacing * (i, j)`. The grid covers from `origin` to `origin + spacing * (dims - 1)`.

**World-space mapping:** The terrain shader computes `world_xy = (tex_coord - 0.5) * spacing` (line 1037), ranging from `[-spacing/2, +spacing/2]`. The probe grid `origin` must be in the same coordinate space. When auto-placed, `origin` defaults to the terrain bounds minimum corner in this coordinate space (i.e., `(-terrain_span/2, -terrain_span/2)` for a centered terrain).

```wgsl
@group(6) @binding(1)
var<uniform> probe_grid: ProbeGridUniforms;

@group(6) @binding(2)
var<storage, read> probe_data: array<GpuProbeData>;

struct ProbeIrradianceResult {
    irradiance: vec3<f32>,   // SH-evaluated irradiance in normal direction
    weight: f32,             // 0.0 at grid edge / disabled, 1.0 at grid interior
};

fn evaluate_sh_l2(n: vec3<f32>, probe: GpuProbeData) -> vec3<f32> {
    // SH L2 basis functions
    let Y00  = 0.282095;
    let Y1m1 = 0.488603 * n.y;
    let Y10  = 0.488603 * n.z;
    let Y11  = 0.488603 * n.x;
    let Y2m2 = 1.092548 * n.x * n.y;
    let Y2m1 = 1.092548 * n.y * n.z;
    let Y20  = 0.315392 * (3.0 * n.z * n.z - 1.0);
    let Y21  = 1.092548 * n.x * n.z;
    let Y22  = 0.546274 * (n.x * n.x - n.y * n.y);

    let basis_01 = vec4<f32>(Y00, Y1m1, Y10, Y11);
    let basis_23 = vec4<f32>(Y2m2, Y2m1, Y20, Y21);

    var result: vec3<f32>;
    result.r = dot(probe.sh_r_01, basis_01) + dot(probe.sh_r_23, basis_23) + probe.sh_r_4.x * Y22;
    result.g = dot(probe.sh_g_01, basis_01) + dot(probe.sh_g_23, basis_23) + probe.sh_g_4.x * Y22;
    result.b = dot(probe.sh_b_01, basis_01) + dot(probe.sh_b_23, basis_23) + probe.sh_b_4.x * Y22;

    return max(result, vec3(0.0));
}

fn sample_probe_irradiance(world_pos: vec3<f32>, normal: vec3<f32>) -> ProbeIrradianceResult {
    var result: ProbeIrradianceResult;
    result.irradiance = vec3(0.0);
    result.weight = 0.0;

    if (probe_grid.grid_origin.w < 0.5) {
        return result;  // Probes disabled
    }

    let dims = vec2<u32>(u32(probe_grid.grid_params.z), u32(probe_grid.grid_params.w));
    let spacing = probe_grid.grid_params.xy;

    // Map world XY to fractional grid coordinates
    let grid_uv = (world_pos.xy - probe_grid.grid_origin.xy) / spacing;

    // Bilinear: find 4 nearest probes
    let cell = clamp(grid_uv - vec2(0.5), vec2(0.0), vec2<f32>(dims - vec2(1u)));
    let i0 = vec2<u32>(floor(cell));
    let frac = fract(cell);

    let idx00 = i0.y * dims.x + i0.x;
    let idx10 = i0.y * dims.x + min(i0.x + 1u, dims.x - 1u);
    let idx01 = min(i0.y + 1u, dims.y - 1u) * dims.x + i0.x;
    let idx11 = min(i0.y + 1u, dims.y - 1u) * dims.x + min(i0.x + 1u, dims.x - 1u);

    let sh00 = evaluate_sh_l2(normal, probe_data[idx00]);
    let sh10 = evaluate_sh_l2(normal, probe_data[idx10]);
    let sh01 = evaluate_sh_l2(normal, probe_data[idx01]);
    let sh11 = evaluate_sh_l2(normal, probe_data[idx11]);

    let bilinear = mix(mix(sh00, sh10, frac.x), mix(sh01, sh11, frac.x), frac.y);

    // Edge blend: fade to 0 weight at grid boundary (world-space distance).
    // The vertex-domain extent is (0, 0) to (dims - 1, dims - 1) in grid coords.
    // Degenerate axes (dims == 1, extent == 0) get full weight along that axis.
    let fallback_dist = probe_grid.blend_params.x;
    let grid_extent = vec2<f32>(dims - vec2(1u));  // Vertex-domain max

    // Per-axis edge distance: for a degenerate axis (extent == 0), use a large
    // sentinel so it never constrains the min(). For normal axes, compute
    // distance to the nearer edge in grid-cell units.
    let sentinel = fallback_dist + 1.0;  // Always > fallback_dist, so saturate = 1.0
    let dist_x = select(min(grid_uv.x, grid_extent.x - grid_uv.x) * spacing.x,
                         sentinel, grid_extent.x < 0.5);
    let dist_y = select(min(grid_uv.y, grid_extent.y - grid_uv.y) * spacing.y,
                         sentinel, grid_extent.y < 0.5);
    let dist_to_edge = min(dist_x, dist_y);
    let edge_weight = saturate(dist_to_edge / max(fallback_dist, 1e-6));

    result.irradiance = bilinear;
    result.weight = edge_weight;
    return result;
}
```

### 4.2 Integration point in `fs_main`

The probe irradiance replaces the IBL diffuse irradiance term **upstream** of shadow/occlusion application. The mix happens **outside** `eval_ibl_split` — that function is not modified and does not gain a `world_pos` parameter. Instead, the mix happens in `fs_main` at the call site (~line 2960-2981) where `ibl_split.diffuse` is already available.

The current code at lines 2967-2981:
```wgsl
let ibl_split = eval_ibl_split(rotated_normal, rotated_view, ibl_albedo, metallic, roughness, f0);
// ... shadow applied to ibl_split.diffuse ...
let ibl_diffuse_with_shadow = ibl_split.diffuse * shadow_factor;
```

With probes enabled, the diffuse term is blended **before** shadow application:
```wgsl
let ibl_split = eval_ibl_split(rotated_normal, rotated_view, ibl_albedo, metallic, roughness, f0);

// TV5: Blend probe irradiance into diffuse IBL term
let probe_result = sample_probe_irradiance(world_position, N);
// Recompute diffuse with blended irradiance: probe replaces global cubemap irradiance
let kS_ibl = ibl_split.fresnel;
let kD_ibl = (vec3<f32>(1.0) - kS_ibl) * (1.0 - metallic);
let global_diffuse = ibl_split.diffuse;  // kD * base_color * global_irradiance
let probe_diffuse = kD_ibl * ibl_albedo * probe_result.irradiance;
let blended_diffuse = mix(global_diffuse, probe_diffuse, probe_result.weight);

let ibl_diffuse_with_shadow = blended_diffuse * shadow_factor;
```

`world_position` is already available in `fs_main` as a fragment-stage variable (carried from the vertex shader or reconstructed from `tex_coord` depending on the render path). `eval_ibl_split` is unchanged — no signature change, no new parameter.

When probes are disabled (`weight = 0.0`), `blended_diffuse = global_diffuse` — pixel-identical to current behavior. Shadow factor and occlusion continue to apply downstream as before.

The specular path (`prefiltered_color` at line 2220) is **not modified** in TV5.1-5.2. Specular continues to use the global IBL cubemap. TV5.3 would add a separate reflection probe system alongside.

### 4.3 Debug modes

Two new debug mode constants (next available slots after existing modes):

- `DBG_PROBE_IRRADIANCE`: Visualize raw probe irradiance contribution (`probe_result.irradiance * probe_result.weight`) with standard tonemap. Shows what probes are adding.
- `DBG_PROBE_WEIGHT`: Visualize `probe_result.weight` as grayscale. Shows grid coverage and edge blending.

---

## 5. Blending Units

All blending distances are in **world-space meters**, consistent with the terrain shader's world-space coordinate system (`world_xy` at line 1036-1037).

- `ProbeGridDesc.influence_radius`: Reserved for future per-probe dynamic falloff (not used in v1 bilinear sampling). World-space meters.
- `ProbeGridUniforms.blend_params.x` (`fallback_blend_distance`): Distance in world-space meters over which probe weight fades from 1.0 to 0.0 at grid edges. The shader converts grid-cell edge distance to world units by multiplying by `min(spacing.x, spacing.y)` before comparing against this threshold.
- `ProbeSettings.fallback_blend_distance`: Python-facing parameter, same world-space meters. Default: `spacing * 2.0` (auto-computed from grid spacing if not specified).

`influence_radius` and `fallback_blend_distance` are separate concepts. `influence_radius` is per-probe and unused in v1. `fallback_blend_distance` is a grid-level edge-blending control.

---

## 6. HeightfieldAnalytical Baker

CPU-side bake that runs once when probes are configured or invalidated.

### 6.1 Placement

For each grid cell `(i, j)`:
1. Compute world XY: `origin + spacing * (i, j)`
2. Sample heightfield at that XY via bilinear interpolation
3. Set world Z: `terrain_height * z_scale + height_offset`

### 6.2 Hemisphere ray-march

For each probe, cast `ray_count` rays distributed on the upper hemisphere (stratified Fibonacci spiral):

1. For each ray direction `d`:
   - March along the heightfield in discrete steps
   - If `heightfield(ray_xy) > ray_z` at any step, mark occluded
   - If unoccluded: contribute `sky_color * sky_intensity * cos_theta` weighted by SH basis
   - If occluded: contribute zero (v1 — no terrain color bleeding)
2. Accumulate: `sh_coeffs[l][channel] += color * Y_lm(d) * cos_theta * solid_angle`
3. Normalize by total solid angle

This is conceptually identical to what `height_ao` does (ray-march heightfield for occlusion) but accumulated into 9 SH L2 coefficients per channel instead of a scalar AO value.

### 6.3 Performance

256 probes x 64 rays x ~100 march steps = ~1.6M heightfield lookups. CPU-side, this completes in single-digit milliseconds — not a bottleneck.

---

## 7. Python API

### 7.1 ProbeSettings dataclass

```python
@dataclass
class ProbeSettings:
    """Irradiance probe configuration for terrain scenes."""
    enabled: bool = False
    grid_dims: Tuple[int, int] = (8, 8)
    origin: Optional[Tuple[float, float]] = None  # World XY; None = auto (terrain bounds min corner)
    spacing: Optional[Tuple[float, float]] = None  # World-space meters; None = auto from terrain_span / grid_dims
    height_offset: float = 5.0                     # Meters above terrain surface
    ray_count: int = 64                            # Hemisphere samples per probe
    fallback_blend_distance: Optional[float] = None  # World meters; None = auto (spacing * 2.0)
    sky_color: Tuple[float, float, float] = (0.6, 0.75, 1.0)  # Linear RGB sky tint
    sky_intensity: float = 1.0

    def __post_init__(self) -> None:
        if self.enabled:
            cols, rows = self.grid_dims
            if cols < 1 or rows < 1:
                raise ValueError("grid_dims must be >= (1, 1)")
            if cols * rows > 4096:
                raise ValueError("grid_dims product must be <= 4096 (probe count limit)")
```

### 7.2 Integration

Added to `make_terrain_params_config()` as an optional `probes=ProbeSettings(...)` parameter. When `enabled=True`:
1. The renderer resolves placement from grid + heightfield
2. Bakes SH coefficients via `HeightfieldAnalyticalBaker`
3. Uploads to SSBO and binds into group(6)

When `enabled=False` (default): fallback bind group with zeroed data — no behavioral change.

### 7.3 Auto-placement

When `origin` is `None` and `spacing` is `None`, auto-placement is resolved per-axis:

- **Normal axis** (`dims > 1`): `spacing = terrain_span / (dims - 1)`, `origin = terrain_min`. The first and last probes sit exactly on the terrain boundary (since probe index `i` maps to `origin + spacing * i`, covering `terrain_min` to `terrain_min + spacing * (dims - 1)` = `terrain_max`).
- **Degenerate axis** (`dims == 1`): `spacing = terrain_span` (arbitrary, unused for indexing since `i` is always 0), `origin = terrain_min + terrain_span / 2`. This places the single probe at the terrain center along that axis, not at the min corner.

This ensures a `(1, 1)` grid places one probe at the terrain center, and a `(1, N)` grid places a single column centered along the degenerate axis.

---

## 8. Memory Tracking

Two separate mechanisms:

1. **Global resource registry** (`src/core/memory_tracker/registry.rs` line 41): `track_buffer_allocation()` / `free_buffer_allocation()` called when the probe SSBO and uniform buffer are created/destroyed. This contributes to the global aggregate totals (`buffer_count`, `buffer_bytes`) so the system-wide view stays accurate. The registry does **not** provide per-category breakdowns — it only tracks totals.

2. **Probe-local counters**: The probe resource object on `TerrainScene` tracks its own allocation sizes (`grid_uniform_bytes`, `probe_ssbo_bytes`) as plain fields. These are set at upload time and zeroed on teardown. `get_probe_memory_report()` reads from these local counters, not from the global registry.

Note: `check_budget()` in the reporting module is a host-visible budget check. Normal probe SSBO/uniform buffers are device-local (not `MAP_READ`/`MAP_WRITE`), so `check_budget()` is **not** called for probe allocations — it would be a no-op. The global registry's `track_buffer_allocation(size, is_host_visible=false)` correctly records the allocation without inflating the host-visible budget counter.

`get_probe_memory_report()` on `TerrainRenderer` returns:
```python
{
    "probe_count": int,
    "grid_uniform_bytes": int,    # 48
    "probe_ssbo_bytes": int,      # probe_count * 144
    "total_bytes": int,           # sum
}
```

At maximum practical scale (256 probes), probe memory is ~37 KB — negligible against the 512 MiB budget, but tracked for architectural correctness.

---

## 9. Invalidation

Probes rebake when any of these change:
- `ProbeSettings` fields: `grid_dims`, `spacing`, `height_offset`, `origin`
- Heightfield data (different DEM loaded)
- `sky_color` or `sky_intensity` (baker inputs that directly affect irradiance output)

Probes do **not** rebake when:
- Camera moves (probes are view-independent)
- Directional light changes (v1 baker uses sky color, not directional light)
- Shadow settings change
- Material layer settings change
- `fallback_blend_distance` changes (runtime-only shader uniform, no rebake needed)

Invalidation is explicit: any change to an invalidating parameter triggers a full rebake. No incremental updates in TV5.1. The `ProbeBaker::bake()` returns `Result` to support fallible backends — `HeightfieldAnalyticalBaker` is infallible in practice, but `SceneCaptureBaker` can fail on GPU resource allocation or timeout.

The `HeightfieldAnalyticalBaker` handles NaN/nodata values in the heightfield gracefully: during ray-march, heightfield samples that are NaN or infinite are treated as transparent (no occlusion), which is conservative — a nodata region will not erroneously darken probes.

---

## 10. Testing Strategy

### 10.1 Unit tests

- `test_probe_placement_invariant`: `positions_ws.len() == dims[0] * dims[1]` enforced
- `test_probe_bake_deterministic`: Same inputs produce identical SH coefficients
- `test_probe_gpu_layout_size`: `size_of::<GpuProbeData>() == 144`, `size_of::<ProbeGridUniformsGpu>() == 48`
- `test_probe_sh_packing_roundtrip`: `SHL2 -> GpuProbeData -> SHL2` preserves coefficients

### 10.2 Pixel-identity test

- `test_probe_fallback_pixel_identical`: Probes disabled (`enabled=False`, fallback probe bindings active with zeroed data and `weight=0`) produces pixel-identical output to current baseline

### 10.3 Visual regression

- `test_probe_valley_darker`: Valley scene with probes enabled shows measurably lower luminance in valleys vs probes disabled
- `test_probe_ridge_brighter`: Ridge scene with probes shows higher relative luminance on exposed ridges

### 10.4 Memory test

- `test_probe_memory_tracked`: After bake, `get_probe_memory_report()["total_bytes"]` matches expected `48 + probe_count * 144`

### 10.5 Edge blend test

- `test_probe_edge_blend_smooth`: Fragments at grid boundary blend smoothly (no hard seam between probe irradiance and global IBL)

### 10.6 Out-of-bounds and degenerate tests

- `test_probe_out_of_bounds_weight_zero`: World positions far outside the grid return `weight = 0.0` (full IBL fallback)
- `test_probe_single_probe_grid`: `grid_dims = (1, 1)` works correctly (all 4 bilinear samples collapse to index 0)
- `test_probe_grid_dims_validation`: `grid_dims = (0, 0)` raises `ValueError`; `grid_dims > (64, 64)` is allowed up to 4096 total probes
- `test_probe_nodata_heightfield`: Baker handles NaN values in heightfield without producing NaN SH coefficients

### 10.7 Invalidation behavior test

- `test_probe_invalidation_triggers`: Changing `sky_color` or `grid_dims` triggers rebake; changing camera position does not

### 10.8 Example

`examples/terrain_tv5_probe_lighting_demo.py` — renders a real DEM (Gore Range or Rainier) with:
- Offscreen render: probes disabled vs probes enabled side-by-side comparison
- Probe debug mode visualization
- Memory report printed
- PNG output verified non-zero and saved

---

## 11. TV5.3 Upgrade Path (Reflection Probes)

TV5.3 adds local reflection probes **as a separate system** that shares `ProbePlacement` but has its own GPU resources:

- Reflection probes store low-res prefiltered cubemaps (not SH) — specular needs directional data
- A new `ReflectionProbeSet` with cubemap array texture, separate from `ProbeIrradianceSet`
- `SceneCaptureBaker` implements `ProbeBaker` for diffuse (cubemap → SH projection) and a parallel `ReflectionProbeBaker` for specular (cubemap → prefilter)
- Reflection probe bindings would be additional entries in group(6) or, if binding count is a concern, a new group(7) — decided at TV5.3 design time
- The diffuse probe system does not change when reflection probes are added
- TV5.3 DoD requires "at least one terrain + structure scene demonstrates better local specular than global IBL alone" — that test scene will be designed in the TV5.3 spec

---

## 12. File Plan

| New file | Purpose |
|----------|---------|
| `src/terrain/probes/mod.rs` | Module root, re-exports |
| `src/terrain/probes/types.rs` | `ProbeGridDesc`, `ProbePlacement`, `SHL2`, `ProbeIrradianceSet`, `ProbeError` |
| `src/terrain/probes/baker.rs` | `ProbeBaker` trait |
| `src/terrain/probes/heightfield_baker.rs` | `HeightfieldAnalyticalBaker` implementation |
| `src/terrain/probes/gpu.rs` | `GpuProbeData`, `ProbeGridUniformsGpu`, `upload()`, memory tracking |
| `src/shaders/terrain_probes.wgsl` | `ProbeGridUniforms`, `GpuProbeData`, `evaluate_sh_l2`, `sample_probe_irradiance` |
| `python/forge3d/terrain_params.py` | `ProbeSettings` dataclass (extend existing) |
| `examples/terrain_tv5_probe_lighting_demo.py` | Example with real DEM |
| `tests/test_terrain_probes.py` | Unit + visual + memory tests |

| Modified file | Change |
|---------------|--------|
| `src/terrain/renderer/core.rs` | Add probe SSBO, uniform buffer, fallback fields to `TerrainScene` |
| `src/terrain/renderer/constructor.rs` | Initialize probe fallback resources |
| `src/terrain/renderer/bind_groups/layouts.rs` | Extend `create_material_layer_bind_group_layout()` with bindings 1-2 |
| `src/terrain/renderer/bind_groups/terrain_pass.rs` | Build probe bind group entries |
| `src/terrain/renderer/pipeline_cache.rs` | Update `preprocess_terrain_shader()` to concatenate `terrain_probes.wgsl` include |
| `src/terrain/renderer/draw/execute.rs` | No change needed (set_bind_group(6) already bound) |
| `src/terrain/renderer/aov.rs` | No change needed (group(6) already bound) |
| `src/shaders/terrain_pbr_pom.wgsl` | `#include "terrain_probes.wgsl"`, modify irradiance sampling at line 2213, add debug modes |
| `src/render/params/gi.rs` | No structural change; `IrradianceProbes` variant already exists |
| `src/terrain/mod.rs` | Add `pub mod probes;` |

---

## 13. What This Design Does NOT Do

- Does not add a second rendering path or per-probe cubemap capture (that's SceneCaptureBaker, TV5.3)
- Does not modify the specular IBL path (global cubemap remains for specular)
- Does not add terrain color bleeding in v1 (occluded rays contribute zero, not terrain albedo)
- Does not add per-probe dynamic radius or weighted blending (v1 is uniform grid, bilinear)
- Does not change existing draw call structure, pipeline layout group count, or AOV paths
