# TV22 — Scatter Wind Animation Design

**Date:** 2026-03-23
**Epic:** TV22 (Scatter Wind Animation)
**Feasibility:** F1–F2
**Estimate:** 10–18 pd
**Priority:** P3

---

## 1. Problem Statement

Terrain scatter (TV3) handles deterministic placement, culling, and multi-LOD selection, but every placed instance is permanently static. Vegetation that never moves reads as synthetic, especially in populated scenes. TV22 adds GPU-driven wind deformation for scatter batches with per-batch controls, distance-aware stability, and deterministic replay.

## 2. Scope

### In scope

- GPU vertex-shader wind deformation for scatter batches.
- Per-batch wind controls: direction, speed, amplitude, rigidity, bend region, gust, and distance fade.
- Explicit caller-owned `time_seconds` for deterministic replay.
- Shared conversion helper used by both offscreen and viewer scatter paths.
- Distance-aware fade to suppress animation at range.
- Cheap normal tilt approximation for correct lighting during sway.
- Python `ScatterWindSettings` dataclass on `TerrainScatterBatch`.
- Native Rust `ScatterWindSettingsNative` + decode path.
- Example using real DEM assets.
- Regression and visual tests.

### Out of scope

- Cross-path spatial-phase parity (see §8 Accepted Limitations).
- Fixing the existing translation/basis frame mismatch in `pack_instance_transforms`.
- Per-instance wind variation (instance buffer extension).
- Branch/leaf secondary motion.
- Interactive wind painting or wind-field textures.
- CPU-side transform animation.

---

## 3. Python API Surface

### ScatterWindSettings

Location: `python/forge3d/terrain_scatter.py`

```python
@dataclass(frozen=True)
class ScatterWindSettings:
    """Per-batch wind animation controls for scatter vegetation."""
    enabled: bool = False

    # Global wind field
    direction_deg: float = 0.0        # Azimuth: 0 = +X, 90 = +Z in contract XZ (CW viewed from +Y, matches yaw convention)
    speed: float = 1.0                # Sway frequency scalar (>= 0)

    # Per-batch deformation
    amplitude: float = 0.0            # Max bend displacement in contract-space units (>= 0); 0 = no-op
    rigidity: float = 0.5             # [0, 1] — 0 = fully flexible, 1 = rigid (no sway)

    # Bend weighting (normalized against mesh-local Y bounds, y=0 at base)
    bend_start: float = 0.0           # Normalized height [0, 1] where bending begins
    bend_extent: float = 1.0          # Normalized height range above bend_start that bends (> 0)

    # Gust
    gust_strength: float = 0.0        # Additive gust amplitude in contract-space units (>= 0)
    gust_frequency: float = 0.3       # Gust temporal frequency scalar (>= 0)

    # Distance fade (approximate render-distance thresholds, see §8)
    fade_start: float = 0.0           # Distance where fade begins (>= 0)
    fade_end: float = 0.0             # Distance where wind reaches zero (>= 0)
```

**Validation (`__post_init__`):**

| Field | Constraint |
|-------|-----------|
| `speed` | `>= 0` |
| `amplitude` | `>= 0` |
| `rigidity` | `in [0, 1]` |
| `bend_start` | `in [0, 1]` |
| `bend_extent` | `> 0` |
| `gust_strength` | `>= 0` |
| `gust_frequency` | `>= 0` |
| `fade_start` | `>= 0` |
| `fade_end` | `>= 0` |

**No-fade rule:** wind is un-faded when `fade_end <= fade_start`. Both `None` fields are not used; the sentinel is the inequality.

### Integration with TerrainScatterBatch

```python
@dataclass
class TerrainScatterBatch:
    levels: Sequence[TerrainScatterLevel]
    transforms: np.ndarray
    name: str | None = None
    color: Sequence[float] = (0.85, 0.85, 0.85, 1.0)
    max_draw_distance: float | None = None
    wind: ScatterWindSettings = field(default_factory=ScatterWindSettings)  # NEW
```

`wind.enabled == False` or `wind.amplitude == 0.0` → strict no-op. Existing batches that do not set `wind` get the default (disabled) and behave identically to pre-TV22.

### Serialization

`to_native_dict()` and `to_viewer_payload()` always include a `"wind"` key with the settings dict (for debuggability). When wind is disabled, the dict contains default values. The Rust parser treats a missing or null `"wind"` key as all-zero (no-op) for backward compatibility.

### time_seconds

- **Offscreen renderer:** new optional keyword argument `time_seconds: float = 0.0` on `render_terrain_pbr_pom` and `render_with_aov`.
- **Viewer:** accumulated from the viewer frame loop's `dt`.
- **Contract:** the caller owns the clock. Same `time_seconds` + same wind settings = same deformation (deterministic).

---

## 4. Rust Native Layer

### ScatterWindSettingsNative

Location: `src/terrain/scatter.rs` (co-located with `TerrainScatterBatch` and `ScatterWindUniforms`)

```rust
#[derive(Clone, Debug)]
pub struct ScatterWindSettingsNative {
    pub enabled: bool,
    pub direction_deg: f32,
    pub speed: f32,
    pub amplitude: f32,
    pub rigidity: f32,
    pub bend_start: f32,
    pub bend_extent: f32,
    pub gust_strength: f32,
    pub gust_frequency: f32,
    pub fade_start: f32,
    pub fade_end: f32,
}
```

Default: all zeroes / `enabled = false`.

### Decode path

`parse_wind_settings()` in a new `decode_wind.rs` or inline in `py_api.rs` scatter batch parsing. Reads `"wind"` dict key from the batch dict, extracts fields with defaults, returns `ScatterWindSettingsNative`.

### TerrainScatterUploadBatch extension

```rust
pub(super) struct TerrainScatterUploadBatch {
    // ... existing fields ...
    pub(super) wind: ScatterWindSettingsNative,  // NEW
}
```

### TerrainScatterBatch extension

The GPU-side `TerrainScatterBatch` in `src/terrain/scatter.rs` stores `wind: ScatterWindSettingsNative` alongside existing fields.

### GpuScatterLevel extension

`mesh_height_max: f32` is computed once in `build_gpu_level` as `max(vertex.y for all vertices in the mesh)`. A log warning is emitted if `min(vertex.y)` deviates from zero by more than 5% of the mesh Y extent (`max_y - min_y`).

---

## 5. Uniform Layout

### Rename

`SceneUniforms` → `ScatterBatchUniforms` in both `src/render/mesh_instanced.rs` and `src/shaders/mesh_instanced.wgsl`.

### Extended struct (160 → 208 bytes)

```wgsl
struct ScatterBatchUniforms {
  view: mat4x4<f32>,                  // 64 bytes
  proj: mat4x4<f32>,                  // 64 bytes
  color: vec4<f32>,                   // 16 bytes
  light_dir_ws: vec4<f32>,            // 16 bytes  (xyz: dir, w: intensity)
  // --- wind (3 × vec4, 48 bytes) ---
  wind_phase: vec4<f32>,              // x: temporal_phase, y: gust_phase, z: gust_strength, w: rigidity
  wind_vec_bounds: vec4<f32>,         // xyz: local wind_dir * amplitude (contract units), w: mesh_height_max
  wind_bend_fade: vec4<f32>,          // x: bend_start, y: bend_extent, z: fade_start (render-space), w: fade_end (render-space)
}
```

512 pre-allocated per-draw uniform slots grow by 24,576 bytes total. Negligible.

### All-zero sentinel

When `length(wind_vec_bounds.xyz) < 1e-6`, the vertex shader skips the entire wind block. Zero cost when wind is disabled.

### draw_batch_params signature extension

The existing `draw_batch_params` in `src/render/mesh_instanced.rs` gains three new `[f32; 4]` parameters for the wind uniform fields:

```rust
pub fn draw_batch_params<'rp>(
    &'rp self,
    _device: &Device,
    pass: &mut RenderPass<'rp>,
    queue: &Queue,
    view: Mat4,
    proj: Mat4,
    color: [f32; 4],
    light_dir: [f32; 3],
    light_intensity: f32,
    // --- wind (new) ---
    wind_phase: [f32; 4],
    wind_vec_bounds: [f32; 4],
    wind_bend_fade: [f32; 4],
    // --- buffers ---
    vbuf: &'rp Buffer,
    ibuf: &'rp Buffer,
    instbuf: &'rp Buffer,
    index_count: u32,
    instance_count: u32,
)
```

When wind is disabled, callers pass `[0.0; 4]` for all three wind parameters. The Rust struct representation:

```rust
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScatterBatchUniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    color: [f32; 4],
    light_dir_ws: [f32; 4],
    wind_phase: [f32; 4],
    wind_vec_bounds: [f32; 4],
    wind_bend_fade: [f32; 4],
}
```

---

## 6. CPU-Side Conversion

### Shared helper

Location: `src/terrain/scatter.rs`

```rust
pub struct ScatterWindUniforms {
    pub wind_phase: [f32; 4],
    pub wind_vec_bounds: [f32; 4],
    pub wind_bend_fade: [f32; 4],
}

/// Compute shader-ready wind uniform fields.
///
/// Returns all-zero fields when `wind.enabled` is false or `wind.amplitude` is zero.
/// Batch-constant fields come from `wind` and `time_seconds`.
/// `mesh_height_max` is per-draw (per LOD level) and must be injected by the caller.
/// `instance_scale` is used only for fade distance conversion.
pub fn compute_wind_uniforms(
    wind: &ScatterWindSettingsNative,
    time_seconds: f32,
    mesh_height_max: f32,
    instance_scale: f32,
) -> ScatterWindUniforms
```

### Conversion rules

| Uniform field | Computation | Notes |
|---------------|-------------|-------|
| `temporal_phase` | `time_seconds * speed * 2π` | CPU folds speed into phase |
| `gust_phase` | `time_seconds * gust_frequency * 2π` | CPU folds frequency into phase |
| `wind_vec_bounds.xyz` | `(cos(az), 0, sin(az)) * amplitude` | Local/contract frame, no `render_from_contract` |
| `wind_vec_bounds.w` | `mesh_height_max` | Per-draw, from `GpuScatterLevel` |
| `gust_strength` | `gust_strength` (passthrough) | Contract units, scaled by M in shader |
| `rigidity` | `rigidity` (passthrough) | [0, 1] |
| `bend_start` | `bend_start` (passthrough) | Normalized [0, 1] |
| `bend_extent` | `bend_extent` (passthrough) | Normalized, > 0 |
| `fade_start` | `fade_start * instance_scale` | Approximate render-space (see §8) |
| `fade_end` | `fade_end * instance_scale` | Approximate render-space (see §8) |

### Per-batch vs per-draw split

- **Once per batch:** `temporal_phase`, `gust_phase`, `wind_vec_bounds.xyz`, `gust_strength`, `rigidity`, `bend_start`, `bend_extent`, `fade_start`, `fade_end`.
- **Per draw (per LOD level):** `wind_vec_bounds.w` (`mesh_height_max`), injected right before `draw_batch_params`.

### Render path integration

Both `src/terrain/renderer/scatter.rs` (offscreen) and `src/viewer/terrain/scene/scatter.rs` (viewer) call `compute_wind_uniforms`. The `ScatterRenderState` struct gains a `time_seconds: f32` field.

---

## 7. Vertex Shader

### Wind deformation in mesh_instanced.wgsl

```wgsl
@vertex
fn vs_main(in: VsIn) -> VsOut {
  var out: VsOut;
  let M = mat4x4<f32>(in.i_m0, in.i_m1, in.i_m2, in.i_m3);
  var pos_ws = M * vec4<f32>(in.position, 1.0);
  var n_ws = normalize((M * vec4<f32>(in.normal, 0.0)).xyz);

  let wind_local = U.wind_vec_bounds.xyz;
  let wind_amp = length(wind_local);

  if (wind_amp > 1e-6) {
    // Bend weight from mesh-local normalized Y height
    let norm_h = clamp(in.position.y / max(U.wind_vec_bounds.w, 1e-4), 0.0, 1.0);
    let bend_weight = smoothstep(
      U.wind_bend_fade.x,
      U.wind_bend_fade.x + U.wind_bend_fade.y,
      norm_h
    );

    // Wind direction in world space (for spatial phase variety between instances)
    let wind_dir_ws = normalize((M * vec4<f32>(wind_local, 0.0)).xyz);

    // Deterministic sway + gust
    let spatial = dot(pos_ws.xyz, wind_dir_ws) * 0.1;
    let sway = sin(U.wind_phase.x + spatial) * (1.0 - U.wind_phase.w) * wind_amp;
    let gust = sin(U.wind_phase.y + spatial * 0.37) * U.wind_phase.z;  // 0.37: decorrelation factor to prevent gust locking with sway

    // Displacement in local frame, transformed to world through M
    let wind_dir_local = wind_local / wind_amp;
    let disp_local = wind_dir_local * (sway + gust) * bend_weight;
    var disp_ws = (M * vec4<f32>(disp_local, 0.0)).xyz;

    // Distance fade (true view-space distance)
    let fade_start = U.wind_bend_fade.z;
    let fade_end = U.wind_bend_fade.w;
    if (fade_end > fade_start) {
      let view_pos = U.view * pos_ws;
      let view_dist = length(view_pos.xyz);
      disp_ws *= 1.0 - smoothstep(fade_start, fade_end, view_dist);
    }

    // Apply displacement
    pos_ws = vec4<f32>(pos_ws.xyz + disp_ws, 1.0);

    // Cheap normal tilt toward wind, proportional to displacement
    let tilt = length(disp_ws) * 0.3;
    let up_ws = normalize(in.i_m1.xyz);
    n_ws = normalize(n_ws + wind_dir_ws * tilt * max(dot(n_ws, up_ws), 0.0));
  }

  out.pos = U.proj * U.view * pos_ws;
  out.n_ws = n_ws;
  return out;
}
```

### Properties

- **Deterministic:** same `time_seconds` + same wind settings = same deformation.
- **Default-off:** `wind_amp < ε` skips the entire block.
- **Normal tilt:** tilts normal toward wind proportional to displacement magnitude, weighted by how much the normal faces the instance up direction.
- **Mesh contract:** `position.y` is mesh-local height; meshes should have `y ≈ 0` at the base (see §8).

---

## 8. Accepted Limitations

### 8.1 Spatial phase is render-path dependent

`dot(pos_ws.xyz, wind_dir_ws)` uses render-space position for spatial phase. The offscreen and viewer scatter paths produce different render-space coordinates for the same contract-space instance, so the per-vertex phase offsets will differ between paths. Cross-path visual parity of per-vertex phase is a non-goal for TV22. If parity becomes a requirement in a future epic, phase should be computed from contract-space position or a CPU-side per-instance phase seed.

### 8.2 Fade distances are approximate render-space thresholds

`fade_start * instance_scale` and `fade_end * instance_scale` apply a uniform scale factor, but `render_from_contract` in the offscreen path is anisotropic (different horizontal and vertical scale). The fade values are documented as **approximate render-distance thresholds**, not exact contract-space distance conversions. For scenes where the anisotropy is significant, authors should tune fade values empirically.

### 8.3 Mesh Y=0 base convention

Wind bend weighting normalizes `position.y` against `mesh_height_max`. Meshes that do not have `y ≈ 0` at their base will bend from an incorrect baseline. `build_gpu_level` logs a warning if `mesh_y_min` deviates from zero by more than 5% of the mesh Y extent. Meshes violating this convention are documented as unsupported for wind animation — they will still render, but bend behavior will be incorrect.

### 8.4 Translation/basis frame mismatch

The existing `pack_instance_transforms` in `src/terrain/scatter.rs` remaps instance translation through `render_from_contract` but does not remap the instance basis columns. TV22 works around this by operating entirely in the mesh-local frame (wind direction, displacement, bend height). A future cleanup could unify the instance matrix frame, but that is out of TV22 scope and would affect all scatter rendering.

### 8.5 Per-instance scale interaction

Instance `scale_range` affects wind displacement proportionally — a scatter instance at `scale=5` sways 5× more than one at `scale=1`. This is physically realistic for vegetation (taller trees sway more) and is documented behavior, not a bug.

---

## 9. File Inventory

### New files

| File | Purpose |
|------|---------|
| `tests/test_tv22_scatter_wind.py` | Python-side validation, integration, and visual regression tests |
| `examples/terrain_tv22_scatter_wind_demo.py` | Example using real DEM assets |

### Modified files

| File | Change |
|------|--------|
| `python/forge3d/terrain_scatter.py` | Add `ScatterWindSettings`, extend `TerrainScatterBatch` |
| `python/forge3d/__init__.py` | Export `ScatterWindSettings` |
| `python/forge3d/__init__.pyi` | Type stub for `ScatterWindSettings` |
| `src/render/mesh_instanced.rs` | Rename `SceneUniforms` → `ScatterBatchUniforms`, add wind fields, extend `draw_batch_params` |
| `src/shaders/mesh_instanced.wgsl` | Rename struct, add wind deformation to `vs_main` |
| `src/terrain/scatter.rs` | Add `ScatterWindSettingsNative`, `ScatterWindUniforms`, `compute_wind_uniforms`, `mesh_height_max` on `GpuScatterLevel`, wind field on `TerrainScatterBatch` |
| `src/terrain/renderer/scatter.rs` | Extend `ScatterRenderState` with `time_seconds`, call `compute_wind_uniforms`, pass wind to `draw_batch_params` |
| `src/terrain/renderer/py_api.rs` | Parse `"wind"` dict from scatter batch, add `time_seconds` kwarg |
| `src/viewer/terrain/scene/scatter.rs` | Call shared `compute_wind_uniforms`, pass `time_seconds` from frame loop |
| `src/terrain/render_params/mod.rs` | Re-export `ScatterWindSettingsNative` if needed for cross-module access |
| `tests/test_api_contracts.py` | Add `ScatterWindSettings` to contract surface |

---

## 10. Test Plan

### Unit tests (Python)

- `ScatterWindSettings` defaults: all fields have expected values, `enabled=False`.
- Validation: `speed < 0`, `rigidity > 1`, `amplitude < 0`, `bend_extent <= 0`, etc. all raise `ValueError`.
- `TerrainScatterBatch` with `wind=ScatterWindSettings()` serializes correctly (native + viewer payloads).
- `TerrainScatterBatch` without explicit `wind` field matches default-factory behavior.

### Unit tests (Rust)

- `compute_wind_uniforms` with default/disabled wind returns all-zero fields.
- `compute_wind_uniforms` with enabled wind returns expected phase, direction, fade values.
- `mesh_height_max` is correctly computed from vertex data.
- `mesh_y_min` warning fires for meshes with non-zero base.

### Integration tests

- Render a scatter scene with wind enabled, verify non-zero pixel difference from the static baseline (same scene, `time_seconds=0` vs `time_seconds=1`).
- Render with `wind.enabled=False`, verify pixel-exact match with pre-TV22 baseline.
- Render with `amplitude=0.0`, verify pixel-exact match with static baseline.
- Render with `rigidity=1.0`, verify pixel-exact match with static baseline (`(1.0 - rigidity)` zeros out sway).
- Render at two different `time_seconds` values with the same wind settings, verify different output (animation is happening).
- Render at the same `time_seconds` twice, verify identical output (determinism).
- Render at large view-space distance beyond `fade_end`, verify pixel-exact match with static baseline (fade suppresses all motion).
- Render with `bend_start=0.0` vs `bend_start=0.8` at the same `time_seconds`, verify different output (bend region affects which mesh portion sways).
- Render with `gust_strength=0.0` vs `gust_strength > 0` at the same `time_seconds`, verify different output (gust adds additional motion).

### Visual regression

- Example scene rendered at multiple `time_seconds` values, output PNGs compared for plausible vegetation motion.

---

## 11. Definition of Done (per epic spec)

| Task | Definition of done |
|------|-------------------|
| **TV22.1** Add deterministic wind deformation for scatter | Scatter batches sway entirely on GPU; same seed and wind settings produce the same motion; disabling wind restores the static baseline. |
| **TV22.2** Add bounded motion controls per batch or asset class | Rigidity, amplitude, bend_start/extent, and gust response tunable per scatter batch; defaults are conservative and stable. |
| **TV22.3** Add distance-aware stability and tests | Distant vegetation motion reduced by fade policy; tests verify stable animation under camera motion and wind changes. |
