# TV22 Scatter Wind Animation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add GPU vertex-shader wind deformation for scatter batches with per-batch controls, distance fade, and deterministic replay.

**Architecture:** Wind settings live on each `TerrainScatterBatch` (Python dataclass), flow through dict serialization to Rust native structs, get converted to 3 `vec4` uniform fields by a shared `compute_wind_uniforms` helper, and drive sinusoidal vertex displacement in `mesh_instanced.wgsl`. Both offscreen and viewer scatter paths share the same conversion helper and shader.

**Tech Stack:** Python 3 (dataclasses, numpy), Rust (wgpu, PyO3, bytemuck, glam), WGSL shaders

**Spec:** `docs/superpowers/specs/2026-03-23-tv22-scatter-wind-animation-design.md`

---

## File Map

### New files

| File | Responsibility |
|------|---------------|
| `tests/test_tv22_scatter_wind.py` | Integration tests: render with/without wind, determinism, fade, controls |
| `examples/terrain_tv22_scatter_wind_demo.py` | Example using real DEM assets showing wind animation |

### Modified files

| File | Change summary |
|------|---------------|
| `python/forge3d/terrain_scatter.py` | Add `ScatterWindSettings` dataclass, extend `TerrainScatterBatch` with `wind` field and serialization |
| `python/forge3d/__init__.py` | Export `ScatterWindSettings` in `__all__` |
| `python/forge3d/__init__.pyi` | Type stub for `ScatterWindSettings`; `time_seconds` on render methods |
| `src/terrain/scatter.rs` | Add `ScatterWindSettingsNative`, `ScatterWindUniforms`, `compute_wind_uniforms()`, `mesh_height_max` on `GpuScatterLevel`, wind on `TerrainScatterBatch` |
| `src/render/mesh_instanced.rs` | Rename `SceneUniforms` → `ScatterBatchUniforms`, add 3 wind `[f32;4]` fields, extend `draw_batch_params` signature |
| `src/shaders/mesh_instanced.wgsl` | Rename struct, add wind deformation block to `vs_main` |
| `src/terrain/renderer/scatter.rs` | Add `time_seconds` to `ScatterRenderState`, call `compute_wind_uniforms`, pass wind to `draw_batch_params` |
| `src/terrain/renderer/py_api.rs` | Parse `"wind"` dict from scatter batch, add `time_seconds` kwarg to `render_terrain_pbr_pom` and `render_with_aov` |
| `src/viewer/ipc/protocol/payloads.rs` | Add wind fields to `IpcTerrainScatterBatch` |
| `src/viewer/viewer_enums/config.rs` | Add wind to `ViewerTerrainScatterBatchConfig` |
| `src/viewer/ipc/protocol/translate/terrain.rs` | Parse wind from IPC payload in `map_terrain_scatter_batch` |
| `src/viewer/terrain/scene/scatter.rs` | Call `compute_wind_uniforms`, pass wind + `time_seconds` to `draw_batch_params` |
| `src/scene/render_paths/png.rs` | Update `draw_batch_params` callsite with zero wind params |
| `tests/test_terrain_scatter.py` | Add `ScatterWindSettings` default, validation, and serialization tests |
| `tests/test_viewer_ipc.py` | Add wind field to viewer scatter IPC round-trip tests |

---

## Task 1: Python ScatterWindSettings Dataclass

**Files:**
- Modify: `python/forge3d/terrain_scatter.py`
- Test: `tests/test_terrain_scatter.py`

- [ ] **Step 1: Write failing tests for ScatterWindSettings defaults and validation**

Add a new test class at the end of `tests/test_terrain_scatter.py`:

```python
class TestScatterWindSettings:
    """Tests for ScatterWindSettings dataclass."""

    def test_defaults(self):
        s = ts.ScatterWindSettings()
        assert s.enabled is False
        assert s.direction_deg == 0.0
        assert s.speed == 1.0
        assert s.amplitude == 0.0
        assert s.rigidity == 0.5
        assert s.bend_start == 0.0
        assert s.bend_extent == 1.0
        assert s.gust_strength == 0.0
        assert s.gust_frequency == 0.3
        assert s.fade_start == 0.0
        assert s.fade_end == 0.0

    def test_enabled_with_amplitude(self):
        s = ts.ScatterWindSettings(enabled=True, amplitude=2.0)
        assert s.enabled is True
        assert s.amplitude == 2.0

    def test_speed_rejects_negative(self):
        with pytest.raises(ValueError, match="speed"):
            ts.ScatterWindSettings(speed=-1.0)

    def test_amplitude_rejects_negative(self):
        with pytest.raises(ValueError, match="amplitude"):
            ts.ScatterWindSettings(amplitude=-0.1)

    def test_rigidity_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="rigidity"):
            ts.ScatterWindSettings(rigidity=1.5)
        with pytest.raises(ValueError, match="rigidity"):
            ts.ScatterWindSettings(rigidity=-0.1)

    def test_bend_start_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="bend_start"):
            ts.ScatterWindSettings(bend_start=-0.1)
        with pytest.raises(ValueError, match="bend_start"):
            ts.ScatterWindSettings(bend_start=1.1)

    def test_bend_extent_rejects_non_positive(self):
        with pytest.raises(ValueError, match="bend_extent"):
            ts.ScatterWindSettings(bend_extent=0.0)
        with pytest.raises(ValueError, match="bend_extent"):
            ts.ScatterWindSettings(bend_extent=-1.0)

    def test_gust_strength_rejects_negative(self):
        with pytest.raises(ValueError, match="gust_strength"):
            ts.ScatterWindSettings(gust_strength=-1.0)

    def test_gust_frequency_rejects_negative(self):
        with pytest.raises(ValueError, match="gust_frequency"):
            ts.ScatterWindSettings(gust_frequency=-1.0)

    def test_fade_start_rejects_negative(self):
        with pytest.raises(ValueError, match="fade_start"):
            ts.ScatterWindSettings(fade_start=-1.0)

    def test_fade_end_rejects_negative(self):
        with pytest.raises(ValueError, match="fade_end"):
            ts.ScatterWindSettings(fade_end=-1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_terrain_scatter.py::TestScatterWindSettings -v`
Expected: FAIL (ScatterWindSettings not defined)

- [ ] **Step 3: Implement ScatterWindSettings**

Add to `python/forge3d/terrain_scatter.py` before `TerrainScatterFilters`:

```python
@dataclass(frozen=True)
class ScatterWindSettings:
    """Per-batch wind animation controls for scatter vegetation."""

    enabled: bool = False
    direction_deg: float = 0.0
    speed: float = 1.0
    amplitude: float = 0.0
    rigidity: float = 0.5
    bend_start: float = 0.0
    bend_extent: float = 1.0
    gust_strength: float = 0.0
    gust_frequency: float = 0.3
    fade_start: float = 0.0
    fade_end: float = 0.0

    def __post_init__(self) -> None:
        if float(self.speed) < 0.0:
            raise ValueError("speed must be >= 0")
        if float(self.amplitude) < 0.0:
            raise ValueError("amplitude must be >= 0")
        if not (0.0 <= float(self.rigidity) <= 1.0):
            raise ValueError("rigidity must be in [0, 1]")
        if not (0.0 <= float(self.bend_start) <= 1.0):
            raise ValueError("bend_start must be in [0, 1]")
        if float(self.bend_extent) <= 0.0:
            raise ValueError("bend_extent must be > 0")
        if float(self.gust_strength) < 0.0:
            raise ValueError("gust_strength must be >= 0")
        if float(self.gust_frequency) < 0.0:
            raise ValueError("gust_frequency must be >= 0")
        if float(self.fade_start) < 0.0:
            raise ValueError("fade_start must be >= 0")
        if float(self.fade_end) < 0.0:
            raise ValueError("fade_end must be >= 0")
```

Add `"ScatterWindSettings"` to the `__all__` list at the bottom of the file.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_terrain_scatter.py::TestScatterWindSettings -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add python/forge3d/terrain_scatter.py tests/test_terrain_scatter.py
git commit -m "feat(tv22): add ScatterWindSettings dataclass with validation"
```

---

## Task 2: Extend TerrainScatterBatch with Wind Field + Serialization

**Files:**
- Modify: `python/forge3d/terrain_scatter.py`
- Test: `tests/test_terrain_scatter.py`

- [ ] **Step 1: Write failing tests for batch wind field and serialization**

Add to `TestBatchSerialization` in `tests/test_terrain_scatter.py`:

```python
def test_batch_default_wind_is_disabled(self):
    batch = ts.TerrainScatterBatch(
        levels=[ts.TerrainScatterLevel(mesh=_trivial_mesh())],
        transforms=_trivial_transforms(),
    )
    assert batch.wind.enabled is False
    assert batch.wind.amplitude == 0.0

def test_batch_with_explicit_wind(self):
    wind = ts.ScatterWindSettings(enabled=True, amplitude=1.5, rigidity=0.3)
    batch = ts.TerrainScatterBatch(
        levels=[ts.TerrainScatterLevel(mesh=_trivial_mesh())],
        transforms=_trivial_transforms(),
        wind=wind,
    )
    assert batch.wind.enabled is True
    assert batch.wind.amplitude == 1.5

def test_native_dict_includes_wind(self):
    wind = ts.ScatterWindSettings(enabled=True, amplitude=2.0, direction_deg=45.0)
    batch = ts.TerrainScatterBatch(
        levels=[ts.TerrainScatterLevel(mesh=_trivial_mesh())],
        transforms=_trivial_transforms(),
        wind=wind,
    )
    d = batch.to_native_dict()
    assert "wind" in d
    assert d["wind"]["enabled"] is True
    assert d["wind"]["amplitude"] == 2.0
    assert d["wind"]["direction_deg"] == 45.0

def test_viewer_payload_includes_wind(self):
    wind = ts.ScatterWindSettings(enabled=True, amplitude=1.0)
    batch = ts.TerrainScatterBatch(
        levels=[ts.TerrainScatterLevel(mesh=_trivial_mesh())],
        transforms=_trivial_transforms(),
        wind=wind,
    )
    p = batch.to_viewer_payload()
    assert "wind" in p
    assert p["wind"]["enabled"] is True

def test_disabled_wind_still_serializes(self):
    batch = ts.TerrainScatterBatch(
        levels=[ts.TerrainScatterLevel(mesh=_trivial_mesh())],
        transforms=_trivial_transforms(),
    )
    d = batch.to_native_dict()
    assert "wind" in d
    assert d["wind"]["enabled"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_terrain_scatter.py::TestBatchSerialization::test_batch_default_wind_is_disabled -v`
Expected: FAIL

- [ ] **Step 3: Implement batch wind field and serialization**

In `python/forge3d/terrain_scatter.py`, modify `TerrainScatterBatch`:

```python
from dataclasses import dataclass, field

@dataclass
class TerrainScatterBatch:
    levels: Sequence[TerrainScatterLevel]
    transforms: np.ndarray
    name: str | None = None
    color: Sequence[float] = (0.85, 0.85, 0.85, 1.0)
    max_draw_distance: float | None = None
    wind: ScatterWindSettings = field(default_factory=ScatterWindSettings)
    # ... existing __post_init__ unchanged ...
```

Add wind serialization to `to_native_dict()`:

```python
def to_native_dict(self) -> dict[str, Any]:
    return {
        "name": self.name,
        "color": tuple(self.color),
        "max_draw_distance": self.max_draw_distance,
        "transforms": self.transforms,
        "levels": [
            {
                "mesh": _mesh_to_py(level.mesh),
                "max_distance": level.max_distance,
            }
            for level in self.levels
        ],
        "wind": {
            "enabled": self.wind.enabled,
            "direction_deg": self.wind.direction_deg,
            "speed": self.wind.speed,
            "amplitude": self.wind.amplitude,
            "rigidity": self.wind.rigidity,
            "bend_start": self.wind.bend_start,
            "bend_extent": self.wind.bend_extent,
            "gust_strength": self.wind.gust_strength,
            "gust_frequency": self.wind.gust_frequency,
            "fade_start": self.wind.fade_start,
            "fade_end": self.wind.fade_end,
        },
    }
```

Add the same `"wind"` dict to `to_viewer_payload()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_terrain_scatter.py::TestBatchSerialization -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add python/forge3d/terrain_scatter.py tests/test_terrain_scatter.py
git commit -m "feat(tv22): add wind field to TerrainScatterBatch with serialization"
```

---

## Task 3: Python Exports

**Files:**
- Modify: `python/forge3d/__init__.py`
- Modify: `python/forge3d/__init__.pyi`

- [ ] **Step 1: Add ScatterWindSettings to __init__.py**

Import `ScatterWindSettings` from `terrain_scatter` alongside existing scatter imports. Add to `__all__`.

- [ ] **Step 2: Add type stub to __init__.pyi**

Add `ScatterWindSettings` import from `.terrain_scatter`. Add `time_seconds: float = ...` parameter to `render_terrain_pbr_pom` and `render_with_aov` stubs.

- [ ] **Step 3: Verify import works**

Run: `python -c "from forge3d import ScatterWindSettings; print(ScatterWindSettings())"`
Expected: prints the default ScatterWindSettings

- [ ] **Step 4: Commit**

```bash
git add python/forge3d/__init__.py python/forge3d/__init__.pyi
git commit -m "feat(tv22): export ScatterWindSettings and update type stubs"
```

---

## Task 4: Rust ScatterWindSettingsNative + compute_wind_uniforms

**Files:**
- Modify: `src/terrain/scatter.rs`

- [ ] **Step 1: Add ScatterWindSettingsNative struct**

Add to `src/terrain/scatter.rs` near the top (after existing structs):

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

impl Default for ScatterWindSettingsNative {
    fn default() -> Self {
        Self {
            enabled: false,
            direction_deg: 0.0,
            speed: 1.0,
            amplitude: 0.0,
            rigidity: 0.5,
            bend_start: 0.0,
            bend_extent: 1.0,
            gust_strength: 0.0,
            gust_frequency: 0.3,
            fade_start: 0.0,
            fade_end: 0.0,
        }
    }
}
```

- [ ] **Step 2: Add ScatterWindUniforms and compute_wind_uniforms**

```rust
#[derive(Clone, Copy, Debug, Default)]
pub struct ScatterWindUniforms {
    pub wind_phase: [f32; 4],
    pub wind_vec_bounds: [f32; 4],
    pub wind_bend_fade: [f32; 4],
}

/// Compute shader-ready wind uniform fields.
///
/// Returns all-zero fields when `wind.enabled` is false or `wind.amplitude` is zero.
/// `mesh_height_max` is per-draw (per LOD level), injected by the caller.
/// `instance_scale` is used only for fade distance conversion.
pub fn compute_wind_uniforms(
    wind: &ScatterWindSettingsNative,
    time_seconds: f32,
    mesh_height_max: f32,
    instance_scale: f32,
) -> ScatterWindUniforms {
    if !wind.enabled || wind.amplitude <= 0.0 {
        return ScatterWindUniforms::default();
    }

    let tau = std::f32::consts::TAU;
    let az = wind.direction_deg.to_radians();
    let dir_x = az.cos();
    let dir_z = az.sin();

    ScatterWindUniforms {
        wind_phase: [
            time_seconds * wind.speed * tau,         // temporal_phase
            time_seconds * wind.gust_frequency * tau, // gust_phase
            wind.gust_strength,                       // gust_strength (contract units)
            wind.rigidity,                            // rigidity [0,1]
        ],
        wind_vec_bounds: [
            dir_x * wind.amplitude,  // wind_dir.x * amplitude
            0.0,                     // wind_dir.y (zero, XZ ground plane)
            dir_z * wind.amplitude,  // wind_dir.z * amplitude
            mesh_height_max,         // per-draw mesh height
        ],
        wind_bend_fade: [
            wind.bend_start,                      // bend_start [0,1]
            wind.bend_extent,                     // bend_extent (> 0)
            wind.fade_start * instance_scale,     // fade_start (approx render-space)
            wind.fade_end * instance_scale,       // fade_end (approx render-space)
        ],
    }
}
```

- [ ] **Step 3: Add wind field to TerrainScatterBatch**

In `src/terrain/scatter.rs`, add `pub wind: ScatterWindSettingsNative` to the `TerrainScatterBatch` struct (around line 74). Update the `new()` constructor to accept and store it.

- [ ] **Step 4: Add unit tests for compute_wind_uniforms**

```rust
#[test]
fn compute_wind_disabled_returns_zeros() {
    let wind = ScatterWindSettingsNative::default();
    let u = compute_wind_uniforms(&wind, 1.0, 10.0, 1.0);
    assert_eq!(u.wind_phase, [0.0; 4]);
    assert_eq!(u.wind_vec_bounds, [0.0; 4]);
    assert_eq!(u.wind_bend_fade, [0.0; 4]);
}

#[test]
fn compute_wind_zero_amplitude_returns_zeros() {
    let wind = ScatterWindSettingsNative { enabled: true, amplitude: 0.0, ..Default::default() };
    let u = compute_wind_uniforms(&wind, 1.0, 10.0, 1.0);
    assert_eq!(u.wind_vec_bounds[0], 0.0);
    assert_eq!(u.wind_vec_bounds[2], 0.0);
}

#[test]
fn compute_wind_direction_and_amplitude() {
    let wind = ScatterWindSettingsNative {
        enabled: true,
        direction_deg: 0.0,  // +X direction
        amplitude: 3.0,
        ..Default::default()
    };
    let u = compute_wind_uniforms(&wind, 0.0, 5.0, 2.0);
    // direction = (cos(0), 0, sin(0)) * 3.0 = (3.0, 0, 0)
    assert!((u.wind_vec_bounds[0] - 3.0).abs() < 1e-6);
    assert!((u.wind_vec_bounds[1]).abs() < 1e-6);
    assert!((u.wind_vec_bounds[2]).abs() < 1e-6);
    assert!((u.wind_vec_bounds[3] - 5.0).abs() < 1e-6); // mesh_height_max
}

#[test]
fn compute_wind_fade_scales_by_instance_scale() {
    let wind = ScatterWindSettingsNative {
        enabled: true,
        amplitude: 1.0,
        fade_start: 10.0,
        fade_end: 20.0,
        ..Default::default()
    };
    let u = compute_wind_uniforms(&wind, 0.0, 1.0, 3.0);
    assert!((u.wind_bend_fade[2] - 30.0).abs() < 1e-6); // 10 * 3
    assert!((u.wind_bend_fade[3] - 60.0).abs() < 1e-6); // 20 * 3
}

#[test]
fn compute_wind_phase_folds_speed() {
    let wind = ScatterWindSettingsNative {
        enabled: true,
        amplitude: 1.0,
        speed: 2.0,
        gust_frequency: 0.5,
        ..Default::default()
    };
    let u = compute_wind_uniforms(&wind, 1.0, 1.0, 1.0);
    let tau = std::f32::consts::TAU;
    assert!((u.wind_phase[0] - 2.0 * tau).abs() < 1e-4); // time * speed * tau
    assert!((u.wind_phase[1] - 0.5 * tau).abs() < 1e-4); // time * gust_freq * tau
}
```

- [ ] **Step 5: Run Rust tests**

Run: `cargo test -p forge3d --lib terrain::scatter::tests -- compute_wind`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/terrain/scatter.rs
git commit -m "feat(tv22): add ScatterWindSettingsNative, compute_wind_uniforms, and wind on TerrainScatterBatch"
```

---

## Task 5: mesh_height_max on GpuScatterLevel

**Files:**
- Modify: `src/terrain/scatter.rs`

- [ ] **Step 1: Add mesh_height_max to GpuScatterLevel**

In `src/terrain/scatter.rs`, add `mesh_height_max: f32` to the `GpuScatterLevel` struct (around line 56).

- [ ] **Step 2: Compute mesh_height_max in build_gpu_level**

In `build_gpu_level()` (around line 365), after building the vertices vector, scan for Y bounds:

```rust
let (y_min, y_max) = vertices.iter().fold(
    (f32::INFINITY, f32::NEG_INFINITY),
    |(mn, mx), v| (mn.min(v.position[1]), mx.max(v.position[1])),
);
let mesh_height_max = y_max;
let y_extent = y_max - y_min;
if y_extent > 1e-6 && y_min.abs() > 0.05 * y_extent {
    eprintln!(
        "[terrain.scatter] warning: mesh y_min={y_min:.3} deviates from zero \
         by >{:.0}% of y_extent={y_extent:.3}; wind bend weighting may be incorrect",
        (y_min.abs() / y_extent) * 100.0
    );
}
```

Store `mesh_height_max` on the returned `GpuScatterLevel`.

- [ ] **Step 3: Add accessor**

Add a public method to `TerrainScatterBatch`:

```rust
pub fn level_mesh_height_max(&self, level_index: usize) -> f32 {
    self.levels[level_index].mesh_height_max
}
```

- [ ] **Step 4: Add test**

```rust
#[test]
fn mesh_height_max_uses_vertex_y_max() {
    // Vertices: y values [0.0, 1.5, 3.0]
    let positions = vec![[0.0, 0.0, 0.0], [1.0, 1.5, 0.0], [0.0, 3.0, 1.0]];
    let y_max = positions.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);
    assert!((y_max - 3.0).abs() < 1e-6);
}

#[test]
fn mesh_y_min_warning_threshold() {
    // Mesh with y_min=0.5, y_max=3.0, extent=2.5 → 0.5/2.5 = 20% > 5% → should warn
    let y_min = 0.5_f32;
    let y_max = 3.0_f32;
    let y_extent = y_max - y_min;
    let deviates = y_extent > 1e-6 && y_min.abs() > 0.05 * y_extent;
    assert!(deviates, "y_min=0.5 should trigger the >5% warning");

    // Mesh with y_min=0.01, y_max=3.0, extent=2.99 → 0.01/2.99 ≈ 0.3% < 5% → no warn
    let y_min = 0.01_f32;
    let y_extent = 3.0 - y_min;
    let deviates = y_extent > 1e-6 && y_min.abs() > 0.05 * y_extent;
    assert!(!deviates, "y_min=0.01 should not trigger warning");
}
```

- [ ] **Step 5: Commit**

```bash
git add src/terrain/scatter.rs
git commit -m "feat(tv22): compute mesh_height_max per LOD level with Y=0 base warning"
```

---

## Task 6: Rename SceneUniforms → ScatterBatchUniforms

**Files:**
- Modify: `src/render/mesh_instanced.rs`
- Modify: `src/shaders/mesh_instanced.wgsl`

This is a non-breaking rename step. No new fields yet.

- [ ] **Step 1: Rename in Rust**

In `src/render/mesh_instanced.rs`, rename `SceneUniforms` → `ScatterBatchUniforms` everywhere (struct definition at line 22, `impl Default`, all usages in the file).

- [ ] **Step 2: Rename in WGSL**

In `src/shaders/mesh_instanced.wgsl`, rename `SceneUniforms` → `ScatterBatchUniforms` (line 4).

- [ ] **Step 3: Run existing tests**

Run: `cargo test -p forge3d --lib render::mesh_instanced`
Expected: all existing tests PASS (rename is non-breaking)

- [ ] **Step 4: Commit**

```bash
git add src/render/mesh_instanced.rs src/shaders/mesh_instanced.wgsl
git commit -m "refactor(tv22): rename SceneUniforms to ScatterBatchUniforms"
```

---

## Task 7: Extend ScatterBatchUniforms + draw_batch_params Signature

**Files:**
- Modify: `src/render/mesh_instanced.rs`
- Modify: `src/terrain/renderer/scatter.rs` (lines 182-196)
- Modify: `src/viewer/terrain/scene/scatter.rs` (lines 88-102)
- Modify: `src/scene/render_paths/png.rs` (lines 134-148)

- [ ] **Step 1: Add wind fields to ScatterBatchUniforms**

In `src/render/mesh_instanced.rs`, extend the struct:

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

Update `Default` impl to zero-initialize the three new fields.

- [ ] **Step 2: Extend draw_batch_params signature**

Add three `[f32; 4]` parameters after `light_intensity`:

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
    wind_phase: [f32; 4],
    wind_vec_bounds: [f32; 4],
    wind_bend_fade: [f32; 4],
    vbuf: &'rp Buffer,
    ibuf: &'rp Buffer,
    instbuf: &'rp Buffer,
    index_count: u32,
    instance_count: u32,
)
```

Inside, populate `u.wind_phase`, `u.wind_vec_bounds`, `u.wind_bend_fade` from the new parameters.

- [ ] **Step 3: Update all callers with zero wind**

Update **all four** callsites of `draw_batch_params` to pass `[0.0; 4], [0.0; 4], [0.0; 4]` as the three new wind parameters:

1. `src/terrain/renderer/scatter.rs` (line 182)
2. `src/viewer/terrain/scene/scatter.rs` (line 88)
3. `src/scene/render_paths/png.rs` (line 134)
4. `src/render/mesh_instanced.rs` (line 663) — the internal `draw_batch_params_renders_pixels` test

Also update the `render()` method in `mesh_instanced.rs` if it populates uniforms (it does at line 381 — ensure the new fields are zeroed).

- [ ] **Step 4: Build**

Run: `cargo build -p forge3d --features enable-gpu-instancing`
Expected: compiles with no errors

- [ ] **Step 5: Run existing tests**

Run: `cargo test -p forge3d --lib render::mesh_instanced`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/render/mesh_instanced.rs src/terrain/renderer/scatter.rs src/viewer/terrain/scene/scatter.rs src/scene/render_paths/png.rs
git commit -m "feat(tv22): extend ScatterBatchUniforms and draw_batch_params with wind fields"
```

---

## Task 8: WGSL Vertex Shader Wind Deformation

**Files:**
- Modify: `src/shaders/mesh_instanced.wgsl`

- [ ] **Step 1: Extend WGSL struct**

```wgsl
struct ScatterBatchUniforms {
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
  color: vec4<f32>,
  light_dir_ws: vec4<f32>,
  wind_phase: vec4<f32>,
  wind_vec_bounds: vec4<f32>,
  wind_bend_fade: vec4<f32>,
}
```

- [ ] **Step 2: Add wind deformation to vs_main**

Replace the existing `vs_main`:

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

    // Wind direction in world space (for spatial phase variety)
    let wind_dir_ws = normalize((M * vec4<f32>(wind_local, 0.0)).xyz);

    // Deterministic sway + gust
    let spatial = dot(pos_ws.xyz, wind_dir_ws) * 0.1;
    let sway = sin(U.wind_phase.x + spatial) * (1.0 - U.wind_phase.w) * wind_amp;
    let gust = sin(U.wind_phase.y + spatial * 0.37) * U.wind_phase.z;  // 0.37: decorrelation

    // Displacement in local frame, transformed to world through M
    let wind_dir_local = wind_local / wind_amp;
    let disp_local = wind_dir_local * (sway + gust) * bend_weight;
    var disp_ws = (M * vec4<f32>(disp_local, 0.0)).xyz;

    // Distance fade (view-space distance)
    let fade_start = U.wind_bend_fade.z;
    let fade_end = U.wind_bend_fade.w;
    if (fade_end > fade_start) {
      let view_pos = U.view * pos_ws;
      let view_dist = length(view_pos.xyz);
      disp_ws *= 1.0 - smoothstep(fade_start, fade_end, view_dist);
    }

    pos_ws = vec4<f32>(pos_ws.xyz + disp_ws, 1.0);

    // Cheap normal tilt
    let tilt = length(disp_ws) * 0.3;
    let up_ws = normalize(in.i_m1.xyz);
    n_ws = normalize(n_ws + wind_dir_ws * tilt * max(dot(n_ws, up_ws), 0.0));
  }

  out.pos = U.proj * U.view * pos_ws;
  out.n_ws = n_ws;
  return out;
}
```

- [ ] **Step 3: Build**

Run: `cargo build -p forge3d --features enable-gpu-instancing`
Expected: compiles (shader is inlined via `include_str!`)

- [ ] **Step 4: Run existing render tests**

Run: `cargo test -p forge3d --lib render::mesh_instanced::tests`
Expected: PASS (zero wind = identical behavior)

- [ ] **Step 5: Commit**

```bash
git add src/shaders/mesh_instanced.wgsl
git commit -m "feat(tv22): add wind deformation to mesh_instanced.wgsl vertex shader"
```

---

## Task 9: Offscreen Scatter Path Integration

**Files:**
- Modify: `src/terrain/renderer/scatter.rs`

- [ ] **Step 1: Add time_seconds to ScatterRenderState**

Add `pub(super) time_seconds: f32` to the `ScatterRenderState` struct (line 12).

Update `build_scatter_render_state` in the same file to accept and pass through `time_seconds`.

- [ ] **Step 2: Call compute_wind_uniforms in render_scatter_pass**

In `render_scatter_pass()` (line 117), after the batch loop begins, compute wind uniforms per batch and inject `mesh_height_max` per draw:

```rust
for batch in &mut self.scatter_batches {
    let (batch_stats, draws) = batch.prepare_draws(
        device, queue, state.eye_contract, state.render_from_contract, state.instance_scale,
    )?;
    accumulate_frame_stats(&mut frame_stats, &batch_stats);

    // Compute batch-constant wind fields
    let base_wind = crate::terrain::scatter::compute_wind_uniforms(
        &batch.wind,
        state.time_seconds,
        0.0,  // placeholder, overridden per-draw
        state.instance_scale,
    );

    for draw in draws {
        let Some(instbuf) = batch.level_instbuf(draw.level_index) else { continue; };
        // Inject per-draw mesh_height_max
        let mut wind = base_wind;
        wind.wind_vec_bounds[3] = batch.level_mesh_height_max(draw.level_index);

        renderer.draw_batch_params(
            device, &mut pass, queue,
            state.view, state.proj,
            batch.color, state.light_dir, state.light_intensity,
            wind.wind_phase, wind.wind_vec_bounds, wind.wind_bend_fade,
            batch.level_vbuf(draw.level_index),
            batch.level_ibuf(draw.level_index),
            instbuf,
            batch.level_index_count(draw.level_index),
            draw.instance_count,
        );
    }
}
```

- [ ] **Step 3: Pass time_seconds from build_scatter_render_state callers**

Update `build_scatter_render_state` signature to accept `time_seconds: f32`:

```rust
pub(super) fn build_scatter_render_state(
    &self,
    params: &crate::terrain::render_params::TerrainRenderParams,
    decoded: &crate::terrain::render_params::DecodedTerrainSettings,
    heightmap_width: u32,
    heightmap_height: u32,
    view: glam::Mat4,
    proj: glam::Mat4,
    eye_render: glam::Vec3,
    time_seconds: f32,  // NEW
) -> ScatterRenderState {
    // ... existing body ...
    ScatterRenderState { /* ... existing fields ... */ time_seconds }
}
```

Update **both** callers:
1. `src/terrain/renderer/draw/mod.rs` (line 199) — pass `time_seconds` from the render call chain
2. `src/terrain/renderer/aov.rs` (line 582) — pass `time_seconds` from the AOV render call chain

Both callers receive `time_seconds` from the py_api kwarg (Task 10), but for now pass `0.0` to keep things compiling.

- [ ] **Step 4: Build**

Run: `cargo build -p forge3d --features enable-gpu-instancing`
Expected: compiles

- [ ] **Step 5: Commit**

```bash
git add src/terrain/renderer/scatter.rs src/terrain/renderer/core.rs src/terrain/renderer/draw/ src/terrain/renderer/aov.rs
git commit -m "feat(tv22): wire wind uniforms through offscreen scatter render path"
```

---

## Task 10: py_api.rs Wind Parsing + time_seconds

**Files:**
- Modify: `src/terrain/renderer/py_api.rs`

- [ ] **Step 1: Add wind parsing to set_scatter_batches**

In `set_scatter_batches()` (line 132), after parsing existing batch fields, add wind parsing:

```rust
let wind = match batch_dict
    .get_item("wind")
    .map_err(|e| PyRuntimeError::new_err(format!("batch {batch_index}: {e}")))?
    .filter(|v| !v.is_none())
{
    Some(wind_dict) => {
        let wind_dict = wind_dict.downcast::<PyDict>().map_err(|_| {
            PyRuntimeError::new_err(format!("batch {batch_index}: 'wind' must be a dict"))
        })?;
        crate::terrain::scatter::ScatterWindSettingsNative {
            enabled: wind_dict.get_item("enabled").ok().flatten()
                .and_then(|v| v.extract().ok()).unwrap_or(false),
            direction_deg: wind_dict.get_item("direction_deg").ok().flatten()
                .and_then(|v| v.extract().ok()).unwrap_or(0.0),
            speed: wind_dict.get_item("speed").ok().flatten()
                .and_then(|v| v.extract().ok()).unwrap_or(1.0),
            amplitude: wind_dict.get_item("amplitude").ok().flatten()
                .and_then(|v| v.extract().ok()).unwrap_or(0.0),
            rigidity: wind_dict.get_item("rigidity").ok().flatten()
                .and_then(|v| v.extract().ok()).unwrap_or(0.5),
            bend_start: wind_dict.get_item("bend_start").ok().flatten()
                .and_then(|v| v.extract().ok()).unwrap_or(0.0),
            bend_extent: wind_dict.get_item("bend_extent").ok().flatten()
                .and_then(|v| v.extract().ok()).unwrap_or(1.0),
            gust_strength: wind_dict.get_item("gust_strength").ok().flatten()
                .and_then(|v| v.extract().ok()).unwrap_or(0.0),
            gust_frequency: wind_dict.get_item("gust_frequency").ok().flatten()
                .and_then(|v| v.extract().ok()).unwrap_or(0.3),
            fade_start: wind_dict.get_item("fade_start").ok().flatten()
                .and_then(|v| v.extract().ok()).unwrap_or(0.0),
            fade_end: wind_dict.get_item("fade_end").ok().flatten()
                .and_then(|v| v.extract().ok()).unwrap_or(0.0),
        }
    }
    None => crate::terrain::scatter::ScatterWindSettingsNative::default(),
};
```

Add `wind` to the `TerrainScatterUploadBatch` construction.

- [ ] **Step 2: Add time_seconds kwarg to render_terrain_pbr_pom**

In `render_terrain_pbr_pom` (line 60), add `time_seconds: f32 = 0.0` to the pyo3 signature:

```rust
#[pyo3(signature = (material_set, env_maps, params, heightmap, target=None, water_mask=None, time_seconds=0.0))]
pub fn render_terrain_pbr_pom<'py>(
    &mut self,
    py: Python<'py>,
    material_set: &crate::render::material_set::MaterialSet,
    env_maps: &crate::lighting::ibl_wrapper::IBL,
    params: &render_params::TerrainRenderParams,
    heightmap: PyReadonlyArray2<'py, f32>,
    target: Option<&Bound<'_, PyAny>>,
    water_mask: Option<PyReadonlyArray2<'py, f32>>,
    time_seconds: f32,
) -> PyResult<Py<crate::Frame>> {
```

Thread `time_seconds` through to the internal render call.

- [ ] **Step 3: Do the same for render_with_aov**

Add `time_seconds: f32 = 0.0` kwarg to `render_with_aov`.

- [ ] **Step 4: Build**

Run: `cargo build -p forge3d --features enable-gpu-instancing,extension-module`
Expected: compiles

- [ ] **Step 5: Commit**

```bash
git add src/terrain/renderer/py_api.rs
git commit -m "feat(tv22): parse wind dict and add time_seconds kwarg to render methods"
```

---

## Task 11: Viewer IPC Wiring

**Files:**
- Modify: `src/viewer/ipc/protocol/payloads.rs`
- Modify: `src/viewer/viewer_enums/config.rs`
- Modify: `src/viewer/ipc/protocol/translate/terrain.rs`

- [ ] **Step 1: Add wind to IPC payload**

In `src/viewer/ipc/protocol/payloads.rs`, add wind fields to `IpcTerrainScatterBatch` (line 200):

```rust
pub struct IpcTerrainScatterBatch {
    pub name: Option<String>,
    pub color: Option<[f32; 4]>,
    pub max_draw_distance: Option<f32>,
    pub transforms: Vec<[f32; 16]>,
    pub levels: Vec<IpcTerrainScatterLevel>,
    pub wind: Option<IpcScatterWind>,  // NEW
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct IpcScatterWind {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub direction_deg: f32,
    #[serde(default = "default_speed")]
    pub speed: f32,
    #[serde(default)]
    pub amplitude: f32,
    #[serde(default = "default_rigidity")]
    pub rigidity: f32,
    #[serde(default)]
    pub bend_start: f32,
    #[serde(default = "default_bend_extent")]
    pub bend_extent: f32,
    #[serde(default)]
    pub gust_strength: f32,
    #[serde(default = "default_gust_frequency")]
    pub gust_frequency: f32,
    #[serde(default)]
    pub fade_start: f32,
    #[serde(default)]
    pub fade_end: f32,
}
```

Add the necessary default helper functions (`default_speed() -> f32 { 1.0 }`, etc.).

- [ ] **Step 2: Add wind to viewer config**

In `src/viewer/viewer_enums/config.rs`, add `pub wind: ScatterWindSettingsNative` to `ViewerTerrainScatterBatchConfig` (line 160), importing from `crate::terrain::scatter`.

- [ ] **Step 3: Map wind in IPC translation**

In `src/viewer/ipc/protocol/translate/terrain.rs`, in `map_terrain_scatter_batch()` (line 254), convert `IpcScatterWind` → `ScatterWindSettingsNative`:

```rust
let wind = batch.wind.map(|w| ScatterWindSettingsNative {
    enabled: w.enabled,
    direction_deg: w.direction_deg,
    speed: w.speed,
    amplitude: w.amplitude,
    rigidity: w.rigidity,
    bend_start: w.bend_start,
    bend_extent: w.bend_extent,
    gust_strength: w.gust_strength,
    gust_frequency: w.gust_frequency,
    fade_start: w.fade_start,
    fade_end: w.fade_end,
}).unwrap_or_default();
```

- [ ] **Step 4: Build**

Run: `cargo build -p forge3d --features enable-gpu-instancing`
Expected: compiles

- [ ] **Step 5: Commit**

```bash
git add src/viewer/ipc/protocol/payloads.rs src/viewer/viewer_enums/config.rs src/viewer/ipc/protocol/translate/terrain.rs
git commit -m "feat(tv22): add wind to viewer IPC payload, config, and translation"
```

---

## Task 12: Viewer Scatter Render Path

**Files:**
- Modify: `src/viewer/terrain/scene/scatter.rs`

- [ ] **Step 1: Add time accumulation**

Add `elapsed_time: f32` field to whatever state struct the viewer scatter path uses (or pass it through from the viewer frame loop's accumulated time). The viewer frame loop in `src/viewer/event_loop/runner.rs` already computes `dt` — accumulate into a total elapsed time and thread it to the scatter render.

- [ ] **Step 2: Pass wind to draw_batch_params**

In the viewer scatter render function (line 88), compute wind uniforms per batch and inject per-draw `mesh_height_max`, matching the offscreen pattern from Task 9:

```rust
let base_wind = crate::terrain::scatter::compute_wind_uniforms(
    &batch.wind, elapsed_time, 0.0, 1.0,  // viewer uses identity scale
);
// Per-draw: override mesh_height_max
let mut wind = base_wind;
wind.wind_vec_bounds[3] = batch.level_mesh_height_max(draw.level_index);

renderer.draw_batch_params(
    device, &mut pass, queue,
    view, proj, batch.color, light_dir, light_intensity,
    wind.wind_phase, wind.wind_vec_bounds, wind.wind_bend_fade,
    vbuf, ibuf, instbuf, index_count, instance_count,
);
```

- [ ] **Step 3: Build**

Run: `cargo build -p forge3d --features enable-gpu-instancing`
Expected: compiles

- [ ] **Step 4: Commit**

```bash
git add src/viewer/terrain/scene/scatter.rs src/viewer/event_loop/
git commit -m "feat(tv22): wire wind uniforms through viewer scatter render path"
```

---

## Task 13: Viewer IPC Tests

**Files:**
- Modify: `tests/test_viewer_ipc.py`

- [ ] **Step 1: Add wind to scatter IPC format test**

In the existing `test_set_terrain_scatter_format` test (line 169), add a `"wind"` key to the batch payload:

```python
"wind": {
    "enabled": True,
    "direction_deg": 45.0,
    "speed": 1.0,
    "amplitude": 2.0,
    "rigidity": 0.3,
    "bend_start": 0.0,
    "bend_extent": 1.0,
    "gust_strength": 0.5,
    "gust_frequency": 0.3,
    "fade_start": 100.0,
    "fade_end": 200.0,
},
```

- [ ] **Step 2: Run IPC tests**

Run: `python -m pytest tests/test_viewer_ipc.py -k scatter -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_viewer_ipc.py
git commit -m "test(tv22): add wind field to viewer scatter IPC tests"
```

---

## Task 14: Integration Tests

**Files:**
- Create: `tests/test_tv22_scatter_wind.py`

- [ ] **Step 1: Write integration test file**

```python
"""TV22 Scatter Wind Animation — integration tests."""
from __future__ import annotations

import numpy as np
import pytest

f3d = pytest.importorskip("forge3d")
from forge3d import terrain_scatter as ts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trivial_mesh():
    """Simple tree-like mesh with y=0 base and y=2 tip."""
    positions = np.array([
        [0, 0, 0], [0.5, 0, 0], [-0.5, 0, 0],
        [0, 2, 0], [0.3, 1.5, 0], [-0.3, 1.5, 0],
    ], dtype=np.float32)
    normals = np.array([
        [0, 0, 1], [0, 0, 1], [0, 0, 1],
        [0, 0, 1], [0, 0, 1], [0, 0, 1],
    ], dtype=np.float32)
    indices = np.array([0, 1, 3, 1, 4, 3, 0, 3, 2, 2, 3, 5], dtype=np.uint32)
    return f3d.geometry.MeshBuffers(positions=positions, normals=normals, indices=indices)


def _make_scatter_source():
    heightmap = np.ones((64, 64), dtype=np.float32) * 100.0
    return ts.TerrainScatterSource(heightmap, z_scale=1.0)


def _place_instances(source, count=20, seed=42):
    return ts.seeded_random_transforms(
        source, count=count, seed=seed,
        scale_range=(0.5, 1.5),
    )


def _render_frame(renderer, material_set, ibl, params, heightmap, batches, time_seconds=0.0):
    ts.apply_to_renderer(renderer, batches)
    frame = renderer.render_terrain_pbr_pom(
        material_set, ibl, params, heightmap, time_seconds=time_seconds,
    )
    return np.array(frame)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWindNoOp:
    """Wind disabled or zero-amplitude must produce identical output to static baseline."""

    @pytest.fixture
    def setup(self):
        # Set up renderer, material_set, ibl, params, heightmap
        # (Follow pattern from test_terrain_scatter.py::TestNativeScatterIntegration)
        pytest.skip("requires GPU — run with enable-gpu-instancing")

    def test_disabled_wind_matches_static(self, setup):
        renderer, ms, ibl, params, hm = setup
        source = _make_scatter_source()
        transforms = _place_instances(source)
        batch_static = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_trivial_mesh())],
            transforms=transforms,
        )
        batch_disabled = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_trivial_mesh())],
            transforms=transforms,
            wind=ts.ScatterWindSettings(enabled=False, amplitude=5.0),
        )
        frame_static = _render_frame(renderer, ms, ibl, params, hm, [batch_static])
        frame_disabled = _render_frame(renderer, ms, ibl, params, hm, [batch_disabled])
        np.testing.assert_array_equal(frame_static, frame_disabled)

    def test_zero_amplitude_matches_static(self, setup):
        renderer, ms, ibl, params, hm = setup
        source = _make_scatter_source()
        transforms = _place_instances(source)
        batch_static = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_trivial_mesh())],
            transforms=transforms,
        )
        batch_zero = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_trivial_mesh())],
            transforms=transforms,
            wind=ts.ScatterWindSettings(enabled=True, amplitude=0.0),
        )
        frame_static = _render_frame(renderer, ms, ibl, params, hm, [batch_static])
        frame_zero = _render_frame(renderer, ms, ibl, params, hm, [batch_zero])
        np.testing.assert_array_equal(frame_static, frame_zero)

    def test_rigidity_one_matches_static(self, setup):
        renderer, ms, ibl, params, hm = setup
        source = _make_scatter_source()
        transforms = _place_instances(source)
        batch_static = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_trivial_mesh())],
            transforms=transforms,
        )
        batch_rigid = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_trivial_mesh())],
            transforms=transforms,
            wind=ts.ScatterWindSettings(enabled=True, amplitude=3.0, rigidity=1.0),
        )
        frame_static = _render_frame(renderer, ms, ibl, params, hm, [batch_static])
        frame_rigid = _render_frame(renderer, ms, ibl, params, hm, [batch_rigid])
        np.testing.assert_array_equal(frame_static, frame_rigid)


class TestWindAnimation:
    """Wind enabled must produce visible, deterministic animation."""

    @pytest.fixture
    def setup(self):
        pytest.skip("requires GPU — run with enable-gpu-instancing")

    def test_different_times_differ(self, setup):
        renderer, ms, ibl, params, hm = setup
        source = _make_scatter_source()
        transforms = _place_instances(source)
        wind = ts.ScatterWindSettings(enabled=True, amplitude=2.0, speed=1.0)
        batch = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_trivial_mesh())],
            transforms=transforms, wind=wind,
        )
        f0 = _render_frame(renderer, ms, ibl, params, hm, [batch], time_seconds=0.0)
        f1 = _render_frame(renderer, ms, ibl, params, hm, [batch], time_seconds=1.0)
        assert not np.array_equal(f0, f1), "wind at different times should differ"

    def test_same_time_is_deterministic(self, setup):
        renderer, ms, ibl, params, hm = setup
        source = _make_scatter_source()
        transforms = _place_instances(source)
        wind = ts.ScatterWindSettings(enabled=True, amplitude=2.0, speed=1.0)
        batch = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_trivial_mesh())],
            transforms=transforms, wind=wind,
        )
        f1 = _render_frame(renderer, ms, ibl, params, hm, [batch], time_seconds=0.5)
        f2 = _render_frame(renderer, ms, ibl, params, hm, [batch], time_seconds=0.5)
        np.testing.assert_array_equal(f1, f2)

    def test_bend_start_affects_output(self, setup):
        renderer, ms, ibl, params, hm = setup
        source = _make_scatter_source()
        transforms = _place_instances(source)
        batch_low = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_trivial_mesh())],
            transforms=transforms,
            wind=ts.ScatterWindSettings(enabled=True, amplitude=2.0, bend_start=0.0),
        )
        batch_high = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_trivial_mesh())],
            transforms=transforms,
            wind=ts.ScatterWindSettings(enabled=True, amplitude=2.0, bend_start=0.8),
        )
        f_low = _render_frame(renderer, ms, ibl, params, hm, [batch_low], time_seconds=0.5)
        f_high = _render_frame(renderer, ms, ibl, params, hm, [batch_high], time_seconds=0.5)
        assert not np.array_equal(f_low, f_high), "different bend_start should differ"

    def test_gust_affects_output(self, setup):
        renderer, ms, ibl, params, hm = setup
        source = _make_scatter_source()
        transforms = _place_instances(source)
        batch_no_gust = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_trivial_mesh())],
            transforms=transforms,
            wind=ts.ScatterWindSettings(enabled=True, amplitude=2.0, gust_strength=0.0),
        )
        batch_gust = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_trivial_mesh())],
            transforms=transforms,
            wind=ts.ScatterWindSettings(enabled=True, amplitude=2.0, gust_strength=1.5),
        )
        f_no = _render_frame(renderer, ms, ibl, params, hm, [batch_no_gust], time_seconds=0.5)
        f_yes = _render_frame(renderer, ms, ibl, params, hm, [batch_gust], time_seconds=0.5)
        assert not np.array_equal(f_no, f_yes), "gust should affect output"


class TestWindFade:
    """Distance fade must suppress wind at range."""

    @pytest.fixture
    def setup(self):
        pytest.skip("requires GPU — run with enable-gpu-instancing")

    def test_far_instances_match_static_with_fade(self, setup):
        """Place instances far from camera with tight fade; verify wind suppressed."""
        renderer, ms, ibl, params, hm = setup
        source = _make_scatter_source()
        # Place instances at known distant positions by using a fixed transform
        # at the far corner of the terrain
        far_transform = ts.make_transform_row_major(
            (source.terrain_width * 0.95, 0.0, source.terrain_width * 0.95),
            scale=1.0,
        ).reshape(1, 16)
        batch_static = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_trivial_mesh())],
            transforms=far_transform,
        )
        # Wind with fade that ends well before the instance distance
        batch_faded = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_trivial_mesh())],
            transforms=far_transform,
            wind=ts.ScatterWindSettings(
                enabled=True, amplitude=3.0,
                fade_start=1.0, fade_end=2.0,  # fade ends at 2 contract units
            ),
        )
        f_static = _render_frame(renderer, ms, ibl, params, hm, [batch_static])
        f_faded = _render_frame(renderer, ms, ibl, params, hm, [batch_faded], time_seconds=1.0)
        np.testing.assert_array_equal(f_static, f_faded,
            "wind should be fully suppressed beyond fade_end")
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_tv22_scatter_wind.py -v`
Expected: Python unit tests PASS; GPU tests skip without GPU feature

- [ ] **Step 3: Commit**

```bash
git add tests/test_tv22_scatter_wind.py
git commit -m "test(tv22): add wind integration tests for no-op, animation, controls, and fade"
```

---

## Task 15: Example

**Files:**
- Create: `examples/terrain_tv22_scatter_wind_demo.py`

- [ ] **Step 1: Write example**

Follow the pattern from `examples/terrain_tv3_scatter_demo.py`. Use `Mount_Fuji_30m.tif` DEM. Create scatter batches with wind enabled, render multiple frames at different `time_seconds` values, save PNGs.

Key elements:
- Load DEM from `assets/tif/Mount_Fuji_30m.tif`
- Create `TerrainScatterSource` from heightmap
- Place vegetation with `grid_jitter_transforms` using slope/elevation filters
- Create batch with `ScatterWindSettings(enabled=True, amplitude=1.5, speed=0.8, gust_strength=0.5)`
- Render frames at `time_seconds` = 0.0, 0.5, 1.0, 1.5
- Save each frame as PNG to `examples/out/tv22_wind_t{time}.png`
- Print frame dimensions and non-black pixel counts

- [ ] **Step 2: Run example**

Run: `python examples/terrain_tv22_scatter_wind_demo.py`
Expected: PNGs saved, visible pixel output reported, different frames show different vegetation positions

- [ ] **Step 3: Verify output images differ between time steps**

Visually inspect or compare the PNGs. Frames at different `time_seconds` should show different vegetation positions.

- [ ] **Step 4: Commit**

```bash
git add examples/terrain_tv22_scatter_wind_demo.py
git commit -m "feat(tv22): add scatter wind animation example with Mount Fuji DEM"
```

---

## Task 16: Documentation

**Files:**
- Create or modify: `docs/tv22-scatter-wind-animation.md`

- [ ] **Step 1: Write feature documentation**

Document:
- What TV22 achieves (GPU wind animation for scatter vegetation)
- The `ScatterWindSettings` API with all fields and their meanings
- The `time_seconds` parameter on render methods
- How to enable wind on a scatter batch (code example)
- The mesh Y=0 base convention
- Per-batch vs global wind controls
- Distance fade behavior
- Accepted limitations (spatial phase parity, fade approximation)

- [ ] **Step 2: Update the epics document**

In `docs/plans/2026-03-16-terrain-viz-epics.md`, update the TV22 row in the status matrix from "Missing" to "Implemented" with a brief note about what shipped.

- [ ] **Step 3: Commit**

```bash
git add docs/
git commit -m "docs(tv22): add feature documentation and update epic status"
```

---

## Execution Order

Tasks 1-3 are Python-only and can run without Rust compilation.
Tasks 4-5 are Rust-only core logic.
Task 6 is a non-breaking rename.
Task 7 is the signature change that touches all callers.
Task 8 is the shader change.
Tasks 9-10 wire the offscreen path.
Tasks 11-12 wire the viewer path.
Tasks 13-15 are tests and example.
Task 16 is documentation.

**Critical path:** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 14 → 15

Tasks 11-12 (viewer) can run in parallel with 9-10 (offscreen) after Task 7.
Task 13 (viewer IPC tests) depends on 11.
Task 16 can run at any point after 15.
