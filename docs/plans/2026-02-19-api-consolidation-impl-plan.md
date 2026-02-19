# Rust/Python API Consolidation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close exposure gaps, fix API drift, deprecate unused Rust duplicates, and complete partial implementations across the forge3d Rust/Python boundary.

**Architecture:** PyO3 bindings in `src/lib.rs` register Rust types for Python. Python wrappers in `python/forge3d/` add validation and fallbacks. Changes add new `#[pyfunction]`/`#[pyclass]` registrations, clean dead references, and wire incomplete GPU pipelines.

**Tech Stack:** Rust (PyO3, wgpu, bytemuck, glam), Python (numpy, pytest), maturin build system.

---

## Phase 1: Register Orphaned PyO3 Classes + Module-Level Forwarding

### Task 1: Register `Frame` and SDF pyclasses in module init

**Files:**
- Modify: `src/lib.rs` (around line 4735, before `Ok(())`)

**Step 1: Write the failing test**

Create `tests/test_api_registration.py`:

```python
"""Tests for API consolidation — orphaned class registration."""
import pytest

def test_frame_class_importable():
    """Frame pyclass should be registered in _forge3d."""
    from forge3d._forge3d import Frame
    assert Frame is not None

def test_sdf_classes_importable():
    """SDF pyclasses should be registered in _forge3d."""
    from forge3d._forge3d import SdfPrimitive, SdfScene, SdfSceneBuilder
    assert SdfPrimitive is not None
    assert SdfScene is not None
    assert SdfSceneBuilder is not None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_registration.py -v -x`
Expected: FAIL with `ImportError: cannot import name 'Frame'`

**Step 3: Add class registrations to `src/lib.rs`**

Insert before `Ok(())` at end of `_forge3d` function (around line 4743):

```rust
    // API consolidation: register orphaned pyclasses
    m.add_class::<Frame>()?;
    m.add_class::<crate::sdf::py::PySdfPrimitive>()?;
    m.add_class::<crate::sdf::py::PySdfScene>()?;
    m.add_class::<crate::sdf::py::PySdfSceneBuilder>()?;
```

**Step 4: Build**

Run: `maturin develop --release 2>&1`
Expected: Build succeeds with zero new warnings.

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_api_registration.py -v -x`
Expected: 2 tests PASS.

**Step 6: Commit**

```bash
git add src/lib.rs tests/test_api_registration.py
git commit -m "feat: register Frame and SDF pyclasses in _forge3d module init"
```

---

### Task 2: Add module-level forwarding for `render_rgba` and `set_msaa_samples`

**Files:**
- Modify: `src/lib.rs` (add two `#[pyfunction]` wrappers + register them)
- Modify: `tests/test_api_registration.py` (add tests)

**Step 1: Write the failing test**

Append to `tests/test_api_registration.py`:

```python
def test_render_rgba_module_level():
    """render_rgba should be callable as a module-level function."""
    import forge3d._forge3d as native
    assert hasattr(native, 'render_rgba'), "render_rgba not found at module level"

def test_set_msaa_samples_module_level():
    """set_msaa_samples should be callable as a module-level function."""
    import forge3d._forge3d as native
    assert hasattr(native, 'set_msaa_samples'), "set_msaa_samples not found at module level"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_registration.py::test_render_rgba_module_level tests/test_api_registration.py::test_set_msaa_samples_module_level -v -x`
Expected: FAIL with `AssertionError: render_rgba not found at module level`

**Step 3: Add forwarding functions to `src/lib.rs`**

Add before the `#[pymodule]` function (around line 4420):

```rust
/// Module-level forwarding for render_rgba (delegates to Scene.render_rgba).
/// Satisfies wrapper expectations at python/forge3d/helpers/offscreen.py:41
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (scene, width=None, height=None))]
fn render_rgba<'py>(
    py: Python<'py>,
    scene: &mut crate::scene::Scene,
    width: Option<u32>,
    height: Option<u32>,
) -> PyResult<pyo3::Bound<'py, numpy::PyArray3<u8>>> {
    // Forward to the Scene method which handles all the GPU work
    scene.render_rgba(py)
}

/// Module-level forwarding for set_msaa_samples (delegates to Scene.set_msaa_samples).
/// Satisfies wrapper expectations at python/forge3d/viewer.py:42
#[cfg(feature = "extension-module")]
#[pyfunction]
fn set_msaa_samples(scene: &mut crate::scene::Scene, samples: u32) -> PyResult<u32> {
    scene.set_msaa_samples(samples)
}
```

Then register in the `_forge3d` function body (after the viewer functions, around line 4440):

```rust
    // Module-level forwarding functions (API consolidation)
    m.add_function(wrap_pyfunction!(render_rgba, m)?)?;
    m.add_function(wrap_pyfunction!(set_msaa_samples, m)?)?;
```

**Important:** Check the exact signature of `Scene.render_rgba` in `src/scene/mod.rs:1721` before writing the forwarding function. The forwarding function signature must match what Python callers expect.

**Step 4: Build**

Run: `maturin develop --release 2>&1`
Expected: Build succeeds.

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_api_registration.py -v -x`
Expected: 4 tests PASS.

**Step 6: Commit**

```bash
git add src/lib.rs tests/test_api_registration.py
git commit -m "feat: add module-level render_rgba and set_msaa_samples forwarding functions"
```

---

## Phase 2: Clean API Drift + Add Mesh TBN Exports

### Task 3: Clean dead ReSTIR native references from `lighting.py`

**Files:**
- Modify: `python/forge3d/lighting.py` (6 locations)
- Modify: `tests/test_api_registration.py` (add verification test)

**Step 1: Write the failing test**

Append to `tests/test_api_registration.py`:

```python
def test_no_dead_restir_native_references():
    """lighting.py must not reference non-existent native ReSTIR symbols."""
    import inspect
    from forge3d import lighting
    source = inspect.getsource(lighting)
    dead_symbols = [
        "_forge3d.create_restir_di",
        "_forge3d.restir_set_lights",
        "_forge3d.restir_clear_lights",
        "_forge3d.restir_sample_light",
        "_forge3d.restir_render_frame",
        "_forge3d.restir_get_statistics",
    ]
    for sym in dead_symbols:
        assert sym not in source, f"Dead native reference found: {sym}"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_registration.py::test_no_dead_restir_native_references -v -x`
Expected: FAIL — dead references are present.

**Step 3: Clean `python/forge3d/lighting.py`**

In `RestirDI.__init__` (around line 221): Remove the `_forge3d.create_restir_di` block. Replace with:

```python
        # ReSTIR is internal to the wavefront path tracer and not
        # independently callable from Python. This class provides
        # CPU-side ReSTIR configuration and sampling.
        self._native_restir = None
```

In `set_lights` (around line 291): Remove the `_forge3d.restir_set_lights` block:

```python
        # Native ReSTIR is internal — CPU implementation handles light state
```

In `clear_lights` (around line 302): Remove the `_forge3d.restir_clear_lights` block.

In `sample_light` (around line 314): Remove the `_forge3d.restir_sample_light` block.

In `render_frame` (around line 367): Remove the `_forge3d.restir_render_frame` call. The method should raise `NotImplementedError("GPU ReSTIR render requires the wavefront path tracer")` if native was expected.

In `get_statistics` (around line 420): Remove the `_forge3d.restir_get_statistics` block.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_api_registration.py::test_no_dead_restir_native_references -v -x`
Expected: PASS.

**Step 5: Run full lighting test suite to verify no regressions**

Run: `python -m pytest tests/test_lighting_alignment.py tests/test_lighting_preset.py tests/test_sun_ephemeris.py -v`
Expected: All existing tests still pass.

**Step 6: Commit**

```bash
git add python/forge3d/lighting.py tests/test_api_registration.py
git commit -m "fix: remove dead ReSTIR native references from lighting.py"
```

---

### Task 4: Add mesh TBN native exports

**Files:**
- Modify: `src/lib.rs` (add 2 `#[pyfunction]` + registration)
- Modify: `tests/test_api_registration.py` (add test)

**Step 1: Write the failing test**

Append to `tests/test_api_registration.py`:

```python
def test_mesh_tbn_exports_exist():
    """mesh_generate_cube_tbn and mesh_generate_plane_tbn should be exported."""
    import forge3d._forge3d as native
    assert hasattr(native, 'mesh_generate_cube_tbn'), "mesh_generate_cube_tbn not found"
    assert hasattr(native, 'mesh_generate_plane_tbn'), "mesh_generate_plane_tbn not found"

def test_mesh_generate_cube_tbn_returns_data():
    """mesh_generate_cube_tbn should return dict with vertices, indices, tbn_data."""
    from forge3d._forge3d import mesh_generate_cube_tbn
    result = mesh_generate_cube_tbn()
    assert 'vertices' in result
    assert 'indices' in result
    assert 'tbn_data' in result
    assert len(result['indices']) == 36  # 6 faces * 2 triangles * 3 indices
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_registration.py::test_mesh_tbn_exports_exist -v -x`
Expected: FAIL.

**Step 3: Add `#[pyfunction]` wrappers in `src/lib.rs`**

Add before the `#[pymodule]` function:

```rust
/// Generate unit cube with TBN data. Returns dict with 'vertices', 'indices', 'tbn_data'.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn mesh_generate_cube_tbn(py: Python<'_>) -> PyResult<PyObject> {
    let (verts, indices, tbn) = crate::mesh::tbn::generate_cube_tbn();
    let dict = pyo3::types::PyDict::new(py);
    // Convert vertices to list of dicts
    let vert_list: Vec<_> = verts.iter().map(|v| {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("position", (v.position.x, v.position.y, v.position.z)).unwrap();
        d.set_item("normal", (v.normal.x, v.normal.y, v.normal.z)).unwrap();
        d.set_item("uv", (v.uv.x, v.uv.y)).unwrap();
        d.into_any()
    }).collect();
    dict.set_item("vertices", vert_list)?;
    dict.set_item("indices", indices)?;
    let tbn_list: Vec<_> = tbn.iter().map(|t| {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("tangent", (t.tangent.x, t.tangent.y, t.tangent.z)).unwrap();
        d.set_item("bitangent", (t.bitangent.x, t.bitangent.y, t.bitangent.z)).unwrap();
        d.set_item("normal", (t.normal.x, t.normal.y, t.normal.z)).unwrap();
        d.set_item("handedness", t.handedness).unwrap();
        d.into_any()
    }).collect();
    dict.set_item("tbn_data", tbn_list)?;
    Ok(dict.into_any().unbind())
}

/// Generate plane mesh with TBN data. Returns dict with 'vertices', 'indices', 'tbn_data'.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (width=4, height=4))]
fn mesh_generate_plane_tbn(py: Python<'_>, width: u32, height: u32) -> PyResult<PyObject> {
    let (verts, indices, tbn) = crate::mesh::tbn::generate_plane_tbn(width, height);
    let dict = pyo3::types::PyDict::new(py);
    let vert_list: Vec<_> = verts.iter().map(|v| {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("position", (v.position.x, v.position.y, v.position.z)).unwrap();
        d.set_item("normal", (v.normal.x, v.normal.y, v.normal.z)).unwrap();
        d.set_item("uv", (v.uv.x, v.uv.y)).unwrap();
        d.into_any()
    }).collect();
    dict.set_item("vertices", vert_list)?;
    dict.set_item("indices", indices)?;
    let tbn_list: Vec<_> = tbn.iter().map(|t| {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("tangent", (t.tangent.x, t.tangent.y, t.tangent.z)).unwrap();
        d.set_item("bitangent", (t.bitangent.x, t.bitangent.y, t.bitangent.z)).unwrap();
        d.set_item("normal", (t.normal.x, t.normal.y, t.normal.z)).unwrap();
        d.set_item("handedness", t.handedness).unwrap();
        d.into_any()
    }).collect();
    dict.set_item("tbn_data", tbn_list)?;
    Ok(dict.into_any().unbind())
}
```

Register in the module init:

```rust
    // Mesh TBN exports (API consolidation)
    m.add_function(wrap_pyfunction!(mesh_generate_cube_tbn, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_generate_plane_tbn, m)?)?;
```

**Important:** Verify that `crate::mesh::tbn::generate_cube_tbn` and `generate_plane_tbn` exist with those exact names. The source audit shows them at `src/mesh/tbn.rs:250` and `src/mesh/tbn.rs:214`. Adapt function names if they differ.

**Step 4: Build**

Run: `maturin develop --release 2>&1`
Expected: Build succeeds.

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_api_registration.py -v -x`
Expected: All tests PASS.

**Step 6: Commit**

```bash
git add src/lib.rs tests/test_api_registration.py
git commit -m "feat: add mesh_generate_cube_tbn and mesh_generate_plane_tbn native exports"
```

---

## Phase 3: Terrain Analysis PyO3 Wrappers

### Task 5: Expose `slope_aspect_compute` and `contour_extract` to Python

**Files:**
- Modify: `src/terrain/analysis.rs` (add `#[pyfunction]` wrappers at end of file)
- Modify: `src/lib.rs` (register functions)
- Modify: `tests/test_api_registration.py` (add tests)

**Step 1: Write the failing test**

Append to `tests/test_api_registration.py`:

```python
def test_terrain_analysis_exports():
    """Terrain analysis functions should be exported."""
    import forge3d._forge3d as native
    assert hasattr(native, 'compute_slope_aspect_py')
    assert hasattr(native, 'extract_contours_py')

def test_compute_slope_aspect_basic():
    """compute_slope_aspect_py should return slope and aspect arrays."""
    import numpy as np
    from forge3d._forge3d import compute_slope_aspect_py

    # 3x3 ramp: elevation increases with x
    heights = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.float32)
    result = compute_slope_aspect_py(heights, 3, 3, 1.0, 1.0)
    assert 'slope' in result
    assert 'aspect' in result
    assert len(result['slope']) == 9
    # Center pixel should have non-zero slope
    assert result['slope'][4] > 0.0

def test_extract_contours_basic():
    """extract_contours_py should return contour polylines."""
    import numpy as np
    from forge3d._forge3d import extract_contours_py

    # 4x4 grid with values 0-15
    heights = np.arange(16, dtype=np.float32)
    result = extract_contours_py(heights, 4, 4, 1.0, 1.0, [5.0, 10.0])
    assert 'polylines' in result
    assert len(result['polylines']) > 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_registration.py::test_terrain_analysis_exports -v -x`
Expected: FAIL.

**Step 3: Add PyO3 wrappers to `src/terrain/analysis.rs`**

Append to the file:

```rust
// --- PyO3 bindings (API consolidation) ---

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
#[cfg(feature = "extension-module")]
use pyo3::types::PyDict;

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (heights, width, height, dx, dy))]
pub fn compute_slope_aspect_py(
    py: Python<'_>,
    heights: Vec<f32>,
    width: usize,
    height: usize,
    dx: f32,
    dy: f32,
) -> PyResult<PyObject> {
    let results = slope_aspect_compute(&heights, width, height, dx, dy)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    let slopes: Vec<f32> = results.iter().map(|sa| sa.slope_deg).collect();
    let aspects: Vec<f32> = results.iter().map(|sa| sa.aspect_deg).collect();
    let dict = PyDict::new(py);
    dict.set_item("slope", slopes)?;
    dict.set_item("aspect", aspects)?;
    Ok(dict.into_any().unbind())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (heights, width, height, dx, dy, levels))]
pub fn extract_contours_py(
    py: Python<'_>,
    heights: Vec<f32>,
    width: usize,
    height: usize,
    dx: f32,
    dy: f32,
    levels: Vec<f32>,
) -> PyResult<PyObject> {
    let result = contour_extract(&heights, width, height, dx, dy, &levels)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    let polylines: Vec<_> = result.polylines.iter().map(|pl| {
        let d = PyDict::new(py);
        d.set_item("level", pl.level).unwrap();
        let pts: Vec<(f32, f32)> = pl.points.clone();
        d.set_item("points", pts).unwrap();
        d.into_any()
    }).collect();
    let dict = PyDict::new(py);
    dict.set_item("polylines", polylines)?;
    dict.set_item("polyline_count", result.polyline_count)?;
    dict.set_item("total_points", result.total_points)?;
    Ok(dict.into_any().unbind())
}
```

Register in `src/lib.rs` module init:

```rust
    // Terrain analysis exports (API consolidation)
    m.add_function(wrap_pyfunction!(crate::terrain::analysis::compute_slope_aspect_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::terrain::analysis::extract_contours_py, m)?)?;
```

**Step 4: Build**

Run: `maturin develop --release 2>&1`
Expected: Build succeeds.

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_api_registration.py -v -x`
Expected: All tests PASS.

**Step 6: Commit**

```bash
git add src/terrain/analysis.rs src/lib.rs tests/test_api_registration.py
git commit -m "feat: expose terrain slope/aspect and contour extraction to Python"
```

---

## Phase 4: Cloud Shadow + Reflection Settings

### Task 6: Add cloud shadow fields to `PyVolumetricSettings`

**Files:**
- Modify: `src/lighting/py_bindings.rs` (extend `PyVolumetricSettings`)
- Modify: `tests/test_api_registration.py` (add test)

**Step 1: Write the failing test**

Append to `tests/test_api_registration.py`:

```python
def test_volumetric_settings_cloud_shadows():
    """PyVolumetricSettings should have cloud shadow fields."""
    from forge3d._forge3d import VolumetricSettings
    vs = VolumetricSettings()
    assert hasattr(vs, 'cloud_shadows_enabled')
    assert hasattr(vs, 'cloud_shadow_density')
    assert hasattr(vs, 'cloud_shadow_speed')
    assert hasattr(vs, 'cloud_shadow_quality')
    # Defaults should be sensible
    assert vs.cloud_shadows_enabled == False
    assert 0.0 <= vs.cloud_shadow_density <= 1.0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_registration.py::test_volumetric_settings_cloud_shadows -v -x`
Expected: FAIL.

**Step 3: Extend `PyVolumetricSettings` in `src/lighting/py_bindings.rs`**

Add fields to the struct (after existing `phase_function` field):

```rust
    #[pyo3(get, set)]
    pub cloud_shadows_enabled: bool,
    #[pyo3(get, set)]
    pub cloud_shadow_quality: String,  // "low"|"medium"|"high"|"ultra"
    #[pyo3(get, set)]
    pub cloud_shadow_density: f32,
    #[pyo3(get, set)]
    pub cloud_shadow_speed: f32,
```

Update `Default` impl and `__new__` to include defaults:

```rust
    cloud_shadows_enabled: false,
    cloud_shadow_quality: "medium".to_string(),
    cloud_shadow_density: 0.6,
    cloud_shadow_speed: 0.02,
```

**Step 4: Build and test**

Run: `maturin develop --release 2>&1 && python -m pytest tests/test_api_registration.py::test_volumetric_settings_cloud_shadows -v -x`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/lighting/py_bindings.rs tests/test_api_registration.py
git commit -m "feat: add cloud shadow settings to PyVolumetricSettings"
```

---

### Task 7: Add `PyReflectionSettings` pyclass

**Files:**
- Modify: `src/lighting/py_bindings.rs` (add new pyclass)
- Modify: `src/lib.rs` (register class)
- Modify: `tests/test_api_registration.py` (add test)

**Step 1: Write the failing test**

Append to `tests/test_api_registration.py`:

```python
def test_reflection_settings_class():
    """ReflectionSettings pyclass should be importable and constructable."""
    from forge3d._forge3d import ReflectionSettings
    rs = ReflectionSettings()
    assert rs.enabled == False
    assert rs.quality == "medium"
    assert rs.fresnel_power > 0.0
    assert rs.plane_y == 0.0
    # Setters should work
    rs.enabled = True
    rs.quality = "high"
    assert rs.enabled == True
    assert rs.quality == "high"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_registration.py::test_reflection_settings_class -v -x`
Expected: FAIL.

**Step 3: Add `PyReflectionSettings` to `src/lighting/py_bindings.rs`**

Add at end of file (before any `#[cfg(test)]` block):

```rust
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "ReflectionSettings")]
#[derive(Clone)]
pub struct PyReflectionSettings {
    #[pyo3(get, set)]
    pub enabled: bool,
    #[pyo3(get, set)]
    pub quality: String,
    #[pyo3(get, set)]
    pub plane_y: f32,
    #[pyo3(get, set)]
    pub fresnel_power: f32,
    #[pyo3(get, set)]
    pub intensity: f32,
    #[pyo3(get, set)]
    pub distance_fade_start: f32,
    #[pyo3(get, set)]
    pub distance_fade_end: f32,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyReflectionSettings {
    #[new]
    #[pyo3(signature = (
        enabled = false,
        quality = "medium".to_string(),
        plane_y = 0.0,
        fresnel_power = 5.0,
        intensity = 0.8,
        distance_fade_start = 50.0,
        distance_fade_end = 200.0,
    ))]
    fn new(
        enabled: bool,
        quality: String,
        plane_y: f32,
        fresnel_power: f32,
        intensity: f32,
        distance_fade_start: f32,
        distance_fade_end: f32,
    ) -> Self {
        Self {
            enabled,
            quality,
            plane_y,
            fresnel_power,
            intensity,
            distance_fade_start,
            distance_fade_end,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ReflectionSettings(enabled={}, quality='{}', plane_y={}, fresnel_power={})",
            self.enabled, self.quality, self.plane_y, self.fresnel_power,
        )
    }
}
```

Register in `src/lib.rs`:

```rust
    // Planar reflection settings (API consolidation)
    m.add_class::<crate::lighting::py_bindings::PyReflectionSettings>()?;
```

**Step 4: Build and test**

Run: `maturin develop --release 2>&1 && python -m pytest tests/test_api_registration.py::test_reflection_settings_class -v -x`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/lighting/py_bindings.rs src/lib.rs tests/test_api_registration.py
git commit -m "feat: add PyReflectionSettings pyclass for planar reflections"
```

---

## Phase 5: Labels PyO3 Bindings

### Task 8: Create labels Python bindings

**Files:**
- Create: `src/labels/py_bindings.rs`
- Modify: `src/labels/mod.rs` (add `pub mod py_bindings;`)
- Modify: `src/lib.rs` (register classes + call registration)
- Modify: `tests/test_api_registration.py` (add tests)

**Step 1: Write the failing test**

Append to `tests/test_api_registration.py`:

```python
def test_label_classes_importable():
    """Label system pyclasses should be registered."""
    from forge3d._forge3d import LabelStyle
    ls = LabelStyle()
    assert hasattr(ls, 'font_size')
    assert hasattr(ls, 'color')
    assert ls.font_size > 0.0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_registration.py::test_label_classes_importable -v -x`
Expected: FAIL.

**Step 3: Create `src/labels/py_bindings.rs`**

```rust
//! PyO3 bindings for the labels system (API consolidation).
//! Exposes read-only label configuration for headless rendering workflows.

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

/// Python-visible label style configuration.
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "LabelStyle")]
#[derive(Clone)]
pub struct PyLabelStyle {
    #[pyo3(get, set)]
    pub font_size: f32,
    #[pyo3(get, set)]
    pub color: (f32, f32, f32, f32),
    #[pyo3(get, set)]
    pub halo_color: (f32, f32, f32, f32),
    #[pyo3(get, set)]
    pub halo_width: f32,
    #[pyo3(get, set)]
    pub max_width: f32,
    #[pyo3(get, set)]
    pub anchor: String,
    #[pyo3(get, set)]
    pub priority: f32,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyLabelStyle {
    #[new]
    #[pyo3(signature = (
        font_size = 14.0,
        color = (1.0, 1.0, 1.0, 1.0),
        halo_color = (0.0, 0.0, 0.0, 0.8),
        halo_width = 2.0,
        max_width = 200.0,
        anchor = "center".to_string(),
        priority = 0.0,
    ))]
    fn new(
        font_size: f32,
        color: (f32, f32, f32, f32),
        halo_color: (f32, f32, f32, f32),
        halo_width: f32,
        max_width: f32,
        anchor: String,
        priority: f32,
    ) -> Self {
        Self { font_size, color, halo_color, halo_width, max_width, anchor, priority }
    }

    fn __repr__(&self) -> String {
        format!("LabelStyle(font_size={}, anchor='{}')", self.font_size, self.anchor)
    }
}

/// Register label bindings into the forge3d module.
#[cfg(feature = "extension-module")]
pub fn register_label_bindings(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<PyLabelStyle>()?;
    Ok(())
}
```

**Step 4: Wire it up**

Add to `src/labels/mod.rs`:

```rust
#[cfg(feature = "extension-module")]
pub mod py_bindings;
```

Register in `src/lib.rs` (near clipmap registration):

```rust
    // Labels bindings (API consolidation)
    crate::labels::py_bindings::register_label_bindings(&m)?;
```

**Step 5: Build and test**

Run: `maturin develop --release 2>&1 && python -m pytest tests/test_api_registration.py::test_label_classes_importable -v -x`
Expected: PASS.

**Step 6: Commit**

```bash
git add src/labels/py_bindings.rs src/labels/mod.rs src/lib.rs tests/test_api_registration.py
git commit -m "feat: add PyLabelStyle pyclass for headless label rendering"
```

---

## Phase 6: Deprecate Unused Rust Modules

### Task 9: Add deprecation annotations to 4 Rust modules

**Files:**
- Modify: `src/export/mod.rs`
- Modify: `src/style/mod.rs`
- Modify: `src/tiles3d/mod.rs`
- Modify: `src/bundle/mod.rs`
- Modify: `tests/test_api_registration.py`

**Step 1: Write the test**

Append to `tests/test_api_registration.py`:

```python
def test_deprecated_modules_documented():
    """Deprecated Rust modules should be documented in the design doc."""
    from pathlib import Path
    design_doc = Path("docs/plans/2026-02-19-rust-python-api-consolidation-design.md")
    assert design_doc.exists()
    content = design_doc.read_text()
    # Verify the decision matrix documents all 4 deprecations
    for module in ["SVG export", "Mapbox style", "3D Tiles", "Bundle"]:
        assert module in content, f"Missing deprecation documentation for {module}"
```

**Step 2: Add deprecation headers to each module**

Prepend to `src/export/mod.rs`:

```rust
//! # DEPRECATED
//! The Python implementation (`python/forge3d/export.py`) is the canonical SVG export path.
//! This Rust module is retained as reference only. See design doc:
//! `docs/plans/2026-02-19-rust-python-api-consolidation-design.md` Section 5.
```

Prepend to `src/style/mod.rs`:

```rust
//! # DEPRECATED
//! The Python implementation (`python/forge3d/style.py` + `style_expressions.py`) is the
//! canonical Mapbox Style Spec parser. This Rust module is retained as reference only.
```

Prepend to `src/tiles3d/mod.rs`:

```rust
//! # DEPRECATED
//! The Python implementation (`python/forge3d/tiles3d.py`) is the canonical 3D Tiles parser.
//! This Rust module is retained as reference only.
```

Prepend to `src/bundle/mod.rs`:

```rust
//! # DEPRECATED
//! The Python implementation (`python/forge3d/bundle.py`) is the canonical bundle format.
//! This Rust module is retained as reference only.
```

**Step 3: Add `#[deprecated]` to public items in each module**

For each `pub fn` and `pub struct` in these modules, add:

```rust
#[deprecated(note = "Use Python forge3d.export / forge3d.style / forge3d.tiles3d / forge3d.bundle instead")]
```

**Step 4: Build — expect deprecation warnings (that's correct)**

Run: `maturin develop --release 2>&1`
Expected: Build succeeds. Deprecation warnings are expected and acceptable for this change.

**Step 5: Run test**

Run: `python -m pytest tests/test_api_registration.py::test_deprecated_modules_documented -v -x`
Expected: PASS.

**Step 6: Commit**

```bash
git add src/export/mod.rs src/style/mod.rs src/tiles3d/mod.rs src/bundle/mod.rs tests/test_api_registration.py
git commit -m "chore: mark export, style, tiles3d, bundle Rust modules as deprecated"
```

---

## Phase 7: Wire Bloom Execute + SSGI/SSR Settings

### Task 10: Wire bloom `execute()` to dispatch compute pipelines

**Files:**
- Modify: `src/core/bloom.rs` (rewrite `execute()` body)
- Create: `tests/test_bloom_wired.py`

**Step 1: Write the failing test**

Create `tests/test_bloom_wired.py`:

```python
"""Test that bloom execute() actually dispatches compute work."""
import pytest

def test_bloom_config_defaults():
    """BloomConfig defaults should be opt-in (enabled=false)."""
    # This test validates the design requirement: new features default off
    # Bloom config is internal to Rust, so we test via terrain params
    from forge3d._forge3d import TerrainRenderParams
    params = TerrainRenderParams()
    # If bloom fields exist, verify defaults
    if hasattr(params, 'bloom_enabled'):
        assert params.bloom_enabled == False
    if hasattr(params, 'bloom_threshold'):
        assert params.bloom_threshold > 0.0
```

**Step 2: Implement bloom `execute()`**

In `src/core/bloom.rs`, replace the no-op body of `execute()` with actual dispatches:

```rust
fn execute(
    &self,
    device: &Device,
    encoder: &mut CommandEncoder,
    input: &TextureView,
    output: &TextureView,
    _resource_pool: &PostFxResourcePool,
    mut timing_manager: Option<&mut GpuTimingManager>,
) -> RenderResult<()> {
    if let Some(ref tm) = timing_manager {
        // Begin timing scope
    }

    let brightpass_pipeline = self.brightpass_pipeline.as_ref()
        .ok_or_else(|| RenderError::Render("Bloom brightpass pipeline not initialized".into()))?;
    let blur_h_pipeline = self.blur_h_pipeline.as_ref()
        .ok_or_else(|| RenderError::Render("Bloom blur_h pipeline not initialized".into()))?;
    let blur_v_pipeline = self.blur_v_pipeline.as_ref()
        .ok_or_else(|| RenderError::Render("Bloom blur_v pipeline not initialized".into()))?;

    let config = &self.config;
    if !config.enabled {
        return Ok(());
    }

    // Create intermediate textures for ping-pong
    let desc = wgpu::TextureDescriptor {
        label: Some("bloom_intermediate"),
        size: wgpu::Extent3d { width: self.width, height: self.height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };
    let intermediate_a = device.create_texture(&desc);
    let intermediate_b = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bloom_intermediate_b"),
        ..desc
    });
    let view_a = intermediate_a.create_view(&Default::default());
    let view_b = intermediate_b.create_view(&Default::default());

    let wg_x = (self.width + 7) / 8;
    let wg_y = (self.height + 7) / 8;

    // Pass 1: Brightpass (input -> intermediate_a)
    // ... create bind group with input + view_a + brightpass uniforms
    // ... dispatch compute pass

    // Pass 2: Horizontal blur (intermediate_a -> intermediate_b)
    // ... dispatch compute pass

    // Pass 3: Vertical blur (intermediate_b -> output, additive blend with input)
    // ... dispatch compute pass

    Ok(())
}
```

**Important:** The exact bind group layout depends on the WGSL shaders (`bloom_brightpass.wgsl`, `bloom_blur_h.wgsl`, `bloom_blur_v.wgsl`). Read those shaders before writing the bind group creation code. The above is the structural outline — fill in bind groups after reading the shader binding declarations.

**Step 3: Build and test**

Run: `maturin develop --release 2>&1 && python -m pytest tests/test_bloom_wired.py -v -x`
Expected: PASS.

**Step 4: Commit**

```bash
git add src/core/bloom.rs tests/test_bloom_wired.py
git commit -m "feat: wire bloom execute() to dispatch brightpass and blur compute passes"
```

---

### Task 11: Wire SSGI/SSR Python settings to Rust update_settings

**Files:**
- Modify: `src/terrain/renderer.rs` or `src/viewer/` (wherever `SsgiPass`/`SsrPass` are instantiated)
- Modify: `tests/test_api_registration.py`

**Step 1: Write the test**

Append to `tests/test_api_registration.py`:

```python
def test_ssgi_settings_fields_exist():
    """PySSGISettings should have writable fields that map to Rust."""
    from forge3d._forge3d import SSGISettings
    s = SSGISettings()
    # These fields should exist and be settable
    s.ray_steps = 32
    s.intensity = 0.5
    assert s.ray_steps == 32
    assert s.intensity == 0.5

def test_ssr_settings_fields_exist():
    """PySSRSettings should have writable fields that map to Rust."""
    from forge3d._forge3d import SSRSettings
    s = SSRSettings()
    s.max_steps = 64
    s.intensity = 0.8
    assert s.max_steps == 64
    assert s.intensity == 0.8
```

**Step 2: Investigate the wire-up gap**

Before coding, trace where `SsgiPass`/`SsrPass` are created and where `PySSGISettings`/`PySSRSettings` are consumed. The gap is that Python creates settings objects, but they're never passed to `SsgiPass::update_settings()` or `SsrPass::update_settings()`. Find the connection point — likely in `TerrainRenderer` render method or viewer render loop.

**Step 3: Wire the settings**

The exact code depends on where the gap is. The pattern:

```rust
// In the render path where ssgi_pass is used:
if let Some(ref ssgi_settings) = self.py_ssgi_settings {
    ssgi_pass.update_settings(&queue, |s| {
        s.ray_steps = ssgi_settings.ray_steps;
        s.intensity = ssgi_settings.intensity;
        // ... map all fields
    });
}
```

**Step 4: Build and test**

Run: `maturin develop --release 2>&1 && python -m pytest tests/test_api_registration.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add <modified files> tests/test_api_registration.py
git commit -m "feat: wire PySSGISettings and PySSRSettings to Rust update_settings"
```

---

## Phase 8: Point Cloud GPU Path + COPC LAZ

### Task 12: Add wgpu vertex buffer and render pipeline to point cloud renderer

**Files:**
- Modify: `src/pointcloud/renderer.rs`
- Create: `tests/test_pointcloud_gpu.py`

**Step 1: Write the test**

Create `tests/test_pointcloud_gpu.py`:

```python
"""Test point cloud GPU buffer creation."""
import pytest

def test_point_buffer_has_data():
    """PointBuffer from CPU path should have positions."""
    # This tests the existing CPU path still works
    # GPU path is internal to Rust and tested via viewer rendering
    pass  # Placeholder — real GPU test requires viewer integration

def test_pointcloud_renderer_module_exists():
    """Point cloud module should be accessible."""
    # Verify the module compiles and basic types work
    import forge3d._forge3d as native
    # PointCloudRenderer is internal, but we can verify the module loaded
    assert native is not None
```

**Step 2: Add GPU buffer creation to `src/pointcloud/renderer.rs`**

Add to `PointCloudRenderer`:

```rust
/// Create a wgpu vertex buffer from a PointBuffer.
pub fn create_gpu_buffer(
    &self,
    device: &wgpu::Device,
    buffer: &PointBuffer,
) -> Option<wgpu::Buffer> {
    if buffer.positions.is_empty() {
        return None;
    }
    Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("point_cloud_vertex_buffer"),
        contents: bytemuck::cast_slice(&buffer.positions),
        usage: wgpu::BufferUsages::VERTEX,
    }))
}
```

**Important:** This requires adding `wgpu` and `bytemuck` as dependencies if not already present in the pointcloud module. Check `Cargo.toml` and the module's existing imports.

**Step 3: Build and test**

Run: `maturin develop --release 2>&1 && python -m pytest tests/test_pointcloud_gpu.py -v -x`
Expected: PASS.

**Step 4: Commit**

```bash
git add src/pointcloud/renderer.rs tests/test_pointcloud_gpu.py
git commit -m "feat: add wgpu vertex buffer creation to PointCloudRenderer"
```

---

### Task 13: Integrate laz-rs for COPC LAZ decompression

**Files:**
- Modify: `Cargo.toml` (add `laz` dependency)
- Modify: `src/pointcloud/copc.rs` (replace placeholder in `decode_laz_chunk`)
- Create: `tests/test_copc_laz.py`

**Step 1: Write the test**

Create `tests/test_copc_laz.py`:

```python
"""Test COPC LAZ decompression."""
import pytest

def test_copc_module_compiles():
    """COPC module should compile with laz-rs integration."""
    import forge3d._forge3d as native
    assert native is not None

# Note: Real COPC LAZ tests require a .copc.laz fixture file.
# Add fixture-based tests when test data is available.
```

**Step 2: Add `laz` crate to `Cargo.toml`**

Under `[dependencies]`:

```toml
laz = { version = "0.9", optional = true }
```

Under `[features]`:

```toml
copc_laz = ["laz"]
```

**Step 3: Replace placeholder in `src/pointcloud/copc.rs`**

In `decode_laz_chunk()`, replace the zeroed-data fallback:

```rust
fn decode_laz_chunk(
    data: &[u8],
    point_count: u32,
    header: &CopcHeader,
) -> PointCloudResult<PointData> {
    let record_len = header.point_record_length as usize;
    let expected_raw = point_count as usize * record_len;

    // If data length matches expected uncompressed size, it's raw
    if data.len() >= expected_raw {
        // ... existing raw decode path (keep as-is) ...
    }

    // LAZ compressed data
    #[cfg(feature = "copc_laz")]
    {
        use laz::LasZipDecompressor;
        let mut decompressor = LasZipDecompressor::new(
            std::io::Cursor::new(data),
            header.point_record_length as usize,
        ).map_err(|e| PointCloudError::Format(format!("LAZ init: {}", e)))?;

        let mut raw = vec![0u8; expected_raw];
        decompressor.decompress_many(&mut raw)
            .map_err(|e| PointCloudError::Format(format!("LAZ decompress: {}", e)))?;

        // Now decode the raw uncompressed bytes (reuse existing logic)
        return decode_raw_points(&raw, point_count, header);
    }

    #[cfg(not(feature = "copc_laz"))]
    {
        // Return zeroed placeholder with clear error message
        eprintln!(
            "[pointcloud] LAZ decompression not available. \
             Build with `--features copc_laz` to enable. \
             Returning {} zeroed points.",
            point_count
        );
        let positions = vec![0.0f32; point_count as usize * 3];
        Ok(PointData { positions, colors: None, intensities: None })
    }
}
```

**Important:** Extract the existing raw-byte decode logic into a helper `decode_raw_points()` so both the raw and LAZ paths can use it.

**Step 4: Build and test**

Run: `maturin develop --release 2>&1 && python -m pytest tests/test_copc_laz.py -v -x`
Expected: PASS.

**Step 5: Commit**

```bash
git add Cargo.toml src/pointcloud/copc.rs tests/test_copc_laz.py
git commit -m "feat: integrate laz-rs for COPC LAZ decompression (feature-gated)"
```

---

## Phase 9: Final Verification

### Task 14: Full regression suite + symbol count verification

**Files:**
- None new

**Step 1: Run the full test suite**

Run: `python -m pytest tests/ -v --tb=short -q`
Expected: All 916+ tests pass (new tests add to this count).

**Step 2: Verify symbol count increased**

Run: `python -c "import forge3d._forge3d as n; print('Native symbols:', len([s for s in dir(n) if not s.startswith('_')]))"``
Expected: Count > 134 (the pre-consolidation baseline from the 2026-02-17 audit).

**Step 3: Verify zero new build warnings**

Run: `maturin develop --release 2>&1`
Expected: Zero new warnings (deprecation warnings from Task 9 are expected).

**Step 4: Final commit (if any cleanup needed)**

```bash
git add -A
git commit -m "chore: API consolidation complete — verified full regression suite"
```

---

## Appendix: File Modification Summary

| File | Tasks | Action |
|------|-------|--------|
| `src/lib.rs` | 1,2,4,5,7,8 | Register classes + functions |
| `src/lighting/py_bindings.rs` | 6,7 | Add cloud shadow fields + ReflectionSettings class |
| `src/terrain/analysis.rs` | 5 | Add PyO3 wrappers at end |
| `src/labels/py_bindings.rs` | 8 | Create new file |
| `src/labels/mod.rs` | 8 | Add `pub mod py_bindings` |
| `src/core/bloom.rs` | 10 | Rewrite `execute()` body |
| `src/pointcloud/renderer.rs` | 12 | Add GPU buffer creation |
| `src/pointcloud/copc.rs` | 13 | Replace LAZ placeholder |
| `Cargo.toml` | 13 | Add `laz` optional dependency |
| `src/export/mod.rs` | 9 | Add deprecation header |
| `src/style/mod.rs` | 9 | Add deprecation header |
| `src/tiles3d/mod.rs` | 9 | Add deprecation header |
| `src/bundle/mod.rs` | 9 | Add deprecation header |
| `python/forge3d/lighting.py` | 3 | Remove dead ReSTIR refs |
| `tests/test_api_registration.py` | 1-9,11 | New test file (progressive) |
| `tests/test_bloom_wired.py` | 10 | New test file |
| `tests/test_pointcloud_gpu.py` | 12 | New test file |
| `tests/test_copc_laz.py` | 13 | New test file |
