# TV12 — Terrain Offline Render Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic offline terrain accumulation pipeline with adaptive sampling and optional OIDN denoising, producing production-quality terrain renders.

**Architecture:** Rust/GPU engine exposes batch-oriented accumulation primitives (begin/accumulate/metrics/resolve/tonemap). Python controller drives the render loop, adaptive stopping, and OIDN post-processing. HDR accumulation happens on GPU; tonemapping happens exactly once at finalize time.

**Tech Stack:** Rust (wgpu, PyO3), WGSL compute shaders, Python (NumPy), optional Intel OIDN

**Spec:** `docs/superpowers/specs/2026-03-22-tv12-terrain-offline-render-quality-design.md`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `python/forge3d/terrain_params.py` | Add `OfflineQualitySettings` dataclass, extend `DenoiseSettings` with `'oidn'` |
| `python/forge3d/offline.py` | New: `render_offline()` controller, `OfflineResult`, `OfflineProgress` |
| `python/forge3d/denoise_oidn.py` | New: `oidn_available()`, `oidn_denoise()` wrappers |
| `python/forge3d/__init__.py` | Export new types |
| `python/forge3d/__init__.pyi` | Add `HdrFrame` type stub |
| `src/py_types/hdr_frame.rs` | New: `HdrFrame` Python type with `to_numpy_f32()`, `save()` |
| `src/py_types/mod.rs` | Register `hdr_frame` module |
| `src/py_module/classes.rs` | Add `m.add_class::<HdrFrame>()?;` registration |
| `src/terrain/accumulation.rs` | Extend with `OfflineAccumulationState` (ping-pong buffers, cached GPU resources, `prev_tile_means`) |
| `src/terrain/renderer/offline.rs` | New: all 7 offline `#[pymethods]` on `TerrainRenderer` |
| `src/terrain/renderer/core.rs` | Add offline state fields to `TerrainScene`, HDR pipeline cache |
| `src/terrain/renderer/mod.rs` | Register `offline` submodule |
| `src/terrain/renderer/draw/setup/pipeline.rs` | Support `Rgba16Float` render target for HDR output |
| `src/terrain/renderer/pipeline_cache.rs` | Add HDR pipeline variant |
| `src/terrain/render_params/decode_postfx.rs` | Add `DenoiseMethodNative::Oidn` variant |
| `src/shaders/offline_accumulate.wgsl` | New: ping-pong additive accumulation compute shader |
| `src/shaders/offline_resolve.wgsl` | New: divide-by-N resolve + normal renormalization |
| `src/shaders/offline_luminance.wgsl` | New: quarter-res luminance for convergence metrics |
| `src/terrain/renderer/upload.rs` | Pack `offline_hdr_output` flag into overlay uniforms `params5[3]` |
| `src/shaders/terrain_pbr_pom.wgsl` | Read `offline_hdr_output` flag; skip tonemap+sRGB when set |
| `tests/test_tv12_offline_quality.py` | All TV12 tests |
| `examples/terrain_tv12_offline_quality_demo.py` | Demo with real DEM, comparison renders |
| `docs/tv12-terrain-offline-render-quality.md` | Feature documentation |

---

## Task 1: Python Configuration

**Files:**
- Modify: `python/forge3d/terrain_params.py`
- Test: `tests/test_tv12_offline_quality.py`

- [ ] **Step 1: Write failing test for OfflineQualitySettings**

Create `tests/test_tv12_offline_quality.py`:

```python
from __future__ import annotations
import pytest
from forge3d.terrain_params import OfflineQualitySettings, DenoiseSettings


def test_offline_quality_settings_defaults():
    s = OfflineQualitySettings()
    assert s.enabled is False
    assert s.adaptive is False
    assert s.target_variance == 0.001
    assert s.max_samples == 64
    assert s.min_samples == 4
    assert s.batch_size == 4
    assert s.tile_size == 16
    assert s.convergence_ratio == 0.95


def test_offline_quality_settings_validation():
    with pytest.raises(ValueError):
        OfflineQualitySettings(max_samples=0)
    with pytest.raises(ValueError):
        OfflineQualitySettings(min_samples=0)
    with pytest.raises(ValueError):
        OfflineQualitySettings(batch_size=0)
    with pytest.raises(ValueError):
        OfflineQualitySettings(tile_size=0)
    with pytest.raises(ValueError):
        OfflineQualitySettings(convergence_ratio=1.5)
    with pytest.raises(ValueError):
        OfflineQualitySettings(target_variance=-0.1)


def test_denoise_settings_accepts_oidn():
    s = DenoiseSettings(enabled=True, method="oidn")
    assert s.method == "oidn"


def test_denoise_settings_rejects_invalid():
    with pytest.raises(ValueError):
        DenoiseSettings(method="magic")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .worktrees/epic-12 && python -m pytest tests/test_tv12_offline_quality.py -v -x`
Expected: ImportError — `OfflineQualitySettings` does not exist yet.

- [ ] **Step 3: Implement OfflineQualitySettings and update DenoiseSettings**

In `python/forge3d/terrain_params.py`, after the `DenoiseSettings` class (~line 870):

1. Update `DenoiseSettings.__post_init__` validator: change `valid_methods = ("atrous", "bilateral", "none")` to `valid_methods = ("atrous", "bilateral", "oidn", "none")`.

2. Add new dataclass:

```python
@dataclass
class OfflineQualitySettings:
    """TV12: Offline accumulation and adaptive sampling policy.

    Does NOT duplicate aa_samples/aa_seed (use existing params) or
    denoiser selection (use DenoiseSettings.method).
    """
    enabled: bool = False
    adaptive: bool = False
    target_variance: float = 0.001
    max_samples: int = 64
    min_samples: int = 4
    batch_size: int = 4
    tile_size: int = 16
    convergence_ratio: float = 0.95

    def __post_init__(self) -> None:
        if self.max_samples < 1:
            raise ValueError("max_samples must be >= 1")
        if self.min_samples < 1:
            raise ValueError("min_samples must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.tile_size < 1:
            raise ValueError("tile_size must be >= 1")
        if self.convergence_ratio < 0.0 or self.convergence_ratio > 1.0:
            raise ValueError("convergence_ratio must be in [0.0, 1.0]")
        if self.target_variance < 0.0:
            raise ValueError("target_variance must be >= 0.0")
```

3. Add `offline: Optional[OfflineQualitySettings] = None` parameter to `make_terrain_params_config()` signature (~line 1648, after `denoise`).

4. Pass it through in the config dict construction.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd .worktrees/epic-12 && python -m pytest tests/test_tv12_offline_quality.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add python/forge3d/terrain_params.py tests/test_tv12_offline_quality.py
git commit -m "feat(tv12): add OfflineQualitySettings config and extend DenoiseSettings with oidn"
```

---

## Task 2: OIDN Integration Module

**Files:**
- Create: `python/forge3d/denoise_oidn.py`
- Test: `tests/test_tv12_offline_quality.py`

- [ ] **Step 1: Write failing tests for OIDN module**

Append to `tests/test_tv12_offline_quality.py`:

```python
import numpy as np
from forge3d.denoise_oidn import oidn_available, oidn_denoise


def test_oidn_available_returns_bool():
    result = oidn_available()
    assert isinstance(result, bool)


def test_oidn_denoise_fallback_when_unavailable():
    """When oidn is not installed, oidn_denoise raises ImportError."""
    if oidn_available():
        pytest.skip("oidn is installed, cannot test fallback")
    beauty = np.random.rand(64, 64, 3).astype(np.float32)
    with pytest.raises(ImportError):
        oidn_denoise(beauty)


def test_oidn_denoise_validates_input_shape():
    """Even without oidn, shape validation should run first."""
    if oidn_available():
        pytest.skip("test targets missing-oidn path")
    bad = np.random.rand(64, 64, 4).astype(np.float32)
    with pytest.raises(ValueError, match="beauty must be.*H, W, 3"):
        oidn_denoise(bad)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd .worktrees/epic-12 && python -m pytest tests/test_tv12_offline_quality.py::test_oidn_available_returns_bool -v`
Expected: ImportError — `denoise_oidn` module does not exist.

- [ ] **Step 3: Implement denoise_oidn.py**

Create `python/forge3d/denoise_oidn.py`:

```python
"""TV12: Optional OIDN (Open Image Denoise) integration.

Requires: pip install pyoidn   (or: pip install oidn)
Falls back cleanly when the package is not installed.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def oidn_available() -> bool:
    """Return True if an OIDN Python binding is importable."""
    try:
        import oidn  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import pyoidn  # noqa: F401
        return True
    except ImportError:
        pass
    return False


def oidn_denoise(
    beauty: np.ndarray,
    albedo: Optional[np.ndarray] = None,
    normal: Optional[np.ndarray] = None,
    *,
    hdr: bool = True,
    quality: str = "high",
) -> np.ndarray:
    """Denoise using Intel Open Image Denoise.

    Args:
        beauty: (H, W, 3) float32 linear HDR color.
        albedo: optional (H, W, 3) float32 albedo guide.
        normal: optional (H, W, 3) float32 normal guide.
        hdr: True if beauty is HDR (values can exceed 1.0).
        quality: "default" or "high".

    Returns:
        Denoised (H, W, 3) float32 array.

    Raises:
        ValueError: if input shapes are wrong.
        ImportError: if no OIDN package is available.
    """
    if beauty.ndim != 3 or beauty.shape[2] != 3:
        raise ValueError("beauty must be (H, W, 3) float32")
    h, w = beauty.shape[:2]
    if albedo is not None and albedo.shape != (h, w, 3):
        raise ValueError(f"albedo shape {albedo.shape} must match beauty ({h}, {w}, 3)")
    if normal is not None and normal.shape != (h, w, 3):
        raise ValueError(f"normal shape {normal.shape} must match beauty ({h}, {w}, 3)")

    # Try importing OIDN
    oidn_mod = None
    try:
        import oidn as oidn_mod  # noqa: F811
    except ImportError:
        try:
            import pyoidn as oidn_mod  # noqa: F811
        except ImportError:
            raise ImportError(
                "OIDN denoising requires the 'oidn' or 'pyoidn' package. "
                "Install with: pip install pyoidn"
            )

    device = oidn_mod.NewDevice()
    device.Commit()

    beauty_f32 = np.ascontiguousarray(beauty, dtype=np.float32)
    output = np.empty_like(beauty_f32)

    filt = device.NewFilter("RT")
    filt.SetImage("color", beauty_f32)
    filt.SetImage("output", output)
    if albedo is not None:
        filt.SetImage("albedo", np.ascontiguousarray(albedo, dtype=np.float32))
    if normal is not None:
        filt.SetImage("normal", np.ascontiguousarray(normal, dtype=np.float32))
    filt.SetBool("hdr", hdr)
    if quality == "high":
        filt.SetInt("quality", 1)
    filt.Commit()
    filt.Execute()

    return output
```

- [ ] **Step 4: Run tests**

Run: `cd .worktrees/epic-12 && python -m pytest tests/test_tv12_offline_quality.py -k oidn -v`
Expected: All 3 oidn tests PASS (or skip if oidn is installed).

- [ ] **Step 5: Commit**

```bash
git add python/forge3d/denoise_oidn.py tests/test_tv12_offline_quality.py
git commit -m "feat(tv12): add OIDN denoiser integration module with runtime detection"
```

---

## Task 3: HdrFrame Rust Type

**Files:**
- Create: `src/py_types/hdr_frame.rs`
- Modify: `src/py_types/mod.rs` (add `pub mod hdr_frame; pub use hdr_frame::*;`)
- Modify: `src/py_module/classes.rs` (add `m.add_class::<HdrFrame>()?;` after Frame registration ~line 28)
- Modify: `python/forge3d/__init__.py` (add `"HdrFrame"` to native exports list ~line 56)
- Modify: `python/forge3d/__init__.pyi` (add `HdrFrame` type stub)
- Test: `tests/test_tv12_offline_quality.py`

- [ ] **Step 1: Write failing test for HdrFrame**

Append to `tests/test_tv12_offline_quality.py`:

```python
from _terrain_runtime import terrain_rendering_available
GPU_AVAILABLE = terrain_rendering_available()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU terrain rendering not available")
def test_hdr_frame_type_exists():
    assert hasattr(f3d, "HdrFrame")
```

Add `import forge3d as f3d` to the top imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .worktrees/epic-12 && python -m pytest tests/test_tv12_offline_quality.py::test_hdr_frame_type_exists -v`
Expected: FAIL — `HdrFrame` not in forge3d.

- [ ] **Step 3: Implement HdrFrame**

Create `src/py_types/hdr_frame.rs`:

```rust
use super::super::*;

/// TV12: HDR frame for offline render quality pipeline.
///
/// Always Rgba16Float format. Provides float32 numpy export for OIDN handoff
/// and EXR save. Owns its texture independently — survives offline session teardown.
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "HdrFrame")]
pub struct HdrFrame {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) texture: wgpu::Texture,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl HdrFrame {
    pub(crate) fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        texture: wgpu::Texture,
        width: u32,
        height: u32,
    ) -> Self {
        Self { device, queue, texture, width, height }
    }
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl HdrFrame {
    #[new]
    fn py_new() -> PyResult<Self> {
        Err(PyRuntimeError::new_err(
            "HdrFrame objects are constructed internally by forge3d",
        ))
    }

    #[getter]
    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Export as float32 numpy array (H, W, 4) in linear HDR space.
    fn to_numpy_f32<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray3<f32>> {
        let data = crate::core::hdr::read_hdr_texture(
            &self.device,
            &self.queue,
            &self.texture,
            self.width,
            self.height,
            wgpu::TextureFormat::Rgba16Float,
        )
        .map_err(|err| PyRuntimeError::new_err(format!("HDR readback failed: {err}")))?;

        let arr = ndarray::Array3::from_shape_vec(
            (self.height as usize, self.width as usize, 4),
            data,
        )
        .map_err(|_| PyRuntimeError::new_err("failed to reshape HDR buffer into numpy array"))?;

        Ok(arr.into_pyarray_bound(py).into_gil_ref())
    }

    /// Save to EXR file.
    fn save(&self, path: &str) -> PyResult<()> {
        let path_obj = std::path::Path::new(path);
        if let Some(ext) = path_obj.extension().and_then(|ext| ext.to_str()) {
            if !ext.eq_ignore_ascii_case("exr") {
                return Err(PyValueError::new_err(format!(
                    "HdrFrame.save() requires .exr extension, got .{}", ext
                )));
            }
        }
        #[cfg(feature = "images")]
        {
            let data = crate::core::hdr::read_hdr_texture(
                &self.device, &self.queue, &self.texture,
                self.width, self.height, wgpu::TextureFormat::Rgba16Float,
            )
            .map_err(|err| PyRuntimeError::new_err(format!("HDR readback failed: {err}")))?;

            exr_write::write_exr_rgba_f32(path_obj, self.width, self.height, &data, "beauty")
                .map_err(|err| PyRuntimeError::new_err(format!("EXR write failed: {err:#}")))?;
            Ok(())
        }
        #[cfg(not(feature = "images"))]
        Err(PyRuntimeError::new_err("saving HdrFrame requires the 'images' feature"))
    }

    fn __repr__(&self) -> String {
        format!("HdrFrame(width={}, height={}, format=Rgba16Float)", self.width, self.height)
    }
}
```

Update `src/py_types/mod.rs` to add `pub mod hdr_frame; pub use hdr_frame::*;`.

Update `python/forge3d/__init__.py` to add `"HdrFrame"` to the native exports list.

- [ ] **Step 4: Build and run test**

Run: `cd .worktrees/epic-12 && cargo build --features extension-module && python -m pytest tests/test_tv12_offline_quality.py::test_hdr_frame_type_exists -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/py_types/hdr_frame.rs src/py_types/mod.rs python/forge3d/__init__.py
git commit -m "feat(tv12): add HdrFrame type with float32 numpy export and EXR save"
```

---

## Task 4: Terrain Shader HDR Output Mode + Pipeline Support

**Files:**
- Modify: `src/shaders/terrain_pbr_pom.wgsl` (~lines 3718-3733, 3840-3851)
- Modify: `src/terrain/renderer/upload.rs` (~line 89, pack flag into `params5[3]`)
- Modify: `src/terrain/render_params/core.rs` (add `offline_hdr_output` field)
- Modify: `src/terrain/render_params/decode_core.rs` (decode the field)
- Modify: `src/terrain/render_params/decode_postfx.rs` (~line 263, add `"oidn" => DenoiseMethodNative::Oidn`)
- Modify: `src/terrain/renderer/draw/setup/pipeline.rs` (support Rgba16Float render target)
- Modify: `src/terrain/renderer/pipeline_cache.rs` (add HDR pipeline variant)
- Test: `tests/test_tv12_offline_quality.py`

- [ ] **Step 1: Write failing test for HDR output shader flag**

Append to `tests/test_tv12_offline_quality.py`:

```python
ROOT = Path(__file__).resolve().parents[1]
TERRAIN_SHADER = ROOT / "src" / "shaders" / "terrain_pbr_pom.wgsl"

def test_terrain_shader_has_offline_hdr_flag():
    source = TERRAIN_SHADER.read_text(encoding="utf-8")
    assert "offline_hdr_output" in source, "terrain shader must read offline_hdr_output flag"
```

Add `from pathlib import Path` to the imports.

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — `offline_hdr_output` not in shader source.

- [ ] **Step 3: Implement shader HDR mode**

In `src/shaders/terrain_pbr_pom.wgsl`:

1. Find the overlay uniforms struct where `params5` is read. The `output_srgb_eotf` is at `params5.z` (overlay_uniforms.params5[2]). Use `params5.w` (currently 0.0) for `offline_hdr_output`.

2. In the fragment shader's final output section (~line 3718), add the HDR bypass. Find:
```wgsl
let tonemapped = tonemap_filmic_terrain(shaded);
final_color = tonemapped;
```

Replace with:
```wgsl
let offline_hdr = overlay_uniforms.params5.w;
if (offline_hdr > 0.5) {
    // TV12: Output linear HDR (post-exposure, post-fog, pre-tonemap)
    final_color = shaded;
} else {
    let tonemapped = tonemap_filmic_terrain(shaded);
    final_color = tonemapped;
}
```

3. In the sRGB encoding section (~line 3840), guard sRGB encoding similarly:
```wgsl
if (offline_hdr < 0.5) {
    // existing sRGB encoding
    encoded_color = linear_to_srgb(final_color);
} else {
    encoded_color = final_color;
}
```

In `src/terrain/renderer/upload.rs` (~line 89):

Change `params5: [detail_fade_start, detail_fade_end, output_srgb_eotf, 0.0]` to use `params5[3]` for the offline flag. Add to the same function:
```rust
let offline_hdr_output = if params.offline_hdr_output { 1.0 } else { 0.0 };
// ...
params5: [detail_fade_start, detail_fade_end, output_srgb_eotf, offline_hdr_output],
```

In `src/terrain/render_params/core.rs`, add field: `pub offline_hdr_output: bool,`

In `src/terrain/render_params/decode_core.rs`, decode it (default false):
```rust
let offline_hdr_output = params
    .getattr("offline_hdr_output").ok()
    .and_then(|v| v.extract::<bool>().ok())
    .unwrap_or(false);
```

In `src/terrain/render_params/decode_postfx.rs` (~line 263), add OIDN variant to denoise method matching:
```rust
method: match method_str.as_str() {
    "bilateral" => DenoiseMethodNative::Bilateral,
    "oidn" => DenoiseMethodNative::Oidn,  // TV12: pass through to Python controller
    "none" => DenoiseMethodNative::None,
    _ => DenoiseMethodNative::Atrous,
},
```
Add `Oidn` variant to the `DenoiseMethodNative` enum (wherever it's defined).

In `src/terrain/renderer/pipeline_cache.rs`, add an HDR pipeline variant. The terrain pipeline takes `color_format` as a parameter — create a second cached entry with `TextureFormat::Rgba16Float` for offline HDR output.

In `src/terrain/renderer/draw/setup/pipeline.rs`, ensure the `RenderTargets` creation supports `Rgba16Float` as the internal texture format. The offline path should create targets with `Rgba16Float` instead of `self.color_format` (which is `Rgba8UnormSrgb`).

- [ ] **Step 4: Build and test**

Run: `cd .worktrees/epic-12 && cargo build --features extension-module && python -m pytest tests/test_tv12_offline_quality.py::test_terrain_shader_has_offline_hdr_flag -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/shaders/terrain_pbr_pom.wgsl src/terrain/renderer/upload.rs \
  src/terrain/render_params/core.rs src/terrain/render_params/decode_core.rs \
  tests/test_tv12_offline_quality.py
git commit -m "feat(tv12): add offline_hdr_output flag to terrain shader for linear HDR output"
```

---

## Task 5: Accumulation Compute Shaders

**Files:**
- Create: `src/shaders/offline_accumulate.wgsl`
- Create: `src/shaders/offline_resolve.wgsl`
- Create: `src/shaders/offline_luminance.wgsl`
- Test: `tests/test_tv12_offline_quality.py`

- [ ] **Step 1: Write failing test for shader existence**

```python
def test_offline_shaders_exist():
    assert (ROOT / "src" / "shaders" / "offline_accumulate.wgsl").exists()
    assert (ROOT / "src" / "shaders" / "offline_resolve.wgsl").exists()
    assert (ROOT / "src" / "shaders" / "offline_luminance.wgsl").exists()
```

- [ ] **Step 2: Implement offline_accumulate.wgsl**

```wgsl
// TV12: Additive ping-pong accumulation for offline terrain rendering.
// Reads current sample + previous accumulation, writes sum to next accumulation.
// Division by sample count happens at resolve time, not here.

@group(0) @binding(0) var current_sample: texture_2d<f32>;
@group(0) @binding(1) var prev_accumulation: texture_2d<f32>;
@group(0) @binding(2) var next_accumulation: texture_storage_2d<rgba32float, write>;

struct AccumParams {
    width: u32,
    height: u32,
    sample_index: u32,
    _pad: u32,
}
@group(0) @binding(3) var<uniform> params: AccumParams;

@compute @workgroup_size(8, 8)
fn accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let current = textureLoad(current_sample, coords, 0);

    var result: vec4<f32>;
    if (params.sample_index == 0u) {
        result = current;
    } else {
        let prev = textureLoad(prev_accumulation, coords, 0);
        result = prev + current;
    }
    textureStore(next_accumulation, coords, result);
}
```

- [ ] **Step 3: Implement offline_resolve.wgsl**

```wgsl
// TV12: Resolve accumulated HDR buffer by dividing by sample count.
// Beauty: divide by N, write to Rgba16Float output.
// Normal variant: divide by N then renormalize.

struct ResolveParams {
    width: u32,
    height: u32,
    sample_count: u32,
    mode: u32,  // 0 = beauty/albedo (divide only), 1 = normal (divide + renormalize)
}

@group(0) @binding(0) var accumulated: texture_2d<f32>;
@group(0) @binding(1) var resolved: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: ResolveParams;

@compute @workgroup_size(8, 8)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let accum = textureLoad(accumulated, coords, 0);
    let n = f32(max(params.sample_count, 1u));
    var result = accum / n;

    if (params.mode == 1u) {
        // Normal: renormalize after averaging
        let len = length(result.xyz);
        if (len > 1e-6) {
            result = vec4<f32>(result.xyz / len, result.w);
        }
    }

    textureStore(resolved, coords, result);
}
```

- [ ] **Step 4: Implement offline_luminance.wgsl**

```wgsl
// TV12: Quarter-resolution luminance extraction for convergence metrics.
// Reads the accumulated HDR buffer, divides by sample count, averages 4x4 pixel blocks,
// writes luminance to a quarter-res R32Float output.

struct LuminanceParams {
    src_width: u32,
    src_height: u32,
    sample_count: u32,
    _pad: u32,
}

@group(0) @binding(0) var accumulated: texture_2d<f32>;
@group(0) @binding(1) var luminance_out: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: LuminanceParams;

@compute @workgroup_size(8, 8)
fn extract_luminance(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_w = (params.src_width + 3u) / 4u;
    let dst_h = (params.src_height + 3u) / 4u;
    if (gid.x >= dst_w || gid.y >= dst_h) { return; }

    let n = f32(max(params.sample_count, 1u));
    var sum_lum: f32 = 0.0;
    var count: f32 = 0.0;

    for (var dy: u32 = 0u; dy < 4u; dy++) {
        for (var dx: u32 = 0u; dx < 4u; dx++) {
            let sx = gid.x * 4u + dx;
            let sy = gid.y * 4u + dy;
            if (sx < params.src_width && sy < params.src_height) {
                let accum = textureLoad(accumulated, vec2<i32>(i32(sx), i32(sy)), 0);
                let avg = accum / n;
                sum_lum += 0.2126 * avg.r + 0.7152 * avg.g + 0.0722 * avg.b;
                count += 1.0;
            }
        }
    }

    let mean_lum = sum_lum / max(count, 1.0);
    textureStore(luminance_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(mean_lum, 0.0, 0.0, 0.0));
}
```

- [ ] **Step 5: Run test**

Expected: PASS (files exist).

- [ ] **Step 6: Commit**

```bash
git add src/shaders/offline_accumulate.wgsl src/shaders/offline_resolve.wgsl \
  src/shaders/offline_luminance.wgsl tests/test_tv12_offline_quality.py
git commit -m "feat(tv12): add offline accumulation, resolve, and luminance compute shaders"
```

---

## Task 6: Offline Accumulation State in Rust

**Files:**
- Modify: `src/terrain/accumulation.rs`
- Modify: `src/terrain/renderer/core.rs` (~line 80-85)
- Test: builds successfully

This task adds the state structures without the methods. The methods come in Tasks 7-8.

- [ ] **Step 1: Extend accumulation.rs with OfflineAccumulationState**

Add to `src/terrain/accumulation.rs`:

```rust
/// TV12: Complete offline accumulation session state.
///
/// Owns all GPU resources for an active offline render session.
/// Only one session can be active per TerrainRenderer at a time.
pub struct OfflineAccumulationState {
    /// Ping-pong beauty accumulation textures (Rgba32Float)
    pub beauty_accum: [wgpu::Texture; 2],
    pub beauty_views: [wgpu::TextureView; 2],
    /// Which ping-pong buffer is the "current" accumulation (0 or 1)
    pub current_buffer: usize,
    /// Albedo accumulation (Rgba32Float, additive)
    pub albedo_accum: wgpu::Texture,
    pub albedo_view: wgpu::TextureView,
    /// Normal accumulation (Rgba32Float, additive, renormalized at resolve)
    pub normal_accum: wgpu::Texture,
    pub normal_view: wgpu::TextureView,
    /// Depth reference (R32Float, single sample from sample 0)
    pub depth_ref: wgpu::Texture,
    pub depth_ref_view: wgpu::TextureView,
    /// Scratch HDR render target for each sample (Rgba16Float)
    pub scratch_beauty: wgpu::Texture,
    pub scratch_beauty_view: wgpu::TextureView,
    /// Total accumulated samples so far
    pub total_samples: u32,
    /// Jitter sequence for camera offsets
    pub jitter: JitterSequence,
    /// Image dimensions
    pub width: u32,
    pub height: u32,
    /// TV12.2: Per-tile mean luminance from previous metrics call (for temporal convergence)
    pub prev_tile_means: Vec<f32>,
    /// Quarter-res luminance texture for metrics
    pub luminance_texture: wgpu::Texture,
    pub luminance_view: wgpu::TextureView,
    pub luminance_width: u32,
    pub luminance_height: u32,
}
```

Add a factory method:

```rust
impl OfflineAccumulationState {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        max_samples: u32,
        seed: Option<u64>,
    ) -> Self {
        let create_rgba32f = |label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };

        let create_rgba16f = |label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };

        let beauty_a = create_rgba32f("tv12.beauty_accum.a");
        let beauty_b = create_rgba32f("tv12.beauty_accum.b");
        let albedo = create_rgba32f("tv12.albedo_accum");
        let normal = create_rgba32f("tv12.normal_accum");
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tv12.depth_ref"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let scratch = create_rgba16f("tv12.scratch_beauty");

        let default_view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            beauty_views: [default_view(&beauty_a), default_view(&beauty_b)],
            beauty_accum: [beauty_a, beauty_b],
            current_buffer: 0,
            albedo_view: default_view(&albedo),
            albedo_accum: albedo,
            normal_view: default_view(&normal),
            normal_accum: normal,
            depth_ref_view: default_view(&depth),
            depth_ref: depth,
            scratch_beauty_view: default_view(&scratch),
            scratch_beauty: scratch,
            total_samples: 0,
            jitter: JitterSequence::new(max_samples, seed),
            width,
            height,
        }
    }

    /// Swap ping-pong buffers after accumulation.
    pub fn swap_buffers(&mut self) {
        self.current_buffer = 1 - self.current_buffer;
    }

    /// Index of the buffer that holds the current accumulated sum.
    pub fn read_buffer_idx(&self) -> usize {
        self.current_buffer
    }

    /// Index of the buffer to write the next accumulation result.
    pub fn write_buffer_idx(&self) -> usize {
        1 - self.current_buffer
    }
}
```

- [ ] **Step 2: Add offline state to TerrainScene**

In `src/terrain/renderer/core.rs`, add after the existing accumulation fields (~line 85):

```rust
    /// TV12: Offline accumulation session state (None when no session is active)
    pub(super) offline_state: Mutex<Option<crate::terrain::accumulation::OfflineAccumulationState>>,
    /// TV12: Offline accumulation compute pipeline
    pub(super) offline_accumulate_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) offline_accumulate_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// TV12: Offline resolve compute pipeline
    pub(super) offline_resolve_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) offline_resolve_bind_group_layout: Option<wgpu::BindGroupLayout>,
```

Initialize these as `Mutex::new(None)` / `None` in the `TerrainScene::new()` constructor.

- [ ] **Step 3: Build**

Run: `cd .worktrees/epic-12 && cargo build --features extension-module`
Expected: Builds successfully.

- [ ] **Step 4: Commit**

```bash
git add src/terrain/accumulation.rs src/terrain/renderer/core.rs
git commit -m "feat(tv12): add OfflineAccumulationState and TerrainScene offline fields"
```

---

## Task 7: Offline Render Methods (Rust)

**Files:**
- Create: `src/terrain/renderer/offline.rs`
- Modify: `src/terrain/renderer/mod.rs` (add `mod offline;`)
- Test: `tests/test_tv12_offline_quality.py`

This is the largest task. It implements all 7 offline `#[pymethods]` on `TerrainRenderer`.

- [ ] **Step 1: Write integration test for the offline pipeline**

Append to `tests/test_tv12_offline_quality.py`:

```python
@pytest.fixture(scope="module")
def offline_render_env():
    if not GPU_AVAILABLE:
        pytest.skip("GPU terrain rendering not available")

    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()
    heightmap = _build_bowl_heightmap()
    overlay = _build_overlay()

    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
        hdr_path = Path(tmp.name)
    try:
        _write_test_hdr(hdr_path)
        ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    finally:
        hdr_path.unlink(missing_ok=True)

    return {"renderer": renderer, "material_set": material_set,
            "ibl": ibl, "heightmap": heightmap, "overlay": overlay}


class TestOfflineAccumulation:
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_begin_accumulate_resolve_tonemap(self, offline_render_env):
        """TV12.1: Basic offline pipeline runs without error."""
        env = offline_render_env
        params = _make_offline_params(env["overlay"], aa_samples=4, aa_seed=42)
        r = env["renderer"]

        r.begin_offline_accumulation(params, env["heightmap"], env["material_set"], env["ibl"])
        result = r.accumulate_batch(4)
        assert result.total_samples == 4

        hdr, aov = r.resolve_offline_hdr()
        assert hdr.size == params.size
        hdr_np = hdr.to_numpy_f32()
        assert hdr_np.shape == (params.size[1], params.size[0], 4)
        assert hdr_np.dtype == np.float32

        frame = r.tonemap_offline_hdr(hdr)
        frame_np = frame.to_numpy()
        assert frame_np.shape == (params.size[1], params.size[0], 4)
        assert frame_np.dtype == np.uint8

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_determinism(self, offline_render_env):
        """TV12.1: Same seed produces identical output."""
        env = offline_render_env
        params = _make_offline_params(env["overlay"], aa_samples=4, aa_seed=42)
        r = env["renderer"]

        # First render
        r.begin_offline_accumulation(params, env["heightmap"], env["material_set"], env["ibl"])
        r.accumulate_batch(4)
        hdr1, _ = r.resolve_offline_hdr()
        frame1 = r.tonemap_offline_hdr(hdr1)
        np1 = frame1.to_numpy()

        # Second render (same seed)
        r.begin_offline_accumulation(params, env["heightmap"], env["material_set"], env["ibl"])
        r.accumulate_batch(4)
        hdr2, _ = r.resolve_offline_hdr()
        frame2 = r.tonemap_offline_hdr(hdr2)
        np2 = frame2.to_numpy()

        assert np.array_equal(np1, np2), "Same seed must produce identical output"

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_quality_improvement(self, offline_render_env):
        """TV12.1: Multi-sample accumulation reduces noise vs single sample."""
        env = offline_render_env

        # Single sample
        params1 = _make_offline_params(env["overlay"], aa_samples=1, aa_seed=42)
        r = env["renderer"]
        r.begin_offline_accumulation(params1, env["heightmap"], env["material_set"], env["ibl"])
        r.accumulate_batch(1)
        hdr1, _ = r.resolve_offline_hdr()
        frame1 = r.tonemap_offline_hdr(hdr1)

        # 16 samples
        params16 = _make_offline_params(env["overlay"], aa_samples=16, aa_seed=42)
        r.begin_offline_accumulation(params16, env["heightmap"], env["material_set"], env["ibl"])
        r.accumulate_batch(16)
        hdr16, _ = r.resolve_offline_hdr()
        frame16 = r.tonemap_offline_hdr(hdr16)

        # Multi-sample should be different (smoother edges)
        diff = _mean_abs_diff(frame1.to_numpy(), frame16.to_numpy())
        assert diff > 0.1, f"Multi-sample should visibly differ from single (diff={diff:.3f})"

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_session_guard_double_begin(self, offline_render_env):
        """TV12.1: begin_offline_accumulation while active raises error."""
        env = offline_render_env
        params = _make_offline_params(env["overlay"], aa_samples=4, aa_seed=42)
        r = env["renderer"]

        r.begin_offline_accumulation(params, env["heightmap"], env["material_set"], env["ibl"])
        with pytest.raises(RuntimeError, match="offline.*active"):
            r.begin_offline_accumulation(params, env["heightmap"], env["material_set"], env["ibl"])
        # Clean up: resolve + tonemap to end session
        hdr, _ = r.resolve_offline_hdr()
        r.tonemap_offline_hdr(hdr)
```

Add test helpers at the top of the file:

```python
import tempfile

def _build_bowl_heightmap(size: int = 128) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    bowl = 0.6 * (xx * xx + yy * yy)
    ridge = 0.2 * np.exp(-((xx - 0.5) ** 2 * 20.0 + (yy + 0.1) ** 2 * 13.0))
    heightmap = bowl + ridge
    heightmap -= float(heightmap.min())
    heightmap /= max(float(heightmap.max()), 1e-6)
    return heightmap.astype(np.float32)

def _build_overlay():
    cmap = f3d.Colormap1D.from_stops(
        stops=[(0.0, "#16341a"), (0.35, "#40692f"), (0.65, "#8b7b53"), (1.0, "#f3f6fb")],
        domain=(0.0, 1.0),
    )
    return f3d.OverlayLayer.from_colormap1d(cmap, strength=1.0)

def _write_test_hdr(path, width=8, height=4):
    with open(str(path), "wb") as handle:
        handle.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode())
        for _ in range(height * width):
            handle.write(bytes([128, 128, 128, 128]))

def _mean_abs_diff(a, b):
    return float(np.mean(np.abs(a[..., :3].astype(np.float32) - b[..., :3].astype(np.float32))))

def _make_offline_params(overlay, *, aa_samples=4, aa_seed=42):
    from forge3d.terrain_params import PomSettings, make_terrain_params_config
    config = make_terrain_params_config(
        size_px=(224, 224), render_scale=1.0, terrain_span=4.0, msaa_samples=1,
        z_scale=2.0, exposure=1.0, domain=(0.0, 1.0), albedo_mode="colormap",
        colormap_strength=1.0, ibl_enabled=True, light_azimuth_deg=138.0,
        light_elevation_deg=16.0, sun_intensity=0.8, cam_radius=6.2,
        cam_phi_deg=138.0, cam_theta_deg=58.0, fov_y_deg=48.0,
        camera_mode="screen", overlays=[overlay],
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        aa_samples=aa_samples, aa_seed=aa_seed,
    )
    return f3d.TerrainRenderParams(config)
```

- [ ] **Step 2: Implement offline.rs with all 6 methods**

Create `src/terrain/renderer/offline.rs`. This file implements `begin_offline_accumulation`, `accumulate_batch`, `read_accumulation_metrics`, `resolve_offline_hdr`, `upload_hdr_frame`, and `tonemap_offline_hdr` as `#[pymethods]` on `TerrainRenderer`.

Key implementation notes:
- `begin_offline_accumulation`: Check `offline_state` is None (error if active), create `OfflineAccumulationState` with **owned GPU resources** (upload heightmap texture, create IBL bind group, copy decoded params — do NOT store borrowed Python references), lazy-create offline compute pipelines if not yet compiled.
- `accumulate_batch`: Loop N times — for each sample: set `offline_hdr_output=true` on params, get jitter offset, apply jitter to projection, render terrain to the scratch Rgba16Float target using the **HDR pipeline variant** (not the standard Rgba8UnormSrgb pipeline), dispatch `offline_accumulate` compute shader to blend scratch into accum buffer (ping-pong), increment sample count.
- `read_accumulation_metrics`: Dispatch `offline_luminance.wgsl` to produce quarter-res luminance map, read back via staging buffer, compute per-tile mean luminance on CPU, compare to `prev_tile_means` for **temporal convergence** (not spatial variance), update `prev_tile_means`.
- `resolve_offline_hdr`: Dispatch `offline_resolve` compute shader for beauty (Rgba32Float→Rgba16Float)/albedo/normal (with renormalization), copy depth reference. Return `(HdrFrame, AovFrame)` — both own their textures independently.
- `upload_hdr_frame`: Create Rgba16Float texture, write numpy data via `queue.write_texture`.
- `tonemap_offline_hdr`: Run existing `postprocess_tonemap.wgsl` on the HdrFrame input, write to Rgba8UnormSrgb texture, create Frame, call `end_offline_accumulation()` internally.
- `end_offline_accumulation`: Release all offline session resources, set `offline_session_active = false`. Idempotent (no-op when no session active). Called by `tonemap_offline_hdr` automatically, but also available explicitly for cleanup on error or HDR-only export.

Register the module in `src/terrain/renderer/mod.rs`: add `mod offline;`.

- [ ] **Step 3: Build**

Run: `cd .worktrees/epic-12 && cargo build --features extension-module`
Expected: Builds successfully.

- [ ] **Step 4: Run integration tests**

Run: `cd .worktrees/epic-12 && python -m pytest tests/test_tv12_offline_quality.py::TestOfflineAccumulation -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/terrain/renderer/offline.rs src/terrain/renderer/mod.rs \
  tests/test_tv12_offline_quality.py
git commit -m "feat(tv12): implement offline accumulation pipeline (begin/accumulate/metrics/resolve/tonemap)"
```

---

## Task 8: Adaptive Sampling Tests

**Files:**
- Test: `tests/test_tv12_offline_quality.py`

- [ ] **Step 1: Write adaptive sampling tests**

```python
class TestAdaptiveSampling:
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_metrics_queryable(self, offline_render_env):
        """TV12.2: Metrics are valid after accumulation."""
        env = offline_render_env
        params = _make_offline_params(env["overlay"], aa_samples=8, aa_seed=42)
        r = env["renderer"]
        r.begin_offline_accumulation(params, env["heightmap"], env["material_set"], env["ibl"])
        r.accumulate_batch(4)
        # First call establishes baseline (prev_tile_means starts empty)
        metrics1 = r.read_accumulation_metrics(0.001)
        assert metrics1.total_samples == 4
        # Second batch + metrics shows temporal convergence
        r.accumulate_batch(4)
        metrics2 = r.read_accumulation_metrics(0.001)
        assert metrics2.total_samples == 8
        assert metrics2.mean_delta >= 0.0
        assert metrics2.p95_delta >= 0.0
        assert metrics2.max_tile_delta >= metrics2.p95_delta
        assert 0.0 <= metrics2.converged_tile_ratio <= 1.0
        hdr, _ = r.resolve_offline_hdr()
        r.tonemap_offline_hdr(hdr)

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_max_samples_respected(self, offline_render_env):
        """TV12.2: Accumulation respects max_samples."""
        env = offline_render_env
        params = _make_offline_params(env["overlay"], aa_samples=4, aa_seed=42)
        r = env["renderer"]
        r.begin_offline_accumulation(params, env["heightmap"], env["material_set"], env["ibl"])
        r.accumulate_batch(4)
        result = r.accumulate_batch(4)
        assert result.total_samples == 8  # Batches add up
        hdr, _ = r.resolve_offline_hdr()
        r.tonemap_offline_hdr(hdr)
```

- [ ] **Step 2: Run tests**

Expected: PASS (they exercise the metrics and batch accumulation from Task 7).

- [ ] **Step 3: Commit**

```bash
git add tests/test_tv12_offline_quality.py
git commit -m "test(tv12): add adaptive sampling metric and batch tests"
```

---

## Task 9: Python Offline Controller

**Files:**
- Create: `python/forge3d/offline.py`
- Modify: `python/forge3d/__init__.py` (export `render_offline`, `OfflineResult`, `OfflineProgress`)
- Test: `tests/test_tv12_offline_quality.py`

- [ ] **Step 1: Write failing test for render_offline**

```python
from forge3d.offline import render_offline, OfflineResult
from forge3d.terrain_params import OfflineQualitySettings

class TestOfflineController:
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_render_offline_basic(self, offline_render_env):
        env = offline_render_env
        params = _make_offline_params(env["overlay"], aa_samples=8, aa_seed=42)
        settings = OfflineQualitySettings(enabled=True, batch_size=4)
        result = render_offline(
            env["renderer"], env["material_set"], env["ibl"],
            params, env["heightmap"], settings=settings,
        )
        assert isinstance(result, OfflineResult)
        assert result.frame is not None
        assert result.hdr_frame is not None
        assert result.aov_frame is not None
        assert result.metadata["samples_used"] == 8
        assert result.metadata["denoiser_used"] == "none"

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_render_offline_with_progress(self, offline_render_env):
        env = offline_render_env
        params = _make_offline_params(env["overlay"], aa_samples=8, aa_seed=42)
        settings = OfflineQualitySettings(enabled=True, batch_size=4)
        progress_log = []
        result = render_offline(
            env["renderer"], env["material_set"], env["ibl"],
            params, env["heightmap"], settings=settings,
            progress_callback=lambda p: progress_log.append(p),
        )
        assert len(progress_log) == 2  # 2 batches of 4
        assert progress_log[-1].samples_so_far == 8

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_render_offline_adaptive(self, offline_render_env):
        """Adaptive mode with a flat terrain should converge quickly."""
        env = offline_render_env
        flat = np.full((128, 128), 0.5, dtype=np.float32)
        params = _make_offline_params(env["overlay"], aa_samples=4, aa_seed=42)
        settings = OfflineQualitySettings(
            enabled=True, adaptive=True, min_samples=4,
            max_samples=32, batch_size=4, target_variance=0.1,
        )
        result = render_offline(
            env["renderer"], env["material_set"], env["ibl"],
            params, flat, settings=settings,
        )
        assert result.metadata["samples_used"] <= 32
```

- [ ] **Step 2: Implement offline.py**

Create `python/forge3d/offline.py` with `render_offline()`, `OfflineResult`, `OfflineProgress` as specified in the design doc Section 8. The controller loop:

1. Calls `renderer.begin_offline_accumulation()`
2. Loops with `accumulate_batch()` + optional `read_accumulation_metrics()` for convergence
3. Calls `resolve_offline_hdr()`
4. Optionally applies OIDN or A-trous denoising on the HDR numpy data
5. Calls `tonemap_offline_hdr()`
6. Returns `OfflineResult`

- [ ] **Step 3: Run tests**

Run: `cd .worktrees/epic-12 && python -m pytest tests/test_tv12_offline_quality.py::TestOfflineController -v`
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add python/forge3d/offline.py python/forge3d/__init__.py tests/test_tv12_offline_quality.py
git commit -m "feat(tv12): implement Python offline render controller with adaptive sampling"
```

---

## Task 10: OIDN Integration Tests

**Files:**
- Test: `tests/test_tv12_offline_quality.py`

- [ ] **Step 1: Write OIDN integration tests**

```python
class TestOidnIntegration:
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_oidn_fallback_to_atrous(self, offline_render_env):
        """When oidn not installed, method='oidn' falls back to atrous."""
        if oidn_available():
            pytest.skip("oidn is installed")
        env = offline_render_env
        params = _make_offline_params(
            env["overlay"], aa_samples=4, aa_seed=42,
            denoise_method="oidn",
        )
        settings = OfflineQualitySettings(enabled=True, batch_size=4)
        with pytest.warns(UserWarning, match="oidn.*falling back"):
            result = render_offline(
                env["renderer"], env["material_set"], env["ibl"],
                params, env["heightmap"], settings=settings,
            )
        assert result.metadata["denoiser_used"] == "atrous"

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_no_denoise_baseline(self, offline_render_env):
        """method='none' matches non-denoised resolve."""
        env = offline_render_env
        params = _make_offline_params(env["overlay"], aa_samples=4, aa_seed=42)
        settings = OfflineQualitySettings(enabled=True, batch_size=4)
        result = render_offline(
            env["renderer"], env["material_set"], env["ibl"],
            params, env["heightmap"], settings=settings,
        )
        assert result.metadata["denoiser_used"] == "none"
```

Update `_make_offline_params` to accept an optional `denoise_method` kwarg and pass it as `denoise=DenoiseSettings(enabled=True, method=denoise_method)` when provided.

- [ ] **Step 2: Run tests**

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_tv12_offline_quality.py
git commit -m "test(tv12): add OIDN integration and fallback tests"
```

---

## Task 11: Example Demo

**Files:**
- Create: `examples/terrain_tv12_offline_quality_demo.py`

- [ ] **Step 1: Create the demo script**

Follow the pattern of `terrain_tv5_probe_lighting_demo.py`. The demo should:

1. Load a real DEM from `assets/tif/dem_rainier.tif`
2. Render 3 outputs:
   - Single-sample baseline (aa_samples=1)
   - Multi-sample offline quality (aa_samples=16)
   - Multi-sample + A-trous denoised
3. Compose a side-by-side comparison PNG
4. Save individual PNGs and an EXR of the HDR output
5. Print convergence stats and sample counts
6. Support CLI args: `--dem`, `--output-dir`, `--width`, `--height`, `--samples`

- [ ] **Step 2: Run the demo**

Run: `cd .worktrees/epic-12 && python examples/terrain_tv12_offline_quality_demo.py --output-dir examples/out/terrain_tv12_offline_quality`
Expected: PNGs and EXR written to output directory. Verify the multi-sample output is visually smoother than the single-sample baseline.

- [ ] **Step 3: Commit**

```bash
git add examples/terrain_tv12_offline_quality_demo.py
git commit -m "feat(tv12): add terrain offline quality demo with real DEM comparison renders"
```

---

## Task 12: Image Output Verification

**Files:**
- Test: `tests/test_tv12_offline_quality.py`

- [ ] **Step 1: Write image output tests**

```python
class TestImageOutput:
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_png_save(self, offline_render_env, tmp_path):
        env = offline_render_env
        params = _make_offline_params(env["overlay"], aa_samples=4, aa_seed=42)
        settings = OfflineQualitySettings(enabled=True, batch_size=4)
        result = render_offline(
            env["renderer"], env["material_set"], env["ibl"],
            params, env["heightmap"], settings=settings,
        )
        png_path = tmp_path / "test_offline.png"
        result.frame.save(str(png_path))
        assert png_path.exists()
        assert png_path.stat().st_size > 100

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_exr_save(self, offline_render_env, tmp_path):
        env = offline_render_env
        params = _make_offline_params(env["overlay"], aa_samples=4, aa_seed=42)
        settings = OfflineQualitySettings(enabled=True, batch_size=4)
        result = render_offline(
            env["renderer"], env["material_set"], env["ibl"],
            params, env["heightmap"], settings=settings,
        )
        exr_path = tmp_path / "test_offline.exr"
        result.hdr_frame.save(str(exr_path))
        assert exr_path.exists()
        assert exr_path.stat().st_size > 100

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_hdr_numpy_values_plausible(self, offline_render_env):
        env = offline_render_env
        params = _make_offline_params(env["overlay"], aa_samples=4, aa_seed=42)
        settings = OfflineQualitySettings(enabled=True, batch_size=4)
        result = render_offline(
            env["renderer"], env["material_set"], env["ibl"],
            params, env["heightmap"], settings=settings,
        )
        hdr = result.hdr_frame.to_numpy_f32()
        assert hdr.dtype == np.float32
        assert np.all(np.isfinite(hdr))
        assert np.max(hdr[..., :3]) > 0.0  # Not all black
```

- [ ] **Step 2: Run tests**

Run: `cd .worktrees/epic-12 && python -m pytest tests/test_tv12_offline_quality.py::TestImageOutput -v`
Expected: All PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_tv12_offline_quality.py
git commit -m "test(tv12): add image output verification tests (PNG, EXR, HDR numpy)"
```

---

## Task 13: Documentation

**Files:**
- Create: `docs/tv12-terrain-offline-render-quality.md`

- [ ] **Step 1: Write feature documentation**

Cover:
- What TV12 adds (offline accumulation, adaptive sampling, OIDN denoising)
- API reference: `render_offline()`, `OfflineQualitySettings`, `HdrFrame`
- Quality tiering: none → atrous → oidn
- Usage examples with code snippets
- Configuration options and their defaults
- How to install OIDN (`pip install pyoidn`)

- [ ] **Step 2: Commit**

```bash
git add docs/tv12-terrain-offline-render-quality.md
git commit -m "docs(tv12): add terrain offline render quality feature documentation"
```

---

## Task 14: Final Integration and Cleanup

- [ ] **Step 1: Run the full test suite**

Run: `cd .worktrees/epic-12 && python -m pytest tests/test_tv12_offline_quality.py -v`
Expected: All tests pass.

- [ ] **Step 2: Run existing tests to verify no regressions**

Run: `cd .worktrees/epic-12 && python -m pytest tests/ -v --timeout=120`
Expected: No regressions. Existing tests continue to pass.

- [ ] **Step 3: Run the demo and verify output images**

Run: `cd .worktrees/epic-12 && python examples/terrain_tv12_offline_quality_demo.py`
Expected: PNGs written, visually correct (multi-sample smoother than single-sample).

- [ ] **Step 4: Bump version**

Update `python/forge3d/__init__.py`: `__version__ = "1.17.0"`.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: bump version to 1.17.0 for TV12 terrain offline render quality release"
```
