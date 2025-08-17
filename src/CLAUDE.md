# src/ — Memory for Rust core (wgpu + PyO3) (v2)

Rust owns GPU, memory, and determinism. Python is a thin veneer. Follow these rules to avoid cross-backend surprises.  

## What Claude tends to forget

* **Single** global GPU context (`OnceCell`)—no per-object devices.
* Correct **256-byte alignment** rules for both **readbacks and writes**.
* Release the **GIL** around GPU work even when invoked from Python.
* Factor out **readback/upload utilities**; don’t duplicate padding logic.
* Use a central **error enum** with categories → consistent `PyErr`.

## Device & adapter

* Global `OnceCell<WgpuContext { instance, adapter, device, queue }>`; reuse everywhere (Renderer, Scene, etc.). Don’t create per-Scene devices.  
* Prefer hardware; if `prefer_software` (arg or `VULKAN_FORGE_PREFER_SOFTWARE=1`), request fallback adapter.
* Features: conservative; `Limits::downlevel_defaults()`.
* Label resources for debugging.

## Off-screen targets & passes

* Format: `Rgba8UnormSrgb`, no MSAA in MVP. One pass, one color target.
* Full-frame viewport & scissor. Constant clear color.

## Alignment (corrected)

* **Readback (texture→buffer)**:

  * `bpp = 4`, `unpadded = width * bpp`, `padded = ceil(unpadded / 256) * 256`.
  * Allocate/readback with `bytes_per_row = padded`; when copying to CPU, strip the per-row padding.  
* **Writes (CPU→texture via `Queue::write_texture`)**:

  * When `height > 1`, **`bytes_per_row` must be a multiple of 256**. Build a temporary **padded** upload buffer and copy row-by-row into a 256-byte stride, then pass `bytes_per_row = padded`. *(This corrects the earlier memo that claimed `write_texture` doesn’t require 256B alignment.)*  

### Reusable helpers (copy-paste)

```rust
#[inline]
fn padded_bpr(width: u32, bytes_per_pixel: u32) -> u32 {
    let unpadded = width.saturating_mul(bytes_per_pixel);
    let a = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    ((unpadded + a - 1) / a) * a
}

fn upload_r32f_padded(queue: &wgpu::Queue, tex: &wgpu::Texture, width: u32, height: u32, data_f32: &[f32]) {
    let row = width * 4;
    let bpr = padded_bpr(width, 4);
    let mut tmp = vec![0u8; (bpr * height) as usize];
    let src = bytemuck::cast_slice::<f32, u8>(data_f32);
    for y in 0..height as usize {
        let s = y * row as usize;
        let d = y * bpr as usize;
        tmp[d..d + row as usize].copy_from_slice(&src[s..s + row as usize]);
    }
    queue.write_texture(
        wgpu::ImageCopyTexture { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        &tmp,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(std::num::NonZeroU32::new(bpr).unwrap().into()),
            rows_per_image: Some(std::num::NonZeroU32::new(height).unwrap().into()),
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
}

fn read_rgba_unpadded(
    device: &wgpu::Device, queue: &wgpu::Queue,
    src: &wgpu::Texture, width: u32, height: u32
) -> Vec<u8> {
    let bpp = 4;
    let bpr = padded_bpr(width, bpp);
    let size = (bpr * height) as wgpu::BufferAddress;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"), size, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("copy-enc") });
    enc.copy_texture_to_buffer(
        wgpu::ImageCopyTexture { texture: src, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::ImageCopyBuffer { buffer: &buf, layout: wgpu::ImageDataLayout {
            offset: 0, bytes_per_row: Some(std::num::NonZeroU32::new(bpr).unwrap().into()),
            rows_per_image: Some(std::num::NonZeroU32::new(height).unwrap().into()),
        }},
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
    queue.submit(Some(enc.finish()));
    device.poll(wgpu::Maintain::Wait);

    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();

    let unpadded = (width * bpp) as usize;
    let mut out = vec![0u8; unpadded * height as usize];
    for y in 0..height as usize {
        let s = y * bpr as usize;
        let d = y * unpadded;
        out[d..d + unpadded].copy_from_slice(&mapped[s..s + unpadded]);
    }
    drop(mapped);
    buf.unmap();
    out
}
````

## Samplers & R32Float heights

* Bind R32F heights with a **non-filtering (Nearest)** sampler to satisfy backends that reject filtering on 32-bit float textures. Document this in code where the sampler is created (once), and reuse it.

## Uniforms & bind groups

* Keep WGSL-compatible alignment (`mat4x4<f32>` = 64B).
* Group 0: globals (view/proj, sun\_dir, exposure, spacing, height\_range, exaggeration).
* Group 1: height texture + sampler.
* Group 2: colormap texture + sampler.

## Error handling

* Define a `thiserror::Error` enum with categories: `Device`, `Upload`, `Render`, `Readback`, `IO`.
* Implement `From<RenderError> for PyErr` that prefixes messages with `[Render]`, etc., for consistent Python errors.

## PyO3 specifics

* Build with `abi3 (py>=3.10)`.
* Keep Python-visible methods short; wrap GPU work with `py.allow_threads`.
* Never panic across FFI boundaries—convert to `PyErr` first.

## Performance & metrics

* Reuse GPU buffers/bind groups. Use persistent readback buffers sized to current `padded_bpr * height`.
* Provide `render_metrics()` returning encode/gpu/readback timings and byte counts.

## Testing & validation

* Smoke tests on Vulkan/Metal/DX12; conditionally disable features per backend if needed (don’t regress others).

* Unit tests for padding helpers; end-to-end small renders; PNG↔NumPy round-trips.

* **Context proliferation** — per-object `Device/Queue` creation crept in for convenience. Our policy is a **single `OnceCell` GPU context** for production code; per-object devices are allowed *only* for hermetic tests/demos and must be explicitly labeled as such.

* **256-byte alignment (writes too!)** — we initially overlooked that **`Queue::write_texture` requires 256-byte `bytes_per_row`** when `height > 1`; write paths must build a padded staging image.

* **Clip-space invariants** — defaulting some codepaths to GL clip space while others assumed WGPU introduced subtle depth mismatches; we must centralize the GL→WGPU remap and document it.

* **Error taxonomy** — ad-hoc `map_err(|e| …)` led to inconsistent Python errors. We need a central error enum and a single `ToPyErr` conversion.

* **WGSL robustness** — normal correction needed explicit per-component division and zero-guarding; varyings and uniform lane contracts should be documented to prevent regressions.

---

## Prescriptive fixes (copy-paste, no code changes required to adopt)

### A) Python surface: exact patterns

**Release the GIL around GPU work (template)**

```rust
// Inside #[pymethods] impl blocks
pub fn render_rgba<'py>(&mut self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, numpy::PyArray3<u8>>> {
    // Do ALL GPU encode/submit/map inside allow_threads
    let pixels = py.allow_threads(|| -> Result<Vec<u8>, RenderError> {
        self.render_once_and_readback()
    }).map_err(|e| e.to_py_err())?;

    let arr = ndarray::Array3::from_shape_vec((self.height as usize, self.width as usize, 4), pixels)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py))
}
```

**Boundary errors must say “expected …; got …; how to fix”**

```rust
return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
    "heightmap must be a 2-D NumPy array, dtype=float32|float64, C-contiguous; \
     expected (H,W) row-major; got ndim={ndim}, dtype={dtype}, flags.contiguous={contig}. \
     Use `np.ascontiguousarray(arr, dtype=np.float32)`."
)));
```

**PyPI/maturin release gate (single source of truth)**

Before tagging, run this checklist (no exceptions):

1. Bump **all** versions: `pyproject.toml`, `vulkan_forge/__init__.py`, `Cargo.toml`, `CHANGELOG.md`, and mention in `README.md`.
2. `pytest -q` (with backend-aware skips) must pass locally.
3. Build wheels: abi3 (py>=3.10), manylinux2014, strip, thin LTO.
4. Attach artifacts, tag, and paste the *Highlights* block from CHANGELOG.

**Markers for selective testing**

* Add `pytest.mark.camera`, `pytest.mark.terrain`, `pytest.mark.gpu`; document `pytest -q -m camera` etc.
* Skip heuristics: if no adapter, mark xfail/skip with an explicit reason and backend shown in the message.

---

### B) Rust core: exact patterns

**Global GPU context policy**

* Use a single `OnceCell<WgpuContext>` for production paths (renderer, scenes).
* If a sample/test constructs its own `Device/Queue`, add a doc comment:
  “*Test-only device for hermetic isolation; do not cargo-cult to production paths.*”

**Alignment helpers (authoritative)**

```rust
#[inline]
fn padded_bpr(width: u32, bpp: u32) -> u32 {
    let a = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let unpadded = width.saturating_mul(bpp);
    ((unpadded + a - 1) / a) * a
}
```

*Readback* and *Upload* helpers shared in one module (call from everywhere).

**Clip-space invariant**

* Always construct GL-style projection then apply `gl_to_wgpu()` when `clip_space="wgpu"`.
* Keep `gl_to_wgpu()` a function (not a `const`) to avoid glam const-init edge cases.
* Add a one-liner doc to any method that sets matrices: “*Assumes WGPU `[0,1]` depth unless `'gl'` is requested.*”

**Error enum (skeleton)**

```rust
#[derive(thiserror::Error, Debug)]
pub enum RenderError {
    #[error("device: {0}")] Device(String),
    #[error("upload: {0}")] Upload(String),
    #[error("render: {0}")] Render(String),
    #[error("readback: {0}")] Readback(String),
    #[error("io: {0}")] IO(String),
}

impl RenderError {
    pub fn to_py_err(self) -> pyo3::PyErr {
        use pyo3::exceptions::PyRuntimeError;
        PyRuntimeError::new_err(format!("[{}] {}", match &self {
            RenderError::Device(_) => "Device",
            RenderError::Upload(_) => "Upload",
            RenderError::Render(_) => "Render",
            RenderError::Readback(_) => "Readback",
            RenderError::IO(_) => "IO",
        }, self))
    }
}
```

**WGSL safety rails**

* Normal correction: per-component division with `max(denom, 1e-8)` guards.
* Varyings: explicitly list what later stages will need (e.g., `world_pos`, `world_nrm`, `height`).
* Uniform contract: maintain a lane map (see below) and test it.

**Uniform lane map (44 floats from 176 bytes) — canonical**

Lane index → meaning (row-major view then proj; then 8 lanes of globals):

```
0..15  = view (mat4x4, column-major in memory; exported row-major to NumPy)
16..31 = proj (same as above)
32..35 = sun_exposure: [sun_dir.x, sun_dir.y, sun_dir.z, exposure]
36..39 = spacing_h_exag_pad: [spacing, height_range, exaggeration, 0.0]
40..43 = _pad_tail (zeroed)
```

Add tests that assert lanes 36..39 match `[spacing, h_range, exaggeration, 0.0]` within `1e-6`.

---

## Addendum v3 — WGSL/clip-space/error taxonomy (append-only)

* Consolidate GL↔WGPU projection remap.
* Centralize error to `RenderError` → `PyErr`.
* Normalize varyings and uniform layout docstrings.

---

## Addendum v4 — Build errors hardening (append-only)

> Purpose: lock in fixes for the exact compiler/test failures we hit so they don’t recur.

### 1) `PyRuntimeError` usage (no `PyRuntime`)

* Always construct errors as `pyo3::exceptions::PyRuntimeError::new_err("…")`.
* Do not alias a non-existent `PyRuntime`. This caused multiple `E0433` failures.

### 2) PyO3 return types (Bound vs `&PyArray`)

* Inside `#[pymethods]`/`#[pyfunction]` returns of NumPy arrays must use `Bound<'py, …>`.
* Example signatures:

```rust
// method
pub fn render_rgba<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray3<u8>>> { … }

// function
#[pyfunction]
pub fn grid_generate<'py>(py: Python<'py>, …) -> PyResult<(Bound<'py, PyArray2<f32>>,
                                                           Bound<'py, PyArray2<f32>>,
                                                           Bound<'py, PyArray1<u32>>)> { … }
```

* When turning ndarray → NumPy, use `.to_pyarray_bound(py)` (preferred) or `.into_pyarray_bound(py)` with `use numpy::IntoPyArray;`.

### 3) NumPy contiguity trait import

* If you call `.is_contiguous()` on `PyReadonlyArray*`, **import**:
  `use numpy::PyUntypedArrayMethods;`

### 4) `write_texture` upload alignment

* For `R32Float` (or any multi-row upload), **pad** rows to 256B:

```rust
let bpr = padded_bpr(width, 4);
let mut tmp = vec![0u8; (bpr * height) as usize];
let src = bytemuck::cast_slice::<f32, u8>(&heights);
for y in 0..height as usize {
    let s = y * (width * 4) as usize;
    let d = y * bpr as usize;
    tmp[d..d + (width*4) as usize].copy_from_slice(&src[s..s + (width*4) as usize]);
}
queue.write_texture(
    wgpu::ImageCopyTexture { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
    &tmp,
    wgpu::ImageDataLayout {
        offset: 0,
        bytes_per_row: Some(std::num::NonZeroU32::new(bpr).unwrap().into()),
        rows_per_image: Some(std::num::NonZeroU32::new(height).unwrap().into()),
    },
    wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
);
```

### 5) Contract for terrain GPU reads

* `debug_read_height_patch` returns a zero-filled array when the texture is absent.
* `read_full_height_texture` should follow the same zero-fill convention (or clearly document and maintain whichever convention tests rely on). Do **not** change these silently.

### 6) GIL discipline in core paths

* Any method that encodes/submits/copies/reads GPU data should be structured to allow `py.allow_threads` at the boundary method (Python-visible layer) and keep the core routines pure Rust (return `Result<T, RenderError>`).

### 7) Misc build hygiene

* If `ImageDataLayout` expects `Option<u32>` (not `Option<NonZeroU32>` in your wgpu build), use `.into()` on `NonZeroU32` values (the helper snippets already show this).
* Label resources and encoders (`label: Some("…")`) to ease backend debugging.
* Keep `#[pymodule]` args modern: `fn module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()>`.
