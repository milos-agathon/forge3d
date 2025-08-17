# python/ — Memory for Python surface & packaging

This directory is the thin Python API for `vulkan_forge`. The Rust core is exposed via **PyO3** as `vulkan_forge._vulkan_forge` and re-exported from `vulkan_forge/__init__.py`. Keep the surface minimal, explicit, and deterministic.  

## What Claude tends to forget

* Release the **GIL** during heavy GPU work (encode/submit/map/upload/download).
* Keep **API + docs in sync** (constructor flags like `prefer_software`).
* Enforce **dtype/shape/contiguity** at the boundary with precise, actionable errors.
* Keep **versions** synchronized across README, CHANGELOG, `__init__.py`, `pyproject.toml`, and `Cargo.toml`.

## API principles

* No hidden threads; no implicit globals in Python.
* Validate **dtypes/shapes/contiguity** and fail fast with `PyRuntimeError("…; expected …; got …")`.
* Prefer **zero-copy** or single-copy paths from NumPy to Rust.
* Expose lightweight **adapter/device info** and **metrics** functions.

## GIL discipline (copy-paste)

Wrap GPU work from Rust with `py.allow_threads(|| { … })`:

```rust
#[pymethods]
impl Renderer {
    pub fn render_rgba<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray3<u8>>> {
        let pixels = py.allow_threads(|| -> Result<Vec<u8>, RenderError> {
            self.render_once_and_readback()
        }).map_err(|e| e.to_py_err())?;
        let arr = ndarray::Array3::from_shape_vec((self.h as usize, self.w as usize, 4), pixels)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray_bound(py))
    }
}
````

(Apply the same pattern to PNG writes, height uploads, and debug readbacks.)

## Boundary checks

* DEM: accept `float32` (preferred) and `float64`; cast once in Rust.
* Enforce **C-contiguous**: `(H,W)` for DEM; `(H,W,3|4)` for images.
* Raise `PyRuntimeError` with a message that states *expected vs got* and the fix.

## NumPy ↔ PNG rules

* `png_to_numpy(path) → (H,W,4) uint8` (RGBA).
* `numpy_to_png(path, arr)` accepts `(H,W)`, `(H,W,3)`, `(H,W,4)` `uint8`.
* Reject non-contiguous arrays with a clear “array must be C-contiguous” message.
* Do **not** apply color transforms; treat bytes as sRGB payload.

## Examples & metrics

* Deterministic examples with fixed seeds. Save `env.json` and small **metrics** dict (encode/gpu/readback ms, bytes) next to outputs for reproducibility.

## Packaging (maturin) — exact checklist

1. **Version bump** in:

   * `pyproject.toml` (`version = …`)
   * `vulkan_forge/__init__.py` (`__version__ = "…"`)
   * `Cargo.toml` (`version = …`)
   * `CHANGELOG.md`, `README.md`
2. Build wheels:

   * abi3 (py>=3.10), `manylinux2014` on Linux.
   * `strip = true`, `lto = "thin"` in Cargo profiles.
3. `pytest -q` (with skips when backends unavailable).
4. Upload wheels and tag release.

(These rules extend the prior packaging guidance with a concrete, error-proof release checklist.)

## Style & lint

* Type-hint public APIs; docstrings short & imperative.
* Run `ruff` + `black` in pre-commit.

# Addendum v3 — Lessons from T2/T3 (Do-not-delete; append-only)

This addendum captures *observed blindspots* from our recent back-and-forth and turns them into **hard rules**, **copy-paste snippets**, and **review checklists**. It does **not** change any source files; it improves our shared memory so future edits land correctly on the first try.

## What actually tripped us up (explicit blindspots)

### 1) `python/` — Memory for Python surface & packaging

* **GIL discipline gaps** — we sometimes returned NumPy without `py.allow_threads` around the GPU work (encode/submit/map/upload/readback). This can stall other Python threads during long GPU IO.
* **Boundary enforcement drift** — dtype/shape/contiguity checks were not always enforced with precise *expected vs got* errors, which made debugging user inputs harder.
* **Version coherency** — version bumps were not propagated across `pyproject.toml`, `__init__.py`, `Cargo.toml`, and `CHANGELOG.md`; changelog entries occasionally lagged the code.
* **Test selection clarity** — camera/terrain subsets were run informally; we should provide a stable marker‐based selection (`-m gpu` / `-m camera`) and document skips for backends not present.
* **Docs/API mismatch risk** — READMEs and docstrings did not always reflect defaults (e.g., clip-space, colormap names), which confused users reading the Python surface first.

---

# Addendum v4 — Fixes from recent build & test failures (append-only)

> Purpose: encode concrete rules that prevent the specific errors we saw (PyO3 signatures, NumPy trait imports, platform-specific commands, and test contracts). Copy/paste these into PRs as acceptance criteria.

## A) PyO3 return types & signatures (Bound everywhere)

* In `#[pymethods]` functions that **return NumPy arrays**, always return:

  * `PyResult<Bound<'py, numpy::PyArrayN<T>>>` — **not** `&PyArray…`.
  * Use `arr.into_pyarray_bound(py)` or `arr.to_pyarray_bound(py)` consistently.
  * If you use `into_pyarray_bound`, **import the trait**:
    `use numpy::IntoPyArray;`
  * Prefer `to_pyarray_bound` for clarity (it doesn’t require moving the ndarray).

**Template:**

```rust
use numpy::{PyArray1, PyArray2, PyArray3, IntoPyArray}; // or prefer .to_pyarray_bound

#[pyfunction]
pub fn make_mesh<'py>(py: Python<'py>) -> PyResult<(Bound<'py, PyArray2<f32>>,
                                                    Bound<'py, PyArray2<f32>>,
                                                    Bound<'py, PyArray1<u32>>)> {
    let pos = ndarray::Array2::<f32>::zeros((n,3));
    let uv  = ndarray::Array2::<f32>::zeros((n,2));
    let idx = vec![0u32; m];
    Ok((
        pos.to_pyarray_bound(py),
        uv.to_pyarray_bound(py),
        PyArray1::from_vec_bound(py, idx),
    ))
}
```

## B) NumPy contiguity checks (use the right trait)

* To call `.is_contiguous()` / `.is_c_contiguous()` on **readonly arrays**, bring the trait into scope:

```rust
use numpy::PyUntypedArrayMethods;

let a32: numpy::PyReadonlyArray2<f32> = arr.readonly();
if !a32.is_contiguous() { // portable across 0.21 line; prefer this name
    return Err(PyRuntimeError::new_err("array must be C-contiguous (row-major)"));
}
```

* Stick to `.is_contiguous()` (preferred spelling). Only use `.is_c_contiguous()` if you must target pre-0.21 conventions.

## C) Error type: never spell `PyRuntime`

* Always raise via `pyo3::exceptions::PyRuntimeError::new_err(...)`.
  Don’t invent `PyRuntime`; don’t alias a non-existing type. This was the root of several `E0433` errors.

## D) Test contracts for terrain reads

* `debug_read_height_patch(...)` — **returns zeros** if no GPU height texture is present.
* `read_full_height_texture()` — **also returns zeros** until `upload_height_r32f()` runs.
  (This avoids order-dependent test failures. If you need a stricter contract, add *another* method that raises instead, but do not change these.)

## E) Cross-platform packaging/setup snippets

* Avoid shelling out to platform-specific tools (`sysctl`, `wmic`) from Python packaging or tests. Prefer:

```python
def detect_memory_gb() -> float | None:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        return None
```

* Use `pytest -m "not gpu"` as a sane default on CI without adapters.

## F) Review checklist (Python surface)

* [ ] All NumPy returns are `Bound<'py, ...>` and constructed with `to_pyarray_bound`/`into_pyarray_bound`.
* [ ] `use numpy::PyUntypedArrayMethods;` is present where contiguity is checked.
* [ ] No `PyRuntime` typos; only `PyRuntimeError::new_err`.
* [ ] GIL released around GPU work (`py.allow_threads`).
* [ ] Platform-portable helpers (no `sysctl`, `wmic`) in packaging/tests.