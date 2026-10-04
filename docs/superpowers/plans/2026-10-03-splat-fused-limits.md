# SPLAT-FUSED Limits Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **superpowers:executing-plans** (inline, in this session). **Subagents are forbidden by the owner** — do not use superpowers:subagent-driven-development, the Agent tool, or any delegation. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the ten limits found while producing the SPLAT-FUSED demo videos (Mount St. Helens, Lauterbrunnen) so a fused render can be 1080p/4K, use a terrain colour map, shade cliffs smoothly, light dense LiDAR correctly at low sun, and be driven fully from Python.

**Architecture:** All work stays inside the existing fused specialization: Rust driver `src/path_tracing/hybrid_compute/render_fused.rs`, fused WGSL appended through `shader_sources::fused_kernel()` (anchored, exactly-once edits; the default hybrid kernel stays byte-identical), paging in `src/splat/`, and the Python surface in `python/forge3d/splat.py` + `src/py_functions/splat.rs`. Every fix is proven by a test whose numeric threshold is fixed in this plan before implementation.

**Tech Stack:** Rust 2021 + wgpu 0.19 + WGSL (naga-validated), PyO3/numpy, Python 3.13, pytest.

**Spec:** the limits reported at the end of the demo session (2026-10-03), restated with measured root causes in the "Limit → root cause → task" table below; current behaviour documented in `docs/splat-fused.md` ("Limits") and `docs/prompts/fable5-moonshots/12-splat-fused.md` (original DoD that must keep passing — the original prompt lives at `D:/forge3d/docs/prompts/fable5-moonshots/12-splat-fused.md`).

## Global Constraints

- Worktree: `C:/tmp/splat-fused`, branch `splat-fused`, base commit `b255a990`. Never edit `C:/Users/milos/forge3d` or `D:/forge3d` sources.
- No subagents, no Agent tool, no delegation of any step.
- No `git commit`, `git push`, or PR unless the owner explicitly asks. Each task ends with a verification checkpoint, not a commit.
- No new crates or Python packages.
- No new `.wgsl` files (each would need a `tests/shader_proofs_ledger.toml` entry). New WGSL goes into `src/shaders/fusion/unified_occlusion.wgsl` or is an anchored edit in `src/shader_sources.rs`.
- No new test files (an untracked `tests/test_*.py` fails `tests/test_no_silent_degradation.py`). Extend existing ones: `tests/test_splat_fusion_occlusion.rs`, `tests/test_splat_api.py`, inline `#[cfg(test)]` modules.
- The default hybrid kernel must stay byte-identical: `fused_kernel_tests::default_hybrid_kernel_is_untouched_by_the_fused_specialization` must keep passing.
- World-coordinate f32 inventory is frozen: `tests/test_world_coord_f32_gate.py` must pass **unchanged**. Do not add `as f32` casts; use `crate::splat::num::{f32_from_u32, f32_from_usize}`. If a new narrowing is unavoidable, STOP and ask the owner (ledger edits need owner review).
- Budget, gates and tolerances are fixed: 512 MiB budget, shadow IoU gate `> 0.9`, golden `SSIM >= 0.995` / `mean abs <= 2.0`. Never edit these constants.
- Golden regeneration is allowed only in tasks marked **[changes default image]**, only via `FORGE3D_UPDATE_SPLAT_FUSION_GOLDENS=1`. Record the pre-regeneration SSIM in the evidence file. After regenerating, the golden test must pass twice in a row without the env var.
- Every new public symbol: register in `src/py_module/*`, re-export in `python/forge3d/__init__.py` `__all__`, add to `python/forge3d/splat.pyi` and `python/forge3d/__init__.pyi`, add to `EXPECTED_FUNCTIONS`/`EXPECTED_CLASSES` in `tests/test_api_contracts.py`. Every new public `render_*` accepts `certificate=` and `cache=` (swept by `tests/test_render_certificate_contract.py`).
- Lint: `cargo forge3d-clippy` **and** `cargo forge3d-clippy-acceptance`. Only the second alias compiles `splat-fusion`; both must be clean. Run `cargo fmt --all -- --check`.
- Evidence: append each task's DoD command outputs (the decisive lines) to `docs/superpowers/plans/2026-10-03-splat-fused-limits-evidence.md` under a `## Task N` heading. A task is not done until its evidence block exists.

## Shared commands (Git Bash; use exactly these)

```bash
export CARGO_TARGET_DIR=D:/forge3d/artifacts/build/splat-fused/target
export PYO3_PYTHON=C:/Users/milos/AppData/Local/Programs/Python/Python313/python.exe
export F=default,async_readback,copc_laz,cog_streaming,gis-remote,geos-topology,weighted-oit,wsI_bigbuf,wsI_double_buf,enable-pbr,enable-tbn,enable-normal-mapping,enable-hdr-offscreen,enable-renderer-config,enable-staging-rings,enable-inverse-pt,shader-contract-asserts,enable-globe,splat-fusion
export M=C:/tmp/splat-fused/Cargo.toml
```

- **BUILD_PYD** (after any Rust/WGSL change, before any pytest):
  ```bash
  cargo build --release --lib --manifest-path $M --features extension-module,weighted-oit,enable-pbr,enable-tbn,enable-gpu-instancing,enable-staging-rings,enable-inverse-pt,copc_laz,cog_streaming,gis-remote,geos-topology,atmosphere-bake,shader-contract-asserts,enable-globe,splat-fusion && cp "$CARGO_TARGET_DIR/release/forge3d.dll" C:/tmp/splat-fused/python/forge3d/_forge3d.pyd
  ```
- **UNIT** `<filter>`: `cargo test --release --manifest-path $M --features $F --lib -- --test-threads=1 <filter>`
- **GPU** `<filter>`: `FORGE3D_SPLAT_FUSION_REQUIRED_GPU=1 cargo test --release --manifest-path $M --features $F --test test_splat_fusion_occlusion -- --test-threads=1 --nocapture <filter>`
- **PY** `<args>`: `cd C:/tmp/splat-fused && PYTHONPATH=C:/tmp/splat-fused/python FORGE3D_NO_BOOTSTRAP=1 FORGE3D_SPLAT_FUSION_REQUIRED_GPU=1 D:/forge3d/.venv/Scripts/python -m pytest <args> -v --tb=short`
- Never pipe a background run into `tail`/`head`; write to a log file and grep it (`exit=$?` echoed into the log).

## Limit → root cause → task

| # | Limit (as reported) | Measured root cause | Task |
|---|---|---|---|
| L1 | 1080p cannot render; 720p barely fits | `render_fused.rs:731-797` allocates 392 B/px of persistent GPU memory: 3 reservoirs × 80 B (`:755-757`), Welford 48 B (`:746`), 7 AOV textures 48 B (`:779-788`), 2 G-buffers × 16 B (`:758-759`), accumulation 16 B (`:737`), output 8 B (`:761`); readback staging adds ≤ 80 B. The driver always dispatches one full frame (`camera_flags: 0`, offset 0, `:621-632`), although the kernel already supports seamless tiles (`hybrid_kernel.wgsl:105-156`, `pt_restir_spatial.wgsl:244-247`). | T9 |
| L2 | Terrain costs ≈ 30 B per DEM cell | `terrain_heightfield.rs:290-306`: min-max pyramid is `Rg32Float`, padded to the next power of two per axis (`build_minmax_mips`). A 1250×2500 DEM → 2048×4096 × 8 B × 4/3 = 89.5 MB, against 12.5 MB of heights. | T6 |
| L3 | Terrain takes a single colour | `render_fused.rs:580-592` passes `albedo_rgba = None`. The existing map path is `Rgba32Float` (16 B/cell, `terrain_heightfield.rs:555-571`). | T5 |
| L4 | Steep cliffs show vertical streaks | `hybrid_terrain_traversal.wgsl:334-343`: the shading normal is the per-cell bilinear-patch derivative, discontinuous across every cell edge. | T7 |
| L5 | Dense LiDAR goes dark at low sun | Sphelets are spheres (`unified_occlusion.wgsl:284-297`). Grazing shadow rays leaving a sphelet layer pass through neighbouring spheres; the self-bias (`:517-522`) skips at most `bias / 0.25`. Measured T_lidar 0.70 at 22° elevation (bias 8 → only 0.76). | T8 |
| L6 | Python can't set the LiDAR/splat self-shadow bias | `python/forge3d/splat.py:442-479` lacks `splat_self_bias_sigmas` / `lidar_self_bias_radii` (native has them, `src/py_functions/splat.rs:563-564`); `self_bias` is dropped at `splat.py:607`. | T2 |
| L7 | COPC header min/max read at wrong offsets | `src/pointcloud/copc.rs:298-307` reads max at 179/203/227 and min at 187/211/235. LAS 1.4 puts maxX at 179, minX 187, maxY 195, minY 203, maxZ 211, minZ 219. | T1 |
| L8 | Python renders one view per call (recompiles kernel, re-streams pages) | `src/py_functions/splat.rs:701-703` builds `HybridPathTracer::new_fused()` and a `FusedScene` per call; Rust `render_fused_sequence` (`render_fused.rs:457`) is not exposed. | T10 |
| L9 | Splats can't be georeferenced from Python | `GaussianSplatCloud::apply_similarity` (`src/splat/mod.rs:274`) has no binding. | T3 |
| L10 | Small COPC nodes each waste a whole page slot | `src/splat/stream.rs:1159-1227` makes ≥ 1 page per node. Measured 12,959 pages averaging 232 points at capacity 2700. | T4 |

**Explicitly out of scope** (named so nobody assumes they're fixed): GPU SH degrees 2–3 for splat colour; the power-of-two padding of the min-max pyramid itself (the shared WGSL descent assumes pow2 node coordinates and packs ≤ 8192 cells per axis); heightfields can't represent overhangs, so cliffs stay 2.5D even with smooth normals; capture lighting baked into splat SH.

**Task order** (risk ascending; tasks marked **[changes default image]** regenerate the golden): T1 → T2 → T3 → T6 → T5 → T4 **[changes default image]** → T7 **[changes default image]** → T8 **[changes default image]** → T9 **[changes default image]** → T10 → T11.

---

### Task 1: COPC header bounds read from the LAS 1.4 offsets (L7)

**Files:**
- Modify: `src/pointcloud/copc.rs:298-307` (`read_las_header`), add `CopcDataset::header()` next to `bounds()` (`:264`)
- Test: new inline module at the end of `src/pointcloud/copc.rs`

**Interfaces:**
- Produces: `pub fn header(&self) -> &CopcHeader` on `CopcDataset`.

- [ ] **Step 1: Write the failing tests** (append to `src/pointcloud/copc.rs`)

```rust
#[cfg(test)]
mod header_bounds_tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn las14_bounds_are_read_from_their_specified_offsets() {
        // LAS 1.4 public header: max X @179, min X @187, max Y @195,
        // min Y @203, max Z @211, min Z @219.
        let mut buf = vec![0u8; 375];
        buf[0..4].copy_from_slice(b"LASF");
        for (at, v) in [(179, 11.0f64), (187, 1.0), (195, 22.0), (203, 2.0), (211, 33.0), (219, 3.0)] {
            buf[at..at + 8].copy_from_slice(&v.to_le_bytes());
        }
        let header = read_las_header(&mut Cursor::new(buf)).unwrap();
        assert_eq!(header.max_bounds, [11.0, 22.0, 33.0]);
        assert_eq!(header.min_bounds, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn fixture_header_bounds_enclose_exactly_the_decoded_points() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/splat_fusion/swath.copc.laz");
        let dataset = CopcDataset::open(&path).unwrap();
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for node in dataset.nodes() {
            let data = dataset.read_points(&node.key).unwrap();
            for p in data.positions.chunks_exact(3) {
                for a in 0..3 {
                    lo[a] = lo[a].min(p[a]);
                    hi[a] = hi[a].max(p[a]);
                }
            }
        }
        let header = dataset.header();
        for a in 0..3 {
            assert!((header.min_bounds[a] - lo[a]).abs() <= 1e-3, "min axis {a}: {} vs {}", header.min_bounds[a], lo[a]);
            assert!((header.max_bounds[a] - hi[a]).abs() <= 1e-3, "max axis {a}: {} vs {}", header.max_bounds[a], hi[a]);
        }
    }
}
```

- [ ] **Step 2: Run, expect FAIL.** UNIT `header_bounds_tests` → `las14_bounds_are_read_from_their_specified_offsets` fails (`max_bounds` is `[11.0, 2.0, 0.0]`), and the fixture test fails to compile (`header()` missing).

- [ ] **Step 3: Implement.** Replace `copc.rs:298-307` with:

```rust
    // LAS 1.4 public header: max X @179, min X @187, max Y @195, min Y @203,
    // max Z @211, min Z @219.
    let f = |at: usize| f64::from_le_bytes(buf[at..at + 8].try_into().unwrap());
    let max_bounds = [f(179), f(195), f(211)];
    let min_bounds = [f(187), f(203), f(219)];
```

and add to `impl CopcDataset`:

```rust
    /// Parsed LAS 1.4 public header of the file.
    pub fn header(&self) -> &CopcHeader {
        &self.header
    }
```

- [ ] **Step 4: Run, expect PASS.** UNIT `header_bounds_tests` → 2 passed.

**Definition of Done (all must hold):**
1. UNIT `header_bounds_tests` → `2 passed; 0 failed`.
2. The Step 2 failure was observed before the fix (paste it into the evidence file).
3. `grep -rn "min_bounds\|max_bounds" src/pointcloud` lists only `copc.rs` and `copc_decode.rs`. No other consumer needs a change.
4. GPU `test_splat_fusion_occlusion` still passes (the reader feeds the fused path).

---

### Task 2: Python exposes the self-shadow bias and the `self_bias` AOV (L6)

**Files:**
- Modify: `python/forge3d/splat.py`: `render_fused` signature (`:442-479`) and native call (`:552-591`); `FusedRenderResult` (`:266-305`); the `arrays` tuple and `stats` filter (`:594-609`)
- Modify: `python/forge3d/splat.pyi` (same signature and field)
- Test: `tests/test_splat_api.py`

**Interfaces:**
- Produces: `render_fused(..., splat_self_bias_sigmas: float = 2.0, lidar_self_bias_radii: float = 1.0, ...)`; `FusedRenderResult.self_bias: np.ndarray` (H, W) float32.

- [ ] **Step 1: Write the failing test** (append to `tests/test_splat_api.py`)

```python
@pytest.mark.skipif(not gpu_available(), reason="no usable GPU adapter for the fused render")
def test_self_bias_parameters_reach_the_kernel():
    m = manifest()
    kw = scene_kwargs(m)
    common = dict(samples=4, width=96, height=96, return_aovs=True)
    a = splat.render_fused(**kw, **common, splat_self_bias_sigmas=2.0, lidar_self_bias_radii=1.0)
    b = splat.render_fused(**kw, **common, splat_self_bias_sigmas=0.5, lidar_self_bias_radii=3.0)
    lidar = (a.hit_kind == splat.HIT_LIDAR) & (b.hit_kind == splat.HIT_LIDAR)
    assert lidar.sum() > 50
    np.testing.assert_allclose(a.self_bias[lidar], 1.0 * m["lidar_radius"], rtol=1e-6)
    np.testing.assert_allclose(b.self_bias[lidar], 3.0 * m["lidar_radius"], rtol=1e-6)
    splats = (a.hit_kind == splat.HIT_SPLAT) & (b.hit_kind == splat.HIT_SPLAT)
    assert splats.sum() > 50
    np.testing.assert_allclose(b.self_bias[splats], 0.25 * a.self_bias[splats], rtol=1e-5)
    assert np.all(a.self_bias[a.hit_kind == splat.HIT_TERRAIN] == 0.0)
    with pytest.raises(RuntimeError, match="self-shadow bias"):
        splat.render_fused(**kw, **common, lidar_self_bias_radii=-1.0)
```

- [ ] **Step 2: Run, expect FAIL.** BUILD_PYD; PY `tests/test_splat_api.py::test_self_bias_parameters_reach_the_kernel` → `TypeError: render_fused() got an unexpected keyword argument 'splat_self_bias_sigmas'`.

- [ ] **Step 3: Implement.**
  - Add the two keyword parameters after `sun_angular_radius_deg` in `render_fused` and forward them to the native call as `splat_self_bias_sigmas=float(splat_self_bias_sigmas), lidar_self_bias_radii=float(lidar_self_bias_radii)`.
  - Document both under `Args:`: "secondary rays leaving a splat (point) skip the stretch in which they rise this many standard deviations (sphelet radii) along the hit normal".
  - Add `self_bias: np.ndarray` to `FusedRenderResult` directly after `reservoir_visibility` (before `stats`). Add `"self_bias"` to the `arrays` tuple. Change the stats filter to `key not in arrays`.
  - Mirror both changes in `splat.pyi`.

- [ ] **Step 4: Run, expect PASS.** PY `tests/test_splat_api.py tests/test_api_contracts.py`.

**Definition of Done:**
1. PY `tests/test_splat_api.py::test_self_bias_parameters_reach_the_kernel` passes, not skipped (`FORGE3D_SPLAT_FUSION_REQUIRED_GPU=1`).
2. PY `tests/test_splat_api.py tests/test_api_contracts.py` → 0 failures.
3. `splat.pyi` signature equals `splat.py` (same names, defaults and order). Check by eye, and by `inspect.signature(splat.render_fused)` printed into the evidence.

---

### Task 3: Georeference splats from Python: `GaussianSplatCloud.transformed` (L9)

**Files:**
- Modify: `src/splat/mod.rs` (add `transformed` after `apply_similarity`, `:274-305`)
- Modify: `src/py_functions/splat.rs` (`#[pymethods] impl PyGaussianSplatCloud`, `:65-215`)
- Modify: `python/forge3d/splat.pyi`, `docs/splat-fused.md` (usage line)
- Test: `src/splat/mod.rs` tests module; `tests/test_splat_api.py`

**Interfaces:**
- Produces: Rust `pub fn transformed(&self, scale: f32, rotation: [f32; 4], translation: [f32; 3]) -> Result<GaussianSplatCloud, RenderError>`; Python `GaussianSplatCloud.transformed(scale: float, rotation: tuple[float,float,float,float], translation: tuple[float,float,float]) -> GaussianSplatCloud` (rotation is `(w, x, y, z)`).

- [ ] **Step 1: Write the failing tests.** Rust (in `src/splat/mod.rs` `mod tests`):

```rust
    #[test]
    fn transformed_copies_and_leaves_the_source_untouched() {
        let cloud = cloud_of([0.5, 1.5, 0.25], [0.6, 0.2, -0.5, 0.4]);
        let moved = cloud.transformed(2.0, [0.7, 0.1, 0.7, -0.1], [10.0, -4.0, 3.0]).unwrap();
        assert_eq!(cloud.positions[0], [1.0, 2.0, 3.0]);
        assert_eq!(moved.scales[0], [1.0, 3.0, 0.5]);
        assert_eq!(moved.inv_cov()[0], inverse_covariance(moved.scales[0], moved.rotations[0]));
        assert!(cloud.transformed(0.0, [1.0, 0.0, 0.0, 0.0], [0.0; 3]).is_err());
    }
```

Python (append to `tests/test_splat_api.py`):

```python
def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw*bw - ax*bx - ay*by - az*bz, aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw])


def test_transformed_applies_a_similarity_without_mutating_the_source():
    cloud = splat.load_gaussian_splats(FIXTURE_DIR / manifest()["splat_file"])
    before = np.array(cloud.positions, copy=True)
    q = np.array([0.9, 0.1, -0.3, 0.2]); q /= np.linalg.norm(q)
    t = np.array([100.0, -20.0, 7.0])
    moved = cloud.transformed(2.5, tuple(q), tuple(t))
    R = _quat_to_matrix(q)
    np.testing.assert_allclose(moved.positions, 2.5 * before @ R.T + t, rtol=0, atol=1e-3)
    np.testing.assert_allclose(moved.scales, 2.5 * np.asarray(cloud.scales), rtol=1e-6)
    for qi, qm in zip(np.asarray(cloud.rotations), np.asarray(moved.rotations)):
        assert abs(np.dot(_quat_mul(q, qi), qm)) >= 1.0 - 1e-6
    np.testing.assert_array_equal(np.asarray(cloud.positions), before)
    assert moved.count == cloud.count
    for bad in (dict(scale=0.0), dict(scale=float("nan"))):
        with pytest.raises(ValueError, match="scale"):
            cloud.transformed(bad["scale"], (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="rotation"):
        cloud.transformed(1.0, (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
```

- [ ] **Step 2: Run, expect FAIL** (UNIT `transformed_copies`: method missing; PY: `AttributeError`).

- [ ] **Step 3: Implement.** Rust core:

```rust
    /// A copy of this cloud under a similarity transform (uniform scale,
    /// rotation `(w, x, y, z)`, translation). The source cloud is untouched;
    /// the copy registers its own host bytes with the memory tracker.
    pub fn transformed(
        &self,
        scale: f32,
        rotation: [f32; 4],
        translation: [f32; 3],
    ) -> Result<Self, RenderError> {
        let mut out = Self::from_parts(
            self.positions.clone(),
            self.scales.clone(),
            self.rotations.clone(),
            self.opacities.clone(),
            self.sh0.clone(),
            self.sh_rest.clone(),
        )?;
        out.apply_similarity(scale, rotation, translation)?;
        Ok(out)
    }
```

PyO3 method (inside `#[pymethods] impl PyGaussianSplatCloud`):

```rust
    /// Copy of the cloud under a similarity transform: uniform `scale`,
    /// unit-quaternion `rotation` (w, x, y, z), `translation`.
    fn transformed(&self, scale: f32, rotation: [f32; 4], translation: [f32; 3]) -> PyResult<Self> {
        if !(scale.is_finite() && scale > 0.0) {
            return Err(PyValueError::new_err(format!("scale must be finite and > 0, got {scale}")));
        }
        let norm = rotation.iter().map(|v| v * v).sum::<f32>().sqrt();
        if !(norm.is_finite() && norm > 1e-12) {
            return Err(PyValueError::new_err("rotation must be a finite, non-zero quaternion (w, x, y, z)"));
        }
        if translation.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("translation must be finite"));
        }
        Ok(Self { inner: Arc::new(self.inner.transformed(scale, rotation, translation)?) })
    }
```

Add the stub to `splat.pyi`, and a one-line usage example to `docs/splat-fused.md`: `cloud = cloud.transformed(scale, (w, x, y, z), (tx, ty, tz))  # place a capture in the scene frame`.

- [ ] **Step 4: Run, expect PASS.** UNIT `transformed_copies`; BUILD_PYD; PY `tests/test_splat_api.py`.

**Definition of Done:**
1. UNIT `splat::tests::transformed_copies_and_leaves_the_source_untouched` passes.
2. PY `test_transformed_applies_a_similarity_without_mutating_the_source` passes (CPU-only; it must not be skipped).
3. PY `tests/test_api_contracts.py` passes. `transformed` is a method, so no new `EXPECTED_*` entries are needed.

---

### Task 6: Terrain memory: conservative f16 min-max pyramid for the fused path (L2)

(Runs before T5 because T5 extends the constructor this task introduces.)

**Files:**
- Modify: `src/path_tracing/hybrid_compute/terrain_heightfield.rs`. Add `MinMaxPrecision`, `TerrainAlbedoMap`, `f16_floor`/`f16_ceil`, `TerrainMinMaxPyramid::from_heightfield_with_precision`, and `TerrainPtScene::new_with_options`. The existing `new_with_albedo` delegates with `MinMaxPrecision::F32` (identical behaviour for every existing caller).
- Modify: `src/path_tracing/hybrid_compute/render_fused.rs`. `FusedTerrainDesc` gains `minmax_precision: MinMaxPrecision`; the call at `:580-596` uses `new_with_options`; `FusedRenderOutput` gains `terrain_bytes: u64` and `terrain_minmax_f16: bool`.
- Modify: every `FusedTerrainDesc { .. }` literal: `src/py_functions/splat.rs:440-447`, `tests/test_splat_fusion_occlusion.rs:156-166`, `render_fused.rs:571-578`, and any other hit by `grep -rn "FusedTerrainDesc {" src tests`.
- Modify: `src/py_functions/splat.rs` `output_dict` adds `terrain_bytes`, `terrain_minmax_f16`.
- Test: inline tests in `terrain_heightfield.rs`; `tests/test_splat_fusion_occlusion.rs`.

**Interfaces:**
- Produces:
  ```rust
  #[derive(Clone, Copy, Debug, PartialEq, Eq)]
  pub enum MinMaxPrecision { F32, F16Conservative }
  pub enum TerrainAlbedoMap<'a> { None, Rgba32F(&'a [f32]) } // T5 adds Rgba8Srgb(&'a [u8])
  pub(crate) fn f16_floor(v: f32) -> half::f16;
  pub(crate) fn f16_ceil(v: f32) -> half::f16;
  impl TerrainPtScene { pub fn new_with_options(device, queue, heights, dem_width, dem_height, spacing, exaggeration, albedo: [f32; 3], env_map, env_intensity, albedo_map: TerrainAlbedoMap<'_>, albedo_sampling: AlbedoSampling, turbidity: f32, minmax: MinMaxPrecision) -> Result<Self, RenderError>; pub fn minmax_is_f16(&self) -> bool; pub fn minmax_bytes(&self) -> u64; }
  ```
  `minmax_bytes()` = tracked bytes of the min-max texture alone (all mips). Re-export `MinMaxPrecision` and `TerrainAlbedoMap` from `src/path_tracing/hybrid_compute/mod.rs` next to the existing `pub use terrain_heightfield::{AlbedoSampling, TerrainPtScene};`.
- Rust test helper `terrain_desc` (`tests/test_splat_fusion_occlusion.rs:156-166`) sets `minmax_precision: MinMaxPrecision::F16Conservative`, the shipped setting, so the acceptance suite exercises it.
- Rule: `F16Conservative` is honoured only when every height satisfies `|h| <= 65504`. Otherwise the pyramid is F32 and `minmax_is_f16()` returns `false`. That's reported in the output, not hidden.

- [ ] **Step 1: Write failing CPU tests** (in `terrain_heightfield.rs` tests):

```rust
    #[test]
    fn f16_rounding_is_conservative_and_tight() {
        let mut state = 0x1234_5678u32;
        for _ in 0..100_000 {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            let v = (f32::from_bits(0x3f80_0000 | (state >> 9)) - 1.5) * 2.0e4; // [-1e4, 1e4)
            let (lo, hi) = (f16_floor(v), f16_ceil(v));
            assert!(lo.to_f32() <= v && v <= hi.to_f32(), "{v}: {} {}", lo.to_f32(), hi.to_f32());
            let steps = hi.to_bits().wrapping_sub(lo.to_bits()) & 0x7fff;
            assert!(steps <= 1 || (lo.to_f32() <= 0.0 && hi.to_f32() >= 0.0), "{v}: not adjacent");
        }
        for exact in [0.0f32, 1.0, -2.5, 2048.0, 65504.0, -65504.0] {
            assert_eq!(f16_floor(exact).to_f32(), exact);
            assert_eq!(f16_ceil(exact).to_f32(), exact);
        }
        assert_eq!(f16_floor(f32::INFINITY).to_f32(), f32::INFINITY);
        assert_eq!(f16_ceil(f32::NEG_INFINITY).to_f32(), f32::NEG_INFINITY);
    }
```

- [ ] **Step 2: Write failing GPU tests** (append to `tests/test_splat_fusion_occlusion.rs`; reuse `gpu_available`, `fixture_sources`, `fused_desc`, `fixture_params`):

```rust
fn bits(v: &[f32]) -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() }

#[test]
fn terrain_f16_minmax_changes_no_output_bit() {
    if !gpu_available() { return; }
    let fixture = FixtureScene::load(fixture_dir()).unwrap();
    let params = fixture_params(&fixture);
    let scene = FusedScene::new(fixture_sources(&fixture, &params), params).unwrap();
    let tracer = HybridPathTracer::new_fused().unwrap();
    let mut desc = fused_desc(&scene, &fixture, 160, 2, 4);
    desc.terrain.as_mut().unwrap().minmax_precision = MinMaxPrecision::F32;
    let a = tracer.render_fused(&desc).unwrap();
    desc.terrain.as_mut().unwrap().minmax_precision = MinMaxPrecision::F16Conservative;
    let b = tracer.render_fused(&desc).unwrap();
    assert!(!a.terrain_minmax_f16 && b.terrain_minmax_f16);
    assert_eq!(a.rgba, b.rgba);
    assert_eq!(a.hit_kind, b.hit_kind);
    for (name, x, y) in [("radiance", &a.radiance, &b.radiance), ("depth", &a.depth, &b.depth),
        ("transmittance", &a.transmittance, &b.transmittance), ("normal", &a.normal, &b.normal)] {
        assert_eq!(bits(x), bits(y), "{name} differs");
    }
}

#[test]
fn terrain_f16_minmax_halves_the_pyramid_bytes() {
    if !gpu_available() { return; }
    // Lauterbrunnen-shaped DEM: 1250 x 2500 (pow2-padded to 2048 x 4096 cells).
    let (w, h) = (1250u32, 2500u32);
    let heights: Vec<f32> = (0..w * h).map(|i| 700.0 + ((i % w) as f32 * 0.37).sin() * 900.0 + (i / w) as f32 * 0.5).collect();
    let ctx = forge3d::core::gpu::try_ctx().unwrap();
    let mk = |p| TerrainPtScene::new_with_options(&ctx.device, &ctx.queue, &heights, w, h, (6.0, 6.0), 1.0,
        [0.4; 3], None, 1.0, TerrainAlbedoMap::None, AlbedoSampling::Nearest, 1.0, p).unwrap();
    let full = mk(MinMaxPrecision::F32);
    let half = mk(MinMaxPrecision::F16Conservative);
    assert_eq!(half.minmax_bytes() * 2, full.minmax_bytes(), "f16 pyramid must be exactly half");
    assert_eq!(full.byte_size() - half.byte_size(), half.minmax_bytes(), "nothing else may change");
    // pow2-padded 2048 x 4096 level 0 -> 8 B/texel (f32) over the full mip chain.
    assert!(full.minmax_bytes() >= 2048 * 4096 * 8);
    assert!(half.minmax_is_f16() && !full.minmax_is_f16());
}
```

(The `(i % w) as f32` casts are test code under `tests/`, outside the production inventory the f32 gate freezes. Confirm with `tests/test_world_coord_f32_gate.py` in the DoD. If the gate counts them, use `forge3d::splat::num::f32_from_u32` instead, which requires making that module `pub` behind `#[doc(hidden)]`.)

- [ ] **Step 3: Run, expect FAIL** (missing items: compile errors).

- [ ] **Step 4: Implement.**
  - `f16_floor`/`f16_ceil`:
    ```rust
    pub(crate) fn f16_floor(v: f32) -> half::f16 {
        let h = half::f16::from_f32(v);
        if h.is_nan() || h.to_f32() <= v { h } else { f16_step_down(h) }
    }
    pub(crate) fn f16_ceil(v: f32) -> half::f16 {
        let h = half::f16::from_f32(v);
        if h.is_nan() || h.to_f32() >= v { h } else { f16_step_up(h) }
    }
    fn f16_step_up(h: half::f16) -> half::f16 {
        let b = h.to_bits();
        if h == half::f16::INFINITY { return h; }
        if b == 0x8000 { return half::f16::from_bits(0x0001); }
        half::f16::from_bits(if b & 0x8000 == 0 { b + 1 } else { b - 1 })
    }
    fn f16_step_down(h: half::f16) -> half::f16 {
        let b = h.to_bits();
        if h == half::f16::NEG_INFINITY { return h; }
        if b == 0x0000 { return half::f16::from_bits(0x8001); }
        half::f16::from_bits(if b & 0x8000 == 0 { b - 1 } else { b + 1 })
    }
    ```
  - `from_heightfield_with_precision(device, queue, heights, w, h, precision)`. For `F16Conservative` (when in range), create the min-max texture as `Rg16Float` and upload per level `[f16_floor(min), f16_ceil(max)]` as `[u16; 2]` texels (`bytes_per_row = lw * 4`). `byte_size` sums 4 B per texel instead of 8. `from_heightfield` calls it with `F32`. Change nothing in the bind-group layout: `Rg16Float` binds to the existing unfilterable-float `texture_2d<f32>` entry.
  - `new_with_options`: body of `new_with_albedo` with the two new parameters; `new_with_albedo` becomes a delegating wrapper (`TerrainAlbedoMap::Rgba32F` when `Some`, `MinMaxPrecision::F32`).
  - Fused driver: `FusedTerrainDesc.minmax_precision` (Python `build_scene` sets `F16Conservative`); report `terrain_bytes = terrain_scene.byte_size()` and `terrain_minmax_f16 = terrain_scene.minmax_is_f16()`.

- [ ] **Step 5: Run, expect PASS.** UNIT `f16_rounding`; GPU `terrain_f16_minmax`; GPU full `test_splat_fusion_occlusion`; PY `tests/test_hybrid_terrain_pt.py` (shared terrain code, F32 path).

**Definition of Done:**
1. UNIT `f16_rounding_is_conservative_and_tight` passes.
2. GPU `terrain_f16_minmax_changes_no_output_bit` passes: rgba, hit_kind, radiance, depth, transmittance and normal are **bit-identical**. Rationale, verified in `hybrid_terrain_traversal.wgsl:401-404`: min-max values only cull nodes; leaf spans come from xz slabs and the exact patch test uses the `R32Float` heights.
3. GPU `terrain_f16_minmax_halves_the_pyramid_bytes` passes (`pyramid16 * 2 == pyramid32` exactly).
4. GPU full `test_splat_fusion_occlusion` passes with no golden regeneration (output is bit-identical).
5. PY `tests/test_hybrid_terrain_pt.py tests/test_world_coord_f32_gate.py` pass unchanged.

---

### Task 5: Terrain colour map (8-bit sRGB, 4 B/cell) (L3)

**Files:**
- Modify: `terrain_heightfield.rs`. Add `TerrainAlbedoMap::Rgba8Srgb(&[u8])` (texture `Rgba8UnormSrgb`, w×h×4 bytes; alpha 255 = mapped, anything else falls back to the uniform albedo, exactly as the kernel's `c.a >= 1.0` test at `hybrid_terrain_traversal.wgsl:544`). `albedo_bytes = 4 * w * h`.
- Modify: `render_fused.rs`. `FusedTerrainDesc` gains `albedo_map: Option<Vec<u8>>` (RGBA8 sRGB, `width*height*4`, row-major, row 0 = first heights row) and `albedo_sampling: AlbedoSampling`; pass `TerrainAlbedoMap::Rgba8Srgb` when present.
- Modify: every `FusedTerrainDesc { .. }` literal (same list as Task 6: `grep -rn "FusedTerrainDesc {" src tests`) adds `albedo_map: None, albedo_sampling: AlbedoSampling::Bilinear`; the fused driver's placeholder DEM (`render_fused.rs:571-578`) uses `AlbedoSampling::Nearest` as today.
- Modify: `src/py_functions/splat.rs` `build_scene`/`render_fused`/`render_fused_reference_py`: new kwargs `terrain_albedo_map: Option<PyReadonlyArray3<u8>>` (shape `(rows, cols, 3|4)`), `terrain_albedo_sampling: &str = "bilinear"` (`"bilinear"|"nearest"`, else `ValueError`).
- Modify: `python/forge3d/splat.py`. `FusedTerrain` gains `albedo_map: Optional[np.ndarray] = None` and `albedo_sampling: str = "bilinear"`; `_resolve_terrain` validates `dtype == uint8` (else `TypeError`) and shape `(rows, cols, 3|4)` equal to `heights.shape` (else `ValueError("terrain albedo_map shape ...")`); forwarded via `_scene_kwargs`. Update `splat.pyi`.
- Test: `tests/test_splat_fusion_occlusion.rs`, `tests/test_splat_api.py`.

- [ ] **Step 1: Write the failing GPU test** (Rust):

```rust
fn srgb_to_linear(c: u8) -> f32 { let c = f32::from(c) / 255.0; if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) } }
fn linear_to_srgb_u8(v: f32) -> i32 { let v = v.clamp(0.0, 1.0); let s = if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }; (s * 255.0).round() as i32 }

#[test]
fn terrain_albedo_map_drives_the_albedo_aov() {
    if !gpu_available() { return; }
    let fixture = FixtureScene::load(fixture_dir()).unwrap();
    let params = fixture_params(&fixture);
    let scene = FusedScene::new(fixture_sources(&fixture, &params), params).unwrap();
    let m = &fixture.manifest;
    let (w, h) = (m.dem_width as usize, m.dem_height as usize);
    let stripes = [[200u8, 40, 40], [40, 60, 200]];
    let mut map = vec![0u8; w * h * 4];
    for r in 0..h { for c in 0..w {
        let s = stripes[(c / 8) % 2];
        let alpha = if r < h / 2 { 255 } else { 0 }; // southern half: masked -> uniform albedo
        map[(r * w + c) * 4..(r * w + c) * 4 + 4].copy_from_slice(&[s[0], s[1], s[2], alpha]);
    }}
    let mut desc = fused_desc(&scene, &fixture, 192, 1, 1);
    let t = desc.terrain.as_mut().unwrap();
    t.albedo_map = Some(map);
    t.albedo_sampling = AlbedoSampling::Nearest;
    let out = HybridPathTracer::new_fused().unwrap().render_fused(&desc).unwrap();
    let (sx, sz) = (m.dem_spacing[0], m.dem_spacing[1]);
    let (ox, oz) = (-0.5 * (m.dem_width - 1) as f32 * sx, -0.5 * (m.dem_height - 1) as f32 * sz);
    let (mut mapped, mut masked) = (0, 0);
    for i in 0..out.hit_kind.len() {
        if out.hit_kind[i] != HIT_TERRAIN { continue; }
        let col = (out.position[3 * i] - ox) / sx;
        let row = (out.position[3 * i + 2] - oz) / sz;
        let (cn, rn) = (col.round() as usize, row.round() as usize);
        let in_stripe = (col.round() - col).abs() < 0.4 && (1..7).contains(&(cn % 8));
        if !in_stripe || rn == h / 2 || rn + 1 == h / 2 { continue; }
        let rgb = [out.albedo[3 * i], out.albedo[3 * i + 1], out.albedo[3 * i + 2]];
        let expect: [u8; 3] = if rn < h / 2 { mapped += 1; stripes[(cn / 8) % 2] } else {
            masked += 1; m.terrain_albedo.map(|v| linear_to_srgb_u8(v) as u8) };
        for ch in 0..3 {
            assert!((linear_to_srgb_u8(rgb[ch]) - i32::from(expect[ch])).abs() <= 1, "pixel {i} ch {ch}");
        }
    }
    assert!(mapped >= 500 && masked >= 300, "mapped {mapped} masked {masked}");
}
```

(Uniform-albedo comparison: `m.terrain_albedo` is linear, so the masked half expects `linear_to_srgb_u8(terrain_albedo)` per channel. `srgb_to_linear` is kept for the Python test below.)

- [ ] **Step 2: Write the failing Python tests** (`tests/test_splat_api.py`): `FusedTerrain(heights=h, albedo_map=np.zeros(h.shape + (3,), np.float32))` → `TypeError`; a map of shape `(rows+1, cols, 3)` → `ValueError` matching `"albedo_map shape"`. A GPU test renders the fixture with a uniform-colour map `(rows, cols, 3)` of sRGB `(128, 128, 128)` and asserts `result.albedo[terrain]` ≈ `srgb_to_linear(128)` within `1/255` after sRGB encoding.

- [ ] **Step 3: Run, expect FAIL**, then **Step 4: implement** as listed in Files. Also add a unit test asserting `albedo_bytes == 4 * w * h` for an `Rgba8Srgb` map on a 37×23 DEM.

- [ ] **Step 5: Run, expect PASS.** GPU `terrain_albedo_map`; BUILD_PYD; PY `tests/test_splat_api.py`; GPU full `test_splat_fusion_occlusion` (default output unchanged: no map).

**Definition of Done:**
1. GPU `terrain_albedo_map_drives_the_albedo_aov` passes, with ≥ 500 mapped and ≥ 300 masked pixels, every one within ±1 sRGB code value.
2. Unit test `albedo_bytes == 4*w*h` passes (memory is 4 B/cell, not 16).
3. PY tests: `TypeError` for non-uint8, `ValueError` for shape mismatch, and the uniform-map GPU check all pass.
4. GPU full `test_splat_fusion_occlusion` passes with no golden regeneration.
5. PY `tests/test_hybrid_terrain_pt.py` passes (the `Rgba32F` path is unchanged).

---

### Task 4: Pack small COPC nodes into shared pages (L10) **[changes default image]**

**Files:**
- Modify: `src/splat/stream.rs`. `CopcPage` becomes `{ parts: Vec<CopcPart>, count: u32, aabb: Aabb }` with `struct CopcPart { key: OctreeKey, first: u32, count: u32 }`; add `fn node_morton(key: &OctreeKey) -> u64` (uses `spread_bits_21`, `:122`) and `fn pack_copc_nodes(nodes: Vec<(OctreeKey, u32, Aabb)>, capacity: u32) -> Vec<CopcPage>`; `CopcPageSource::open` (`:1160-1227`) calls it; `load` concatenates the parts.
- Modify: `src/splat/fixture.rs:101-102`: make `COPC_ORIGIN` and `COPC_HALFSIZE` `pub const`.
- Test: `src/splat/stream.rs` tests module.

**Packing rule (normative):**
1. A node with `count >= capacity / 2` keeps standalone pages: runs of ≤ `capacity` points, each with the node's cell AABB (today's behaviour).
2. Nodes with `count < capacity / 2` are grouped by `depth`. Within a depth, sort by `(node_morton(key), key.x, key.y, key.z)` and pack greedily: append to the open page while `open.count + node.count <= capacity`, otherwise close it and start a new one. Page AABB = union of member cells.
3. Page order: standalone pages first, sorted by `(depth, x, y, z, first)` as today; then packed pages in creation order.

```rust
fn pack_copc_nodes(nodes: Vec<(OctreeKey, u32, Aabb)>, capacity: u32) -> Vec<CopcPage> {
    let mut pages = Vec::new();
    let mut small: Vec<(OctreeKey, u32, Aabb)> = Vec::new();
    let mut big: Vec<(OctreeKey, u32, Aabb)> = Vec::new();
    for node in nodes {
        if node.1 >= capacity / 2 { big.push(node) } else if node.1 > 0 { small.push(node) }
    }
    big.sort_by(|a, b| (a.0.depth, a.0.x, a.0.y, a.0.z).cmp(&(b.0.depth, b.0.x, b.0.y, b.0.z)));
    for (key, count, aabb) in big {
        let mut first = 0;
        while first < count {
            let run = (count - first).min(capacity);
            pages.push(CopcPage { parts: vec![CopcPart { key: key.clone(), first, count: run }], count: run, aabb });
            first += run;
        }
    }
    small.sort_by(|a, b| (a.0.depth, node_morton(&a.0), a.0.x, a.0.y, a.0.z)
        .cmp(&(b.0.depth, node_morton(&b.0), b.0.x, b.0.y, b.0.z)));
    let mut open: Option<CopcPage> = None;
    let mut open_depth = u32::MAX;
    for (key, count, aabb) in small {
        let fits = open.as_ref().is_some_and(|p| open_depth == key.depth && p.count + count <= capacity);
        if !fits {
            pages.extend(open.take());
            open = Some(CopcPage { parts: Vec::new(), count: 0, aabb: Aabb::EMPTY });
            open_depth = key.depth;
        }
        let page = open.as_mut().unwrap();
        page.parts.push(CopcPart { key, first: 0, count });
        page.count += count;
        page.aabb = page.aabb.union(aabb);
    }
    pages.extend(open);
    pages
}
```

- [ ] **Step 1: Write failing unit tests** (in `stream.rs` tests):

```rust
    #[test]
    fn small_copc_nodes_are_packed_without_losing_or_duplicating_points() {
        use crate::splat::fixture::{write_copc, LasPoint, COPC_HALFSIZE, COPC_ORIGIN};
        let dir = std::env::temp_dir().join(format!("forge3d-copc-pack-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("many_nodes.copc.laz");
        let n = 3000u32;
        let points: Vec<LasPoint> = (0..n).map(|i| {
            let u = |k: u32| f64::from(hash_unit(i.wrapping_mul(3).wrapping_add(k))) * 1.98 - 0.99;
            LasPoint { xyz: [COPC_ORIGIN[0] + u(1) * COPC_HALFSIZE, COPC_ORIGIN[1] + u(2) * COPC_HALFSIZE,
                COPC_ORIGIN[2] + u(3) * COPC_HALFSIZE], rgb: [100, 120, 90], classification: 2 }
        }).collect();
        write_copc(&path, &points).unwrap();
        let capacity = 512u32;
        let source = CopcPageSource::open(&path, PointCloudFrame { origin: COPC_ORIGIN, z_up: true }, capacity).unwrap();
        let nodes = crate::pointcloud::CopcDataset::open(&path).unwrap().nodes().len() as u32;
        let pages = source.page_count();
        let floor = n.div_ceil(capacity);
        assert!(pages >= floor && pages <= 2 * floor, "pages {pages}, floor {floor}");
        assert!(pages < nodes, "pages {pages} must be fewer than nodes {nodes}");
        let mut loaded: Vec<[u32; 3]> = Vec::new();
        for page in 0..pages {
            let meta = source.meta(page);
            assert!(meta.count <= capacity);
            let PagePayload::Points { positions, .. } = source.load(page).unwrap() else { panic!() };
            assert_eq!(positions.len() as u32, meta.count);
            for p in &positions {
                for a in 0..3 { assert!(p[a] >= meta.aabb.min[a] - 1e-3 && p[a] <= meta.aabb.max[a] + 1e-3); }
                loaded.push(p.map(f32::to_bits));
            }
        }
        assert_eq!(loaded.len() as u32, n);
        loaded.sort_unstable();
        loaded.dedup();
        assert_eq!(loaded.len() as u32, n, "a point was duplicated or lost");
        std::fs::remove_dir_all(&dir).ok();
    }
```

(If two generated points collide bit-exactly, `dedup` would mis-report. The hash makes that negligible, and the assertion message says what to inspect.)

- [ ] **Step 2: Run, expect FAIL** (today `pages == nodes`, about 65 > `2 * floor` = 12).
- [ ] **Step 3: Implement** the rule above; `load(page)` iterates `parts`, decoding each node with `decode_node` and taking `[first, first + count)`.
- [ ] **Step 4: Run.** UNIT `small_copc_nodes_are_packed`; then GPU full `test_splat_fusion_occlusion`. If the golden SSIM falls below 0.995 (primitive ids change, so the stochastic acceptance noise changes), record the SSIM and regenerate with `FORGE3D_UPDATE_SPLAT_FUSION_GOLDENS=1`, then rerun twice without it.

**Definition of Done:**
1. UNIT `small_copc_nodes_are_packed_without_losing_or_duplicating_points` passes: `ceil(N/cap) <= pages <= 2*ceil(N/cap)`, `pages < nodes`, every point exactly once, every point inside its page AABB.
2. GPU full `test_splat_fusion_occlusion`: all five IoU cases `> 0.9`; the CPU oracle test passes with its existing threshold; the golden passes (after a recorded regeneration if needed).
3. Evidence also records the page count of `D:/forge3d_data/splat_fused_demos/lauterbrunnen` rebuilt at voxel 3.3 m with adaptive depth (`process_lidar.py 3.3` after reverting its `depth=3` to `max_points_per_node=2600`): page count at capacity 2700 must be ≤ `2*ceil(points/2700)` (was 12,959).

---

### Task 7: Smooth terrain shading normals on the fused path (L4) **[changes default image]**

**Files:**
- Modify: `src/shader_sources.rs`. Add to `FUSED_TERRAIN_EDITS` the rename `"fn terrain_normal_at(p: vec3<f32>, cx: u32, cz: u32) -> vec3<f32> {"` → `"fn terrain_normal_at_base(p: vec3<f32>, cx: u32, cz: u32) -> vec3<f32> {"` (what: `"terrain shading normal seam"`).
- Modify: `src/shaders/fusion/unified_occlusion.wgsl`. Add `const FUSION_FLAG_SMOOTH_TERRAIN: u32 = 4u;`, `fusion_terrain_vertex_normal`, and the fused `terrain_normal_at`.
- Modify: `src/splat/fusion.rs`. `FusionParams.terrain_smooth_normals: bool` (default `true`); `pools[3] |= u32::from(params.terrain_smooth_normals) << 2`.
- Modify: `src/splat/kernel.rs`. Add CPU mirrors `terrain_patch_normal` and `terrain_smooth_normal`.
- Modify: native `render_fused`/`render_fused_reference_py` + Python `render_fused(..., terrain_smooth_normals: bool = True)` + `.pyi`.
- Test: `src/splat/kernel.rs` tests; `tests/test_splat_fusion_occlusion.rs`; `src/shader_sources.rs` `fused_kernel_tests`.

WGSL (append to `unified_occlusion.wgsl`):

```wgsl
const FUSION_FLAG_SMOOTH_TERRAIN: u32 = 4u;

// Central-difference normal at DEM vertex (ix, iz), clamped at the borders.
fn fusion_terrain_vertex_normal(ix: i32, iz: i32) -> vec3<f32> {
    let w = i32(terrain.dims.x);
    let h = i32(terrain.dims.y);
    let x0 = clamp(ix - 1, 0, w - 1);
    let x1 = clamp(ix + 1, 0, w - 1);
    let z0 = clamp(iz - 1, 0, h - 1);
    let z1 = clamp(iz + 1, 0, h - 1);
    let ex = terrain.h_params.z;
    let hx0 = textureLoad(terrain_height_tex, vec2<i32>(x0, iz), 0).r * ex;
    let hx1 = textureLoad(terrain_height_tex, vec2<i32>(x1, iz), 0).r * ex;
    let hz0 = textureLoad(terrain_height_tex, vec2<i32>(ix, z0), 0).r * ex;
    let hz1 = textureLoad(terrain_height_tex, vec2<i32>(ix, z1), 0).r * ex;
    let dhdx = (hx1 - hx0) / (f32(max(x1 - x0, 1)) * terrain.origin_spacing.z);
    let dhdz = (hz1 - hz0) / (f32(max(z1 - z0, 1)) * terrain.origin_spacing.w);
    return normalize(vec3<f32>(-dhdx, 1.0, -dhdz));
}

// Fused shading normal: the four cell-corner vertex normals interpolated
// bilinearly (C0-continuous across cells). The hit itself stays the exact
// bilinear-patch intersection; a smooth normal facing away from the patch
// falls back to the patch normal.
fn terrain_normal_at(p: vec3<f32>, cx: u32, cz: u32) -> vec3<f32> {
    let base = terrain_normal_at_base(p, cx, cz);
    if ((fusion.pools.w & FUSION_FLAG_SMOOTH_TERRAIN) == 0u) { return base; }
    let u = clamp((p.x - terrain.origin_spacing.x) / terrain.origin_spacing.z - f32(cx), 0.0, 1.0);
    let v = clamp((p.z - terrain.origin_spacing.y) / terrain.origin_spacing.w - f32(cz), 0.0, 1.0);
    let ix = i32(cx);
    let iz = i32(cz);
    let n = mix(
        mix(fusion_terrain_vertex_normal(ix, iz), fusion_terrain_vertex_normal(ix + 1, iz), u),
        mix(fusion_terrain_vertex_normal(ix, iz + 1), fusion_terrain_vertex_normal(ix + 1, iz + 1), u),
        v);
    let s = normalize(n);
    return select(base, s, dot(s, base) > 0.0);
}
```

CPU mirrors (`kernel.rs`), statement for statement (`heights` row-major `w×h`, `spacing = (sx, sz)`, `ex`):

```rust
pub fn terrain_patch_normal(heights: &[f32], w: usize, spacing: (f32, f32), ex: f32, u: f32, v: f32, cx: usize, cz: usize) -> [f32; 3] {
    let at = |x: usize, z: usize| heights[z * w + x] * ex;
    let (h00, h10, h01, h11) = (at(cx, cz), at(cx + 1, cz), at(cx, cz + 1), at(cx + 1, cz + 1));
    let dh_du = (h10 - h00) * (1.0 - v) + (h11 - h01) * v;
    let dh_dv = (h01 - h00) * (1.0 - u) + (h11 - h10) * u;
    normalize3([-dh_du / spacing.0, 1.0, -dh_dv / spacing.1])
}
pub fn terrain_smooth_normal(heights: &[f32], w: usize, h: usize, spacing: (f32, f32), ex: f32, u: f32, v: f32, cx: usize, cz: usize) -> [f32; 3] {
    let vertex = |ix: isize, iz: isize| {
        let cl = |a: isize, n: usize| a.clamp(0, n as isize - 1) as usize;
        let (x0, x1, z0, z1) = (cl(ix - 1, w), cl(ix + 1, w), cl(iz - 1, h), cl(iz + 1, h));
        let at = |x: usize, z: usize| heights[z * w + x] * ex;
        let dhdx = (at(x1, iz as usize) - at(x0, iz as usize)) / ((x1 - x0).max(1) as f32 * spacing.0);
        let dhdz = (at(ix as usize, z1) - at(ix as usize, z0)) / ((z1 - z0).max(1) as f32 * spacing.1);
        normalize3([-dhdx, 1.0, -dhdz])
    };
    let lerp = |a: [f32; 3], b: [f32; 3], t: f32| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
    let (ix, iz) = (cx as isize, cz as isize);
    let n = lerp(lerp(vertex(ix, iz), vertex(ix + 1, iz), u), lerp(vertex(ix, iz + 1), vertex(ix + 1, iz + 1), u), v);
    let s = normalize3(n);
    let base = terrain_patch_normal(heights, w, spacing, ex, u, v, cx, cz);
    if s[0] * base[0] + s[1] * base[1] + s[2] * base[2] > 0.0 { s } else { base }
}
```

(`(x1 - x0).max(1) as f32` is a `usize → f32` cast. Use `num::f32_from_usize`, gated `#[cfg(test)]` today: lift that gate. Keep the gate test green.)

- [ ] **Step 0: Capture the pre-task image** (for DoD 6). With the pre-task tree built (BUILD_PYD), run:
  `PYTHONPATH=C:/tmp/splat-fused/python FORGE3D_NO_BOOTSTRAP=1 D:/forge3d/.venv/Scripts/python -c "import sys,numpy as np; sys.path.insert(0,'tests'); from _splat_fusion import manifest, scene_kwargs; from forge3d import splat; np.save('D:/forge3d/cache/tmp/splat_limits_pre_t7.npy', splat.render_fused(**scene_kwargs(manifest()), samples=8, width=128, height=128, seed=0))"`
- [ ] **Step 1: Write failing CPU tests** (`kernel.rs`):
  - `smooth_terrain_normal_is_continuous_across_cell_edges`: seeded random DEM 33×33 (`hash_unit` from `stream`), spacing (6, 6), heights in [0, 300). For every interior edge between cells `(cx, cz)` and `(cx+1, cz)`, at `v ∈ {0.1, 0.5, 0.9}`: `|smooth(cx, u=1) − smooth(cx+1, u=0)|∞ ≤ 1e-6`. Same along z. Sensitivity: the max jump of `terrain_patch_normal` across the same edges is `≥ 1e-2`.
  - `smooth_terrain_normal_tracks_the_analytic_surface`: `h(x,z) = 120·tanh(x/25) + 6·sin(z/17)` sampled at 6 m on 65×65 centred at 0. At 10,000 hashed points, mean angular error vs the analytic normal: `smooth ≤ 0.7 × patch`.
- [ ] **Step 2: Write failing GPU tests** (`test_splat_fusion_occlusion.rs`):
  - `gpu_smooth_terrain_normal_matches_cpu_mirror`: fixture render 160², spp 1, frames 1. For every terrain centre-ray pixel, recover `(cx, cz, u, v)` from `out.position`; the angle between `out.normal` and `terrain_smooth_normal(...)` is `≤ 1e-3` rad for 100% of pixels (≥ 2000 pixels).
  - `smooth_terrain_normals_add_no_self_shadow_acne`: fixture 160², `terrain_smooth_normals` false vs true (via `FusionParams`). Terrain pixels where `transmittance[4i+3]` goes from `1.0` (off) to `< 1.0` (on) are `≤ 0.1%` of terrain pixels.
- [ ] **Step 3: Write the failing kernel-source test** (`fused_kernel_tests`): `source.matches("fn terrain_normal_at_base(").count() == 1`, `source.matches("fn terrain_normal_at(").count() == 1`, and `hybrid_kernel()` still contains `fn terrain_normal_at(` and not `_base`.
- [ ] **Step 4: Run, expect FAIL**, then **Step 5: implement** as listed.
- [ ] **Step 6: Run.** UNIT `smooth_terrain` + `fused_kernel_tests`; GPU `smooth_terrain`; GPU full `test_splat_fusion_occlusion` (regenerate the golden per the global rule if needed); BUILD_PYD; PY `tests/test_splat_fusion_occlusion.py tests/test_splat_api.py`.

**Definition of Done:**
1. Both CPU tests pass with the stated thresholds (continuity `≤ 1e-6` with base jump `≥ 1e-2`; analytic error ratio `≤ 0.7`).
2. GPU mirror test: 100% of ≥ 2000 terrain pixels within `1e-3` rad of the CPU mirror.
3. Acne test: `≤ 0.1%` of terrain pixels newly shadowed.
4. `fused_kernel_tests` (all, including `default_hybrid_kernel_is_untouched...`) pass.
5. GPU full `test_splat_fusion_occlusion`: five IoU cases `> 0.9`; golden passes (regenerated with a recorded pre-SSIM if needed).
6. Python `terrain_smooth_normals=False` reproduces the pre-task image: rendering the Step 0 call again with `terrain_smooth_normals=False` gives rgba identical byte-for-byte to `D:/forge3d/cache/tmp/splat_limits_pre_t7.npy` (`np.array_equal` printed `True` in the evidence).

---

### Task 8: LiDAR surfels: oriented discs for surface points (L5) **[changes default image]**

**Files:**
- Create: `src/splat/surfel.rs` (Rust only, not WGSL): `estimate_point_surfels`, `sym_eigen3`, `oct_encode`, `oct_decode`, `SURFEL_SPHERE: u32 = 0xFFFF_FFFF`.
- Modify: `src/splat/mod.rs` (`pub mod surfel;`).
- Modify: `src/splat/stream.rs`. `PointGpu` becomes `{ pos_radius: [f32; 4], color: [f32; 3], normal_oct: u32 }` (still 32 B, `Pod`); `DecodedPage::from_payload(page, payload, lidar_radius, surfels: bool)` fills `normal_oct` per page via `estimate_point_surfels` (or `SURFEL_SPHERE` when `surfels == false`). Update every caller (`grep -rn "from_payload(" src tests`).
- Modify: `src/shaders/fusion/unified_occlusion.wgsl`. `FusionPoint { pos_radius: vec4<f32>, color: vec3<f32>, normal_oct: u32 }`; add `fusion_oct_decode`; `fusion_sphelet` handles discs; `fusion_point_page_closest` uses the disc normal; replace `point.color.rgb` with `point.color`.
- Modify: `src/splat/kernel.rs`. `disc_coverage` CPU mirror.
- Modify: `src/splat/fusion.rs`. `FusionParams.lidar_surfels: bool` (default `true`) feeds `from_payload`.
- Modify: `src/path_tracing/fused_reference.rs:365-377`. Surfel points get a two-sided 16-gon proxy (radius `sphelet_proxy_radius(...)` × area-equivalence factor `sqrt(2π / (16·sin(2π/16)))`) in the disc plane; sphere points keep the icosahedron.
- Modify: `tests/test_splat_fusion_occlusion.rs` `fused_transmittance_matches_brute_force_cpu_oracle` (`:614-715`). The oracle reads the `PointGpu` records produced by `DecodedPage::from_payload` (the same data the GPU sees) and evaluates `disc_coverage`/`sphelet_coverage` accordingly.
- Native + Python `lidar_surfels: bool = True`; `.pyi`.

**Normative estimator** (`estimate_point_surfels(positions: &[[f32; 3]], radius: f32) -> Vec<u32>`, returns `normal_oct` per point):
- Grid hash with cell `2·radius`. Neighbours are the points within `3·radius` (excluding self).
- If fewer than 6 neighbours → `SURFEL_SPHERE`.
- Covariance of `{self} ∪ neighbours` (f64); eigenvalues `λ0 ≤ λ1 ≤ λ2` via `sym_eigen3` (cyclic Jacobi, ≤ 32 sweeps, stop when `|a01|+|a02|+|a12| < 1e-18`).
- Planar iff `λ0 ≤ 0.08·λ1` **and** `λ1 ≥ 0.2·λ2`; else `SURFEL_SPHERE`.
- Normal = eigenvector of `λ0`, flipped so `n.y ≥ 0`; for `|n.y| < 1e-6`, flipped so the first non-zero component is positive. Returns `oct_encode(n)`.

Oct packing (Rust; the WGSL decode must be its exact inverse):

```rust
pub fn oct_encode(n: [f32; 3]) -> u32 {
    let l1 = n[0].abs() + n[1].abs() + n[2].abs();
    let (mut x, mut y) = (n[0] / l1, n[1] / l1);
    if n[2] < 0.0 {
        let (ox, oy) = (x, y);
        x = (1.0 - oy.abs()) * if ox >= 0.0 { 1.0 } else { -1.0 };
        y = (1.0 - ox.abs()) * if oy >= 0.0 { 1.0 } else { -1.0 };
    }
    let q = |v: f32| ((v.clamp(-1.0, 1.0) * 0.5 + 0.5) * 65535.0).round() as u32;
    q(x) | (q(y) << 16)
}
```

```wgsl
const FUSION_SURFEL_SPHERE: u32 = 0xffffffffu;
fn fusion_oct_decode(packed: u32) -> vec3<f32> {
    let e = vec2<f32>(f32(packed & 0xffffu), f32(packed >> 16u)) * (2.0 / 65535.0) - vec2<f32>(1.0);
    var n = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    let t = max(-n.z, 0.0);
    n.x = n.x + select(t, -t, n.x >= 0.0);
    n.y = n.y + select(t, -t, n.y >= 0.0);
    return normalize(n);
}
// In fusion_sphelet: when point.normal_oct != FUSION_SURFEL_SPHERE, intersect the disc
// (centre pos_radius.xyz, normal n, radius r): denom = dot(d, n); |denom| < 1e-6 -> no coverage;
// t = dot(c - o, n) / denom; q = o + t d - c; b2 = dot(q, q);
// coverage = lidar_opacity * (1 - b2 / r^2) when tmin < t < tmax and b2 < r^2; t_hit = t.
```

- [ ] **Step 0: Capture the pre-task image** (for DoD 7): the Task 7 Step 0 command with the post-Task-7 tree, saved to `D:/forge3d/cache/tmp/splat_limits_pre_t8.npy`.
- [ ] **Step 1: Write failing CPU tests** (`surfel.rs`):
  - `oct_round_trip_error_is_below_0_02_degrees` (10,000 hashed unit vectors, max error `≤ 0.02°`); `SURFEL_SPHERE` is never produced by `oct_encode` (also check `(1,0,0)`, `(0,1,0)`, `(0,0,±1)`).
  - `sym_eigen3_decomposes_symmetric_matrices`: `‖A v_i − λ_i v_i‖ ≤ 1e-9·‖A‖` for 1000 random SPD matrices; eigenvectors orthonormal within `1e-9`.
  - `flat_grid_becomes_horizontal_surfels`: 40×40 grid, spacing 1, jitter ±0.05 in xz and ±0.02 in y, radius 0.8 → for points ≥ 3 from the edge: ≥ 95% are surfels with `|n·ŷ| ≥ 0.995`.
  - `vertical_wall_becomes_vertical_surfels`: same grid in the x = const plane → ≥ 95% surfels with `|n·x̂| ≥ 0.99`.
  - `isotropic_cloud_stays_spheres`: 2000 hashed points in a ball of radius 6, radius 0.8 → `≤ 5%` surfels.
  - `estimator_is_deterministic`: two runs give identical `Vec<u32>`.
  - In `kernel.rs`, `disc_coverage_closed_forms`: centre hit → `opacity`; `b = r/2` → `0.75·opacity`; parallel ray → `0`; and a ray leaving a horizontal grid of coplanar discs (`o = p + 1e-3·ŷ`, elevation 10°) has coverage 0 against every disc, while `sphelet_coverage` for the same grid is `> 0` (sensitivity).
- [ ] **Step 2: Write failing GPU tests** (`test_splat_fusion_occlusion.rs`). Build point page stores with `PageStoreWriter::create(path, PageKind::Points, 4096, 0)` + `push_point_page(&positions, &rgba)` + `finish()`, then `PageStoreFile::open`. No terrain.
  - `dense_lidar_plane_does_not_shadow_itself`: 120×120 grid, spacing 1.0, y = 0, radius 0.85, opacity 1. Camera at `(0, 60, -70)` looking at `(0, 0, 0)`. For sun elevations 10°, 22° and 40° (azimuth 45°), with `lidar_surfels = true`: the mean `T_lidar` (`transmittance[4i+2]`) over LiDAR centre-ray hits with `|x|, |z| ≤ 50` is `≥ 0.98`, over ≥ 2000 pixels. With `lidar_surfels = false` at 22°: mean `≤ 0.90` (proves the test is sensitive).
  - `lidar_wall_still_casts_its_shadow`: the plane above plus a wall of points on `x = 0`, `y ∈ [0, 8]`, `z ∈ [-20, 20]`, spacing 0.5. Sun azimuth 0° (toward +x), elevation 30°. Ground pixels with `x ∈ [-12, -2]`, `|z| ≤ 15` (inside the analytic shadow `x ∈ [-8/tan30°, 0)`) have `T_lidar ≤ 0.05` for ≥ 95% of them (≥ 200 pixels), with surfels on.
- [ ] **Step 3: Run, expect FAIL**, then **Step 4: implement** as listed.
- [ ] **Step 5: Run.** UNIT `surfel`, `disc_coverage`; GPU `dense_lidar_plane`, `lidar_wall`, `fused_transmittance_matches_brute_force_cpu_oracle`; GPU full `test_splat_fusion_occlusion` (regenerate the golden per the global rule if needed); BUILD_PYD; PY suite.

**Definition of Done:**
1. All six `surfel.rs` tests and `disc_coverage_closed_forms` pass with the stated thresholds.
2. `dense_lidar_plane_does_not_shadow_itself`: `≥ 0.98` at 10°/22°/40° with surfels; `≤ 0.90` at 22° without.
3. `lidar_wall_still_casts_its_shadow`: `≥ 95%` of ≥ 200 shadow pixels have `T_lidar ≤ 0.05`.
4. Oracle test passes with its existing threshold, now comparing against `disc_coverage` for surfels.
5. GPU full `test_splat_fusion_occlusion`: five IoU cases `> 0.9` **with surfels on and disc proxies in the reference**. If any case is ≤ 0.9, the task is not done: fix the model or the proxies, never the gate.
6. `size_of::<PointGpu>() == 32` asserted in a unit test (pool sizing unchanged).
7. PY `lidar_surfels=False` vs `True` on the fixture: the transmittance AOV differs on LiDAR pixels (`np.any(...)`), and the Step 0 call with `lidar_surfels=False` equals `D:/forge3d/cache/tmp/splat_limits_pre_t8.npy` byte-for-byte.

---

### Task 9: Resolution-independent memory: seamless tiles + AOV opt-out (L1) **[changes default image]**

**Owner decision (2026-10-03): Option A, always seamless.** Every fused render, tiled or not, runs with `camera_flags = 1`, so ReSTIR spatial reuse between neighbouring pixels is off everywhere and a tiled render is bit-identical to the monolithic one. Rejected: B (spatial reuse only for single-tile renders; preview and final would differ) and C (tile-local reuse; possible faint seams at low sample counts, untestable for identity). Accepted cost: slightly grainier soft-shadow edges at low sample counts; final renders compensate with samples.

**Files:**
- Modify: `src/shader_sources.rs`. `FUSED_TERRAIN_EDITS` gains the edit what=`"seamless reservoir sun direction"`, from `"        if (uniforms.camera_flags == 0u && prev_valid) {\n            sun_dir = normalize(prev_r.sample.direction);"` to `"        if (prev_valid) {\n            sun_dir = normalize(prev_r.sample.direction);"`. Rationale comment: in seamless mode spatial reuse is self-only (`pt_restir_spatial.wgsl:244-247`) and temporal reuse is per pixel, so the reservoir's sun-disc sample is tile-safe; without this edit tiles would shade from the disc centre only (hard shadows).
- Modify: `src/path_tracing/hybrid_compute/render_fused.rs`:
  - `FusedRenderDesc` gains `tile: Option<(u32, u32)>` and `aovs: bool`.
  - Add `pub(crate) const MAX_TILE_PIXELS: u64 = 1 << 20;`, `pub(crate) fn default_tile(w: u32, h: u32) -> (u32, u32)` (= `(w, h)` if `w*h <= MAX_TILE_PIXELS`, else `(w.min(1024), h.min(1024))`), and `pub(crate) fn fused_tiles(w: u32, h: u32, tile: (u32, u32)) -> Vec<(u32, u32, u32, u32)>` (row-major `(x, y, tw, th)`; the last column/row may be smaller).
  - Split `render_fused_view` into a view-level part (env bake, `TerrainPtScene`, lighting/hybrid/terrain/earth-curvature UBOs, scene/light buffers: built once per view) and `fn render_fused_tile(&self, view: &FusedViewResources, desc: &FusedRenderDesc, fused: &mut FusedGpu, tick: &mut u64, rect: (u32, u32, u32, u32)) -> Result<FusedTileOutput, RenderError>` containing today's `:730-1340` per-pixel code with `Uniforms { width: tw, height: th, full_width: desc.width, full_height: desc.height, pixel_offset_x: x, pixel_offset_y: y, camera_flags: 1, sensor_rect: [x/W, y/H, (x+tw)/W, (y+th)/H], cam_aspect: W/H, .. }`. Compute the sensor rect with `f32_from_u32`; validate as `render_terrain.rs:1086-1131` does.
  - Assemble tile outputs into full-size vectors (`rgba` 4 B/px, `radiance/albedo/normal/direct/position` 3 f32/px, `transmittance` 4 f32/px, `depth/sun_cosine/self_bias/reservoir_visibility` 1 f32/px, `hit_kind` 1 B/px). Aggregates: `frames` (equal across tiles), `restarts` and `stale_frames` (sums), `variance` (max), `reservoir_valid_count` (sum).
  - `aovs == false`: `aov_flags = 0` on every frame; `AovFrames::new(device, 1, 1, &aovs_all)`; skip the AOV, G-buffer and reservoir readbacks; AOV output vectors are empty. Beauty is unaffected: the AOV block runs after accumulation (`hybrid_terrain_traversal.wgsl:884`) and draws no beauty random numbers.
- Modify: native `render_fused` gains `tile_width: Option<u32> = None`, `tile_height: Option<u32> = None`, `aovs: bool = true`. Python `render_fused(..., tile: Optional[Tuple[int, int]] = None)` validates `1 <= tw <= width`, `1 <= th <= height` (`ValueError`) and passes `aovs=return_aovs`. When `return_aovs` is false only `rgba` is read back. Update `.pyi`. Update every `FusedRenderDesc { .. }` literal (`grep -rn "FusedRenderDesc {" src tests`) with `tile: None, aovs: true`.
- Test: `render_fused.rs` (pure tiling unit tests), `fused_kernel_tests`, `tests/test_splat_fusion_occlusion.rs`, `tests/test_splat_api.py`.

- [ ] **Step 1: Write failing pure tests** (`render_fused.rs` `#[cfg(test)]`):
  - `fused_tiles_cover_every_pixel_exactly_once` for `(300, 200, (128, 96))`, `(1920, 1080, (1024, 1024))`, `(7, 5, (7, 5))`: build a coverage counter and assert all counts are 1; every tile is within bounds.
  - `default_tile_is_single_up_to_one_megapixel`: `default_tile(1024, 1024) == (1024, 1024)`, `default_tile(1920, 1080) == (1024, 1024)`, `default_tile(3840, 2160) == (1024, 1024)`.
- [ ] **Step 2: Write the failing kernel-source test** (`fused_kernel_tests`): `body(&source, "main_terrain")` contains `"if (prev_valid) {\n            sun_dir = normalize(prev_r.sample.direction);"` and does not contain `"camera_flags == 0u && prev_valid"`.
- [ ] **Step 3: Write failing GPU tests** (`test_splat_fusion_occlusion.rs`):

```rust
#[test]
fn tiled_fused_render_equals_the_monolithic_render_bit_for_bit() {
    if !gpu_available() { return; }
    let fixture = FixtureScene::load(fixture_dir()).unwrap();
    let params = fixture_params(&fixture);
    let scene = FusedScene::new(fixture_sources(&fixture, &params), params).unwrap();
    let tracer = HybridPathTracer::new_fused().unwrap();
    let mut desc = fused_desc(&scene, &fixture, 0, 2, 4);
    desc.width = 300;
    desc.height = 200;
    let whole = tracer.render_fused(&desc).unwrap();
    for tile in [(150, 100), (128, 96)] {
        desc.tile = Some(tile);
        let tiled = tracer.render_fused(&desc).unwrap();
        assert_eq!(whole.rgba, tiled.rgba, "{tile:?} rgba");
        assert_eq!(whole.hit_kind, tiled.hit_kind, "{tile:?} hit_kind");
        for (name, a, b) in [("radiance", &whole.radiance, &tiled.radiance), ("albedo", &whole.albedo, &tiled.albedo),
            ("normal", &whole.normal, &tiled.normal), ("position", &whole.position, &tiled.position),
            ("direct", &whole.direct, &tiled.direct), ("transmittance", &whole.transmittance, &tiled.transmittance),
            ("depth", &whole.depth, &tiled.depth), ("sun_cosine", &whole.sun_cosine, &tiled.sun_cosine),
            ("self_bias", &whole.self_bias, &tiled.self_bias),
            ("reservoir_visibility", &whole.reservoir_visibility, &tiled.reservoir_visibility)] {
            assert_eq!(bits(a), bits(b), "{tile:?} {name}");
        }
        assert_eq!(whole.variance.to_bits(), tiled.variance.to_bits());
        assert_eq!(whole.reservoir_valid_count, tiled.reservoir_valid_count);
    }
}

#[test]
fn fused_peak_memory_does_not_grow_with_output_resolution() {
    if !gpu_available() { return; }
    let fixture = FixtureScene::load(fixture_dir()).unwrap();
    let params = fixture_params(&fixture);
    let scene = FusedScene::new(fixture_sources(&fixture, &params), params).unwrap();
    let tracer = HybridPathTracer::new_fused().unwrap();
    let run = |w: u32, h: u32, tile: Option<(u32, u32)>, aovs: bool| {
        let mut d = fused_desc(&scene, &fixture, 0, 1, 1);
        d.width = w; d.height = h; d.tile = tile; d.aovs = aovs;
        let out = tracer.render_fused(&d).unwrap();
        assert_eq!(out.rgba.len(), (w * h * 4) as usize);
        out.peak_total_bytes
    };
    const MIB: u64 = 1 << 20;
    let base = run(960, 540, None, true);
    assert!(run(1920, 1080, Some((960, 540)), true) <= base + MIB);
    let base_lean = run(960, 540, None, false);
    assert!(run(3840, 2160, Some((960, 540)), false) <= base_lean + MIB);
    assert!(base_lean + 48 * 960 * 540 <= base, "AOV textures must be freed when aovs = false");
    assert!(base <= BUDGET_BYTES);
}

#[test]
fn aov_opt_out_leaves_the_beauty_untouched() {
    if !gpu_available() { return; }
    let fixture = FixtureScene::load(fixture_dir()).unwrap();
    let params = fixture_params(&fixture);
    let scene = FusedScene::new(fixture_sources(&fixture, &params), params).unwrap();
    let tracer = HybridPathTracer::new_fused().unwrap();
    let mut d = fused_desc(&scene, &fixture, 192, 2, 4);
    let with = tracer.render_fused(&d).unwrap();
    d.aovs = false;
    let without = tracer.render_fused(&d).unwrap();
    assert_eq!(with.rgba, without.rgba);
    assert!(without.hit_kind.is_empty() && without.albedo.is_empty());
}
```

(`fused_desc(&scene, &fixture, 0, ...)` sets size 0, then width/height are assigned explicitly; `validate` runs at render time, so a zero placeholder is harmless.)

- [ ] **Step 4: Run, expect FAIL**, then **Step 5: implement** as listed.
- [ ] **Step 6: Run.** UNIT `fused_tiles`, `default_tile`, `fused_kernel_tests`; GPU `tiled_fused_render`, `fused_peak_memory`, `aov_opt_out`; GPU full `test_splat_fusion_occlusion` (regenerate the golden per the global rule; the seamless edit changes ReSTIR reuse); BUILD_PYD; PY: add and run `test_render_fused_1080p_fits_the_budget` (fixture at 1920×1080, `samples=2`, `return_aovs=True` → shape `(1080, 1920)` AOVs, `stats["peak_total_bytes"] <= 512 MiB`) and `test_render_fused_rejects_invalid_tiles` (`tile=(0, 10)` → `ValueError`).

**Definition of Done:**
1. Pure tests pass (exact single coverage; default-tile rule).
2. Kernel-source test passes, and the default kernel stays untouched.
3. `tiled_fused_render_equals_the_monolithic_render_bit_for_bit` passes for both tile sizes, including the ragged 128×96.
4. `fused_peak_memory_does_not_grow_with_output_resolution` passes: 1080p and 4K peaks ≤ the 960×540 peak + 1 MiB, and AOV opt-out frees ≥ 48 B/px.
5. `aov_opt_out_leaves_the_beauty_untouched` passes.
6. GPU full `test_splat_fusion_occlusion`: five IoU cases `> 0.9`; golden passes after a recorded regeneration.
7. PY 1080p test: shape correct, peak ≤ 512 MiB; invalid tile → `ValueError`.

---

### Task 10: Python multi-view `render_fused_sequence` (L8)

**Files:**
- Modify: `src/py_functions/splat.rs`. Add `#[pyfunction] render_fused_sequence(py, *, views: Vec<Bound<PyDict>>, <every scene/option kwarg of render_fused except camera and sun fields>, certificate = None, cache = None) -> PyResult<Py<PyList>>`. Each view dict needs `cam_origin` and `cam_look_at`; optional `cam_up` (default `(0, 1, 0)`), `fov_y_deg` (45), `sun_azimuth_deg` (135), `sun_elevation_deg` (45), `sun_intensity` (2.5), `sun_color` (`(1, 0.97, 0.92)`), `exposure` (1), `seed` (0); unknown keys → `ValueError`. Build `SceneInputs` once, `HybridPathTracer::new_fused()` once, call `tracer.render_fused_sequence(&descs)`, return a list of `output_dict`s.
- Modify: `src/py_module/functions/rendering.rs` (register), `python/forge3d/splat.py` (`@dataclass(frozen=True) class FusedView: camera: FusedCamera; sun_azimuth_deg: float = 135.0; sun_elevation_deg: float = 45.0; sun_intensity: float = 2.5; sun_color: Tuple[float, float, float] = (1.0, 0.97, 0.92); exposure: float = 1.0; seed: int = 0`, and `def render_fused_sequence(*, splats=None, pointcloud=None, terrain=None, views: Sequence[FusedView], samples: int = 64, ...same options as render_fused..., return_aovs: bool = False, certificate=False, cache=None) -> List[Union[np.ndarray, FusedRenderResult]]`), `python/forge3d/__init__.py` (`__all__` += `"render_fused_sequence"`, `"FusedView"`), both `.pyi`, and `tests/test_api_contracts.py` `EXPECTED_FUNCTIONS` += `"render_fused_sequence"`.
- Test: `tests/test_splat_api.py`.

- [ ] **Step 1: Write the failing tests:**

```python
@pytest.mark.skipif(not gpu_available(), reason="no usable GPU adapter for the fused render")
def test_render_fused_sequence_matches_independent_renders_and_reuses_pages():
    m = manifest()
    kw = scene_kwargs(m)
    cam = kw.pop("camera")
    sun = {k: kw.pop(k) for k in ("sun_azimuth_deg", "sun_elevation_deg", "sun_intensity", "sun_color")}
    other = splat.FusedCamera(origin=tuple(np.array(cam.origin) + [6.0, 0.0, -4.0]), look_at=cam.look_at, up=cam.up, fov_y_deg=cam.fov_y_deg)
    views = [splat.FusedView(camera=cam, **sun), splat.FusedView(camera=other, **sun), splat.FusedView(camera=cam, **sun)]
    common = dict(samples=8, width=128, height=128, return_aovs=True)
    seq = splat.render_fused_sequence(**kw, views=views, **common)
    assert len(seq) == 3
    for view, got in zip(views, seq):
        alone = splat.render_fused(**kw, camera=view.camera, **sun, **common)
        np.testing.assert_array_equal(got.rgba, alone.rgba)
        np.testing.assert_array_equal(got.radiance.view(np.uint32), alone.radiance.view(np.uint32))
    loads = [r.stats["paging"]["loads"] for r in seq]  # cumulative over the sequence
    assert loads[2] == loads[1], f"repeated view re-streamed pages: {loads}"
    assert max(r.stats["peak_total_bytes"] for r in seq) <= 512 * 1024 * 1024
    with pytest.raises(ValueError, match="unknown view key"):
        splat._native("render_fused_sequence")(views=[{"cam_origin": (0, 1, 2), "cam_look_at": (0, 0, 0), "bogus": 1}],
                                               heights=np.zeros((4, 4), np.float32))
```

- [ ] **Step 2: Run, expect FAIL**, then **Step 3: implement**, then **Step 4: run** PY `tests/test_splat_api.py tests/test_api_contracts.py tests/test_render_certificate_contract.py`.

**Definition of Done:**
1. The test passes, not skipped: each sequence result is byte-identical (rgba, radiance bits) to the independent render, and the repeated view adds zero page loads.
2. PY `tests/test_api_contracts.py tests/test_render_certificate_contract.py` pass (the new `render_*` accepts `certificate=` and `cache=`).
3. `forge3d.render_fused_sequence` and `forge3d.FusedView` import from the package root.

---

### Task 11: Final verification gate, docs, demo evidence

**Files:** `docs/splat-fused.md` (rewrite "Limits"; add "Measured" with the numbers below), `CHANGELOG.md` (one entry per fixed limit), the evidence file.

- [ ] **Step 1: Docs.** Remove the fixed limits from `docs/splat-fused.md` "Limits". Keep exactly the out-of-scope list from this plan, and document `tile`, `albedo_map`, `albedo_sampling`, `terrain_smooth_normals`, `lidar_surfels`, the self-bias kwargs, `GaussianSplatCloud.transformed` and `render_fused_sequence`.
- [ ] **Step 2: Full gate** (each command's tail goes into the evidence file):

```bash
cargo fmt --all --manifest-path $M -- --check
cd C:/tmp/splat-fused && cargo forge3d-clippy && cargo forge3d-clippy-acceptance
cargo test --release --manifest-path $M --workspace --features $F -- --test-threads=1 --skip gpu_extrusion --skip brdf_tile
FORGE3D_SPLAT_FUSION_REQUIRED_GPU=1 cargo test --release --manifest-path $M --features $F --test test_splat_fusion_occlusion -- --test-threads=1 --nocapture
```
then BUILD_PYD and PY `tests/test_splat_fusion_occlusion.py tests/test_splat_api.py tests/test_api_contracts.py tests/test_render_certificate_contract.py tests/test_no_silent_degradation.py tests/test_world_coord_f32_gate.py tests/test_hybrid_terrain_pt.py`.

- [ ] **Step 3: Demo evidence** (manual; scripts under `D:/forge3d_data/splat_fused_demos`, outside the repo):
  - Mount St. Helens: one frame at 1920×1080 with default tiling → record `peak_total_bytes` and seconds.
  - Lauterbrunnen: one frame at 1920×1080 with `terrain_6m.npy`, an albedo map derived procedurally from swissALTI3D slope/elevation (meadow < 30°, rock ≥ 30°, snow > 2600 m; labelled "procedural" in the caption), `lidar_surfels=True`, `terrain_smooth_normals=True` → record peak, seconds and `terrain_bytes`.

**Definition of Done (whole plan):**
1. Every task's DoD is met, with an evidence block per task in the evidence file.
2. All Step 2 commands exit 0: fmt clean, both clippy aliases with 0 warnings, the curated workspace test run, the GPU acceptance run (5 IoU cases > 0.9, billion-primitive and ≤ 512 MiB assertions), and every listed pytest file with 0 failures and **0 skips** on the NVIDIA machine.
3. `git status --short` shows only files named in this plan (plus `docs/superpowers/plans/*`), and nothing is committed.
4. Demo evidence: both 1920×1080 frames rendered with peak ≤ 512 MiB.
5. The final report to the owner lists each limit L1–L10 with its test name and measured number, the golden regenerations made (task and pre-SSIM), and the out-of-scope items.
