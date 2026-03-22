# TV12 — Terrain Offline Render Quality

**Date:** 2026-03-22
**Epic:** TV12 (from `docs/plans/2026-03-16-terrain-viz-epics.md`)
**Priority:** P1 (Phase 1 — Render Quality)
**Feasibility:** F2 (18-32 pd)

---

## 1. Problem

Forge3D has terrain AOV export, TAA, and A-trous denoising, but it lacks a coherent offline terrain-quality pipeline. The current terrain shader (`terrain_pbr_pom.wgsl`) applies exposure, atmospheric fog, filmic tonemapping, and sRGB encoding inline before writing beauty output. There is no way to accumulate multiple jittered samples in HDR space, adaptively allocate samples, or apply a learned denoiser to the linear result.

The existing `accumulation_blend.wgsl` overwrites the accumulator rather than performing read-modify-write accumulation (lines 34-37: `let result = current; textureStore(accumulation, coords, result)`). `Frame.to_numpy()` rejects Rgba16Float (frame.rs:196-198). These gaps must be closed before offline quality is viable.

---

## 2. Architecture

### 2.1 Controller / Engine Split

**Rust/GPU (accumulation engine):** Owns HDR sample accumulation, jitter application, tile variance computation, HDR resolve, and tonemapping. Exposes batch-oriented primitives. Never decides adaptive policy.

**Python (offline controller):** Owns the render loop, adaptive stopping decisions, progress reporting, and OIDN post-processing. Calls Rust batch primitives.

### 2.2 Pipeline

```
begin_offline_accumulation(params, heightmap, material_set, env_maps)
  │
  ├─ loop:
  │    ├─ accumulate_batch(sample_count)        ← GPU renders N jittered samples, blends into HDR accum
  │    ├─ read_accumulation_metrics()            ← small tile-variance buffer readback
  │    └─ if converged or max_samples: break
  │
  ├─ resolve_offline_hdr()                       ← divide accum by count → HdrFrame + AovFrame
  │
  ├─ [optional] Python: oidn_denoise(hdr, albedo, normal)
  │
  └─ tonemap_offline_hdr(hdr_frame)              ← apply tonemap → Frame (uint8 PNG-ready)
```

### 2.3 Key Constraints

1. **TAA/reprojection bypassed.** Offline accumulation and temporal filtering must not stack. The render pass sets `taa_enabled = false` during accumulation.
2. **AOVs rendered during accumulation.** Beauty, albedo, and normal accumulate alongside each other so OIDN inputs stay spatially aligned with beauty.
3. **Convergence checks use a small GPU-produced tile-variance buffer** (per-tile Welford stats), not full-resolution readback.
4. **`Frame.to_numpy()` is export-only.** Never participates in the accumulation loop.
5. **Single-active-session semantics.** `begin_offline_accumulation()` while a session is already active, or calling `render_terrain_pbr_pom()` / `render_with_aov()` during an active offline session, raises an error rather than silently clobbering cached state.
6. **Scope is narrow.** The offline pipeline promises: accumulation → HDR resolve + AOV resolve → denoise → tonemap. It does not promise DoF, motion blur, lens effects, or bloom in the offline path. Those remain interactive-only until a future epic integrates them.

---

## 3. Rust GPU Primitives

Six methods on `TerrainRenderer`:

### 3.1 `begin_offline_accumulation(params, heightmap, material_set, env_maps)`

- Validates no offline session is already active; errors if so.
- Allocates or resizes:
  - **Beauty accumulation:** Two ping-pong Rgba32Float textures (A and B) for additive accumulation without ReadWrite storage.
  - **Albedo accumulation:** One Rgba32Float texture (additive, divided at resolve).
  - **Normal accumulation:** One Rgba32Float texture (additive, renormalized at resolve).
  - **Depth reference:** One R32Float texture (written once on sample 0, not accumulated).
  - **Per-sample luminance scratch:** One Rgba16Float texture (same size as beauty scratch) used by the variance compute shader to write per-pixel luminance. Read back as part of tile variance computation on CPU.
- Creates `JitterSequence` from `aa_seed` in params (R2 sequence). Pre-generates `max_samples` offsets when `adaptive=True`, or `aa_samples` offsets when `adaptive=False`. The existing implementation wraps via modulo, so this is safe.
- Caches all render inputs (material_set, env_maps, heightmap, decoded params) so subsequent calls to `accumulate_batch` do not re-pass them. Same `TerrainRenderParams` type used by `render_terrain_pbr_pom()`.
- Resets sample count to 0. Clears accumulation textures via `encoder.clear_texture()` (newly allocated textures are zero-initialized by wgpu, but resized/reused textures require explicit clearing).
- Sets internal flag `offline_session_active = true`.

### 3.2 `accumulate_batch(sample_count: u32) → OfflineBatchResult`

- Errors if no offline session is active.
- For each of `sample_count` iterations:
  1. Get next jitter offset from `JitterSequence`.
  2. Apply jitter to cached projection matrix via `apply_jitter_to_projection()`.
  3. Render main terrain pass to a scratch Rgba16Float target with **HDR output mode** (exposure + fog applied, tonemapping + sRGB encoding skipped). TAA disabled.
  4. Render AOV passes (albedo, normal) to scratch AOV targets with same jitter.
  5. On sample 0 only: copy depth to the depth reference texture.
  6. Dispatch `offline_accumulate.wgsl` compute shader: reads scratch beauty, reads current accumulation texture A, writes sum to accumulation texture B. Swap A↔B (ping-pong).
  7. Similarly accumulate albedo and normal into their buffers (additive, same ping-pong pattern).
  8. Increment `total_samples`.
- Returns `OfflineBatchResult { total_samples: u32, batch_time_ms: f64 }`.

### 3.3 `read_accumulation_metrics(target_variance: f32) → OfflineMetrics`

- Errors if no offline session is active.
- Reads back the current beauty accumulation buffer via a small staging buffer.
- **CPU-side tile variance computation** (not GPU atomics — WGSL has no `atomicAdd` on `f32`):
  - Divides the resolved image into tiles of `tile_size × tile_size` pixels.
  - Per tile: computes luminance variance using Welford's algorithm over the tile's pixels from the current accumulated average (sum / total_samples).
  - This readback is the resolved average at the current sample count, not the full per-sample history. The cost is one staging buffer copy (~33KB for tile stats, computed from a downsampled luminance map).
- Returns `OfflineMetrics`:
  - `total_samples: u32`
  - `mean_variance: f32` — average tile variance
  - `p95_variance: f32` — 95th percentile tile variance
  - `max_tile_variance: f32` — worst tile (diagnostic only, not used in stopping rule)
  - `converged_tile_ratio: f32` — fraction of tiles with variance < `target_variance`

### 3.4 `resolve_offline_hdr() → (HdrFrame, AovFrame)`

- Errors if no offline session is active.
- Dispatches resolve compute shader (`offline_resolve.wgsl`):
  - **Beauty:** reads from Rgba32Float accumulation buffer, divides by `total_samples`, writes to a **newly allocated** Rgba16Float texture → `HdrFrame`. The format conversion (Rgba32Float → Rgba16Float) happens in the shader; the resolve shader reads `texture_2d<f32>` and writes `texture_storage_2d<rgba16float, write>`.
  - **Albedo:** divides by `total_samples` → averaged albedo → AovFrame albedo channel.
  - **Normal:** divides by `total_samples`, then **renormalizes** each pixel (`normalize()` the averaged normal vector) → AovFrame normal channel.
  - **Depth:** copies the single-sample reference depth → AovFrame depth channel.
- Does **not** tonemap. Does **not** apply bloom/DoF/lens effects.
- The returned `HdrFrame` **owns its texture independently** (allocated fresh, not referencing the accumulation buffer). It survives offline session teardown in `tonemap_offline_hdr()`.
- Returns `(HdrFrame, AovFrame)`.

### 3.5 `upload_hdr_frame(data: np.ndarray, size: (u32, u32)) → HdrFrame`

- Creates a new Rgba16Float texture of the given size.
- Uploads the float32 numpy array `(H, W, 3)` or `(H, W, 4)` to the texture via `queue.write_texture()`.
- Returns a new `HdrFrame` that owns the texture independently.
- Used by the Python controller to re-upload denoised HDR data for tonemapping.

### 3.6 `tonemap_offline_hdr(hdr_frame: HdrFrame) → Frame`

- Runs `postprocess_tonemap.wgsl` on the HDR input using the tonemap settings from the cached params (operator, exposure, white point, gamma, LUT if configured).
- Writes to Rgba8UnormSrgb texture → `Frame`.
- Cleans up offline session state: releases accumulation buffers, clears cached render inputs, sets `offline_session_active = false`. The `HdrFrame` passed in (and any previously returned `HdrFrame` from `resolve_offline_hdr()`) survives teardown because it owns its texture independently.
- The returned `Frame` is a normal Frame: `save()` writes PNG, `to_numpy()` returns uint8 RGBA.

---

## 4. HdrFrame Type

New Rust/Python type alongside `Frame`:

```rust
#[pyclass]
pub struct HdrFrame {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    texture: wgpu::Texture,
    width: u32,
    height: u32,
    // Always Rgba16Float
}
```

Python methods:
- `to_numpy_f32() → np.ndarray` — returns `(H, W, 4)` float32 array (linear HDR). Uses existing `read_rgba_f32()` internal path.
- `save(path: str)` — writes `.exr` only. Errors on other extensions.
- `size → (int, int)`

This is the OIDN handoff point: Python calls `hdr_frame.to_numpy_f32()`, slices `[:,:,:3]` for beauty RGB, and passes to OIDN alongside AOV albedo/normal from `AovFrame`.

---

## 5. New Shaders

### 5.1 `offline_accumulate.wgsl`

Compute shader (8x8 workgroups). Replaces the broken `accumulation_blend.wgsl` for offline use.

```
Inputs:  current_sample (texture_2d<f32>), prev_accumulation (texture_2d<f32>)
Output:  next_accumulation (texture_storage_2d<rgba32float, write>)
Uniform: { sample_index: u32, width: u32, height: u32 }

Operation: next = prev + current  (additive; division happens at resolve)
```

Ping-pong between two textures avoids the ReadWrite storage limitation noted in the existing shader comments.

### 5.2 Tile variance computation (CPU-side)

No GPU variance shader. WGSL does not support `atomicAdd` on `f32`, and the workarounds (float-as-uint bit tricks, workgroup reductions) add complexity without meaningful benefit for an offline quality path.

Instead, `read_accumulation_metrics()` performs tile variance on the CPU:

1. Resolve the current accumulation buffer to a temporary HDR average (divide by N).
2. Read back a **downsampled luminance map** via a staging buffer — either the full resolved average (already needed for the final resolve) or a quarter-resolution luminance computed by a trivial `offline_luminance.wgsl` compute shader that writes `lum = 0.2126*R + 0.7152*G + 0.0722*B` per pixel to a quarter-res R32Float texture.
3. On the CPU (Rust side), divide the luminance map into tiles and compute per-tile variance using Welford's algorithm.
4. Return `OfflineMetrics` with sorted tile variances for p95 and ratio computation.

This keeps the GPU path simple and the convergence logic auditable.

### 5.3 Terrain shader HDR output mode

Add a uniform flag `offline_hdr_output: u32` to `TerrainUniforms` (or pack into an existing padding slot). When set to 1, the fragment shader's final output path changes from:

```wgsl
let tonemapped = tonemap_filmic_terrain(shaded);
final_color = tonemapped;
// + linear_to_srgb encoding
```

to:

```wgsl
final_color = shaded;  // post-exposure, post-fog, pre-tonemap, linear HDR
// no sRGB encoding
```

The render target format for offline passes is Rgba16Float instead of Rgba8UnormSrgb.

---

## 6. AOV Accumulation Semantics

| Buffer | Accumulation | Resolve | OIDN role |
|--------|-------------|---------|-----------|
| **Beauty** | Additive HDR (sum of linear samples) | Divide by N | Primary input |
| **Albedo** | Additive (sum) | Divide by N | Auxiliary guide |
| **Normal** | Additive (sum of unit vectors) | Divide by N, then renormalize | Auxiliary guide |
| **Depth** | Single sample (sample 0 reference) | Copy as-is | Not passed to OIDN core; export-only |

OIDN's core contract is `beauty + albedo + normal`. Depth is available in the AovFrame for export and EXR multichannel output but is not part of the denoiser input.

---

## 7. OIDN Integration

### 7.1 Module: `python/forge3d/denoise_oidn.py`

```python
def oidn_available() -> bool:
    """Runtime check for oidn package availability."""

def oidn_denoise(
    beauty: np.ndarray,        # (H, W, 3) float32, linear HDR
    albedo: np.ndarray | None,  # (H, W, 3) float32
    normal: np.ndarray | None,  # (H, W, 3) float32
    *,
    hdr: bool = True,
    quality: str = "high",      # "default", "high"
) -> np.ndarray:
    """Denoise using Intel Open Image Denoise."""
```

### 7.2 Configuration

Extend existing `DenoiseSettings.method` (terrain_params.py:838) to accept `'oidn'`:

```python
@dataclass
class DenoiseSettings:
    enabled: bool = False
    method: str = "atrous"  # 'atrous', 'bilateral', 'oidn', 'none'
    # ... existing fields unchanged ...
```

The existing `__post_init__` validator (`valid_methods = ("atrous", "bilateral", "none")`) must be updated to include `"oidn"` in the valid set.

No new `OfflineQualitySettings` dataclass for denoiser selection — it lives in the existing `DenoiseSettings`. Offline-specific settings (adaptive policy, batching) are new:

```python
@dataclass
class OfflineQualitySettings:
    """Offline accumulation and adaptive sampling policy.

    Does NOT duplicate aa_samples/aa_seed (use existing params) or
    denoiser selection (use DenoiseSettings.method).
    """
    enabled: bool = False
    adaptive: bool = False           # Enable adaptive sample allocation
    target_variance: float = 0.001   # Per-tile convergence threshold
    max_samples: int = 64            # Upper bound for adaptive mode
    min_samples: int = 4             # Minimum before first convergence check
    batch_size: int = 4              # Samples per accumulate_batch call
    tile_size: int = 16              # Variance tile size in pixels
    convergence_ratio: float = 0.95  # Fraction of tiles that must converge
```

`aa_samples` (existing field in `make_terrain_params_config`) serves as the fixed sample count when `adaptive=False`. `aa_seed` serves as the jitter seed. No duplication.

### 7.3 Quality Tiering and Fallback

| `DenoiseSettings.method` | Behavior |
|--------------------------|----------|
| `'none'` | No denoising |
| `'atrous'` | A-trous wavelet (always available, existing implementation) |
| `'oidn'` | OIDN if `oidn` package installed; warns and falls back to `'atrous'` if unavailable |

The offline controller logs which denoiser was actually used in `OfflineResult.metadata`.

---

## 8. Python Offline Controller

### 8.1 Module: `python/forge3d/offline.py`

```python
@dataclass
class OfflineProgress:
    samples_so_far: int
    max_samples: int
    mean_variance: float
    p95_variance: float
    converged_ratio: float
    elapsed_ms: float

@dataclass
class OfflineResult:
    frame: Frame              # Tonemapped uint8 beauty
    hdr_frame: HdrFrame       # Linear HDR beauty (for EXR export)
    aov_frame: AovFrame       # Albedo, normal, depth
    metadata: dict             # samples_used, final_variance, denoiser_used, etc.

def render_offline(
    renderer: TerrainRenderer,
    material_set: MaterialSet,
    env_maps: IBL,
    params: TerrainRenderParams,
    heightmap: np.ndarray,
    *,
    settings: OfflineQualitySettings,
    progress_callback: Callable[[OfflineProgress], None] | None = None,
) -> OfflineResult:
```

### 8.2 Controller Loop

```python
def render_offline(...):
    renderer.begin_offline_accumulation(params, heightmap, material_set, env_maps)

    total = settings.max_samples if settings.adaptive else params.aa_samples
    rendered = 0

    while rendered < total:
        batch = min(settings.batch_size, total - rendered)
        result = renderer.accumulate_batch(batch)
        rendered = result.total_samples

        # Single metrics readback per iteration (avoid redundant GPU staging copies)
        metrics = None
        if progress_callback or (settings.adaptive and rendered >= settings.min_samples):
            metrics = renderer.read_accumulation_metrics(settings.target_variance)

        if progress_callback and metrics is not None:
            progress_callback(OfflineProgress(
                samples_so_far=rendered, max_samples=total,
                mean_variance=metrics.mean_variance,
                p95_variance=metrics.p95_variance,
                converged_ratio=metrics.converged_tile_ratio,
                elapsed_ms=result.batch_time_ms,
            ))

        if settings.adaptive and rendered >= settings.min_samples and metrics is not None:
            if (metrics.converged_tile_ratio >= settings.convergence_ratio
                    or metrics.p95_variance < settings.target_variance):
                break

    hdr_frame, aov_frame = renderer.resolve_offline_hdr()

    # Denoise between resolve and tonemap (operates in linear HDR space)
    denoiser_used = "none"
    if params.denoise.enabled and params.denoise.method != "none":
        method = params.denoise.method
        if method == "oidn" and not oidn_available():
            warnings.warn("oidn package not installed; falling back to atrous denoiser")
            method = "atrous"

        beauty_hdr = hdr_frame.to_numpy_f32()[:, :, :3]
        albedo_np = aov_frame.albedo()   # (H, W, 3) float32
        normal_np = aov_frame.normal()   # (H, W, 3) float32

        if method == "oidn":
            denoised = oidn_denoise(beauty_hdr, albedo=albedo_np, normal=normal_np)
            denoiser_used = "oidn"
        elif method == "atrous":
            denoised = atrous_denoise(
                beauty_hdr, albedo=albedo_np, normal=normal_np,
                iterations=params.denoise.iterations,
                sigma_color=params.denoise.sigma_color,
                sigma_normal=params.denoise.sigma_normal,
            )
            denoiser_used = "atrous"

        # Re-upload denoised HDR to GPU for tonemapping
        hdr_frame = renderer.upload_hdr_frame(denoised, hdr_frame.size)

    frame = renderer.tonemap_offline_hdr(hdr_frame)

    return OfflineResult(
        frame=frame,
        hdr_frame=hdr_frame,  # HdrFrame owns its texture; survives session teardown
        aov_frame=aov_frame,
        metadata={
            "samples_used": rendered,
            "denoiser_used": denoiser_used,
            "final_variance": metrics.p95_variance if metrics else None,
            "converged_ratio": metrics.converged_tile_ratio if metrics else None,
        },
    )
```

---

## 9. Adaptive Stopping Rule

The default stopping policy uses **two conditions (OR)**:

1. `converged_tile_ratio >= convergence_ratio` (default 0.95) — at least 95% of tiles have per-tile variance below `target_variance`.
2. `p95_variance < target_variance` — the 95th percentile tile variance is below threshold.

Either condition being true stops accumulation. This prevents one hot tile from pinning the entire render to `max_samples`.

`max_tile_variance` is reported in `OfflineMetrics` as a diagnostic but does not participate in the stopping decision.

---

## 10. Testing

### TV12.1 — Deterministic Offline Accumulation

| Test | Assertion |
|------|-----------|
| **Determinism** | Same `aa_seed`, same backend/adapter/driver → identical output bytes. Not claimed across different GPU drivers. |
| **Quality improvement** | Multi-sample (N=16) measurably reduces aliasing over single-sample on a high-frequency terrain scene (PSNR or edge-variance metric). |
| **TAA bypass** | Offline accumulation does not activate TAA reprojection. Verified by checking that TAA history is not consulted during offline batches. |
| **Single-sample baseline** | `aa_samples=1` through the offline path produces the offline single-sample baseline (same HDR output + tonemap path, not necessarily identical to `render_terrain_pbr_pom()` which uses the inline tonemap). |
| **Session guard** | `begin_offline_accumulation()` while active raises error. `render_terrain_pbr_pom()` during active offline session raises error. |

### TV12.2 — Adaptive Sampling

| Test | Assertion |
|------|-----------|
| **Early stop on uniform scene** | A flat-color terrain converges in fewer samples than `max_samples`. |
| **More samples on complex scene** | A high-frequency terrain uses more samples than `min_samples`. |
| **Max samples respected** | Adaptive never exceeds `max_samples`. |
| **Metrics queryable** | `read_accumulation_metrics()` returns valid `OfflineMetrics` with plausible values after each batch. |
| **Convergence trend** | After `min_samples`, the reported `converged_tile_ratio` is generally non-decreasing (small fluctuations from noisy batches are tolerated; the trend over 3+ batches must be upward). |

### TV12.3 — OIDN Integration

| Test | Assertion |
|------|-----------|
| **OIDN produces different output** | OIDN-denoised frame differs from undenoised frame (pixel diff > threshold). |
| **Graceful fallback** | When `oidn` not installed, method='oidn' falls back to `'atrous'` with a logged warning. |
| **Denoiser metadata** | `OfflineResult.metadata['denoiser_used']` reports the actually-used denoiser. |
| **No-denoise baseline** | `method='none'` output matches the non-denoised resolve. |
| **AOV alignment** | Albedo and normal passed to OIDN have same dimensions as beauty HDR. |

---

## 11. Scope Exclusions

The following are **explicitly not part of TV12**:

- Bloom, DoF, motion blur, lens effects in the offline path (interactive-only for now).
- Full-resolution per-pixel adaptive sampling (tile-based only).
- GPU-side OIDN execution (CPU/Python only).
- Audio, timeline, or shot-queue integration (TV18).
- Modification of the existing `render_terrain_pbr_pom()` or `render_with_aov()` paths.

---

## 12. File Change Summary

| File | Change |
|------|--------|
| `src/terrain/accumulation.rs` | Extend with ping-pong buffer pair, CPU tile variance computation, session state flag |
| `src/terrain/renderer/py_api.rs` | Add 6 new `#[pymethods]`: begin/accumulate/metrics/resolve/upload_hdr/tonemap |
| `src/terrain/renderer/core.rs` | Add offline state to `TerrainScene`, HDR render target, session guards |
| `src/shaders/offline_accumulate.wgsl` | New: additive ping-pong accumulation compute shader |
| `src/shaders/offline_resolve.wgsl` | New: divide-by-N resolve (Rgba32Float→Rgba16Float) + normal renormalization |
| `src/shaders/terrain_pbr_pom.wgsl` | Add `offline_hdr_output` uniform flag; skip tonemap+sRGB when set |
| `src/py_types/frame.rs` | Add `HdrFrame` type with `to_numpy_f32()` |
| `python/forge3d/__init__.py` | Export `HdrFrame`, `OfflineQualitySettings` |
| `python/forge3d/terrain_params.py` | Add `OfflineQualitySettings` dataclass; extend `DenoiseSettings.method` with `'oidn'` |
| `python/forge3d/offline.py` | New: `render_offline()` controller, `OfflineResult`, `OfflineProgress` |
| `python/forge3d/denoise_oidn.py` | New: `oidn_available()`, `oidn_denoise()` |
| `examples/terrain_tv12_offline_quality_demo.py` | New: demo with real DEM, comparison renders |
| `tests/test_tv12_offline_quality.py` | New: all tests from Section 10 |
| `docs/tv12-terrain-offline-render-quality.md` | New: feature documentation |
