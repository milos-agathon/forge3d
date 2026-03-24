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
begin_offline_accumulation(params, heightmap, material_set, env_maps, water_mask=None)
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

Seven methods on `TerrainRenderer`:

### 3.1 `begin_offline_accumulation(params, heightmap, material_set, env_maps, water_mask=None, jitter_sequence_samples=None)`

- Validates no offline session is already active; errors if so.
- Accepts the same optional `water_mask` input as the one-shot terrain render path and forwards it unchanged into terrain shading.
- Allocates or resizes:
  - **Beauty accumulation:** Two ping-pong Rgba32Float textures (A and B) for additive accumulation without ReadWrite storage.
  - **Albedo accumulation:** One Rgba32Float texture (additive, divided at resolve).
  - **Normal accumulation:** One Rgba32Float texture (additive, renormalized at resolve).
  - **Depth reference:** One R32Float texture (written once on sample 0, not accumulated).
  - **Per-sample luminance scratch:** One Rgba16Float texture (same size as beauty scratch) used by the variance compute shader to write per-pixel luminance. Read back as part of tile variance computation on CPU.
- Creates `JitterSequence` from `aa_seed` in params (R2 sequence). `render_offline()` passes `jitter_sequence_samples=settings.max_samples` when `adaptive=True`, or `params.aa_samples` when `adaptive=False`, so the pre-generated low-discrepancy sequence covers the full planned budget without wrapping.
- **Caches owned GPU resources**, not borrowed Python references. Specifically: copies the decoded `DecodedTerrainSettings` struct, uploads and owns the heightmap texture + view, creates and owns IBL bind group, creates and owns material bind groups. The Python-side `PyReadonlyArray`, `MaterialSet`, and `IBL` objects are consumed during `begin_offline_accumulation` and not referenced again — all subsequent `accumulate_batch` calls use the cached GPU resources. Same `TerrainRenderParams` type used by `render_terrain_pbr_pom()`.
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
- **Measures temporal convergence, not spatial detail.** The metric tracks how much the per-tile mean luminance changes relative to a short history window of prior calls, not how much variance exists within a tile. A sharp cliff edge that is fully converged will read as "converged" even though it has high spatial contrast.
- Implementation:
  1. Resolve current accumulation to a temporary quarter-resolution luminance buffer: `lum = 0.2126*R + 0.7152*G + 0.0722*B` over `(accum / total_samples)`. This uses a trivial `offline_luminance.wgsl` compute shader writing to a quarter-res R32Float texture.
  2. Read back the quarter-res luminance via staging buffer. Cost: `(W/4) * (H/4) * 4` bytes (e.g., ~130KB for 1920x1080).
  3. On CPU: divide into tiles, compute per-tile mean luminance.
  4. Compare to the **average of the compatible prior calls' per-tile means** (stored in `OfflineAccumulationState.prev_tile_mean_history`). Per-tile temporal delta: `|current_mean - history_mean| / max(max(current_mean, history_mean), 1e-4)`.
  5. A tile is "converged" when its temporal delta < `target_variance`.
  6. Append the current means into the bounded history window for the next call.
  7. The Python bindings release the GIL while waiting on the staging-buffer readback so `device.poll(...)` does not pin Python callbacks or signals.
- Returns `OfflineMetrics`:
  - `total_samples: u32`
  - `mean_delta: f32` — average per-tile temporal delta
  - `p95_delta: f32` — 95th percentile temporal delta
  - `max_tile_delta: f32` — worst tile (diagnostic only, not used in stopping rule)
  - `converged_tile_ratio: f32` — fraction of tiles with temporal delta < `target_variance`

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

- Applies the **same terrain filmic tonemap curve** used by the live `terrain_pbr_pom.wgsl` path (`tonemap_filmic_terrain()`), not the generic `postprocess_tonemap.wgsl` operators (ACES/Reinhard/etc). This ensures offline output has the same look as the live path. Implemented as a new `tonemap_terrain_offline.wgsl` fullscreen compute shader that reads Rgba16Float HDR input, applies `tonemap_filmic_terrain()` + `linear_to_srgb()`, and writes to Rgba8Unorm output. The terrain filmic curve parameters are extracted from the existing shader to avoid duplication.
- If the user has configured a `TonemapSettings` operator override (e.g., ACES), the offline path respects it by dispatching the appropriate curve. But the **default** is the terrain filmic curve, matching the live path.
- Writes to Rgba8Unorm texture → `Frame`.
- Calls `end_offline_accumulation()` internally to clean up session state.
- The returned `Frame` is a normal Frame: `save()` writes PNG, `to_numpy()` returns uint8 RGBA.

### 3.7 `end_offline_accumulation()`

- Releases all offline session resources: accumulation buffers, cached render inputs, and convergence history.
- Sets `offline_session_active = false`.
- **Idempotent**: calling it when no session is active is a no-op.
- This method exists so Python can clean up if it errors during denoise/upload or intentionally stops at HDR/EXR export without tonemapping. `tonemap_offline_hdr()` calls this internally, so the normal path does not require an explicit call.
- The `HdrFrame` and `AovFrame` returned by `resolve_offline_hdr()` survive teardown because they own their textures independently.

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
- `save(path: str)` — writes `.exr` only. Errors on missing or non-EXR extensions. Releases the GIL during readback and EXR encode/write.
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

### 5.2 `offline_luminance.wgsl` — quarter-res luminance extraction

Compute shader (8x8 workgroups). Reads the current accumulation buffer, divides by sample count, computes luminance, writes to a quarter-resolution R32Float texture.

```
Input:   accumulated (texture_2d<f32>)   — Rgba32Float accumulation buffer
Output:  luminance (texture_storage_2d<r32float, write>) — quarter-res
Uniform: { width: u32, height: u32, sample_count: u32, _pad: u32 }

Per output pixel (at quarter res):
  avg over 4x4 source pixels of: (accum / sample_count)
  lum = 0.2126*R + 0.7152*G + 0.0722*B
  write lum
```

### 5.3 Convergence metric computation (CPU-side)

`read_accumulation_metrics()` measures **temporal convergence** — how much per-tile mean luminance changes relative to a short history window of prior calls:

1. Dispatch `offline_luminance.wgsl` to produce quarter-res luminance map.
2. Read back via staging buffer. Cost: `(W/4) * (H/4) * 4` bytes (~130KB for 1920x1080).
3. On CPU: divide into tiles, compute per-tile mean luminance.
4. Compare to the average of `prev_tile_mean_history` (stored in `OfflineAccumulationState`). Per-tile temporal delta: `|current - history_mean| / max(max(current, history_mean), 1e-4)`.
5. A tile is "converged" when its temporal delta < `target_variance`.
6. Append current means to the bounded history window for the next call.
7. Release the Python GIL while waiting on the readback so progress callbacks and interrupts remain responsive.

This measures whether adding more samples changes the result, not whether the image has spatial detail. A sharp cliff edge that is fully converged reads as "converged" (low temporal delta). A noisy flat patch reads as "not converged" (high temporal delta) until it stabilizes.

### 5.3 Terrain shader HDR output mode

Pack `offline_hdr_output` into the existing `OverlayUniforms.params5.w` slot (currently hardcoded to `0.0` at `upload.rs:89`). This avoids changing the `TerrainUniforms` struct layout. The Rust side sets `params5[3] = 1.0` when rendering in offline HDR mode. The WGSL side reads `overlay_uniforms.params5.w`. When set to `> 0.5`, the fragment shader's final output path changes from:

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

The render target format for offline passes is Rgba16Float instead of Rgba8Unorm.

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
    enabled: bool = False          # Explicit opt-in; render_offline() raises if False
    method: str = "atrous"  # 'atrous', 'oidn', 'none'
    # ... existing fields unchanged ...
```

The existing `__post_init__` validator (`valid_methods = ("atrous", "none")`) must be updated to include `"oidn"` in the valid set.

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
    mean_delta: float       # Average per-tile temporal delta
    p95_delta: float        # 95th percentile temporal delta
    converged_ratio: float  # Fraction of tiles below target_variance
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
    water_mask: np.ndarray | None = None,
) -> OfflineResult:
```

### 8.2 Controller Loop

```python
def render_offline(...):
    if not settings.enabled:
        raise ValueError("render_offline requires OfflineQualitySettings(enabled=True)")

    total = settings.max_samples if settings.adaptive else params.aa_samples
    renderer.begin_offline_accumulation(
        params,
        heightmap,
        material_set,
        env_maps,
        water_mask=water_mask,
        jitter_sequence_samples=total,
    )
    try:
        rendered = 0
        metric_history = []

        while rendered < total:
            batch = min(settings.batch_size, total - rendered)
            result = renderer.accumulate_batch(batch)
            rendered = result.total_samples

            # Single metrics readback per iteration (avoid redundant GPU staging copies)
            metrics = None
            if progress_callback or (settings.adaptive and rendered >= settings.min_samples):
                metrics = renderer.read_accumulation_metrics(settings.target_variance)
                metric_history.append(metrics)

            if progress_callback and metrics is not None:
                progress_callback(OfflineProgress(
                    samples_so_far=rendered, max_samples=total,
                    mean_delta=metrics.mean_delta,
                    p95_delta=metrics.p95_delta,
                    converged_ratio=metrics.converged_tile_ratio,
                    elapsed_ms=result.batch_time_ms,
                ))

            if settings.adaptive and rendered >= settings.min_samples and metrics is not None:
                has_upward_trend = len(metric_history) >= 3 and (
                    metric_history[-1].converged_tile_ratio >= metric_history[-3].converged_tile_ratio
                )
                if has_upward_trend and (
                    metrics.converged_tile_ratio >= settings.convergence_ratio
                        or metrics.p95_delta < settings.target_variance):
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
        # tonemap_offline_hdr calls end_offline_accumulation() internally

    except Exception:
        renderer.end_offline_accumulation()  # Guaranteed cleanup on error
        raise

    return OfflineResult(
        frame=frame,
        hdr_frame=hdr_frame,  # HdrFrame owns its texture; survives session teardown
        aov_frame=aov_frame,
        metadata={
            "samples_used": rendered,
            "denoiser_used": denoiser_used,
            "final_p95_delta": metrics.p95_delta if metrics else None,
            "converged_ratio": metrics.converged_tile_ratio if metrics else None,
        },
    )
```

---

## 9. Adaptive Stopping Rule

The metric is **temporal convergence**: how much the per-tile mean luminance changes relative to a short history window of `read_accumulation_metrics()` calls. This measures whether adding more samples changes the result, not whether the image has spatial detail.

The default stopping policy uses **two conditions (OR)**, but only after the controller has observed an upward 3-snapshot convergence trend:

1. `converged_tile_ratio >= convergence_ratio` (default 0.95) — at least 95% of tiles have temporal delta below `target_variance`.
2. `p95_delta < target_variance` — the 95th percentile temporal delta is below threshold.

Either condition being true stops accumulation. This prevents one hot tile from pinning the entire render to `max_samples`.

`max_tile_delta` is reported in `OfflineMetrics` as a diagnostic but does not participate in the stopping decision.

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
| **Progress callback** | `progress_callback` receives `OfflineProgress` updates whose final values match the returned metadata. |
| **Cleanup on error** | Exceptions during batching, metrics, resolve, or tonemap release the active offline session. |

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
| `src/terrain/accumulation.rs` | Extend with `OfflineAccumulationState` (ping-pong buffers, cached GPU resources, convergence history) |
| `src/terrain/renderer/offline.rs` | New: all 7 offline `#[pymethods]` on `TerrainRenderer` |
| `src/terrain/renderer/core.rs` | Add offline state fields to `TerrainScene`, HDR pipeline cache entry, session guards |
| `src/terrain/renderer/mod.rs` | Register `offline` submodule |
| `src/terrain/renderer/draw/setup/pipeline.rs` | Support `Rgba16Float` render target for offline HDR output |
| `src/terrain/renderer/pipeline_cache.rs` | Add HDR pipeline variant keyed by `(color_format, sample_count)` |
| `src/terrain/renderer/upload.rs` | Pack `offline_hdr_output` flag into overlay uniforms `params5[3]` |
| `src/terrain/render_params/decode_postfx.rs` | Add `DenoiseMethodNative::Oidn` variant; map `"oidn"` string |
| `src/shaders/offline_accumulate.wgsl` | New: additive ping-pong accumulation compute shader |
| `src/shaders/offline_resolve.wgsl` | New: divide-by-N resolve (Rgba32Float→Rgba16Float) + normal renormalization |
| `src/shaders/offline_luminance.wgsl` | New: quarter-res luminance extraction for convergence metrics |
| `src/shaders/tonemap_terrain_offline.wgsl` | New: terrain filmic tonemap as fullscreen compute (matches live path) |
| `src/shaders/terrain_pbr_pom.wgsl` | Read `offline_hdr_output` flag; skip tonemap+sRGB when set |
| `src/py_types/hdr_frame.rs` | New: `HdrFrame` type with `to_numpy_f32()` and EXR save |
| `src/py_types/mod.rs` | Register `hdr_frame` module |
| `src/py_module/classes.rs` | Add `m.add_class::<HdrFrame>()?;` registration |
| `python/forge3d/__init__.py` | Export `HdrFrame`, `OfflineQualitySettings` |
| `python/forge3d/__init__.pyi` | Add `HdrFrame` type stub |
| `python/forge3d/terrain_params.py` | Add `OfflineQualitySettings` dataclass; extend `DenoiseSettings.method` with `'oidn'` |
| `python/forge3d/offline.py` | New: `render_offline()` controller, `OfflineResult`, `OfflineProgress` |
| `python/forge3d/denoise_oidn.py` | New: `oidn_available()`, `oidn_denoise()` |
| `examples/terrain_tv12_offline_quality_demo.py` | New: demo with real DEM, comparison renders |
| `tests/test_tv12_offline_quality.py` | New: all tests from Section 10 |
| `docs/tv12-terrain-offline-render-quality.md` | New: feature documentation |
