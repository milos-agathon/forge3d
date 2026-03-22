# TV12 — Terrain Offline Render Quality

Forge3D 1.17.0 adds a deterministic offline terrain rendering pipeline with multi-sample accumulation, adaptive convergence, and optional OIDN denoising.

## Overview

The offline quality pipeline produces terrain renders that are measurably better than the single-frame interactive path by accumulating multiple jittered samples in HDR space before tonemapping. The system is designed for production-quality terrain output where render time is not the primary constraint.

**Three capabilities:**

1. **Deterministic multi-sample accumulation (TV12.1)** — R2 quasi-random jitter sequence produces stable, repeatable output for the same seed. Accumulation happens in Rgba32Float precision.

2. **Adaptive sampling (TV12.2)** — The renderer tracks per-tile temporal convergence (how much the result changes between batches) and stops early when the image stabilizes. Prevents over-sampling uniform regions while spending more on high-frequency detail.

3. **Optional OIDN denoising (TV12.3)** — Intel Open Image Denoise can be applied to the accumulated HDR result before tonemapping, using beauty + albedo + normal as inputs. Falls back to the built-in A-trous denoiser when OIDN is not installed.

## Quick Start

```python
import forge3d as f3d
from forge3d.offline import render_offline
from forge3d.terrain_params import OfflineQualitySettings, make_terrain_params_config

session = f3d.Session(window=False)
renderer = f3d.TerrainRenderer(session)
material_set = f3d.MaterialSet.terrain_default()
ibl = f3d.IBL.from_hdr("env.hdr", intensity=1.0)

params = f3d.TerrainRenderParams(make_terrain_params_config(
    size_px=(1920, 1080),
    render_scale=1.0,
    terrain_span=5000.0,
    msaa_samples=1,
    z_scale=1.0,
    exposure=1.0,
    domain=(0.0, 3000.0),
    aa_samples=16,       # Number of accumulation samples
    aa_seed=42,          # Deterministic seed
))

settings = OfflineQualitySettings(enabled=True, batch_size=4)
result = render_offline(renderer, material_set, ibl, params, heightmap, settings=settings)

result.frame.save("terrain_offline.png")       # Tonemapped PNG
result.hdr_frame.save("terrain_offline.exr")   # Linear HDR EXR
```

## API Reference

### `render_offline(renderer, material_set, env_maps, params, heightmap, *, settings, progress_callback=None)`

High-level controller that runs the full offline pipeline.

**Returns:** `OfflineResult` with:
- `frame` — Tonemapped `Frame` (uint8 RGBA, PNG-saveable)
- `hdr_frame` — Linear HDR `HdrFrame` (float32, EXR-saveable)
- `aov_frame` — `AovFrame` with albedo, normal, depth
- `metadata` — dict with `samples_used`, `denoiser_used`, `final_p95_delta`, `converged_ratio`

### `OfflineQualitySettings`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | False | Enable offline quality pipeline |
| `adaptive` | bool | False | Enable adaptive sample allocation |
| `target_variance` | float | 0.001 | Temporal convergence threshold |
| `max_samples` | int | 64 | Upper bound for adaptive mode |
| `min_samples` | int | 4 | Minimum before convergence check |
| `batch_size` | int | 4 | Samples per GPU batch |
| `tile_size` | int | 16 | Variance tile size in pixels |
| `convergence_ratio` | float | 0.95 | Fraction of tiles that must converge |

### `HdrFrame`

Linear HDR texture wrapper (always Rgba16Float).

- `to_numpy_f32() -> np.ndarray` — Returns `(H, W, 4)` float32 array
- `save(path)` — Saves as EXR (errors on non-.exr extension)
- `size` — `(width, height)` tuple

### Low-Level Rust Methods on `TerrainRenderer`

For advanced use cases, the batch primitives are exposed directly:

```python
renderer.begin_offline_accumulation(params, heightmap, material_set, env_maps)
result = renderer.accumulate_batch(4)       # OfflineBatchResult
metrics = renderer.read_accumulation_metrics(0.001)  # OfflineMetrics
hdr, aov = renderer.resolve_offline_hdr()   # (HdrFrame, AovFrame)
hdr = renderer.upload_hdr_frame(data, size) # Re-upload denoised HDR
frame = renderer.tonemap_offline_hdr(hdr)   # Frame
renderer.end_offline_accumulation()         # Explicit cleanup
```

## Quality Tiering

| `DenoiseSettings.method` | Availability | Quality |
|--------------------------|-------------|---------|
| `'none'` | Always | No denoising — raw accumulated result |
| `'atrous'` | Always | A-trous wavelet denoiser (built-in) |
| `'oidn'` | Requires `pip install pyoidn` | Intel OIDN learned denoiser |

When `method='oidn'` is requested but OIDN is not installed, the system falls back to `'atrous'` with a warning. The actually-used denoiser is reported in `OfflineResult.metadata['denoiser_used']`.

## Adaptive Convergence

The adaptive sampling metric measures **temporal convergence**, not spatial detail. After each batch, the system computes per-tile mean luminance and compares to the previous batch. When the result stops changing (temporal delta below threshold), the tile is converged.

Stopping rule (two conditions, OR):
1. `converged_tile_ratio >= convergence_ratio` (default 0.95)
2. `p95_delta < target_variance`

This means sharp cliff edges converge correctly (they stop changing), while noisy regions get more samples until they stabilize.

## Session Semantics

Only one offline session can be active per renderer. `begin_offline_accumulation()` errors if a session is already active. Normal `render_terrain_pbr_pom()` calls during an active session are blocked.

`tonemap_offline_hdr()` automatically ends the session. `end_offline_accumulation()` provides explicit cleanup for error paths or HDR-only export without tonemapping.
