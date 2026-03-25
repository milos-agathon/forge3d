# TV12: Terrain Offline Render Quality

Epic TV12 is shipped for deterministic offline terrain accumulation, adaptive sampling, and optional learned denoising through the public Python API.

## What shipped

- `forge3d.render_offline(...)` drives the offline terrain render loop from Python.
- `TerrainRenderer` exposes the batch primitives behind that controller:
  - `begin_offline_accumulation(...)`
  - `accumulate_batch(...)`
  - `read_accumulation_metrics(...)`
  - `resolve_offline_hdr()`
  - `upload_hdr_frame(...)`
  - `tonemap_offline_hdr(...)`
  - `end_offline_accumulation()`
- `render_offline(..., water_mask=...)` forwards the terrain water mask unchanged so offline renders honor the same water shading inputs as one-shot terrain renders.
- `HdrFrame` exposes linear HDR readback with `to_numpy_f32()` and `.save(".exr")`.
- `OfflineQualitySettings` controls adaptive stopping, batching, and convergence policy; `render_offline()` requires `enabled=True` as an explicit opt-in.
- `begin_offline_accumulation(..., jitter_sequence_samples=...)` optionally overrides the precomputed R2 jitter budget; `render_offline()` passes the full planned sample count automatically.
- `DenoiseSettings.method` accepts `"none"`, `"atrous"`, and `"oidn"`.
- When OIDN is requested but unavailable at runtime, the controller warns and falls back to A-trous.

## Public API

```python
import forge3d as f3d
from forge3d.terrain_params import (
    DenoiseSettings,
    OfflineQualitySettings,
    make_terrain_params_config,
)

config = make_terrain_params_config(
    size_px=(1280, 720),
    terrain_span=4000.0,
    z_scale=1.0,
    cam_radius=5200.0,
    cam_phi_deg=142.0,
    cam_theta_deg=64.0,
    aa_samples=16,
    aa_seed=11,
    denoise=DenoiseSettings(enabled=True, method="oidn", iterations=2),
)

params = f3d.TerrainRenderParams(config)
result = f3d.render_offline(
    renderer,
    material_set,
    ibl,
    params,
    heightmap,
    settings=OfflineQualitySettings(
        enabled=True,
        adaptive=True,
        max_samples=16,
        min_samples=4,
        batch_size=4,
        target_variance=0.002,
        tile_size=8,
        convergence_ratio=0.9,
    ),
)

result.frame.save("terrain_offline.png")
result.hdr_frame.save("terrain_offline_hdr.exr")
result.aov_frame.save_all("out", "terrain_offline")
print(result.metadata)
```

## Output Contract

- `OfflineResult.frame` is the final tonemapped `Frame`.
- `OfflineResult.hdr_frame` is the resolved linear HDR beauty buffer.
- `OfflineResult.aov_frame` exposes aligned albedo, normal, and depth outputs for the same render.
- `OfflineResult.metadata` reports the actual sample count, denoiser used, final convergence ratio, and final `p95` delta.
- `render_offline()` raises `ValueError` if `OfflineQualitySettings.enabled` is left at its default `False`.
- `method="none"` preserves the resolved HDR beauty unchanged.
- OIDN guidance images must match the beauty dimensions; the controller rejects mismatched AOVs explicitly.
- Adaptive early-stop requires both a satisfied threshold and an upward 3-snapshot convergence trend.

## Notes

- The current implementation keeps the user-facing TV12 contract but performs accumulation and convergence analysis on the CPU after HDR sample readback. This favors adapter compatibility over a GPU-only accumulation path.
- Blocking GPU waits in offline metrics, HDR/AOV numpy readbacks, and HDR EXR saves release the Python GIL so callbacks, signals, and other Python threads are not pinned behind `device.poll(...)`.
- `HdrFrame.save()` requires an explicit `.exr` suffix; missing or non-EXR extensions fail fast before any GPU readback or file write.
- Offline accumulation is single-session. Starting a second session or calling one-shot terrain rendering while a session is active raises an error.
- Adaptive stopping uses an OR rule once the controller has seen three convergence snapshots with an upward trend:
  - `converged_tile_ratio >= convergence_ratio`
  - `p95_delta < target_variance`

## Example And Tests

- Example: `python examples/terrain_tv12_offline_quality_demo.py`
- Runtime tests: `tests/test_tv12_offline_quality.py`
- Controller and OIDN tests: `tests/test_tv12_oidn.py`

Use the example when you want one command that writes:

- a single-sample baseline PNG
- an offline accumulated PNG
- resolved albedo, normal, and depth outputs
- an HDR EXR beauty file when the build includes EXR support
