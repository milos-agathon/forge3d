# Offline Terrain Rendering

This page documents the repo's terrain offline-rendering path. It is the
high-quality accumulation workflow behind `forge3d.offline.render_offline()`
and the lower-level `TerrainRenderer` accumulation API.

## When To Use It

Use the offline path when an interactive viewer snapshot is not enough and you
need progressive refinement, denoise guidance, and a clear separation between
HDR accumulation and final tonemapping.

The typical stack is:

- `Session`
- `MaterialSet`
- `IBL`
- `TerrainRenderParams`
- `TerrainRenderer`
- `OfflineQualitySettings`

## Public Accumulation Primitives

`TerrainRenderer` exposes the batch-oriented primitives directly. The public
offline sequence is:

1. `begin_offline_accumulation(...)`
2. `accumulate_batch(sample_count)`
3. `read_accumulation_metrics(target_variance, tile_size)`
4. `resolve_offline_hdr()`
5. `tonemap_offline_hdr(hdr_frame)`
6. `end_offline_accumulation()`

`render_offline()` is a convenience wrapper around those same calls. Use it if
you want the built-in adaptive loop, progress callback, and optional denoise
integration instead of managing the batches yourself.

## Output Contract

`resolve_offline_hdr()` returns the resolved linear HDR beauty buffer together
with aligned albedo, normal, and depth outputs packaged as the matching AOV
frame. That alignment matters because CPU denoisers and downstream analysis
expect the beauty, guidance, and depth buffers to describe the same resolved
pixel grid.

At the Python layer, `render_offline()` returns:

- `OfflineResult.frame`: the final tonemapped `Frame`
- `OfflineResult.hdr_frame`: the resolved linear HDR beauty buffer
- `OfflineResult.aov_frame`: aligned albedo, normal, and depth outputs
- `OfflineResult.metadata`: samples used, denoiser choice, and convergence data

The offline pipeline is intentionally documented in terms of linear HDR and
separate AOV data. It should not be treated as an sRGB-only screenshot path.

## Adaptive Accumulation

`OfflineQualitySettings` controls the stop conditions and batch behavior:

- `enabled` opts into the offline path explicitly
- `adaptive` switches from a fixed sample count to metric-driven stopping
- `max_samples` and `min_samples` bound the accumulation window
- `batch_size` controls how many samples each `accumulate_batch()` call adds
- `target_variance`, `tile_size`, and `convergence_ratio` drive early stopping

When adaptive mode is active, `render_offline()` polls
`read_accumulation_metrics(...)` and stops once the convergence trend and
variance thresholds say more samples are not buying enough visible improvement.

## Denoise Flow

The denoise path runs after `resolve_offline_hdr()` and before
`tonemap_offline_hdr(...)`.

- Beauty data comes from `OfflineResult.hdr_frame`
- Guidance data comes from the aligned AOV buffers
- OIDN is used when available and requested
- The A-Trous fallback stays available when OIDN is unavailable

If a denoiser updates the HDR beauty image, the wrapper re-uploads that data
before tonemapping so the final `Frame` matches the denoised HDR result.

## Relationship To The Rest Of The Docs

See `guides/rendering_and_analysis.md` for the broader rendering map and
`guides/output_and_integration.md` for packaging, snapshots, notebooks, and
downstream export workflows.
