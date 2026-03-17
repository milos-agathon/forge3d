# Async Readback System Guide

forge3d contains an internal Rust async readback implementation in
`src/core/async_readback.rs`, gated by the `async_readback` Cargo feature.
There is **no** public Python module named `forge3d.async_readback` today.

## Public Python alternatives

- `forge3d.Scene.render_rgba()` for native scene readback
- `forge3d.render_offscreen_rgba()` for the offscreen helper path
- `ViewerHandle.snapshot()` for interactive-viewer captures

## What this means

- Older examples using `AsyncRenderer`, `AsyncReadbackConfig`, or
  `AsyncReadbackContext` from Python are outdated.
- If you need async readback at the Rust layer, work directly with the feature-
  gated implementation in `src/core/async_readback.rs`.

## Recommended path

For current Python code, treat readback as a synchronous operation unless you
add a dedicated binding for the async subsystem yourself.
