---
paths: ["src/**/*.rs"]
---

# Rust engine conventions

- Python-bound errors use `RenderError` / `RenderResult<T>` and convert through
  `From<RenderError> for PyErr`; match a leaf module's existing result type.
- `GpuContext`, `ResourceRegistry`, and `GLOBAL_CSM_STATE` are process-wide.
  GPU backend selection is fixed by the first context initialization.
- `Session(backend=...)` sets `WGPU_BACKENDS` before first GPU use, is a no-op
  when it matches the live backend, and raises on a conflicting live backend.
- Device creation preserves native limits and negotiates optional capabilities.
  Do not introduce private `DeviceDescriptor::default()` request paths.
- The 512 MiB host-visible budget defaults to ENFORCE. All wgpu buffers and
  textures route through `src/core/resource_tracker.rs`; the allocation gate
  rejects raw creation elsewhere.
- New native feature gates must be reflected in the maturin wheel feature list
  or their public APIs must raise `DegradedCapability`.
- CENSOR deleted the flat framegraph shim, legacy postfx chain/resource pool,
  standalone bloom effect, render bundles, and parallel render memory budget.
  `framegraph_impl`, `BloomConfig`, Scene CPU bloom, and util IBL estimators are
  the surviving implementations; do not recreate the deleted structures.
