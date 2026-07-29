# Virtual Texturing Support Matrix

| Capability | Support level | Scope | Diagnostics |
| --- | --- | --- | --- |
| Albedo terrain VT family | `supported` | Runtime pages the albedo family; this is the current albedo-only VT runtime path. | `estimated_gpu_memory` where budget risk is knowable. |
| Normal terrain VT family | `unsupported` | Python accepts `normal` for forward compatibility, but native runtime pages only `albedo`; `MapScene.validate` reports `vt.normal` before render. | `vt_unsupported_family`. |
| Mask terrain VT family | `unsupported` | Python accepts `mask` for forward compatibility, but native runtime pages only `albedo`; `MapScene.validate` reports `vt.mask` before render. | `vt_unsupported_family`. |
| Runtime residency stats | `underdeveloped` | Lower-level stats exist; product validation integration is reported through large-scene diagnostics where metadata is available. | `unavailable_cache_lod_stats` diagnostic when unavailable. |

Non-albedo family requests must not silently skip. They are diagnosed before render
through `vt_unsupported_family`; this non-MVP-blocking deferral is an explicit
unsupported runtime status, not a runtime implementation for those families.

## Memory budgets

The terrain virtual-texture runtime is governed by **two independent budgets**.
They are enforced separately and are not interchangeable.

| Budget | Value | Enforcement |
| --- | --- | --- |
| Host-visible (staging rings, upload buffers, readbacks) | **512 MiB** | CENSOR's resource tracker, enforce-by-default (`BUDGET_POLICY_ENFORCE`); an over-budget host-visible allocation fails with `RenderError::Budget` |
| Device-local VT atlas (the resident tile atlas itself) | **256 MiB by default**, set by `TerrainVTSettings.residency_budget_mb` | Per-family LRU eviction in `FamilyResidencyTracker`, with the shared tile-cache capacity as the global backstop |
| Per-frame VT upload (bytes copied into the atlas per frame) | Caller-supplied, `vt_upload_budget_bytes` | Highest-priority-first: a request whose upload would push the frame over the budget is skipped and stays in the retained request set for a later frame |

The device-local atlas budget is deliberately **not** counted against the
512 MiB host-visible budget: atlas tiles are device-local, so they never
occupy the host-visible heap that CENSOR's tracker governs. Compressed
(BC7/BC5/BC4) tiles occupy roughly a quarter of the equivalent RGBA8
footprint, so the same 256 MiB holds about 4x the resident texels.

`residency_budget_mb` is split **evenly across the enabled families**
(albedo, normal, mask, height). With all four enabled at the 256 MiB default
each family gets 64 MiB, and one family's paging pressure evicts only its own
least-recently-used tiles while it stays inside that share.

Observed values are reported by `forge3d.diagnostics.vt_stats()`:
`cache_budget_mb`, `atlas_device_local_bytes`,
`atlas_uncompressed_equivalent_bytes`, `atlas_compression_ratio`,
`evictions`, `tiles_streamed`, and `retained_requests`.
