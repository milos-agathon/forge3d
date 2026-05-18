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
