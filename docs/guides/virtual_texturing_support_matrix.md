# Virtual Texturing Support Matrix

| Capability | Support level | Scope | Diagnostics |
| --- | --- | --- | --- |
| Albedo terrain VT family | `supported` | Runtime pages the albedo-only family and `MapScene.validate` records budget risk. | `estimated_gpu_memory` where budget risk is knowable. |
| Normal terrain VT family | `missing` | Python accepts the family for forward compatibility, but native runtime does not page it. | `vt_unsupported_family`. |
| Mask terrain VT family | `missing` | Python accepts the family for forward compatibility, but native runtime does not page it. | `vt_unsupported_family`. |
| Runtime residency stats | `underdeveloped` | Lower-level stats exist; product validation integration remains diagnostic-bearing. | `estimated_gpu_memory`. |

Non-albedo family requests must be surfaced before render instead of being
only log-visible.
