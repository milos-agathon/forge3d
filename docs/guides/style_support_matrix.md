# Style Support Matrix

forge3d style support is scoped to local/provided features. It is not streamed MVT
rendering and it is not complete Mapbox style parity.

| Capability | Support level | Scope | Diagnostics |
| --- | --- | --- | --- |
| `fill` layers | `supported` | Local/provided polygon features with `fill-color`, `fill-opacity`, and `fill-outline-color`. | `unsupported_style_field` for unsupported paint/layout fields. |
| `line` layers | `supported` | Local/provided line features with `line-color`, `line-width`, and `line-opacity`. | `unsupported_style_field` for unsupported paint/layout fields. |
| `circle` layers | `supported` | Local/provided point features with `circle-color`, `circle-radius`, and `circle-opacity`. | `unsupported_style_field` for unsupported paint/layout fields. |
| `symbol` text layers | `underdeveloped` | Existing style-to-label helpers can preserve some text styling, but production LabelLayer integration is owned by later features. | `experimental_feature` or label diagnostics when used in production workflows. |
| Other style layer types | `unsupported` | Heatmap, raster, hillshade, fill-extrusion, background, and other layer types are outside the P0 local-feature subset. | `unsupported_style_layer_type`. |
| Streamed vector tiles | `non-goal` | Hosted/live tile delivery is outside this feature. | Documentation boundary, not a render path. |

Unsupported style fields must be reported before render; they must not be
silently dropped in PRD-scoped workflows.
