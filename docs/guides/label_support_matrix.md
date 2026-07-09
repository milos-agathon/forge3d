# Label Support Matrix

| Capability | Support level | Scope | Diagnostics |
| --- | --- | --- | --- |
| Point labels | `supported` | `ViewerHandle.add_label(...)` returns a stable id for accepted ASCII point-label text and records public label state. | `missing_glyphs` for known atlas coverage gaps; `placeholder_fallback` for empty text. |
| Batch point labels | `supported` | `ViewerHandle.add_labels(...)` preserves input order, returns stable ids for accepted labels, and returns per-label diagnostics for rejected entries. | `placeholder_fallback`, `missing_glyphs`. |
| Line labels | `supported` | `ViewerHandle.add_line_label(...)` returns a stable id for valid flat paths and records deterministic glyph instances with tangent rotation. | `placeholder_fallback` for invalid paths; `experimental_feature` for unavailable terrain sampling. |
| Curved labels | `experimental` | `ViewerHandle.add_curved_label(...)` is public but returns a typed diagnostic until curved glyph rendering is production-stable. | `experimental_feature`. |
| Callouts | `supported` | `ViewerHandle.add_callout(...)` returns a stable id for accepted callout text and anchor data. | `missing_glyphs`, `placeholder_fallback`. |
| Typography controls | `supported` | `ViewerHandle.set_label_typography(...)` updates native label-manager typography state, records serializable settings, and exposes deterministic layout metrics for the configured tracking, kerning, line-height, and word-spacing controls. | Invalid future controls must fail with typed diagnostics instead of no-op success. |
| Decluttering controls | `supported` | `ViewerHandle.set_declutter_algorithm(...)` updates native label-manager declutter state and records a deterministic placement policy for `greedy` or `annealing` with seed and max-iteration settings. | Unsupported algorithm names return `placeholder_fallback` rather than no-op success. |
| Atlas loading | `supported` | `ViewerHandle.load_label_atlas(...)` records active atlas PNG and metrics paths in serializable viewer label state. | Future atlas validation may add `missing_glyphs` or atlas-format diagnostics. |
| Missing glyph diagnostics | `supported` | Known non-ASCII glyph gaps are reported before sending create commands to the viewer. | `missing_glyphs`. |
| Upside-down line handling | `supported` | Reverse line paths are normalized so glyph rotations stay upright, with `upside_down_adjusted` recorded per glyph. | Documentation boundary; no diagnostic for supported flat line handling. |
| Terrain-elevated line labels | `experimental` | Terrain sampling is not owned by feature `002`; requests such as `terrain_mode="sample"` return diagnostics instead of silently flattening. | `experimental_feature`. |
| Deterministic LabelPlan | `supported` | `LabelPlan.compile(...)` produces deterministic accepted/rejected plan data, point and polygon candidates, terrain-aware point elevations where sampled, keepout and priority rejection details, and render/export payloads. | `missing_glyphs`, `label_rejection_summary`; unsupported backends return `placeholder_fallback`. |
| Advanced repeated line labels | `underdeveloped` | P2 product planning adds deterministic repeat-distance output where line geometry can be compiled; unavailable render integration remains diagnosed before render. | `experimental_feature` or `label_rejection_summary` where applicable. |
| Arabic joining | `supported` | `LabelPlan.compile(...)` uses native `rustybuzz` plus `unicode-bidi` when the glyph atlas supplies a font path, records glyph IDs/clusters/advances and RTL metadata, and maps shaped Arabic runs to the current presentation-form atlas keys for MapScene rendering. | `missing_glyphs` if the atlas does not include the shaped render glyphs. |
| Advanced curved labels and Indic shaping | `experimental` | Curved path text, Indic conjunct shaping, and complex-script shaping beyond the current Arabic joining path remain non-MVP-blocking unless prioritized with end-to-end tests. They must be diagnosed before render rather than treated as silent success. | `experimental_feature`. |

The feature `002` public workflow uses high-level `ViewerHandle` methods and
does not require raw IPC for basic label creation, configuration, line labels,
callouts, vector-overlay creation, or clear/remove workflows. Existing raw IPC
helpers remain an advanced compatibility surface, but they are not the MVP
label API truth contract.

Successful create calls return a stable id where users need update, inspect,
remove, export, or review workflows. No label command should report success
while doing nothing. Unsupported or unverified label behavior must produce
typed diagnostics or explicit failure.
