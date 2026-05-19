# Label Support Matrix

| Capability | Support level | Scope | Diagnostics |
| --- | --- | --- | --- |
| Point labels | `supported` | `ViewerHandle.add_label` creates stable ids for point labels through the public workflow, not raw IPC. | `missing_glyphs` diagnostic for atlas gaps. |
| Line labels | `supported` | `ViewerHandle.add_line_label` validates line geometry and emits deterministic glyph ordering keys. | `placeholder_fallback` diagnostic for invalid geometry. |
| Curved labels | `experimental` | `ViewerHandle.add_curved_label` is public but not production-stable. | `experimental_feature` diagnostic. |
| Callouts | `supported` | `ViewerHandle.add_callout` creates stable ids for point-attached callout labels. | `missing_glyphs` diagnostic for atlas gaps. |
| Typography controls | `supported` | `ViewerHandle.set_label_typography` updates layout metrics for tracking, line height, and alignment. | Typography layout metrics are returned for review. |
| Decluttering controls | `supported` | `ViewerHandle.set_declutter_algorithm` records deterministic placement policy settings. | Placement policy state is returned for review. |
| Atlas loading | `supported` | `ViewerHandle.load_label_atlas` records atlas and metrics paths for subsequent label validation. | `missing_glyphs` diagnostic when coverage is insufficient. |
| Missing glyph diagnostics | `supported` | Known glyph coverage gaps are reported through typed diagnostics. | `missing_glyphs` diagnostic. |
| Upside-down line handling | `supported` | Line glyph rotation and ordering are normalized for deterministic line-label placement. | Diagnostic-bearing invalid geometry remains `placeholder_fallback`. |
| Terrain-elevated line labels | `experimental` | Terrain-elevated line-label behavior is exposed only as an experimental public path. | `experimental_feature` diagnostic. |
| Deterministic LabelPlan | `supported` | Offline point and polygon candidates compile through `LabelPlan.compile` with reviewable accepted/rejected output. | `label_rejection_summary`; unsupported backends return `placeholder_fallback`. |

No label command should report success while doing nothing. Unsupported or
unverified label behavior must produce typed diagnostics or explicit failure.
