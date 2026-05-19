# Diagnostics Reference

forge3d map-production diagnostics are structured records for validation and
bundle-ready review data. They are not log messages.

Each diagnostic contains:

| Field | Meaning |
| --- | --- |
| `code` | Stable diagnostic identifier. |
| `severity` | One of `info`, `warning`, `error`, or `fatal`. |
| `message` | Short user-facing description. |
| `remediation` | Action or support boundary. |
| `support_level` | PRD classification where applicable. |
| `layer_id` | Affected layer when known. |
| `object_id` | Affected object, label, tile, or asset when known. |
| `details` | Deterministic JSON-ready context. |

## Required Codes

| Code | Minimum output-affecting severity | Support level | Trigger |
| --- | --- | --- | --- |
| `missing_crs` | `error` | `unsupported` | Terrain, layer, or asset CRS is required for a render path and was not provided. |
| `missing_source_identity` | `error` | `unsupported` | A layer or asset lacks a stable source identifier needed for bundle review. |
| `missing_renderable_data` | `error` | `placeholder/fallback` | A layer has no geometry, raster pixels, labels, tiles, or other renderable payload. |
| `missing_external_asset` | `error` | `missing` | A referenced file or bundled external asset cannot be resolved. |
| `unsupported_asset_format` | `error` | `unsupported` | An asset extension or declared format is outside the public MapScene subset. |
| `unsupported_output_format` | `error` | `unsupported` | Requested output is outside the supported offline PNG path. |
| `unsupported_layer_type` | `error` | `unsupported` | A MapScene layer type is not part of the typed public scene contract. |
| `unsupported_feature` | `error` | `unsupported` | A requested feature is known but not supported by the public workflow. |
| `crs_mismatch` | `error` | `unsupported` | Layer CRS differs from scene or terrain CRS and no transform was provided. |
| `missing_glyphs` | `warning` | `underdeveloped` | Labels reference glyphs missing from the active atlas. |
| `unsupported_style_field` | `warning` | `unsupported` | A supported style layer uses paint or layout fields forge3d does not consume. |
| `unsupported_style_layer_type` | `error` | `unsupported` | A style layer type is outside the supported local feature subset. |
| `pro_gated_path` | `error` | `Pro-gated` | The requested workflow requires a native or Pro path. |
| `placeholder_fallback` | `error` | `placeholder/fallback` | A path would produce zero geometry or non-renderable placeholder output. |
| `experimental_feature` | `warning` | `experimental` | A feature exists but is not production-stable. |
| `vt_unsupported_family` | `error` | `unsupported` | A non-albedo VT family is requested while runtime pages only albedo. |
| `python_public_3dtiles_incomplete` | `error` | `underdeveloped` | Public Python 3D Tiles workflow cannot complete render preparation. |
| `estimated_gpu_memory` | `warning` | `supported` | Estimated GPU memory exceeds the configured budget. |
| `label_rejection_summary` | `warning` | `underdeveloped` | Label placement rejected candidates and exposes reason counts. |
| `missing_crs` | `error` | `unsupported` | A layer or scene component requires CRS-aware validation but no CRS was supplied. |
| `missing_source_identity` | `error` | `unsupported` | A recipe layer has no stable source path, inline data, or source identity for validation and bundle review. |
| `missing_renderable_data` | `error` | `unsupported` | A layer declares a render path but provides no renderable source data. |
| `missing_external_asset` | `error` | `unsupported` | A scene or bundle references an external asset that cannot be found. |
| `unsupported_asset_format` | `error` | `unsupported` | A source asset uses a file format outside the documented public MapScene workflow. |
| `unsupported_output_format` | `error` | `unsupported` | A requested output format is outside the current PNG-oriented public render path. |
| `unsupported_layer_type` | `error` | `unsupported` | A recipe contains a layer type that MapScene validation cannot route. |
| `unsupported_feature` | `error` | `unsupported` | A requested feature exists as intent but has no supported public render implementation. |

## Render Policy

`continue_on_warning` permits warning-only reports to proceed while preserving
diagnostics. `fail_on_warning` blocks warning reports. `error` and `fatal`
diagnostics always block successful render completion.
