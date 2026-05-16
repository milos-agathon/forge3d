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

## Render Policy

`continue_on_warning` permits warning-only reports to proceed while preserving
diagnostics. `fail_on_warning` blocks warning reports. `error` and `fatal`
diagnostics always block successful render completion.
