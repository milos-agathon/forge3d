# Data Model: Diagnostics and Support Matrices

## Diagnostic

- `code`: stable string from required inventory or feature-local extension.
- `severity`: one of `info`, `warning`, `error`, `fatal`.
- `message`: concise user-facing description.
- `remediation`: actionable next step or support boundary.
- `support_level`: optional PRD Appendix B classification.
- `layer_id`: optional affected layer ID.
- `object_id`: optional affected object, feature, label, tile, or asset ID.
- `details`: JSON-serializable deterministic mapping for extra context.

Validation: unknown severities and support levels are invalid. `details` must be serializable and sorted for output.

## ValidationReport

- `status`: highest blocking status derived from diagnostics and render policy.
- `diagnostics`: ordered list of `Diagnostic`.
- `layer_summaries`: ordered `LayerSummary` entries.
- `estimated_gpu_memory_bytes`: optional integer estimate.
- `supported_features`: deterministic mapping.
- `unsupported_features`: deterministic mapping.
- `render_blocked`: boolean.

Ordering: severity rank, code, layer ID, object ID, message, then stable details hash.

## LayerSummary

- `layer_id`
- `layer_type`
- `support_level`
- `diagnostic_codes`
- `object_count`
- `bounds`
- `memory_estimate_bytes`
- `details`

## SupportMatrixEntry

- `area`
- `capability`
- `support_level`
- `scope`
- `limitations`
- `diagnostic_codes`
- `remediation`
- `evidence`

## RenderFailurePolicy

- `continue_on_warning`: default.
- `fail_on_warning`: warnings block successful render.
- Errors and fatals always block.

## SeverityPolicy

Maps severities to report status and render blocking behavior. Informational diagnostics never block by themselves.
