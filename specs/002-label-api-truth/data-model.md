# Data Model: Label API Truth

## LabelId

Stable public identifier for a label or label-like object. It must be usable in diagnostics and later update/inspect/remove/export/review workflows where those workflows exist.

## LabelBatchResult

- `ids`: ordered list of accepted label IDs or null entries.
- `diagnostics`: ordered per-input diagnostics.
- `input_count`: number of requested labels.

Validation: output order must match input order.

## LabelConfigurationState

- `enabled`
- `active_atlas`
- `typography`
- `declutter_algorithm`
- `support_status`
- `diagnostics`

State must be inspectable, render-visible, layout-measurable, serializable, or diagnosed as unsupported.

## LineLabelPath

- `points`
- `path_type`: horizontal, vertical, diagonal, curved, terrain-elevated, invalid.
- `repeat_distance`
- `terrain_mode`

## GlyphInstance

- `label_id`
- `glyph`
- `position`
- `rotation`
- `bounds`
- `ordering_key`

## LabelDiagnostic

Structured diagnostic compatible with feature `001`, with affected label/layer/operation fields in `details`.
