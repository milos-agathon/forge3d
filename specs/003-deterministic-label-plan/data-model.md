# Data Model: Deterministic LabelPlan

## LabelPlan

- `accepted`: ordered `AcceptedLabel` list.
- `rejected`: ordered `RejectedLabel` list.
- `diagnostics`: ordered diagnostics compatible with feature `001`.
- `bounds`: plan and per-label bounds where applicable.
- `seed`: deterministic seed used for candidate generation and tie-breaks.
- `payload_version`: serialization version.

## AcceptedLabel

- `label_id`
- `source_id`
- `text`
- `geometry_type`
- `candidate`
- `priority_class`
- `screen_bounds`
- `world_bounds`
- `typography`
- `glyphs`
- `ordering_key`

## RejectedLabel

- `label_id`
- `source_id`
- `candidate_id`
- `reason`: one of `collision`, `outside_view`, `missing_glyph`, `priority_lost`, `keepout_region`, `terrain_occluded`, `invalid_geometry`, `unsupported_geometry_type`, `empty_text`.
- `diagnostic_refs`
- `ordering_key`

## LabelCandidate

- `candidate_id`
- `candidate_type`: center, above, below, left, right, radial, centroid, visual_center.
- `anchor`
- `score`
- `bounds`
- `terrain_sample`
- `ordering_key`

## KeepoutRegion

- `region_id`
- `kind`: title, legend, scale_bar, north_arrow, manual_rectangle.
- `bounds`
- `priority`

## PriorityClass

- `name`
- `rank`
- `tie_break_policy`

## TerrainSample

- `position`
- `elevation`
- `visible`
- `source`
- `diagnostics`
