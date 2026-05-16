# Deterministic LabelPlan

`LabelPlan.compile` builds a deterministic offline label plan from source
labels, camera/output context, optional terrain samples, keepouts, priority
rules, typography, glyph coverage, and a seed. The public model is designed for
reviewable map production: successful plans expose accepted labels, rejected
labels, diagnostics, bounds, and the seed that shaped candidate ordering.

Core public objects:

- `LabelPlan`: compiled plan with `accepted`, `rejected`, `diagnostics`,
  `bounds`, `seed`, `to_dict()`, `from_dict()`, `to_render_payload()`, and
  `to_export_payload()`.
- `AcceptedLabel`: selected label with stable identity, text, geometry type,
  selected `LabelCandidate`, generated candidates, typography, glyphs, and
  screen/world bounds.
- `RejectedLabel`: rejected source label or candidate with a reason code,
  candidate identity where known, diagnostic references, and structured
  details.
- `LabelCandidate`: candidate anchor, type, score, bounds, terrain sample, and
  deterministic ordering key.
- `KeepoutRegion`: title, legend, scale bar, north arrow, or manual rectangle
  bounds that reject intersecting candidates with `keepout_region`.
- `PriorityClass`: rank and tie-break policy used by collision solving.

## Candidate Generation

Point labels generate `center`, `above`, `below`, `left`, `right`, and
`radial` candidates. Radial candidates use the plan seed for deterministic
angle and jitter details, so fixed inputs produce stable candidate order.

Polygon labels generate `centroid` and `visual_center` candidates. If the
centroid is unsuitable, the selected candidate falls back to `visual_center`.
Invalid polygon rings or zero-area polygons are rejected with
`invalid_geometry`.

Terrain-aware point labels can request sampling with `requires_terrain` or
`terrain_mode`. Accepted labels use sampled elevation. Invisible samples or
unavailable required samplers reject with `terrain_occluded`; unavailable
samplers also emit `placeholder_fallback`.

## Rejection Reasons

Every rejected label or candidate uses one of these reason codes:

- `collision`
- `outside_view`
- `missing_glyph`
- `priority_lost`
- `keepout_region`
- `terrain_occluded`
- `invalid_geometry`
- `unsupported_geometry_type`
- `empty_text`

Plans emit `missing_glyphs` diagnostics for glyph coverage gaps and
`label_rejection_summary` diagnostics with deterministic reason counts.

## Payloads

`to_render_payload()` and `to_export_payload()` preserve accepted labels,
rejected labels, diagnostics, bounds, seed, typography, glyphs, and candidate
data. Unsupported render or export backends do not return empty success:
payloads keep the plan data and append a typed `placeholder_fallback`
diagnostic with `supported` set to false.

The current public compiler covers deterministic offline point and polygon
planning. Advanced curved/repeated label rendering, complex script shaping,
and renderer-specific backend execution remain bounded by diagnostics and later
feature work rather than being claimed here.
