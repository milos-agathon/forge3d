# Quickstart: Deterministic LabelPlan

These scenarios are validation targets for the feature `003` implementation.
They use the public `LabelPlan.compile`, `KeepoutRegion`, and `PriorityClass`
API rather than raw viewer IPC.

## Reproducible Compile

1. Build a fixed label source with point and polygon labels.
2. Use fixed camera, viewport, typography, glyph atlas, keepouts, priority
   classes, terrain sampler, and seed.
3. Compile twice with `LabelPlan.compile(...)`.
4. Compare `accepted`, `rejected`, `diagnostics`, `bounds`, `seed`,
   `to_render_payload()`, `to_export_payload()`, and serialized payloads
   exactly.

```python
from forge3d import KeepoutRegion, LabelPlan, PriorityClass

labels = [
    {
        "id": "capital",
        "text": "Capital",
        "geometry": {"type": "Point", "coordinates": (40.0, 40.0, 0.0)},
        "priority_class": "capital",
    },
    {
        "id": "local",
        "text": "Local",
        "geometry": {"type": "Point", "coordinates": (40.0, 40.0, 0.0)},
        "priority_class": "local",
    },
    {
        "id": "legend-hit",
        "text": "Legend",
        "geometry": {"type": "Point", "coordinates": (10.0, 10.0, 0.0)},
    },
    {
        "id": "glyph-gap",
        "text": "Glyph!",
        "geometry": {"type": "Point", "coordinates": (70.0, 10.0, 0.0)},
    },
]

plan = LabelPlan.compile(
    labels=labels,
    camera={"name": "fixed"},
    viewport=(100, 100),
    keepouts=[
        KeepoutRegion(
            region_id="legend",
            kind="legend",
            bounds=(0.0, 0.0, 20.0, 20.0),
        )
    ],
    priority_rules=[
        PriorityClass(name="local", rank=10),
        PriorityClass(name="capital", rank=20),
    ],
    glyph_atlas={"glyphs": set("CapitalLocalLegendGlyph")},
    seed=17,
)
```

## Rejection Reason Coverage

Create fixtures for every required reason and assert each rejected item keeps a
stable label ID, reason, and candidate ID where available:

- `collision`
- `outside_view`
- `missing_glyph`
- `priority_lost`
- `keepout_region`
- `terrain_occluded`
- `invalid_geometry`
- `unsupported_geometry_type`
- `empty_text`

Plans with glyph gaps include `missing_glyphs`. Plans with rejected labels
include `label_rejection_summary` with deterministic reason counts.

## Keepout And Priority

Use title, legend, scale bar, north arrow, and manual rectangle keepouts. Place
labels that overlap each keepout and confirm rejected candidates use
`keepout_region`.

Use colliding labels with different `PriorityClass` ranks and confirm the
higher rank wins deterministically. Equal-priority ties use stable label
ordering and reject the loser with `collision`.

## Render/Export Payload

Use `to_render_payload()` and `to_export_payload()` after compilation. Payloads
must include accepted labels, rejected labels, bounds, typography, glyph
references, diagnostics, candidates, and seed.

Unsupported backend requests must return typed diagnostics instead of
placeholder success. For example, `plan.to_render_payload(backend="native-gpu")`
keeps the plan data and appends a `placeholder_fallback` diagnostic with
`supported` set to false.
