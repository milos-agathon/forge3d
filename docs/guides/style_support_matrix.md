# Style Support Matrix

forge3d style support is scoped to local/provided features. It is not streamed MVT
rendering and it is not complete Mapbox style parity.

| Capability | Support level | Scope | Diagnostics |
| --- | --- | --- | --- |
| `fill` layers | `supported` | Local/provided polygon features with `fill-color`, `fill-opacity`, and `fill-outline-color`. | `unsupported_style_field` for unsupported paint/layout fields. |
| `line` layers | `supported` | Local/provided line features with `line-color`, `line-width`, and `line-opacity`. | `unsupported_style_field` for unsupported paint/layout fields. |
| Local/viewer line ribbons | `supported` | `VectorLineLayer` / `ViewerHandle.add_vector_line`: world-unit width and halo, butt/square caps, bevel/miter joins, native vertex drape and display-height offset. | `VectorLineStyleError.diagnostics` uses `unsupported_style_field`; invalid geometry/numbers raise `ValueError` before IPC. |
| `circle` layers | `supported` | Local/provided point features with `circle-color`, `circle-radius`, and `circle-opacity`. | `unsupported_style_field` for unsupported paint/layout fields. |
| `symbol` text layers | `underdeveloped` | Existing style-to-label helpers can preserve some text styling, and production `LabelPlan` / `LabelLayer` workflows are validated through `MapScene.validate`. | `experimental_feature`, `missing_glyphs`, or `label_rejection_summary` diagnostics when used in production workflows. |
| Other style layer types | `unsupported` | Heatmap, raster, hillshade, fill-extrusion, background, and other layer types are outside the P0 local-feature subset. | `unsupported_style_layer_type`. |
| Streamed vector tiles | `non-goal` | Hosted/live tile delivery is outside this feature. | Documentation boundary, not a render path. |

Unsupported style fields must be reported before render; they must not be
silently dropped in PRD-scoped workflows.

## Local/viewer stroke and drape

```python
from forge3d import VectorLineLayer

line = VectorLineLayer(
    "river", [(25, 0, 25), (70, 0, 25), (100, 0, 95)],
    width=6, halo=2, color=(0, 0.3, 1, 1),
    cap="square", join="bevel", drape=True, z_offset=0.5,
)
overlay_id = viewer.add_vector_line(line)  # Existing, terrain-loaded ViewerHandle.
```

The helper builds triangle geometry and a disjoint halo border for the existing
native overlay submission. Callers provide only the path and styling. Existing
`vector_overlay_configs_from_style`, low-level overlay defaults and style
serialization remain unchanged.

- Points are absolute viewer world `(X, display Y, Z)`, with `Z=-map Y` for
  projected terrain. Convert source coordinates before calling the helper.
- `width` is the full horizontal ribbon width, at least `0.1`, following
  `VectorOverlayConfig`. `halo` is a nonnegative border width on each side.
  Both use world units (metres on metric terrain), not screen pixels.
- Caps are `butt` (stop at the endpoint) and `square` (extend half the width).
  A halo surrounds the endpoint too. Joins are `bevel` (cut the outer corner)
  and `miter` (intersect the offset edges, clamping miter length at four times
  the stroke half-width or the halo's outer radius). Defaults are `butt` caps
  and `miter` joins. The ratio of `4.0` and clamping behavior follow
  `src/vector/line_helpers.rs`; this helper does not switch long miters to bevel.
- Native drape samples the heightfield bilinearly at every generated vertex,
  clamps outside-terrain samples and adds
  `(height - domain_min) * z_scale + z_offset` to the input display Y.
  `z_offset` defaults to `0.5`; despite its name it offsets vertical **Y**.
  Without drape it adds only `z_offset`. Triangles interpolate vertex heights;
  provide more path points when terrain curvature needs finer sampling, while
  leaving room for the width and halo at corners. Densifying near a sharp turn
  can fold the offset edges and is rejected. For an unclamped corner with turn
  angle θ, its offset edge reaches `r * tan(θ / 2)` along each adjacent segment,
  where `r = width / 2 + halo`; segments shorter than this can fold. A curve
  with radius smaller than `r` can fold too. Reduce width/halo or simplify the
  corner when this happens; adding points alone does not resolve the geometry.
- Open paths with finite nonzero horizontal segments are supported. Closed,
  reversing, crossing or folded ribbons are rejected. Nonfinite dimensions,
  coordinates and colors, invalid opacity and non-u32 feature IDs are rejected.

`VectorLineLayer.from_style(layer, points, properties=..., zoom=..., halo=...,
drape=..., z_offset=...)` evaluates the existing local color/number expression
helpers for `line-color`, `line-width`, `line-opacity`, `line-cap` and `line-join`.
This explicitly interprets the translated width in world units. When omitted,
width is `1`, cap is `butt`, and join is `miter` with the renderer's fixed ratio
`4.0` clamp. `line-miter-limit` remains unsupported and is diagnosed; no style
schema or existing translation defaults change. The layer must
be visible and match its feature filter/zoom. Unsupported paint/layout fields,
dashes and round caps/joins raise `VectorLineStyleError` with the existing
structured diagnostics before geometry or rendering. Unresolvable expressions
raise `ValueError`. Halo, drape and offset are helper options, not new serialized
Mapbox fields.

Run [the local road example](../../examples/vector_line_drape.py) with
`python examples/vector_line_drape.py --output artifacts/road.png` after building
the viewer with
`cargo build --release --bin interactive_viewer --features async_readback,enable-gpu-instancing`.
The example creates asymmetric terrain and submits a two-segment road with a halo.
Viewer snapshots use the documented interactive-viewer exclusion from offscreen
render certificates. This helper does not enable SUTURA `MapScene` projection,
streamed MVT or full Mapbox parity.
