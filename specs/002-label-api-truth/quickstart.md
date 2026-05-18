# Quickstart: Label API Truth

These scenarios are validation targets for implementation. The runnable smoke
example is:

```powershell
python examples/label_api_truth_basic.py --json
```

## Basic High-Level Workflow

1. Create a viewer handle through the existing public viewer startup path.
2. Call `ViewerHandle.load_label_atlas(...)`.
3. Call `ViewerHandle.add_label(...)` and assert a stable ID is returned.
4. Call `ViewerHandle.add_labels([...])` and assert ordered IDs/diagnostics.
5. Call `ViewerHandle.add_line_label(...)` and inspect deterministic glyph
   ordering and tangent rotations for supported flat paths.
6. Call `ViewerHandle.add_callout(...)` and
   `ViewerHandle.add_vector_overlay(...)` and assert stable IDs.
7. Toggle `ViewerHandle.set_labels_enabled(False)` and `True`.
8. Call `ViewerHandle.clear_labels()` and confirm state changes.
9. Confirm no raw `viewer_ipc` call is required in the scenario.

## No-Op Setter Check

1. Prepare two labels whose layout changes under typography or declutter settings.
2. Call `ViewerHandle.set_label_typography(...)` and compare layout/render or
   serialized state. In feature `002`, the call updates native label-manager
   typography state and exposes deterministic layout metrics for the configured
   tracking, kerning, line-height, and word-spacing controls.
3. Call `ViewerHandle.set_declutter_algorithm(...)` and compare placement
   behavior. In feature `002`, the call updates native label-manager declutter
   state and records a deterministic placement policy for supported algorithms.

## Line And Curved Labels

1. Run horizontal, vertical, diagonal, and curved path fixtures.
2. For supported line paths, inspect glyph instances and tangent rotations.
3. For reverse line paths, confirm upside-down avoidance through normalized
   glyph rotation and `upside_down_adjusted` state.
4. For unsupported curved behavior, expect `experimental_feature` diagnostics
   before render.

## Negative Path Checks

- Empty text does not return successful create without a diagnostic.
- Invalid path geometry does not store unused metadata as success.
- Non-ASCII glyph gaps return `missing_glyphs` before render where the input
  makes them knowable.
- Terrain-elevated label request without terrain sampling returns an
  `experimental_feature` diagnostic.

## Determinism Check

Repeated runs of `examples/label_api_truth_basic.py --json` with fixed inputs
must produce the same stable IDs, ordered batch IDs, diagnostic codes, and line
glyph ordering keys. This is a feature `002` regression check only; the full
offline deterministic `LabelPlan` compiler remains feature `003`.
