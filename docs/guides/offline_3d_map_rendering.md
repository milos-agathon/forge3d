# Offline 3D Map Rendering

forge3d's product direction is an offline 3D map-production workflow:

```text
MapScene + LabelPlan + ValidationReport + Bundle
```

Feature `004` establishes the typed `MapScene` MVP, and feature `005` extends
that workflow with asset bundle guardrails. Deterministic `LabelPlan`
placement is part of the public offline map-production path.

| Area | Support level in this feature | Notes |
| --- | --- | --- |
| Structured diagnostics | `supported` | `Diagnostic` and `ValidationReport` are public Python objects. |
| `MapScene.render` PNG path | `supported` | Fixture-backed scenes can render via native/offscreen PNG output or source-derived fallback with `last_render_backend` recorded. |
| `MapScene.save_bundle` | `supported` | Bundles persist recipes, review payloads, layer sources, label sources, and validation reports. |
| Deterministic `LabelPlan` | `supported` | Label candidates and accepted IDs are stable for equivalent scene inputs. |
| Unsupported-path validation | `underdeveloped` | Diagnostics are emitted before render or bundle success for incomplete paths. |
| Web-first hosted tile delivery | `non-goal` | Offline map production remains the scope. |

`unsupported`, `Pro-gated`, `placeholder/fallback`, `experimental`, or
`underdeveloped` paths must be reported before successful render completion.

Canonical examples:

- `examples/mapscene_terrain_raster.py`
- `examples/mapscene_vector_labels.py`
- `examples/mapscene_buildings_labels.py`

Related guides:

- `guides/label_plan_guide`
- `guides/diagnostics_reference`
- `guides/style_support_matrix`
- `guides/building_support_matrix`
- `guides/tiles3d_support_matrix`
- `guides/virtual_texturing_support_matrix`
- `guides/competitive_positioning`

Verification:

```bash
python -m pytest tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py -q
```
