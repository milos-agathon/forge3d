# Offline 3D Map Rendering

forge3d's product direction is an offline 3D map-production workflow:

```text
MapScene + LabelPlan + ValidationReport + Bundle
```

Feature `001` establishes the diagnostics contract and support matrices used by
that workflow. Feature `003` adds deterministic `LabelPlan`, and feature `004`
adds the public `MapScene` / `SceneRecipe` API for typed validation,
native/offscreen PNG output for fixture-backed MVP recipes, deterministic
source-derived compatibility output for symbolic fixture recipes, and
deterministic review bundles.

| Area | Support level | Notes |
| --- | --- | --- |
| Structured diagnostics | `supported` | `Diagnostic` and `ValidationReport` are public Python objects. |
| Typed MapScene recipe | `underdeveloped` | `MapScene`, `SceneRecipe`, terrain, raster, vector, label, point-cloud, building-intent, camera, lighting, output, and map-furniture recipe objects are public. |
| `MapScene.validate` | `supported` | Returns deterministic structured reports with source-data, CRS, style, glyph, label-plan, memory, support-status, VT, 3D Tiles, and building diagnostics where applicable. |
| `MapScene.render` PNG path | `supported` | Fixture-backed `.npy` terrain, PNG raster, inline vector, and label recipes render through the native/offscreen `Scene.render_rgba` path; symbolic fixture recipes keep deterministic source-derived compatibility output. Unsupported layer paths still block with typed diagnostics. |
| `MapScene.save_bundle` review bundle | `underdeveloped` | Writes deterministic review metadata, recipe intent, validation report, label plans, and label source references. Blocking diagnostics are preserved and marked non-renderable. |
| Deterministic LabelPlan | `supported` | `LabelPlan.compile` produces stable accepted/rejected labels, diagnostics, bounds, seed, and render/export payloads for fixed inputs. |
| Unsupported-path validation | `underdeveloped` | `Pro-gated`, `placeholder/fallback`, `experimental`, `unsupported`, and incomplete integration paths must remain diagnostic-bearing. |
| Material, VT, and large-scene polish | `underdeveloped` | P2 gaps are diagnosed before render: VT normal/mask as `vt_unsupported_family`, textured building material intent through texture/UV/fallback diagnostics, and large scenes through memory/cache/LOD/instancing summaries. |
| Web-first hosted tile delivery | `non-goal` | Offline map production remains the scope. |

Unsupported, `Pro-gated`, `placeholder/fallback`, `experimental`, or
`underdeveloped` paths must be reported before successful render completion.
The current product does not render textured PBR buildings end to end; scalar
fallback is not textured PBR support and must remain diagnostic-bearing.

`MapScene.render("map.png")` performs validation if needed and writes PNG output
for supported terrain/raster/vector/label recipes. Real `.npy` terrain and PNG
raster assets prefer the native/offscreen path; symbolic fixture recipes remain
on deterministic source-derived compatibility output. `MapScene.save_bundle("map.forge3d")`
records the recipe, diagnostics, support summaries, label plans, source
references, camera, lighting, output, render backend, and renderability status
through the public Python API.

## Canonical Examples

The MVP workflow is represented by three typed examples:

- `examples/mapscene_terrain_raster.py`: terrain plus raster overlay validates,
  writes native/offscreen PNG output from generated local assets, and saves a
  review bundle.
- `examples/mapscene_vector_labels.py`: terrain plus vector overlays and labels
  compiles deterministic `LabelPlan` data, writes native/offscreen PNG output,
  and saves a review bundle with label diagnostics.
- `examples/mapscene_buildings_labels.py`: terrain plus building intent and
  labels reports honest building diagnostics when the current packaging cannot
  render the building path.

Run the example and quickstart coverage with:

```powershell
python -m pytest tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py -q
```

## Support References

Use the linked guides as the support contract for MVP review:

- `guides/label_plan_guide`
- `guides/diagnostics_reference`
- `guides/style_support_matrix`
- `guides/building_support_matrix`
- `guides/tiles3d_support_matrix`
- `guides/virtual_texturing_support_matrix`
- `guides/large_scene_support`
- `guides/competitive_positioning`
