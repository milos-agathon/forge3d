# Contract: MapScene MVP

Final module path is TBD. Public behavior:

```python
scene = MapScene(
    terrain=TerrainSource(...),
    camera=OrbitCamera(...),
    lighting=LightingPreset(...),
    layers=[RasterOverlay(...), VectorOverlay(...), LabelLayer(...), PointCloudLayer(...), BuildingLayer(...)],
    output=OutputSpec(width=3840, height=2160, format="png"),
    map_furniture=MapFurnitureLayer(...),
)

report = scene.validate()
image_or_result = scene.render("map.png")
bundle_result = scene.save_bundle("map.forge3d")
```

Contract rules:

- `validate()` returns `ValidationReport` before render.
- `render()` performs validation if needed and writes PNG only for supported non-blocked scenes.
- Warning diagnostics continue by default; fail-on-warning blocks warnings; errors/fatals always block.
- No implicit CRS transforms are applied.
- `save_bundle()` writes deterministic review intent for supported layer types and preserves diagnostics.
- Canonical examples must not call raw IPC.
- P1/P2 layer intent may be represented but unavailable behavior must validate as `Pro-gated`, `placeholder/fallback`, `experimental`, or `unsupported`.
