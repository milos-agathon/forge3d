# Contract: P1 Assets And Bundles

Final module paths extend the P0 product APIs established by features `001` through `004`.

```python
LabelLayer.from_features(features, *, text, crs=None, terrain_sampling="auto", typography=None)
LabelLayer.from_geodataframe(gdf, *, text, crs=None, terrain_sampling="auto", typography=None)
LabelLayer.from_style_layer(features, style_layer, *, crs=None)

FontAtlas.default_latin()
FontAtlas.from_font(path, *, ranges=None)

BuildingLayer.from_geojson(path, **options)
BuildingLayer.from_cityjson(path, **options)
Tiles3DLayer(path, *, lod=None, cache_budget=None)

bundle = MapScene.load_bundle(path)
scene.save_bundle(path)
```

Contract rules:

- Ingestion produces real label/building/tile state or typed diagnostics.
- Label expression evaluation is deterministic and reports missing fields.
- Bundles store both source labels and compiled `LabelPlan` payloads where available.
- Bundle load reports missing external assets with structured diagnostics.
- Unavailable building/3D Tiles rendering remains `Pro-gated`, `placeholder/fallback`, `experimental`, or `unsupported`; diagnostic support alone is not render support.
