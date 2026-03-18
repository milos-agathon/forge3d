# 3D Buildings

> **Pro Feature:** The buildings import pipeline in this recipe requires a
> [Pro license](https://forge3d.dev/pro).

![3D buildings preview](images/05-3d-buildings.png)

This example uses the real CityJSON tile
`assets/geojson/10-270-592.city.json`. It is parsed into triangle meshes and
previewed directly from those triangles instead of routing them through the
terrain vector-overlay path.

## Ingredients

- `forge3d.add_buildings_cityjson()`
- `Building.positions`
- `Building.indices`

## Sketch

```python
import forge3d as f3d

layer = f3d.add_buildings_cityjson("assets/geojson/10-270-592.city.json")
for building in layer.buildings:
    vertices = building.positions.reshape(-1, 3)
    triangles = building.indices.reshape(-1, 3)
    print(building.id, vertices.shape[0], triangles.shape[0], building.height)
```

Once you have triangle positions and indices, render them through a mesh path or
your own preview renderer. Do not send CityJSON building meshes through
`add_vector_overlay`; that path is for draped overlays, not volumetric buildings.
