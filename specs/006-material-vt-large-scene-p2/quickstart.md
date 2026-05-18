# Quickstart: Material, VT, and Large-Scene P2

These scenarios are validation targets for implementation.

## VT Family Validation

1. Validate an albedo-only VT scene.
2. Validate a scene requesting albedo plus normal and mask families.
3. If normal/mask are not implemented, expect `vt_unsupported_family`.
4. Confirm render does not silently skip requested families while reporting success.

## Textured Buildings

1. Add a building fixture with albedo texture and UVs.
2. Validate and render or receive typed diagnostics explaining unavailable texture support.
3. Add missing-UV and missing-texture fixtures.
4. Expect `missing_uvs` and `missing_texture_path` before render.

## Advanced Labels

1. Compile repeated line labels with fixed repeat distance and seed.
2. Compile curved road/river labels where supported.
3. Apply priority presets for capitals, cities, rivers, peaks, roads, and annotations.
4. Compare accepted/rejected outputs exactly or expect typed experimental/unsupported diagnostics.

## Large-Scene Diagnostics

1. Validate a scene with large terrain, point cloud, buildings, and tiles.
2. Expect memory budget estimates where metadata exists.
3. Expect cache/LOD/instancing stats where available.
4. Missing stats must produce unavailable/unsupported diagnostics, not invented values.
