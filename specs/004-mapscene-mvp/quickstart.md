# Quickstart: MapScene MVP

These scenarios are validation targets for implementation.

## Terrain Plus Raster

Run:

```powershell
python examples/mapscene_terrain_raster.py --output-dir examples/out/mapscene_terrain_raster --json
```

1. Create `MapScene` with `TerrainSource`, `RasterOverlay`, camera, lighting, and PNG `OutputSpec`.
2. Run `validate()` and assert a structured `ValidationReport`.
3. Run `render("terrain_raster.png")` and assert deterministic native/offscreen PNG output is written for the generated `.npy` terrain and PNG raster recipe.
4. Run `save_bundle("terrain_raster.forge3d")` and compare deterministic manifest fields plus review status.

## Terrain Plus Vector Plus Labels

Run:

```powershell
python examples/mapscene_vector_labels.py --output-dir examples/out/mapscene_vector_labels --json
```

1. Create `VectorOverlay` from provided local features and style.
2. Create `LabelLayer` with point/polygon labels and fixed seed.
3. Validate style support and glyph coverage before render.
4. Compile labels through `LabelPlan` during render preparation.
5. Confirm render writes native/offscreen PNG output while preserving label-plan diagnostics in the review bundle.

## Terrain Plus Buildings Plus Labels

Run:

```powershell
python examples/mapscene_buildings_labels.py --output-dir examples/out/mapscene_buildings_labels --json
```

1. Add `BuildingLayer` intent with a known fixture.
2. If native/public building path is available, validate geometry counts and renderability.
3. If unavailable, expect `pro_gated_path`, `placeholder_fallback`, or unsupported diagnostics.
4. The example must not claim successful building rendering when diagnostics block it.

## Negative Path Checks

- CRS mismatch without explicit compatible policy emits `crs_mismatch`.
- Unsupported style fields emit `unsupported_style_field`.
- Missing glyphs emit `missing_glyphs`.
- Render before explicit validation still exposes validation report.
- Bundle save with blocking diagnostics preserves diagnostics and does not imply successful render.
