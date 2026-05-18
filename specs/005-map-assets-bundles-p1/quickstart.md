# Quickstart: Map Assets And Bundle Round-Trip P1

These scenarios are validation targets for implementation.

## Data-Driven Labels

1. Create `LabelLayer` from point, line, and polygon local fixtures.
2. Evaluate `{name}`, `get`, `concat`, `coalesce`, and casing expressions.
3. Transform CRS through forge3d utilities where available or emit `crs_mismatch`.
4. Compile into `LabelPlan` and compare deterministic accepted/rejected output.

## Typography And Font Coverage

1. Use bundled default Latin atlas and validate coverage.
2. Generate or load a TTF/OTF atlas where supported.
3. Declare fallback ranges.
4. Validate Unicode coverage gaps before render.

## Buildings And 3D Tiles

1. Add CityJSON/GeoJSON building fixtures to `MapScene`.
2. Validate native, Pro-gated, placeholder/fallback, and unsupported paths.
3. Add supported local 3D Tiles fixture.
4. Validate unsupported formats/features produce typed diagnostics.

## Bundle Round-Trip

1. Save a scene with terrain, labels, diagnostics, and supported asset metadata.
2. Load the bundle with all assets present and validate equivalent intent.
3. Remove or move one external asset and load again.
4. Expect `missing_external_asset` diagnostics and no false render success.
