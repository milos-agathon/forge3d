# Recipe Family Manifest Schema

This document describes the P1-1 recipe-family manifest schema implemented in
`python/forge3d/recipe_manifest.py`. The manifest is metadata only: it records
provenance, inputs, layers, defaults, support status, diagnostics, tests, and
fixture intent for existing recipe families. It does not execute recipes,
render images, fetch data, or replace `SceneRecipe` / `MapScene`.

## Public Python Surface

- `RecipeManifest`
- `RecipeInput`
- `RecipeOutput`
- `RecipeLayer`
- `SourceEvidence`
- `GoldenFixtureIntent`
- `manifest_from_dict`
- `manifest_to_dict`
- `manifest_to_json`
- `manifest_from_json`
- `validate_manifest`
- `load_manifest`
- `save_manifest`

P1-1 intentionally does not add top-level `forge3d.__all__` exports. Import
from `forge3d.recipe_manifest`.

## Required Manifest Fields

- `schema_version`: string, currently `"1"`.
- `recipe_family`: known recipe-family token.
- `recipe_id`: stable lowercase recipe identifier.
- `status`: audit vocabulary value.
- `source_examples`: repo-relative example/source paths.
- `source_evidence`: objects with `path`, optional `line_start`, optional
  `line_end`, and optional `note`.
- `required_inputs`: `RecipeInput` objects.
- `optional_inputs`: `RecipeInput` objects.
- `produced_outputs`: `RecipeOutput` objects.
- `layers`: ordered `RecipeLayer` objects.
- `alignment`: declarative CRS, extent, resolution, transform, or notes.
- `preprocessing`: JSON-compatible metadata.
- `camera_defaults`: JSON-compatible metadata.
- `lighting_defaults`: JSON-compatible metadata.
- `styling_defaults`: JSON-compatible metadata.
- `annotations_defaults`: JSON-compatible metadata.
- `render_export_defaults`: JSON-compatible metadata.
- `support_status`: capability/status mapping.
- `diagnostics`: expected validation diagnostic tokens.
- `tests`: test names or fixture references.
- `golden_fixture_intent`: `GoldenFixtureIntent` object.
- `non_goals`: explicit exclusions.
- `open_questions`: blockers only; first-batch fixtures use `[]`.

JSON serialization is deterministic: sorted keys, two-space indentation, and a
trailing newline.

## Descriptor Shapes

`RecipeInput`:

- `name`: string.
- `kind`: string.
- `crs`: optional string.
- `shape`: optional JSON-compatible value.
- `format`: optional string.
- `role`: optional string.

`RecipeOutput`:

- `kind`: string.
- `format`: string.
- `path`: optional string. Use `"example_defined"` when the path policy is
  script-local.
- `deterministic`: optional boolean.

`RecipeLayer`:

- `layer_id`: string.
- `layer_type`: allowed layer token.
- `role`: string.
- `required`: boolean.
- `style_ref`: optional string.

`SourceEvidence`:

- `path`: repo-relative path.
- `line_start`: optional integer.
- `line_end`: optional integer.
- `note`: optional string.

`GoldenFixtureIntent`:

- `status`: `exists`, `missing`, or `deferred`.
- `source_test`: optional repo-relative test path.
- `source_file`: optional repo-relative evidence file.
- `note`: optional string.

## Allowed Families

First-batch fixture families:

- `terrain_demo`
- `terrain_label`
- `landcover_esri_terrain_viewer`
- `climate_bivariate`
- `hydrology_river`
- `mapscene_showcases`

Registered later-family tokens:

- `terrain_relief_rem`
- `population_spike_worldpop`
- `population_ghsl_3d`
- `builtup_cover_3d`
- `pointcloud_cog`
- `urban_osm_city`
- `luxembourg_rail_overlay`

## Status Vocabulary

- `proven_in_forge3d`
- `partially_proven`
- `exists_only_as_example_or_script_logic`
- `exists_but_not_exposed_as_public_api`
- `exists_but_not_tested`
- `not_found`
- `evidence_missing`
- `unclear_requires_human_confirmation`

`support_status` values use the same vocabulary.

## Layer Type Tokens

- `terrain_dem`
- `raster_continuous`
- `raster_categorical`
- `raster_bivariate`
- `vector_polygon`
- `vector_line`
- `label_annotation`
- `map_furniture`
- `mapscene_recipe`
- `pointcloud`
- `cog_raster`
- `building_footprint`
- `temporal_sequence`
- `globe_raster`

P1-1 validates token membership only. It does not claim render support for
every token.

## Diagnostic Tokens

- `recipe_manifest_missing_field`: required manifest key is absent.
- `recipe_manifest_invalid_field`: field shape or type is invalid.
- `recipe_manifest_invalid_status`: `status` or `support_status` uses an
  unknown status.
- `recipe_manifest_unknown_family`: `recipe_family` is not registered.
- `recipe_manifest_missing_source`: `source_examples` or `source_evidence.path`
  does not exist locally.
- `recipe_manifest_unsupported_layer`: unknown `layer_type`.
- `recipe_manifest_alignment_unspecified`: multiple spatial inputs omit CRS or
  alignment assumptions.
- `recipe_manifest_render_path_unspecified`: render/export output omits a path
  and no `"example_defined"` path policy is declared.
- `recipe_manifest_golden_not_selected`: fixture intent is `missing` or
  `deferred`.
- `recipe_manifest_mapscene_partial`: manifest declares partial MapScene
  compatibility.
- `recipe_manifest_example_only`: recipe or support status exists only as
  example/script logic.
- `recipe_manifest_schema_version_unsupported`: schema version is not `"1"`.

## First-Batch Fixtures

Fixture manifests live under `tests/fixtures/recipe_manifests/`:

- `terrain_demo.json`
- `terrain_label.json`
- `landcover_esri_terrain_viewer.json`
- `climate_bivariate.json`
- `hydrology_river.json`
- `mapscene_showcases.json`

Validation checks local source-path existence only. It performs no network
access, imports no GIS/runtime backend libraries, and performs no rendering.

## Non-Goals Preserved

- No raster/vector/terrain alignment diagnostics implementation.
- No G-003 pass compositing or blend modes.
- No categorical, bivariate, line-stroke, map-plate, building, COG, point-cloud,
  urban-context, or label-default implementation.
- No recipe execution behavior.
- No visual goldens.
- No rendering tests.
- No Rust, PyO3, shader, renderer, GIS runtime, COG, pointcloud, terrain,
  building, label, legend, scale bar, or north arrow changes.
