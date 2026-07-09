# Fable 5 Prompt: SUTURA Final Hardening

Use this prompt with Claude Fable 5 from the repository root.

Boiler basis: `docs/prompts/fable5-moonshots/01-aequitas-final-hardening-fable5-prompt.md`.

```xml
<role>
You are Claude Fable 5 acting as a senior forge3d implementation engineer and product-integrity auditor. Your task is to harden SUTURA from "zero-placeholder gate currently passes" to an honestly durable MapScene implementation.
</role>

<effort>
Use high effort. This is a correctness-sensitive Python/MapScene bundle-integrity task. Keep the diff tightly scoped to SUTURA hardening. Prefer the smallest root-cause fix over new renderer work.
</effort>

<objective>
Make the current SUTURA implementation satisfy docs/prompts/fable5-moonshots/10-sutura.md under a strict product-integrity reading, or return an explicit blocked verdict with code-backed reasons.

Do not treat the current passing SUTURA test gate as sufficient by itself. The implementation must also prevent stale compiled plans after recipe mutation, stop presenting deterministic CPU raster/vector compositor paths as native-only, and make MapScene bundle version handling match the v3/v2 contract.
</objective>

<current_verified_state>
Fresh Codex audit evidence from 2026-07-09:

- `python -m pytest tests/test_mapscene_sutura_integrity.py -v --tb=short`: 15 passed, including the SSIM/re-render cases.
- `python -m pytest tests/test_mapscene_label_occlusion.py tests/test_mapscene_save_bundle.py tests/test_mapscene_render_policy.py tests/test_p1_bundle_roundtrip.py -q`: 18 passed.
- `python -m pytest tests/test_mapscene_render_png.py -q`: 26 passed.
- `python -m pytest tests/test_mapscene_vector_strokes.py -q`: 12 passed.
- `python -m pytest tests/test_api_contracts.py tests/test_mapscene_docs.py tests/test_recipe_manifest.py -q`: 377 passed, 1 skipped.
- `rg -n "allow_placeholder" python/forge3d`: no matches.

The core SUTURA implementation is real: `allow_placeholder` is gone, `MapSceneNativeUnavailable` exists, `CompiledScenePlan` is required by `_render_native_offscreen_result`, `RecipeManifest.compiled_label_plans` and `depth_cull` persist compile-phase state, `BUNDLE_VERSION` is 3, and `scene/compiled_plan.json` is written and rehydrated.
</current_verified_state>

<already_present>
Do not redo these from scratch unless verification proves they regressed:

1. `MapScene.render()` no longer accepts `allow_placeholder`.
2. `MapSceneNativeUnavailable` carries `.diagnostics` blocks shaped by `_map_scene_validation.diagnostic_block`.
3. `_render_native_offscreen_result(recipe, compiled, ...)` rejects non-`CompiledScenePlan` input.
4. `MapScene.render()` compiles on demand when `self.compiled_plan is None`.
5. `MapScene.compile_plan()` freezes label plans and depth-cull visibility into `RecipeManifest`.
6. `manifest_to_json()` canonicalizes finite floats and rejects NaN/Inf.
7. v3 bundles persist `scene/compiled_plan.json`; missing compiled plans are recompiled once for v2-style bundles.
8. The four SUTURA recipes live in `tests/_sutura_recipes.py`.
</already_present>

<non_negotiable_remaining_gaps>
1. Stale compiled plans can survive recipe mutation.

   Evidence:
   - `python/forge3d/map_scene.py`: `render()` reads `compiled = self.compiled_plan` and only recompiles when it is `None`.
   - `python/forge3d/map_scene.py`: `save_bundle()` does the same.
   - `CompiledScenePlan.recipe_hash` already exists and is computed from `self.recipe.to_dict()` in `compile_plan()`.
   - Reproducer from the audit:
     `scene.compile_plan(); scene.recipe.layers = (); scene.save_bundle(...)` produced a bundle whose serialized recipe had 0 layers while `compiled_plan.json` still had `compiled_label_plans: ["labels"]`.

   Required outcome:
   - Before every public render or bundle save, reuse a compiled plan only if `compiled.recipe_hash == _stable_hash(self.recipe.to_dict())`.
   - If the hash differs, recompile once from the current recipe.
   - Use the existing `_stable_hash` and `CompiledScenePlan.recipe_hash`; do not add a new dirty-tracking system.
   - The recompiled plan must be the one rendered and saved.

   Preferred minimal shape:
   ```python
   def _compiled_plan_for_current_recipe(self) -> CompiledScenePlan:
       current_hash = _stable_hash(self.recipe.to_dict())
       compiled = self.compiled_plan
       if compiled is None or compiled.recipe_hash != current_hash:
           return self.compile_plan()
       return compiled
   ```

   Use that helper from `render()` and `save_bundle()`, or inline the same logic in both places if that produces a smaller clear diff.

2. Deterministic CPU compositor exceptions are still described as native-only success.

   Evidence:
   - `python/forge3d/_map_scene_validation.py` says every MapScene layer is native-required.
   - `python/forge3d/map_scene.py` still composites raster overlays through `_load_native_raster_overlay()` plus `_composite_recipe_layers(..., include_raster=True)`.
   - `python/forge3d/map_scene.py` routes dashed/mitered precise vectors through `_composite_recipe_layers(..., include_vectors=True)` and returns `composited=True`.
   - `tests/test_mapscene_vector_strokes.py::test_styled_vector_layers_route_to_precise_raster_path` intentionally locks this precise Python raster path.

   Required outcome:
   - Do not implement a new native vector tessellator or native raster compositor in this prompt.
   - Do not block the existing precise vector/raster behavior unless a focused test proves it is placeholder output.
   - Make the contract honest: these paths are deterministic CPU compositors, not native-only compositors and not placeholders.
   - Surface that truth in render metadata and support features so future docs/tests cannot claim "all layers native-only" while these paths remain.

   Preferred minimal metadata:
   - When raster overlays are present and composited by `_composite_recipe_layers`, set native-result metadata:
     `raster_overlay_backend = "python_resample_composite"` and `raster_overlay_layer_count = <count>`.
   - When `_vector_layer_requires_precise_raster(layer)` routes vectors through `_composite_recipe_layers`, set:
     `vector_backend = "python_precise_raster"`.
   - Preserve existing native OIT metadata for ordinary vector, point-cloud, and 3D Tiles paths.
   - Include these metadata keys in `MapScene.last_render_metadata`.
   - Add supported feature markers such as `mapscene.raster_overlay_composite` and `mapscene.vector_precise_raster_composite` when those paths run.

3. `MapScene.load_bundle()` bypasses bundle manifest version rejection.

   Evidence:
   - `python/forge3d/bundle.py::BundleManifest.load()` rejects `manifest.version > BUNDLE_VERSION`.
   - `python/forge3d/map_scene.py::MapScene.load_bundle()` reads `scene/mapscene_recipe.json` and `scene/compiled_plan.json` directly.
   - Audit reproducer: change a saved MapScene bundle `manifest.json` version to `999`; `MapScene.load_bundle(bundle)` still succeeds.

   Required outcome:
   - `MapScene.load_bundle()` must read `manifest.json` with `BundleManifest.load()` when the manifest exists.
   - A bundle with `version > BUNDLE_VERSION` must raise `ValueError`.
   - Keep the v2 read path: version 2 or missing `scene/compiled_plan.json` recompiles once from the serialized recipe.
   - Do not break existing v3 `save_bundle()` / `load_bundle()` round-trips.
</non_negotiable_remaining_gaps>

<implementation_rules>
- Start by reading `docs/prompts/fable5-moonshots/10-sutura.md`, this prompt, and the current `python/forge3d/map_scene.py` flow around `compile_plan()`, `render()`, `save_bundle()`, and `load_bundle()`.
- Run `git status --short` before editing. Do not revert unrelated dirty files.
- Expected scope is Python tests and `python/forge3d/map_scene.py`; avoid Rust/WGSL changes unless fresh evidence proves they are required.
- No new dependencies.
- No reintroduction of `allow_placeholder`.
- No CPU placeholder branch. Deterministic CPU compositor paths may remain only if explicitly named as such and covered by tests.
- Do not lower SSIM thresholds, skip gates, or rename tests to hide incomplete behavior.
- Do not fix unrelated full-suite failures unless they directly block SUTURA verification.
</implementation_rules>

<required_checks>
Add or update the smallest tests that prove the repaired behavior:

1. A test that compiles a scene, mutates `scene.recipe.layers`, calls `save_bundle()`, and proves `scene/compiled_plan.json` matches the mutated recipe rather than the stale plan.
   - Concrete assertion: after mutating layers to `()`, `compiled_label_plans` is absent or empty and `scene/mapscene_review.json["compiled_label_plan_ids"] == []`.

2. A test that compiles a scene, mutates the recipe, monkeypatches `_render_native_offscreen_result`, calls `render()`, and proves the compiled plan passed into the renderer has `recipe_hash == _stable_hash(scene.recipe.to_dict())`.

3. A test that renders a raster-overlay scene and asserts `scene.last_render_metadata["raster_overlay_backend"] == "python_resample_composite"` and the support report marks `mapscene.raster_overlay_composite` supported.

4. Preserve or update `tests/test_mapscene_vector_strokes.py::test_styled_vector_layers_route_to_precise_raster_path` so it also asserts the precise vector route is not hidden as native OIT when run through public `MapScene.render()`.

5. A test that saves a MapScene bundle, edits `manifest.json` to `version: 999`, and asserts `MapScene.load_bundle()` raises `ValueError`.

6. Preserve `tests/test_mapscene_sutura_integrity.py::test_no_allow_placeholder_symbol`.
</required_checks>

<verification>
Run these commands after implementation:

1. `python -m pytest tests/test_mapscene_sutura_integrity.py -v --tb=short`
2. `python -m pytest tests/test_mapscene_vector_strokes.py tests/test_mapscene_render_png.py tests/test_mapscene_save_bundle.py tests/test_p1_bundle_roundtrip.py -q`
3. `python -m pytest tests/test_api_contracts.py tests/test_mapscene_docs.py tests/test_recipe_manifest.py -q`
4. `rg -n "allow_placeholder" python/forge3d`
5. `git status --short`

If Rust or WGSL was touched, also run:

1. `maturin develop --release`
2. `cargo fmt --check`
3. `cargo forge3d-clippy`

Read the full output. Report exit codes and pass/fail counts. For `rg -n "allow_placeholder" python/forge3d`, success means no matches and exit code 1.
</verification>

<final_output>
Return:

- Outcome: complete, incomplete, or blocked.
- Files changed.
- Requirement-by-requirement SUTURA hardening status.
- Exact verification command results.
- Explicit answer to: "Can a stale compiled plan be rendered or saved after recipe mutation?"
- Explicit answer to: "Are raster and precise-vector CPU compositor paths still present, and if so are they honestly reported?"
- Explicit answer to: "Does MapScene.load_bundle reject future bundle versions?"
- Any unrelated full-suite failures or skipped checks.
</final_output>
```
