<role>
You are Codex acting as a senior implementation engineer for forge3d.
</role>

<objective>
Close the remaining P0/P1 implementation gaps identified by the independent re-audit in:

`docs/3d-map-rendering-quality-blender-outmatch-plan.md`

Do not re-implement already-full work. The target is to move the nine `partial` tasks to `full` with execution-backed evidence, keep the seven verified-full tasks green, and update the plan's `Implementation audit` blocks only when the evidence supports the new status.
</objective>

<current_audit_baseline>
Independent re-audit date: 2026-07-03.

Status counts:
- `full`: 7
- `partial`: 9
- `none`: 0

Already `full`; treat as regression gates only unless touched:
- `BOP-P0-01`
- `BOP-P0-03`
- `BOP-P0-04`
- `BOP-P0-05`
- `BOP-P1-02`
- `BOP-P1-05`
- `BOP-P1-08`

Must close:
- `BOP-P0-02`
- `BOP-P0-06`
- `BOP-P1-01`
- `BOP-P1-03`
- `BOP-P1-04`
- `BOP-P1-06`
- `BOP-P1-07`
- `BOP-P1-09`
- `BOP-P1-10`
</current_audit_baseline>

<operating_rules>
- Start with `git status --short`; the worktree is dirty. Do not revert unrelated changes.
- Read `AGENTS.md`, `CONTRIBUTING.md`, and the P0/P1 sections of `docs/3d-map-rendering-quality-blender-outmatch-plan.md` before editing.
- Use repo skills when the work touches their surface: `slice-review-loop`, `render-quality-validation`, `gis-carto-contract`, `pyo3-native-surface`, `wgsl-shader-pipeline`, and `example-gallery-recipes`.
- Fix root causes in shared paths after grepping callsites. Do not patch only the named failing test.
- Keep scope to P0/P1 gap closure. Do not broaden into P2/P3 unless a P0/P1 definition of done explicitly requires it.
- Do not add dependencies unless a gap cannot be closed without one. `pyproj`/`mypy` are allowed because the audit explicitly identified them as missing gates.
- Do not weaken visual thresholds or convert real GPU checks into mock/source-grep checks.
- Rebuild the native extension with `maturin develop` before making runtime claims about Rust/WGSL changes.
</operating_rules>

<first_slice_preflight>
This is the highest-leverage blocker. Do it before feature work.

1. Verify the interpreter and extension freshness:
   ```powershell
   git status --short
   python -c "import sys, forge3d; print(sys.executable); print(forge3d.__file__)"
   Get-ChildItem -Recurse -Include *.rs,*.wgsl | Where-Object { $_.LastWriteTime -gt (Get-Item python/forge3d/_forge3d*.pyd).LastWriteTime } | Select-Object FullName
   ```
   Expected: no Rust/WGSL source newer than the `.pyd`, or run `maturin develop`.

2. Fix the CRS backend in this environment:
   ```powershell
   python -c "import pyproj, sys; print(pyproj.__file__); print(hasattr(pyproj, 'Geod')); print(pyproj.Geod(ellps='WGS84').inv(0,0,1,0))"
   ```
   If this imports an empty namespace or a stale `D:\forge3d` `.pth` shadow, remove/rename the stale `.pth` outside the repo and install a real package:
   ```powershell
   python -m pip install pyproj
   ```
   Do not commit environment-local `.pth` edits.

3. Confirm recipe goldens are tracked or intentionally added:
   ```powershell
   git status --short tests/golden/recipes tests/golden/presets
   ```
   If these are intended baselines, add them in the final change set. Do not leave P1-10 depending on untracked goldens.

4. Reproduce the red gate:
   ```powershell
   python -m pytest tests/test_mapscene_alignment.py tests/test_mapscene_furniture.py tests/test_recipe_goldens.py -q
   ```
   Expected current blockers: pyproj-related skips/failures if CRS is broken, and deterministic `mapscene_furniture_graticule` SSIM failure around `0.8146`. Fix or intentionally re-baseline after diff evidence.
</first_slice_preflight>

<gap_closure_slices>
Work in this order. Each slice ends with focused tests, plan audit update, and a small status note.

## Slice 1 - Environment, CRS, Furniture, Recipe Red Gate

Task IDs: `BOP-P1-01`, `BOP-P1-07`, `BOP-P1-10`.

Close:
- Real `pyproj` or native `proj` backend is functional in the test environment.
- Geodesic guards require functional `pyproj.Geod`, not merely an importable `pyproj` module.
- `tests/test_mapscene_alignment.py` runs its CRS assertions instead of skipping due to the stale stub.
- Add a bundle-roundtrip assertion for alignment provenance fields.
- Fix or re-baseline `mapscene_furniture_graticule` only after generating compare artifacts and deciding which backend is intended.
- Re-run the UTM graticule golden once CRS works.
- Ensure `tests/golden/recipes/` and `tests/golden/presets/` intended baselines are committed.

Required evidence:
```powershell
python -m pytest tests/test_mapscene_alignment.py tests/test_mapscene_furniture.py -q
$env:FORGE3D_RUN_TERRAIN_GOLDENS="1"; python -m pytest tests/test_recipe_goldens.py -q
python scripts/compare_images.py <expected.png> <actual.png> --ssim --diff <diff.png> --json <metrics.json>
```

## Slice 2 - MapScene Label/Vector Default Path

Task ID: `BOP-P0-02`.

Close:
- The CPU label/vector fallback must not silently run on the `gpu_terrain` backend. Gate it behind `allow_placeholder=True` or fail with a diagnostic that names the missing native capability.
- Add a public `MapScene.render()` determinism test: two renders of the same label/vector recipe produce bit-identical accepted-label sets and bit-identical pixels.
- Add a real-path no-placeholder-rectangle regression test; do not use only `FakeNativeTextScene`.
- Bind `FontAtlas` on `LabelLayer` or update the task contract and docs if the existing `glyph_atlas` binding is the intended public API.
- Migrate `examples/fuji_labels_demo.py` to the public GPU label path.

Required evidence:
```powershell
python -m pytest tests/test_mapscene_render_png.py tests/test_mapscene_examples.py tests/test_text_atlas.py -q
$env:FORGE3D_RUN_TERRAIN_GOLDENS="1"; python -m pytest tests/test_recipe_goldens.py -k "vector_labels or label_halo" -q
```

## Slice 3 - Quickstart and Typing Gate

Task ID: `BOP-P0-06`.

Close:
- Add a test that extracts and executes the actual `docs/start/quickstart.md` MapScene snippet. Placeholder mode is acceptable only if the snippet explicitly asks for it.
- Make `mypy` or `pyright` installed in the test lane that runs `tests/test_mapscene_typing.py`; skipping because the tool is absent no longer counts.
- Keep `.pyi`, `__all__`, docs, and runtime exports consistent.

Required evidence:
```powershell
python -m pytest tests/test_mapscene_quickstart.py tests/test_mapscene_typing.py -q
python -m mypy --version
```

## Slice 4 - Depth-Tested Label Occlusion

Task ID: `BOP-P1-03`.

Close:
- Labels without explicit `z` derive occlusion depth from the projected anchor against the depth AOV.
- Add an end-to-end test of real `render_with_aov` depth feeding label occlusion. Do not use `FakeAovFrame` for the closure proof.
- Add a real-ridge-depth golden with a declutter-slot-release assertion.
- Add dense-point determinism coverage and either a curved-line golden or a documented unsupported substitution.
- Implement or delete the dead Rust `project_with_occlusion` stub; do not leave a misleading stub in the API surface.

Required evidence:
```powershell
python -m pytest tests/test_mapscene_label_occlusion.py -q
$env:FORGE3D_RUN_TERRAIN_GOLDENS="1"; python -m pytest tests/test_recipe_goldens.py -k "label_occlusion or ridge" -q
```

## Slice 5 - Vector Joins, Dashes, and Supersample Golden

Task ID: `BOP-P1-04`.

Close:
- Geometric miter joins with a miter limit and bevel joins are implemented in the path that reaches rendered pixels.
- Joined/dashed stroke behavior is identical on the GPU OIT path, or dashed/joined strokes route exclusively through the correct path.
- Add a join-shape-differentiating test or golden that would fail if miter/bevel were approximated as the old fragment-only join.
- Add the 4x supersample torture golden variant.

Required evidence:
```powershell
python -m pytest tests/test_mapscene_vector_strokes.py -q
$env:FORGE3D_RUN_TERRAIN_GOLDENS="1"; python -m pytest tests/test_recipe_goldens.py -k "stroke or vector" -q
```

## Slice 6 - Building CSM Shadows and Kept Claims

Task ID: `BOP-P1-06`.

Close:
- Building meshes render into the cascaded shadow-map pass and sample cascades in mesh shading, replacing projected-quad shadow approximation.
- Add a real rendered test/golden where a building casts and receives shadow.
- Either add footprint-aware ridge generation for non-rectangular footprints or document the bounding-box limitation and remove any stronger claim.
- Wire per-building IDs into picking if the task's picking claim is kept; otherwise remove the claim from the task/docs.

Required evidence:
```powershell
python -m pytest tests/test_mapscene_buildings.py -q
cargo test --lib building --all-features
```
If WGSL or bind layouts change, also run the relevant runtime pipeline-creation test and `maturin develop`.

## Slice 7 - Tonemap and Color Contract

Task ID: `BOP-P1-09`.

Close:
- Replace the self-comparison tonemap test with a real cross-path test: render or tonemap the same HDR frame through the offline path and postfx path, then assert <= 1 LSB after quantization.
- Delete or compile-guard `src/shaders/tonemap.wgsl` so there is no dead, ambiguous tonemap shader.
- Commit `docs/guides/color-management.md` if it is intended documentation.

Required evidence:
```powershell
python -m pytest tests/test_color_management.py tests/test_exr_output.py -q
cargo test --lib tonemap --all-features
```

## Slice 8 - CI Quality Gates and Red-Gate Documentation

Task ID: `BOP-P1-10`.

Close:
- CI recipe lane installs the CRS backend (`pyproj` or a `proj`-built wheel) so CRS failures are real failures.
- Hosted CI must either run a GPU-capable render-and-match lane or the plan/docs must explicitly scope hosted CI to meta-validation while the GPU lane runs on an identified self-hosted/software adapter path.
- Add or document the broken-shader red-gate procedure so it is reproducible, not an oral claim.
- Confirm shader-sensitive recipe checks run rather than skip in the intended lane.

Required evidence:
```powershell
python -m pytest tests/test_recipe_manifest.py tests/test_recipe_goldens.py -q
python scripts/terrain_ci_probe.py --mode terrain
```
Also inspect `.github/workflows/ci.yml` and record exactly which lane runs which gate.
</gap_closure_slices>

<hard_requirements>
- A task is `full` only when its stated definition of done is met through the public/intended runtime path.
- Mock tests may cover wiring, but they cannot be the only proof for render-quality or GPU-path closure.
- Any visual claim needs backend label plus numeric evidence: SSIM/MAE or equivalent, diff image path, and artifact paths.
- Do not regenerate goldens just to green a failing test. Produce diff evidence first and explain why the new baseline is correct.
- Do not leave `MapScene.render()` default behavior silently using placeholders for premium recipe output.
- Do not leave verification theater: no expression-compared-to-itself tests, no "skip if tool missing" gate for required typing checks, no source-only shader validation for pixel-affecting changes.
- If native APIs move, update PyO3 registration, Python wrappers, `.pyi`, `__all__`, package data, and contract tests in the same slice.
- If WGSL bindings/layouts move, trace declarations through Rust bind-group layouts and run a runtime pipeline-creation/render test.
</hard_requirements>

<plan_update_requirements>
After each slice, update only the affected `Implementation audit` block(s) in `docs/3d-map-rendering-quality-blender-outmatch-plan.md`.

Use the existing block shape:
```markdown
- **Implementation audit status:** `full|partial|none`
- **Audit date:** YYYY-MM-DD
- **Audited evidence:** ...
- **Why this status:** ...
- **Remaining coding to close:** ...
```

When all nine partial tasks are closed, update `## P0/P1 Implementation Audit Summary` before `## P2`:
- counts must become `full 16`, `partial 0`, `none 0`
- list the commands actually run
- list visual artifact paths and metrics for changed goldens
- preserve P2/P3 text
</plan_update_requirements>

<final_validation>
Run focused tests per slice, then run the smallest broad gate that covers touched surfaces:

```powershell
git status --short
python -m pytest tests/test_api_contracts.py -q
python -m pytest tests/test_mapscene_render_png.py tests/test_mapscene_examples.py tests/test_mapscene_quickstart.py tests/test_mapscene_typing.py -q
python -m pytest tests/test_mapscene_alignment.py tests/test_mapscene_furniture.py -q
python -m pytest tests/test_text_atlas.py tests/test_mapscene_label_occlusion.py tests/test_mapscene_vector_strokes.py -q
python -m pytest tests/test_mapscene_buildings.py -q
python -m pytest tests/test_color_management.py tests/test_exr_output.py -q
python -m pytest tests/test_recipe_manifest.py tests/test_recipe_goldens.py -q
cargo forge3d-clippy
cargo test --workspace --all-features
$env:FORGE3D_RUN_TERRAIN_GOLDENS="1"; python -m pytest tests/test_terrain_visual_goldens.py tests/test_terrain_tv10_goldens.py tests/test_recipe_goldens.py -q
```

If a command is unavailable or skipped, record the exact reason and which task remains unsupported by that missing evidence.
</final_validation>

<final_response_format>
Return only:
- changed files grouped by slice
- P0/P1 task status table
- validation commands run and outcomes
- visual evidence/golden artifact paths
- skipped checks or blockers
- next exact slice only if anything remains non-full
</final_response_format>
