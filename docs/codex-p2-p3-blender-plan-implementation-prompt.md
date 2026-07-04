# Prompt: Codex P2/P3 Blender-Outmatch Implementation

Use this prompt in a fresh Codex session rooted at `C:\Users\milos\forge3d`.

```text
<role>
You are Codex acting as a senior implementation engineer for forge3d.
</role>

<objective>
Accurately, surgically, and rigorously implement the remaining P2 and P3 tasks from:

`docs/3d-map-rendering-quality-blender-outmatch-plan.md`

Close the P2 scale/polish tasks as real runtime capabilities, then handle P3 by evidence: implement only after prerequisites are satisfied and the spike outcome supports it; otherwise delete/defer with a recorded decision exactly as the plan allows.
</objective>

<operating_rules>
- Start with `git status --short`; the worktree may be dirty. Do not revert unrelated changes.
- Read `AGENTS.md`, `CONTRIBUTING.md`, and the full P2/P3 sections of the plan before editing.
- Confirm P0/P1 status first. Do not start a P2/P3 slice whose prerequisites are still partial unless the slice explicitly exists to unblock it.
- Use repo skills when their surface is touched: `slice-review-loop`, `render-quality-validation`, `pyo3-native-surface`, `wgsl-shader-pipeline`, `gis-carto-contract`, and `example-gallery-recipes`.
- Work one bounded slice at a time. Each slice must name task IDs, runtime path, focused tests, and stop conditions before editing.
- Prefer wiring existing code over adding parallel systems. Prefer deleting dead code over preserving misleading capability.
- Do not add dependencies except the P3 text-shaping dependency explicitly allowed by the plan (`rustybuzz` + `unicode-bidi`), and only in the `BOP-P3-03` slice.
- Rebuild the native extension with `maturin develop` before making runtime claims about Rust/WGSL changes.
- Visual/render-quality claims require backend labels, artifacts, metrics, and diff evidence.
</operating_rules>

<scope>
Implement or evidence-close:

P2:
- `BOP-P2-01` ship COG streaming, bounded caches, TIFF predictor correctness, optional on-disk cache.
- `BOP-P2-02` streaming clipmap terrain in the render path.
- `BOP-P2-03` native 3D Tiles and point-cloud Python exposure, plus EDL point shading.
- `BOP-P2-04` SSAO/SSGI/SSR and TAA reachable from terrain/MapScene.
- `BOP-P2-05` non-blocking VT feedback, staging-ring uploads, dirty page-table updates.
- `BOP-P2-06` Gerstner water, automatic water masks, cloud settings/shadows for map scenes.
- `BOP-P2-07` memory-budget enforcement and terrain-path GPU timing.
- `BOP-P2-08` renderer consolidation/delete/merge sweep.
- `BOP-P2-09` complete glTF PBR import for map assets.
- `BOP-P2-10` thematic classification and data-driven GPU vector styling.

P3:
- `BOP-P3-01` terrain-integrated path tracing or delete/record the wavefront/ReSTIR decision.
- `BOP-P3-02` camera-relative rendering for continental precision, only after clipmap scale proves need.
- `BOP-P3-03` complex text shaping with the approved dependency, only after Latin label pipeline is full.
- `BOP-P3-04` GPU-driven culling and multi-draw-indirect, only after city/object counts justify it.
</scope>

<recommended_slice_order>
1. Preflight: confirm P0/P1 full status, native extension freshness, GPU availability, CRS/raster deps, and current dirty files.
2. COG and cache foundation: `BOP-P2-01`.
3. Runtime measurement and budget foundation: `BOP-P2-07`, then `BOP-P2-05` where it needs timing/memory proof.
4. Streaming terrain: `BOP-P2-02`, using local tile pyramids before remote COG if simpler.
5. Native content cores: `BOP-P2-03`, then `BOP-P2-09` for textured assets.
6. City/terrain post effects: `BOP-P2-04`, after mesh/building paths exist.
7. Water/cloud map-scene upgrades: `BOP-P2-06`.
8. Thematic GPU styling: `BOP-P2-10`, coordinated with any vector-format changes from P1.
9. Consolidation sweep: `BOP-P2-08`, after deciding which dead shaders/files were wired by earlier slices.
10. P3 decision slices in order: `BOP-P3-01`, `BOP-P3-02`, `BOP-P3-03`, `BOP-P3-04`.
</recommended_slice_order>

<per_slice_method>
For each slice:

1. Restate the task ID, prerequisites, definition of done, and files/modules likely touched.
2. Grep every caller of the function/class/shader path before editing.
3. Write or update the smallest test/golden that fails for the current gap.
4. Implement the smallest shared-path fix; avoid one-off demo-only wiring.
5. If native APIs move, update PyO3 registration, Python wrapper, `.pyi`, `__all__`, package data, and `tests/test_api_contracts.py` together.
6. If WGSL bindings/layouts move, trace declarations through Rust bind-group layouts and run a runtime pipeline creation/render test.
7. Run focused tests and record exact output.
8. For pixel changes, render before/after at identical settings and run `scripts/compare_images.py` with SSIM/diff/json artifacts.
9. Update only affected `Implementation audit` blocks in the plan. Mark `full` only when the public/intended runtime path satisfies the definition of done.
10. Stop and reslice if requirements become ambiguous or cross an unrelated subsystem contract.
</per_slice_method>

<p2_closure_requirements>
- `BOP-P2-01`: default wheel exposes `CogDataset`; byte cache and disk cache have hard budgets; predictor-compressed COG fixture decodes correctly.
- `BOP-P2-02`: clipmap terrain feeds `TerrainRenderer`, shares the PBR terrain shader, renders >=100 km x 100 km with no cracks and bounded memory; single-tile goldens stay stable.
- `BOP-P2-03`: Python uses native 3D Tiles and point decode; `NotImplementedError` paths are deleted; MapScene renders real 3D Tiles/point-cloud content; EDL has a golden.
- `BOP-P2-04`: SSAO/SSGI/SSR settings reach terrain/MapScene; contact/reflection goldens prove effect; TAA is wired only where it helps the viewer.
- `BOP-P2-05`: no per-frame VT `Maintain::Wait`; VT feedback is async; staging rings and dirty page-table updates are measured.
- `BOP-P2-06`: water renders through GPU water from explicit or auto masks; cloud shadows reach terrain presets/examples; PIL water fakes are retired where replaced.
- `BOP-P2-07`: memory policy can warn/enforce at chokepoints; tracked bytes approximate real allocations; terrain/VT/offline timings surface in bench/diagnostics.
- `BOP-P2-08`: zero unreferenced WGSL unless explicitly allowlisted; one AO tree; dead stub VT/pipeline/API paths deleted or replaced by the real path.
- `BOP-P2-09`: glTF import handles all relevant primitives, transforms, material factors/textures/tangents, and renders a textured GLB landmark in MapScene.
- `BOP-P2-10`: thematic classification has unit tests; data-driven styles become GPU attributes; choropleth renders in one layer draw path, not per-class draw loops.
</p2_closure_requirements>

<p3_decision_rules>
- `BOP-P3-01`: run a bounded spike first. If terrain/building integration with the real path tracer is testable, implement `OutputSpec(renderer="pt")` with determinism and a golden. If not, delete the unused wavefront path or mark it deferred with an explicit plan decision and keep raster accumulation as the offline ceiling.
- `BOP-P3-02`: implement only after `BOP-P2-02` proves large-region clipmap rendering and a jitter test reproduces f32 instability. Otherwise leave deferred with evidence.
- `BOP-P3-03`: implement only after P1 Latin label quality is full. Add only `rustybuzz`/`unicode-bidi`; prove Arabic joining with a golden and retire the experimental diagnostic.
- `BOP-P3-04`: implement only after building/tiles/scatter scenes produce object counts where CPU culling is insufficient. Must have CPU-reference correctness and a 100k-instance perf benchmark.
</p3_decision_rules>

<validation_commands>
Use focused commands first. Broaden only after the slice is stable.

```powershell
git status --short
maturin develop
python -m pytest tests/test_api_contracts.py -q
python -m pytest tests/test_cog*.py tests/test_gis*.py -q
python -m pytest tests/test_tv20_virtual_texturing.py tests/test_terrain_material_maps.py -q
python -m pytest tests/test_3dtiles*.py tests/test_pointcloud*.py -q
python -m pytest tests/test_mapscene_buildings.py tests/test_mapscene_render_png.py -q
python -m pytest tests/test_recipe_manifest.py tests/test_recipe_goldens.py -q
cargo forge3d-clippy
cargo test --workspace --all-features
python scripts/terrain_ci_probe.py --mode terrain
$env:FORGE3D_RUN_TERRAIN_GOLDENS="1"; python -m pytest tests/test_terrain_visual_goldens.py tests/test_terrain_tv10_goldens.py tests/test_recipe_goldens.py -q
```

For visual changes:

```powershell
python scripts/compare_images.py <expected.png> <actual.png> --ssim --diff <diff.png> --json <metrics.json>
```
</validation_commands>

<forbidden_shortcuts>
- Do not close a P2/P3 task with source-grep tests, fake native scenes, placeholder renderers, or demo-only CPU imitation.
- Do not add broad abstractions before checking whether existing clipmap, VT, point-cloud, screen-space, PBR, and memory modules already solve the problem.
- Do not leave public APIs that advertise unsupported capability.
- Do not keep dead WGSL/files after deciding not to wire them.
- Do not weaken golden thresholds or regenerate baselines without diff evidence.
- Do not implement speculative P3 work before its prerequisites and spike evidence exist.
</forbidden_shortcuts>

<plan_update_requirements>
For each affected P2/P3 task in `docs/3d-map-rendering-quality-blender-outmatch-plan.md`, add or update one block:

```markdown
- **Implementation audit status:** `full|partial|none|deferred`
- **Audit date:** YYYY-MM-DD
- **Audited evidence:** ...
- **Why this status:** ...
- **Remaining coding to close:** ...
```

Use `deferred` only for P3 decisions explicitly allowed by the plan, and include the evidence that made implementation premature or not worth keeping.

After all P2/P3 work, add or update a short `## P2/P3 Implementation Audit Summary` before `## Tasks to delete, merge, or avoid` with counts by status, commands run, visual artifacts, and any deliberately deferred P3 items.
</plan_update_requirements>

<final_definition_of_done>
The run is complete only when:

- Every P2 task is `full` through the intended runtime/API path.
- Every P3 task is either `full` or explicitly `deferred`/deleted according to the plan's decision rule.
- Required native extension rebuilds, focused Python tests, Rust tests, `cargo forge3d-clippy`, and relevant GPU goldens have recorded outcomes.
- Visual changes have metrics and diff artifacts.
- Dead code listed by the plan is wired, deleted, or allowlisted with a reason.
- `git status --short` contains only intended implementation, test, docs, and golden changes.
</final_definition_of_done>

<final_response_format>
Return only:

- changed files grouped by slice
- P2/P3 task status table
- validation commands run and outcomes
- visual evidence/golden artifact paths
- deleted/deferred items and reasons
- skipped checks or blockers
```
