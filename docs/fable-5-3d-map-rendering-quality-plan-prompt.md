You are working in the local `forge3d` repository.

Objective:
Sweep through `src/`, `src/shaders/`, `python/`, and `examples/` to diagnose the current quality, architecture, and missing capabilities of forge3d's 3D map rendering stack. Think hard about how forge3d could become better than Blender specifically for high-end 3D map rendering quality, not for general 3D content creation. Write one accurate, surgically precise, strict, evidence-backed implementation plan and save it as:

`docs/3d-map-rendering-quality-blender-outmatch-plan.md`

This is a planning and diagnosis task only. Do not edit source code, tests, shaders, examples, manifests, fixtures, or existing docs other than the single markdown plan file above.

Operating rules:
- Inspect the actual repository state. Do not rely on memory.
- Start with `git status --short`; identify unrelated dirty files and leave them alone.
- Use `rg --files`, `rg`, and targeted file reads. Prefer reading the real call flow over broad summaries.
- Inspect at minimum:
  - Rust rendering architecture under `src/`
  - WGSL shaders under `src/shaders/`
  - Python public API and wrappers under `python/forge3d/`
  - Rendering examples under `examples/`
  - Existing relevant docs only when they clarify current intent, especially `docs/terrain/offline-render-quality.md`, `docs/superpowers/plans/3d-map-rendering-gaps-assessment.md`, and rendering/gallery docs.
- If you make a concrete claim about Blender's current rendering capabilities, verify it from official Blender documentation or clearly mark it as general background knowledge.
- Do not claim forge3d can "beat Blender" globally. The target is narrower: outmatch Blender on reproducible, geospatially aware, cartographic 3D map rendering quality.
- Do not ask me to choose priorities unless blocked by information only I can provide. Make a defensible recommendation.
- Do not expose hidden chain-of-thought. Give conclusions, evidence, rationale, and tradeoffs.

Primary quality axes to audit:
- Terrain fidelity: DEM ingest, clipmaps, displacement, normals, geomorphing, erosion/relief cues, terrain PBR, parallax/height detail, LOD, cracks, precision, and large-area streaming.
- Geospatial correctness: CRS handling, Earth curvature/geodesy where relevant, vertical datums, units, coordinate precision, tiled data alignment, reprojection, and metadata preservation.
- Materials: physically based material system, texture support, terrain splat/VT layers, albedo/normal/roughness/mask support, building materials, water, snow/ice/vegetation, thematic materials, and map-specific palettes.
- Lighting: sun/sky model, HDR/IBL, atmosphere, shadows, cascades, contact shadows, ambient occlusion, SSGI, SSR, path tracing, temporal stability, exposure, tone mapping, and color management.
- Cartography: labels, decluttering, halos, leader lines, scale bars, legends, north arrows, vector overlay antialiasing, line joins/caps, polygon fills, transparency/OIT, typography, and print/export quality.
- Scene content: buildings, CityJSON/OSM/glTF/3D Tiles, point clouds, instancing, vegetation/scatter, water/shorelines, clouds/smoke/fog, and animated/time-varying map layers.
- Offline quality: accumulation, AOVs, denoising, supersampling, reproducible camera paths, deterministic renders, visual regression tests, HDR/EXR/PNG output, and artifact-free exports.
- API ergonomics: Python APIs, `.pyi` stubs, examples, presets, recipe/scene manifests, and whether high-quality results are reachable without private knowledge.
- Performance and scalability: GPU memory, streaming, virtual texturing, batching, cache behavior, async loading, large datasets, tile traversal, and profiling hooks.

Audit method:
1. Build a quick map of rendering modules and public APIs.
2. Trace the end-to-end paths for terrain, scene rendering, offline rendering, shaders, labels/vector overlays, buildings/3D assets, point clouds, lighting/GI, clouds/volumetrics, and examples.
3. For each axis, classify current capability as:
   - `shipping`: implemented and reachable from Python/examples
   - `partial`: implemented internally but incomplete, weakly integrated, gated, undocumented, or missing tests
   - `stub`: placeholder API or fallback behavior only
   - `missing`: no meaningful implementation found
4. Record evidence with file paths and line numbers where possible. Prefer exact symbols/functions/modules over vague file references.
5. Identify root causes, not symptoms. Example: if a feature exists in Rust but not Python, call it an API exposure gap; if Python exists but examples cannot use it, call it an integration/example gap.
6. Compare against the map-rendering target, not generic Blender feature parity.
7. Turn the findings into a phased implementation plan that is realistic for this codebase.

Plan requirements:
- The markdown file must start with:
  - title
  - date
  - scope
  - one-paragraph executive summary
  - explicit statement of what "outmatch Blender for 3D map rendering quality" means
- Include a capability matrix with rows for the primary quality axes and columns:
  - current status
  - evidence
  - gap
  - target state
  - priority
- Include a ranked task plan. Every task must include:
  - stable task ID
  - priority (`P0`, `P1`, `P2`, `P3`)
  - title
  - description
  - why this change is needed
  - what needs to be coded
  - likely files/modules to modify
  - public API or shader changes, if any
  - tests/validation to add
  - definition of done
  - risks and dependencies
- Make tasks implementation-grade. Do not write vague tasks such as "improve lighting" or "add better materials." Say exactly what must be added or wired.
- Prefer high-leverage sequencing:
  - P0: unlocks visible quality or exposes existing hidden capability
  - P1: core rendering quality and map-specific differentiation
  - P2: scalability, advanced effects, polish, tooling
  - P3: speculative or expensive work
- Separate "quick wins from existing code" from "new rendering systems."
- Call out tasks that should be deleted, merged, or avoided if the repo already has enough machinery.
- Do not add new dependencies unless the plan explains why existing Rust/Python dependencies and platform APIs are insufficient.
- Do not invent support that is not in the repo. If unsure, say exactly what evidence is missing.

Minimum implementation themes to consider, but only include if evidence supports them:
- End-to-end high-quality map scene preset that wires terrain, sky, sun, shadows, AO/GI, tone mapping, labels, overlays, and offline output.
- Full virtual texture layer support beyond albedo if current runtime only pages albedo.
- Terrain material layering with normal/roughness/mask data and map-specific material presets.
- Deterministic cartographic label compiler and visual regression tests.
- Building/city asset material pipeline with textured PBR and CityJSON/OSM/glTF integration.
- Better atmospheric scattering, cloud shadows, fog, water/shorelines, snow/ice/vegetation cues where map scenes need them.
- Offline path tracing or hybrid accumulation improvements only if they integrate with terrain/maps and are testable.
- Golden image gallery and quality gates that prevent regression.
- Python recipe/preset APIs that make premium output reachable in one script.

Strict non-goals:
- No implementation during this run.
- No broad rewrite plan.
- No generic Blender clone.
- No UI/editor roadmap unless it directly improves reproducible 3D map render quality.
- No vague marketing claims.
- No tasks whose definition of done cannot be tested.

Verification before final response:
- Confirm the plan file exists at `docs/3d-map-rendering-quality-blender-outmatch-plan.md`.
- Run a markdown sanity check by reading the file back.
- Report only:
  - plan file path
  - number of tasks by priority
  - top 3 highest-leverage tasks
  - any skipped checks or blockers
