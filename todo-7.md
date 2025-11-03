I'll open the two referenced files to extract the requirements and the desired structure. I'll read `p7.md` to gather the requirements and `todo-6.md` to mirror its structure for the new task list.

To mirror the structure and reference specific files accurately, I'll quickly search the codebase to locate the `Renderer` class and relevant Python config/CLI entry points, and confirm if `examples/terrain_demo.py` already has CLI handling for `--preset`. This ensures the tasks reference correct paths and integration points.

I'll open [python/forge3d/__init__.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/__init__.py:0:0-0:0) to confirm where `Renderer` is defined and how to add `apply_preset`, and check [python/forge3d/config.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:0:0-0:0) for config schema to merge presets cleanly.

I'll scan key files to tailor the task list precisely:
- Open [python/forge3d/config.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:0:0-0:0) to understand the current config schema and how presets should merge.
- Check if `examples/terrain_demo.py` exists and whether a `--preset` flag is already wired.
- Search for any existing `preset` parsing to avoid duplicating work.

I'll open [examples/terrain_demo.py](cci:7://file:///Users/mpopovic3/forge3d/Users/mpopovic3/forge3d/examples/terrain_demo.py:0:0-0:0) to see current CLI options and whether a `--preset` flag exists, so I can reference precise integration points and avoid redundancy.



I'll search for the render module and supporting functions to choose the best primitives for the examples.

To tailor the engineering tasks to actual APIs and ensure realistic integration points, I'll open [python/forge3d/render.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/Users/mpopovic3/forge3d/python/forge3d/render.py:0:0-0:0) to see what rendering helpers exist for examples like lighting and shadow galleries.

# P7 implementation plan: high-level presets, validation, and examples

Below is a concrete execution plan to deliver P7 “Python UX polish” with high-level presets, clean merging into the existing config system, new example galleries, and docs. Tasks mirror the P6 structure in [todo-6.md](cci:7://file:///Users/mpopovic3/forge3d/todo-6.md:0:0-0:0) and reference current files (e.g., [python/forge3d/config.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:0:0-0:0), [python/forge3d/__init__.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/__init__.py:0:0-0:0), [examples/terrain_demo.py](cci:7://file:///Users/mpopovic3/forge3d/examples/terrain_demo.py:0:0-0:0), [python/forge3d/render.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/render.py:0:0-0:0), `docs/`).

## Milestone 1 — Presets API and config integration

- P7-01: `forge3d.presets` module scaffolding (High, 0.5–1 day)
  - Create `python/forge3d/presets.py` exporting:
    - `studio_pbr()` → directional key light + IBL preset, `shadows.technique="pcf"`, `shading.brdf="disney-principled"`, `gi.modes=["ibl"]`, `atmosphere.sky="hdri"` (no default HDR path; allow CLI to override).
    - `outdoor_sun()` → `atmosphere.sky="hosek-wilkie"`, primary directional “sun”, `shadows.technique="csm"`, `shading.brdf="cooktorrance-ggx"`.
    - `toon_viz()` → `shading.brdf="toon"`, `shadows.technique="hard"`, `gi.modes=[]` (no GI).
    - `get(name)`, `available()`, and case-insensitive resolution of preset names.
  - Ensure returned data maps cleanly to [RendererConfig.from_mapping()](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:190:4-211:19) structure (lighting/shading/shadows/gi/atmosphere).
  - Exit criteria: `from forge3d import presets; presets.get("outdoor_sun")` returns a valid mapping consumed by [RendererConfig.from_mapping()](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:190:4-211:19) without errors.

- P7-02: `Renderer.apply_preset(name, **overrides)` (High, 0.5 day)
  - Add `Renderer.apply_preset` to [python/forge3d/__init__.py::Renderer](cci:2://file:///Users/mpopovic3/forge3d/python/forge3d/__init__.py:388:0-497:32) to:
    - Resolve the preset via `forge3d.presets.get(name)`.
    - Merge preset into the current [RendererConfig](cci:2://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:320:0-418:19) instance using [RendererConfig.from_mapping(preset, current_cfg)](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:190:4-211:19) semantics.
    - Apply any `overrides` after the preset using existing normalization in [load_renderer_config(current_cfg, overrides)](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:477:0-494:14).
    - Call [_apply_config()](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/__init__.py:476:4-492:56) to refresh cached state.
  - Exit criteria: Calling `Renderer.apply_preset("toon_viz", brdf="toon")` updates internal config and affects subsequent renders; invalid preset raises helpful `ValueError`.

- P7-03: Terrain demo CLI `--preset` wiring (High, 0.5 day)
  - Update [examples/terrain_demo.py](cci:7://file:///Users/mpopovic3/forge3d/examples/terrain_demo.py:0:0-0:0):
    - Add `--preset <name>` to the CLI parser.
    - In [_build_renderer_config(args)](cci:1://file:///Users/mpopovic3/forge3d/examples/terrain_demo.py:196:0-226:48), if `args.preset` is provided:
      - Fetch preset mapping (`forge3d.presets.get(args.preset)`).
      - Apply preset first to form a base config, then merge existing CLI overrides (brdf, shadows, cascades, sky, hdr, volumetric, etc.).
    - Keep `--hdr` override compatible with both atmosphere.hdr_path and environment light.
  - Exit criteria: The one-liner acceptance works and overrides take precedence over preset defaults:
    - python examples/terrain_demo.py --preset outdoor_sun --brdf cooktorrance-ggx --shadows csm --cascades 4 --hdr assets/sky.hdr

- P7-04: Unit tests for presets and merging (High, 0.5–1 day)
  - Add `tests/test_presets.py`:
    - Validate each preset returns a mapping accepted by [RendererConfig.from_mapping()](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:190:4-211:19).
    - Validate expected fields: e.g., `outdoor_sun()` sets `atmosphere.sky="hosek-wilkie"`, `shadows.technique="csm"`, `shading.brdf="cooktorrance-ggx"`.
  - Add `tests/test_renderer_apply_preset.py`:
    - `Renderer.apply_preset` applies settings and respects overrides (override wins).
  - Exit criteria: Tests pass on CI; invalid preset names raise clean errors.

Notes
- Keep Python API import-safe in CPU-only environments (no hard dependency on native symbols).
- Don’t hardcode HDR files in presets; let CLI `--hdr` or environment light provide path.

## Milestone 2 — Example galleries

- P7-05: `examples/lighting_gallery.py` (Medium, 1 day)
  - Render a grid that sweeps BRDFs and basic light setups (directional, environment IBL) using high-level helpers from [python/forge3d/render.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/render.py:0:0-0:0) or the terrain path to keep consistent with repo assets.
  - Support small sizes for fast CI (e.g., 512×512 tiles) and an `--outdir` parameter.
  - Exit criteria: Script runs locally without GPU dependencies hard-failing; saves a grid image and/or per-cell images.

- P7-06: `examples/shadow_gallery.py` (Medium, 1 day)
  - Side-by-side comparison of `hard`, `pcf`, `pcss`, `vsm`, `evsm`, `msm`, `csm` at the same camera/light.
  - Overlays labels per panel; include `--size` and `--outdir` flags.
  - Exit criteria: Produces a labeled mosaic image demonstrating each technique’s look.

- P7-07: `examples/ibl_gallery.py` (Medium, 1 day)
  - HDRI rotation sweep and a roughness sweep (vary `shading.roughness`).
  - Use `IBL.from_hdr` with repository HDR asset; vary `rotation` and `roughness` in a grid.
  - Exit criteria: Two mosaics (rotation and roughness) saved; runs with current assets.

Notes
- Prefer synthetic or existing assets under `assets/` to avoid adding large files.
- Ensure examples degrade gracefully if GPU features are unavailable.

## Milestone 3 — Docs and how-to pages

- P7-08: Presets how-to pages (High, 0.5–1 day)
  - Add Sphinx docs pages (e.g., `docs/user/presets_overview.rst`, `docs/user/preset_studio_pbr.rst`, `docs/user/preset_outdoor_sun.rst`, `docs/user/preset_toon_viz.rst`).
  - Include code snippets (Python and CLI) and small thumbnails from galleries.
  - Exit criteria: `make html` builds successfully; presets pages linked from the main user docs index.

- P7-09: Example galleries docs (Medium, 0.5 day)
  - Short how-tos referencing `examples/lighting_gallery.py`, `examples/shadow_gallery.py`, `examples/ibl_gallery.py`.
  - Include CLI flags and tips for reproducibility.
  - Exit criteria: Gallery pages build; navigation link added.

## Milestone 4 — Acceptance, tests, and CI wiring

- P7-10: Acceptance one-liner test and goldens (High, 0.5–1 day)
  - Add a small integration test that constructs [_build_renderer_config()](cci:1://file:///Users/mpopovic3/forge3d/examples/terrain_demo.py:196:0-226:48) with `--preset outdoor_sun` + overrides and asserts expected config fields.
  - Optionally record tiny golden(s) under `tests/golden/p7/` for at least one preset configuration (skip when GPU is unavailable).
  - Exit criteria: CI asserts “one command” acceptance is reproducible; goldens stable.

- P7-11: Lint/style and imports (Low, 0.25 day)
  - Ensure `python/forge3d/presets.py` is covered by linters configured in `pyproject.toml`.
  - Add top-level re-exports if desired: `from . import presets` in [python/forge3d/__init__.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/__init__.py:0:0-0:0) for discoverability.
  - Exit criteria: CI lint passes; import paths are stable.

---

## Recommended implementation order

1) P7‑01 presets module → P7‑02 `Renderer.apply_preset` → P7‑03 CLI `--preset`  
2) P7‑04 unit tests for presets/merge  
3) P7‑05/06/07 example galleries  
4) P7‑08/09 docs pages  
5) P7‑10 acceptance + goldens → P7‑11 lint/style

Rationale
- Presets + merge API first to unblock CLI and examples.
- Examples next provide the visual content for docs and acceptance.
- Docs trail implementation; tests/goldens stabilize CI and regressions.

## Estimates summary (single engineer; add 30–40% overlap for review/CI)

- P7‑01: 0.5–1 day  
- P7‑02: 0.5 day  
- P7‑03: 0.5 day  
- P7‑04: 0.5–1 day  
- P7‑05: 1 day  
- P7‑06: 1 day  
- P7‑07: 1 day  
- P7‑08: 0.5–1 day  
- P7‑09: 0.5 day  
- P7‑10: 0.5–1 day  
- P7‑11: 0.25 day

Base “ship”: ~6–9 days

## Risks and mitigations

- Hosek/Preetham vs HDRI interactions: Ensure preset doesn’t require assets that may be missing at runtime. Mitigation: presets do not hardcode HDR paths; CLI `--hdr` or environment light can supply one; add clear error messages in validation.
- Config merge precedence: Preset defaults must never override explicit CLI flags. Mitigation: apply preset first, then merge CLI overrides; add tests.
- GPU availability: Gallery scripts should not hard-fail on CPU-only machines. Mitigation: keep fallbacks or skip heavy paths; assert shapes and write small images.

## Deliverables per milestone

- M1: `python/forge3d/presets.py`; `Renderer.apply_preset`; terrain demo `--preset` integrated; unit tests passing.  
- M2: `examples/lighting_gallery.py`, `examples/shadow_gallery.py`, `examples/ibl_gallery.py` producing mosaics.  
- M3: Sphinx how-to pages for each preset + gallery documentation.  
- M4: Acceptance one-liner validated in CI; optional small goldens; lint/style clean.

## File references

- Presets and merging
  - `python/forge3d/presets.py` (new)
  - [python/forge3d/__init__.py::Renderer](cci:2://file:///Users/mpopovic3/forge3d/python/forge3d/__init__.py:388:0-497:32) (add `apply_preset`)
  - [python/forge3d/config.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:0:0-0:0) (existing merge/validation helpers)
- CLI integration
  - [examples/terrain_demo.py](cci:7://file:///Users/mpopovic3/forge3d/examples/terrain_demo.py:0:0-0:0) (add `--preset`, merge order)
- Examples and helpers
  - [python/forge3d/render.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/render.py:0:0-0:0) (helpers usable in galleries)
- Docs
  - `docs/user/` (new preset and gallery pages)
- Tests
  - `tests/test_presets.py`, `tests/test_renderer_apply_preset.py`, small integration under `tests/` + `tests/golden/p7/` as applicable

If you want, I can draft the `python/forge3d/presets.py` skeleton and the `Renderer.apply_preset` method next, followed by the `--preset` CLI wiring in [examples/terrain_demo.py](cci:7://file:///Users/mpopovic3/forge3d/examples/terrain_demo.py:0:0-0:0).