You are ChatGPT 5.1 (high reasoning), acting as a **coding agent inside the `forge3d` repository** (Rust + Python, with PyO3 bindings).  
Your work must strictly follow the repository rules and architecture guidance laid out in [AGENTS.md](cci:7://file:///Users/mpopovic3/forge3d/AGENTS.md:0:0-0:0).

#### 0. Global Constraints (from [AGENTS.md](cci:7://file:///Users/mpopovic3/forge3d/AGENTS.md:0:0-0:0))

You MUST obey all of the following:

- **Rust crate first, Python package second.**
  - Rust `src/` is the source of truth; Python `python/forge3d/` is a façade that must stay aligned (e.g. [RendererConfig](cci:2://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:481:0-579:19) vs `src/render/params.rs`).

- **Tests + docs define behavior.**
  - Never silently change semantics without adding/adjusting tests in `tests/` and updating Sphinx docs in `docs/`.

- **Clean code discipline.**
  - Small, single‑responsibility functions.
  - Clear, intention‑revealing names.
  - No speculative flexibility (YAGNI).
  - Refactor in **tiny, safe steps** with tests green at each step.

- **Memory & GPU budgets are strict.**
  - Respect the **shadow atlas budget** and the **512 MiB host‑visible memory budget**.
  - Do not introduce hidden large allocations in Python paths.

- **Prompt‑driven, but not design‑free.**
  - Treat AI‑generated code as immature; reason about correctness, testability, and architecture.

You must **read and respect**:

- [AGENTS.md](cci:7://file:///Users/mpopovic3/forge3d/AGENTS.md:0:0-0:0)
- [python/forge3d/__init__.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/__init__.py:0:0-0:0)
- [python/forge3d/config.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:0:0-0:0)
- [python/forge3d/presets.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/presets.py:0:0-0:0)
- [examples/terrain_demo.py](cci:7://file:///Users/mpopovic3/forge3d/examples/Users/mpopovic3/forge3d/examples/terrain_demo.py:0:0-0:0)
- [examples/lighting_gallery.py](cci:7://file:///Users/mpopovic3/forge3d/examples/lighting_gallery.py:0:0-0:0)
- [examples/shadow_gallery.py](cci:7://file:///Users/mpopovic3/forge3d/examples/shadow_gallery.py:0:0-0:0)
- [examples/ibl_gallery.py](cci:7://file:///Users/mpopovic3/forge3d/examples/ibl_gallery.py:0:0-0:0)
- Relevant tests under `tests/` that touch renderer config, presets, and examples.

---

### 1. Objective (P7)

Implement and harden **Python UX polish for rendering presets & examples (P7)**:

- Provide **friendly high‑level presets** while keeping low‑level flags.
- Ensure users can reproduce canonical figures and demos with **one command** (e.g. `terrain_demo.py --preset outdoor_sun ...`).
- Provide lighting/shadow/IBL galleries that align with the rest of the system and degrade gracefully on CPU‑only/no‑GPU environments.

---

### 2. Functional Requirements

Use these as **non‑negotiable acceptance criteria**.

#### 2.1 `forge3d.presets` module

You MUST ensure [python/forge3d/presets.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/presets.py:0:0-0:0) satisfies:

- **R1 – Preset functions**
  - Export:
    - [studio_pbr() -> dict](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/presets.py:52:0-85:5)
    - [outdoor_sun() -> dict](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/presets.py:88:0-117:5)
    - [toon_viz() -> dict](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/presets.py:120:0-145:5)
  - Each returns a mapping compatible with [RendererConfig.from_mapping()](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:548:4-579:19):
    - Uses only `lighting`, `shading`, `shadows`, `gi`, `atmosphere`, optional `brdf_override`.
    - Values must match the expectations and validation rules in [config.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:0:0-0:0).

- **R2 – Preset intent and shape**
  - [studio_pbr](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/presets.py:52:0-85:5):
    - Disney principled BRDF: `brdf="disney-principled"`.
    - One studio directional key light (slightly warm).
    - `shadows.technique="pcf"`, `map_size` power of two, `cascades=1`.
    - `gi.modes` is empty by default.
    - `atmosphere.enabled=False`.
  - [outdoor_sun](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/presets.py:88:0-117:5):
    - GGX BRDF: `brdf="cooktorrance-ggx"`.
    - Directional “sun” light with reasonable intensity.
    - `shadows.technique="csm"`, `cascades` in `[2,4]`, `map_size` power of two and within [atlas_memory_bytes](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:446:4-448:89) budget.
    - `atmosphere.enabled=True`, `sky="hosek-wilkie"`.
    - `gi.modes` empty by default; user can enable `["ibl"]` when they add HDR.
  - [toon_viz](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/presets.py:120:0-145:5):
    - `brdf="toon"`, `normal_maps=False`.
    - Hard shadows: `technique="hard"`, moderate `map_size` (e.g. 1024), `cascades=1`.
    - `atmosphere.enabled=False`, `gi.modes` empty.

- **R3 – Registry & lookup**
  - `_PRESETS` and `_ALIASES` must allow:
    - [presets.available() -> list[str]](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/presets.py:167:0-169:34) returning sorted canonical names (`"studiopbr"`, `"outdoorsun"`, `"toonviz"`).
    - [presets.get(name: str) -> dict](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/presets.py:172:0-188:20):
      - Name normalization: case/space/hyphen/underscore‑insensitive.
      - Alias resolution: e.g. `"studio"`, `"pbr"`, `"outdoor"`, `"sun"`, `"toon"`.
      - Raises a clear `ValueError` for unknown names, listing available presets.
      - Returns a **copy**, not shared mutable state.

- **R4 – Validation safety**
  - For every preset:
    ```python
    from forge3d.config import RendererConfig
    cfg = RendererConfig.from_mapping(presets.get(name))
    cfg.validate()
    ```
    must succeed under default constraints.
  - No preset may:
    - Break [ShadowParams](cci:2://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:395:0-448:89) validation (map size, cascades, moment bias, atlas bytes).
    - Configure GI IBL (`gi.modes` containing `"ibl"`) without an HDR path reachable by [RendererConfig.validate()](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:505:4-546:113) (either `atmosphere.hdr_path` or an environment light with `hdr_path`).

#### 2.2 [Renderer.apply_preset](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/__init__.py:477:4-517:28)

In [python/forge3d/__init__.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/__init__.py:0:0-0:0), the [Renderer](cci:2://file:///Users/mpopovic3/forge3d/python/forge3d/__init__.py:393:0-576:32) class MUST satisfy:

- **R5 – API**
  - Method present:
    ```python
    def apply_preset(self, name: str, **overrides) -> None:
        ...
    ```

- **R6 – Merge semantics**
  - Merge order is **exactly**:
    1. Current `self._config` (a [RendererConfig](cci:2://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:481:0-579:19)).
    2. Preset: [presets.get(name)](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/presets.py:172:0-188:20).
    3. Overrides supplied by the caller.
  - Overrides can be:
    - **Nested**:
      - Keys: `lighting`, `shading`, `shadows`, `gi`, `atmosphere`, `brdf_override`.
      - Values: mappings compatible with [RendererConfig.from_mapping()](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:548:4-579:19).
    - **Flat**:
      - Any keys recognized by [split_renderer_overrides](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:666:0-705:31) (`brdf`, `shadows`, `shadow_map_res`, `cascades`, `sky`, `hdr`, `gi`, etc.).
  - Implementation pattern (conceptual):
    - [preset_map = presets.get(name)](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/presets.py:172:0-188:20)
    - [cfg = RendererConfig.from_mapping(preset_map, self._config)](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:462:4-478:19)
    - Partition `overrides` into nested vs flat (`nested_keys = {"lighting", "shading", ...}`).
    - If `nested`: [cfg = RendererConfig.from_mapping(nested, cfg)](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:462:4-478:19).
    - Then `cfg = load_renderer_config(cfg, flat or None)` to apply & validate.
    - Commit only if validation passes:
      - `self._config = cfg`
      - [self._apply_config()](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/__init__.py:555:4-571:56) to refresh cached lighting/shading state.

- **R7 – Error behavior**
  - Unknown preset name:
    - [presets.get](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/presets.py:172:0-188:20) raises `ValueError`; [apply_preset](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/__init__.py:477:4-517:28) must **not** mutate `self._config`.
  - Malformed overrides:
    - Let [RendererConfig.validate()](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:505:4-546:113) or normalization raise (e.g., invalid `cascades`, `map_size`).
    - On failure, `self._config` and cached state must remain unchanged.

#### 2.3 CLI & Examples

- **R8 – Terrain preset CLI acceptance**
  - [examples/terrain_demo.py](cci:7://file:///Users/mpopovic3/forge3d/examples/Users/mpopovic3/forge3d/examples/terrain_demo.py:0:0-0:0) must accept:
    - `--preset <name>` where `<name>` is any valid preset or alias.
    - At least the overrides:
      - `--brdf`, `--shadows`, `--cascades`, `--hdr`.
  - For the acceptance command (from P7 spec):
    ```bash
    python examples/terrain_demo.py --preset outdoor_sun \
      --brdf cooktorrance-ggx --shadows csm --cascades 4 --hdr assets/sky.hdr
    ```
    you must ensure that:
    - The combined config is:
      - Based on [outdoor_sun](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/presets.py:88:0-117:5).
      - Has `shading.brdf == "cooktorrance-ggx"`.
      - Has `shadows.technique == "csm"` and `shadows.cascades == 4`.
      - Satisfies all validation rules regarding HDR/IBL (if GI is enabled).
    - The demo runs without config validation errors on a correctly set up machine.

- **R9 – [lighting_gallery.py](cci:7://file:///Users/mpopovic3/forge3d/examples/lighting_gallery.py:0:0-0:0)**
  - Keep its current shape (mesh path tracer, variants list).
  - Guarantee:
    - It uses `_import_shim.ensure_repo_import()`.
    - It degrades gracefully:
      - On `forge3d` import failure: prints diagnostics and exits with a clear message.
      - On per‑variant failure: logs the failure and falls back to [_placeholder_tile()](cci:1://file:///Users/mpopovic3/forge3d/examples/lighting_gallery.py:140:0-151:35).

- **R10 – [shadow_gallery.py](cci:7://file:///Users/mpopovic3/forge3d/examples/shadow_gallery.py:0:0-0:0)**
  - Ensure:
    - `SHADOW_TECHNIQUES` remains the single source of technique names and descriptions.
    - [_build_params_for_tech()](cci:1://file:///Users/mpopovic3/forge3d/examples/shadow_gallery.py:110:0-170:46) constructs `TerrainRenderParamsConfig` using `*Settings` types from `terrain_params.py` in a way consistent with their validation.
    - When GPU/native is unavailable or types are missing, the script uses labeled placeholder tiles and still completes successfully.

- **R11 – [ibl_gallery.py](cci:7://file:///Users/mpopovic3/forge3d/examples/ibl_gallery.py:0:0-0:0)**
  - Maintain the M4 IBL CPU pipeline:
    - Uses `m4_generate.py` helpers to build IBL resources.
    - Requires BRDF tile renderer availability ([has_gpu()](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/__init__.py:724:0-726:25) and [render_brdf_tile_full](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/__init__.py:1659:0-1661:87)).
  - Fail **early and clearly** if:
    - `m4_generate.py` is missing, or
    - The BRDF tile renderer isn’t available.

#### 2.4 Documentation

- **R12 – Sphinx docs**
  - Add or update Sphinx docs to:
    - Describe `forge3d.presets`, listing:
      - [studio_pbr](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/presets.py:52:0-85:5), [outdoor_sun](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/presets.py:88:0-117:5), [toon_viz](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/presets.py:120:0-145:5), their intent, and typical usage.
    - Document [Renderer.apply_preset](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/__init__.py:477:4-517:28) and its merge semantics.
    - Provide CLI snippets for:
      - `terrain_demo.py --preset outdoor_sun ...` (one‑command reproduction).
      - [lighting_gallery.py](cci:7://file:///Users/mpopovic3/forge3d/examples/lighting_gallery.py:0:0-0:0), [shadow_gallery.py](cci:7://file:///Users/mpopovic3/forge3d/examples/shadow_gallery.py:0:0-0:0), [ibl_gallery.py](cci:7://file:///Users/mpopovic3/forge3d/examples/ibl_gallery.py:0:0-0:0).

---

### 3. Non‑Functional Requirements

You MUST:

- Keep all changes **small and test‑driven**:
  - Never leave the test suite red for more than a few minutes.
  - Distinguish structural refactoring from feature changes.

- Maintain **backwards compatibility** unless a test/spec explicitly allows a change.

- Maintain or improve **code clarity** per [AGENTS.md](cci:7://file:///Users/mpopovic3/forge3d/AGENTS.md:0:0-0:0):
  - Short functions, clear names, one level of abstraction per function.
  - No “clever” but opaque logic; prefer explicitly named helpers.

---

### 4. Milestones & Deliverables

You MUST organize your work into the following milestones, committing code and tests that pass at each milestone.

1. **Milestone 1 – Analysis & Baseline**
   - Verify current behavior of:
     - [presets.available()](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/presets.py:167:0-169:34), [presets.get()](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/presets.py:172:0-188:20).
     - [Renderer.apply_preset](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/__init__.py:477:4-517:28) (if present).
     - [terrain_demo.py](cci:7://file:///Users/mpopovic3/forge3d/examples/Users/mpopovic3/forge3d/examples/terrain_demo.py:0:0-0:0) handling of any existing config flags.
   - Add a minimal test that:
     - Iterates over current presets and checks [RendererConfig.from_mapping(...).validate()](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:548:4-579:19) passes (or documents why not, if currently impossible).

2. **Milestone 2 – Preset Contract Hardening**
   - Adjust [python/forge3d/presets.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/presets.py:0:0-0:0) to fully satisfy R1–R4.
   - Add Python tests under `tests/` to:
     - Confirm names & aliases resolve correctly.
     - Confirm each preset validates independently.
     - Confirm unknown names raise the correct `ValueError`.

3. **Milestone 3 – [Renderer.apply_preset](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/__init__.py:477:4-517:28) Semantics**
   - Implement or refine [Renderer.apply_preset](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/__init__.py:477:4-517:28) to conform to R5–R7.
   - Add tests covering:
     - Merge order with a simple scenario (baseline config → preset → overrides).
     - Nested vs flat overrides.
     - Error isolation (invalid overrides do not mutate `self._config`).

4. **Milestone 4 – Terrain Demo Preset CLI**
   - Extend [examples/terrain_demo.py](cci:7://file:///Users/mpopovic3/forge3d/examples/Users/mpopovic3/forge3d/examples/terrain_demo.py:0:0-0:0) to wire `--preset` and low‑level overrides into the same config logic as [Renderer.apply_preset](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/__init__.py:477:4-517:28).
   - Add at least one test (can be a small, environment‑skippable test) or script that:
     - Asserts the acceptance command from [p7.md](cci:7://file:///Users/mpopovic3/forge3d/p7.md:0:0-0:0) produces a valid [RendererConfig](cci:2://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:481:0-579:19) (no validation errors).

5. **Milestone 5 – Gallery Examples QA**
   - Smoke‑test:
     - [lighting_gallery.py](cci:7://file:///Users/mpopovic3/forge3d/examples/lighting_gallery.py:0:0-0:0)
     - [shadow_gallery.py](cci:7://file:///Users/mpopovic3/forge3d/examples/shadow_gallery.py:0:0-0:0)
     - [ibl_gallery.py](cci:7://file:///Users/mpopovic3/forge3d/examples/ibl_gallery.py:0:0-0:0)
   - Add minimal tests or scripted checks (skipped if GPU/native unavailable) that:
     - Run each script with tiny tile sizes and confirm an output file (PNG or NPY) exists.
     - Confirm graceful failure modes where appropriate.

6. **Milestone 6 – Documentation & Finalization**
   - Update Sphinx docs to fulfill R12.
   - Run full test suite + docs build.
   - Produce a short summary (commit message or doc note) explaining:
     - How presets work,
     - How [apply_preset](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/__init__.py:477:4-517:28) merges,
     - How to reproduce the P7 acceptance example.

---


Assess the current level of implementation of Phase 5 from todo-6.md. If the requirements are met, do nothing. If the requirements are not met, think very hard to turn these into an extremely surgically precise and strict prompt for ChatGPT 5.1 (high reasoning) to accomplish the missing requirements

you must read fully @AGENTS.md to get familiar with my codebase. Next, you must carefully read @todo-5.md as a whole. Then you must fully implement P5.8. Test after every change. Do not stop until you meet all the requirements for P5.8

you must read fully @AGENTS.md  to get familiar with my codebase. Next, you must carefully read @p6.md as a whole. Then you must design an extremely surgically precise and accurate and specific set of requirements for ChatGPT 5.1. (high reasoning)  with coherent milestones and clear deliverables