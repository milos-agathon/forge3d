# FORGE3D SKILLSET — The Definitive Skill Library Design

> Produced 2026-07-09 by evidence audit of: 31 session transcripts (`C:\Users\milos\.claude\projects\C--Users-milos-forge3d\*.jsonl`, ~75 MB main sessions + subagent/workflow logs), `AGENTS.md` (7 reflection campaigns), 9 memory files, `CLAUDE.md`, `.claude/rules/*.md` (6 files), the live codebase surface, and all 826 commits (`git log`). Every skill below is traceable to ≥2 independent evidence occurrences; citations use `[sessionid]` = transcript file prefix, `sha` = commit, or `file:line`.

---

## 1. EXECUTIVE SUMMARY

Six deep skills replace ~80% of routine forge3d work, because nearly every session is one of three motions — **wire a native feature through the PyO3 bridge**, **push a rendering capability through wgpu/WGSL**, or **defend a gate** (golden image, allocation budget, CI) — and all three funnel through one chokepoint: the **rebuild-and-prove loop** (`maturin develop` appears 567 times across 113 transcript files; its failure modes — stale `.pyd`, wrong interpreter, wrong build profile — are the top recorded time-wasters). The set is: (1) **forge3d-rebuild-verify**, the loop itself with its five documented traps; (2) **forge3d-native-symbol**, the 7-site registration chain that AGENTS.md P0.2/P0.3/P0.4/P2.3 each relearned separately; (3) **forge3d-wgsl-contract**, the WGSL↔Rust stride/bind-group/format contract whose violations surface only at first GPU dispatch; (4) **forge3d-golden-gates**, the regen/wobble/provenance workflow for six golden directories and five regen env vars; (5) **forge3d-gpu-test-gating**, making GPU-touching tests actually run locally and skip honestly in CI; (6) **forge3d-tracked-allocation**, the CENSOR-era allocation discipline now enforced by a source-level gate. Not more: spec-writing, TDD, debugging, and worktrees are already covered by global skills — these six encode only what is forge3d-specific.

---

## 2. EVIDENCE LEDGER

Frequency-ranked archetypes of routine/painful work, with citations. Counts are ripgrep hits across the transcript corpus measured 2026-07-09.

### A1. The rebuild-and-prove loop (stale builds, wrong python, wrong profile) — **highest frequency**
The single most repeated activity and the most-relearned failure cluster.
- **Frequency:** `maturin develop` — 567 hits / 113 transcript files (top: `[c10ba4ed]`:68, `[bd16a6f0]`:29, `[8c94e77c]`:29, `[c2b543e3]`:26). `cargo forge3d-clippy` — 516 hits / 95 files. Curated `cargo test … --skip gpu_extrusion` — 147 hits / 58 files. CLAUDE.md's first IMPORTANT rule is this loop — it exists because it was violated.
- `[69571db4]` (BOP audit): "The recurring closure gaps: the stale compiled `.pyd` (Rust/WGSL changes source-only), goldens generated through a forced-placeholder monkeypatch instead of the GPU path…" — stale-build gaps recur enough to headline an audit; the same audit found the shadowed broken `pyproj` at `D:\forge3d\python\pyproj` made a golden gate unrunnable.
- Memory `project_python_env_venv` (origin `[8c94e77c]`): bare `python` on PATH is r-miniconda with a stale editable install (`forge3d.pth` → `D:\forge3d\python`) — "`hasattr(forge3d, new_symbol)` returns False and old code runs. **Why:** Cost a debugging round on 2026-07-07."
- Memory `project_debug_build_gpu_probe_crash` (origin `[03834849]`, literal crash log in-transcript at `tests/_terrain_runtime.py`): "dev-profile `maturin develop` builds crash the GPU probe on this machine (access violation); use `--release`". Sibling memory `feedback_pipe_tail_masks_failures` (origin `[583bee8a]`, PROMETHEUS: "bit this session twice"; live `| tail -40` background pytest in `[8148bb44]`) covers the masked-exit-code trap in the same loop.

### A2. Native symbol registration — the multi-site chain
- **Frequency:** `EXPECTED_FUNCTIONS|EXPECTED_CLASSES` — 254 hits / 54 files (top: `[03834849]`:22, `[8c94e77c]`:21, `[0adf4a9b]`:18, `[29995b0b]`:18).
- AGENTS.md P0.3: "a `#[pyclass]` with no `m.add_class::<T>()` … is invisible to Python even though it compiles fine." P0.4: feature-gated Rust absent from `pyproject.toml [tool.maturin].features` "compiles but the functions are excluded from the extension." P0.2: dead `hasattr(_native, "render_rgba")` probe persisted silently. P2.3: the `PyLabelStyle` binding pattern.
- `[25f8ddb3]`: a completed chain in one quote — "New native symbols — `seal_provenance`, `verify_provenance` (registered in `src/py_module/functions/provenance.rs`, re-exported in `__all__`, in `EXPECTED_FUNCTIONS`/`EXPECTED_PACKAGE_ATTRS`, stubbed in the `.pyi`)".
- Live surface verified today: `tests/test_api_contracts.py` — `EXPECTED_CLASSES` at :61, `EXPECTED_FUNCTIONS` at :115, `EXPECTED_PACKAGE_ATTRS` at :695; `pyproject.toml:109` features list; `.claude/rules/python-api.md` names the four hand-edited sites and flags `__init__.pyi` as KNOWN-DRIFTED.

### A3. WGSL↔Rust GPU contract violations (stride, bind groups, formats)
- **Frequency:** wgpu-validation fingerprints (`Buffer is bound with size|max_bind_groups|shader expects…`) — 116 hits / 34 files (top: `[583bee8a]`:27, `[8c94e77c]`:12, `[f7ec505b]`:7).
- `[583bee8a]` (PROMETHEUS), verbatim wgpu error: "Buffer is bound with size 4 where the shader expects 16 in group[1] compact index 2". AGENTS.md PROMETHEUS: ad-hoc `TerrainReservoir` had "an 80-byte WGSL stride against a 64-byte Rust allocation… verify the STRIDE, not the field list"; and "when a shader comment claims 'consolidated to stay within max_bind_groups=4', count the actual pipeline-layout groups — the claim had drifted."
- `[ecbc2c0c]`: "`R8Unorm` vs `rgba8unorm` vs `Rgba8Unorm` will trigger a wgpu validation error at pipeline or bind-group creation. The visibility-AOV path tracer is broken on any real device and nothing in CI catches [it]."
- `[f7ec505b]`: "the terrain pipeline silently swaps to `terrain_minimal.wgsl` when `max_bind_groups < 6`, dropping cloud shadows and planar reflections with zero diagnostic."
- Live surface: `src/shaders/terrain.wgsl:5` — "Globals UBO (176 bytes total, must match Rust)"; 124 `.wgsl` files; `.claude/rules/shaders.md`: validation happens only at pipeline build after `maturin develop`, never at `cargo build`.

### A4. Golden-image / adjudication churn
- **Frequency:** `golden` — >1,000 raw hits (top main sessions: `[3c57e35d]`:103, `[127f97ba]`:47, `[4393a059]`:45); regen-specific cluster tops: `[127f97ba]`:12, `[69571db4]`:11.
- `[69571db4]`: one red gate re-run across an entire audit — "`mapscene_furniture_graticule` SSIM 0.814600 < 0.995 … **SSIM 0.814600 identical on both runs**" (deterministic failure chased repeatedly).
- Memory `project_adjudication_gate_workflow` (origin `[c2b543e3]`): regen via `FORGE3D_UPDATE_ADJUDICATION_GOLDENS=1`, ~50–95 s full gate, "PT reference has a small run-to-run wobble (~0.01 pp…) — don't chase byte-identical PT reruns", the spp=1 raster-parity iteration trick, and the transient `TerrainRenderer` segfault that silently skips the gate.
- AGENTS.md BOP-P2-02: "Ten P2 recipe goldens … were explicitly listed in `.gitignore` — they existed only on this machine while the plan claimed 'intended baselines are committed'. Check `git ls-files`, not just the working tree."
- Live surface: 5 regen env vars in `tests/` (`FORGE3D_UPDATE_TERRAIN_GOLDENS` ×4 sites, `_RECIPE_`, `_PROVENANCE_FIXTURE`, `_HYBRID_TERRAIN_`, `_ADJUDICATION_` ×2 each); 6 golden dirs (`tests/golden/{adjudication,hybrid_terrain,labels,presets,recipes,terrain}`, `tests/goldens/determinism`).

### A5. GPU-gated tests that silently don't run (or panic in CI)
- Memory `project_curated-cargo-test-skips-renderer` (origin `[584481c7]`): the curated `cargo test` lacks `extension-module`, so unit tests inside `src/terrain/renderer/**` are "silently skipped by that gate (0 tests, no error)."
- AGENTS.md BOP-P2-02: "`cargo test <filter> --lib` without `--features extension-module` silently skips the entire `src/terrain/renderer/` tree. A run that reports 'N passed, hundreds filtered out' can still mean *your new tests never ran*."
- Commit cluster: `cade5179` ("24 tests across 11 source files call GPU device initialization … panic with exit code 101" on GPU-less runners), `2583c7fe` (GPU helpers → `Option` + graceful skip), `cf0f5c0d` (`--all-features` pulls `pyo3/extension-module` into `cargo check` — replaced with the curated list), plus `ab635915`, `b9c5ae2f`, `eb042334`, `e13118f6`, `7a2316c2`, `a8649856`.
- `[ee174b8b]`: Vulkan `TerrainRenderer` construction segfault bisect ("Windows fatal exception: access violation … exit=139"); `[c2b543e3]`: the same transient segfault made the adjudication gate "silently skip via its availability probe; an immediate retry succeeded."

### A6. Memory-budget / allocation-tracking discipline (CENSOR)
- **Frequency:** `tracked_create_buffer|MemoryBudget|host_visible_bytes` — 1,056 hits / 81 files, concentrated in `[c10ba4ed]` (77 in main + 20–75 in each of ~30 batch subagents — the CENSOR campaign).
- Commit series (10 commits, 2026-07-08/09): `fe0327f2` (allocation ledger, tracked wrappers, enforce-by-default budget) → `fde3323b`/`f7a069bd`/`bfc4992b`/`798c6abf`/`9771335b` (route each subtree through wrappers) → `8c5ef8da` ("zero raw sites") → `2e11f2c0` (source-level gate) → `f488592d`/`fb6d746b` (fallbacks recorded as degradations).
- AGENTS.md PROMETHEUS: "wrap every tracked allocation in a Drop guard (`TrackedGpu`) … so `?`/early-return paths cannot leak tracker state. Gate the budget on tracker metrics captured AFTER all allocations."
- Live surface verified: `src/core/resource_tracker.rs` — `tracked_create_buffer`:343, `tracked_create_buffer_init`:375, `tracked_create_texture`:407; `tests/test_allocation_gate.py` + `tests/allocation_allowlist.toml`; `.claude/rules/rust-core.md` (512 MiB budget, ENFORCE default since 2026-07-09).

### A7. (Sub-archetype, folded) Worktree friction
- Memory `project_nested_worktree_clippy_alias` (origin `[c10ba4ed]`): `cargo forge3d-clippy` fails with "Unrecognized option: 'workspace'" in worktrees nested inside the repo (duplicate `.cargo/config.toml` alias merge), forcing `--no-verify` commits; fix = worktrees under `C:/tmp` or the expanded clippy command.
- Memory `feedback_worktree_vs_main`: reviewing stale `.worktrees/epic-24` produced two false bug reports against `main`.
- **Disposition:** folded as pitfalls into forge3d-rebuild-verify (not a standalone skill — the decision logic is two rules, and `superpowers:using-git-worktrees` covers the generic mechanics).

### A8. (Rejected) Spec → plan → implement → review cycle
- Evidence exists (commit prefix clusters: `docs(tv13)` ×8, `docs(tv11)` ×8, `docs(tv22)` ×5, `docs(tv20)` ×5, each spec followed by "fix spec review findings" / "fix plan review findings" commits), but the workflow is already fully covered by global skills (`superpowers:brainstorming`, `superpowers:writing-plans`, `superpowers:executing-plans`, `tdd`). The forge3d-specific residue (verification commands, gate lists) lives inside the six skills below. **Rejected per Gate 4 (ORIGINAL).**

---

## 3. THE SKILLSET (ranked by leverage)

| Rank | Skill name | Axis(es) | Trigger phrase | What it automates | Evidence count | Est. uses/week | Failure-mode it prevents |
|---|---|---|---|---|---|---|---|
| 1 | `forge3d-rebuild-verify` | DEVELOP + FIX | "I edited Rust/WGSL, now test it" / "results look stale" | The edit→build→prove loop: profile choice, interpreter choice, freshness proof, background-run hygiene, correct lint/test invocations | 6 independent (A1) | 20–40 | Stale `.pyd` runs old code; dev-build GPU probe crash; r-miniconda shadow; `pipe\|tail` masking failures; nested-worktree clippy breakage |
| 2 | `forge3d-native-symbol` | DEVELOP | "expose X to Python" / "add a native function/class" / "hasattr returns False" | The full 7-site registration chain with rebuild + contract verification | 6 independent (A2) | 2–5 | Invisible-but-compiling symbols (P0.3); wheel-excluded features (P0.4); contract-test and `.pyi` drift |
| 3 | `forge3d-wgsl-contract` | INNOVATE + FIX | "new shader/pipeline/binding" / "Buffer is bound with size…" / "wgpu validation error" | WGSL↔Rust stride/layout/bind-group/format contract checks + validation-error triage | 5 independent (A3) | 3–8 | 80-vs-64-byte stride corruption; bind-group-budget drift; format-case mismatches; features silently dropped on minimal layout |
| 4 | `forge3d-golden-gates` | FIX + DEVELOP | "golden test failed" / "regen goldens" / "SSIM below threshold" | Classify → iterate cheaply → regen intentionally → prove goldens are tracked | 5 independent (A4) | 5–10 (during visual epics) | Casual regen masking regressions; gitignored goldens; chasing PT wobble; transient-probe false skips |
| 5 | `forge3d-gpu-test-gating` | DEVELOP + FIX | "add a renderer test" / "0 tests ran" / "CI panics with no GPU" | Placing tests where they actually run; honest GPU skip logic locally + CI | 5 independent (A5) | 2–4 | Silently-skipped test trees; `.expect()` panics on GPU-less runners; `--all-features` breakage |
| 6 | `forge3d-tracked-allocation` | DEVELOP + FIX | "create a buffer/texture" / "allocation gate failed" / "MemoryBudgetExceeded" | Tracked-wrapper allocation, Drop-guard hygiene, degradation recording, allowlist policy | 4 independent (A6) | 2–5 | Raw allocation sites failing the source gate; tracker leaks on early return; budget checks against the wrong metric |

All six are orchestrations over `.claude/rules/*.md` facts plus decision logic and verification steps that the rules files deliberately do not contain (rules state facts; skills sequence decisions and prove outcomes). None duplicates a global skill: `tdd`/`systematic-debugging`/`writing-plans` are process-generic, while every step below names a forge3d file, command, or threshold.

---

## 4. FULL SPECS

### 4.1 `forge3d-rebuild-verify` (DEVELOP + FIX)

- **name:** `forge3d-rebuild-verify`
- **description:** "Use when any `src/**/*.rs` or `*.wgsl` file was edited and behavior must be verified, when Python results look stale or a freshly added symbol seems missing, when running pytest/cargo suites in this repo, or when a background test run needs a trustworthy exit code."
- **WHEN TO USE:** after every Rust/WGSL edit; before claiming any test result; when `hasattr`/import results contradict the code; when choosing between dev and `--release`; when starting a background build/test.
- **WHEN NOT TO USE:** pure-Python edits under `python/forge3d/` with no native rebuild needed (pytest picks them up directly); doc-only changes.
- **INPUTS:** the set of edited files; the intended test scope (GPU-probing vs pure-Python vs Rust-only).
- **REPO SURFACE:** `pyproject.toml`, `.cargo/config.toml` (alias), `tests/conftest.py` (auto-bootstrap, `FORGE3D_NO_BOOTSTRAP`), `tests/_terrain_runtime.py` (GPU probe), `.venv/Scripts/python`, `CLAUDE.md` command list.
- **STEP-BY-STEP:** see the fully-authored draft in §6.1.
- **SUCCESS CHECK:** `.venv/Scripts/python -c "import forge3d; print(forge3d.__file__)"` resolves inside this repo's `.venv`, AND the targeted test command's real exit code is captured from a full log file (`cmd > log 2>&1; echo exit=$?`), AND (for new symbols) `hasattr` is `True` post-build.
- **PITFALLS PREVENTED:** all five A1 traps (stale `.pyd`, r-miniconda shadow, dev-profile probe crash, `pipe|tail` masking, nested-worktree clippy alias) plus the transient post-GPU-run segfault (retry-once rule from `[c2b543e3]`).
- **SIBLING FILES:** `commands.md` — the four canonical invocations (maturin dev/release; curated cargo test verbatim from CLAUDE.md; forge3d-clippy + its expanded fallback form; pytest with `.venv` python) so the skill body stays short.

### 4.2 `forge3d-native-symbol` (DEVELOP)

- **name:** `forge3d-native-symbol`
- **description:** "Use when exposing a new Rust function or class to Python in forge3d, when a `#[pyfunction]`/`#[pyclass]` compiles but is missing from `forge3d._forge3d`, or when `tests/test_api_contracts.py` fails after an API change."
- **WHEN TO USE:** adding/renaming/removing any native symbol; diagnosing "module has no attribute X" after a successful build.
- **WHEN NOT TO USE:** pure-Python API additions (only `__all__`, contract lists, and `.pyi` apply — no Rust sites); internal Rust refactors with no Python surface change.
- **INPUTS:** symbol kind (function vs class), owning domain (terrain/gis/lighting/…), whether it sits behind a new cargo feature.
- **REPO SURFACE:** `src/py_functions/<domain>.rs`, `src/py_module/functions/<domain>.rs`, `src/py_types/*.rs`, `src/py_module/classes.rs`, `src/lib.rs` (pymodule), `pyproject.toml:109` features, `python/forge3d/__init__.py` (guarded re-export + `__all__`), `tests/test_api_contracts.py` (:61/:115/:695), matching `.pyi`.
- **STEP-BY-STEP:** see the fully-authored draft in §6.2.
- **SUCCESS CHECK:** `maturin develop` then `.venv/Scripts/python -m pytest tests/test_api_contracts.py -q` green, and `.venv/Scripts/python -c "import forge3d as f; print(hasattr(f, '<symbol>'))"` prints `True`.
- **PITFALLS PREVENTED:** P0.2 (dead module-level probes for instance methods), P0.3 (orphaned pyclass), P0.4 (feature absent from maturin features), P2.3 (nested pyclass needs `Clone` + registration; `f32::MAX` must be a literal in signature defaults), python-api rule (the `.pyi` is drifted — never trust it as source of truth).
- **SIBLING FILES:** none needed — the checklist is the skill.

### 4.3 `forge3d-wgsl-contract` (INNOVATE + FIX)

- **name:** `forge3d-wgsl-contract`
- **description:** "Use when authoring or editing WGSL shaders, bind group layouts, or GPU pipelines in forge3d, and when triaging wgpu validation errors such as 'Buffer is bound with size X where the shader expects Y', bind-group-count failures, stride mismatches, or texture format errors."
- **WHEN TO USE:** any change touching `src/shaders/**/*.wgsl`, `src/viewer/**/*.wgsl`, a `BindGroupLayout`, a `#[repr(C)] Pod` struct bound to a shader, or pipeline creation; any wgpu validation panic.
- **WHEN NOT TO USE:** shader *content* changes that alter math but no struct layout, binding index, format, or entry point (rebuild + visual check suffices via `forge3d-rebuild-verify`).
- **INPUTS:** the WGSL file + the Rust pipeline-creation site pair; the target device class (portable ≤4 bind groups vs desktop 6).
- **REPO SURFACE:** 124 `.wgsl` files under `src/shaders/` + `src/viewer/`; `src/terrain/pipeline/creation.rs` (minimal-layout switch); `src/path_tracing/restir/types.rs` (canonical 80-byte reservoir); `tests/test_shader_reachability.py` (ALLOWLIST); `src/core/gpu.rs` (limits: `max_storage_buffers_per_shader_stage ≥ 8`).
- **STEP-BY-STEP:** see the fully-authored draft in §6.3.
- **SUCCESS CHECK:** `maturin develop` (dev ok if no GPU probe in the test; else `--release`) then a test that actually *builds the pipeline* passes with `WGPU_BACKEND=dx12` — a clean `cargo build` proves nothing (shaders validate only at `create_shader_module`/pipeline creation).
- **PITFALLS PREVENTED:** stride-not-field-list (PROMETHEUS 80/64), comment-vs-layout drift on `max_bind_groups`, dummy buffers smaller than one element of the largest runtime array (48-byte `BvhNode` vs 4-byte dummy), format-case mismatch (`[ecbc2c0c]`), silent minimal-layout feature drop (`[f7ec505b]`), orphaned `.wgsl` failing reachability.
- **SIBLING FILES:** `layout-rules.md` — a compact WGSL size/alignment table (vec3 padding, uniform vs storage array stride, the 176-byte `Globals` worked example) too heavy for the body.

### 4.4 `forge3d-golden-gates` (FIX + DEVELOP)

- **name:** `forge3d-golden-gates`
- **description:** "Use when a golden-image or adjudication test fails, when a visual change is intended and baselines need regeneration, when SSIM/lit-pass thresholds are missed, or when adding a new golden-backed test to forge3d."
- **WHEN TO USE:** any failure in `tests/test_recipe_goldens.py`, `test_terrain_visual_goldens.py`, `test_terrain_tv10_goldens.py`, `test_adjudication_gate.py`, `test_hybrid_terrain_pt.py`, `test_provenance_*.py`; any intentional renderer-output change.
- **WHEN NOT TO USE:** non-image numeric test failures; CI lanes skipped by the GPU probe (that's `forge3d-gpu-test-gating`).
- **INPUTS:** failing test id; whether the visual change is intentional; which gate family (recipe/terrain/adjudication/hybrid/provenance).
- **REPO SURFACE:** `tests/golden/{adjudication,hybrid_terrain,labels,presets,recipes,terrain}/`, `tests/goldens/determinism/`; env vars `FORGE3D_UPDATE_RECIPE_GOLDENS`, `FORGE3D_UPDATE_TERRAIN_GOLDENS`, `FORGE3D_UPDATE_ADJUDICATION_GOLDENS`, `FORGE3D_UPDATE_HYBRID_TERRAIN_GOLDENS`, `FORGE3D_UPDATE_PROVENANCE_FIXTURE`; diff artifact dir `FORGE3D_RECIPE_GOLDEN_ARTIFACT_DIR`.
- **STEP-BY-STEP:**
  1. **Classify the failure** before touching anything: (a) environment (missing pyproj / GPU probe skip / stale `.pyd` — route to `forge3d-rebuild-verify`), (b) transient probe segfault (`TerrainRenderer` exit-139 after heavy GPU runs → **rerun once** before debugging, per `[c2b543e3]`), (c) stochastic wobble (PT lit-pass ±~0.01 pp — thresholds absorb it; do NOT chase byte-identity), (d) real regression, (e) intentional change.
  2. **Prove the baseline exists in git**: `git ls-files tests/golden <failing-golden-path>` — BOP-P2-02 found ten goldens that existed only locally while docs claimed they were committed. A golden not in `git ls-files` is a red flag, not a baseline.
  3. **Iterate cheaply** (adjudication family): use the spp=1 raster-parity trick — `render_adjudication_pair(512, 512, 1)` scored against the committed `pt_reference.png` — instead of re-rendering PT at 4096 spp (~50–95 s per run).
  4. **Inspect diffs, then regen intentionally**: set the matching `FORGE3D_UPDATE_*` env var for exactly one run, with the diff/actual/expected artifacts reviewed first ("Only update a golden after the diff explains an intentional renderer change" — `[127f97ba]` fixture README). Never regen to make a red gate green without a diff explanation.
  5. **Re-run the gate twice** (wobble check) and commit goldens together with the renderer change and a one-line provenance note.
- **SUCCESS CHECK:** gate green on two consecutive runs without the update env var set; regenerated files appear in `git status` AND `git ls-files` after commit; diff artifacts reviewed (state what changed and why).
- **PITFALLS PREVENTED:** casual-regen masking regressions (`[127f97ba]` GT-04), gitignored goldens (BOP-P2-02), monkeypatched/placeholder goldens (`[69571db4]`: "goldens generated through a forced-placeholder monkeypatch instead of the GPU path"), chasing deterministic-looking SSIM 0.8146 without classifying first.
- **SIBLING FILES:** `gates.md` — table of gate family → test file → golden dir → env var → typical runtime → threshold semantics.

### 4.5 `forge3d-gpu-test-gating` (DEVELOP + FIX)

- **name:** `forge3d-gpu-test-gating`
- **description:** "Use when writing tests that touch the GPU, renderer, or feature-gated modules in forge3d, when a test run reports suspiciously few collected tests, when CI panics on GPU-less runners, or when a gate silently skips."
- **WHEN TO USE:** adding Rust unit tests near `src/terrain/renderer/**` or any `#[cfg(feature = "extension-module")]` tree; adding pytest tests that construct `Scene`/`TerrainRenderer`/`Session`; CI red with exit 101 / adapter panics; "N passed, hundreds filtered out".
- **WHEN NOT TO USE:** device-free pure-logic tests in unconditional modules (they just run); golden threshold failures (that's `forge3d-golden-gates`).
- **INPUTS:** what the test needs (device? extension-module? specific backend?); target gate (curated cargo test, local pytest, CI lane).
- **REPO SURFACE:** curated cargo test command (CLAUDE.md — lacks `extension-module` by design), `src/terrain/mod.rs` cfg-gating, `tests/_terrain_runtime.py::terrain_rendering_available` (:89), `scripts/terrain_ci_probe.py`, `.github/workflows/` (probe-gated lanes, `WGPU_BACKEND=dx12`), `pytest.ini` markers + `tests/conftest.py` (`pro`, `slow`).
- **STEP-BY-STEP:**
  1. **Place Rust device-free logic in an unconditional module** (pattern: `src/terrain/vt_family_residency.rs`, declared un-gated in `src/terrain/mod.rs`, imported by the renderer behind the cfg) — anything inside the gated renderer tree is invisible to the curated gate.
  2. **Prove collection, not just passing**: `cargo test --lib <module> -- --list` must print the new test names under the curated feature set. "0 tests, no error" is the documented failure mode.
  3. **Rust GPU helpers return `Option`, never `.expect()`**: on `None`, skip with a message (commit `2583c7fe` converted 24 panicking call sites across 11 files after CI exit-101 failures).
  4. **pytest GPU gating goes through the probe**: guard with `terrain_rendering_available()`; remember it caches per-process and can cache `False` after a transient segfault — rerun the suite once before debugging (`[c2b543e3]`).
  5. **Never `--all-features`** in any check/test/lint invocation — it enables `pyo3/extension-module` (link failures, `cf0f5c0d`) and pulls system PROJ (build-and-ci rule).
  6. **CI expectations**: Python CI runs install-smoke + 7 contract files only, not `tests/`; visual goldens run only on probe success AND terrain-path changes. Write tests knowing which lane will actually execute them.
- **SUCCESS CHECK:** new test names appear in `-- --list` output under the curated command (Rust) or are collected and either run or `SKIPPED` with an explicit GPU reason (pytest) — never silently absent; CI lane green on a GPU-less runner.
- **PITFALLS PREVENTED:** silently-skipped renderer tests (memory `curated-cargo-test-skips-renderer`), CI adapter panics, `--all-features` breakage, probe-cache false skips.
- **SIBLING FILES:** none — cross-reference `forge3d-rebuild-verify` `commands.md` for invocations.

### 4.6 `forge3d-tracked-allocation` (DEVELOP + FIX)

- **name:** `forge3d-tracked-allocation`
- **description:** "Use when creating any wgpu buffer or texture in forge3d Rust code, when `tests/test_allocation_gate.py` fails with a raw-allocation finding, or when Python raises `forge3d.MemoryBudgetExceeded`."
- **WHEN TO USE:** any new `Buffer`/`Texture` allocation in `src/`; allocation-gate red; budget errors; adding a fallback allocation path.
- **WHEN NOT TO USE:** bind groups, views, samplers, pipelines (not tracked resources); Python-side memory questions (use `forge3d.mem` diagnostics directly).
- **INPUTS:** allocation site, resource kind, whether host-visible (`MAP_READ|MAP_WRITE`), whether a fallback/degraded path exists.
- **REPO SURFACE:** `src/core/resource_tracker.rs` (`tracked_create_buffer`:343, `tracked_create_buffer_init`:375, `tracked_create_texture`:407), `tests/test_allocation_gate.py`, `tests/allocation_allowlist.toml`, the global degradation sink (`412ad76b`), `src/core/memory_tracker/` registry, `.claude/rules/rust-core.md` (512 MiB host-visible budget, ENFORCE default).
- **STEP-BY-STEP:**
  1. **Never call `device.create_buffer`/`create_texture` raw** — use the tracked wrappers; the source-level gate (`2e11f2c0`) fails on any new raw site not in `tests/allocation_allowlist.toml`.
  2. **Own the guard**: hold the returned tracked guard for the resource's lifetime so `?`/early-return paths release tracker state (PROMETHEUS `TrackedGpu` lesson); implement `Drop` on wrapper types owning multiple tracked resources.
  3. **Fallbacks are degradations, not silence**: when an allocation falls back (smaller size, different format, WBOIT snapshot path), record it in the degradation sink (`f488592d`, `fb6d746b`) instead of manual ad-hoc tracking.
  4. **Budget-check the right metric**: the 512 MiB budget compares `host_visible_bytes` only; device-local resources never appear there — gate features on tracker metrics captured AFTER all allocations, not on `peak_host_visible_bytes` alone.
  5. **Allowlist only with a documented reason** — the allowlist is for deliberate exceptions, not for making the gate pass.
- **SUCCESS CHECK:** `.venv/Scripts/python -m pytest tests/test_allocation_gate.py -q` green; for budget-sensitive paths, a mem report showing expected tracked deltas (and `MemoryBudgetExceeded` raised, not a crash, when over).
- **PITFALLS PREVENTED:** raw-site gate failures, tracker leaks on early return, budget gates reading the wrong metric, silent fallbacks.
- **SIBLING FILES:** none — the wrapper signatures are three functions in one file.

---

## 5. BUILD ORDER

Dependencies: every skill's SUCCESS CHECK routes through the rebuild-and-prove loop, so `forge3d-rebuild-verify` must exist first and the others cross-reference it (never duplicate its commands).

1. **`forge3d-rebuild-verify`** — foundation; unblocks honest verification for all others. *(Top-3 leverage)*
2. **`forge3d-native-symbol`** — highest per-use error-prevention after the loop; references #1 for its success check. *(Top-3 leverage)*
3. **`forge3d-wgsl-contract`** — deepest failure cost (silent memory corruption / device-only breakage); references #1. *(Top-3 leverage)*
4. **`forge3d-golden-gates`** — references #1 (freshness) and #5 (probe skips); build after #5's probe rules are written, or accept a forward reference.
5. **`forge3d-gpu-test-gating`** — references #1's commands file.
6. **`forge3d-tracked-allocation`** — self-contained; lowest urgency because the source-level gate already fails loudly with the fix in the error path.

**Instantiation discipline (per `superpowers:writing-skills`):** the transcript evidence in §2 is the documented RED baseline — real agents, without these skills, produced exactly the failures each skill targets (verbatim rationalizations included, e.g. running bare `python`, regenerating goldens to green a gate, trusting `cargo build` for shader validity). Before deploying each SKILL.md, run the GREEN verification: a fresh subagent given a matching task with the skill loaded must hit the success check; close any new rationalizations (REFACTOR) before moving to the next skill. Do not batch-deploy all six untested.

---

## 6. FULLY-AUTHORED TOP-3 DRAFTS

### 6.1 `forge3d-rebuild-verify/SKILL.md`

```markdown
---
name: forge3d-rebuild-verify
description: Use when any forge3d src/*.rs or *.wgsl file was edited and behavior must be verified, when Python results look stale, when a freshly added native symbol appears missing, when choosing dev vs release builds, or when running pytest/cargo/clippy in this repo (including background runs).
---

# forge3d Rebuild & Verify Loop

## Overview

Python imports a **compiled** `_forge3d.pyd`. Nothing you edit in `src/` exists for Python
until `maturin develop` rebuilds it, and three local traps (stale interpreter, wrong build
profile, masked exit codes) make "it passed" claims false. This skill is the loop that
turns an edit into a *proven* result.

**Core principle: no claim without a freshness proof and a real exit code.**

## When to Use

- After editing any `src/**/*.rs` or `*.wgsl` file, before running anything in Python
- A symbol you just registered shows `hasattr(...) == False`, or behavior matches old code
- Before running pytest suites that construct `Scene` / `TerrainRenderer` / `Session`
- When launching any build/test as a background task

**Not for:** pure edits under `python/forge3d/` (no rebuild needed) or docs-only changes.

## The Loop

1. **Pick the build profile.**
   - Will anything downstream touch a GPU probe (mapscene, terrain, occlusion, goldens,
     anything importing `tests/_terrain_runtime.py`)? → `maturin develop --release`.
     Dev-profile builds crash the DX12 adapter probe on this machine
     (Windows fatal access violation in `terrain_rendering_available`).
   - Pure-Python / solver-only tests after a Rust edit → plain `maturin develop` is fine.
2. **Build.** `maturin develop [--release]`. A clean `cargo build` is NOT a substitute:
   WGSL validates only when wgpu builds the pipeline at runtime, and Python still imports
   the old `.pyd` until maturin reinstalls it.
3. **Prove freshness with the right interpreter.**
   ```
   .venv/Scripts/python -c "import forge3d; print(forge3d.__file__)"
   ```
   Must resolve inside `C:\Users\milos\forge3d\.venv`. The bare `python` on PATH is
   r-miniconda with a stale editable install (`forge3d.pth` -> D:\forge3d\python) that
   silently shadows your build. For a new symbol, also check
   `.venv/Scripts/python -c "import forge3d as f; print(hasattr(f, '<symbol>'))"`.
4. **Run the right command** (see commands.md for the verbatim forms):
   - Python tests: `.venv/Scripts/python -m pytest tests/... -v --tb=short`
     (set `FORGE3D_NO_BOOTSTRAP=1` if you already built — conftest otherwise rebuilds
     with `maturin develop --release` at session start).
   - Rust tests: the curated `cargo test --workspace --features <curated list> --
     --test-threads=1 --skip gpu_extrusion --skip brdf_tile` from CLAUDE.md. Never
     `--all-features` (pulls system PROJ + pyo3/extension-module link failures).
   - Lint: `cargo forge3d-clippy` — never plain `cargo clippy`.
5. **Capture real exit codes.** For background or long runs:
   ```
   cmd > "$SCRATCHPAD/run.log" 2>&1; echo exit=$?
   ```
   then grep the log. Never `cmd | tail -N`: it truncates the retained log AND reports
   tail's exit code (0), so failures look like passes.
6. **Interpret flakes before debugging.** `TerrainRenderer(session)` can transiently
   segfault (exit 139) right after heavy back-to-back GPU runs, which caches the GPU
   probe as False and silently skips gates for that pytest process. Re-run once; only
   debug if it reproduces.

## Quick Reference

| Situation | Do |
|---|---|
| Edited `.rs`/`.wgsl`, next step is GPU-probing pytest | `maturin develop --release` |
| Edited `.rs`, next step is pure-logic pytest | `maturin develop` |
| Any pytest/spike script | `.venv/Scripts/python`, never bare `python` |
| Already built this session | `FORGE3D_NO_BOOTSTRAP=1` before pytest |
| Rust lint | `cargo forge3d-clippy` (from the MAIN checkout) |
| In a worktree nested inside the repo | Alias breaks ("Unrecognized option: 'workspace'"): use a `C:/tmp` worktree, or run the expanded clippy command from commands.md |
| Background build/test | Redirect to a log file; echo `$?`; grep the log |
| Probe segfault right after heavy GPU runs | Re-run once before debugging |

## Rationalizations — all false here

| Excuse | Reality |
|---|---|
| "cargo build passed, the shader is fine" | naga validates at pipeline creation, not compile; rebuild via maturin and run a pipeline-touching test |
| "python found forge3d, so it's my build" | PATH python is a stale D:\forge3d editable; check `forge3d.__file__` |
| "dev build is faster and probably fine" | dev-profile crashes the GPU adapter probe with an access violation |
| "the background run printed the last lines, looked green" | `| tail` masked the exit code; the failure was above the fold |
| "clippy alias errored, I'll commit --no-verify" | Run the expanded command or relocate the worktree; don't skip the gate |

## Red Flags — STOP and re-run the loop

- You are about to write "tests pass" without a captured exit code
- `hasattr` False right after "successful" build → wrong interpreter until proven otherwise
- A gate "passed" in far less time than its known runtime (probe-skipped, not passed)
```

*(ships with `commands.md`: the four verbatim invocations — dev/release maturin, curated cargo test, forge3d-clippy + expanded fallback, `.venv` pytest form)*

### 6.2 `forge3d-native-symbol/SKILL.md`

```markdown
---
name: forge3d-native-symbol
description: Use when exposing a new Rust function or class to Python in forge3d, when a #[pyfunction] or #[pyclass] compiles but is missing from forge3d._forge3d, when tests/test_api_contracts.py fails after an API change, or when removing/renaming a native symbol.
---

# forge3d Native Symbol Registration

## Overview

A `#[pyfunction]`/`#[pyclass]` that compiles is **invisible to Python** until it is
registered, re-exported, contract-locked, and stubbed — seven sites across Rust, config,
Python, and tests. Every partial chain in this repo's history became a silent bug
(P0.2 dead probes, P0.3 orphaned classes, P0.4 wheel-excluded features).

**Core principle: the chain is done when the contract test passes and `hasattr` is True —
not when Rust compiles.**

## When to Use

- Adding, renaming, or removing any native function or class
- "module 'forge3d._forge3d' has no attribute X" after a successful build
- `tests/test_api_contracts.py` failures

**Not for:** pure-Python API (only steps 5–7 apply); Rust-internal refactors with no
Python-visible change.

## The Chain (functions)

1. **Body** — `#[pyfunction] pub(crate) fn ...` in `src/py_functions/<domain>.rs`
   (new file ⇒ add `pub mod` in `src/py_functions/mod.rs`). Import PyO3 names through the
   crate-root re-exports (`use super::*;` / `use super::super::*;`), never `use pyo3::...`
   directly. Return `PyResult<T>`; `RenderError` auto-converts via `?`. Ad-hoc argument
   validation raises `PyValueError`/`PyRuntimeError` directly — no new error enums per site.
2. **Register** — `m.add_function(wrap_pyfunction!(name, m)?)?` in
   `src/py_module/functions/<domain>.rs`. Registration lives behind
   `#[cfg(feature = "extension-module")]`; group related registrations in one cfg block.
3. **Feature gate check** — if the symbol sits behind ANY new cargo feature, that feature
   MUST be added to `[tool.maturin].features` in `pyproject.toml` (~line 109), or it
   compiles fine and ships absent from the wheel. This list is the single source of truth
   for wheel contents; the CI cargo-test feature list is different and does NOT prove
   wheel inclusion.
4. **(Classes instead of 1–2)** — `#[pyclass(module = "forge3d._forge3d", name = "...")]`
   usually in `src/py_types/<x>.rs`; one `m.add_class::<T>()?` in
   `src/py_module/classes.rs`. Nested pyclasses used as fields need `Clone` AND their own
   `add_class` registration. `f32::MAX` defaults must be the literal `3.4028235e38` in
   `#[pyo3(signature)]`.
5. **Package surface** — in `python/forge3d/__init__.py`: guarded re-export inside the
   `if _NATIVE_MODULE is not None:` block (with `hasattr` guard) AND an entry in the
   explicit `__all__` list.
6. **Contract locks** — `tests/test_api_contracts.py`: `EXPECTED_CLASSES` (~:61),
   `EXPECTED_FUNCTIONS` (~:115), `EXPECTED_PACKAGE_ATTRS` (~:695). Update in lockstep,
   including on removals.
7. **Stubs** — the matching `.pyi`. It is hand-maintained and `__init__.pyi` is
   known-drifted: verify against `__all__` and `dir(_forge3d)`, never trust the stub as
   source of truth. Sync the signature to the Rust `#[pyo3(text_signature)]` /
   `#[pyo3(signature)]`, not to guesswork.

## Verify (REQUIRED — evidence before claims)

Use forge3d-rebuild-verify, then:

```
.venv/Scripts/python -c "import forge3d as f; print(f.__file__); print(hasattr(f, '<symbol>'))"
.venv/Scripts/python -m pytest tests/test_api_contracts.py -q
```

`hasattr` must be True with `__file__` inside the repo `.venv` — a False here after a
green build means wrong interpreter (stale editable shadow), not a missing registration.

## Common Mistakes

| Mistake | Consequence | Fix |
|---|---|---|
| Class compiled, never `add_class`-ed | Invisible to Python, no error anywhere | Step 4; contract test pins it |
| New cfg feature, pyproject untouched | Wheel silently excludes the symbol | Step 3 first, then rebuild |
| Probing `hasattr(_native, "method")` for an instance method | Dead probe, fallback path runs forever | Read the Rust signature; methods live on classes |
| Contract lists updated but `.pyi` skipped | Type-checker lies to users | Step 7 every time |
| Verified with bare `python` | False negative, wasted debugging round | `.venv/Scripts/python` only |
```

### 6.3 `forge3d-wgsl-contract/SKILL.md`

```markdown
---
name: forge3d-wgsl-contract
description: Use when authoring or editing WGSL shaders, bind group layouts, Pod uniform/storage structs, or pipeline creation in forge3d, and when triaging wgpu validation errors like "Buffer is bound with size X where the shader expects Y", bind group count/layout mismatches, array stride errors, or texture format mismatches.
---

# forge3d WGSL <-> Rust Pipeline Contract

## Overview

Every WGSL binding is a byte-level contract with a Rust `BindGroupLayout` and `Pod`
struct, enforced by wgpu **only at pipeline creation or first dispatch — never at
`cargo build`**. The recorded failures are stride drift (an 80-byte WGSL struct over a
64-byte Rust allocation writing out of range), bind-group budgets that drifted from
their comments, and format-case mismatches that break only on real devices.

**Core principle: verify the STRIDE and the ACTUAL layout, not the field list or the
comment.**

## When to Use

- Any change to `src/shaders/**/*.wgsl` or `src/viewer/**/*.wgsl` touching struct
  layouts, binding indices, groups, formats, or entry points
- Any change to a `BindGroupLayout`, `PipelineLayout`, or a `#[repr(C)]` Pod struct
  bound to a shader
- Any wgpu validation error at pipeline build or dispatch

**Not for:** pure math changes inside an existing entry point with untouched interfaces
(just rebuild and visually verify via forge3d-rebuild-verify).

## Contract Checklist

1. **Find both sides first.** The WGSL file and its pipeline-creation site (e.g.
   `src/shaders/terrain.wgsl` <-> `src/terrain/pipeline/creation.rs`). Change one side,
   update the other in the same edit.
2. **Stride, not fields.** For every struct in a storage array: compute the WGSL stride
   (alignment rules: vec3 pads to 16; struct size rounds up to its alignment) and compare
   against `std::mem::size_of::<RustPod>()`. Two structs can share every field name and
   still differ in stride. Canonical precedent: the ReSTIR `Reservoir` in
   `src/path_tracing/restir/types.rs` is the 80-byte layout that replaced an ad-hoc
   64-byte twin. wgpu will NOT catch a short runtime-sized array until the last pixels
   write out of bounds.
3. **Count actual bind groups.** Portability budget is `max_bind_groups = 4`; the terrain
   path additionally switches to `terrain_minimal.wgsl` (4 groups, no cloud shadows /
   planar reflections) when the device reports `< 6`. Count groups in the
   PIPELINE LAYOUT, not in shader comments — comments have drifted before. New group-4/5
   bindings silently vanish on minimal-layout devices: guard features so they degrade
   with a diagnostic instead of assuming the binding exists.
4. **Storage-buffer budget:** <= 8 per compute stage (device raises the limit to >= 8
   deliberately in `src/core/gpu.rs`). If a new binding would exceed it, split the entry
   point and give it its OWN pipeline layout (precedent: `main_terrain_gbuffer`).
5. **Dummy/placeholder buffers** must cover ONE ELEMENT of the largest runtime-sized
   array they stand in for (WGSL `BvhNode` = 48 bytes). A 4-byte dummy passes creation
   and fails at dispatch: "Buffer is bound with size 4 where the shader expects 16".
6. **Formats are case- and letter-exact across three sites:** WGSL texture declaration,
   the Rust `TextureFormat`, and the layout entry. `R8Unorm` vs `Rgba8Unorm` vs WGSL
   `rgba8unorm` mismatches validate only on a real device.
7. **Uniform structs carry documented byte sizes** — e.g. terrain `Globals` is
   "176 bytes total, must match Rust" (`src/shaders/terrain.wgsl` header). Changing a
   field means updating BOTH sides and the documented size.
8. **New entry points are free; new files are not.** Extra entry points in a shared
   module cost nothing until a pipeline references them (interface validates per entry
   point). But every new `.wgsl` file must be referenced from `src/**/*.rs` or explicitly
   allowlisted in `tests/test_shader_reachability.py`, else that test fails.

## Verify (REQUIRED)

`cargo build` proves nothing about WGSL. Run the loop from forge3d-rebuild-verify, then
execute a test that actually BUILDS the pipeline (a minimal `Scene` render or the
feature's pytest) with `WGPU_BACKEND=dx12`. The contract holds only when a pipeline
using every touched binding has been created and dispatched once without validation
errors.

## Error Triage

| wgpu error | Likely cause | Check |
|---|---|---|
| "Buffer is bound with size X where the shader expects Y" | Dummy/short buffer, or stride drift | Steps 2 and 5 |
| Bind group layout mismatch at pipeline creation | Rust layout != WGSL group/binding indices | Steps 1 and 3 |
| Pipeline uses more bind groups than the device allows | Budget drift, comment trusted | Step 3 |
| Texture format error at bind-group creation | Case/format mismatch across the three sites | Step 6 |
| Wrong pixels, no error, only at high indices | Short runtime-sized array (stride) | Step 2 |
| Works locally, broken on another backend/device | Minimal-layout swap or native-limit dependence | Step 3 |
```

*(ships with `layout-rules.md`: WGSL size/align table + the 176-byte `Globals` worked example)*

---

## What I might be missing (self-critique)

The weakest-evidence archetype is **`forge3d-tracked-allocation`**: its 1,056 transcript hits are overwhelmingly one campaign (`[c10ba4ed]`, the CENSOR wave of 2026-07-08/09) rather than many independent recurrences, and now that `tests/test_allocation_gate.py` fails loudly at the source level, the gate's own error message may already teach the fix well enough that a skill adds little beyond the Drop-guard and degradation-sink subtleties. To confirm or cut it, I would read next: (a) the gate test's actual failure output — if it names the tracked-wrapper alternative per site, demote the skill to a two-line rule; (b) the next two or three feature sessions that allocate GPU resources, checking whether agents reach for raw `create_buffer` and how long the gate failure detains them; and (c) a grep of future transcripts for `MemoryBudgetExceeded` to see whether the budget-vs-device-local distinction (the genuinely non-obvious part) recurs without prompting. Secondarily, my uses/week estimates for `forge3d-golden-gates` assume the current visual-epic cadence continues; if the roadmap shifts to GIS/vector work, rank 4 and rank 5 should swap.
