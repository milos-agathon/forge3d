# forge3d Codebase Refactor Playbook

This document replaces the prior “no-code-change” audit script with guidance that fits the forge3d rules in `AGENTS.md`. Use it to plan and execute safe, high-quality refactors across Rust, WGSL, and Python.

## Core guardrails (read first)
- **Rust crate first, Python package second.** Keep Rust core and PyO3 bindings aligned; Python tests define behavior.
- **Tests and docs define semantics.** Never change runtime behavior without checking relevant tests and docs. Add/adjust tests in Python (not Rust) to cover new behavior.
- **Memory/GPU discipline.** Respect the 512 MiB host-visible budget and existing GPU feature flags. Reuse buffers/textures where possible.
- **Small, clean steps.** Extract small functions, clarify names, remove duplication, and keep files orderly. Leave every touched file cleaner.
- **Stay in bounds.** Keep bind group layouts, shader bindings, and Python params in sync. Avoid speculative hooks (YAGNI).

## Safe-by-default workflow
1. **Snapshot the workspace**
   - `pwd`
   - `git status --porcelain`
   - `git rev-parse --show-toplevel`
   - `git rev-parse HEAD`

2. **Read the intent before editing**
   - Re-read `AGENTS.md` rules.
   - Open relevant docs/tests (e.g., `docs/`, `tests/`) to understand expected behavior and invariants.

3. **Scope the change**
   - State the goal, impacted domains (Rust core, Python facade, WGSL), and constraints (memory, feature flags).
   - Identify the exact files and bindings that must stay in sync (e.g., `src/render/params.rs` ↔ `python/forge3d/config.py`, WGSL bind layouts ↔ Rust bind groups).

4. **Design the refactor**
   - Prefer extraction over in-place complexity.
   - Keep functions small and single-responsibility; avoid flag arguments.
   - Maintain the stepdown rule in files: high-level first, details below.
   - Keep public APIs stable unless the change demands otherwise; use expand/contract when altering public surfaces.

5. **Code with guardrails**
   - Use `apply_patch`-style minimal diffs.
   - Align names with intent; avoid ambiguous or cryptic identifiers.
   - Keep terrain, GI, and memory bindings consistent across Rust/WGSL/Python.
   - For third-party boundaries, use thin adapters; isolate vendor-specific calls.

6. **Validate early and often**
   - Prefer targeted, fast checks: `cargo check -q`, `cargo fmt -- --check`, `cargo clippy` (warnings as appropriate), `python -m compileall -q .`, `pytest -q tests/<scope>`.
   - For rendering changes, run the smallest relevant Python test or example; keep golden outputs stable unless intentionally updating them.
   - If behavior changes, add/adjust Python tests; do not add Rust tests for behavior coverage.

7. **Document what changed**
   - Update nearby docs/schemas/config comments when behavior or expectations change.
   - Keep doc updates concise and aligned with actual code paths.

8. **Finish clean**
   - Rerun `git status --porcelain` to ensure only intended files changed.
   - Summarize the change, risks, and validation steps in your PR/commit message.

## Refactor checklists

### Bindings & API sync
- When updating renderer/config params: change Rust structs/enums, Python mirrors, default values, and any serialization/deserialization.
- When touching WGSL layouts: update Rust bind group layouts, descriptor set creation, and any Python-side size/shape assumptions.

### Memory & performance
- Reuse GPU resources where possible; avoid per-frame allocations.
- Confirm texture formats and sizes respect the memory budget; prefer reuse over duplication.
- Avoid O(n) per-frame work in hot paths; consider precomputation or caching if safe.

### Testing discipline
- Add regression tests in Python for changed behavior.
- Use deterministic seeds/fixtures where possible.
- Keep tests fast and isolated; avoid shared mutable fixtures.

### Naming and structure
- Make intent obvious through names.
- Group related code; maintain vertical whitespace for readability.
- Extract polymorphism/strategies when type-conditionals spread; avoid premature abstractions.

## Command palette (safe defaults)
- Build/check: `cargo check -q`
- Lint: `cargo fmt -- --check`, `cargo clippy`
- Python sanity: `python -m compileall -q .`
- Tests: `pytest -q tests/<target>` (only what’s needed)

Run only what is necessary for your change; do not install new system dependencies.

## Change log template (for PRs/commits)
1. **Summary:** What changed and why.
2. **Behavior:** What users/tests observe; note any intentional diffs.
3. **Risk/Surface:** Rust modules, Python APIs, WGSL shaders touched.
4. **Validation:** Commands/tests/examples run and results.
5. **Follow-ups:** Any deferred cleanups (with issue/plan).

## Final reminder
Code is the design. Keep it expressive, small, and test-backed. Respect forge3d’s layering (Python facade ↔ PyO3 bindings ↔ Rust core ↔ WGSL) and the memory/GPU constraints on every change.