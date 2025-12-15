You are Codex CLI acting as:
- Senior Engineering Program Manager + elite graphics/runtime engineer
- Repo: forge3d (Rust backend + Python frontend via PyO3)
- Constraints (non-negotiable): Vulkan ≥1.2 compatible design, WebGPU/WGSL primary, RAII, GPU host-visible heap budget ≤ 512 MiB, cross-platform win_amd64/linux_x86_64/macos_universal2.

TASK (PLAN ONLY — DO NOT IMPLEMENT):
Create a surgically precise refactoring plan to split `src/viewer/mod.rs` (~8k LOC) into many Rust source files, where every new/modified file is ≤ 300 lines (excluding license headers if any). The plan must preserve behavior and public API. This is a refactor, not a feature.

ABSOLUTE RULES (from AGENTS.md — treat as binding):
1) Keep files small (≤ ~300 LOC). Split large modules logically.
2) Refactor safely in tiny steps; keep a working baseline; run the canonical verification suite frequently.
3) Do not change behavior while refactoring. Any behavior change must be treated as a separate, explicitly-scoped task.
4) Respect Rust/Python boundary: core GPU/rendering stays in Rust; Python orchestrates/tests.
5) No “DONE/PASS” claims in implementation work without raw command output — for this task you produce a PLAN, but you MUST still specify exact commands and checkpoints for each step as if someone will execute them.

WHAT YOU MUST DO FIRST (EVIDENCE REQUIRED IN THE PLAN):
A) Read `AGENTS.md` and summarize ONLY the refactoring-relevant rules you will enforce (bullet list, max ~15 bullets).
B) Inspect `src/viewer/mod.rs` fully and produce a structured inventory:
   - All top-level public exports (pub structs/enums/fns/modules/re-exports) and their intended consumers.
   - Major responsibilities currently mixed in the file (e.g., init/device/surface, event loop, input, camera, render graph/passes, SSR/GI/P5 scenes, snapshots/readback, IPC, HUD, shaders/embedded WGSL, etc.).
   - Identify embedded WGSL sources (raw strings/include_str) and where they should live after refactor.
C) Inspect upstream and downstream dependencies:
   - Upstream: find every Rust module/crate location that imports `crate::viewer::*` or references Viewer/run_viewer/run_viewer_with_ipc/IpcUserEvent/etc. Include a list of file paths + symbols used.
   - Downstream: enumerate viewer submodules already present under `src/viewer/` and how `mod.rs` depends on them.
   - If there are Python entrypoints or CLI commands that call viewer functions, identify them and list the call chain to `src/viewer/mod.rs`.

DELIVERABLES (PLAN OUTPUT MUST INCLUDE ALL):
1) Target module tree (directory + file names):
   - A proposed `src/viewer/` structure where `mod.rs` becomes a small orchestrator/re-export file (≤ 300 LOC).
   - New files grouped by responsibility (examples: viewer_state.rs, viewer_init.rs, viewer_event_loop.rs, viewer_input.rs, viewer_render.rs, viewer_snapshot.rs, viewer_ipc_bridge.rs, pipelines/*, passes/*, shaders/*.wgsl, etc.).
   - Every file in the proposed tree must be ≤ 300 LOC; if a responsibility is too large, split further.

2) Public API preservation strategy:
   - Which symbols stay re-exported from `src/viewer/mod.rs`.
   - Whether to introduce internal modules with `pub(crate)` and keep external API unchanged.
   - “Parallel change / expand-contract” steps if any renames are unavoidable (prefer: no renames).

3) A precise “move map”:
   - A table mapping chunks of `src/viewer/mod.rs` → new file path(s).
   - For each moved block: list key types/functions/constants moved, and any cross-module types they depend on.
   - Highlight any cycles that could appear and how you avoid them (acyclic deps).

4) Step-by-step execution plan (small, reversible increments):
   For EACH step, include:
   - The exact code movement action (what to extract/move first, second, third…).
   - The mechanical edits required (module declarations, imports, visibility changes, re-exports).
   - The exact verification commands to run after the step (must be concrete, not “run tests”).
     At minimum include: `cargo fmt`, `cargo check`, and the canonical Python test command(s) used by this repo (discover from AGENTS.md / existing CI config; do not guess).
   - Expected risks/failure modes for that step (e.g., borrow checker fallout, lifetime issues around winit surface, feature-gated code, shader include paths, visibility errors).
   - A rollback note (how to revert if it breaks).

5) Shader extraction plan:
   - Identify each embedded WGSL chunk (label it by its Rust label string if present, e.g., "viewer.gbuf.geom.shader") and propose a `.wgsl` file name/location.
   - Specify how it should be loaded (prefer compile-time include_str with stable paths, or a central shader registry module).
   - Ensure the refactor does not change shader content (byte-identical if possible). If exact identity is hard, specify how to validate equivalence (e.g., hash the WGSL strings before/after).

6) Dependency hygiene rules you will enforce:
   - Boundaries between viewer orchestration vs passes/pipelines vs scene presets vs IO/readback.
   - Where GPU resources live (RAII owners) vs where command encoding happens.
   - How you avoid “god structs” getting worse (e.g., splitting Viewer into sub-structs like RenderPipelines, SceneState, CaptureState, etc., each in its own module, then composed).

7) Final acceptance criteria for the refactor (plan-level):
   - `src/viewer/mod.rs` ≤ 300 LOC.
   - No new/modified viewer-related file exceeds 300 LOC.
   - No public API breakage for external callers (Rust and Python).
   - All tests pass and the viewer still runs (define exact smoke command and what constitutes success).

OUTPUT FORMAT REQUIREMENTS:
- Your response MUST be a single structured Markdown document with these headings exactly:
  1. Constraints and rules enforced
  2. Inventory of current `src/viewer/mod.rs`
  3. Upstream callers and downstream dependencies
  4. Target module tree (≤ 300 LOC per file)
  5. Move map (old → new)
  6. Step-by-step refactor plan with verification commands
  7. Shader extraction plan
  8. Risks, unknowns, and stop-conditions
- Include at least one table in sections (3), (5), and (6).
- Do not write any code patches. Do not implement. Plan only.
- If you find that “≤ 300 LOC per file” conflicts with existing viewer submodules already in the tree, call it out explicitly and include those files in the plan scope with splits.

Begin by reading AGENTS.md, then proceed.
