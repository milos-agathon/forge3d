You are working in the local `forge3d` repository.

Objective:
Sweep through `src/`, `src/shaders/`, `python/`, and `examples/` to identify the repo-specific skills that would most improve speed, correctness, rendering quality work, and day-to-day implementation efficiency in forge3d. Then create those skills inside the repo under `.codex/skills/`.

This is a skill-creation task, not a feature implementation task. Do not edit Rust, WGSL, Python runtime code, examples, tests, fixtures, manifests, or existing docs except for the report named below. The deliverables are repo-local skill folders and one concise audit report.

Deliverables:
- Create selected skills under `.codex/skills/<skill-name>/SKILL.md`.
- Write a report to `docs/forge3d-recommended-skills-audit.md`.
- Do not create placeholder skills. If a skill is not clearly useful after inspection, do not create it.

Operating rules:
- Start with `git status --short`; identify unrelated dirty files and leave them alone.
- Inspect the actual repo. Do not rely on memory.
- Use `rg --files`, `rg`, and targeted reads.
- Inspect at minimum:
  - `src/`
  - `src/shaders/`
  - `python/forge3d/`
  - `examples/`
  - `tests/` only enough to understand validation patterns
  - existing relevant docs under `docs/`
- Prefer the smallest useful set of skills. Target 3-6 skills. More than 6 requires a strong written reason.
- Each skill must accelerate repeated work that is real in this repo. Do not create generic skills like "Rust coding", "Python testing", or "documentation writing".
- Avoid overlap with globally available skills unless the forge3d-specific procedure, file map, or validation rules are genuinely different.
- Do not create scripts, references, or assets unless they materially reduce repeated work. A focused `SKILL.md` is enough for most skills.
- Do not expose hidden chain-of-thought. Give conclusions, evidence, rationale, and tradeoffs.

External workflow pattern finding to incorporate:
- Treat `dzhng/skills` as a workflow pattern library only, not as a drop-in forge3d workflow.
- Reuse ideas, not files, unless direct inspection proves copying a selected skill is better than a short forge3d-specific rewrite.
- Strong patterns to adapt:
  - feature slicing into independently verifiable slices with API seams, typed contracts, focused tests, screenshots/browser checkpoints where relevant, and reslicing when hidden complexity appears
  - renderer/WebGPU review discipline for shader contracts, bind groups, buffers, pass orchestration, depth semantics, GPU resource ownership, visual validation, and performance gates
  - screenshot critique for full-frame screenshots plus zoomed crops instead of subjective visual claims
  - refactor-clean discipline that prefers deletion, convergence on existing abstractions, and avoiding architectural sediment
  - generic docs discipline when it supports implementation correctness
- Risky pattern to constrain:
  - Do not import an autonomous "keep going until everything is done" loop unmodified. forge3d has sensitive Rust/Python, native/WASM, renderer/Studio, RenderSpec/state, and GIS correctness boundaries. A skill may allow sustained execution only inside one bounded slice with explicit verification and stop conditions.
- Missing from the external repo and therefore requiring forge3d-specific skill content:
  - Rust/PyO3 registration and wrapper contracts
  - WGSL/wgpu details as implemented in forge3d
  - GIS correctness, CRS, raster/vector boundaries, and no-Python-backend rules
  - forge3d Studio runtime constraints if Studio-specific files exist
- Preferred development loop to encode when relevant:
  - roadmap/doc -> feature-slicing spec -> one Codex slice -> focused tests -> review bundle -> external review -> fixes -> PR/merge.
  - Do not encode a broad "let the agent run across the whole feature until it thinks it is done" workflow.

Skill format:
Each skill folder must contain exactly one required file unless an extra resource is justified:

`.codex/skills/<skill-name>/SKILL.md`

Use this shape:

```markdown
---
name: <lowercase-hyphen-name>
description: <clear trigger description that says exactly when to use this forge3d-specific skill>
---

# <Human Title>

## Purpose
<One short paragraph.>

## Workflow
<Concise ordered workflow. Include repo-specific files and commands.>

## Evidence To Read
<Only the minimum recurring files/modules that agents should inspect before acting.>

## Validation
<Smallest relevant checks. Prefer existing repo checks. Include when to skip expensive checks.>

## Boundaries
<What this skill must not do.>
```

Skill quality requirements:
- The `description` must include trigger language because it is the only always-visible discovery text.
- Keep each `SKILL.md` lean. Remove generic advice an expert coding agent already knows.
- Include exact repo paths, symbols, commands, and hazards where useful.
- Include lessons from `AGENTS.md` when they affect the skill's domain.
- Include current dirty-worktree caution in workflows: inspect status and avoid unrelated user changes.
- Every skill must have a validation section with at least one concrete check or a reason no check is appropriate.
- Every skill must have boundaries to prevent scope creep.
- If two proposed skills overlap heavily, merge them.

Skill candidates to consider, but only create if the repo evidence supports them:
- Forge3D feature-slicing skill for turning roadmap/docs into one independently verifiable implementation slice with explicit API seams, tests, visual checkpoints, and reslicing rules.
- Forge3D architecture skill for Rust core ownership, thin Python wrappers, native/browser or Studio boundaries, RenderSpec/state separation where present, and preventing architectural sediment.
- Rendering-quality-audit skill for diagnosing terrain, lighting, post-processing, offline render, and gallery image quality issues.
- WGSL/render-pipeline skill for tracing shader-to-Rust bindings, bind groups, pipelines, uniforms, and feature gates.
- Python-native-surface skill for PyO3 registration, Python wrappers, `.pyi` stubs, `__all__`, and API contract tests.
- Visual-golden-validation skill for deterministic image generation, pixel/statistical checks, gallery examples, and regression evidence.
- GIS/carto-engine skill for GIS helper implementation, CRS/raster/vector boundaries, and carto-engine docs alignment.
- Example-gallery skill for keeping examples, docs gallery pages, generated images, and recipe manifests aligned.
- Performance/memory skill for GPU memory, virtual texturing, point clouds, tile caches, streaming, and profiling.
- Review-bundle skill for git status, scoped diff, untracked files, focused test logs, generated artifacts, and "do not commit temporary review bundles" hygiene.
- Studio-runtime skill only if repo evidence shows forge3d Studio files or specs: immutable `RenderSpec`, prepared artifacts, sandboxed worker, and `RenderSpec -> forge3d.MapScene` compiler boundaries.

Audit method:
1. Build a quick map of repeated high-friction workflows in the repo.
2. Identify where future agents are likely to make expensive mistakes:
   - PyO3 class/function registration mismatches
   - wrapper/native API drift
   - feature-gated Rust functions missing from maturin features
   - shader layout/bind-group drift
   - hidden capabilities not exposed to Python
   - examples that do not validate the real render path
   - rendering quality claims not tied to measurable output
3. Use the existing `AGENTS.md` lessons as evidence, especially around PyO3 registrations, wrapper/native mismatches, TBN feature gates, rendering settings, and resource-pool wiring.
4. Rank skills by expected leverage:
   - How often this workflow repeats
   - How costly mistakes are
   - Whether a skill can prevent mistakes with a short file map and checklist
   - Whether existing repo docs/tests already cover the workflow
5. Compare proposed skills against the external workflow pattern finding above. Adapt high-fit patterns and reject drop-in autonomous workflows that would be unsafe for forge3d.
6. Create only the skills with clear positive ROI.

Report requirements:
Write `docs/forge3d-recommended-skills-audit.md` with:
- date
- inspected areas
- skill selection rationale
- how the external workflow pattern finding was applied, including which patterns were adopted, customized, or rejected
- created skills table with:
  - skill name
  - trigger/use case
  - why needed
  - evidence from repo
  - key validation commands
- rejected skill candidates table with:
  - candidate
  - reason rejected or deferred
- maintenance notes:
  - when to update each skill
  - what repo changes would make a skill obsolete

Validation before final response:
- Confirm every created skill has valid YAML frontmatter with only `name` and `description`.
- Confirm every skill name is lowercase hyphen-case.
- Confirm every created skill has Purpose, Workflow, Evidence To Read, Validation, and Boundaries sections.
- Read back the report and all created `SKILL.md` files.
- Run `git status --short`.
- Final response must include only:
  - report path
  - created skill paths
  - rejected/deferred count
  - any skipped checks or blockers
