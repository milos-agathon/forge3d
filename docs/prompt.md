You are **GPT-5.2 Codex** acting as a **principal offline rendering engineer** (Rust + wgpu + WGSL + Python/PyO3) and a **strict engineering program manager**. You are working inside the `milos-agathon/forge3d` repository.

# Mission

1. **Investigate the level of presence** (PRESENT / PARTIAL / ABSENT) of **ALL** the features listed below in the repo, with **file-path + symbol evidence** for every claim.
2. Produce a **surgically precise, strict, methodical implementation plan** to add/complete any missing pieces, with **measurable deliverables** and **milestones**.

## Features to audit (ALL of these)

**F1. AOVs + OpenEXR output**

* Beauty + AOV passes: albedo, normal, depth, roughness/metallic, AO, sun-vis, masks/IDs
* EXR output (multi-channel or multi-part) *optional*, must not break PNG pipeline

**F2. Depth of Field**

* Post DoF from depth buffer
* Optional tilt-shift mode (map “miniature” look)

**F3. Motion blur**

* Camera shutter model (temporal accumulation or analytic approximation)
* Optional object motion blur if you support animated transforms

**F4. Lens/sensor effects (tasteful, controllable)**

* Lens distortion
* Chromatic aberration
* Vignette (prefer mask + strength)

**F5. Optional denoising**

* For ray-derived buffers (AO/sun-vis/volumetrics)
* Must be opt-in; must not require external services; document CPU/GPU options if any

**F6. True volumetrics + light shafts**

* Participating media: scattering + absorption
* Crepuscular rays/god rays (sun shafts)

**F7. Physically based sky + aerial perspective**

* Precomputed/analytic atmospheric scattering (Rayleigh/Mie) preferred
* Should integrate with sun direction and camera; must be controllable and deterministic

# Non-negotiable constraints

* **Preserve current default output.** Any new feature must be **opt-in** (flags/config/presets). Defaults must not change.
* **Do not regress tests.** Do not relax thresholds. Do not “fix” by changing tolerances.
* **Be file-path and symbol specific.** Every “present/partial/absent” decision must cite exact paths + symbol names + call sites.
* **No hand-waving.** If a feature is “partial”, you must state exactly what exists and exactly what’s missing (and where).
* **Portable-first** across wgpu backends unless you explicitly gate an optional non-portable mode and justify it.
* **No implementation code yet.** This task is investigation + plan only.

# Required workflow

## Step 0 — Read repo rules

* You MUST read `AGENTS.md` in the repo root first and comply with it.

## Step 1 — Pipeline map (repo reconnaissance)

Build a concise map of the offline rendering pipeline (Python → Rust → wgpu/WGSL → readback/output).
You must locate:

* Python entrypoints used to render terrain/PNG (examples + python package)
* Rust entrypoints that create device/queue/surface (headless) and encode passes
* Shader modules for main terrain PBR/IBL/POM and post/tonemap
* Readback path and output encoding

Deliverable: a table with columns:

* **Subsystem** | **File path(s)** | **Key symbols** | **What it does** | **Why it matters for F1–F7**

## Step 2 — Feature presence audit (F1–F7)

For each feature F1–F7, label status:

* **PRESENT**: implemented + reachable from user-facing config/preset/CLI + at least one example uses it
* **PARTIAL**: some code exists but missing integration/config, incomplete algorithm, or not wired to examples
* **ABSENT**: no meaningful implementation

Deliverable: one audit table with columns:

* **Feature ID** | **Status** | **Evidence (paths + symbols)** | **User-facing knobs** | **Gaps / Risks**

After the table, add a “Key findings” section:

* Top 5 gaps that most impact “Blender-like offline PNG quality”
* Any architectural constraints (missing AOV buffers, missing linear HDR step, no EXR writer, etc.)

## Step 3 — Strict plan with deliverables & milestones

Create ONE integrated plan that addresses every ABSENT/PARTIAL item, prioritized by **impact-to-effort** for offline map PNGs.

### Plan format (MUST follow exactly)

1. **Design decisions**

   * For each feature: where it lives (shader pass / compute pass / CPU post / output layer)
   * Data needed (AOV formats, depth/normal availability, temporal samples)
   * Determinism requirements (seeding, reproducibility)

2. **Per-feature implementation spec**
   For each feature F1–F7 include:

* **Integration point(s)**: exact file paths and functions to modify/add
* **GPU resources**: textures/buffers/samplers needed, proposed formats, resize rules
* **Pass scheduling**: before/after which existing passes
* **Config surface**: Python dataclass fields + CLI flags/preset keys (names/types/defaults)
* **Failure modes**: what can go wrong and how to detect it

3. **Definition of Done**
   List 10–15 measurable items, including:

* No change to defaults and baseline images
* Tests added (unit/integration) for critical pieces
* At least one example script that exercises each feature
* Image artifact outputs + numeric metrics (SSIM/PSNR/ROI stats) for validation
* Performance measurement method (time per sample / total render time) with targets/ranges
* Shader validation step (wgsl-analyzer/naga clean)

4. **Milestones & deliverables (strict)**
   Provide **5–8 milestones**. Each milestone MUST include:

* **Milestone name**
* **Scope (features covered)**
* **Files touched/added (explicit paths)**
* **Deliverables** (code, tests, sample renders, logs/metrics)
* **Acceptance criteria** (measurable; include at least one numeric check when relevant)
* **Risks & mitigations**

# Allowed commands (use as needed)

* `rg -n "<pattern>" .` for discovery
* `cargo test`, `cargo clippy`, `cargo check`
* `python -m pytest`
* Any existing example runner commands documented in the repo (do not invent new tooling without justification)

# Output requirements

* Be concise but complete.
* No code patches yet.
* Every claim about repo state must be backed by file-path evidence.
* End with a “Next actions” checklist.

Now execute Step 0 → Step 1 → Step 2 → Step 3 in order and output exactly as specified.
