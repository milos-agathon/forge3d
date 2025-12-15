You are acting as a **principal real-time rendering engineer** (Rust + WGSL + WebGPU/wgpu + Python) working in the `forge3d` repo. You MUST fix the remaining unmet requirements while preserving the frozen production default.

## CRITICAL RULES

* You must fully read AGENTS.md to learn repo rules.
* Do not regress existing passing tests.
* After every edit: **run tests** and **re-render all required evidence images in the same session**. No “generated earlier” exceptions.
* Do NOT modify thresholds in any strict validation profiles to “make tests pass”.
* Preserve existing behavior as the default unless a new opt-in flag/mode is introduced.
* No handwaving: provide concrete diffs, exact files/lines, and reproducible commands with raw output.

---

## CONTEXT (WHAT IS ALREADY MET — MUST KEEP)

These must remain true:

1. HARD / PCF / PCSS are wired end-to-end and produce different output (hash/pixel-diff tests exist).
2. CLI no longer crashes for shadow technique values.
3. [SHADOW] technique=... logging exists.
4. Tests are green.
5. Decisions already taken and implemented: **S2 (fail-fast VSM/EVSM/MSM)** and **P1 (mesh-based perspective heightfield)**.

Do not break any of the above.

---

## WHAT IS STILL NOT ACCEPTED (THE REAL GAP)

Your current evidence **does not prove** the user’s real issue is solved:

* The user’s “Rainier looks flat” complaint is about the **actual Rainier scene**, not the synthetic step DEM.
* Shadow-technique evidence currently leans on **step_dem.tif** because Rainier “still looks similar across techniques” — that is precisely what we must fix (or explain with concrete, measured evidence).
* Evidence discipline is incomplete: any required probe images must be freshly regenerated and the md5/diff tooling must be fully reproducible (no placeholder “md5/diff script” line).

This milestone is not done until Rainier relief is clearly non-flat **under low-angle lighting** and we can prove it reproducibly.

---

## OBJECTIVE (DO NOT MISINTERPRET)

The renderer must produce **Blender-like relief** on the **Rainier** terrain under **low sun elevation (< 30°)** while honoring the user’s camera controls (theta/phi/fov) and without relying on “sun = camera + 90°” heuristics.

You must **find and fix root causes**, not recommend different angles.

---

## NON-NEGOTIABLE INVESTIGATION TASKS (DO THESE FIRST)

### I0) Confirm the Rainier scene is actually using mesh perspective mode

**Deliverable I0:** In the Rainier preset and CLI path, prove (via log + config dump) that the Rainier render is using the mesh/perspective mode you added — not silently falling back to screen-mode/fullscreen reconstruction.

* Add a single debug log line (guarded by `FORGE3D_DEBUG_CAMERA=1`) that prints:

  * `render_mode` (screen vs mesh) and the exact code path taken
  * camera eye/target, derived `view_dir` (normalized)
  * raw CLI values used (phi/theta/fov)
  * sun azimuth/elevation, derived `sun_dir` (normalized)
  * `dot(view_dir, sun_dir)`
  * DEM texel size (dx, dy) + elevation scale used in normal/mesh generation

### I1) Audit unit/scale correctness for Rainier DEM

Flat relief is often caused by **XY/Z unit mismatch** (degrees treated as meters, wrong texel size, wrong z-scale, wrong height normalization).

**Deliverable I1:** Identify and fix (if needed) at least one of the following (with code references):

* degrees vs meters mismatch
* swapped dx/dy
* radians/degrees mixup in camera or sun math
* incorrect normal derivation scale (slope too small → normals near-up → flat shading)
* mesh height scale not matching world XY scale

If Rainier DEM is geographic (lat/lon), you MUST either:

* (Preferred) enforce/auto-reproject to a projected CRS for shading/mesh scale, OR
* implement a correct meters-per-degree approximation based on latitude and use it consistently.

No silent behavior: warn/error clearly.

### I2) Add a Rainier-only relief metric (measured, not vibes)

Add a small deterministic analysis script that quantifies “relief” from an image **without manual judgment**.

**Deliverable I2:** Add `tools/relief_metric.py` (or similar) that outputs:

* luminance stddev over non-background pixels
* edge/gradient magnitude percentiles (e.g., p50/p90)
* optional “shadowed pixel ratio” if you have a shadow factor buffer in debug mode

This tool must take `--input <png>` and print a stable, parseable summary.

---

## IMPLEMENTATION REQUIREMENTS (STRICT)

### R1 — Rainier must show real relief under low sun elevation

You must produce a Rainier preset output that is obviously non-flat **without** “sun = camera + 90°” logic.

**Deliverable R1A (Preset):** Update or create:

* `presets/rainier_relief_low_sun.json` (or equivalent)

  * explicit `sun_elevation_deg` in **[10°, 25°]**
  * camera theta/phi/fov that is clearly perspective (not top-down)
  * explicitly uses mesh/perspective mode (no ambiguity)
  * add a comment block explaining the intent and constraints (low sun, no heuristic offset)

**Deliverable R1B (Output):**

* `examples/out/rainier_relief_low_sun.png`

**Acceptance (Measured):**

* `tools/relief_metric.py` must report:

  * luminance stddev ≥ a threshold you justify from baselines (and you must include baseline numbers)
  * edge magnitude p90 ≥ threshold
* You must provide the baseline measurements for:

  * the previous “flat” Rainier render
  * the new “relief” Rainier render
    so we can see objective improvement.

### R2 — Shadow techniques must be meaningfully different on Rainier (not just step DEM)

If Rainier still looks “similar”, you must either:

* fix it (preferred), OR
* prove via quantitative evidence why it is physically expected in that configuration and adjust the Rainier preset so differences become visible **without breaking defaults**.

**Deliverable R2A (Outputs):**
Render the *same Rainier scene* with:

* `examples/out/rainier_shadow_hard.png`
* `examples/out/rainier_shadow_pcf.png`
* `examples/out/rainier_shadow_pcss.png`

**Acceptance:**

* md5 hashes must differ
* pixel-diff counts must be non-zero and above a tiny floor (to avoid “1 pixel differs” flukes)

### R3 — Keep step DEM, but demote it to tests-only evidence

Step DEM is fine for deterministic tests, but it cannot be the primary justification for the user-facing Rainier fix.

**Deliverable R3:**

* If `step_dem.tif` is required for tests, it must be:

  * either generated deterministically by a script checked into `tools/` and produced during tests, OR
  * checked into an appropriate test asset location (not `examples/out/`)
* `examples/out/` must not be treated as a source-controlled dependency unless AGENTS.md explicitly allows it.

### R4 — Evidence reproducibility: no placeholders, no “generated earlier”

You must re-render **all** required evidence images in this session and include raw outputs.

**Deliverables R4:**

1. A committed script `tools/md5_and_diff.py` (or similar) that:

   * prints md5 for each image
   * prints pixel-diff counts for specified pairs
2. In your final report, include:

   * the exact command lines used
   * the raw stdout of md5/diff script
   * the raw stdout of `python -m pytest -q`

---

## REQUIRED EVIDENCE OUTPUTS (MUST BE FRESHLY GENERATED)

### E1 — Geometry probes (lighting-independent; regenerate all)

* probe_fov30.png, probe_fov60.png, probe_fov90.png
* probe_theta25.png, probe_theta75.png
* probe_phi0.png, probe_phi90.png

### E2 — Rainier relief (this is the real user problem)

* rainier_relief_low_sun.png

### E3 — Rainier shadow technique sanity

* rainier_shadow_hard.png, rainier_shadow_pcf.png, rainier_shadow_pcss.png

### E4 — Optional but strongly recommended (diagnostic)

* rainier_depth_probe.png (Rainier using the geometry-only depth/debug mode)

---

## ACCEPTANCE CRITERIA (MUST ALL PASS)

1. Rainier preset uses mesh/perspective mode (proven via debug log).
2. Rainier low-sun render is measurably higher-relief than prior baseline (relief metric tool output included).
3. Rainier HARD/PCF/PCSS outputs differ (md5 + pixel diffs).
4. All geometry probes are regenerated in this session and still differ as required.
5. VSM/EVSM/MSM remain fail-fast with clear error (no regressions).
6. `python -m pytest -q` passes.

---

## OUTPUT FORMAT (STRICT)

Return:

1. A short “What was wrong” section (max 6 bullets, each bullet must cite file/line or value).
2. A short plan (max 10 bullets).
3. List of edited files.
4. Exact commands run (every render + metrics + tests).
5. Raw output blocks:

   * relief metric output (baseline + new)
   * md5 + pixel-diff output
   * pytest output
6. Final test results summary.

No extra commentary.
