## Prompt for ChatGPT 5.1 Codex Max (paste verbatim)

You are **ChatGPT 5.1 Codex Max**. You must implement a **minimal-change remediation** that fixes ONLY the failing requirements while **preserving all already-passing Milestone B/C/D results**. You must not “improve” anything not required. You must not handwave. You must produce the exact deliverables listed below, with exact file names and JSON schemas.

### 0) Hard rules (non-negotiable)

1. **Do not change** any code path that affects the following already-passing metrics unless explicitly required to fix the failing item:

   * Milestone B: sentinel, nonuniform, mode distinctness, artifacts JSON results.
   * Milestone C: mode25_metrics, normal angle errors, normal diff raw metrics.
   * Milestone D: blend curve, orbit file counts, temporal metrics, compare fade on/off.
2. All fixes must be **surgical**: constrain changes to the smallest possible surface area.
3. Every fix must be validated by **re-running the same metric scripts** that generated the JSON artifacts.
4. If you must adjust a threshold or constant, you MUST justify why it is required and show it does not break passing milestones.
5. Output must be deterministic:

   * Seed any stochastic process.
   * Pin config values in a single config file.
6. **No new output formats.** Only the required JSON files and any explicitly requested logs.

### 1) Current failing items (must be fixed)

#### (B.4) Attribution requirement fails

* Current failure: `mode24_vs_mode0 = 0.8594` must be **≤ 0.85**.
* You MUST bring `mode24_vs_mode0` to **≤ 0.85** while keeping all other Milestone B JSONs still passing.
* You MUST preserve B.4 ordering constraint: `hf_energy(mode23) < hf_energy(mode24)` remains true.

#### (C) Missing required output (outputs_present.json fail)

* Current failure: `outputs_present.json` is missing one required item (reported missing path in the JSON).
* You MUST make `outputs_present.json` pass by ensuring the required output exists at the expected location(s).
* You MUST NOT alter the already-pass normal metrics by changing normal computation.

#### (D4) Orbit integrity duplicates detected

* Current failure: `orbit_integrity.json` has `no_duplicates: false` and `pass: false`.
* You MUST eliminate duplicates so that `orbit_integrity.json` passes with `no_duplicates: true`.
* You MUST preserve all already-passing Milestone D metrics including:

  * Blend curve constraints
  * Orbit frame counts
  * Temporal metrics thresholds
  * compare_fade_on_off.json ratios remain within threshold and do not worsen unexpectedly.

### 2) Required repo workflow (you must follow exactly)

You MUST implement the remediation in the following phases, in order, without skipping.

#### Phase A — Baseline snapshot (do not modify anything yet)

1. Locate the scripts/entrypoints that generate each of these JSONs:

   * `metrics_attribution.json` (Milestone B)
   * `outputs_present.json` (Milestone C)
   * `orbit_integrity.json` (Milestone D)
2. Run the existing pipeline to reproduce current failures and confirm they match:

   * B.4: mode24_vs_mode0 ≈ 0.8594
   * C: outputs_present missing_count == 1
   * D: orbit_integrity pass == false due to duplicates
3. Save the raw console logs to:

   * `reports/remediation/baseline_run.log`

Deliverable A1:

* `reports/remediation/baseline_run.log` (plain text)

#### Phase B — Fix B.4 (attribution) with minimal change

Objective: reduce `mode24_vs_mode0` to **≤ 0.85** while leaving everything else unchanged.

Constraints:

* You may ONLY touch the attribution pipeline portion that feeds mode24/mode0 comparison. You may not touch mode26/mode27 or sentinel generation.
* You may not change the metric computation itself unless it is clearly wrong (and if you do, you must demonstrate all other B metrics still pass).

Implementation requirements:

1. Identify the cause of mode24 leaking too much high-frequency energy relative to mode0 (e.g., fade weight, incorrect blend, missing clamp, wrong channel).
2. Apply the smallest correction that:

   * reduces HF energy for mode24 slightly OR increases baseline appropriately without affecting other modes.
3. Add a single config knob (if needed) under an existing config system, defaulting to current behavior, and set it only for mode24 attribution run. Do not create many knobs.

Validation requirements:

* Re-run the full Milestone B JSON generation and show:

  * `metrics_attribution.json` now passes:

    * `mode23_vs_mode0` still ≤ 0.70
    * `mode24_vs_mode0` now ≤ 0.85
    * `hf_energy(mode23) < hf_energy(mode24)` still true
  * All other Milestone B JSONs remain PASS (unchanged or within existing thresholds).

Deliverables B:

* Updated `metrics_attribution.json`
* `reports/remediation/b_fix_report.json` with schema:

  ```json
  {
    "b4_before": {"mode23_vs_mode0": 0.0, "mode24_vs_mode0": 0.0},
    "b4_after":  {"mode23_vs_mode0": 0.0, "mode24_vs_mode0": 0.0},
    "changed_files": ["..."],
    "config_changes": {"key": "value"},
    "regression_check": {
      "metrics_sentinel_pass": true,
      "metrics_nonuniform_pass": true,
      "metrics_mode_distinctness_pass": true,
      "metrics_artifacts_pass": true
    }
  }
  ```

#### Phase C — Fix Milestone C missing output (outputs_present)

Objective: make `outputs_present.json` report **pass: true** with `missing_count: 0`.

Constraints:

* You must not change normal computation or rendering.
* This is a **pipeline hygiene** fix: file path, writing location, naming, or stage ordering.

Implementation requirements:

1. Identify why the expected output path is missing (wrong directory, wrong filename, conditional skip, non-created folder, etc.).
2. Ensure the output is generated and written exactly where `outputs_present` expects it.
3. Add a guard: if output writing fails, the pipeline must error loudly (non-zero exit).

Validation requirements:

* Re-run the Milestone C pipeline and ensure:

  * `outputs_present.json` pass: true, missing_count: 0
  * `mode25_metrics.json`, `normal_angle_error_summary.json`, `normal_diff_raw.json` remain PASS and unchanged beyond noise.

Deliverables C:

* Updated `outputs_present.json`
* `reports/remediation/c_fix_report.json` with schema:

  ```json
  {
    "missing_before": {"pass": false, "missing_count": 0, "missing": []},
    "missing_after":  {"pass": true,  "missing_count": 0, "missing": []},
    "changed_files": ["..."],
    "path_resolution": {
      "expected_paths": ["..."],
      "actual_written_paths": ["..."]
    },
    "regression_check": {
      "mode25_metrics_pass": true,
      "normal_angle_error_pass": true,
      "normal_diff_raw_pass": true
    }
  }
  ```

#### Phase D — Fix Milestone D duplicates (orbit_integrity)

Objective: produce orbits with **no duplicate frames** per the integrity checker.

Constraints:

* You must not change the blend curve table or any temporal metric thresholds.
* You must not reduce orbit length (still 36 frames per set as currently passing).
* You must not “hide” duplicates by removing checking. The checker must remain intact.

Implementation requirements:

1. Determine the root cause of duplicates (e.g., orbit camera path repeats identical pose, wrong time step, rounding, integer frame index reuse, overwritten frame files, sorting collision, same RNG seed reuse, etc.).
2. Fix it in the orbit generation stage by ensuring:

   * Unique frame index -> unique camera pose/time parameter
   * Unique output filename mapping
   * Stable deterministic ordering
3. Implement a preventive assertion:

   * During orbit generation, compute a lightweight per-frame signature (e.g., camera pose hash + frame index) and assert uniqueness; fail fast otherwise.

Validation requirements:

* Re-run Milestone D generation and validate:

  * `orbit_integrity.json` now pass: true, no_duplicates: true
  * `orbit_files.json` remains PASS with all 36-count sets
  * `temporal_metrics_synth.json`, `temporal_metrics_by_frame.json`, `compare_fade_on_off.json`, `blend_curve_table.json` remain PASS (or improve, but must not regress beyond thresholds).

Deliverables D:

* Updated `orbit_integrity.json`
* Updated `orbit_files.json` (only if contents must change due to correct regeneration; counts must remain 36)
* `reports/remediation/d_fix_report.json` with schema:

  ```json
  {
    "duplicates_before": {"pass": false, "no_duplicates": false, "duplicates": []},
    "duplicates_after":  {"pass": true,  "no_duplicates": true,  "duplicates": []},
    "root_cause": "string",
    "changed_files": ["..."],
    "uniqueness_assertion": {"enabled": true, "signature": "string"},
    "regression_check": {
      "blend_curve_pass": true,
      "orbit_files_pass": true,
      "temporal_metrics_by_frame_pass": true,
      "temporal_metrics_synth_pass": true,
      "compare_fade_on_off_pass": true
    }
  }
  ```

#### Phase E — Final proofpack summary

After all fixes:

1. Re-run all Milestone B/C/D pipelines needed to regenerate the relevant JSON artifacts.
2. Produce a single summary JSON that states PASS/FAIL per milestone and points to the updated JSON files.

Deliverable E:

* `reports/remediation/proofpack_summary_final.json` with schema:

  ```json
  {
    "milestone_b": {"status": "PASS", "jsons": ["..."]},
    "milestone_c": {"status": "PASS", "jsons": ["..."]},
    "milestone_d": {"status": "PASS", "jsons": ["..."]},
    "changed_files": ["..."],
    "notes": ["no regressions observed"]
  }
  ```

### 3) Output format constraints for your response

In your final response to me, you MUST output:

1. A **file tree** of all changed/new files under `reports/remediation/`
2. A **single bullet list** of code files changed, with 1-line reason each
3. A **verification checklist** showing the PASS/FAIL status you observed (must be all PASS)

No extra commentary. No tables. No “should”. Only factual statements and paths.

---
