# Moonshot Round 3 Grounding Fixes

**Date:** 2026-07-12

## Goal

Correct every issue identified in `AUDIT-REPORT-round-3.md` so prompts 23–42 all satisfy the existing groundedness rubric.

## Changes

- `24-diluvium.md`: scope the zero-hit claim to hydrodynamic solver code and remove the false comparison to PROBATUM's safe accumulation division.
- `25-orogenesis.md`: scope the zero-hit claim to geomorphic flow/erosion implementation, excluding unrelated test prose.
- `33-probatum.md`: replace the false live-defect claim with an explicit seeded unsafe shader fixture; update the NaN-guard inventory without brittle exact counts. Preserve the proof, ablation, and zero-suppression requirements.
- `34-terminus.md`: remove `arbitrary` from the zero-hit expression while retaining the absence of property/fuzz tooling.
- `35-pulsus.md`: acknowledge the constant `frame_time_ms` placeholder and require the task to create a dedicated real-hardware CI lane for the 60-minute on/off soak.
- `36-compendium.md`: scope the codec inventory to terrain/DEM paths and acknowledge the unrelated unimplemented KTX2 Zstd branch.
- `README-round-3.md`: align EUCLIDEA consumer wording and state that PULSUS creates its hardware CI gate.
- `AUDIT-REPORT-round-3.md`: re-audit the corrected text, record applied edits, and update the verdict table to 20 grounded.

## Constraints

- Do not alter moonshot objectives, measurable thresholds, non-goals, or implementation scope except where necessary to make a previously ungated acceptance check genuinely gated.
- Do not edit source code, tests, CI, fixtures, or prompts already rated grounded.
- Do not claim tests/builds/renders passed; this remains a source audit.

## Verification

- Search for every stale phrase called out by the audit and confirm it is gone.
- Confirm prompts 23–42 still have their required sections and curated Cargo command.
- Confirm the audit report contains exactly 20 findings and reports 20 grounded, 0 mostly-grounded, 0 flawed.
- Run `git diff --check` or an equivalent no-index whitespace check for ignored prompt files, then inspect scoped status/diffs without touching unrelated worktree changes.
