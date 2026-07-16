# Open Pull Request Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every pull request that was open in `milos-agathon/forge3d` on 2026-07-16, preserving only work that is not already in `main`, and merge the surviving work in a dependency-safe order.

**Architecture:** Treat current `origin/main` as authoritative and reconstruct surviving changes from it. Do not merge stale aggregate branches or resolve conflicts by taking an old branch wholesale. Preserve unique GIS work in a rewritten #109, preserve unique LITTERA work in a reconstructed #111, and close the remaining PRs only after their useful commits or documents have been accounted for.

**Tech Stack:** Git/GitHub, Rust/wgpu/PyO3, Python/pytest, maturin, repository source gates, GPU golden/certificate CI.

## Global Constraints

- Audit snapshot: 2026-07-16, `origin/main` at `dc09e5b2` after merged PR #122.
- Open PR scope is exactly #109 through #121 (13 PRs).
- Current `main` already contains CENSOR through #106, MENSURA M-01–M-05/M-07 through #108 plus merge `20829e39`, and full M-06 through #122.
- No current open PR has a human review. #110–#121 have no CI checks. #109 has a failed aggregate because its required M-06 NVIDIA/Vulkan job was cancelled.
- A PR is not mergeable merely because its old stacked base reports `CLEAN`; every retained PR must be rebuilt or rebased onto current `main`, retargeted to `main`, and run exact-head CI.
- Preserve current-main implementations when resolving stale-branch conflicts. In particular, do not replace the stronger #122 anchoring contracts with #109/#118 versions.
- Do not weaken numeric, image, conformance, or zero-skip thresholds to make a branch green.
- Do not accept known failures as “pre-existing” when the PR changes the same subsystem.
- `docs/prompts/` is locally present but git-ignored. A PR body must cite a tracked plan/spec that reviewers can inspect.
- Global retained-PR gate: `git diff --check`, `cargo fmt --check`, `cargo forge3d-clippy`, `maturin develop --release`, the curated Rust matrix from `CLAUDE.md`, `python scripts/ci_pytest_lane.py -q`, relevant GPU/golden/certificate tests, and green required `CI Success` on the final head.

---

## Executive Decision

Only two existing PR numbers should ultimately merge:

1. **#109**, rewritten as the remaining Rust GIS closure only.
2. **#111**, reconstructed as a clean LITTERA core PR on current `main`.

The other eleven open PRs should close without merge:

- **Superseded/already in main:** #110, #112, #113, #114, #117, #118.
- **Already absorbed by updated #111:** #119, #120.
- **Unreviewable stale aggregate; salvage first:** #121.
- **Mixed/orphan documentation bundle:** #115.
- **Contradicts tracked example/gallery plans:** #116.

After #111, LITTERA still needs one new closure PR for the original hard curved-text, three-surface, hidden-dependency, and cross-platform gates. #111 must not claim full LITTERA completion before that PR is green.

## Evidence Snapshot

| PR | Live state | Checks/reviews | Decision |
|---|---|---|---|
| #109 | `CONFLICTING/DIRTY`, base `main`, 130 files | Aggregate failed; required M-06 hardware job cancelled; 0 reviews | Rewrite GIS-only, then merge |
| #110 | Clean only against stale `codex/censor-closure` | No checks, 0 reviews | Verify current main, salvage any unique fix, close |
| #111 | Clean only against stale `codex/censor-closure`, 145 files | No checks, 0 reviews | Reconstruct on current main, restore spec gates, then merge |
| #112 | Clean only against stale base | No checks, 0 reviews | Functionality is in main; extract tracked design doc, verify, close |
| #113 | Clean only against stale base | No checks, 0 reviews | Functionality is in main; verify live GPU contract, close |
| #114 | Clean only against stale base | No checks, 0 reviews | Changes are already in current main, close |
| #115 | Clean only against stale base | No checks, 0 reviews | Close mixed bundle; republish wanted docs separately |
| #116 | Clean only against stale base, deletes 46,836 lines | No checks, 0 reviews | Reject and close |
| #117 | Clean only against #110 | No checks, 0 reviews | Superseded by main/#109, close after inventory |
| #118 | Clean only against #110 | No checks, 0 reviews | Superseded by #122, close |
| #119 | `CONFLICTING/DIRTY`, three commits behind #111 | No checks, 0 reviews | Updated #111 already contains the surface; close after #111 proof |
| #120 | `CONFLICTING/DIRTY`, three commits behind #111 | No checks, 0 reviews | Updated #111 already contains the surface; close after #111 proof |
| #121 | `CONFLICTING/DIRTY`, 609 files, +708,745/-13,728 | No checks, 0 reviews, empty body | Salvage LITTERA/docs, then close |

## Per-PR Audit and Definition of Done

### PR #109 — Remaining Rust GIS plan and viewer anchoring

**Tracked authority:** `docs/carto-engine/rust-gis-implementation-plan.md`; `docs/carto-engine/mensura-m06-world-coord-anchoring.md`; current-main #122 evidence.

**What is missing now:** #109 overlaps #122 on 92 paths and its M-06 implementation is obsolete. Thirty-eight paths are #109-only, but some of those are also M-06 support. The useful remainder is the GIS work: rasterization burn/merge semantics, boundary filtering/union/reprojection, remote OSM/Terrarium fetch policy, destination-CRS building ingestion, metadata-only `warped_vrt_info`, and error-policy refinements. The plan status table is internally stale and must be reconciled with its later checked items.

**Definition of done:** #109 is rebuilt from current `main`; every viewer/M-06 hunk now owned by #122 is absent; only tested GIS closure remains; the plan has one consistent status per capability; no unsupported format is advertised; focused GIS/API/remote local-server tests, global gates, and exact-head CI are green with no required skip or known failure.

### PR #110 — MENSURA geodesy engine

**Tracked authority:** Rust GIS plan M-01–M-03 and the MENSURA evidence already merged through #108/`20829e39`.

**What is missing now:** Nothing proven unique. Current main has the typed geodesy, projections, EGM96, Karney/geodesic behavior, stronger tests, and later fixes. The open PR's own verification excluded cross-PR conservation/height tests. Its 1,377-line hand-ported geodesic solver is also larger than the already locked `geographiclib-rs` path in current main.

**Definition of done:** Run the current-main six-win MENSURA verification (doctests, 10k conservation, EPSG examples/oracle, 20-point EGM96, geodesic residuals, typed height boundary). Move only a demonstrably missing fix into rewritten #109 or a tiny current-main PR. Then close #110 without merge, linking #108 and the verification evidence.

### PR #111 — LITTERA native text core

**Tracked authority:** extract and track `docs/superpowers/specs/2026-07-12-littera-design.md` and `docs/superpowers/plans/2026-07-12-littera-implementation.md`; source acceptance thresholds are summarized below.

**What is missing now:** The PR is not standalone: LITTERA Tasks 1–7 live in #121's ancestry. Its current tests weaken the required 12 px IoU from 0.995 to 0.90, Hausdorff from 0.5 px to 1.0 px, and 96 px SSIM from 0.999 to 0.985. The single-channel ablation does not prove failure of the 0.5 px bound. `test_text_three_surfaces.py` does not implement the required pairwise Delta E/SSIM gate. Curved-text Task 11 and the complete hidden-dependency/cross-platform Task 12 remain incomplete. The 145-file diff also mixes broad render timing and 22 golden/certificate changes that need explicit provenance.

**Definition of done for this PR:** Reconstruct from current main with only LITTERA foundation commits `702dd63e` through `bec49fdf` plus #111's four commits; exclude CENSOR and HDR ancestry. Carry the tracked LITTERA design/plan. Restore the exact source thresholds (IoU >= 0.995, Hausdorff <= 0.5 px, 96 px SSIM >= 0.999, and single-channel SDF must fail the Hausdorff bound); require >=200 exact HarfBuzz cases and >=2,000 lines from each bidi corpus; remove all known failures; preserve the already integrated CPU and SVG outline surfaces from the updated #111 head; regenerate only genuinely affected label goldens/certificates and pass protected signature/golden gates. Merge it as **LITTERA core**, not full LITTERA closure.

### PR #112 — HDR mood and EXR decoding

**Tracked authority:** `docs/superpowers/specs/2026-07-13-general-hdr-terrain-mood-design.md` (currently stranded in #121/#112 history and absent from main).

**What is missing now:** The code is already in current main through #122 (`environment_mood_tint`, `apply_luminance_preserving_tint`, `src/formats/exr.rs`, and `IBL.from_hdr` dispatch). The tracked design document is missing from main, and the PR body does not prove the no-default-feature EXR error path.

**Definition of done:** Extract the final design doc into a small current-main docs PR. On current main run the 23 focused tests without required skips, Rust F16/U32 shuffled-channel decoder tests, `.hdr/.rgbe/.exr`/unsupported-extension Python contract, and `cargo test --no-default-features` for the explicit EXR feature error. If green, close #112 as superseded; if not, open a minimal fix PR for only the observed gap.

### PR #113 — Hybrid terrain `sun_color`

**Tracked authority:** HDR mood design section 1.

**What is missing now:** The full call chain is already in current main through #122, including wrapper/stubs/native validation and direct/ReSTIR factoring. The open PR proved only 5 tests while 10 GPU tests skipped. It also has a direct `pyo3::types` import that violates the repository PyO3 bridge convention.

**Definition of done:** After the tracked HDR design is landed, verify on current main: wrapper and both stubs; exactly-three finite non-negative validation including zero; direct light `I*c`; ReSTIR raw `c` plus separate intensity; a live custom-color GPU render; unchanged legacy-default golden. Close #113 as superseded. Do not transplant its stale bridge import.

### PR #114 — CENSOR cross-platform hardening

**Tracked authority:** `docs/superpowers/specs/2026-07-10-censor-closure-design.md` and merged PR #106.

**What is missing now:** Nothing unique has been shown. Current main contains CRLF-to-LF WGSL normalization before hashing/compiling, descriptor-indexed LUT replication, the Python 3.10 feature-table parser, and truthful terrain-capability skips.

**Definition of done:** Run the focused certificate/hash, no-silent-degradation, BRDF certificate, and terrain availability tests on current main; retain the #106/#122 implementation if results differ. Close #114 as superseded.

### PR #115 — Research notes, reflections, and CI markers

**Tracked authority:** only the `UNRUN.toml` hunk maps to CENSOR slice 6; the remaining files have no single originating plan.

**What is missing now:** The COG UNRUN entry and bench contract already exist on main. The PR mixes source, test suppression, local-machine skill research, two unrelated specs, and an orphan rendering memo. Some citations are machine-local and not reproducible.

**Definition of done:** Close #115 without merge. If a document is still wanted, publish it in a separate docs-only PR after refreshing facts, removing machine-local citations, giving it an owner/use, linking it from the relevant index or implementation plan, and passing the docs build.

### PR #116 — Stop tracking examples

**Tracked authority:** no plan; it conflicts with `docs/audits/2026-07-07-08-examples-gallery-taste-honesty-audit.md`, whose T-08-07 requires curating/tracking documented examples rather than deleting them, and with CENSOR's test-accounting rules.

**What is missing now:** It deletes all 40 example files and 21 runner tests, converts surviving coverage to absence skips, and leaves docs, gallery regeneration, tutorials, API references, and planning evidence pointing at deleted files.

**Definition of done:** Close #116 unmerged. Any future removal requires an explicit ADR that replaces the current example/gallery and CENSOR contracts, updates every reference, defines where examples live, preserves essential behavioral tests, and proves clone/sdist/docs consistency. A skip is not replacement coverage.

### PR #117 — MENSURA GIS reproject/measure/CRS

**Tracked authority:** Rust GIS plan M-04–M-05 and current-main MENSURA evidence.

**What is missing now:** Its intended behavior is already in main and refined by #109. The PR admits a failing test that expects `BackendUnavailable`, but the authoritative contract requires `TransformFailed` for a parseable unsupported transform. The public `measure_geometries` compatibility middleman also needs a real compatibility justification or deletion.

**Definition of done:** Inventory only: confirm current main/#109 covers densified bounds, geodesic units, dateline topology, structured transform attributes, and the source gate forbidding silent `.ok()` suppression. Preserve only a unique tested fix in #109. Close #117 without merge.

### PR #118 — Earth-scale f64 world coordinates

**Tracked authority:** MENSURA M-06 and merged #122.

**What is missing now:** It is an older, weaker anchoring implementation. Current main has stronger epsilon/non-finite guards, a frozen per-frame camera, one deterministic rebase, all viewer consumers, stronger narrowing inventory, and required hardware evidence.

**Definition of done:** Compare only animation, Scene, and 3D-Tiles non-viewer deltas against current main. Preserve a genuinely absent behavior only through a tiny test-first current-main fix. Then close #118 as superseded by #122.

### PR #119 — LITTERA SVG outline export

**Tracked authority:** LITTERA Task 8 SVG surface.

**What is missing now:** The updated #111 already contains `_native_label_path`, native `svg_path_data`, real SVG `<path>` output, and three-surface tests. #119 is based on #111's first commit, conflicts with the remediated head, and would delete hundreds of newer lines if taken wholesale.

**Definition of done:** In reconstructed #111, prove Latin/Arabic/Devanagari SVG uses deterministic filled outlines, halo reuses the same path, no `<text>` is emitted, and outline errors do not panic. Once that exact-head evidence is green, close #119 as absorbed by #111.

### PR #120 — LITTERA MapScene CPU labels

**Tracked authority:** LITTERA Task 8 CPU surface.

**What is missing now:** The updated #111 already contains `_composite_text_mask`, native `shape`, and `rasterize_shaped_run`; #120 is stale and conflicts in both files. Its own report left seven required MapScene failures.

**Definition of done:** In reconstructed #111, prove no `_draw_text_fallback`, `ImageFont`, or `ImageDraw` text path remains; native multiscript shaping and rasterization are live; and the terrain-capable MapScene PNG suite has no required failure/skip. Then close #120 as absorbed by #111.

### PR #121 — `codex/censor-closure` aggregate

**Tracked authority:** CENSOR closure design, LITTERA design/plan, and HDR mood design—but these are three separate campaigns.

**What is missing now:** CENSOR is already merged through #106. The open PR is 95 commits behind current main, has 75 merge conflicts by `git merge-tree`, has no body/check/review, and mixes CENSOR, LITTERA, Unicode corpora, examples, HDR design, CI, and unrelated docs. Its useful unmerged content is the LITTERA foundation and the tracked LITTERA/HDR design documents.

**Definition of done:** Record the salvage inventory. Move LITTERA commits `702dd63e`–`bec49fdf` and the LITTERA design/plan into reconstructed #111; move the final HDR design into the docs-only extraction described under #112; prove CENSOR is already represented by #106/current main. Then close #121 without merge and retarget no child PR to it.

## Required Closure and Merge Order

### Phase 0 — Preserve evidence before closing stale branches

- [ ] **Step 1:** Record exact commit/file inventories for #109, #111, and #121.
- [ ] **Step 2:** Extract the tracked LITTERA design/plan and final HDR mood design from #121 history.
- [ ] **Step 3:** Verify current-main supersession gates for #110, #112, #113, #114, #117, and #118.
- [ ] **Step 4:** Close #115 and #116 unmerged with links to this audit and their violated/missing plans.
- [ ] **Step 5:** Close #110, #112, #113, #114, #117, and #118 after linking the current-main verification or salvage commit.
- [ ] **Step 6:** Close #121 only after LITTERA and HDR artifacts are preserved.

### Phase 1 — Rewrite and merge #109 first

- [ ] **Step 1:** Rebuild #109 from current `main` with only non-M-06 GIS closure files.
- [ ] **Step 2:** Remove all stale viewer/anchor/CI evidence owned by #122.
- [ ] **Step 3:** Reconcile `docs/carto-engine/rust-gis-implementation-plan.md` status rows and acceptance evidence.
- [ ] **Step 4:** Run focused GIS/API/remote tests and the global retained-PR gate.
- [ ] **Step 5:** Retarget #109 to `main`, require green exact-head `CI Success` and review, then merge.

### Phase 2 — Reconstruct and merge #111 second

- [ ] **Step 1:** Start from `main` after #109 and apply only the LITTERA foundation plus #111 commits.
- [ ] **Step 2:** Carry the tracked LITTERA design/plan; exclude CENSOR and HDR ancestry.
- [ ] **Step 3:** Restore the original LITTERA conformance thresholds and hard ablation.
- [ ] **Step 4:** Preserve the updated #111 CPU/SVG implementations that supersede #119/#120.
- [ ] **Step 5:** Review every changed golden/certificate and run the protected signature/golden gates.
- [ ] **Step 6:** Run focused shaping/bidi/MSDF/API/MapScene/SVG tests and the global retained-PR gate.
- [ ] **Step 7:** Retarget #111 to `main`, require green exact-head `CI Success` and review, then merge as LITTERA core.
- [ ] **Step 8:** Close #119 and #120 as absorbed, linking the passing #111 tests.

### Phase 3 — Finish LITTERA honestly

- [ ] **Step 1:** Open one focused current-main closure PR for LITTERA Task 11 and the missing Task 12 gates.
- [ ] **Step 2:** Require curved MapScene text with <=0.25 px normal deviation, upright reverse path, and correct RTL direction.
- [ ] **Step 3:** Require GPU/CPU/SVG pairwise Delta E 2000 <2 on >=99% of covered pixels and SSIM >0.99.
- [ ] **Step 4:** Require zero hidden text dependencies/fallbacks and identical cross-platform shaped hashes.
- [ ] **Step 5:** Claim full LITTERA completion only when all six original measurable wins pass without weakened thresholds.

## Final Merge Graph

```text
current main (#106 + #108/20829e39 + #122)
    |
    +-- rewritten #109 (GIS-only)
            |
            +-- reconstructed #111 (LITTERA core; includes #121 salvage and #119/#120 surfaces)
                    |
                    +-- new LITTERA closure PR (curved + hard three-surface/cross-platform gates)
```

The HDR design-doc extraction is documentation-only and may merge after #109 or #111, but it must land before #112/#113 are declared fully accounted for. No other open PR belongs in the merge graph.

## Self-Review Checklist

- [ ] Every one of the 13 open PRs has a disposition and definition of done.
- [ ] Every close-without-merge action identifies what must be preserved first.
- [ ] Every merge candidate starts from current `main` and has exact-head CI/review gates.
- [ ] #122 remains authoritative for M-06.
- [ ] #106 remains authoritative for CENSOR.
- [ ] LITTERA thresholds match the source spec and are not weakened.
- [ ] Example deletion is not accepted without a replacement policy and coverage.
- [ ] PR bodies cite tracked, reviewable planning documents.
- [ ] The final merge order is unambiguous: #109, then #111, then a new LITTERA closure PR.
