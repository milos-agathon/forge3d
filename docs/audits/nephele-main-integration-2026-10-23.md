# NEPHELE main integration — 2026-10-23

Prepared main `d6b9e556586cceddb9c08f16ffc35f7d71fb4d88` on release parent `ca9ad2cec1d9b890e9f50178dd279c4b322dd781` for PR #172. This is a reviewable preparation, not release readiness. The protected values remain unchanged until Milos approves this exact batch. No physical capture or local GPU work occurred.

Milos authorized: “Yes — prepare the main integration before capture, push it to PR #172, then stop with one exact approval batch (shader pin, contract changes, f32 ledger regeneration, any certificates). Release still requires physical green plus the 1500² DEM frame-time measurement.” The earlier approval was “approved for the pin and range — but before committing, remove the per-frame readback from the viewer diagnostics path. Before release: decouple the surface cache from DEM resolution and measure frame time on a realistic DEM (≥1024²); report or enforce convergence instead of a fixed 21 orders.” That earlier pin/range approval is already consumed; it does not approve this combined source.

## Exact approval batch

- Terrain source pin: `0x3afe_8ce7_1c2d_1a4a` → `0x17ad_c1fb_40ed_8189`.
- Inventory: 1931 / `eb22339b6de627ada37d1f0068d7306bfa152ae339997d7c6ee106dee09951a8` → 1936 / `8f9f378ce8e27a98e9b99961073ae70128584485778adf1dc2c597c464c0cee7`. The ledger preserves both parent inventories and every historical freeze; the scanner and exclusions are unchanged. All exact added/removed sites and old/new file SHA-256 values are in the JSON.
- Contracts: import main's three already reviewed CONCORDIA unproven records, integrate five existing test-contract files, and apply the source-builder analyzer proposal. No further NEPHELE cloudlight range change is proposed. Preserve the existing mesh-unavailable test.
- Import 23 certificates and 24 pixel products byte-for-byte from main; every old/new hash and path is in the JSON. These are main's existing products, not refreshed attestations for this candidate. No signing or golden regeneration is proposed.

The text changes are in `nephele-main-integration-2026-10-23-protected.patch` (SHA-256 `c132d1db70832e5ac89838bfeb4038a8f5fb3e8c92bdd0d2fa3626cf6cd55e07`). The 58-file exact manifest is `nephele-main-integration-2026-10-23.json`. Neither the patch nor those protected values is applied. Product bytes are obtained by exact main commit/path and verified against the manifest, without duplicating binary goldens in the patch.

## Implementation and evidence

Both camera conventions survive the merge: NEPHELE Y-up normals and shadow bounds, and main's centered north-up shared frame. Main's new interactive calls receive the existing environment arguments and their real HDR target format. All other imported files and their reasons are listed in the JSON.

The original 1500×1500 Gore Range DEM is now downloaded once by the shared fixture job, hash checked, packaged separately and restored by NEPHELE. Its verified download is 9,012,888 bytes, SHA-256 `bcacfdfabefc7ef4e7b9a9cefe12375b491fe1769218ad3c3ddd3ae30f9d93bc`. The four-file Python bundle and existing test assertions are unchanged.

Clippy acceptance and formatting pass. Workflow checks: 44 passed, zero failed/skipped (`FORGE3D_NO_BOOTSTRAP=1`). Current f32 checks: 20 passed / 13 failed because the prior approved freeze is deliberately retained. The proposed ledger passed 36 functional checks; its only first-run failure was a draft approval-word assertion. After replacing that new assertion with provenance/review presence, the affected exact-parent-restoration check passed. Existing approval assertions remain intact. Four camera-mode checks pass; the first incorrect core filter ran zero and is not counted as proof. Four frame-graph and three media shader checks executed successfully. The current source-pin test executed and failed with the exact new value above, as expected.

The proposed analyzer executes the current pure Rust shader builders and both boolean variants where applicable. Naga validates 356 assemblies, zero unresolved sources, zero compute comparison-sampling calls (117.69 seconds). The current analyzer remains blocked until this proposal is approved. Environment: Windows, Rust 1.97.1, Python 3.13.14, worktree-owned Cargo target; Rust feature and environment details are in the JSON. No new installed-wheel SHA claim is made.

## Metrics and remaining release proof

| Metric | Last physical 68cc04e9 | CPU forecast, not a new capture |
|---|---:|---:|
| B1 | 0.7128 fail | 1.0000 |
| Shaft SSIM | 0.9916 | 0.99235 |
| G4a | 0.7110 fail | 0.99615 |
| Medium response | 0.3216 | 0.37997 |
| G4b relative error | 0.1027 | 0.00762 |

All margins, backcasts, sensitivity, G1/G2 statistics and unmeasured G5/G6 physical status are retained in the JSON. No new numerical transport change or forecast was made during this integration. The directional fields and convergence remain unchanged from ca9ad2ce: 19/19 CPU fixed-point orders, independent surface-cache dimensions, no fixed 21-order limit.

After approval, release still requires exact-head installed-wheel checks, NVIDIA Vulkan physical green, the 1500² DEM timing, and full acceptance CI. Imported main certificates do not certify this combined source. The protected-main-only certificate refresh path is a concrete remaining governance gap; no candidate signature or unknown future certificate value is proposed. D3D12, Metal and WASM are ABSENT. Reference RNG overlap is reported and unchanged. Proposed steering fixes are listed in the JSON and were not applied.

All old/new hashes in this batch identify canonical Git blob bytes; text proposals use LF. Windows working-copy CRLF bytes are not used as prospective Git hashes. The preparation was first pushed at d590df3e; this audit correction changes no production source or proposed behavior.
