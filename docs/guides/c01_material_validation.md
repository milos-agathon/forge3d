# C01 validation on current main

The reviewed C01 integration is in [draft PR #308](https://github.com/milos-agathon/forge3d/pull/308).
It is based on main `f80dba3d613af33290689e49b9de18b167e49988` and preserves
the original candidate `4a251d086cf95df62274963a489c6af76fcca092` in its
worktree. The isolated checkout retains the [validation receipts and commands](../../artifacts/c01/review/REPORT.md).

The November 15 validation covers mixed terrain and raster channels and
native cloud and screen-space exports. On the original draft's RTX 3070/Vulkan
runtime `d3ae07ab5c43980ebb1951e8cd06e7ca0cc84cd089c6c7901aeb37c40fb2d391`,
379 C01 cases, 309 one-shot pipeline cases and all 17 selected
physical publication cases passed with zero skips. These runs used
`WGPU_BACKEND=vulkan` and `FORGE3D_NO_BOOTSTRAP=1`. Physical publication
also set `FORGE3D_RUN_GPU_ANAMNESIS=1`. All 23 default PNGs matched clean
main byte for byte. The computed terrain pin `0x1e4e931cfb13547f` passed
the strict check, and all seven shader assembly tests and eight Python
shader proofs passed.

Milos accepted the prepared original and snow revision for step 3. Soil
and rock blend together, and he still needs the legend to distinguish
snow from the summit zone. The October 19 decision waived hands-on
authoring and comparative effort for this step.
[Owner assessment](c01_human_trial.md).

C01 remains partial. F6 requires an exclusive-host timing run and measured
pass-scoped hardware texture-fetch counts. The 23 signed shader-identity
checks still fail against the existing certificates. The
[certificate workflow](../../.github/workflows/certificate-refresh.yml)
runs on protected `main` after an authorized integration. Completing
that gate requires separately authorized signing, review and promotion
of the certificates, and a passing rerun. Independent physical
cross-backend qualification remains absent.

The sections below retain historical runtimes, revisions and acceptance
records.

# Integrated C01 validation — 2026-10-19

The single local C01 commit is rebased onto fetched `origin/main` `797ca32a`.
The current candidate and private runtime are recorded in
`artifacts/c01/review/owner-assessment-2026-10-19T200921Z/candidate.txt` and
`artifacts/c01/review/local-integration-2026-10-19T190454Z/runtime.txt`.
Native SHA-256 is `16e325f371eb1b158f2d8a123695710a831720ab1fd5e852100dfe7b10f186cf`;
all 1,418 captured native inputs stayed unchanged during the rebuild. The documented
Rust rewrite computed terrain pin `0xdc9478469d595255`; other pins, committed
goldens and certificates were not refreshed.

On Python 3.13.14 / RTX 3070 / Vulkan / NVIDIA 610.60, the source runner passed
369 C01 cases and 8 shader proofs in a separate process, both with zero skips.
Environment: `WGPU_BACKEND=vulkan`, `FORGE3D_NO_BOOTSTRAP=1`,
`FORGE3D_REPO_ROOT` is this worktree and `FORGE3D_C01_NATIVE` is the new private
`native/_forge3d.pyd` under the evidence directory. Formatting, canonical Clippy,
3 source pins, the computed-pin dry check, 7 shader assemblies and 72 renderer
Rust tests passed; the filtered tests actually executed.

All 22 default PNGs match the approved historical base in encoded bytes and
pixels. Compatibility has 22/1/6 failures, with no new failure IDs. Historical
22/1/9 is retained: the three resolved extra failures rejected untracked C01 test
files, which the authorized commit now tracks. The same three source-only checks
pass on a clean `797ca32a` checkout; that comparison claims no clean-main native
or renderer qualification. Full logs and the exact cause are in
`artifacts/c01/review/local-integration-2026-10-19T190454Z/compatibility-delta-cause.json`.

820 raw readbacks are byte-identical to the prior reviewed candidate; six changed
clipmap fly-through readbacks were checked numerically and visually. Their maximum
channel deltas are 2–23 codes, with no visible holes; the streaming test passed.
This is asynchronous streaming evidence, not a determinism qualification.

C01 remains **partial** for Milos's assessment of the prepared renders, exclusive-machine F6 timing
with actual hardware texture-fetch counters, and final approved integration.
Hosted CI, other physical backends, OIDN, terrain DOF and reference rerenders are
unrun/unqualified; portability remains ABSENT. Viewer compositor and world-scale
mesh material texture work remain deferred. The sections below retain the prior
historical evidence; this section supersedes their integration/runtime status.

On 2026-10-19 Milos confirmed that his assessment of the prepared results is
sufficient for step 3; hands-on authoring and comparative effort are no longer
required for that step. His actual assessment is pending. The current acceptance
record is in `artifacts/c01/review/owner-assessment-2026-10-19T200921Z/acceptance-change.json`.
Earlier references below to required hands-on effort are historical and superseded.
F6, final approval and the deferred/unqualified capabilities above are unchanged.

# C01 material validation

## Review follow-up — 2026-10-19

The independent review accepted all four owner decisions with no new defects.
Its two warning observations are now corrected: MapScene attributes ignored-key
warnings to the external call, and `dataclasses.replace` does not warn again for
unchanged copied address values. New declarations, assignments and replacements
that choose a different non-default address value still warn; values and
serialization are preserved. These paths are tested on Python 3.13.14.

Five new cases failed before the fixes. All **369 C01 tests and eight separate
shader proofs pass with zero skips**. Failure IDs still equal base (22 gate,
one offline, nine extra), and all 22 default PNGs are byte-identical. Native
98455954 and all 1,394 native input files are unchanged. Evidence and exact
commands: [follow-up report](../../artifacts/c01/review/review-followup-2026-10-19/REPORT.md).

The [human-trial guide](c01_human_trial.md) now selects the final retained runtime
and explicit Python executable; its help command succeeds. This is preparation,
not a completed human trial.

Integration is prepared only: observed origin/main 5b0a23c3 is 33 commits ahead.
After text line-ending normalization, 15 overlapping files have four conflicts;
three non-pin resolutions are saved as unapplied artifacts. The remaining
combined-shader pin must be computed after explicit owner approval. No commit,
rebase, merge, pin refresh or signing occurred. These tests qualify the unmerged
checkout; post-integration gates are still required.

C01 remains **partial** for Milos's hands-on trial, F6 exclusive-host timing and
hardware texture-fetch counters, and integration. Viewer, deferred world-scale
textures, OIDN and other backend/CI lanes remain deferred or unqualified; no
capability is inferred from the passing Vulkan tests. Earlier records below are
historical where superseded by this section.

## Owner decisions 1–4 — current contract, 2026-10-16

Milos accepted the legacy-building samples>1/HDR output change and the clamped edges on scaled direct renders. The building-output acknowledgement is **settled**. Both changes are recorded in the [changelog](../../CHANGELOG.md) and [release notes](c01_release_notes.md).

`SamplingSettings.address_u/v/w` are retained, serialized unchanged and have no effect. Non-default `ClampToEdge`/`MirrorRepeat` values warn with `DeprecationWarning` on construction or assignment; the historical `Repeat` defaults remain silent. Material tiling is unchanged. The fields will be removed in a future release.

MapScene uses fixed MSAA 1 / render scale 1, as base did. Recipe `cli_params.msaa_samples` and `render_scale` are preserved but ignored with `UserWarning` whenever present. They no longer reject otherwise supported native provenance or add a second one-shot visibility draw. The direct renderer owns scaled/MSAA testing; native offline accumulation still forces MSAA 1. Existing offline/HDR provenance restrictions are unchanged. A future recipe-controlled sampling feature would need measured quality benefit and provenance support.

The scaled building matrix now calls `render_offline` directly, alongside the vegetation matrix. Separate mesh/fullscreen MapScene cases retain fixed-sampling silhouette/filtering coverage. Paint/vector MapScene controls compare ignored-key output with the key-free recipe, assert the fixed native parameters and verify one visibility draw per render. See [decision evidence](../../artifacts/c01/review/merge-decisions-1-4/REPORT.md).

C01 remains **partial** for the human trial, roadmap F6 hardware counters/exclusivity and integration. These owner decisions do not authorize a merge, commit, push, golden/certificate refresh or signing. OIDN, other adapters/backends, viewer and the previously listed unavailable lanes remain unqualified.

## Qualification before the owner sampling decision — historical

The evidence below is retained history. Its recipe-controlled MapScene sampling behavior, scaled MapScene claims, provenance rejection and pending building-output acknowledgement are superseded by decisions 1–4 above. Its native/framebuffer/scatter proofs remain finite evidence; current checks are linked in the decision report.


N1–N3 from [the independent review](../../artifacts/c01/review/fixes-n1-n3/CONTRACT.md) are corrected on NVIDIA RTX 3070 / Vulkan / NVIDIA 610.60. C01 remains **partial**: human trial, roadmap F6 hardware counters/exclusivity, merge and **Milos's acknowledgement of the legacy-building samples>1/HDR output change** remain outstanding.

## Corrections and independent evidence

- **N1:** visibility references always use unjittered MSAA 1 / scale 1 parameters while beauty retains requested settings. The eight paint/vector × samples 1/4 × buildings absent/present cases now render with requested MSAA 4 / scale 0.5. Unsupported scaled/multisampled provenance raises a clear `ValueError` before native resources are created; supported provenance retains its existing one-shot contract. Red crash evidence: `red-valid-0/1/2/4/5/6.log`, `red-isolated-oracle-2.log`; early-rejection red: `red-valid-8.log`. The samples-4/building-present cases are passing compatibility controls.
- **N2:** the conservative guide margin is one output pixel when scale is 1 (no frame resize), otherwise `max(1, ceil(1.5/scale))`, accounting for half an internal pixel of jitter plus one internal pixel of linear-filter support. The measured scale-0.5 margin is three pixels. Both beauty and AOV framebuffer blits now use ClampToEdge; material texture tiling and filter selection are unchanged. The unused framebuffer address translator was removed. No WGSL changed. Before-code `red-isolated-oracle-0.log` detects opposite-edge wrap, and `red-isolated-oracle-1.log` detects building samples in guides. The new oracle compares native scatter-on/off HDR with shadows disabled solely to isolate visible fragments from legitimate terrain-shadow differences; the original shadow-enabled matrix still checks passthrough and real filtering. All 16 requested cases have **zero guided pixels with visible scatter contribution**, exact HDR passthrough on those samples, nonempty guides, and changed terrain outside the exclusion. An independent measured local-support assertion also rejects opposite-edge wrap in the direct fixtures. See [matrix measurements](../../artifacts/c01/review/fixes-n1-n3/regression-evidence.json).
- **N3:** the mesh building fixture now has substantial visible terrain; every guide is nonempty and atrous must actually change terrain. There are **eight effective configurations**, with 16 requested combinations. All eight requested MSAA 1/4 pairs are byte-identical controls because native offline accumulation forces `msaa_samples=1`; the scale axis changes output. Scatter-denoiser metadata now records requested/effective MSAA separately and includes terrain margins in passthrough coverage. The exact metadata test was updated to this corrected wording. `red-13.log` proves the original empty-guide defect. Paint on/off equality remains an **integration consistency** check using a shared reference, not an independent visibility proof. Independent native coverage/hue/fault tests are separate.

The broader pre-compatibility comparison caught five exact-dictionary failures from globally added MSAA metadata fields (`worktree-offline.xml`). The fields are now restricted to scatter-denoiser results; existing tests are unchanged. Final result files use the `-compat` suffix and all required failure sets equal base.

Rejected setup attempts are retained: `red-0..3` used incorrect sample-centre bounds, and `red-valid-3` included an unintended legend after material binding. The corrected fixture removes that legend; the preserved before-code samples-4/building-present control passes (`red-isolated-oracle-3.log`). Early mesh scatter-on/off comparisons included real cast-shadow changes; the final isolated silhouette oracle separates them. These rejected interpretations are not evidence of a silhouette defect.

## Preserved F1–F6 contract

The original edge probes and `ring_analysis.py` were rerun. Ring-1 has zero pixels changing by more than 8 codes and maximum 0; the legacy-building maximum is 0 against the required base maximum of 4. Those ring metrics are supplemental: the independent scatter-disabled oracle above supplies the leakage proof.

Protected EXR paint still comes only from final pixels that equal saved exact paint. Title/label pixels retain HDR radiance, and untouched paint equals linear(swatch). The original overlay probe reports **0 overlaid mask pixels**. The scaled-building on/off fixture retains 11,902 identical protected pixels, subject to the integration-consistency limit above.

Native-visible paint coverage remains 100% in the three ridge fixtures; dropped-row and UV-shift fault probes each reject all three. The ≥99.6% hue and immediate-boundary-only requirements are unchanged. At 12°/1-cell, 62 disagreements against 64 allowed leave a **2-pixel margin**, requiring requalification per adapter/driver. This does not qualify portability.

The previous 1,298 changed pixels included mixed scatter silhouettes; they were not all terrain. Edge changes were possible at scale 1 as well as 0.5. Historical `fixes-r1-r4/base-*` launch records name rejected native `5975822d`, but their explicit runner actually loaded `1ddb7a4c`, as annotated in the prior correction. Old logs remain unchanged. Prior F1–F6 completion claims about two-pixel support, 16 effective cases and actual offline MSAA4 are superseded here.

## Conservative terrain-margin cost

Matched default-shadow scatter-disabled renders are checked against retained HDR. Terrain whose HDR is identical with scatter disabled, yet is excluded from guiding, is unnecessary passthrough. A clean scatter-disabled atrous render measures the filtering correction forgone there. This includes noise/detail; it is **not ground-truth noise MSE**. The denominator is independently retained native-visible terrain. Both vegetation and MapScene buildings are measured; duplicate requested-MSAA4 controls add no effective measurements. [Raw data and four-panel previews](../../artifacts/c01/review/fixes-n1-n3/noise-cost.json).

| Case (effective MSAA 1) | Extra terrain passthrough | Forgone RGB correction RMS | Maximum |
| --- | --- | --- | --- |
| vegetation-mesh-msaa1-scale1.0 | 1146/17956 (6.38%) | 3.673 | 26 |
| vegetation-screen-msaa1-scale1.0 | 727/39721 (1.83%) | 0.737 | 4 |
| vegetation-mesh-msaa1-scale0.5 | 2890/17956 (16.09%) | 3.320 | 21 |
| vegetation-screen-msaa1-scale0.5 | 1567/39721 (3.95%) | 0.611 | 3 |
| legacy-buildings-mesh-msaa1-scale1.0 | 862/20065 (4.30%) | 1.874 | 14 |
| legacy-buildings-screen-msaa1-scale1.0 | 254/7148 (3.55%) | 0.511 | 3 |
| legacy-buildings-mesh-msaa1-scale0.5 | 2495/20065 (12.43%) | 1.592 | 10 |
| legacy-buildings-screen-msaa1-scale0.5 | 453/7148 (6.34%) | 0.507 | 3 |

## Validation and runtime

The **19-file C01 cohort: 343 passed, 0 failed, 0 skipped** ([XML](../../artifacts/c01/review/fixes-n1-n3/cohort-final-compat.xml)). Shader proofs ran in a separate process: **8 passed, 0 failed, 0 skipped** ([XML](../../artifacts/c01/review/fixes-n1-n3/shader-proofs.xml)). Tests run through `artifacts/c01/run_source_tests.py`. The separate shader process predates the final metadata-only compatibility correction; its native/shader inputs are unchanged and its evidence is reused. Rust: fmt and strict clippy passed; renderer 72/72, pins 3/3, shader assembly 7/7, computed pin dry-run 1/1. Pin remains `0xd5cf_1a32_4d74_6c08`; no refresh/re-signing occurred.

Fresh base is `git archive d81c3e2b python/forge3d` plus native `1ddb7a4c9a600fc4891d15dde8455cbbe97331174a5a20056b731076700a1375`. The eight-suite failure IDs equal base (22); all 34 selected offline/denoise modules show equal failure sets. Existing golden, clipmap-budget and CI/provenance failures remain disclosed. [Failure IDs](../../artifacts/c01/review/fixes-n1-n3/failure-comparison.json), [module inventory](../../artifacts/c01/review/fixes-n1-n3/inventory.json).

| Group | Base P/F/S | Candidate P/F/S | New failures |
| --- | --- | --- | --- |
| gate | 219/22/2 | 219/22/2 | 0 |
| offline | 376/1/7 | 376/1/7 | 0 |
| extra | 203/9/0 | 203/9/0 | 0 |

All **22 default recipe PNGs are byte-identical to fresh base**, encoded files and decoded pixels ([proof](../../artifacts/c01/review/fixes-n1-n3/all22-default-proof-final.json)). Readbacks: 446 unchanged, 37 changed and 62 unmatched/new; every changed/new capture was checked numerically and visually ([records](../../artifacts/c01/review/fixes-n1-n3/readback-comparison.json)). Matrix and noise-cost previews were also inspected. Finite visual evidence does not establish other adapters or workloads.

Worktree: `C:\Users\milos\.codex\worktrees\c01-geospatial-materials\forge3d` (uncommitted, base `d81c3e2b`). Python: `C:/Users/milos/AppData/Local/Programs/Python/Python313/python.exe` (3.13.14). Environment: `WGPU_BACKEND=vulkan`, `FORGE3D_NO_BOOTSTRAP=1`, `FORGE3D_REPO_ROOT=C:\Users\milos\.codex\worktrees\c01-geospatial-materials\forge3d`, `FORGE3D_C01_NATIVE=C:\Users\milos\.codex\worktrees\c01-geospatial-materials\forge3d\artifacts\c01\review\fixes-n1-n3\native\_forge3d.pyd`. Native SHA256: `984559542b8695bad5a14386dec4bd83571ce484cd471404817349c77eec40c2`. Candidate source snapshot: `C:\Users\milos\.codex\worktrees\c01-geospatial-materials\forge3d\artifacts\c01\runtimes\d0bd1c395268280e059a14b81f9d89689f399363ef7e959878ade8ec1d5732e9`. Build features and all 1394 input hashes are in [native identity](../../artifacts/c01/review/fixes-n1-n3/native-identity.json) / [input manifest](../../artifacts/c01/review/fixes-n1-n3/native-inputs.json); inputs were unchanged during the build. Only the recorded four Rust files differ from the prior private native input manifest.

Every validation command has argv, relevant environment, exit code and full log in [COMMANDS.md](../../artifacts/c01/review/fixes-n1-n3/COMMANDS.md) / [commands.json](../../artifacts/c01/review/fixes-n1-n3/commands.json). Each pre-existing touched file has a new `before/` snapshot. Prior uncommitted C01 work is preserved. No commit, push, dependency, golden, certificate, determinism-hash refresh, pin refresh or signing was performed.

## Roadmap scope and remaining limits

Only the six C01 objects are spliced into a fresh backup of `D:/forge3d/docs/moonshot-roadmap.html`, with raw JSON round trips, `<` escaping, unchanged states/statuses and identical non-target spans. The validator reports **VALID atlas/work/research/view**. N4's later C05/WF changes belong to other work and are preserved; the scope proof applies to this edit-time backup, not to a claim that the live file never changes.

**Unrun/unqualified:** native OIDN; Metal; DX12; WASM; hosted CI; viewer; terrain DOF; Blender/rayshader reference renders; the broader 94-module sweep. Cross-backend qualification remains ABSENT. Human trial, roadmap F6 counter/exclusivity evidence, merge and Milos's legacy-building output acknowledgement remain open.

Reusable steering proposals only: independent leak oracles must isolate cast shadows; matrix axes must demonstrate native effects; guide fixtures must contain filtering work. Keep sample-centre terrain bounds and remove incidental furniture after binding. No AGENTS.md, skill or registry change was applied; promoted skill targets were absent.

## Earlier qualification history — superseded



# C01 material validation

Historical R1–R4 correction record: the independent F1–F6 review found unresolved R1/R2 defects. The corrections and new acceptance evidence are recorded in `artifacts/c01/review/fixes-f1-f6/REPORT.md`. The earlier verdict is superseded. F6: the older base command records retained a launch variable pointing to rejected native 5975822d, but the baseline runner actually loaded 1ddb7a4c; the individual records now carry explicit runtime-selection annotations. Old logs are unchanged.

R1–R4 are implemented and verified on the RTX 3070/Vulkan offscreen route. C01 remains **partial** for the human trial, F6 and merge. All prior uncommitted C01 work is preserved.

## Runtime and native checks

Private native: `C:\Users\milos\.codex\worktrees\c01-geospatial-materials\forge3d\artifacts\c01\review\fixes-r1-r4\native-final\_forge3d.pyd`. SHA256: `055da50a00af607062f1a588ceb1d9f5ed696ca5b5a00f0f103d9a0ea6dacc1f`.
All 1394 recorded native inputs rehash identically to the preserved owner-qualified native. No Rust or WGSL change remains in this correction. The feature list and native source selection are in [native-identity.json](../../artifacts/c01/review/fixes-r1-r4/native-identity.json).

Terrain pin remains `0xd5cf_1a32_4d74_6c08`. No shader, determinism pin, committed golden, signed certificate, dependency, commit or hosted state was changed. During investigation the computed pin dry run, 3 source-pin checks, 7 shader-assembly checks, formatting, clippy and 72 focused renderer Rust tests passed. An experimental fullscreen depth-write approach was privately rebuilt, then rejected when the existing C01 depth-contract test failed; those native changes were completely reverted. Its binary is not the acceptance runtime. One earlier stale compile was cancelled after its source changed. Native source and input hashes are restored to the original qualified build, and all final Python/GPU acceptance uses that native.

Python is the requested Python 3.13 executable. GPU test environment: `WGPU_BACKEND=vulkan`, `FORGE3D_NO_BOOTSTRAP=1`, `FORGE3D_C01_NATIVE=<private native above>`, `FORGE3D_REPO_ROOT=<actual C01 worktree>`. The source runner now also sets `PYTHONPATH` to its selected snapshot for child GPU probes, correcting two false skips from child imports of an unrelated package. Real rasterio is imported first. Base Python was freshly extracted with `git archive d81c3e2b python/forge3d`, using native SHA256 `1ddb7a4c9a600fc4891d15dde8455cbbe97331174a5a20056b731076700a1375`.

## Corrections

**R1 — HDR radiance.** EXR replacement is restricted to `protected_mask`. Non-C01 cloud shadows leave every HDR value intact; mixed protected/cloud/HDR output has exact linear swatches on protected pixels and original radiance elsewhere. The metadata assertion is covered. Tests failed on the original code, then passed on the correction.

**R2 — native scatter denoising.** MapScene builds the terrain-depth equality mask for every nonempty building batch. Direct `render_offline` detects uploaded scatter through the native memory report. Both routes share output-centre guide coverage, including MSAA/internal-scaling inputs. The native identity attachment excludes all actually visible scatter fragments; this is essential for fullscreen's established Always comparison without depth writes. The strict depth contract and native depth policy remain intact. At this stage the tests proved only centre-owned scatter and sky passthrough; the later F1 review found mixed scatter silhouettes still filtered. The original regressions failed before implementation. Final tests cover legacy buildings at samples=4 plus direct vegetation in both mesh/fullscreen modes, at MSAA 1/scale 1 and MSAA 4/scale 0.5. The initially used colour-difference test mask also covered terrain; the 1,298 residual changes were outside centre-owned scatter, but included mixed silhouette samples; the earlier terrain-only interpretation was incorrect (independent review F1/F4). See [mask diagnosis](../../artifacts/c01/review/fixes-r1-r4/legacy-mask-diagnosis/measurement.json); the final test uses actual native identities and verifies exact passthrough.

The approved native offline backend is retained. **Legacy building outputs at samples>1/HDR now match samples=1 geometry. This default-output change still needs Milos's acknowledgement.**

**R3 — native triplanar motion.** All 48 colour views at 1280×720 and both separate native-disabled/repeat-scale control renders use `MapScene.render`. MapScene forwards the existing native `TriplanarSettings` through `LightingPreset.settings['triplanar']`; absence preserves the previous default. The fixture uses native layers alone: C01-authored rasters intentionally require opacity=1 and occlude native albedo, so mixing them would defeat this proof or break their palette contract. That contract remains unchanged. Every native albedo layer uses a photographic source, including the former flat snow placeholder. Native rock/snow use [Rock 3](https://polyhaven.com/a/rock_3), and native grass/dirt use [Brown Mud Leaves 01](https://polyhaven.com/a/brown_mud_leaves_01), under Poly Haven's [CC0 asset licence](https://polyhaven.com/license), checked 2026-10-16. Asset paths and hashes are retained in [configuration.json](../../artifacts/c01/review/fixes-r1-r4/triplanar/configuration.json).

Only the separate projection-weight diagnostic calls the native renderer directly: it reads sampler XYZ weights rather than presenting a product colour frame. All three axes are present, and disabling native albedo or changing its repeat changes visible pixels on every axis. C01-authored textures remain **ground-planar**; this evidence makes no claim of C01-authored XYZ texture sampling.

Every view was inspected at original size; the inspection record is in [evidence.json](../../artifacts/c01/review/fixes-r1-r4/triplanar/evidence.json). The [lossless movie](../../artifacts/c01/review/fixes-r1-r4/triplanar.mp4) contains all 48 source RGB frames, verified byte-for-byte after decoding. Texture visibility uses the review's 5×5 luminance-standard-deviation threshold of 2 display codes, on fully covered neighbourhoods, avoiding silhouette contrast. Across all 48 frames:

| Native dominant axis | Fraction below 2 codes, range | Median local std, range |
| --- | --- | --- |
| X | 6.46%–9.03% | 3.169–3.515 |
| Y | 4.79%–7.60% | 3.367–3.875 |
| Z | 1.66%–2.65% | 4.397–4.718 |

This finite fixture demonstrates visible native textures through the blend and motion. It does not establish universal material quality or hardware fetch counts.

**R4 — independent native class oracle.** The ridge test compares protected red/blue paint with native physical sampling of the same categorical raster. The predeclared 99.6% limit comes from the independent review's observed 99.61–99.94% agreement. Every discrepancy must also be an immediate native class-boundary pixel. The old test passed with deliberately shifted UV; the replacement rejected that same fault.

| View/pattern | Hue agreement | Disagreements | Nonboundary disagreements |
| --- | --- | --- | --- |
| 40°, 4-cell | 99.926099% | 12 | 0 |
| 12°, 4-cell | 99.944054% | 9 | 0 |
| 12°, 1-cell | 99.614596% | 62 | 0 |

## Required validation

The complete 18-file C01 cohort plus the new regression module passed: **321 passed, 0 failed, 0 skipped**. `tests/test_shader_proofs.py`: **8 passed, 0 failed, 0 skipped**. These zero-skip claims use the environment above.

Fresh base/candidate comparisons cover all eight requested gate suites and all 34 tracked modules containing offline/denoise references or explicitly requested by the owner. Counts below are passed / failed / skipped; full case IDs, errors and commands are retained.

| Group | Base P / F / S | Candidate P / F / S | New failures |
| --- | --- | --- | --- |
| gate | 219 / 22 / 2 | 219 / 22 / 2 | 0 |
| offline | 376 / 1 / 7 | 376 / 1 / 7 | 0 |
| extra | 203 / 9 / 0 | 203 / 9 / 0 | 0 |

The eight-suite failure sets are equal. Offline/denoise groups introduce no new failure IDs. Existing failures are not waived or changed. See [failure-comparison.json](../../artifacts/c01/review/fixes-r1-r4/failure-comparison.json).

All **22 default recipe PNGs are byte-identical to freshly rendered base**, including both file bytes and decoded pixel bytes; no golden or certificate refresh occurred. See [all22-default-proof.json](../../artifacts/c01/review/fixes-r1-r4/all22-default-proof.json).

All 36 changed and 62 newly present captured readbacks were numerically compared and visually inspected through 38 unique retained previews; identical raw array pairs reuse their inspection. The other 447 captured readbacks are byte-identical. Depth previews have a documented shared contrast stretch; raw arrays are unchanged. See [readback-comparison.json](../../artifacts/c01/review/fixes-r1-r4/readback-comparison.json), which lists arrays, tests, hashes, differences, numeric ranges and inspected images. Unmatched readbacks remain identified rather than being counted as unchanged. Vector/label edge changes, newly visible direct scatter, and rock/sky differences relative to base remain visible from the pre-existing C01 changes; no native source change remains in this correction.

Passthrough is defined at native unjittered output pixel centres, as in the requested depth/identity mask. Before F1 was corrected, mixed silhouette pixels changed at both scale 1 and scale 0.5. The original tests asserted only native identity-selected pixel centres and missed these sample-coverage edges; their passthrough claim did not cover the silhouettes.

Every executed validation command has argv, relevant environment, working directory, exit code and full log in [COMMANDS.md](../../artifacts/c01/review/fixes-r1-r4/COMMANDS.md) and [commands.json](../../artifacts/c01/review/fixes-r1-r4/commands.json). Expected red failures, setup failures and the cancelled stale build remain visible.

## Roadmap delivery

The live `D:/forge3d/docs/moonshot-roadmap.html` reports **VALID atlas/work/research/view** under the required validator. The six C01 target objects retain their requested states and C01 stays partial. See [validator log](../../artifacts/c01/review/fixes-r1-r4/roadmap-valid.log) and [final raw-span scope proof](../../artifacts/c01/review/fixes-r1-r4/roadmap-final-scope.json).

The pre-edit backup also failed on C05. Milos explicitly authorized a scoped C05 roadmap repair. C05's seven capability claims were restored from the existing frozen review at commit `53f8282b11067a37dda5bb77743ceabcc697b6cc`; all 20 restored references/excerpts were verified against seven actual Git files. Its separate 2026-10-16 local implementation report remains unchanged. The linked preparation and Landscape cards had also replaced their frozen assessments with unpinned local reports. Their pinned assessments were restored at research revision `a5e37efd17ed9c5a2cb0a94c7314876d0aa93c30`, preserving the complete local reports separately. The Landscape card's pinned review status is **none**, while its separate local implementation status remains **partial**; those are different review contexts. No local C05 file is presented as evidence from either pinned commit.

Every mutation has a preceding backup, an exact per-object JSON round trip and less-than escaping. Apart from the original six C01 objects and these three directly linked C05 objects, every raw byte is unchanged. Repair details: [C05 capability](../../artifacts/c01/review/fixes-r1-r4/c05-roadmap-repair.json), [preparation workflow](../../artifacts/c01/review/fixes-r1-r4/c05-workflow-roadmap-repair.json), [Landscape extension](../../artifacts/c01/review/fixes-r1-r4/c05-landscape-roadmap-repair.json). No C05 implementation code or test was modified or newly qualified by this repair. One pre-write scope assertion failed before mutation because the component counters already matched; that log remains retained alongside the final successful repair.

## Remaining scope and steering

- C01 stays partial: Milos's human trial, exclusive-host/hardware-counter F6 evidence and merge are outstanding. Existing Blender/rayshader silhouette/class-edge discrepancies remain disclosed in the earlier review; those reference lanes were not rerun for this correction.
- Physical cross-backend qualification is ABSENT. Metal, DX12, WASM, hosted CI, interactive viewer C01 composition, terrain DOF and native OIDN qualification were not exercised as correction acceptance lanes. Metal absence and the opt-in viewer skip remain recorded separately from the zero-skip cohort.
- Deferred world-scale mesh textures and vector/building normal/roughness work remain outside C01. No commit, push, PR, deployment, signing or dependency change was performed.
- Reusable steering gaps: use native identity coverage for fullscreen Always draws instead of changing physical depth semantics; use actual surface ownership to isolate scatter, since a colour-difference mask can also include terrain changes; propagate runtime snapshots to child GPU probes through `PYTHONPATH`. Proposed render-validation guidance only; no agent contract or skill was modified. Manifest targets under `D:/forge3d/.agents/skills/` are absent in this checkout, so no user-global or quarantined skill was used as fallback authority.

Changed implementation/test files: `python/forge3d/map_scene.py`, `python/forge3d/offline.py`, `tests/test_c01_review_gpu.py`, new `tests/test_c01_review_regressions.py`. The local source-test runner, validation guide, sequence generator, six live C01 roadmap objects and three separately authorized C05 roadmap objects are updated; all other pre-existing C01 changes remain intact. Rust/WGSL are byte-identical to this turn's starting C01 sources.

## Prior C01 qualification — historical record

The following predates the R1–R4 corrections. Its prior triplanar sequence is superseded by the evidence above.

# C01 material authoring validation

## Final local report under Milos’s decisions

All eight colour combinations are implemented and locally exercised. C01 remains **partial and unmerged** because the human trial and performance proof are incomplete. This report supersedes the historical reports archived with the evidence. No production dependency, commit, PR, committed golden or signed certificate was changed.

### Per-criterion verdicts

| Requirement | Verdict and observed limit |
| --- | --- |
| F1: one map/recipe | met — lit terrain/raster/vector/building colours and exact protected colours coexist. Class membership is independent of relief; relief combined with lighting is declared cartographic shading. |
| Native visibility | met — same-frame terrain/scene Depth32 and optional winning building batch ID; terrain pixel-centre class rule, no class blending. Buildings, ridges and sky occlude terrain/fills; elevated strokes use scene depth. |
| Protected buildings | met — each authored feature owns a native identity batch; exact display swatches on visible pixels, lit neighbours remain lit. Overlapping buildings and terrain occlusion are tested. |
| Physical vector paint | met — pixel-centre ground rasterization, linear-sRGB straight-alpha source-over into terrain albedo before native lighting. Protected polygons use cell centres; protected strokes/outlines use the declared half-coverage rule and complete swatches. |
| Offline/HDR/denoise | met for exercised paths — unjittered same-camera/geometry visibility reference plus exact paint after resolve, tone mapping and denoise. Exact PNG means uint8 display swatches; exact EXR means their float32 linear-sRGB values, retaining radiance elsewhere. Terrain guide AOVs are not scene-wide; visible buildings/sky pass through. |
| F2 debug/corner | met — product `debug_mode=0`; explicit 8×8 terrain-covered corner checks and original red tests pass. |
| Scatter correctness | met for exercised mesh/clipmap/AOV/offline routes — Less+depth write; fullscreen retains overlay. All 15 requested before/after pairs inspected; a thin light fringe in one legacy outline is retained as an observed coverage difference, without a parity/improvement claim. The isolated patch is reverse-checkable. See downstream-depth-review.json: fog/water run before scatter, contact samples DEM, NumPy SSAO uses DEM, and terrain DOF has no executing pass. |
| Atrous black sky | met — base samples=4 plain-terrain defect reproduced; invalid/zero-normal sky passes through each iteration and cannot contaminate surface neighbours. Separate reverse-checkable patch and CPU/GPU regressions. |
| F3 normal effect | met for controlled 20° north/east fixtures — same direction as real tilted geometry, about ±6 display codes. Supplied probe measures +4.81/-5.00 on terrain; its old 4.17% terrain crop diluted the average to ~0.21. This is not universal asset/light quality certification. |
| F5 tiling/motion | met for retained workloads — full 48 nominal/grazing views, adjacent/halo continuity, persisted world phase/seed, and all 48 native triplanar material-boundary views inspected. Actual geometric XYZ projection is regression-tested; phase-varied ground preparation and native periodic triplanar are distinct. |
| F6 timing/fetches | partial — repeated matched samples and all warmups retained. Owner declared the interval ready and our queue was stopped; external Python/browser CPU activity and unassigned GPU activity prevent an exclusive-host claim. Fetches are inferred. |
| F7/reference comparison | partial — actual 97 frames/tool (48 nominal+48 grazing+revision), source/geometry/camera/assets/size recorded and tolerances fixed before comparison. Blender projected controls pass; strict silhouette/class-edge equivalence does not. Human expertise/effort/preference are unverified. |
| Legend | met — one attached 11-entry legend, each meaning-carrying binding has its own caption; proposals and art direction are identified. |
| Preview/final/F8 | met for declared offscreen paths — real Reykjanes data, all eight combinations,1280×720 preview (samples=1) and final (samples=4/HDR/atrous), same-camera full sequences and revision inspected. Viewer excludes the Python precise compositor and is concretely out of this authoring path. |
| Human effort | unverified — Milos must perform [c01_human_trial.md](c01_human_trial.md); script timing is not human effort. |
| Deferred channels | recorded owner scope decision — wall/roof world textures and vector/building normal/roughness remain open in MAP-MESH-MATERIAL-CHANNELS. |

Roadmap acceptance order:

| Item | Verdicts |
| --- | --- |
| SPEC-GEOSPATIAL-MATERIALS | 1 met; 2 met; 3 met; 4 met; 5 met; 6 met; 7 partial |
| MAP-MATERIAL-MEANING | 1 unverified; 2 met; 3 met; 4 partial |
| CAND-RELIEF-STYLES | 1 met; 2 met; 3 met |
| CAND-MATERIAL-TILING | 1 met; 2 met; 3 partial |

SPEC 7 and MAP 1 require actual human work; MAP 4 includes that effort. Tiling 3 explicitly requests measured fetches, so inferred source-level calls do not close it. C01 atlas and research card remain partial.

### Verification

311 C01 tests, 8 isolated shader proofs and 10 supplied probes pass, with zero pytest skips in those cohorts. Format, cargo forge3d-clippy, executed focused Rust tests, shader assembly/source-pin/proof and read-only determinism gates pass. Rust tests actually execute: renderer 72, scatter 25, mesh 1, decode 2, assembly 7, source pins 3, rewrite-check 1 and determinism lint 1. Exact selectors/features are in the command records.

| Existing-test lane | Base failed / passed / skipped / errors | Worktree failed / passed / skipped / errors | Failure set |
| --- | --- | --- | --- |
| gate | 22 / 219 / 2 / 0 | 22 / 219 / 2 / 0 | subset of base |
| other | 15 / 1428 / 38 / 0 | 9 / 1434 / 38 / 0 | subset of base |

All 94 additional tracked affected modules ran on each binary. Per-module logs retain exact Python/native identities; the final compatibility correction is rechecked by the complete eight-suite gate and full C01/probe rerun, while unchanged early-module evidence is reused. The eight requested suites are a separate gate. Complete failure IDs, skip reasons, every observed readback difference and all 15 image pairs are in `base-vs-worktree-final.json` and `readback-review/`. Observed arrays: base 3087, worktree 3138; equal arrays 3051; tests with a readback difference 23. There are 36 changed array hashes across 20 tests and 51 added readbacks in 3 tests. Five additional changed-test IDs are float-HDR staged failure/recovery cases; ordinal alignment can differ when reads are added. Complete raw records are retained; positional hash differences do not imply semantic equivalence between different passes.

All 22 default recipe PNGs are identical in pixels **and complete file bytes**. The 22 existing committed-golden failures are also present on base (21 pixel SSIM, 1 shader-identity); no committed image or signed certificate is refreshed to make them disappear. Explicit NVIDIA/Vulkan pixel selection runs separately. Existing platform/capability selector skips remain visible; the zero-skip claim applies only to the named C01/shader/probe cohorts.

Ruff: 27 changed Python/stub files are clean. The wider 32-file check retains 128 diagnostics versus 132 on base, with no new semantic diagnostic (numeric source-line relocations normalized; raw diagnostics retained). The new public exports are placed before executable statements; existing late imports and other pre-existing lint are retained. No rule was suppressed. git diff --check and both isolated patch reverse checks pass.

### Native identity, commands and environment

Base commit: `d81c3e2b5fb90c809d5f79a695ce98ee411b8eac`. Native SHA256: `055da50a00af607062f1a588ceb1d9f5ed696ca5b5a00f0f103d9a0ea6dacc1f`. Wheel SHA256: `99dcc19758cc4f7fc6f787f19e5b762e289610bf1509a00055aea1d4beb42593`. All 1394 native build inputs still match the build manifest.

Private binary: `C:\Users\milos\.codex\worktrees\c01-geospatial-materials\forge3d\artifacts\c01\review\owner-decisions\native-triplanar\_forge3d.pyd`. Python runtime: `C:\Users\milos\.codex\worktrees\c01-geospatial-materials\forge3d\artifacts\c01\runtimes\9f03ddca3b55957585294401df33d53cd874fb99541b13995b510f04325d4d3f`.

Python 3.13.14; Windows; RTX 3070 (8192 MiB), NVIDIA 610.60/Vulkan; Ryzen 9 5900X (12 cores / 24 threads). Environment: WGPU_BACKEND=vulkan, FORGE3D_NO_BOOTSTRAP=1, FORGE3D_C01_NATIVE=the private binary above, FORGE3D_C01_ARTIFACT_DIR=owner-decisions/cohort-release. Inherited FORGE3D_REPO_ROOT=D:/forge3d/main is explicitly retained. Build/test PYO3_PYTHON and CARGO_TARGET_DIR are recorded.

Features: `extension-module,weighted-oit,enable-pbr,enable-tbn,enable-gpu-instancing,enable-staging-rings,enable-inverse-pt,copc_laz,cog_streaming,gis-remote,geos-topology,atmosphere-bake,shader-contract-asserts,enable-globe,splat-fusion`.

Exact build argv:

```json
{
  "argv": [
    "maturin",
    "build",
    "--release",
    "--locked",
    "--target-dir",
    "C:\\Users\\milos\\.codex\\worktrees\\c01-geospatial-materials\\forge3d\\artifacts\\c01\\review\\native-target",
    "--out",
    "C:\\Users\\milos\\.codex\\worktrees\\c01-geospatial-materials\\forge3d\\artifacts\\c01\\review\\owner-decisions\\wheels3",
    "--interpreter",
    "C:\\Users\\milos\\AppData\\Local\\Programs\\Python\\Python313\\python.exe"
  ],
  "exit_code": 0,
  "environment": {
    "PYO3_PYTHON": "C:\\Users\\milos\\AppData\\Local\\Programs\\Python\\Python313\\python.exe",
    "CARGO_TARGET_DIR": "C:\\Users\\milos\\.codex\\worktrees\\c01-geospatial-materials\\forge3d\\artifacts\\c01\\review\\native-target"
  }
}
```

Validated outputs: [final still](../../artifacts/c01/review/owner-decisions/qualified/final/frame-0000.png), [nominal sequence](../../artifacts/c01/review/owner-decisions/qualified/final-nominal.mp4), [grazing sequence](../../artifacts/c01/review/owner-decisions/qualified/final-grazing.mp4), [triplanar sequence](../../artifacts/c01/review/owner-decisions/qualified/triplanar.mp4).

Every actual command array, selected environment, exit code and full log is retained under `artifacts/c01/review/owner-decisions/`:

- `native-build-final-command.json`
- `triplanar-gate-commands.json`
- `final-python-commands.json`
- `final-comparison-commands.json`
- `final-remaining-commands.json`
- `final-qualification-commands.json`
- `offline-fix-commands.json`
- `reference-final-commands.json`
- `offline-call-fix-command.json`
- `timing-owner-ready-run.json`

### Performance samples and limits

timing-evidence.json / material_preparation_samples:

| Phase | Case | Wall seconds |
| --- | --- | --- |
| warmup | baseline | 2.621800 |
| warmup | corrected | 9.015516 |
| measured | baseline | 2.683712 |
| measured | corrected | 9.199848 |
| measured | baseline | 2.529605 |
| measured | corrected | 11.458104 |

timing-evidence.json / samples:

| Phase | Case | Wall seconds |
| --- | --- | --- |
| warmup | baseline | 10.706375 |
| warmup | corrected | 0.571339 |
| measured | baseline | 0.571506 |
| measured | corrected | 0.623369 |
| measured | baseline | 0.636869 |
| measured | corrected | 0.650501 |

integrated-timing-evidence.json / samples:

| Phase | Case | Wall seconds |
| --- | --- | --- |
| warmup | baseline | 22.912894 |
| warmup | corrected | 27.059906 |
| measured | baseline | 17.689597 |
| measured | corrected | 18.080913 |
| measured | baseline | 18.046287 |
| measured | corrected | 19.031769 |

All GPU pass samples remain in timing-evidence.json, including the main-pass spikes (~25/44ms). The integrated measured draws reuse CPU preparations (zero preparation evaluations after warmup) but execute fresh native work; measured wall includes visibility, resolve, denoise, exact paint, furniture and EXR/PNG encoding/I/O. Two observations per case establish repetition, not confidence or superiority. Phase variation uses four CPU phase samples versus one and adds no source-level native sample call per draw; full offline C01 also draws a visibility reference. No hardware fetch counter was available. All counts are inferred.

### Rendered reference assessment

Blender 4.5.0 EEVEE_NEXT uses the matched mesh/camera, Principled physical colours and protected emission; rayshader 0.37.3 on R 4.5.1/rayvertex 0.14 uses genuine software 3D frames with independently authored Lambert/cartographic materials. Both retain all 96 sequence frames and the 20m snow revision. Automated setup/revision/render and peak memory records are retained; their wall times overlapped other work and establish no speed ranking. The same legend is applied downstream for reference publication frames, separately from raw 3D comparison.

Rejected preflights remain in the logs: the initial sideways-building export and malformed R UV setup were corrected before qualifying renders. The first triplanar diagnostic lacked a requested AOV; the subsequent flat-normal projection failure produced the failing-first regression and opt-in geometric-normal correction. The eight-suite run also caught the unused private offline keyword; that adapter defect was fixed before the final full rerun. None of these failed attempts counts as successful evidence.

Tolerance was fixed before comparison at [1.0001624301470429,1.0001311126777637] pixels per axis: two half-pixel sampling centres plus measured float32 projection error. Blender geographic controls pass. Full-frame silhouettes have exceptions up to 5 px at Blender alpha fringes and 4 px at a native/rayshader faint tip; those do not pass strict silhouette equivalence. Reference AA/bilinear filtering dilutes narrow exact-RGB classes (up to 54 px to another exact-colour pixel); that is not a 54 px geographic displacement. Physical display differences are descriptive (~45 codes Blender/~32 codes R), with no equivalence tolerance or winner.

The earlier actual GDAL 3.9.3 multidirectional comparison is retained: its predeclared 1/255 equivalence tolerance fails (mean 0.112579,max 0.366787). The documented derivative/ambient conventions differ; this is assessed separately from ridge/valley readability. The QGIS application was not run. The published method, existing implementation reuse and rayshader equivalent are documented in terrain_material_recipes.md; no exact QGIS/GDAL appearance equivalence is claimed.

Agent visual judgement: class swatches and proposed buildings are readable; the display preview has a navy background while the HDR/tonemapped final has a grey background; physical final terrain is darker/smoother with less photographic detail than preview, with visible geometric faceting in the steep triplanar fixture. No abrupt seam/reset was observed in the retained motion. This finite fixture is not universal material/adapter qualification or Milos’s preference. All 7 movies retain every 48 source RGB frame losslessly; encoding equality proves transport, separately from the recorded original-size/motion inspection.

### Automated render/encode records

All values below are observed automated execution. Human setup/revision effort is unverified. Peaks are process working set (RAM); per-renderer GPU VRAM was not measured. These workloads use different lighting/AA/denoise models, so times are not a speed ranking.

| Tool | Automated setup (s) | Revision authoring (s) | Revision render (s) | Peak RAM (bytes) |
| --- | --- | --- | --- | --- |
| blender | 1.648851 | 0.157893 | 0.566910 | 861822976 |
| rayshader | 2.950000 | 0.670000 | 4.990000 | 2079494144 |
| Native | not instrumented as a whole | not instrumented | 17.578824 | 595509248 |

| Render workload | Frames | Wall min/median/max (s) | Peak RAM (bytes) |
| --- | --- | --- | --- |
| Native preview | 1 | 11.059630/11.059630/11.059630 | 462536704 |
| Native final | 1 | 17.964885/17.964885/17.964885 | 595615744 |
| Native final-nominal | 48 | 12.678943/13.970616/18.638007 | 658145280 |
| Native final-grazing | 48 | 13.012665/13.832337/18.794270 | 662069248 |
| blender nominal+grazing | 96 | 0.243203/0.538422/1.270172 | 861822976 |
| rayshader nominal+grazing | 96 | 4.330000/4.860000/6.580000 | 2079494144 |

| Encoding | Wall seconds | Peak encoder RAM (bytes) |
| --- | --- | --- |
| final-nominal | 0.307940 | 614346752 |
| final-grazing | 0.307167 | 613720064 |
| triplanar | 0.307714 | 616275968 |
| blender-nominal | 0.408367 | 601374720 |
| blender-grazing | 0.508330 | 614125568 |
| rayshader-nominal | 0.508744 | 598757376 |
| rayshader-grazing | 0.508201 | 614227968 |

Every individual frame/encode sample, actual tool command line, Blender GPU platform and R library paths are retained in the corresponding evidence/command JSON. Reference launch records now cover all 96 views plus each revision.

### Pin, certificates and remaining owner work

Terrain source pin: `0xf8f90a5fda9933d0` → `0x0c59680ae8d34d4b` → `0xd5cf1a324d746c08`, through the [documented computed rewrite](shader_pin_refresh.md). Later shader changes are opt-in C01 branches; all 22 default PNGs remain identical. Signed certificates embed shader identity and remain unchanged. Re-signing requires Milos’s separate approval.

Unrun or incomplete lanes:

- MAP-MATERIAL-MEANING criterion 1: Milos has not performed docs/guides/c01_human_trial.md. Human expertise, assistance, active setup/revision effort, readability and preference remain unverified.
- F6: all repeated owner-ready-interval samples are retained, but external Python/headless-browser activity and unassigned WDDM GPU activity prevent an exclusive-host claim. Hardware texture fetches remain inferred, not measured.
- Interactive viewer does not execute the Python precise C01 compositor on each camera update. Only the declared offscreen preview/final authoring paths are qualified; terrain DOF has no executing pass on this route.
- Cross-backend portability is ABSENT; Metal/other adapters, WASM and full hosted CI were not exercised. Existing selector/capability skips are reported separately from the zero-skip C01 cohort.
- World-scale wall/roof textures and vector/building normal/roughness are explicitly deferred by Milos to MAP-MESH-MATERIAL-CHANNELS; they are not claimed as delivered.
- C01 remains unmerged. No commit, PR, golden refresh, certificate refresh or signing was authorized or performed.

Remaining owner actions are the human trial, any further exclusive-host/hardware-counter qualification, and explicit integration authorization. The deferred follow-on remains open. No new scope decision is needed for the implemented colour routes.

Steering gap observed: PowerShell does not use backslash quote escaping, and rg positional wildcard paths fail on Windows. Proposed guidance only: use literal here-strings for multiline Python and rg -g for file globs. No AGENTS.md or skill changes were applied.

### Roadmap validation

The live file passed `validate_roadmap.js`: `VALID atlas/work/research/view`. Only the six target objects were spliced; per-object `ensure_ascii` reproduction and `<` escaping were checked, with unrelated bytes preserved. The `matchedEvidence` string schema mismatch found by the first run was corrected in the target card, then the live validator passed. `CAND-RELIEF-STYLES` is done; the other three work items remain open, and WF-MATERIAL-MEANING / C01 remain partial. Follow-on `MAP-MESH-MATERIAL-CHANNELS` is open.

Exact validator command/log: `artifacts/c01/review/owner-decisions/roadmap-final-validation-command.json` and `roadmap-final-validation.log`.

### Changed files

```text
.gitignore
docs/guides/feature_map.md
python/forge3d/__init__.py
python/forge3d/__init__.pyi
python/forge3d/_map_scene_common.py
python/forge3d/_map_scene_compiler.py
python/forge3d/_map_scene_projection.py
python/forge3d/_map_scene_rasters.py
python/forge3d/_map_scene_render.py
python/forge3d/denoise.py
python/forge3d/map_scene.py
python/forge3d/offline.py
python/forge3d/raster_style.py
python/forge3d/terrain_params.py
python/forge3d/widgets.py
src/py_types/aov.rs
src/render/mesh_instanced.rs
src/shaders/mesh_instanced.wgsl
src/shaders/terrain_pbr_pom.wgsl
src/terrain/render_params/decode_materials.rs
src/terrain/render_params/decode_postfx.rs
src/terrain/render_params/native_material.rs
src/terrain/render_params/native_postfx/quality.rs
src/terrain/renderer/aov.rs
src/terrain/renderer/bind_groups/terrain_pass.rs
src/terrain/renderer/constructor.rs
src/terrain/renderer/core.rs
src/terrain/renderer/draw/execute.rs
src/terrain/renderer/offline.rs
src/terrain/renderer/py_api.rs
src/terrain/renderer/scatter.rs
src/terrain/scatter.rs
src/verify/det_rewrite.rs
src/verify/ir/engine.rs
src/verify/ir/tests.rs
docs/guides/c01_human_trial.md
docs/guides/c01_material_validation.md
docs/guides/geospatial_materials.md
docs/guides/shader_pin_refresh.md
docs/guides/terrain_material_recipes.md
examples/terrain_material_recipe.py
python/forge3d/_map_scene_materials.py
python/forge3d/_material_preparation.py
python/forge3d/geospatial_materials.py
python/forge3d/geospatial_materials.pyi
python/forge3d/terrain_materials.py
python/forge3d/terrain_materials.pyi
tests/test_atrous_sky.py
tests/test_c01_completion.py
tests/test_c01_completion_gpu.py
tests/test_c01_owner_decisions.py
tests/test_c01_review_corrections.py
tests/test_c01_review_gpu.py
tests/test_geospatial_materials.py
tests/test_geospatial_materials_gpu.py
tests/test_material_preparation_reuse.py
tests/test_material_workflow.py
tests/test_terrain_material_recipes.py
tests/test_terrain_material_recipes_gpu.py
D:/forge3d/docs/moonshot-roadmap.html (live ignored roadmap; target object splices only)
```

Full machine-readable result: `artifacts/c01/review/owner-decisions/execution-final.json`. Frame, comparison, command and timing artifacts are local evidence, not committed generated products.


Historical reports are preserved separately in `artifacts/c01/review/owner-decisions/superseded-validation.md`; they do not describe this final build.

R4 adapter risk (F5): the 12°/1-cell fixture had 62 disagreements against 64 allowed, a 2-pixel margin on RTX 3070/Vulkan/NVIDIA 610.60. Requalify coverage and the unchanged hue/boundary limits per adapter and driver; this is not a portability certificate.
