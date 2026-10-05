# TERRA-DETERMINATA acceptance evidence — final closeout 2026-10-04

Nodes 04 and 04b are marked **complete** locally, contingent on PR #206 merge,
and TERRA-DET-LINT-01 is **done**, under
Milos's revised criterion: no new CI failures compared with scheduled main
run 37110762671. PR-head run 37153410227 meets that comparison and passes the
named TERRA jobs. Owner decision E limits acceptance to physical NVIDIA
DX12/Vulkan PNG identity and browser canary identity accepted by decision D.
No absent vendor is credited as passed. Milos authorized committing and pushing
this updated report to PR #206; merging remains Milos's action.

This review uses local worktree
`D:\forge3d\.worktrees\terra-determinata-local`, branch
`codex/terra-determinata-local`, based on `origin/main`
`3a6ec9f83f36919f79a1b79b5740bee267cd1b58`. Earlier local validation used no
subagents. Closeout used `gpt-5.6-sol` at `xhigh` for independent Python/API
review and initial roadmap analysis. After Milos instructed "Don't use subagents",
the active roadmap lane was stopped before edits. The parent completed the
roadmap implementation and all subsequent work using its current session model; this interface does not expose a
parent model/effort override or verified runtime identifier.
Milos authorized committing/pushing the test and report, opening a PR and
dispatching full CI. Merge remains Milos's action. No golden, certificate,
proof pin, threshold, timeout, workflow or dependency-manifest change is made.

## Owner decision and remaining authority

On 2026-10-03 Milos selected **“Accept browser canary identity as evidence.”**
The browser requirement therefore uses the existing compute and rgba32float
raster canaries, rather than a canonical terrain PNG. This is recorded as
`TERRA-DET-WEBGPU-01` in the current local roadmap. It does not change
native PNG equality or introduce tolerance.

On 2026-10-03 Milos made **decision E**:

> Option 2 and remove from roadmap the Apple and Intel and AMD requirements.

The available hardware is one NVIDIA GeForce RTX 3070, with an AMD Ryzen CPU
and no integrated GPU. Intel, AMD and Apple GPU requirements and unmeasured
vendor arithmetic are **out of scope by decision E**, not passed. Record E as
`TERRA-DET-SCOPE-01` (decision, done), and close `TERRA-DET-HARDWARE-01` as
`done` / `wontfix`. The repository runner inventory previously showed only
`forge3d-rtx3070`, labelled `self-hosted, Windows, X64, forge3d-gpu, gpu-nvidia`.

On 2026-10-04 Milos replaced the first-post-merge-scheduled-run and overall-green
criterion with no new CI failures against scheduled main 37110762671, using
PR-head run 37153410227. The pre-existing ORBIS SSIM, TESSELLA, Windows Python
cancellations and optional Metal failures are out of scope. The named Rust,
determinism summary and ANAMNESIS Seed/Consume jobs pass in both runs; LINT-01
is closed. The existing workflow configures AETHER offline-bake only on Ubuntu;
macOS/Windows do not execute that step. No workflow change is authorized.
The earlier local and hosted evidence below is retained and reused.
TESSELLA is now fixed on main by merged PR #207, verified by run 37160548024,
job 111315204953. ORBIS SSIM and Windows cancellations remain open follow-ups
and have not been rechecked on current main.

## Observed physical identity

The fresh release wheel rendered the canonical 512×512 scene in four separate
processes: Vulkan twice and DX12 twice. Every PNG SHA-256 was:

```text
ba4b97b6517be71ff080ac74b0a6a37012b888aa2fe9cc48840590d950b26675
```

The decoded RGBA arrays also match exactly: **zero differing pixels**.
Both tracked `terra_determinata_v1.sha256` and
`terra_determinata_v1.dx12.sha256` already contain this hash on main.
The earlier 11-pixel discrepancy does not reproduce on this base and driver;
there is no measured offending lookup to replace. No scene or golden change
was necessary. Local records, PNGs and byte comparisons are in
`artifacts/terra-determinata/repeatability/`.

Adapter: **NVIDIA GeForce RTX 3070**, DiscreteGpu, vendor 4318, device 9348,
Vulkan/Dx12 as requested, `software_fallback=false`; driver **610.60**.
Windows x64, Python **3.13.14**, NumPy **2.5.3**, pytest **9.1.1**, Pillow
**12.3.0**, Rust **1.97.1**, maturin **1.15.0**.

The installed wheel is
`target/wheels/forge3d-1.40.1-cp310-abi3-win_amd64.whl`, SHA-256
`56402162632921cc0d0f7e28129504c3efa51512e7a6ee6a875bb3a0d22d1cfb`.
Both Python and `_forge3d.pyd` imports resolved inside this worktree's
`.venv/Lib/site-packages/forge3d/`.

## Local checks and changes

`tests/test_determinism_hash.py` now respects the installed-wheel lane in its
three subprocess environment setups. The adapter-attribution unit test also
marks its mocked render's canary `not_run`, instead of initializing a real GPU
under its temporary deterministic environment. Real canary execution and
byte assertions remain in the dedicated subprocess tests.

Before the second fix the combined command produced 71 passes and a failure
in `test_runtime_contract_asserts_observed_gpu_inputs`: the preceding unit test
latched deterministic mode, and the unrelated hybrid render reported no valid
ReSTIR reservoirs. After isolating that unit test, the same ordered pair
passed, and the complete requested combined command passed **72 tests, zero
skips**. No assertion, test, capability gate or safety check was removed.

| Check | Observed result |
|---|---|
| `maturin build --release --interpreter .venv/Scripts/python.exe` | Passed, release build 10m 50s |
| Install wheel with `python -m pip install --no-deps --force-reinstall` | Passed; installed package/native paths verified |
| `pytest tests/test_determinism_hash.py tests/test_determinism_matrix.py tests/test_shader_proofs.py -v --tb=short` | 72 passed, zero skips; saved [command/environment](../artifacts/terra-determinata/pytest-combined-final-command-environment.json), [JUnit](../artifacts/terra-determinata/pytest-combined-final.xml), and [zero-skip verifier output](../artifacts/terra-determinata/pytest-combined-final-zero-skips.log) |
| DX12 `pytest tests/test_determinism_hash.py -v --tb=short` | 9 passed, 1 existing Vulkan-only SIDERA replay skip; saved [command/environment](../artifacts/terra-determinata/pytest-hash-dx12-pinned-command-environment.json) and [JUnit](../artifacts/terra-determinata/pytest-hash-dx12-pinned.xml); SIDERA was exercised separately on Vulkan |
| `pytest tests/test_budget_enforce.py tests/test_memory_budget_policy.py -v --tb=short` | 7 passed, zero skips; existing 512 MiB enforce policy and rejection of a 600 MiB request |
| `cargo test --lib verify:: --features extension-module,shader-contract-asserts -- --nocapture` | 108 passed, 1 existing ignored diagnostic, 1461 filtered; terrain proof/ablation tests executed |
| `det_instrument_rewrite` in that Rust run | Passed: 0 edits; all residual skips are `hardware_sample` advisories |
| `cargo forge3d-clippy` | Passed |
| `cargo fmt --check` | Passed |
| NVIDIA visual selection | 27 passed, zero skips |
| SIDERA `tests/test_astro_night_golden.py` | 5 passed, zero skips |
| AEQUITAS `tests/test_adjudication_gate.py` | 34 passed, zero skips |
| PROMETHEUS `tests/test_hybrid_terrain_pt.py tests/test_prometheus_dem_reference.py` | 32 passed, zero skips |

The ignored Rust diagnostic is
`verify::ir::tests::inspect_deterministic_kernel_ir_shape`; it was not run.
Full logs and JUnit files are under `artifacts/terra-determinata/`.

Environment for local installed-wheel tests:

```powershell
$env:CARGO_TARGET_DIR = 'D:\forge3d\.worktrees\terra-determinata-local\target'
$env:FORGE3D_NO_BOOTSTRAP = '1'
$env:FORGE3D_TEST_INSTALLED_WHEEL = '1'
$env:PYTHONUTF8 = '1'
$env:WGPU_BACKEND = 'vulkan'
$env:FORGE3D_DETERMINISM_TEST_BACKEND = 'vulkan'
$env:TEMP = 'D:\forge3d\.worktrees\terra-determinata-local\artifacts\terra-determinata\temp'
$env:TMP = $env:TEMP
$env:TMPDIR = $env:TEMP
```

Each identity-render child additionally sets `FORGE3D_DETERMINISTIC=1` and
`WGPU_BACKENDS` to its one backend **before GPU initialization**.
No software-adapter override was enabled. Combined pytest uses
`--basetemp=artifacts/terra-determinata/temp/combined-final` and
`--junitxml=artifacts/terra-determinata/pytest-combined-final.xml`.
The two saved command/environment JSON records contain the exact recorded
shell invocations from task `01a100b2-2652-77f0-8823-174c5bbc171b`, with their
source command IDs retained in each JSON. Only explicitly assigned environment
variables were captured in those invocations; a full inherited environment
snapshot is unavailable. The combined pytest subcommand exited 0, although
its compound shell later exited 5 on the separate, superseded all-DX12 attempt.

On closeout, `.venv/Scripts/python.exe scripts/assert_junit_zero_skips.py
artifacts/terra-determinata/pytest-combined-final.xml` was executed with
`PYTHONUTF8=1`, exit 0: 72 tests, 0 failures, 0 errors, 0 skipped. Its output is
saved without rerunning pytest. `local-checks.json` is explicitly labelled
**superseded first attempt**; its failed combined-Vulkan check is not evidence
for `pytest-combined-final`.

An initial test attempt failed to create the machine-global
`D:\forge3d\cache\tmp\pytest-of-milos` directory (56 setup errors).
Moving TEMP/TMP/TMPDIR and pytest scratch into the worktree resolved those
errors. The initial failed logs are retained; they are not claimed as passes.

For the DX12 hash test file, `FORGE3D_DETERMINISM_TEST_BACKEND=dx12` pins the
actual tested renders and canaries to DX12 in fresh subprocesses, while the
general collection-time terrain availability probe uses `WGPU_BACKEND=vulkan`.
An initial all-DX12 probe exceeded `_terrain_runtime.py`'s existing 120-second
limit and skipped the module (pytest exit 5); that attempt is not passing
evidence. No timeout was changed. The subsequent pinned DX12 tests passed,
including the test that verifies an active DX12 adapter despite a later Vulkan
probe argument.

The NVIDIA lanes run serially with `WGPU_BACKEND=vulkan`,
`WGPU_BACKENDS=vulkan`, `FORGE3D_RUN_TERRAIN_GOLDENS=1`,
`FORGE3D_TERRAIN_GOLDEN_VARIANT=nvidia-vulkan` and
`FORGE3D_RECIPE_GOLDEN_VARIANT=nvidia-vulkan`, plus the installed-wheel and
local-temp environment above. All golden/certificate update flags are `0`:
`FORGE3D_UPDATE_TERRAIN_GOLDENS`, `FORGE3D_UPDATE_RECIPE_GOLDENS`,
`FORGE3D_UPDATE_RECIPE_CERTIFICATES`, `FORGE3D_UPDATE_SIDERA_GOLDEN`,
`FORGE3D_UPDATE_ADJUDICATION_GOLDENS`, and
`FORGE3D_UPDATE_HYBRID_TERRAIN_GOLDENS`.

Exact GPU selections (using `.venv/Scripts/python.exe`):

```text
scripts/run_nvidia_visual_acceptance.py --suite visual --junit artifacts/terra-determinata/nvidia/visual.xml
-m pytest tests/test_astro_night_golden.py -v --tb=short
-m pytest tests/test_adjudication_gate.py -v --tb=short
-m pytest tests/test_hybrid_terrain_pt.py tests/test_prometheus_dem_reference.py -v --tb=short
```

Each pytest lane also writes its named JUnit XML and uses its own
`artifacts/terra-determinata/nvidia/temp/<lane>` basetemp; the visual runner's
temporary files are rooted by TEMP/TMP/TMPDIR in that directory tree.
`scripts/assert_junit_zero_skips.py` passed for every required NVIDIA XML.
The exact argument lists and exits are retained in
`artifacts/terra-determinata/nvidia/results.json` and `nvidia-gates.ps1`.

AEQUITAS's actual physical render pair recorded process peak tracked
host-visible memory **67,109,888 bytes**, total tracked peak **467,668,100
bytes**, `budget_policy=enforce`, `within_budget=true`. Its adapter was the
physical NVIDIA Vulkan device. This is saved in
`artifacts/terra-determinata/nvidia/adjudication/capture.json`.

## Sampling and numerical policy

The inventory covers `shader_sources::terrain_parts()` and the three IBL
precompute shader files. After stripping comments, the requested live files
have **zero raw `dot/normalize/pow/exp/log/mix/cross` calls**.
`csm.wgsl` is deleted; its current terrain shadow path is in
`terrain_pbr_pom.wgsl`. The raw regex matches found by `rg` are comments.

`artifacts/terra-determinata/sampling-inventory.json` lists all 39 source
sample sites with file, line, function, texture and sampler. This includes
disabled optional/debug branches; it is not a claim that every branch ran.
The lint's full advisory inventory is retained in `cargo-verify.log`.

Canonical active sampling includes the IBL equirectangular/precompute inputs,
PCF shadow taps, material/colormap lookups and the diffuse/specular/BRDF IBL
lookups in `eval_ibl_split`. Nearest filtering is applied where resource
construction calls `deterministic_filter_mode`; shadow tap coordinates and
accumulation use pinned arithmetic, while comparison sampling remains a
hardware operation. In particular, the **shared runtime IBL sampler** in
`src/lighting/ibl_wrapper.rs` still explicitly requests **linear filtering**
and is bound by `prepare_ibl_bind_group`. The source policy's broad nearest
wording must not be read as proof that this sampler is pinned or that all
sampling is software reconstruction. This remaining hardware arithmetic is
measured on the RTX 3070 but unproven on the missing vendors.

The polynomial float32 DEM avoids platform libm transcendentals. Its POM,
PCF shadows and IBL are enabled in the main canonical configuration.
`det_*` barriers and the zero-edit Naga-IR lint cover the source arithmetic;
DX12 pins FXC and the DEBUG/skip-optimization compilation path. Compiler
fast-math behavior on unmeasured physical Apple/Intel/AMD devices, rasterizer
interpolation, derivatives and hardware texel selection are not certified by
the local NVIDIA result. No cross-vendor conclusion is inferred from them.

## Hosted evidence reused, not redispatched

[Full run 37029612721](https://github.com/milos-agathon/forge3d/actions/runs/37029612721)
tested PR #202 head `bbbad1dc`, now merged on main:

- Test Rust logs show `det_instrument_rewrite ... ok` on Ubuntu, macOS and
  Windows (jobs 110913553362, 110913553033, 110913553200). That Windows job
  failed elsewhere in ORBIS; it is not claimed as a green whole Rust job.
- Hosted Render Determinism summary **110931276459** passed. Real NVIDIA
  Vulkan PNG hash equals the current golden. Browser compute/raster canaries
  equal the unchanged committed canaries. Rechecking the downloaded real
  artifacts with `scripts/check_determinism_hashes.py` also passed.
- The summary explicitly records **2/6 legs with evidence**: NVIDIA PNG and
  browser canaries. AMD/Intel are absent, Apple is documented absent and wasm
  is policy-only. This is **not** a full physical portability proof.
- Browser metadata reports SwiftShader/software, `raster_format=rgba32float`,
  compute `f94de1bffbf3383447e63c8db2fa6f9c891af406303303ec0cc9c2fc9d729aeb`
  and raster `298a84780b271bfa0633e50507bc1209ed17fbed4a68c1afb5a7d38f0b23875a`.
- Relaxed-SIMD job **110922976893** passed: no build configuration enables it.
- Physical Seed **110919056142** passed, with the common Vulkan hash. Physical
  Consume **110996347770** independently produced the same DX12 PNG hash:
  4 hits, 0 misses, hit rate 1.0. The mismatch check had 0 hits, 4 misses,
  hit rate 0.0. Producer and consumer were the same physical machine; this is
  cross-backend evidence, not a distinct-machine claim.

[Full run 37031903053](https://github.com/milos-agathon/forge3d/actions/runs/37031903053)
on the now-merged ORBIS branch passed all three Test Rust jobs, including
Windows **110920747301**, whose log also shows the determinism lint passing.
That run lacked PR #202's golden refresh and failed its determinism summary;
it is used only as Rust/lint evidence.

Local hosted downloads and recheck logs are under
`artifacts/terra-determinata/hosted-*`.

## Honesty check

Read-only searches of `README.md`, `docs/start/architecture.md` and
`docs/guides/feature_map.md` in both the current checkout and the base worktree
found **no claims of cross-vendor TERRA identity**; there are no matching
path:line findings to report. These documents are unchanged. Search records
are saved in `artifacts/terra-determinata/honesty-check.json`.

## Final closeout — 2026-10-04

Milos revised the criterion to **no new CI failures compared with scheduled
main run 37110762671**. The existing ORBIS SSIM, TESSELLA, Windows Python
cancellations and optional Metal failures are out of scope. This replaces the
previous overall-green and first-post-merge-scheduled-run blockers; it does
not change or suppress any executable gate. Run 37110762671 is the scheduled
baseline at `3a6ec9f83f36919f79a1b79b5740bee267cd1b58`; it predates PR #206
and is not claimed to include that PR.

The authorized test/report commit is
`e719399653ab0cca2040f9b55e6ca28fdc26801b`, pushed on
`codex/terra-determinata-local` in [PR #206](https://github.com/milos-agathon/forge3d/pull/206).
The initial published report was the pre-dispatch snapshot. Milos now
authorizes committing and pushing this updated evidence report to the same PR
branch. Only this report changes in the follow-up commit; existing executable
and physical evidence inputs are unchanged. No direct main commit or merge
is authorized.

PR #206 open at e7193996; nodes 04/04b become complete when it merges.

The earlier "merged" statement came from a prompt template, not from Milos.
The API confirms PR #206 is open, mergeable and CLEAN at that pre-update head.
The old discrepancy record is superseded by this correction. The previously
tested head remains the reference for full physical CI; scoped PR CI verifies
the subsequent report-only head.

### Completed CI evidence

[Scheduled main 37110762671](https://github.com/milos-agathon/forge3d/actions/runs/37110762671)
completed with overall `failure`.
[PR-head full CI 37153410227](https://github.com/milos-agathon/forge3d/actions/runs/37153410227)
is `workflow_dispatch`, `scope=full`, on the exact published PR head and
completed with overall `cancelled`. Neither overall run is claimed green.
The completed job lists have identical coverage, identical failed/cancelled
job names and conclusions, and **no baseline-success job regressed**. The two
HELIOS jobs improved from skipped to success. The named TERRA evidence jobs
below all passed in both runs. Comparison and assertions are saved in
`artifacts/terra-determinata/closeout-ci-comparison.json`.

| Required evidence | Scheduled-main job | PR-head job | Result in both |
|---|---:|---:|---|
| Test Rust, Ubuntu | 111168136429 | 111291832051 | success |
| Test Rust, macOS | 111168136426 | 111291832021 | success |
| Test Rust, Windows | 111168136468 | 111291832418 | success |
| Render Determinism Acceptance Summary | 111180189604 | 111308086752 | success |
| ANAMNESIS Physical Seed, Vulkan | 111170313842 | 111293788948 | success |
| ANAMNESIS Physical Consume, DX12 | 111175907032 | 111303156846 | success |

All six Rust logs contain actual
`verify::det_rewrite::instrument::det_instrument_rewrite ... ok` output.
Scheduled-main logs show 1540 library tests passed, zero failed, two ignored
on each OS. Step metadata confirms doctests and clippy executed successfully
on all three OSes in both runs. The configured AETHER offline-bake acceptance
step executed successfully on Ubuntu; macOS/Windows skipped that step under
the existing workflow configuration and are not credited with executing it.
Saved logs: `closeout-main-rust-{ubuntu,macos,windows}.log` and reused
`pr206-test-rust-{ubuntu,macos,windows}.log`.

Both summary logs report `determinism diff: OK` and physical
**NVIDIA GeForce RTX 3070**, Vulkan/DiscreteGpu, PNG SHA-256 equal to the
committed golden:

```text
ba4b97b6517be71ff080ac74b0a6a37012b888aa2fe9cc48840590d950b26675
```

Browser compute and rgba32float raster canaries equal the committed hashes
in both summary logs:

```text
compute f94de1bffbf3383447e63c8db2fa6f9c891af406303303ec0cc9c2fc9d729aeb
raster  298a84780b271bfa0633e50507bc1209ed17fbed4a68c1afb5a7d38f0b23875a
```

PR-head NVIDIA render job **111294523463** and browser job **111294523578**
passed. Reused downloaded browser metadata reports BrowserWebGpu, SwiftShader,
`software_fallback=true`, `raster_format=rgba32float`; its hashes are verified
in `pr206-browser-identity.json`. The native adapter metadata reports vendor
4318, device 9348, driver 610.60 and `software_fallback=false`.
Intel/AMD are explicitly absent in both summaries and Apple is documented
absent. Their wrapper jobs' success does not prove a physical render;
they remain **out of scope by decision E, never passed**. The wasm policy
marker is not a render. No six-vendor or canonical browser-terrain claim is
made.

Seed and Consume logs in each run show the common PNG hash above, with
RTX 3070 physical Vulkan and DX12 adapters respectively. Each Consume records
`backend_goldens_match=true`, four hits and zero misses. Each compatibility
mismatch control records zero hits and four misses. Producer and consumer
use the same physical machine; this is cross-backend evidence. Saved logs:
`closeout-{main,pr206}-{seed,consume}.log` and
`closeout-{main,pr206}-determinism-summary.log`.

PR-event core run **37153402166** also passed (preflight **111291722025**,
Fast Contract **111291798428**, PR Core Success **111292629261**).
All four Linux and all four macOS installed-wheel full jobs passed. The four
Windows jobs were cancelled in both runs, and remain cancelled in the record:

| Installed-wheel full lane | Scheduled-main job | PR-head job | Result in both |
|---|---:|---:|---|
| Linux 3.10 | 111169276383 | 111293149193 | success |
| Linux 3.11 | 111169276483 | 111293149233 | success |
| Linux 3.12 | 111169276461 | 111293149238 | success |
| Linux 3.13 | 111169276432 | 111293149218 | success |
| macOS 3.10 | 111170820651 | 111294523514 | success |
| macOS 3.11 | 111170820641 | 111294523564 | success |
| macOS 3.12 | 111170820686 | 111294523508 | success |
| macOS 3.13 | 111170820766 | 111294523485 | success |
| Windows 3.10 | 111170313952 | 111293789019 | cancelled |
| Windows 3.11 | 111170313997 | 111293789022 | cancelled |
| Windows 3.12 | 111170313960 | 111293789076 | cancelled |
| Windows 3.13 | 111170314001 | 111293789065 | cancelled |

### Pre-existing failures and current follow-up status

| Follow-up | Scheduled-main job | PR-head job | Observed existing failure |
|---|---:|---:|---|
| Visual Goldens NVIDIA / ORBIS SSIM — open | 111170313814 | 111293788855 | Same sole failing test: `test_orbis_ground_snapshot_matches_committed_nvidia_vulkan_golden`; SSIM 0.993359476432247 below existing 0.995 threshold; 39 passed in each historical run; not rechecked on current main |
| TESSELLA — fixed on main by PR #207 | 111170313895 | 111293788945 | Historical failures: five streaming-budget tests and one missing CPU BVH oracle. Merged PR #207 fixes them; run 37160548024, job 111315204953 passed |
| Windows Python matrix — open | Four jobs listed above | Four jobs listed above | All four versions cancelled in both runs; not rechecked on current main |
| SUBSTRATIA optional Metal | 111170820518 | 111294523382 | `terrain-ci-probe: ABSENT — no CI-safe hardware adapter on this runner`, before tests |
| Full Acceptance Summary | 111180391817 | 111308284181 | failure in both completed runs; no whole-run-green claim |

The ORBIS and TESSELLA final failing-test signatures, including assertion
messages, match exactly between the two historical logs. Saved logs:
`closeout-{main,pr206}-{orbis,tessella}.log`, `closeout-main-metal.log` and
reused `pr206-substratia-metal-failure.log`.
Windows 3.11's saved PR log reached only 41% in unchanged globe tests and
reports cancellation; it does not prove execution of `test_determinism_hash.py`.
The timeout explanation remains an inference because the earlier annotation
lookup returned HTTP 503. No workflow timeout was changed.

TESSELLA is **fixed on main by PR #207**, merged as
`6d85ce0e7cda22b16d525aad17254146cbe90efd` on 2026-10-04.
[Scoped run 37160548024](https://github.com/milos-agathon/forge3d/actions/runs/37160548024)
on PR #207 head `4c166e6c4bdc2a9622f5050e9f644fbff2bf5f5f` passed;
[TESSELLA job 111315204953](https://github.com/milos-agathon/forge3d/actions/runs/37160548024/job/111315204953)
passed the exact NVIDIA Vulkan adapter check, GPU/CPU LOD differential,
real-shader HZB differential, all six acceptance gates and consolidated nine
verification results. Saved records: `pr207-main-fix-run.json` and
`pr207-tessella-job.json`. The run tested the PR #207 head before its merge;
the merged fix is now on main.

**ORBIS SSIM and Windows Python cancellations remain open follow-ups. Neither
has been rechecked on current main.** Their baseline results are historical,
not current-main failures or passes. Optional Metal remains out of scope.
No unrelated follow-up was fixed by this TERRA session, no other roadmap node
or work item was edited for it, and no executable test or gate was weakened.

### Roadmap closure and scope proof

The current **`D:\forge3d\docs\moonshot-roadmap.html`** now records nodes
**04 and 04b as `full` (complete)** and **TERRA-DET-LINT-01 as `done`**.
Node 04's review carries the exact note **"contingent on PR #206 merge"**;
the complete markers are retained as instructed, with integration contingent
on Milos merging the open PR.
The prior pending scheduled-run row is replaced by the owner-revised
criterion and the run/job evidence above, as an implemented row. The duplicate
04b implemented claim is removed; all its distinct evidence is retained in
the surviving row. Component counts now match actual rows:

| Node | Complete | Partial | Missing |
|---|---:|---:|---:|
| 04 | 7 | 0 | 0 |
| 04b | 5 | 0 | 0 |

Decision E, `TERRA-DET-SCOPE-01` and `TERRA-DET-HARDWARE-01` done/wontfix remain
as previously recorded. No Intel/AMD/Apple requirement is reintroduced.
Only nodes 04/04b and work item TERRA-DET-LINT-01 change in this final step.
The other atlas data, source summary table, work items, outer HTML/CSS/JS,
code comments and frozen excerpts are preserved. Before/after snapshots,
the scoped diff and assertions are saved as
`roadmap-final-closeout-{before,after}.html`, `roadmap-final-closeout.diff`
and `roadmap-final-closeout-validation.json`. The worktree roadmap copy is
untouched. The main roadmap is a local ignored tracker; it is not committed.

After this scoped write, a concurrent writer changed node 19's summary and
unverified fields and normalized outer line endings. Those changes are
preserved; no restore or overwrite was performed. Current nodes 04/04b and
all work items still equal the saved closeout result. The built-in
`validateAtlas()` and `validateWork()` both pass on the current file.
`roadmap-final-current-verification.json` distinguishes this concurrent
change from this session's saved, scoped before/after diff.

### Checks performed and not rerun

Read the two completed GitHub run/job records and necessary logs; verified
the job outcome comparison, executed Rust lint/step evidence, native hashes,
canary hashes and Seed/Consume records; verified the scoped roadmap diff,
component counts and preserved frozen/outer content. Existing local pytest,
GPU, resource-budget, Rust, browser and honesty-check evidence is reused.
The roadmap's actual built-in validators and `git diff --check` passed.
No local test, render, build or full CI was rerun. No post-merge scheduled run
was sought or substituted; the owner explicitly selected the earlier
scheduled baseline. No timer polling, subagent or merge was used. The prior
roadmap closure had no commit or push; this merge-prep correction is explicitly
authorized for a report-only commit and push to PR #206. No golden, pin,
certificate, threshold, timeout,
workflow or executable test changed.

The final implementation and review lane is the parent only. The exact
current runtime model and effort are not exposed by this interface, so they
are not guessed. Historical independent Python/API review and initial
roadmap analysis used `gpt-5.6-sol` / `xhigh`, before the owner's prohibition
on subagents. No model escalation or new lane was used for final closeout.

Reusable command note, proposed only: in PowerShell, give `rg` a directory
and `--glob` instead of an unexpanded wildcard path; print scoped evidence
fields rather than frozen JSON excerpts. No AGENTS.md or skill edit applied.

Closeout evidence meets **Milos's revised criterion**. Nodes 04/04b remain
marked complete locally, contingent on PR #206 merge. The updated report is
authorized for publication to the PR branch; scoped CI on that report-only
head is the remaining merge-readiness check. **Milos must merge PR #206** to
complete integration. No merge is performed by this session.
