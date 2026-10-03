# TERRA-DETERMINATA acceptance evidence — 2026-10-03

Nodes 04 and 04b remain **partial**, pending the first scheduled full main run
that includes this PR after Milos merges it. Owner decision E limits acceptance
to physical NVIDIA DX12/Vulkan PNG identity and the browser canary identity
accepted by decision D. No absent vendor is credited as passed.

This review uses local worktree
`D:\forge3d\.worktrees\terra-determinata-local`, branch
`codex/terra-determinata-local`, based on `origin/main`
`3a6ec9f83f36919f79a1b79b5740bee267cd1b58`. Earlier local validation used no
subagents. This closeout uses `gpt-5.6-sol` at `xhigh` for independent Python/API
review and the roadmap JSON/HTML update. The parent integrates and reviews the
changes using its current session model; this interface does not expose a
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

The first scheduled full main run including this test fix after merge remains
unverified. Existing full dispatch evidence below is distinguished from that
required run. `TERRA-DET-LINT-01` stays open until the scheduled run proves the
three Test Rust jobs, the configured AETHER offline-bake step (Ubuntu), doctests
and clippy, NVIDIA golden identity, browser canaries and ANAMNESIS Seed/Consume.
The existing workflow configures AETHER offline-bake only on Ubuntu; macOS and
Windows do not have that step selected. No workflow change is authorized.

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

## Outstanding evidence and reusable setup knowledge

No physical Intel/AMD/Apple matrix was run: out of scope by decision E.
No canonical browser terrain PNG was built or run, per decision D.
The current `D:\forge3d\docs\moonshot-roadmap.html` is edited locally under E;
the stale worktree copy is not used or committed. Nodes 04/04b remain partial
with scheduled-main confirmation as the only acceptance blocker.
PR-head full CI and the post-merge scheduled run are separate proofs; IDs and
observed outcomes are recorded during closeout. Milos must merge the green PR
before the required scheduled-main proof can exist. No scheduled-run timer
polling or replacement manual-main run is used.

Proposed setup-rule improvement, not applied to `AGENTS.md`: alongside the
per-worktree venv and CARGO_TARGET_DIR, set TEMP/TMP/TMPDIR and pytest basetemp
inside the worktree, and use UTF-8 explicitly for source/evidence scripts.
PowerShell `rg` searches should pass a directory and `--glob`, rather than
unexpanded wildcard path arguments.
