# Opt-in DEM mesh density — 2026-10-20

Work started from remote `main` **5db4989df267e65efc242e689efa04dd7116da3c**,
verified by read-only `git ls-remote`. Local `main` was older and diverged;
the primary checkout also contained unrelated changes. Implementation is in
the isolated `D:/forge3d/.worktrees/terrain-dem-resolution` worktree.
The measurements and local closure below preceded publication; Milos later
authorized committing, pushing and opening a PR. No subagents were used.

## Cost diagnosis and retained fix

The October 12 cost fix, **3fabfb47**, is already an ancestor of current main.
The CPU AO implementation is unchanged between that commit and current main.
The retained [cost acceptance report](terrain-native-cost-acceptance-evidence.md)
and its local instrumentation patch identify
`TerrainScene::compute_coarse_ao_from_heightmap`: a radius-8 scan of up to
288 neighbours per texel, including square roots and arctangents. At 3000²,
the recorded matched baseline spent 17.610 of 17.988 seconds in that CPU span.
The existing fix skips it when neither weighted AO nor raw-AO debug consumes
the result; explicit consumers retain the original arithmetic.

That historical profile is reused, not claimed as a new instrumented run.
This task rebuilt current main before editing Rust. The same existing
`tools/benchmark_terrain_upload.py` measured the 1080×1080 mesh:zup workload:

| Input | Size | Current-main native wall s | Process CPU s | Timestamped GPU passes ms |
| --- | ---: | ---: | ---: | ---: |
| random | 1490² | 0.280199 | 0.250000 | 252.854 |
| random | 2080² | 0.109158 | 0.093750 | 64.020 |
| random | 3000² | 0.144872 | 0.140625 | 64.900 |
| smooth | 1490² | 0.025541 | 0.031250 | 9.485 |
| smooth | 2080² | 0.049860 | 0.046875 | 14.795 |
| smooth | 3000² | 0.140085 | 0.140625 | 51.365 |

These are individual calls, not latency distributions. Pipeline priming is
outside the measured calls. The first random call has a larger measured GPU
time; no statistical speedup claim is based on that sample. CPU and GPU work
can overlap, and GPU timestamps do not cover all transfers or synchronization.
No additional cost fix or cache shortcut was necessary.

## Geometry and compatibility

Milos chose **“Add opt-in DEM-resolution mesh mode”** on 2026-10-20.
`camera_mode="mesh:zup:dem"` (or `"mesh:dem"`) selects
`max(width, 2)` by `max(height, 2)` vertices. Both axes visit their DEM samples;
a singleton axis repeats its only sample to form a quad. The initial
square-grid implementation missed rectangular sample peaks and was corrected
after review. The measurements in the initial-validation sections below
describe that earlier binary, not the corrected per-axis shader.
See the [public usage guide](terrain/dem-mesh.md).

The procedural beauty/water and shadow shaders now support rectangular grids.
The existing uniform buffer sizes and bind-group layouts are retained. A shared Rust calculation
supplies matching draw counts and uniforms to beauty, AOV, offline
accumulation, planar reflections and shadow casters. Height dimensions are
explicit inputs, rather than retained renderer state. `u32` draw-count overflow
raises an error. No CPU vertex/index arrays are added.

Existing mesh modes retain their 512² beauty grid and 1024² shadow grid.
Public defaults, signatures, registration, exports, serialized formats,
goldens, signed certificates and dependencies are unchanged. The exact
owner-approved conversion-inventory gate transition is recorded below.
The WGSL source pin was refreshed after explicit owner approval;
approval and final validation are recorded separately from the initial results.
The existing `camera_mode` getter is now represented in the
Python stub; no new exported symbol or contract-list entry is introduced.

The shadow cache has an independently declared input projection. The new mode
must enter that projection because it changes caster density. A focused
physical-GPU regression against the pre-correction binary reproduced a shadow
cache hit on the density switch. The correction adds camera mode only for
DEM mesh modes, retaining existing modes' projection bytes.

## Environment and evidence placement

Windows, Python 3.13.14, NumPy 2.5.1, Forge3D 1.45.0; physical NVIDIA GeForce
RTX 3070, Vulkan, NVIDIA driver 610.60. The adapter probe reports
`DiscreteGpu` and `software_fallback=False`. Venv-local pip was verified before
building. The original checkout's root and nested Rust/WGSL/Python/test/docs
contracts were followed; the registry's promoted skill directories were absent.

Logs and measurements are under `artifacts/terrain-dem-resolution/` in the
worktree and the authorized C: output directory
`C:/Users/milos/.codex/visualizations/2026/10/20/01a159d1-e91d-7a12-b657-203fe03db1d5/terrain-dem-resolution/`.
All 146 baseline files (144 arrays plus metadata and HDR) were moved to C:
and hash-verified. D: later filled and caused one Clippy run to fail with
`no space on device`. Final compiler output uses C: instead.

Milos then authorized freeing D:. Only this task's two completed Rust test
products, `forge3d-88aa39d3cb1c490e.exe` and `.pdb`, were archived to C:;
507,523,072 bytes were moved and both SHA-256 values verified. Their manifest
is retained beside them. Tests and gates were not removed or changed by this
archive. Shared build products otherwise remain untouched.

The app-managed worktree helper stalled before registration; its checkout was
left untouched. The local isolated worktree above was created through Git.

Maturin also regenerated the tracked native PDB. Its generated bytes were
archived and hash-verified on C:, then Git restored the original committed
LFS pointer. The rebuilt `.pyd` remains installed. The final source diff
contains no generated products. A full diff check initially hit the LFS
filter on that generated PDB; it passed after restoration.

## Initial validation results (before rectangular correction)

- **4 Rust tests passed** with
  `cargo test --lib --features extension-module,default,cog_streaming,shader-contract-asserts,splat-fusion terrain::renderer::core::camera_mode_tests -- --test-threads=1`.
  The new density/count test and all three existing camera tests executed.
- **`cargo fmt --check` and `cargo forge3d-clippy` passed**, including the final
  shadow-cache correction. The C: target was used for final Clippy.
- **`maturin develop --release --offline` passed** for the baseline, geometry
  change and final cache correction. The final build uses the unchanged
  release features in `pyproject.toml`, the worktree venv and the C: target.
- **17 selected tests passed** on the geometry binary: all 14 initial DEM mesh
  cases, the named single-sample and deterministic offline tests, and
  `test_proven_list_clean_and_large_enough` from the shader proof suite.
- **The new cache regression passed on the final binary.** The old binary
  failed its expected shadow-miss assertion; the corrected binary recomputes
  on a density switch, matches an uncached render, restores on repetition and
  retains the original mode's cache hit. The unchanged uncached tests above
  are reused; final-binary comparisons below independently exercise that path.
- **Both JUnit reports passed `scripts/assert_junit_zero_skips.py`.** Runs set
  `WGPU_BACKEND=vulkan`, `FORGE3D_NO_BOOTSTRAP=1`,
  `FORGE3D_TESSELLA_REQUIRED_GPU=1` and `VF_COLOR_DEBUG_MODE=0`.
  `CI` and `GITHUB_ACTIONS` were unset.
- **All 144 default arrays match shape, dtype and bytes** against freshly
  rebuilt current main, including 60 offline arrays. The frozen existing
  benchmark covers the stated large random/smooth inputs plus all three
  camera modes and explicit AO/debug consumers. No golden update mode was used.
- **Final `git diff --check` passed.** WGSL, goldens/certificates, dependency
  manifests, workflow gates and `skills-lock.json` have no changes.

Native SHA-256 provenance:

| Binary | SHA-256 |
| --- | --- |
| Current-main baseline | `7e69eb8faf56cd658ebb1213baeb2b07e4492f8b8efb9b7096176f44624ad154` |
| Geometry test binary | `f7b4481d5d56d00c70859d479f97cc4d17196a805d4133ba74a44dc641080c1d` |
| Final cache-corrected binary | `0bed8f8d6b5db289640ebcb63e3303b0c489d06b50fa74b40b655e74b395a100` |

## Initial silhouette, memory and observed GPU work

A 1025² DEM has a one-column ridge between the default grid's central
vertices. The existing independent CPU `project_terrain_depth` rasterizer
projects every DEM triangle to a 192×128 reference mask. The reference covers
3192 pixels. The default GPU grid covers 2570, with **622 mismatched pixels**;
the DEM grid covers 3192, with **zero mismatches**. Both captures use the same
camera and input. Their beauty/depth/normal/albedo arrays, reference mask and
`silhouette-comparison.png` are retained in the C: evidence directory.

The larger input probe uses 1080×1080, smooth heights, 41.6 km span, MSAA 1,
no AO/POM/detail/shadows/reflections and identical settings within each pair.
Only the mode changes. Rows are single calls; GPU values sum executed pass
timestamps:

| DEM | Default GPU ms | DEM-mode GPU ms | Default / DEM wall s | Default / DEM process CPU s |
| --- | ---: | ---: | --- | --- |
| 1490² | 109.561 | 129.487 | 0.133998 / 0.150510 | 0.109375 / 0.140625 |
| 2080² | 55.700 | 21.947 | 0.087953 / 0.047028 | 0.062500 / 0.046875 |
| 3000² | 57.794 | 26.979 | 0.120124 / 0.076814 | 0.125000 / 0.078125 |

**Timing limit:** unrelated native builds were observed during these final
runs. Wall times are contended, initial samples vary substantially, and these
numbers establish executed GPU work rather than a controlled speedup or
regression claim. They do not show that increasing density is generally
faster. Vertex work increases from 1,566,726 to 53,964,006 invocations at 3000²;
the full grid is drawn every frame.

All measured render captures report a **1024-byte tracked host-visible peak**,
within the existing 512 MiB render budget. The largest paired large-input
device-local peak is 342,755,162 bytes. These captures precede array readback;
ordinary CPU vectors, process RSS and later readback are not covered by that
host-visible figure. No unused coarse-AO allocation appears in these calls.

## Scope, remaining limits and reusable notes

Owned source paths are the renderer's grid/count/uniform plumbing in
`src/terrain/renderer{.rs,/core.rs,/geometry.rs,/upload.rs,/aov.rs,/offline.rs,
/draw/mod.rs,/draw/execute.rs,/water_reflection/bind_group.rs,/shadows/render.rs,
/shadows/setup.rs,/py_api.rs}`, the native camera-mode comments in
`src/terrain/render_params/{core.rs,py_api.rs}`, Python parameter/stub comments,
the focused regression file, this report and the two terrain usage pages,
plus the approved source pin in `src/verify/ir/engine.rs` and docs index entry.
The existing native registration, exports and API symbol lists need no new
entry because this is an option on the existing string property.

Review reopened the local `MAP-TERRAIN-SCALE` item because rectangular
silhouette sampling was unproven. The correction, explicit source-pin approval
and final native proof/GPU verification are recorded below; publication
acceptance also requires green checks on the latest PR commit.
The independent 05 ORBIS and 19 TESSELLA atlas assessments are unchanged.

The initial verification did not exercise the full Cargo/Python/hosted CI matrix,
a fresh golden lane, the film compositor, another physical adapter, DX12,
Metal or WASM. Remediation below adds fresh physical golden and real-DEM checks;
historical cost-stage evidence remains historical.
Cross-backend portability evidence is **ABSENT**. The corrected grid preserves
the renderer's existing single span for both world axes; per-axis sample
density does not add rectangular geospatial extents.

Reusable command notes, proposed for the local validation guide rather than
applied to contracts or skills: select registry/roadmap JSON entries through a
parser, since a text match can print their entire single-line payload; use
`rg -g` file filters in PowerShell; set writable task `TEMP`/`TMP` before venv
creation; use a private target on a volume with sufficient space when the
shared target fills. Agent contracts and skills were not edited.

## Review remediation — 2026-10-20

Milos requested **“fix all”** after the independent review reopened the item.
Remote main was reverified as `5db4989d` with a read-only Git query. Work
continued in the same isolated worktree, preserving primary-checkout changes.

- **F1:** width and height now independently determine the DEM mesh, shadow
  caster and water ray-traversal grids. The mode flag uses an existing uniform
  component; no buffer size or bind-group layout changes. The 512/1024 legacy
  grids remain. The regression first failed on the old `0bed8f8d` binary.
  Both rectangular orientations now reach zero silhouette mismatches in the
  focused tests. The reviewed 1025-by-700, column-343 ridge changes from
  **566 mismatches to zero**. Square-ridge coverage is retained.
- **F2:** the usage guide explicitly describes fewer shadow vertices on DEM
  axes below 1024 samples. The parity test is named
  `test_square_512_dem_matches_legacy_with_shadows_disabled`; obsolete
  rectangular legacy-equality cases are replaced by full-DEM reference tests.
- **F3:** `terrain/dem-mesh` is included directly in `docs/index.rst`.
  Sphinx's dummy build returned zero and its environment confirms that index
  relationship. There are no warnings for the usage page. Other documents
  produce 111 warning lines; see `sphinx/warnings.log`. The first post-build
  assertion also matched this separate acceptance report's filename; the
  exact-page check in `sphinx/index-check.json` corrects that assertion.

All new logs, probes and arrays are under
`artifacts/terrain-dem-resolution/remediation/` in the worktree.
The tested rectangular-correction native SHA-256 is
`b5d4c16c10d3c9847e64b2d813936bd6908b25b42052be099062b0e1bdce175c`.
The test-file SHA-256 is
`136781553dc421579f2a39887c17c4d17b2ca617ba1f0139d24840ecf2cf9aa9`;
the complete current file ran after its final edit.

Observed checks:

- `maturin develop --release --offline` passed, with the same pyproject
  release features and private C: compiler target (6m36s compilation).
  `python -m maturin` initially could not locate the executable in this
  system-site-packages venv; invoking the installed CLI with `VIRTUAL_ENV`
  explicitly set completed the required build.
- `cargo forge3d-clippy`, `cargo fmt --check` and `git diff --check` passed.
  Compiler incremental-cache hard-link warnings were logged; there were no
  Clippy failures. No dependency or gate changed.
- Four camera/grid Rust tests, six shader-assembly tests (including Naga
  validation), and the dry-run `det_instrument_rewrite` test passed. The latter
  required no WGSL rewrites; its existing hardware-sampling advisories remain.
- **30 tests passed with zero skips** on the rebuilt extension: all 19 cases
  in `tests/test_terrain_dem_mesh.py`, two named TV12 offline checks, and all
  nine cases from the existing terrain visual/TV10 golden files. The golden
  cases also passed on the old binary before rebuilding. Update mode was
  unset and no reference image changed. The canonical JUnit zero-skip checker
  passed for both reports. The GPU suite sets `WGPU_BACKEND=vulkan`,
  `FORGE3D_NO_BOOTSTRAP=1`, `FORGE3D_TESSELLA_REQUIRED_GPU=1`,
  `VF_COLOR_DEBUG_MODE=0`, `FORGE3D_RUN_TERRAIN_GOLDENS=1` and
  `FORGE3D_TERRAIN_GOLDEN_VARIANT=nvidia-vulkan`; `CI` and `GITHUB_ACTIONS`
  are unset. It includes 700-by-1025 and 1025-by-700 shadow/reflection cases
  with explicit water masks, and singleton-axis GPU cases.
- **144 arrays remain byte-identical** to freshly rebuilt current main using
  the original frozen benchmark. **30 additional arrays are byte-identical**
  across this shader correction with shadows, water and reflections enabled,
  for screen, mesh and mesh:zup through beauty/AOV/offline paths. Script,
  settings and input hashes match within their respective comparisons. These
  are exact array comparisons; golden tests retain their existing tolerances.

### Real steep-relief source and measured error

`probe_relief.py` reads an unresampled, entirely finite source window from
`D:/forge3d/data/tracy_arm/TracyArmMerged.tif`: EPSG:26908, 5 m source pixels,
column 6926, row 3188, width 700, height 1025, bounds
`(611000, 6410995, 614500, 6416120)`. Raw float32 elevations span
**-273.25458 to 1770.90210 m**, approximately 2044 m of relief.
Input SHA-256: `c8a583fa2400158336af6dd1eb5affba6e5e239204a3391e7705901833051074`.

The 512-by-384 render uses span 5120, z-scale 1, radius 8500, theta 75°,
phi 0°, clip `(1, 30000)`, the measured raw height clamp, MSAA 1, and the
same material/lighting in each mode. The existing single square world span
is retained; this probe tests sample fidelity, not new rectangular extents.
The independent CPU full-DEM projection covers 12,755 pixels:

| Mode | Silhouette mismatches | Tracked host-visible peak |
| --- | ---: | ---: |
| mesh:zup | 9 | 1024 B |
| mesh:zup:dem | 4 | 1024 B |

The remaining four mismatches are reported, not treated as exact reference
agreement. The earlier square-grid opt-in also measured four errors on this
real crop; the rectangular spike regression exposes the corrected sampling
loss directly. The tracked peaks precede array readback and satisfy the
existing 512 MiB render budget; they do not measure total process memory.
Adapter metadata proves physical RTX 3070 / Vulkan with no software fallback.
Raw inputs, reference mask, render arrays and allocation details are retained
in `relief-before/` and `relief-after/`.

### Approved source pin and final verification

The WGSL correction changes the normalized assembled terrain source hash from
`0xeb4175bc1e2b5963` to **`0x944106271abcdd3c`**. Both an independent assembly
calculation and the existing Rust source-pin test confirm that value. On
2026-10-20 Milos explicitly answered **“Approve this source-hash update”**
to the exact old/new value request. Only `PINNED_TERRAIN_SOURCE_HASH` and its
approval comment in `src/verify/ir/engine.rs` were changed. The earlier red
source-pin log is retained as historical evidence; that test now passes.

The approved final native SHA-256 is
`2fc382967e606368b2b34ff32eecd43711b273f293517e19b005feca17ce659c`.
Logs and same-process adapter/binary provenance are under
`artifacts/terrain-dem-resolution/remediation/final-approval/`:

- `maturin develop --release --offline` passed with the unchanged pyproject
  feature set and private C: target (8m57s compilation); the rebuilt native
  extension was installed before final Python verification.
- `terrain_renderer_source_hash_is_pinned` executed and passed. Canonical
  Clippy, formatting and the final whitespace check also passed.
- **Four final tests passed with zero skips** in 113.97s: the complete proven
  list, unsafe-fixture rejection, seeded guard-removal controls, and a physical
  700-by-1025 DEM render with shadows, explicit water and planar reflections.
  `shader-report.json` reports **127 proven entries across 12 modules** with
  no suppressions. The negative controls remain unproven as required.
  The canonical JUnit zero-skip assertion passed. Same-process metadata confirms
  physical RTX 3070 / Vulkan, NVIDIA 610.60, `software_fallback=False`.
  Environment: `WGPU_BACKEND=vulkan`, `FORGE3D_NO_BOOTSTRAP=1`,
  `FORGE3D_TESSELLA_REQUIRED_GPU=1`, `VF_COLOR_DEBUG_MODE=0`; `CI` and
  `GITHUB_ACTIONS` unset.

The 30-test GPU/offline/golden run and 174 exact array comparisons above are
retained for the unchanged WGSL/geometry. Between those captures and final
verification, only the approved verifier pin/comment changed; the geometry
test-file hash remains identical. No golden, signed certificate, other
determinism reference or dependency was changed. The conversion-inventory gate
transition approved during publication is recorded below. Publication was
subsequently authorized. Unrelated file deletions
present on resumption were preserved, not restored or included in this change.

Full acceptance CI, other physical backends and the film compositor remain unrun. No
fresh certificate/signing claim is made. Reusable guidance proposed, not
applied to agent contracts: geometry-density proof should include a non-square
input in each orientation and test alignment to source samples on both axes.

### Approved conversion-inventory transition and PR checks

Publication is PR [#261](https://github.com/milos-agathon/forge3d/pull/261).
The first hosted CI run passed formatting, Clippy and the native build, then
Fast Contract reported eight failures from the frozen conversion inventory
(583 passed, 29 skipped). The DEM grid changes three cast-site records and
removes four old records, changing the raw inventory from 1856 to 1855 sites.

On 2026-10-20 Milos explicitly approved this exact transition in
`tests/test_world_coord_f32_gate.py` and
`tests/data/world_coord_f32_ledger.json`. The new record pins both complete
inventories, checks the exact three additions and four removals, reconstructs
the main inventory, then runs the existing historical checks. Every earlier
ledger record, frozen digest, scanner rule and rejecting probe is retained.
New negative controls reject missing, duplicate, changed and extra sites.
The operands are integer render-grid dimensions; no world-coordinate cast is
introduced.

The active command
`python -m pytest tests/test_world_coord_f32_gate.py -q --junitxml=artifacts/terrain-dem-resolution/remediation/ci-inventory-active.xml`
passed **31 tests in 71.67s**, with zero skips. It used the worktree venv,
`FORGE3D_NO_BOOTSTRAP=1` and `PYTHONPATH` set to the worktree's `python/`.
The log and JUnit report are retained under `remediation/`. These publication
changes affect tests and documentation only; the recorded native/GPU evidence
above applies to the unchanged production source. Hosted checks must pass on
the latest PR commit before the local roadmap item is closed.
