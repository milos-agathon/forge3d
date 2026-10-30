# NEPHELE scheduler inputs and real-DEM timing — 2026-10-24

Head **024c7803d17814c447637208cb4535112c9ac166** is pushed to PR #172.
[NEPHELE run 40075114543](https://github.com/milos-agathon/forge3d/actions/runs/40075114543)
is green on this exact checkout and installed-wheel revision. Physical JUnit is
**6 passed, 0 failed, 0 errors, 0 skipped**. All six unchanged gate functions
also pass when rescored locally from the downloaded artifact with CPU-only code.

Milos accepted the measured performance costs for integration on 2026-10-24, with the slowdown to be stated explicitly in CHANGELOG. His exact words were:

> Accept and document it: merge with the slowdown stated in the CHANGELOG. I'll then write the remaining merge-prep instructions: the certificate fix, CHANGELOG, roadmap, full CI, and a stop before merge.

This records performance acceptance and the disclosure requirement. Certificate remediation, CHANGELOG, roadmap and full CI await his remaining merge-prep instructions. Merge remains a separate hard stop; no merge is authorized by this record.

The original **1500×1500 Gore Range DEM**, at native 1 m spacing and full relief,
rendered through `render_terrain_pbr_pom` at **1920×1080** on NVIDIA RTX 3070,
Vulkan, driver 610.60, with software fallback false. The input SHA256 is
`bcacfdfabefc7ef4e7b9a9cefe12375b491fe1769218ad3c3ddd3ae30f9d93bc`.

| Observation | Render call (ms) | Completed frame, including output readback (ms) | Output readback (ms) |
|---|---:|---:|---:|
| First frame | 76,146.792 | 76,151.004 | 4.211 |
| Unchanged frame | 324.768 | 328.481 | 3.713 |

These are two individual observations, without an average, percentile, FPS
target or before-fix speedup claim. Before this fix the API raised before
returning its first frame. The first cloudlight timestamp interval is
75,656.008 ms; the unchanged interval is 0.001024 ms. That interval includes
GPU idle gaps while the CPU reads convergence progress, so it is not a pure
compute-time measurement or a measurement of any single dispatch.

The explicit **untimed capture after both observations** reports:

- `orders=28`, `converged=true`, `cache_reused=true`;
- `maximum_source_change=0.0`, `maximum_surface_change=0.0`;
- surface cache **242×137**, following the viewport froxel grid rather than
  the 1500² DEM;
- tracked host-visible peak **44,535,312 bytes**, below the existing
  **536,870,912-byte (512 MiB)** limit;
- froxel device-local allocation **1,676,418,092 bytes**.

The producer does not save separate convergence/cache-reuse diagnostics for
each timed frame. The flags above belong to its final untimed capture. No
iteration cap was introduced. The frozen 64² physical capture converged at
order 22 with progress `[0, 0, 0, 0]` and both reported changes zero.

Two commits implement the fix:

- `3c132f933f474f6a7433ff065a545ba355c6a7e4`: content keys and normal-draw
  integration. [Its run](https://github.com/milos-agathon/forge3d/actions/runs/40072668949)
  passed all six physical gates, then exposed the resolve validation error.
- `024c7803d17814c447637208cb4535112c9ac166`: resolve validation uses the actual
  compiled source type. A sampled media composite does not require the
  attachment transition that the legacy beauty source requires.

Changed source files and their purpose:

- [media.rs](../../src/terrain/renderer/media.rs): hash the exact uploaded
  extinction, light-transmittance and world-medium bytes with canonical medium
  identity, density/resource versions, dimensions and sun input. Carry producer
  identities for previous history, radiance and directional state without GPU
  readback. Initial state identifies wgpu's defined zero contents and extent.
- [draw/mod.rs](../../src/terrain/renderer/draw/mod.rs): supply all graph inputs
  that are read before their first writer. Validate the actual shadow transition
  at the first shadow reader, which can be injection in directional mode, and
  validate resolve according to its compiled input type.
- [draw/leaf_keys.rs](../../src/terrain/renderer/draw/leaf_keys.rs): one production
  key map shared with CPU tests for both transport modes, downstream key
  invalidation, first-reader shadow validation and sampled/attachment resolve.

The CPU regression first failed against the height-only map with the same
`nephele.media.world.medium` missing-key error as the benchmark. It now traverses
the scheduler successfully in both modes. Mutating each supplied leaf changes
the resolved-frame key. Media keeps its existing disabled content-store path;
the scheduler still computes and validates keys on that path.

The resolve follow-up follows the owner's existing rule: “Don't assert barriers
for `SampledTexture`.” The non-media beauty source still requires its own actual
transition, and CPU coverage rejects an absent or unrelated transition. No
existing assertion was changed.

Image metrics are exactly unchanged from `98cc1df1`:

| Criterion | Before | After | Existing criterion |
|---|---:|---:|---|
| B1 pass fraction | 1.000000 | 1.000000 | ≥ 0.95 |
| Shaft SSIM | 0.992584 | 0.992584 | > 0.95 |
| G4a pass fraction | 0.997431 | 0.997431 | ≥ 0.95 |
| Medium changed fraction | 0.387035 | 0.387035 | ≥ 0.10 |
| G4b relative error | 0.008058 | 0.008058 | ≤ 0.20 |

G4b's measured signed response is 0.022053710105061672 against the unchanged
open-surface target 0.02187741965233074. Gates 1, 2 and 5 also have exactly
unchanged metrics; all raw before/after values and margins are in the JSON.
Gate 5 validates 356 shader assemblies with zero compute comparison-sampling
calls. Gate 6 proves identical diagnostics and bitwise-identical independent
physical frames, with hash
`de2e1b8020ea809eddcdf37e803b2da8412c1db7d6bb5d3950fdba578ea1f3bc`.
Its tracked host-visible peak remains 2,242,684 bytes.

Validation actually run:

- Final CPU scheduler/resolve suite: **2 passed**, exercising both transport
  modes and the legacy non-media resolve guard; build/run 130 seconds.
- Upload-identity CPU test: **1 passed**; media frame-graph tests: **5 passed**;
  media shader unit test: **1 passed**.
- `cargo fmt --check`: passed, 0.778 seconds.
- `cargo forge3d-clippy-acceptance`: passed, 42.666 seconds; the required
  routine pre-commit hook also passed.
- f32 source inventory: **1,936 sites**, unchanged SHA256
  `8f9f378ce8e27a98e9b99961073ae70128584485778adf1dc2c597c464c0cee7`.
- Installed-wheel fast JUnit: **661/0/0**, including the 600 MiB named-budget
  error case passed. Shader/environment JUnit: **10/0/0**, including both GPU
  environment cases. Physical JUnit: **6/0/0**. All three zero-skip checks passed.
- Exact checkout, both physical capture revisions, benchmark revision and both
  timing-certificate engine revisions match the final head.

Rust unit features were
`extension-module,default,cog_streaming,shader-contract-asserts,splat-fusion`.
Local Cargo used this worktree's `target`. The acceptance Clippy feature set is
the repository's unchanged `forge3d-clippy-acceptance` alias. Full commands,
durations and environment are preserved in the JSON and evidence logs.
Zero-skip CI used `FORGE3D_ALLOW_HOSTED_WINDOWS_TERRAIN=1`,
`FORGE3D_NO_BOOTSTRAP=1`, `FORGE3D_TEST_INSTALLED_WHEEL=1`,
`FORGE3D_NEPHELE_LANE=windows-nvidia-vulkan`, candidate SHA equal to the final
head, and `WGPU_BACKEND=vulkan` / `WGPU_BACKENDS=vulkan`. The direct trace step
also used `FORGE3D_NEPHELE_TRACE=1`.

Terrain, hybrid and determinism pins remain respectively
`0x17ad_c1fb_40ed_8189`, `0x32c0_583b_cf1a_f865` and
`0xd904_56ee_98b5_6d96`. Shader contracts, f32 ledger, committed goldens and
certificates, fixture, gates, G4 policy, existing tests/assertions and reference
RNG overlap were not changed. No new GPU allocation or dependency was added.

The owner's action authorization was: “Commit, push, dispatch `scope=nephele`,
and report the physical JUnit (expect 6/0/0) and the 1500² timing: first and
unchanged frames, with the convergence and cache-reuse fields.” No new protected
value approval was needed.

Residual limits remain visible. The unsigned runtime certificates report
`shader_provenance_missing` for nine first-frame and three unchanged-frame
NEPHELE pipeline uses. Those degradation records are preserved; registration,
certificate and ledger remediation were not applied. The real-DEM cold and
unchanged RGBA hashes differ, and no large-DEM reference-quality or warm-frame
bitwise-identity claim is made. The physical Gate 6 proof is for independent
64² captures. D3D12, Metal and WASM are **ABSENT**. Local GPU execution, full
release CI, merge, tag and publishing were not run.

A reusable steering gap was observed in the f32 source scanner: a Rust lifetime
can pair with an apostrophe in a new test comment and mask a `cfg(test)` marker.
Comparing the inventory tuples identified the false positive. Wording the new
comment without that apostrophe restored the identical production inventory;
no scanner or ledger was changed. An owner-reviewed lexer correction is proposed
separate work, not applied here.

[Full JSON](nephele-scheduler-leaf-keys-2026-10-24.json) and
[CPU-only reproducer](nephele-scheduler-leaf-keys-2026-10-24.py) are saved beside
this note. Downloaded official evidence is in
`../../.release-evidence/2026-10-24/ci-40075114543/`; command logs and source
manifests are in `../../.release-evidence/2026-10-24/media-leaf-keys/`.
These audit files are kept locally without a post-capture commit, preserving
the exact green head.

Reproduce from `D:/forge3d/.worktrees/nephele-146-release` in PowerShell:

```powershell
& .release-evidence/2026-10-22/analysis-venv/Scripts/python.exe docs/audits/nephele-scheduler-leaf-keys-2026-10-24.py
```
