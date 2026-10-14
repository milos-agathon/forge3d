# Coherent viewer viewport targets

Architecture review item 24 is implemented in the existing viewer API. The fresh
worktree started at `a5e37efd17ed9c5a2cb0a94c7314876d0aa93c30`; the PR branch
was then based on current main (`1d057506`) before committing.
`Viewer::resize_render_targets` prepares one complete `PreparedViewportTargets`
without changing the live viewer, checks GPU validation and allocation errors,
then installs it without allocation or fallible work. A failed preparation drops
its partial replacement and preserves live textures, views, caches, settings,
histories, accounting and surface configuration. Zero and device-limit-exceeding
explicit target sizes return errors; minimized window events remain ignored.

`TargetKind` shares initial and replacement descriptors for lit output, GI
baseline/diffuse/specular/output/debug, sky, geometry depth, and full/half fog.
The set also owns replacement G-buffer, HZB mips, allocated SSAO/SSGI/SSR targets
and their histories, SSGI scene histories, and viewer TAA histories. Effects retain
pipelines, uniforms, enabled flags, parameters, environment inputs, frame state
and half-resolution mode. GI dispatch dimensions follow the new viewport.
Terrain keeps its separately owned screen targets and has its history invalidated.

Installation clears bind groups that reference replaced attachments, including
sky presentation. One allocation-free `invalidate_viewport_histories(width,
height)` owns terrain/GI/TAA/fog resets, previous projection and jitter for both
resize and anchor rebase. The IPC `history_invalidation_count` retains its original
rebase-only meaning; viewport resets do not increment it. The source contract test
now checks both callers and every reset in this shared helper.

Window resize configures the surface after successful installation. Same-size
resize and `Lost`/`Outdated` recovery configure the surface without allocation or
history reset. A failed new native extent suspends incompatible frames and redraw
requests until another resize event; surface recovery never retries the allocation.
Returning the native window to the live extent allows presentation immediately.
Offscreen target resize continues to keep surface configuration separate.

## Descriptor and rendering corrections

GI debug consistently uses the shader-required `Rgba16Float`. GI diffuse and
specular now have `COPY_SRC` at initialization as well as resize. Half-resolution
fog keeps `COPY_SRC`; half sizes clamp floor division to one texel. Fog dispatch
and history copies use the allocated half texture dimensions, including 1x1.

TAA neighborhood bounds use a WGSL struct with unchanged min/max calculations.
This fixes the FXC array-return compilation error on Direct3D12. Pipeline creation
returns its validation error instead of retaining an invalid pipeline. After TAA
execution swaps its histories, the viewer presents the newly written read slot;
the former write-slot lookup presented an empty first frame after reset. The
existing GI reexecution path also supplies the lit output storage binding and
uses the current lighting layout's environment/sampler/uniform binding numbers.

## Memory trade-off

Preparation deliberately holds old and new viewport resources together to make
failure atomic. Tracked resident estimates therefore include both sets and exclude
driver overhead; this is not measured hardware VRAM. The 512 MiB host-visible
budget does not bound device-local textures. Native out-of-memory error scopes
remain necessary, and the tests inject failures rather than exhaust real VRAM.

SSAO noise and SSGI/SSR environment textures and samplers are created only during
initialization. Resize prepares only viewport textures. The capture verifies their
per-label peaks do not grow. For the same enabled 320x240 fixture preparing 101x67,
tracked live usage was **84,155,876 bytes**, the replacement **1,441,308 bytes**,
and the simultaneous peak **85,597,184 bytes**. Dropping preparation returned to
**84,155,876 bytes**; viewer shutdown returned the owned group to zero.

## Tests and evidence

The descriptor test uses independent expected format/usage constants. The Windows
GPU test injects at viewer targets, G-buffer, HZB, SSAO, SSGI scene history, SSR
history and the final TAA texture through the real tracked allocator. It checks
all live allocation identities and released replacement bytes, plus actual wgpu
validation failure. It exercises native window allocation failure, non-retrying
surface recovery, same-size recovery, offscreen resize, odd sizes, 1x1, and restore.
HZB dimensions/mips, every effect attachment and history, GI dispatch size and TAA
textures are checked. Nondefault GI/TAA settings survive replacement.

The fixture has a white material, a camera distance of 3 and near plane of 1,
320x240 captures, all screen-space effects, half-resolution SSGI and TAA. It saves
lit, full/half fog, sky and a clamped visualization of the actual GI HDR output.
Both fog modes change the lit image; GI/fog readbacks contain finite radiance;
sky and lit captures contain spatial detail. Terrain-owned TAA is populated with
an executed GPU resolve, then invalidated without replacing its allocation IDs.
These are focused behavioral checks, not golden-image or visual-quality certificates.

Local results on Windows, 14 October 2026:

- NVIDIA GeForce RTX 3070, Direct3D12: descriptor and physical test passed, with
  no wgpu validation error in the checked frames, GI execution and readbacks.
- Microsoft Basic Render Driver, Direct3D12 WARP: both tests passed with explicit
  fallback selection and a CPU-adapter assertion. This is software evidence.
- RTX 3070, Vulkan, NVIDIA driver 610.60: both tests passed, but Vulkan logged
  `VUID-vkQueueSubmit-pSignalSemaphores-00067` presentation errors, including before
  resize. This remains a pre-existing presentation limitation; no clean Vulkan
  validation claim is made. Clean native frame evidence comes from Direct3D12.
- Canonical `cargo forge3d-clippy`, `cargo fmt --check`, formatting checks on all
  changed Rust sources, `git diff --check` and CI YAML parsing passed.
- The 23 allocation, temporal and exact matrix-inventory source checks passed
  with `FORGE3D_NO_BOOTSTRAP=1` and `PYTHONPATH=tests`. The inventory retains the
  same four reset-matrix operations at their new shared owner, which consumes the
  frozen frame. Python native bindings were not exercised.

The exact narrowing inventory records the ten new u32 viewport-size conversions,
with Milos's explicit approval on 14 October 2026. Removing those exact entries
restores the previous frozen inventory; historical hashes and rejection probes
remain intact. No world-coordinate conversion or scanner exemption was added.

The existing physical NVIDIA Windows CI job now runs these Rust tests with its
required portable feature matrix on Direct3D12 with an explicit NVIDIA/discrete-adapter assertion, preserves commands,
logs and captures, and requires both tests to execute without skips. Its existing
M-06 NVIDIA Vulkan gate is restored for the subsequent acceptance steps. The
hosted Windows test is not skipped; the local WARP run checks its software path.
Hosted CI and that modified self-hosted job have not been dispatched locally.

Run commands from inside the worktree so Cargo aliases resolve. Use its own
`CARGO_TARGET_DIR` to avoid other checkouts' build locks. For a physical run:

```powershell
$env:WGPU_BACKENDS = 'dx12'
$env:FORGE3D_VIEWPORT_REQUIRE_PHYSICAL = '1'
cargo test --lib --features default,async_readback viewer::render_targets::tests -- --test-threads=1 --nocapture
cargo forge3d-clippy
$env:FORGE3D_NO_BOOTSTRAP = '1'
$env:PYTHONPATH = 'tests'
python -m pytest tests/test_allocation_gate.py tests/test_m06_temporal_resource_contract.py -q --basetemp .task-evidence/pytest
```

For WARP, set `FORGE3D_VIEWPORT_REQUIRE_PHYSICAL=0` and
`FORGE3D_VIEWPORT_TEST_FALLBACK=1`. The fallback switch exists only in Windows Rust
test builds. `FORGE3D_VIEWPORT_EVIDENCE_DIR` selects the capture directory.
When checking individual changed Rust files, restrict untracked source selection
to `src/`; pytest creates intentionally malformed Rust fixtures in temporary data.
A repository-wide untracked `*.rs` selection incorrectly includes those fixtures. No
production dependency, golden, determinism hash, public serialized format or agent
contract was changed. Cross-backend portability qualification is ABSENT: no
qualifying committed golden was exercised. Other OS lanes, the full Python/live
M-06 matrix and real GPU memory exhaustion were not run.

Saved final logs, input hashes, commands and captures are in
`C:/Users/milos/.codex/visualizations/2026/10/14/01a13a3b-bffa-7ac3-b78e-44e59c14631e/viewer-resize-evidence/review-fixes/`.
Use the `final-*` logs and backend subdirectories. Earlier diagnosis failures
remain there for provenance and are not final validation results.

## Review prompt

Review the PR diff against its merge base with main for item 24 in
`D:/forge3d/docs/architecture-review-forge3d-50.html`. The architecture assessment
used source `a5e37efd`; the PR starts from main `1d057506`. Do not edit or use subagents. Verify that
one internal module owns allocation, texture/view pairing and coordinated history
reset; preparation failure preserves and releases resources; installation preserves
settings, formats/usages and accounting; surface recovery does not retry allocation;
all attachments/mips/dispatch sizes agree after odd, repeated and 1x1 resize; and
post-reset frames use fresh history. Inspect the focused tests, physical/WARP
captures and final command/input-hash evidence described here. Distinguish clean
Direct3D12 evidence from Vulkan presentation errors and unrun CI. Report only
concrete requirement failures with location, evidence, impact and correction,
then a requirement-by-requirement verdict.
