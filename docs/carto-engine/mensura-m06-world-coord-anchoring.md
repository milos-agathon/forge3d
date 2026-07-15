# MENSURA M-06 Option 2 — Full Geospatial Viewer Anchoring

Status: implementation complete on `codex/m06-fgv-audit-remediation`; merge
acceptance remains pending until the required NVIDIA/Vulkan PR job is green with
zero skipped M-06 tests. No signed recipe certificate refresh is required: the
packaged target recipe passes and its shared signed shader hashes are unchanged.

The authoritative contract is
[`mensura-m06-full-geospatial-viewer-spec.md`](./mensura-m06-full-geospatial-viewer-spec.md).
This note describes the implementation and the evidence that the PR must
produce. It does not substitute local preflight results for the required CI
result.

## Runtime contract

The interactive viewer owns one persistent `camera_anchor`. At the beginning of
each frame it selects one complete camera pose using terrain, point-cloud, then
general-camera precedence. Terrain rebases against the horizontal camera target;
the other camera modes rebase against the eye. The selected `FrameCamera` copies
the anchor and supplies view, projection, screen render, snapshot, motion blur,
labels, point clouds, fog, GI, sky, objects, and picking for that frame.

Absolute world positions remain `f64` until subtraction from the copied anchor.
`Anchor::narrow` in `src/camera/anchor.rs` is the only world-coordinate
`f64 -> f32` implementation. Render vertices, colors, normalized UV extents,
local density-volume coordinates, and local scatter transforms remain `f32`
because they are not absolute world coordinates.

The production rebase site is `Viewer::prepare_frame_anchor`. A rebase performs
one deterministic refresh before encoding:

- point-cloud, vector, label, object, and picking caches are repacked from their
  persistent absolute or local sources against the new anchor;
- point-cloud and vector GPU buffers are updated through existing `COPY_DST`
  allocations;
- vector BVHs are rebuilt from the actual render vertices and feature IDs;
- TAA, SSGI, SSAO, SSR, fog, motion, and terrain histories are invalidated in
  place; and
- the previous view-projection state is reset to the current copied frame.

SSGI and TAA history reset APIs are infallible and allocation-free. The resource
ledger, rebase count, history-invalidation count, active camera, effect flags,
point-cloud CPU/render/device bytes, and host-visible budget state are exposed by
`get_stats` for live adjudication.

## Coordinate ownership

| Source | Persistent representation | Render boundary |
|---|---|---|
| General, terrain, and point-cloud cameras | `DVec3` eye/target | copied `FrameCamera` through `Anchor::view_look_at` |
| Object transforms | local mesh plus `DVec3` translation | one anchor-relative model translation; rotation/scale remain local |
| Terrain | `RasterInfo`, `DVec2` origin/span | physical footprint `(c,-f)` and `(a*w,-e*h)` packed from the copied anchor |
| Point cloud | LAS-derived `Vec<PointSource3D>` with `DVec3` positions | persistent f32 render cache and existing `COPY_DST` instance buffer |
| Vector overlay | typed 8-lane rows: XYZ `f64`, RGBA `f32`, ID `u32` | anchor-packed render vertices; drape UVs use the physical terrain footprint |
| Labels/callouts | `DVec3` point or `Vec<DVec3>` polyline | anchor-packed projection cache |
| CityJSON | transformed positions and tessellation in `f64` | local direction conversion through `Anchor` |
| Picks | render-frame intersections | restored to absolute `f64` with the exact copied frame anchor |
| Scatter/density volumes | terrain-local `f32` contract | scaled through the same non-square physical origin/span as terrain |

IPC absolute-world fields use `f64`. Vector rows reject any length other than
eight, non-finite XYZ/RGBA, non-integral IDs, negative IDs, and IDs above
`u32::MAX`. Commands are stable-partitioned so frame-establishing loads/camera
changes run before content commands in the same batch. Camera, content, object,
point-cloud, vector, label, and scene-review changes validate against one copied
prospective frame before publishing state.

## Terrain metadata policy

Only a genuinely absent GeoTIFF transform selects the legacy local footprint:
origin `(0,0)`, span `(width,height)`, and no invented CRS. A present transform
must be finite, north-up, axis-aligned, positive in map X, negative in map Y, and
produce finite positive spans. Rotation, shear, mirroring, south-up axes, zero
span, malformed metadata, and metadata-read failures are typed errors.

`load_terrain` validates metadata synchronously at the IPC trust boundary before
the command is enqueued, before a terrain scene is constructed, and before GPU
allocation. Simple relief, PBR, CSM, vector draping, density volumes, scatter,
DoF, screen rendering, and snapshots use the same physical footprint and copied
anchor.

## Atomicity and bounds

Earth-scale coordinates are valid. The safety boundary applies only to the
anchor-relative residual and rejects any camera or content point more than
1,000,000 m from its prospective frame. Non-finite data is always rejected.

Scene-review installation is transactional. It validates the complete effective
scene, stages all new raster/vector/label/scatter runtime objects, rolls the
staged IDs back on any failure, removes the previous runtime only after the stage
succeeds, and publishes the registry snapshot last. A rejected finite-but-distant
label or vector therefore cannot partially update the runtime or query surface.

## Required evidence

The required CI job is `M-06 Full Geospatial Viewer (NVIDIA Vulkan)` in
`.github/workflows/ci.yml`. It runs on the self-hosted NVIDIA Windows runner with
`WGPU_BACKEND=vulkan`, checks out LFS data, installs the current wheel, builds a
fresh release `interactive_viewer`, requires a real hardware terrain probe, and
runs the source gates plus `tests/test_m06_full_geospatial_viewer.py`. A standard
JUnit parser fails the job for zero collected tests, any failure/error, or any
skip. `ci-success` requires this job.

The live file records backend, images, frame metrics, resource counts, and viewer
logs under `tests/artifacts/m06`. It measures:

- pixel parity between missing-transform local terrain and the same terrain at
  translated Earth-scale coordinates;
- fail-closed signed-transform rejection with unchanged GPU ledger counts;
- non-square PBR/CSM/DoF/volumetric/scatter rendering with enabled
  TAA/SSGI-temporal/SSR/fog telemetry;
- a full terrain orbit with zero rebase/history churn plus consecutive-frame
  no-flash SSIM/MAE;
- object-transform and 500,000-point coexistence under terrain camera ownership,
  exact source/render/device byte reporting, and ten allocation-free rebases;
- live one-millimetre red/blue separation at Earth magnitude with a zero-distance
  negative control; and
- scene-review rollback after a render-frame residual rejection.

Source gates independently reject hidden component/index/helper narrowing,
multiline or duplicate matrix producers, additional production rebase sites,
raw GPU allocations, and historical local-frame enforcement patterns. Unit tests
also cover arbitrary/missing CRS, malformed metadata, non-square scatter and
density mapping, exact typed-vector parsing, vector precision across repeated
rebases, point-cloud buffer reuse, active-camera precedence, snapshot
composition, mouse picking, and transaction ordering.

## Local preflight evidence

The pre-PR Metal preflight used the normally packaged ABI3 wheel and a freshly
built release viewer, not the editable development extension. The required live
file completed `4 passed` with a zero-skip JUnit result. These results are useful
regression evidence, but they do not replace the required NVIDIA/Vulkan job:

- local/translated-terrain parity: SSIM `0.9999999993148123`, MAE
  `7.978e-09`, maximum channel difference one byte; the signed-transform
  negative control was rejected without changing the `1047` buffer / `26`
  texture ledger;
- all required effects reported enabled; a full orbit produced exactly one
  initial rebase/history invalidation and no additional churn, steady resources
  at `1075` buffers / `54` textures, and consecutive-frame no-flash SSIM `1.0`
  with MAE `0.0`;
- host-visible peak was `656404` bytes and the tracked total was approximately
  `325.96` MB, both within policy;
- the 500,000-point case reported exact source/render/device totals of
  `24000000` bytes each, retained terrain camera ownership, and held resource
  counts at `1051` buffers / `26` textures through ten rebases;
- the live millimetre case rendered one red and one blue pixel exactly one pixel
  apart, while the zero-distance control produced no red separation; and
- the scene-review residual rejection preserved the prior runtime and registry
  exactly.

The signed `mapscene_terrain_raster` recipe golden also passes against the
packaged wheel on Metal. Viewer shadow depth now has its own 128-byte camera ABI
shader, while the offscreen signed recipe keeps its original shared 112-byte
shader. The shared `terrain.shadow_depth.shader` and `terrain_pbr_pom.shader`
hashes therefore remain unchanged and no protected certificate-refresh run is
needed for this change.

## Certificate policy

This change does not weaken recipe certificate verification and does not write a
private signing key or unsigned replacement certificates. The protected
certificate-refresh workflow remains the only authorized signing path. The
packaged target recipe verification above establishes that this remediation did
not change its signed dependency hashes. Any future reviewed WGSL dependency
change that does alter a signed recipe hash remains blocked until the protected
workflow produces and verifies the refreshed certificate set.
