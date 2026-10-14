# Scene pick assembly

Architecture review item 36 is implemented by
`src/viewer/input/scene_picking.rs`. Mouse release and the `pick_at` IPC command
both enter `Viewer::pick_at_screen` with the same frozen frame camera.

The scene collector reads BVH, object, visible point-cloud and label hits
without changing selection. It restores render-relative BVH and object
positions through that frame's f64 anchor. Point-cloud hits keep their exact
source f64 position, and labels keep their retained f64 world position.
Feature IDs, layer names, attributes and terrain information are retained.

Ray hits are ordered by distance, with stable ordering for equal distances.
The existing screen-label priority is retained: a picked label precedes ray
hits. Label depth is normalized screen depth, so it is not compared with ray
distance. No label or geometry capability is added by this change.

After collection, `UnifiedPickingSystem::apply_assembled_pick_event` applies
the click policy once. Both the viewer's selected feature/layer and its
selection manager use the first assembled result. Milos approved this behavior
on 14 October 2026; previously object and point hits could be reported without
being selected, and a label could cause a second selection mutation.

- A normal click replaces selection with the first hit.
- Shift toggles the first hit when multi-select is enabled; otherwise it uses
  the normal replacement policy.
- Ctrl is retained in the event and has no separate selection policy.
- An empty click clears selection unless Shift is held. Empty Shift clicks
  retain the selection set, while the viewer's selected feature/layer clear.
- The viewer emits its existing event shape for nonempty picks. The public
  BVH-only `handle_pick_event` route retains its previous behavior, including
  leaving selection alone when no BVH hit is found.

The Rust regressions cover mixed sources, depth and label priority, identity,
attributes, exact point/label coordinates, rebasing and modifier combinations.
On Windows, the acceptance test also calls the shipped screen-pick route using
a hidden viewer, a real LAS fixture and the actual BVH, object, point-cloud and
label pick implementations. It checks the returned event, viewer state and
selection set before and after rebasing. This exercises picking, not image
quality or cross-backend render equivalence.

The existing world-coordinate source guard follows the assembly function.
Its narrowing inventory reverses only the two exact screen-coordinate cast
locations moved into the collector before checking the unchanged freeze
hashes. Missing, duplicated or altered casts still fail the guard.

## Local validation — 14 October 2026

Worktree: `D:/forge3d/.worktrees/scene-pick-assembly`, based on `origin/main`
revision `a5e37efd17ed9c5a2cb0a94c7314876d0aa93c30` for the initial validation.
This records implementation evidence; merge status is tracked by the PR.

Changed files: `src/viewer/input/scene_picking.rs`,
`src/viewer/input/mod.rs`, `src/viewer/input/viewer_input.rs`,
`src/picking/unified.rs`, `tests/test_world_coord_f32_gate.py` and this document.

- `picking::unified::tests`: 5 passed, including every Shift/Ctrl/multi-select
  combination, empty hits and non-click events.
- `viewer::input::scene_picking`: 5 passed. The Windows screen-pick acceptance
  used NVIDIA GeForce RTX 3070, Vulkan, driver 610.60. It loaded a three-point
  LAS fixture, collected real object/BVH/point/label hits, checked selected
  state and modifiers, and rebased onto the moved camera eye. Point and label
  positions remained exact; geometry retained the M-06 0.5 mm comparison.
- 39 source checks passed across the focused runs: the anchoring and matrix
  guards, the world-coordinate inventory, and the exact cast-location move.
  Source-only pytest used `FORGE3D_NO_BOOTSTRAP=1`. The separate native Python
  `test_public_camera_helper_preserves_earth_scale_offset` was not run.
- `cargo forge3d-clippy`, focused `rustfmt --check`, and `git diff --check`
  passed after the final source changes.

The Rust test feature set was
`default,async_readback,copc_laz,cog_streaming,gis-remote,geos-topology,weighted-oit,wsI_bigbuf,wsI_double_buf,enable-pbr,enable-tbn,enable-normal-mapping,enable-hdr-offscreen,enable-renderer-config,enable-staging-rings,shader-contract-asserts`.
The final scene run used `CARGO_INCREMENTAL=0` and
`CARGO_PROFILE_TEST_DEBUG=0` to avoid the local cache hard-link failure and
debug-symbol generation; debug assertions and shader contract assertions
remained enabled. All build output and logs are local to this worktree.
Full logs are under `out/scene-picking/`.

Python bindings, the full workspace/CI matrix, Python viewer end-to-end tests,
image-quality checks and independent cross-backend renders were not run.
