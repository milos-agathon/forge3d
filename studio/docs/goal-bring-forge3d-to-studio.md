# Goal: bring Forge3D's surface into Studio, honestly

## Why this exists

Studio has one render path and about twenty settings. Forge3D has hundreds.
Every "add a feature" attempt fails the same two ways: a control that cannot
change a picture ships, or a mechanism gets built twice (once in the engine,
once in the shell). Both look like progress; both cost a session to unwind.
This is the contract that prevents it.

## Non-negotiable rules

1. **The engine is the source of truth.** Before writing shell code, find the
   engine's own mechanism. If it exists — `render_scale` for supersampling,
   `thematic.apply_palette` for class rasters, `camera_render_view` for the
   camera — use it. A second implementation is a bug even when it works.
2. **No control ships without a measured pixel change.** For every setting,
   render once with and once without, and quote the changed-sample count. A
   setting that changes zero pixels is dead configuration: delete it, do not
   document it. (`pom` and `quality.samples` were both shipped dead this way.)
3. **Measure under the user's conditions**, not a convenient one. The sparkle
   bug survived three rounds because every test flew a wide orbit, where the
   DEM fits one cell per pixel; the user was in a close-up, where it does not.
   Reproduce at their camera, sun, frame size and device-pixel ratio.
4. **Every number in prose must be re-derivable.** The still anchor moved from
   `848cea2e` to `716555b1` the moment `render_scale` changed. Quote the
   command that produced it, and keep both values when one supersedes another.
5. **Every UI range must sit inside the engine's range.** Studio allowed a
   vertical scale of 0.05 where `TerrainRenderParams` raises below 0.1: a value
   the renderer refuses, offered as a slider.
6. **One increment per commit**, with its tests, and the tree green.

## The recipe for a setting (five places, three tests)

1. `crates/studio-core/src/project.rs` — the field on `RenderSettings` with
   `#[serde(default…)]`, a range check in `Project::validate`, the default.
2. `src/bridge/types.ts` — the mirror, with a JSDoc that says what it means.
3. `src-tauri/src/render.rs` — the request field, `with_settings`, and
   `to_payload`. This is the only place a preview may differ from a final, and
   it must say why.
4. `src-tauri/src/studio_render.py` — map it onto the engine's dataclass or
   function. If the engine has no consumer, stop and report it.
5. `src/components/Inspector.tsx` — one control, in the tab the thing belongs
   to: Render is the output, Atmosphere is weather, Lighting is sun and sky,
   Object is the selection.
6. Tests: one schema-contract line, one payload assertion, one service test.
   Then a GPU render proving the pixels moved.

## The audit to run before every release

- Every `RenderSettings` field reaches the payload; the service reads no key
  the payload does not send.
- Every payload flag is read by the service (this is how the two dead knobs
  were found).
- Every UI range is inside the engine's range.
- No client-side re-implementation of an engine mechanism.
- No comment that contradicts the code — that is how the next person misprices
  a frame.

## Where the surface stands

Built: project model, undo, schema v4; Solid/Material/Rendered; stills and
sequence export; timeline with camera-path recording; render queue; AOV and
OIDN denoise; sky and atmosphere; shadows; water (physical optics, blob
filter); fog (reach and tint); bloom and vignette; supersampled frames.

Added since: **colormaps** (the engine's own catalogue; needs
`albedo_mode="mix"`, and one colormap LUT per frame), **terrain sun shadows**
(`SunVisibilitySettings`; reach and steps), a **per-DEM vertical scale** (6%
relief; Bryce stays 1.0, the Swiss DEM 5.25), and a **despeckle** pass for
class rasters.

Measured and **blocked**, with the numbers in the roadmap: material layers
(`vs_main` writes a constant `world_normal`, so the layer slope is ~0), tonemap
operators and LUT, SSAO/SSGI/SSR, planar water reflections — all 0 changed
samples on this path. Clouds wait on the interactive path's stale shader
contract.

Added later: an **elevation legend** — the engine's own `Legend` (gradient bar,
ticks in metres) composed into the frame's corner with the colormap.

Not built: the **map plate**. It exists in the engine, but
`forge3d.map_plate.MapPlate` calls `_check_pro_access`, so it raises without a
Pro licence; and a *categorical* legend still needs a class scheme the project
does not store. Guessing one would be the dishonesty this page forbids.

## Definition of done

One setting or one behaviour; the five-place diff; the three tests; a measured
pixel delta under the user's conditions; the roadmap line; and the tree green —
`cargo test`, clippy at zero warnings, `cargo fmt --check`, `tsc`, `npm test`,
`pytest`. Nothing else counts as done.
