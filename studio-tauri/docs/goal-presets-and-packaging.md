# Goal: presets a stranger expects, and a deb a stranger can run

**Selection:** (A) renderer presets as tested Studio recipes; (B) stranger-installable deb with honest engine discovery. **Exit:** every control that shipped moved ≥1 pixel or says so; gates green; docs moved with the work.

## Verified ground truth

- Engine `forge3d.presets`: `outdoor_sun`, `rainier_relief`, `rainier_showcase`, `studio_pbr`, `toon_viz` → dicts keyed `lighting/shading/shadows/gi/atmosphere`.
- Studio maps the scene via `RenderRequest`/`build_request` (exposure, shadows/technique/resolution, iblIntensity, environment, skyModel/Turbidity/GroundAlbedo/Exposure, AOV/denoise flags).
- The deb bundles in CI (`studio-tauri.yml`, deb artifact on main; locally `target/release/bundle/deb/Forge3D Studio_0.1.0_amd64.deb`, 4.8 MB, /usr/bin, .desktop entry).
- `interpreter_candidates()`: env var → ancestors' `.venv/bin/python` → system `python3`. Fresh `python3` **has no `import forge3d`** (ModuleNotFoundError confirmed) → probe says unavailable; UI says it can't see the engine; no remedy offered today.
- `forge3d` is on PyPI (1.48.0, "Rust/WebGPU 3D maps").

## Track A — renderer presets as Studio recipes

Named recipes in the look workflow; no engine changes; no parallel renderer; one undo step per recipe (Batch of commands).

1. **Map what each preset means in Studio settings; document what Studio cannot express.** `studio_pbr` (exposure 1.0, directional key 6.0, PCF 2048) and `outdoor_sun` (lights + empty-GI default) → exposure + shadow technique/resolution. `rainier_relief/showcase`, `toon_viz`: read `forge3d/presets.py` line-by-line; exclude what has no honest mapping (e.g. `toon_viz` → no `lambert_physical` equivalent) with a tooltip saying so. Never a silent no-op.
2. **Mapping is tested, not trusted.** Per recipe: apply, assert resulting request payload (exposure, technique, IBL intensity, …). Pixel proof: byte/SSIM compare of the same scene before/after; changed-sample counts in the commit message.
3. **One Batch per recipe** (`SetRenderSettings` ± `SetTransform` for the sun) so undo reverts atomically and restore returns the exact pre-recipe frame.
4. **Look only** — recipes never touch camera, DEM, layers, selection. Tooltip per recipe names the source preset and known deviations.
5. **UI:** small "Presets" group in the Render section. Failure = one helpful tooltip, never a modal.

## Track B — the stranger-installable deb

1. **Discovery, ordered, verified:** `FORGE3D_STUDIO_PYTHON` (validate import, never trust blindly) → ancestors' `.venv/bin/python` (repo path, keep) → PATH `python3`/`python` with the same import check → **clear failure** naming the remedy (`pip install --user forge3d`, or set `FORGE3D_STUDIO_PYTHON`). No auto-installs, no sudo, no bare-python fallback.
2. **`--engine-check` CLI flag:** prints every candidate tried, each result, verdict, exit 0/1. Stranger- and CI-verifiable.
3. **postinst must not require the engine.** One line: "forge3d not found; `pip install --user forge3d` or set FORGE3D_STUDIO_PYTHON." No pip in the script.
4. **Docs:** README gains install — dpkg, `pip install forge3d`, first launch, reading `--engine-check`, troubleshooting. Verify `rpm`/`appimage` alongside deb or gate on deb with a note.
5. **Required verification (this machine):** (a) `dpkg -i` builds deb on a repo-less env; binary launches and probe **finds** a fresh `uv`-venv `forge3d`; (b) with bare PATH, `--engine-check` prints the intended remedy; (c) with `FORGE3D_STUDIO_PYTHON` set to the venv, `--engine-check` passes; (d) desktop entry valid (`desktop-file-validate`); (e) if CI runs it headless, say plainly the gate is the probe, not a GPU render; add a deb/audit CI job doing the same.
6. **No engine bundling; no forge3d fork; no `pip install` into the user's environment.**

## Hard rules (both tracks)

- MapScene remains the render path everywhere; recipes are inputs, never a parallel renderer.
- Every user-visible control analysed against pixels (diffs measured, counts in commit messages) or honestly marked "no measurable change" with a reason to ship it anyway.
- Gates: `cargo fmt/clippy`, `npm test`, `tsc`, studio-core + shell suites, fast-lane pytest where the touched area hints at it; engine untouched unless a failing-test-driven fix is unavoidable.
- Docs (`feature-coverage.md`, roadmap) land with each increment, not at the end; `.local/` files untracked and unquoted.
- One increment per commit; green checkpoints; no impossible promises ("installs everywhere" is measured per this doc, never asserted globally).

## Out of scope

New `forge3d` wheel; tile-laddered streaming preview; meshes in the still; anything in `.local/`.

## Definition of done

Presets ship as named recipes (UI + tests + measured deltas) and the deb passes `dpkg -i` on a repo-less machine, reports engine status honestly with the documented remedy. This document is the checklist; amend it in the same commits.

## Status (amended with the work)

**Track A — shipped in `ab14551f`, pushed to `fork/studio-tauri`.** All five
presets are named recipes under Render → "Look presets"; one undoable `Batch`
per recipe; tooltip names the engine preset and deviations; conformance tests
with measured deltas (outdoor_sun 159,556, rainier_showcase 206,585,
rainier_relief 66,613, studio_pbr 158,376, toon_viz 159,585 changed samples at
512², Fuji). The still honours `zScale` (render session keyed on
`(dem, zScale)`) so the relief recipes are real in the Rendered path, not
just the viewport. Deviation notes shipped in the recipes and in the roadmap
§5.33: toon BRDF → closest-honest PCF; engine-frame sun vectors not ported;
cascades/light-size/AO not ported.

**Track B — shipped with this amendment.** `--engine-check` flag on the
binary; labeled, import-validated candidates (env → repo `.venv` → PATH
python3); postinst names the remedy and nothing else; deb `Depends` merged
(libgbm1, mesa-vulkan-drivers) with a strict dpkg audit in CI; desktop entry
validates; README gains the stranger install path, the real `--engine-check`
output, a troubleshooting table, and the rpm/appimage caveat (built,
unaudited). CI `deb-audit` job: build deb → `dpkg -i` → validate desktop
entry → `--engine-check` in both scenarios. **The gate is the probe, not a
GPU render** — the hosted runner has no terrain-safe GPU, so GPU-side visual
verification on a fresh stranger machine remains unverified by design.

**Honest residuals:**

- `dpkg -i` itself was not run on this machine (no sudo password); the control
  checks were done by extracting the deb with `ar` and reading its members.
  CI runs the real `dpkg -i`.
- On this machine, a bare `PATH python3` resolved `python3` successfully but
  the probe reports the truth about a broken interpreter — e.g. a system
  `python3` with `forge3d` but no `numpy` fails with
  `import forge3d failed: No module named 'numpy'`. That is the probe
  working as designed, and it is in the troubleshooting table.
- `rpm` and `appimage` remain unaudited (documented in the README; not
  promised).
