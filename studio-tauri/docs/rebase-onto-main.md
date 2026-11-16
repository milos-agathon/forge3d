# Rebasing the Studio branch onto `main` (measured 2026-11-16)

`studio-tauri` has never been merged to `main`: **main has no `studio-tauri/`
tree at all** (`git ls-tree origin/main --name-only | grep -c studio-tauri` → 0).
The branch carries the whole Studio product surface — 269 files, 81,175
insertions — and `main` has moved 325+ commits since the branch point
(`687f5e0b`, the 1.49.0 residency release).

A pull request is the only way to get the `deb-audit` job onto a real
runner (the workflow's `push` trigger is `main` only). **But a pull request
whose base conflicts produces no CI evidence at all** — the workflow's
preflight merge never happens, so the job never runs. That is exactly what
`git merge-tree --write-tree origin/main studio-tauri` reports today.

## What actually conflicts

```
$ git merge-tree --write-tree origin/main studio-tauri
CONFLICT (content): Merge conflict in python/forge3d/map_scene.py
CONFLICT (content): Merge conflict in tests/UNRUN.toml
CONFLICT (content): Merge conflict in tests/data/world_coord_f32_ledger.json
```

Three files, six hunks. **The Studio tree itself merges cleanly** — every
`studio-tauri/` file auto-merges, including this session's work.

The map_scene.py hunks are not mechanical: both sides moved the same engine
code, and each side's version is *newer in its own direction*.

### Hunk 1 — `python/forge3d/map_scene.py` (~1488): the colour source

| Side | Shape |
|---|---|
| `main` | adds `projection_only` (skip the overlay work when only projecting) and collects `overlays` as a **list** |
| branch | widens the condition to `preset_name or settings.get("colormap")` (the Studio names a colormap in the recipe settings) and adds the `light_azimuth_deg` / `light_elevation_deg` override (the Studio sends the sun's angles explicitly) |

**Resolution:** keep main's `thematic` / `overlays` / `projection_only`
structure, widen the condition to `if preset_name or settings.get("colormap"):`,
and keep the branch's explicit sun-angle override — main's version has no
such override, and the Studio's still depends on it.

### Hunk 2 — `python/forge3d/map_scene.py` (~1538): the clip range

| Side | Shape |
|---|---|
| `main` | `clip = (0.1, max(6000.0, terrain_span * 1.5))`, overridden by the camera's near/far when a `statistical_height_domain` is set |
| branch | `clip_far` plus a `clip_override` taken from the recipe settings (`"clip": [near, far]` — what the Studio sends) |

**Resolution:** compose them — the branch's `clip_override` when the recipe
carries one, main's `statistical_domain` override on top of it.

### Hunk 3 — `python/forge3d/map_scene.py` (~1598): shadows and overlays

| Side | Shape |
|---|---|
| `main` | `shadows=_mapscene_shadow_settings(renderer_config.shadows)` — **one argument**; `overlays=overlays` (the computed list) |
| branch | `shadows=_mapscene_shadow_settings(renderer_config.shadows, raw=raw_shadows)` — main's function no longer takes `raw`; `overlays=(... _invalid_overlays(...))` — honours the recipe's explicit `overlays` key, where `[]` is the Studio's sentinel for "no built-in colour source" |

**Resolution:** main's one-argument shadow call (main replaced the `raw=`
plumbing) and an overlays expression that honours the recipe's explicit
`overlays` when it is present, falling back to main's computed list.

### Hunk 4 — `python/forge3d/map_scene.py` (~3553): the native label composite

| Side | Shape |
|---|---|
| `main` | `if not _needs_native_label_composite(recipe, plans):` — the newer guard |
| branch | the docstring plus `if not batches:` — no accepted label, no native scene |

**Resolution:** main's guard with the branch's early-out; both mean the same
thing, and the branch's comment explains why the legacy compositor must not
be reached for.

### `tests/UNRUN.toml`

Both sides appended entries; the file is a registry of tests deliberately
outside the fast lane. Keep both sides' entries.

### `tests/data/world_coord_f32_ledger.json`

**The ledger is an order-sensitive digest, not a mergeable file.** Its
documented rule: a branch's own transition must be undone *before* the chain
runs, and the branch's entries must be re-recorded against the new baseline
rather than left on the old one. So: take main's ledger, re-record the
branch's entries on top of it, and re-run the ledger gate
(`tests/test_world_coord_f32_gate.py`) — not a textual merge.

## After the rebase

1. `cargo fmt --check`, `cargo clippy --workspace --all-targets`,
   `cargo test --workspace` (engine), `npx tsc --noEmit`, `npm test`.
2. The fast lane (`python scripts/ci_pytest_lane.py --profile fast`) — 730
   tests, and it is what proves the engine's render path did not move.
3. The determinism gate (`tests/test_determinism_hash.py`) — the recipe
   goldens move with the engine, so the hashes are re-recorded on the
   engine's lane, not by hand.
4. `git merge-tree --write-tree origin/main <branch>` must come back clean
   **before** the pull request is opened.
5. Then open the pull request (`studio-tauri` → `main`). The `studio-tauri`
   workflow's `deb-audit` job builds the deb, `dpkg -i`s it, validates the
   desktop entry and runs `--engine-check` in both interpreter scenarios —
   the engine-discovery evidence that only a runner can produce.
