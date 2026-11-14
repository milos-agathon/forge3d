# Phase 17 — the engine's inputs: plan (not code)

**Status:** plan for review; nothing is built. The roadmap names this phase;
its own triggers are unmet. This document exists so the trigger decision can
be made on facts, and so the work — if greenlit — lands as contained
increments, never a big-bang "fetch" button.

**Triggers (roadmap §8, verbatim):**
| Trigger | State today |
|---|---|
| Datasets/presets — "onboarding gap measured" | **Unmet** — no user, including you, has reported an onboarding gap; starter scenes exist and open offline |
| COG streaming — "a URL-based DEM is requested" | **Unmet** — no request seen |
| Bundles — "sharing/verification requested" | Unmet — unrelated to this phase |

**Decision this plan gates on:** do we build it *with triggers unmet* because
the user asked for it (this session asked for the plan, not the build), or
do we hold the phase until an onboarding/URL-DEM request actually arrives?
Recommendation: **build the two items whose risk is low and value is real
(starter scenes from the engine's dataset registry + dataset-resolved asset
pick), hold COG streaming and preset exposure behind their own triggers.**
Reasons below.

## 1. What the engine already offers (verified in this tree)

* **`forge3d.datasets`** — a registry of bundled + remote datasets:
  `bundled()` → `mini_dem`, `sample_boundaries`; `remote()` →
  `rainier`, `fuji`, `swiss`, `luxembourg`, `swiss-land-cover`,
  `luxembourg-rail`, `mount-fuji-places`, `sample-buildings`,
  `mount-fuji-buildings`, `mt-st-helens`. Fetches go through **pooch**,
  SHA-256-pinned against `known_hash`, cached under `~/.forge3d/datasets/`,
  fallback order: repo checkout → `FORGE3D_REPO_ROOT` →
  `FORGE3D_DATASETS_BASE_URL` (default: the repo's media endpoint).
  `fetch_dem`/`fetch_cityjson`/`fetch_copc` filter by kind.
  `pip install forge3d[datasets]` required; missing pooch raises a clear
  ImportError telling the user to install it.
* **`forge3d.open_cog`** — HTTP(S) COG access with a native range-streaming
  reader (feature `cog_streaming`, compiled in this build) that fetches only
  requested windows via HTTP range requests, plus a rasterio fallback when
  compression is LZW (prediction 5 cannot stream; the wrapper returns a
  `CogDatasetFallback` in that case). Requests are block-aligned 16 KiB.
* **`forge3d.presets`** — five renderer presets (`outdoor_sun`,
  `rainier_relief`, `rainier_showcase`, `studio_pbr`, `toon_viz`) returned as
  plain dicts (`lighting/shading/shadows/gi/atmosphere`). **These do not map
  1:1 onto the studio's request shape** — the studio speaks exposure,
  shadow-technique/resolution, IBL intensity, sky model, AOV flags, not a
  RendererConfig. A "preset" in the studio would be a **recipe** over the
  settings the studio already has, not a pass-through of the engine's dicts.
* **`forge3d.gis.cog_*` / `raster_window`** — windowed COG reads land in the
  same asset pipeline the GeoTIFF import uses once bytes are local.

## 2. What Phase 17 means for Studio (scope, narrowed to one increment per commit)

### 2.1 Datasets as "sample projects" in New Project (increment 1)
* New Project gains an entry *"From the engine's datasets…"* listing the
  remote registry (name, kind, description), **not** the repo assets.
* Picking one: the shell calls a tiny `datasets.py`-backed IPC
  (`dataset_info()`, `fetch_dem(name)`), gets a local path, then does exactly
  what `import_terrain_dem` does today (asset registry, `TerrainObject`,
  world span, `fitTerrainRelief`).
* The `FORGE3D_DATASETS_BASE_URL` override and the repo-checkout fallback
  make this work **off the network** (this repo's `assets/` already covers
  fuji/swiss/luxembourg — the fallback path resolves them locally, so the
  demo path is offline-safe).
* Failure modes, all supported and testable: no network (fallback to local),
  missing pooch (clear error naming `pip install forge3d[datasets]`),
  corrupt cache (pooch hash mismatch → re-download), unknown name (engine
  raises, shell surfaces it), disk full (pooch raises, shell surfaces it).
* **Measure:** render the new starter project headlessly against the
  byte-identical-baseline path (Phase 15 machinery) and count changed
  samples vs opening the same DEM from a file picker — expect zero.

### 2.2 Open-by-name for import/imagery/vector (increment 2)
* The import dialogs (DEM, imagery, vector, labels source) gain
  *"… or from Forge3D datasets"*, reusing 2.1's IPC. One code path for all
  asset kinds (DEM/overlay/vector/cityjson). Same fallback semantics; same
  failure modes.
* Building footprints (`sample-buildings`, `mount-fuji-buildings`) work as
  vector layers only after Phase C's ingest — they are **out of scope here**
  (no mesh path in the still); opening them is refused honestly, same as now.

### 2.3 Engine presets exposure (increment 3 — hold behind trigger)
* Only after 2.1/2.2 are proven, and only as **named Studio recipes**:
  `Outdoor sun`, `Studio PBR`, … Each maps to studio-native settings
  (exposure, light kind/azimuth, shadow technique/resolution, IBL intensity,
  sky model) via one mapping table with unit tests proving the mapping holds
  its invariants (engine ranges respected, per §9 of the schema policy).
* The engine dicts are **not** passed through — they cannot reach the still's
  service untranslated. No half-applied preset: applying a recipe sets every
  field atomically, one undoable `Batch` command.
* Explicitly out: `toon_viz` (no equivalent in the studio's light model),
  any part of the dict with no studio counterpart (documented).

### 2.4 COG streaming (increment 4 — hold behind trigger, larger)
* "Open URL…" for DEM/imagery: `open_cog(url)` server-side, window reads only
  for the preview mesh and the still's tiles; no full download.
* Hard requirements before this ships: offline fallback to full fetch when
  the file is not streamable (LZW/predict 5 → rasterio fallback, already
  built into `open_cog`), a cache budget, a UI affordance showing "streaming"
  state, and cancellation.
* The preview mesh then needs the COG resolution ladder (overviews) instead
  of the single-grid 96×96 sample — that is the real work; do not sneak it
  into 2.1/2.2.

### 2.5 Not in this phase, on purpose
* Point clouds/COPC UI (viewer tier, §8), buildings/meshes in Rendered
  (Phase C), bundles/sharing, terrain_params preset exposure beyond what the
  studio already wires (28 settings), mobile/browser fetching.

## 3. Staged increments (each shippable, none destructive)

1. `datasets_registry_ipc` — Rust: `dataset_list()` (pure read of
   `dataset_info()`), `dataset_fetch(name)` → path. No UI yet; tested against
   the offline fallback (repo checkout) so CI passes without network.
2. New Project → "From the engine's datasets". One dialog group; no new
   project types; `import_terrain_dem` path unchanged.
3. Import-dialogs reuse. Same IPC; three call sites.
4. (Gate) presets exposure, behind its own trigger.
5. (Gate) COG streaming, behind its own trigger + the resolution-ladder work.

## 4. What this does NOT change

* Stills stay MapScene; the recipe contract is untouched.
* No network call happens unless the user invokes the dataset path and the
  local repo checkout is absent.
* Every "fetch" is an explicit user action; the app never phones home on its
  own. pooch honours the hash; we surface failures, we do not retry silently.
* Menus/palette get one new entry group; nothing else moves.

## 5. Risks and how each fails

| Risk | Failure mode | Mitigation |
|---|---|---|
| Offline machine | Remote registry unusable | Repo-checkout fallback is first-class and is what CI exercises |
| Missing pooch | ImportError with install instructions | Surface the engine's message verbatim; never bundle pooch silently |
| Corrupt/truncated download | Hash mismatch | pooch raises; shell shows "download corrupt — retry" |
| Baked-in base URL drift | 404 from upstream media | `FORGE3D_DATASETS_BASE_URL` override documented; failing URL shown in the error |
| "Preset" means five different things to five people | Scope creep into 2.3→2.4 | 2.3 emits Studio-native recipes only; engine dicts never become project state |
| COG on 4 GB+ DEM | Streaming appears to hang | cache budget + progress + cancel, and full-fetch fallback is one click |

## 6. Verification plan

* Unit: `datasets` IPC output shape; name→kind filtering; the offline
  fallback path (assert the repo candidate is used); cache-hit path (fetch
  once, fetch twice, second returns instantly with hash verified).
* Shell: new-project-from-dataset creates a project whose terrain asset
  resolves; undo of the creation command removes it.
* Pixel: Phase 15 byte-identical baseline vs the same DEM opened from disk
  — counts must match exactly (same engine load path).
* Manual: New Project from each of the four terrain DEMs in the registry,
  offline and online; error paths (kill network; rename `assets/` to force a
  download).
* Time to budget: increments 1–3 are ~1–2 days each; 4 and 5 are real phases
  on their own and ship behind their own triggers.

## 7. What the user must decide

1. Build order: **2.1 → 2.2 only, trigger-flagged** (recommended), or
2. Full Phase 17 including presets and COG now, or
3. Hold Phase 17 entirely until the triggers are met.

The rest of this plan assumes (1); nothing in increments 1–3 forecloses 4/5.
