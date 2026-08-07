# Swiss Land Cover Capsule — Design Specification

Date: 2026-08-07 (rev 3: relocated to the primary checkout; provenance filled from Milos's answers + in-repo evidence; render path switched to the Iberia PT template; signing deferred; single-vendor calibration)
Status: Draft for review
Scope: The first Verified Map Capsule (Swiss terrain + draped land-cover overlay), which becomes the authoritative template for the other launch capsules and for useforge3d.com. This spec covers the capsule only, not the full website.

Why this subject: terrain + a classified raster drape demonstrates a representative cartographic workflow (elevation, overlay, palette/classification, legend-ready classes) rather than a bare DEM. Mount Rainier remains a later capsule as the simplest onboarding case.

## 1. Context and settled decisions

useforge3d.com is being rebuilt from scratch as a conversion system for forge3d: existing audience/channels supply acquisition; the site converts visitors into successful local renders. Settled constraints (2026-08-07 design review cycle):

- No hosted rendering, GPU workers, queues, uploads, or remote render API. The site serves static content only.
- Local Python rendering is the official v1 path. Browser-side forge3d never gates launch.
- Visually persuasive for humans; operationally precise for agents.
- Verified Map Capsules are the content unit. Exact recipes outrank prompts; prompts are derived, versioned, matrix-tested artifacts.
- Navigation is task/capability-first; geography (Atlas) is a later, secondary mode.
- Publishing uses finite named seasons, not an indefinite weekly promise.
- The first onboarding capsule must use only the open core (Apache-2.0/MIT). Pro capsules are allowed later, explicitly marked.

## 2. What a capsule is

A capsule is a self-contained, versioned directory proving one claim:

> This map was rendered by forge3d from this recipe and this data; you can reproduce it on your own hardware and check what ran.

```text
examples/capsules/swiss-landcover/
├── README.md                  # human instructions (section 9)
├── recipe.py                  # the exact executable recipe (section 4)
├── environment.lock.md        # pinned forge3d version, Python range, platform notes
├── data-manifest.json         # datasets: name, sha256, byte size, fetch method (section 3)
├── provenance.json            # upstream source & license record, one entry per dataset (section 3)
├── recipe_result.json         # written by recipe.py: sha256 of final PNG + inputs + recipe + cert (rev 4)
├── verify_config.json         # pinned SSIM/mean-abs thresholds + calibration record
├── expected/
│   ├── reference.png          # CANONICAL lossless reference — verify.py compares against THIS (rev 4)
│   ├── preview.avif           # lossy web derivative (display only, never compared)
│   ├── preview.webp           # lossy web derivative (display only, never compared)
│   ├── thumb.webp             # 640px thumbnail derivative
│   └── reference.certificate.json   # reference RenderCertificate (section 6)
├── VERIFY.md                  # local verification walkthrough (section 7)
└── AGENT_TASK.md              # agent contract (section 10)
```

Schema follows existing repo conventions (`tests/golden/certificates/`, `RECIPE_GOLDENS` in `tests/test_recipe_goldens.py`, `tools/europe_smoke/provenance.py`) rather than inventing a parallel format.

## 3. Dataset provenance

The capsule uses two registry datasets (`python/forge3d/datasets.py`):

| Registry name | File | Pinned hash | Source (per Milos, 2026-08-07) |
|---|---|---|---|
| `swiss` (dem) | `switzerland_dem.tif` | `sha256:d09d229f…0bd478` | 30 m Terrarium elevation (AWS Terrain Tiles) |
| `swiss-land-cover` (overlay) | `switzerland_land_cover.tif` | `sha256:6b254585…39a8f7f` | Esri 10 m Sentinel-2 Land Use/Land Cover |

Both files were produced by Milos. `provenance.json` records, per dataset: provider, product, original URL, license basis, required attribution, acquisition date, transformation history, bbox/CRS/resolution.

Known values to encode:

- **Land cover**: Sentinel-2 10 m Land Use/Land Cover — Esri, Impact Observatory, and Microsoft. License CC BY 4.0; the credit line already codified at `examples/swiss_terrain_landcover_viewer.py:22` ("Data: Sentinel-2 10m Land Use/Land Cover – Esri, Impact Observatory, and Microsoft") is rendered verbatim in the capsule README and on the capsule web page.
- **Elevation**: Terrarium-encoded AWS Terrain Tiles at ~30 m. Attribution follows the Mapzen/AWS terrain-tiles composite notice (upstream sources for the Swiss extent include EU-DEM/SRTM); exact string finalized during capsule build from the creating script.
- **Remaining detail** (acquisition year/zoom, crop/resample steps): extracted during capsule build from the Python script that created the files (the repo's Terrarium fetch pattern appears in `tools/europe_smoke/basemap.py` and the land-cover viewer family). If the exact creating script can't be identified, re-derivation from these documented sources reproduces provenance by construction.

## 4. Recipe and certified render path

The recipe MUST use a certified offscreen render path. The existing Swiss example (`examples/swiss_terrain_landcover_viewer.py`) is interactive — `open_viewer_async` + overlay IPC — and interactive viewer output is explicitly outside the RenderCertificate contract (`python/forge3d/viewer.py:1204,1242`). It remains the site's interactive on-ramp, but it is not the verified artifact.

The render follows the **Iberia PT template** (`examples/iberia_landcover_pt/iberia_landcover_pt_v1.py`), per Milos's direction — the proven premium look for land-cover relief:

1. Fetch both datasets through the registry (hash-checked): `swiss` DEM and `swiss-land-cover` raster.
2. Build the classified overlay deterministically, reusing the palette + despeckle logic from `swiss_terrain_landcover_viewer.py` (`build_overlay`: classes → despeckle → RGBA palette PNG). Fixed palette, fixed parameters, no randomness.
3. Path-trace the terrain light field with PROMETHEUS using the Iberia template's approach: quasi-orthographic nadir camera (`CAMERA_FOV_Y = 8.0`), square camera-cell tiling sized for the Swiss extent, fixed `SUN_AZIMUTH`/`SUN_ELEVATION`/`SUN_INTENSITY` constants, feathered tile stitching, and the template's terrain-hit sanity check.
4. Modulate the light-free land-cover overlay by the traced light field (Iberia template's compositing), producing the final plate.
5. Render with `certificate=<path>` — the path-tracing entry points support the certificate contract (`python/forge3d/path_tracing.py:1015`) — writing `swiss_landcover.png` plus the certificate JSON.
6. Exit nonzero with a readable message if the native GPU path is unavailable, the tracer reports no terrain hits, or the run degrades (no silent fallback — a run that didn't exercise the real renderer must not look like success).

PT is the ONLY render route — there is no MapScene or other fallback, and the agent contract forbids reporting success on any alternative path (rev 4: the earlier fallback sentence contradicted §10 and is withdrawn). The authoritative PT register is VENDORED into `recipe.py` verbatim with per-block source citations: the untracked `swiss_landcover_pt_3d.py` / Iberia scripts are historical references only, not runtime dependencies, because they are gitignored and target APIs absent from the tracked tree — importing them would break clean-clone reproducibility.

Recipe properties: no CLI args required (constants at top of file), no network access beyond the initial dataset fetches, no Pro imports, deterministic parameters, and a header comment stating the pinned forge3d version it was verified against.

## 5. Open-core boundary enforcement

- The recipe must import nothing from Pro-gated modules (`map_plate`, SVG/PDF export, building import, Mapbox-style import). Terrain, raster overlays, and path tracing are open-core per the README's workflow split.
- Verification: the clean-machine reproduction (section 12) runs with no license key set (`forge3d.set_license_key` never called, no key env var). If the recipe completes, the open-core claim holds by construction.
- The capsule page lists tier explicitly: `Tier: Free (open core)`.

## 6. Reference certificate — production signing DEFERRED

Per Milos (2026-08-07): the production-signed certificate is not required for v1. Rationale accepted — for human visitors, reproduction + pinned hashes + image comparison carry the trust story; the cryptographic signature matters chiefly for agents fetching capsules programmatically and for capsule files re-shared outside the site.

v1 ships:

- `expected/reference.certificate.json` — the RenderCertificate emitted by the reference render, committed AS EMITTED (rev 4: no fields are added to the certificate JSON — the schema is not ours to extend; the development-trust label lives in VERIFY.md, verify.py output, and verify_config.json instead). The capsule page and VERIFY.md state plainly that v1 reference certificates are not production-signed.
- The full certificate payload is still valuable unsigned: engine version, per-module WGSL hashes, adapter identity, capabilities, timing/allocation ledgers, degradation list — everything a local run is compared against.

Deferred (post-v1, when agent-facing distribution justifies it): signing the reference certificate through the existing protected lane (`FORGE3D_CERT_SIGNING_KEY`, policy in `tests/golden/certificates/README.md`), verified offline via `python -m forge3d.certificate verify … --pubkey signing.pub`. Nothing in the capsule layout changes when signing is added — only the signature block and the label.

## 7. Local verification model (user-side)

`VERIFY.md` walks the user (or agent) through:

1. Run `recipe.py` locally → produces `swiss_landcover.png` + a local execution report (certificate JSON).
2. Compare local report vs reference certificate: engine version, WGSL module hashes, degradation list (empty expected), adapter/backend class.
3. Compare local image vs reference using the scoped method in section 8.
4. Labels stay honest: both reference and local certificates are development-signed in v1 (section 6); neither is presented as production trust.

No telemetry. Nothing leaves the user's machine (section 13).

## 8. Image comparison policy

Adopt the existing golden policy rather than inventing one: SSIM over RGB with a per-scene minimum plus a mean-absolute-difference cap, exactly as `tests/test_recipe_goldens.py` does (`ssim_min` per scene, e.g. 0.990, plus `mean_abs_max`; `tests/_ssim.py`). The capsule pins its own `ssim_min`/`mean_abs_max`.

Per Milos (2026-08-07): calibration is **single-vendor NVIDIA**. Thresholds are calibrated on NVIDIA hardware and documented as such ("Expected-output thresholds calibrated on NVIDIA; other vendors untested"). A PT-accumulated render adds sampling variance on top of vendor variance; the pilot fixes the sample count and measures run-to-run SSIM spread on the same machine before setting thresholds.

Explicitly NOT claimed: byte-identical output, or validated thresholds on non-NVIDIA hardware. If user reports later show other vendors falling outside thresholds, the thresholds are widened and documented — never silently.

## 9. Human instructions (README.md)

Fixed sequence, mirroring the future capsule page: **See it → understand it → reproduce it locally → verify the result.**

1. The finished render (preview derivative) and one paragraph: what this map is, what forge3d capabilities it demonstrates (path-traced terrain light field, classified raster drape, certificate-bearing offscreen rendering), tier (Free), hardware expectations (native GPU required; NVIDIA-calibrated; approximate VRAM; expected PT wall-clock range).
2. Reproduce: `pip install "forge3d==<pinned>"` → `python recipe.py` → expected wall-clock range and output description.
3. Verify: pointer to `VERIFY.md`.
4. Data: sources, licenses, checksums, and the verbatim attribution lines (from `provenance.json`), including the Esri/Impact Observatory/Microsoft credit.
5. Known limitations, failure modes by platform, and where to report issues.

## 10. Agent contract (AGENT_TASK.md)

Addressed to a coding agent operating on the user's machine:

- Goal statement (reproduce and verify this capsule), success criteria (recipe exit 0; SSIM/mean-abs within thresholds; certificate comparison passes), and explicit non-goals (do not modify recipe parameters; do not fetch alternative datasets; do not claim success on fallback paths).
- Environment preflight: Python version, `pip install "forge3d==<pinned>"`, GPU presence check (`f3d.has_gpu()`), disk space for both rasters, expected PT runtime.
- Ordered steps with the exact commands, expected outputs, and per-step failure diagnostics (what to check, what to report back to the human).
- Reporting format: a short structured summary (platform, adapter, forge3d version, SSIM score, certificate comparison results, wall-clock).
- A version stamp tying the task file to the capsule version and tested matrix.

The derived "paste this" prompt for the website is generated from this file and carries the tested-matrix disclaimer ("Tested with these named agents and versions; results may differ outside this matrix").

## 11. Web preview derivatives

Generated from the reference render at capsule build time, checked into the capsule:

- `preview.avif` + `preview.webp` hero: 1920px wide, target ≤ 350 KB (AVIF) / ≤ 500 KB (WebP).
- Thumbnail derivative: 640px, ≤ 60 KB.
- The multi-MB `docs/assets` PNGs (including `03-swiss-landcover.png`) are documentation assets and are never served on capsule pages.

## 12. Acceptance criteria (pilot)

The capsule ships only when all pass:

1. `provenance.json` complete for both datasets; attribution lines render in the README.
2. Clean-machine reproduction succeeds from a fresh venv on the primary platform (Windows + NVIDIA), no license key set, following README.md verbatim.
3. Certificate flow: reference certificate emitted and honestly labeled (section 6); local report generated; WGSL hashes match; degradations empty.
4. Image comparison passes thresholds calibrated on NVIDIA (run-to-run PT variance measured first; single-vendor scope documented).
5. Agent matrix: one terminal agent (Claude Code) and one IDE agent (Cursor) complete AGENT_TASK.md end-to-end, versions recorded. No public compatibility claims beyond the recorded matrix.
6. Derivatives meet the size budgets.

(External pilot tester: deferred per Milos, 2026-08-07 — revisit before public launch of the full site.)

## 13. Measurement policy

v1 is strictly private: no telemetry, no beacons, nothing phoned home from recipes or verifiers. Proxies measured site-side only: capsule downloads, copy-button events, page progression. A voluntary post-success "I reproduced this" action may be added later. Site copy never reports proxies as "successful renders."

Baseline metrics to start collecting at launch (targets set only after a baseline exists): clean-install success rate, median time to first local render, capsule completion rate (voluntary confirmations), failure categories by platform, agent-task completion by named host/version.

## 14. Interactive dashboards (resolved status)

Both dashboards are **single-file, self-contained HTML** — directly hostable on the static site with no server dependency:

- `europe_smoke_basemap.html` — generated by `examples/europe_smoke_dashboard/gen_interactive.py` ("Build the self-contained dashboard HTML", writes to its assets dir); regenerate on demand.
- `europe_wind_heat.html` — a built copy exists at `.worktrees/europe-wind-camera/output/playwright/wind-dashboard-final/europe_wind_heat.html`; canonical rebuild via the wind dashboard template pipeline.

They become live exhibits on the site (a later work item, after the capsule pilot): serve from the static CDN, sandboxed iframe if embedded in pages, payload budgets checked at publish time.

## 15. Out of scope / deferred

Hosted rendering of any kind; the full website IA (separate spec after this capsule survives its pilot); MCP server; browser-side forge3d; Atlas; Terra; production certificate signing (section 6); external pilot tester; multi-vendor GPU calibration; the Rainier capsule (later season slot).

## 16. Open items

| # | Item | Owner |
|---|---|---|
| 1 | Identify the exact creating script(s) for the two Swiss GeoTIFFs and extract acquisition details into `provenance.json` (fallback: re-derive from documented sources) | Implementation |
| 2 | Fix PT parameters for the Swiss extent (camera cells, sun constants, exposure, sample budget) from `iberia_landcover_pt_v1.py` + `swiss_terrain_landcover_viewer.py` | Implementation |
| 3 | Measure PT run-to-run SSIM variance on NVIDIA, then pin `ssim_min` / `mean_abs_max` | Implementation |
| 4 | Pin the forge3d version for `environment.lock.md` (current crate v1.30.0; confirm the PyPI wheel to pin) | Implementation |
| 5 | ~~Decide capsule home in the repo~~ Resolved 2026-08-07: `examples/capsules/` | — |

## 17. Rev 4 — specification-closure addendum (2026-08-07, external review reconciliation)

Resolutions of the seven closure items; where an item conflicts with earlier sections, this addendum governs.

1. **Authoritative template.** The PT register is vendored verbatim into `recipe.py` (per-block citations); the untracked `swiss_landcover_pt_3d.py` / `iberia_landcover_pt_v1.py` scripts are historical references, not dependencies. (These files exist on the Windows working machine as untracked files; they are absent from origin/main — hence vendoring, not importing.)
2. **PT only.** No MapScene or any other fallback route; §4 amended. Failure is a nonzero exit, never an alternative renderer.
3. **Canonical reference.** `expected/reference.png` (lossless) is the SSIM/mean-abs comparison target. AVIF/WebP previews and `thumb.webp` are display-only derivatives.
4. **Provenance claim language.** v1 provenance is "documented, with stated gaps" — never advertised as "verified" while any gap note exists. The recorded facts: DEM from Terrarium (AWS Terrain Tiles) at ~30 m, land cover from Esri/IO/Microsoft Sentinel-2 10 m LULC (CC BY 4.0), hashes pinned; unrecoverable details (tile ids, zoom, upstream download date) are stated as unknown in `notes`. Re-derivation from documented sources creates a NEW dataset version with new hashes and regenerated reference artifacts (a v2 path), not retroactive provenance for the existing bytes.
5. **Trust envelope.** The RenderCertificate covers the final PT tile pass (GPU light field) — it does NOT bind the composed PNG. The binding artifact is `recipe_result.json`, written by `recipe.py` on success: sha256 of the final PNG, of both fetched inputs, of `recipe.py` itself, the certificate file's sha256, and the forge3d version. `verify.py` recomputes and cross-checks all of these. This binding is by hash manifest, not signature; cryptographic signing of the manifest is deferred with production signing (§6). The certificate JSON is never mutated.
6. **Install contract.** Engine/package version is 1.34.0 (Windows x86-64 wheel on PyPI). The pinned install command is `pip install "forge3d==1.34.0" rasterio pillow scipy numpy` (exact tested versions recorded in environment.lock.md at Task 5; add any dataset-fetch dependency observed in the acceptance run).
7. **Executable verifier.** `python verify.py` is the single acceptance command: validates both JSON manifests, recomputes dataset + PNG + recipe hashes against `recipe_result.json`, compares `out/swiss_landcover.png` to `expected/reference.png` (RGB SSIM + mean-abs, pinned thresholds), checks mandatory certificate fields (engine version, WGSL module hashes, empty degradations) vs informational ones (adapter, timings — reported, never gating), writes machine-readable `out/verify_result.json`, and exits nonzero on any failure. Calibration facts (hardware, driver, backend, runs, threshold derivation) live in `verify_config.json`.

**NOT_PROVEN discipline:** the NVIDIA calibration and each agent-host run remain `NOT_PROVEN` until executed against the exact frozen capsule version; README's tested matrix records only executed combinations and marks the rest `untested`.
