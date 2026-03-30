# forge3d Developer Platform — Design Specification

**Date:** 2026-03-10
**Status:** Approved
**Timeline:** 6 weeks (Approach C — Foundation → Experience → Monetization)

---

## Overview

Ship forge3d as a polished, pip-installable developer platform with dual-track documentation, interactive Jupyter widgets, a visual gallery, and an open-core licensing model. The goal: someone discovers forge3d on PyPI, installs it, follows a tutorial, and gets a beautiful 3D terrain render in their notebook in under 5 minutes.

### Scope

**In scope:**
- Pre-built abi3 wheels for 4 platform targets (Windows x86_64, Linux x86_64, Linux aarch64, macOS universal2)
- PyPI publication with automated CI/CD
- Dual-track tutorials (GIS professionals + Python developers)
- Interactive Jupyter ipywidgets (controls that re-render, not a live 3D canvas)
- Visual gallery with 10 showcase entries
- Open-core licensing (Apache-2.0 base + Pro tier with Ed25519 offline keys)

**Out of scope:**
- Hosted rendering API / serverless endpoint (separate project)
- Full embedded WebGPU viewer in Jupyter (requires WASM export, Priority 7)
- musllinux wheels (Alpine Docker — tiny audience overlap with GPU rendering)

### Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Approach | C: Foundation → Experience → Monetization | Sequential phases with clean exit points de-risk solo launch |
| Jupyter interactivity | ipywidgets with re-render (not live 3D canvas) | Achievable without WASM; ipywidgets is standard |
| License model | Open-core (Apache-2.0 + Pro) | Maximizes adoption while capturing commercial value |
| Target audience | Both GIS professionals and Python developers | Dual tutorial tracks, shared gallery |
| Python floor | 3.10+ | 3.8/3.9 EOL; aligns with scientific Python ecosystem |
| Platforms | Win x86_64, Linux x86_64, Linux aarch64, macOS universal2 | Covers desktop + cloud GPU (Graviton) |
| Timeline | 4–6 weeks, polished launch | First impressions on PyPI are sticky |

---

## Phase 1: Foundation (Weeks 1–2)

**Exit criterion:** `pip install forge3d` works from the 4 abi3 wheel targets. Package is on PyPI with correct metadata.

### 1.1 Python version floor bump

**Current:** `pyproject.toml` requires `>=3.8`, classifiers list 3.8–3.13, CI tests 3.9/3.11/3.12.

**Changes:**
- Set `requires-python = ">=3.10"` in `pyproject.toml`
- Remove 3.8/3.9 classifiers, ensure 3.10/3.11/3.12/3.13 are listed
- CI test matrix: 3.10, 3.11, 3.12, 3.13
- Audit Python source: remove `from __future__ import annotations` shims, replace `typing.Union`/`typing.Optional` with `X | Y` syntax throughout public modules — signals modern Python to developers browsing source

**Impact:** Reduces wheel matrix, enables modern type syntax in docs and source.
**Definition of done:** `pyproject.toml` updated, classifiers correct, CI matrix green on 3.10–3.13, no 3.8/3.9 compat code remains in public modules.
**Effort:** Small (half day).

### 1.2 Add Linux aarch64 wheel build

**Current:** CI builds wheels for 3 platforms (Windows x86_64, Linux x86_64, macOS universal2). No aarch64.

**Changes:**
- Add `aarch64-unknown-linux-gnu` target to the wheel build matrix
- Use `maturin-action` with QEMU cross-compilation or `--zig` linker for the C toolchain
- wgpu backend resolution: verify Vulkan backend compiles correctly for aarch64 in cross environment — wgpu build scripts may probe for system libraries not present during cross-compilation. Budget extra debugging time here.
- Smoke-test the aarch64 wheel in a QEMU-emulated container in CI

**Risk:** This is the highest-risk item in Phase 1. wgpu's GPU backend linking during cross-compilation may require workarounds (feature flags, build script patches, or a dedicated aarch64 runner). Tackling it early means problems surface in week 1, not week 5.

**Impact:** Enables cloud GPU batch workloads on AWS Graviton (cheaper than x86), ARM Linux Docker containers.
**Definition of done:** aarch64 wheel builds in CI, passes import smoke test in QEMU container, wheel is included in the PyPI publish matrix.
**Effort:** Medium (2–3 days, possibly more if wgpu cross-compilation issues arise).

### 1.3 Fix stale metadata

**Current:** `docs/conf.py` hardcodes version `1.9.1`. `pyproject.toml` has placeholder URLs (`https://github.com/example/forge3d`). Copyright says 2025.

**Changes:**
- `conf.py`: source version from `pyproject.toml` via regex parse (no Python import — avoids chicken-and-egg with maturin build requiring compiled extension)
- `pyproject.toml`: replace placeholder URLs with real GitHub/docs URLs
- Version single source of truth: `pyproject.toml` `version` field. Maturin reads it via `version = { from = "pyproject.toml" }` in Cargo.toml. `__init__.py` keeps `__version__` in sync (already does).
- Update copyright to 2025–2026
- Set classifier to `Development Status :: 5 - Production/Stable`

**Impact:** Eliminates version drift, makes PyPI listing professional.
**Definition of done:** `conf.py` dynamically reads version, all URLs resolve, copyright updated, classifier updated.
**Effort:** Small (half day).

### 1.4 PyPI publish workflow

**Current:** No publish step in CI. Wheels are built as artifacts but never uploaded to PyPI.

**Changes:**
- New workflow: `.github/workflows/publish.yml`
- Trigger: push of version tags (`v*`)
- Build: uses `maturin-action` for 4 abi3 wheels (one per platform, covering all Python 3.10+) + 1 sdist
- Publish: `twine upload` via PyPI OIDC trusted publisher (no API token stored in secrets)
- Dry-run: on PRs, build all wheels but skip upload (validates the matrix)
- Artifact retention: upload wheels as GitHub release assets alongside the PyPI publish

**Impact:** Automated, reproducible releases. Tag → wheels → PyPI in one step.
**Definition of done:** Tag push triggers workflow, all 4 abi3 wheels + sdist build successfully, publish to TestPyPI succeeds (real PyPI publish gated on Phase 3 launch).
**Effort:** Medium (1–2 days).

### 1.5 Install smoke tests

**Two tiers:**

**Tier 1 — Import/API surface (gates publish, runs on all 4 abi3 wheel targets):**
```python
import forge3d
assert forge3d.__version__
assert hasattr(forge3d, 'open_viewer_async')
assert hasattr(forge3d, 'open_viewer')
adapters = forge3d.enumerate_adapters()  # may be empty on CI, but shouldn't crash
```

**Tier 2 — Rendering integration (runs on CI runners with software GPU or self-hosted with GPU):**
- Install `mesa-vulkan-drivers` (lavapipe) on Linux CI for software Vulkan backend
- Render a 64×64 triangle, verify output is a valid RGBA numpy array
- This tier is `continue-on-error` — it doesn't block publish but reports failures

**Impact:** Catches broken wheels before they reach PyPI. No user's first experience is a failed import.
**Definition of done:** Tier 1 passes on all 4 wheel targets. Tier 2 passes on at least Linux x86_64 with lavapipe.
**Effort:** Small–Medium (1 day).

### 1.6 Pro boundary decision log

A living document: `docs/product/pro-boundary-notes.md`. During Phase 1 and 2 work, log observations about which features feel open vs. Pro as they come up. No code changes — just capture thinking for Phase 3.

**Impact:** Phase 3 starts with a decision log rather than a blank page.
**Definition of done:** File exists, has at least 5 entries by end of Phase 2.
**Effort:** Negligible (ongoing notes).

---

## Phase 2: Developer Experience (Weeks 2–4)

**Exit criterion:** Both tutorial tracks complete, Jupyter widget works in JupyterLab and Colab, gallery has 10 entries, API reference auto-generates.

### 2.1 Documentation site overhaul

**Current:** Sphinx with RTD theme, 121 doc files, feature-oriented structure, stale version.

**Changes:**

**Structure reorganization** — four top-level sections in `docs/index.rst`:
1. **Getting Started** — install instructions, "5-minute quickstart" (audience-neutral), architecture overview
2. **Tutorials** — dual-track (2.2)
3. **API Reference** — auto-generated via autodoc/autosummary
4. **Gallery** — visual showcase (2.4)

**Architecture page** (`docs/start/architecture.md`):
- One-page overview: Rust engine → PyO3 bindings → headless wgpu rendering → PNG/PDF output
- Explains why pre-built wheels mean no Rust toolchain needed
- Diagram of the rendering pipeline (text-based, Mermaid or ASCII)
- Preempts "why Rust?" and "do I need to compile?" questions

**Doc build promotion:** Change CI from `continue-on-error: true` to hard failure. Docs that don't build block the merge.

**Impact:** Transforms docs from a feature reference into a user-journey guide.
**Definition of done:** New structure live, architecture page written, doc build is a required CI check.
**Effort:** Medium (2–3 days).

### 2.2 Dual-track tutorials

Each tutorial is a `.md` file in `docs/tutorials/` with complete, copy-pasteable code blocks. Every code block runs standalone. Each tutorial ends with a rendered image showing expected output.

**GIS Professional Track** (knows rasterio/geopandas, new to 3D rendering):

| # | Title | Covers | Data Source |
|---|-------|--------|-------------|
| 1 | Visualize your first DEM | Load GeoTIFF with rasterio, launch viewer, orbit camera, take snapshot | Bundled mini DEM (~500KB) |
| 2 | Drape vector data on terrain | Load GeoJSON boundaries, Mapbox Style Spec styling, terrain overlay | Bundled GeoJSON (~100KB) |
| 3 | Build a map plate | Legend, scale bar, north arrow, PNG/PDF export | Same as tutorial 1+2 |
| 4 | 3D buildings from OSM | Building footprints, roof inference, PBR materials | Fetched CityJSON (~2MB) |

**Python Developer Track** (knows numpy/matplotlib, new to geospatial):

| # | Title | Covers | Data Source |
|---|-------|--------|-------------|
| 1 | Your first 3D terrain | Generate numpy heightmap (sine waves), render, explain what a DEM is | Generated in-line |
| 2 | Camera and lighting | Orbit camera, sun position, shadow techniques, MP4 export | Generated in-line |
| 3 | Point clouds and 3D Tiles | Load COPC file, explain point clouds, visualize | Fetched COPC (~3MB) |
| 4 | Scene bundles | Save/load reproducible scenes, share with colleagues | From tutorial 1 output |

**Pro boundary note:** GIS tutorial 3 (map plate) and GIS tutorial 4 (buildings) use features that may become Pro in Phase 3. Flag in `docs/product/pro-boundary-notes.md`. During Phase 2 these work without a key. In Phase 3, add callout boxes: "This tutorial uses Pro features. [Get a Pro key →]"

**Impact:** Two clear onramps for the two target audiences. Nobody hits a dead end.
**Definition of done:** All 8 tutorials written, code blocks tested, output images rendered and embedded.
**Effort:** Large (4–5 days across both tracks).

### 2.3 Sample datasets module

**New module:** `python/forge3d/datasets.py`

**Bundled in wheel (small, always available):**
- `mini_dem.tif` — 256×256 GeoTIFF DEM (~500KB) for GIS tutorial 1
- `sample_boundaries.geojson` — simple polygon boundaries (~100KB) for GIS tutorial 2

**Fetched on demand (larger, downloaded once and cached):**
- Uses `pooch` library for content-addressed downloads with SHA256 verification
- Cache location: `~/.forge3d/datasets/`
- `fetch_dem(name)` — downloads larger DEMs from GitHub releases
- `fetch_cityjson(name)` — downloads CityJSON building data
- `fetch_copc(name)` — downloads sample point cloud

**API:**
```python
from forge3d.datasets import mini_dem, sample_boundaries, fetch_dem
dem = mini_dem()           # → numpy array, always works (bundled)
dem_large = fetch_dem("rainier")  # → downloads on first call, cached after
```

**New optional extra:** `pip install forge3d[datasets]` installs `pooch>=1.6`. Bundled data works without it.

**Impact:** Tutorials run standalone without users sourcing their own data.
**Definition of done:** Bundled data loads without network. `fetch_*` functions download, cache, and return correct data. Total bundled data < 1MB.
**Effort:** Medium (1–2 days).

### 2.4 Jupyter ipywidgets integration

**New module:** `python/forge3d/widgets.py`

**Dependencies:** `ipywidgets>=8.0` as optional extra `pip install forge3d[jupyter]`

**Widget: `ViewerWidget`**
```python
from forge3d.widgets import ViewerWidget

w = ViewerWidget(terrain_path="dem.tif", width=1280, height=720)
w  # displays in notebook cell — launches viewer, shows snapshot inline
```

Methods mirror the `ViewerHandle` API:
- `w.set_camera(phi_deg, theta_deg, radius)` — orbit camera
- `w.set_sun(azimuth_deg, elevation_deg)` — sun position
- `w.send_ipc(cmd_dict)` — send any IPC command directly
- `w.snapshot(path, width, height)` — take high-res snapshot and display inline
- `w.close()` — shut down the viewer subprocess

The widget wraps the same viewer binary used in scripts. It launches the viewer subprocess, connects via IPC, and displays snapshots inline in the notebook output. For headless environments (Colab, CI), the viewer renders offscreen via wgpu.

**Type preservation:** The `@requires_pro` decorator (Phase 3) and any internal decorators use `functools.wraps` + `typing.ParamSpec` (3.10+) to preserve full type signatures for IDE autocompletion.

**Testing:** `tests/test_widgets.py` — test widget class existence and method presence without a notebook kernel (ipywidgets supports headless mode). No visual testing.

**Notebook examples** in `examples/notebooks/`:
1. `quickstart.ipynb` — install, import, launch viewer, take first snapshot
2. `terrain_explorer.ipynb` — ViewerWidget with programmatic camera/sun control
3. `map_plate.ipynb` — compose a map plate, export from notebook

**Impact:** The "wow moment" — interactive 3D terrain in a notebook cell.
**Definition of done:** `ViewerWidget` launches the viewer, takes snapshots, and displays them inline in JupyterLab. Works in headless mode (Colab/CI). 3 notebooks run end-to-end.
**Effort:** Medium (3–4 days).

### 2.5 Gallery

**Location:** `docs/gallery/`

Each entry is a `.md` file with: hero image (800px wide), 2–3 sentence description, complete Python script, tags.

| # | Entry | Tags | Pro? |
|---|-------|------|------|
| 1 | Mount Rainier — PBR terrain with shadows | `terrain`, `pbr`, `shadows` | No |
| 2 | Mount Fuji with place labels | `terrain`, `labels` | No |
| 3 | Swiss landcover with legend | `terrain`, `overlays`, `legend` | Yes (map plate) |
| 4 | Luxembourg rail network | `vector`, `styling` | No |
| 5 | Amsterdam 3D buildings | `buildings`, `cityjson`, `pbr` | Yes (buildings) |
| 6 | COPC point cloud visualization | `pointcloud`, `copc` | No |
| 7 | Camera flyover still frame | `animation`, `camera` | No |
| 8 | Vector SVG export | `export`, `svg` | Yes (vector export) |
| 9 | Shadow technique comparison | `shadows`, `comparison` | Partial (PCF open, rest Pro) |
| 10 | Composed map plate | `map-plate`, `legend`, `scale-bar` | Yes (map plate) |

Sphinx renders as an HTML gallery page with thumbnail grid. Each thumbnail links to detail page. Pro entries get a small badge on the thumbnail.

**Impact:** Visual proof of capability. People share gallery images, driving organic discovery.
**Definition of done:** 10 entries with images and working scripts. Gallery page renders in Sphinx with thumbnail grid.
**Effort:** Medium (2–3 days — most renders already exist, need scripts and writeup).

### 2.6 API reference population

**Scope:** Add Google-style docstrings to public functions/classes in the 15 most-used modules: `render`, `terrain_params`, `config`, `map_plate`, `legend`, `scale_bar`, `export`, `buildings`, `style`, `bundle`, `crs`, `colormaps`, `animation`, `viewer`, `widgets`.

Each docstring includes: one-line summary, params with types, return type, raises, one-line example.

Generate `.rst` stubs via `sphinx-apidoc` for the full `forge3d` package. Internal modules (`_native`, `_validate`, `_gpu`, `mem`) get a one-line description only.

**This is the pressure valve.** If time runs short, the auto-generated stubs from `sphinx-apidoc` still produce pages with function signatures — ugly but functional. Tutorials and gallery are higher priority. Don't let docstring perfectionism delay widget or gallery work.

**Impact:** Searchable, interlinked API reference.
**Definition of done:** Top 15 modules have docstrings, `sphinx-apidoc` generates navigable reference, builds without warnings.
**Effort:** Medium (2–3 days, cuttable to 1 day with minimal docstrings).

---

## Phase 3: Monetization & Launch (Weeks 4–6)

**Exit criterion:** Open/Pro boundary implemented, license key mechanism works, beta testers have validated, launch blog post published, package live on PyPI.

### 3.1 Open/Pro feature boundary

**Open (Apache-2.0) — the complete "hello world to useful result" journey:**
- Interactive viewer with terrain rendering, snapshots, and offscreen/headless mode
- Full colormap library (all providers — cmocean, cmcrameri, colorcet, palettable)
- Basic lighting and PCF shadows
- Vector overlays (load, style, drape on terrain)
- COPC/LAZ point cloud reading and rendering
- 3D Tiles loading and rendering
- CRS reprojection
- PNG export (single image, no artificial limits)
- Camera animation preview (no MP4 export)
- Jupyter widget (`ViewerWidget`)
- `forge3d.datasets` sample data
- All configuration classes and presets

**Pro (commercial license) — production output and premium rendering:**
- Map plate compositor (legends, scale bars, north arrow, multi-element layout)
- SVG/PDF vector export
- Scene bundles (.forge3d save/load)
- Advanced shadow techniques (PCSS, VSM, EVSM, MSM)
- 3D buildings pipeline (roof inference, PBR material auto-assignment, CityJSON import)
- `batch_render()` API — shared GPU context, parallel scheduling, progress callbacks (5–10x faster than naive looping of viewer snapshots)
- MP4 camera animation export
- Mapbox Style Spec import
- All post-processing effects (DoF, motion blur, volumetrics, lens effects)
- Priority support

**Design principle:** The open tier includes everything needed to explore, learn, and produce useful single renders. The Pro tier gates production output (map plates, PDF, video, batch performance) and premium rendering quality (advanced shadows, postfx, buildings). No petty restrictions — colormaps, presets, and the full API surface are open.

### 3.2 License key mechanism

**Architecture — offline, no server required:**
- Key set via `forge3d.set_license_key("F3D-XXXX-XXXX-XXXX")` or `FORGE3D_LICENSE_KEY` env var
- Key format: `F3D-{tier}-{expiry_yyyymmdd}-{signature}`
- Validation: Ed25519 signature verified locally. Public key embedded in package. No phone-home.
- State: cached in module-level singleton after first validation (no repeated crypto per call)

**Grace period on expiry:**
- Expired key: Pro features continue working for 14 days with a stderr warning on each call: "Your forge3d Pro license expired on {date}. Renew at forge3d.dev/pro. Pro features will stop working on {grace_date}."
- After grace period: raises `forge3d.LicenseError` with renewal URL

**`@requires_pro` decorator:**
- Applied to Pro-gated functions
- Uses `functools.wraps` + `typing.ParamSpec` to preserve full type signatures (IDE autocompletion works correctly in VS Code / PyCharm)
- Checks cached license state, raises `LicenseError` if invalid/missing

**Error experience:**
```python
forge3d.export_pdf(scene, "output.pdf")
# → forge3d.LicenseError: PDF export requires a Pro license.
#   Set your key: forge3d.set_license_key("F3D-...")
#   Get a key at: https://forge3d.dev/pro
```

**No obfuscation.** The decorator is a simple check, not DRM. The license is a social contract with honest commercial users, not a technical fortress.

**Key generation:** Standalone script (not shipped in package) that signs keys with private Ed25519 key. Lives in a private repo.

**Implementation:** New module `python/forge3d/_license.py` — key parsing, Ed25519 verification, tier/expiry extraction, grace period logic.

**Impact:** Monetization without hostile UX.
**Definition of done:** `set_license_key()` works, `@requires_pro` gates correct functions, expired keys get 14-day grace, `LicenseError` messages are clear and include URL.
**Effort:** Medium (2–3 days).

### 3.3 Docs Pro badges

- Tutorials using Pro features get a callout box at top: "This tutorial uses Pro features (map plate, PDF export). [Get a Pro key →](https://forge3d.dev/pro)"
- Gallery thumbnails for Pro entries show a small "Pro" badge
- API reference: Pro-gated functions show `[Pro]` marker
- Getting Started and first tutorial in each track are 100% open — zero Pro friction during onboarding

**Impact:** Transparent monetization, no surprises.
**Definition of done:** All Pro tutorials have callout, gallery badges render, API reference markers present.
**Effort:** Small (half day).

### 3.4 Package extras update

**Current extras:** `raster`, `matplotlib`, `tiles`, `cartopy`, `colormaps`, `all`

**Add:**
- `jupyter`: `ipywidgets>=8.0`
- `datasets`: `pooch>=1.6`

**Update `all`:** include `jupyter` and `datasets`

Pro features use the same native extension — gated by license key, not by package extra.

**Impact:** Clean install paths for different use cases.
**Definition of done:** `pip install forge3d[jupyter]` and `pip install forge3d[datasets]` work, `[all]` includes them.
**Effort:** Small (half day).

### 3.5 Launch prep

**Blog post** (written week 5, published at launch):
- Lead with the hero image — a stunning 3D terrain render that makes people stop scrolling
- Then the 5-line code that produced it
- Then "here's how this works" — architecture, open-core model, what you get
- Conclude with: install command, tutorial link, Pro info
- Published on docs site + cross-posted to dev.to, relevant subreddits (r/Python, r/gis, r/datascience)

**README overhaul:**
- Hero image, one-liner description, install command, 5-line example
- "Features" section with open vs Pro distinction
- "Gallery" thumbnails (4–6 links)
- Badges: PyPI version, Python versions, license, CI status
- Quick links: tutorials, API reference, Pro

**GitHub repo polish:**
- Replace placeholder URLs in `pyproject.toml` with real URLs
- `CONTRIBUTING.md` — development setup (Rust toolchain, maturin develop, running tests)
- Issue templates: bug report, feature request
- `SECURITY.md` — responsible disclosure process
- Repo description and GitHub topics set

**Impact:** Professional first impression on PyPI and GitHub.
**Definition of done:** README has hero image and working code example, blog post draft reviewed, repo metadata complete.
**Effort:** Medium (2–3 days).

### 3.6 Beta testing (week 5)

**Tester profiles (5–8 people):**
- 2–3 GIS professionals (rasterio/QGIS users, likely macOS/Linux)
- 2–3 Python developers (data science/visualization, likely macOS/Colab)
- 1 Windows user with a real GPU (historically the most wheel/driver issues)
- 1 existing forge3d user (regression check)

**Protocol:**
- Give them `pip install forge3d` (from TestPyPI) + a tutorial link, no other instructions
- Collect: did install work, how far in tutorial, what confused them, would they use again
- Fix critical issues. Cosmetic issues → backlog.

**Impact:** Catches real-world install/UX issues before public launch.
**Definition of done:** All beta testers successfully install and complete at least one tutorial. Critical issues fixed.
**Effort:** Medium (1 week elapsed, ~2 days of active fix work).

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| wgpu cross-compilation fails for aarch64 | Medium | High | Tackle in week 1. Fallback: drop aarch64 from launch, add post-launch |
| Jupyter widget doesn't work in Colab | Medium | Medium | Test in week 2. Fallback: Colab gets "Re-render" button UI |
| Docstring population takes longer than expected | High | Low | Pressure valve: auto-generated stubs are ugly but functional. Cut scope here first |
| Beta testers find critical wheel bugs | Medium | High | Budget 2 days fix time in week 5. Worst case: delay launch by 1 week |
| Pro boundary feels wrong after beta feedback | Low | Medium | Decision log captures thinking throughout. Boundary can be adjusted before launch |

---

## Success Metrics

| Metric | Target (launch + 30 days) |
|--------|--------------------------|
| PyPI installs | 500+ |
| GitHub stars | 100+ |
| Tutorial completion (analytics) | 50+ users reach "first render" |
| Pro key requests | 10+ |
| Critical bugs reported | < 5 |
| Wheel install failures | < 1% of attempts |
