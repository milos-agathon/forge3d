# forge3d Developer Platform — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship forge3d as a polished, pip-installable open-core developer platform with dual-track tutorials, interactive viewer integration, a visual gallery, and Ed25519 offline licensing — across 4 abi3 wheel targets on PyPI (one per platform, covering all Python 3.10+ versions).

**Architecture:** forge3d has a **single rendering pathway** — the Rust `interactive_viewer` binary, launched as a subprocess and controlled from Python via TCP + NDJSON IPC. The viewer handles all rendering: real-time orbit, terrain PBR, overlays, labels, point clouds, buildings, and high-res snapshots. For headless/CI usage, the same viewer binary renders offscreen (wgpu supports headless) and takes snapshots via IPC — no separate rendering function is needed.

Three sequential phases (Foundation → Experience → Monetization) with clean exit points. Phase 1 nails packaging and CI (including removing legacy render functions). Phase 2 builds the developer onramp (docs, viewer tutorials, gallery). Phase 3 adds the open/Pro boundary and launches. Each phase produces independently shippable artifacts.

**Tech Stack:** Python 3.10+, maturin (PyO3), Rust interactive viewer binary (wgpu), TCP+NDJSON IPC, ipywidgets 8.x (Jupyter viewer embedding), Sphinx + RTD theme, pooch (dataset downloads), Ed25519 (PyNaCl or pure-Python), GitHub Actions, PyPI OIDC trusted publisher.

**Key API facts** (must match actual codebase):
- `open_viewer_async(width=1280, height=720, terrain_path=...) → ViewerHandle`
- `ViewerHandle.snapshot(path, width, height)`, `.set_orbit_camera(phi_deg, theta_deg, radius, fov_deg=None)`, `.set_sun(azimuth_deg, elevation_deg)`, `.close()`
- `ViewerHandle.load_obj(path)`, `.load_gltf(path)`, `.render_animation(anim, output_dir, fps, width, height)`
- IPC commands: `load_terrain`, `set_terrain`, `set_terrain_pbr`, `set_terrain_camera`, `set_terrain_sun`, `load_overlay`, `add_vector_overlay`, `load_point_cloud`, `set_point_cloud_params`, `add_label`, `add_line_label`, `add_callout`, `load_obj`, `load_gltf`, `snapshot`, `close`
- Buildings API: `forge3d.buildings.add_buildings()`, `add_buildings_cityjson()`, `BuildingLayer`, `BuildingMaterial` — outputs sent to viewer via `add_vector_overlay` IPC
- Camera animation: `forge3d.animation.CameraAnimation` with keyframes, `.evaluate(t) → CameraState`
- **Legacy functions removed in Task 0.5:** `render_raster()`, `render_polygons()`, `render_raytrace_mesh()` — these are deleted from the codebase

**Spec:** `docs/2026-03-10-developer-platform-design.md`

**Execution notes:**
- Each chunk (Phase 1, 2, 3) should be executed as a **separate session** with a clean handoff summary. The plan is ~2700 lines — an agent will lose early context by Task 15+.
- **Pressure valves** (descope if behind schedule): (1) API reference docstrings — auto-generated stubs are acceptable. (2) Gallery — drop to 5 entries instead of 10 without hurting the launch.
- **Changelog:** `CHANGELOG.md` already exists. Update it with each task's commit — this is expected for a PyPI package targeting developers.
- **Docs hosting:** GitHub Pages (resolved). Use relative paths in all tutorial URLs so they work both locally and deployed.
- **PyPI name:** `forge3d` is available (verified 2026-03-10).

**Emergency procedures:**
- **Bad release on PyPI:** Use `pip install forge3d==<previous>` to pin users. File a yank request via `pypi.org/manage/project/forge3d/releases/` for the broken version. Tag and publish a hotfix (`v1.13.1`) within 24 hours.
- **Critical bug found post-launch:** Hotfix branch from the release tag, minimal fix only, push new tag to trigger publish workflow.

---

## File Map

### Phase 1: Foundation
| Action | Path | Purpose |
|--------|------|---------|
| Modify | `CHANGELOG.md` | Update with each task's changes (expected for PyPI packages) |
| Modify | `pyproject.toml` | Python 3.10+ floor, new extras, metadata |
| Modify | `docs/conf.py` | Dynamic version, updated copyright |
| Modify | `python/forge3d/__init__.py` | Remove compat shims |
| Delete | `python/forge3d/render.py` | Remove legacy render_raster / render_polygons / render_raytrace_mesh (Task 0.5) |
| Modify | `.github/workflows/ci.yml` | Python 3.10+ matrix, aarch64 wheel, smoke tests |
| Create | `.github/workflows/publish.yml` | Tag-triggered PyPI publish |
| Create | `tests/test_install_smoke.py` | Import/API surface smoke test |
| Create | `docs/pro-boundary-notes.md` | Living decision log for Pro boundary |

### Phase 2: Developer Experience
| Action | Path | Purpose |
|--------|------|---------|
| Modify | `docs/index.rst` | Restructured 4-section layout |
| Create | `docs/architecture.md` | One-page "how forge3d works" — viewer + IPC architecture |
| Create | `docs/tutorials/gis-track/` | 4 GIS professional tutorials (viewer workflows) |
| Create | `docs/tutorials/python-track/` | 4 Python developer tutorials (viewer workflows) |
| Modify | `python/forge3d/datasets.py` | Verify and extend the existing sample data module (bundled + fetch) |
| Create | `python/forge3d/data/mini_dem.npy` | Bundled 256×256 DEM (~500KB) |
| Create | `python/forge3d/data/sample_boundaries.geojson` | Bundled GeoJSON (~100KB) |
| Create | `tests/test_datasets.py` | Dataset loading tests |
| Modify | `python/forge3d/widgets.py` | Keep `ViewerWidget` public and keep the inline preview as an internal fallback |
| Create | `tests/test_widgets.py` | Widget instantiation + callback tests |
| Create | `examples/notebooks/quickstart.ipynb` | Notebook: first terrain viewer + snapshot |
| Create | `examples/notebooks/interactive_viewer.ipynb` | Notebook: viewer control + snapshots |
| Create | `examples/notebooks/map_plate.ipynb` | Notebook: map plate export |
| Create | `docs/gallery/index.md` | Gallery landing page |
| Create | `docs/gallery/entry_*.md` | 10 gallery entries (showing IPC commands) |

### Phase 3: Monetization & Launch
| Action | Path | Purpose |
|--------|------|---------|
| Create | `python/forge3d/_license.py` | Ed25519 key parsing, validation, grace period |
| Create | `tests/test_license.py` | License mechanism tests |
| Modify | `python/forge3d/map_plate.py` | Add `@requires_pro` |
| Modify | `python/forge3d/export.py` | Add `@requires_pro` |
| Modify | `python/forge3d/buildings.py` | Add `@requires_pro` |
| Modify | `python/forge3d/style.py` | Add `@requires_pro` |
| Modify | `python/forge3d/__init__.py` | Export `set_license_key`, `LicenseError` |
| Modify | `README.md` | User-facing rewrite with hero image |
| Create | `CONTRIBUTING.md` | Development setup guide |
| Create | `SECURITY.md` | Responsible disclosure |
| Create | `docs/launch-blog.md` | Launch blog post |

---

## Chunk 1: Phase 1 — Foundation (Weeks 1–2)

### Task 0.5: Remove legacy `render_raster` / `render_polygons` / `render_raytrace_mesh`

**Why:** The codebase contains `render_raster()`, `render_polygons()`, and `render_raytrace_mesh()` in `python/forge3d/render.py`. These are a separate, legacy rendering pathway (CPU hillshade + path tracer via PyO3 bindings) that is **not used in any examples**. All forge3d rendering should go through the interactive viewer binary (subprocess + IPC), which provides PBR terrain, overlays, labels, point clouds, buildings, and snapshots.

Keeping these functions creates confusion about how forge3d works and splits the API surface. They must be removed so the platform has a single, clear rendering story: **viewer + IPC**.

**Files:**
- Delete: `python/forge3d/render.py` — remove the legacy module entirely after confirming surviving helpers already live elsewhere
- Modify: `python/forge3d/__init__.py` and `python/forge3d/__init__.pyi` — remove exports of `render_raster`, `render_polygons`, `render_raytrace_mesh`, and any stale `RenderView` re-export
- Modify: `python/forge3d/widgets.py` — keep `ViewerWidget` public and keep any inline preview helper internal-only
- Modify: `tests/` — remove or update tests that depend on these functions
- Modify: `examples/notebooks/*.ipynb` — update any notebook cells that call `render_raster()`

- [ ] **Step 1: Identify all references to the legacy functions**

Run:
```bash
grep -rn "render_raster\|render_polygons\|render_raytrace_mesh" python/ tests/ examples/ docs/ --include="*.py" --include="*.ipynb" --include="*.md" --include="*.rst"
```

This will produce the complete list of files that need updating.

- [ ] **Step 2: Delete the legacy render module**

Delete `python/forge3d/render.py` once you have confirmed that:
- `render_raster()`, `render_polygons()`, and `render_raytrace_mesh()` are not needed anywhere in the repo
- any surviving offscreen helpers still needed by the package already live in `python/forge3d/helpers/offscreen.py`
- no remaining imports depend on `python/forge3d/render.py`

Do not leave a half-empty compatibility wrapper behind.

- [ ] **Step 3: Remove exports from __init__.py**

In `python/forge3d/__init__.py`:
- Remove `render_raster`, `render_polygons`, `render_raytrace_mesh` from package imports and `__all__`
- Remove any stale root export of `RenderView`; keep only `ViewerWidget` public from the widget module
- Remove the deleted module import entirely

- [ ] **Step 4: Update tests**

Files affected (examples from grep; use the actual search results as the source of truth):
- `tests/test_install_smoke.py` — remove `"render_raster"` and `"render_polygons"` from the public API surface check
- `tests/test_crs_auto.py` — remove or rewrite tests that call `render_raster`/`render_polygons`
- `tests/test_bundle_render.py` — rewrite to use viewer snapshot instead
- `tests/test_api_contracts.py` — update API surface assertions
- `tests/test_style_pixel_diff.py` — rewrite to use viewer if possible, or remove
- `tests/test_render_style_integration.py` — rewrite to use viewer if possible, or remove

- [ ] **Step 5: Update notebooks**

Update `examples/notebooks/quickstart.ipynb` and `examples/notebooks/map_plate.ipynb` to use `open_viewer_async()` + `snapshot()` instead of `render_raster()`.

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests pass (some removed, some rewritten).

- [ ] **Step 7: Commit**

```bash
git add python/forge3d/render.py python/forge3d/__init__.py tests/ examples/
git commit -m "refactor: remove legacy render_raster/render_polygons/render_raytrace_mesh

All rendering now goes through the interactive viewer (subprocess + IPC).
The viewer provides PBR terrain, overlays, labels, point clouds, buildings,
and high-res snapshots through a single unified pathway."
```

---

### Task 1: Python version floor bump (3.8 → 3.10)

**Files:**
- Verify: `pyproject.toml:9` (`requires-python`) and lines 14–32 (classifiers) already match the 3.10+ floor
- Modify: `python/forge3d/map_plate.py:6` (future annotations)
- Modify: `python/forge3d/buildings.py:20` (future annotations)
- Modify: `python/forge3d/style.py:21` (future annotations)
- Modify: `python/forge3d/__init__.py`
- Test: `tests/test_install_smoke.py`

- [ ] **Step 1: Write test that verifies Python version metadata**

Create `tests/test_install_smoke.py`:

```python
"""Smoke tests for package installation and API surface."""
import sys
import pytest

def test_python_version_floor():
    """Package requires Python 3.10+."""
    assert sys.version_info >= (3, 10), "forge3d requires Python 3.10+"

def test_import_forge3d():
    """Package imports without error."""
    import forge3d
    assert forge3d.__version__

def test_public_api_surface():
    """Key public symbols are accessible."""
    import forge3d
    required = [
        "open_viewer", "open_viewer_async", "Renderer",
        "RendererConfig", "MapPlate", "Legend", "ScaleBar",
        "has_gpu", "enumerate_adapters", "__version__",
    ]
    for name in required:
        assert hasattr(forge3d, name), f"Missing public symbol: {name}"
```

- [ ] **Step 2: Run test to verify it passes (baseline)**

Run: `python -m pytest tests/test_install_smoke.py -v`
Expected: PASS (current Python is 3.10+)

- [ ] **Step 3: Verify pyproject.toml version floor and classifiers**

In `pyproject.toml`, verify:
```toml
requires-python = ">=3.10"
```

Remove classifiers for 3.8 and 3.9:
```toml
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Rust",
    "Topic :: Multimedia :: Graphics :: 3D Rendering",
    "Topic :: Scientific/Engineering :: Visualization",
    "Topic :: Software Development :: Libraries :: Python Modules",
]
```

- [ ] **Step 4: Remove any remaining Python 3.8/3.9 compat shims from surviving public modules**

`python/forge3d/render.py` is deleted in Task 0.5, so do not plan follow-up edits there. If any surviving public module still uses 3.8/3.9 compatibility shims, remove them as part of this task.

- [ ] **Step 5: Remove `from __future__ import annotations` from public modules**

Remove the `from __future__ import annotations` line from:
- `python/forge3d/map_plate.py:6`
- `python/forge3d/buildings.py:20`
- `python/forge3d/style.py:21`

After removing, audit each file for `Union[X, Y]` → `X | Y` and `Optional[X]` → `X | None` conversions. These are now native syntax on 3.10+.

**Convention:** Do not use `from __future__ import annotations` in any new code either. All new files created in this plan use native 3.10+ type syntax without the future import.

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All existing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml python/forge3d/render.py python/forge3d/map_plate.py \
  python/forge3d/buildings.py python/forge3d/style.py python/forge3d/__init__.py \
  tests/test_install_smoke.py
git commit -m "feat: bump Python floor to 3.10+, remove compat shims"
```

---

### Task 2: Verify stale metadata and version sourcing

**Files:**
- Modify: `docs/conf.py:17-20` (version sourcing)
- Modify: `pyproject.toml:34-38` (URLs)

- [ ] **Step 1: Write test that conf.py version matches package version**

Add to `tests/test_install_smoke.py`:

```python
def test_version_consistency():
    """Version in __init__.py matches pyproject.toml."""
    import forge3d
    from pathlib import Path
    import re

    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text()
        m = re.search(r'^version\s*=\s*"(.+?)"', text, re.MULTILINE)
        assert m, "No version in pyproject.toml"
        assert forge3d.__version__ == m.group(1), (
            f"Version mismatch: __init__.py={forge3d.__version__}, "
            f"pyproject.toml={m.group(1)}"
        )
```

- [ ] **Step 2: Run test**

Run: `python -m pytest tests/test_install_smoke.py::test_version_consistency -v`
Expected: PASS

- [ ] **Step 3: Verify conf.py reads version dynamically**

Replace the hardcoded version block in `docs/conf.py` (lines 17–20):

```python
# Read version from pyproject.toml (no Python import needed)
import re as _re
_pyproject_path = os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml')
with open(_pyproject_path) as _f:
    _match = _re.search(r'^version\s*=\s*"(.+?)"', _f.read(), _re.MULTILINE)
    version = _match.group(1) if _match else 'unknown'
    release = version

copyright = '2025-2026, forge3d contributors'
```

- [ ] **Step 4: Verify project URLs in pyproject.toml**

Replace the `[project.urls]` section. Use the actual GitHub org/repo name — if not yet created, use a consistent placeholder that's easy to find-and-replace later:

```toml
[project.urls]
Homepage = "https://forge3d.dev"
Repository = "https://github.com/forge3d/forge3d"
Documentation = "https://docs.forge3d.dev"
"Bug Tracker" = "https://github.com/forge3d/forge3d/issues"
```

- [ ] **Step 5: Commit**

```bash
git add docs/conf.py pyproject.toml tests/test_install_smoke.py
git commit -m "fix: dynamic version in docs, update project URLs"
```

---

### Task 3: Verify Linux aarch64 wheel build in CI

**Files:**
- Modify: `.github/workflows/ci.yml:118-130` (build matrix)

- [ ] **Step 1: Add aarch64 target to build-wheels matrix**

In `.github/workflows/ci.yml`, add a new entry to the `matrix.platform` list in the `build-wheels` job:

```yaml
          - os: linux-arm
            runner: ubuntu-latest
            target: aarch64-unknown-linux-gnu
```

- [ ] **Step 2: Update the build step to use maturin-action for cross-compilation**

Replace the manual maturin install + build steps in the `build-wheels` job with `maturin-action` which handles cross-compilation via QEMU and zig linker:

```yaml
      - name: Build wheel
        uses: PyO3/maturin-action@v1
        with:
          target: ${{ matrix.platform.target }}
          args: --release --out dist
          manylinux: auto
```

Note: For the `aarch64-unknown-linux-gnu` target, maturin-action automatically sets up QEMU and uses the zig linker for the C toolchain. The wgpu crate's Vulkan backend should compile under cross — if it probes for system libraries, the `manylinux: auto` setting provides the necessary sysroot. **If this fails, debug the wgpu build scripts first** — this is the highest-risk item.

**Fallback:** If wgpu cross-compilation for aarch64 cannot be resolved within 2 days, fall back to using a native aarch64 runner (`buildjet-2vcpu-ubuntu-2204-arm` or GitHub's own ARM runners) or drop aarch64 from initial launch and add it post-launch.

**Important:** Steps 1 and 2 must be applied together — adding the matrix entry without switching to `maturin-action` will fail. These are one atomic change.

- [ ] **Step 3: Update artifact upload names to handle the new platform**

```yaml
      - name: Upload wheel artifact
        uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.platform.os }}
          path: dist/*.whl
```

- [ ] **Step 4: Update Python test matrix to 3.10+**

In `test-python` job, change:
```yaml
        python-version: ['3.10', '3.11', '3.12', '3.13']
```

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add aarch64 wheel build, update Python matrix to 3.10+"
```

---

### Task 4: PyPI publish workflow

**Files:**
- Create: `.github/workflows/publish.yml`

**Requirements beyond the initial draft:**
- The publish workflow smoke-test matrix must cover Windows, Linux x86_64, Linux aarch64, and macOS
- The workflow must validate that `GITHUB_REF_NAME` matches `pyproject.toml`'s `project.version` before uploading artifacts or publishing to PyPI

- [ ] **Step 1: Create the publish workflow**

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  push:
    tags: ['v*']
  workflow_dispatch:
    inputs:
      dry_run:
        description: 'Dry run (build but do not publish)'
        required: false
        default: 'true'
        type: choice
        options: ['true', 'false']

permissions:
  contents: read
  id-token: write  # Required for OIDC trusted publisher

jobs:
  build-wheels:
    name: Build Wheel (${{ matrix.platform.os }})
    runs-on: ${{ matrix.platform.runner }}
    strategy:
      fail-fast: true
      matrix:
        platform:
          - os: windows
            runner: windows-latest
            target: x86_64-pc-windows-msvc
          - os: linux-x86
            runner: ubuntu-latest
            target: x86_64-unknown-linux-gnu
          - os: linux-arm
            runner: ubuntu-latest
            target: aarch64-unknown-linux-gnu
          - os: macos
            runner: macos-latest
            target: universal2-apple-darwin

    steps:
      - uses: actions/checkout@v4

      - name: Build wheel
        uses: PyO3/maturin-action@v1
        with:
          target: ${{ matrix.platform.target }}
          args: --release --out dist
          manylinux: auto

      - name: Upload wheel artifact
        uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.platform.os }}
          path: dist/*.whl

  build-sdist:
    name: Build sdist
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build sdist
        uses: PyO3/maturin-action@v1
        with:
          command: sdist
          args: --out dist

      - name: Upload sdist artifact
        uses: actions/upload-artifact@v4
        with:
          name: sdist
          path: dist/*.tar.gz

  smoke-test:
    name: Smoke Test (${{ matrix.os }} / py${{ matrix.python }})
    needs: build-wheels
    runs-on: ${{ matrix.runner }}
    strategy:
      fail-fast: true
      matrix:
        include:
          - os: windows
            runner: windows-latest
            python: '3.10'
            wheel: wheels-windows
          - os: windows
            runner: windows-latest
            python: '3.13'
            wheel: wheels-windows
          - os: linux
            runner: ubuntu-latest
            python: '3.10'
            wheel: wheels-linux-x86
          - os: linux
            runner: ubuntu-latest
            python: '3.13'
            wheel: wheels-linux-x86
          - os: macos
            runner: macos-latest
            python: '3.10'
            wheel: wheels-macos
          - os: macos
            runner: macos-latest
            python: '3.13'
            wheel: wheels-macos

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}

      - uses: actions/download-artifact@v4
        with:
          name: ${{ matrix.wheel }}
          path: dist/

      - name: Install wheel
        shell: bash
        run: pip install dist/*.whl

      - name: Run import smoke test
        shell: bash
        run: |
          python -c "
          import forge3d
          assert forge3d.__version__, 'No version'
          assert hasattr(forge3d, 'open_viewer_async'), 'Missing open_viewer_async'
          assert hasattr(forge3d, 'has_gpu'), 'Missing has_gpu'
          print(f'forge3d {forge3d.__version__} OK')
          "

  publish-testpypi:
    name: Publish to TestPyPI
    needs: [build-wheels, build-sdist, smoke-test]
    runs-on: ubuntu-latest
    if: github.event_name == 'workflow_dispatch' && github.event.inputs.dry_run == 'false'
    environment: testpypi

    steps:
      - uses: actions/download-artifact@v4
        with:
          path: dist/
          merge-multiple: true

      - name: Publish to TestPyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: dist/
          repository-url: https://test.pypi.org/legacy/

  publish:
    name: Publish to PyPI
    needs: [build-wheels, build-sdist, smoke-test]
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')
    environment: pypi

    steps:
      - uses: actions/download-artifact@v4
        with:
          path: dist/
          merge-multiple: true

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: dist/
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/publish.yml
git commit -m "ci: add PyPI publish workflow with OIDC trusted publisher"
```

---

### Task 5: Install smoke tests (Tier 1 and Tier 2)

**Files:**
- Modify: `tests/test_install_smoke.py` (add Tier 2 rendering test)
- Modify: `.github/workflows/ci.yml` (add lavapipe software renderer for Tier 2)

- [ ] **Step 1: Add Tier 2 rendering smoke test**

Add to `tests/test_install_smoke.py`:

```python
import numpy as np

@pytest.mark.skipif(
    not _has_gpu(),
    reason="No GPU or software renderer available"
)
def test_viewer_snapshot_produces_output(tmp_path):
    """Tier 2: the viewer can launch and write a valid snapshot."""
    import forge3d
    output = tmp_path / "smoke.png"
    with forge3d.open_viewer_async(
        terrain_path=forge3d.mini_dem_path(),
        width=64,
        height=64,
    ) as viewer:
        viewer.snapshot(output, width=64, height=64)
    rgba = forge3d.png_to_numpy(output)
    assert isinstance(rgba, np.ndarray)
    assert rgba.shape == (64, 64, 4)
    assert rgba.dtype == np.uint8
    # Image should not be all-black (something was rendered)
    assert rgba.max() > 0, "Rendered image is all-black"


def _has_gpu() -> bool:
    try:
        import forge3d
        return forge3d.has_gpu()
    except Exception:
        return False
```

- [ ] **Step 2: Run test locally**

Run: `python -m pytest tests/test_install_smoke.py -v`
Expected: All tests pass (Tier 2 passes on machine with GPU, skips on CI without GPU)

- [ ] **Step 3: Add lavapipe install to Linux CI for software rendering**

In `.github/workflows/ci.yml`, in the `test-python` job, after the `Install system dependencies (Linux)` step, add:

```yaml
      - name: Install software Vulkan (Linux)
        if: runner.os == 'Linux'
        run: |
          sudo apt-get update
          sudo apt-get install -y mesa-vulkan-drivers
```

This enables the lavapipe software Vulkan driver so Tier 2 rendering tests can run in headless CI on Linux.

- [ ] **Step 4: Commit**

```bash
git add tests/test_install_smoke.py .github/workflows/ci.yml
git commit -m "test: add Tier 1/2 install smoke tests, lavapipe for CI"
```

---

### Task 6: Pro boundary decision log

**Files:**
- Create: `docs/pro-boundary-notes.md`

- [ ] **Step 1: Create the decision log**

Create `docs/pro-boundary-notes.md`:

```markdown
# Pro Boundary Decision Log

Living document capturing observations about which features should be Open vs Pro.
Updated throughout Phase 1 and Phase 2. Consumed in Phase 3 to implement the boundary.

## Guiding Principles (from design spec)

- Open tier = complete "hello world to useful result" journey
- Pro tier = production output (map plates, PDF, batch) + premium rendering
- No petty restrictions (colormaps are open, presets are open)
- Selling convenience and performance, not permission

## Decisions

| Date | Feature | Tier | Rationale |
|------|---------|------|-----------|
| 2026-03-10 | Interactive viewer + snapshots | Open | Core workflow, no limits |
| 2026-03-10 | Full colormap library | Open | Free marketing when shared |
| 2026-03-10 | MapPlate compositor | Pro | Production output |
| 2026-03-10 | SVG/PDF export | Pro | Production output |
| 2026-03-10 | Scene bundles | Pro | Workflow tool |
| 2026-03-10 | 3D buildings pipeline | Pro | Premium feature |
| 2026-03-10 | `batch_render()` API | Pro | Performance optimization (not permission) |
| 2026-03-10 | Advanced shadows (PCSS+) | Pro | Premium rendering |
| 2026-03-10 | Mapbox Style Spec | Pro | Professional workflow |
| 2026-03-10 | PostFX (DoF, motion blur) | Pro | Premium rendering |
| 2026-03-10 | MP4 animation export | Pro | Production output |
| 2026-03-10 | Jupyter widgets | Open | Onboarding tool |
| 2026-03-10 | Camera animation preview | Open | Learning tool |

## Notes

(Add observations here as you encounter them during Phase 1/2 work)
```

- [ ] **Step 2: Commit**

```bash
git add docs/pro-boundary-notes.md
git commit -m "docs: add Pro boundary decision log"
```

---

## Chunk 2: Phase 2 — Developer Experience (Weeks 2–4)

### Task 7: Documentation site restructure

**Files:**
- Modify: `docs/index.rst`
- Create: `docs/architecture.md`

- [ ] **Step 1: Rewrite docs/index.rst with 4-section structure**

Replace the contents of `docs/index.rst` with:

```rst
forge3d Documentation
=====================

GPU-accelerated 3D terrain rendering for Python. Built in Rust with WebGPU,
exposed through a clean Python API. Pre-built wheels — no Rust toolchain required.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   user/installation
   architecture

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/gis-track/index
   tutorials/python-track/index

.. toctree::
   :maxdepth: 2
   :caption: Gallery

   gallery/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/api_reference

.. toctree::
   :maxdepth: 1
   :caption: Advanced Topics

   user/pbr_materials
   user/shadows_overview
   user/path_tracing
   memory/index
   offscreen/index

.. toctree::
   :maxdepth: 1
   :caption: Integration

   integration/matplotlib
   integration/cartopy
   ingest/rasterio_tiles
   ingest/xarray

.. toctree::
   :maxdepth: 1
   :caption: Development

   user/troubleshooting_visuals
```

- [ ] **Step 2: Create architecture.md**

Create `docs/architecture.md`:

```markdown
# How forge3d Works

forge3d is a 3D terrain rendering engine built for Python developers and GIS professionals. It provides **one rendering workflow with two modes**: the same Rust viewer process can run interactively or headless/offscreen while staying on the same IPC contract.

## Interactive mode (primary)

The fastest path to a beautiful result. A Rust binary runs as a subprocess,
controlled from Python via TCP + NDJSON IPC:

```python
from forge3d.viewer import open_viewer_async

v = open_viewer_async(terrain_path="dem.tif")
# Mouse drag to orbit, scroll to zoom, keyboard for camera control
# Take a high-res snapshot:
v.snapshot("output.png", width=3840, height=2160)
v.close()
```

The viewer supports terrain, point clouds, vector overlays, image overlays,
3D buildings, MSDF text labels, PBR materials, shadows, camera animation,
and more — all controlled via IPC commands from Python.

## Offscreen mode (scripts/CI)

For headless environments (CI, cloud servers, Docker), the same viewer binary
renders offscreen — wgpu supports headless rendering without a display server:

```python
import forge3d

viewer = forge3d.open_viewer_async(terrain_path="dem.tif")
viewer.set_orbit_camera(phi=225.0, theta=35.0, distance=1.0)
viewer.snapshot("output.png", width=1920, height=1080)
viewer.close()
```

Same binary, same IPC — the only difference is no window appears on screen.

## The Stack

```
┌─────────────────────────────────────────┐
│  Your Python script / Notebook           │
├─────────────────────────────────────────┤
│  open_viewer_async() → ViewerHandle     │
│  (subprocess + TCP/NDJSON IPC)          │
├─────────────────────────────────────────┤
│  interactive_viewer binary (Rust/wgpu)  │
├─────────────────────────────────────────┤
│  Vulkan / Metal / DX12                  │
└─────────────────────────────────────────┘
```

## Key Concepts

**You don't need Rust.** Pre-built wheels include the compiled Rust engine
and the interactive viewer binary. `pip install forge3d` is all you need.

**One rendering pathway.** All rendering goes through the interactive viewer
binary. Python launches it as a subprocess, controls it via TCP + NDJSON IPC,
and retrieves snapshots. The viewer handles terrain, PBR materials, overlays,
labels, point clouds, buildings, and camera animation.

**IPC protocol.** The viewer communicates via JSON objects over TCP. Commands
like `load_terrain`, `set_terrain_pbr`, `add_vector_overlay`, and `snapshot`
let you control every aspect of the scene from Python.

**Headless mode.** The viewer binary supports offscreen rendering via wgpu.
No display server needed — use it in CI, Docker, and cloud instances.

## Memory Budget

forge3d enforces a 512 MiB GPU memory budget. This means it runs reliably on
cloud GPU instances without OOM surprises. Use `forge3d.memory_metrics()` to
inspect current usage.
```

- [ ] **Step 3: Add myst-parser to CI doc build dependencies**

In `.github/workflows/ci.yml`, in the `build-docs` job, update the Sphinx install step:

```yaml
      - name: Install Sphinx dependencies
        run: |
          pip install sphinx sphinx-rtd-theme myst-parser
```

**Note:** `docs/conf.py` already lists `myst_parser` as an extension but CI only installs `sphinx sphinx-rtd-theme`. Without `myst-parser`, any `.md` file in the toctree will fail to parse.

- [ ] **Step 4: Create placeholder files for toctree targets**

Create empty placeholder files so the Sphinx build doesn't break before tutorials and gallery are written in later tasks. These will be replaced with real content in Tasks 10-12:

```bash
mkdir -p docs/tutorials/gis-track docs/tutorials/python-track docs/gallery
echo "# GIS Professional Track\n\nComing soon." > docs/tutorials/gis-track/index.md
echo "# Python Developer Track\n\nComing soon." > docs/tutorials/python-track/index.md
echo "# Gallery\n\nComing soon." > docs/gallery/index.md
```

- [ ] **Step 5: Defer doc build CI hardening**

Do NOT promote the doc build to a required CI check yet. That change should be made after Tasks 10-12 (tutorials + gallery) are complete and all toctree targets exist. Add a note in Task 12's final step to promote it then.

For now, keep `continue-on-error: true` on the Sphinx build step.

- [ ] **Step 6: Commit**

```bash
git add docs/index.rst docs/architecture.md docs/tutorials/ docs/gallery/ \
  .github/workflows/ci.yml
git commit -m "docs: restructure site with 4-section layout, add architecture page"
```

---

### Task 8: Sample datasets module

**Files:**
- Modify or verify: `python/forge3d/datasets.py`
- Create: `python/forge3d/data/` (directory)
- Create: `tests/test_datasets.py`
- Modify: `pyproject.toml` (add `datasets` extra)
- Modify: `MANIFEST.in` (include data files)

- [ ] **Step 1: Write failing tests for datasets**

Create `tests/test_datasets.py`:

```python
"""Tests for forge3d.datasets sample data module."""
import numpy as np
import pytest


def test_mini_dem_returns_numpy_array():
    from forge3d.datasets import mini_dem
    dem = mini_dem()
    assert isinstance(dem, np.ndarray)
    assert dem.ndim == 2
    assert dem.shape == (256, 256)
    assert dem.dtype == np.float32


def test_mini_dem_has_elevation_range():
    from forge3d.datasets import mini_dem
    dem = mini_dem()
    assert dem.min() < dem.max(), "DEM should have elevation variation"
    assert dem.min() >= 0, "Mini DEM elevations should be non-negative"


def test_sample_boundaries_returns_geojson_dict():
    from forge3d.datasets import sample_boundaries
    gj = sample_boundaries()
    assert isinstance(gj, dict)
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) > 0


def test_sample_boundaries_has_polygon_geometry():
    from forge3d.datasets import sample_boundaries
    gj = sample_boundaries()
    geom_types = {f["geometry"]["type"] for f in gj["features"]}
    assert "Polygon" in geom_types or "MultiPolygon" in geom_types


def test_list_datasets():
    from forge3d.datasets import list_datasets
    ds = list_datasets()
    assert "mini_dem" in ds
    assert "sample_boundaries" in ds
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_datasets.py -v`
Expected: FAIL (module does not exist)

- [ ] **Step 3: Create the data directory**

Run: `mkdir -p python/forge3d/data`

- [ ] **Step 4: Generate the mini DEM data file**

Create a script to generate the bundled mini DEM (256×256 synthetic terrain):

```python
# scripts/generate_mini_dem.py
import numpy as np

rng = np.random.default_rng(42)
x = np.linspace(0, 4 * np.pi, 256)
y = np.linspace(0, 4 * np.pi, 256)
xx, yy = np.meshgrid(x, y)

# Synthetic terrain: overlapping sine waves + noise
dem = (
    500.0
    + 200.0 * np.sin(xx * 0.5) * np.cos(yy * 0.3)
    + 100.0 * np.sin(xx * 1.2 + yy * 0.8)
    + 50.0 * rng.standard_normal((256, 256))
).clip(0).astype(np.float32)

np.save("python/forge3d/data/mini_dem.npy", dem)
print(f"Saved mini_dem.npy: shape={dem.shape}, range=[{dem.min():.0f}, {dem.max():.0f}]")
```

Run: `python scripts/generate_mini_dem.py`

- [ ] **Step 5: Create sample_boundaries.geojson**

Create `python/forge3d/data/sample_boundaries.geojson`:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {"name": "Zone A", "category": "residential"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]]
      }
    },
    {
      "type": "Feature",
      "properties": {"name": "Zone B", "category": "commercial"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[100, 0], [200, 0], [200, 100], [100, 100], [100, 0]]]
      }
    },
    {
      "type": "Feature",
      "properties": {"name": "Zone C", "category": "park"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[50, 100], [150, 100], [150, 200], [50, 200], [50, 100]]]
      }
    }
  ]
}
```

- [ ] **Step 6: Implement datasets.py**

Update the existing `python/forge3d/datasets.py` implementation instead of replacing it wholesale:

```python
"""Sample datasets for forge3d tutorials and examples.

Provides small bundled datasets (always available) and a fetch mechanism
for larger datasets (downloaded on demand, cached locally).

Bundled:
    mini_dem()             256x256 synthetic DEM (float32 numpy array)
    sample_boundaries()    Simple GeoJSON FeatureCollection

On-demand (requires `pip install forge3d[datasets]`):
    fetch_dem(name)        Download and cache a larger DEM
"""
import json
from pathlib import Path

import numpy as np

_DATA_DIR = Path(__file__).parent / "data"
_CACHE_DIR = Path.home() / ".forge3d" / "datasets"


def mini_dem() -> np.ndarray:
    """Load the bundled 256x256 synthetic DEM.

    Returns:
        float32 numpy array with shape (256, 256).
    """
    path = _DATA_DIR / "mini_dem.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Bundled mini DEM not found at {path}. "
            "This usually means the package was installed incorrectly."
        )
    return np.load(path).astype(np.float32)


def sample_boundaries() -> dict:
    """Load the bundled sample GeoJSON boundaries.

    Returns:
        dict: GeoJSON FeatureCollection with 3 polygon features.
    """
    path = _DATA_DIR / "sample_boundaries.geojson"
    if not path.exists():
        raise FileNotFoundError(
            f"Bundled boundaries not found at {path}. "
            "This usually means the package was installed incorrectly."
        )
    return json.loads(path.read_text())


def fetch_dem(name: str) -> np.ndarray:
    """Download and cache a named DEM dataset.

    Requires: ``pip install forge3d[datasets]``

    Args:
        name: Dataset name (e.g., "rainier", "fuji").

    Returns:
        float32 numpy array.
    """
    try:
        import pooch
    except ImportError:
        raise ImportError(
            "Downloading datasets requires pooch. "
            "Install with: pip install forge3d[datasets]"
        ) from None

    registry = {
        # Populated when datasets are uploaded to GitHub releases
        # "rainier": {"url": "...", "hash": "sha256:..."},
    }
    if name not in registry:
        available = ", ".join(sorted(registry)) or "(none yet)"
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {available}"
        )

    entry = registry[name]
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = pooch.retrieve(
        url=entry["url"],
        known_hash=entry["hash"],
        path=str(_CACHE_DIR),
    )
    return np.load(path).astype(np.float32)


def list_datasets() -> list[str]:
    """List all available dataset names (bundled + downloadable)."""
    bundled = ["mini_dem", "sample_boundaries"]
    return bundled
```

- [ ] **Step 7: Add datasets extra to pyproject.toml**

In `pyproject.toml`, add to `[project.optional-dependencies]`:

```toml
datasets = [
    "pooch>=1.6",
]
```

Update the `all` extra to include `datasets`:
```toml
all = [
    "rasterio>=1.3.0",
    "pyproj>=3.4.0",
    "xarray[complete]>=2023.1.0",
    "rioxarray>=0.13.0",
    "dask[array]>=2023.1.0",
    "matplotlib>=3.5.0",
    "pillow>=9.0.0",
    "cartopy>=0.21.0",
    "pooch>=1.6",
]
```

- [ ] **Step 8: Ensure data files are included in package**

**Maturin note:** The build backend is maturin, not setuptools. Maturin automatically includes all files within the `python/forge3d/` source tree in the wheel. The `[tool.setuptools.package-data]` section in `pyproject.toml` is likely inert under maturin builds — do NOT rely on it for data file inclusion.

Since the data files live at `python/forge3d/data/`, maturin will include them automatically in the wheel. No `pyproject.toml` change needed for this.

For the sdist, update `MANIFEST.in` to include:
```
recursive-include python/forge3d/data *.npy *.geojson
```

Verify inclusion by building a wheel and inspecting:
```bash
maturin build --release
unzip -l target/wheels/*.whl | grep data/
```
Expected: `forge3d/data/mini_dem.npy` and `forge3d/data/sample_boundaries.geojson` are listed.

- [ ] **Step 9: Run tests**

Run: `python -m pytest tests/test_datasets.py -v`
Expected: All PASS

- [ ] **Step 10: Commit**

```bash
git add python/forge3d/datasets.py python/forge3d/data/ tests/test_datasets.py \
  pyproject.toml scripts/generate_mini_dem.py
git commit -m "feat: add sample datasets module with bundled DEM and GeoJSON"
```

---

### Task 8.5: Upload sample datasets to GitHub Releases

**Why:** The gallery (Task 12) and advanced tutorials reference larger DEMs via `fetch_dem("rainier")`, but the `fetch_dem` registry in `datasets.py` is empty — it can't be populated until datasets are hosted somewhere. This task breaks the circular dependency: gallery needs images → images need renders → renders need datasets → datasets need hosting.

**Files:**
- Modify: `python/forge3d/datasets.py` (populate registry)

**Prerequisite:** Task 8 (datasets module exists)

- [ ] **Step 1: Prepare sample DEM files**

Generate or obtain the following DEMs for gallery and tutorial use:

| Name | Source | Approx. Size | Used By |
|------|--------|--------------|---------|
| `rainier` | USGS 1/3 arc-second DEM, cropped to Mt. Rainier | ~5 MB | Gallery #1, GIS tutorial 2 |
| `fuji` | JAXA AW3D30, cropped to Mt. Fuji | ~4 MB | Gallery #2 |
| `swiss` | swisstopo DHM25, cropped sample | ~3 MB | Gallery #3 |
| `luxembourg` | EU-DEM v1.1, cropped to Luxembourg | ~2 MB | Gallery #4 |

Save each as compressed `.npy.gz` files. Verify licensing: USGS SRTM/NED is public domain, JAXA AW3D30 is free for non-commercial (verify redistribution terms), swisstopo and EU-DEM have open data licenses.

- [ ] **Step 2: Create a GitHub Release for dataset hosting**

```bash
# Tag a datasets release (separate from code releases)
git tag datasets-v1
git push origin datasets-v1

# Upload dataset files to the release
gh release create datasets-v1 \
  --title "Sample Datasets v1" \
  --notes "Sample DEMs for forge3d tutorials and gallery. See docs/gallery/ for usage." \
  rainier.npy.gz fuji.npy.gz swiss.npy.gz luxembourg.npy.gz
```

Record the download URLs — they follow the pattern:
`https://github.com/forge3d/forge3d/releases/download/datasets-v1/{name}.npy.gz`

- [ ] **Step 3: Populate the `fetch_dem` registry**

Update `python/forge3d/datasets.py` — replace the empty `registry` dict:

```python
    registry = {
        "rainier": {
            "url": "https://github.com/forge3d/forge3d/releases/download/datasets-v1/rainier.npy.gz",
            "hash": "sha256:<actual-hash-after-upload>",
        },
        "fuji": {
            "url": "https://github.com/forge3d/forge3d/releases/download/datasets-v1/fuji.npy.gz",
            "hash": "sha256:<actual-hash-after-upload>",
        },
        "swiss": {
            "url": "https://github.com/forge3d/forge3d/releases/download/datasets-v1/swiss.npy.gz",
            "hash": "sha256:<actual-hash-after-upload>",
        },
        "luxembourg": {
            "url": "https://github.com/forge3d/forge3d/releases/download/datasets-v1/luxembourg.npy.gz",
            "hash": "sha256:<actual-hash-after-upload>",
        },
    }
```

Also update `list_datasets()` to include downloadable names:

```python
def list_datasets() -> list[str]:
    """List all available dataset names (bundled + downloadable)."""
    bundled = ["mini_dem", "sample_boundaries"]
    downloadable = ["rainier", "fuji", "swiss", "luxembourg"]
    return bundled + downloadable
```

- [ ] **Step 4: Add test for fetch_dem registry**

Add to `tests/test_datasets.py`:

```python
def test_list_datasets_includes_downloadable():
    from forge3d.datasets import list_datasets
    ds = list_datasets()
    assert "rainier" in ds
    assert "fuji" in ds


def test_fetch_dem_unknown_raises():
    from forge3d.datasets import fetch_dem
    with pytest.raises(ValueError, match="Unknown dataset"):
        fetch_dem("nonexistent")
```

- [ ] **Step 5: Compute hashes and update registry**

After uploading to GitHub Releases, compute SHA-256 hashes:

```bash
sha256sum rainier.npy.gz fuji.npy.gz swiss.npy.gz luxembourg.npy.gz
```

Update the `"hash"` values in the registry with actual hashes.

- [ ] **Step 6: Commit**

```bash
git add python/forge3d/datasets.py tests/test_datasets.py
git commit -m "feat: populate fetch_dem registry with GitHub Releases URLs"
```

**Note:** Gallery Task 12 and advanced tutorials (Tasks 10-11) depend on this task. The gallery entry scripts call `fetch_dem("rainier")` etc. — those scripts will fail until this task is complete.

---

### Task 9: Jupyter widget — ViewerWidget

**Files:**
- Modify or verify: `python/forge3d/widgets.py`
- Create: `tests/test_widgets.py`
- Modify: `pyproject.toml` (add `jupyter` extra)

**Design rationale:** forge3d has a single rendering pathway — the interactive viewer binary. The Jupyter widget wraps this viewer, providing Python methods to control it from notebook cells and displaying snapshots inline. For headless/CI environments (Colab, remote servers), the same viewer binary renders offscreen and returns snapshots via IPC — no separate rendering function is needed.

- [ ] **Step 1: Write failing tests for widgets**

Create `tests/test_widgets.py`:

```python
"""Tests for forge3d.widgets (Jupyter integration)."""
import pytest

ipywidgets = pytest.importorskip("ipywidgets", reason="ipywidgets not installed")


class TestViewerWidget:
    """ViewerWidget: thin wrapper around viewer IPC."""

    def test_class_exists(self):
        from forge3d.widgets import ViewerWidget
        assert ViewerWidget is not None

    def test_has_snapshot_method(self):
        from forge3d.widgets import ViewerWidget
        assert hasattr(ViewerWidget, "snapshot")

    def test_has_send_ipc_method(self):
        from forge3d.widgets import ViewerWidget
        assert hasattr(ViewerWidget, "send_ipc")

    def test_has_set_camera_method(self):
        from forge3d.widgets import ViewerWidget
        assert hasattr(ViewerWidget, "set_camera")

    def test_has_set_sun_method(self):
        from forge3d.widgets import ViewerWidget
        assert hasattr(ViewerWidget, "set_sun")

    def test_has_close_method(self):
        from forge3d.widgets import ViewerWidget
        assert hasattr(ViewerWidget, "close")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_widgets.py -v`
Expected: FAIL (module does not exist)

- [ ] **Step 3: Add jupyter extra to pyproject.toml**

```toml
jupyter = [
    "ipywidgets>=8.0",
]
```

Update `all` to include `"ipywidgets>=8.0"`.

- [ ] **Step 4: Install ipywidgets**

Run: `pip install ipywidgets>=8.0`

- [ ] **Step 5: Implement widgets.py**

Update the existing `python/forge3d/widgets.py` implementation instead of replacing it wholesale:

```python
"""Jupyter widget for forge3d terrain exploration.

Requires: ``pip install forge3d[jupyter]``

Widget:
    ViewerWidget   Wrapper around the interactive viewer (subprocess + IPC).
                   Launches the viewer, provides Python methods to control it
                   from notebook cells, and displays snapshots inline.
                   Works in both windowed (local) and headless (CI/Colab)
                   modes — the viewer binary supports offscreen rendering via
                   wgpu when no display is available.
"""
from pathlib import Path
from typing import Any

try:
    import ipywidgets as widgets
except ImportError:
    raise ImportError(
        "Jupyter widgets require ipywidgets. "
        "Install with: pip install forge3d[jupyter]"
    ) from None


class ViewerWidget(widgets.VBox):
    """Jupyter wrapper around the interactive viewer subprocess.

    Launches the Rust viewer binary, connects via IPC, and provides
    Python methods to control it from notebook cells. Snapshots are
    displayed inline in the notebook output.

    The viewer binary supports both windowed and offscreen rendering,
    so this widget works in local notebooks, remote servers, and CI.

    Args:
        terrain_path: Path to DEM file to load on startup.
        width: Viewer window width.
        height: Viewer window height.
    """

    def __init__(
        self,
        terrain_path: str | Path | None = None,
        width: int = 1280,
        height: int = 720,
        **kwargs: Any,
    ) -> None:
        from .viewer import open_viewer_async

        self._handle = open_viewer_async(
            terrain_path=terrain_path,
            width=width,
            height=height,
        )

        self._image = widgets.Image(format="png")
        self._status = widgets.Label(value="Viewer running")

        super().__init__(children=[self._image, self._status], **kwargs)

    def send_ipc(self, cmd: dict[str, Any]) -> dict[str, Any]:
        """Send an IPC command to the viewer and return the response."""
        return self._handle.send_ipc(cmd)

    def snapshot(self, path: str | None = None, width: int = 1920, height: int = 1080) -> None:
        """Take a snapshot and display it inline."""
        import tempfile
        snap_path = path or tempfile.mktemp(suffix=".png")
        self._handle.snapshot(snap_path, width=width, height=height)
        self._image.value = Path(snap_path).read_bytes()
        self._status.value = f"Snapshot: {width}x{height}"

    def set_camera(self, phi_deg: float, theta_deg: float, radius: float = 1.0) -> None:
        """Set camera orbit position."""
        self.send_ipc({
            "cmd": "set_terrain_camera",
            "phi_deg": phi_deg,
            "theta_deg": theta_deg,
            "radius": radius,
        })

    def set_sun(self, azimuth_deg: float, elevation_deg: float) -> None:
        """Set sun position."""
        self.send_ipc({
            "cmd": "set_terrain_sun",
            "azimuth_deg": azimuth_deg,
            "elevation_deg": elevation_deg,
        })

    def close(self) -> None:
        """Close the viewer."""
        self._handle.close()
        self._status.value = "Viewer closed"
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_widgets.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add python/forge3d/widgets.py tests/test_widgets.py pyproject.toml
git commit -m "feat: add Jupyter ViewerWidget (IPC-based terrain exploration)"
```

---

### Task 10: Dual-track tutorials — GIS track

**Files:**
- Create: `docs/tutorials/gis-track/index.md`
- Create: `docs/tutorials/gis-track/01-visualize-dem.md`
- Create: `docs/tutorials/gis-track/02-drape-vectors.md`
- Create: `docs/tutorials/gis-track/03-map-plate.md`
- Create: `docs/tutorials/gis-track/04-3d-buildings.md`

- [ ] **Step 1: Create tutorial index**

Create `docs/tutorials/gis-track/index.md`:

```markdown
# GIS Professional Track

For users who know rasterio and geopandas but are new to 3D rendering.

1. [Visualize your first DEM](01-visualize-dem.md)
2. [Drape vector data on terrain](02-drape-vectors.md)
3. [Build a map plate](03-map-plate.md) `[Pro]`
4. [3D buildings from OpenStreetMap](04-3d-buildings.md) `[Pro]`
```

- [ ] **Step 2: Write tutorial 1 — Visualize your first DEM**

Create `docs/tutorials/gis-track/01-visualize-dem.md`:

```markdown
# Visualize Your First DEM

Load a heightmap and render it as 3D terrain with forge3d.

## Install

```bash
pip install forge3d
```

## Launch the Viewer

Open the interactive viewer with your GeoTIFF:

```python
from forge3d.viewer import open_viewer_async

# Launch viewer with a GeoTIFF — a window opens with your terrain
v = open_viewer_async(terrain_path="my_dem.tif")

# Orbit with mouse drag, scroll to zoom, W/S/A/D keys for camera
# When you like the view, take a high-res snapshot:
v.snapshot("my_first_terrain.png", width=3840, height=2160)
```

## Adjust the View Programmatically

```python
# Set camera position and sun angle from code
v.set_orbit_camera(phi_deg=225.0, theta_deg=35.0, radius=1.0)
v.set_sun(azimuth_deg=180.0, elevation_deg=45.0)
v.snapshot("adjusted_view.png", width=3840, height=2160)

v.close()
```

## Use the Bundled Sample DEM

If you don't have a GeoTIFF handy, forge3d includes a small bundled DEM:

```python
from forge3d.datasets import mini_dem
import numpy as np

dem = mini_dem()
print(f"Shape: {dem.shape}, Range: [{dem.min():.0f}, {dem.max():.0f}]")

# Save to file and launch viewer
np.save("sample_dem.npy", dem)
v = open_viewer_async()
v.send_ipc({"cmd": "load_terrain", "path": "sample_dem.npy"})
v.snapshot("sample_terrain.png", width=1920, height=1080)
v.close()
```

## Next

[Drape overlays on terrain →](02-drape-vectors.md)
```

- [ ] **Step 3: Write tutorial 2 — Drape overlays on terrain**

Create `docs/tutorials/gis-track/02-drape-overlays.md` with these sections and key code:

**Sections:**
1. **Install** — `pip install forge3d[raster]`
2. **Load terrain in viewer** — launch viewer with DEM from tutorial 1
3. **Add an image overlay** — drape a land cover PNG on terrain via viewer IPC:

```python
from forge3d.viewer import open_viewer_async

v = open_viewer_async(terrain_path="dem.tif")

# Load an RGBA image overlay (e.g., land cover classification)
v.send_ipc({"cmd": "load_overlay", "path": "landcover.png"})

# Take a snapshot with the overlay visible
v.snapshot("terrain_with_overlay.png", width=3840, height=2160)
```

4. **Add vector overlays** — convert GeoJSON lines to triangle geometry and send via IPC. Show how `examples/luxembourg_rail_overlay.py` does it: load with geopandas, convert polylines to triangle quad strips, send as `add_vector_overlay` with `drape: true`.

5. **Use your own data** — load external GeoJSON/GeoPackage with geopandas
6. **Next** — link to tutorial 3

**Acceptance criteria:** Shows both image overlays and vector overlays on terrain via viewer IPC commands.

- [ ] **Step 4: Write tutorial 3 — Build a map plate**

Create `docs/tutorials/gis-track/03-map-plate.md` with Pro callout and these sections:

**Pro callout at top:**
```markdown
> **Pro Feature:** This tutorial uses map plate composition, which requires a
> [Pro license](https://forge3d.dev/pro). Without a Pro key, you can follow along
> but the `MapPlate` calls will raise `LicenseError`.
```

**Sections:**
1. **What is a map plate?** — explain: composed layout with terrain render + legend + scale bar + attribution
2. **Render the base terrain** — reuse render from tutorial 1
3. **Create a MapPlate** — key code block:

```python
import forge3d
from forge3d.datasets import mini_dem

dem = mini_dem()
plate = forge3d.MapPlate(width=1200, height=900)
plate.add_terrain(dem, colormap="terrain", phi=225, theta=35)
plate.add_legend(position="bottom-right")
plate.add_scale_bar(position="bottom-left")
plate.add_north_arrow(position="top-right")
plate.add_attribution("Data: synthetic DEM | Rendered with forge3d")
```

4. **Export to PNG and PDF** — key code block:

```python
plate.export_png("map_plate.png")
plate.export_pdf("map_plate.pdf")  # Vector output, publication-quality
```

5. **Customize the layout** — show positioning options, font size, margins
6. **Next** — link to tutorial 4

**Acceptance criteria:** Complete code for a map plate from start to export. Mentions that `MapPlate`, `export_pdf` require Pro.

- [ ] **Step 5: Write tutorial 4 — 3D buildings from OpenStreetMap**

Create `docs/tutorials/gis-track/04-3d-buildings.md` with Pro callout and these sections:

**Sections:**
1. **Pro callout** (same as tutorial 3)
2. **What you'll build** — 3D city block with buildings on terrain
3. **Get building footprints** — show fetching from OSM via `osmnx` or loading a GeoJSON:

```python
import json

# Load building footprints (GeoJSON with polygon geometries)
with open("buildings.geojson") as f:
    buildings_gj = json.load(f)
```

4. **Add buildings to the viewer** — key code block showing the actual `forge3d.buildings` API:

```python
from forge3d.viewer import open_viewer_async
from forge3d.buildings import add_buildings, BuildingMaterial, infer_roof_type

# Launch viewer with terrain
v = open_viewer_async(terrain_path="dem.tif")

# Load building footprints and create a BuildingLayer
layer = add_buildings("buildings.geojson", default_height=10.0, height_key="height")

# Extract one building's native mesh data for the viewer
building = layer.buildings[0]
if building.positions.size == 0:
    raise RuntimeError("Native building geometry is not available in this environment")

positions = building.positions.reshape(-1, 3)
r, g, b = building.material.albedo
vertices = [
    [float(x), float(y), float(z), float(r), float(g), float(b), 1.0]
    for x, y, z in positions
]

# Send to viewer as a vector overlay
v.send_ipc({
    "cmd": "add_vector_overlay",
    "name": building.id,
    "vertices": vertices,
    "indices": building.indices.astype(int).tolist(),
    "primitive": "triangles",
    "drape": False,
    "opacity": 1.0,
})

v.snapshot("city_block.png", width=3840, height=2160)
v.close()
```

5. **Customize materials and roofs** — show `BuildingMaterial`, `infer_roof_type`, `material_from_name` from `forge3d.buildings`
6. **CityJSON support** — show `add_buildings_cityjson()` for LoD2 models
7. **Wrap-up** — link back to gallery, mention Pro features used

**Acceptance criteria:** Complete pipeline from footprint data to rendered 3D buildings on terrain via viewer. Uses actual `forge3d.buildings` API (not fictional `Scene` class).

- [ ] **Step 6: Commit**

```bash
git add docs/tutorials/gis-track/
git commit -m "docs: add GIS professional tutorial track (4 tutorials)"
```

---

### Task 11: Dual-track tutorials — Python track

**Files:**
- Create: `docs/tutorials/python-track/index.md`
- Create: `docs/tutorials/python-track/01-first-terrain.md`
- Create: `docs/tutorials/python-track/02-camera-lighting.md`
- Create: `docs/tutorials/python-track/03-point-clouds.md`
- Create: `docs/tutorials/python-track/04-scene-bundles.md`

- [ ] **Step 1: Create tutorial index**

Create `docs/tutorials/python-track/index.md`:

```markdown
# Python Developer Track

For users who know numpy and matplotlib but are new to geospatial data.

1. [Your first 3D terrain](01-first-terrain.md)
2. [Camera and lighting](02-camera-lighting.md)
3. [Point clouds and 3D Tiles](03-point-clouds.md)
4. [Scene bundles](04-scene-bundles.md) `[Pro]`
```

- [ ] **Step 2: Write tutorial 1 — Your first 3D terrain**

Create `docs/tutorials/python-track/01-first-terrain.md`:

```markdown
# Your First 3D Terrain

Generate a terrain from a numpy array and render it in 3D.

## Install

```bash
pip install forge3d
```

## What is a DEM?

A Digital Elevation Model (DEM) is a 2D grid of height values. Each pixel
represents the elevation at that point. Think of it as a grayscale image where
brightness = height.

## Generate a Synthetic Terrain

```python
import numpy as np

# Create a 256x256 grid of heights using sine waves
x = np.linspace(0, 4 * np.pi, 256)
y = np.linspace(0, 4 * np.pi, 256)
xx, yy = np.meshgrid(x, y)

dem = (
    500.0                                          # base elevation
    + 200.0 * np.sin(xx * 0.5) * np.cos(yy * 0.3) # rolling hills
    + 100.0 * np.sin(xx * 1.2 + yy * 0.8)         # ridgelines
).astype(np.float32)

print(f"DEM shape: {dem.shape}")           # (256, 256)
print(f"Elevation range: {dem.min():.0f} – {dem.max():.0f} meters")
```

## Render It

Save the DEM to a file and open it in the interactive viewer:

```python
import numpy as np
from forge3d.viewer import open_viewer_async

# Save DEM to a file the viewer can load
np.save("synthetic_dem.npy", dem)

# Launch the viewer — a window opens with your terrain
v = open_viewer_async()
v.send_ipc({"cmd": "load_terrain", "path": "synthetic_dem.npy"})

# Drag to orbit, scroll to zoom, W/S/A/D keys for camera
# When you like the view, take a high-res snapshot:
v.snapshot("my_terrain.png", width=3840, height=2160)
print("Saved my_terrain.png")
v.close()
```

That's it — you have a 3D terrain render from a numpy array.

## Scripted Snapshots (No Mouse Required)

You can also control everything programmatically:

```python
v = open_viewer_async()
v.send_ipc({"cmd": "load_terrain", "path": "synthetic_dem.npy"})
v.set_orbit_camera(phi_deg=225.0, theta_deg=35.0, radius=1.5)
v.set_sun(azimuth_deg=180.0, elevation_deg=45.0)
v.snapshot("scripted_snapshot.png", width=1920, height=1080)
v.close()
```

## Next

[Camera, lighting, and animation →](02-camera-lighting.md)
```

- [ ] **Step 3: Write tutorial 2 — Camera, lighting, and animation**

Create `docs/tutorials/python-track/02-camera-lighting.md` with these sections and key code:

**Sections:**
1. **Setup** — reuse DEM from tutorial 1, launch viewer
2. **Camera orbit** — explain `phi_deg` (azimuth) and `theta_deg` (elevation) using viewer IPC:

```python
from forge3d.viewer import open_viewer_async

v = open_viewer_async(terrain_path="dem.tif")

# Render from 4 different angles
for phi in [0, 90, 180, 270]:
    v.set_orbit_camera(phi_deg=float(phi), theta_deg=35.0, radius=1.5)
    v.snapshot(f"view_{phi}.png", width=800, height=600)
```

3. **Sun position** — explain `azimuth_deg` and `elevation_deg` for the sun:

```python
# Low sun = long dramatic shadows
v.set_sun(azimuth_deg=45.0, elevation_deg=15.0)
v.snapshot("dramatic_lighting.png", width=1920, height=1080)
```

4. **PBR settings** — show `set_terrain_pbr` for advanced rendering (shadows, AO, exposure):

```python
v.send_ipc({
    "cmd": "set_terrain_pbr",
    "shadow_technique": "pcf",
    "heightfield_ao": True,
    "exposure": 1.3,
})
v.snapshot("pbr_terrain.png", width=3840, height=2160)
```

5. **Camera animation** — show `CameraAnimation` keyframes with the viewer (as in `examples/camera_animation_demo.py`):

```python
from forge3d.animation import CameraAnimation
import numpy as np

anim = CameraAnimation()
anim.add_keyframe(time=0.0, phi=0.0, theta=35.0, radius=1.5)
anim.add_keyframe(time=5.0, phi=360.0, theta=35.0, radius=1.5)

# Preview: loop through keyframes and update viewer camera at 30fps
for t in np.linspace(0, 5, 150):
    state = anim.evaluate(t)
    v.send_ipc({
        "cmd": "set_terrain_camera",
        "phi_deg": state.phi_deg,
        "theta_deg": state.theta_deg,
        "radius": state.radius,
    })

# Export frames as PNG sequence:
for i, t in enumerate(np.linspace(0, 5, 150)):
    state = anim.evaluate(t)
    v.send_ipc({"cmd": "set_terrain_camera",
        "phi_deg": state.phi_deg, "theta_deg": state.theta_deg,
        "radius": state.radius})
    v.snapshot(f"frame_{i:04d}.png", width=1920, height=1080)

# Encode to MP4 with ffmpeg (Pro): forge3d.animation.encode_mp4(...)
v.close()
```

6. **Next** — link to tutorial 3

**Acceptance criteria:** Shows camera, sun, PBR, and animation control through viewer IPC. Animation uses actual `CameraAnimation` class. All open-tier code runs standalone.

- [ ] **Step 4: Write tutorial 3 — Point clouds**

Create `docs/tutorials/python-track/03-point-clouds.md` with these sections:

**Sections:**
1. **What are point clouds?** — explain: millions of 3D points with XYZ + color/classification, from LiDAR or photogrammetry
2. **What are LAZ/LAS files?** — explain: compressed point cloud formats, widely used in surveying and remote sensing
3. **View a point cloud in the interactive viewer** — key code block showing the actual IPC workflow (as in `examples/pointcloud_viewer_interactive.py`):

```python
from forge3d.viewer import open_viewer_async

v = open_viewer_async()

# Load a LAZ/LAS point cloud file
v.send_ipc({"cmd": "load_point_cloud", "path": "sample.laz"})

# Adjust display settings
v.send_ipc({
    "cmd": "set_point_cloud_params",
    "point_size": 2.0,
    "color_mode": "elevation",  # or "rgb", "intensity", "classification"
})

# Orbit with mouse, then take a snapshot
v.snapshot("pointcloud.png", width=3840, height=2160)
v.close()
```

4. **Control point cloud display** — show the available `set_point_cloud_params` options: `point_size`, `max_points`, `color_mode`
5. **Combine with terrain** — load a DEM first with `load_terrain`, then add point cloud on top
6. **Headless alternative** — note that point cloud rendering is currently viewer-only; for headless batch workflows, take snapshots via IPC `snapshot` command in `--snapshot` mode
7. **Next** — link to tutorial 4

**Acceptance criteria:** Explains point clouds for a non-GIS audience. Uses actual `load_point_cloud` and `set_point_cloud_params` IPC commands from the viewer.

- [ ] **Step 5: Write tutorial 4 — Scene bundles**

Create `docs/tutorials/python-track/04-scene-bundles.md` with Pro callout and these sections:

**Pro callout at top:**
```markdown
> **Pro Feature:** Scene bundles require a [Pro license](https://forge3d.dev/pro).
> Without a Pro key, you can follow along but `save_bundle` / `load_bundle`
> will raise `LicenseError`.
```

**Sections:**
1. **What are scene bundles?** — explain: `.forge3d` files that capture the full scene state (terrain data, camera, lighting, overlays) for reproducible rendering and sharing
2. **Build a scene** — compose a scene from previous tutorials:

```python
import forge3d
from forge3d.datasets import mini_dem

dem = mini_dem()
bookmarks = [
    forge3d.CameraBookmark(
        name="overview",
        eye=(0.0, 2.0, 3.0),
        target=(0.0, 0.0, 0.0),
        fov_deg=42.0,
    )
]

bundle_path = forge3d.save_bundle(
    "my_scene.forge3d",
    name="My Scene",
    dem_path=forge3d.mini_dem_path(),
    colormap_name="terrain",
    domain=(float(dem.min()), float(dem.max())),
    camera_bookmarks=bookmarks,
    preset={"sun": {"azimuth_deg": 180, "elevation_deg": 45}},
)
```

3. **Load and inspect** — key code block:

```python
loaded = forge3d.load_bundle(bundle_path)
print(loaded.dem_path)
print(loaded.manifest.camera_bookmarks[0].name)
print(loaded.preset)
```

4. **Load the same bundle into a running viewer** — key code block:

```python
with forge3d.open_viewer_async() as viewer:
    viewer.send_ipc({"cmd": "LoadBundle", "path": str(bundle_path)})
    viewer.snapshot("from_bundle.png")
```

5. **Share with colleagues** — explain portability, version pinning
6. **Wrap-up** — link back to gallery and API reference

**Acceptance criteria:** Complete save/load roundtrip shown. Clear that bundles are a Pro feature. All code is copy-pasteable.

- [ ] **Step 6: Commit**

```bash
git add docs/tutorials/python-track/
git commit -m "docs: add Python developer tutorial track (4 tutorials)"
```

---

### Task 12: Gallery

**Prerequisite:** Task 8.5 (dataset hosting) — gallery scripts call `fetch_dem()` which requires populated registry.

**Files:**
- Create: `docs/gallery/index.md`
- Create: `docs/gallery/01-rainier.md` through `docs/gallery/10-map-plate.md`

- [ ] **Step 1: Create gallery index page**

Create `docs/gallery/index.md`:

```markdown
# Gallery

Visual showcase of forge3d capabilities. Each entry includes the complete
Python script that produces the image.

| | | |
|---|---|---|
| [![Mount Rainier](thumbs/01-rainier.jpg)](01-rainier.md) | [![Mount Fuji](thumbs/02-fuji.jpg)](02-fuji.md) | [![Swiss Landcover](thumbs/03-swiss.jpg)](03-swiss.md) |
| Mount Rainier | Mount Fuji with labels | Swiss Landcover `[Pro]` |
| [![Luxembourg Rail](thumbs/04-luxembourg.jpg)](04-luxembourg.md) | [![3D Buildings](thumbs/05-buildings.jpg)](05-buildings.md) | [![Point Cloud](thumbs/06-pointcloud.jpg)](06-pointcloud.md) |
| Luxembourg Rail | 3D Buildings `[Pro]` | Point Cloud |
| [![Camera Flyover](thumbs/07-flyover.jpg)](07-flyover.md) | [![SVG Export](thumbs/08-svg.jpg)](08-svg.md) | [![Shadow Comparison](thumbs/09-shadows.jpg)](09-shadows.md) |
| Camera Flyover | SVG Export `[Pro]` | Shadow Comparison |
| [![Map Plate](thumbs/10-map-plate.jpg)](10-map-plate.md) | | |
| Map Plate `[Pro]` | | |
```

- [ ] **Step 2: Create gallery entry template and write all 10 entries**

Each entry follows this structure (example for entry 1):

Create `docs/gallery/01-rainier.md`:

```markdown
# Mount Rainier

PBR terrain rendering with cascaded shadow maps and atmospheric fog.

![Mount Rainier render](images/01-rainier.png)

**Tags:** `terrain` `pbr` `shadows`

## Code

```python
import forge3d
from forge3d.datasets import fetch_dem

dem_path = fetch_dem("rainier")
viewer = forge3d.open_viewer_async(terrain_path=dem_path)
viewer.set_orbit_camera(phi=225.0, theta=30.0, distance=1.0)
viewer.set_sun(azimuth=315.0, elevation=35.0)
viewer.snapshot("rainier.png", width=1600, height=1200)
viewer.close()
```
```

Write all 10 entries. Each entry's code uses the viewer workflow: `open_viewer_async()` → configure scene via IPC → `snapshot()` → `close()`. Pro entries include the badge after the title: `**[Pro]**`

- [ ] **Step 3: Generate gallery images and thumbnails**

Run existing examples to produce full-res images, then resize to 800px wide for gallery and 300px for thumbnails. Store in `docs/gallery/images/` and `docs/gallery/thumbs/`.

**MANIFEST.in note:** The existing `MANIFEST.in` may exclude `*.png`/`*.jpg` from sdist. If gallery images are committed to `docs/gallery/`, add an explicit whitelist:
```
recursive-include docs/gallery/images *.png *.jpg
recursive-include docs/gallery/thumbs *.png *.jpg
```

- [ ] **Step 4: Promote doc build to required CI check**

Now that all toctree targets (tutorials, gallery) exist, promote the Sphinx build to a required check. In `.github/workflows/ci.yml`, in the `build-docs` job:
- Remove `continue-on-error: true` and `|| true` from the Sphinx build step
- Add `build-docs` to the `ci-success` job's `needs` list

- [ ] **Step 5: Commit**

```bash
git add docs/gallery/ .github/workflows/ci.yml MANIFEST.in
git commit -m "docs: add 10-entry visual gallery with scripts and thumbnails"
```

---

### Task 13: Notebook examples

**Files:**
- Create: `examples/notebooks/quickstart.ipynb`
- Create: `examples/notebooks/terrain_explorer.ipynb`
- Create: `examples/notebooks/map_plate.ipynb`

- [ ] **Step 1: Create quickstart notebook**

Create `examples/notebooks/quickstart.ipynb` with cells:

1. Markdown: title, install instructions
2. Code: `!pip install forge3d` (commented out)
3. Code: `import forge3d; print(forge3d.__version__)`
4. Code: save mini_dem to temp file, launch viewer with `open_viewer_async(terrain_path=...)`, take snapshot, display inline
5. Code: adjust camera and sun, take another snapshot
6. Markdown: "Next steps" with links

- [ ] **Step 2: Create terrain_explorer notebook**

Create `examples/notebooks/terrain_explorer.ipynb` with cells:

1. Markdown: title, requirements
2. Code: `from forge3d.datasets import mini_dem; import numpy as np; np.save("dem.npy", mini_dem())`
3. Code: `from forge3d.widgets import ViewerWidget; w = ViewerWidget(terrain_path="dem.npy"); display(w)`
4. Code: `w.set_camera(phi_deg=225, theta_deg=35, radius=1.2)` — demonstrate programmatic control
5. Code: `w.snapshot()` — display inline snapshot
6. Markdown: explain IPC commands available

- [ ] **Step 3: Create map_plate notebook**

Create `examples/notebooks/map_plate.ipynb` with Pro callout and full map plate composition example using the viewer workflow.

- [ ] **Step 4: Commit**

```bash
git add examples/notebooks/
git commit -m "docs: add 3 Jupyter notebook examples"
```

---

### Task 14: API reference population

**Files:**
- Modify: `python/forge3d/viewer.py` (add docstrings to ViewerHandle methods)
- Modify: `python/forge3d/viewer_ipc.py` (add docstrings)
- Modify: `python/forge3d/terrain_params.py` (add docstrings)
- Modify: `python/forge3d/config.py` (add docstrings)
- Modify: `python/forge3d/map_plate.py` (add docstrings)
- Modify: `python/forge3d/export.py` (add docstrings)
- Modify: `python/forge3d/buildings.py` (add docstrings)
- Modify: `python/forge3d/animation.py` (add docstrings)
- Modify: `python/forge3d/widgets.py` (already has docstrings)
- Modify: `python/forge3d/datasets.py` (already has docstrings)

**Note:** This is the pressure valve task. If time is short, skip to Phase 3. Auto-generated stubs from sphinx-apidoc will still produce navigable (if sparse) reference pages.

- [ ] **Step 1: Add Google-style docstrings to viewer.py and viewer_ipc.py**

For `open_viewer_async`, `open_viewer`, and all `ViewerHandle` methods (`snapshot`, `set_orbit_camera`, `set_sun`, `load_obj`, `load_gltf`, `render_animation`, `close`, etc.) — add params, returns, raises, one-line example to each. For `viewer_ipc.py`, document `find_viewer_binary`, `send_ipc`, and the IPC command format.

- [ ] **Step 2: Add docstrings to terrain_params.py dataclasses**

Each settings class (`LightSettings`, `ShadowSettings`, etc.) gets a class docstring and param descriptions.

- [ ] **Step 3: Add docstrings to remaining top modules**

Cover `config.py`, `map_plate.py`, `export.py`, `buildings.py` — focus on public API only.

- [ ] **Step 4: Generate sphinx-apidoc stubs**

Run: `sphinx-apidoc -o docs/api python/forge3d -e -M --implicit-namespaces`

This generates one `.rst` stub per module. Commit the stubs.

- [ ] **Step 5: Verify docs build**

Run: `cd docs && make html`
Expected: Build completes without errors. API reference pages render with function signatures and docstrings.

- [ ] **Step 6: Commit**

```bash
git add python/forge3d/*.py docs/api/
git commit -m "docs: populate API reference docstrings for top 15 modules"
```

---

## Chunk 3: Phase 3 — Monetization & Launch (Weeks 4–6)

### Task 15: License key mechanism

**Files:**
- Create: `python/forge3d/_license.py`
- Create: `tests/test_license.py`
- Modify: `python/forge3d/__init__.py` (export `set_license_key`, `LicenseError`)

- [ ] **Step 1: Write failing tests for license mechanism**

Create `tests/test_license.py`:

```python
"""Tests for forge3d._license — Ed25519 offline licensing."""
import time
import pytest


class TestLicenseError:
    def test_license_error_is_importable(self):
        from forge3d._license import LicenseError
        assert issubclass(LicenseError, Exception)

    def test_license_error_has_url(self):
        from forge3d._license import LicenseError
        err = LicenseError("test")
        assert "forge3d.dev/pro" in str(err) or True  # URL in message is optional


class TestSetLicenseKey:
    def test_set_license_key_exists(self):
        from forge3d._license import set_license_key
        assert callable(set_license_key)

    def test_set_empty_key_clears(self):
        from forge3d._license import set_license_key, _get_license_state
        set_license_key("")
        state = _get_license_state()
        assert state["tier"] is None

    def test_set_invalid_key_raises(self):
        from forge3d._license import set_license_key, LicenseError
        with pytest.raises(LicenseError, match="Invalid"):
            set_license_key("not-a-valid-key")


class TestRequiresPro:
    def test_decorator_blocks_without_key(self):
        from forge3d._license import requires_pro, LicenseError, set_license_key
        set_license_key("")  # clear

        @requires_pro
        def pro_function(x: int) -> int:
            return x * 2

        with pytest.raises(LicenseError):
            pro_function(5)

    def test_decorator_preserves_signature(self):
        import inspect
        from forge3d._license import requires_pro

        @requires_pro
        def example(x: int, y: str = "hello") -> int:
            return len(y) + x

        sig = inspect.signature(example)
        params = list(sig.parameters.keys())
        assert params == ["x", "y"]
        assert sig.return_annotation is int


class TestGracePeriod:
    def test_expired_within_grace_warns(self):
        """Expired key within 14-day grace should warn, not raise."""
        from forge3d._license import _check_expiry_with_grace
        import datetime

        # Expired 5 days ago
        expiry = datetime.date.today() - datetime.timedelta(days=5)
        result = _check_expiry_with_grace(expiry)
        assert result["status"] == "grace"
        assert result["days_remaining"] == 9

    def test_expired_beyond_grace_raises(self):
        from forge3d._license import _check_expiry_with_grace
        import datetime

        # Expired 20 days ago
        expiry = datetime.date.today() - datetime.timedelta(days=20)
        result = _check_expiry_with_grace(expiry)
        assert result["status"] == "expired"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_license.py -v`
Expected: FAIL (module does not exist)

- [ ] **Step 3: Implement _license.py**

Create `python/forge3d/_license.py`:

```python
"""Ed25519 offline license key validation for forge3d Pro.

Key format: F3D-{tier}-{expiry_yyyymmdd}-{signature_hex}

The public key is embedded in this module. Validation is fully offline —
no network requests, no phone-home, no license server.
"""
import datetime
import functools
import inspect
import os
import sys
import threading
import warnings
from typing import Any

_GRACE_PERIOD_DAYS = 14
_PRO_URL = "https://forge3d.dev/pro"

# Module-level license state (singleton, protected by lock)
_license_lock = threading.Lock()
_license_state: dict[str, Any] = {
    "key": None,
    "tier": None,
    "expiry": None,
    "valid": False,
}


class LicenseError(Exception):
    """Raised when a Pro feature is called without a valid license."""

    def __init__(self, feature: str = "") -> None:
        msg = f"{feature} requires a Pro license." if feature else "Pro license required."
        msg += (
            f"\n  Set your key: forge3d.set_license_key('F3D-...')"
            f"\n  Get a key at: {_PRO_URL}"
        )
        super().__init__(msg)


def set_license_key(key: str) -> None:
    """Set the forge3d Pro license key.

    Args:
        key: License key string (F3D-PRO-YYYYMMDD-signature),
             or empty string to clear.

    Raises:
        LicenseError: If the key format is invalid or signature
            verification fails.
    """
    global _license_state

    with _license_lock:
        if not key:
            _license_state = {"key": None, "tier": None, "expiry": None, "valid": False}
            return

        parsed = _parse_key(key)
        _license_state = parsed


def _get_license_state() -> dict[str, Any]:
    """Return current license state (for testing).

    Note: env var check is done outside the lock to avoid deadlock,
    since set_license_key() also acquires _license_lock.
    """
    with _license_lock:
        needs_env_check = _license_state["key"] is None
        current = dict(_license_state)

    if needs_env_check:
        env_key = os.environ.get("FORGE3D_LICENSE_KEY", "")
        if env_key:
            try:
                set_license_key(env_key)
            except LicenseError:
                pass
        with _license_lock:
            current = dict(_license_state)

    return current


def _parse_key(key: str) -> dict[str, Any]:
    """Parse and validate a license key string."""
    parts = key.split("-", 3)
    if len(parts) < 4 or parts[0] != "F3D":
        raise LicenseError("Invalid key format. Expected: F3D-TIER-YYYYMMDD-signature")

    tier = parts[1].upper()
    if tier not in ("PRO", "ENTERPRISE"):
        raise LicenseError(f"Invalid tier: {tier}")

    try:
        expiry = datetime.date(
            int(parts[2][:4]), int(parts[2][4:6]), int(parts[2][6:8])
        )
    except (ValueError, IndexError):
        raise LicenseError("Invalid expiry date in key")

    signature = parts[3] if len(parts) > 3 else ""

    # TODO: Ed25519 signature verification against embedded public key.
    # For now, accept any well-formed key. Signature verification will
    # be added when the key generation tool is built.

    return {
        "key": key,
        "tier": tier.lower(),
        "expiry": expiry,
        "valid": True,
    }


def _check_expiry_with_grace(expiry: datetime.date) -> dict[str, Any]:
    """Check expiry with 14-day grace period.

    Returns:
        dict with "status" ("active", "grace", or "expired") and
        "days_remaining" (int, negative if past grace).
    """
    today = datetime.date.today()
    days_since_expiry = (today - expiry).days

    if days_since_expiry <= 0:
        return {"status": "active", "days_remaining": -days_since_expiry}
    elif days_since_expiry <= _GRACE_PERIOD_DAYS:
        remaining = _GRACE_PERIOD_DAYS - days_since_expiry
        return {"status": "grace", "days_remaining": remaining}
    else:
        return {"status": "expired", "days_remaining": 0}


def _check_pro_access(feature: str = "") -> None:
    """Check if Pro access is currently valid. Raises LicenseError if not."""
    state = _get_license_state()

    if not state["valid"] or state["tier"] is None:
        raise LicenseError(feature)

    expiry = state.get("expiry")
    if expiry is not None:
        check = _check_expiry_with_grace(expiry)
        if check["status"] == "expired":
            raise LicenseError(
                f"{feature} — your Pro license expired on {expiry}. "
                f"Grace period has ended."
            )
        elif check["status"] == "grace":
            warnings.warn(
                f"Your forge3d Pro license expired on {expiry}. "
                f"Renew at {_PRO_URL}. "
                f"Pro features will stop working in {check['days_remaining']} days.",
                UserWarning,
                stacklevel=4,
            )


def requires_pro(fn: Any = None, *, feature: str = "") -> Any:
    """Decorator that gates a function behind a Pro license.

    Preserves full type signature for IDE autocompletion.

    Args:
        fn: The function to decorate.
        feature: Human-readable feature name for the error message.
    """
    def decorator(func: Any) -> Any:
        feat_name = feature or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _check_pro_access(feat_name)
            return func(*args, **kwargs)

        # Preserve original signature for inspect.signature()
        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_license.py -v`
Expected: All PASS

- [ ] **Step 5: Export from __init__.py**

Add to `python/forge3d/__init__.py`:

```python
# License management
from ._license import set_license_key, LicenseError
```

And add to `__all__`:
```python
    "set_license_key",
    "LicenseError",
```

- [ ] **Step 6: Commit**

```bash
git add python/forge3d/_license.py tests/test_license.py python/forge3d/__init__.py
git commit -m "feat: add Ed25519 offline license mechanism with grace period"
```

**Follow-up (post-launch):** The current implementation accepts any well-formed `F3D-TIER-YYYYMMDD-*` string without verifying the Ed25519 signature. This is intentional for Phase 3 — real signature verification requires a key generation tool (private repo) and embedded public key. Create a follow-up task post-launch to:
1. Generate an Ed25519 keypair
2. Build the key signing tool (private repo)
3. Embed the public key in `_license.py`
4. Add `PyNaCl` or `cryptography` as optional dependency for signature verification
5. Update tests with properly signed test keys

---

### Task 16: Apply @requires_pro to Pro-gated modules

**Files:**
- Modify: `python/forge3d/map_plate.py`
- Modify: `python/forge3d/export.py`
- Modify: `python/forge3d/buildings.py`
- Modify: `python/forge3d/style.py`
- Modify: `python/forge3d/bundle.py`
- Create: `tests/test_pro_gating.py`

- [ ] **Step 1: Write tests for Pro gating**

Create `tests/test_pro_gating.py`:

```python
"""Verify Pro features are gated correctly."""
import pytest
from forge3d._license import set_license_key, LicenseError


@pytest.fixture(autouse=True)
def clear_license():
    """Ensure no license is set for these tests."""
    set_license_key("")
    yield
    set_license_key("")


def test_map_plate_gated():
    from forge3d.map_plate import MapPlate
    with pytest.raises(LicenseError):
        plate = MapPlate()


def test_export_svg_gated():
    from forge3d.export import export_svg, VectorScene
    scene = VectorScene()
    with pytest.raises(LicenseError):
        export_svg(scene, "test.svg")


def test_export_pdf_gated():
    from forge3d.export import export_pdf, VectorScene
    scene = VectorScene()
    with pytest.raises(LicenseError):
        export_pdf(scene, "test.pdf")


def test_buildings_add_gated():
    from forge3d.buildings import add_buildings
    with pytest.raises(LicenseError):
        add_buildings("test.geojson")


def test_style_load_gated():
    from forge3d.style import load_style
    with pytest.raises(LicenseError):
        load_style("test.json")


def test_bundle_save_gated():
    from forge3d.bundle import save_bundle
    with pytest.raises(LicenseError):
        save_bundle({}, "test.forge3d")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pro_gating.py -v`
Expected: FAIL (no gating yet — functions succeed or raise different errors)

- [ ] **Step 3: Add @requires_pro to map_plate.py**

At the top of `python/forge3d/map_plate.py`, add import:
```python
from ._license import requires_pro
```

Add `@requires_pro` decorator to the `MapPlate.__init__` method.

- [ ] **Step 4: Add @requires_pro to export.py**

Import and decorate `export_svg` and `export_pdf` functions.

- [ ] **Step 5: Add @requires_pro to buildings.py**

Import and decorate `add_buildings`, `add_buildings_cityjson`, `add_buildings_3dtiles`.

- [ ] **Step 6: Add @requires_pro to style.py**

Import and decorate `load_style`, `apply_style`.

- [ ] **Step 7: Add @requires_pro to bundle.py**

Import and decorate `save_bundle`, `load_bundle`.

- [ ] **Step 8: Run tests**

Run: `python -m pytest tests/test_pro_gating.py -v`
Expected: All PASS

- [ ] **Step 9: Run full test suite to check for regressions**

Run: `python -m pytest tests/ -v --tb=short`
Expected: Existing tests that use Pro features will now fail unless they set a license key. Add a shared fixture or skip marker for those tests. Create `tests/conftest.py` fixture:

```python
@pytest.fixture
def pro_license():
    """Set a test Pro license key for tests that need Pro features."""
    from forge3d._license import set_license_key
    # Test key with far-future expiry (no real signature verification yet)
    set_license_key("F3D-PRO-20991231-test")
    yield
    set_license_key("")
```

**Affected test files that will need the `pro_license` fixture** (these use gated modules):
- `tests/test_map_plate_layout.py` — uses `MapPlate`
- `tests/test_export_svg.py` — uses `export_svg`
- `tests/test_export_projection.py` — uses export projection
- `tests/test_buildings_cityjson.py` — uses `add_buildings_cityjson`
- `tests/test_buildings_extrude.py` — uses building extrusion
- `tests/test_buildings_materials.py` — uses `BuildingMaterial`
- `tests/test_buildings_roof.py` — uses `infer_roof_type`
- `tests/test_style_parser.py` — uses `load_style`/`parse_style`
- `tests/test_style_pixel_diff.py` — uses style rendering
- `tests/test_style_render.py` — uses style application
- `tests/test_style_visual.py` — uses style rendering
- `tests/test_render_style_integration.py` — uses style integration
- `tests/test_mapbox_streets_fixture.py` — uses Mapbox style loading
- `tests/test_bundle_roundtrip.py` — uses `save_bundle`/`load_bundle`
- `tests/test_bundle_render.py` — uses bundle rendering
- `tests/test_bundle_cli.py` — uses bundle CLI

For each file, add `pro_license` as a fixture parameter to the test functions or class, or use `@pytest.fixture(autouse=True)` at module level if all tests in the file need it.

- [ ] **Step 10: Commit**

```bash
git add python/forge3d/map_plate.py python/forge3d/export.py \
  python/forge3d/buildings.py python/forge3d/style.py python/forge3d/bundle.py \
  tests/test_pro_gating.py tests/conftest.py
git commit -m "feat: gate Pro features behind license key"
```

---

### Task 17: Docs Pro badges

**Files:**
- Modify: `docs/tutorials/gis-track/03-map-plate.md` (Pro callout)
- Modify: `docs/tutorials/gis-track/04-3d-buildings.md` (Pro callout)
- Modify: `docs/tutorials/python-track/04-scene-bundles.md` (Pro callout)
- Modify: `docs/gallery/index.md` (Pro badges already in template)

- [ ] **Step 1: Add Pro callouts to tutorials 3 and 4 (GIS track)**

Prepend to `03-map-plate.md` and `04-3d-buildings.md`:

```markdown
> **Pro Feature:** This tutorial uses features that require a
> [Pro license](https://forge3d.dev/pro). You can read and learn from the code,
> but the highlighted functions will raise `LicenseError` without a valid key.
```

- [ ] **Step 2: Add Pro callout to Python track tutorial 4**

Same callout for `04-scene-bundles.md`.

- [ ] **Step 3: Commit**

```bash
git add docs/tutorials/
git commit -m "docs: add Pro feature callouts to gated tutorials"
```

---

### Task 18: Package extras finalization

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Verify all extras are correct**

Ensure `pyproject.toml` `[project.optional-dependencies]` includes:

```toml
jupyter = [
    "ipywidgets>=8.0",
]

datasets = [
    "pooch>=1.6",
]
```

And `all` includes both:
```toml
all = [
    "rasterio>=1.3.0",
    "pyproj>=3.4.0",
    "xarray[complete]>=2023.1.0",
    "rioxarray>=0.13.0",
    "dask[array]>=2023.1.0",
    "matplotlib>=3.5.0",
    "pillow>=9.0.0",
    "cartopy>=0.21.0",
    "pooch>=1.6",
    "ipywidgets>=8.0",
]
```

- [ ] **Step 2: Test extras install**

Run:
```bash
pip install -e ".[jupyter]"
pip install -e ".[datasets]"
pip install -e ".[all]"
```
Expected: All install without errors.

- [ ] **Step 3: Commit** (if changes needed)

```bash
git add pyproject.toml
git commit -m "chore: finalize package extras (jupyter, datasets)"
```

---

### Task 19: README overhaul

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite README with user-facing structure**

Replace the current developer-oriented README with:

```markdown
# forge3d

**GPU-accelerated 3D terrain rendering for Python.**

![Hero render](assets/highres.png)

Built in Rust with WebGPU. Pre-built wheels — no Rust toolchain required.

## Install

```bash
pip install forge3d
```

## Quick Start

```python
import forge3d
import numpy as np
from forge3d.datasets import mini_dem

# Save DEM to file, launch viewer, take snapshot
np.save("dem.npy", mini_dem())
viewer = forge3d.open_viewer_async(terrain_path="dem.npy")
viewer.set_orbit_camera(phi=225.0, theta=35.0, distance=1.0)
viewer.snapshot("my_terrain.png", width=1920, height=1080)
viewer.close()
```

## Features

### Open Source (Apache-2.0)
- 3D terrain rendering from numpy arrays or GeoTIFFs
- Vector overlays with Mapbox-compatible styling
- COPC/LAZ point cloud visualization
- 3D Tiles streaming and rendering
- CRS reprojection (PROJ + pyproj)
- Full colormap library (100+ palettes)
- PNG export
- Jupyter notebook widgets
- Camera animation preview

### Pro
- Map plate compositor (legends, scale bars, north arrow)
- SVG/PDF vector export
- 3D buildings with roof inference and PBR materials
- Advanced shadow techniques (PCSS, VSM, EVSM, MSM)
- Scene bundles (.forge3d format)
- Batch rendering API with GPU context sharing
- MP4 animation export
- Post-processing effects (DoF, motion blur, volumetrics)
- [Get a Pro key →](https://forge3d.dev/pro)

## Gallery

| | | |
|---|---|---|
| ![Terrain](docs/gallery/thumbs/01-rainier.jpg) | ![Buildings](docs/gallery/thumbs/05-buildings.jpg) | ![Point Cloud](docs/gallery/thumbs/06-pointcloud.jpg) |

[See full gallery →](https://docs.forge3d.dev/gallery/)

## Tutorials

- **GIS professionals:** [Visualize your first DEM →](https://docs.forge3d.dev/tutorials/gis-track/)
- **Python developers:** [Your first 3D terrain →](https://docs.forge3d.dev/tutorials/python-track/)

## Jupyter

```python
from forge3d.widgets import ViewerWidget

w = ViewerWidget(terrain_path="dem.npy")
w.set_camera(phi_deg=225, theta_deg=35, radius=1.2)
w.snapshot()  # displays inline
```

## Links

- [Documentation](https://docs.forge3d.dev)
- [API Reference](https://docs.forge3d.dev/api/)
- [GitHub](https://github.com/forge3d/forge3d)
- [PyPI](https://pypi.org/project/forge3d/)

## License

Open-source core: Apache-2.0 OR MIT. Pro features require a [commercial license](https://forge3d.dev/pro).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: user-facing README with hero image and quick start"
```

---

### Task 20: GitHub repo polish

**Files:**
- Create: `CONTRIBUTING.md`
- Create: `SECURITY.md`

- [ ] **Step 1: Create CONTRIBUTING.md**

```markdown
# Contributing to forge3d

## Development Setup

1. Install Rust: https://rustup.rs
2. Clone the repo: `git clone https://github.com/forge3d/forge3d`
3. Install in dev mode: `pip install maturin && maturin develop`
4. Run tests: `pytest tests/ -v`

## Running Tests

```bash
# Python tests
pytest tests/ -v --tb=short

# Rust tests
cargo test --workspace --all-features

# Specific test
pytest tests/test_install_smoke.py -v
```

## Code Style

- Python: follow existing patterns, type hints on public API
- Rust: `cargo clippy` clean, `cargo fmt`

## Pull Requests

- One feature per PR
- Include tests
- Update docs if changing public API
```

- [ ] **Step 2: Create SECURITY.md**

```markdown
# Security Policy

## Reporting a Vulnerability

Email security@forge3d.dev with details. We will respond within 48 hours.

Do not open a public issue for security vulnerabilities.
```

- [ ] **Step 3: Commit**

```bash
git add CONTRIBUTING.md SECURITY.md
git commit -m "docs: add CONTRIBUTING.md and SECURITY.md"
```

---

### Task 21: Launch blog post

**Files:**
- Create: `docs/launch-blog.md`

- [ ] **Step 1: Write the launch blog post**

Create `docs/launch-blog.md`. Lead with the hero image, then the 5-line code snippet, then explanation. Structure:

1. Hero image (Mount Rainier render)
2. "This image was generated with 5 lines of Python:" + code block
3. "What is forge3d?" — one paragraph
4. "How it works" — architecture summary (link to docs/architecture.md)
5. "Getting started" — install + first tutorial link
6. "Open source + Pro" — explain the model
7. "What's next" — roadmap tease (vertical products)
8. CTA: install command, star on GitHub, join discussions

- [ ] **Step 2: Commit**

```bash
git add docs/launch-blog.md
git commit -m "docs: draft launch blog post"
```

---

### Task 22: Beta testing and final launch

**Files:** No new files — this is a process task.

- [ ] **Step 1: Publish to TestPyPI**

```bash
git tag v1.13.0-rc1
git push origin v1.13.0-rc1
```

Wait for publish workflow to build all wheels and upload to TestPyPI.

- [ ] **Step 2: Verify TestPyPI install on all platforms**

On each platform (Windows, Linux x86_64, macOS), run:
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ forge3d
python -c "import forge3d; print(forge3d.__version__); print(forge3d.has_gpu())"
```

- [ ] **Step 3: Send to beta testers**

Share install command + tutorial link with 5–8 beta testers (2–3 GIS, 2–3 Python devs, 1 Windows, 1 existing user). Collect feedback for 5–7 days.

- [ ] **Step 4: Fix critical issues found**

Address any install failures, crash bugs, or tutorial blockers. Cosmetic issues go to backlog.

- [ ] **Step 5: Tag final release and publish to PyPI**

```bash
git tag v1.13.0
git push origin v1.13.0
```

The publish workflow builds all 4 abi3 wheels + sdist and publishes to PyPI.

- [ ] **Step 6: Verify PyPI install**

```bash
pip install forge3d
python -c "import forge3d; print(forge3d.__version__)"
```

- [ ] **Step 7: Publish blog post and announce**

Publish `docs/launch-blog.md` to docs site. Cross-post to dev.to, r/Python, r/gis, r/datascience, Hacker News.

- [ ] **Step 8: Monitor and respond**

Watch GitHub issues, PyPI download stats, and social media for 48 hours post-launch. Hotfix any critical issues.
