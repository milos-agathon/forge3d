# Khumbu Sentinel Timelapse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a forge3d example that renders a polished winter-to-winter Sentinel-2 time-series animation of the Khumbu Icefall with live public data and local caching.

**Architecture:** Add one self-contained example script with small pure-Python helpers for date defaults, STAC payloads, scene selection, Copernicus DEM URL construction, RGB normalization, and frame output. Keep network/GPU paths behind the CLI runtime while unit tests cover deterministic helper behavior without live services.

**Tech Stack:** Python 3.10+, `requests`, `rasterio`, `numpy`, `Pillow`, `pyproj`, forge3d viewer IPC, `ffmpeg` for MP4 encoding.

---

## File Structure

- Create `examples/khumbu_icefall_sentinel_timelapse.py`: CLI, data discovery, cache paths, raster prep, forge3d frame rendering, MP4 encoding.
- Create `tests/test_khumbu_icefall_sentinel_timelapse.py`: import the example with a stubbed `forge3d`, and test deterministic helpers.
- Modify `docs/examples/index.md`: add the new example to the examples catalog.
- Modify `docs/superpowers/specs/2026-04-25-khumbu-sentinel-timelapse-design.md`: keep the approved one-year winter-to-winter defaults.

---

### Task 1: CLI And Selection Contracts

**Files:**
- Create: `tests/test_khumbu_icefall_sentinel_timelapse.py`
- Create: `examples/khumbu_icefall_sentinel_timelapse.py`

- [ ] **Step 1: Write failing tests for defaults, STAC payload, scene de-duplication, and frame names**

```python
def test_parse_args_defaults_to_recent_winter_year(monkeypatch):
    module = load_module()
    monkeypatch.setattr(sys, "argv", ["khumbu_icefall_sentinel_timelapse.py"])
    args = module.parse_args()
    assert args.start_date == "2025-02-28"
    assert args.end_date == "2026-02-28"
    assert args.max_scenes is None
    assert args.cloud_cover == pytest.approx(35.0)

def test_stac_payload_uses_bbox_dates_and_cloud_filter():
    module = load_module()
    payload = module.build_stac_payload(
        bbox=module.KHUMBU_BBOX,
        start_date="2025-02-28",
        end_date="2026-02-28",
        cloud_cover=35.0,
        limit=100,
    )
    assert payload["collections"] == ["sentinel-2-l2a"]
    assert payload["bbox"] == list(module.KHUMBU_BBOX)
    assert payload["datetime"] == "2025-02-28T00:00:00Z/2026-02-28T23:59:59Z"
    assert payload["query"]["eo:cloud_cover"]["lte"] == 35.0

def test_select_scenes_deduplicates_dates_and_evenly_caps():
    module = load_module()
    features = [
        feature("tile-b", "2025-03-02T04:51:00Z", 12.0, "45RVM"),
        feature("tile-a", "2025-03-02T04:51:00Z", 8.0, "45RVL"),
        feature("tile-c", "2025-04-01T04:51:00Z", 3.0, "45RVL"),
        feature("tile-d", "2025-05-01T04:51:00Z", 9.0, "45RVL"),
        feature("tile-e", "2025-06-01T04:51:00Z", 2.0, "45RVL"),
    ]
    selected = module.select_scenes(features, max_scenes=3)
    assert [scene.date for scene in selected] == ["2025-03-02", "2025-04-01", "2025-06-01"]
    assert selected[0].item_id == "tile-a"

def test_frame_path_is_deterministic(tmp_path):
    module = load_module()
    assert module.frame_path(tmp_path, 7).name == "frame_0007.png"
```

- [ ] **Step 2: Run tests and verify they fail because the module does not exist yet**

Run: `python -m pytest tests/test_khumbu_icefall_sentinel_timelapse.py -q`

Expected: FAIL during import or missing functions.

- [ ] **Step 3: Implement minimal CLI and scene helper code**

Create constants, `SceneItem`, `parse_args()`, `build_stac_payload()`, `select_scenes()`, and `frame_path()` in `examples/khumbu_icefall_sentinel_timelapse.py`.

- [ ] **Step 4: Run tests and verify Task 1 passes**

Run: `python -m pytest tests/test_khumbu_icefall_sentinel_timelapse.py -q`

Expected: Task 1 tests pass.

---

### Task 2: DEM URL And RGB Overlay Helpers

**Files:**
- Modify: `tests/test_khumbu_icefall_sentinel_timelapse.py`
- Modify: `examples/khumbu_icefall_sentinel_timelapse.py`

- [ ] **Step 1: Write failing tests for Copernicus tile naming and RGB normalization**

```python
def test_copernicus_dem_urls_cover_default_bbox():
    module = load_module()
    urls = module.copernicus_dem_urls(module.KHUMBU_BBOX)
    assert len(urls) == 2
    assert urls[0].endswith(
        "/Copernicus_DSM_COG_10_N27_00_E086_00_DEM/"
        "Copernicus_DSM_COG_10_N27_00_E086_00_DEM.tif"
    )
    assert urls[1].endswith(
        "/Copernicus_DSM_COG_10_N28_00_E086_00_DEM/"
        "Copernicus_DSM_COG_10_N28_00_E086_00_DEM.tif"
    )

def test_normalize_rgb_to_rgba_preserves_alpha_and_dtype():
    module = load_module()
    rgb = np.dstack([
        np.linspace(0, 10000, 9, dtype=np.float32).reshape(3, 3),
        np.full((3, 3), 5000, dtype=np.float32),
        np.flipud(np.linspace(0, 10000, 9, dtype=np.float32).reshape(3, 3)),
    ])
    rgba = module.normalize_rgb_to_rgba(rgb)
    assert rgba.shape == (3, 3, 4)
    assert rgba.dtype == np.uint8
    assert np.all(rgba[..., 3] == 255)
    assert rgba[..., :3].max() <= 255
```

- [ ] **Step 2: Run tests and verify they fail because helpers are missing**

Run: `python -m pytest tests/test_khumbu_icefall_sentinel_timelapse.py -q`

Expected: FAIL for missing `copernicus_dem_urls` and `normalize_rgb_to_rgba`.

- [ ] **Step 3: Implement DEM URL and RGB helpers**

Add deterministic Copernicus GLO-30 URL generation, percentile stretch, gamma, and RGBA conversion.

- [ ] **Step 4: Run tests and verify Task 2 passes**

Run: `python -m pytest tests/test_khumbu_icefall_sentinel_timelapse.py -q`

Expected: all current tests pass.

---

### Task 3: Runtime Data And Rendering Pipeline

**Files:**
- Modify: `examples/khumbu_icefall_sentinel_timelapse.py`

- [ ] **Step 1: Add cache and dependency checks**

Implement `require_runtime_dependencies()`, cache directory creation, and clear `SystemExit` messages for missing `requests`, `rasterio`, `PIL`, or native forge3d viewer support.

- [ ] **Step 2: Add STAC search and signing**

Implement `search_sentinel_scenes()` using `build_stac_payload()` and `sign_planetary_computer_url()`.

- [ ] **Step 3: Add DEM preparation**

Implement `prepare_dem()` to read intersecting Copernicus COG windows, merge them, reproject to `EPSG:32645`, and save a cached terrain GeoTIFF.

- [ ] **Step 4: Add Sentinel overlay preparation**

Implement `prepare_overlay()` to read the `visual` asset or `B04/B03/B02`, reproject to the DEM grid, normalize to RGBA, and cache one overlay PNG per selected scene.

- [ ] **Step 5: Add forge3d frame rendering**

Implement `render_frames()` using `open_viewer_async()`, terrain IPC, overlay loading, orbit camera motion, date-label post-processing, and deterministic frame paths.

- [ ] **Step 6: Add MP4 encoding and `main()`**

Implement `encode_mp4()`, `render_timelapse()`, and `main()` with `--preview-only` and `--frames-only`.

- [ ] **Step 7: Run focused tests**

Run: `python -m pytest tests/test_khumbu_icefall_sentinel_timelapse.py -q`

Expected: all tests pass.

---

### Task 4: Documentation

**Files:**
- Modify: `docs/examples/index.md`

- [ ] **Step 1: Add a catalog row**

Add `khumbu_icefall_sentinel_timelapse.py` under "Animation And Camera Automation" with Sentinel-2, Copernicus DEM, forge3d viewer, and MP4 output as the main APIs/workflow.

- [ ] **Step 2: Verify docs text includes the new example**

Run: `rg "khumbu_icefall_sentinel_timelapse" docs/examples/index.md`

Expected: one matching row.

---

### Task 5: Final Verification

**Files:**
- All touched files

- [ ] **Step 1: Run focused unit tests**

Run: `python -m pytest tests/test_khumbu_icefall_sentinel_timelapse.py -q`

Expected: all tests pass.

- [ ] **Step 2: Run import/CLI smoke**

Run: `python examples/khumbu_icefall_sentinel_timelapse.py --help`

Expected: exits 0 and shows the winter-to-winter date defaults.

- [ ] **Step 3: Review git diff**

Run: `git diff -- examples/khumbu_icefall_sentinel_timelapse.py tests/test_khumbu_icefall_sentinel_timelapse.py docs/examples/index.md docs/superpowers/specs/2026-04-25-khumbu-sentinel-timelapse-design.md docs/superpowers/plans/2026-04-25-khumbu-sentinel-timelapse-implementation.md`

Expected: only intended files changed.

