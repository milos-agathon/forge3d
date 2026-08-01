# Humanity Globe PT Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recreate the rotating GPW-v4 population globe video with PROMETHEUS path tracing — real 3D spikes casting soft shadows on the sphere — plus the full data-honesty and cartography fixes, 360 frames @ 30 fps, 1080×1080, seamless loop.

**Architecture:** An orthographic view of a globe is a heightfield (`z = √(R²−r²)` over the disc), so every video frame is traced with the proven legacy PT pipeline (uniform-albedo light field × light-free overlay, Archetype 3 spike recipe). Spikes are stamped as radial *ramps* (thin fins along the screen-projection of the surface normal) so they read as foreshortened needles: dots at disc center, streaks near the limb. The PT frame IS the final view — no viewer pass. Spec: `docs/superpowers/specs/2026-08-01-humanity-globe-pt-design.md`.

**Tech Stack:** Python 3.10+, NumPy, PIL, rasterio, `forge3d.path_tracing.hybrid_render_terrain_reference` (pre-OBLIQUA: no albedo_map / no orthographic camera / no render_terrain_poster), ffmpeg.

## Global Constraints

- Python interpreter: ALWAYS `C:\Users\milos\forge3d\.venv\Scripts\python` (PATH python shadows a stale install).
- Implementation happens in a fresh worktree at `C:/tmp/humanity-globe-pt` branched off `main` (main checkout is dirty). Create with `git -C C:\Users\milos\forge3d worktree add C:/tmp/humanity-globe-pt -b humanity-globe-pt main`. Use `git -C C:/tmp/humanity-globe-pt ...` and absolute paths everywhere (Bash cwd is unstable in worktrees).
- No Rust/WGSL changes anywhere in this plan → no `maturin develop` needed; the main venv's `_forge3d` is current.
- Memory gate: tracked PT bytes ≈ `19·grid² + 366·render_px²` must stay < 512 MiB. Defaults grid 1536², render 1080² → 471 MB. Quality mode 2×2 @1008+48 (render 1104²) → 491 MB.
- PT light compass = `sun_azimuth_deg + 90`. `SUN_AZIMUTH = 225.0` → NW light, SE shadows. Verify with the quadrant probe before any full run (Task 8), never eyeball.
- Animation flicker rule: fixed accumulation count per frame (`min_frames == max_frames == N_ACC`); the variance-gate early exit is a proven flicker source.
- Max-pool spike data, never bilinear (spikes are 1–2 texels).
- Long renders: background, output redirected to a **log file** (never `cmd | tail`), chain with `&&` never `;`.
- New render iterations go to NEW filenames; per-cell npz caches are keyed by all trace parameters.
- If `git check-ignore` says a new file is ignored, commit it with `git add -f` (examples/** is partially gitignored).
- Original script `examples/population_global_gpw/humanity_globe_video.py` stays untouched and runnable.
- The legend/text copy is exact: title `The Humanity Globe: World Population Density`, legend title `Population density (people per km²)`, labels `<1, 1+, 5+, 10+, 50+, 100+, 500+, 1000+`.

## File Structure

- Create `examples/population_global_gpw/humanity_globe_pt_video.py` — the whole PT pipeline (constants, projection/dome math, data prep, spike stamping, trace, modulation, cartographic finish, CLI/runner/probes). Monolithic single-file example, matching repo convention. Imports reusable helpers from sibling `humanity_globe_video.py` (`classify_density`, `roma_class_palette`, `_font`, `frame_path`, `write_frame`, `build_ffmpeg_command`, `encode_mp4`, `GPW_30SEC_URL`, `EXPECTED_30SEC_SHAPE`).
- Create `tests/test_humanity_globe_pt.py` — pure-NumPy unit tests (no GPU, no network, no rasterio requirement at import time).

Data caches reuse the existing GPW 30-arcsec download: pass `--cache-dir C:\Users\milos\forge3d\examples\.cache\humanity_globe` (the worktree's own default path would re-download ~1 GB).

---

### Task 1: Scaffold, projection math, dome heightfield

**Files:**
- Create: `examples/population_global_gpw/humanity_globe_pt_video.py`
- Create: `tests/test_humanity_globe_pt.py`

**Interfaces:**
- Consumes: `humanity_globe_video.sphere_lat_lon` (only in tests, as the round-trip oracle).
- Produces (used by every later task):
  - module constants listed below
  - `view_from_latlon(lat_deg, lon_deg, center_lon) -> (vx, vy, nz, visible)` — arrays float64; unit-sphere view coords, `vx` right/east, `vy` up/north, `nz` toward camera; `visible = nz > 0`.
  - `frame_view_arrays(size: int, center_lon: float) -> dict` with keys `vx, vy, rr, nz, visible, lat, lon` — per-pixel (size×size) arrays; `rr = vx²+vy²` in unit-disc coords.
  - `dome_heightfield(grid_size: int) -> np.ndarray` — (G, G) float32, normalized 0..1 (world height = value × `RELIEF_WORLD`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_humanity_globe_pt.py
import sys
from pathlib import Path

import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "population_global_gpw"
sys.path.insert(0, str(EXAMPLE_DIR))

import humanity_globe_pt_video as pt  # noqa: E402
import humanity_globe_video as hg  # noqa: E402


def test_view_roundtrip_against_original_forward_mapping():
    # The original renderer maps frame pixels -> (lat, lon). Feeding those
    # back through view_from_latlon must reproduce the pixel's view coords.
    size = 128
    center = -100.0
    visible, lat, lon, normals = hg.sphere_lat_lon(size, center)
    vx, vy, nz, vis = pt.view_from_latlon(lat[visible], lon[visible], center)
    assert np.allclose(vx, normals[visible][:, 0], atol=1e-4)
    assert np.allclose(vy, normals[visible][:, 1], atol=1e-4)
    assert np.allclose(nz, normals[visible][:, 2], atol=1e-4)
    assert vis.all()


def test_view_from_latlon_farside_invisible():
    _, _, nz, vis = pt.view_from_latlon(0.0, 0.0, 180.0)
    assert not vis and nz < 0


def test_dome_heightfield_profile():
    grid = pt.dome_heightfield(512)
    assert grid.shape == (512, 512) and grid.dtype == np.float32
    center = grid[256, 256]
    assert abs(center - pt.R_WORLD / pt.RELIEF_WORLD) < 2e-3
    assert grid[0, 0] == 0.0  # corner: outside the disc
    row = grid[256, 256:]  # radially monotone decreasing
    assert (np.diff(row) <= 1e-6).all()


def test_frame_view_arrays_shapes_and_disc():
    fa = pt.frame_view_arrays(200, 30.0)
    assert fa["visible"].shape == (200, 200)
    assert fa["visible"][100, 100]
    assert not fa["visible"][0, 0]
    assert abs(fa["lat"][100, 100]) < 1.0
    dlon = ((fa["lon"][100, 100] - 30.0 + 180.0) % 360.0) - 180.0
    assert abs(dlon) < 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\milos\forge3d\.venv\Scripts\python -m pytest C:/tmp/humanity-globe-pt/tests/test_humanity_globe_pt.py -v --tb=short`
Expected: collection error `ModuleNotFoundError: No module named 'humanity_globe_pt_video'`.

- [ ] **Step 3: Write the scaffold + math**

```python
#!/usr/bin/env python3
"""Path-traced Humanity Globe: rotating GPW-v4 population-density globe.

Each video frame is a PROMETHEUS trace of the orthographic dome-as-heightfield
with population spikes stamped as radial ramps. Spec:
docs/superpowers/specs/2026-08-01-humanity-globe-pt-design.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import humanity_globe_video as hg  # noqa: E402  (sibling module, reused helpers)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "examples" / "out" / "humanity_globe_pt"
DEFAULT_CACHE_DIR = REPO_ROOT / "examples" / ".cache" / "humanity_globe"

# ---- scene geometry -------------------------------------------------------
SPAN_X = 100.0            # world span of the PT grid
DISC_FRAC = 0.88          # globe disc diameter as fraction of the span
R_WORLD = 0.5 * SPAN_X * DISC_FRAC          # 44.0 world units
SPIKE_MIN_WORLD = 0.15    # display floor for the smallest spike
SPIKE_MAX_WORLD = 1.8     # tallest spike (~4% of globe radius)
RELIEF_WORLD = R_WORLD + SPIKE_MAX_WORLD    # tracer exaggeration (0..1 -> world)
HEIGHT_GAMMA = 3.0        # continuous log-height law (WorldPop recipe)
DENSITY_MIN = 1.0         # persons/km2 below which no spike exists
DENSITY_REF_MAX = 10000.0  # density mapped to SPIKE_MAX_WORLD
NS_RAMP = 24              # samples along each spike ramp
STAMP_OFFSETS = ((0, 0), (0, 1), (1, 0))  # texel footprint of a ramp sample
SPIKE_NZ_MIN = 0.05       # drop spikes closer to the limb than this

# ---- PT light (Archetype 3, hard poster light) ---------------------------
CAMERA_FOV_Y = 8.0
CAMERA_MARGIN = 1.06
SUN_AZIMUTH = 225.0       # light compass = azimuth + 90 -> NW light, SE shadows
SUN_ELEVATION = 32.0      # probe-tunable: raise toward 44 if disc dark frac > 0.12
SUN_INTENSITY = 2.6
ENV_INTENSITY = 0.80
PT_ALBEDO = (0.62, 0.62, 0.62)
N_ACC = 800               # fixed accumulation frames (min == max: no flicker gate)

# ---- modulation (Archetype 3 flat-ground register) -----------------------
SHADE_LOW_PCT = 0.5
SHADE_HIGH_PCT = 99.0
SHADE_GAMMA = 0.90
TERRAIN_FLOOR = 0.52
TERRAIN_GAIN = 0.48
OCEAN_FLOOR = 0.42
OCEAN_GAIN = 0.58
OCEAN_BLUR_SIGMA_PX = 8.0
OCEAN_BLUR_MIX = 0.35
LIGHT_TINT_STRENGTH = 1.0
LIGHT_TINT_CLAMP = (0.78, 1.25)
SPIKE_MASK_DILATE_PX = 2

# ---- cartography ---------------------------------------------------------
OCEAN_RGB = (16, 26, 38)
LAND_RGB = (168, 170, 166)
GRATICULE_STEP_DEG = 15.0
GRATICULE_HALF_WIDTH_DEG = 0.22
GRATICULE_LIFT = 14           # additive RGB lift on the dark ocean base
BG_RGB = (8, 10, 14)
RIM_RGB = (140, 170, 220)
RIM_STRENGTH = 0.35
RIM_SIGMA = 0.030         # in unit-disc radius units

# ---- video ---------------------------------------------------------------
DEFAULT_SIZE = 1080
DEFAULT_FPS = 30
DEFAULT_FRAMES = 360      # 360 frames x 1 deg = seamless loop
INITIAL_CENTER_LONGITUDE = -100.0

TITLE_TEXT = "The Humanity Globe: World Population Density"
LEGEND_TITLE = "Population density (people per km²)"
LEGEND_LABELS = ("<1", "1+", "5+", "10+", "50+", "100+", "500+", "1000+")
CAPTION_TEXT = hg.CAPTION_TEXT

AGG_FACTOR_5MIN = 10      # 30 arcsec -> 5 arcmin
SHAPE_5MIN = (2160, 4320)
NE_LAND_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_land.geojson"
)


def view_from_latlon(lat_deg, lon_deg, center_lon):
    """Lat/lon (deg) -> unit-sphere view coords for a globe centered on
    ``center_lon``. Returns (vx, vy, nz, visible): vx east/right, vy
    north/up, nz toward camera. The surface normal IS (vx, vy, nz)."""
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    dlon = np.radians(
        ((np.asarray(lon_deg, dtype=np.float64) - float(center_lon) + 180.0) % 360.0)
        - 180.0
    )
    cos_lat = np.cos(lat)
    vx = cos_lat * np.sin(dlon)
    vy = np.sin(lat)
    nz = cos_lat * np.cos(dlon)
    return vx, vy, nz, nz > 0.0


def frame_view_arrays(size: int, center_lon: float) -> dict:
    """Per-pixel view/geo arrays for a size x size frame of the globe disc."""
    size = int(size)
    c = (np.arange(size, dtype=np.float64) + 0.5) / size
    vx = np.broadcast_to((c[None, :] - 0.5) * 2.0 / DISC_FRAC, (size, size))
    vy = np.broadcast_to((0.5 - c[:, None]) * 2.0 / DISC_FRAC, (size, size))
    rr = vx * vx + vy * vy
    visible = rr <= 1.0
    nz = np.sqrt(np.clip(1.0 - rr, 0.0, None))
    lat = np.degrees(np.arcsin(np.clip(vy, -1.0, 1.0)))
    lon = float(center_lon) + np.degrees(np.arctan2(vx, np.maximum(nz, 1e-9)))
    lon = ((lon + 180.0) % 360.0) - 180.0
    return {"vx": vx, "vy": vy, "rr": rr, "nz": nz, "visible": visible,
            "lat": lat, "lon": lon}


def dome_heightfield(grid_size: int) -> np.ndarray:
    """Normalized (0..1) hemisphere dome over the PT grid; world height =
    value x RELIEF_WORLD. Zero outside the disc (flat apron)."""
    g = int(grid_size)
    c = (np.arange(g, dtype=np.float64) + 0.5) / g
    wx = (c[None, :] - 0.5) * SPAN_X
    wz = (c[:, None] - 0.5) * SPAN_X
    rr = wx * wx + wz * wz
    dome = np.sqrt(np.clip(R_WORLD * R_WORLD - rr, 0.0, None))
    return np.ascontiguousarray(dome / RELIEF_WORLD, dtype=np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\Users\milos\forge3d\.venv\Scripts\python -m pytest C:/tmp/humanity-globe-pt/tests/test_humanity_globe_pt.py -v --tb=short`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git -C C:/tmp/humanity-globe-pt add examples/population_global_gpw/humanity_globe_pt_video.py tests/test_humanity_globe_pt.py
git -C C:/tmp/humanity-globe-pt commit -m "feat(examples): humanity globe PT scaffold - projection + dome heightfield"
```

(If `git -C C:/tmp/humanity-globe-pt check-ignore examples/population_global_gpw/humanity_globe_pt_video.py` prints the path, use `git add -f` instead.)

---

### Task 2: Data prep — 5-min aggregate, NE land mask, equal-area spike sites

**Files:**
- Modify: `examples/population_global_gpw/humanity_globe_pt_video.py` (append)
- Modify: `tests/test_humanity_globe_pt.py` (append)

**Interfaces:**
- Consumes: `hg.GPW_30SEC_URL`, `hg.EXPECTED_30SEC_SHAPE`, constants from Task 1.
- Produces:
  - `aggregate_30sec(source_path: Path, out_path: Path, factor: int) -> None` — mean-aggregates the 30-arcsec GeoTIFF by `factor` (10 → 2160×4320 float32 GeoTIFF).
  - `ensure_data(cache_dir: Path, force: bool = False) -> dict` — returns `{"density": (2160,4320) float32, "land": (2160,4320) bool, "sites": dict}` (downloads GPW + NE GeoJSON on first use, caches npz `globe_pt_data_v1.npz`).
  - `land_mask_5min(cache_dir: Path) -> np.ndarray` — (2160,4320) bool from NE 110m land polygons.
  - `build_spike_sites(density: np.ndarray) -> dict` — keys `lat`, `lon`, `density` (1-D float64/float32 arrays), equal-area thinned, only `density >= DENSITY_MIN`.
  - `spike_height_world(density) -> np.ndarray` — world-unit spike heights, 0 below `DENSITY_MIN`.

- [ ] **Step 1: Write the failing tests** (append to the test file)

```python
def test_spike_sites_equal_area_thinning():
    # Uniform density 100 everywhere: candidate sites per row must shrink
    # ~proportionally to cos(lat), i.e. sites-per-row * block_width == w.
    h, w = 180, 360  # 1-degree synthetic grid
    density = np.full((h, w), 100.0, dtype=np.float32)
    sites = pt.build_spike_sites(density)
    assert (sites["density"] == 100.0).all()
    lat_edges = np.linspace(90.0, -90.0, h + 1)
    for row, (top, bot) in enumerate(zip(lat_edges[:-1], lat_edges[1:])):
        lat_c = 0.5 * (top + bot)
        n_row = int(((sites["lat"] > bot) & (sites["lat"] <= top)).sum())
        width = int(np.ceil(1.0 / max(np.cos(np.radians(lat_c)), 1e-3)))
        expected = int(np.ceil(w / width))
        assert n_row == expected, (row, n_row, expected)


def test_spike_sites_threshold_and_max_pool():
    density = np.zeros((4, 8), dtype=np.float32)
    density[2, 3] = 500.0   # single hot cell near the equator
    density[0, 0] = 0.5     # below DENSITY_MIN -> no site
    sites = pt.build_spike_sites(density)
    assert len(sites["density"]) == 1
    assert sites["density"][0] == 500.0
    # cell (2,3) of a 4x8 grid: lat center 0.. -45+..: rows span 90..-90
    assert -45.0 < sites["lat"][0] < 0.0
    assert -45.0 < sites["lon"][0] < 0.0


def test_spike_height_law():
    s = pt.spike_height_world(np.array([0.0, 0.5, 1.0, 10000.0, 1e6]))
    assert s[0] == 0.0 and s[1] == 0.0
    assert abs(s[2] - pt.SPIKE_MIN_WORLD) < 1e-9      # density==MIN -> floor
    assert abs(s[3] - pt.SPIKE_MAX_WORLD) < 1e-9      # density==REF_MAX -> cap
    assert abs(s[4] - pt.SPIKE_MAX_WORLD) < 1e-9      # clamped above cap
    mids = pt.spike_height_world(np.array([10.0, 100.0, 1000.0]))
    assert (np.diff(mids) > 0).all()                  # strictly increasing
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\milos\forge3d\.venv\Scripts\python -m pytest C:/tmp/humanity-globe-pt/tests/test_humanity_globe_pt.py -v --tb=short -k "spike_sites or height_law"`
Expected: FAIL with `AttributeError: ... has no attribute 'build_spike_sites'`.

- [ ] **Step 3: Implement** (append to the script)

```python
def spike_height_world(density) -> np.ndarray:
    d = np.clip(np.asarray(density, dtype=np.float64), 0.0, None)
    n = np.clip(
        np.log(np.maximum(d, DENSITY_MIN) / DENSITY_MIN)
        / np.log(DENSITY_REF_MAX / DENSITY_MIN),
        0.0,
        1.0,
    )
    s = SPIKE_MIN_WORLD + (SPIKE_MAX_WORLD - SPIKE_MIN_WORLD) * n ** HEIGHT_GAMMA
    return np.where(d >= DENSITY_MIN, s, 0.0)


def build_spike_sites(density: np.ndarray) -> dict:
    """Equal-area spike candidates: per latitude row, MAX-pool density along
    longitude in blocks of ceil(1/cos(lat)) cells so candidate sites are
    ~uniform per unit of Earth surface area (a lat-lon grid alone would
    visually over-spike high latitudes)."""
    h, w = density.shape
    d_lat = 180.0 / h
    d_lon = 360.0 / w
    lats, lons, dens = [], [], []
    for row in range(h):
        lat_c = 90.0 - (row + 0.5) * d_lat
        width = int(np.ceil(1.0 / max(np.cos(np.radians(lat_c)), 1e-3)))
        line = density[row]
        pad = (-w) % width
        padded = np.pad(line, (0, pad), mode="constant") if pad else line
        blocks = padded.reshape(-1, width)
        block_max = blocks.max(axis=1)
        keep = block_max >= DENSITY_MIN
        if not keep.any():
            continue
        arg = blocks.argmax(axis=1)
        cols = np.arange(blocks.shape[0]) * width + arg
        cols = np.minimum(cols, w - 1)[keep]
        lats.append(np.full(cols.shape, lat_c))
        lons.append(-180.0 + (cols + 0.5) * d_lon)
        dens.append(block_max[keep])
    if not lats:
        raise RuntimeError("No spike sites — density grid is empty")
    return {
        "lat": np.concatenate(lats),
        "lon": np.concatenate(lons),
        "density": np.concatenate(dens).astype(np.float32),
    }


def aggregate_30sec(source_path: Path, out_path: Path, factor: int) -> None:
    """Mean-aggregate the 30-arcsec GPW GeoTIFF by ``factor`` (generalizes the
    original script's fixed factor-30 aggregator)."""
    import rasterio
    from rasterio.transform import from_origin

    factor = int(factor)
    out_shape = (hg.EXPECTED_30SEC_SHAPE[0] // factor, hg.EXPECTED_30SEC_SHAPE[1] // factor)
    with rasterio.open(source_path) as src:
        if (src.height, src.width) != hg.EXPECTED_30SEC_SHAPE:
            raise ValueError(
                f"Expected 30-arc-second GPW raster shape {hg.EXPECTED_30SEC_SHAPE}, "
                f"got {(src.height, src.width)}"
            )
        profile = src.profile.copy()
        out = np.zeros(out_shape, dtype=np.float32)
        for out_row in range(out_shape[0]):
            window = rasterio.windows.Window(0, out_row * factor, src.width, factor)
            block = src.read(1, window=window, masked=True).astype(np.float32)
            row = block.reshape(factor, out_shape[1], factor).mean(axis=(0, 2))
            out[out_row, :] = np.asarray(
                row.filled(0.0) if hasattr(row, "filled") else row, dtype=np.float32
            )
    res = 360.0 / out_shape[1]
    profile.update(
        width=out_shape[1], height=out_shape[0], dtype="float32", count=1,
        compress="deflate", transform=from_origin(-180.0, 90.0, res, res),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)


def _ensure_30sec_source(cache_dir: Path, *, force: bool = False) -> Path:
    source_path = cache_dir / "Global_2020_PopulationDensity30sec_GPWv4.tiff"
    if source_path.exists() and not force:
        return source_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    download_path = source_path.with_suffix(source_path.suffix + ".download")
    if download_path.exists():
        download_path.unlink()
    try:
        urllib.request.urlretrieve(hg.GPW_30SEC_URL, download_path)
        download_path.replace(source_path)
    except Exception as exc:
        if download_path.exists():
            download_path.unlink()
        raise SystemExit(
            f"GPW 30-arcsec download failed from {hg.GPW_30SEC_URL}: {exc}"
        ) from exc
    return source_path


def land_mask_5min(cache_dir: Path) -> np.ndarray:
    """Natural Earth 110m land polygons rasterized onto the 5-min grid.
    Plain-JSON geometry -> rasterio.features (no geopandas)."""
    import rasterio.features
    from rasterio.transform import from_origin

    geojson_path = cache_dir / "ne_110m_land.geojson"
    if not geojson_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(NE_LAND_URL, geojson_path)
        except Exception as exc:
            raise SystemExit(
                f"Natural Earth land download failed from {NE_LAND_URL}: {exc}"
            ) from exc
    features = json.loads(geojson_path.read_text(encoding="utf-8"))["features"]
    geoms = [f["geometry"] for f in features if f.get("geometry")]
    res = 360.0 / SHAPE_5MIN[1]
    mask = rasterio.features.rasterize(
        [(g, 1) for g in geoms],
        out_shape=SHAPE_5MIN,
        transform=from_origin(-180.0, 90.0, res, res),
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    return mask.astype(bool)


def ensure_data(cache_dir: Path, *, force: bool = False) -> dict:
    cache_dir = Path(cache_dir)
    bundle = cache_dir / "globe_pt_data_v1.npz"
    if bundle.exists() and not force:
        loaded = np.load(bundle)
        return {
            "density": loaded["density"],
            "land": loaded["land"],
            "sites": {"lat": loaded["site_lat"], "lon": loaded["site_lon"],
                      "density": loaded["site_density"]},
        }
    agg_path = cache_dir / "gpw_v4_density_5min.tif"
    if not agg_path.exists() or force:
        aggregate_30sec(_ensure_30sec_source(cache_dir, force=force), agg_path,
                        AGG_FACTOR_5MIN)
    import rasterio

    with rasterio.open(agg_path) as src:
        density = src.read(1).astype(np.float32)
    density[~np.isfinite(density)] = 0.0
    density = np.clip(density, 0.0, None)
    if density.shape != SHAPE_5MIN:
        raise ValueError(f"Expected 5-min grid {SHAPE_5MIN}, got {density.shape}")
    land = land_mask_5min(cache_dir)
    sites = build_spike_sites(density)
    np.savez_compressed(
        bundle, density=density, land=land, site_lat=sites["lat"],
        site_lon=sites["lon"], site_density=sites["density"],
    )
    return {"density": density, "land": land, "sites": sites}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\Users\milos\forge3d\.venv\Scripts\python -m pytest C:/tmp/humanity-globe-pt/tests/test_humanity_globe_pt.py -v --tb=short`
Expected: 7 passed. (Network/rasterio paths are not unit-tested; they run in Task 8.)

- [ ] **Step 5: Commit**

```bash
git -C C:/tmp/humanity-globe-pt add examples/population_global_gpw/humanity_globe_pt_video.py tests/test_humanity_globe_pt.py
git -C C:/tmp/humanity-globe-pt commit -m "feat(examples): globe PT data prep - 5-min aggregate, NE land mask, equal-area sites"
```

---

### Task 3: Spike ramp stamping (grid heightfield + frame-space class/mask)

**Files:**
- Modify: `examples/population_global_gpw/humanity_globe_pt_video.py` (append)
- Modify: `tests/test_humanity_globe_pt.py` (append)

**Interfaces:**
- Consumes: Task 1 math, Task 2 `spike_height_world`, `hg.classify_density`.
- Produces:
  - `stamp_spikes_grid(height_norm: np.ndarray, sites: dict, center_lon: float) -> np.ndarray` — returns a copy of the (G,G) normalized heightfield with spike ramps max-stamped in.
  - `stamp_spike_pixels(size: int, sites: dict, center_lon: float) -> tuple[np.ndarray, np.ndarray]` — `(class_img uint8 (size,size), spike_mask bool (size,size))` in frame space; class 0 = no spike.

Geometry: a spike is a radial needle of world length `S` at surface point `p` with unit normal `n = (vx, vy, nz)`. Its tip sits at `p·(1 + S/R_WORLD)` — view coords scale by `(1 + S/R_WORLD)`, tip height = `dome_z + S·nz`. The heightfield stamp is a thin ramp (fin) from base to tip ground-track: at parameter `t`, position `base_uv + t·(tip_uv − base_uv)`, normalized height `nz·(R_WORLD + t·S)/RELIEF_WORLD`, combined with `np.maximum.at`. From the nadir camera this reads exactly as the foreshortened needle: a dot at disc center, a streak near the limb.

- [ ] **Step 1: Write the failing tests** (append)

```python
def _one_site(lat, lon, density=10000.0):
    return {"lat": np.array([lat]), "lon": np.array([lon]),
            "density": np.array([density], dtype=np.float32)}


def test_stamp_center_spike_is_a_dot_of_full_height():
    g = 512
    dome = pt.dome_heightfield(g)
    stamped = pt.stamp_spikes_grid(dome, _one_site(0.0, 0.0), center_lon=0.0)
    peak = float(stamped.max())
    expected = (pt.R_WORLD + pt.SPIKE_MAX_WORLD) / pt.RELIEF_WORLD
    assert abs(peak - expected) < 2e-3
    # footprint stays tiny at disc center (a dot, not a streak)
    changed = (stamped - dome) > 1e-4
    assert 0 < changed.sum() <= len(pt.STAMP_OFFSETS) * 4


def test_stamp_off_axis_spike_leaves_a_radial_streak():
    g = 512
    dome = pt.dome_heightfield(g)
    stamped = pt.stamp_spikes_grid(dome, _one_site(0.0, 60.0), center_lon=0.0)
    changed = (stamped - dome) > 1e-4
    ys, xs = np.nonzero(changed)
    # expected ground-track length: S*sin(60deg) world units -> texels
    track_world = pt.SPIKE_MAX_WORLD * math.sin(math.radians(60.0))
    track_px = track_world / pt.SPAN_X * g
    extent = xs.max() - xs.min() + 1
    assert extent >= max(2, int(track_px * 0.5))
    assert ys.max() - ys.min() <= 3  # streak is radial (here: along +x)


def test_stamp_pixels_class_and_mask_match():
    class_img, mask = pt.stamp_spike_pixels(400, _one_site(0.0, 0.0, 2000.0), 0.0)
    assert mask.any()
    assert class_img[mask].max() == hg.classify_density(np.array([2000.0]))[0]
    assert not mask[0, 0]


def test_stamp_farside_site_is_ignored():
    dome = pt.dome_heightfield(256)
    stamped = pt.stamp_spikes_grid(dome, _one_site(0.0, 180.0), center_lon=0.0)
    assert np.array_equal(stamped, dome)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\milos\forge3d\.venv\Scripts\python -m pytest C:/tmp/humanity-globe-pt/tests/test_humanity_globe_pt.py -v --tb=short -k stamp`
Expected: FAIL with `AttributeError: ... 'stamp_spikes_grid'`.

- [ ] **Step 3: Implement** (append)

```python
def _ramp_samples(sites: dict, center_lon: float, size: int):
    """Shared ramp sampling for grid and frame stamping. Returns
    (rows, cols, z_norm, site_index) flat arrays for all ramp samples, in a
    size x size raster whose full width spans SPAN_X world units."""
    vx, vy, nz, vis = view_from_latlon(sites["lat"], sites["lon"], center_lon)
    keep = vis & (nz > SPIKE_NZ_MIN)
    if not keep.any():
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=np.float64), empty
    vx, vy, nz = vx[keep], vy[keep], nz[keep]
    s_w = spike_height_world(np.asarray(sites["density"], dtype=np.float64)[keep])
    scale = 1.0 + s_w / R_WORLD
    # base/tip in raster uv (0..1); v axis runs north-up -> row-down
    u0 = 0.5 + 0.5 * DISC_FRAC * vx
    v0 = 0.5 - 0.5 * DISC_FRAC * vy
    u1 = 0.5 + 0.5 * DISC_FRAC * vx * scale
    v1 = 0.5 - 0.5 * DISC_FRAC * vy * scale
    t = np.linspace(0.0, 1.0, NS_RAMP, dtype=np.float64)[:, None]
    us = u0[None, :] + t * (u1 - u0)[None, :]
    vs = v0[None, :] + t * (v1 - v0)[None, :]
    z = nz[None, :] * (R_WORLD + t * s_w[None, :]) / RELIEF_WORLD
    cols = np.clip((us * size).astype(np.int64), 0, size - 1)
    rows = np.clip((vs * size).astype(np.int64), 0, size - 1)
    idx = np.broadcast_to(np.nonzero(keep)[0][None, :], rows.shape)
    return rows.ravel(), cols.ravel(), z.ravel(), idx.ravel()


def stamp_spikes_grid(height_norm: np.ndarray, sites: dict, center_lon: float) -> np.ndarray:
    out = height_norm.copy()
    g = out.shape[0]
    rows, cols, z, _ = _ramp_samples(sites, center_lon, g)
    for dr, dc in STAMP_OFFSETS:
        r = np.clip(rows + dr, 0, g - 1)
        c = np.clip(cols + dc, 0, g - 1)
        np.maximum.at(out, (r, c), z.astype(np.float32))
    return out


def stamp_spike_pixels(size: int, sites: dict, center_lon: float):
    class_img = np.zeros((size, size), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=bool)
    rows, cols, _, idx = _ramp_samples(sites, center_lon, size)
    if rows.size:
        classes = hg.classify_density(np.asarray(sites["density"], dtype=np.float32))
        sample_cls = classes[idx]
        for dr, dc in STAMP_OFFSETS:
            r = np.clip(rows + dr, 0, size - 1)
            c = np.clip(cols + dc, 0, size - 1)
            np.maximum.at(class_img, (r, c), sample_cls)
            mask[r, c] = True
    return class_img, mask
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\Users\milos\forge3d\.venv\Scripts\python -m pytest C:/tmp/humanity-globe-pt/tests/test_humanity_globe_pt.py -v --tb=short`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git -C C:/tmp/humanity-globe-pt add examples/population_global_gpw/humanity_globe_pt_video.py tests/test_humanity_globe_pt.py
git -C C:/tmp/humanity-globe-pt commit -m "feat(examples): globe PT spike ramp stamping (grid + frame space)"
```

---

### Task 4: PT trace per frame — camera, budget gate, feathered tiles, cell cache

**Files:**
- Modify: `examples/population_global_gpw/humanity_globe_pt_video.py` (append)
- Modify: `tests/test_humanity_globe_pt.py` (append)

**Interfaces:**
- Consumes: `forge3d.path_tracing.hybrid_render_terrain_reference` (imported inside `_pt_pass`), Task 1 constants.
- Produces:
  - `camera_for_grid(*, tiles=1, tx=0, ty=0, expand=1.0) -> dict` — nadir quasi-ortho camera (square grid; Swiss-v6 pattern with `expand` for feather margins).
  - `cell_weights(size: int, margin: int) -> np.ndarray` — 1-D trapezoid weights.
  - `budget_bytes(grid: int, frame: int, margin: int, tiles: int) -> int` and `check_budget(...)` which raises `SystemExit` over 512 MiB.
  - `trace_params_key(args, frame_index: int) -> str` — sha1-fingerprint over every trace-relevant parameter.
  - `trace_frame(height_grid: np.ndarray, frame_index: int, args, cache_dir: Path) -> tuple[np.ndarray, np.ndarray]` — `(rgb float32 (F,F,3) with NaN misses, hit bool (F,F))`, F = `args.tiles * args.frame`; per-cell npz cache; fixed accumulation (`min_frames == max_frames == args.acc_frames`).

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_camera_matches_template_at_one_tile():
    cam = pt.camera_for_grid()
    half = 0.5 * pt.SPAN_X * pt.CAMERA_MARGIN  # 53.0
    dist = half / math.tan(math.radians(pt.CAMERA_FOV_Y / 2.0))
    assert cam["origin"] == (0.0, dist, 0.0)
    assert cam["look_at"] == (0.0, 0.0, 0.0)
    assert cam["up"] == (0.0, 0.0, -1.0)
    assert abs(cam["fov_y"] - pt.CAMERA_FOV_Y) < 1e-9


def test_camera_tiles_partition_and_expand():
    c00 = pt.camera_for_grid(tiles=2, tx=0, ty=0)
    c11 = pt.camera_for_grid(tiles=2, tx=1, ty=1)
    half = 0.5 * pt.SPAN_X * pt.CAMERA_MARGIN
    assert abs(c00["origin"][0] + half / 2.0) < 1e-9
    assert abs(c11["origin"][0] - half / 2.0) < 1e-9
    wide = pt.camera_for_grid(tiles=2, expand=1.2)
    assert wide["fov_y"] > c00["fov_y"]


def test_cell_weights_trapezoid():
    w = pt.cell_weights(10, 3)
    assert w[5] == 1.0 and 0.0 < w[0] < 1.0
    assert np.allclose(w, w[::-1])
    assert (pt.cell_weights(8, 0) == 1.0).all()


def test_budget_gate():
    assert pt.budget_bytes(1536, 1080, 0, 1) == 19 * 1536**2 + 366 * 1080**2
    assert pt.budget_bytes(1536, 1008, 48, 2) == 19 * 1536**2 + 366 * 1104**2
    pt.check_budget(1536, 1080, 0, 1)  # must not raise (471 MB)
    try:
        pt.check_budget(4096, 1248, 0, 1)  # 770 MB — must refuse
        raise AssertionError("check_budget accepted an over-budget config")
    except SystemExit:
        pass


def test_trace_params_key_changes_with_params():
    class A:  # minimal args stand-in
        grid = 1536; frame = 1080; tiles = 1; margin = 48
        acc_frames = 800; spp = 1; seed = 7
    k1 = pt.trace_params_key(A, 0)
    A.acc_frames = 900
    k2 = pt.trace_params_key(A, 0)
    A.acc_frames = 800
    k3 = pt.trace_params_key(A, 1)
    assert k1 != k2 and k1 != k3 and len(k1) >= 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\milos\forge3d\.venv\Scripts\python -m pytest C:/tmp/humanity-globe-pt/tests/test_humanity_globe_pt.py -v --tb=short -k "camera or weights or budget or params_key"`
Expected: FAIL with `AttributeError: ... 'camera_for_grid'`.

- [ ] **Step 3: Implement** (append; feather-blend copied from the Swiss v6 template)

```python
MEMORY_GATE_BYTES = 512 * 2**20
BYTES_PER_TEXEL = 19
BYTES_PER_FRAME_PX = 366


def camera_for_grid(*, tiles: int = 1, tx: int = 0, ty: int = 0, expand: float = 1.0) -> dict:
    """Nadir quasi-ortho camera over one cell of the square PT grid
    (Swiss v6 pattern; ``expand`` > 1 widens framing for feather margins)."""
    half_extent = 0.5 * SPAN_X * CAMERA_MARGIN
    distance = half_extent / math.tan(math.radians(CAMERA_FOV_Y / 2.0))
    tile_half = half_extent / tiles
    center_x = -half_extent + (tx + 0.5) * 2.0 * tile_half
    center_z = -half_extent + (ty + 0.5) * 2.0 * tile_half
    fov_y = math.degrees(
        2.0 * math.atan(math.tan(math.radians(CAMERA_FOV_Y / 2.0)) / tiles * expand)
    )
    return {"origin": (center_x, distance, center_z),
            "look_at": (center_x, 0.0, center_z),
            "up": (0.0, 0.0, -1.0), "fov_y": fov_y, "exposure": 1.0}


def cell_weights(size: int, margin: int) -> np.ndarray:
    """1D trapezoid: 1.0 in the cell interior, linear 0..1 across the margin."""
    if margin <= 0:
        return np.ones(size, dtype=np.float32)
    i = np.arange(size, dtype=np.float32) + 0.5
    return np.minimum(1.0, np.minimum(i / margin, (size - i) / margin)).astype(np.float32)


def budget_bytes(grid: int, frame: int, margin: int, tiles: int) -> int:
    render = frame + (2 * margin if tiles > 1 else 0)
    return BYTES_PER_TEXEL * grid * grid + BYTES_PER_FRAME_PX * render * render


def check_budget(grid: int, frame: int, margin: int, tiles: int) -> None:
    total = budget_bytes(grid, frame, margin, tiles)
    if total >= MEMORY_GATE_BYTES:
        raise SystemExit(
            f"PT budget {total / 2**20:.0f} MiB exceeds the 512 MiB gate "
            f"(grid {grid}, frame {frame}, margin {margin}, tiles {tiles}). "
            "Shrink --frame or --grid."
        )


def _neutral_sky_env(height: int = 32, width: int = 64) -> np.ndarray:
    # Cool neutral sky: blue-ish zenith is what makes shadows go cool under
    # the hard poster light (Swiss register), warm-white horizon, dim ground.
    zenith = np.array([0.55, 0.65, 0.95], dtype=np.float32)
    horizon = np.array([0.90, 0.92, 0.98], dtype=np.float32)
    ground = np.array([0.35, 0.33, 0.30], dtype=np.float32)
    rows = np.linspace(1.0, -1.0, height, dtype=np.float32)
    env_rows = np.empty((height, 3), dtype=np.float32)
    up = np.clip(rows, 0.0, 1.0)[:, None] ** 0.65
    env_rows[:] = horizon[None, :] * (1.0 - up) + zenith[None, :] * up
    below = rows < 0.0
    down = np.clip(-rows[below], 0.0, 1.0)[:, None] ** 0.5
    env_rows[below] = horizon[None, :] * (1.0 - down) + ground[None, :] * down
    return np.repeat(env_rows[:, None, :], width, axis=1)


def _pt_pass(dem_grid, render_px, camera, args, *, label="full"):
    from forge3d.path_tracing import hybrid_render_terrain_reference

    rows, cols = dem_grid.shape
    spacing = SPAN_X / (cols - 1)
    out = hybrid_render_terrain_reference(
        dem_grid, render_px, render_px, camera,
        spacing=(spacing, spacing),
        exaggeration=RELIEF_WORLD,
        albedo=PT_ALBEDO,
        sun_azimuth_deg=SUN_AZIMUTH,
        sun_elevation_deg=float(args.sun_elevation),
        sun_intensity=SUN_INTENSITY,
        env_map=_neutral_sky_env(),
        env_intensity=ENV_INTENSITY,
        spp=int(args.spp),
        max_frames=int(args.acc_frames),   # fixed accumulation:
        min_frames=int(args.acc_frames),   # min == max -> no variance-gate flicker
        variance_threshold=1e-12,
        seed=int(args.seed),
    )
    print(
        f"[PT:{label}] frames={out['frames']} variance={out['variance']:.3e} "
        f"peak_host_visible={out.get('peak_host_visible_bytes', 0) / 2**20:.1f} MiB",
        flush=True,
    )
    rgba = out["rgba"].astype(np.float32) / 255.0
    hit = np.isfinite(out["depth"])
    rgb = np.where(hit[:, :, None], rgba[:, :, :3], np.nan).astype(np.float32)
    return rgb, hit


def trace_params_key(args, frame_index: int) -> str:
    payload = json.dumps({
        "grid": int(args.grid), "frame": int(args.frame), "tiles": int(args.tiles),
        "margin": int(args.margin), "acc": int(args.acc_frames),
        "spp": int(args.spp), "seed": int(args.seed),
        "sun": (SUN_AZIMUTH, float(getattr(args, "sun_elevation", SUN_ELEVATION)),
                SUN_INTENSITY, ENV_INTENSITY),
        "relief": (RELIEF_WORLD, DISC_FRAC, SPIKE_MIN_WORLD, SPIKE_MAX_WORLD,
                   HEIGHT_GAMMA, DENSITY_REF_MAX, NS_RAMP),
        "frame_index": int(frame_index),
        "data": "globe_pt_data_v1",
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def trace_frame(height_grid: np.ndarray, frame_index: int, args, cache_dir: Path):
    """Feather-blended 2x2 (or single-tile) PT light field for one video frame,
    with per-cell npz caching (GPU hiccups resume instead of losing the run)."""
    check_budget(int(args.grid), int(args.frame), int(args.margin), int(args.tiles))
    tiles = max(1, int(args.tiles))
    frame = int(args.frame)
    margin = int(args.margin) if tiles > 1 else 0
    key = trace_params_key(args, frame_index)
    cell_dir = Path(cache_dir) / "pt_cells" / f"f{frame_index:04d}_{key}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    if tiles == 1:
        cache = cell_dir / "cell_0_0.npz"
        if cache.is_file():
            loaded = np.load(cache)
            return loaded["rgb"], loaded["hit"]
        rgb, hit = _pt_pass(height_grid, frame, camera_for_grid(), args,
                            label=f"frame {frame_index}")
        if not hit.any():
            raise RuntimeError("Path tracer produced no hits — check camera framing")
        np.savez_compressed(cache, rgb=rgb, hit=hit)
        return rgb, hit
    render = frame + 2 * margin
    expand = render / float(frame)
    field = tiles * frame
    rgb_acc = np.zeros((field, field, 3), dtype=np.float64)
    w_acc = np.zeros((field, field), dtype=np.float64)
    w1d = cell_weights(render, margin)
    w2d = np.minimum(w1d[:, None], w1d[None, :]).astype(np.float32)
    for ty in range(tiles):
        for tx in range(tiles):
            cache = cell_dir / f"cell_{tx}_{ty}.npz"
            if cache.is_file():
                loaded = np.load(cache)
                tile_rgb, tile_hit = loaded["rgb"], loaded["hit"]
                print(f"[PT:f{frame_index} tile {tx},{ty}] cached", flush=True)
            else:
                camera = camera_for_grid(tiles=tiles, tx=tx, ty=ty, expand=expand)
                tile_rgb, tile_hit = _pt_pass(
                    height_grid, render, camera, args,
                    label=f"f{frame_index} tile {tx},{ty}")
                np.savez_compressed(cache, rgb=tile_rgb, hit=tile_hit)
            w_tile = np.where(tile_hit, w2d, 0.0).astype(np.float32)
            y0 = ty * frame - margin
            x0 = tx * frame - margin
            sy0, sx0 = max(0, -y0), max(0, -x0)
            dy0, dx0 = max(0, y0), max(0, x0)
            sy1 = render - max(0, y0 + render - field)
            sx1 = render - max(0, x0 + render - field)
            src_rgb = np.nan_to_num(tile_rgb[sy0:sy1, sx0:sx1], nan=0.0)
            src_w = w_tile[sy0:sy1, sx0:sx1]
            rgb_acc[dy0:dy0 + sy1 - sy0, dx0:dx0 + sx1 - sx0] += src_rgb * src_w[:, :, None]
            w_acc[dy0:dy0 + sy1 - sy0, dx0:dx0 + sx1 - sx0] += src_w
    hit = w_acc > 1e-6
    rgb = np.full((field, field, 3), np.nan, dtype=np.float32)
    rgb[hit] = (rgb_acc[hit] / w_acc[hit][:, None]).astype(np.float32)
    if not hit.any():
        raise RuntimeError("Path tracer produced no hits — check camera framing")
    return rgb, hit
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\Users\milos\forge3d\.venv\Scripts\python -m pytest C:/tmp/humanity-globe-pt/tests/test_humanity_globe_pt.py -v --tb=short`
Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
git -C C:/tmp/humanity-globe-pt add examples/population_global_gpw/humanity_globe_pt_video.py tests/test_humanity_globe_pt.py
git -C C:/tmp/humanity-globe-pt commit -m "feat(examples): globe PT trace - camera, 512MiB gate, feathered tiles, cell cache"
```

---

### Task 5: Shade/tint extraction + overlay modulation

**Files:**
- Modify: `examples/population_global_gpw/humanity_globe_pt_video.py` (append)
- Modify: `tests/test_humanity_globe_pt.py` (append)

**Interfaces:**
- Consumes: Task 4 outputs `(rgb, hit)`; constants from Task 1.
- Produces:
  - `shade_and_tint(rgb_field, hit, out_size: int) -> tuple[np.ndarray, np.ndarray]` — `(shade (S,S) float32 0..1, tint (S,S,3) float32)` resized to `out_size`. Full-frame (no bbox crop — the PT frame IS the final view; the flat apron around the disc is masked later by the disc alpha).
  - `modulate_overlay(overlay_rgba: np.ndarray, shade, tint, spike_mask, ocean_mask) -> np.ndarray` — uint8 RGBA; land: `TERRAIN_FLOOR + TERRAIN_GAIN·shade`; ocean: own floor/gain over `(1−OCEAN_BLUR_MIX)·shade + OCEAN_BLUR_MIX·blur(shade)` (calms PT noise on the dark ocean but keeps spike shadows falling on the sea — a heavy full blur would erase them); spike pixels exempt (scale=1, tint=1).

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_shade_and_tint_bounds_and_shapes():
    rng = np.random.default_rng(0)
    rgb = rng.uniform(0.05, 0.9, (64, 64, 3)).astype(np.float32)
    hit = np.ones((64, 64), dtype=bool)
    shade, tint = pt.shade_and_tint(rgb, hit, 128)
    assert shade.shape == (128, 128) and tint.shape == (128, 128, 3)
    assert 0.0 <= shade.min() and shade.max() <= 1.0
    assert tint.min() >= pt.LIGHT_TINT_CLAMP[0] - 1e-6
    assert tint.max() <= pt.LIGHT_TINT_CLAMP[1] + 1e-6


def test_modulate_respects_masks():
    s = 32
    overlay = np.full((s, s, 4), 200, dtype=np.uint8)
    shade = np.zeros((s, s), dtype=np.float32)      # full shadow everywhere
    tint = np.ones((s, s, 3), dtype=np.float32)
    spike = np.zeros((s, s), dtype=bool); spike[4, 4] = True
    ocean = np.zeros((s, s), dtype=bool); ocean[10:, :] = True
    out = pt.modulate_overlay(overlay, shade, tint, spike, ocean)
    land_px = out[2, 2, 0] / 200.0
    ocean_px = out[20, 20, 0] / 200.0
    assert abs(land_px - pt.TERRAIN_FLOOR) < 0.02    # land floor in shadow
    assert abs(ocean_px - pt.OCEAN_FLOOR) < 0.02     # ocean floor in shadow
    assert out[4, 4, 0] == 200                       # spike exempt
    assert (out[:, :, 3] == overlay[:, :, 3]).all()  # alpha untouched
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\milos\forge3d\.venv\Scripts\python -m pytest C:/tmp/humanity-globe-pt/tests/test_humanity_globe_pt.py -v --tb=short -k "shade_and_tint or modulate"`
Expected: FAIL with `AttributeError: ... 'shade_and_tint'`.

- [ ] **Step 3: Implement** (append; full-frame variant of the SE-Europe template)

```python
def shade_and_tint(rgb_field, hit, out_size: int):
    lum = np.where(hit, rgb_field @ np.array([0.2126, 0.7152, 0.0722],
                                             dtype=np.float32), np.nan)
    finite = np.isfinite(lum)
    lum = np.where(finite, lum, float(np.nanmedian(lum)))
    low = float(np.percentile(lum[finite], SHADE_LOW_PCT))
    high = float(np.percentile(lum[finite], SHADE_HIGH_PCT))
    shade = np.clip((lum - low) / max(high - low, 1e-6), 0.0, 1.0)
    shade = np.power(shade, SHADE_GAMMA, dtype=np.float32)

    tint = np.nan_to_num(rgb_field, nan=1.0) / np.maximum(lum, 1e-4)[:, :, None]
    tint = np.where(finite[:, :, None], tint, 1.0)
    lit = shade > 0.75
    if lit.any():
        anchor = np.median(tint[lit], axis=0)
        tint = tint / np.maximum(anchor[None, None, :], 1e-4)
    tint = np.clip(tint, LIGHT_TINT_CLAMP[0], LIGHT_TINT_CLAMP[1]).astype(np.float32)

    def _resize(field: np.ndarray) -> np.ndarray:
        img = Image.fromarray(field, mode="F").resize(
            (out_size, out_size), Image.Resampling.BICUBIC)
        return np.asarray(img, dtype=np.float32)

    shade_hi = np.clip(_resize(shade), 0.0, 1.0)
    tint_hi = np.stack(
        [_resize(np.ascontiguousarray(tint[:, :, c])) for c in range(3)], axis=-1)
    return shade_hi, np.clip(tint_hi, LIGHT_TINT_CLAMP[0], LIGHT_TINT_CLAMP[1])


def modulate_overlay(overlay_rgba, shade, tint, spike_mask, ocean_mask):
    overlay = np.asarray(overlay_rgba, dtype=np.uint8).copy()
    rgb = overlay[:, :, :3].astype(np.float32) / 255.0
    blur = np.asarray(
        Image.fromarray(shade, mode="F").filter(
            ImageFilter.GaussianBlur(OCEAN_BLUR_SIGMA_PX)),
        dtype=np.float32)
    ocean_shade = (1.0 - OCEAN_BLUR_MIX) * shade + OCEAN_BLUR_MIX * blur
    scale = np.where(
        ocean_mask, OCEAN_FLOOR + OCEAN_GAIN * ocean_shade,
        TERRAIN_FLOOR + TERRAIN_GAIN * shade)
    tint_mix = 1.0 + LIGHT_TINT_STRENGTH * (tint - 1.0)
    scale = np.where(spike_mask, 1.0, scale)
    tint_mix = np.where(spike_mask[:, :, None], 1.0, tint_mix)
    rgb = np.clip(rgb * scale[:, :, None] * tint_mix, 0.0, 1.0)
    overlay[:, :, :3] = np.round(rgb * 255.0).astype(np.uint8)
    return overlay
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\Users\milos\forge3d\.venv\Scripts\python -m pytest C:/tmp/humanity-globe-pt/tests/test_humanity_globe_pt.py -v --tb=short`
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
git -C C:/tmp/humanity-globe-pt add examples/population_global_gpw/humanity_globe_pt_video.py tests/test_humanity_globe_pt.py
git -C C:/tmp/humanity-globe-pt commit -m "feat(examples): globe PT shade/tint + masked overlay modulation"
```

---

### Task 6: Overlay build, graticule, rim/background, downscale, corrected text

**Files:**
- Modify: `examples/population_global_gpw/humanity_globe_pt_video.py` (append)
- Modify: `tests/test_humanity_globe_pt.py` (append)

**Interfaces:**
- Consumes: `frame_view_arrays`, `stamp_spike_pixels`, `hg.roma_class_palette`, `hg._font`, Task 2 data dict.
- Produces:
  - `build_overlay(size: int, center_lon: float, data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — `(overlay_rgba uint8 (S,S,4), spike_mask bool, ocean_mask bool)`. Ocean/land base from the NE mask (nearest-sampled per pixel), graticule pre-baked as lightened ocean (so it modulates naturally), spike ramp pixels in Roma class colors, alpha = 255 inside the disc else 0. `spike_mask` is MaxFilter-dilated by `SPIKE_MASK_DILATE_PX`.
  - `finish_frame(modulated_rgba: np.ndarray, out_size: int) -> np.ndarray` — composites over `BG_RGB` using the disc alpha, adds the atmospheric rim glow, LANCZOS-downscales to `out_size`, returns uint8 RGBA (opaque).
  - `compose_text(frame: np.ndarray) -> np.ndarray` — title/legend/caption with the corrected copy (Global Constraints), Roma swatches, `hg._font`.

- [ ] **Step 1: Write the failing tests** (append)

```python
def _tiny_data():
    density = np.zeros((36, 72), dtype=np.float32)  # 5-deg synthetic grid
    density[17, 36] = 1200.0                        # near (0N, 2.5E)
    land = np.zeros((36, 72), dtype=bool)
    land[17, 36] = True
    sites = pt.build_spike_sites(density)
    return {"density": density, "land": land, "sites": sites}


def test_build_overlay_layers():
    data = _tiny_data()
    overlay, spike_mask, ocean_mask = pt.build_overlay(300, 0.0, data)
    assert overlay.shape == (300, 300, 4)
    assert overlay[0, 0, 3] == 0                    # corner: outside disc
    assert overlay[150, 150, 3] == 255              # center: on disc
    assert spike_mask.any() and ocean_mask.any()
    assert not (spike_mask & ~pt.frame_view_arrays(300, 0.0)["visible"]).any()
    # spike pixels carry a Roma class color, not the ocean base
    ys, xs = np.nonzero(spike_mask & (overlay[:, :, 3] == 255))
    palette = hg.roma_class_palette()
    cls = hg.classify_density(np.array([1200.0]))[0]
    assert (overlay[ys, xs, :3] == palette[cls]).all(axis=1).any()


def test_graticule_on_ocean_only():
    data = _tiny_data()
    overlay, spike_mask, ocean_mask = pt.build_overlay(400, 0.0, data)
    base = np.array(pt.OCEAN_RGB, dtype=np.int32)
    lighter = (overlay[:, :, :3].astype(np.int32) > base + 8).all(axis=2)
    grat = lighter & ocean_mask & ~spike_mask
    assert grat.any()                                # equator+meridian visible
    land_px = ~ocean_mask & (overlay[:, :, 3] == 255) & ~spike_mask
    land_colors = np.unique(overlay[land_px][:, :3], axis=0)
    assert len(land_colors) <= 1                     # land stays flat base


def test_finish_frame_background_and_size():
    rgba = np.zeros((200, 200, 4), dtype=np.uint8)
    fa = pt.frame_view_arrays(200, 0.0)
    rgba[:, :, 3] = np.where(fa["visible"], 255, 0)
    rgba[:, :, :3] = 90
    out = pt.finish_frame(rgba, 100)
    assert out.shape == (100, 100, 4)
    assert tuple(out[0, 0, :3]) == pt.BG_RGB         # corner = background
    assert out[50, 50, 0] > 60                       # disc content survives


def test_compose_text_legend_copy():
    frame = np.full((540, 540, 4), 10, dtype=np.uint8)
    out = pt.compose_text(frame)
    assert out.shape == frame.shape
    assert pt.LEGEND_TITLE == "Population density (people per km²)"
    assert "30 km" not in pt.LEGEND_TITLE
    assert pt.LEGEND_LABELS[0] == "<1" and pt.LEGEND_LABELS[-1] == "1000+"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\milos\forge3d\.venv\Scripts\python -m pytest C:/tmp/humanity-globe-pt/tests/test_humanity_globe_pt.py -v --tb=short -k "overlay_layers or graticule or finish_frame or compose_text"`
Expected: FAIL with `AttributeError: ... 'build_overlay'`.

- [ ] **Step 3: Implement** (append)

```python
def _sample_grid_nearest(grid: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    row = np.clip(((90.0 - lat) / 180.0 * h).astype(np.int64), 0, h - 1)
    col = np.clip(((lon + 180.0) / 360.0 * w).astype(np.int64), 0, w - 1)
    return grid[row, col]


def build_overlay(size: int, center_lon: float, data: dict):
    fa = frame_view_arrays(size, center_lon)
    visible = fa["visible"]
    land = np.zeros((size, size), dtype=bool)
    land[visible] = _sample_grid_nearest(
        data["land"].astype(np.uint8), fa["lat"][visible], fa["lon"][visible]) > 0
    ocean_mask = visible & ~land

    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[ocean_mask] = OCEAN_RGB
    rgb[land] = LAND_RGB

    # graticule: pre-baked lightened ocean so it modulates with the light
    half = GRATICULE_HALF_WIDTH_DEG
    lat_frac = np.abs(((fa["lat"] + 0.5 * GRATICULE_STEP_DEG) % GRATICULE_STEP_DEG)
                      - 0.5 * GRATICULE_STEP_DEG)
    lon_frac = np.abs(((fa["lon"] + 0.5 * GRATICULE_STEP_DEG) % GRATICULE_STEP_DEG)
                      - 0.5 * GRATICULE_STEP_DEG)
    grat = ocean_mask & (fa["nz"] > 0.2) & ((lat_frac < half) | (lon_frac < half * np.maximum(np.cos(np.radians(fa["lat"])), 0.2)))
    rgb[grat] = np.clip(
        np.array(OCEAN_RGB, dtype=np.int32) + GRATICULE_LIFT, 0, 255
    ).astype(np.uint8)

    class_img, spike_mask = stamp_spike_pixels(size, data["sites"], center_lon)
    palette = hg.roma_class_palette()
    spike_px = spike_mask & visible & (class_img > 0)
    rgb[spike_px] = palette[class_img[spike_px]]

    dil = Image.fromarray((spike_px * 255).astype(np.uint8), mode="L").filter(
        ImageFilter.MaxFilter(2 * SPIKE_MASK_DILATE_PX + 1))
    spike_mask_out = (np.asarray(dil) > 127) & visible

    alpha = np.where(visible, 255, 0).astype(np.uint8)
    return np.dstack([rgb, alpha]), spike_mask_out, ocean_mask


def finish_frame(modulated_rgba: np.ndarray, out_size: int) -> np.ndarray:
    size = modulated_rgba.shape[0]
    fa = frame_view_arrays(size, 0.0)  # rim geometry is rotation-invariant
    alpha = modulated_rgba[:, :, 3].astype(np.float32) / 255.0
    rgb = modulated_rgba[:, :, :3].astype(np.float32)
    bg = np.array(BG_RGB, dtype=np.float32)
    out = rgb * alpha[:, :, None] + bg[None, None, :] * (1.0 - alpha[:, :, None])
    rd = np.sqrt(fa["rr"])
    glow = np.exp(-((rd - 1.0) / RIM_SIGMA) ** 2) * (rd >= 1.0)
    out = np.clip(
        out + np.array(RIM_RGB, dtype=np.float32)[None, None, :]
        * (RIM_STRENGTH * glow)[:, :, None], 0.0, 255.0)
    img = Image.fromarray(
        np.dstack([out.astype(np.uint8),
                   np.full((size, size, 1), 255, dtype=np.uint8)]), mode="RGBA")
    if out_size != size:
        img = img.resize((out_size, out_size), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def compose_text(frame: np.ndarray) -> np.ndarray:
    image = Image.fromarray(frame, mode="RGBA")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    title_font = hg._font(max(16, int(w * hg.TITLE_FONT_SCALE)))
    label_font = hg._font(max(12, int(w * hg.LEGEND_LABEL_FONT_SCALE)), bold=True)
    value_font = hg._font(max(9, int(w * hg.LEGEND_VALUE_FONT_SCALE)))
    small_font = hg._font(max(8, int(w * hg.CAPTION_FONT_SCALE)))
    draw.text((int(w * 0.02), int(h * 0.018)), TITLE_TEXT,
              fill=(238, 238, 238, 255), font=title_font)
    legend_x = int(w * 0.078)
    legend_y = int(h * 0.872)
    cell_w = max(10, int(w * 0.105))
    cell_h = max(6, int(h * 0.05))
    draw.text((legend_x, legend_y - int(h * 0.042)), LEGEND_TITLE,
              fill=(245, 245, 245, 255), font=label_font)
    palette = hg.roma_class_palette()
    for idx, label in enumerate(LEGEND_LABELS):
        x0 = legend_x + idx * cell_w
        color = tuple(int(v) for v in palette[idx]) + (255,)
        draw.rectangle((x0, legend_y, x0 + cell_w, legend_y + cell_h), fill=color)
        draw.text((x0 + int(cell_w * 0.30), legend_y + int(cell_h * 0.18)),
                  label, fill=(0, 0, 0, 255), font=value_font)
    draw.text((int(w * 0.02), int(h * 0.956)), CAPTION_TEXT,
              fill=(220, 220, 220, 255), font=small_font)
    return np.asarray(image, dtype=np.uint8)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\Users\milos\forge3d\.venv\Scripts\python -m pytest C:/tmp/humanity-globe-pt/tests/test_humanity_globe_pt.py -v --tb=short`
Expected: 22 passed.

- [ ] **Step 5: Commit**

```bash
git -C C:/tmp/humanity-globe-pt add examples/population_global_gpw/humanity_globe_pt_video.py tests/test_humanity_globe_pt.py
git -C C:/tmp/humanity-globe-pt commit -m "feat(examples): globe PT overlay, graticule, rim, corrected legend copy"
```

---

### Task 7: CLI, single-frame renderer, subprocess runner, ffmpeg, probes

**Files:**
- Modify: `examples/population_global_gpw/humanity_globe_pt_video.py` (append)
- Modify: `tests/test_humanity_globe_pt.py` (append)

**Interfaces:**
- Consumes: everything above; `hg.frame_path`, `hg.write_frame`, `hg.encode_mp4`.
- Produces:
  - `parse_args(argv=None) -> argparse.Namespace` — flags: `--output-dir`, `--cache-dir`, `--size` (1080), `--fps` (30), `--frames` (360), `--grid` (1536), `--frame` (1080), `--tiles` (1), `--margin` (48), `--acc-frames` (800), `--spp` (1), `--seed` (7), `--sun-elevation` (32.0), `--frame-index N` (single-frame worker mode), `--probe {quadrant,timing,flicker,still}`, `--start-frame` (0), `--force`, `--keep-frames`.
  - `orbit_longitude_pt(frame_index: int, total_frames: int) -> float` — `INITIAL_CENTER_LONGITUDE + 360·i/total` (frame `total` ≡ frame 0 → seamless).
  - `render_one_frame(frame_index: int, args, data: dict) -> Path` — full per-frame pipeline; writes `frames/frame_%04d.png` under `--output-dir`, returns the path.
  - `frames_to_render(frames_dir: Path, total: int, start: int, force: bool) -> list[int]` — resume logic (skips existing PNGs unless `--force`).
  - `worker_command(args, frame_index: int) -> list[str]` — subprocess argv for one frame.
  - `run_all_frames(args) -> None` — parent loop: subprocess per frame (TDR isolation), ≤3 retries each, appends to `<output-dir>/render.log`, then ffmpeg encode.
  - `main(argv=None) -> int`.

Per-frame pipeline inside `render_one_frame` (master size `M = args.tiles * args.frame`):
1. `center = orbit_longitude_pt(i, args.frames)`
2. `grid = stamp_spikes_grid(dome_heightfield(args.grid), data["sites"], center)`
3. `rgb, hit = trace_frame(grid, i, args, cache_dir)`
4. `overlay, spike_mask, ocean_mask = build_overlay(M, center, data)`
5. `shade, tint = shade_and_tint(rgb, hit, M)`
6. `mod = modulate_overlay(overlay, shade, tint, spike_mask, ocean_mask)`
7. `frame = compose_text(finish_frame(mod, args.size))`
8. `hg.write_frame(hg.frame_path(frames_dir, i), frame)`

Probes (all run through the same code path, small overrides):
- `--probe quadrant`: overrides grid 768 / frame 640 / tiles 1 / acc 400; stamps ONLY the 10 highest-density sites visible at frame 0; traces; samples mean luminance in 7-px boxes ±10 px diagonally (SE/SW/NE/NW) around each spike base pixel; prints the per-quadrant means and the darkest quadrant. Gate: darkest must be SE.
- `--probe timing`: renders frames 0, 120, 240 at production settings with `time.perf_counter`, prints s/frame and the projected total for `args.frames`.
- `--probe still`: renders frame 0 at production settings to `probe_still.png`; prints the disc dark fraction `float((shade < 0.15)[disc].mean())` where `disc` = visible pixels of `frame_view_arrays(M, center)`.
- `--probe flicker`: renders frames 0–9, computes mean |Δ luminance| between consecutive frames over ocean pixels (recomputed mask per frame), prints per-pair values and the max.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_orbit_longitude_seamless():
    assert pt.orbit_longitude_pt(0, 360) == pt.INITIAL_CENTER_LONGITUDE
    full = pt.orbit_longitude_pt(360, 360)
    assert abs((full - pt.orbit_longitude_pt(0, 360)) % 360.0) < 1e-9


def test_frames_to_render_resume(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "frame_0000.png").write_bytes(b"x")
    (frames_dir / "frame_0002.png").write_bytes(b"x")
    todo = pt.frames_to_render(frames_dir, total=4, start=0, force=False)
    assert todo == [1, 3]
    assert pt.frames_to_render(frames_dir, total=4, start=2, force=False) == [3]
    assert pt.frames_to_render(frames_dir, total=3, start=0, force=True) == [0, 1, 2]


def test_worker_command_roundtrips_through_parser():
    args = pt.parse_args([])
    cmd = pt.worker_command(args, 17)
    assert cmd[0] == sys.executable
    assert "--frame-index" in cmd and "17" in cmd
    parsed = pt.parse_args(cmd[2:])  # strip interpreter + script path
    assert parsed.frame_index == 17
    assert parsed.grid == args.grid and parsed.acc_frames == args.acc_frames
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\milos\forge3d\.venv\Scripts\python -m pytest C:/tmp/humanity-globe-pt/tests/test_humanity_globe_pt.py -v --tb=short -k "orbit or frames_to_render or worker_command"`
Expected: FAIL with `AttributeError: ... 'orbit_longitude_pt'`.

- [ ] **Step 3: Implement** (append)

```python
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--size", type=int, default=DEFAULT_SIZE)
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    p.add_argument("--grid", type=int, default=1536)
    p.add_argument("--frame", type=int, default=1080)
    p.add_argument("--tiles", type=int, default=1)
    p.add_argument("--margin", type=int, default=48)
    p.add_argument("--acc-frames", type=int, default=N_ACC)
    p.add_argument("--spp", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--sun-elevation", type=float, default=SUN_ELEVATION)
    p.add_argument("--frame-index", type=int, default=None)
    p.add_argument("--probe", choices=("quadrant", "timing", "flicker", "still"),
                   default=None)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep-frames", action="store_true")
    return p.parse_args(argv)


def orbit_longitude_pt(frame_index: int, total_frames: int) -> float:
    return INITIAL_CENTER_LONGITUDE + 360.0 * (int(frame_index) / float(max(1, total_frames)))


def render_one_frame(frame_index: int, args, data: dict) -> Path:
    center = orbit_longitude_pt(frame_index, int(args.frames))
    master = int(args.tiles) * int(args.frame)
    grid = stamp_spikes_grid(dome_heightfield(int(args.grid)), data["sites"], center)
    rgb, hit = trace_frame(grid, frame_index, args, Path(args.cache_dir))
    overlay, spike_mask, ocean_mask = build_overlay(master, center, data)
    shade, tint = shade_and_tint(rgb, hit, master)
    mod = modulate_overlay(overlay, shade, tint, spike_mask, ocean_mask)
    frame = compose_text(finish_frame(mod, int(args.size)))
    frames_dir = Path(args.output_dir) / "frames"
    out_path = hg.frame_path(frames_dir, frame_index)
    hg.write_frame(out_path, frame)
    return out_path


def frames_to_render(frames_dir: Path, total: int, start: int, force: bool) -> list[int]:
    todo = []
    for i in range(int(start), int(total)):
        if force or not hg.frame_path(frames_dir, i).exists():
            todo.append(i)
    return todo


def worker_command(args, frame_index: int) -> list[str]:
    return [
        sys.executable, str(Path(__file__).resolve()),
        "--frame-index", str(int(frame_index)),
        "--output-dir", str(args.output_dir), "--cache-dir", str(args.cache_dir),
        "--size", str(args.size), "--fps", str(args.fps),
        "--frames", str(args.frames), "--grid", str(args.grid),
        "--frame", str(args.frame), "--tiles", str(args.tiles),
        "--margin", str(args.margin), "--acc-frames", str(args.acc_frames),
        "--spp", str(args.spp), "--seed", str(args.seed),
        "--sun-elevation", str(args.sun_elevation),
    ]


def run_all_frames(args) -> None:
    frames_dir = Path(args.output_dir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.output_dir) / "render.log"
    todo = frames_to_render(frames_dir, int(args.frames), int(args.start_frame),
                            bool(args.force))
    print(f"[GlobePT] {len(todo)} frames to render "
          f"({int(args.frames) - len(todo)} cached); log: {log_path}")
    with open(log_path, "a", encoding="utf-8") as log:
        for n, i in enumerate(todo):
            for attempt in range(3):
                t0 = time.perf_counter()
                result = subprocess.run(
                    worker_command(args, i), stdout=log, stderr=subprocess.STDOUT)
                if result.returncode == 0:
                    dt = time.perf_counter() - t0
                    print(f"[GlobePT] frame {i} done in {dt:.0f}s "
                          f"({n + 1}/{len(todo)})", flush=True)
                    break
                print(f"[GlobePT] frame {i} FAILED (attempt {attempt + 1}/3, "
                      f"rc={result.returncode}) — retrying", flush=True)
            else:
                raise SystemExit(f"Frame {i} failed 3 times — see {log_path}")
    missing = frames_to_render(frames_dir, int(args.frames), 0, False)
    if missing:
        raise SystemExit(f"{len(missing)} frames still missing: {missing[:8]}...")
    output = Path(args.output_dir) / "humanity_globe_pt.mp4"
    if hg.encode_mp4(frames_dir, output, fps=int(args.fps)):
        print(f"[GlobePT] MP4: {output}")
    print("[GlobePT] frames kept on disk (source of truth for re-encode)")


def _probe_quadrant(args, data: dict) -> None:
    order = np.argsort(data["sites"]["density"])[::-1]
    center = orbit_longitude_pt(0, int(args.frames))
    vx, vy, nz, vis = view_from_latlon(
        data["sites"]["lat"][order], data["sites"]["lon"][order], center)
    good = np.nonzero(vis & (nz > 0.5))[0][:10]
    sites = {k: data["sites"][k][order][good] for k in ("lat", "lon", "density")}
    grid = stamp_spikes_grid(dome_heightfield(int(args.grid)), sites, center)
    rgb, hit = trace_frame(grid, 0, args, Path(args.cache_dir) / "probe_quadrant")
    lum = np.nan_to_num(rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
    master = int(args.tiles) * int(args.frame)
    u = 0.5 + 0.5 * DISC_FRAC * vx[good]
    v = 0.5 - 0.5 * DISC_FRAC * vy[good]
    px = (u * master).astype(int)
    py = (v * master).astype(int)
    d = 10
    means = {}
    for name, (dr, dc) in {"SE": (d, d), "SW": (d, -d),
                           "NE": (-d, d), "NW": (-d, -d)}.items():
        vals = []
        for x, y in zip(px, py):
            r0, c0 = np.clip(y + dr - 3, 0, master - 7), np.clip(x + dc - 3, 0, master - 7)
            vals.append(float(lum[r0:r0 + 7, c0:c0 + 7].mean()))
        means[name] = float(np.mean(vals))
    darkest = min(means, key=means.get)
    print(f"[probe:quadrant] {means} -> darkest {darkest} "
          f"({'OK: SE' if darkest == 'SE' else 'FAIL: expected SE'})")


def _probe_timing(args, data: dict) -> None:
    for i in (0, 120, 240):
        t0 = time.perf_counter()
        render_one_frame(i, args, data)
        dt = time.perf_counter() - t0
        print(f"[probe:timing] frame {i}: {dt:.1f}s")
    print(f"[probe:timing] projected total for {args.frames} frames: "
          f"~{dt * int(args.frames) / 3600.0:.1f}h (from last frame)")


def _probe_still(args, data: dict) -> None:
    center = orbit_longitude_pt(0, int(args.frames))
    master = int(args.tiles) * int(args.frame)
    grid = stamp_spikes_grid(dome_heightfield(int(args.grid)), data["sites"], center)
    rgb, hit = trace_frame(grid, 0, args, Path(args.cache_dir))
    shade, _ = shade_and_tint(rgb, hit, master)
    disc = frame_view_arrays(master, center)["visible"]
    dark = float((shade[disc] < 0.15).mean())
    print(f"[probe:still] disc dark fraction {dark:.3f} "
          f"({'OK' if dark <= 0.12 else 'RAISE --sun-elevation toward 44'})")
    out = Path(args.output_dir) / "probe_still.png"
    render_one_frame(0, args, data)
    frames_dir = Path(args.output_dir) / "frames"
    Path(hg.frame_path(frames_dir, 0)).replace(out)
    print(f"[probe:still] wrote {out}")


def _probe_flicker(args, data: dict) -> None:
    prev = None
    deltas = []
    for i in range(10):
        path = render_one_frame(i, args, data)
        frame = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        lum = frame @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        center = orbit_longitude_pt(i, int(args.frames))
        fa = frame_view_arrays(lum.shape[0], center)
        land = np.zeros_like(fa["visible"])
        land[fa["visible"]] = _sample_grid_nearest(
            data["land"].astype(np.uint8), fa["lat"][fa["visible"]],
            fa["lon"][fa["visible"]]) > 0
        ocean = fa["visible"] & ~land
        if prev is not None:
            deltas.append(float(np.abs(lum[ocean] - prev[ocean]).mean()))
        prev = lum
    print(f"[probe:flicker] per-pair mean|dLum| {['%.4f' % d for d in deltas]} "
          f"max {max(deltas):.4f} "
          f"({'OK' if max(deltas) < 0.004 else 'RAISE --acc-frames'})")


def main(argv=None) -> int:
    args = parse_args(argv)
    data = ensure_data(Path(args.cache_dir))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.probe == "quadrant":
        args.grid, args.frame, args.tiles, args.acc_frames = 768, 640, 1, 400
        _probe_quadrant(args, data)
        return 0
    if args.probe == "timing":
        _probe_timing(args, data)
        return 0
    if args.probe == "still":
        _probe_still(args, data)
        return 0
    if args.probe == "flicker":
        _probe_flicker(args, data)
        return 0
    if args.frame_index is not None:
        render_one_frame(int(args.frame_index), args, data)
        return 0
    run_all_frames(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Note for the flicker probe: the ocean flicker metric intentionally includes moving spike shadows on the ocean (they rotate 1°/frame — genuine motion, small). The 0.004 bound was chosen with that in mind; if it trips, FIRST check whether the delta is spatially concentrated at shadow edges (fine) or disc-wide (raise `--acc-frames`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\Users\milos\forge3d\.venv\Scripts\python -m pytest C:/tmp/humanity-globe-pt/tests/test_humanity_globe_pt.py -v --tb=short`
Expected: 25 passed.

- [ ] **Step 5: Commit**

```bash
git -C C:/tmp/humanity-globe-pt add examples/population_global_gpw/humanity_globe_pt_video.py tests/test_humanity_globe_pt.py
git -C C:/tmp/humanity-globe-pt commit -m "feat(examples): globe PT CLI, per-frame worker, resume runner, probes"
```

---

### Task 8: GPU gate — data fetch, quadrant/still/timing/flicker probes, tune constants

This task runs on the GPU (RTX 3070) and produces judgments, not just code. Run everything from the worktree with the main venv and the SHARED cache dir. **GPU renders run one at a time.**

**Files:**
- Modify: `examples/population_global_gpw/humanity_globe_pt_video.py` (constant tuning only, if probes demand it)

- [ ] **Step 1: Data prep smoke** (network + rasterio; ~1 GB download if not cached)

```bash
cd C:/tmp/humanity-globe-pt && C:/Users/milos/forge3d/.venv/Scripts/python -c "import sys; sys.path.insert(0, 'examples/population_global_gpw'); import humanity_globe_pt_video as pt; from pathlib import Path; d = pt.ensure_data(Path('C:/Users/milos/forge3d/examples/.cache/humanity_globe')); print(d['density'].shape, d['land'].sum(), len(d['sites']['density']))"
```
Expected: `(2160, 4320) <several hundred thousand land cells> <site count ~1e5-3e5>`. If the site count exceeds ~5e5, raise `DENSITY_MIN` is NOT the fix (it changes the map) — check `build_spike_sites` for a bug first.

- [ ] **Step 2: Quadrant probe** (shadow direction; <2 min)

```bash
cd C:/tmp/humanity-globe-pt && C:/Users/milos/forge3d/.venv/Scripts/python examples/population_global_gpw/humanity_globe_pt_video.py --probe quadrant --cache-dir C:/Users/milos/forge3d/examples/.cache/humanity_globe > C:/tmp/humanity-globe-pt/probe_quadrant.log 2>&1
```
Gate: log ends with `darkest SE ... OK`. If not, STOP — re-read the sun-convention memory before touching any constant.

- [ ] **Step 3: Still probe** (look + dark fraction; ~2–5 min)

```bash
cd C:/tmp/humanity-globe-pt && C:/Users/milos/forge3d/.venv/Scripts/python examples/population_global_gpw/humanity_globe_pt_video.py --probe still --cache-dir C:/Users/milos/forge3d/examples/.cache/humanity_globe > C:/tmp/humanity-globe-pt/probe_still.log 2>&1
```
Gates, in order:
1. Dark fraction ≤ 0.12. If over: re-run with `--sun-elevation 36`, then `40`, then `44`; adopt the first passing value by editing `SUN_ELEVATION` in the script (and note it in the commit message).
2. Visual judgment of `probe_still.png` (the reviewing agent reads the image): spikes visible as dots/streaks with SE shadows, coastlines readable, graticule subtle, no seams, limb clean, legend/title correct. Spike visibility knobs if needed: `STAMP_OFFSETS` (footprint), `SPIKE_MIN_WORLD`, `SPIKE_MAX_WORLD` (with `RELIEF_WORLD` recomputed — it is defined as `R_WORLD + SPIKE_MAX_WORLD`).
3. Numeric spike-shadow check: mean luminance in the 10 tallest spikes' SE 7-px boxes < their NW boxes (reuse the quadrant-probe printout from Step 2 — it already proves this).

- [ ] **Step 4: Timing probe** (~3 frames at production settings)

```bash
cd C:/tmp/humanity-globe-pt && C:/Users/milos/forge3d/.venv/Scripts/python examples/population_global_gpw/humanity_globe_pt_video.py --probe timing --cache-dir C:/Users/milos/forge3d/examples/.cache/humanity_globe > C:/tmp/humanity-globe-pt/probe_timing.log 2>&1
```
Decision rule (spec): if s/frame × 360 ≤ ~10 h, KEEP defaults (tiles 1, frame 1080). If ≤ ~10 h would also hold for quality mode (4× the per-frame cost, i.e. s/frame × 4 × 360 ≤ 10 h), UPGRADE to `--tiles 2 --frame 1008 --margin 48` (master 2016). If even single-tile exceeds ~16 h, drop `--acc-frames` to 500 and re-run the flicker probe.

- [ ] **Step 5: Flicker probe** (10 frames at the chosen production settings)

```bash
cd C:/tmp/humanity-globe-pt && C:/Users/milos/forge3d/.venv/Scripts/python examples/population_global_gpw/humanity_globe_pt_video.py --probe flicker --cache-dir C:/Users/milos/forge3d/examples/.cache/humanity_globe > C:/tmp/humanity-globe-pt/probe_flicker.log 2>&1
```
Gate: max per-pair mean|Δlum| < 0.004 on ocean. If over and disc-wide (not shadow-edge-local): raise `--acc-frames` ×1.5 and repeat (re-check timing budget).

- [ ] **Step 6: Commit tuned constants + probe logs summary**

```bash
git -C C:/tmp/humanity-globe-pt add examples/population_global_gpw/humanity_globe_pt_video.py
git -C C:/tmp/humanity-globe-pt commit -m "feat(examples): globe PT probe-tuned constants (sun/acc per measured gates)"
```
(Record the measured numbers — dark fraction, s/frame, flicker max — in the commit body.)

---

### Task 9: Full 360-frame run, encode, verify, final review

- [ ] **Step 1: Launch the overnight run in the background** (settings = Task 8's chosen config; example shown for defaults)

```bash
cd C:/tmp/humanity-globe-pt && C:/Users/milos/forge3d/.venv/Scripts/python examples/population_global_gpw/humanity_globe_pt_video.py --cache-dir C:/Users/milos/forge3d/examples/.cache/humanity_globe > C:/tmp/humanity-globe-pt/full_run.log 2>&1
```
Run via the Bash tool with `run_in_background: true` and a Monitor/wakeup on completion. The runner resumes: if the machine hiccups, re-launch the same command — cached cells + existing PNGs skip instantly.

- [ ] **Step 2: Verify completeness + encode**

Check `examples/out/humanity_globe_pt/frames/` holds exactly `frame_0000.png … frame_0359.png` (the runner already refuses to encode when frames are missing and encodes automatically when complete — confirm `humanity_globe_pt.mp4` exists and `render.log` has no unresolved FAILED lines).

- [ ] **Step 3: Numeric spot checks**
- Open frames 0, 90, 180, 270 (reviewing agent reads the images): rotation advances westward continuously; no seam pops; legend static and correct.
- Loop seam: `np.abs(frame_0000 − frame_0359)` mean < the mean of `np.abs(frame_0000 − frame_0001)` × 1.5 (frame 359 is 1° before frame 0, so consecutive-frame-sized difference).
- Full-video flicker: compute the ocean mean|Δlum| series across all 360 frames (same metric as the probe); max < 0.006.

- [ ] **Step 4: Final review + ship gate**
- Run the whole unit suite once more: `C:\Users\milos\forge3d\.venv\Scripts\python -m pytest C:/tmp/humanity-globe-pt/tests/test_humanity_globe_pt.py -v --tb=short` → all pass.
- Confirm the original script still imports: `C:/Users/milos/forge3d/.venv/Scripts/python -c "import sys; sys.path.insert(0, 'C:/tmp/humanity-globe-pt/examples/population_global_gpw'); import humanity_globe_video"`.
- `git -C C:/tmp/humanity-globe-pt status` — only intended files.
- Send the MP4 + a representative frame to the user; **ask whether to open a PR** (per CLAUDE.md — do not open one unprompted).
