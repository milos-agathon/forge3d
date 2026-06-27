# Humanity Globe Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `examples/humanity_globe_video.py` so forge3d can recreate the reference Humanity Globe MP4 from GPW-v4 population density data.

**Architecture:** Keep the globe renderer as a focused offline Python example because forge3d does not currently expose a native textured-globe renderer. The example owns data loading/aggregation, density classification, orthographic sphere rasterization, frame composition, and ffmpeg encoding while using the repo's example import shim and `forge3d.numpy_to_png` for PNG output.

**Tech Stack:** Python 3.10+, `numpy`, `Pillow`, `rasterio`, optional `matplotlib`, `pytest`, `ffmpeg`, forge3d Python package utilities.

---

## Scope Check

The approved spec is a single deliverable: one example script plus focused tests and docs. It does not require native Rust/WebGPU changes, viewer changes, or changes to existing Poland/Turkey population examples.

## File Structure

- Create `examples/humanity_globe_video.py`
  - Owns CLI parsing, GPW data resolution, optional aggregation from 30-arc-second GPW raster to 15-minute cells, density classification, palette selection, orthographic sphere sampling, frame composition, PNG writing, and MP4 encoding.
  - Keep pure helpers import-safe so tests can load the module without network, ffmpeg, or GPU.
- Create `tests/test_humanity_globe_video.py`
  - Loads the example via `importlib.util.spec_from_file_location`.
  - Stubs `forge3d.numpy_to_png` so pure tests do not require native loading.
  - Covers CLI defaults, classification, grid validation, coordinate sampling, orbit progression, palette fallback, legend labels, ffmpeg command construction, and a tiny preview render.
- Modify `docs/examples/index.md`
  - Add a row for `humanity_globe_video.py` in the animation/data visualization catalog.

## Implementation Tasks

### Task 1: Import-Safe Example Shell And CLI Defaults

**Files:**
- Create: `examples/humanity_globe_video.py`
- Create: `tests/test_humanity_globe_video.py`

- [ ] **Step 1: Write failing import and CLI default tests**

Add `tests/test_humanity_globe_video.py` with this initial content:

```python
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "humanity_globe_video.py"


def load_module():
    spec = importlib.util.spec_from_file_location("humanity_globe_video", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    forge3d_stub = types.ModuleType("forge3d")

    def numpy_to_png(path, array):
        from PIL import Image

        Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGBA").save(path)

    forge3d_stub.numpy_to_png = numpy_to_png
    examples_dir = str(EXAMPLE_PATH.parent)
    added_examples_dir = False
    previous_forge3d = sys.modules.get("forge3d")
    previous_module = sys.modules.get(spec.name)
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
        added_examples_dir = True
    sys.modules["forge3d"] = forge3d_stub
    sys.modules[spec.name] = module
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        if previous_module is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous_module
        if previous_forge3d is None:
            sys.modules.pop("forge3d", None)
        else:
            sys.modules["forge3d"] = previous_forge3d
        if added_examples_dir:
            sys.path.remove(examples_dir)
    return module


def test_parse_args_defaults_match_reference_video(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    monkeypatch.setattr(sys, "argv", ["humanity_globe_video.py"])

    args = module.parse_args()

    assert args.size == 720
    assert args.fps == 25
    assert args.duration == pytest.approx(28.8)
    assert args.frames is None
    assert module.frame_count(args) == 720
    assert args.output == module.DEFAULT_OUTPUT
    assert args.preview == module.DEFAULT_PREVIEW


def test_parse_args_accepts_explicit_frame_override(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    monkeypatch.setattr(sys, "argv", ["humanity_globe_video.py", "--frames", "25", "--size", "360"])

    args = module.parse_args()

    assert args.frames == 25
    assert args.size == 360
    assert module.frame_count(args) == 25
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_humanity_globe_video.py -q
```

Expected: FAIL because `examples/humanity_globe_video.py` does not exist.

- [ ] **Step 3: Create the import-safe example shell**

Create `examples/humanity_globe_video.py` with:

```python
#!/usr/bin/env python3
"""Recreate the Humanity Globe population-density video with forge3d helpers.

Reference concept and original rayshader/rayrender recipe:
https://gist.github.com/tylermorganwall/3ee1c6e2a5dff19aca7836c05cbbf9ac

Data source family: GPW-v4 population density, revision 11, 2020.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _import_shim import ensure_repo_import

ensure_repo_import()

import forge3d as f3d


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "examples" / "out" / "humanity_globe"
DEFAULT_CACHE_DIR = REPO_ROOT / "examples" / ".cache" / "humanity_globe"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "humanity_globe_forge3d.mp4"
DEFAULT_PREVIEW = DEFAULT_OUTPUT_DIR / "humanity_globe_preview.png"
DEFAULT_GPW_15MIN = REPO_ROOT / "data" / "gpw_v4_population_density_rev11_2020_15_min.tif"
DEFAULT_SIZE = 720
DEFAULT_FPS = 25
DEFAULT_DURATION = 28.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a forge3d recreation of the Humanity Globe population-density MP4."
    )
    parser.add_argument("--gpw-tif", type=Path, default=None, help="Explicit GPW-v4 2020 population-density GeoTIFF.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--frames-only", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def frame_count(args: argparse.Namespace) -> int:
    if args.frames is not None:
        return max(1, int(args.frames))
    return max(1, int(round(float(args.fps) * float(args.duration))))


def main() -> int:
    args = parse_args()
    raise SystemExit("Rendering is implemented in later tasks.")


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests and verify Task 1 passes**

Run:

```powershell
python -m pytest tests/test_humanity_globe_video.py -q
```

Expected: 2 passed.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add examples/humanity_globe_video.py tests/test_humanity_globe_video.py
git commit -m "Add humanity globe example CLI shell"
```

Expected: commit succeeds.

### Task 2: Density Classification, Palette, Grid Validation, And GPW Data Resolution

**Files:**
- Modify: `tests/test_humanity_globe_video.py`
- Modify: `examples/humanity_globe_video.py`

- [ ] **Step 1: Add failing pure data tests**

Append these tests to `tests/test_humanity_globe_video.py`:

```python
def test_density_classification_matches_reference_thresholds() -> None:
    module = load_module()
    values = np.array([[0.0, 0.5, 1.0, 1.1, 5.1, 10.1, 50.1, 100.1, 500.1, 1000.1]], dtype=np.float32)

    classes = module.classify_density(values)

    assert classes.tolist() == [[0, 0, 0, 1, 2, 3, 4, 5, 6, 7]]


def test_validate_15min_grid_accepts_reference_shape() -> None:
    module = load_module()
    data = np.zeros((720, 1440), dtype=np.float32)

    module.validate_15min_grid(data)


def test_validate_15min_grid_rejects_wrong_shape() -> None:
    module = load_module()
    data = np.zeros((10, 20), dtype=np.float32)

    with pytest.raises(ValueError, match="Expected 15-minute GPW grid"):
        module.validate_15min_grid(data)


def test_turbo_palette_has_reference_class_count_and_uint8_values() -> None:
    module = load_module()

    palette = module.turbo_class_palette()

    assert palette.shape == (8, 3)
    assert palette.dtype == np.uint8
    assert palette[0].tolist() == [175, 175, 175]
    assert len({tuple(row) for row in palette[1:]}) == 7


def test_legend_labels_match_reference_text() -> None:
    module = load_module()

    assert module.LEGEND_TITLE == "People per 30km^2"
    assert module.LEGEND_LABELS == ("0", "1>", "5>", "10>", "50>", "100>", "500>", "1000>")
```

- [ ] **Step 2: Run the new data tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_humanity_globe_video.py -q
```

Expected: FAIL because `classify_density`, `validate_15min_grid`, `turbo_class_palette`, and legend constants are missing.

- [ ] **Step 3: Implement classification, palette, validation, and data path resolution**

Add these imports near the top of `examples/humanity_globe_video.py`:

```python
import json
import shutil
import urllib.request
from dataclasses import dataclass
from typing import Iterable

import numpy as np
```

Add these constants and helpers below the default path constants:

```python
GPW_30SEC_URL = "https://pacific-data.sprep.org/system/files/Global_2020_PopulationDensity30sec_GPWv4.tiff"
EXPECTED_15MIN_SHAPE = (720, 1440)
EXPECTED_30SEC_SHAPE = (21600, 43200)
AGGREGATION_FACTOR = 30
DENSITY_THRESHOLDS = np.array([1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0], dtype=np.float32)
LEGEND_TITLE = "People per 30km^2"
LEGEND_LABELS = ("0", "1>", "5>", "10>", "50>", "100>", "500>", "1000>")
FALLBACK_TURBO7 = np.array(
    [
        [109, 76, 134],
        [116, 173, 239],
        [52, 214, 207],
        [183, 230, 121],
        [247, 222, 91],
        [246, 142, 64],
        [209, 55, 36],
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class GpwData:
    path: Path
    derived: bool
    source: str
```

Add these pure helpers:

```python
def classify_density(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    safe = np.where(np.isfinite(data), data, 0.0)
    classes = np.zeros(safe.shape, dtype=np.uint8)
    for idx, threshold in enumerate(DENSITY_THRESHOLDS, start=1):
        classes[safe > threshold] = idx
    return classes


def validate_15min_grid(data: np.ndarray) -> None:
    if tuple(data.shape) != EXPECTED_15MIN_SHAPE:
        raise ValueError(
            f"Expected 15-minute GPW grid with shape {EXPECTED_15MIN_SHAPE}, got {tuple(data.shape)}"
        )


def turbo_class_palette() -> np.ndarray:
    base = np.array([[175, 175, 175]], dtype=np.uint8)
    try:
        import matplotlib.colormaps as colormaps

        cmap = colormaps["turbo"]
        xs = np.linspace(0.04, 0.92, 7, dtype=np.float32)
        turbo = np.round(np.asarray(cmap(xs), dtype=np.float32)[:, :3] * 255.0).astype(np.uint8)
    except Exception:
        turbo = FALLBACK_TURBO7.copy()
    return np.vstack([base, turbo]).astype(np.uint8)


def resolve_gpw_source(args: argparse.Namespace) -> GpwData:
    candidates = []
    if args.gpw_tif is not None:
        candidates.append(Path(args.gpw_tif))
    candidates.extend([DEFAULT_GPW_15MIN, Path(args.cache_dir) / DEFAULT_GPW_15MIN.name])
    for candidate in candidates:
        if candidate.exists():
            return GpwData(candidate, derived=False, source=str(candidate))
    derived_path = derive_15min_gpw(Path(args.cache_dir), force=bool(args.force))
    return GpwData(derived_path, derived=True, source=GPW_30SEC_URL)


def derive_15min_gpw(cache_dir: Path, *, force: bool = False) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / DEFAULT_GPW_15MIN.name
    metadata_path = out_path.with_suffix(out_path.suffix + ".json")
    if out_path.exists() and not force:
        return out_path
    source_path = cache_dir / "Global_2020_PopulationDensity30sec_GPWv4.tiff"
    if not source_path.exists() or force:
        download_path = source_path.with_suffix(source_path.suffix + ".download")
        if download_path.exists():
            download_path.unlink()
        try:
            urllib.request.urlretrieve(GPW_30SEC_URL, download_path)
            download_path.replace(source_path)
        except Exception as exc:
            if download_path.exists():
                download_path.unlink()
            raise SystemExit(
                "Missing GPW-v4 15-minute raster. Provide --gpw-tif pointing to "
                "gpw_v4_population_density_rev11_2020_15_min.tif, or allow download "
                f"from {GPW_30SEC_URL}. Download failed: {exc}"
            ) from exc
    aggregate_30sec_to_15min(source_path, out_path)
    metadata_path.write_text(
        json.dumps({"source": GPW_30SEC_URL, "aggregation": "mean 30x30 cells", "shape": list(EXPECTED_15MIN_SHAPE)}, indent=2),
        encoding="utf-8",
    )
    return out_path
```

Add the aggregation helper, using chunked reads to avoid loading the full 30-arc-second raster at once:

```python
def aggregate_30sec_to_15min(source_path: Path, out_path: Path) -> None:
    try:
        import rasterio
    except ModuleNotFoundError as exc:
        raise SystemExit("rasterio is required to derive the 15-minute GPW raster. Install rasterio or pass --gpw-tif.") from exc

    with rasterio.open(source_path) as src:
        if (src.height, src.width) != EXPECTED_30SEC_SHAPE:
            raise ValueError(f"Expected 30-arc-second GPW raster shape {EXPECTED_30SEC_SHAPE}, got {(src.height, src.width)}")
        profile = src.profile.copy()
        out = np.zeros(EXPECTED_15MIN_SHAPE, dtype=np.float32)
        for out_row in range(EXPECTED_15MIN_SHAPE[0]):
            window = rasterio.windows.Window(0, out_row * AGGREGATION_FACTOR, src.width, AGGREGATION_FACTOR)
            block = src.read(1, window=window, masked=True).astype(np.float32)
            row = block.reshape(AGGREGATION_FACTOR, EXPECTED_15MIN_SHAPE[1], AGGREGATION_FACTOR).mean(axis=(0, 2))
            out[out_row, :] = np.asarray(row.filled(0.0) if hasattr(row, "filled") else row, dtype=np.float32)
    profile.update(width=EXPECTED_15MIN_SHAPE[1], height=EXPECTED_15MIN_SHAPE[0], dtype="float32", count=1, compress="deflate")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)
```

- [ ] **Step 4: Run data tests**

Run:

```powershell
python -m pytest tests/test_humanity_globe_video.py -q
```

Expected: all current tests pass.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add examples/humanity_globe_video.py tests/test_humanity_globe_video.py
git commit -m "Add humanity globe data classification helpers"
```

Expected: commit succeeds.

### Task 3: Orthographic Projection, Orbit, And Tiny Preview Rendering

**Files:**
- Modify: `tests/test_humanity_globe_video.py`
- Modify: `examples/humanity_globe_video.py`

- [ ] **Step 1: Add failing projection and preview tests**

Append these tests:

```python
def test_orbit_longitude_completes_single_rotation() -> None:
    module = load_module()

    values = [module.orbit_longitude(i, 720) for i in (0, 180, 360, 540, 719)]

    assert values[0] == pytest.approx(-100.0)
    assert values[1] == pytest.approx(-10.0)
    assert values[2] == pytest.approx(80.0)
    assert values[3] == pytest.approx(170.0)
    assert values[4] < 260.0


def test_sphere_lat_lon_center_tracks_orbit_longitude() -> None:
    module = load_module()

    visible, lat, lon, normals = module.sphere_lat_lon(size=9, center_lon=-100.0)

    assert visible[4, 4]
    assert lat[4, 4] == pytest.approx(0.0, abs=1e-6)
    assert lon[4, 4] == pytest.approx(-100.0, abs=1e-6)
    assert normals.shape == (9, 9, 3)


def test_render_frame_with_tiny_synthetic_grid_returns_rgba() -> None:
    module = load_module()
    density = np.zeros((18, 36), dtype=np.float32)
    density[7:11, 14:20] = 1200.0

    frame = module.render_frame(density, frame_index=0, total_frames=4, size=96, include_text=False)

    assert frame.shape == (96, 96, 4)
    assert frame.dtype == np.uint8
    assert int(frame[:, :, 3].min()) == 255
    assert int(frame[:, :, :3].max()) > 175
```

- [ ] **Step 2: Run tests and verify projection/render failures**

Run:

```powershell
python -m pytest tests/test_humanity_globe_video.py -q
```

Expected: FAIL for missing `orbit_longitude`, `sphere_lat_lon`, and `render_frame`.

- [ ] **Step 3: Implement projection and base renderer**

Add these constants:

```python
TITLE_TEXT = "The Humanity Globe: World Population Density, 30km^2 Grid"
DATA_CREDIT = "Data: 2020 GPW-v4"
CREATED_CREDIT = "Created with forge3d"
REFERENCE_CREDIT = "Reference: @tylermorganwall"
INITIAL_CENTER_LONGITUDE = -100.0
```

Add these helpers before `main()`:

```python
def orbit_longitude(frame_index: int, total_frames: int) -> float:
    total = max(1, int(total_frames))
    return INITIAL_CENTER_LONGITUDE + 360.0 * (int(frame_index) / float(total))


def sphere_lat_lon(size: int, center_lon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords = (np.arange(int(size), dtype=np.float32) + 0.5) / float(size)
    x = (coords[None, :] - 0.5) * 2.0
    y = (0.5 - coords[:, None]) * 2.0
    radius = 0.72
    sx = x / radius
    sy = y / radius
    rr = sx * sx + sy * sy
    visible = rr <= 1.0
    z = np.sqrt(np.clip(1.0 - rr, 0.0, 1.0))
    lat = np.degrees(np.arcsin(np.clip(sy, -1.0, 1.0))).astype(np.float32)
    lon_offset = np.degrees(np.arctan2(sx, np.maximum(z, 1e-6))).astype(np.float32)
    lon = ((float(center_lon) + lon_offset + 180.0) % 360.0 - 180.0).astype(np.float32)
    normals = np.dstack([sx, sy, z]).astype(np.float32)
    normals[~visible] = 0.0
    return visible, lat, lon, normals


def sample_classes(classes: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    h, w = classes.shape
    row = np.clip(((90.0 - lat) / 180.0 * h).astype(np.int32), 0, h - 1)
    col = np.clip(((lon + 180.0) / 360.0 * w).astype(np.int32), 0, w - 1)
    return classes[row, col]


def _light(normals: np.ndarray) -> np.ndarray:
    light_dir = np.array([-0.45, 0.35, 0.82], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    lambert = np.clip((normals * light_dir.reshape(1, 1, 3)).sum(axis=2), 0.0, 1.0)
    rim = np.power(np.clip(1.0 - normals[:, :, 2], 0.0, 1.0), 2.0)
    return np.clip(0.38 + 0.52 * lambert + 0.12 * rim, 0.0, 1.0)


def render_frame(
    density: np.ndarray,
    *,
    frame_index: int,
    total_frames: int,
    size: int = DEFAULT_SIZE,
    include_text: bool = True,
) -> np.ndarray:
    classes = classify_density(density)
    visible, lat, lon, normals = sphere_lat_lon(size, orbit_longitude(frame_index, total_frames))
    sampled = sample_classes(classes, lat, lon)
    palette = turbo_class_palette()
    rgb = np.zeros((size, size, 3), dtype=np.float32)
    base = palette[0].astype(np.float32)
    light = _light(normals)
    rgb[visible] = base * light[visible, None]
    positive = visible & (sampled > 0)
    if positive.any():
        color = palette[sampled]
        boost = np.clip(0.92 + 0.20 * (1.0 - normals[:, :, 2]), 0.85, 1.18)
        rgb[positive] = np.clip(color[positive].astype(np.float32) * boost[positive, None], 0.0, 255.0)
    alpha = np.full((size, size, 1), 255, dtype=np.uint8)
    frame = np.dstack([np.clip(rgb, 0.0, 255.0).astype(np.uint8), alpha])
    if include_text:
        frame = compose_frame_text(frame)
    return frame
```

Add a temporary no-op text compositor so tests pass before the full layout task:

```python
def compose_frame_text(frame: np.ndarray) -> np.ndarray:
    return frame
```

- [ ] **Step 4: Run projection/render tests**

Run:

```powershell
python -m pytest tests/test_humanity_globe_video.py -q
```

Expected: all current tests pass.

- [ ] **Step 5: Commit Task 3**

Run:

```powershell
git add examples/humanity_globe_video.py tests/test_humanity_globe_video.py
git commit -m "Add humanity globe projection renderer"
```

Expected: commit succeeds.

### Task 4: Text Layout, Frame Writing, ffmpeg Command, And Render Orchestration

**Files:**
- Modify: `tests/test_humanity_globe_video.py`
- Modify: `examples/humanity_globe_video.py`

- [ ] **Step 1: Add failing orchestration tests**

Append these tests:

```python
def test_frame_path_is_deterministic(tmp_path: Path) -> None:
    module = load_module()

    assert module.frame_path(tmp_path, 7).name == "frame_0007.png"


def test_ffmpeg_command_uses_reference_encoding_settings(tmp_path: Path) -> None:
    module = load_module()

    cmd = module.build_ffmpeg_command(tmp_path / "frames", tmp_path / "out.mp4", fps=25)

    assert cmd[:4] == ["ffmpeg", "-y", "-framerate", "25"]
    assert "-c:v" in cmd
    assert "libx264" in cmd
    assert "-pix_fmt" in cmd
    assert "yuv420p" in cmd
    assert "+faststart" in cmd
    assert str(tmp_path / "out.mp4") == cmd[-1]


def test_write_frame_uses_forge3d_png_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    calls = []

    def fake_numpy_to_png(path, array):
        calls.append((Path(path), np.asarray(array).shape))

    monkeypatch.setattr(module.f3d, "numpy_to_png", fake_numpy_to_png)

    frame = np.zeros((12, 12, 4), dtype=np.uint8)
    module.write_frame(tmp_path / "frame_0000.png", frame)

    assert calls == [(tmp_path / "frame_0000.png", (12, 12, 4))]


def test_render_preview_writes_preview_png(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    written = []

    def fake_write_frame(path, frame):
        written.append((Path(path), frame.shape))

    monkeypatch.setattr(module, "write_frame", fake_write_frame)
    density = np.zeros((18, 36), dtype=np.float32)

    module.render_preview(density, tmp_path / "preview.png", size=64)

    assert written == [(tmp_path / "preview.png", (64, 64, 4))]
```

- [ ] **Step 2: Run tests and verify orchestration failures**

Run:

```powershell
python -m pytest tests/test_humanity_globe_video.py -q
```

Expected: FAIL for missing frame/ffmpeg/orchestration helpers.

- [ ] **Step 3: Implement text composition and orchestration helpers**

Add imports:

```python
import subprocess
from PIL import Image, ImageDraw, ImageFont
```

Replace `compose_frame_text()` with:

```python
def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    font_name = "segoeuib.ttf" if bold else "segoeui.ttf"
    font_path = Path("C:/Windows/Fonts") / font_name
    try:
        return ImageFont.truetype(str(font_path), size)
    except OSError:
        return ImageFont.load_default()


def compose_frame_text(frame: np.ndarray) -> np.ndarray:
    image = Image.fromarray(frame, mode="RGBA")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    title_font = _font(max(14, int(w * 0.028)))
    label_font = _font(max(10, int(w * 0.019)), bold=True)
    small_font = _font(max(8, int(w * 0.014)))
    draw.text((int(w * 0.02), int(h * 0.018)), TITLE_TEXT, fill=(238, 238, 238, 255), font=title_font)
    legend_x = int(w * 0.078)
    legend_y = int(h * 0.872)
    cell_w = int(w * 0.105)
    cell_h = int(h * 0.05)
    draw.text((legend_x, legend_y - int(h * 0.042)), LEGEND_TITLE, fill=(245, 245, 245, 255), font=label_font)
    palette = turbo_class_palette()
    for idx, label in enumerate(LEGEND_LABELS):
        x0 = legend_x + idx * cell_w
        color = tuple(int(v) for v in palette[idx]) + (255,)
        draw.rectangle((x0, legend_y, x0 + cell_w, legend_y + cell_h), fill=color)
        draw.text((x0 + int(cell_w * 0.38), legend_y + int(cell_h * 0.18)), label, fill=(0, 0, 0, 255), font=small_font)
    credit_y = int(h * 0.952)
    draw.text((int(w * 0.02), credit_y), DATA_CREDIT, fill=(220, 220, 220, 255), font=small_font)
    draw.text((int(w * 0.02), credit_y + int(h * 0.022)), CREATED_CREDIT, fill=(220, 220, 220, 255), font=small_font)
    ref_text = REFERENCE_CREDIT
    bbox = draw.textbbox((0, 0), ref_text, font=label_font)
    draw.text((w - (bbox[2] - bbox[0]) - int(w * 0.02), credit_y + int(h * 0.012)), ref_text, fill=(0, 140, 255, 255), font=label_font)
    return np.asarray(image, dtype=np.uint8)
```

Add orchestration helpers:

```python
def frame_path(frames_dir: Path, frame_index: int) -> Path:
    return Path(frames_dir) / f"frame_{int(frame_index):04d}.png"


def write_frame(path: Path, frame: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    f3d.numpy_to_png(path, np.ascontiguousarray(frame, dtype=np.uint8))


def build_ffmpeg_command(frames_dir: Path, output_path: Path, *, fps: int) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-framerate",
        str(int(fps)),
        "-i",
        str(Path(frames_dir) / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def encode_mp4(frames_dir: Path, output_path: Path, *, fps: int) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[HumanityGlobe] ffmpeg was not found; leaving PNG frames on disk.")
        return False
    cmd = build_ffmpeg_command(frames_dir, output_path, fps=fps)
    cmd[0] = ffmpeg
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg failed:\n{result.stderr[-1200:]}")
    return True


def render_preview(density: np.ndarray, preview_path: Path, *, size: int = DEFAULT_SIZE) -> None:
    write_frame(preview_path, render_frame(density, frame_index=0, total_frames=1, size=size, include_text=True))
```

- [ ] **Step 4: Run orchestration tests**

Run:

```powershell
python -m pytest tests/test_humanity_globe_video.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 4**

Run:

```powershell
git add examples/humanity_globe_video.py tests/test_humanity_globe_video.py
git commit -m "Add humanity globe frame composition"
```

Expected: commit succeeds.

### Task 5: Raster Loading, CLI Render Path, Docs, And Verification

**Files:**
- Modify: `tests/test_humanity_globe_video.py`
- Modify: `examples/humanity_globe_video.py`
- Modify: `docs/examples/index.md`

- [ ] **Step 1: Add final render path tests**

Append:

```python
def test_load_density_reads_valid_15min_grid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    rasterio = pytest.importorskip("rasterio")
    path = tmp_path / "gpw.tif"
    data = np.zeros(module.EXPECTED_15MIN_SHAPE, dtype=np.float32)
    with rasterio.open(path, "w", driver="GTiff", width=data.shape[1], height=data.shape[0], count=1, dtype="float32") as dst:
        dst.write(data, 1)

    loaded = module.load_density(path)

    assert loaded.shape == module.EXPECTED_15MIN_SHAPE
    assert loaded.dtype == np.float32


def test_render_video_preview_only_resolves_data_and_writes_preview(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    args = types.SimpleNamespace(
        gpw_tif=None,
        output=tmp_path / "out.mp4",
        preview=tmp_path / "preview.png",
        cache_dir=tmp_path / "cache",
        size=32,
        fps=25,
        duration=28.8,
        frames=None,
        preview_only=True,
        frames_only=False,
        keep_frames=False,
        force=False,
    )
    density = np.zeros((18, 36), dtype=np.float32)
    calls = []

    monkeypatch.setattr(module, "resolve_gpw_source", lambda _args: module.GpwData(tmp_path / "synthetic.tif", False, "synthetic"))
    monkeypatch.setattr(module, "load_density", lambda _path: density)
    monkeypatch.setattr(module, "render_preview", lambda data, path, size: calls.append((data.shape, Path(path), size)))

    output_path, preview_path = module.render_video(args)

    assert output_path is None
    assert preview_path == tmp_path / "preview.png"
    assert calls == [((18, 36), tmp_path / "preview.png", 32)]
```

- [ ] **Step 2: Run tests and verify final failures**

Run:

```powershell
python -m pytest tests/test_humanity_globe_video.py -q
```

Expected: FAIL for missing `load_density` and `render_video`.

- [ ] **Step 3: Implement raster loading and full CLI render path**

Add:

```python
def load_density(path: Path) -> np.ndarray:
    try:
        import rasterio
    except ModuleNotFoundError as exc:
        raise SystemExit("rasterio is required to read GPW GeoTIFF data. Install rasterio or provide a prepared NumPy path in a future extension.") from exc
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
    validate_15min_grid(data)
    data[~np.isfinite(data)] = 0.0
    data = np.clip(data, 0.0, None)
    if float(data.max()) <= 0.0:
        raise ValueError("GPW density raster contains no positive population-density values.")
    return data


def clear_frames(frames_dir: Path) -> None:
    if frames_dir.exists():
        for path in frames_dir.glob("frame_*.png"):
            path.unlink()
    frames_dir.mkdir(parents=True, exist_ok=True)


def render_frames(density: np.ndarray, frames_dir: Path, *, size: int, total_frames: int) -> None:
    clear_frames(frames_dir)
    for index in range(total_frames):
        frame = render_frame(density, frame_index=index, total_frames=total_frames, size=size, include_text=True)
        write_frame(frame_path(frames_dir, index), frame)
        print(f"\r[HumanityGlobe] frame {index + 1}/{total_frames}", end="", flush=True)
    print()


def render_video(args: argparse.Namespace) -> tuple[Path | None, Path]:
    gpw = resolve_gpw_source(args)
    print(f"[HumanityGlobe] GPW source: {gpw.source}")
    density = load_density(gpw.path)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.preview_only:
        render_preview(density, args.preview, size=int(args.size))
        return None, args.preview
    frames_dir = args.output.parent / "frames"
    total_frames = frame_count(args)
    render_frames(density, frames_dir, size=int(args.size), total_frames=total_frames)
    render_preview(density, args.preview, size=int(args.size))
    if args.frames_only:
        return None, args.preview
    encoded = encode_mp4(frames_dir, args.output, fps=int(args.fps))
    if encoded and not args.keep_frames:
        for path in frames_dir.glob("frame_*.png"):
            path.unlink()
    return (args.output if encoded else None), args.preview
```

Replace `main()` with:

```python
def main() -> int:
    args = parse_args()
    output_path, preview_path = render_video(args)
    print(f"[HumanityGlobe] Preview: {preview_path}")
    if output_path is not None:
        print(f"[HumanityGlobe] MP4: {output_path}")
    return 0
```

- [ ] **Step 4: Update examples catalog**

In `docs/examples/index.md`, add this row under `## Animation And Camera Automation` after `khumbu_icefall_sentinel_timelapse.py`:

```markdown
| `humanity_globe_video.py` | Offline rotating-globe recreation of the Humanity Globe GPW-v4 population-density video with stepped turbo colors and MP4 output. | GPW-v4 population density, `numpy_to_png()`, `ffmpeg` |
```

- [ ] **Step 5: Run unit tests**

Run:

```powershell
python -m pytest tests/test_humanity_globe_video.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Run import/help smoke**

Run:

```powershell
python examples/humanity_globe_video.py --help
```

Expected: exits 0 and shows `--gpw-tif`, `--preview-only`, `--frames-only`, `--keep-frames`, and `--force`.

- [ ] **Step 7: Run tiny synthetic preview smoke without real GPW data**

Run:

```powershell
@'
from pathlib import Path
import importlib.util
import numpy as np

path = Path("examples/humanity_globe_video.py")
spec = importlib.util.spec_from_file_location("humanity_globe_video", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
density = np.zeros((18, 36), dtype=np.float32)
density[6:12, 12:22] = 1200.0
module.render_preview(density, Path("examples/out/humanity_globe/synthetic_preview.png"), size=160)
print("wrote synthetic preview")
'@ | python -
```

Expected: exits 0 and writes `examples/out/humanity_globe/synthetic_preview.png`.

- [ ] **Step 8: Optional short frame export if real GPW data is available**

Run only if `data/gpw_v4_population_density_rev11_2020_15_min.tif` exists or the public GPW download is acceptable:

```powershell
python examples/humanity_globe_video.py --frames 25 --frames-only --size 360 --keep-frames
```

Expected: writes 25 frames under `examples/out/humanity_globe/frames/`.

- [ ] **Step 9: Commit Task 5**

Run:

```powershell
git add examples/humanity_globe_video.py tests/test_humanity_globe_video.py docs/examples/index.md
git commit -m "Add humanity globe video example"
```

Expected: commit succeeds.

### Task 6: Full Artifact Verification And Final Polish

**Files:**
- Modify only if verification reveals defects:
  - `examples/humanity_globe_video.py`
  - `tests/test_humanity_globe_video.py`
  - `docs/examples/index.md`

- [ ] **Step 1: Run focused tests**

Run:

```powershell
python -m pytest tests/test_humanity_globe_video.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run relevant docs/catalog smoke**

Run:

```powershell
python -m pytest tests/test_mapscene_docs.py tests/test_datasets.py -q
```

Expected: tests pass. If unrelated failures occur, capture output and continue with focused verification.

- [ ] **Step 3: Generate a production preview**

Run:

```powershell
python examples/humanity_globe_video.py --preview-only
```

Expected: if GPW data is available or download succeeds, writes `examples/out/humanity_globe/humanity_globe_preview.png`. If the public download fails, rerun with an explicit local `--gpw-tif` path once available.

- [ ] **Step 4: Generate the full MP4**

Run:

```powershell
python examples/humanity_globe_video.py
```

Expected: writes `examples/out/humanity_globe/humanity_globe_forge3d.mp4`.

- [ ] **Step 5: Verify MP4 metadata**

Run:

```powershell
ffprobe -v error -show_entries stream=width,height,avg_frame_rate,nb_frames,duration -of default=noprint_wrappers=1 examples/out/humanity_globe/humanity_globe_forge3d.mp4
```

Expected output includes:

```text
width=720
height=720
avg_frame_rate=25/1
duration=28.800000
nb_frames=720
```

- [ ] **Step 6: Extract contact sheet for visual comparison**

Run:

```powershell
ffmpeg -y -v error -i examples/out/humanity_globe/humanity_globe_forge3d.mp4 -vf "fps=1/4,scale=180:180,tile=4x2" -frames:v 1 examples/out/humanity_globe/humanity_globe_contact_sheet.png
```

Expected: writes `examples/out/humanity_globe/humanity_globe_contact_sheet.png` showing a full rotating globe with colored population layers and visible title/legend/credits.

- [ ] **Step 7: Fix any verification defects and commit**

If defects were found, patch the relevant file, rerun the failing command, then commit:

```powershell
git add examples/humanity_globe_video.py tests/test_humanity_globe_video.py docs/examples/index.md
git commit -m "Polish humanity globe video output"
```

Expected: commit only if files changed.

## Plan Self-Review

Spec coverage:

- Output size, fps, duration, frame count: Task 1 tests and Task 6 ffprobe.
- GPW-v4 15-minute data path and 30-arc-second derivation: Task 2.
- Classification thresholds and turbo colors: Task 2.
- Orthographic rotating globe: Task 3.
- Title, legend, and credits: Task 4.
- `forge3d.numpy_to_png` use: Task 4 tests and implementation.
- MP4 encoding: Task 4 and Task 6.
- Docs catalog: Task 5.
- Out-of-scope native globe/runtime work: no Rust or viewer tasks included.

Red-flag scan:

- No incomplete-marker steps remain.
- Every code-writing step includes concrete code.
- Every verification step has an exact command and expected result.

Type consistency:

- `frame_count(args)`, `classify_density(values)`, `validate_15min_grid(data)`, `turbo_class_palette()`, `orbit_longitude(frame_index, total_frames)`, `sphere_lat_lon(size, center_lon)`, `render_frame(...)`, `frame_path(...)`, `write_frame(...)`, `build_ffmpeg_command(...)`, `load_density(...)`, and `render_video(args)` are introduced before later tasks reference them.
