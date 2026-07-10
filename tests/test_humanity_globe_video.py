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
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
        added_examples_dir = True
    previous_forge3d = sys.modules.get("forge3d")
    sys.modules["forge3d"] = forge3d_stub
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        if previous_forge3d is None:
            sys.modules.pop("forge3d", None)
        else:
            sys.modules["forge3d"] = previous_forge3d
        if added_examples_dir and examples_dir in sys.path:
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


def test_load_density_reads_valid_15min_grid(tmp_path: Path) -> None:
    module = load_module()
    rasterio = pytest.importorskip("rasterio")
    path = tmp_path / "gpw.tif"
    data = np.ones(module.EXPECTED_15MIN_SHAPE, dtype=np.float32)
    with rasterio.open(path, "w", driver="GTiff", width=data.shape[1], height=data.shape[0], count=1, dtype="float32") as dst:
        dst.write(data, 1)

    loaded = module.load_density(path)

    assert loaded.shape == module.EXPECTED_15MIN_SHAPE
    assert loaded.dtype == np.float32


def test_render_video_preview_only_resolves_data_and_writes_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
