from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "khumbu_icefall_sentinel_timelapse.py"


def load_module():
    spec = importlib.util.spec_from_file_location("khumbu_icefall_sentinel_timelapse", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    forge3d_stub = types.ModuleType("forge3d")
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


def feature(item_id: str, timestamp: str, cloud_cover: float, tile: str = "45RVL") -> dict[str, object]:
    return {
        "id": item_id,
        "properties": {
            "datetime": timestamp,
            "eo:cloud_cover": cloud_cover,
            "s2:mgrs_tile": tile,
        },
        "assets": {
            "visual": {"href": f"https://example.test/{item_id}/visual.tif"},
            "B04": {"href": f"https://example.test/{item_id}/B04.tif"},
            "B03": {"href": f"https://example.test/{item_id}/B03.tif"},
            "B02": {"href": f"https://example.test/{item_id}/B02.tif"},
        },
    }


def test_parse_args_defaults_to_curated_hundred_frame_2025_to_2026_render(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    monkeypatch.setattr(sys, "argv", ["khumbu_icefall_sentinel_timelapse.py"])

    args = module.parse_args()

    assert args.start_date == "2025-01-01"
    assert args.end_date == "2026-02-28"
    assert args.max_scenes == 24
    assert args.cloud_cover == pytest.approx(35.0)
    assert args.fps == 10
    assert args.duration == pytest.approx(10.0)
    assert int(round(args.fps * args.duration)) == 100
    assert args.dem_resolution == pytest.approx(5.0)


def test_stac_payload_uses_bbox_dates_and_cloud_filter() -> None:
    module = load_module()

    payload = module.build_stac_payload(
        bbox=module.KHUMBU_BBOX,
        start_date="2023-02-28",
        end_date="2026-02-28",
        cloud_cover=35.0,
        limit=100,
    )

    assert payload["collections"] == ["sentinel-2-l2a"]
    assert payload["bbox"] == list(module.KHUMBU_BBOX)
    assert payload["datetime"] == "2023-02-28T00:00:00Z/2026-02-28T23:59:59Z"
    assert payload["query"]["eo:cloud_cover"]["lte"] == 35.0


def test_parse_args_accepts_dem_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    monkeypatch.setattr(sys, "argv", ["khumbu_icefall_sentinel_timelapse.py", "--dem-resolution", "7.5"])

    args = module.parse_args()

    assert args.dem_resolution == pytest.approx(7.5)


def test_select_scenes_deduplicates_dates_and_evenly_caps() -> None:
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


def test_select_scenes_without_cap_keeps_all_unique_dates() -> None:
    module = load_module()
    features = [
        feature("tile-a", "2025-03-02T04:51:00Z", 8.0),
        feature("tile-b", "2025-03-02T04:52:00Z", 12.0),
        feature("tile-c", "2025-03-07T04:51:00Z", 3.0),
    ]

    selected = module.select_scenes(features, max_scenes=None)

    assert [scene.item_id for scene in selected] == ["tile-a", "tile-c"]


def test_select_scenes_with_cap_prefers_low_cloud_snow_season_scenes() -> None:
    module = load_module()
    features = [
        feature("winter-cloudy", "2025-01-05T04:51:00Z", 18.0),
        feature("winter-clear", "2025-01-10T04:51:00Z", 3.0),
        feature("monsoon-clear", "2025-07-10T04:51:00Z", 1.0),
        feature("autumn-clear", "2025-10-15T04:51:00Z", 4.0),
    ]

    selected = module.select_scenes(features, max_scenes=2)

    assert [scene.item_id for scene in selected] == ["winter-clear", "autumn-clear"]


def test_search_sentinel_scenes_paginates_stac_results(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = pytest.importorskip("requests")

    module = load_module()
    calls: list[dict[str, object]] = []
    pages = [
        {
            "features": [feature("tile-a", "2023-03-03T04:51:00Z", 8.0)],
            "links": [
                {
                    "rel": "next",
                    "href": module.STAC_SEARCH_URL,
                    "method": "POST",
                    "body": {"token": "page-2"},
                }
            ],
        },
        {
            "features": [feature("tile-b", "2023-03-08T04:51:00Z", 7.0)],
            "links": [],
        },
    ]

    class FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_post(url: str, *, json: dict[str, object], timeout: int) -> FakeResponse:
        assert url == module.STAC_SEARCH_URL
        assert timeout == 60
        calls.append(json)
        return FakeResponse(pages.pop(0))

    monkeypatch.setattr(requests, "post", fake_post)

    selected = module.search_sentinel_scenes(
        bbox=module.KHUMBU_BBOX,
        start_date="2023-02-28",
        end_date="2026-02-28",
        cloud_cover=35.0,
        max_scenes=None,
    )

    assert [scene.item_id for scene in selected] == ["tile-a", "tile-b"]
    assert calls == [
        module.build_stac_payload(
            bbox=module.KHUMBU_BBOX,
            start_date="2023-02-28",
            end_date="2026-02-28",
            cloud_cover=35.0,
            limit=100,
        ),
        {"token": "page-2"},
    ]
    assert pages == []


def test_frame_path_is_deterministic(tmp_path: Path) -> None:
    module = load_module()

    assert module.frame_path(tmp_path, 7).name == "frame_0007.png"


def test_format_date_label_prints_clear_year_month_day() -> None:
    module = load_module()

    assert module._format_date_label("2026-02-25") == "2026 February 25"


def test_frame_plan_crossfades_between_dates_and_updates_label_at_midpoint() -> None:
    module = load_module()
    scenes = [
        module.SceneItem("a", "2025-01-01", "2025-01-01T04:51:00Z", 1.0, "45RVL", {}),
        module.SceneItem("b", "2025-02-01", "2025-02-01T04:51:00Z", 1.0, "45RVL", {}),
        module.SceneItem("c", "2025-03-01", "2025-03-01T04:51:00Z", 1.0, "45RVL", {}),
    ]

    plan = module.build_frame_plan(scenes, fps=10, duration=3.0, preview_only=False, crossfade_frames=6)

    assert len(plan) == 30
    assert plan[0].scene.item_id == "a"
    assert plan[0].blend_scene is None
    blended = [item for item in plan if item.blend_scene is not None]
    assert blended
    assert min(item.label_opacity for item in blended) < 0.80
    midpoint_items = [item for item in blended if item.blend_alpha >= 0.5]
    assert midpoint_items
    assert midpoint_items[0].label_scene.item_id == midpoint_items[0].blend_scene.item_id
    assert any(item.blend_peer_index is not None for item in blended)


def test_frame_plan_uses_short_crossfade_window_between_keyframes() -> None:
    module = load_module()
    scenes = [
        module.SceneItem("a", "2025-01-01", "2025-01-01T04:51:00Z", 1.0, "45RVL", {}),
        module.SceneItem("b", "2025-02-01", "2025-02-01T04:51:00Z", 1.0, "45RVL", {}),
        module.SceneItem("c", "2025-03-01", "2025-03-01T04:51:00Z", 1.0, "45RVL", {}),
    ]

    plan = module.build_frame_plan(scenes, fps=10, duration=3.0, preview_only=False, crossfade_frames=4)

    assert plan[1].scene.item_id == "a"
    assert plan[1].blend_scene is None
    assert plan[9].scene.item_id == "a"
    assert plan[9].blend_scene is None
    assert plan[10].blend_scene.item_id == "b"
    assert plan[10].blend_alpha == pytest.approx(0.2)
    assert plan[13].blend_scene.item_id == "b"
    assert plan[13].blend_alpha == pytest.approx(0.8)
    assert plan[14].scene.item_id == "b"
    assert plan[14].blend_scene is None
    assert plan[-6].scene.item_id == "b"
    assert plan[-6].blend_scene is None
    assert plan[-5].blend_scene.item_id == "c"
    assert plan[-5].blend_alpha == pytest.approx(0.2)
    assert plan[-1].blend_scene is None


def test_frame_plan_uses_constant_blend_steps_for_smooth_cadence() -> None:
    module = load_module()
    scenes = [
        module.SceneItem("a", "2025-01-01", "2025-01-01T04:51:00Z", 1.0, "45RVL", {}),
        module.SceneItem("b", "2025-02-01", "2025-02-01T04:51:00Z", 1.0, "45RVL", {}),
        module.SceneItem("c", "2025-03-01", "2025-03-01T04:51:00Z", 1.0, "45RVL", {}),
    ]

    plan = module.build_frame_plan(scenes, fps=10, duration=3.0, preview_only=False)

    segment = [
        item.blend_alpha
        for item in plan
        if item.scene.item_id == "a" and item.blend_scene is not None and item.blend_scene.item_id == "b"
    ]
    steps = np.diff(np.array([0.0, *segment, 1.0], dtype=np.float64))

    assert steps.max() - steps.min() < 1e-12


def test_default_frame_plan_uses_longer_crossfade_window_for_date_handoffs() -> None:
    module = load_module()
    scenes = [
        module.SceneItem("a", "2025-01-01", "2025-01-01T04:51:00Z", 1.0, "45RVL", {}),
        module.SceneItem("b", "2025-02-01", "2025-02-01T04:51:00Z", 1.0, "45RVL", {}),
        module.SceneItem("c", "2025-03-01", "2025-03-01T04:51:00Z", 1.0, "45RVL", {}),
    ]

    plan = module.build_frame_plan(scenes, fps=10, duration=3.0, preview_only=False)
    segment = [
        item.blend_alpha
        for item in plan
        if item.scene.item_id == "a" and item.blend_scene is not None and item.blend_scene.item_id == "b"
    ]

    assert module.DEFAULT_CROSSFADE_FRAMES >= 10
    assert len(segment) == 10
    assert segment == pytest.approx([(index + 1) / 11.0 for index in range(10)])


def test_transition_blend_softens_high_contrast_changes_without_snap() -> None:
    image_module = pytest.importorskip("PIL.Image")

    module = load_module()
    current = np.zeros((2, 3, 3), dtype=np.uint8)
    peer = np.zeros((2, 3, 3), dtype=np.uint8)
    current[:, 0, :] = 255
    peer[:, 2, :] = 255

    early = np.asarray(
        module._blend_transition_images(
            image_module.fromarray(current, mode="RGB"),
            image_module.fromarray(peer, mode="RGB"),
            0.4,
        )
    )
    late = np.asarray(
        module._blend_transition_images(
            image_module.fromarray(current, mode="RGB"),
            image_module.fromarray(peer, mode="RGB"),
            0.6,
        )
    )

    assert 90 < int(early[0, 0, 0]) < 220
    assert 35 < int(early[0, 2, 0]) < 150
    assert 35 < int(late[0, 0, 0]) < 150
    assert 90 < int(late[0, 2, 0]) < 220
    assert int(early[0, 0, 0]) > int(late[0, 0, 0])
    assert int(early[0, 2, 0]) < int(late[0, 2, 0])


def test_label_frame_uses_fixed_single_row_badge_for_all_months(tmp_path: Path) -> None:
    image_module = pytest.importorskip("PIL.Image")

    module = load_module()

    def render_and_dark_bounds(name: str, date_text: str) -> tuple[int, int, int, int]:
        path = tmp_path / name
        image_module.new("RGB", (1600, 1000), (245, 245, 245)).save(path)
        module._label_frame(path, date_text)
        arr = np.asarray(image_module.open(path).convert("RGB"))
        dark = arr.mean(axis=2) < 105
        ys, xs = np.where(dark)
        assert xs.size > 0
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    february_bounds = render_and_dark_bounds("february.png", "2026-02-25")
    september_bounds = render_and_dark_bounds("september.png", "2026-09-25")

    assert february_bounds == september_bounds
    assert february_bounds[3] - february_bounds[1] < 68


def test_label_frame_supports_transition_opacity(tmp_path: Path) -> None:
    image_module = pytest.importorskip("PIL.Image")

    module = load_module()
    full_path = tmp_path / "full.png"
    faded_path = tmp_path / "faded.png"
    image_module.new("RGB", (1600, 1000), module.FRAME_BACKGROUND_RGB).save(full_path)
    image_module.new("RGB", (1600, 1000), module.FRAME_BACKGROUND_RGB).save(faded_path)

    module._label_frame(full_path, "2026-02-25", opacity=1.0)
    module._label_frame(faded_path, "2026-02-25", opacity=0.45)

    full = np.asarray(image_module.open(full_path).convert("RGB"), dtype=np.float32)
    faded = np.asarray(image_module.open(faded_path).convert("RGB"), dtype=np.float32)
    background = np.array(module.FRAME_BACKGROUND_RGB, dtype=np.float32)
    assert float(np.mean(np.abs(faded - background))) < float(np.mean(np.abs(full - background))) * 0.65


def test_lighten_map_image_uses_restrained_shadow_lift_without_changing_background() -> None:
    image_module = pytest.importorskip("PIL.Image")

    module = load_module()
    image = image_module.new("RGB", (5, 4), module.FRAME_BACKGROUND_RGB)
    pixels = image.load()
    pixels[1, 1] = (12, 14, 16)
    pixels[2, 1] = (96, 101, 106)
    pixels[3, 1] = (238, 239, 240)

    lightened = np.asarray(module._lighten_map_image(image).convert("RGB"), dtype=np.float32)

    assert tuple(int(value) for value in lightened[0, 0]) == module.FRAME_BACKGROUND_RGB
    assert 35.0 < float(np.mean(lightened[1, 1])) < 50.0
    assert 12.0 < float(np.mean(lightened[1, 2]) - np.mean([96.0, 101.0, 106.0])) < 24.0
    assert float(np.mean(lightened[1, 3])) < np.mean([238.0, 239.0, 240.0]) + 3.0


def test_render_frames_rejects_empty_frame_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()

    def fail_open_viewer_async(**_kwargs: object) -> object:
        raise AssertionError("viewer should not open for empty frame plans")

    monkeypatch.setattr(module.f3d, "open_viewer_async", fail_open_viewer_async, raising=False)

    with pytest.raises(ValueError, match="^frame_plan must contain at least one frame$"):
        module.render_frames(
            dem_path=tmp_path / "dem.tif",
            overlays={},
            frame_plan=[],
            frames_dir=tmp_path / "frames",
            size=(32, 24),
        )


def test_render_frames_uses_extended_viewer_timeout_for_heavy_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_module = pytest.importorskip("PIL.Image")

    module = load_module()
    scene = module.SceneItem("a", "2025-01-01", "2025-01-01T04:51:00Z", 1.0, "45RVL", {})
    plan = module.build_frame_plan([scene], fps=1, duration=1.0, preview_only=False)
    open_kwargs: dict[str, object] = {}

    class FakeViewer:
        def __enter__(self) -> "FakeViewer":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            pass

        def send_ipc(self, _payload: dict[str, object]) -> None:
            pass

        def load_overlay(self, *_args: object, **_kwargs: object) -> None:
            pass

        def snapshot(self, path: Path, *, width: int, height: int) -> None:
            image_module.new("RGB", (width, height), (96, 100, 104)).save(path)

    def fake_open_viewer_async(**kwargs: object) -> FakeViewer:
        open_kwargs.update(kwargs)
        return FakeViewer()

    monkeypatch.setattr(module.f3d, "open_viewer_async", fake_open_viewer_async, raising=False)
    monkeypatch.setattr(module, "build_render_scene", lambda _path: module.RenderScene(3.0, 3.0, 0.3, (1.0, 0.0, 1.0), 9.0, 30.0))
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(module, "_ground_frame", lambda _path: None)
    monkeypatch.setattr(module, "_label_image", lambda image, _date, *, opacity=1.0: image.convert("RGB"))

    module.render_frames(
        dem_path=tmp_path / "dem.tif",
        overlays={scene.item_id: tmp_path / "a.png"},
        frame_plan=plan,
        frames_dir=tmp_path / "frames",
        size=(1600, 1000),
    )

    assert float(open_kwargs["timeout"]) >= 120.0


def test_render_frames_applies_final_luminance_lift_to_saved_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_module = pytest.importorskip("PIL.Image")

    module = load_module()
    scene = module.SceneItem("a", "2025-01-01", "2025-01-01T04:51:00Z", 1.0, "45RVL", {})
    plan = module.build_frame_plan([scene], fps=1, duration=1.0, preview_only=False)

    class FakeViewer:
        def __enter__(self) -> "FakeViewer":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            pass

        def send_ipc(self, _payload: dict[str, object]) -> None:
            pass

        def load_overlay(self, *_args: object, **_kwargs: object) -> None:
            pass

        def snapshot(self, path: Path, *, width: int, height: int) -> None:
            image_module.new("RGB", (width, height), (12, 14, 16)).save(path)

    monkeypatch.setattr(module.f3d, "open_viewer_async", lambda **_kwargs: FakeViewer(), raising=False)
    monkeypatch.setattr(module, "build_render_scene", lambda _path: module.RenderScene(3.0, 3.0, 0.3, (1.0, 0.0, 1.0), 9.0, 30.0))
    monkeypatch.setattr(module, "_terrain_state", lambda _scene, *, progress: {"cmd": "set_terrain", "progress": progress})
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(module, "_ground_frame", lambda _path: None)
    monkeypatch.setattr(module, "_label_image", lambda image, _date, *, opacity=1.0: image.convert("RGB"))

    module.render_frames(
        dem_path=tmp_path / "dem.tif",
        overlays={scene.item_id: tmp_path / "a.png"},
        frame_plan=plan,
        frames_dir=tmp_path / "frames",
        size=(8, 6),
    )

    saved = np.asarray(image_module.open(tmp_path / "frames" / "frame_0000.png").convert("RGB"), dtype=np.float32)
    assert 35.0 < float(np.mean(saved[0, 0])) < 50.0


def test_render_frames_snapshots_every_output_frame_with_smooth_camera_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_module = pytest.importorskip("PIL.Image")

    module = load_module()
    scenes = [
        module.SceneItem("a", "2025-01-01", "2025-01-01T04:51:00Z", 1.0, "45RVL", {}),
        module.SceneItem("b", "2025-02-01", "2025-02-01T04:51:00Z", 1.0, "45RVL", {}),
    ]
    plan = module.build_frame_plan(scenes, fps=4, duration=2.0, preview_only=False, crossfade_frames=0)
    snapshots: list[Path] = []
    grounded: list[Path] = []
    progress_values: list[float] = []

    class FakeViewer:
        def __enter__(self) -> "FakeViewer":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            pass

        def send_ipc(self, payload: dict[str, object]) -> None:
            if payload.get("cmd") == "set_terrain":
                progress_values.append(float(payload["progress"]))

        def load_overlay(self, *_args: object, **_kwargs: object) -> None:
            pass

        def snapshot(self, path: Path, *, width: int, height: int) -> None:
            snapshots.append(Path(path))
            shade = 80 + len(snapshots) * 12
            image_module.new("RGB", (width, height), (shade, shade, shade)).save(path)

    def fake_terrain_state(_scene: object, *, progress: float) -> dict[str, object]:
        return {"cmd": "set_terrain", "progress": float(progress)}

    monkeypatch.setattr(module.f3d, "open_viewer_async", lambda **_kwargs: FakeViewer(), raising=False)
    monkeypatch.setattr(module, "build_render_scene", lambda _path: module.RenderScene(3.0, 3.0, 0.3, (1.0, 0.0, 1.0), 9.0, 30.0))
    monkeypatch.setattr(module, "_terrain_state", fake_terrain_state)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(module, "_ground_frame", lambda path: grounded.append(Path(path)))

    module.render_frames(
        dem_path=tmp_path / "dem.tif",
        overlays={scene.item_id: tmp_path / f"{scene.item_id}.png" for scene in scenes},
        frame_plan=plan,
        frames_dir=tmp_path / "frames",
        size=(32, 24),
    )

    assert len(plan) == 8
    assert len(snapshots) == len(plan)
    assert len(grounded) == len(plan)
    assert len(progress_values) == len(plan)
    assert progress_values == pytest.approx([module._camera_progress_for_frame(index, len(plan)) for index in range(len(plan))])
    assert len(set(round(value, 6) for value in progress_values)) == len(plan)
    assert len(list((tmp_path / "frames").glob("frame_*.png"))) == len(plan)


def test_render_frames_renders_transition_peers_at_matching_camera_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_module = pytest.importorskip("PIL.Image")

    module = load_module()
    scenes = [
        module.SceneItem("a", "2025-01-01", "2025-01-01T04:51:00Z", 1.0, "45RVL", {}),
        module.SceneItem("b", "2025-02-01", "2025-02-01T04:51:00Z", 1.0, "45RVL", {}),
    ]
    plan = module.build_frame_plan(scenes, fps=4, duration=2.0, preview_only=False, crossfade_frames=2)
    snapshot_records: list[tuple[str, float, Path]] = []
    current_progress = 0.0
    current_overlay = ""

    class FakeViewer:
        def __enter__(self) -> "FakeViewer":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            pass

        def send_ipc(self, payload: dict[str, object]) -> None:
            nonlocal current_progress
            if payload.get("cmd") == "set_terrain":
                current_progress = float(payload["progress"])

        def load_overlay(self, _name: str, path: Path, **_kwargs: object) -> None:
            nonlocal current_overlay
            current_overlay = Path(path).stem

        def snapshot(self, path: Path, *, width: int, height: int) -> None:
            snapshot_records.append((current_overlay, current_progress, Path(path)))
            shade = 90 + len(snapshot_records) * 10
            image_module.new("RGB", (width, height), (shade, shade, shade)).save(path)

    def fake_terrain_state(_scene: object, *, progress: float) -> dict[str, object]:
        return {"cmd": "set_terrain", "progress": float(progress)}

    monkeypatch.setattr(module.f3d, "open_viewer_async", lambda **_kwargs: FakeViewer(), raising=False)
    monkeypatch.setattr(module, "build_render_scene", lambda _path: module.RenderScene(3.0, 3.0, 0.3, (1.0, 0.0, 1.0), 9.0, 30.0))
    monkeypatch.setattr(module, "_terrain_state", fake_terrain_state)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(module, "_ground_frame", lambda _path: None)

    module.render_frames(
        dem_path=tmp_path / "dem.tif",
        overlays={scene.item_id: tmp_path / f"{scene.item_id}.png" for scene in scenes},
        frame_plan=plan,
        frames_dir=tmp_path / "frames",
        size=(32, 24),
    )

    blended_indexes = [index for index, item in enumerate(plan) if item.blend_scene is not None]
    assert blended_indexes == [5, 6]
    assert len(snapshot_records) == len(plan) + len(blended_indexes)

    by_frame: dict[int, list[tuple[str, float, Path]]] = {}
    for record in snapshot_records:
        frame_index = int(record[2].stem.split("_")[1])
        by_frame.setdefault(frame_index, []).append(record)

    for frame_index in blended_indexes:
        records = by_frame[frame_index]
        assert [record[0] for record in records] == ["a", "b"]
        assert records[0][1] == pytest.approx(records[1][1])
        assert records[0][1] == pytest.approx(module._camera_progress_for_frame(frame_index, len(plan)))


def test_copernicus_dem_urls_cover_default_bbox() -> None:
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


def test_dem_grid_cache_token_changes_with_resolution() -> None:
    module = load_module()
    coarse = {
        "crs": "EPSG:32645",
        "transform": (10.0, 0.0, 483770.0, 0.0, -10.0, 3098870.0),
        "width": 688,
        "height": 554,
    }
    fine = {
        "crs": "EPSG:32645",
        "transform": (5.0, 0.0, 483770.0, 0.0, -5.0, 3098870.0),
        "width": 1376,
        "height": 1108,
    }

    assert module._dem_grid_cache_token(coarse) != module._dem_grid_cache_token(fine)


def test_normalize_rgb_to_rgba_preserves_alpha_and_dtype() -> None:
    module = load_module()
    rgb = np.dstack(
        [
            np.linspace(0, 10000, 9, dtype=np.float32).reshape(3, 3),
            np.full((3, 3), 5000, dtype=np.float32),
            np.flipud(np.linspace(0, 10000, 9, dtype=np.float32).reshape(3, 3)),
        ]
    )

    rgba = module.normalize_rgb_to_rgba(rgb)

    assert rgba.shape == (3, 3, 4)
    assert rgba.dtype == np.uint8
    assert np.all(rgba[..., 3] == 255)
    assert rgba[..., :3].max() <= 255


def test_normalize_rgb_to_rgba_can_use_global_stretch() -> None:
    module = load_module()
    dark = np.dstack(
        [
            np.full((3, 3), 20.0, dtype=np.float32),
            np.full((3, 3), 25.0, dtype=np.float32),
            np.full((3, 3), 30.0, dtype=np.float32),
        ]
    )
    bright = dark + 60.0
    stretch = module.RgbStretch(low=(0.0, 0.0, 0.0), high=(100.0, 100.0, 100.0))

    dark_rgba = module.normalize_rgb_to_rgba(dark, stretch=stretch)
    bright_rgba = module.normalize_rgb_to_rgba(bright, stretch=stretch)

    assert float(np.mean(bright_rgba[..., :3])) > float(np.mean(dark_rgba[..., :3])) + 80.0


def test_khumbu_color_grade_turns_beige_snow_cooler_and_brighter() -> None:
    module = load_module()
    sample = np.array(
        [
            [[0.72, 0.68, 0.58], [0.42, 0.39, 0.34]],
            [[0.86, 0.83, 0.76], [0.20, 0.19, 0.18]],
        ],
        dtype=np.float32,
    )

    graded = module._apply_khumbu_color_grade(sample)
    input_luma = np.tensordot(sample, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axes=([2], [0]))
    output_luma = np.tensordot(graded, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axes=([2], [0]))

    assert graded[0, 0, 2] > graded[0, 0, 0]
    assert graded[1, 0, 2] >= graded[1, 0, 0] - 0.015
    assert float(np.mean(output_luma)) > float(np.mean(input_luma)) + 0.045
    assert output_luma[1, 0] > input_luma[1, 0]
    assert output_luma[1, 1] > input_luma[1, 1]


def test_cool_alpine_hdri_writer_creates_radiance_file(tmp_path: Path) -> None:
    module = load_module()

    path = module.ensure_cool_alpine_hdri(tmp_path)

    data = path.read_bytes()
    assert path.name == module.COOL_ALPINE_HDRI_NAME
    assert data.startswith(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 64 +X 128\n")
    assert len(data) == len(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 64 +X 128\n") + 128 * 64 * 4


def test_unique_scenes_by_id_handles_unhashable_asset_dicts() -> None:
    module = load_module()
    scenes = [
        module.SceneItem("a", "2025-03-02", "2025-03-02T04:51:00Z", 3.0, "45RVL", {"visual": "one"}),
        module.SceneItem("a", "2025-03-02", "2025-03-02T04:51:00Z", 3.0, "45RVL", {"visual": "one"}),
        module.SceneItem("b", "2025-03-07", "2025-03-07T04:51:00Z", 4.0, "45RVL", {"visual": "two"}),
    ]

    unique = module.unique_scenes_by_id(scenes)

    assert [scene.item_id for scene in unique] == ["a", "b"]


def test_build_render_scene_targets_relative_relief(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    module = load_module()
    dem_path = tmp_path / "high_mountain.tif"
    dem = np.array([[5000.0, 5200.0], [6400.0, 7000.0]], dtype=np.float32)
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:32645",
        transform=from_origin(0.0, 20.0, 10.0, 10.0),
    ) as dst:
        dst.write(dem, 1)

    scene = module.build_render_scene(dem_path)

    expected = (float(np.percentile(dem, 58.0)) - float(np.percentile(dem, 2.0))) * scene.zscale
    absolute = float(np.percentile(dem, 58.0)) * scene.zscale
    assert scene.target[1] == pytest.approx(expected)
    assert scene.target[1] < absolute * 0.5


def test_build_render_scene_uses_demo_framing_with_grounding_margin(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    module = load_module()
    dem_path = tmp_path / "wide_view_dem.tif"
    dem = np.linspace(5000.0, 7000.0, 12, dtype=np.float32).reshape(3, 4)
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=4,
        height=3,
        count=1,
        dtype="float32",
        crs="EPSG:32645",
        transform=from_origin(0.0, 30.0, 10.0, 10.0),
    ) as dst:
        dst.write(dem, 1)

    scene = module.build_render_scene(dem_path)

    assert scene.radius == pytest.approx(4.0 * 2.48)
    assert scene.fov_deg == pytest.approx(29.0)


def test_terrain_state_uses_oblique_camera_to_reveal_relief() -> None:
    module = load_module()
    scene = module.RenderScene(
        terrain_width=1376.0,
        terrain_height=1107.0,
        zscale=0.34,
        target=(715.5, 220.0, 509.2),
        radius=3715.2,
        fov_deg=30.0,
    )

    state = module._terrain_state(scene, progress=0.5)

    assert state["theta"] == pytest.approx(34.0)
    assert state["phi"] == pytest.approx(module.TIMELAPSE_ORBIT_START_PHI + module.TIMELAPSE_ORBIT_DEGREES * 0.5)
    assert state["ambient"] == pytest.approx(0.48)
    assert state["shadow"] <= 0.60
    assert state["background"] == pytest.approx([value / 255.0 for value in module.FRAME_BACKGROUND_RGB])


def test_camera_progress_for_frame_is_bounded_eased_and_deterministic() -> None:
    module = load_module()

    values = [module._camera_progress_for_frame(index, 6) for index in range(6)]

    assert values[0] == pytest.approx(0.0)
    assert values[-1] == pytest.approx(1.0)
    assert values == pytest.approx([0.0, 0.104, 0.352, 0.648, 0.896, 1.0])
    assert all(0.0 <= value <= 1.0 for value in values)
    assert all(left < right for left, right in zip(values, values[1:]))
    assert module._camera_progress_for_frame(0, 1) == pytest.approx(0.5)


def test_terrain_state_uses_continuous_orbit_and_light_reference_style() -> None:
    module = load_module()
    scene = module.RenderScene(
        terrain_width=1376.0,
        terrain_height=1107.0,
        zscale=0.34,
        target=(715.5, 220.0, 509.2),
        radius=3715.2,
        fov_deg=30.0,
    )

    start = module._terrain_state(scene, progress=0.0)
    mid = module._terrain_state(scene, progress=0.5)
    end = module._terrain_state(scene, progress=1.0)
    pbr = module._terrain_pbr_state(hdri_path=Path("cache") / "cool.hdr")

    assert start["phi"] < mid["phi"] < end["phi"]
    assert end["phi"] - start["phi"] == pytest.approx(module.TIMELAPSE_ORBIT_DEGREES)
    assert start["theta"] == pytest.approx(34.0)
    assert start["background"] == pytest.approx([value / 255.0 for value in module.FRAME_BACKGROUND_RGB])
    assert module.FRAME_BACKGROUND_RGB == (252, 253, 255)
    assert module.BASE_SLAB_RGB == (24, 27, 30)
    assert start["sun_elevation"] >= 35.0
    assert 0.0 < float(start["sun_intensity"]) <= 1.0
    assert start["ambient"] == pytest.approx(0.48)
    assert start["shadow"] <= 0.60
    assert pbr["hdr_path"].endswith("cool.hdr")
    assert pbr["ibl_intensity"] >= 0.40
    assert pbr["exposure"] >= 1.30
    assert pbr["height_ao"]["strength"] <= 0.20
    assert pbr["tonemap"]["temperature"] >= 7000.0


def test_ground_frame_adds_light_background_base_and_contact_shadow(tmp_path: Path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    draw_module = pytest.importorskip("PIL.ImageDraw")

    module = load_module()
    path = tmp_path / "floating_terrain.png"
    image = image_module.new("RGB", (400, 250), module.FRAME_BACKGROUND_RGB)
    draw = draw_module.Draw(image)
    draw.polygon([(120, 45), (275, 70), (330, 150), (205, 205), (85, 145)], fill=(246, 247, 248))
    draw.polygon([(180, 75), (265, 92), (240, 145), (155, 132)], fill=(45, 58, 63))
    image.save(path)

    module._ground_frame(path)

    grounded = np.asarray(image_module.open(path).convert("RGB"))
    assert tuple(int(value) for value in grounded[0, 0]) == module.FRAME_BACKGROUND_RGB
    dark_pixels = np.all(grounded < np.array([32, 34, 34], dtype=np.uint8), axis=2)
    assert int(dark_pixels.sum()) > 600
    assert grounded[70, 130, 0] > 220


def test_prepare_overlay_prefers_raw_reflectance_bands_over_clipped_visual_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_module = pytest.importorskip("PIL.Image")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    module = load_module()
    dem_path = tmp_path / "dem.tif"
    dem = np.array([[5000.0, 5200.0], [6400.0, 7000.0]], dtype=np.float32)
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:32645",
        transform=from_origin(0.0, 20.0, 10.0, 10.0),
    ) as dst:
        dst.write(dem, 1)

    scene = module.SceneItem(
        "scene-with-clipped-visual",
        "2025-10-23",
        "2025-10-23T04:48:51Z",
        12.0,
        "45RVM",
        {
            "visual": "https://example.test/visual.tif",
            "B04": "https://example.test/B04.tif",
            "B03": "https://example.test/B03.tif",
            "B02": "https://example.test/B02.tif",
        },
    )
    calls: list[str] = []

    def fake_read_reprojected_asset(href: str, _dem_profile: dict[str, object], *, indexes: int | list[int]) -> np.ndarray:
        calls.append(href.rsplit("/", 1)[-1])
        if href.endswith("visual.tif"):
            return np.full((3, 2, 2), 255, dtype=np.uint8)
        if href.endswith("B04.tif"):
            return np.array([[1200, 3800], [7200, 13200]], dtype=np.uint16)
        if href.endswith("B03.tif"):
            return np.array([[1300, 4200], [7600, 14100]], dtype=np.uint16)
        if href.endswith("B02.tif"):
            return np.array([[1500, 4700], [8300, 15200]], dtype=np.uint16)
        raise AssertionError(f"unexpected href {href}")

    monkeypatch.setattr(module, "_read_reprojected_asset", fake_read_reprojected_asset)

    overlay_path = module.prepare_overlay(scene, dem_path=dem_path, cache_dir=tmp_path / "cache", force=False)

    rgba = np.asarray(image_module.open(overlay_path).convert("RGBA"))
    assert calls == ["B04.tif", "B03.tif", "B02.tif"]
    assert float(np.median(rgba[..., :3])) < 245.0
    assert float(np.std(rgba[..., :3])) > 20.0


def test_sign_planetary_computer_url_retries_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = pytest.importorskip("requests")

    module = load_module()
    sleeps: list[float] = []
    responses: list[object] = []

    class FakeResponse:
        def __init__(self, status_code: int, payload: dict[str, str] | None = None, retry_after: str | None = None) -> None:
            self.status_code = status_code
            self._payload = payload or {}
            self.headers = {"Retry-After": retry_after} if retry_after is not None else {}

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code} error", response=self)

        def json(self) -> dict[str, str]:
            return self._payload

    responses.extend([FakeResponse(429, retry_after="0.25"), FakeResponse(200, {"href": "https://signed.example.test/B04.tif"})])

    def fake_get(url: str, *, params: dict[str, str], timeout: int) -> FakeResponse:
        assert url == module.PLANETARY_COMPUTER_SIGN_URL
        assert params == {"href": "https://example.test/B04.tif"}
        assert timeout == 30
        return responses.pop(0)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(module.time, "sleep", lambda delay: sleeps.append(float(delay)))

    signed = module.sign_planetary_computer_url("https://example.test/B04.tif")

    assert signed == "https://signed.example.test/B04.tif"
    assert sleeps == [0.25]
    assert responses == []
