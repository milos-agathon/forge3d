# Khumbu Smooth Orbit Light Render Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Khumbu Sentinel timelapse render a smooth per-frame camera orbit with a lighter reference-style look.

**Architecture:** Keep the change inside the existing Khumbu example and its focused test module. Add a deterministic camera progress helper, tune the example-owned lighting/background constants, and change `render_frames()` from unique-date still snapshots to per-output-frame viewer snapshots with camera-matched transition peers.

**Tech Stack:** Python 3.10+, `numpy`, `Pillow`, forge3d viewer IPC, `pytest`, `ffmpeg` for the final MP4.

---

## File Structure

- Modify `examples/khumbu_icefall_sentinel_timelapse.py`: light render constants, `_camera_progress_for_frame()`, `_terrain_state()`, `_terrain_pbr_state()`, `_ground_frame()`, and `render_frames()`.
- Modify `tests/test_khumbu_icefall_sentinel_timelapse.py`: replace fixed-camera/keyframe snapshot expectations with smooth-orbit and per-frame render tests.
- Keep `docs/superpowers/specs/2026-05-05-khumbu-smooth-orbit-light-render-design.md` unchanged during implementation.
- Leave unrelated dirty files alone unless a failing test proves they are directly involved.

---

### Task 1: Camera Progress And Light Render Contracts

**Files:**
- Modify: `tests/test_khumbu_icefall_sentinel_timelapse.py`
- Modify: `examples/khumbu_icefall_sentinel_timelapse.py`

- [ ] **Step 1: Write failing tests for smooth camera progress and lighter terrain state**

Add these tests near the existing terrain state tests in `tests/test_khumbu_icefall_sentinel_timelapse.py`:

```python
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
    pbr = module._terrain_pbr_state()

    assert start["phi"] < mid["phi"] < end["phi"]
    assert end["phi"] - start["phi"] == pytest.approx(module.TIMELAPSE_ORBIT_DEGREES)
    assert start["theta"] == pytest.approx(34.0)
    assert start["background"] == pytest.approx([value / 255.0 for value in module.FRAME_BACKGROUND_RGB])
    assert module.FRAME_BACKGROUND_RGB == (250, 251, 250)
    assert module.BASE_SLAB_RGB == (18, 19, 20)
    assert start["ambient"] >= 0.34
    assert start["shadow"] <= 0.78
    assert pbr["exposure"] >= 1.05
    assert pbr["height_ao"]["strength"] <= 0.28
```

- [ ] **Step 2: Run the new tests and verify they fail for missing helper and old constants**

Run:

```bash
python -m pytest tests/test_khumbu_icefall_sentinel_timelapse.py::test_camera_progress_for_frame_is_bounded_eased_and_deterministic tests/test_khumbu_icefall_sentinel_timelapse.py::test_terrain_state_uses_continuous_orbit_and_light_reference_style -q
```

Expected: FAIL because `_camera_progress_for_frame` is not defined and the current terrain style remains darker.

- [ ] **Step 3: Add light constants and camera progress helper**

In `examples/khumbu_icefall_sentinel_timelapse.py`, replace the existing fixed camera and render color constants near the top of the file with:

```python
DEFAULT_CROSSFADE_FRAMES = 4
DEFAULT_SIZE = (1600, 1000)
FIXED_TIMELAPSE_CAMERA_PROGRESS = 0.5
TIMELAPSE_ORBIT_START_PHI = 202.0
TIMELAPSE_ORBIT_DEGREES = 44.0
TRUE_COLOR_BAND_ASSETS = ("B04", "B03", "B02")
OVERLAY_PIPELINE_VERSION = "raw-true-color-v2"
FRAME_BACKGROUND_RGB = (250, 251, 250)
TERRAIN_BACKGROUND_COLOR = tuple(value / 255.0 for value in FRAME_BACKGROUND_RGB)
BASE_SLAB_RGB = (18, 19, 20)
```

Add this helper immediately after `_smoothstep_scalar()`:

```python
def _camera_progress_for_frame(frame_index: int, total_frames: int) -> float:
    if int(total_frames) <= 1:
        return 0.5
    linear = min(max(float(frame_index) / float(int(total_frames) - 1), 0.0), 1.0)
    return _smoothstep_scalar(linear)
```

- [ ] **Step 4: Update terrain and PBR state for continuous orbit and lighter render**

Replace `_terrain_state()` with:

```python
def _terrain_state(scene: RenderScene, *, progress: float) -> dict[str, object]:
    progress = min(max(float(progress), 0.0), 1.0)
    theta = 34.0
    phi = TIMELAPSE_ORBIT_START_PHI + progress * TIMELAPSE_ORBIT_DEGREES
    return {
        "cmd": "set_terrain",
        "phi": phi,
        "theta": theta,
        "radius": scene.radius,
        "fov": scene.fov_deg,
        "sun_azimuth": 312.0,
        "sun_elevation": 24.0,
        "sun_intensity": 1.52,
        "ambient": 0.36,
        "zscale": scene.zscale,
        "shadow": 0.74,
        "background": list(TERRAIN_BACKGROUND_COLOR),
        "water_level": -999999.0,
        "target": list(scene.target),
    }
```

Replace `_terrain_pbr_state()` with:

```python
def _terrain_pbr_state() -> dict[str, object]:
    return {
        "cmd": "set_terrain_pbr",
        "enabled": True,
        "shadow_technique": "pcss",
        "shadow_map_res": 4096,
        "exposure": 1.08,
        "msaa": 8,
        "ibl_intensity": 0.08,
        "normal_strength": 1.24,
        "height_ao": {"enabled": True, "directions": 8, "steps": 24, "max_distance": 320.0, "strength": 0.26, "resolution_scale": 0.78},
        "tonemap": {"operator": "aces", "white_point": 5.2, "white_balance_enabled": True, "temperature": 6250.0, "tint": 0.0},
        "sky": {"enabled": False},
    }
```

- [ ] **Step 5: Update the existing terrain state test to the new light style**

In `test_terrain_state_uses_oblique_camera_to_reveal_relief`, replace the assertions after `state = module._terrain_state(scene, progress=0.5)` with:

```python
    assert state["theta"] == pytest.approx(34.0)
    assert state["phi"] == pytest.approx(module.TIMELAPSE_ORBIT_START_PHI + module.TIMELAPSE_ORBIT_DEGREES * 0.5)
    assert state["ambient"] >= 0.34
    assert state["shadow"] <= 0.78
    assert state["background"] == pytest.approx([value / 255.0 for value in module.FRAME_BACKGROUND_RGB])
```

- [ ] **Step 6: Run the Task 1 tests and verify they pass**

Run:

```bash
python -m pytest tests/test_khumbu_icefall_sentinel_timelapse.py::test_camera_progress_for_frame_is_bounded_eased_and_deterministic tests/test_khumbu_icefall_sentinel_timelapse.py::test_terrain_state_uses_continuous_orbit_and_light_reference_style tests/test_khumbu_icefall_sentinel_timelapse.py::test_terrain_state_uses_oblique_camera_to_reveal_relief -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

Run:

```bash
git add examples/khumbu_icefall_sentinel_timelapse.py tests/test_khumbu_icefall_sentinel_timelapse.py
git commit -m "Tune Khumbu orbit and light render state"
```

Expected: commit succeeds with only the Khumbu example and test file staged.

---

### Task 2: Per-Frame Snapshot Tests

**Files:**
- Modify: `tests/test_khumbu_icefall_sentinel_timelapse.py`
- Modify: `examples/khumbu_icefall_sentinel_timelapse.py`

- [ ] **Step 1: Replace the unique-keyframe snapshot test with per-frame snapshot coverage**

In `tests/test_khumbu_icefall_sentinel_timelapse.py`, replace `test_render_frames_snapshots_only_unique_keyframes` with:

```python
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
```

- [ ] **Step 2: Replace the fixed-camera test with transition peer progress coverage**

Replace `test_render_frames_keeps_camera_fixed_across_keyframes_to_avoid_shadow_ghosts` with:

```python
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
```

- [ ] **Step 3: Run the replacement tests and verify they fail on old render behavior**

Run:

```bash
python -m pytest tests/test_khumbu_icefall_sentinel_timelapse.py::test_render_frames_snapshots_every_output_frame_with_smooth_camera_progress tests/test_khumbu_icefall_sentinel_timelapse.py::test_render_frames_renders_transition_peers_at_matching_camera_progress -q
```

Expected: FAIL because `render_frames()` still snapshots unique keyframes and uses fixed camera progress.

---

### Task 3: Per-Frame Render Implementation

**Files:**
- Modify: `examples/khumbu_icefall_sentinel_timelapse.py`
- Test: `tests/test_khumbu_icefall_sentinel_timelapse.py`

- [ ] **Step 1: Add a reusable snapshot helper**

Add this helper above `render_frames()` in `examples/khumbu_icefall_sentinel_timelapse.py`:

```python
def _snapshot_scene_frame(
    viewer: Any,
    *,
    render_scene: RenderScene,
    scene: SceneItem,
    overlay_path: Path,
    raw_path: Path,
    width: int,
    height: int,
    camera_progress: float,
    settle: float,
) -> None:
    viewer.send_ipc(_terrain_state(render_scene, progress=camera_progress))
    viewer.load_overlay("khumbu_sentinel", overlay_path, extent=(0.0, 0.0, 1.0, 1.0), opacity=1.0, preserve_colors=True)
    viewer.send_ipc({"cmd": "set_overlays_enabled", "enabled": True})
    viewer.send_ipc({"cmd": "set_overlay_solid", "solid": False})
    viewer.send_ipc({"cmd": "set_overlay_preserve_colors", "preserve_colors": True})
    time.sleep(float(settle))
    viewer.snapshot(raw_path, width=width, height=height)
```

- [ ] **Step 2: Replace `render_frames()` with per-output-frame rendering**

Delete this temporary compatibility constant from the top of `examples/khumbu_icefall_sentinel_timelapse.py`:

```python
FIXED_TIMELAPSE_CAMERA_PROGRESS = 0.5
```

Replace the body of `render_frames()` with:

```python
def render_frames(
    *,
    dem_path: Path,
    overlays: dict[str, Path],
    frame_plan: list[FramePlanItem],
    frames_dir: Path,
    size: tuple[int, int],
    settle: float = 0.08,
) -> Path:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for existing in frames_dir.glob("frame_*.png"):
        existing.unlink()
    raw_dir = frames_dir / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for existing in raw_dir.glob("raw_*.png"):
        existing.unlink()

    render_scene = build_render_scene(dem_path)
    width, height = int(size[0]), int(size[1])
    preview_path = frames_dir.parent / "khumbu_icefall_preview.png"
    total_frames = len(frame_plan)

    from PIL import Image

    with f3d.open_viewer_async(terrain_path=dem_path, width=width, height=height, timeout=60.0) as viewer:
        viewer.send_ipc(_terrain_pbr_state())
        time.sleep(1.2)
        for index, item in enumerate(frame_plan):
            camera_progress = _camera_progress_for_frame(index, total_frames)
            current_raw_path = raw_dir / f"raw_{index:04d}_a.png"
            _snapshot_scene_frame(
                viewer,
                render_scene=render_scene,
                scene=item.scene,
                overlay_path=overlays[item.scene.item_id],
                raw_path=current_raw_path,
                width=width,
                height=height,
                camera_progress=camera_progress,
                settle=settle,
            )
            _ground_frame(current_raw_path)

            if item.blend_scene is not None:
                peer_raw_path = raw_dir / f"raw_{index:04d}_b.png"
                _snapshot_scene_frame(
                    viewer,
                    render_scene=render_scene,
                    scene=item.blend_scene,
                    overlay_path=overlays[item.blend_scene.item_id],
                    raw_path=peer_raw_path,
                    width=width,
                    height=height,
                    camera_progress=camera_progress,
                    settle=settle,
                )
                _ground_frame(peer_raw_path)
                with Image.open(current_raw_path) as current_image:
                    current = current_image.convert("RGB")
                with Image.open(peer_raw_path) as peer_image:
                    peer = peer_image.convert("RGB")
                image = _blend_transition_images(current, peer, float(item.blend_alpha))
            else:
                with Image.open(current_raw_path) as current_image:
                    image = current_image.convert("RGB")

            image = _label_image(image, item.label_scene.date, opacity=item.label_opacity)
            out_path = frame_path(frames_dir, index)
            image.save(out_path, compress_level=1)
            if index == total_frames // 2:
                shutil.copyfile(out_path, preview_path)
            if index == 0 or (index + 1) % 24 == 0 or index == total_frames - 1:
                _safe_progress(f"\r[Khumbu] finished frame {index + 1}/{total_frames} {item.label_scene.date}", end="")
    _safe_progress("")
    return preview_path
```

- [ ] **Step 3: Run the per-frame render tests and verify they pass**

Run:

```bash
python -m pytest tests/test_khumbu_icefall_sentinel_timelapse.py::test_render_frames_snapshots_every_output_frame_with_smooth_camera_progress tests/test_khumbu_icefall_sentinel_timelapse.py::test_render_frames_renders_transition_peers_at_matching_camera_progress -q
```

Expected: PASS.

- [ ] **Step 4: Run all Khumbu tests and capture regressions**

Run:

```bash
python -m pytest tests/test_khumbu_icefall_sentinel_timelapse.py -q
```

Expected: tests unrelated to fixed camera pass or identify exact assertions that still encode the old dark/fixed-camera behavior.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add examples/khumbu_icefall_sentinel_timelapse.py tests/test_khumbu_icefall_sentinel_timelapse.py
git commit -m "Render Khumbu orbit frames per output frame"
```

Expected: commit succeeds with the per-frame rendering change and tests.

---

### Task 4: Adjust Grounding Test For Lighter Style

**Files:**
- Modify: `tests/test_khumbu_icefall_sentinel_timelapse.py`
- Modify: `examples/khumbu_icefall_sentinel_timelapse.py`

- [ ] **Step 1: Keep the grounding test aligned with the lighter background**

In `test_ground_frame_adds_gray_background_base_and_contact_shadow`, rename the test to:

```python
def test_ground_frame_adds_light_background_base_and_contact_shadow(tmp_path: Path) -> None:
```

Keep the existing final assertions:

```python
    assert tuple(int(value) for value in grounded[0, 0]) == module.FRAME_BACKGROUND_RGB
    dark_pixels = np.all(grounded < np.array([32, 34, 34], dtype=np.uint8), axis=2)
    assert int(dark_pixels.sum()) > 600
    assert grounded[70, 130, 0] > 220
```

- [ ] **Step 2: Soften `_ground_frame()` shadows if the lighter-style tests show excessive darkness**

If the grounding test fails because the dark base remains too heavy for the lighter render target, use these exact alpha constants inside `_ground_frame()`:

```python
        contact_alpha = np.clip(np.asarray(contact, dtype=np.float32) * 0.30, 0, 72).astype(np.uint8)
```

and:

```python
    shadow_alpha = np.clip(np.asarray(shadow_mask, dtype=np.float32) * 0.32, 0, 84).astype(np.uint8)
```

The expected result is a visible slab/contact shadow with the corner background equal to `(250, 251, 250)`.

- [ ] **Step 3: Run all Khumbu tests**

Run:

```bash
python -m pytest tests/test_khumbu_icefall_sentinel_timelapse.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit Task 4**

Run:

```bash
git add examples/khumbu_icefall_sentinel_timelapse.py tests/test_khumbu_icefall_sentinel_timelapse.py
git commit -m "Align Khumbu tests with lighter render style"
```

Expected: commit succeeds if Task 4 changed files after Task 3.

---

### Task 5: Output Verification And Final Render

**Files:**
- Verify: `examples/khumbu_icefall_sentinel_timelapse.py`
- Verify: `tests/test_khumbu_icefall_sentinel_timelapse.py`
- Output: `examples/out/khumbu_icefall_sentinel_timelapse/khumbu_icefall_sentinel_timelapse.mp4`

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
python -m pytest tests/test_khumbu_icefall_sentinel_timelapse.py -q
```

Expected: PASS.

- [ ] **Step 2: Run CLI smoke**

Run:

```bash
python examples/khumbu_icefall_sentinel_timelapse.py --help
```

Expected: exits 0 and shows `--fps`, `--duration`, `--size`, `--preview-only`, `--frames-only`, and `--force`.

- [ ] **Step 3: Render a short smoke sequence**

Run:

```bash
python examples/khumbu_icefall_sentinel_timelapse.py --max-scenes 3 --duration 2 --fps 12 --size 800 500 --frames-only
```

Expected: exits 0 and writes 24 PNGs under `examples/out/khumbu_icefall_sentinel_timelapse/frames`.

- [ ] **Step 4: Inspect the short smoke sequence metadata**

Run:

```bash
(Get-ChildItem examples/out/khumbu_icefall_sentinel_timelapse/frames/frame_*.png).Count
```

Expected: `24`.

- [ ] **Step 5: Render the requested MP4**

Run:

```bash
python examples/khumbu_icefall_sentinel_timelapse.py
```

Expected: exits 0 and writes `examples/out/khumbu_icefall_sentinel_timelapse/khumbu_icefall_sentinel_timelapse.mp4`.

- [ ] **Step 6: Verify MP4 frame rate, frame count, and duration**

Run:

```bash
ffprobe -v error -show_entries format=duration -show_entries stream=codec_type,width,height,r_frame_rate,avg_frame_rate,nb_frames -of json examples/out/khumbu_icefall_sentinel_timelapse/khumbu_icefall_sentinel_timelapse.mp4
```

Expected: `width` is `1600`, `height` is `1000`, `avg_frame_rate` is `24/1`, `nb_frames` is `384`, and duration is close to `16.000000`.

- [ ] **Step 7: Extract visual check frames**

Run:

```bash
New-Item -ItemType Directory -Force -Path examples/out/khumbu_icefall_sentinel_timelapse/diagnostics | Out-Null
ffmpeg -y -ss 0 -i examples/out/khumbu_icefall_sentinel_timelapse/khumbu_icefall_sentinel_timelapse.mp4 -frames:v 1 examples/out/khumbu_icefall_sentinel_timelapse/diagnostics/khumbu_orbit_start.png
ffmpeg -y -ss 8 -i examples/out/khumbu_icefall_sentinel_timelapse/khumbu_icefall_sentinel_timelapse.mp4 -frames:v 1 examples/out/khumbu_icefall_sentinel_timelapse/diagnostics/khumbu_orbit_mid.png
ffmpeg -y -ss 15.9 -i examples/out/khumbu_icefall_sentinel_timelapse/khumbu_icefall_sentinel_timelapse.mp4 -frames:v 1 examples/out/khumbu_icefall_sentinel_timelapse/diagnostics/khumbu_orbit_end.png
```

Expected: three PNGs exist and show a visible camera orbit with a white/light background.

- [ ] **Step 8: Review the final diff**

Run:

```bash
git diff -- examples/khumbu_icefall_sentinel_timelapse.py tests/test_khumbu_icefall_sentinel_timelapse.py
```

Expected: diff is limited to smooth orbit rendering, lighter render constants, and matching tests.
