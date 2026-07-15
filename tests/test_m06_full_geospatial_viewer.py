"""Required live acceptance for M-06 Option-2 full geospatial viewer.

The required CI lane sets RUN_M06_VIEWER_CI=1, builds a fresh release viewer,
and rejects every skip through scripts/assert_junit_zero_skips.py.  These tests
use only public NDJSON IPC and emitted artifacts so the evidence represents the
shipped executable rather than an in-process test double.
"""

from __future__ import annotations

import json
import os
import queue
import re
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.environ.get("FORGE3D_M06_ARTIFACT_DIR", ROOT / "tests/artifacts/m06"))
RUN_REQUIRED = os.environ.get("RUN_M06_VIEWER_CI") == "1"
pytestmark = [
    pytest.mark.interactive_viewer,
    pytest.mark.skipif(not RUN_REQUIRED, reason="required M-06 hardware lane only"),
]


def _viewer_binary() -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    path = ROOT / "target" / "release" / f"interactive_viewer{suffix}"
    assert path.is_file(), f"fresh release viewer is required: {path}"
    return path


class ViewerProcess:
    def __init__(self, *, effects: bool = False) -> None:
        args = [str(_viewer_binary()), "--ipc-port", "0", "--size", "640x400"]
        if effects:
            args += [
                "--gi",
                "ssgi:on",
                "--gi",
                "ssr:on",
                "--ssgi-temporal-enable",
                "true",
                "--ssgi-temporal-alpha",
                "0.88",
                "--ssr-enable",
                "true",
                "--fog",
                "on",
                "--fog-temporal",
                "0.8",
            ]
        env = os.environ.copy()
        env.setdefault("WGPU_BACKEND", "vulkan")
        self.process = subprocess.Popen(
            args,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert self.process.stdout is not None
        self.lines: list[str] = []
        self._line_queue: queue.Queue[str] = queue.Queue()

        def drain() -> None:
            assert self.process.stdout is not None
            for line in iter(self.process.stdout.readline, ""):
                self.lines.append(line.rstrip())
                self._line_queue.put(line)

        threading.Thread(target=drain, daemon=True).start()
        ready = re.compile(r"FORGE3D_VIEWER_READY\s+port=(\d+)")
        deadline = time.monotonic() + 45.0
        port: int | None = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise AssertionError(f"viewer exited before READY\n{self.log_tail()}")
            try:
                line = self._line_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            match = ready.search(line)
            if match:
                port = int(match.group(1))
                break
        assert port is not None, f"viewer READY timeout\n{self.log_tail()}"
        self.socket = socket.create_connection(("127.0.0.1", port), timeout=20.0)
        self._recv_buffer = b""
        self.wait_frames(2)

    def log_tail(self, count: int = 100) -> str:
        return "\n".join(self.lines[-count:])

    def send(self, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        assert self.process.poll() is None, f"viewer is not running\n{self.log_tail()}"
        self.socket.settimeout(timeout)
        self.socket.sendall((json.dumps(payload, allow_nan=False) + "\n").encode("utf-8"))
        while b"\n" not in self._recv_buffer:
            chunk = self.socket.recv(65536)
            assert chunk, f"viewer IPC closed\n{self.log_tail()}"
            self._recv_buffer += chunk
        line, self._recv_buffer = self._recv_buffer.split(b"\n", 1)
        response = json.loads(line.decode("utf-8"))
        return response

    def ok(self, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        response = self.send(payload, timeout)
        assert response.get("ok"), f"IPC rejected {payload.get('cmd')}: {response}\n{self.log_tail()}"
        return response

    def stats(self) -> dict[str, Any]:
        return self.ok({"cmd": "get_stats"})["stats"]

    def wait_until(
        self, predicate: Callable[[], Any], *, timeout: float = 45.0, description: str
    ) -> Any:
        deadline = time.monotonic() + timeout
        last: Any = None
        while time.monotonic() < deadline:
            assert self.process.poll() is None, f"viewer crashed while waiting for {description}\n{self.log_tail()}"
            last = predicate()
            if last is not None:
                return last
            time.sleep(0.08)
        raise AssertionError(f"timeout waiting for {description}; last={last!r}\n{self.log_tail()}")

    def wait_frames(self, count: int = 2, timeout: float = 30.0) -> dict[str, Any]:
        start = self.stats()["frame_count"] if hasattr(self, "socket") else 0
        return self.wait_until(
            lambda: _matching_stats(self, lambda stats: stats["frame_count"] >= start + count),
            timeout=timeout,
            description=f"{count} new frames",
        )

    def snapshot(self, path: Path, *, width: int = 512, height: int = 320) -> np.ndarray:
        path.unlink(missing_ok=True)
        self.ok({"cmd": "snapshot", "path": str(path), "width": width, "height": height})

        def loaded() -> np.ndarray | None:
            if not path.is_file() or path.stat().st_size < 1000:
                return None
            try:
                with Image.open(path) as image:
                    image.load()
                    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            except (OSError, ValueError):
                return None

        return self.wait_until(loaded, timeout=45.0, description=f"snapshot {path.name}")

    def close(self, artifact_name: str) -> None:
        try:
            if self.process.poll() is None:
                self.ok({"cmd": "close"}, timeout=5.0)
        except (OSError, AssertionError):
            pass
        try:
            self.socket.close()
        except OSError:
            pass
        try:
            self.process.wait(timeout=8.0)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=8.0)
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_DIR / artifact_name).write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def _heightfield(width: int, height: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    field = (
        50.0
        + 20.0 * np.sin(xx / max(width, 1) * np.pi * 3.0)
        + 14.0 * np.cos(yy / max(height, 1) * np.pi * 2.0)
        + xx * 0.08
    )
    return np.asarray(field, dtype=np.float32)


def _write_local_tiff(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values).save(path)


def _write_geotiff(
    path: Path,
    values: np.ndarray,
    *,
    origin_x: float,
    origin_y: float,
    span_x: float,
    span_y: float,
    crs: str | None = "EPSG:32633",
    mirrored_x: bool = False,
    south_up: bool = False,
) -> None:
    import rasterio
    from affine import Affine

    height, width = values.shape
    pixel_x = span_x / width
    pixel_y = span_y / height
    transform = Affine(
        -pixel_x if mirrored_x else pixel_x,
        0.0,
        origin_x,
        0.0,
        pixel_y if south_up else -pixel_y,
        origin_y,
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        transform=transform,
        crs=crs,
    ) as dataset:
        dataset.write(values, 1)


def _metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    assert a.shape == b.shape
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    x = a.astype(np.float64).ravel()
    y = b.astype(np.float64).ravel()
    c1 = 0.01**2
    c2 = 0.03**2
    mean_x, mean_y = float(x.mean()), float(y.mean())
    var_x, var_y = float(x.var()), float(y.var())
    cov = float(np.mean((x - mean_x) * (y - mean_y)))
    ssim = ((2 * mean_x * mean_y + c1) * (2 * cov + c2)) / (
        (mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2)
    )
    return {"ssim": ssim, "mae": float(diff.mean()), "max_abs": float(diff.max())}


def _write_json(name: str, payload: dict[str, Any]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _matching_stats(
    viewer: ViewerProcess, predicate: Callable[[dict[str, Any]], bool]
) -> dict[str, Any] | None:
    stats = viewer.stats()
    return stats if predicate(stats) else None


def _active_volumetrics(viewer: ViewerProcess) -> dict[str, Any] | None:
    report = viewer.ok({"cmd": "get_terrain_volumetrics_report"})[
        "terrain_volumetrics_report"
    ]
    return report if report["active_volume_count"] > 0 else None


def _load_terrain(viewer: ViewerProcess, path: Path) -> dict[str, Any]:
    before = viewer.stats()["frame_count"]
    viewer.ok({"cmd": "load_terrain", "path": str(path)}, timeout=45.0)
    return viewer.wait_until(
        lambda: _matching_stats(
            viewer,
            lambda stats: stats["frame_count"] >= before + 2
            and stats["active_camera"] == "terrain",
        ),
        timeout=45.0,
        description=f"terrain {path.name}",
    )


def _camera(viewer: ViewerProcess, *, target: list[float], phi: float, radius: float) -> None:
    viewer.ok(
        {
            "cmd": "set_terrain_camera",
            "phi_deg": phi,
            "theta_deg": 52.0,
            "radius": radius,
            "fov_deg": 50.0,
            "target": target,
        }
    )
    viewer.wait_frames(2)


def _write_obj(path: Path) -> None:
    path.write_text(
        """v -1 0 -1
v 1 0 -1
v 1 0 1
v -1 0 1
v -1 2 -1
v 1 2 -1
v 1 2 1
v -1 2 1
f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 5 1 4 8
""",
        encoding="utf-8",
    )


def _write_las(path: Path, *, count: int, center_x: float, center_y: float) -> None:
    import laspy

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([center_x, center_y, 0.0])
    las = laspy.LasData(header)
    side = int(np.ceil(np.sqrt(count)))
    idx = np.arange(count, dtype=np.int64)
    gx = idx % side
    gy = idx // side
    las.x = center_x + (gx / max(side - 1, 1) - 0.5) * 900.0
    las.y = center_y + (gy / max(side - 1, 1) - 0.5) * 600.0
    las.z = 90.0 + 30.0 * np.sin(gx / 21.0) + 20.0 * np.cos(gy / 17.0)
    las.red = np.full(count, 65535, dtype=np.uint16)
    las.green = np.asarray((gy % 256) * 257, dtype=np.uint16)
    las.blue = np.asarray((gx % 256) * 257, dtype=np.uint16)
    las.intensity = np.asarray(idx % 65535, dtype=np.uint16)
    las.write(path)


def _scatter_payload(width: int, height: int) -> dict[str, Any]:
    positions = [[-0.6, 0.0, -0.6], [0.6, 0.0, -0.6], [0.0, 2.2, 0.0], [0.6, 0.0, 0.6], [-0.6, 0.0, 0.6]]
    normals = [[0.0, 1.0, 0.0]] * len(positions)
    indices = [0, 1, 2, 1, 3, 2, 3, 4, 2, 4, 0, 2, 0, 4, 3, 0, 3, 1]
    transforms = []
    for x, z, scale in [(0.28, 0.3, 9.0), (0.5, 0.5, 12.0), (0.72, 0.64, 8.0)]:
        transforms.append(
            [scale, 0.0, 0.0, width * x, 0.0, scale, 0.0, 80.0, 0.0, 0.0, scale, height * z, 0.0, 0.0, 0.0, 1.0]
        )
    return {
        "cmd": "set_terrain_scatter",
        "batches": [
            {
                "name": "m06-nonsquare",
                "color": [0.05, 0.95, 0.1, 1.0],
                "max_draw_distance": 10000.0,
                "transforms": transforms,
                "levels": [{"positions": positions, "normals": normals, "indices": indices, "max_distance": 10000.0}],
            }
        ],
    }


def test_m06_georeferencing_fail_closed_and_local_pixel_lock(tmp_path: Path) -> None:
    viewer = ViewerProcess()
    try:
        values = _heightfield(160, 96)
        local = tmp_path / "local-no-transform.tif"
        translated = tmp_path / "translated-arbitrary-crs.tif"
        invalid = tmp_path / "invalid-mirrored.tif"
        _write_local_tiff(local, values)
        _write_geotiff(translated, values, origin_x=6_378_000.0, origin_y=1_000_000.0, span_x=160.0, span_y=96.0, crs="EPSG:32633")
        _write_geotiff(invalid, values, origin_x=6_378_000.0, origin_y=1_000_000.0, span_x=160.0, span_y=96.0, mirrored_x=True)

        _load_terrain(viewer, local)
        _camera(viewer, target=[80.0, 45.0, 48.0], phi=137.0, radius=280.0)
        local_frame = viewer.snapshot(tmp_path / "local.png")

        _load_terrain(viewer, translated)
        _camera(viewer, target=[6_378_080.0, 45.0, -999_952.0], phi=137.0, radius=280.0)
        translated_frame = viewer.snapshot(tmp_path / "translated.png")
        parity = _metrics(local_frame, translated_frame)
        assert parity["ssim"] >= 0.999, parity
        assert parity["mae"] <= 0.5 / 255.0, parity

        before = viewer.stats()
        rejected = viewer.send({"cmd": "load_terrain", "path": str(invalid)})
        assert rejected.get("ok") is False, rejected
        assert "unsupported_axis_orientation" in rejected.get("error", ""), rejected
        after = viewer.stats()
        assert after["tracked_buffer_count"] == before["tracked_buffer_count"]
        assert after["tracked_texture_count"] == before["tracked_texture_count"]
        _write_json("georeferencing.json", {"backend": os.environ.get("WGPU_BACKEND"), "parity": parity, "rejection": rejected, "before": before, "after": after})
    finally:
        viewer.close("georeferencing-viewer.log")


def test_m06_nonzero_origin_effects_orbit_no_flash_and_scatter(tmp_path: Path) -> None:
    viewer = ViewerProcess(effects=True)
    try:
        values = _heightfield(192, 96)
        terrain = tmp_path / "non-square-effects.tif"
        # Pixel aspect is 2:1 while the physical footprint is 4:3; this catches
        # any path that reconstructs depth from raster dimensions.
        _write_geotiff(terrain, values, origin_x=6_378_000.0, origin_y=1_000_000.0, span_x=4000.0, span_y=3000.0)
        _load_terrain(viewer, terrain)
        target = [6_380_000.0, 50.0, -998_500.0]
        _camera(viewer, target=target, phi=0.0, radius=6000.0)
        viewer.ok({"cmd": "set_taa_enabled", "enabled": True})
        viewer.ok(
            {
                "cmd": "set_terrain_pbr",
                "enabled": True,
                "shadow_technique": "csm",
                "shadow_map_res": 1024,
                "dof": {"enabled": True, "f_stop": 8.0, "focus_distance": 6000.0, "quality": "medium"},
                "volumetrics": {
                    "enabled": True,
                    "mode": "heterogeneous",
                    "density": 0.006,
                    "steps": 24,
                    "half_res": True,
                    "density_volumes": [
                        {"preset": "fog_bank", "center": [96.0, 16.0, 48.0], "size": [60.0, 28.0, 28.0], "resolution": [32, 16, 16], "density_scale": 0.35, "seed": 7}
                    ],
                },
            }
        )
        enabled = viewer.wait_until(
            lambda: _matching_stats(
                viewer,
                lambda stats: all(
                    stats[key]
                    for key in (
                        "taa_enabled",
                        "ssgi_enabled",
                        "ssgi_temporal_enabled",
                        "ssr_enabled",
                        "fog_enabled",
                    )
                ),
            ),
            timeout=45.0,
            description="all temporal/effect telemetry flags",
        )
        report = viewer.wait_until(
            lambda: _active_volumetrics(viewer),
            timeout=45.0,
            description="heterogeneous density volume",
        )

        viewer.wait_frames(8)
        no_overlay = viewer.snapshot(tmp_path / "no-overlay.png")
        overlay = tmp_path / "terrain-overlay.png"
        Image.new("RGBA", (64, 64), (238, 44, 166, 255)).save(overlay)
        viewer.ok(
            {
                "cmd": "load_overlay",
                "name": "m06-overlay",
                "path": str(overlay),
                "extent": [0.2, 0.2, 0.8, 0.8],
                "opacity": 0.9,
                "z_order": 0,
            }
        )
        viewer.ok({"cmd": "set_overlays_enabled", "enabled": True})
        viewer.wait_frames(4)
        with_overlay = viewer.snapshot(tmp_path / "with-overlay.png")
        overlay_delta = _metrics(no_overlay, with_overlay)
        assert overlay_delta["mae"] > 0.00005, overlay_delta

        viewer.ok(
            {
                "cmd": "set_scene_review_state",
                "state": {
                    "review_layers": [
                        {
                            "id": "effects-label",
                            "name": "Effects label",
                            "labels": [
                                {
                                    "kind": "point",
                                    "text": "M06",
                                    "world_pos": [target[0], 80.0, target[2]],
                                }
                            ],
                        }
                    ],
                    "variants": [
                        {
                            "id": "effects",
                            "active_layer_ids": ["effects-label"],
                        }
                    ],
                    "active_variant_id": "effects",
                },
            }
        )
        labels = viewer.wait_until(
            lambda: viewer.ok({"cmd": "list_review_layers"})["review_layers"]
            or None,
            timeout=45.0,
            description="non-zero-origin label layer",
        )

        no_scatter = viewer.snapshot(tmp_path / "no-scatter.png")
        viewer.ok(_scatter_payload(192, 96))
        viewer.wait_frames(5)
        with_scatter = viewer.snapshot(tmp_path / "with-scatter.png")
        scatter_delta = _metrics(no_scatter, with_scatter)
        assert scatter_delta["mae"] > 0.00005, scatter_delta

        viewer.wait_frames(10)
        start = viewer.stats()
        orbit_frames = []
        for phi in range(0, 361, 30):
            _camera(viewer, target=target, phi=float(phi), radius=6000.0)
            orbit_frames.append(viewer.stats())
        end = viewer.stats()
        assert end["camera_rebase_count"] == start["camera_rebase_count"]
        assert end["history_invalidation_count"] == start["history_invalidation_count"]
        resource_counts = {
            (stats["tracked_buffer_count"], stats["tracked_texture_count"])
            for stats in orbit_frames
        }
        assert len(resource_counts) == 1, resource_counts

        frames = []
        for index in range(3):
            viewer.wait_frames(3)
            frames.append(viewer.snapshot(tmp_path / f"stationary-{index}.png"))
        no_flash = [_metrics(frames[index], frames[index + 1]) for index in range(2)]
        assert min(metric["ssim"] for metric in no_flash) >= 0.97, no_flash
        assert max(metric["mae"] for metric in no_flash) <= 0.03, no_flash
        assert enabled["within_host_visible_budget"] is True
        _write_json("effects-orbit.json", {"backend": os.environ.get("WGPU_BACKEND"), "enabled": enabled, "volumetrics": report, "labels": labels, "overlay_delta": overlay_delta, "scatter_delta": scatter_delta, "start": start, "end": end, "orbit": orbit_frames, "no_flash": no_flash})
    finally:
        viewer.close("effects-orbit-viewer.log")


def test_m06_object_pointcloud_coexistence_and_allocation_free_rebases(tmp_path: Path) -> None:
    viewer = ViewerProcess()
    try:
        values = _heightfield(192, 128)
        terrain = tmp_path / "coexistence.tif"
        origin_x, origin_y = 6_378_000.0, 1_000_000.0
        span_x, span_y = 4000.0, 2600.0
        center_x, center_z = origin_x + span_x / 2.0, -origin_y + span_y / 2.0
        _write_geotiff(terrain, values, origin_x=origin_x, origin_y=origin_y, span_x=span_x, span_y=span_y)
        _load_terrain(viewer, terrain)
        _camera(viewer, target=[center_x, 60.0, center_z], phi=145.0, radius=5000.0)

        obj = tmp_path / "anchored-box.obj"
        _write_obj(obj)
        viewer.ok({"cmd": "load_obj", "path": str(obj)})
        viewer.ok({"cmd": "set_transform", "translation": [center_x, 170.0, center_z], "scale": [120.0, 120.0, 120.0]})
        object_stats = viewer.wait_until(
            lambda: _matching_stats(
                viewer,
                lambda stats: stats["scene_has_mesh"]
                and not stats["transform_is_identity"],
            ),
            timeout=45.0,
            description="anchored object transform",
        )
        assert object_stats["active_camera"] == "terrain"

        point_count = 500_000
        las = tmp_path / "earth-scale-500k.las"
        _write_las(las, count=point_count, center_x=center_x, center_y=center_z)
        viewer.ok({"cmd": "load_point_cloud", "path": str(las), "point_size": 5.0, "max_points": point_count, "color_mode": "rgb"}, timeout=90.0)
        loaded = viewer.wait_until(
            lambda: _matching_stats(
                viewer, lambda stats: stats["point_cloud_point_count"] == point_count
            ),
            timeout=120.0,
            description="500k point cloud",
        )
        assert loaded["active_camera"] == "terrain"
        assert loaded["point_cloud_source_bytes"] == point_count * 48
        assert loaded["point_cloud_render_cache_bytes"] == point_count * 48
        assert loaded["point_cloud_gpu_instance_bytes"] == point_count * 48
        assert loaded["within_host_visible_budget"] is True

        viewer.ok({"cmd": "set_point_cloud_params", "visible": False})
        hidden = viewer.snapshot(tmp_path / "point-hidden.png")
        viewer.ok({"cmd": "set_point_cloud_params", "visible": True, "point_size": 6.0})
        visible = viewer.snapshot(tmp_path / "point-visible.png")
        point_delta = _metrics(hidden, visible)
        assert point_delta["mae"] > 0.0001, point_delta

        stable = viewer.stats()
        counts = []
        anchors = []
        for index in range(10):
            target_x = center_x + (1500.0 if index % 2 == 0 else 0.0)
            _camera(viewer, target=[target_x, 60.0, center_z], phi=145.0, radius=5000.0)
            stats = viewer.stats()
            counts.append([stats["tracked_buffer_count"], stats["tracked_texture_count"]])
            anchors.append(stats["camera_anchor_origin"])
        assert all(count == counts[0] for count in counts), counts
        assert counts[0] == [stable["tracked_buffer_count"], stable["tracked_texture_count"]]
        assert viewer.stats()["camera_rebase_count"] >= stable["camera_rebase_count"] + 9
        _write_json("coexistence-rebases.json", {"backend": os.environ.get("WGPU_BACKEND"), "object": object_stats, "loaded": loaded, "point_delta": point_delta, "counts": counts, "anchors": anchors})
    finally:
        viewer.close("coexistence-viewer.log")


def _color_centroid(frame: np.ndarray, channel: int) -> tuple[int, np.ndarray | None]:
    other = [index for index in range(3) if index != channel]
    mask = (frame[..., channel] > 0.75) & (frame[..., other[0]] < 0.35) & (frame[..., other[1]] < 0.35)
    coords = np.argwhere(mask)
    return int(coords.shape[0]), None if coords.size == 0 else coords.mean(axis=0)


def test_m06_live_millimetre_vectors_and_scene_review_transaction(tmp_path: Path) -> None:
    viewer = ViewerProcess()
    try:
        values = np.zeros((128, 256), dtype=np.float32)
        terrain = tmp_path / "millimetre.tif"
        origin_x, origin_y = 6_378_137.0, 1_000_000.0
        span_x, span_y = 0.02, 0.01
        _write_geotiff(terrain, values, origin_x=origin_x, origin_y=origin_y, span_x=span_x, span_y=span_y, crs=None)
        _load_terrain(viewer, terrain)
        center = [origin_x + span_x / 2.0, 0.002, -origin_y + span_y / 2.0]
        _camera(viewer, target=center, phi=90.0, radius=0.03)
        separated = {
            "cmd": "add_vector_overlay",
            "id": 700,
            "name": "one-millimetre",
            "vertices": [
                [center[0] - 0.0005, center[1], center[2], 1.0, 0.0, 0.0, 1.0, 71],
                [center[0] + 0.0005, center[1], center[2], 0.0, 0.0, 1.0, 1.0, 72],
            ],
            "indices": [0, 1],
            "primitive": "points",
            "point_size": 12.0,
            "drape": False,
        }
        viewer.ok(separated)
        viewer.wait_frames(3)
        precise = viewer.snapshot(tmp_path / "one-mm.png", width=640, height=400)
        red_count, red_centroid = _color_centroid(precise, 0)
        blue_count, blue_centroid = _color_centroid(precise, 2)
        assert red_count > 0 and blue_count > 0, (red_count, blue_count)
        assert red_centroid is not None and blue_centroid is not None
        separation_px = float(np.linalg.norm(red_centroid - blue_centroid))
        assert separation_px >= 1.0, separation_px

        viewer.ok({"cmd": "remove_vector_overlay", "id": 700})
        viewer.ok(
            {
                **separated,
                "id": 701,
                "name": "zero-millimetre-control",
                "vertices": [
                    [center[0], center[1], center[2], 1.0, 0.0, 0.0, 1.0, 73],
                    [center[0], center[1], center[2], 0.0, 0.0, 1.0, 1.0, 74],
                ],
            }
        )
        viewer.wait_frames(3)
        control = viewer.snapshot(tmp_path / "zero-mm.png", width=640, height=400)
        control_red, _ = _color_centroid(control, 0)
        control_blue, _ = _color_centroid(control, 2)
        assert min(control_red, control_blue) == 0, (control_red, control_blue)

        baseline_state = {
            "cmd": "set_scene_review_state",
            "state": {
                "review_layers": [{"id": "accepted", "name": "Accepted", "labels": [{"kind": "point", "text": "valid", "world_pos": center}]}],
                "variants": [{"id": "baseline", "active_layer_ids": ["accepted"]}],
                "active_variant_id": "baseline",
            },
        }
        viewer.ok(baseline_state)
        baseline = viewer.wait_until(
            lambda: {
                "layers": viewer.ok({"cmd": "list_review_layers"})["review_layers"],
                "variants": viewer.ok({"cmd": "list_scene_variants"})["scene_variants"],
                "active": viewer.ok({"cmd": "get_active_scene_variant"})["active_scene_variant"],
            }
            if viewer.ok({"cmd": "get_active_scene_variant"})["active_scene_variant"] == "baseline"
            else None,
            timeout=45.0,
            description="baseline scene-review transaction",
        )
        rejected_state = {
            "cmd": "set_scene_review_state",
            "state": {
                "review_layers": [{"id": "poison", "labels": [{"kind": "point", "text": "too far", "world_pos": [center[0] + 2_000_000.0, center[1], center[2]]}]}],
                "variants": [{"id": "rejected", "active_layer_ids": ["poison"]}],
                "active_variant_id": "rejected",
            },
        }
        viewer.ok(rejected_state)
        viewer.wait_frames(5)
        after = {
            "layers": viewer.ok({"cmd": "list_review_layers"})["review_layers"],
            "variants": viewer.ok({"cmd": "list_scene_variants"})["scene_variants"],
            "active": viewer.ok({"cmd": "get_active_scene_variant"})["active_scene_variant"],
        }
        assert after == baseline
        _write_json("millimetre-transaction.json", {"backend": os.environ.get("WGPU_BACKEND"), "red_pixels": red_count, "blue_pixels": blue_count, "separation_px": separation_px, "control_red_pixels": control_red, "control_blue_pixels": control_blue, "baseline": baseline, "after_rejected": after})
    finally:
        viewer.close("millimetre-viewer.log")
