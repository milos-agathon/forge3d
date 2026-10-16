"""Exercise the shared sky state through the shipped viewer and GPU frames.

Each visual claim needs a control that responds to the tested property:
direction, zero intensity, a Moon-direction sky, or a sprite-free dark sky.
Restoration references precede the first observation. Native SIDERA goldens
are separate regression coverage; they do not exercise these viewer adapters.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from forge3d import astro, geometry

from tests import test_m06_full_geospatial_viewer as live
from tests.test_terrain_viewer_pbr import find_viewer_binary


pytestmark = pytest.mark.interactive_viewer


def _manual_body(body: str) -> dict:
    azimuth, elevation, _ = astro.body_position(
        body, datetime(2026, 12, 24, 22, tzinfo=timezone.utc), 52.3676, 4.9041
    )
    # Astronomical north is -z; the legacy manual convention starts at +z.
    return {
        "cmd": "set_terrain_sun",
        "azimuth_deg": (180.0 - azimuth) % 360.0,
        "elevation_deg": elevation,
    }


def _write_terrain(tmp_path):
    terrain = tmp_path / "terrain.tif"
    live._write_geotiff(
        terrain,
        live._heightfield(192, 128),
        origin_x=0,
        origin_y=0,
        span_x=4000,
        span_y=2600,
    )
    return terrain


def _terrain(viewer, tmp_path, *, sky=False):
    terrain = _write_terrain(tmp_path)
    live._load_terrain(viewer, terrain)
    viewer.ok(
        {
            "cmd": "set_terrain_camera",
            "phi_deg": 145.0,
            # theta is measured from zenith: 85 degrees exposes the actual sky.
            "theta_deg": 85.0 if sky else 52.0,
            "radius": 5000.0,
            "fov_deg": 50.0,
            "target": [2000, 50, 1300],
        }
    )
    return terrain


def _terrain_params(viewer):
    start = len(viewer.lines)
    viewer.ok({"cmd": "get_terrain_params"})
    return viewer.wait_until(
        lambda: next(
            (
                line
                for line in viewer.lines[start:]
                if line.startswith("[terrain] phi=")
            ),
            None,
        ),
        description="executed terrain parameter query",
    )


def _moonlight_intensity():
    # The public ephemeris plus the committed night.rs illuminance model give
    # an independent manual-light reference for the actual GPU upload path.
    _, phase, _ = astro.moon_phase(datetime(2026, 12, 24, 22, tzinfo=timezone.utc))
    altitude = _manual_body("moon")["elevation_deg"]
    assert 0.0 < altitude < 90.0
    airmass = (1.0 - 0.96 * math.sin(math.radians(90.0 - altitude)) ** 2) ** -0.5
    attenuation = 10.0 ** (-0.4 * (0.026 * phase + 4.0e-9 * phase**4))
    return float(
        np.float32(
            attenuation * 10.0 ** (-0.4 * 0.172 * airmass) / 10.0 ** (-0.4 * 0.172)
        )
    )


def _assert_moon_terrain_params(params, *, daylight=1.0):
    moon = _manual_body("moon")
    assert f"sun_az={moon['azimuth_deg']:.1f}" in params
    assert f"sun_el={moon['elevation_deg']:.1f}" in params
    assert f"intensity={daylight * _moonlight_intensity():.2f}" in params
    assert "source=solar_time" in params


class _ConsoleViewer:
    """The console event loop owns stdin; the IPC event loop does not read it."""

    log_tail = live.ViewerProcess.log_tail
    wait_until = live.ViewerProcess.wait_until

    def __init__(self, terrain):
        self.process = subprocess.Popen(
            [
                str(find_viewer_binary()),
                "--size",
                "640x400",
                "--sky",
                "off",
                "--lit-sun",
                "2",
                "--terrain",
                str(terrain),
            ],
            cwd=live.ROOT,
            env=os.environ.copy(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self.lines = []

        def drain():
            for line in self.process.stdout:
                self.lines.append(line.rstrip())

        self.reader = threading.Thread(target=drain, daemon=True)
        self.reader.start()

    def ready(self):
        self.wait_until(
            lambda: next(
                (line for line in self.lines if line.startswith("[terrain] Loaded:")),
                None,
            ),
            description="console terrain attachment",
        )

    def stats(self):
        line = self.wait_until(
            lambda: next(
                (
                    line
                    for line in self.lines
                    if line.startswith("FORGE3D_VIEWER_ADAPTER ")
                ),
                None,
            ),
            description="console physical adapter selection",
        )
        values = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', line))
        return {
            f"adapter_{key}": (
                int(value, 16)
                if key in {"vendor", "device"}
                else json.loads(value) if value.startswith('"') else value
            )
            for key, value in values.items()
        }

    def command(self, text):
        self.process.stdin.write(text + "\n")
        self.process.stdin.flush()

    def snapshot(self, path):
        path.unlink(missing_ok=True)
        self.command(f":snapshot {path} 512x320")

        def loaded():
            try:
                with Image.open(path) as image:
                    image.load()
                    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            except (OSError, ValueError):
                return None

        pixels = self.wait_until(loaded, description=f"console snapshot {path.name}")
        shutil.copy2(path, live.ARTIFACT_DIR / path.name)
        return pixels

    def close(self):
        try:
            if self.process.poll() is None:
                self.command(":quit")
                # Same shutdown contract as the existing ViewerProcess.
                self.process.wait(timeout=8.0)
        except (OSError, subprocess.TimeoutExpired):
            self.process.terminate()
            self.process.wait(timeout=8.0)
        finally:
            self.reader.join()
            self.process.stdin.close()
            live.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            (live.ARTIFACT_DIR / "sky-lighting-intensity-command.log").write_text(
                "\n".join(self.lines) + "\n", encoding="utf-8"
            )


def _record_adapter(viewer, name):
    stats = viewer.stats()
    assert stats["adapter_device_type"] in {"DiscreteGpu", "IntegratedGpu"}, stats
    executable = Path(viewer.process.args[0]).resolve()
    stats["viewer_executable"] = str(executable)
    stats["viewer_binary_sha256"] = hashlib.sha256(executable.read_bytes()).hexdigest()
    live.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (live.ARTIFACT_DIR / f"sky-lighting-{name}-adapter.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )


def _observation(hour: int) -> dict:
    return {
        "cmd": "set_observation",
        "year": 2026,
        # Deep night with a high, nearly full Moon gives independent, visible
        # signals for celestial sprites and directional geometry lighting.
        "month": 12,
        "day": 24,
        "hour": hour,
        "minute": 0,
        "second": 0.0,
        "latitude_deg": 52.3676,
        "longitude_deg": 4.9041,
    }


@pytest.mark.parametrize("scene", ["sky", "mesh", "terrain"])
def test_manual_night_manual_gpu_transitions(scene, tmp_path, monkeypatch):
    # Reuse the execution-completed IPC, frame barriers, and artifact capture
    # already exercised by the hardware viewer lane.
    monkeypatch.setenv("FORGE3D_DETERMINISTIC", "1")
    monkeypatch.setattr(live, "_viewer_binary", find_viewer_binary)
    viewer = live.ViewerProcess(
        extra_args=[
            "--viz",
            "lit",
            "--lit-ibl",
            "0",
            "--brdf",
            "ggx",
            "--sky",
            "on" if scene == "sky" else "off",
        ]
    )
    try:
        viewer.ok({"cmd": "set_taa_enabled", "enabled": False})
        _record_adapter(viewer, scene)
        if scene == "mesh":
            # A curved surface has a moving GGX highlight even with the
            # viewer's black untextured albedo. Controls below must prove
            # sensitivity to both direction and moonlight intensity.
            sphere = geometry.primitive_mesh("sphere")
            obj = tmp_path / "lit-sphere.obj"
            obj.write_text(
                "\n".join(
                    ["v " + " ".join(map(str, p)) for p in sphere.positions]
                    + ["vn " + " ".join(map(str, n)) for n in sphere.normals]
                    + [
                        "f " + " ".join(f"{int(i)+1}//{int(i)+1}" for i in face)
                        for face in np.asarray(sphere.indices).reshape(-1, 3)
                    ]
                ),
                encoding="utf-8",
            )
            live._record_input(obj)
            viewer.ok({"cmd": "load_obj", "path": str(obj)})
            moon = _manual_body("moon")
            azimuth, elevation = np.deg2rad(
                [moon["azimuth_deg"], moon["elevation_deg"]]
            )
            moon_direction = np.array(
                [
                    np.cos(elevation) * np.sin(azimuth),
                    np.sin(elevation),
                    np.cos(elevation) * np.cos(azimuth),
                ]
            )
            # set_look_at reduces a requested eye through the existing orbit
            # mapping (-z, -y, -x). Face the actual eye toward this surface.
            viewer.ok(
                {
                    "cmd": "cam_lookat",
                    "eye": (3 * moon_direction[[2, 1, 0]]).tolist(),
                    "target": [0, 0, 0],
                }
            )
        else:
            _terrain(viewer, tmp_path, sky=scene == "sky")

        def capture(name):
            image = viewer.snapshot(tmp_path / f"{scene}-{name}.png")
            return image[: image.shape[0] // 4] if scene == "sky" else image

        manual = {"cmd": "lit_sun", "azimuth_deg": 90.0, "elevation_deg": 30.0}
        viewer.ok(
            {
                "cmd": "set_terrain_sun",
                "azimuth_deg": 90.0,
                "elevation_deg": 30.0,
                "intensity": 1.0,
            }
        )
        baseline = capture("manual")
        # These references are captured before ANY observation. A broken
        # transition cannot contaminate both reference and restored frames.
        viewer.ok({**manual, "azimuth_deg": 270.0})
        direction_control = capture("direction-control")
        assert not np.array_equal(baseline, direction_control)
        viewer.ok(
            {
                "cmd": "set_terrain_sun",
                "azimuth_deg": 90.0,
                "elevation_deg": 30.0,
                "intensity": 2.0,
            }
        )
        brighter = capture("intensity-two")
        if scene != "sky":
            assert not np.array_equal(brighter, baseline)
        viewer.ok(_observation(22))
        night = capture("night")
        assert not np.array_equal(baseline, night)
        if scene != "sky":
            if scene == "terrain":
                _assert_moon_terrain_params(_terrain_params(viewer))
            else:
                # Match the observed frame with an independently requested
                # Moon direction AND modeled intensity. A stale manual
                # direction at the same dim intensity must visibly differ.
                moonlight = _moonlight_intensity()
                assert 0.0 < moonlight < 1.0
                viewer.ok({**_manual_body("moon"), "intensity": moonlight})
                np.testing.assert_array_equal(capture("manual-moonlight"), night)
                viewer.ok(
                    {
                        "cmd": "set_terrain_sun",
                        "azimuth_deg": 90.0,
                        "elevation_deg": 30.0,
                        "intensity": moonlight,
                    }
                )
                assert not np.array_equal(capture("stale-direction-moonlight"), night)
            viewer.ok({**_manual_body("moon"), "intensity": 0.0})
            zero = capture("zero-intensity")
            assert not np.array_equal(night, zero)
            assert night.sum() > zero.sum()
            viewer.ok(_observation(22))
        viewer.ok(manual)
        restored = capture("restored")
        np.testing.assert_array_equal(restored, baseline)

        # Reapplying the same sky/site restores the celestial overlay after
        # manual control cleared it, and a day observation restores daylight.
        viewer.ok(_observation(22))
        night_again = capture("night-again")
        np.testing.assert_array_equal(night_again, night)
        viewer.ok(_observation(8 if scene == "mesh" else 12))
        day = capture("day")
        assert not np.array_equal(day, night)

        if scene == "terrain":
            viewer.ok(_observation(22))
            viewer.ok(
                {"cmd": "set_terrain", "sun_azimuth": 90.0, "sun_elevation": 30.0}
            )
            partial = viewer.snapshot(tmp_path / "terrain-partial-manual.png")
            np.testing.assert_array_equal(partial, baseline)
    finally:
        viewer.close(f"sky-lighting-{scene}.log")


def test_sky_retains_true_sun_and_clears_visible_night_sprites(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE3D_DETERMINISTIC", "1")
    monkeypatch.setattr(live, "_viewer_binary", find_viewer_binary)
    viewer = live.ViewerProcess(extra_args=["--sky", "on"])
    try:
        _record_adapter(viewer, "true-sun")
        viewer.ok({"cmd": "set_taa_enabled", "enabled": False})
        _terrain(viewer, tmp_path, sky=True)

        def capture(name):
            image = viewer.snapshot(tmp_path / f"true-sun-{name}.png")
            return image[: image.shape[0] // 4]

        # A dark manual reference before the first observation makes stale
        # sprites observable instead of hiding them under a daylight sky.
        solar = _manual_body("sun")
        viewer.ok({**solar, "intensity": 1.0})
        clear = capture("no-sprites")
        viewer.ok(_observation(22))
        night = capture("night")
        assert (
            night.sum() > clear.sum()
        ), "the sampled sky must contain visible celestial sprites"

        # Clear directly from observation into a dark manual sky, before a
        # bright Moon-direction control could conceal surviving instances.
        viewer.ok({**solar, "intensity": 1.0})
        np.testing.assert_array_equal(capture("sprites-cleared"), clear)
        viewer.ok(_observation(22))
        np.testing.assert_array_equal(capture("night-again"), night)

        viewer.ok({**_manual_body("moon"), "intensity": 1.0})
        moon_control = capture("moon-direction")
        assert not np.array_equal(night, moon_control)
        assert (
            night.mean() < moon_control.mean()
        ), "true Sun below horizon must keep sky dark"

    finally:
        viewer.close("sky-lighting-true-sun.log")


def test_observation_survives_late_terrain_and_unrelated_edits(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE3D_DETERMINISTIC", "1")
    monkeypatch.setattr(live, "_viewer_binary", find_viewer_binary)
    viewer = live.ViewerProcess(extra_args=["--sky", "off"])
    try:
        _record_adapter(viewer, "late-terrain")
        viewer.ok({"cmd": "set_taa_enabled", "enabled": False})
        viewer.ok(_observation(22))
        _terrain(viewer, tmp_path)
        # No sun command may intervene between terrain attachment and capture.
        attached = viewer.snapshot(tmp_path / "late-terrain-night.png")
        assert "source=solar_time" in _terrain_params(viewer)
        viewer.ok(_observation(22))
        np.testing.assert_array_equal(
            viewer.snapshot(tmp_path / "late-terrain-night-again.png"), attached
        )
        lighting = _terrain_params(viewer).split(" | ")[1]
        viewer.ok({"cmd": "set_terrain", "zscale": 2.0})
        edited_params = _terrain_params(viewer)
        assert "zscale=2.00" in edited_params
        assert edited_params.split(" | ")[1] == lighting
        edited = viewer.snapshot(tmp_path / "unrelated-terrain-edit.png")
        assert not np.array_equal(edited, attached)
        viewer.ok(_observation(22))
        np.testing.assert_array_equal(
            viewer.snapshot(tmp_path / "unrelated-terrain-reobserved.png"), edited
        )
        viewer.ok({"cmd": "set_terrain", "zscale": 1.0})
        assert _terrain_params(viewer).split(" | ")[1] == lighting
        np.testing.assert_array_equal(
            viewer.snapshot(tmp_path / "unrelated-terrain-restored.png"), attached
        )
        viewer.ok({**_manual_body("moon"), "intensity": 0.0})
        zero = viewer.snapshot(tmp_path / "late-terrain-zero.png")
        assert attached.sum() > zero.sum()
    finally:
        viewer.close("sky-lighting-late-terrain.log")


def test_terrain_source_labels_and_scene_preset(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE3D_DETERMINISTIC", "1")
    monkeypatch.setattr(live, "_viewer_binary", find_viewer_binary)
    viewer = live.ViewerProcess(extra_args=["--sky", "off"])
    try:
        _record_adapter(viewer, "source-preset")
        viewer.ok({"cmd": "set_taa_enabled", "enabled": False})
        terrain = _terrain(viewer, tmp_path)
        # solar_time is an accepted explicit source label even for manual
        # angles; arbitrary custom strings are rejected at the IPC boundary.
        custom = {
            "cmd": "set_terrain_sun",
            "azimuth_deg": 90.0,
            "elevation_deg": 30.0,
            "intensity": 1.0,
            "source": "solar_time",
        }
        # Preserve the operator log's unscaled daylight meaning below horizon,
        # even though the terrain receives zero effective lighting there.
        start = len(viewer.lines)
        viewer.ok({**custom, "elevation_deg": -30.0, "intensity": 2.0})
        sun_log = viewer.wait_until(
            lambda: next(
                (
                    line
                    for line in viewer.lines[start:]
                    if line.startswith("[terrain] Sun:")
                ),
                None,
            ),
            description="executed below-horizon sun log",
        )
        assert sun_log.endswith("int=2.00")
        assert "intensity=0.00" in _terrain_params(viewer)
        viewer.ok(custom)
        assert "source=solar_time" in _terrain_params(viewer)
        viewer.ok({"cmd": "set_terrain", "sun_intensity": 2.0})
        assert "intensity=2.00" in _terrain_params(viewer)
        assert "source=manual_angles" in _terrain_params(viewer)
        viewer.ok(custom)
        live._load_terrain(viewer, terrain)
        assert "source=manual_angles" in _terrain_params(viewer)
        live._camera(viewer, target=[2000, 50, 1300], phi=145, radius=5000)
        viewer.ok(custom)
        baseline = viewer.snapshot(tmp_path / "preset-manual-reference.png")
        viewer.ok(_observation(22))
        viewer.ok(
            {
                "cmd": "set_scene_review_state",
                "state": {
                    "base_state": {
                        "preset": {"sun_azimuth": 90.0, "sun_elevation": 30.0}
                    },
                },
            }
        )
        assert "source=manual_angles" in _terrain_params(viewer)
        np.testing.assert_array_equal(
            viewer.snapshot(tmp_path / "preset-restored.png"), baseline
        )
    finally:
        viewer.close("sky-lighting-source-preset.log")


def test_lit_sun_text_and_startup_commands_update_terrain(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE3D_DETERMINISTIC", "1")
    terrain = _write_terrain(tmp_path)
    viewer = _ConsoleViewer(terrain)
    try:
        viewer.ready()
        _record_adapter(viewer, "intensity-command")
        startup_two = viewer.snapshot(tmp_path / "lit-sun-startup-two.png")
        viewer.command(":lit-sun 1")
        one = viewer.snapshot(tmp_path / "lit-sun-text-one.png")
        assert not np.array_equal(one, startup_two)
        assert startup_two.sum() > one.sum()
        viewer.command(":lit-sun 0")
        zero = viewer.snapshot(tmp_path / "lit-sun-text-zero.png")
        assert one.sum() > zero.sum()
        viewer.command(":lit-sun 2")
        np.testing.assert_array_equal(
            viewer.snapshot(tmp_path / "lit-sun-text-two.png"), startup_two
        )
    finally:
        viewer.close()
