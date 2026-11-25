"""Continuity at the native frame and durable poster-tile boundaries."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import threading
from time import perf_counter

import numpy as np
import pytest

import forge3d as f3d
from _terrain_runtime import _adapter_is_terrain_safe


WINDOW = 32  # WELFORD_WINDOW in render_terrain.rs
FRAME_BOUNDARIES = (1, WINDOW - 1, WINDOW + 1, 2 * WINDOW - 1, 2 * WINDOW)
ARRAYS = ("rgba", "albedo", "normal", "depth", "radiance", "luminance_variance")
STATISTICS = (
    "frames", "variance", "converged", "convergence_metric", "traversal",
    "reservoir_valid_count", "reservoir_m_min", "reservoir_m_max",
    "minmax_pyramid_bytes", "gpu_resource_bytes",
)


def _scene():
    # The split environment produces nonzero Welford moments. Sunlight keeps
    # reservoir history live. The first eligible variance check is frame 64.
    dem = np.arange(16, dtype=np.float32).reshape(4, 4) / 16
    env = np.zeros((2, 4, 3), dtype=np.float32)
    env[:, 2:] = 2
    camera = {
        "origin": (0.0, 10.0, 0.0), "look_at": (0.0, 0.0, 0.0),
        "up": (0.0, 0.0, -1.0), "fov_y": 30.0,
    }
    settings = dict(
        spacing=(100.0, 100.0), albedo=(1.0, 1.0, 1.0), env_map=env,
        env_intensity=1.0, spp=1, min_frames=2 * WINDOW, max_frames=3 * WINDOW,
        variance_threshold=1e30,
    )
    return dem, camera, settings


def _terrain(**controls):
    dem, camera, settings = _scene()
    camera = controls.pop("camera", camera)
    return f3d.hybrid_render_terrain_reference(dem, 8, 8, camera, **(settings | controls))


def _poster(**controls):
    dem, camera, settings = _scene()
    return f3d.render_terrain_poster(dem, 8, 8, camera, **({"tile": 4} | settings | controls))


def _current_allocations():
    metrics = f3d.memory_metrics()
    return {key: metrics[key] for key in ("buffer_bytes", "texture_bytes", "host_visible_bytes")}


def _capture_terrain(result, path):
    np.savez(path, **{field: result[field] for field in ARRAYS})
    path.with_suffix(".json").write_text(
        json.dumps({field: result[field] for field in STATISTICS}), encoding="utf-8",
    )


def _command(mode, path, *extra):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), mode, str(path), *map(str, extra)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.splitlines()[-1])


def _kill_at_boundary(mode, checkpoint, certificate):
    command = [sys.executable, str(Path(__file__).resolve()), mode, str(checkpoint), str(certificate)]
    with subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
    ) as process:
        messages = []
        for line in process.stdout:
            if line.rstrip() == "CONTINUA_READY":
                process.kill()
                assert process.wait() != 0
                return
            messages.append(line)
        pytest.fail(f"Worker exited before its checkpoint boundary: {''.join(messages)}")


def _await_termination():
    print("CONTINUA_READY", flush=True)
    sys.stdin.read(1)
    raise AssertionError("checkpoint worker should have been terminated")


@pytest.fixture(scope="module")
def gpu():
    probe = f3d.device_probe(os.environ.get("WGPU_BACKEND"))
    if (
        not _adapter_is_terrain_safe(probe)
        or probe.get("backend") != "Vulkan"
        or not probe.get("driver_info")
    ):
        message = f"CONTINUA recovery requires a physical adapter with a driver identity: {probe}"
        if os.environ.get("FORGE3D_TESSELLA_REQUIRED_GPU") == "1":
            pytest.fail(message)
        pytest.skip(message)


def test_public_controls_are_keyword_only():
    for render in (f3d.hybrid_render_terrain_reference, f3d.render_terrain_poster):
        signature = inspect.signature(render)
        for name, default in (
            ("on_progress", None), ("checkpoint_path", None),
            ("checkpoint_every", None), ("resume", False),
        ):
            parameter = signature.parameters[name]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
            assert parameter.default is default
    assert issubclass(f3d.TerrainRenderCancelled, RuntimeError)


def test_terrain_restart_preserves_arrays_statistics_and_absolute_frames(gpu, tmp_path):
    reference = _terrain()
    assert reference["frames"] == 2 * WINDOW
    assert reference["variance"] > 0
    assert reference["reservoir_valid_count"] > 0
    current = _current_allocations()
    for frame in FRAME_BOUNDARIES:
        checkpoint = tmp_path / f"{frame}.frame"
        artifact = tmp_path / f"{frame}.npz"
        cert = tmp_path / f"{frame}.cancelled.cert.json"
        with pytest.raises(f3d.TerrainRenderCancelled) as cancelled:
            _terrain(
                checkpoint_path=checkpoint, checkpoint_every=frame, certificate=str(cert),
                on_progress=lambda event: event.completed_frames != frame,
            )
        assert _current_allocations() == current
        assert not cert.exists()
        error = cancelled.value
        assert error.code == "terrain_render_cancelled"
        assert error.checkpoint_path == checkpoint
        assert checkpoint.is_file()
        assert error.progress.completed_frames == frame
        measured_at = frame // WINDOW * WINDOW
        assert error.progress.variance_at_frame == (measured_at or None)
        assert error.progress.variance_age_frames == (frame - measured_at if measured_at else None)
        if measured_at:
            assert error.progress.variance > 0
        else:
            assert error.progress.variance is None

        metadata = _command("resume-terrain", checkpoint, artifact)
        assert metadata["resume_frame"] == frame
        assert metadata["dispatched"] == reference["frames"] - frame
        assert metadata["first_callback"] == (frame + 1 if frame < reference["frames"] else None)
        with np.load(artifact, allow_pickle=False) as resumed:
            for field in ARRAYS:
                np.testing.assert_array_equal(resumed[field], reference[field], err_msg=field)
        stats = json.loads(artifact.with_suffix(".json").read_text(encoding="utf-8"))
        assert stats == {field: reference[field] for field in STATISTICS}
        certificate = json.loads(artifact.with_suffix(".cert.json").read_text(encoding="utf-8"))
        assert certificate["inputs"]["continua_resumed"] == "true"
        assert certificate["inputs"]["continua_resume_frame"] == str(frame)
        assert certificate["inputs"]["continua_frames_dispatched"] == str(reference["frames"] - frame)
        if frame == reference["frames"]:
            assert not certificate["passes"], "converged restore has no terrain or geometry dispatch"


def test_poster_skips_published_tiles_and_resumes_after_worker_exit(gpu, tmp_path):
    checkpoint = tmp_path / "poster"
    cert = tmp_path / "poster.cert.json"
    _kill_at_boundary("stop-after-receipt", checkpoint, cert)
    assert len(list(checkpoint.glob("*.rgba"))) == 2
    assert not cert.exists()
    _kill_at_boundary("stop-in-third-tile", checkpoint, cert)
    assert len(list(checkpoint.glob("*.rgba"))) == 2
    assert (checkpoint / "tile-0-1.frame").is_file()
    assert not cert.exists()

    artifact = tmp_path / "poster.npy"
    result = _command("resume-poster", checkpoint, artifact, cert)
    reference = _poster()
    np.testing.assert_array_equal(np.load(artifact, allow_pickle=False), reference["rgba"])
    assert result["recovered"] == 2
    assert result["resumed"]
    assert result["dispatches_per_tile"] == [0, 0, WINDOW - 1, 2 * WINDOW]
    assert result["frames_dispatched"] == sum(result["dispatches_per_tile"])
    assert result["resume_frames"] == [2 * WINDOW, 2 * WINDOW, WINDOW + 1, 0]
    assert result["resumed_per_tile"] == [True, True, True, False]
    inputs = json.loads(cert.read_text(encoding="utf-8"))["inputs"]
    assert inputs["continua_resumed"] == "true"
    assert inputs["continua_recovered_tiles"] == "2"
    assert inputs["continua_resumed_tiles"] == "1"
    assert inputs["continua_frames_dispatched"] == str(result["frames_dispatched"])

    with pytest.raises(RuntimeError, match="tile_layout mismatch"):
        _poster(checkpoint_path=checkpoint, resume=True, tile=8)
    receipt = checkpoint / "tile-0-0.rgba"
    damaged = bytearray(receipt.read_bytes())
    damaged[-1] ^= 1
    receipt.write_bytes(damaged)
    with pytest.raises(RuntimeError, match="tile checksum mismatch"):
        _poster(checkpoint_path=checkpoint, resume=True)


def test_invalid_restore_and_callback_failures_leave_no_success_artifacts(gpu, tmp_path):
    checkpoint = tmp_path / "valid.frame"
    certificate = tmp_path / "failed.cert.json"
    # Warm the reusable GPU context before examining job allocations.
    _terrain()
    current = _current_allocations()

    def abort(_event):
        raise ValueError("callback failure")

    with pytest.raises(ValueError, match="callback failure"):
        _terrain(
            on_progress=abort, checkpoint_path=checkpoint, checkpoint_every=1,
            certificate=str(certificate),
        )
    assert checkpoint.is_file()
    assert not certificate.exists()
    assert _current_allocations() == current

    with pytest.raises(f3d.TerrainRenderCancelled) as cancelled:
        _terrain(on_progress=lambda _: False, certificate=str(certificate))
    assert cancelled.value.checkpoint_path is None
    assert _current_allocations() == current

    data = checkpoint.read_bytes()
    for field, filename, content in (
        ("length", "short.frame", data[:-1]),
        ("checksum", "checksum.frame", data[:-1] + bytes([data[-1] ^ 1])),
        ("native_build", "build.frame", data[:8] + bytes([data[8] ^ 1]) + data[9:]),
        ("shader", "shader.frame", data[:40] + bytes([data[40] ^ 1]) + data[41:]),
    ):
        path = tmp_path / filename
        path.write_bytes(content)
        with pytest.raises(RuntimeError, match=field):
            _terrain(checkpoint_path=path, resume=True, certificate=str(certificate))
        assert _current_allocations() == current
    dem, camera, _ = _scene()
    camera["fov_y"] = 31
    with pytest.raises(RuntimeError, match="camera mismatch"):
        _terrain(checkpoint_path=checkpoint, resume=True, camera=camera, certificate=str(certificate))
    dem[0, 0] += 1
    with pytest.raises(RuntimeError, match="heightmap mismatch"):
        f3d.hybrid_render_terrain_reference(dem, 8, 8, checkpoint_path=checkpoint, resume=True)
    assert not certificate.exists()
    assert _current_allocations() == current

def test_checkpointed_poster_owns_inputs_without_a_progress_callback(gpu, tmp_path, monkeypatch):
    from forge3d._terrain_checkpoint import PosterCheckpoint

    dem, camera, settings = _scene()
    original_env = settings["env_map"].copy()
    expected = f3d.render_terrain_poster(dem, 8, 8, camera, tile=4, **settings)["rgba"]
    saved = PosterCheckpoint.save_tile
    published, changed = threading.Event(), threading.Event()

    def mutate():
        published.wait()
        settings["env_map"].fill(0)
        dem.fill(9)
        changed.set()

    def save_then_signal(self, *args):
        saved(self, *args)
        if not published.is_set():
            published.set()
            changed.wait()

    monkeypatch.setattr(PosterCheckpoint, "save_tile", save_then_signal)
    writer = threading.Thread(target=mutate)
    writer.start()
    job = tmp_path / "poster"
    cert = tmp_path / "poster.cert.json"
    try:
        actual = f3d.render_terrain_poster(
            dem, 8, 8, camera, tile=4, checkpoint_path=job, certificate=cert, **settings,
        )
    finally:
        published.set()
        writer.join()
    np.testing.assert_array_equal(actual["rgba"], expected)
    assert cert.is_file()
    recovered = _poster(checkpoint_path=job, resume=True, env_map=original_env)
    assert recovered["recovered_tiles"] == 4
    assert recovered["frames_dispatched"] == 0
    np.testing.assert_array_equal(recovered["rgba"], expected)


def test_receipt_only_certificate_binds_source_and_pixels(gpu, tmp_path):
    digests, posters = [], []
    for index, albedo in enumerate(((1.0, 1.0, 1.0), (0.12, 0.16, 0.05))):
        job = tmp_path / str(index)
        original_cert = tmp_path / f"{index}-original.json"
        restored_cert = tmp_path / f"{index}-restored.json"
        original = _poster(albedo=albedo, checkpoint_path=job, certificate=original_cert)
        restored = _poster(albedo=albedo, checkpoint_path=job, resume=True, certificate=restored_cert)
        np.testing.assert_array_equal(restored["rgba"], original["rgba"])
        digests.append(restored["certificate_digest"])
        posters.append(restored["rgba"])
        cert = json.loads(restored_cert.read_text(encoding="utf-8"))
        primary = json.loads(original_cert.read_text(encoding="utf-8"))
        manifest = job.joinpath("poster.continua").read_bytes()
        assert cert["inputs"]["continua_manifest_sha256"] == manifest[-32:].hex()
        assert cert["inputs"]["continua_rgba_sha256"] == hashlib.sha256(restored["rgba"]).hexdigest()
        assert cert["inputs"]["albedo"] == primary["inputs"]["albedo"]
        assert cert["inputs"]["continua_frames_dispatched"] == "0"
        assert {entry["label"] for entry in cert["passes"]} == {"hybrid_pt.poster"}
        for field in ("native_build", "adapter", "backend", "driver", "shader"):
            key = f"continua_source_{field}"
            assert len(cert["inputs"][key]) == 64
            assert cert["inputs"][key] == primary["inputs"][key]
    assert not np.array_equal(*posters)
    assert digests[0] != digests[1]


@pytest.mark.parametrize("outer_render, inner_render", [(_terrain, _poster), (_poster, _terrain)])
def test_callback_render_has_its_own_certificate(gpu, tmp_path, outer_render, inner_render):
    outer_cert = tmp_path / "outer.json"
    inner_cert = tmp_path / "inner.json"
    inner_runs = []

    def nested(_event):
        if not inner_runs:
            inner_runs.append(inner_render(albedo=(0.1, 0.2, 0.3), certificate=inner_cert))

    output = outer_render(on_progress=nested, certificate=outer_cert)
    assert len(inner_runs) == 1
    np.testing.assert_array_equal(output["rgba"], outer_render()["rgba"])
    for render, path, color in (
        (outer_render, outer_cert, "1,1,1"),
        (inner_render, inner_cert, "0.1,0.2,0.3"),
    ):
        cert = json.loads(path.read_text(encoding="utf-8"))
        assert cert["inputs"]["albedo"] == color
        labels = [entry["label"] for entry in cert["passes"]]
        assert labels.count("hybrid_pt.poster") == (1 if render is _poster else 0)
        assert labels.count("hybrid_pt.terrain") == (4 if render is _poster else 1)


@pytest.mark.parametrize("caller_has_rendered", [False, True], ids=["fresh-caller", "prior-render"])
def test_public_certificate_writer_exports_latest_process_render(gpu, tmp_path, caller_has_rendered):
    from forge3d.certificate import emit_render_certificate, payload_sha256
    from forge3d.diagnostics import render_certificate

    path = tmp_path / "worker.json"

    def export_from_other_thread():
        if caller_has_rendered:
            _terrain()
        workers.submit(_terrain, albedo=(0.1, 0.2, 0.3)).result()
        expected = render_certificate()
        assert expected["inputs"]["albedo"] == "0.1,0.2,0.3"
        assert emit_render_certificate(path) == payload_sha256(expected)
        assert json.loads(path.read_text(encoding="utf-8")) == expected

    with ThreadPoolExecutor(max_workers=2) as workers:
        workers.submit(export_from_other_thread).result()


@pytest.mark.parametrize("pause", ["progress", "native", "signing"])
def test_concurrent_posters_keep_their_certificates(gpu, tmp_path, monkeypatch, pause):
    from forge3d import _degradation, path_tracing
    from forge3d._terrain_checkpoint import PosterCheckpoint

    a_ready = threading.Event()
    b_saved = threading.Event()
    a_done = threading.Event()
    b_done = threading.Event()
    a_cert = tmp_path / "a.json"
    b_cert = tmp_path / "b.json"

    save = PosterCheckpoint.save_tile
    emit = path_tracing._emit_current_render_certificate

    def save_tile(self, path, window, rgba, stats):
        marker = path.parent.name
        _degradation.record("test", marker, "recorded by this poster")
        if pause == "progress" and marker == "b" and not b_saved.is_set():
            b_saved.set()
            a_done.wait()
        save(self, path, window, rgba, stats)

    def sign(certificate):
        if pause == "signing" and certificate == a_cert:
            a_ready.set()
            b_done.wait()
        return emit(certificate)

    def progress(_event):
        if pause != "signing" and not a_ready.is_set():
            a_ready.set()
            b_saved.wait()

    def release_a(_event):
        b_saved.set()

    def poster_a():
        try:
            return _poster(checkpoint_path=tmp_path / "a", certificate=a_cert, on_progress=progress)
        finally:
            a_ready.set()
            a_done.set()

    def poster_b():
        a_ready.wait()
        try:
            return _poster(
                checkpoint_path=tmp_path / "b", certificate=b_cert, albedo=(0.1, 0.2, 0.3),
                on_progress=release_a if pause == "native" else None,
            )
        finally:
            b_saved.set()
            b_done.set()

    monkeypatch.setattr(PosterCheckpoint, "save_tile", save_tile)
    monkeypatch.setattr(path_tracing, "_emit_current_render_certificate", sign)
    with ThreadPoolExecutor(max_workers=2) as workers:
        a = workers.submit(poster_a)
        b = workers.submit(poster_b)
        a_result, b_result = a.result(), b.result()
        assert not np.array_equal(a_result["rgba"], b_result["rgba"])

    for path, result, color in ((a_cert, a_result, "1,1,1"), (b_cert, b_result, "0.1,0.2,0.3")):
        cert = json.loads(path.read_text(encoding="utf-8"))
        assert cert["inputs"]["albedo"] == color
        assert cert["inputs"]["continua_rgba_sha256"] == hashlib.sha256(result["rgba"]).hexdigest()
        labels = [entry["label"] for entry in cert["passes"]]
        assert labels.count("hybrid_pt.terrain") == 4
        assert labels.count("hybrid_pt.poster") == 1
        assert cert["allocations"]["peak_host_visible_bytes"] > 0
        assert [row["name"] for row in cert["degradations"] if row["kind"] == "test"] == [path.stem]


def _windows_linker_id(binary):
    """Read the CodeView ID using the on-disk PE offsets."""
    image = binary.read_bytes()
    pe = struct.unpack_from("<I", image, 60)[0]
    section_count, = struct.unpack_from("<H", image, pe + 6)
    optional_size, = struct.unpack_from("<H", image, pe + 20)
    optional = pe + 24
    magic, = struct.unpack_from("<H", image, optional)
    directory = {0x10B: 96, 0x20B: 112}[magic]
    debug_rva, debug_size = struct.unpack_from("<II", image, optional + directory + 6 * 8)
    for i in range(section_count):
        section = optional + optional_size + i * 40
        rva, size, offset = struct.unpack_from("<III", image, section + 12)
        if rva <= debug_rva < rva + size:
            start = offset + debug_rva - rva
            for entry in range(start, start + debug_size, 28):
                kind, size, _, data = struct.unpack_from("<IIII", image, entry + 12)
                if kind == 2 and size >= 24 and image[data:data + 4] == b"RSDS":
                    return image[data + 4:data + 24]
    raise AssertionError("The Windows extension has no CodeView build ID")


def test_build_identity_comes_from_the_loaded_extension(gpu, tmp_path):
    expected = _command("build-id", tmp_path)
    if os.name != "posix":
        if sys.platform == "win32":
            import forge3d._forge3d as native
            linker_id = _windows_linker_id(Path(native.__file__))
            assert expected == hashlib.sha256(
                b"forge3d.continua.linked-image.v1windows" + linker_id,
            ).hexdigest()
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"replacement installation file")
        assert _command("build-id", replacement, "redirect-path") == expected
        return

    # Linux can replace an extension's file while the old inode is mapped.
    # Copy the package so the test never modifies the loaded test installation.
    source = Path(f3d.__file__).resolve().parent
    runtime = tmp_path / "runtime"
    shutil.copytree(source, runtime / "forge3d")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(runtime), env.get("PYTHONPATH")]))
    command = [sys.executable, str(Path(__file__).resolve()), "build-id", str(tmp_path), "wait"]
    with subprocess.Popen(command, env=env, text=True, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as worker:
        line = worker.stdout.readline().rstrip()
        assert line.startswith("LOADED "), line
        loaded = Path(line.removeprefix("LOADED "))
        assert loaded.resolve().parent == (runtime / "forge3d").resolve()
        replacement = loaded.with_suffix(".new")
        replacement.write_bytes(b"replacement installation file")
        os.replace(replacement, loaded)
        stdout, _ = worker.communicate("continue")
    assert worker.returncode == 0, stdout
    assert json.loads(stdout.splitlines()[-1]) == expected


def _worker(mode, path, args):
    if mode == "resume-terrain":
        frames = []
        start = perf_counter()
        artifact = Path(args[0])
        result = _terrain(
            checkpoint_path=path, resume=True,
            on_progress=lambda event: frames.append(event.completed_frames),
            certificate=str(artifact.with_suffix(".cert.json")),
        )
        elapsed = perf_counter() - start
        _capture_terrain(result, artifact)
        print(json.dumps({
            "resume_frame": result["resume_frame"], "dispatched": result["frames_dispatched"],
            "first_callback": frames[0] if frames else None, "seconds": elapsed,
        }))
    elif mode == "stop-after-receipt":
        def stop(event):
            if event.tile_complete and event.tile == (1, 0):
                _await_termination()
        _poster(checkpoint_path=path, on_progress=stop, certificate=args[0])
    elif mode == "stop-in-third-tile":
        def stop(event):
            if event.tile == (0, 1) and event.completed_frames == WINDOW + 1:
                _await_termination()
        _poster(
            checkpoint_path=path, resume=True, checkpoint_every=WINDOW + 1,
            on_progress=stop, certificate=args[0],
        )
    elif mode == "resume-poster":
        result = _poster(checkpoint_path=path, resume=True, certificate=args[1])
        np.save(args[0], result["rgba"], allow_pickle=False)
        print(json.dumps({
            "recovered": result["recovered_tiles"],
            "resumed": result["resumed"],
            "dispatches_per_tile": [tile["frames_dispatched"] for tile in result["tiles"]],
            "frames_dispatched": result["frames_dispatched"],
            "resume_frames": [tile["resume_frame"] for tile in result["tiles"]],
            "resumed_per_tile": [tile["resumed"] for tile in result["tiles"]],
        }))
    elif mode == "build-id":
        import forge3d._forge3d as native
        if args == ["redirect-path"]:
            native.__file__ = str(path)
        elif args == ["wait"]:
            print(f"LOADED {native.__file__}", flush=True)
            sys.stdin.read()
        print(json.dumps(native._terrain_checkpoint_runtime()["native_build"]))
    elif mode == "baseline":
        start = perf_counter()
        _capture_terrain(_terrain(), path)
        print(json.dumps({"seconds": perf_counter() - start, "device": f3d.device_probe()}))
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    _worker(sys.argv[1], Path(sys.argv[2]), sys.argv[3:])
