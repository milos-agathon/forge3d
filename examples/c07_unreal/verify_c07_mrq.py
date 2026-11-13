"""Verify the external C07 PNG returns and their evaluated Unreal camera samples."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image

from prepare_c07_mrq import write
from prepare_unreal_job import digest, read
from verify_unreal_return import verify


def project(point, eye, basis, width, height, half_x, half_y):
    depth, right, up = basis @ (point - eye)
    if depth <= 0:
        raise ValueError("an evaluated landmark is behind the camera")
    return np.array([(0.5 + right / (2 * depth * half_x)) * width,
                     (0.5 - up / (2 * depth * half_y)) * height])


def check_camera(job, samples, scene, sequence_name):
    count = job["frame_count"]
    if [s["mrq_root_frame"] for s in samples] != list(range(count)):
        raise ValueError("expected one ordered MRQ camera sample for every output frame")
    width, height = job["size"]
    tick_n, tick_d = scene["sequences"][sequence_name]["tick_rate"]
    projected = {}
    for lm in job["landmarks"]:
        projected.setdefault(lm["frame"], []).append(lm)
    evidence = []
    for frame, sample in zip(job["frames"], samples, strict=True):
        index = frame["frame"]
        num, den = sample["sequence_rate"]
        seconds = (sample["sequence_frame"] + sample["sequence_sub_frame"]) * den / num
        time_error = abs(seconds - frame["time_s"])
        # Allow the centre to be rounded to its nearest representable Sequencer tick.
        tick = tick_d / tick_n
        if time_error > tick / 2 + math.ulp(frame["time_s"]):
            raise ValueError(f"frame {index}: sample {seconds}s does not match the frame centre")
        eye = np.array(sample["eye_cm"])
        basis = np.array([sample[key] for key in ("forward", "right", "up")])
        ref_basis = np.array([frame[key] for key in ("forward", "right", "up")])
        angle = np.arctan2(np.linalg.norm(np.cross(ref_basis, basis), axis=1),
                           np.sum(ref_basis * basis, axis=1))
        half_x, half_y = np.array(sample["filmback_mm"]) / (2 * sample["focal_length_mm"])
        ref_x, ref_y = np.array(job["filmback_mm"]) / (2 * frame["focal_length_mm"])
        actual_half_x = math.tan(math.radians(sample["horizontal_fov_deg"]) / 2)
        edge = max(abs(half_x / ref_x - 1) * width / 2, abs(half_y / ref_y - 1) * height / 2,
                   abs(actual_half_x / ref_x - 1) * width / 2)
        aspect_error = abs((half_x / half_y) / (width / height) - 1) * width / 2
        eye_error = np.linalg.norm(eye - frame["eye_cm"])
        landmarks = projected.get(index, [])
        if not landmarks:
            raise ValueError(f"frame {index}: missing independent landmark reference")
        min_depth = min(np.dot(np.asarray(lm["world_cm"]) - frame["eye_cm"], ref_basis[0]) for lm in landmarks)
        position_half_pixel_cm = min_depth * min(ref_x / width, ref_y / height)
        # Pose, projection and filmback must each be within half of one output pixel.
        direction_limit_rad = math.atan(min(ref_x / width, ref_y / height))
        if eye_error > position_half_pixel_cm or max(angle) > direction_limit_rad or max(edge, aspect_error) > 0.5:
            raise ValueError(f"frame {index}: camera exceeds its half-pixel footprint")
        lm_rows = []
        for lm in landmarks:
            registered = scene["landmarks"][lm["feature_id"]]
            actual_xy = project(np.array(registered["trace_cm"]), eye, basis, width, height, half_x, half_y)
            reference_xy = np.array(lm["expected_viewport_xy"]) * [width, height]
            error = actual_xy - reference_xy
            if np.max(abs(error)) > 0.5:
                raise ValueError(f"frame {index}: {lm['feature_id']} projects {error.tolist()} px from reference")
            lm_rows.append({"feature_id": lm["feature_id"], "residual_px": error.tolist()})
        evidence.append({
            "frame": index, "actual_time_s": seconds, "time_error_s": time_error,
            "eye_error_cm": float(eye_error), "axis_error_deg": np.degrees(angle).tolist(),
            "fov_edge_error_px": edge, "aspect_error_px": aspect_error, "landmarks": lm_rows,
        })
    return evidence


def frame_metrics(folder, count):
    result = []
    previous = None
    for index in range(count):
        with Image.open(folder / f"frame_{index:08d}.png") as image:
            frame = np.asarray(image.convert("RGB"))
        if np.all(frame == frame[0, 0]):
            raise ValueError(f"frame {index} has a uniform image")
        delta = None if previous is None else float(np.abs(frame.astype(np.int16) - previous).mean())
        result.append({"frame": index, "min_rgb": int(frame.min()), "max_rgb": int(frame.max()),
                       "mean_absolute_rgb_delta": delta})
        previous = frame
    return result


def inspect(base, editor_log):
    request = read(base / "request.json")
    scene = read(base / "scene.json")
    terminals = {"render": "succeeded", "cancel": "cancelled", "error": "error"}
    report = {}
    for run in request["runs"]:
        shot, name, mode = run["shot"], run["name"], run["mode"]
        folder = base / name
        job_path = Path(request["jobs"][shot])
        job = read(job_path)
        status = verify(job_path, folder)
        returned = read(folder / "result.json")
        samples = read(folder / "samples.json")
        if (returned.get("scene_sha256") != digest(base / "scene.json")
                or returned.get("evaluated_samples_sha256") != digest(folder / "samples.json")):
            raise ValueError(f"{name} did not bind the scene and evaluated sample bytes")
        if status["status"] != terminals[mode]:
            raise ValueError(f"{name} returned {status['status']}, expected {terminals[mode]}")
        if returned["renderer"]["id"] != "unreal-mrq-external":
            raise ValueError(f"wrong renderer identity for {name}")
        if mode == "cancel" and not samples:
            raise ValueError("cancellation must be exercised during a real render")
        if mode == "error":
            before, marker, _ = editor_log.rpartition(f"C07 {name}: error")
            job_log = before.rpartition("MoviePipelineLinearExecutorBase starting job")[2]
            if (returned["input_sequence_exists"] or returned["job_finished"] is not None
                    or samples or status["files"] or not marker
                    or "MoviePipelineLinearExecutorBase finished 0 jobs" not in job_log):
                raise ValueError("the engine did not reject the deliberately missing sequence")
        report[name] = {**status, "measured_render_seconds": returned["render_seconds"]}
        if mode == "render":
            evidence = check_camera(job, samples, scene, shot)
            write(folder / "verification.json", {
                "camera": evidence, "images": frame_metrics(folder, job["frame_count"]),
                "max_landmark_error_px": max(abs(value) for row in evidence
                                             for lm in row["landmarks"] for value in lm["residual_px"]),
            })
    if "eiger" in report and "eiger-repeat" in report:
        original = read(base / "eiger/result.json")
        repeat = read(base / "eiger-repeat/result.json")
        report["repeat_identical_png_bytes"] = original["frames"] == repeat["frames"]
    return report


def export_cache_context(base, engine_root, editor_log):
    """Bind verified external frames to the renderer and its captured inputs."""
    def canonical_hash(value):
        return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    verified = read(base / "verification.json")
    request = read(base / "request.json")
    scene = read(base / "scene.json")
    build = read(engine_root / "Build/Build.version")
    if str(build["Changelist"]) not in scene["engine"] or scene["engine"] not in editor_log:
        raise ValueError("the installation and editor log do not identify the rendering engine")
    plugins = {}
    for directory, name in (("Experimental", "PythonScriptPlugin"), ("Editor", "EditorScriptingUtilities"),
                            ("MovieScene", "SequencerScripting"), ("MovieScene", "MovieRenderPipeline"),
                            ("Runtime", "GeometryScripting")):
        path = engine_root / "Plugins" / directory / name / f"{name}.uplugin"
        descriptor = path.read_text(encoding="utf-8-sig")
        plugins[name] = {"sha256": digest(path),
                         "version": int(re.search(r'"Version"\s*:\s*(\d+)', descriptor)[1]),
                         "version_name": re.search(r'"VersionName"\s*:\s*"([^"]*)"', descriptor)[1]}
    adapter = Path(__file__).with_name("execute_unreal_mrq.py")
    platform = re.search(r'shaderplatform="([^"]+)"', editor_log)
    gpu = re.search(r'gpu="([^"]+)"', editor_log)
    driver = re.search(r"LogRHI:\s+Driver Version: ([^\r\n]+)", editor_log)
    if not (platform and platform[1] == "VULKAN_SM5" and gpu and driver):
        raise ValueError("the editor log does not identify the qualified Vulkan SM5 device and driver")
    fingerprint = {"id": "unreal-mrq-external", "build": build, "plugins": plugins,
                   "shader_platform": platform[1], "gpu": gpu[1], "driver": driver[1],
                   "executable_sha256": digest(engine_root / "Binaries/Win64/UnrealEditor-Cmd.exe"),
                   "adapter_sha256": digest(adapter)}
    runs = {}
    for item in request["runs"]:
        name = item["name"]
        if not verified[name]["usable_movie_frames"]:
            continue
        result_path = base / name / "result.json"
        returned = read(result_path)
        job = read(Path(request["jobs"][item["shot"]]))
        context = {"source_sha256": {key: value for key, value in job.items() if key.endswith("_sha256")},
                   "job_sha256": returned["job_sha256"], "scene_sha256": returned["scene_sha256"],
                   "project_assets_sha256": scene["asset_sha256"], "level": scene["level"],
                   "sequence": returned["sequence"], "preset": returned["preset"], "settings": returned["settings"],
                   "evaluated_samples_sha256": returned["evaluated_samples_sha256"],
                   "fps": job["fps"], "size": job["size"], "frame_count": job["frame_count"]}
        runs[name] = {"render_frame_context": context, "external_renderer_context_sha256": canonical_hash(context),
                      "raw_result_sha256": digest(result_path)}
    write(base / "external-cache-context.json", {
        "render_frame_fingerprint": fingerprint, "external_renderer_sha256": canonical_hash(fingerprint),
        "runs": runs,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    parser.add_argument("--editor-log", type=Path, required=True)
    parser.add_argument("--engine-root", type=Path, help="also write the verified external renderer cache context")
    args = parser.parse_args()
    log = args.editor_log.read_text(encoding="utf-8")
    write(args.attempt / "verification.json", inspect(args.attempt, log))
    if args.engine_root is not None:
        export_cache_context(args.attempt, args.engine_root, log)
