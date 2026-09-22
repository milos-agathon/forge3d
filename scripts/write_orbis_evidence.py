#!/usr/bin/env python3
"""Bind measured ORBIS acceptance output to its exact runtime revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EvidenceError(RuntimeError):
    """Raised when an input cannot support the claimed ORBIS evidence."""


ORBIS_MIN_JITTER_SAMPLES = 32
ORBIS_MIN_CRACK_BOUNDARY_SAMPLES = 64
ORBIS_EXPECTED_DESCENT_FRAMES = 25
# Progress is a state-transition count whose value depends on cold COG decode
# latency versus first-frame shader compilation, so only existence is gated;
# nonblocking behaviour is gated by the bounded-poll and pending-work counts.
ORBIS_MIN_STREAMING_PROGRESS_FRAMES = 1


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must contain a JSON object")
    return value


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise EvidenceError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise EvidenceError(f"{name} must be finite")
    return number


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EvidenceError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise EvidenceError(f"{name} exceeds {maximum}")
    return value


def _sha256(path: Path, label: str) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise EvidenceError(f"cannot read {label} {path}: {error}") from error


def _validate_adapter(
    probe_record: dict[str, Any], metrics: dict[str, Any]
) -> dict[str, Any]:
    adapter = probe_record.get("probe")
    if not isinstance(adapter, dict):
        raise EvidenceError("adapter probe is missing its probe object")
    required = {
        "status": "ok",
        "vendor": 0x10DE,
        "backend": "Vulkan",
        "device_type": "DiscreteGpu",
        "software_fallback": False,
    }
    for name, expected in required.items():
        if adapter.get(name) != expected:
            raise EvidenceError(
                f"adapter {name}={adapter.get(name)!r}, expected {expected!r}"
            )
    if "nvidia" not in str(adapter.get("name", "")).casefold():
        raise EvidenceError("adapter name does not identify NVIDIA")
    metric_fields = {
        "adapter_name": "name",
        "adapter_vendor": "vendor",
        "adapter_backend": "backend",
        "adapter_device_type": "device_type",
        "software_fallback": "software_fallback",
    }
    for metric_name, probe_name in metric_fields.items():
        if metrics.get(metric_name) != adapter.get(probe_name):
            raise EvidenceError(
                f"metrics {metric_name} does not match adapter probe {probe_name}"
            )
    return adapter


def _validate_measurements(measurements: dict[str, Any]) -> dict[str, Any]:
    metrics = measurements.get("metrics")
    golden = measurements.get("golden")
    if not isinstance(metrics, dict) or not isinstance(golden, dict):
        raise EvidenceError("measurement input requires metrics and golden objects")

    jitter = _finite_number(metrics.get("max_vertex_jitter_px"), "max_vertex_jitter_px")
    naive = _finite_number(
        metrics.get("naive_max_vertex_jitter_px"), "naive_max_vertex_jitter_px"
    )
    peak = _finite_number(
        metrics.get("peak_gpu_visible_bytes"), "peak_gpu_visible_bytes"
    )
    cracks = _finite_number(metrics.get("lod_crack_pixels"), "lod_crack_pixels")
    if not 0.0 <= jitter < 0.5:
        raise EvidenceError(
            "camera-relative jitter does not satisfy 0 <= jitter < 0.5 px"
        )
    if not naive > 10.0 * jitter:
        raise EvidenceError("naive absolute-f32 jitter is not more than 10x production")
    if not 0.0 < peak < 512 * 1024 * 1024:
        raise EvidenceError("GPU-visible peak does not satisfy 0 < peak < 512 MiB")
    if cracks != 0.0:
        raise EvidenceError("LOD crack count is not zero")
    jitter_samples = _positive_int(
        metrics.get("jitter_sample_count"), "jitter_sample_count"
    )
    if jitter_samples < ORBIS_MIN_JITTER_SAMPLES:
        raise EvidenceError(
            f"jitter_sample_count must be at least {ORBIS_MIN_JITTER_SAMPLES}"
        )
    boundary_samples = _positive_int(
        metrics.get("crack_boundary_samples"), "crack_boundary_samples"
    )
    if boundary_samples < ORBIS_MIN_CRACK_BOUNDARY_SAMPLES:
        raise EvidenceError(
            "crack_boundary_samples must be at least "
            f"{ORBIS_MIN_CRACK_BOUNDARY_SAMPLES}"
        )
    if _finite_number(metrics.get("crack_depth_variance"), "crack_depth_variance") <= 0:
        raise EvidenceError("crack_depth_variance contains no measured depth evidence")
    rendered = _positive_int(metrics.get("rendered_frames"), "rendered_frames")
    bounded = _positive_int(metrics.get("bounded_poll_frames"), "bounded_poll_frames")
    progress = _positive_int(
        metrics.get("streaming_progress_frames"), "streaming_progress_frames"
    )
    if (
        rendered != ORBIS_EXPECTED_DESCENT_FRAMES
        or bounded != ORBIS_EXPECTED_DESCENT_FRAMES
    ):
        raise EvidenceError(
            "rendered_frames and bounded_poll_frames must both be exactly "
            f"{ORBIS_EXPECTED_DESCENT_FRAMES}"
        )
    if (
        not ORBIS_MIN_STREAMING_PROGRESS_FRAMES
        <= progress
        <= ORBIS_EXPECTED_DESCENT_FRAMES
    ):
        raise EvidenceError(
            "streaming_progress_frames must be between "
            f"{ORBIS_MIN_STREAMING_PROGRESS_FRAMES} and {ORBIS_EXPECTED_DESCENT_FRAMES}"
        )
    _positive_int(metrics.get("pending_streaming_frames"), "pending_streaming_frames")
    _positive_int(metrics.get("coarse_fallback_frames"), "coarse_fallback_frames")
    _positive_int(
        metrics.get("max_stream_uploads_per_frame"),
        "max_stream_uploads_per_frame",
        maximum=64,
    )
    if _finite_number(golden.get("ssim"), "golden.ssim") < 0.995:
        raise EvidenceError("golden SSIM is below 0.995")
    if (
        _finite_number(
            golden.get("mean_absolute_difference"), "golden.mean_absolute_difference"
        )
        > 2.0
    ):
        raise EvidenceError("golden mean absolute difference exceeds 2")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--adapter-probe", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--actual-image", type=Path, required=True)
    parser.add_argument("--command", action="append", required=True)
    parser.add_argument("--candidate-sha")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        repository = args.repository.resolve()
        repository_sha = _git(repository, "rev-parse", "HEAD")
        if len(repository_sha) != 40:
            raise EvidenceError(
                "runtime repository SHA is not a full 40-character commit"
            )
        if args.candidate_sha and repository_sha != args.candidate_sha:
            raise EvidenceError(
                f"runtime SHA {repository_sha} does not match candidate {args.candidate_sha}"
            )
        tracked_status = _git(
            repository, "status", "--porcelain", "--untracked-files=no"
        )
        if args.require_clean and tracked_status:
            raise EvidenceError(f"tracked worktree is dirty: {tracked_status}")

        measurements = _read_object(args.metrics, "ORBIS measurements")
        metrics = _validate_measurements(measurements)
        golden = measurements["golden"]
        golden_sha = _sha256(args.golden, "committed golden")
        if golden.get("sha256") != golden_sha:
            raise EvidenceError(
                "recorded golden SHA-256 does not match the supplied file"
            )
        actual_sha = _sha256(args.actual_image, "actual snapshot")
        if measurements.get("snapshot_sha256") != actual_sha:
            raise EvidenceError(
                "recorded snapshot SHA-256 does not match the supplied file"
            )
        probe_record = _read_object(args.adapter_probe, "adapter probe")
        adapter = _validate_adapter(probe_record, metrics)
        evidence = {
            "schema": "forge3d.orbis.acceptance.v1",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository_sha": repository_sha,
            "tracked_worktree_clean": not bool(tracked_status),
            "commands": args.command,
            "adapter": adapter,
            "measurements": measurements,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (EvidenceError, OSError, ValueError) as error:
        parser.exit(1, f"ORBIS evidence rejected: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
