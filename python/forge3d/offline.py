from __future__ import annotations

from dataclasses import dataclass
import time
import warnings
from typing import Any, Callable, Optional

import numpy as np

from .denoise import atrous_denoise
from .denoise_oidn import oidn_available, oidn_denoise
from .terrain_params import OfflineQualitySettings

_CONVERGENCE_TREND_WINDOW = 3


@dataclass
class OfflineProgress:
    samples_so_far: int
    max_samples: int
    mean_delta: float
    p95_delta: float
    converged_ratio: float
    elapsed_ms: float


@dataclass
class OfflineResult:
    frame: Any
    hdr_frame: Any
    aov_frame: Any
    metadata: dict


def _as_metric_dict(metrics: Any) -> dict[str, float]:
    def _value(key: str) -> Any:
        try:
            return metrics[key]
        except Exception:
            return getattr(metrics, key)

    return {
        "total_samples": int(_value("total_samples")),
        "mean_delta": float(_value("mean_delta")),
        "p95_delta": float(_value("p95_delta")),
        "max_tile_delta": float(_value("max_tile_delta")),
        "converged_tile_ratio": float(_value("converged_tile_ratio")),
    }


def _require_matching_aov_shape(
    name: str,
    array: np.ndarray,
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    arr = np.ascontiguousarray(array, dtype=np.float32)
    if arr.shape != expected_shape:
        raise RuntimeError(f"{name} shape {arr.shape} does not match beauty HDR shape {expected_shape}")
    return arr


def _has_upward_convergence_trend(history: list[dict[str, float]]) -> bool:
    if len(history) < _CONVERGENCE_TREND_WINDOW:
        return False

    window = history[-_CONVERGENCE_TREND_WINDOW:]
    ratios = [entry["converged_tile_ratio"] for entry in window]
    return ratios[-1] >= ratios[0] - 1e-3 and sum(
        curr - prev for prev, curr in zip(ratios, ratios[1:])
    ) >= -1e-3


def render_offline(
    renderer: Any,
    material_set: Any,
    env_maps: Any,
    params: Any,
    heightmap: np.ndarray,
    *,
    settings: OfflineQualitySettings,
    progress_callback: Optional[Callable[[OfflineProgress], None]] = None,
    water_mask: Optional[np.ndarray] = None,
) -> OfflineResult:
    """Render terrain through the TV12 offline accumulation pipeline.

    `settings.enabled` must be True so callers opt into the offline pipeline
    deliberately rather than relying on the dataclass defaults.

    `water_mask` is forwarded unchanged to the terrain renderer so offline renders
    honor the same terrain-water shading inputs as one-shot renders.
    """

    if not settings.enabled:
        raise ValueError("render_offline requires OfflineQualitySettings(enabled=True)")

    target_samples = int(settings.max_samples if settings.adaptive else params.aa_samples)
    target_samples = max(target_samples, 1)
    renderer.begin_offline_accumulation(
        material_set=material_set,
        env_maps=env_maps,
        params=params,
        heightmap=heightmap,
        water_mask=water_mask,
        jitter_sequence_samples=target_samples,
    )

    rendered = 0
    metrics: dict[str, float] | None = None
    metric_history: list[dict[str, float]] = []
    denoiser_used = "none"
    started = time.perf_counter()

    try:
        while rendered < target_samples:
            batch = min(int(settings.batch_size), target_samples - rendered)
            batch_result = renderer.accumulate_batch(batch)
            try:
                rendered = int(batch_result["total_samples"])
            except Exception:
                rendered = int(batch_result.total_samples)

            need_metrics = progress_callback is not None or (
                settings.adaptive and rendered >= settings.min_samples
            )
            if need_metrics:
                metrics = _as_metric_dict(
                    renderer.read_accumulation_metrics(
                        settings.target_variance,
                        settings.tile_size,
                    )
                )
                metric_history.append(metrics)

            if progress_callback is not None and metrics is not None:
                progress_callback(
                    OfflineProgress(
                        samples_so_far=rendered,
                        max_samples=target_samples,
                        mean_delta=metrics["mean_delta"],
                        p95_delta=metrics["p95_delta"],
                        converged_ratio=metrics["converged_tile_ratio"],
                        elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    )
                )

            if (
                settings.adaptive
                and rendered >= settings.min_samples
                and metrics is not None
                and _has_upward_convergence_trend(metric_history)
                and (
                    metrics["converged_tile_ratio"] >= settings.convergence_ratio
                    or metrics["p95_delta"] < settings.target_variance
                )
            ):
                break

        hdr_frame, aov_frame = renderer.resolve_offline_hdr()

        if getattr(params, "denoise", None) is not None and params.denoise.enabled:
            method = str(params.denoise.method).lower()
            if method != "none":
                beauty_hdr = np.asarray(hdr_frame.to_numpy_f32(), dtype=np.float32)[..., :3]
                albedo_np = _require_matching_aov_shape(
                    "albedo",
                    np.asarray(aov_frame.albedo(), dtype=np.float32),
                    beauty_hdr.shape,
                )
                normal_np = _require_matching_aov_shape(
                    "normal",
                    np.asarray(aov_frame.normal(), dtype=np.float32),
                    beauty_hdr.shape,
                )

                if method == "oidn":
                    if oidn_available():
                        beauty_hdr = oidn_denoise(
                            beauty_hdr,
                            albedo=albedo_np,
                            normal=normal_np,
                        )
                        denoiser_used = "oidn"
                    else:
                        warnings.warn(
                            "oidn package not installed; falling back to atrous denoiser",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        method = "atrous"

                if method == "atrous":
                    beauty_hdr = atrous_denoise(
                        beauty_hdr,
                        albedo=albedo_np,
                        normal=normal_np,
                        iterations=int(params.denoise.iterations),
                        sigma_color=float(params.denoise.sigma_color),
                        sigma_normal=float(params.denoise.sigma_normal),
                        sigma_depth=float(params.denoise.sigma_depth),
                    )
                    denoiser_used = "atrous"

                if denoiser_used != "none":
                    hdr_frame = renderer.upload_hdr_frame(beauty_hdr, hdr_frame.size)

        frame = renderer.tonemap_offline_hdr(hdr_frame)
    except Exception:
        renderer.end_offline_accumulation()
        raise

    return OfflineResult(
        frame=frame,
        hdr_frame=hdr_frame,
        aov_frame=aov_frame,
        metadata={
            "samples_used": rendered,
            "denoiser_used": denoiser_used,
            "final_p95_delta": None if metrics is None else metrics["p95_delta"],
            "converged_ratio": None if metrics is None else metrics["converged_tile_ratio"],
            "target_samples": target_samples,
            "adaptive": bool(settings.adaptive),
        },
    )


__all__ = ["OfflineProgress", "OfflineResult", "render_offline"]
