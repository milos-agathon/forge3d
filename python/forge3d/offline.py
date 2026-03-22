"""TV12: Offline terrain render quality controller.

Drives the Rust/GPU accumulation engine with adaptive stopping and optional OIDN denoising.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


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
    frame: object       # Frame (tonemapped uint8)
    hdr_frame: object   # HdrFrame (linear HDR)
    aov_frame: object   # AovFrame (albedo, normal, depth)
    metadata: dict


def render_offline(
    renderer,
    material_set,
    env_maps,
    params,
    heightmap: np.ndarray,
    *,
    settings,  # OfflineQualitySettings
    progress_callback: Optional[Callable[[OfflineProgress], None]] = None,
) -> OfflineResult:
    """Run the offline terrain accumulation pipeline.

    The renderer must have begin_offline_accumulation, accumulate_batch,
    read_accumulation_metrics, resolve_offline_hdr, upload_hdr_frame,
    tonemap_offline_hdr, and end_offline_accumulation methods (TV12 Rust API).
    """
    from .denoise_oidn import oidn_available, oidn_denoise
    from .denoise import atrous_denoise

    renderer.begin_offline_accumulation(params, heightmap, material_set, env_maps)
    try:
        total = settings.max_samples if settings.adaptive else params.aa_samples
        rendered = 0

        while rendered < total:
            batch = min(settings.batch_size, total - rendered)
            result = renderer.accumulate_batch(batch)
            rendered = result.total_samples

            metrics = None
            if progress_callback or (settings.adaptive and rendered >= settings.min_samples):
                metrics = renderer.read_accumulation_metrics(settings.target_variance)

            if progress_callback and metrics is not None:
                progress_callback(OfflineProgress(
                    samples_so_far=rendered,
                    max_samples=total,
                    mean_delta=metrics.mean_delta,
                    p95_delta=metrics.p95_delta,
                    converged_ratio=metrics.converged_tile_ratio,
                    elapsed_ms=result.batch_time_ms,
                ))

            if settings.adaptive and rendered >= settings.min_samples and metrics is not None:
                if (metrics.converged_tile_ratio >= settings.convergence_ratio
                        or metrics.p95_delta < settings.target_variance):
                    break

        hdr_frame, aov_frame = renderer.resolve_offline_hdr()

        denoiser_used = "none"
        denoise_cfg = getattr(params, 'denoise', None)
        if denoise_cfg is not None and getattr(denoise_cfg, 'enabled', False):
            method = getattr(denoise_cfg, 'method', 'none')
            if method != "none":
                if method == "oidn" and not oidn_available():
                    warnings.warn(
                        "oidn package not installed; falling back to atrous denoiser",
                        UserWarning,
                        stacklevel=2,
                    )
                    method = "atrous"

                beauty_hdr = hdr_frame.to_numpy_f32()[:, :, :3]
                albedo_np = aov_frame.albedo()
                normal_np = aov_frame.normal()

                if method == "oidn":
                    denoised = oidn_denoise(beauty_hdr, albedo=albedo_np, normal=normal_np)
                    denoiser_used = "oidn"
                elif method == "atrous":
                    denoised = atrous_denoise(
                        beauty_hdr,
                        albedo=albedo_np,
                        normal=normal_np,
                        iterations=getattr(denoise_cfg, 'iterations', 3),
                        sigma_color=getattr(denoise_cfg, 'sigma_color', 0.1),
                        sigma_normal=getattr(denoise_cfg, 'sigma_normal', 0.1),
                    )
                    denoiser_used = "atrous"

                hdr_frame = renderer.upload_hdr_frame(denoised, hdr_frame.size)

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
            "final_p95_delta": metrics.p95_delta if metrics else None,
            "converged_ratio": metrics.converged_tile_ratio if metrics else None,
        },
    )
