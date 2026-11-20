from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
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


def _params_python_object(params: Any) -> Any:
    return getattr(params, "python_object", params)


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
    certificate: "bool | str | Any" = False,
    cache: "str | Any | None" = None,
    _denoise_terrain_mask: Optional[np.ndarray] = None,
) -> OfflineResult:
    """Render terrain through the TV12 offline accumulation pipeline.

    `settings.enabled` must be True so callers opt into the offline pipeline
    deliberately rather than relying on the dataclass defaults.

    `water_mask` is forwarded unchanged to the terrain renderer so offline renders
    honor the same terrain-water shading inputs as one-shot renders.

    When ``certificate`` is truthy a signed RenderCertificate is assembled for
    the completed render; a path value additionally writes the signed JSON
    there, and the deterministic payload SHA256 is stashed into the result
    metadata under ``certificate_payload_sha256``.
    """

    # The native accumulation object is stateful and currently cannot be
    # reconstructed from a blob without losing its AOV/HDR type contract.
    # Soundness therefore chooses an explicit false miss for this entry point;
    # `forge3d.anamnesis.render_sequence` provides pass-granular sequence
    # caching, while accepting `cache=` here keeps the render API uniform.
    from ._cache_policy import report_cache_bypass
    report_cache_bypass(cache, "offline.render_offline")
    from . import certificate as _certificate

    with _certificate._render_capture(
        "python.offline.render", "terrain.offline.finalize", draw_calls=1
    ):
        result = _render_offline_impl(
            renderer,
            material_set,
            env_maps,
            params,
            heightmap,
            settings=settings,
            progress_callback=progress_callback,
            water_mask=water_mask,
            _denoise_terrain_mask=_denoise_terrain_mask,
        )
    if certificate:
        sha = _certificate.emit_render_certificate(certificate)
        if sha is not None:
            result.metadata["certificate_payload_sha256"] = sha
    return result


class _OfflineRenderSession:
    """Own one successful native begin through completion or failure cleanup.

    Native tonemapping still ends a successful low-level session. Explicit end
    is idempotent, so this scope also closes renderers that leave it active and
    covers every failure, including callback cancellation.
    """

    def __init__(
        self,
        renderer: Any,
        material_set: Any,
        env_maps: Any,
        params: Any,
        heightmap: np.ndarray,
        settings: OfflineQualitySettings,
        water_mask: Optional[np.ndarray],
        denoise_terrain_mask: Optional[np.ndarray] = None,
    ) -> None:
        self.renderer = renderer
        self.params = params
        self.settings = settings
        self.denoise_terrain_mask = denoise_terrain_mask
        self.target_samples = max(
            int(settings.max_samples if settings.adaptive else params.aa_samples), 1
        )
        self._begin_kwargs = {
            "material_set": material_set,
            "env_maps": env_maps,
            "params": params,
            "heightmap": heightmap,
            "water_mask": water_mask,
            "jitter_sequence_samples": self.target_samples,
        }

    def __enter__(self) -> _OfflineRenderSession:
        # A rejected begin must not close a session owned by another caller.
        self.renderer.begin_offline_accumulation(**self._begin_kwargs)
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        try:
            self.renderer.end_offline_accumulation()
        except Exception as cleanup_error:
            if exc_value is None:
                raise
            # Keep the render failure primary while exposing a cleanup failure.
            raise exc_value.with_traceback(traceback) from cleanup_error

    def run(
        self,
        progress_callback: Optional[Callable[[OfflineProgress], None]],
    ) -> OfflineResult:
        renderer = self.renderer
        settings = self.settings
        target_samples = self.target_samples
        rendered = 0
        metrics: dict[str, float] | None = None
        metric_history: list[dict[str, float]] = []
        started = time.perf_counter()

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

        return self._finish(rendered, metrics)

    def _finish(
        self,
        rendered: int,
        metrics: Optional[dict[str, float]],
    ) -> OfflineResult:
        renderer = self.renderer
        params = self.params
        settings = self.settings
        target_samples = self.target_samples
        denoiser_used = "none"
        hdr_frame, aov_frame = renderer.resolve_offline_hdr()

        params_python = _params_python_object(params)
        denoise = getattr(params_python, "denoise", None)
        if denoise is not None and denoise.enabled:
            method = str(denoise.method).lower()
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
                passthrough = None
                if self.denoise_terrain_mask is not None:
                    valid = np.asarray(self.denoise_terrain_mask)
                    if valid.dtype != np.bool_ or valid.shape != beauty_hdr.shape[:2]:
                        raise ValueError("denoise terrain guidance must be an aligned boolean pixel mask")
                    passthrough = ~valid
                    original_unguided = beauty_hdr[passthrough].copy()
                    normal_np = normal_np.copy()
                    normal_np[passthrough] = 0

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
                        iterations=int(denoise.iterations),
                        sigma_color=float(denoise.sigma_color),
                        sigma_normal=float(denoise.sigma_normal),
                        sigma_depth=float(denoise.sigma_depth),
                    )
                    denoiser_used = "atrous"

                if denoiser_used != "none":
                    if passthrough is not None:
                        beauty_hdr[passthrough] = original_unguided
                    hdr_frame = renderer.upload_hdr_frame(beauty_hdr, hdr_frame.size)

        frame = renderer.tonemap_offline_hdr(hdr_frame)

        metadata: dict[str, Any] = {
            "samples_used": rendered,
            "denoiser_used": denoiser_used,
            "final_p95_delta": None if metrics is None else metrics["p95_delta"],
            "converged_ratio": None if metrics is None else metrics["converged_tile_ratio"],
            "target_samples": target_samples,
            "adaptive": bool(settings.adaptive),
        }
        if self.denoise_terrain_mask is not None:
            # Native scatter denoising uses jittered single-sample passes.
            metadata["offline_requested_msaa_samples"] = getattr(_params_python_object(params), "msaa_samples", 1)
            metadata["offline_effective_msaa_samples"] = 1
            metadata["denoiser_guide_coverage"] = 'terrain normals; accumulated scatter, sky and their sample-footprint terrain margins pass through without filtering'
        return OfflineResult(
            frame=frame,
            hdr_frame=hdr_frame,
            aov_frame=aov_frame,
            metadata=metadata,
        )


def _visibility_reference_params(params):
    """Keep visibility at unjittered output pixel centres, independent of beauty sampling."""
    from dataclasses import replace
    from . import TerrainRenderParams

    config = _params_python_object(params)
    return TerrainRenderParams(replace(
        config, msaa_samples=1, render_scale=1.0,
        aov=replace(config.aov, enabled=True, building_id=True),
    ))


def _scatter_terrain_reference(renderer, material_set, env_maps, params, heightmap, water_mask):
    """Terrain guide coverage, excluding every actually drawn scatter fragment.

    3D scatter owns scene depth. Fullscreen scatter deliberately uses Always
    without depth writes; its existing native identity attachment owns coverage.
    """
    config = _params_python_object(params)
    reference_params = _visibility_reference_params(params)
    _, reference = renderer.render_with_aov(
        material_set=material_set, env_maps=env_maps, params=reference_params,
        heightmap=heightmap, water_mask=water_mask,
    )
    terrain_depth, scene_depth = reference._visibility_depth()
    terrain = (terrain_depth < 1) & (terrain_depth == scene_depth)
    identities = np.asarray(reference._building_ids())
    if identities.dtype != np.uint32 or identities.shape != terrain.shape:
        raise RuntimeError("native scatter identity must share the denoise guide viewport")
    terrain &= identities == 0
    # Native jitter reaches half an internal pixel; a linear frame resolve
    # adds one internal pixel of support. At scale .5 the measured footprint
    # is three output pixels. Scale 1 bypasses that resolve and needs one.
    # Frame blits clamp at edges; material texture addressing remains separate.
    # Undefined sky guides share the same conservative exclusion.
    from math import ceil
    scale = float(config.render_scale)
    radius = 1 if scale == 1.0 else max(1, ceil(1.5 / scale))
    excluded = ~terrain
    height, width = terrain.shape
    for _ in range(radius):
        padded = np.pad(excluded, 1, mode="constant", constant_values=False)
        excluded = np.logical_or.reduce([
            padded[y:y+height, x:x+width] for y in range(3) for x in range(3)
        ])
    return ~excluded, reference


def _render_offline_impl(
    renderer: Any,
    material_set: Any,
    env_maps: Any,
    params: Any,
    heightmap: np.ndarray,
    *,
    settings: OfflineQualitySettings,
    progress_callback: Optional[Callable[[OfflineProgress], None]] = None,
    water_mask: Optional[np.ndarray] = None,
    _denoise_terrain_mask: Optional[np.ndarray] = None,
) -> OfflineResult:
    if not settings.enabled:
        raise ValueError("render_offline requires OfflineQualitySettings(enabled=True)")

    denoise = getattr(_params_python_object(params), "denoise", None)
    scatter_report = getattr(renderer, "get_scatter_memory_report", None)
    if (_denoise_terrain_mask is None and denoise is not None and denoise.enabled
            and str(denoise.method).lower() != "none" and callable(scatter_report)
            and scatter_report()["batch_count"] > 0):
        _denoise_terrain_mask, _ = _scatter_terrain_reference(
            renderer, material_set, env_maps, params, heightmap, water_mask,
        )

    with _OfflineRenderSession(
        renderer, material_set, env_maps, params, heightmap, settings, water_mask,
        _denoise_terrain_mask,
    ) as session:
        return session.run(progress_callback)


__all__ = ["OfflineProgress", "OfflineResult", "render_offline"]
