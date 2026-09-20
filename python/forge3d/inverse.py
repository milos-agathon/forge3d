# python/forge3d/inverse.py
"""DIFFERENTIA: differentiable inverse path tracing.

Recovers terrain material albedo, sun direction/intensity, and atmospheric
turbidity from a single observed image by reverse-mode gradient descent
through the GPU ReSTIR/hybrid primal (PROMETHEUS). The DEM is a known input;
only materials, sun, and atmosphere are recovered.

The heavy lifting lives in the native ``inverse_solve``/``inverse_render_primal``
symbols (``enable-inverse-pt`` wheel feature); this module is the typed,
documented public surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._native import get_native_module as _get_native_module


class InverseSolveUnavailable(RuntimeError):
    """Raised when the native inverse solver is not built into this wheel."""


def _native_inverse(name: str):
    mod = _get_native_module()
    fn = getattr(mod, name, None) if mod is not None else None
    if fn is None:
        raise InverseSolveUnavailable(
            f"forge3d._forge3d.{name} is unavailable: the extension is not "
            "built, or was built without the 'enable-inverse-pt' Cargo feature"
        )
    return fn


@dataclass
class RecoveredScene:
    """Result of :func:`recover_scene`.

    Attributes:
        albedo: (demH, demW, 3) float32 linear per-texel albedo, clipped [0, 1].
        sun_dir: (3,) float32 unit direction toward the sun.
        sun_intensity: recovered scalar sun intensity.
        turbidity: recovered atmospheric turbidity in [1, 10].
        loss_history: per-iteration mean pixel loss (floats).
        rgba: (H, W, 4) uint8 forward-space render of the recovered scene.
        iterations_run: number of solver iterations actually executed.
        peak_host_visible_bytes: render-scoped allocation-ledger peak — the
            solve's own host-visible bytes (must stay under the 512 MiB
            budget).
        albedo_delta_e2000_median: median per-texel CIEDE2000 between the
            recovered albedo and the `reference_albedo` passed to
            :func:`recover_scene`, computed with the canonical AEQUITAS
            metric (``forge3d._deltae``); ``None`` when no reference was
            given.
    """

    albedo: np.ndarray
    sun_dir: np.ndarray
    sun_intensity: float
    turbidity: float
    loss_history: List[float]
    rgba: np.ndarray
    iterations_run: int
    peak_host_visible_bytes: int
    albedo_delta_e2000_median: Optional[float] = None

    @property
    def sun_angles_deg(self) -> Tuple[float, float]:
        """(azimuth_deg, elevation_deg) of the recovered sun direction."""
        d = np.asarray(self.sun_dir, dtype=np.float64)
        n = float(np.linalg.norm(d)) or 1.0
        d = d / n
        az = float(np.degrees(np.arctan2(d[2], d[0])))
        el = float(np.degrees(np.arcsin(np.clip(d[1], -1.0, 1.0))))
        return az, el


def recover_scene(
    target: np.ndarray,
    dem: np.ndarray,
    *,
    cam: Optional[Dict[str, Any]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    spacing: Tuple[float, float] = (1.0, 1.0),
    exaggeration: float = 1.0,
    init_albedo: Optional[Any] = None,
    sun_azimuth_deg: float = 315.0,
    sun_elevation_deg: float = 45.0,
    sun_intensity: float = 2.5,
    turbidity: float = 2.5,
    sun_color: Tuple[float, float, float] = (1.0, 0.97, 0.92),
    env_map: Optional[np.ndarray] = None,
    env_intensity: float = 0.35,
    iters: int = 32,
    spp: int = 4,
    frames: int = 2,
    tile_size: int = 32,
    seed: int = 7,
    lr_albedo: float = 0.05,
    lr_sun: float = 0.02,
    lr_turbidity: float = 0.02,
    early_stop_tol: float = 1e-4,
    early_stop_patience: int = 8,
    spatial_reuse: bool = True,
    edge_term: bool = True,
    score_correction: bool = True,
    reference_albedo: Optional[Any] = None,
) -> RecoveredScene:
    """Recover albedo/sun/turbidity from a single observed terrain image.

    The solver gets ONLY the beauty image — the observation is the forward
    renderer's beauty output (Reinhard-space u8, no sRGB encode), e.g. the
    return of :func:`forge3d.path_tracing.hybrid_render_terrain_reference`.
    The inverse model is flat-Earth: render observations with
    ``earth_model="flat", refraction_model="none"``.

    Args:
        target: (H, W, 3|4) uint8 observed beauty image (the "photo").
        dem: (demH, demW) float32 heightfield — known geometry input.
        cam: camera dict ``{"origin": (x,y,z), "look_at": (x,y,z),
            "up": (x,y,z), "fov_y": deg, "exposure": float}``.
        init_albedo: (demH, demW, 3) float32 initial guess or (r,g,b) triple;
            defaults to flat 0.5.
        frames: accumulation frames re-rendered per gradient evaluation
            (the forward ReSTIR loop length).
        tile_size: frames of reservoir history kept per adjoint checkpoint
            tile (>= 8); when ``frames > tile_size`` the solver replays the
            seed-deterministic forward chain per tile instead of caching
            every per-frame reservoir snapshot.
        spatial_reuse / edge_term / score_correction: estimator switches
            (ReSTIR spatial reuse, terrain and shadow-boundary handling,
            categorical likelihood-score correction).
        reference_albedo: optional (demH, demW, 3) ground-truth linear
            albedo; when given, the result carries
            ``albedo_delta_e2000_median`` — the AEQUITAS CIEDE2000 score,
            computed by the canonical ``forge3d._deltae`` implementation
            (the metric is never reimplemented natively).
    """
    fn = _native_inverse("inverse_solve")

    dem = np.ascontiguousarray(dem, dtype=np.float32)
    if dem.ndim != 2:
        raise ValueError(f"dem must be (H, W); got shape {dem.shape}")
    target = np.ascontiguousarray(target)
    if target.dtype == np.float32 or target.dtype == np.float64:
        if target.max(initial=0.0) <= 1.0:
            target = (np.clip(target, 0.0, 1.0) * 255.0 + 0.5)
        target = np.clip(target, 0.0, 255.0).astype(np.uint8)
    target = np.ascontiguousarray(target, dtype=np.uint8)
    if target.ndim == 3 and target.shape[2] == 3:
        alpha = np.full(target.shape[:2] + (1,), 255, dtype=np.uint8)
        target = np.concatenate([target, alpha], axis=2)
    if target.ndim != 3 or target.shape[2] != 4:
        raise ValueError(f"target must be (H, W, 3|4); got shape {target.shape}")
    width = int(width or target.shape[1])
    height = int(height or target.shape[0])

    cam = dict(cam or {})
    cam.setdefault("origin", (0.0, 50.0, 120.0))
    cam.setdefault("look_at", (0.0, 0.0, 0.0))
    cam.setdefault("up", (0.0, 1.0, 0.0))
    cam.setdefault("fov_y", 45.0)
    cam.setdefault("exposure", 1.0)

    if init_albedo is not None and not isinstance(init_albedo, tuple):
        init_albedo = np.ascontiguousarray(init_albedo, dtype=np.float32)
    if reference_albedo is not None:
        reference_albedo = np.ascontiguousarray(reference_albedo, dtype=np.float32)
        if reference_albedo.shape != dem.shape + (3,):
            raise ValueError(
                "reference_albedo must have shape "
                f"{dem.shape + (3,)}, got {reference_albedo.shape}"
            )
    if env_map is not None:
        env_map = np.ascontiguousarray(env_map, dtype=np.float32)

    out = fn(
        target,
        dem,
        width,
        height,
        cam,
        spacing=spacing,
        exaggeration=exaggeration,
        init_albedo=init_albedo,
        sun_azimuth_deg=sun_azimuth_deg,
        sun_elevation_deg=sun_elevation_deg,
        sun_intensity=sun_intensity,
        turbidity=turbidity,
        sun_color=sun_color,
        env_map=env_map,
        env_intensity=env_intensity,
        iters=iters,
        spp=spp,
        frames=frames,
        tile_size=tile_size,
        seed=seed,
        lr_albedo=lr_albedo,
        lr_sun=lr_sun,
        lr_turbidity=lr_turbidity,
        early_stop_tol=early_stop_tol,
        early_stop_patience=early_stop_patience,
        spatial_reuse=spatial_reuse,
        edge_term=edge_term,
        score_correction=score_correction,
    )
    albedo_delta_e2000_median = None
    if reference_albedo is not None:
        # AEQUITAS contract: score with the canonical CIEDE2000 in the
        # production package module — never a native port or reimplementation.
        albedo_delta_e2000_median = _canonical_deltae().median_delta_e2000_linear(
            out["albedo"], reference_albedo
        )
    return RecoveredScene(
        albedo=out["albedo"],
        sun_dir=out["sun_dir"],
        sun_intensity=float(out["sun_intensity"]),
        turbidity=float(out["turbidity"]),
        loss_history=list(out["loss_history"]),
        rgba=out["rgba"],
        iterations_run=int(out["iterations_run"]),
        peak_host_visible_bytes=int(out["peak_host_visible_bytes"]),
        albedo_delta_e2000_median=albedo_delta_e2000_median,
    )


def _canonical_deltae():
    """Load the shipped AEQUITAS CIEDE2000 implementation on demand."""
    from . import _deltae

    return _deltae


def render_primal(
    dem: np.ndarray,
    albedo: Any,
    *,
    cam: Optional[Dict[str, Any]] = None,
    width: int = 256,
    height: int = 256,
    spacing: Tuple[float, float] = (1.0, 1.0),
    exaggeration: float = 1.0,
    sun_azimuth_deg: float = 315.0,
    sun_elevation_deg: float = 45.0,
    sun_intensity: float = 2.5,
    turbidity: float = 2.5,
    sun_color: Tuple[float, float, float] = (1.0, 0.97, 0.92),
    env_map: Optional[np.ndarray] = None,
    env_intensity: float = 0.35,
    spp: int = 8,
    frames: int = 2,
    seed: int = 7,
) -> np.ndarray:
    """Render the differentiable primal at explicit parameters.

    Dispatches the same forward terrain chain the solver dual-dispatches —
    identical to ``hybrid_render_terrain_reference`` at matched
    frames/seed/spp with ``earth_model="flat", refraction_model="none"``.
    Returns the (H, W, 4) uint8 forward beauty image.
    """
    fn = _native_inverse("inverse_render_primal")

    dem = np.ascontiguousarray(dem, dtype=np.float32)
    if not isinstance(albedo, tuple):
        albedo = np.ascontiguousarray(albedo, dtype=np.float32)
    if env_map is not None:
        env_map = np.ascontiguousarray(env_map, dtype=np.float32)

    cam = dict(cam or {})
    cam.setdefault("origin", (0.0, 50.0, 120.0))
    cam.setdefault("look_at", (0.0, 0.0, 0.0))
    cam.setdefault("up", (0.0, 1.0, 0.0))
    cam.setdefault("fov_y", 45.0)
    cam.setdefault("exposure", 1.0)

    out = fn(
        dem,
        width,
        height,
        cam,
        albedo,
        spacing=spacing,
        exaggeration=exaggeration,
        sun_azimuth_deg=sun_azimuth_deg,
        sun_elevation_deg=sun_elevation_deg,
        sun_intensity=sun_intensity,
        turbidity=turbidity,
        sun_color=sun_color,
        env_map=env_map,
        env_intensity=env_intensity,
        spp=spp,
        frames=frames,
        seed=seed,
    )
    return out["rgba"]


__all__ = [
    "InverseSolveUnavailable",
    "RecoveredScene",
    "recover_scene",
    "render_primal",
]
