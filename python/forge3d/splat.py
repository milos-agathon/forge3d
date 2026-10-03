# python/forge3d/splat.py
"""SPLAT-FUSED: path-traced Gaussian splats + LiDAR/COPC + terrain fusion.

One physically lit integrator — the hybrid ReSTIR path tracer — renders three
heterogeneous scene representations as ONE scene:

* anisotropic 3D Gaussian splats, intersected *analytically* (closest
  Mahalanobis approach along the ray, no rasterised splatting), acting as
  soft occluders ``T = exp(-kappa * rho)``;
* LiDAR / COPC points, fixed-radius sphelets whose coverage accumulates as
  ``1 - prod(1 - c_i)``, so a dense swath saturates to opaque while a thin
  one stays translucent;
* a terrain heightfield, traced exactly.

Their transmittances multiply (``T_total = T_splat * T_lidar * T_terrain``),
so a splat cloud shadows terrain and terrain shadows a LiDAR swath through
one code path. Primitives live in pages that are streamed on demand into a
fixed GPU residency pool, so the logical scene can hold billions of
primitives while the render stays inside the 512 MiB budget.

The heavy lifting lives in the native ``render_fused`` / ``load_gaussian_splats``
symbols (``splat-fusion`` wheel feature); this module is the typed, documented
public surface.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ._native import get_native_module as _get_native_module

__all__ = [
    "FusedCamera",
    "FusedPointCloud",
    "FusedRenderResult",
    "FusedTerrain",
    "GaussianSplatCloud",
    "HIT_LIDAR",
    "HIT_MISS",
    "HIT_SPLAT",
    "HIT_TERRAIN",
    "SplatFusionUnavailable",
    "build_splat_page_store",
    "load_gaussian_splats",
    "ray_gaussian",
    "render_fused",
    "render_fused_reference",
    "shadow_iou",
    "splat_fusion_available",
    "write_synthetic_point_field",
]

#: ``hit_kind`` values of :class:`FusedRenderResult`.
HIT_MISS = 0
HIT_TERRAIN = 3
HIT_SPLAT = 4
HIT_LIDAR = 5

_PAGE_STORE_MAGIC = b"F3DPGST1"
PathLike = Union[str, "os.PathLike[str]"]


class SplatFusionUnavailable(RuntimeError):
    """Raised when the native fused path is not built into this wheel."""


def _native(name: str):
    mod = _get_native_module()
    fn = getattr(mod, name, None) if mod is not None else None
    if fn is None:
        raise SplatFusionUnavailable(
            f"forge3d._forge3d.{name} is unavailable: the extension is not "
            "built, or was built without the 'splat-fusion' Cargo feature"
        )
    return fn


def splat_fusion_available() -> bool:
    """Whether this build carries the fused splat/LiDAR/terrain integrator."""
    mod = _get_native_module()
    return mod is not None and hasattr(mod, "render_fused")


def _resolve_cloud_class():
    mod = _get_native_module()
    cls = getattr(mod, "GaussianSplatCloud", None) if mod is not None else None
    if cls is not None:
        return cls

    class GaussianSplatCloud:  # type: ignore[no-redef]
        """Placeholder for builds without the ``splat-fusion`` feature."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _native("GaussianSplatCloud")

        @staticmethod
        def from_arrays(*args: Any, **kwargs: Any) -> "GaussianSplatCloud":
            _native("GaussianSplatCloud")
            raise AssertionError("unreachable")

    return GaussianSplatCloud


#: Structure-of-arrays Gaussian splat cloud (native class). Attributes:
#: ``count``, ``sh_degree``, ``byte_size``, ``bounds``, ``positions`` (N, 3),
#: ``scales`` (N, 3 per-axis sigma), ``rotations`` (N, 4 as w, x, y, z),
#: ``opacities`` (N,), ``sh0`` (N, 3) and ``inverse_covariance`` (N, 6, the
#: packed upper triangle the intersection kernel consumes).
GaussianSplatCloud = _resolve_cloud_class()


def load_gaussian_splats(path: PathLike) -> "GaussianSplatCloud":
    """Load a 3D Gaussian Splatting ``.ply``.

    The standard 3DGS layout is expected: ``x y z``, ``scale_0..2`` (log
    sigma), ``rot_0..3`` (w, x, y, z), ``opacity`` (logit), ``f_dc_0..2`` and
    optional ``f_rest_*`` (SH degree 1-3). Activations are applied on load and
    the inverse covariance ``R diag(1/sigma^2) R^T`` is precomputed per splat.

    The cloud is resident (and registered with the memory tracker); a file
    too large for the 512 MiB host budget raises ``MemoryBudgetExceeded`` —
    page it with :func:`build_splat_page_store` instead.
    """
    return _native("load_gaussian_splats")(os.fspath(path))


def build_splat_page_store(
    ply_path: PathLike,
    out_path: PathLike,
    *,
    page_capacity: int = 4096,
    memory_budget_bytes: int = 256 * 1024 * 1024,
) -> Dict[str, int]:
    """Convert a ``.ply`` of any size into an out-of-core splat page store.

    The file is never held in memory: an external bucketed sort bounded by
    ``memory_budget_bytes`` groups splats into spatially compact pages. Pass
    the resulting path as ``splats=`` to :func:`render_fused`.
    """
    return dict(
        _native("build_splat_page_store")(
            os.fspath(ply_path),
            os.fspath(out_path),
            int(page_capacity),
            int(memory_budget_bytes),
        )
    )


def write_synthetic_point_field(
    path: PathLike,
    *,
    page_capacity: int = 4096,
    template_pages: int = 16,
    grid: Tuple[int, int] = (496, 496),
    cell_size: float = 16.0,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    thickness: float = 1.5,
    hole: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    seed: int = 0,
) -> Dict[str, int]:
    """Write a tiled point field whose index aliases a few on-disk pages.

    ``grid[0] * grid[1]`` tiles of ``page_capacity`` points each are indexed
    while only ``template_pages`` payloads are stored, so a billion-point
    *index* costs a few megabytes. Tiles overlapping the xz rectangle
    ``hole = (min_x, min_z, max_x, max_z)`` are left out. Used to exercise
    out-of-core residency at scale.
    """
    return dict(
        _native("write_synthetic_point_field")(
            os.fspath(path),
            page_capacity=int(page_capacity),
            template_pages=int(template_pages),
            grid=(int(grid[0]), int(grid[1])),
            cell_size=float(cell_size),
            origin=tuple(float(v) for v in origin),
            thickness=float(thickness),
            hole=tuple(float(v) for v in hole),
            seed=int(seed),
        )
    )


def ray_gaussian(
    origin: Sequence[float],
    direction: Sequence[float],
    center: Sequence[float],
    scale: Sequence[float],
    rotation: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    opacity: float = 1.0,
    *,
    tmin: float = 0.0,
    tmax: float = math.inf,
    kappa: float = 4.0,
) -> Dict[str, float]:
    """Analytic ray / anisotropic-Gaussian intersection for one splat.

    Returns ``t_star`` (ray parameter of closest Mahalanobis approach,
    ``-(D^T S d) / (d^T S d)``), ``g_star`` (squared Mahalanobis distance
    there), ``response`` (``opacity * exp(-g_star / 2)`` times the fraction of
    the Gaussian's line integral inside ``[tmin, tmax]``; zero beyond 3
    sigma), ``t_hit`` (entry into the 1-sigma shell) and ``transmittance``
    (``exp(-kappa * response)``). This is the CPU mirror of the WGSL kernel
    the GPU traversal runs.
    """
    return dict(
        _native("splat_ray_gaussian")(
            [float(v) for v in origin],
            [float(v) for v in direction],
            [float(v) for v in center],
            [float(v) for v in scale],
            [float(v) for v in rotation],
            float(opacity),
            tmin=float(tmin),
            tmax=float(tmax),
            kappa=float(kappa),
        )
    )


@dataclass(frozen=True)
class FusedCamera:
    """Pinhole camera of a fused render (world units, y up)."""

    origin: Tuple[float, float, float]
    look_at: Tuple[float, float, float]
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov_y_deg: float = 45.0


@dataclass(frozen=True)
class FusedTerrain:
    """Terrain heightfield of a fused render.

    ``heights`` is a ``(rows, cols)`` float array centred on the world
    origin: column ``c`` sits at ``x = (c - (cols - 1) / 2) * spacing[0]``,
    row ``r`` at ``z = (r - (rows - 1) / 2) * spacing[1]``, height along y.

    ``albedo_map`` is an optional ``(rows, cols, 3|4)`` uint8 sRGB colour map
    on the same grid (4 bytes per cell on the GPU). With four channels, alpha
    255 marks a mapped cell and any other alpha falls back to ``albedo``.
    ``albedo_sampling`` is ``"bilinear"`` or ``"nearest"`` (categorical maps).
    """

    heights: np.ndarray
    spacing: Tuple[float, float] = (1.0, 1.0)
    exaggeration: float = 1.0
    albedo: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    albedo_map: Optional[np.ndarray] = None
    albedo_sampling: str = "bilinear"


@dataclass(frozen=True)
class FusedPointCloud:
    """A LiDAR source: a COPC file or a forge3d point page store.

    For COPC, ``origin`` is the LAS coordinate (f64) that maps to the scene
    origin; with ``z_up`` (the LAS convention) points are re-axed to the
    tracer's y-up frame as ``(east, height, -north)``.
    """

    path: PathLike
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    z_up: bool = True


@dataclass
class FusedRenderResult:
    """Beauty, AOVs and paging/memory statistics of a fused render.

    Attributes:
        rgba: (H, W, 4) uint8 tonemapped beauty.
        radiance: (H, W, 3) float32 mean linear radiance.
        albedo, normal, position, direct: (H, W, 3) float32 centre-ray AOVs
            (``direct`` is the sun radiance reaching the surface).
        depth: (H, W) float32 hit distance (NaN on a miss).
        transmittance: (H, W, 4) float32 unified occlusion toward the sun at
            the centre-ray hit: ``(T_total, T_splat, T_lidar, T_terrain)``.
        hit_kind: (H, W) uint8 — :data:`HIT_MISS`, :data:`HIT_TERRAIN`,
            :data:`HIT_SPLAT` or :data:`HIT_LIDAR`.
        sun_cosine: (H, W) float32 N.L at the hit.
        reservoir_visibility: (H, W) float32 last-known sun visibility
            carried by each pixel's merged ReSTIR reservoir (-1 if empty).
        self_bias: (H, W) float32 self-shadow skip distance (world units)
            applied to secondary rays leaving the centre-ray hit (0 on
            terrain and misses).
        stats: frames, restarts, stale_frames, variance, logical_primitives,
            page_count, tlas_node_count, reservoir_valid_count, ``paging``
            (miss_events, loads, evictions, peak_resident_pages, pool_bytes,
            ...) and the memory-tracker peaks (``peak_total_bytes``,
            ``peak_host_visible_bytes``, ``tracker_limit_bytes``, ...).
    """

    rgba: np.ndarray
    radiance: np.ndarray
    albedo: np.ndarray
    normal: np.ndarray
    position: np.ndarray
    direct: np.ndarray
    depth: np.ndarray
    transmittance: np.ndarray
    hit_kind: np.ndarray
    sun_cosine: np.ndarray
    reservoir_visibility: np.ndarray
    self_bias: np.ndarray
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def shadow_mask(self) -> np.ndarray:
        """(H, W) bool: surfaces the unified occlusion model shadows
        (``T_total < 0.5`` at a hit)."""
        return (self.transmittance[..., 0] < 0.5) & (self.hit_kind != HIT_MISS)


def _is_page_store(path: str) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(len(_PAGE_STORE_MAGIC)) == _PAGE_STORE_MAGIC
    except OSError as exc:
        raise FileNotFoundError(f"cannot open fused scene input {path!r}: {exc}") from exc


def _as_items(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _resolve_splats(splats: Any) -> Tuple[Optional[Any], List[str]]:
    """Split ``splats`` into (resident cloud, page-store paths)."""
    cloud = None
    stores: List[str] = []
    for item in _as_items(splats):
        if isinstance(item, (str, os.PathLike)):
            path = os.fspath(item)
            if _is_page_store(path):
                stores.append(path)
                continue
            item = load_gaussian_splats(path)
        if not hasattr(item, "inverse_covariance"):
            raise TypeError(
                "splats must be a GaussianSplatCloud, a .ply path, a splat page "
                f"store path, or a list of those; got {type(item).__name__}"
            )
        if cloud is not None:
            raise ValueError(
                "at most one resident GaussianSplatCloud can be rendered per call; "
                "page additional clouds with GaussianSplatCloud.write_page_store()"
            )
        cloud = item
    return cloud, stores


def _resolve_pointclouds(pointcloud: Any) -> List[Tuple[str, Tuple[float, float, float], bool]]:
    resolved = []
    for item in _as_items(pointcloud):
        if isinstance(item, Mapping):
            item = FusedPointCloud(**item)
        elif isinstance(item, (str, os.PathLike)):
            item = FusedPointCloud(path=item)
        if not isinstance(item, FusedPointCloud):
            raise TypeError(
                "pointcloud must be a path, a FusedPointCloud, a mapping with "
                f"path/origin/z_up, or a list of those; got {type(item).__name__}"
            )
        origin = tuple(float(v) for v in item.origin)
        if len(origin) != 3 or not all(math.isfinite(v) for v in origin):
            raise ValueError("pointcloud origin must be three finite numbers")
        resolved.append((os.fspath(item.path), origin, bool(item.z_up)))
    return resolved


def _resolve_terrain(terrain: Any) -> Optional[FusedTerrain]:
    if terrain is None:
        return None
    if isinstance(terrain, Mapping):
        terrain = FusedTerrain(**terrain)
    elif not isinstance(terrain, FusedTerrain):
        terrain = FusedTerrain(heights=np.asarray(terrain))
    heights = np.ascontiguousarray(terrain.heights, dtype=np.float32)
    if heights.ndim != 2 or min(heights.shape) < 2:
        raise ValueError(
            f"terrain heights must be a 2-D array of at least 2x2 samples, got shape {heights.shape}"
        )
    if not np.isfinite(heights).all():
        raise ValueError("terrain heights contain non-finite samples")
    albedo_map = terrain.albedo_map
    if albedo_map is not None:
        albedo_map = np.asarray(albedo_map)
        if albedo_map.dtype != np.uint8:
            raise TypeError(
                f"terrain albedo_map must be uint8 sRGB, got dtype {albedo_map.dtype}"
            )
        if albedo_map.ndim != 3 or albedo_map.shape[:2] != heights.shape or albedo_map.shape[2] not in (3, 4):
            raise ValueError(
                f"terrain albedo_map shape {albedo_map.shape} must be "
                f"({heights.shape[0]}, {heights.shape[1]}, 3|4)"
            )
        albedo_map = np.ascontiguousarray(albedo_map)
    if terrain.albedo_sampling not in ("bilinear", "nearest"):
        raise ValueError(
            f"terrain albedo_sampling must be 'bilinear' or 'nearest', got {terrain.albedo_sampling!r}"
        )
    return FusedTerrain(
        heights=heights,
        spacing=(float(terrain.spacing[0]), float(terrain.spacing[1])),
        exaggeration=float(terrain.exaggeration),
        albedo=tuple(float(v) for v in terrain.albedo),
        albedo_map=albedo_map,
        albedo_sampling=terrain.albedo_sampling,
    )


def _resolve_camera(camera: Any) -> FusedCamera:
    if isinstance(camera, FusedCamera):
        return camera
    if isinstance(camera, Mapping):
        if "origin" not in camera or "look_at" not in camera:
            raise ValueError("camera mapping requires 'origin' and 'look_at'")
        fov = camera.get("fov_y_deg", camera.get("fov_y", 45.0))
        return FusedCamera(
            origin=tuple(float(v) for v in camera["origin"]),
            look_at=tuple(float(v) for v in camera["look_at"]),
            up=tuple(float(v) for v in camera.get("up", (0.0, 1.0, 0.0))),
            fov_y_deg=float(fov),
        )
    raise TypeError(
        "camera must be a FusedCamera or a mapping with origin/look_at[/up/fov_y_deg]; "
        f"got {type(camera).__name__}"
    )


def _scene_kwargs(
    splats: Any, pointcloud: Any, terrain: Any
) -> Tuple[Dict[str, Any], Optional[FusedTerrain]]:
    cloud, stores = _resolve_splats(splats)
    clouds = _resolve_pointclouds(pointcloud)
    resolved_terrain = _resolve_terrain(terrain)
    if cloud is None and not stores and not clouds and resolved_terrain is None:
        raise ValueError(
            "render_fused needs at least one representation: splats, pointcloud or terrain"
        )
    kwargs: Dict[str, Any] = {
        "splats": cloud,
        "splat_stores": stores,
        "pointclouds": clouds,
    }
    if resolved_terrain is not None:
        kwargs.update(
            heights=resolved_terrain.heights,
            spacing=resolved_terrain.spacing,
            exaggeration=resolved_terrain.exaggeration,
            terrain_albedo_map=resolved_terrain.albedo_map,
            terrain_albedo_sampling=resolved_terrain.albedo_sampling,
        )
    return kwargs, resolved_terrain


def _certificate_arg(certificate: Any) -> Any:
    """Normalise a ``certificate=`` value for the native seam (bool or str)."""
    if certificate is None or isinstance(certificate, bool):
        return bool(certificate)
    return os.fspath(certificate)


def render_fused(
    *,
    splats: Any = None,
    pointcloud: Any = None,
    terrain: Any = None,
    camera: Any,
    samples: int = 64,
    samples_per_frame: Optional[int] = None,
    width: int = 512,
    height: int = 512,
    seed: int = 0,
    exposure: float = 1.0,
    sun_azimuth_deg: float = 135.0,
    sun_elevation_deg: float = 45.0,
    sun_intensity: float = 2.5,
    sun_color: Tuple[float, float, float] = (1.0, 0.97, 0.92),
    sky_turbidity: float = 2.5,
    sky_ground_albedo: float = 0.2,
    sky_intensity: float = 0.35,
    kappa: float = 4.0,
    lidar_radius: float = 0.2,
    lidar_opacity: float = 1.0,
    ibl_occlusion_distance: float = 12.0,
    sun_angular_radius_deg: float = 0.2665,
    splat_self_bias_sigmas: float = 2.0,
    lidar_self_bias_radii: float = 1.0,
    terrain_smooth_normals: bool = True,
    lidar_surfels: bool = True,
    tile: Optional[Tuple[int, int]] = None,
    brdf: str = "lambert",
    roughness: float = 0.6,
    metallic: float = 0.0,
    fog_density: float = 0.0,
    fog_height_falloff: float = 0.0,
    page_capacity: int = 4096,
    splat_page_size: Optional[int] = None,
    splat_slots: int = 96,
    point_slots: int = 96,
    policy: str = "exact",
    return_aovs: bool = False,
    certificate: "bool | str | os.PathLike[str]" = False,
    cache: Optional[str] = None,
) -> Union[np.ndarray, FusedRenderResult]:
    """Path-trace splats, a LiDAR point cloud and terrain as one scene.

    Args:
        splats: a :class:`GaussianSplatCloud`, a ``.ply`` path, a splat page
            store path, or a list of those (at most one resident cloud).
        pointcloud: a COPC path, a point page store path, a
            :class:`FusedPointCloud` (path + f64 origin + z-up flag), or a
            list of those.
        terrain: a 2-D height array or a :class:`FusedTerrain`.
        camera: a :class:`FusedCamera` or a mapping with ``origin``,
            ``look_at`` and optional ``up`` / ``fov_y_deg``.
        samples: camera samples per pixel (accumulated over frames).
        samples_per_frame: camera samples taken per accumulation frame
            (default ``min(samples, 4)``); the frame count is
            ``ceil(samples / samples_per_frame)``.
        sun_*: direction (azimuth measured in the xz plane from +x toward +z,
            elevation above the horizon), intensity and colour of the sun.
        sky_*: Hosek-Wilkie sky turbidity, ground albedo and intensity.
        kappa: splat extinction scale — the optical depth of a fully opaque
            splat through its centre (``T = exp(-kappa * rho)``).
        lidar_radius, lidar_opacity: sphelet radius (world units) and peak
            coverage of a LiDAR return.
        ibl_occlusion_distance: splats/points occlude sky light only within
            this distance (terrain occludes at any distance).
        splat_self_bias_sigmas, lidar_self_bias_radii: secondary rays leaving
            a splat (point) skip the stretch in which they rise this many
            standard deviations (sphelet radii) along the hit normal.
        lidar_surfels: render LiDAR returns on locally planar neighbourhoods
            (ground, roofs, walls) as oriented discs instead of spheres, so a
            dense surface swath does not shadow itself at low sun.
        terrain_smooth_normals: shade terrain with interpolated vertex
            normals (continuous across DEM cells) instead of the per-cell
            patch normal; the intersection is the exact patch either way.
        brdf: surface response evaluated through the shared BRDF dispatcher
            (``lambert``, ``oren_nayar``, ``cook_torrance_ggx``, ...).
        fog_density, fog_height_falloff: exponential height fog along sun
            rays (``VolumetricParams`` convention; composes with splat
            transmittance as a separate Beer-Lambert factor).
        page_capacity, splat_slots, point_slots: residency pool geometry —
            primitives per page and resident pages per kind.
        splat_page_size: primitives per page when a resident cloud is
            paginated (default: the page capacity). Smaller pages give tighter
            traversal boxes.
        policy: ``"exact"`` (a frame that touched a missing page is redone
            once it is streamed in; a working set larger than the pool is an
            error) or ``"progressive"`` (missing pages are requested
            asynchronously and affected samples reuse the reservoir's
            last-known visibility).
        tile: ``(width, height)`` of the seamless tiles the frame is rendered
            in (default: the whole frame up to one megapixel, else 1024x1024).
            Per-pixel GPU memory is bounded by one tile, so the output size
            is not limited by the 512 MiB budget; a tiled render equals the
            single-tile render bit for bit.
        return_aovs: return a :class:`FusedRenderResult` instead of the image.
            Without it only the beauty is read back and the AOV targets are
            not allocated.
        certificate: ``True`` assembles the signed CENSOR render certificate
            of this render (read it with
            :func:`forge3d.diagnostics.render_certificate`); a path also
            writes it there.
        cache: accepted for the ANAMNESIS render contract. The fused
            integrator streams its scene per view and has no render-graph
            cache, so the value is ignored.

    Returns:
        ``(height, width, 4)`` uint8 RGBA, or a :class:`FusedRenderResult`
        when ``return_aovs`` is true.

    Raises:
        ValueError / TypeError: missing or malformed inputs. A representation
            that cannot be rendered is never skipped silently.
        SplatFusionUnavailable: the wheel was built without ``splat-fusion``.
        forge3d.MemoryBudgetExceeded: the render would exceed the 512 MiB
            budget, or (exact policy) its working set does not fit the pool.
        forge3d.DegradedCapability: the device grants too few storage
            buffers per stage for the fused kernel.
    """
    native = _native("render_fused")
    if samples < 1:
        raise ValueError(f"samples must be >= 1, got {samples}")
    if width < 1 or height < 1:
        raise ValueError(f"width and height must be >= 1, got {width}x{height}")
    cam = _resolve_camera(camera)
    scene, resolved_terrain = _scene_kwargs(splats, pointcloud, terrain)
    spp = min(int(samples), 4) if samples_per_frame is None else int(samples_per_frame)
    if not 1 <= spp <= 64:
        raise ValueError(f"samples_per_frame must be in 1..=64, got {spp}")
    frames = int(math.ceil(samples / spp))
    tile_kwargs: Dict[str, int] = {}
    if tile is not None:
        tw, th = (int(v) for v in tile)
        if not (1 <= tw <= width and 1 <= th <= height):
            raise ValueError(
                f"tile {tw}x{th} must be at least 1x1 and fit the {width}x{height} image"
            )
        tile_kwargs = dict(tile_width=tw, tile_height=th)
    out = native(
        cam_origin=cam.origin,
        cam_look_at=cam.look_at,
        cam_up=cam.up,
        fov_y_deg=float(cam.fov_y_deg),
        width=int(width),
        height=int(height),
        spp=spp,
        frames=frames,
        seed=int(seed),
        exposure=float(exposure),
        terrain_albedo=(
            resolved_terrain.albedo if resolved_terrain is not None else (0.5, 0.5, 0.5)
        ),
        sun_azimuth_deg=float(sun_azimuth_deg),
        sun_elevation_deg=float(sun_elevation_deg),
        sun_intensity=float(sun_intensity),
        sun_color=tuple(float(v) for v in sun_color),
        sky_turbidity=float(sky_turbidity),
        sky_ground_albedo=float(sky_ground_albedo),
        sky_intensity=float(sky_intensity),
        kappa=float(kappa),
        lidar_radius=float(lidar_radius),
        lidar_opacity=float(lidar_opacity),
        ibl_occlusion_distance=float(ibl_occlusion_distance),
        sun_angular_radius_deg=float(sun_angular_radius_deg),
        splat_self_bias_sigmas=float(splat_self_bias_sigmas),
        lidar_self_bias_radii=float(lidar_self_bias_radii),
        terrain_smooth_normals=bool(terrain_smooth_normals),
        lidar_surfels=bool(lidar_surfels),
        brdf=str(brdf),
        roughness=float(roughness),
        metallic=float(metallic),
        fog_density=float(fog_density),
        fog_height_falloff=float(fog_height_falloff),
        page_capacity=int(page_capacity),
        splat_page_size=None if splat_page_size is None else int(splat_page_size),
        splat_slots=int(splat_slots),
        point_slots=int(point_slots),
        policy=str(policy),
        aovs=bool(return_aovs),
        certificate=_certificate_arg(certificate),
        cache=cache,
        **tile_kwargs,
        **scene,
    )
    return _fused_result(out, return_aovs)


_RESULT_ARRAYS = (
    "rgba",
    "radiance",
    "albedo",
    "normal",
    "position",
    "direct",
    "depth",
    "transmittance",
    "hit_kind",
    "sun_cosine",
    "reservoir_visibility",
    "self_bias",
)


def _fused_result(out: Mapping[str, Any], return_aovs: bool) -> Union[np.ndarray, FusedRenderResult]:
    """The public result of one native fused render dict."""
    if not return_aovs:
        return out["rgba"]
    stats = {key: value for key, value in out.items() if key not in _RESULT_ARRAYS}
    stats["paging"] = dict(stats["paging"])
    return FusedRenderResult(**{name: out[name] for name in _RESULT_ARRAYS}, stats=stats)


@dataclass(frozen=True)
class FusedView:
    """One view of :func:`render_fused_sequence`: camera, sun, exposure and
    seed (everything else is shared by the sequence)."""

    camera: FusedCamera
    sun_azimuth_deg: float = 135.0
    sun_elevation_deg: float = 45.0
    sun_intensity: float = 2.5
    sun_color: Tuple[float, float, float] = (1.0, 0.97, 0.92)
    exposure: float = 1.0
    seed: int = 0


def render_fused_sequence(
    *,
    splats: Any = None,
    pointcloud: Any = None,
    terrain: Any = None,
    views: Sequence[FusedView],
    samples: int = 64,
    samples_per_frame: Optional[int] = None,
    width: int = 512,
    height: int = 512,
    sky_turbidity: float = 2.5,
    sky_ground_albedo: float = 0.2,
    sky_intensity: float = 0.35,
    kappa: float = 4.0,
    lidar_radius: float = 0.2,
    lidar_opacity: float = 1.0,
    ibl_occlusion_distance: float = 12.0,
    sun_angular_radius_deg: float = 0.2665,
    splat_self_bias_sigmas: float = 2.0,
    lidar_self_bias_radii: float = 1.0,
    terrain_smooth_normals: bool = True,
    lidar_surfels: bool = True,
    tile: Optional[Tuple[int, int]] = None,
    brdf: str = "lambert",
    roughness: float = 0.6,
    metallic: float = 0.0,
    fog_density: float = 0.0,
    fog_height_falloff: float = 0.0,
    page_capacity: int = 4096,
    splat_page_size: Optional[int] = None,
    splat_slots: int = 96,
    point_slots: int = 96,
    policy: str = "exact",
    return_aovs: bool = False,
    certificate: "bool | str | os.PathLike[str]" = False,
    cache: Optional[str] = None,
) -> List[Union[np.ndarray, FusedRenderResult]]:
    """Render several views of one fused scene (a fly-through or sun sweep).

    The scene is built once, the fused kernel is compiled once, and every
    view shares one residency pool: pages streamed in for one view stay
    resident for the next and are evicted least-recently-used when the pool
    is full. Each result equals :func:`render_fused` of that view on its own
    (exact policy). Options mean what they mean for :func:`render_fused`;
    ``stats["paging"]`` counters are cumulative over the sequence and the
    memory peaks cover the whole sequence.

    Returns:
        One ``(height, width, 4)`` uint8 image, or one
        :class:`FusedRenderResult` with ``return_aovs``, per view.
    """
    native = _native("render_fused_sequence")
    if samples < 1:
        raise ValueError(f"samples must be >= 1, got {samples}")
    if width < 1 or height < 1:
        raise ValueError(f"width and height must be >= 1, got {width}x{height}")
    views = list(views)
    if not views:
        raise ValueError("render_fused_sequence needs at least one view")
    native_views = []
    for index, view in enumerate(views):
        if not isinstance(view, FusedView):
            raise TypeError(f"views[{index}] must be a FusedView, got {type(view).__name__}")
        cam = _resolve_camera(view.camera)
        native_views.append(
            dict(
                cam_origin=tuple(float(v) for v in cam.origin),
                cam_look_at=tuple(float(v) for v in cam.look_at),
                cam_up=tuple(float(v) for v in cam.up),
                fov_y_deg=float(cam.fov_y_deg),
                sun_azimuth_deg=float(view.sun_azimuth_deg),
                sun_elevation_deg=float(view.sun_elevation_deg),
                sun_intensity=float(view.sun_intensity),
                sun_color=tuple(float(v) for v in view.sun_color),
                exposure=float(view.exposure),
                seed=int(view.seed),
            )
        )
    scene, resolved_terrain = _scene_kwargs(splats, pointcloud, terrain)
    spp = min(int(samples), 4) if samples_per_frame is None else int(samples_per_frame)
    if not 1 <= spp <= 64:
        raise ValueError(f"samples_per_frame must be in 1..=64, got {spp}")
    frames = int(math.ceil(samples / spp))
    tile_kwargs: Dict[str, int] = {}
    if tile is not None:
        tw, th = (int(v) for v in tile)
        if not (1 <= tw <= width and 1 <= th <= height):
            raise ValueError(
                f"tile {tw}x{th} must be at least 1x1 and fit the {width}x{height} image"
            )
        tile_kwargs = dict(tile_width=tw, tile_height=th)
    outs = native(
        views=native_views,
        width=int(width),
        height=int(height),
        spp=spp,
        frames=frames,
        terrain_albedo=(
            resolved_terrain.albedo if resolved_terrain is not None else (0.5, 0.5, 0.5)
        ),
        sky_turbidity=float(sky_turbidity),
        sky_ground_albedo=float(sky_ground_albedo),
        sky_intensity=float(sky_intensity),
        kappa=float(kappa),
        lidar_radius=float(lidar_radius),
        lidar_opacity=float(lidar_opacity),
        ibl_occlusion_distance=float(ibl_occlusion_distance),
        sun_angular_radius_deg=float(sun_angular_radius_deg),
        splat_self_bias_sigmas=float(splat_self_bias_sigmas),
        lidar_self_bias_radii=float(lidar_self_bias_radii),
        terrain_smooth_normals=bool(terrain_smooth_normals),
        lidar_surfels=bool(lidar_surfels),
        brdf=str(brdf),
        roughness=float(roughness),
        metallic=float(metallic),
        fog_density=float(fog_density),
        fog_height_falloff=float(fog_height_falloff),
        page_capacity=int(page_capacity),
        splat_page_size=None if splat_page_size is None else int(splat_page_size),
        splat_slots=int(splat_slots),
        point_slots=int(point_slots),
        policy=str(policy),
        aovs=bool(return_aovs),
        certificate=_certificate_arg(certificate),
        cache=cache,
        **tile_kwargs,
        **scene,
    )
    return [_fused_result(out, return_aovs) for out in outs]


def render_fused_reference(
    *,
    splats: Any = None,
    pointcloud: Any = None,
    terrain: Any = None,
    camera: Any,
    region: Tuple[Sequence[float], Sequence[float]],
    samples: int = 64,
    width: int = 512,
    height: int = 512,
    seed: int = 0,
    sun_azimuth_deg: float = 135.0,
    sun_elevation_deg: float = 45.0,
    sun_intensity: float = 2.5,
    sun_color: Tuple[float, float, float] = (1.0, 0.97, 0.92),
    kappa: float = 4.0,
    lidar_radius: float = 0.2,
    lidar_opacity: float = 1.0,
    page_capacity: int = 4096,
    splat_page_size: Optional[int] = None,
    min_sun_cosine: float = 0.15,
    lidar_surfels: bool = True,
    certificate: "bool | str | os.PathLike[str]" = False,
    cache: Optional[str] = None,
) -> Dict[str, Any]:
    """Path-traced occlusion reference of a fused scene (AEQUITAS tracer).

    Every soft primitive inside ``region = ((min_x, min_y, min_z), (max_x,
    max_y, max_z))`` is replaced by the hard surface the unified occlusion
    model binarises to — a splat by the iso-ellipsoid where its transmittance
    crosses 1/2, a LiDAR return by the sphere where its coverage crosses 1/2,
    the terrain by its triangle mesh — and the scene is rendered by the
    wavefront path tracer, which shares no traversal or paging code with
    :func:`render_fused`.

    Returns a dict with ``radiance``, ``albedo``, ``normal`` (H, W, 3),
    ``depth`` (H, W), ``receiver_class`` (H, W) uint8 using the
    ``HIT_*`` codes, ``shadow`` (H, W) int8 (-1 unclassified, 0 lit, 1
    shadowed, read from the path-traced radiance) and the proxy counts.

    ``certificate`` follows the render-certificate contract of
    :func:`render_fused`; ``cache`` is accepted for the ANAMNESIS render
    contract and ignored (the reference trace has no render-graph cache).
    """
    native = _native("render_fused_reference")
    cam = _resolve_camera(camera)
    scene, _ = _scene_kwargs(splats, pointcloud, terrain)
    lo, hi = region
    return dict(
        native(
            cam_origin=cam.origin,
            cam_look_at=cam.look_at,
            cam_up=cam.up,
            fov_y_deg=float(cam.fov_y_deg),
            width=int(width),
            height=int(height),
            region=(tuple(float(v) for v in lo), tuple(float(v) for v in hi)),
            frames=int(samples),
            seed=int(seed),
            sun_azimuth_deg=float(sun_azimuth_deg),
            sun_elevation_deg=float(sun_elevation_deg),
            sun_intensity=float(sun_intensity),
            sun_color=tuple(float(v) for v in sun_color),
            kappa=float(kappa),
            lidar_radius=float(lidar_radius),
            lidar_opacity=float(lidar_opacity),
            page_capacity=int(page_capacity),
            splat_page_size=None if splat_page_size is None else int(splat_page_size),
            min_sun_cosine=float(min_sun_cosine),
            lidar_surfels=bool(lidar_surfels),
            certificate=_certificate_arg(certificate),
            cache=cache,
            **scene,
        )
    )


def shadow_iou(fused: np.ndarray, reference: np.ndarray, valid: Optional[np.ndarray] = None) -> float:
    """Intersection-over-union of two boolean shadow masks over ``valid``.

    Returns 0.0 when the union is empty (nothing to score).
    """
    fused = np.asarray(fused, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    if fused.shape != reference.shape:
        raise ValueError(f"mask shapes differ: {fused.shape} vs {reference.shape}")
    if valid is None:
        valid = np.ones_like(fused, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != fused.shape:
        raise ValueError(f"valid mask shape {valid.shape} does not match {fused.shape}")
    union = int(np.count_nonzero((fused | reference) & valid))
    if union == 0:
        return 0.0
    return int(np.count_nonzero(fused & reference & valid)) / union
