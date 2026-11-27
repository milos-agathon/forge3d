"""MODUS process memory estimate for the tiled terrain reference renderer."""
from __future__ import annotations

import math
from operator import index
import sys

import numpy as np

from ._memory import aligned_row_size

if sys.platform == "win32":
    import ctypes as ct
    from ctypes import wintypes

    class _Counters(ct.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            *((name, ct.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage",
            )),
        ]

    _query = ct.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    _query.argtypes = [wintypes.HANDLE, ct.POINTER(_Counters), wintypes.DWORD]
    _query.restype = wintypes.BOOL


def process_bytes():
    """Current private commit (Windows) or RSS (Unix), and its kernel high-water mark."""
    if sys.platform == "win32":
        counters = _Counters(cb=ct.sizeof(_Counters))
        if not _query(-1, ct.byref(counters), counters.cb):
            raise ct.WinError(ct.get_last_error())
        return counters.PagefileUsage, counters.PeakPagefileUsage
    if sys.platform.startswith("linux"):
        with open("/proc/self/status", "rb") as status:
            rss = {k: int(v.split()[0]) * 1024 for line in status
                   for k, _, v in [line.partition(b":")] if k in (b"VmRSS", b"VmHWM")}
        return rss[b"VmRSS"], rss[b"VmHWM"]
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak *= 1 if sys.platform == "darwin" else 1024
    if sys.platform == "darwin":
        import ctypes
        import os
        # PROC_PIDTASKINFO, beginning with virtual_size and resident_size.
        info = (ctypes.c_uint64 * 12)()
        query = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True).proc_pidinfo
        query.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                          ctypes.c_void_p, ctypes.c_int)
        if query(os.getpid(), 4, 0, info, ctypes.sizeof(info)) < 16:
            raise OSError(ctypes.get_errno(), "proc_pidinfo(PROC_PIDTASKINFO) failed")
        return info[1], peak
    return peak, peak


class MemoryAdmissionRefused(MemoryError):
    code = "memory_admission_refused"

    def __init__(self, limit, estimate):
        self.limit_bytes = limit
        self.estimate = estimate
        super().__init__(
            f"{self.code}: limit={limit}, predicted_peak={estimate['predicted_peak_bytes']}, "
            f"contributors={estimate['contributors']}"
        )


class UnknownInputShape(TypeError):
    """A finite preflight estimate needs dimensions the source did not expose."""

    code = "memory_estimate_unavailable"
    limit_bytes = None


def _shape(value):
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return tuple(map(index, shape))
        except TypeError:
            raise UnknownInputShape("poster preflight needs integer array dimensions") from None
    # Infer rectangular sequence dimensions without asking NumPy to materialize it.
    if isinstance(value, (list, tuple, range)):
        return (len(value),) + (_shape(value[0]) if len(value) else ())
    if np.isscalar(value):
        return ()
    try:
        return memoryview(value).shape
    except TypeError:
        raise UnknownInputShape("poster preflight needs array shape metadata") from None


def _pyramid_bytes(height, width):
    # One pair of float32 values per power-of-two padded DEM cell at every mip.
    w = 1 << max(0, width - 2).bit_length()
    h = 1 << max(0, height - 2).bit_length()
    size = 8 * w * h
    while w > 1 or h > 1:
        w, h = max(1, w // 2), max(1, h // 2)
        size += 8 * w * h
    return size


# Calibration and derivation in docs/guides/recoverable_terrain_renders.md.
_RUNTIME_FIXED = 221362704
# Extra tracked CPU ownership in native poster phases, measured with the
# compact and calibration jobs. Already covered by the fixed process term.
_TRACER_CPU_RUNTIME = 2_166_094 + (2_166_094 - 1_624_403)
_RUNTIME_PER_TILE_PIXEL = 3359
# Largest measured extra private commit per byte of scene texture payload
# after the fixed and tile terms, rounded up (see the independent DEM sweep).
_RUNTIME_PER_SCENE_BYTE = 7
_TILE_RECORD = 2024
_RECOVERY_TILE_RECORD = 2476


def estimate_poster(heightmap, width, height, *, tile=1024, albedo_map=None,
                    env_map=None, mesh_vertices=None, mesh_indices=None, checkpoint_path=None,
                    on_progress=None):
    width, height, tile = map(index, (width, height, tile))
    if min(width, height, tile) <= 0:
        raise ValueError("width, height, and tile must be positive")
    dem_shape = _shape(heightmap)
    if len(dem_shape) != 2:
        raise ValueError(f"heightmap must be 2D (H, W), got shape {dem_shape}")
    tw, th = min(tile, width), min(tile, height)
    pixels = tw * th
    tile_count = ((width + tile - 1) // tile) * ((height + tile - 1) // tile)
    native_inputs = 0
    conversions = {}
    sizes = {}
    conversion_phases = {}
    for name, value, dtype in (
        ("heightmap", heightmap, np.float32), ("albedo_map", albedo_map, np.float32),
        ("env_map", env_map, np.float32), ("mesh_vertices", mesh_vertices, np.float32),
        ("mesh_indices", mesh_indices, np.uint32),
    ):
        size = 0 if value is None else 4 * math.prod(_shape(value))
        sizes[name] = size
        native_inputs += size  # PyO3 collects each argument into its own Rust Vec.
        if isinstance(value, np.ndarray):
            shares_caller = value.flags.c_contiguous and value.dtype == dtype
        elif isinstance(value, memoryview):
            shares_caller = value.c_contiguous and value.format == np.dtype(dtype).char
        else:
            shares_caller = False
        if name == "heightmap":
            if np.ma.isMaskedArray(value):
                # height_array casts the values and mask, then fills missing
                # samples with NaN. The final float32 image survives the call.
                shares_caller = False
                conversion_phases["height_conversion"] = dict(
                    cast_heightmap=size, cast_mask=size // 4, filled_heightmap=size)
            if checkpoint_path is not None and not shares_caller:
                # The normalized array and immutable checkpoint copy overlap.
                conversion_phases["height_snapshot"] = dict(
                    normalized_heightmap=size, checkpoint_copy=size)
        if size and (checkpoint_path is not None or not shares_caller):
            conversions[name + "_conversion"] = size

    pyramid = _pyramid_bytes(*dem_shape)
    # Shape-only admission cannot tell whether the DEM contains NaNs. The
    # masked terrain upload packs height and coverage into RG32Float texels.
    height_texture = 2 * sizes["heightmap"]
    material = max(16, sizes["albedo_map"])
    env_rgb = max(12, sizes["env_map"])
    env_rgba = env_rgb // 3 * 4
    triangles = sizes["mesh_indices"] // 12
    # Padded vertices, cloned indices and SAH nodes (at most 2N-1, 48 B).
    nodes = 48 * max(0, 2 * triangles - 1)
    mesh = sizes["mesh_vertices"] // 12 * 16 + sizes["mesh_indices"] + nodes
    plate = width * height * 4
    record = _RECOVERY_TILE_RECORD if checkpoint_path is not None or on_progress is not None else _TILE_RECORD
    common = dict(conversions, plate=plate, tile_records=tile_count * record)
    # TerrainMinMaxPyramid owns cpu_mips until the tile's TerrainPtScene drops.
    render = dict(common, native_input_vectors=native_inputs, pyramid=pyramid)
    # render_terrain.rs retains prev_init and Rust readback Vecs until return.
    # NumPy takes ownership of the six result Vecs. Reusing the plate slice
    # and releasing the result before the next tile avoids a second set.
    tile_live = dict(render, mesh=mesh + triangles * 4, reservoir_initial_values=80 * pixels,
                     mapped_timing=1024)
    staging = max(80 * pixels, aligned_row_size(tw * 8) * th)
    phases = {
        **conversion_phases,
        "validate": dict(conversions, validation_masks=max(sizes["heightmap"], sizes["albedo_map"]) // 4),
        "pyramid_upload": dict(render, masked_height_texels=height_texture,
                               upload=pyramid + height_texture),
        "material_environment_upload": dict(render, packed_material=material,
            environment_rgb_rgba=env_rgb + env_rgba, upload=material + env_rgba),
        "mesh_build": dict(render, mesh=mesh, triangles_bounds_indices=triangles * (36 + 32 + 4)),
        "reservoir_readback": dict(tile_live, reservoir_copy=80 * pixels, mapped=80 * pixels),
        "statistics_readback": dict(tile_live, readback_vectors=(80 + 8 + 16 + 48) * pixels,
                                    mapped=48 * pixels),
        "aov_readback": dict(tile_live, readback_vectors=(80 + 8 + 16 + 48 + 8 + 8 + 8 + 4) * pixels,
                             variance_radiance_rgba=(4 + 12 + 4) * pixels,
                             mapped=aligned_row_size(tw * 8) * th),
        "publish": dict(tile_live, readback_vectors=(80 + 8 + 16 + 48 + 8 + 8 + 8 + 4) * pixels,
                        result_vectors=(4 + 12 + 12 + 4 + 12 + 4) * pixels),
    }
    phase = max(phases, key=lambda name: sum(phases[name].values()))
    dominant = {k: v for k, v in phases[phase].items() if v}
    known = sum(dominant.values())
    current, previous_peak = process_bytes()
    scene_bytes = height_texture + pyramid + material + env_rgba
    runtime_scene = scene_bytes * _RUNTIME_PER_SCENE_BYTE
    runtime = _RUNTIME_FIXED + pixels * _RUNTIME_PER_TILE_PIXEL + runtime_scene
    contributors = dict(dominant, process_baseline=current, runtime_fixed=_RUNTIME_FIXED,
                        runtime_tile=pixels * _RUNTIME_PER_TILE_PIXEL, runtime_scene=runtime_scene)
    mapped_at_peak = dominant.get("mapped", 0) + dominant.get("mapped_timing", 0)
    cpu_parts = {}
    for parts in phases.values():
        modeled = sum(parts.values()) - parts.get("mapped", 0) - parts.get("mapped_timing", 0)
        tracer = _TRACER_CPU_RUNTIME if "native_input_vectors" in parts else 0
        if modeled + tracer > sum(cpu_parts.values()):
            cpu_parts = dict(buffers_and_inputs=modeled, tracer_runtime=tracer)
    return {
        "metric": "peak_private_commit_bytes" if sys.platform == "win32" else "peak_resident_bytes",
        "process_baseline_bytes": current,
        "prior_peak_bytes": previous_peak, "peak_phase": phase,
        "phase_bytes": {k: sum(parts.values()) for k, parts in phases.items()},
        "contributors": dict(sorted(contributors.items(), key=lambda pair: -pair[1])),
        "known_growth_bytes": known, "input_conversion_bytes": conversions,
        "cpu_at_peak_bytes": known - mapped_at_peak, "host_visible_at_peak_bytes": mapped_at_peak,
        "cpu_peak_bytes": sum(cpu_parts.values()), "cpu_peak_contributors": cpu_parts,
        "host_visible_peak_bytes": staging + 1024,  # timing query readback
        # Tile storage: accum 16, stats 48, three reservoirs 240, G-buffers
        # 32, beauty 8, seven AOVs 48. Fixed scene/light and timing resources.
        "device_local_peak_bytes": 392 * pixels + scene_bytes + mesh + 504 + 1088,
        "predicted_peak_bytes": current + known + runtime,
        "runtime_fixed_bytes": _RUNTIME_FIXED, "runtime_bytes_per_tile_pixel": _RUNTIME_PER_TILE_PIXEL,
        "scene_texture_payload_bytes": scene_bytes,
        "runtime_bytes_per_scene_byte": _RUNTIME_PER_SCENE_BYTE,
        "calculation": "process baseline + largest live phase + runtime_fixed + "
                       "runtime_bytes_per_tile_pixel * largest_tile_pixels + "
                       "runtime_bytes_per_scene_byte * scene_texture_payload_bytes. "
                       "CPU includes the tracked share of fixed runtime during native tile phases.",
    }
