"""Shape-only working sets for the native terrain and CHRONOS render routes."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
import math
from pathlib import Path
import sys

import numpy as np

from ._memory import aligned_row_size
from ._poster_memory import UnknownInputShape, _shape, process_bytes
from .config import ShadowParams

# Maximum control growth after modeled buffers were deducted, plus the
# measured spread of those cold residuals across the six control workloads.
# See the qualification table in the recoverable-render guide.
_COLD_RUNTIME = 471258376 + (471258376 - 461195747)
# Cold private commit for a terrain render with decoded vector features,
# after subtracting its shape-based working set. Reserve the observed maximum
# plus the spread across fresh Windows/Vulkan jobs, as for ordinary terrain.
_VECTOR_COLD_RUNTIME = 511_209_135 + (511_209_135 - 477_785_820)
# Tracked CPU growth beyond the frame's shape-derived buffers. For each
# renderer, use the largest calibration residual plus its measured spread.
# These are shares of the process runtime allowance already in admission.
_GRID_CPU_RUNTIME = 22_253_053 + (22_253_053 - 19_688_022)
_OFFLINE_CPU_RUNTIME = 25_288_765 + (25_288_765 - 22_287_198)
_CLIPMAP_CPU_RUNTIME = 34_477_432 + (34_477_432 - 28_863_580)
# Ceiling of the remaining cold 2048-square control growth per normalized
# float32 DEM byte. The repeat of that same render had a much lower OS peak.
_RUNTIME_PER_HEIGHT_BYTE = 45
# Largest cold device-local peak in the native captures, including the
# renderer, environment and default shadow resources.
_RENDERER_DEVICE = 222573122
# The measured device peak includes the startup atlas and its default replacement.
_DEFAULT_ATLAS = 2048**2 * 4
_STARTUP_ATLAS = 2048**2 * 4 * (4 + 8)
# Excess private commit above the ordinary working set for the two cold
# four-sample controls, including the observed spread (see qualification).
_OFFLINE_RUNTIME = 1268712862 + (1268712862 - 1247163830)
# Main controls at 256² and 1024² grew by 225.16 private bytes per added
# pixel. Frame arrays and readback already charge 64 of those bytes.
_OFFLINE_RUNTIME_PER_EXTRA_PIXEL = 162
# Cold 64² offline capture: summed per-label high-water marks minus mapped
# bytes, input heightmap, frame targets and the resized reflection targets.
# This bounds shared textures still alive during offline atlas replacement.
_OFFLINE_RENDERER_DEVICE = 254080414 - 17408 - 16384 - 705536 - 8192


def _raster(data, path):
    if isinstance(data, Mapping):
        data = data.get("array")
    if data is not None:
        shape = _shape(data)
        dtype = getattr(data, "dtype", None)
        itemsize = np.dtype(dtype).itemsize if dtype is not None else np.dtype(float).itemsize
        return math.prod(shape), itemsize
    if not path:
        raise UnknownInputShape("scene preflight needs raster dimensions")
    path = Path(path)
    try:
        if path.suffix.lower() == ".npy":
            with path.open("rb") as stream:
                version = np.lib.format.read_magic(stream)
                reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                shape, _, dtype = reader(stream)
            return math.prod(shape), dtype.itemsize
        if path.suffix.lower() in {".tif", ".tiff"}:
            from ._native import get_native_symbol
            read_info = get_native_symbol("read_raster_info")
            if read_info is not None:
                try:
                    info = read_info(str(path))
                    cells = info.width * info.height * info.band_count
                    return cells, max(np.dtype(dtype).itemsize for dtype in info.dtype_per_band)
                except (OSError, TypeError, ValueError, RuntimeError):
                    # Image overlays can also use TIFF variants supported by Pillow.
                    pass
        from PIL import Image, ImageMode
        with Image.open(path) as image:
            return (image.width * image.height * len(image.getbands()),
                    np.dtype(ImageMode.getmode(image.mode).typestr).itemsize)
    except (OSError, TypeError, ValueError) as exc:
        raise UnknownInputShape(f"cannot read raster dimensions from {path}: {exc}") from exc


def _metadata_bytes(value, summarized_dem):
    # A compiled frame serializes its input graph. Walk the containers without
    # serializing arrays, building a second recipe or materializing geometry.
    if isinstance(value, np.ndarray):
        descriptor = sys.getsizeof(value.shape) + sys.getsizeof(value.dtype)
        if value is summarized_dem:
            return descriptor
        # Styled arrays serialize as nested lists of boxed numbers. The
        # unstyled input DEM serializes only shape, type and hash.
        return descriptor + value.size * (np.dtype(object).itemsize + sys.getsizeof(0.0))
    size = sys.getsizeof(value)
    if isinstance(value, Mapping):
        return size + sum(_metadata_bytes(k, summarized_dem) + _metadata_bytes(v, summarized_dem)
                          for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return size + sum(_metadata_bytes(v, summarized_dem) for v in value)
    if is_dataclass(value) and not isinstance(value, type):
        return size + sum(_metadata_bytes(getattr(value, f.name), summarized_dem) for f in fields(value))
    return size


def _environment_bytes(settings):
    ibl = settings.get("ibl")
    if not isinstance(ibl, Mapping):
        return 0
    # MapScene's packaged clear-sky and missing-source fallbacks are both
    # 2x2 HDRs already included in the default renderer measurement.
    for key in ("path", "hdr_path", "environment_path"):
        if ibl.get(key) and (path := Path(str(ibl[key]))).is_file():
            break
    else:
        return 0
    if path.suffix.lower() in {".hdr", ".rgbe"}:
        with path.open("rb") as stream:
            for line in stream:
                if not line.strip():
                    break
            size = stream.readline().split()
        if len(size) == 4 and size[0] in (b"+Y", b"-Y") and size[2] in (b"+X", b"-X"):
            try:
                w, h = int(size[3]), int(size[1])
            except ValueError:
                pass
            else:
                if min(w, h) > 0:
                    # RGBE source, RGB32f decode, RGBA32f conversion/upload.
                    return w * h * (4 + 12 + 16 + 16)
    raise UnknownInputShape(f"cannot preflight decoded environment dimensions for {path}")


def _clipmap_working_set(config):
    """Bound the planner's mesh and the indexed renderer's retained LOD variants."""
    if config is None:
        return 0, 0, 0
    count = int(config.get("ring_count", 4))
    ring = int(config.get("ring_resolution", 64))
    center = int(config.get("center_resolution", ring))

    def mesh_bytes(levels, ring_res, center_res):
        # ClipmapLevel::validate bounds both spliced boundary coordinates and
        # skirt duplicates. Each grid cell has at most six 32-bit indices.
        ring_side = center_res + 2 * ring_res + 8
        vertices = (center_res + 1)**2 + 2 * levels * ring_side**2
        return vertices * (36 + 6 * 4)

    # Metadata planning accepts 'levels' and uses the raw resolutions. Native
    # indexed rendering clamps the camera-mode arguments in clipmap_camera_config.
    planner = mesh_bytes(max(0, int(config.get("levels", count))), max(0, ring), max(0, center))
    count = min(8, max(1, count if 0 <= count <= 0xffffffff else 4))
    ring = min(256, max(4, ring if 0 <= ring <= 0xffffffff else 64))
    center = min(256, max(4, center if 0 <= center <= 0xffffffff else ring))
    variants = sum(mesh_bytes(count, max(2, ring >> lod), max(2, center >> lod))
                   for lod in range(count))
    # CPU mesh, growable Vec capacity, generation scratch and upload staging.
    # CPU planning finishes before the renderer builds its indexed meshes.
    tiles = count + 1  # MapScene's frustum mode has one chunk per region
    selection = 16 + 56 * tiles
    readback = 2 * selection + 24  # GPU LOD selection and visibility statistics
    draw_buffers = 256 + 80 + 16 + tiles * (2 * 56 + 16 * count + 20 + 80)
    return max(4 * planner, 4 * variants), readback, variants + draw_buffers


def _vt_dimensions(size):
    if isinstance(size, Sequence) and not isinstance(size, (str, bytes)):
        return int(size[0]), int(size[-1] if len(size) == 1 else size[1])
    return int(size), int(size)


def _vt_source_bytes(size, virtual_size, family, path):
    width, height = _vt_dimensions(virtual_size)
    if min(width, height) < 1:
        raise ValueError("VT virtual dimensions must be positive")
    mip_bytes = 4 * width * height
    while width > 1 or height > 1:
        width, height = (width + 1) // 2, (height + 1) // 2
        mip_bytes += 4 * width * height
    if path is not None:
        cells, item = _raster(None, path)
        # Decoded input, conversion to contiguous RGBA8, fallback color and
        # the registration copy. PNG conversion may expand one band to four.
        scratch = cells * item + max(cells, math.prod(_vt_dimensions(virtual_size)) * 4) * 7
    else:
        width, height = (max(16, x) for x in _vt_dimensions(size))
        # Float32 coordinates, modulation, RGB/alpha and two RGBA expression
        # temporaries during procedural albedo creation. Other families fill RGBA8.
        scratch = width * height * (72 if family == "albedo" else 8) + (width + height) * 4
    return mip_bytes, scratch


def _vt_working_set(recipe, config):
    from .map_scene import _mapscene_vt_settings
    if config is None or not config.get("enabled", True):
        return 0, 0, 0
    vt = _mapscene_vt_settings(recipe)
    layer = vt.layers[0]
    chains = scratch = 0
    sources = config.get("sources")
    if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes)):
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            dims = source.get("virtual_size_px", config.get("virtual_size_px", (512, 512)))
            chain, temporary = _vt_source_bytes(source.get("size", dims), dims,
                                               source.get("family", "albedo"), source.get("path"))
            chains += chain
            scratch = max(scratch, temporary)
    elif config.get("procedural_sources"):
        count = max(0, int(config.get("source_count", 4)))
        for family in vt.layers:
            chain, temporary = _vt_source_bytes(family.virtual_size_px, family.virtual_size_px,
                                               family.family, None)
            chains += count * chain
            scratch = max(scratch, temporary if count else 0)

    pages_x, pages_y = ((s + layer.tile_size - 1) // layer.tile_size for s in layer.virtual_size_px)
    width, height = (1 << (s - 1).bit_length() for s in (pages_x, pages_y))
    levels = min(vt.max_mip_levels, max(width, height).bit_length())
    tail_width = sum(max(1, width >> mip) for mip in range(1, levels))
    # Three family slots × four shader material slots, with RGBA32Float entries.
    table = 3 * 4 * 16 * max(width, tail_width) * (height + ((height + 1) // 2 if levels > 1 else 0))
    # Preflight doesn't create an adapter. Raw RGBA8 covers the three BC
    # textures (one byte per texel each) and the compatibility atlas.
    atlas = 4 * vt.atlas_size**2
    slot = 4 * layer.slot_size**2
    resident = min(atlas, int(vt.residency_budget_mb * 1024**2))
    feedback = ((1 << 16) + 2) * 4 if vt.use_feedback else 0
    staging = 3 * max(8 * 1024**2, slot)
    # Registered sources share their mip chains with the prepared runtime.
    # A previous albedo fallback float array can remain live at registration.
    cpu = chains + max(2 * scratch, table + resident + slot + feedback)
    return cpu, feedback, atlas + table + feedback + staging


def _scene(scene, opts, previous_atlas):
    from .map_scene import (
        MapScene, RasterOverlay, VectorOverlay, _mapscene_clipmap_config, _mapscene_native_effects,
        _mapscene_needs_offline, _mapscene_shadow_settings, _mapscene_vt_config,
    )
    if not isinstance(scene, MapScene):
        raise TypeError("scene preflight requires a MapScene")
    recipe = scene.recipe
    output = recipe.output
    if output is None:
        raise UnknownInputShape("scene preflight requires output dimensions")
    width, height = max(64, int(output.width)), max(64, int(output.height))
    pixels = width * height
    terrain = recipe.terrain
    clip_cpu, clip_host, clip_device = _clipmap_working_set(_mapscene_clipmap_config(recipe))
    vt_cpu, vt_host, vt_device = _vt_working_set(recipe, _mapscene_vt_config(recipe))
    n, element = _raster(terrain.data, terrain.path)
    # Source, finite mask, float32 normalization, NaN replacement, canonical
    # hash copy and bytes, native height/coverage and upload staging.
    terrain_bytes = n * (element * 2 + 1 + 4 * 3 + 8 * 2)
    layers = 0
    encoded = _metadata_bytes(recipe, terrain.data if terrain.style is None else None)
    for layer in (terrain, *recipe.layers):
        styled = getattr(layer, "style", None)
        if layer is terrain and styled is None:
            continue
        if isinstance(layer, RasterOverlay) or layer is terrain:
            for data_key, path_key in (("data", "path"), ("secondary_data", "secondary_path")):
                data, path = getattr(layer, data_key, None), getattr(layer, path_key, None)
                if data is None and not path:
                    continue
                cells, item = _raster(data, path)
                # Decoded values/mask, RGBA8 color, UV warp and upload.
                layers += cells * (item + 1 + 4) + n * 8
            if hasattr(styled, "recipe"):
                for material in styled.recipe.layers:
                    for tex in material.textures():
                        cells, item = _raster(None, tex.resolved_path)
                        layers += cells * (item + 4)
        elif not getattr(layer, "features", None) and (
            getattr(layer, "path", None) or getattr(layer, "source", None)
        ):
            raise UnknownInputShape("compressed external geometry requires a decoded allocation estimate")
    settings = recipe.lighting.settings or {}
    environment = _environment_bytes(settings)
    config = settings.get("renderer_config")
    shadow_config = config.get("shadows", {}) if isinstance(config, Mapping) else {}
    shadows = _mapscene_shadow_settings(ShadowParams.from_mapping(shadow_config))
    atlas = shadows._estimate_memory_bytes()
    atlas_key = (shadows.resolution, shadows.cascades, atlas)
    # Native replacement creates the new depth/moment textures before dropping
    # the old atlas. The measured cold device floor already includes startup.
    overlap = (max(0, previous_atlas[2] - _STARTUP_ATLAS)
               if previous_atlas is not None and previous_atlas != atlas_key else 0)
    extra_atlas = max(0, atlas - _DEFAULT_ATLAS) + overlap
    raster_recipe = opts.get("raster_recipe")
    if raster_recipe is not None:
        layers += raster_recipe.grid.width * raster_recipe.grid.height * (3 * 4 + 8 + 4)

    samples = int(opts.get("samples", output.samples))
    offline = _mapscene_needs_offline(output, samples) and not _mapscene_native_effects(recipe)
    guides = bool(output.aovs or opts.get("emit_provenance") or offline)
    guides_bpp = (3 + 3 + 1 + 1) * 4 if guides else 0
    frame = pixels * (4 * 4 + guides_bpp)
    host = aligned_row_size(width * (16 if guides else 4)) * height + 1024  # two timing readback slots
    if offline:
        # HDR/depth and three AOVs, three pairs of RGBA32F accumulators,
        # reference depth, four resolved RGBA16F targets, tonemapped RGBA8,
        # and the quarter-resolution R32F luminance texture. Samples reuse them.
        targets = pixels * (8 + 4 + 3*8 + 6*16 + 4 + 4*8 + 4)
        targets += ((width + 3) // 4) * ((height + 3) // 4) * 4
        renderer_device = _OFFLINE_RENDERER_DEVICE
        reflection = (width // 2) * (height // 2) * 8
    else:
        targets = pixels * (8 + 4 + guides_bpp)
        renderer_device = _RENDERER_DEVICE
        reflection = 0
    device = dict(renderer_with_startup_atlas=renderer_device - _DEFAULT_ATLAS,
                  shadow_atlas=atlas, cold_floor_adjustment=max(0, _DEFAULT_ATLAS - atlas),
                  previous_atlas_excess=overlap, reflection_targets=reflection,
                  terrain=n * 8, layers=layers, environment=environment // 3, frame_targets=targets)
    parts = dict(terrain_normalize_hash_upload=terrain_bytes, raster_layers=layers,
                 frame_buffers=frame, mapped_readback=host, environment=environment)
    if clip_device:
        parts["clipmap_mesh_and_upload"] = clip_cpu
        parts["clipmap_selection_readback"] = clip_host
        device["clipmap_indexed_geometry"] = clip_device
    if vt_device:
        parts.update(vt_sources_and_paging=vt_cpu, vt_feedback_readback=vt_host)
        device["vt_atlas_tables_feedback_and_staging"] = vt_device
    runtime = dict(runtime_terrain=n * 4 * _RUNTIME_PER_HEIGHT_BYTE,
                   runtime_offline=_OFFLINE_RUNTIME if offline else 0,
                   runtime_offline_pixels=((pixels - 64**2) * _OFFLINE_RUNTIME_PER_EXTRA_PIXEL
                                           if offline else 0),
                   runtime_shadow=extra_atlas)
    if any(isinstance(layer, VectorOverlay) for layer in recipe.layers):
        runtime["runtime_vector_overlay"] = _VECTOR_COLD_RUNTIME - _COLD_RUNTIME
    if clip_device and not offline:
        # The first indexed control incurred 1.59 GB of growth at 64².
        # Share the measured cold alternate-pipeline allowance once per frame.
        runtime["runtime_streaming_pipeline"] = _OFFLINE_RUNTIME
    return parts, runtime, device, atlas_key, encoded, pixels


def estimate_scene_job(route, options):
    fly = route == "render_flythrough"
    if fly:
        path = options["camera_path"]
        if isinstance(path, Mapping):
            scenes = (path[key] for key in sorted(path, key=int))
        elif isinstance(path, Sequence):
            scenes = path
        else:
            raise UnknownInputShape("flythrough preflight needs a reusable camera sequence")
    else:
        scenes = (options.get("self", options.get("scene")),)
    largest = {}
    live = device = retained = max_descriptor = cache_frames = host = 0
    previous_atlas = None
    previous_clip_device = previous_clip_host = 0
    largest_runtime = device_parts = {}
    cpu_parts = {}
    cpu = 0
    for scene in scenes:
        parts, runtime, gpu, previous_atlas, descriptor, pixels = _scene(scene, options, previous_atlas)
        # Frozen recipe, plans, provenance records, native compiled frames and
        # manifest encoding. Source DEM pixels remain shared between frames.
        if fly:
            # A replacement geometry provider is built before the old indexed
            # mesh, its GPU buffers, and its two selection readbacks are dropped.
            clip_device = gpu.get("clipmap_indexed_geometry", 0)
            clip_host = parts.get("clipmap_selection_readback", 0)
            if previous_clip_device and clip_device:
                parts["previous_clipmap_mesh"] = 2 * previous_clip_device
                parts["clipmap_selection_readback"] += previous_clip_host
                gpu["previous_clipmap_geometry"] = previous_clip_device
            previous_clip_device, previous_clip_host = clip_device, clip_host
            retained += descriptor * 5
            if options.get("cache") is not None:
                cache_frames += pixels * 4
        max_descriptor = max(max_descriptor, descriptor)
        mapped = (parts["mapped_readback"] + parts.get("vt_feedback_readback", 0)
                  + parts.get("clipmap_selection_readback", 0))
        host = max(host, mapped)
        if sum(gpu.values()) > device:
            device_parts, device = gpu, sum(gpu.values())
        if "clipmap_mesh_and_upload" in parts:
            cpu_runtime = _CLIPMAP_CPU_RUNTIME
        elif runtime["runtime_offline"]:
            cpu_runtime = _OFFLINE_CPU_RUNTIME
        else:
            cpu_runtime = _GRID_CPU_RUNTIME
        frame_cpu = sum(parts.values()) - mapped
        if frame_cpu + cpu_runtime > cpu:
            cpu = frame_cpu + cpu_runtime
            cpu_parts = dict(frame_buffers_and_inputs=frame_cpu, renderer_runtime=cpu_runtime)
        size = sum(parts.values()) + sum(runtime.values())
        if size > live:
            largest, largest_runtime, live = parts, runtime, size
    if not largest:
        raise ValueError("CHRONOS flythrough requires at least one camera frame")
    parts = dict(largest, recipe_copies=3 * max_descriptor, compiled_frames=retained,
                 cached_frames=cache_frames)
    cpu_parts.update(recipe_copies=3 * max_descriptor, compiled_frames=retained, cached_frames=cache_frames)
    known = sum(parts.values())
    baseline, prior = process_bytes()
    parts.update(process_baseline=baseline, runtime_cold=_COLD_RUNTIME, **largest_runtime)
    return dict(entry_point=route, metric=("peak_private_commit_bytes" if sys.platform == "win32"
                                          else "peak_resident_bytes"),
                process_baseline_bytes=baseline, prior_peak_bytes=prior,
                predicted_peak_bytes=sum(parts.values()), known_growth_bytes=known,
                cpu_peak_bytes=sum(cpu_parts.values()), cpu_peak_contributors=cpu_parts,
                host_visible_peak_bytes=host + retained,
                device_local_peak_bytes=device, device_local_contributors=device_parts,
                contributors=dict(sorted(parts.items(), key=lambda kv: -kv[1])),
                calculation="baseline + cold renderer + largest concurrent frame including mesh, VT, "
                            "readbacks and its terrain, pipeline and shadow allowances + recipe copies "
                            "+ retained frames and cache payloads. "
                            "The CPU ledger includes the tracked share of runtime once. "
                            "Device-local textures are reported separately.")
