"""Native scene execution for ANAMNESIS; no reference pixels are produced here."""
from __future__ import annotations

import copy
from dataclasses import replace
from functools import cache
from concurrent.futures import Future
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import types
from typing import Any, Mapping

import numpy as np

from ._canonical_json import canonical_json_bytes


class _LazyScratch:
    def __init__(self):
        self.directory = None

    def __enter__(self):
        return self

    def __truediv__(self, name):
        if self.directory is None:
            self.directory = tempfile.TemporaryDirectory(prefix="forge3d-anamnesis-scene-")
        return Path(self.directory.name) / name

    def __exit__(self, *args):
        if self.directory is not None:
            self.directory.cleanup()


_DEFAULT_ENVIRONMENT = threading.local()


@contextmanager
def _native_preparation(cache, *, enabled):
    from ._native import get_native_module
    from .anamnesis import _preload_packed_store, _signature_executor
    initializer = getattr(get_native_module(), "_anamnesis_prepare_context", None) if enabled else None
    previous = getattr(_DEFAULT_ENVIRONMENT, "context_future", None)
    started = None
    if isinstance(initializer, types.BuiltinFunctionType) and previous is None:
        started = _signature_executor().submit(initializer)
        _DEFAULT_ENVIRONMENT.context_future = started
    try:
        with _preload_packed_store(cache if initializer is not None else None):
            yield
    finally:
        import sys
        failed = sys.exc_info()[0] is not None
        try:
            if started is not None:
                try:
                    started.result()
                except Exception:
                    if not failed:
                        raise
        finally:
            _DEFAULT_ENVIRONMENT.context_future = previous

def _prepared_resource_fingerprint(context_future, resources, environment):
    return resources.anamnesis_resource_fingerprint(environment, context_ready=context_future)


def _default_material_identity():
    material_directory = os.environ.get("FORGE3D_MATERIAL_DIR")
    material_root = Path(material_directory if material_directory is not None else "assets/materials")
    material_sources = []
    for name in ("rock_albedo.png", "grass_albedo.png", "dirt_albedo.png", "snow_albedo.png"):
        path = material_root / name
        material_sources.append((name, hashlib.sha256(path.read_bytes()).digest() if path.exists() else None))
    return material_directory, os.getcwd(), str(material_root.absolute()), tuple(material_sources)


def _default_environment_identity(request):
    """Reuse owned default inputs only while their sources remain identical."""
    if request not in (b"null", b"{}"):
        return None
    import forge3d as f3d
    from . import map_scene as ms
    resolver, writer, factory = ms._native_ibl_path, ms._write_minimal_hdr, ms._terrain_resources_for_recipe
    if any(getattr(fn, "__code__", None) is None or fn.__code__.co_filename != ms.__file__
           for fn in (resolver, writer, factory)):
        return None
    resource_methods = tuple(getattr(ms._TerrainRenderResources, name)
                             for name in ("__init__", "env_maps", "anamnesis_resource_fingerprint"))
    if (ms._TerrainRenderResources.__getattribute__ is not object.__getattribute__
            or any(getattr(fn, "__code__", None) is None or fn.__code__.co_filename != ms.__file__
                   for fn in resource_methods)):
        return None
    constructors = (f3d.MaterialSet.terrain_default, f3d.IBL.from_hdr,
                    f3d.TerrainRenderer._anamnesis_default_resource_fingerprint)
    if not all(isinstance(fn, types.BuiltinFunctionType) for fn in constructors):
        return None
    try:
        material_identity = _default_material_identity()
    except OSError:
        # The ordinary resource path retains the renderer's error handling.
        return None
    identity = (resolver, writer, factory, ms._TerrainRenderResources, *resource_methods, *constructors,
                material_identity,
                os.environ.get("VF_FORCE_LUT_UNORM"), os.environ.get("FORGE3D_DETERMINISTIC"))
    entry = getattr(_DEFAULT_ENVIRONMENT, "entry", None)
    if entry is not None and entry[0] == identity:
        try:
            if hashlib.sha256(entry[2].read_bytes()).digest() == entry[3]:
                return entry[2], entry[4], entry
        except OSError:
            pass
    directory = tempfile.TemporaryDirectory(prefix="forge3d-anamnesis-default-ibl-")
    path = Path(directory.name) / "environment.hdr"
    writer(path)
    content = hashlib.sha256(path.read_bytes()).digest()
    resources = ms._TerrainRenderResources(f3d)
    environment = resources.env_maps(str(path))
    context_future = getattr(_DEFAULT_ENVIRONMENT, "context_future", None)
    if context_future is None:
        fingerprint = resources.anamnesis_resource_fingerprint(environment)
    else:
        from .anamnesis import _signature_executor
        fingerprint = _signature_executor().submit(
            _prepared_resource_fingerprint, context_future, resources, environment)
    if fingerprint is None:
        directory.cleanup()
        return None
    _DEFAULT_ENVIRONMENT.entry = (identity, directory, path, content, fingerprint, resources, environment)
    return path, fingerprint, _DEFAULT_ENVIRONMENT.entry


# Each exclusion is justified and physically rejected as a terrain input in
# tests/anamnesis_irrelevant_inputs.toml. Unknown frame fields stay keyed.
_TERRAIN_FRAME_EXCLUSIONS = (
    "labels", "label_set_hash", "scene_hash", "scene.labels",
    "scene.label_fade_frames", "scene.scene.layers",
)


def _terrain_frame_projection(frame_state):
    for path in _TERRAIN_FRAME_EXCLUSIONS:
        parts = path.split(".")
        node = frame_state
        for part in parts[:-1]:
            node = node.get(part) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(parts[-1], None)
    return frame_state


def _scenes(recipe, frames):
    from .map_scene import MapScene, SceneRecipe

    if isinstance(recipe, (str, os.PathLike)):
        payload = json.loads(Path(recipe).read_text(encoding="utf-8"))
        return _scenes(payload, frames)
    if isinstance(recipe, MapScene):
        return {frame: recipe for frame in frames}
    if isinstance(recipe, SceneRecipe):
        scene = MapScene(recipe=recipe)
        return {frame: scene for frame in frames}
    if isinstance(recipe, Mapping):
        if recipe and all(isinstance(value, MapScene) for value in recipe.values()):
            return {frame: recipe[frame] for frame in frames}
        payload = recipe.get("recipe", recipe)
        if isinstance(payload, Mapping) and payload.get("kind") == "scene_recipe":
            scene = MapScene(recipe=MapScene._recipe_from_dict(payload))
            return {frame: scene for frame in frames}
    return None


def _terrain_has_external_metadata(terrain):
    if terrain.data is not None or terrain.path is None:
        return False
    path = Path(terrain.path)
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return False
    try:
        import rasterio
        if getattr(rasterio, "__forge3d_stub__", False):
            return True
        with rasterio.open(path) as dataset:
            source = path.resolve()
            return any(Path(name).resolve() != source for name in dataset.files)
    except (ImportError, OSError):
        # An incomplete source identity cannot qualify for the single-file
        # snapshot. The ordinary fresh renderer retains its own diagnostics.
        return True


def _cache_bypass_reason(scene):
    from .map_scene import LabelLayer, RasterOverlay, _mapscene_material_settings, _mapscene_vt_config, _mapscene_water_settings

    recipe, output = scene.recipe, scene.recipe.output
    if output is None:
        return "scene_output_missing"
    for unsupported, reason in (
        (str(output.format).lower() != "png", "scene_output_format"),
        (output.bit_depth != 8, "scene_output_bit_depth"),
        (output.samples != 1, "scene_accumulation"),
        (bool(output.hdr), "scene_hdr"),
        (bool(output.aovs), "scene_aov"),
        (str(output.denoiser).lower() not in {"", "none", "off"}, "scene_denoiser"),
        (bool(recipe.pass_specs), "scene_render_passes"),
        (recipe.map_furniture is not None, "scene_furniture"),
        (recipe.terrain.style is not None, "scene_terrain_style"),
        (not all(isinstance(layer, (LabelLayer, RasterOverlay)) for layer in recipe.layers), "scene_layer_type"),
    ):
        if unsupported:
            return reason
    # Pinning owns only the TIFF bytes. GDAL companion files can also provide
    # the grid, CRS, nodata or mask, so keep their original source stack fresh.
    if _terrain_has_external_metadata(recipe.terrain):
        return "scene_terrain_external_metadata"
    for layer in recipe.layers:
        if isinstance(layer, RasterOverlay):
            sources = [(layer.path, layer.data)]
            if layer.secondary_path is not None or layer.secondary_data is not None:
                sources.append((layer.secondary_path, layer.secondary_data))
            for path, data in sources:
                # GeoTIFF/VRT sidecar identity is not captured by a single-file
                # snapshot. Keep those stacks fresh rather than dropping inputs.
                if data is None and Path(path or "").suffix.lower() not in {".png", ".npy"}:
                    return "scene_raster_external_metadata"
    if _mapscene_material_settings(recipe) is not None:
        return "scene_materials"
    if _mapscene_vt_config(recipe) is not None:
        return "material_virtual_textures"
    water = _mapscene_water_settings(recipe)
    if water is not None and water.mask_path is not None:
        return "scene_water_mask"
    return None


def _eligible(scene):
    return _cache_bypass_reason(scene) is None


def _path_free(recipe):
    # to_dict owns its normalized output mapping; stripping its path does
    # not modify the recipe or its metadata.
    value = recipe.to_dict()
    if isinstance(value.get("output"), dict):
        for key in ("path", "directory", "filename"):
            value["output"].pop(key, None)
    return value


def _plain_frame_methods():
    from . import map_scene as ms
    if (getattr(copy.deepcopy, "__code__", None) is None
            or copy.deepcopy.__code__.co_filename != copy.__file__
            or copy._deepcopy_dispatch.get(type(None)) is not copy._deepcopy_atomic):
        return None
    methods = []
    for cls in (ms.SceneRecipe, ms.TerrainSource, ms.LightingPreset, ms.OutputSpec, ms.OrbitCamera):
        serializer = cls.to_dict
        if (cls.__getattribute__ is not object.__getattribute__ or cls.__setattr__ is not object.__setattr__
                or getattr(serializer, "__code__", None) is None or serializer.__code__.co_filename != ms.__file__
                or any(hasattr(type(vars(cls).get(field)), "__get__") for field in cls.__dataclass_fields__)):
            return None
        initializer = cls.__init__
        post_init = getattr(cls, "__post_init__", None)
        if (getattr(initializer, "__code__", None) is None
                or initializer.__code__.co_filename not in ("<string>", ms.__file__)
                or initializer.__code__.co_name != "__init__"
                or post_init is not None and (getattr(post_init, "__code__", None) is None
                                             or post_init.__code__.co_filename != ms.__file__)):
            return None
        methods.append(serializer)
    return tuple(methods)


def _dtype_name(dtype, names):
    # Numeric dtype names are immutable. Keep the dtype alive with its spelling
    # so object-id reuse cannot alias a different dtype within this batch.
    if dtype.fields is not None or dtype.subdtype is not None or dtype.metadata is not None:
        return str(dtype)
    key = id(dtype)
    entry = names.get(key)
    if entry is None:
        entry = names[key] = (dtype, str(dtype))
    return entry[1]


def _plain_frame_key(recipe, render_policy, *, methods=None, dtype_names=None):
    """Type-sensitive equality for callback-free, file-free blank scenes.

    Blank-scene validation reads terrain/lighting/output/policies, not camera.
    Terrain samples and the live render policy remain keyed for every frame.
    The digest is an in-batch memo key, never a stored format.
    """
    from . import map_scene as ms
    from ._native import get_native_module
    if type(recipe) is not ms.SceneRecipe or type(recipe).__getattribute__ is not object.__getattribute__:
        return None
    if (recipe.layers or recipe.pass_specs or recipe.pass_inputs or recipe.map_furniture is not None
            or recipe.reproducibility_profile is not None or recipe.terrain.path is not None
            or recipe.terrain.style is not None or recipe.terrain._serialized_data_summary is not None):
        return None
    methods = _plain_frame_methods() if methods is None else methods
    if methods is None:
        return None
    for record, cls in ((recipe.terrain, ms.TerrainSource), (recipe.lighting, ms.LightingPreset),
                        (recipe.output, ms.OutputSpec), (recipe.camera, ms.OrbitCamera)):
        if type(record) is not cls or getattr(record.to_dict, "__func__", None) is not cls.to_dict:
            return None
    data = recipe.terrain.data
    if data is not None and (type(data) is not np.ndarray or data.dtype.hasobject):
        return None
    if type(recipe.output.path) not in (str, type(None)):
        return None
    terrain = {key: value for key, value in vars(recipe.terrain).items() if key != "data"}
    output = {key: value for key, value in vars(recipe.output).items()
              if key not in ("path", "directory", "filename")}
    dtype = (str(data.dtype) if dtype_names is None else _dtype_name(data.dtype, dtype_names)) if data is not None else None
    projection = (terrain, None if data is None else (data.shape, dtype),
                  vars(recipe.lighting), output, recipe.target_crs,
                  recipe.render_policy, recipe.diagnostics_policy)
    fingerprint = getattr(get_native_module(), "_anamnesis_plain_key", None)
    if fingerprint is None:
        return None
    key = fingerprint(projection, vars(recipe.camera))
    return None if key is None else (*methods, key)


def _reject_gpu_errors():
    from ._native import get_native_module
    errors = [entry for entry in get_native_module().native_degradations()
              if entry["kind"] in {"uncaptured_gpu_error", "validation_error", "device_lost"}]
    if errors:
        raise RuntimeError(f"ANAMNESIS refuses invalid native GPU output: {errors}")


@cache
def _implementation_fingerprint():
    # Like native build identity, this belongs to the imported implementation
    # for this runtime. Reloading the adapter creates a new cache/function.
    code = hashlib.sha256()
    for name in ("_anamnesis_scene.py", "anamnesis.py", "map_scene.py", "_map_scene_common.py", "_map_scene_compiler.py", "_canonical_json.py", "_anamnesis_params.py", "_map_scene_validation.py", "_map_scene_projection.py", "_map_scene_labels.py", "_map_scene_rasters.py", "raster_style.py", "terrain_params.py", "label_plan.py", "text_atlas.py", "_png.py", "helpers/offscreen.py"):
        code.update((Path(__file__).parent / name).read_bytes())
    return code.digest()


_LABEL_COMPOSITOR = threading.local()


def _label_compositor(scene_cls, width, height):
    # The active compositor owns its ordinary GPU resources. Replace it on a
    # size/class change; glyphs, atlas and the background are rebound per run.
    identity = (scene_cls, width, height)
    entry = getattr(_LABEL_COMPOSITOR, "entry", None)
    if entry is None or entry[0] != identity:
        _LABEL_COMPOSITOR.entry = None
        entry = None  # Drop the previous allocation before creating its replacement.
        scene = scene_cls(width, height)
        if not hasattr(scene, "clear_native_text"):
            return scene
        entry = (identity, scene)
        _LABEL_COMPOSITOR.entry = entry
    entry[1].clear_native_text()
    return entry[1]


def _decoded_label_atlas(png, metrics):
    # pin() owns these files and names them by their complete source bytes.
    # Reuse only the currently bound asset; a same-path edit changes identity.
    identity = (png.stem, metrics.stem)
    entry = getattr(_LABEL_COMPOSITOR, "decoded_atlas", None)
    if entry is None or entry[0] != identity:
        from ._png import load_png_rgba
        from .text_atlas import load_atlas_metrics
        image = load_png_rgba(png)
        image.setflags(write=False)
        entry = (identity, image, load_atlas_metrics(metrics))
        _LABEL_COMPOSITOR.decoded_atlas = entry
    return entry[1:]


def _snapshot_recipes(recipes, *, with_status=False):
    import copyreg
    from . import map_scene as ms
    from ._native import get_native_module

    hooks = {"__deepcopy__", "__reduce__", "__reduce_ex__", "__getstate__",
             "__setstate__", "__slots__", "__new__", "__getattribute__", "__getattr__",
             "__setattr__", "__delattr__", "__getnewargs__", "__getnewargs_ex__"}
    ordinary = tuple(kind for kind in (ms.SceneRecipe, ms.TerrainSource, ms.OrbitCamera,
                                       ms.LightingPreset, ms.OutputSpec, ms.LabelLayer)
                     if not hooks.intersection(kind.__dict__)
                     and kind not in copyreg.dispatch_table)
    copier = getattr(get_native_module(), "_anamnesis_snapshot_recipes", None)
    if copier is None:
        # Snapshot preparation and dry-run planning also work without native
        # bindings. Pixel execution still requires the real native renderer.
        owned = {frame: copy.deepcopy(recipe) for frame, recipe in recipes.items()}
        return (owned, False) if with_status else owned
    if with_status and isinstance(copier, types.BuiltinFunctionType) and "return_preflight" in (copier.__text_signature__ or ""):
        return copier(recipes, ordinary, return_preflight=True)
    owned = copier(recipes, ordinary)
    return (owned, False) if with_status else owned


class _OwnedScene:
    """Plain owned inputs; construct the ordinary scene only for actual work."""

    def __init__(self, recipe):
        self.recipe, self.render_policy = recipe, recipe.render_policy
        self.compiled_plan, self._scene = None, None

    def _materialize(self):
        if self._scene is None:
            from .map_scene import MapScene
            self._scene = MapScene(recipe=self.recipe)
            self._scene.render_policy = self.render_policy
            if self.compiled_plan is not None:
                self._scene.compiled_plan = self.compiled_plan
        return self._scene

    def compile_plan(self):
        self.compiled_plan = self._materialize().compile_plan()
        return self.compiled_plan

    def render(self, *args, **kwargs):
        return self._materialize().render(*args, **kwargs)


def _base_recipe(recipe, terrain, environment, layers):
    settings = dict(recipe.lighting.settings or {})
    settings["ibl"] = dict(settings.get("ibl") or {}, path=str(environment))
    return replace(recipe, terrain=terrain,
                   lighting=replace(recipe.lighting, settings=settings), layers=layers)


def _plain_uniform_bytes(value, *, error_context):
    # The guarded project serializer has already normalized every key to a
    # string. Keep the ordinary canonical diagnostics and negative-zero rule.
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("utf-8")
        if b"-0.0" not in encoded:
            return encoded
    except (TypeError, ValueError):
        pass
    return canonical_json_bytes(value, error_context=error_context)


def _uniform_resource_fragments(frame, state):
    # Split the owned top-level state structurally, never by searching user
    # JSON for a marker. The future resource identity is the only missing value.
    before = {key: value for key, value in state.items() if key < "native_resources"}
    after = {key: value for key, value in state.items() if key > "native_resources"}
    context = "ANAMNESIS terrain.render uniforms"
    prefix = (b'{"frame":' + str(frame).encode("ascii") + b',"state":'
              + _plain_uniform_bytes(before, error_context=context)[:-1]
              + (b"," if before else b"") + b'"native_resources":')
    suffix = ((b"," + _plain_uniform_bytes(after, error_context=context)[1:] if after else b"}") + b"}")
    return prefix, suffix


class _SceneExecution:
    @property
    def _anamnesis_scene_adapter(self):
        # A stored self-reference delays all owned frame snapshots until cyclic GC.
        return self

    @property
    def executors(self):
        # The active scheduler owns these bound callbacks. The adapter must not
        # own callbacks back to itself after a completed/failed sequence.
        return {
            "terrain.render": self.terrain,
            "label.compile": self.compile_labels,
            "label.composite": self.labels,
            "frame.output": self.output,
        }

    def __init__(self, scenes, frames, scratch, *, cache_requested=False, capture_results=False):
        from . import map_scene as ms
        from .text_atlas import default_latin_atlas_paths
        from ._map_scene_compiler import _validate_scene_recipe
        from . import _map_scene_compiler as compiler

        recipes = {frame: scene.recipe for frame, scene in scenes.items()}
        snapshots, callback_free = (_snapshot_recipes(recipes, with_status=True) if cache_requested else
                                    ({frame: copy.deepcopy(recipe) for frame, recipe in recipes.items()}, False))
        self.frames, self.scratch = frames, scratch if isinstance(scratch, _LazyScratch) else Path(scratch)
        self.render_results = {} if capture_results else None
        self.cache_requested = cache_requested
        self._pinned_sources = {}
        self._atlas_bindings = {}
        self._environment_leases = []
        # Custom recipe serializers retain the general canonical traversal.
        # The project serializer normalizes nested mappings via _json_safe;
        # its two direct policy/CRS fields must also be ordinary JSON scalars.
        self.fast_uniforms = cache_requested and all(
            type(recipe) is ms.SceneRecipe
            and getattr(recipe.to_dict, "__func__", None) is ms.SceneRecipe.to_dict
            and ms.SceneRecipe.to_dict.__code__.co_filename == ms.__file__
            and type(recipe.target_crs) in {str, type(None)}
            and type(recipe.render_policy) is str
            and type(scenes[frame].render_policy) is str
            for frame, recipe in snapshots.items()
        )
        methods = _plain_frame_methods() if callback_free and self.fast_uniforms else None
        dtype_names = {}
        plain_keys = {frame: _plain_frame_key(recipe, scenes[frame].render_policy, methods=methods, dtype_names=dtype_names)
                      if methods is not None else None for frame, recipe in snapshots.items()}
        constructor = ms.MapScene.__init__
        plan_property = vars(ms.MapScene).get("compiled_plan")
        lazy_scenes = (not capture_results and methods is not None
                       and type(ms.MapScene) is type
                       and ms.MapScene.__new__ is object.__new__
                       and ms.MapScene.__getattribute__ is object.__getattribute__
                       and ms.MapScene.__setattr__ is object.__setattr__
                       and not hasattr(ms.MapScene, "__del__")
                       and isinstance(plan_property, property)
                       and all(getattr(fn, "__code__", None) is not None
                               and fn.__code__.co_filename == ms.__file__
                               for fn in (plan_property.fget, plan_property.fset))
                       and getattr(constructor, "__code__", None) is not None
                       and constructor.__code__.co_filename == ms.__file__)
        self.scenes = {frame: (_OwnedScene(recipe) if lazy_scenes and plain_keys[frame] is not None
                              else ms.MapScene(recipe=recipe)) for frame, recipe in snapshots.items()}
        self.cache_bypass_reason = None
        previous_key, previous_reason = None, None
        for frame, snapshot in self.scenes.items():
            key = plain_keys[frame]
            reason = (previous_reason if key is not None and key == previous_key else _cache_bypass_reason(snapshot))
            previous_key, previous_reason = key, reason
            if reason is not None:
                self.cache_bypass_reason = reason
                break
        self.cache_eligible = self.cache_bypass_reason is None
        self.compiled = {}
        self.base_recipes, self.base_states, self._base_inputs = {}, {}, {}
        self._pending_resources = {}
        self._prepared_uniforms = {}
        self._prepared_label_blob = None
        self._semantic_passes = None
        self._pending_compositor = None
        self.label_plans, self.atlases, self.label_state = {}, {}, {}
        self.atlas_data = {}
        self.payload = ({"kind": "native_scene_sequence"} if self.cache_eligible else
                        {"frames": {str(frame): _path_free(scenes[frame].recipe) for frame in frames}})
        # Capture implementation bytes as well as the native engine identity.
        implementation_fingerprint = _implementation_fingerprint()
        self.fingerprints = {name: implementation_fingerprint for name in self.executors}
        self.contexts = {name: b"inputs-are-declared-per-pass/v1" for name in self.fingerprints}
        environments, resources_by_environment, environment_requests = {}, {}, {}
        static_frame = None
        static_validation = (_validate_scene_recipe.__code__.co_filename == compiler.__file__
                             and compiler._validate_with_projection.__code__.co_filename == compiler.__file__)
        for frame in frames:
            scene = scenes[frame]
            snapshot = self.scenes[frame]
            snapshot.render_policy = scene.render_policy
            recipe_state = None
            if cache_requested and self.cache_eligible and not snapshot.recipe.layers and scene.compiled_plan is None:
                # Always perform the same safety/diagnostic validation before
                # looking at the store. Blank scenes need no label/depth plan;
                # materialize their complete CompiledScenePlan only on a miss.
                static_key = plain_keys[frame] if static_validation else None
                if static_key is not None and static_frame is not None and static_frame[0] == static_key:
                    # The guarded project camera serializer already normalizes
                    # target values and coerces numeric fields. Only its two
                    # optional fields can still contain nested plain values.
                    camera = snapshot.recipe.camera.to_dict()
                    for name in ("near", "far"):
                        camera[name] = ms._json_safe(camera[name])
                    recipe_state = dict(static_frame[1], camera=camera)
                    report, label_plans = static_frame[2], {}
                else:
                    report, label_plans = _validate_scene_recipe(snapshot.recipe)
                    if static_key is not None:
                        recipe_state = _path_free(snapshot.recipe)
                        static_frame = (static_key, dict(recipe_state), report)
                if label_plans:
                    raise RuntimeError("label-free terrain unexpectedly compiled label plans")
                compiled = None
            else:
                compiled = (scene.compile_plan() if scene.compiled_plan is None else
                            scene._compiled_plan_for_current_recipe())
                report = compiled.validation_report
            if report.render_blocked(scene.render_policy):
                raise RuntimeError("MapScene.render blocked by diagnostics")
            self.compiled[frame] = compiled
            self.scenes[frame].compiled_plan = compiled
            if not self.cache_eligible:
                continue
            recipe = self.scenes[frame].recipe
            # Preserve the fresh renderer's source representation and metadata.
            # Injecting decoded data or grid dimensions changes span and nodata
            # semantics. Pin path-backed sources before deriving any inputs.
            owned_terrain = recipe.terrain
            if owned_terrain.data is None and owned_terrain.path is not None:
                owned_terrain = replace(owned_terrain, path=self.pin(Path(owned_terrain.path)))
            height = ms._load_native_heightmap(owned_terrain)
            if height is None:
                raise ms.MapSceneNativeUnavailable(ms.diagnostic_block(layer="terrain", reason="native terrain samples are unavailable", required_native="TerrainRenderer"))
            height = np.ascontiguousarray(height)
            # The callback-free copier already owns this plain array. Other
            # paths retain the second copy before freezing native height data.
            if not (isinstance(snapshot, _OwnedScene) and static_validation
                    and getattr(ms._load_native_heightmap, "__code__", None) is not None
                    and ms._load_native_heightmap.__code__.co_filename == ms.__file__
                    and height.flags.owndata):
                height = height.copy()
            height.setflags(write=False)
            # Own the environment file for the whole planned execution. Its
            # original path and source bytes remain part of the derivation.
            plain_key = plain_keys[frame]
            ibl_request = environment_requests.get(plain_key) if plain_key is not None else None
            if ibl_request is None:
                ibl_request = canonical_json_bytes(dict(recipe.lighting.settings or {}).get("ibl"), error_context="scene IBL settings")
                if plain_key is not None:
                    environment_requests[plain_key] = ibl_request
            environment = environments.get(ibl_request)
            if environment is None:
                default = _default_environment_identity(ibl_request) if cache_requested else None
                if default is not None:
                    environment, resources_by_environment[default[0]], lease = default
                    self._environment_leases.append(lease)
                else:
                    source, disposable = ms._native_ibl_path(recipe)
                    try:
                        environment = self.pin(Path(source))
                    finally:
                        if disposable:
                            Path(source).unlink(missing_ok=True)
                environments[ibl_request] = environment
            if plain_keys[frame] is None:
                settings = dict(recipe.lighting.settings or {})
                settings["ibl"] = dict(settings.get("ibl") or {}, path=str(environment))
                lighting = replace(recipe.lighting, settings=settings)
            terrain_layers = []
            for layer in recipe.layers:
                if isinstance(layer, ms.RasterOverlay):
                    updates = {}
                    if layer.data is None and layer.path is not None:
                        updates["path"] = self.pin(Path(layer.path))
                    if layer.secondary_data is None and layer.secondary_path is not None:
                        updates["secondary_path"] = self.pin(Path(layer.secondary_path))
                    terrain_layers.append(replace(layer, **updates))
            # Key the resolved grid separately; it must not become source
            # metadata that changes the native renderer's size/span semantics.
            grid = ms._terrain_alignment_grid(
                owned_terrain, target_crs=recipe.target_crs or recipe.terrain.crs,
                fallback_shape=height.shape,
            )
            if grid is not None:
                grid = dict(grid, transform=list(grid["transform"]))
            if plain_keys[frame] is not None:
                self._base_inputs[frame] = (recipe, owned_terrain, environment, tuple(terrain_layers))
            else:
                self.base_recipes[frame] = replace(
                    recipe, terrain=owned_terrain, lighting=lighting, layers=tuple(terrain_layers))
            resource_key = resources_by_environment.get(environment)
            if isinstance(resource_key, Future) and resource_key.done():
                resource_key = resource_key.result()
                resources_by_environment[environment] = resource_key
            if resource_key is None:
                resources, ibl = ms._terrain_resources_for_recipe(self._base_recipe(frame))
                resource_key = resources.anamnesis_resource_fingerprint(ibl)
                resources_by_environment[environment] = resource_key
            if resource_key is None:
                self.cache_eligible = False
                self.cache_bypass_reason = "native_resource_identity_unavailable"
                continue
            state = recipe_state if recipe_state is not None else _path_free(recipe)
            state["layers"] = [value for layer, value in zip(recipe.layers, state["layers"], strict=True)
                               if not isinstance(layer, ms.LabelLayer)]
            frozen = compiled.frame if compiled is not None else None
            frame_state = json.loads(frozen.to_json()) if frozen is not None else None
            if frame_state is not None and not terrain_layers:
                # Native terrain reads frame seeds, LOD morphs and residency,
                # not label records. The original frozen frame is still passed
                # unchanged to rendering; physical rejection tests cover this
                # projection and every retained frame field stays in the key.
                _terrain_frame_projection(frame_state)
            if isinstance(resource_key, Future):
                self._pending_resources[frame] = resource_key
            self.base_states[frame] = {
                "recipe": state, "frame": frame_state, "render_policy": scene.render_policy,
                "terrain_grid": grid,
                "terrain_span": max(1.0, ms._terrain_scene_diagonal(owned_terrain)),
                "height": {"shape": list(height.shape), "dtype": _dtype_name(height.dtype, dtype_names), "sha256": hashlib.sha256(height.tobytes()).hexdigest()},
                "native_resources": None if isinstance(resource_key, Future) else bytes(resource_key).hex(),
                "environment": {name: os.environ.get(name) for name in (
                    "FORGE3D_TERRAIN_SHADOW_DEBUG", "VF_COLOR_DEBUG_MODE", "VF_ROUGHNESS_MULT",
                    "VF_SPEC_AA_ENABLED", "VF_SPECAA_SIGMA_SCALE", "VF_FORCE_LUT_UNORM",
                )},
            }
            if terrain_layers:
                from ._map_scene_rasters import raster_content_hashes
                self.base_states[frame]["raster_content"] = raster_content_hashes(self._base_recipe(frame))
            plans = (ms._chronos_label_plans(frozen) if frozen is not None else
                     compiled.label_plans if compiled is not None else {})
            self.label_plans[frame] = plans
            sources, identities, decoded_atlases = {}, {}, {}
            for layer in recipe.layers:
                if not isinstance(layer, ms.LabelLayer):
                    continue
                png, metrics = default_latin_atlas_paths()
                atlas = dict(layer.glyph_atlas or {})
                image_path = atlas.get("image_path")
                metrics_path = atlas.get("metrics_path") or atlas.get("source_path")
                custom = bool(image_path and metrics_path and Path(image_path).exists() and Path(metrics_path).exists())
                if custom:
                    png, metrics = Path(image_path), Path(metrics_path)
                plan = plans.get(layer.layer_id)
                if cache_requested and plan is not None and plan.accepted:
                    source, identity, decoded = self.atlas_binding(Path(png), Path(metrics), custom)
                    sources[layer.layer_id], identities[layer.layer_id] = source, identity
                    decoded_atlases[layer.layer_id] = decoded
                else:
                    pinned_png, pinned_metrics = self.pin(Path(png)), self.pin(Path(metrics))
                    sources[layer.layer_id] = (pinned_png, pinned_metrics, custom)
                    identities[layer.layer_id] = [pinned_png.stem, pinned_metrics.stem, custom]
            self.atlas_data[frame] = decoded_atlases
            self.atlases[frame] = sources
            self.label_state[str(frame)] = {
                "layers": [layer.to_dict() for layer in recipe.layers if isinstance(layer, ms.LabelLayer)],
                "plans": {name: plan.to_dict() for name, plan in plans.items()},
                "atlases": identities,
            }
        if self.cache_eligible:
            _reject_gpu_errors()
            from ._native import get_native_module
            native = get_native_module()
            if (self.fast_uniforms and self._pending_resources
                    and isinstance(native.capabilities, types.BuiltinFunctionType)
                    and isinstance(native.anamnesis_engine_fingerprint, types.BuiltinFunctionType)):
                # CPU declarations do not depend on device creation. Prepare
                # owned canonical fragments and graph descriptions while the
                # ordinary device initializes; no key or cache hit is inferred.
                self._semantic_passes = []
                def prepare(label, frame, leaf, inputs):
                    if label == "terrain.render" and frame in self._pending_resources:
                        encoded = _uniform_resource_fragments(frame, leaf)
                    else:
                        encoded = canonical_json_bytes({"frame": frame, "state": leaf},
                                                       error_context=f"ANAMNESIS {label} uniforms")
                    self._prepared_uniforms[label, frame] = encoded
                    prefix = b'{"frame":-1,"state":'
                    if (label == "label.compile" and isinstance(encoded, bytes)
                            and encoded.startswith(prefix) and encoded.endswith(b"}")):
                        # The owned label plan is already canonical inside
                        # these uniforms. Preserve those exact payload bytes.
                        self._prepared_label_blob = (leaf, encoded[len(prefix):-1])
                    self._semantic_passes.append((label, frame, leaf, inputs))
                    return f"{frame}:{label}"
                self._declare_semantics(prepare, frames)
        else:
            self.payload = {"frames": {str(frame): _path_free(scenes[frame].recipe) for frame in frames}}

    def _base_recipe(self, frame):
        if frame not in self.base_recipes:
            self.base_recipes[frame] = _base_recipe(*self._base_inputs.pop(frame))
        return self.base_recipes[frame]

    def to_dict(self):
        return self.payload

    def pin(self, path):
        source = str(path.resolve())
        if source in self._pinned_sources:
            return self._pinned_sources[source]
        data = path.read_bytes()
        target = self.scratch / (hashlib.sha256(data).hexdigest() + path.suffix)
        if not target.exists():
            target.write_bytes(data)
        self._pinned_sources[source] = target
        return target

    def atlas_binding(self, png, metrics, custom):
        # A per-execution spelling key only deduplicates reads. Content hashes
        # below remain the asset identity; aliases may safely be read twice.
        request = (os.fspath(png), os.fspath(metrics), custom)
        if request in self._atlas_bindings:
            return self._atlas_bindings[request]
        contents = (png.read_bytes(), metrics.read_bytes())
        identity = tuple(hashlib.sha256(data).hexdigest() for data in contents)
        entry = getattr(_LABEL_COMPOSITOR, "decoded_atlas", None)
        if entry is not None and entry[0] == identity:
            # Rendering consumes the owned decoded values. These source paths
            # are descriptive only; no later read can observe an in-place edit.
            source, decoded = (png, metrics, custom), entry[1:]
        else:
            pinned = tuple(self.scratch / (digest + path.suffix)
                           for path, digest in zip((png, metrics), identity, strict=True))
            for target, data in zip(pinned, contents, strict=True):
                if not target.exists():
                    target.write_bytes(data)
            source = (*pinned, custom)
            decoded = _decoded_label_atlas(*pinned)
        result = (source, [*identity, custom], decoded)
        self._atlas_bindings[request] = result
        return result

    def encode_uniforms(self, label, value, *, error_context):
        prepared = getattr(self, "_prepared_uniforms", {}).get((label, value["frame"]))
        if prepared is not None:
            if isinstance(prepared, tuple):
                identity = value["state"]["native_resources"].encode("ascii")
                return prepared[0] + b'"' + identity + b'"' + prepared[1]
            return prepared
        if self.fast_uniforms and label in {"terrain.render", "frame.output"}:
            # These leaves are built from recipe.to_dict() (_json_safe),
            # json.loads(frame), and owned scalar resource records. Every key
            # is already a string. The C JSON encoder enforces finite floats;
            # only negative-zero normalization can change its canonical bytes.
            return _plain_uniform_bytes(value, error_context=error_context)
        return canonical_json_bytes(value, error_context=error_context)

    def declare(self, declare, frames):
        resolved = {}
        for frame, resource in self._pending_resources.items():
            if resource not in resolved:
                key = resource.result()
                if key is None:
                    raise RuntimeError("ANAMNESIS native resource identity became unavailable")
                resolved[resource] = bytes(key).hex()
            self.base_states[frame]["native_resources"] = resolved[resource]
        self._pending_resources.clear()
        if self.cache_eligible:
            _reject_gpu_errors()
        if self._semantic_passes is not None:
            instances = {}
            outputs = []
            for label, frame, leaf, inputs in self._semantic_passes:
                resolved_inputs = tuple((binding, instances[instance]) for binding, instance in inputs)
                instance = declare(label, frame, leaf, resolved_inputs)
                instances[f"{frame}:{label}"] = instance
                if label == "frame.output":
                    outputs.append(instance)
            return outputs
        return self._declare_semantics(declare, frames)

    def _declare_semantics(self, declare, frames):
        outputs = []
        if not self.cache_eligible:
            for frame in frames:
                outputs.append(declare("frame.output", frame, self.payload["frames"][str(frame)], ()))
            return outputs
        label_compile = (declare("label.compile", -1, self.label_state, ()) if any(
            plan.accepted for plans in self.label_plans.values() for plan in plans.values()
        ) else None)
        for frame in frames:
            image = declare("terrain.render", frame, self.base_states[frame], ())
            if label_compile is not None and any(plan.accepted for plan in self.label_plans[frame].values()):
                image = declare("label.composite", frame, self.label_state[str(frame)], (("terrain.rgba@0", image), ("label.plan@1", label_compile)))
            outputs.append(declare("frame.output", frame, self.base_states[frame]["recipe"]["output"], (("frame.rgba@0", image),)))
        return outputs

    @contextmanager
    def prepare_executors(self, predicted):
        from . import map_scene as ms
        from ._native import get_native_module
        from .anamnesis import _signature_executor
        native = get_native_module()
        helper = getattr(native, "_anamnesis_prepare_label_scene", None)
        previous = self._pending_compositor
        started = None
        for plan in predicted:
            if plan.label != "label.composite":
                continue
            output = self.scenes[plan.frame].recipe.output
            scene_cls = ms._native_scene_class()
            identity = (scene_cls, output.width, output.height)
            entry = getattr(_LABEL_COMPOSITOR, "entry", None)
            if (isinstance(helper, types.BuiltinFunctionType) and scene_cls is getattr(native, "Scene", None)
                    and (entry is None or entry[0] != identity)):
                started = _signature_executor().submit(helper, output.width, output.height)
                self._pending_compositor = identity, started
            break
        try:
            yield
        finally:
            import sys
            failed = sys.exc_info()[0] is not None
            try:
                if started is not None:
                    try:
                        started.result()
                        if not failed:
                            _reject_gpu_errors()
                    except Exception:
                        if not failed:
                            raise
            finally:
                self._pending_compositor = previous

    def label_compositor(self, scene_cls, width, height):
        pending = self._pending_compositor
        if pending is not None and pending[0] == (scene_cls, width, height):
            _LABEL_COMPOSITOR.entry = (pending[0], pending[1].result())
            self._pending_compositor = None
        return _label_compositor(scene_cls, width, height)

    def terrain(self, state, frame, inputs):
        from .map_scene import _render_native_offscreen_result, MapSceneNativeUnavailable, diagnostic_block
        compiled = self.compiled[frame]
        if compiled is None:
            scene = self.scenes[frame]
            compiled = scene.compile_plan()
            if compiled.validation_report.render_blocked(scene.render_policy):
                raise RuntimeError("MapScene.render blocked by diagnostics")
            self.compiled[frame] = compiled
        result = _render_native_offscreen_result(self._base_recipe(frame), compiled)
        if result is None:
            raise MapSceneNativeUnavailable(diagnostic_block(layer="terrain", reason="native terrain rendering is unavailable", required_native="TerrainRenderer"))
        _reject_gpu_errors()
        if self.render_results is not None:
            self.render_results[frame] = result
        return np.ascontiguousarray(result.rgba, dtype=np.uint8).tobytes()

    def compile_labels(self, state, frame, inputs):
        prepared = self._prepared_label_blob
        if prepared is not None and state is prepared[0]:
            return prepared[1]
        return canonical_json_bytes(state, error_context="native ANAMNESIS frozen label plans")

    def labels(self, state, frame, inputs):
        from .map_scene import _composite_native_label_layers, MapSceneNativeUnavailable, diagnostic_block
        output = self.scenes[frame].recipe.output
        base = np.frombuffer(inputs[0], dtype=np.uint8).reshape(output.height, output.width, 4).copy()
        try:
            rgba, rendered = _composite_native_label_layers(
                base, self.scenes[frame].recipe, self.label_plans[frame],
                atlas_sources=self.atlases[frame],
                scene_factory=self.label_compositor if self.cache_requested else None,
                atlas_data=self.atlas_data[frame] if self.cache_requested else None,
            )
            if not rendered:
                raise MapSceneNativeUnavailable(diagnostic_block(layer="labels", reason="native label composition is unavailable", required_native="native text compositor"))
            _reject_gpu_errors()
            if self.render_results is not None and frame in self.render_results:
                self.render_results[frame] = replace(self.render_results[frame], rgba=rgba)
            return rgba.tobytes()
        except BaseException:
            _LABEL_COMPOSITOR.entry = None
            _LABEL_COMPOSITOR.decoded_atlas = None
            raise

    def output(self, state, frame, inputs):
        from .helpers.offscreen import rgba_to_png_bytes
        output = self.scenes[frame].recipe.output
        suffix = "." + str(output.format).lower() if output is not None else ".png"
        if not self.cache_eligible:
            path = self.scratch / (str(frame) + suffix)
            self.scenes[frame].render(str(path), cache=None)
            return path.read_bytes()
        else:
            rgba = np.frombuffer(inputs[0], dtype=np.uint8).reshape(output.height, output.width, 4)
            return rgba_to_png_bytes(rgba, bit_depth=output.bit_depth)


def try_render_sequence(recipe, frames, cache, *, dry_run, max_bytes, verify_reads, capabilities):
    scenes = _scenes(recipe, frames)
    if scenes is None:
        return None
    from .anamnesis import render_sequence
    from .map_scene import MapScene, _shared_terrain_render_context
    with (_native_preparation(cache, enabled=cache is not None and not dry_run and capabilities is None),
          _LazyScratch() as scratch, _shared_terrain_render_context()):
        capture = isinstance(recipe, MapScene) and len(frames) == 1
        adapter = _SceneExecution(scenes, frames, scratch, cache_requested=cache is not None, capture_results=capture)
        context_future = getattr(_DEFAULT_ENVIRONMENT, "context_future", None)
        if context_future is not None:
            # Future.result releases the GIL while waiting. A synchronous
            # native capabilities query would hold it across try_ctx instead,
            # delaying completion of the context and resource workers.
            context_future.result()
        result = render_sequence(
            adapter, frames=frames, cache=cache if adapter.cache_eligible else None,
            dry_run=dry_run, max_bytes=max_bytes, verify_reads=verify_reads,
            pass_executors=adapter.executors,
            pass_executor_fingerprints=adapter.fingerprints,
            pass_executor_contexts=adapter.contexts, capabilities=capabilities,
        )
        if capture:
            # Carry only one frame's result/diagnostics, never the adapter or
            # callbacks back to it; completed sequence snapshots must release.
            result._scene_native_results = adapter.render_results
            result._scene_render_records = {
                frame: (scene.last_render_backend, dict(scene.last_render_metadata or {}),
                        scene.last_validation_report)
                for frame, scene in adapter.scenes.items()
            }
        if cache is not None and not adapter.cache_eligible:
            result.bypass_reason = adapter.cache_bypass_reason
        return result
