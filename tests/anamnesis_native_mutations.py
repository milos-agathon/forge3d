"""Executed native cache mutations; no reference-executor pixels are used."""
from __future__ import annotations

import ast
from dataclasses import asdict, fields, is_dataclass, replace
import inspect
import json
import math
from pathlib import Path
import random
import shutil
import struct
import subprocess
import tempfile
import textwrap

import numpy as np
from PIL import Image

import forge3d as f3d
from forge3d._canonical_json import canonical_json_bytes
from forge3d.anamnesis import capability_fingerprint
from forge3d.determinism import _canonical_params_config, canonical_heightmap, write_canonical_hdr
from forge3d import terrain_params as tp


def _paths(value, path=()):
    if is_dataclass(value):
        for field in fields(value):
            yield from _paths(getattr(value, field.name), (*path, field.name))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _paths(item, (*path, index))
    else:
        yield path, value


def native_config(width=64, height=64):
    return replace(_canonical_params_config()(width, height), camera_mode="mesh")


def _replaced(value, path, changed):
    if not path:
        return changed
    head, *tail = path
    if isinstance(head, str):
        return replace(value, **{head: _replaced(getattr(value, head), tail, changed)})
    result = list(value)
    result[head] = _replaced(value[head], tail, changed)
    return tuple(result) if isinstance(value, tuple) else result


def _at_path(value, path):
    for part in path:
        value = getattr(value, part) if isinstance(part, str) else value[part]
    return value


def _float_mutations(value):
    original = struct.pack("<f", value)
    # Mutate the low three IEEE-754 bytes. The high exponent/sign byte can
    # violate shader coordinate ranges despite representing a finite float.
    for bit in range((len(original) - 1) * 8):
        changed = bytearray(original)
        changed[bit // 8] ^= 1 << (bit % 8)
        candidate = struct.unpack("<f", changed)[0]
        if math.isfinite(candidate) and candidate != value:
            yield candidate


# camera_mode has no Python membership validator. These are the documented
# decoder modes; arbitrary strings merely select the existing fallback branch.
_DOCUMENTED_CHOICES = {"camera_mode": ("screen", "mesh", "mesh:zup")}
_EXPLICIT_OPTIONAL_CONTROLS = {
    "vt_store": "store-backed material_virtual_textures bypass",
    "sky.lut_handle": "opaque parameter_identity_unavailable bypass",
    "media": "participating_media bypass (NEPHELE renders uncached)",
}


def _optional_paths(value, path=()):
    if is_dataclass(value):
        for field in fields(value):
            child = getattr(value, field.name)
            child_path = (*path, field.name)
            annotation = str(field.type)
            if "Optional[" in annotation or "None" in annotation:
                yield child_path, child
            yield from _optional_paths(child, child_path)
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from _optional_paths(child, (*path, index))


def _finite_choices(config, path):
    """Read membership domains from the actual parent dataclass validator."""
    name = ".".join(map(str, path))
    if name in _DOCUMENTED_CHOICES:
        return _DOCUMENTED_CHOICES[name]
    parent = _at_path(config, path[:-1])
    validator = getattr(type(parent), "__post_init__", None)
    if validator is None or not isinstance(path[-1], str):
        return ()
    tree = ast.parse(textwrap.dedent(inspect.getsource(validator)))
    names = dict(vars(type(parent)))

    def literal(node):
        if isinstance(node, ast.Name):
            return names[node.id]
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            return getattr(parent, node.attr)
        return ast.literal_eval(node)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                names[node.targets[0].id] = literal(node.value)
            except (ValueError, TypeError, KeyError, AttributeError):
                pass
    aliases = set()
    for loop in (node for node in ast.walk(tree) if isinstance(node, ast.For)):
        if any(isinstance(child, ast.Attribute) and child.attr == path[-1] for child in ast.walk(loop.iter)):
            aliases.update(child.id for child in ast.walk(loop.target) if isinstance(child, ast.Name))
    choices = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or not any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
            continue
        if not any((isinstance(child, ast.Attribute) and child.attr == path[-1]) or
                   (isinstance(child, ast.Name) and child.id in aliases) for child in ast.walk(node.left)):
            continue
        for comparator in node.comparators:
            try:
                domain = literal(comparator)
            except (ValueError, TypeError, KeyError, AttributeError):
                continue
            if isinstance(domain, (tuple, list, set, frozenset)):
                choices.extend(sorted(domain, key=repr))
    return tuple(dict.fromkeys(choices))


def _activate_optional_inputs(config, root):
    """Give optional leaves real values so their children enter the corpus."""
    path = root / "optional-input.png"
    Image.fromarray(np.full((2, 2, 4), 128, dtype=np.uint8)).save(path)
    lut = root / "optional-tonemap.cube"
    lut.write_text("LUT_3D_SIZE 2\n" + "".join(f"{r} {g} {b}\n" for b in (0, 1) for g in (0, 1) for r in (0, 1)), encoding="utf-8")
    # Physical-water alternatives require the actual terrain sample grid,
    # not merely a grid accepted by the standalone dataclass constructor.
    water_shape = canonical_heightmap().shape
    values = {
        "height_curve_lut": np.linspace(0.0, 1.0, 256, dtype=np.float32),
        "fog.base_height": 0.0, "water.mask_path": str(path), "water.level": 0.0,
        "water.optics.surface_height": np.full(water_shape, 0.2, dtype=np.float32),
        "water.optics.surface_normals": np.tile((0.0, 0.0, 1.0), (*water_shape, 1)),
        "water.optics.color": (0.1, 0.2, 0.3),
        "probes.origin": (0.0, 0.0), "probes.spacing": (1.0, 1.0),
        "probes.fallback_blend_distance": 1.0,
        "reflection_probes.origin": (0.0, 0.0), "reflection_probes.spacing": (1.0, 1.0),
        "reflection_probes.fallback_blend_distance": (1.0, 1.0),
        "tonemap.lut_path": str(lut), "aov.output_dir": str(root / "aov-output"),
        "vt": tp.TerrainVTSettings(), "terrain_crs": "EPSG:32610",
        # terrain_default() has four layers; native rendering requires one
        # centre per uploaded layer, in addition to Python's range checks.
        "material_layer_centers": (0.125, 0.375, 0.625, 0.875), "nodata_height_below": -1.0,
    }
    for name, value in values.items():
        config = _replaced(config, tuple(name.split(".")), value)
    return config


_U64_FIELDS = {"aa_seed", "motion_blur.seed", "terrain_data_revision", "vt_upload_budget_bytes"}
_I32_FIELDS = {"lod.level"}  # decode_lighting.rs stores this as i32, not u32.
# Existing production uniform contracts, not test-only caps. Constructor-only
# acceptance is insufficient: integer casts can wrap before these checks.
_NATIVE_INTEGER_RANGES = {
    "lod.level": (0, 65_536, "runtime_contract.rs:u_shading.clamp2"),
    "pom.refine_steps": (0, 128, "runtime_contract.rs:u_shading.pom_steps"),
}
_FILE_COLORS = {
    "materials.normal_path": (128, 128, 255, 255),
    "materials.roughness_path": (128, 128, 128, 255),
    "materials.mask_path": (64, 128, 192, 255),
    "materials.albedo_path": (200, 180, 150, 255),
    "detail.detail_normal_path": (128, 128, 255, 255),
}


def _integer_candidates(config, path, value, rng):
    """Owner-approved boundaries and seeded single-byte integer changes.

    Numeric boundaries come from the public validator and native u32/u64
    types, rather than a test-only cap. Texture dimensions also obey the
    physical device limit. Constructor rejection remains explicit evidence.
    """
    name = ".".join(map(str, path))
    bits = 64 if name in _U64_FIELDS else 32
    maximum = (1 << (bits - 1 if name in _I32_FIELDS else bits)) - 1
    candidates = {0, 1, value - 1, value + 1, maximum}
    if name in _NATIVE_INTEGER_RANGES:
        lower, upper, _ = _NATIVE_INTEGER_RANGES[name]
        candidates.update((lower - 1, lower, lower + 1, upper - 1, upper, upper + 1))
    parent = _at_path(config, path[:-1])
    if isinstance(parent, tp.VTLayerFamily) and path[-1] in {"tile_size", "tile_border"}:
        # Valid integer boundaries are constrained by the atlas divisibility
        # contract. Derive them from that actual atlas rather than guessing.
        atlas = config.vt.atlas_size
        slots = set()
        for divisor in range(1, math.isqrt(atlas) + 1):
            if atlas % divisor == 0:
                slots.update((divisor, atlas // divisor))
        if path[-1] == "tile_size":
            candidates.update(slot - 2 * parent.tile_border for slot in slots)
        else:
            candidates.update((slot - parent.tile_size) // 2 for slot in slots
                              if (slot - parent.tile_size) % 2 == 0)
    if isinstance(path[-1], str) and is_dataclass(parent):
        validator = getattr(type(parent), "__post_init__", None)
        if validator is not None:
            tree = ast.parse(textwrap.dedent(inspect.getsource(validator)))
            for compare in (node for node in ast.walk(tree) if isinstance(node, ast.Compare)):
                if not any(isinstance(node, ast.Attribute) and node.attr == path[-1] for node in ast.walk(compare)):
                    continue
                for node in ast.walk(compare):
                    if isinstance(node, ast.Constant) and type(node.value) is int:
                        candidates.update((node.value - 1, node.value, node.value + 1))
    for byte in range(bits // 8):
        candidates.add(value ^ (rng.randrange(1, 256) << (8 * byte)))
    return sorted(candidates - {value})


def _changed_config(config, path, value):
    if path == ("height_curve_mode",) and value == "lut":
        return replace(config, height_curve_mode=value, height_curve_lut=np.linspace(0.0, 1.0, 256, dtype=np.float32))
    if path == ("detail", "detail_normal_path") and value is None:
        # Removing the normal file is valid only when its strength is zero.
        return replace(config, detail=replace(config.detail, detail_normal_path=None, detail_strength=0.0))
    return _replaced(config, path, value)


def _mutation_config(root):
    config = native_config()
    config.screen_space = tp.ScreenSpaceSettings()
    # Screen-camera alternatives require procedural water. The physical
    # alternative is separately mutated while the baseline camera is mesh.
    config.water = replace(config.water, optics=tp.WaterOpticsSettings(mode="procedural"))
    config.aa_seed = config.motion_blur.seed = config.terrain_data_revision = 0
    file_pixels, file_bytes = {}, {}
    for field, color in _FILE_COLORS.items():
        path = root / (field.replace(".", "-") + ".png")
        pixels = np.full((2, 2, 4), color, dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        config = _replaced(config, tuple(field.split(".")), str(path))
        file_pixels[field], file_bytes[field] = pixels, path.read_bytes()
    config.detail = replace(config.detail, enabled=True, detail_strength=0.5)
    config = _activate_optional_inputs(config, root)
    return config, file_pixels, file_bytes


def _parameter_mutations(config, root, rng, max_dimension):
    categories, rejected, bypasses = [], [], []
    inventory = {}
    optional_paths = list(_optional_paths(config))
    for path, value in _paths(config):
        name = ".".join(map(str, path))
        domain = _finite_choices(config, path)
        finite = bool(domain) or isinstance(value, bool)
        if domain:
            choices = domain
        elif isinstance(value, bool):
            choices = (not value,)
        elif isinstance(value, float):
            choices = list(_float_mutations(value))
        elif isinstance(value, int):
            choices = _integer_candidates(config, path, value, rng)
        elif isinstance(value, str):
            if name in _FILE_COLORS:
                inventory[name] = "file-content bytes and optional presence"
                continue
            if name == "terrain_crs":
                choices = ("EPSG:4326",)
            elif name in {"water.mask_path", "tonemap.lut_path", "aov.output_dir"}:
                alternate = root / ("alternative-" + Path(value).name)
                if Path(value).is_file():
                    shutil.copyfile(value, alternate)
                choices = (str(alternate),)
            else:
                raise AssertionError(f"missing string mutation: {name}")
        elif isinstance(value, (np.ndarray, tp._WaterSurfaceGrid)):
            inventory[name] = "array bytes and optional presence"
            categories.append(("array." + name, path, [None], False))
            continue
        elif value is None and name in _EXPLICIT_OPTIONAL_CONTROLS:
            inventory[name] = _EXPLICIT_OPTIONAL_CONTROLS[name]
            continue
        else:
            raise AssertionError(f"unclassified native parameter: {name} ({type(value).__name__})")
        inventory[name] = "validator alternatives" if finite else "numeric boundaries/bytes or string value"
        accepted = []
        for candidate in choices:
            if candidate == value:
                continue
            try:
                changed = _changed_config(config, path, candidate)
                f3d.TerrainRenderParams(changed)
                if _at_path(changed, path) == value:
                    raise ValueError("constructor canonicalized to the original value")
                if name in _NATIVE_INTEGER_RANGES:
                    lower, upper, contract = _NATIVE_INTEGER_RANGES[name]
                    if not lower <= candidate <= upper:
                        raise ValueError(f"outside production {contract} range [{lower}, {upper}]")
                if path[0] == "size_px" and max_dimension is not None and not 1 <= candidate <= max_dimension:
                    raise ValueError(f"outside physical max_texture_dimension_2d={max_dimension}")
            except (ValueError, TypeError, OverflowError) as error:
                rejected.append({"field": name, "value": candidate, "reason": str(error)})
                continue
            if name == "shadows.technique" and candidate in {"VSM", "EVSM", "MSM"}:
                bypasses.append((name, candidate, changed, "moment_shadows"))
                continue
            if name == "vt.enabled" and candidate is True:
                bypasses.append((name, candidate, changed, "material_virtual_textures"))
                continue
            accepted.append(candidate)
        if not accepted and name == "vt.enabled":
            continue  # Its sole bool alternative is executed by the VT control.
        assert accepted, f"no valid mutation for {name}"
        rng.shuffle(accepted)
        categories.append(("params." + name, path, accepted, finite or isinstance(value, int)))
    for path, value in optional_paths:
        name = ".".join(map(str, path))
        if name in _EXPLICIT_OPTIONAL_CONTROLS:
            continue
        assert value is not None, f"optional parameter was not activated: {name}"
        try:
            changed = _changed_config(config, path, None)
            f3d.TerrainRenderParams(changed)
            normalized = _at_path(changed, path)
            if normalized is not None and np.array_equal(normalized, value):
                raise ValueError("constructor canonicalized None to the original value")
        except (ValueError, TypeError, OverflowError) as error:
            rejected.append({"field": "optional." + name, "value": None, "reason": str(error)})
            continue
        categories.append(("optional." + name, path, [None], True))
    assert set(inventory) == {".".join(map(str, path)) for path, _ in _paths(config)}
    categories.extend((name, None, [None], False) for name in (
        "heightmap", "water_mask", "environment", "material", "time_seconds",
        *("file." + name for name in _FILE_COLORS),
    ))
    return categories, rejected, bypasses, inventory


def _optional_bypass_controls(config, root, heights, mask, ibl, material):
    """Execute the two opaque/contextual optional inputs on real resources."""
    handle = f3d.atmosphere_bake_luts()
    sky = tp.SkySettings(enabled=True, model="aether", lut_handle=handle)
    store = root / "optional-material.f3dvt"
    # Use the production packer to create a complete, small, three-family
    # store. No hand-written header or synthetic page payload is accepted.
    from forge3d.media import Medium

    print("ANAMNESIS optional controls: packing the real VT store", flush=True)
    packed = subprocess.run([
        "cargo", "run", "--quiet", "--release", "--bin", "forge3d-vtpack", "--",
        "--procedural", "--output", str(store), "--virtual-width", "128",
        "--virtual-height", "128", "--tile-size", "128", "--tile-border", "0",
        "--seed", "19", "--coarse-min-mip", "0", "--detail-max-mip", "0",
        "--detail-window-pages", "0",
    ], cwd=Path(__file__).resolve().parents[1], check=False, capture_output=True, text=True)
    pack_log = root / "optional-vtpack.log"
    pack_log.write_text(packed.stdout + "\n" + packed.stderr, encoding="utf-8")
    assert packed.returncode == 0, f"VT packer failed; see {pack_log}"
    vt = tp.TerrainVTSettings(enabled=True, atlas_size=256, layers=[
        tp.VTLayerFamily(family="albedo", virtual_size_px=(128, 128), tile_size=128, tile_border=0),
    ])
    cases = (
        ("sky.lut_handle", replace(config, sky=sky), "parameter_identity_unavailable"),
        ("vt_store", replace(config, vt_store=f3d.open_vt_store(store), vt=vt), "material_virtual_textures"),
        ("media", replace(config, media=Medium((0.01, 0.01, 0.01), (0.02, 0.02, 0.02), 1.0), camera_mode="mesh:yup"), "participating_media"),
    )
    controls = []
    for name, changed, reason in cases:
        print(f"ANAMNESIS optional control: {name}", flush=True)
        f3d.TerrainRenderParams(changed)  # Exercise the real boundary validator.
        outputs = []
        destination = root / ("optional-bypass-" + name)
        for cache in (None, destination):
            renderer = f3d.TerrainRenderer(f3d.Session(window=False))
            f3d.clear_native_degradations()
            pixels = renderer.render_terrain_pbr_pom(
                material, ibl, f3d.TerrainRenderParams(changed), heights,
                water_mask=mask, cache=cache,
            ).to_numpy().tobytes()
            errors = [entry for entry in f3d.native_degradations()
                      if entry["kind"] in {"uncaptured_gpu_error", "validation_error"}]
            assert not errors, errors
            outputs.append(pixels)
        report = dict(renderer.last_anamnesis_cache_report)
        assert outputs[0] == outputs[1], f"optional bypass changed fresh pixels: {name}"
        assert report["bypass_reason"] == reason
        assert not report["hits"] and not report["pass_keys"]
        assert report["bytes_read"] == report["bytes_written"] == 0
        assert not destination.exists()
        controls.append({"field": name, "reason": reason, "independent_fresh_equal": True})
    assert {control["field"] for control in controls} == set(_EXPLICIT_OPTIONAL_CONTROLS)
    return controls


def run_native_mutations(root: Path, count: int) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0xA11A_17)
    config, file_pixels, file_bytes = _mutation_config(root)
    heights = np.ascontiguousarray(canonical_heightmap(), dtype=np.float32)
    mask = np.full_like(heights, 0.5)
    hdr_path = root / "environment.hdr"
    write_canonical_hdr(hdr_path)
    hdr_bytes = hdr_path.read_bytes()
    hdr_pixels = hdr_bytes.index(b"\n", hdr_bytes.index(b"-Y ")) + 1
    ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    material = f3d.MaterialSet.terrain_default()
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    adapter = dict(f3d.device_probe())
    assert adapter.get("software_fallback") is False
    assert str(adapter.get("device_type", "")).lower() in {"discretegpu", "integratedgpu"}
    max_dimension = f3d.capabilities()["limits"]["max_texture_dimension_2d"]

    def render(params, height, water, environment, materials, seconds, cache):
        f3d.clear_native_degradations()
        image = renderer.render_terrain_pbr_pom(
            materials, environment, f3d.TerrainRenderParams(params), height,
            water_mask=water, time_seconds=seconds, cache=cache,
        )
        pixels = image.to_numpy().tobytes()
        errors = [entry for entry in f3d.native_degradations()
                  if entry["kind"] in {"uncaptured_gpu_error", "validation_error"}]
        assert not errors, f"native render reported GPU errors: {errors}"
        return pixels, dict(renderer.last_anamnesis_cache_report)

    baseline, baseline_report = render(config, heights, mask, ibl, material, 0.0, root / "baseline")
    baseline_key = baseline_report["pass_keys"]["terrain.resolve"]
    forward_key = baseline_report["pass_keys"]["terrain.forward"]
    from anamnesis_packed_store import metadata as packed_metadata
    metadata = packed_metadata(root / "baseline", forward_key)
    declaration = metadata["derivation"]["material"]
    params_mapping = asdict(config)
    params_mapping["height_curve_lut"] = config.height_curve_lut.tolist()
    assert canonical_json_bytes(params_mapping, error_context="native mutation configuration") in bytes.fromhex(declaration["uniform_hex"])
    assert bytes.fromhex(declaration["capability_fingerprint_hex"]) == capability_fingerprint(dict(f3d.capabilities()))
    categories, rejected, bypasses, inventory = _parameter_mutations(config, root, rng, max_dimension)
    coverage = dict.fromkeys((name for name, _, _, _ in categories), 0)
    alternatives = {name: dict.fromkeys(map(repr, candidates), 0) for name, _, candidates, discrete in categories if discrete}
    # First cover every field, then every remaining discrete alternative.
    # This keeps the full finite corpus independent of per-field domain size.
    required_cases = [(index, 0) for index in range(len(categories))]
    required_cases.extend((index, alternative) for index, (_, _, values, discrete) in enumerate(categories)
                          if discrete for alternative in range(1, len(values)))
    if count == 10_000:
        assert len(required_cases) <= count, "required alternatives exceed the owner-defined corpus"
    changed_pixels = 0
    trace = root / "mutation-cases.jsonl"
    for index in range(count):
        category, alternative = (required_cases[index] if index < len(required_cases) else
                                 (index % len(categories), index // len(categories)))
        name, path, candidates, discrete = categories[category]
        candidate = candidates[alternative % len(candidates)]
        params, height, water, environment, materials, seconds = config, heights, mask, ibl, material, 0.0
        changed_file = None
        detail = {}
        if name.startswith("array."):
            array = np.asarray(_at_path(config, path)).copy()
            offset = rng.randrange(array.size) * array.itemsize + rng.randrange(2)
            byte_mask = rng.randrange(1, 256)
            array.view(np.uint8).reshape(-1)[offset] ^= byte_mask
            detail = {"byte_offset": offset, "xor": byte_mask, "dtype": array.dtype.str, "shape": list(array.shape)}
            params = _replaced(config, path, array)
        elif path is not None:
            params = _changed_config(config, path, candidate)
        elif name.startswith("file."):
            field = name.removeprefix("file.")
            changed_file = Path(_at_path(config, tuple(field.split("."))))
            pixels = file_pixels[field].copy()
            offset, byte_mask = rng.randrange(pixels.size), rng.randrange(1, 256)
            pixels.reshape(-1)[offset] ^= byte_mask
            detail = {"texel_byte_offset": offset, "xor": byte_mask}
            Image.fromarray(pixels).save(changed_file)
        elif name in {"heightmap", "water_mask"}:
            array = (heights if name == "heightmap" else mask).copy()
            byte_index = rng.randrange(array.size) * array.itemsize + rng.randrange(2)
            byte_mask = rng.randrange(1, 256)
            array.view(np.uint8).reshape(-1)[byte_index] ^= byte_mask
            detail = {"byte_offset": byte_index, "xor": byte_mask}
            if name == "heightmap":
                height = array
            else:
                water = array
        elif name == "environment":
            changed_hdr = bytearray(hdr_bytes)
            byte_index = hdr_pixels + rng.randrange((len(hdr_bytes) - hdr_pixels) // 4) * 4 + rng.randrange(3)
            byte_mask = rng.randrange(1, 256)
            changed_hdr[byte_index] ^= byte_mask
            detail = {"byte_offset": byte_index, "xor": byte_mask}
            changed_path = root / "mutated-environment.hdr"
            changed_path.write_bytes(changed_hdr)
            environment = f3d.IBL.from_hdr(str(changed_path), intensity=1.0)
        elif name == "material":
            strength = rng.choice([value for value in _float_mutations(1.0) if value >= 0.0])
            materials = f3d.MaterialSet.terrain_default(normal_strength=strength)
            detail = {"normal_strength": strength}
        else:
            seconds = rng.choice(list(_float_mutations(0.0)))
            detail = {"time_seconds": seconds}
        try:
            with tempfile.TemporaryDirectory(dir=root, prefix="mutation-") as store:
                shutil.copytree(root / "baseline", store, dirs_exist_ok=True)
                cached, first = render(params, height, water, environment, materials, seconds, store)
                assert first["pass_keys"]["terrain.resolve"] != baseline_key, f"production input escaped key: {index}:{name}"
                restored, warm = render(params, height, water, environment, materials, seconds, store)
                fresh, cold = render(params, height, water, environment, materials, seconds, None)
                assert cached == restored == fresh, f"native stale hit: {index}:{name}"
                assert warm["pass_keys"] == first["pass_keys"]
                assert warm["hits"] == baseline_report["misses"]
                assert not warm["misses"] and warm["graph_command_submissions"] == 0
                assert warm["bypass_reason"] is None
                assert not cold["hits"] and cold["graph_command_submissions"] > 0
                changed_pixels += cached != baseline
        except Exception as error:
            raise AssertionError(f"native mutation {index}:{name}={candidate!r}: {error}") from error
        finally:
            if changed_file is not None:
                changed_file.write_bytes(file_bytes[name.removeprefix("file.")])
        coverage[name] += 1
        if discrete:
            alternatives[name][repr(candidate)] += 1
        with trace.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"case": index, "field": name, "value": candidate, "mutation": detail}) + "\n")
        if (index + 1) % len(categories) == 0:
            print(f"ANAMNESIS native mutations {index + 1}/{count}", flush=True)
    assert all(coverage.values()), f"unexercised production fields: {coverage}"
    if count == 10_000:
        assert all(all(values.values()) for values in alternatives.values()), f"unexercised discrete alternatives: {alternatives}"

    bypass_controls = []
    for name, value, changed, reason in bypasses:
        if reason == "material_virtual_textures":
            renderer = f3d.TerrainRenderer(f3d.Session(window=False))
            pixels = np.full((248, 248, 4), (128, 128, 255, 255), dtype=np.uint8)
            for material_index in range(material.material_count):
                renderer.register_material_vt_source(material_index, "albedo", pixels, (248, 248), None)
            changed = replace(changed, vt=tp.TerrainVTSettings(enabled=True, atlas_size=256,
                layers=[tp.VTLayerFamily(family="albedo", virtual_size_px=(248, 248))]))
        fresh, _ = render(changed, heights, mask, ibl, material, 0.0, None)
        result, report = render(changed, heights, mask, ibl, material, 0.0, root / "bypassed")
        assert result == fresh and report["bypass_reason"] == reason
        assert not report["hits"] and not report["pass_keys"]
        assert not (root / "bypassed").exists()
        bypass_controls.append({"field": name, "value": value, "reason": report["bypass_reason"]})

    bypass_controls.extend(_optional_bypass_controls(config, root, heights, mask, ibl, material))
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    visible, _ = render(replace(config, cam_phi_deg=55.0), heights, mask, ibl, material, 0.0, None)
    assert visible != baseline, "mesh camera visible-change control did not affect pixels"
    # A sun edit after an all-hit frame must execute a fresh shadow before
    # forward shading; equality of old restored color alone cannot prove this.
    cached_baseline, cached_report = render(config, heights, mask, ibl, material, 0.0, root / "baseline")
    assert cached_baseline == baseline and cached_report["graph_command_submissions"] == 0
    sun = replace(config, light=replace(config.light, azimuth_deg=config.light.azimuth_deg + 90.0, elevation_deg=25.0))
    sun_cached, sun_report = render(sun, heights, mask, ibl, material, 0.0, root / "baseline")
    sun_fresh, _ = render(sun, heights, mask, ibl, material, 0.0, None)
    assert sun_cached == sun_fresh and sun_cached != baseline
    assert "terrain.shadow" in sun_report["misses"] and "terrain.forward" in sun_report["misses"]
    sun_changed_pixels = int(np.count_nonzero(np.any(np.frombuffer(sun_cached, dtype=np.uint8).reshape(64, 64, 4)[..., :3] != np.frombuffer(baseline, dtype=np.uint8).reshape(64, 64, 4)[..., :3], axis=2)))
    assert sun_changed_pixels > 0
    _, swapped = render(config, mask, heights, ibl, material, 0.0, root / "swapped")
    assert swapped["pass_keys"]["terrain.resolve"] != baseline_key
    return {"seed": 0xA11A_17, "mutations": count, "coverage": coverage, "discrete_alternatives": alternatives,
            "constructor_rejections": rejected, "bypass_controls": bypass_controls, "parameter_inventory": inventory,
            "changed_pixels": changed_pixels, "changed_output_cases": changed_pixels,
            "sun_changed_pixels": sun_changed_pixels, "adapter": adapter}
