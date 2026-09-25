"""CHRONOS deterministic terrain flythrough rendering.

Every frame is compiled once by the native ``compile_frame`` into a canonical,
hash-validated payload (seed, frozen labels/alpha, clipmap ring morphs,
required VT residency). Rendering is a pure reader of that payload, and
``FlythroughManifest.replay_frame`` reproduces any frame's raw RGBA SHA-256
from the stored compiled data alone.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._canonical_json import canonical_json_bytes
from .map_scene import (
    MapScene,
    ReproducibilityProfile,
    _mapscene_clipmap_config,
    _mapscene_effective_camera_mode,
    _mapscene_vt_config,
    _mapscene_vt_settings,
    _shared_terrain_render_context,
    _terrain_scene_diagonal,
)

_CHRONOS_MANIFEST_SCHEMA = "forge3d.chronos.flythrough/1"
_VT_FAMILY_SLOTS = {"albedo": 0, "normal": 1, "mask": 2}


def _native():
    import forge3d as f3d

    required = ("frame_seed", "compile_frame", "render_compiled_frame", "CompiledFrame")
    missing = [name for name in required if not hasattr(f3d, name)]
    if missing:
        raise RuntimeError(
            "CHRONOS requires the native forge3d extension with: " + ", ".join(missing)
        )
    return f3d


def _canonical_json_text(value: Any, error_context: str) -> str:
    return canonical_json_bytes(value, error_context=error_context).decode("utf-8")


def _prepare_frame_scene(
    source: MapScene,
    frame_index: int,
    base_seed: int,
    samples: int,
    out_path: Path,
) -> MapScene:
    recipe = source.recipe
    output = recipe.output
    if output is None:
        raise ValueError("CHRONOS flythrough requires each MapScene recipe to carry an OutputSpec")
    if str(output.format).lower() != "png":
        raise ValueError(
            f"CHRONOS flythrough requires PNG output (raw uint8 RGBA hashing); "
            f"got format={output.format!r}"
        )
    if int(output.bit_depth) != 8:
        raise ValueError(
            f"CHRONOS flythrough requires 8-bit output (raw uint8 RGBA hashing); "
            f"got bit_depth={output.bit_depth}"
        )
    if bool(output.hdr):
        raise ValueError("CHRONOS flythrough requires hdr=False (raw uint8 RGBA hashing)")
    if tuple(output.aovs or ()):
        raise ValueError(
            f"CHRONOS flythrough requires an empty AOV list (raw uint8 RGBA hashing); "
            f"got aovs={tuple(output.aovs)!r}"
        )
    derived_seed = int(_native().frame_seed(int(base_seed), int(frame_index)))
    output = replace(output, path=str(out_path), samples=int(samples))
    profile = recipe.reproducibility_profile
    profile = (
        replace(profile, seed=derived_seed)
        if profile is not None
        else ReproducibilityProfile(seed=derived_seed)
    )
    frame_scene = MapScene(
        replace(recipe, output=output, reproducibility_profile=profile)
    )
    frame_scene.compile_plan()
    return frame_scene


def _accepted_labels(frame_scene: MapScene) -> dict[tuple[str, str], Any]:
    accepted: dict[tuple[str, str], Any] = {}
    for layer_id, plan in frame_scene.compiled_label_plans.items():
        for label in getattr(plan, "accepted", ()) or ():
            accepted[(str(layer_id), str(label.label_id))] = label
    return accepted


def _label_timeline_events(
    per_frame_keys: Mapping[int, set],
    sorted_indices: Sequence[int],
) -> dict[tuple[str, str], dict[int, tuple[Any, Any]]]:
    all_keys: set[tuple[str, str]] = set()
    for keys in per_frame_keys.values():
        all_keys.update(keys)
    events: dict[tuple[str, str], dict[int, tuple[Any, Any]]] = {}
    for key in all_keys:
        visible = [key in per_frame_keys[index] for index in sorted_indices]
        by_frame: dict[int, tuple[Any, Any]] = {}
        i = 0
        while i < len(sorted_indices):
            if not visible[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(sorted_indices) and visible[j + 1]:
                j += 1
            appearance = sorted_indices[i] if i > 0 else None
            disappearance = sorted_indices[j + 1] if j + 1 < len(sorted_indices) else None
            for position in range(i, j + 1):
                by_frame[sorted_indices[position]] = (appearance, disappearance)
            i = j + 1
        events[key] = by_frame
    return events


def _label_records(
    keys: Sequence[tuple[str, str]],
    accepted: Mapping[tuple[str, str], Any],
    events_for: Any,
) -> list[dict[str, Any]]:
    records = []
    for layer_id, label_id in sorted(keys):
        label = accepted.get((layer_id, label_id))
        visible = label is not None
        appearance_frame, disappearance_frame = events_for(layer_id, label_id, visible)
        records.append(
            {
                "layer_id": layer_id,
                "label_id": label_id,
                "visible": visible,
                "appearance_frame": appearance_frame,
                "disappearance_frame": disappearance_frame,
                "payload": label.to_dict() if visible else {},
            }
        )
    return records


def _clipmap_descriptor(recipe: Any) -> dict[str, Any] | None:
    config = _mapscene_clipmap_config(recipe)
    if config is None:
        return None
    return {
        "ring_count": min(max(int(config.get("levels", config.get("ring_count", 4))), 1), 8),
        "morph_range": min(max(float(config.get("morph_range", 0.3)), 0.0), 1.0),
        "terrain_span": float(max(1.0, _terrain_scene_diagonal(recipe.terrain))),
        "camera_distance": float(recipe.camera.distance),
    }


def _vt_descriptor(recipe: Any, render_size: tuple[int, int]) -> dict[str, Any] | None:
    config = _mapscene_vt_config(recipe)
    if not isinstance(config, Mapping) or not bool(config.get("enabled", True)):
        return None
    vt = _mapscene_vt_settings(recipe)
    if vt is None or not getattr(vt, "layers", None):
        return None
    layer = vt.layers[0]
    sources = config.get("sources")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        if bool(config.get("procedural_sources", False)):
            sources = [
                {"material_index": index, "family": "albedo"}
                for index in range(int(config.get("source_count", 4)))
            ]
        else:
            sources = ()
    descriptor_sources = []
    for source in sources or ():
        if not isinstance(source, Mapping):
            continue
        family = str(source.get("family", "albedo"))
        if family not in _VT_FAMILY_SLOTS:
            raise ValueError(f"CHRONOS VT descriptor: unsupported family '{family}'")
        descriptor_sources.append(
            {
                "family_slot": _VT_FAMILY_SLOTS[family],
                "material_index": int(source.get("material_index", 0)),
            }
        )
    target = recipe.camera.target or (0.0, 0.0, 0.0)
    target_xyz = [float(target[i]) if i < len(target) else 0.0 for i in range(3)]
    return {
        "virtual_size_px": [int(layer.virtual_size_px[0]), int(layer.virtual_size_px[1])],
        "tile_size": int(layer.tile_size),
        "max_mip_levels": int(vt.max_mip_levels),
        "sources": descriptor_sources,
        "render_size": [int(render_size[0]), int(render_size[1])],
        "terrain_span": float(max(1.0, _terrain_scene_diagonal(recipe.terrain))),
        "camera_mode": _mapscene_effective_camera_mode(recipe),
        "camera_target": target_xyz,
        "camera_distance": float(recipe.camera.distance),
        "fov_deg": float(recipe.camera.fov_deg),
    }


def _render_size(recipe: Any) -> tuple[int, int]:
    output = recipe.output
    return (max(64, int(output.width)), max(64, int(output.height)))


def _scene_json(frame_scene: MapScene, label_records: list[dict[str, Any]]) -> str:
    recipe_dict = frame_scene.recipe.to_dict()
    output_dict = recipe_dict.get("output")
    if isinstance(output_dict, Mapping):
        recipe_dict["output"] = {
            key: value for key, value in output_dict.items() if key != "path"
        }
    scene_value = {
        "labels": label_records,
        "clipmap": _clipmap_descriptor(frame_scene.recipe),
        "virtual_texture": _vt_descriptor(frame_scene.recipe, _render_size(frame_scene.recipe)),
        "scene": recipe_dict,
    }
    return _canonical_json_text(scene_value, "chronos scene")


def _camera_json(frame_scene: MapScene) -> str:
    return _canonical_json_text(frame_scene.recipe.camera.to_dict(), "chronos camera")


def _frame_png_name(frame_index: int) -> str:
    return f"frame_{int(frame_index):08d}.png"


def _frame_provenance_name(frame_index: int) -> str:
    return f"frame_{int(frame_index):08d}.provenance.json"


def _decode_png_rgba(path: Path):
    import numpy as np

    from ._png import load_png_rgba

    rgba = np.asarray(load_png_rgba(path), dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise RuntimeError(f"decoded PNG is not HxWx4 RGBA: {rgba.shape}")
    return np.ascontiguousarray(rgba)


_FRAME_RECORD_FIELDS = (
    "frame_index",
    "image",
    "provenance",
    "pixel_hash",
    "width",
    "height",
    "compiled_frame",
    "base_seed",
    "frame_seed",
    "samples",
    "residency_hash",
    "label_set_hash",
    "lod_hash",
    "engine_revision",
    "camera_hash",
    "scene_hash",
)


@dataclass(frozen=True)
class FlythroughManifest:
    """Deterministic record of a CHRONOS flythrough render."""

    base_seed: int
    samples: int
    frames: tuple[Mapping[str, Any], ...] = ()
    schema: str = _CHRONOS_MANIFEST_SCHEMA
    _compiled_frames: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema != _CHRONOS_MANIFEST_SCHEMA:
            raise ValueError(
                f"unsupported flythrough manifest schema {self.schema!r}"
            )
        if int(self.samples) <= 0:
            raise ValueError("flythrough manifest samples must be positive")
        frames = tuple(dict(record) for record in self.frames)
        indices = []
        for record in frames:
            missing = [name for name in _FRAME_RECORD_FIELDS if name not in record]
            if missing:
                raise ValueError(
                    f"flythrough frame record is missing fields: {', '.join(missing)}"
                )
            index = int(record["frame_index"])
            indices.append(index)
            pixel_hash = record["pixel_hash"]
            if (
                not isinstance(pixel_hash, str)
                or len(pixel_hash) != 64
                or any(ch not in "0123456789abcdef" for ch in pixel_hash)
            ):
                raise ValueError(
                    f"flythrough frame {index} pixel_hash must be 64 lowercase hex chars"
                )
        if indices != sorted(indices) or len(set(indices)) != len(indices):
            raise ValueError("flythrough frame indices must be sorted and unique")
        if any(index < 0 for index in indices):
            raise ValueError("flythrough frame indices must be non-negative")
        object.__setattr__(self, "frames", frames)
        if self._compiled_frames:
            compiled_frames = tuple(self._compiled_frames)
            if len(compiled_frames) != len(frames):
                raise ValueError(
                    "flythrough manifest _compiled_frames count must match frames"
                )
        else:
            f3d = _native()
            compiled_frames = tuple(
                f3d.CompiledFrame.from_json(str(record["compiled_frame"]))
                for record in frames
            )
        for record, compiled in zip(frames, compiled_frames):
            index = int(record["frame_index"])
            if compiled.to_json() != str(record["compiled_frame"]):
                raise ValueError(
                    f"flythrough frame {index}: compiled payload mismatch"
                )
            payload = json.loads(compiled.to_json())
            if int(compiled.frame_index) != index:
                raise ValueError(
                    f"flythrough frame {index}: compiled frame_index mismatch"
                )
            if int(compiled.base_seed) != int(self.base_seed):
                raise ValueError(
                    f"flythrough frame {index}: compiled base_seed disagrees with manifest"
                )
            if int(compiled.samples) != int(self.samples):
                raise ValueError(
                    f"flythrough frame {index}: compiled samples disagrees with manifest"
                )
            if int(record["base_seed"]) != int(self.base_seed):
                raise ValueError(
                    f"flythrough frame {index}: record base_seed disagrees with manifest"
                )
            if int(record["samples"]) != int(self.samples):
                raise ValueError(
                    f"flythrough frame {index}: record samples disagrees with manifest"
                )
            if int(record["frame_seed"]) != int(compiled.frame_seed):
                raise ValueError(
                    f"flythrough frame {index}: record frame_seed disagrees with payload"
                )
            for name in (
                "residency_hash",
                "label_set_hash",
                "lod_hash",
                "engine_revision",
                "camera_hash",
                "scene_hash",
            ):
                if record[name] != payload[name]:
                    raise ValueError(
                        f"flythrough frame {index}: record {name} disagrees with payload"
                    )
        object.__setattr__(self, "_compiled_frames", compiled_frames)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "base_seed": int(self.base_seed),
            "samples": int(self.samples),
            "frames": [dict(record) for record in self.frames],
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "FlythroughManifest":
        schema = data.get("schema")
        if schema != _CHRONOS_MANIFEST_SCHEMA:
            raise ValueError(f"unsupported flythrough manifest schema {schema!r}")
        frames = data.get("frames")
        if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
            raise ValueError("flythrough manifest 'frames' must be a sequence")
        return FlythroughManifest(
            base_seed=int(data["base_seed"]),
            samples=int(data["samples"]),
            frames=tuple(dict(record) for record in frames),
            schema=str(schema),
        )

    @staticmethod
    def load(path: str | Path) -> "FlythroughManifest":
        return FlythroughManifest.from_dict(json.loads(Path(path).read_text("utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(
            canonical_json_bytes(self.to_dict(), error_context="chronos manifest")
        )

    def frame_record(self, frame_index: int) -> Mapping[str, Any]:
        for record in self.frames:
            if int(record["frame_index"]) == int(frame_index):
                return record
        raise KeyError(f"no CHRONOS compiled frame for frame_index {frame_index}")

    def replay_frame(
        self, scene: MapScene, frame_index: int, out_path: str | Path
    ) -> Mapping[str, Any]:
        """Re-render one frozen frame and verify the stored raw RGBA hash."""
        f3d = _native()
        record = self.frame_record(frame_index)
        canonical = str(record["compiled_frame"])
        compiled = f3d.CompiledFrame.from_json(canonical)
        stored_labels = (
            (json.loads(canonical).get("scene") or {}).get("labels") or []
        )
        stored_events = {
            (str(item["layer_id"]), str(item["label_id"])): (
                item.get("appearance_frame"),
                item.get("disappearance_frame"),
            )
            for item in stored_labels
            if isinstance(item, Mapping)
        }
        out_path = Path(out_path)
        frame_scene = _prepare_frame_scene(
            scene, int(frame_index), self.base_seed, self.samples, out_path
        )
        accepted = _accepted_labels(frame_scene)

        def events_for(layer_id: str, label_id: str, _visible: bool):
            return stored_events.get((layer_id, label_id), (None, None))

        label_records = _label_records(sorted(stored_events), accepted, events_for)
        camera_json = _camera_json(frame_scene)
        scene_json = _scene_json(frame_scene, label_records)
        candidate = f3d.compile_frame(
            int(frame_index), self.base_seed, self.samples, camera_json, scene_json
        )
        if candidate.to_json() != canonical:
            raise ValueError(
                "CHRONOS replay inputs do not match the stored compiled frame "
                f"for frame_index {frame_index}"
            )
        plan = frame_scene.compiled_plan
        frame_scene.compiled_plan = replace(plan, frame=compiled)
        frame_scene.render(str(out_path))
        rgba = _decode_png_rgba(out_path)
        provenance = json.loads(f3d.render_compiled_frame(compiled, rgba))
        if provenance.get("pixel_hash") != record["pixel_hash"]:
            raise RuntimeError(
                "CHRONOS replay pixel hash mismatch for frame_index "
                f"{frame_index}: {provenance.get('pixel_hash')} != {record['pixel_hash']}"
            )
        return provenance


def render_flythrough(
    camera_path: Sequence[MapScene] | Mapping[int, MapScene],
    *,
    base_seed: int,
    samples: int,
    out_dir: str | Path,
    certificate: bool | str | os.PathLike[str] = False,
    cache: str | os.PathLike[str] | None = None,
) -> FlythroughManifest:
    """Render a deterministic CHRONOS flythrough to ``out_dir``.

    ``camera_path`` is either a sequence of ``MapScene`` (frame indices are the
    enumerate order) or a mapping of explicit integer frame indices to
    ``MapScene``. Each frame writes ``frame_<index:08d>.png`` plus a canonical
    ``frame_<index:08d>.provenance.json`` sidecar, and the run writes a
    deterministic ``flythrough_manifest.json``.
    When ``certificate`` is requested, each frame also writes its own
    ``frame_<index:08d>.certificate.json`` sidecar in ``out_dir``. ``cache``
    uses a ``frame_<index:08d>`` subfolder for each frame's ANAMNESIS store;
    certificates disable cache eligibility.

    The whole run shares one native ``Session``/``TerrainRenderer``/IBL set;
    determinism lives in each frame's compiled inputs (seed, camera, scene
    JSON), not in the renderer instance, so reuse preserves the pixel-hash
    contract while avoiding per-frame pipeline construction.
    """
    f3d = _native()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(camera_path, Mapping):
        items = sorted((int(index), scene) for index, scene in camera_path.items())
    else:
        items = list(enumerate(camera_path))
    if not items:
        raise ValueError("CHRONOS flythrough requires at least one camera frame")
    for index, scene in items:
        if not isinstance(scene, MapScene):
            raise TypeError(
                f"CHRONOS camera_path entry {index!r} is not a MapScene"
            )

    sorted_indices = [index for index, _ in items]

    frame_scenes: dict[int, MapScene] = {}
    accepted_maps: dict[int, dict[tuple[str, str], Any]] = {}
    per_frame_keys: dict[int, set] = {}
    for index, source in items:
        frame_scene = _prepare_frame_scene(
            source, index, base_seed, samples, out_dir / _frame_png_name(index)
        )
        accepted = _accepted_labels(frame_scene)
        frame_scenes[index] = frame_scene
        accepted_maps[index] = accepted
        per_frame_keys[index] = set(accepted.keys())

    events = _label_timeline_events(per_frame_keys, sorted_indices)
    all_keys = sorted({key for keys in per_frame_keys.values() for key in keys})

    records: list[dict[str, Any]] = []
    compiled_frames: list[Any] = []
    with _shared_terrain_render_context():
        for index in sorted_indices:
            frame_scene = frame_scenes[index]

            def events_for(layer_id: str, label_id: str, visible: bool, _i: int = index):
                if not visible:
                    return (None, None)
                return events.get((layer_id, label_id), {}).get(_i, (None, None))

            label_records = _label_records(all_keys, accepted_maps[index], events_for)
            compiled = f3d.compile_frame(
                index,
                int(base_seed),
                int(samples),
                _camera_json(frame_scene),
                _scene_json(frame_scene, label_records),
            )
            plan = frame_scene.compiled_plan
            frame_scene.compiled_plan = replace(plan, frame=compiled)
            png_path = out_dir / _frame_png_name(index)
            certificate_path = (
                out_dir / f"frame_{index:08d}.certificate.json"
                if certificate
                else False
            )
            frame_cache = Path(cache) / f"frame_{index:08d}" if cache is not None else None
            frame_scene.render(
                str(png_path), certificate=certificate_path, cache=frame_cache
            )
            rgba = _decode_png_rgba(png_path)
            provenance_json = f3d.render_compiled_frame(compiled, rgba)
            provenance = json.loads(provenance_json)
            provenance_path = out_dir / _frame_provenance_name(index)
            provenance_path.write_bytes(provenance_json.encode("utf-8"))
            record = {
                "frame_index": index,
                "image": _frame_png_name(index),
                "provenance": _frame_provenance_name(index),
                "pixel_hash": provenance["pixel_hash"],
                "width": int(rgba.shape[1]),
                "height": int(rgba.shape[0]),
                "compiled_frame": compiled.to_json(),
            }
            for field_name in (
                "base_seed",
                "frame_seed",
                "samples",
                "residency_hash",
                "label_set_hash",
                "lod_hash",
                "engine_revision",
                "camera_hash",
                "scene_hash",
            ):
                record[field_name] = provenance[field_name]
            records.append(record)
            compiled_frames.append(compiled)

    manifest = FlythroughManifest(
        base_seed=int(base_seed),
        samples=int(samples),
        frames=tuple(records),
        _compiled_frames=tuple(compiled_frames),
    )
    manifest.save(out_dir / "flythrough_manifest.json")
    return manifest


__all__ = [
    "FlythroughManifest",
    "render_flythrough",
]
