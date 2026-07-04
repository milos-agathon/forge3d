"""Deterministic manifest helpers for MapScene recipes."""

from __future__ import annotations

from typing import Any, Mapping

from ._map_scene_common import _layer_id, _stable_hash


def _recipe_payload(value: Any) -> Mapping[str, Any]:
    if hasattr(value, "recipe"):
        value = value.recipe
    if hasattr(value, "to_dict") and callable(value.to_dict):
        payload = value.to_dict()
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise TypeError("recipe_manifest expects a MapScene, SceneRecipe, or recipe mapping")
    recipe = payload.get("recipe") if isinstance(payload.get("recipe"), Mapping) else payload
    if not isinstance(recipe, Mapping):
        raise TypeError("recipe_manifest could not find a recipe mapping")
    return recipe


def _golden_fixture_payload(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    payload = dict(value)
    scene_id = str(payload.get("scene_id") or payload.get("id") or "")
    family = str(payload.get("family") or "")
    golden_path = str(payload.get("golden_path") or "")
    command = str(payload.get("command") or "")
    backend = str(payload.get("backend") or "mapscene")
    status = str(payload.get("status") or "active")
    tolerance = dict(payload.get("tolerance") or {})
    if "ssim_min" not in tolerance:
        tolerance["ssim_min"] = 0.995
    if "mean_abs_max" not in tolerance:
        tolerance["mean_abs_max"] = 2.0
    return {
        "schema": "forge3d.mapscene.golden_fixture_intent.v1",
        "scene_id": scene_id,
        "family": family,
        "status": status,
        "golden_path": golden_path,
        "command": command,
        "backend": backend,
        "tolerance": {
            "ssim_min": float(tolerance["ssim_min"]),
            "mean_abs_max": float(tolerance["mean_abs_max"]),
        },
    }


def recipe_manifest(value: Any, *, golden_fixture_intent: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a deterministic, JSON-safe summary for a MapScene recipe."""

    recipe = dict(_recipe_payload(value))
    terrain = dict(recipe.get("terrain") or {})
    output = dict(recipe.get("output") or {})
    layers = []
    for index, layer in enumerate(recipe.get("layers") or ()):
        if not isinstance(layer, Mapping):
            continue
        layer_id = str(layer.get("layer_id") or _layer_id(layer, f"layer_{index}"))
        layers.append(
            {
                "layer_id": layer_id,
                "kind": str(layer.get("kind") or type(layer).__name__),
                "crs": layer.get("crs"),
                "path": layer.get("path") or layer.get("source"),
                "source_id": (layer.get("metadata") or {}).get("source_id") if isinstance(layer.get("metadata"), Mapping) else None,
                "hash": _stable_hash(layer),
            }
        )
    manifest = {
        "kind": "mapscene_recipe_manifest",
        "schema": "forge3d.mapscene.recipe_manifest.v1",
        "recipe_hash": _stable_hash(recipe),
        "terrain": {
            "path": terrain.get("path"),
            "crs": terrain.get("crs"),
            "source_id": (terrain.get("metadata") or {}).get("source_id") if isinstance(terrain.get("metadata"), Mapping) else None,
            "hash": _stable_hash(terrain),
        },
        "output": {
            "width": int(output.get("width", 0) or 0),
            "height": int(output.get("height", 0) or 0),
            "format": str(output.get("format", "png")),
            "samples": int(output.get("samples", 1) or 1),
            "aovs": list(output.get("aovs") or ()),
            "hdr": bool(output.get("hdr", False)),
        },
        "layers": layers,
    }
    golden_payload = _golden_fixture_payload(golden_fixture_intent)
    if golden_payload is not None:
        manifest["golden_fixture_intent"] = golden_payload
    return manifest


__all__ = ["recipe_manifest"]
