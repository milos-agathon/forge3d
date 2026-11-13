"""C06 composition of actual C07/CHRONOS frames; no observation-time rules."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .chronos import render_flythrough
from .publication import _distinct_ids, _identity, publication_output_support


def _render_publication_flyover(recipe, frames, directory, *, layout, base_seed, cache):
    frames = tuple(frames)
    _distinct_ids([f.page_id for f in frames])
    if recipe.camera_mode != "flat-perspective":
        raise ValueError("Publication flyovers currently require flat-perspective mode")
    if any(f.bounds != frames[0].bounds for f in frames):
        raise ValueError("Publication flyover frames must share one registered terrain extent")
    output = Path(directory)
    # Keep frame manifests unambiguous and never leave old movie frames behind.
    if output.exists() and any(output.iterdir()):
        raise ValueError("Publication flyovers require a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    current = replace(recipe, config=layout.config, layout=None)
    values, valid, grid, resolution, description = current._read()
    classes, style, _, _ = current._classify(values, valid)
    window = current._window(frames[0].bounds, grid, resolution, values.shape)
    top, right, bottom, left = layout.config.margin
    size = current._size(frames[0].bounds, (layout.config.width-right-left, layout.config.height-top-bottom))
    scenes = [current._page_scene(classes[window], style, frame, size, description)[0] for frame in frames]
    scene_ids = [_identity(scene.to_dict()) for scene in scenes]
    # Stage the entire delivery; no complete publication manifest is installed
    # when a native frame or composition fails.
    with tempfile.TemporaryDirectory(prefix=".flyover-", dir=output.parent) as staging:
        stage = Path(staging)
        native_dir = stage / "native"
        manifest = render_flythrough(scenes, base_seed=base_seed, samples=1,
                                    out_dir=native_dir, cache=cache)
        manifest_sha = hashlib.sha256((native_dir / "flythrough_manifest.json").read_bytes()).hexdigest()
        providers = {}
        for frame, scene_id, record in zip(frames, scene_ids, manifest.frames, strict=True):
            providers[frame.page_id] = {
                "scene_id": scene_id, "path": native_dir / record["image"], "pixel_hash": record["pixel_hash"],
                "binding": {"frame_index": record["frame_index"], "camera_hash": record["camera_hash"],
                            "scene_hash": record["scene_hash"], "native_pixel_hash": record["pixel_hash"],
                            "manifest_sha256": manifest_sha, "manifest": "native/flythrough_manifest.json"},
            }
        pages = current._render_layout_series(frames, stage / "frames", cache=cache,
                    insets=tuple(layout.insets), selection=None, flyover_frames=providers)
        for page in pages:
            name = page["page_id"]
            page["output"]["path"] = str(output / "frames" / f"{name}.png")
            (stage / "frames" / f"{name}.json").write_text(json.dumps(page, indent=2, allow_nan=False), encoding="utf8")
        bookends = []
        (stage / "bookends").mkdir()
        for name, endpoint in (("title", pages[0]), ("end", pages[-1])):
            # Exact full-frame copies: matching furniture and native map pixels
            # are inherited from the first/last executed flythrough frame.
            shutil.copyfile(stage / "frames" / f"{endpoint['page_id']}.png", stage / "bookends" / f"{name}.png")
            receipt = {**endpoint, "page_id": name, "source_page_id": endpoint["page_id"],
                       "output": {**endpoint["output"], "path": str(output / "bookends" / f"{name}.png")}}
            (stage / "bookends" / f"{name}.json").write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf8")
            bookends.append(receipt)
        result = {"source": description, "chronos_manifest_sha256": manifest_sha,
                  "chronos_manifest": "native/flythrough_manifest.json", "frames": pages, "bookends": bookends,
                  "output_support": publication_output_support()["publication_png"],
                  "sequence": "Explicit camera frames; no camera or observation interpolation in publication",
                  "encoding": "PNG sequence; movie encoding is outside this route"}
        (stage / "publication-flyover.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
        # Destination was required to be empty. Keep the completion receipt last.
        for name in ("native", "frames", "bookends", "publication-flyover.json"):
            (stage / name).replace(output / name)
    return result
