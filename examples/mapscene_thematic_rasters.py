"""Local D02 fixtures through public styles and MapScene; no data fetching.

Run: python examples/mapscene_thematic_rasters.py --output artifacts/thematic
Requires a terrain-capable physical GPU and installed native bindings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import forge3d as f3d


PALETTE = (
    ((255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)),
    ((255, 255, 0, 255), (0, 255, 255, 255), (255, 0, 255, 255)),
    ((128, 0, 0, 255), (0, 128, 0, 255), (0, 0, 128, 255)),
)


def build_scenes():
    # Repetition enlarges fixture cells without inventing new class values.
    population = np.repeat(np.repeat(np.array([[0, 10, 20], [20, 10, 0], [0, 10, 20]], np.float32), 32, 0), 32, 1)
    x = np.repeat(np.repeat(np.array([[0, 1, 2]] * 3, np.float32), 32, 0), 32, 1)
    y = np.repeat(np.repeat(np.array([[0] * 3, [10] * 3, [20] * 3], np.float32), 32, 0), 32, 1)
    styles = {
        "population": f3d.RasterHeightSurfaceStyle(0.5, "people/km²"),
        "categorical": f3d.CategoricalRasterStyle({i: PALETTE[0][i] for i in range(3)}, {0: "bare", 1: "forest", 2: "water"}),
        "bivariate": f3d.BivariateRasterStyle([1, 2], [10, 20], PALETTE, "Population", "Temperature", x_units="people/km²", y_units="°C"),
    }
    scenes = {}
    for name, style in styles.items():
        # Height output is metres per source unit. The other fixtures use a
        # flat zero-height terrain to make the palette cells easy to inspect.
        data = population if name == "population" else np.zeros_like(x)
        pop_style = style if name == "population" else f3d.RasterHeightSurfaceStyle(1, "people/km²")
        layers = () if name == "population" else (f3d.RasterOverlay(
            layer_id=name, data=x, style=style, secondary_data=y if name == "bivariate" else None,
            crs="EPSG:3857", metadata={"source_id": name}),)
        scenes[name] = f3d.MapScene(
            terrain=f3d.TerrainSource(data=data, style=pop_style, crs="EPSG:3857",
                metadata={"source_id": name, "width": 96, "height": 96, "resolution": [1, 1]}, elevation_sampling_available=True),
            layers=layers, lighting=f3d.LightingPreset(name="daylight", sun_direction=(0.8, 0.6, 0.0), settings={"hue_variation_strength": 0}),
            output=f3d.OutputSpec(width=192, height=192),
        )
    return scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/thematic"))
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, scene in build_scenes().items():
        png = output / f"{name}.png"
        report = scene.render(str(png))
        bundle = output / f"{name}.forge3d"
        scene.save_bundle(bundle)
        restored = f3d.MapScene.load_bundle(bundle)
        replay = output / f"{name}-replay.png"
        restored.render(str(replay))
        if png.read_bytes() != replay.read_bytes():
            raise RuntimeError(f"{name}: bundle replay differs")
        results[name] = {"status": report.status, "sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
                         "backend": scene.last_render_backend, "metadata": scene.last_render_metadata}
    (output / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf8")
    print(json.dumps({name: {"sha256": result["sha256"], "backend": result["backend"]} for name, result in results.items()}, indent=2))


if __name__ == "__main__":
    main()
