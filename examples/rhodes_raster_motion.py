"""Verify class visibility and geographic mask edges on physical Rhodes frames."""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from forge3d import RasterRecipe, map_scene as ms
from forge3d._map_scene_projection import TerrainProjector
from forge3d.preparation import PreparationGrid
from rhodes_raster_recipe import COLORS, scene


def mask_edge_controls(ids, grid):
    """Pick opposing homogeneous samples along registered clear/masked edges."""
    result = []
    # At the 21 km camera range a native pixel spans tens of metres.
    # Offset the samples five 10 m cells from each edge.
    step = 5
    for row in range(step, grid.height - step):
        for col in np.flatnonzero((ids[row, 1:] == 0) != (ids[row, :-1] == 0)):
            left, right = col - step, col + 1 + step
            if left < step or right >= grid.width - step:
                continue
            a, b = int(ids[row, left]), int(ids[row, right])
            if (a == 0) == (b == 0):
                continue
            if not (
                np.all(ids[row - 2 : row + 3, left - 2 : left + 3] == a)
                and np.all(ids[row - 2 : row + 3, right - 2 : right + 3] == b)
            ):
                continue
            if result and row - result[-1][0] <= step * 2:
                continue
            result.append((row, left, right, a, b))
    return result


def verify(root, revised):
    publication = root / ("publication-revised" if revised else "publication")
    declared = json.loads((publication / "frames.json").read_text())
    recipe_file = (
        root
        / "gpu"
        / ("2023-08-17-2.recipe.json" if revised else "2023-08-07-0.recipe.json")
    )
    recipe = RasterRecipe.from_json(recipe_file.read_text(encoding="utf8"))
    grid: PreparationGrid = recipe.grid
    with rasterio.open(publication / "classes.tif") as dataset:
        ids = dataset.read(1)
    edge_pairs = mask_edge_controls(ids, grid)
    assert edge_pairs
    rgb = np.asarray([color[:3] for color in COLORS], dtype=np.float32) / 255
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    native_render = ms._render_terrain_renderer_result
    attachments = []

    def observe(*args, **kwargs):
        result = native_render(*args, **kwargs)
        assert result is not None and result.aov_frame is not None
        attachments.append(np.asarray(result.aov_frame.albedo()))
        return result

    frames = []
    ms._render_terrain_renderer_result = observe
    try:
        for frame in declared["frames"]:
            live = scene(
                root / "prepared",
                grid,
                azimuth=frame["azimuth"],
                elevation=frame["tilt"],
                publication=True,
            )
            live.recipe.output.aovs = ("albedo",)
            output = publication / ("verify-" + frame["frame"])
            live.render(str(output), raster_recipe=recipe)
            assert live.last_render_metadata["thematic_legend"] == declared["legend"]
            albedo = attachments.pop()
            projected = TerrainProjector(live.recipe)
            # The texture hardware supplies the sRGB conversion and the AOV
            # stores binary16. Identify each authored swatch's native value
            # on the first frame, then require those exact values throughout.
            if not frames:
                palette = np.unique(albedo.reshape(-1, 3), axis=0)
                distances = np.linalg.norm(palette[:, None] - linear, axis=-1)
                native_colors = palette[distances.argmin(axis=0)]
                assert len(np.unique(native_colors, axis=0)) == len(COLORS)
                # Each chosen sample lies within its authored swatch's
                # non-overlapping half-neighbour-distance in linear RGB.
                separation = np.linalg.norm(linear[:, None] - linear, axis=-1)
                np.fill_diagonal(separation, np.inf)
                assert np.all(
                    np.linalg.norm(native_colors - linear, axis=-1)
                    < separation.min(axis=0) / 2
                )
            counts = [
                int(np.all(albedo == color, axis=-1).sum()) for color in native_colors
            ]
            assert all(counts), (frame["frame"], counts)
            mask_edges = []
            for row, left, right, a, b in edge_pairs:
                for col, identifier in ((left, a), (right, b)):
                    x, y, _ = projected.project(grid.xy(col, row))
                    observed = albedo[int(y), int(x), :3]
                    if identifier == 0:
                        if np.any(observed):
                            break
                    elif not np.array_equal(observed, native_colors[identifier - 1]):
                        break
                else:
                    mask_edges.append([row, int(left), int(right), a, b])
            assert mask_edges, frame["frame"]
            with (
                Image.open(output) as probe,
                Image.open(publication / frame["frame"]) as beauty,
            ):
                np.testing.assert_array_equal(probe, beauty)
            frames.append(
                {
                    "frame": frame["frame"],
                    "palette_pixels": counts,
                    "mask_edges": mask_edges,
                }
            )
    finally:
        ms._render_terrain_renderer_result = native_render
    persistent_edges = set.intersection(
        *(set(map(tuple, frame["mask_edges"])) for frame in frames)
    )
    assert persistent_edges, "No registered mask boundary survived the full camera move"
    assert recipe.metrics()["readback_bytes"] == 0
    assert recipe.metrics()["classification_dispatches"] == 1
    recipe.close()
    (publication / "motion-controls.json").write_text(
        json.dumps(
            {
                "native_linear_rgb": native_colors.tolist(),
                "frames": frames,
                "persistent_mask_edges": sorted(persistent_edges),
            },
            indent=2,
        )
    )
    print(
        publication.name,
        "all five native albedos + geographic mask edges in",
        len(frames),
        "frames",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    for revised in (False, True):
        verify(args.directory.resolve(), revised)
