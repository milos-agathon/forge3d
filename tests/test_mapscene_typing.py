from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_mapscene_quickstart_typechecks_with_public_stubs(tmp_path: Path) -> None:
    if subprocess.run(
        [sys.executable, "-m", "mypy", "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode != 0:
        pytest.skip("mypy is not installed")

    sample = tmp_path / "mapscene_quickstart_typing.py"
    sample.write_text(
        """
from __future__ import annotations

from pathlib import Path

import forge3d as f3d
import numpy as np
from forge3d.style import RasterHeightSurfaceStyle


def build_scene(output: Path) -> f3d.MapScene:
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(path=f3d.mini_dem_path(), crs="EPSG:3857"),
        lighting=f3d.LightingPreset(name="rainier_showcase"),
        output=f3d.OutputSpec(
            width=1600,
            height=1000,
            path=output,
            samples=4,
            denoiser="atrous",
            aovs=("albedo", "normal", "depth"),
        ),
    )
    manifest: dict[str, object] = f3d.recipe_manifest(scene)
    assert manifest["kind"] == "mapscene_recipe_manifest"
    image = f3d.RenderPassInput(np.zeros((1000, 1600, 4), dtype=np.uint8))
    passes = [f3d.RenderPassSpec("color", "color", ("source",))]
    report = scene.render_passes(passes, {"source": image}, str(output))
    assert report.status is not None
    return scene


def build_thematic_scene() -> f3d.MapScene:
    population = RasterHeightSurfaceStyle(2, "people/km²")
    assert population.high_color[3] == 255
    categorical = f3d.CategoricalRasterStyle({0: (255, 0, 0, 255)})
    bivariate = f3d.BivariateRasterStyle(
        [1, 2], [10, 20], [[(255, 0, 0, 255)] * 3] * 3,
        "Population", "Temperature", x_units="people/km²", y_units="°C",
    )
    assert bivariate.x_units == "people/km²"
    legend: dict[str, object] = bivariate.legend()
    assert legend["kind"] == "bivariate"
    result: f3d.RasterStyleResult = categorical.apply(np.zeros((2, 2)))
    assert result.valid_mask.any()
    return f3d.MapScene(
        terrain=f3d.TerrainSource(data=np.zeros((2, 2)), style=population),
        layers=[f3d.RasterOverlay("theme", data=np.zeros((2, 2)),
                                  secondary_data=np.zeros((2, 2)), style=bivariate)],
    )
""".strip()
        + "\n",
        encoding="utf-8",
    )

    repo_python = Path(__file__).resolve().parents[1] / "python"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_python), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--ignore-missing-imports",
            str(sample),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
