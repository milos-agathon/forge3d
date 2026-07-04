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
    return scene
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
