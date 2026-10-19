"""Preparation CLI destination preflight, exercised through the real process."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from forge3d.preparation import (
    PreparationGrid, PreparationRecipe, PreparationSource, PreparedScene,
)


def test_existing_landscape_is_rejected_before_snapshot_and_retry_succeeds(tmp_path):
    grid = PreparationGrid("EPSG:28992", 8, 8, (100_000, 2, 0, 440_000, 0, -2))
    height = tmp_path / "height.npy"
    classes = tmp_path / "classes.npy"
    np.save(height, np.arange(64, dtype=np.float64).reshape(8, 8))
    np.save(classes, np.ones((8, 8), dtype=np.uint8))
    recipe = tmp_path / "recipe.json"
    PreparationRecipe(grid, (
        PreparationSource("dem", height, "elevation", grid=grid, units="m", vertical_reference="NAP"),
        PreparationSource("cover", classes, "categorical", grid=grid),
    ), "NAP").save(recipe)
    materials = tmp_path / "materials.json"
    materials.write_text('{"1":"Land"}', encoding="utf-8")
    snapshot = tmp_path / "prepared"
    existing = tmp_path / "landscape"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_bytes(b"existing package must remain unchanged")
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ, FORGE3D_NO_BOOTSTRAP="1", GDAL_PAM_ENABLED="NO")

    def run(destination):
        return subprocess.run([
            sys.executable, str(repo / "examples/prepare_scene.py"), str(recipe), str(snapshot),
            "--landscape", str(destination), "--layout", "7", "1", "1", "1",
            "--origin", "100000", "440000", "0", "--categorical-source", "cover",
            "--material-layers", str(materials),
        ], cwd=repo, env=env, capture_output=True, text=True)

    rejected = run(existing)
    assert rejected.returncode == 1
    assert "Traceback" not in rejected.stderr
    diagnostic = json.loads(rejected.stderr)
    assert diagnostic["code"] == "output_exists"
    assert str(existing) in diagnostic["message"]
    assert "new" in diagnostic["remediation"].lower()
    assert not snapshot.exists()
    assert before == {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    fresh = tmp_path / "new-landscape"
    corrected = run(fresh)
    assert corrected.returncode == 0, corrected.stderr
    assert PreparedScene.load(snapshot).layers["dem"].data.shape == (8, 8)
    assert (fresh / "height.png").is_file()
    assert (fresh / "landscape.json").is_file()
    assert sentinel.read_bytes() == before[Path("landscape/keep.txt")]
