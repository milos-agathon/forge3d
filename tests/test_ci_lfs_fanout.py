from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_ci_downloads_verified_lfs_media_once_and_shares_one_artifact() -> None:
    workflow = _workflow()

    assert "git lfs pull" not in workflow
    assert "lfs: true" not in workflow
    assert (
        "https://media.githubusercontent.com/media/milos-agathon/forge3d/"
        "92a86baa3c8f6ba3c3a7368e4f80d4004905a433"
    ) in workflow
    assert workflow.count("sha256sum --check -") == 1
    assert "cff39b4e02d7ba13c48f3d8b1a4080d40ada753ade62fa951459fe4e01e98b48" in workflow
    assert "875b243474b151175f76037acd60c2149ac2e46fba9ba2bbce0c9a6998015dd3" in workflow
    assert "d09d229fa265749720a6b4bd40c440799f43286bf2d401d732ea77f89d0bd478" in workflow
    assert "is still an LFS pointer" in workflow
    assert workflow.count("name: lfs-fixture-bundles") == 3
    assert workflow.count("uses: actions/upload-artifact@v4") >= 1
    assert "retention-days: 1" in workflow


def test_ci_lfs_manifest_contains_only_lane_fixtures() -> None:
    workflow = _workflow()
    prepare = workflow.split("  prepare-lfs-fixtures:", 1)[1].split(
        "  terrain-golden-paths:", 1
    )[0]

    assert "assets/tif/Mount_Fuji_30m.tif" in prepare
    assert "assets/tif/dem_rainier.tif" in prepare
    assert "assets/tif/switzerland_dem.tif" in prepare
    assert "python/forge3d/forge3d.pdb" not in prepare
    assert "assets/highres.png" not in prepare
    assert "assets/swiss-legend.png" not in prepare
    assert "assets/tif/Bryce_Canyon.tif" not in prepare
    assert "assets/tif/luxembourg_dem.tif" not in prepare
    assert "assets/tif/switzerland_land_cover.tif" not in prepare


def test_python_and_m06_restore_only_their_fixture_bundles() -> None:
    workflow = _workflow()
    python_job = workflow.split("  test-python:", 1)[1].split(
        "  test-terminus-fuzz:", 1
    )[0]
    golden_job = workflow.split("  test-golden-images:", 1)[1].split(
        "  refresh-recipe-certificates:", 1
    )[0]
    m06_job = workflow.split("  test-m06-full-geospatial-viewer:", 1)[1].split(
        "  build-docs:", 1
    )[0]

    assert "needs: [build-wheels, prepare-lfs-fixtures]" in python_job
    assert "python-tiffs.zip" in python_job
    assert "forge3d.pdb" not in python_job
    assert "lfs-fixture-bundles" not in golden_job
    assert "needs: [build-wheels, prepare-lfs-fixtures]" in m06_job
    assert "m06-dem.zip" in m06_job
    assert "python-tiffs.zip" not in m06_job
