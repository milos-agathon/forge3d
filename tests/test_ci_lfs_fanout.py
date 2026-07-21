from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_ci_downloads_lfs_once_and_shares_one_artifact() -> None:
    workflow = _workflow()

    assert workflow.count("git lfs pull") == 1
    assert "lfs: true" not in workflow
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
