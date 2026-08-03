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
    assert workflow.count("name: lfs-fixture-bundles") == 4
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
        "\n  # ============================================================================\n  # Accounted slow Python tests (one hosted representative)", 1
    )[0]
    slow_job = workflow.split("  test-python-slow:", 1)[1].split(
        "\n  # ============================================================================\n  # TERMINUS", 1
    )[0]
    golden_job = workflow.split("  test-golden-images:", 1)[1].split(
        "  refresh-recipe-certificates:", 1
    )[0]
    m06_job = workflow.split("  test-m06-full-geospatial-viewer:", 1)[1].split(
        "  build-docs:", 1
    )[0]

    for job in (python_job, slow_job):
        assert "needs: [build-wheels, prepare-lfs-fixtures]" in job
        assert job.count(".zip") == 1
        assert "python-tiffs.zip" in job
        assert "m06-dem.zip" not in job
        assert "forge3d.pdb" not in job
    assert "lfs-fixture-bundles" not in golden_job
    assert "needs: [build-wheels, prepare-lfs-fixtures, terrain-golden-paths]" in m06_job
    assert "m06-dem.zip" in m06_job
    assert "python-tiffs.zip" not in m06_job
    assert "Get-PSDrive -PSProvider FileSystem" in m06_job
    assert "Sort-Object Free -Descending" in m06_job
    assert (
        '$scratchScope = "$env:GITHUB_RUN_ID-$env:GITHUB_RUN_ATTEMPT-$env:GITHUB_JOB"'
        in m06_job
    )
    assert "FORGE3D_M06_SCRATCH_DIR" in m06_job
    assert '"CARGO_TARGET_DIR=$targetDir"' in m06_job
    assert '"FORGE3D_VIEWER_BINARY=$viewerBinary"' in m06_job
    assert "name: Clean M-06 build scratch" in m06_job
    assert "(Split-Path -Leaf $parentDir) -ne 'forge3d-ci-scratch'" in m06_job
    assert m06_job.index("name: Upload M-06 evidence") < m06_job.index(
        "name: Clean M-06 build scratch"
    )


def test_m06_path_filter_and_aggregator_contract() -> None:
    workflow = _workflow()
    paths_job = workflow.split("  terrain-golden-paths:", 1)[1].split(
        "\n  # ============================================================================\n  # Rust Tests", 1
    )[0]
    assert "m06: ${{ steps.filter.outputs.m06 }}" in paths_job
    m06_paths = paths_job.split("            m06:\n", 1)[1].split(
        "\n            f3dz:\n", 1
    )[0]
    for pattern in (
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        ".cargo/config.toml",
        "pyproject.toml",
        "pytest.ini",
        "conftest.py",
        "src/**",
        "python/forge3d/**",
        "scripts/install_compatible_wheel.py",
        "scripts/terrain_ci_probe.py",
        "scripts/assert_junit_zero_skips.py",
        "scripts/summarize_m06_evidence.py",
        "scripts/ci_pytest_lane.py",
        "tests/*.py",
        "tests/*.toml",
        "tests/golden/certificates/**",
        "tests/golden/recipes/mapscene_terrain_raster.png",
        "tests/data/vector_torture/cases.json",
        "assets/lidar/MtStHelens.laz",
        "assets/tif/switzerland_dem.tif",
        "assets/fonts/**",
        "assets/geoid/egm96_n120.bin",
    ):
        assert f"              - '{pattern}'" in m06_paths
    for broad_pattern in (
        ".github/workflows/ci.yml",
        "docs/**",
        "examples/**",
        "python/**",
        "tests/**",
        "assets/**",
    ):
        assert f"              - '{broad_pattern}'" not in m06_paths

    m06_job = workflow.split("  test-m06-full-geospatial-viewer:", 1)[1].split(
        "\n  # ============================================================================\n  # COMPENDIUM F3DZ", 1
    )[0]
    assert "needs: [build-wheels, prepare-lfs-fixtures, terrain-golden-paths]" in m06_job
    assert "if: >-" in m06_job
    for clause in (
        "github.event_name == 'workflow_dispatch'",
        "github.event_name == 'schedule'",
        "github.event_name == 'push' && github.ref == 'refs/heads/main'",
        "needs.terrain-golden-paths.outputs.m06 == 'true'",
    ):
        assert clause in m06_job

    aggregate = workflow.split("  ci-success:", 1)[1]
    assert "m06_required=" in aggregate
    required_branch, skip_tail = aggregate.split(
        'if [ "$m06_required" = "true" ]; then', 1
    )[1].split("\n          else\n", 1)
    skip_branch = skip_tail.split("\n          fi\n          if ", 1)[0]
    success_check = (
        '${{ needs.test-m06-full-geospatial-viewer.result }}" != "success"'
    )
    skipped_check = (
        '${{ needs.test-m06-full-geospatial-viewer.result }}" != "skipped"'
    )
    assert success_check in required_branch
    assert success_check not in skip_branch
    assert skipped_check in skip_branch
    assert skipped_check not in required_branch
    final_gate = aggregate.split(
        'if [ "${{ needs.prepare-lfs-fixtures.result }}" != "success" ] ||', 1
    )[1]
    assert '[ "$m06_failed" -ne 0 ] || \\' in final_gate
    assert "needs.test-m06-full-geospatial-viewer.result" not in final_gate
