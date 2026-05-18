from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_speckit_prerequisite_context_targets_feature_005():
    feature_json = json.loads(_read(".specify/feature.json"))
    assert feature_json["feature_directory"] == "specs/005-map-assets-bundles-p1"

    shell = (shutil.which("pwsh") or shutil.which("powershell")) if os.name == "nt" else None
    if shell is not None:
        result = subprocess.run(
            [
                shell,
                "-NoProfile",
                "-Command",
                ".specify/scripts/powershell/check-prerequisites.ps1 -Json -RequireTasks -IncludeTasks",
            ],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        payload = json.loads(result.stdout)
        assert payload["FEATURE_DIR"].replace("\\", "/").endswith("specs/005-map-assets-bundles-p1")
        assert "tasks.md" in payload["AVAILABLE_DOCS"]
    else:
        assert (ROOT / "specs/005-map-assets-bundles-p1/tasks.md").exists()


def test_completed_phase1_tasks_record_red_yellow_remediation_evidence():
    tasks = _read("specs/005-map-assets-bundles-p1/tasks.md")

    for code in (
        "crs_mismatch",
        "missing_glyphs",
        "pro_gated_path",
        "placeholder_fallback",
        "experimental_feature",
        "python_public_3dtiles_incomplete",
    ):
        assert code in tasks

    for marker in (
        "bundle fixture manifest",
        "compatibility decision",
        "red/yellow remediation",
    ):
        assert marker in tasks


def test_docs_state_and_matrix_record_red_yellow_remediation():
    api_reference = _read("docs/api/api_reference.rst")
    workflow_guide = _read("docs/guides/data_and_scene_workflows.md")
    matrix = _read("docs/superpowers/state/requirements-verification-matrix.md")
    ledger = _read("docs/superpowers/state/implementation-ledger.md")
    context = _read("docs/superpowers/state/current-context-pack.md")

    assert "MapSceneBuildingLayer" in api_reference
    assert "legacy ``forge3d.BuildingLayer``" in api_reference
    assert "full Cesium runtime parity" in workflow_guide
    assert "local/provided feature styling" in workflow_guide
    assert "red/yellow remediation" in matrix
    assert "red/yellow remediation" in ledger
    assert "red/yellow remediation" in context

    next_prompt = context.split("## Next Exact Prompt", 1)[-1]
    assert "005-map-assets-bundles-p1" in next_prompt
    assert "004-mapscene-mvp" not in next_prompt
