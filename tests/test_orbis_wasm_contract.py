"""Build and execute the real ORBIS wasm artifact and production seams."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.wasm


ROOT = Path(__file__).resolve().parents[1]


def test_orbis_wasm_artifact_executes_registry_and_residency_contract() -> None:
    node = shutil.which("node")
    assert node is not None, "Node.js is required for the tracked ORBIS wasm smoke gate"
    command = [
        "cargo",
        "build",
        "--target",
        "wasm32-unknown-unknown",
        "--no-default-features",
        "--features",
        "enable-globe",
        "--lib",
        "--release",
        "--message-format=json",
    ]
    result = subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=360
    )
    artifacts: list[Path] = []
    for line in result.stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("reason") == "compiler-artifact":
            artifacts.extend(
                Path(filename)
                for filename in message.get("filenames", [])
                if filename.endswith(".wasm")
            )
    artifact = next((path for path in artifacts if path.name == "forge3d.wasm"), None)
    assert artifact is not None and artifact.is_file(), "cargo did not emit forge3d.wasm"
    smoke = subprocess.run(
        [node, str(ROOT / "tests" / "wasm_orbis_loader.mjs"), str(artifact)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "ORBIS wasm registry ABI PASS" in smoke.stdout
