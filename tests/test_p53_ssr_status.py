#!/usr/bin/env python3
"""P5.3 SSR regression: ensure final ssr_status == "OK".

This test drives the P5.3 SSR glossy spheres and thickness ablation
examples via `cargo run` and then inspects the aggregated P5 meta
file at reports/p5/p5_meta.json. It is gated behind the same
interactive-viewer CI controls as other viewer-based tests.
"""

import json
import os
import platform
import subprocess
from pathlib import Path

import pytest


# Mark as interactive viewer test and require explicit CI opt-in
pytestmark = pytest.mark.interactive_viewer
pytestmark = [
    pytestmark,
    pytest.mark.skipif(
        not os.environ.get("RUN_INTERACTIVE_VIEWER_CI"),
        reason="Set RUN_INTERACTIVE_VIEWER_CI=1 to enable interactive viewer tests in CI",
    ),
]


# Skip if no GPU or likely headless environment
@pytest.mark.skipif(
    platform.system() == "Linux"
    and not os.environ.get("DISPLAY")
    and not os.environ.get("WAYLAND_DISPLAY"),
    reason="Viewer requires a display; no DISPLAY/WAYLAND_DISPLAY present",
)
@pytest.mark.skipif(
    (not pytest.importorskip("forge3d").has_gpu())
    and (not os.environ.get("RUN_INTERACTIVE_VIEWER_CI")),
    reason="GPU not available for P5.3 SSR test (override with RUN_INTERACTIVE_VIEWER_CI=1)",
)
@pytest.mark.slow
def test_p53_ssr_status_ok():
    """Run P5.3 SSR harness and assert final ssr_status == "OK".

    The sequence mirrors the manual workflow:
      1) Run p5_ssr_glossy to populate SSR stats and stripe contrast.
      2) Run p5_ssr_thickness_ablation to compute thickness metrics.
      3) Read reports/p5/p5_meta.json and require ssr_status == "OK".
    """

    repo_root = Path(__file__).resolve().parents[1]
    meta_path = repo_root / "reports" / "p5" / "p5_meta.json"

    # Remove any stale meta so we always read fresh results from this run.
    if meta_path.exists():
        meta_path.unlink()

    # 1) Glossy spheres capture (stripe contrast, hit-rate, fallback metrics)
    subprocess.run(
        [
            "cargo",
            "run",
            "--release",
            "--example",
            "p5_ssr_glossy",
            "-q",
        ],
        cwd=str(repo_root),
        check=True,
    )

    # 2) Thickness ablation capture (undershoot metrics + edge streaks)
    subprocess.run(
        [
            "cargo",
            "run",
            "--release",
            "--example",
            "p5_ssr_thickness_ablation",
            "-q",
        ],
        cwd=str(repo_root),
        check=True,
    )

    # 3) Verify meta exists and contains an OK SSR status
    assert meta_path.exists(), "p5_meta.json was not written by P5.3 harness"
    data = json.loads(meta_path.read_text())

    ssr_status = data.get("ssr_status")
    assert ssr_status == "OK", f"Expected ssr_status == 'OK', got {ssr_status!r}"
