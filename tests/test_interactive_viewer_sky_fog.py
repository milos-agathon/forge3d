import os
import sys
import time
import platform
import subprocess
import tempfile
from pathlib import Path

import pytest

# Mark as interactive viewer test and require explicit CI opt-in
import pytest

pytestmark = pytest.mark.interactive_viewer

# Skip in CI unless explicitly enabled
pytestmark = [pytestmark, pytest.mark.skipif(not os.environ.get("RUN_INTERACTIVE_VIEWER_CI"), reason="Set RUN_INTERACTIVE_VIEWER_CI=1 to enable interactive viewer tests in CI")]

# Skip if no GPU or likely headless environment
@pytest.mark.skipif(
    platform.system() == "Linux" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"),
    reason="Viewer requires a display; no DISPLAY/WAYLAND_DISPLAY present",
)
@pytest.mark.skipif(
    (not pytest.importorskip("forge3d").has_gpu()) and (not os.environ.get("RUN_INTERACTIVE_VIEWER_CI")),
    reason="GPU not available for interactive viewer test (override with RUN_INTERACTIVE_VIEWER_CI=1)",
)
@pytest.mark.slow
def test_interactive_viewer_sky_fog_snapshot():
    repo_root = Path(__file__).resolve().parents[1]

    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "sky_fog_snapshot.png"

        # Launch the interactive viewer example
        # Note: we use -q to keep output minimal; stdin is used for commands
        proc = subprocess.Popen(
            ["cargo", "run", "--example", "interactive_viewer", "-q"],
            cwd=str(repo_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert proc.stdin is not None

            # Give the event loop a moment to initialize window
            time.sleep(1.5)

            # Issue a few sky/fog commands and a snapshot, then quit
            cmds = [
                ":sky preetham\n",
                ":sky-turbidity 4.0\n",
                ":fog on\n",
                ":fog-density 0.04\n",
                ":fog-mode froxels\n",
                f":snapshot {out_path}\n",
                ":quit\n",
            ]
            for c in cmds:
                proc.stdin.write(c)
                proc.stdin.flush()
                time.sleep(0.1)

            # Wait up to 30s for process to exit and snapshot to be written
            deadline = time.time() + 30.0
            while time.time() < deadline:
                if proc.poll() is not None:
                    break
                if out_path.exists() and out_path.stat().st_size > 0:
                    break
                time.sleep(0.25)

            # Ensure the process has exited by now
            if proc.poll() is None:
                # Final grace period after snapshot
                time.sleep(1.0)
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()

            # Validate snapshot exists and is non-empty
            assert out_path.exists(), "Snapshot file was not created"
            assert out_path.stat().st_size > 0, "Snapshot file is empty"
        finally:
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
