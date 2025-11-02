import os
import time
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.interactive_viewer
pytestmark = [
    pytestmark,
    pytest.mark.skipif(
        not os.environ.get("RUN_INTERACTIVE_VIEWER_CI"),
        reason="Set RUN_INTERACTIVE_VIEWER_CI=1 to enable interactive viewer tests in CI",
    ),
]


@pytest.mark.skipif(
    platform.system() == "Linux" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"),
    reason="Viewer requires a display; no DISPLAY/WAYLAND_DISPLAY present",
)
@pytest.mark.skipif(
    (not pytest.importorskip("forge3d").has_gpu()) and (not os.environ.get("RUN_INTERACTIVE_VIEWER_CI")),
    reason="GPU not available for interactive viewer test (override with RUN_INTERACTIVE_VIEWER_CI=1)",
)
class TestM3Acceptance:
    def _run_viewer(self, repo_root: Path, commands: list[str], out_path: Path, width: int = 640, height: int = 360):
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
            time.sleep(1.5)
            # Set size and apply commands
            pre = [f":size {width} {height}\n"]
            cmds = pre + [c if c.endswith("\n") else c + "\n" for c in commands]
            for c in cmds:
                proc.stdin.write(c)
                proc.stdin.flush()
                time.sleep(0.05)
            # Request snapshot and quit
            proc.stdin.write(f":snapshot {out_path}\n")
            proc.stdin.flush()
            time.sleep(0.1)
            proc.stdin.write(":quit\n")
            proc.stdin.flush()
            # Wait for up to 30s
            deadline = time.time() + 30.0
            while time.time() < deadline:
                if out_path.exists() and out_path.stat().st_size > 0:
                    break
                if proc.poll() is not None:
                    break
                time.sleep(0.2)
            if proc.poll() is None:
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
            assert out_path.exists(), f"Snapshot not created: {out_path}"
            assert out_path.stat().st_size > 0, "Snapshot file empty"
        finally:
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass

    @staticmethod
    def _load_png(path: Path) -> np.ndarray:
        img = Image.open(str(path))
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        return np.array(img, dtype=np.uint8)

    @staticmethod
    def _psnr(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        mse = np.mean((a - b) ** 2)
        if mse <= 1e-8:
            return 100.0
        PIXEL_MAX = 255.0
        return 20.0 * np.log10(PIXEL_MAX) - 10.0 * np.log10(mse)

    def test_sky_model_stability_and_difference(self, tmp_path: Path):
        repo_root = Path(__file__).resolve().parents[1]
        out1 = tmp_path / "sky_preetham_t2.png"
        out2 = tmp_path / "sky_preetham_t2_repeat.png"
        out3 = tmp_path / "sky_hosek_t2.png"

        # Preetham turbidity 2.0
        self._run_viewer(repo_root, [":sky preetham", ":sky-turbidity 2.0", ":sky-exposure 1.0"], out1)
        # Repeat run for stability
        self._run_viewer(repo_root, [":sky preetham", ":sky-turbidity 2.0", ":sky-exposure 1.0"], out2)
        # Hosek-Wilkie same turbidity
        self._run_viewer(repo_root, [":sky hosek-wilkie", ":sky-turbidity 2.0", ":sky-exposure 1.0"], out3)

        a = self._load_png(out1)
        b = self._load_png(out2)
        c = self._load_png(out3)

        psnr_stability = self._psnr(a, b)
        # The same config should be very close (allow tiny differences from noise/jitter)
        assert psnr_stability > 40.0, f"Sky stability PSNR too low: {psnr_stability:.2f} dB"

        psnr_model_diff = self._psnr(a, c)
        # Different models should produce a noticeable difference
        assert psnr_model_diff < 35.0, f"Sky model difference too small (PSNR={psnr_model_diff:.2f} dB)"

    def test_fog_on_off_changes_image(self, tmp_path: Path):
        repo_root = Path(__file__).resolve().parents[1]
        out_off = tmp_path / "fog_off.png"
        out_on = tmp_path / "fog_on.png"

        # Base scene without fog
        self._run_viewer(repo_root, [":fog off", ":sky preetham", ":sky-turbidity 3.0"], out_off)
        # Fog settings that should visibly affect the image
        cmds_on = [
            ":fog on",
            ":fog-density 0.06",
            ":fog-g 0.6",
            ":fog-steps 96",
            ":fog-temporal 0.5",
        ]
        self._run_viewer(repo_root, cmds_on, out_on)

        a = self._load_png(out_off)
        b = self._load_png(out_on)
        psnr = self._psnr(a, b)
        # Fog on/off should produce a significant visual change; expect PSNR well below high similarity
        assert psnr < 30.0, f"Fog on/off change too small (PSNR={psnr:.2f} dB)"
