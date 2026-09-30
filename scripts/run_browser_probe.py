"""TERRA-DETERMINATA browser WebGPU leg.

Runs the det_probe/det_raster WGSL canaries in headless Chrome and writes a
``{scene}.json``-style artifact (probe_sha256, raster_sha256, adapter metadata)
plus ``probe.sha256`` / ``raster.sha256`` files so the aggregate checker can
require the leg.

Usage::

    python scripts/run_browser_probe.py --output hash-out/browser.json \
        --hash-dir hash-out --scene terra_determinata_v1

The script serves only the three WGSL files and the harness page from a temp
directory — the repository is not exposed. Chrome must support WebGPU
(``--enable-unsafe-webgpu``); on hosted runners Chrome falls back to the
SwiftShader software adapter, which is still a real executed leg (its adapter
metadata is reported truthfully as software_fallback=true).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SHADER_DIR = REPO_ROOT / "src" / "shaders"
HARNESS_DIR = REPO_ROOT / "tools" / "determinism_browser"

CHROME_CANDIDATES = [
    os.environ.get("CHROME_PATH", ""),
    shutil.which("chrome"),
    shutil.which("google-chrome"),
    shutil.which("google-chrome-stable"),
    shutil.which("chromium"),
    shutil.which("chromium-browser"),
    r"C:/Program Files/Google/Chrome/Application/chrome.exe",
    r"C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def find_chrome() -> str | None:
    for cand in CHROME_CANDIDATES:
        if cand and Path(cand).exists():
            return str(cand)
    return None


def stage_assets(tmp: Path, probe_source: Path | None = None) -> None:
    """Copy exactly the assets the browser leg needs into the serve root."""
    shutil.copy(SHADER_DIR / "includes" / "determinism.wgsl", tmp / "determinism.wgsl")
    shutil.copy(probe_source or SHADER_DIR / "det_probe.wgsl", tmp / "det_probe.wgsl")
    shutil.copy(SHADER_DIR / "det_raster.wgsl", tmp / "det_raster.wgsl")
    shutil.copy(HARNESS_DIR / "det_probe.js", tmp / "det_probe.js")
    shutil.copy(HARNESS_DIR / "index.html", tmp / "index.html")


def run_browser_probe(timeout_s: int = 60, probe_source: Path | None = None) -> dict:
    chrome = find_chrome()
    if chrome is None:
        return {"status": "absent", "reason": "no Chrome/Chromium binary found"}

    received: dict = {}

    class Handler(SimpleHTTPRequestHandler):
        def do_POST(self) -> None:  # the page pushes its final record here
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length)
            try:
                received["record"] = json.loads(body)
            except json.JSONDecodeError:
                received["record"] = {"status": "failed", "reason": f"bad POST body: {body[:200]!r}"}
            self.send_response(204)
            self.end_headers()

        def log_message(self, *a):  # quiet
            pass

    with tempfile.TemporaryDirectory(prefix="det-browser-") as tmp_str:
        tmp = Path(tmp_str)
        stage_assets(tmp, probe_source)

        handler = partial(Handler, directory=str(tmp))
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        proc = None
        try:
            url = f"http://127.0.0.1:{port}/index.html"
            # Try SwiftShader first: it is the only adapter guaranteed present
            # on GPU-less hosted runners, and the whole point of the leg is
            # that pinned arithmetic gives identical bits even on software.
            # If this Chrome lacks SwiftShader, retry with the default
            # (hardware) adapter so local machines can still run the leg.
            base = [
                chrome,
                "--headless=new",
                "--disable-gpu-sandbox",
                "--no-sandbox",
                "--enable-unsafe-webgpu",
                "--enable-features=Vulkan",
            ]
            deadline = time.time() + timeout_s
            for extra in (
                ["--use-webgpu-adapter=swiftshader", "--use-angle=vulkan"],
                [],
            ):
                received.clear()
                proc = subprocess.Popen(
                    base + extra + [url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                while "record" not in received and time.time() < deadline:
                    if proc.poll() is not None:
                        break  # chrome exited before posting (e.g. crash)
                    time.sleep(0.2)
                rec = received.get("record")
                # A null adapter under the forced SwiftShader flag means this
                # Chrome lacks it — retry on the default (hardware) adapter.
                # Any other outcome is final.
                if rec is not None and "requestAdapter returned null" not in str(
                    rec.get("reason", "")
                ):
                    break
                try:
                    proc.kill()
                except OSError:
                    pass
            record = received.get("record")
            if record is None:
                return {
                    "status": "failed",
                    "reason": "no probe record posted before timeout/chrome exit",
                    "stderr_tail": (proc.stderr.read() if proc and proc.stderr else "")[-2000:],
                }
        finally:
            if proc is not None:
                try:
                    proc.kill()
                except OSError:
                    pass
            server.shutdown()
            thread.join(timeout=5)

    record.setdefault("chrome", Path(chrome).name)
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON record output")
    parser.add_argument("--hash-dir", type=Path, help="Directory for probe.sha256/raster.sha256 files")
    parser.add_argument("--scene", default="terra_determinata_v1")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--probe-source", type=Path, default=None)
    args = parser.parse_args(argv)

    record = run_browser_probe(timeout_s=args.timeout, probe_source=args.probe_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")

    if args.hash_dir and record.get("status") == "ok":
        args.hash_dir.mkdir(parents=True, exist_ok=True)
        (args.hash_dir / "probe.sha256").write_text(record["probe_sha256"] + "\n")
        (args.hash_dir / "raster.sha256").write_text(record["raster_sha256"] + "\n")
        # The aggregate checker attributes the leg via browser.json in the
        # artifact dir; a missing file leaves the leg unattributed.
        (args.hash_dir / "browser.json").write_text(json.dumps(record, indent=2) + "\n")

    status = record.get("status")
    print(json.dumps(record))
    return 0 if status == "ok" else (2 if status == "absent" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
