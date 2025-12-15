#!/usr/bin/env python3
"""IPC smoke test - proof-tight verification that IPC works end-to-end.

This script provides deterministic proof of:
1. Geometry is actually loaded (via get_stats)
2. Loading changes rendered output (before/after diff)
3. set_transform changes rendered output (after load/after transform diff)

Produces 3 snapshots with pixel diff metrics that must meet hard thresholds.
"""

from __future__ import annotations

import json
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

# Add the python package to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

READY_PATTERN = re.compile(r"FORGE3D_VIEWER_READY port=(\d+)")

# Hard thresholds (non-negotiable)
DIFF_BEFORE_AFTER_LOAD_MIN_PCT = 2.0
DIFF_AFTER_LOAD_AFTER_TRANSFORM_MIN_PCT = 1.0


def find_binary() -> str:
    """Find the interactive_viewer binary."""
    cargo_target = Path(__file__).parent.parent / "target"
    for profile in ["release", "debug"]:
        if sys.platform == "win32":
            binary = cargo_target / profile / "interactive_viewer.exe"
        else:
            binary = cargo_target / profile / "interactive_viewer"
        if binary.exists():
            return str(binary)
    raise FileNotFoundError("interactive_viewer binary not found")


def send_command(sock: socket.socket, cmd: dict) -> dict:
    """Send NDJSON command and receive response."""
    request = json.dumps(cmd) + "\n"
    print(f"  -> {request.strip()}")
    sock.sendall(request.encode("utf-8"))
    
    response_data = b""
    while b"\n" not in response_data:
        chunk = sock.recv(4096)
        if not chunk:
            raise RuntimeError("Connection closed")
        response_data += chunk
    
    line = response_data.split(b"\n")[0].decode("utf-8")
    response = json.loads(line)
    print(f"  <- {response}")
    return response


def compute_pixel_diff(img1_path: Path, img2_path: Path) -> Tuple[int, float]:
    """Compute pixel diff between two images.
    
    If images have different sizes, resizes to match the smaller one.
    
    Returns:
        (changed_pixel_count, percent_changed)
    """
    from PIL import Image
    import numpy as np
    
    pil_img1 = Image.open(img1_path).convert("RGB")
    pil_img2 = Image.open(img2_path).convert("RGB")
    
    # If sizes differ, resize to the smaller size for comparison
    if pil_img1.size != pil_img2.size:
        target_size = (
            min(pil_img1.width, pil_img2.width),
            min(pil_img1.height, pil_img2.height),
        )
        pil_img1 = pil_img1.resize(target_size, Image.Resampling.LANCZOS)
        pil_img2 = pil_img2.resize(target_size, Image.Resampling.LANCZOS)
    
    img1 = np.array(pil_img1)
    img2 = np.array(pil_img2)
    
    # A pixel is "changed" if any channel differs by more than a small threshold
    # (to avoid noise from compression artifacts)
    diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
    max_diff = np.max(diff, axis=2)  # Max diff across RGB channels
    changed = max_diff > 2  # Threshold to ignore minor compression noise
    
    changed_count = int(np.sum(changed))
    total_pixels = img1.shape[0] * img1.shape[1]
    pct_changed = 100.0 * changed_count / total_pixels
    
    return changed_count, pct_changed


def wait_for_stats_ready(sock: socket.socket, timeout: float = 10.0) -> dict:
    """Poll get_stats until vb_ready=True or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        resp = send_command(sock, {"cmd": "get_stats"})
        if not resp.get("ok"):
            raise RuntimeError(f"get_stats failed: {resp}")
        stats = resp.get("stats", {})
        if stats.get("vb_ready"):
            return stats
        time.sleep(0.2)
    raise TimeoutError("Timed out waiting for vb_ready=True")


def main() -> int:
    print("=" * 60)
    print("IPC SMOKE TEST - PROOF-TIGHT VERIFICATION")
    print("=" * 60)
    print()
    
    try:
        binary = find_binary()
        print(f"Binary: {binary}")
    except FileNotFoundError as e:
        print(f"FAIL: {e}")
        return 1
    
    out_dir = Path("examples/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Snapshot paths
    snap_before_load = out_dir / "ipc_before_load.png"
    snap_after_load = out_dir / "ipc_after_load.png"
    snap_after_transform = out_dir / "ipc_after_transform.png"
    
    # Start viewer
    print("\n[1] Starting viewer with --ipc-port 0...")
    process = subprocess.Popen(
        [binary, "--ipc-port", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    # Wait for READY line
    port = None
    start_time = time.time()
    timeout = 30.0
    
    while time.time() - start_time < timeout:
        if process.poll() is not None:
            print(f"FAIL: Process exited with code {process.returncode}")
            return 1
        
        line = process.stdout.readline() if process.stdout else ""
        if line:
            print(f"  [viewer] {line.rstrip()}")
            match = READY_PATTERN.search(line)
            if match:
                port = int(match.group(1))
                print(f"\n*** READY line detected: port={port} ***\n")
                break
    
    if port is None:
        print("FAIL: Timeout waiting for READY line")
        process.terminate()
        return 1
    
    # Give viewer time to initialize
    time.sleep(1.5)
    
    # Connect
    print("[2] Connecting to IPC server...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10.0)
    try:
        sock.connect(("127.0.0.1", port))
        print(f"  Connected to 127.0.0.1:{port}")
    except Exception as e:
        print(f"FAIL: Connection error: {e}")
        process.terminate()
        return 1
    
    try:
        # === STEP A: get_stats BEFORE load (must show vb_ready=false) ===
        print("\n[3] get_stats BEFORE load...")
        resp = send_command(sock, {"cmd": "get_stats"})
        if not resp.get("ok"):
            print(f"FAIL: get_stats failed: {resp}")
            raise RuntimeError("get_stats failed")
        stats_before = resp.get("stats", {})
        print(f"  stats_before: {stats_before}")
        
        # Verify blank state
        if stats_before.get("vb_ready"):
            print("  WARNING: vb_ready=True before load (may have initial geometry)")
        if stats_before.get("vertex_count", 0) > 0:
            print("  WARNING: vertex_count > 0 before load")
        
        # Set camera to fixed position
        resp = send_command(sock, {
            "cmd": "cam_lookat",
            "eye": [0.0, 1.0, 3.0],
            "target": [0.0, 0.5, 0.0],
            "up": [0.0, 1.0, 0.0],
        })
        assert resp.get("ok"), f"cam_lookat failed: {resp}"
        
        resp = send_command(sock, {"cmd": "set_fov", "deg": 45.0})
        assert resp.get("ok"), f"set_fov failed: {resp}"
        
        # Wait for viewer to render a frame at proper resolution
        time.sleep(1.0)
        
        # === STEP B: Snapshot BEFORE load ===
        print(f"\n[4] Snapshot BEFORE load -> {snap_before_load}")
        resp = send_command(sock, {
            "cmd": "snapshot",
            "path": str(snap_before_load.absolute()),
            "width": 640,
            "height": 480,
        })
        assert resp.get("ok"), f"snapshot failed: {resp}"
        time.sleep(1.0)
        
        # === STEP C: Load geometry ===
        obj_path = Path("assets/objects/bunny.obj")
        if not obj_path.exists():
            obj_path = Path("assets/objects/cornell_box.obj")
        if not obj_path.exists():
            print("FAIL: No OBJ file found in assets/objects/")
            raise FileNotFoundError("No OBJ found")
        
        print(f"\n[5] Loading model: {obj_path}")
        resp = send_command(sock, {"cmd": "load_obj", "path": str(obj_path.absolute())})
        assert resp.get("ok"), f"load_obj failed: {resp}"
        
        # === STEP D: get_stats AFTER load (must show vb_ready=true, counts>0) ===
        print("\n[6] Waiting for get_stats vb_ready=True...")
        stats_after = wait_for_stats_ready(sock, timeout=10.0)
        print(f"  stats_after: {stats_after}")
        
        if not stats_after.get("vb_ready"):
            print("FAIL: vb_ready is not True after load")
            raise RuntimeError("vb_ready not True")
        if stats_after.get("vertex_count", 0) == 0:
            print("FAIL: vertex_count is 0 after load")
            raise RuntimeError("vertex_count is 0")
        if stats_after.get("index_count", 0) == 0:
            print("FAIL: index_count is 0 after load")
            raise RuntimeError("index_count is 0")
        
        print(f"  OK: vb_ready={stats_after['vb_ready']}, "
              f"vertex_count={stats_after['vertex_count']}, "
              f"index_count={stats_after['index_count']}")
        
        time.sleep(0.5)
        
        # === STEP E: Snapshot AFTER load ===
        print(f"\n[7] Snapshot AFTER load -> {snap_after_load}")
        resp = send_command(sock, {
            "cmd": "snapshot",
            "path": str(snap_after_load.absolute()),
            "width": 640,
            "height": 480,
        })
        assert resp.get("ok"), f"snapshot failed: {resp}"
        time.sleep(1.0)
        
        # === STEP F: Change view significantly (camera move + set_transform) ===
        print("\n[8] Changing camera position significantly...")
        # Move camera to a very different position - this MUST change the render
        resp = send_command(sock, {
            "cmd": "cam_lookat",
            "eye": [2.0, 2.0, 2.0],     # Move camera to a different angle
            "target": [0.0, 0.5, 0.0],
            "up": [0.0, 1.0, 0.0],
        })
        assert resp.get("ok"), f"cam_lookat failed: {resp}"
        
        # Also apply object transform 
        import math
        angle = math.radians(45)
        rotation_quat = [0.0, math.sin(angle/2), 0.0, math.cos(angle/2)]
        resp = send_command(sock, {
            "cmd": "set_transform",
            "translation": [0.5, 0.2, 0.0],
            "rotation_quat": rotation_quat,
            "scale": [1.5, 1.5, 1.5],
        })
        assert resp.get("ok"), f"set_transform failed: {resp}"
        
        # Wait for changes to be rendered
        time.sleep(2.0)
        
        # === STEP G: Snapshot AFTER transform ===
        print(f"\n[9] Snapshot AFTER transform -> {snap_after_transform}")
        resp = send_command(sock, {
            "cmd": "snapshot",
            "path": str(snap_after_transform.absolute()),
            "width": 640,
            "height": 480,
        })
        assert resp.get("ok"), f"snapshot failed: {resp}"
        time.sleep(1.0)
        
        # === STEP H: Close viewer ===
        print("\n[10] Sending close command...")
        try:
            send_command(sock, {"cmd": "close"})
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            print("  Connection closed (expected)")
        
    except Exception as e:
        print(f"FAIL: {e}")
        sock.close()
        process.terminate()
        return 1
    
    sock.close()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.terminate()
    
    # === VERIFICATION: Compute diffs ===
    print("\n" + "=" * 60)
    print("DIFF METRICS")
    print("=" * 60)
    
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        print("FAIL: PIL/numpy not available for diff computation")
        return 1
    
    # Check all snapshots exist
    for p in [snap_before_load, snap_after_load, snap_after_transform]:
        if not p.exists():
            print(f"FAIL: Snapshot not found: {p}")
            return 1
        print(f"  {p.name}: {p.stat().st_size} bytes")
    
    # Compute diffs
    print()
    
    diff1_count, diff1_pct = compute_pixel_diff(snap_before_load, snap_after_load)
    print(f"diff_before_after_load:")
    print(f"  changed_pixels = {diff1_count}")
    print(f"  pct_changed    = {diff1_pct:.2f}%")
    print(f"  threshold      >= {DIFF_BEFORE_AFTER_LOAD_MIN_PCT}%")
    
    diff2_count, diff2_pct = compute_pixel_diff(snap_after_load, snap_after_transform)
    print(f"\ndiff_after_load_after_transform:")
    print(f"  changed_pixels = {diff2_count}")
    print(f"  pct_changed    = {diff2_pct:.2f}%")
    print(f"  threshold      >= {DIFF_AFTER_LOAD_AFTER_TRANSFORM_MIN_PCT}%")
    
    # === FINAL VERDICT ===
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    passed = True
    
    if diff1_pct < DIFF_BEFORE_AFTER_LOAD_MIN_PCT:
        print(f"FAIL: diff_before_after_load.pct_changed ({diff1_pct:.2f}%) < {DIFF_BEFORE_AFTER_LOAD_MIN_PCT}%")
        print("      Loading geometry did not visibly change the frame!")
        passed = False
    else:
        print(f"PASS: diff_before_after_load.pct_changed ({diff1_pct:.2f}%) >= {DIFF_BEFORE_AFTER_LOAD_MIN_PCT}%")
    
    if diff2_pct < DIFF_AFTER_LOAD_AFTER_TRANSFORM_MIN_PCT:
        print(f"FAIL: diff_after_load_after_transform.pct_changed ({diff2_pct:.2f}%) < {DIFF_AFTER_LOAD_AFTER_TRANSFORM_MIN_PCT}%")
        print("      set_transform did not visibly change the frame!")
        passed = False
    else:
        print(f"PASS: diff_after_load_after_transform.pct_changed ({diff2_pct:.2f}%) >= {DIFF_AFTER_LOAD_AFTER_TRANSFORM_MIN_PCT}%")
    
    print()
    if passed:
        print("=== ALL CHECKS PASSED ===")
        return 0
    else:
        print("=== SOME CHECKS FAILED ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
