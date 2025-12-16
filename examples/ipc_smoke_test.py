#!/usr/bin/env python3
"""IPC smoke test - proof-tight verification that IPC works end-to-end.

This script provides deterministic proof of:
1. Geometry is actually loaded (via get_stats: vb_ready, vertex/index counts)
2. Loading changes rendered output (before/after diff with bbox coverage)
3. set_transform changes rendered output (after load/after transform diff)

Produces 3 snapshots at identical resolution with pixel diff metrics.
Diff metric: pixel is "changed" if max(|ΔR|,|ΔG|,|ΔB|) >= 8
Must meet BOTH pct_changed AND bbox_area_pct thresholds.
"""

from __future__ import annotations

import hashlib
import json
import re
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

# Add the python package to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

READY_PATTERN = re.compile(r"FORGE3D_VIEWER_READY port=(\d+)")

# Hard thresholds (non-negotiable) - must meet BOTH conditions
# Diff threshold: max(|ΔR|,|ΔG|,|ΔB|) >= 8
DIFF_THRESHOLD = 8

# before -> after_load: pct_changed >= 2.0% AND bbox_area_pct >= 10.0%
DIFF_BEFORE_AFTER_LOAD_MIN_PCT = 2.0
DIFF_BEFORE_AFTER_LOAD_MIN_BBOX_PCT = 10.0

# after_load -> after_transform: pct_changed >= 1.0% AND bbox_area_pct >= 5.0%
DIFF_AFTER_LOAD_AFTER_TRANSFORM_MIN_PCT = 1.0
DIFF_AFTER_LOAD_AFTER_TRANSFORM_MIN_BBOX_PCT = 5.0

# Fixed snapshot resolution (all 3 must be identical)
SNAPSHOT_WIDTH = 640
SNAPSHOT_HEIGHT = 480


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


def get_png_dimensions(path: Path) -> Tuple[int, int]:
    """Read PNG dimensions from file header without loading full image."""
    with open(path, "rb") as f:
        # PNG signature (8 bytes) + IHDR length (4 bytes) + IHDR type (4 bytes)
        f.read(16)
        # Width and height are 4 bytes each, big-endian
        width = struct.unpack(">I", f.read(4))[0]
        height = struct.unpack(">I", f.read(4))[0]
    return width, height


def compute_md5(path: Path) -> str:
    """Compute MD5 hash of file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


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


def compute_pixel_diff(img1_path: Path, img2_path: Path) -> Tuple[int, float, float]:
    """Compute pixel diff between two images.
    
    Images MUST have identical dimensions (no resizing allowed).
    A pixel is "changed" if max(|ΔR|,|ΔG|,|ΔB|) >= DIFF_THRESHOLD (8).
    
    Returns:
        (changed_pixel_count, pct_changed, bbox_area_pct)
        
    Raises:
        ValueError: if images have different dimensions
    """
    from PIL import Image
    import numpy as np
    
    pil_img1 = Image.open(img1_path).convert("RGB")
    pil_img2 = Image.open(img2_path).convert("RGB")
    
    # HARD REQUIREMENT: images must have identical dimensions
    if pil_img1.size != pil_img2.size:
        raise ValueError(
            f"Image dimensions must match! "
            f"{img1_path.name}: {pil_img1.size}, {img2_path.name}: {pil_img2.size}"
        )
    
    img1 = np.array(pil_img1)
    img2 = np.array(pil_img2)
    
    # A pixel is "changed" if max(|ΔR|,|ΔG|,|ΔB|) >= DIFF_THRESHOLD
    diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
    max_diff = np.max(diff, axis=2)  # Max diff across RGB channels
    changed = max_diff >= DIFF_THRESHOLD
    
    changed_count = int(np.sum(changed))
    total_pixels = img1.shape[0] * img1.shape[1]
    pct_changed = 100.0 * changed_count / total_pixels
    
    # Compute bounding box of changed pixels
    if changed_count > 0:
        rows = np.any(changed, axis=1)
        cols = np.any(changed, axis=0)
        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        min_row, max_row = row_indices[0], row_indices[-1]
        min_col, max_col = col_indices[0], col_indices[-1]
        bbox_height = max_row - min_row + 1
        bbox_width = max_col - min_col + 1
        bbox_area = bbox_height * bbox_width
        bbox_area_pct = 100.0 * bbox_area / total_pixels
    else:
        bbox_area_pct = 0.0
    
    return changed_count, pct_changed, bbox_area_pct


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


def wait_for_transform_version(sock: socket.socket, min_version: int, timeout: float = 5.0) -> dict:
    """Poll get_stats until transform_version >= min_version or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        resp = send_command(sock, {"cmd": "get_stats"})
        if not resp.get("ok"):
            raise RuntimeError(f"get_stats failed: {resp}")
        stats = resp.get("stats", {})
        current_version = stats.get("transform_version", 0)
        if current_version >= min_version:
            return stats
        time.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for transform_version >= {min_version}")


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
    
    # Delete stale files to ensure deterministic output
    for p in [snap_before_load, snap_after_load, snap_after_transform]:
        if p.exists():
            p.unlink()
            print(f"  Deleted stale: {p.name}")
    
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
        print(f"  stats_before: {json.dumps(stats_before)}")
        
        # Verify blank state
        if stats_before.get("vb_ready"):
            print("  WARNING: vb_ready=True before load (may have initial geometry)")
        if stats_before.get("vertex_count", 0) > 0:
            print("  WARNING: vertex_count > 0 before load")
        
        # Set camera to fixed position (for tiny bunny model)
        resp = send_command(sock, {
            "cmd": "cam_lookat",
            "eye": [0.0, 0.1, 0.3],
            "target": [0.0, 0.1, 0.0],
            "up": [0.0, 1.0, 0.0],
        })
        assert resp.get("ok"), f"cam_lookat failed: {resp}"
        
        resp = send_command(sock, {"cmd": "set_fov", "deg": 60.0})
        assert resp.get("ok"), f"set_fov failed: {resp}"
        
        time.sleep(0.5)
        
        # === STEP B: Snapshot BEFORE load (no geometry - tests fallback path) ===
        print(f"\n[4] Snapshot BEFORE load -> {snap_before_load}")
        print(f"    Resolution: {SNAPSHOT_WIDTH}x{SNAPSHOT_HEIGHT}")
        resp = send_command(sock, {
            "cmd": "snapshot",
            "path": str(snap_before_load.absolute()),
            "width": SNAPSHOT_WIDTH,
            "height": SNAPSHOT_HEIGHT,
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
        print(f"  stats_after: {json.dumps(stats_after)}")
        
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
        
        # === STEP E: Snapshot AFTER load (geometry visible) ===
        print(f"\n[7] Snapshot AFTER load -> {snap_after_load}")
        print(f"    Resolution: {SNAPSHOT_WIDTH}x{SNAPSHOT_HEIGHT}")
        resp = send_command(sock, {
            "cmd": "snapshot",
            "path": str(snap_after_load.absolute()),
            "width": SNAPSHOT_WIDTH,
            "height": SNAPSHOT_HEIGHT,
        })
        assert resp.get("ok"), f"snapshot failed: {resp}"
        time.sleep(1.0)
        
        # === STEP F: Test set_transform (CAMERA LOCKED - must change output via object transform) ===
        print("\n[8] Testing set_transform (camera LOCKED)...")
        import math
        angle = math.radians(90)
        rotation_quat = [0.0, math.sin(angle/2), 0.0, math.cos(angle/2)]
        resp = send_command(sock, {
            "cmd": "set_transform",
            "translation": [0.1, 0.0, 0.0],
            "rotation_quat": rotation_quat,
            "scale": [2.0, 2.0, 2.0],
        })
        # set_transform must return ok:true
        if not resp.get("ok"):
            print(f"  FAIL: set_transform returned ok=false")
            print(f"  Error: {resp.get('error', 'no error message')}")
            raise RuntimeError("set_transform must return ok:true")
        print(f"  OK: set_transform returned ok=true")
        print(f"  Transform: translation=[0.1, 0.0, 0.0], rotation=90° Y-axis, scale=[2.0, 2.0, 2.0]")
        
        # CAMERA IS LOCKED - no camera movement after set_transform
        # The diff between after_load and after_transform must be due to object transform alone
        
        # D2: Wait for transform_version to advance (deterministic ack)
        print("\n[8b] Waiting for transform_version to advance...")
        stats_after_transform = wait_for_transform_version(sock, min_version=1, timeout=5.0)
        print(f"  stats_after_transform: {json.dumps(stats_after_transform)}")
        
        # Verify transform was applied
        if stats_after_transform.get("transform_version", 0) < 1:
            print("  FAIL: transform_version did not advance")
            raise RuntimeError("transform_version must advance after set_transform")
        if stats_after_transform.get("transform_is_identity", True):
            print("  FAIL: transform_is_identity should be False after non-identity transform")
            raise RuntimeError("transform_is_identity must be False")
        print(f"  OK: transform_version={stats_after_transform.get('transform_version')}, "
              f"transform_is_identity={stats_after_transform.get('transform_is_identity')}")
        
        # Additional wait for GPU to finish rendering with transformed geometry
        time.sleep(0.5)
        
        # === STEP G: Snapshot AFTER transform ===
        print(f"\n[9] Snapshot AFTER transform -> {snap_after_transform}")
        print(f"    Resolution: {SNAPSHOT_WIDTH}x{SNAPSHOT_HEIGHT}")
        resp = send_command(sock, {
            "cmd": "snapshot",
            "path": str(snap_after_transform.absolute()),
            "width": SNAPSHOT_WIDTH,
            "height": SNAPSHOT_HEIGHT,
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
    print("SNAPSHOT VERIFICATION")
    print("=" * 60)
    
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        print("FAIL: PIL/numpy not available for diff computation")
        return 1
    
    # Check all snapshots exist and have identical dimensions
    snapshot_files = [snap_before_load, snap_after_load, snap_after_transform]
    dimensions = []
    
    print("\nSnapshot listing:")
    for p in snapshot_files:
        if not p.exists():
            print(f"  FAIL: Snapshot not found: {p}")
            return 1
        dims = get_png_dimensions(p)
        md5 = compute_md5(p)
        dimensions.append(dims)
        print(f"  {p.name}: {dims[0]}x{dims[1]}, {p.stat().st_size} bytes, MD5={md5}")
    
    # HARD REQUIREMENT: All snapshots must have identical dimensions
    if not all(d == dimensions[0] for d in dimensions):
        print(f"\nFAIL: Snapshot dimensions do not match!")
        print(f"  Expected all to be {SNAPSHOT_WIDTH}x{SNAPSHOT_HEIGHT}")
        for p, d in zip(snapshot_files, dimensions):
            print(f"  {p.name}: {d[0]}x{d[1]}")
        return 1
    
    if dimensions[0] != (SNAPSHOT_WIDTH, SNAPSHOT_HEIGHT):
        print(f"\nFAIL: Snapshot dimensions {dimensions[0]} != expected {SNAPSHOT_WIDTH}x{SNAPSHOT_HEIGHT}")
        return 1
    
    print(f"\n  OK: All snapshots are {SNAPSHOT_WIDTH}x{SNAPSHOT_HEIGHT}")
    
    print("\n" + "=" * 60)
    print("DIFF METRICS")
    print("=" * 60)
    
    # Compute diffs
    print()
    print(f"Diff threshold: max(|ΔR|,|ΔG|,|ΔB|) >= {DIFF_THRESHOLD}")
    print()
    
    diff1_count, diff1_pct, diff1_bbox = compute_pixel_diff(snap_before_load, snap_after_load)
    print(f"diff_before_after_load:")
    print(f"  changed_pixels = {diff1_count}")
    print(f"  pct_changed    = {diff1_pct:.2f}%  (threshold >= {DIFF_BEFORE_AFTER_LOAD_MIN_PCT}%)")
    print(f"  bbox_area_pct  = {diff1_bbox:.2f}%  (threshold >= {DIFF_BEFORE_AFTER_LOAD_MIN_BBOX_PCT}%)")
    
    diff2_count, diff2_pct, diff2_bbox = compute_pixel_diff(snap_after_load, snap_after_transform)
    print(f"\ndiff_after_load_after_transform:")
    print(f"  changed_pixels = {diff2_count}")
    print(f"  pct_changed    = {diff2_pct:.2f}%  (threshold >= {DIFF_AFTER_LOAD_AFTER_TRANSFORM_MIN_PCT}%)")
    print(f"  bbox_area_pct  = {diff2_bbox:.2f}%  (threshold >= {DIFF_AFTER_LOAD_AFTER_TRANSFORM_MIN_BBOX_PCT}%)")
    
    # === FINAL VERDICT ===
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    passed = True
    
    # Check before -> after_load: MUST meet BOTH thresholds
    print("\n[before -> after_load]")
    pct_ok1 = diff1_pct >= DIFF_BEFORE_AFTER_LOAD_MIN_PCT
    bbox_ok1 = diff1_bbox >= DIFF_BEFORE_AFTER_LOAD_MIN_BBOX_PCT
    
    if pct_ok1:
        print(f"  PASS: pct_changed ({diff1_pct:.2f}%) >= {DIFF_BEFORE_AFTER_LOAD_MIN_PCT}%")
    else:
        print(f"  FAIL: pct_changed ({diff1_pct:.2f}%) < {DIFF_BEFORE_AFTER_LOAD_MIN_PCT}%")
        passed = False
    
    if bbox_ok1:
        print(f"  PASS: bbox_area_pct ({diff1_bbox:.2f}%) >= {DIFF_BEFORE_AFTER_LOAD_MIN_BBOX_PCT}%")
    else:
        print(f"  FAIL: bbox_area_pct ({diff1_bbox:.2f}%) < {DIFF_BEFORE_AFTER_LOAD_MIN_BBOX_PCT}%")
        print("        (Prevents tiny-corner artifact passes)")
        passed = False
    
    # Check after_load -> after_transform: MUST meet BOTH thresholds
    print("\n[after_load -> after_transform]")
    pct_ok2 = diff2_pct >= DIFF_AFTER_LOAD_AFTER_TRANSFORM_MIN_PCT
    bbox_ok2 = diff2_bbox >= DIFF_AFTER_LOAD_AFTER_TRANSFORM_MIN_BBOX_PCT
    
    if pct_ok2:
        print(f"  PASS: pct_changed ({diff2_pct:.2f}%) >= {DIFF_AFTER_LOAD_AFTER_TRANSFORM_MIN_PCT}%")
    else:
        print(f"  FAIL: pct_changed ({diff2_pct:.2f}%) < {DIFF_AFTER_LOAD_AFTER_TRANSFORM_MIN_PCT}%")
        print("        set_transform did not visibly change the frame!")
        passed = False
    
    if bbox_ok2:
        print(f"  PASS: bbox_area_pct ({diff2_bbox:.2f}%) >= {DIFF_AFTER_LOAD_AFTER_TRANSFORM_MIN_BBOX_PCT}%")
    else:
        print(f"  FAIL: bbox_area_pct ({diff2_bbox:.2f}%) < {DIFF_AFTER_LOAD_AFTER_TRANSFORM_MIN_BBOX_PCT}%")
        print("        (Prevents tiny-corner artifact passes)")
        passed = False
    
    print()
    if passed:
        print("=== ALL CHECKS PASSED ===")
        return 0
    else:
        print("=== SOME CHECKS FAILED ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
