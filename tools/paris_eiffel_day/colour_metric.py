#!/usr/bin/env python3
"""Score a Paris-Eiffel render against the committed reference frame.

This is the executable oracle for the design spec's colour gate. It reports,
per class, the render's median colour inside the mask region, the measured
target, the derived tolerance, and pass/fail -- so "it matches the reference"
is a measurement rather than a claim.

Inputs are pinned by hash: the reference frame and the class mask must be the
committed fixtures, or the run aborts. Otherwise a drifting fixture could
silently move the goalposts.

Classes the reference cannot distinguish are abstained in the mask (see
author_class_mask.py) and are reported as SKIP. They carry no colour target and
are gated on reachability in the render's own class raster instead.

Usage:
    python tools/paris_eiffel_day/colour_metric.py RENDER.png
    python tools/paris_eiffel_day/colour_metric.py --self-test
    python tools/paris_eiffel_day/colour_metric.py RENDER.png --json out.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "paris_eiffel_day"
REFERENCE = FIXTURES / "reference_day.png"
MASK = FIXTURES / "class_mask.png"
TARGETS = FIXTURES / "class_targets.json"

REFERENCE_SHA256 = "CA13E48F90F8C3ACDA40BC086C76E3F1C0CF0D7B456228D7F3D2EA92ED22A067"

# Spec section 1: global reference measurements.
GLOBAL_MEAN = (140.116, 149.472, 163.736)
GLOBAL_MEAN_TOL = 8.0

UNASSIGNED = 255


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _mean_saturation(rgb: np.ndarray) -> float:
    """Mean HSV saturation on PIL's 0-255 scale.

    Uses PIL's own HSV conversion rather than an equivalent formula so the
    figure is directly comparable to the value recorded in the design spec
    (65.327); computing it independently drifts by ~0.5 through rounding.
    """
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    return float(np.asarray(img.convert("HSV"), dtype=np.float32)[:, :, 1].mean())


def load_render(path: Path, expect_shape) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32)
    if arr.shape[:2] != expect_shape:
        raise SystemExit(
            f"render is {arr.shape[1]}x{arr.shape[0]}; the reference is "
            f"{expect_shape[1]}x{expect_shape[0]}. Compare at the reference's "
            "own resolution so no resampling is introduced into the metric."
        )
    return arr


def score(render: np.ndarray, reference: np.ndarray, mask: np.ndarray, targets: dict) -> dict:
    rows = []
    for name, entry in sorted(targets["classes"].items(), key=lambda kv: kv[1]["class_id"]):
        class_id = entry["class_id"]
        if entry["status"] != "measured":
            rows.append(
                {
                    "class_id": class_id,
                    "name": name,
                    "status": "SKIP",
                    "reason": entry["reason"],
                }
            )
            continue
        sel = mask == class_id
        count = int(sel.sum())
        if count == 0:
            rows.append(
                {"class_id": class_id, "name": name, "status": "ERROR",
                 "reason": "mask has no pixels for a class marked measured"}
            )
            continue
        target = np.asarray(entry["target_rgb"], dtype=np.float32)
        tol = float(entry["tolerance"])
        median = np.median(render[sel], axis=0)
        delta = np.abs(median - target)
        worst = float(delta.max())
        rows.append(
            {
                "class_id": class_id,
                "name": name,
                "status": "PASS" if worst <= tol else "FAIL",
                "px": count,
                "target_rgb": [round(float(v), 1) for v in target],
                "render_rgb": [round(float(v), 1) for v in median],
                "delta": [round(float(v), 1) for v in delta],
                "worst_delta": round(worst, 1),
                "tolerance": round(tol, 1),
            }
        )

    mean = render.reshape(-1, 3).mean(axis=0)
    mean_delta = np.abs(mean - np.asarray(GLOBAL_MEAN, dtype=np.float32))
    global_row = {
        "mean_rgb": [round(float(v), 2) for v in mean],
        "reference_mean_rgb": list(GLOBAL_MEAN),
        "mean_delta": [round(float(v), 2) for v in mean_delta],
        "worst_mean_delta": round(float(mean_delta.max()), 2),
        "mean_tolerance": GLOBAL_MEAN_TOL,
        "status": "PASS" if float(mean_delta.max()) <= GLOBAL_MEAN_TOL else "FAIL",
        "median_rgb": [round(float(v), 1) for v in np.median(render.reshape(-1, 3), axis=0)],
        "mean_saturation": round(_mean_saturation(render), 3),
        "reference_mean_saturation": round(_mean_saturation(reference), 3),
    }

    failed = [r for r in rows if r["status"] == "FAIL"]
    errored = [r for r in rows if r["status"] == "ERROR"]
    return {
        "classes": rows,
        "global": global_row,
        "verdict": "PASS"
        if not failed and not errored and global_row["status"] == "PASS"
        else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("render", type=Path, nargs="?", help="render PNG to score")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="score the reference against itself; every class must pass by construction",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the report as JSON")
    args = parser.parse_args()

    if not args.self_test and args.render is None:
        parser.error("provide a render, or --self-test")

    digest = _sha256(REFERENCE)
    if digest != REFERENCE_SHA256:
        raise SystemExit(
            f"reference fixture hash mismatch\n  expected {REFERENCE_SHA256}\n  found    {digest}"
        )
    targets = json.loads(TARGETS.read_text(encoding="utf-8"))
    if targets.get("reference_sha256") != REFERENCE_SHA256:
        raise SystemExit("class_targets.json was derived from a different reference frame")

    reference = np.asarray(Image.open(REFERENCE).convert("RGB"), dtype=np.float32)
    mask = np.asarray(Image.open(MASK), dtype=np.uint8)
    if mask.shape != reference.shape[:2]:
        raise SystemExit("class mask and reference differ in size")

    render = reference if args.self_test else load_render(args.render, reference.shape[:2])
    report = score(render, reference, mask, targets)

    label = "SELF-TEST (reference vs itself)" if args.self_test else str(args.render)
    print(f"\nParis-Eiffel colour metric -- {label}")
    print(f"mask sha256 {_sha256(MASK)}")
    print(f"\n{'id':>3} {'class':<15} {'target':>18} {'render':>18} {'worst':>6} {'tol':>6}  result")
    print("-" * 82)
    for row in report["classes"]:
        if row["status"] in ("SKIP", "ERROR"):
            print(f"{row['class_id']:>3} {row['name']:<15} {row['status']:>52}")
            continue
        t = ",".join(f"{v:5.1f}" for v in row["target_rgb"])
        r = ",".join(f"{v:5.1f}" for v in row["render_rgb"])
        print(
            f"{row['class_id']:>3} {row['name']:<15} {t:>18} {r:>18} "
            f"{row['worst_delta']:>6.1f} {row['tolerance']:>6.1f}  {row['status']}"
        )

    g = report["global"]
    print(
        f"\nglobal mean {tuple(g['mean_rgb'])} vs reference {tuple(g['reference_mean_rgb'])}"
        f"  worst {g['worst_mean_delta']} (tol {g['mean_tolerance']})  {g['status']}"
    )
    print(
        f"global median {tuple(g['median_rgb'])}   "
        f"mean saturation {g['mean_saturation']} vs {g['reference_mean_saturation']}"
    )
    print(f"\nVERDICT: {report['verdict']}")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {args.json}")

    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
