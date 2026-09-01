#!/usr/bin/env python3
"""Author the Paris-Eiffel class mask from hand-placed reference regions.

The mask is the acceptance oracle for `examples/paris_eiffel_day.py`: it assigns
pixels of the committed reference frame to the class IDs of the design spec, so
per-class colour targets and tolerances can be derived from measurement rather
than asserted.

Why regions rather than a per-pixel classifier
----------------------------------------------
The reference is a stylised render in which several distinct classes share a
colour. Measured on the frame itself, lit roof is (207, 200, 198) and road is
(203, 191, 186) -- a separation of ~12, below the noise of a screen recording.
Lawn and generic landuse are likewise one green family. A colour classifier
would therefore *invent* boundaries it cannot see, and the resulting targets
would encode that invention.

So this script assigns only hand-placed regions that are unambiguous on
inspection, and leaves everything else UNASSIGNED. The metric ignores
unassigned pixels. Abstention is the honest outcome for a class the reference
cannot distinguish; those classes are gated on reachability in the render's own
class raster instead, where provenance is known.

Every region carries an expectation predicate. The predicate subtracts
contaminants inside a region -- a car on the esplanade, a line marking on the
pitch -- returning them to UNASSIGNED rather than averaging them into the
target. It never assigns a class, so roof and road stay separated by hand
placement, not by colour. A region that falls below the retention floor is a
placement error and aborts the build, so a polygon dragged onto the wrong
surface cannot silently corrupt a target.

Usage:
    python tools/paris_eiffel_day/author_class_mask.py [--qc-dir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE = REPO_ROOT / "tests" / "fixtures" / "paris_eiffel_day" / "reference_day.png"
MASK_OUT = REPO_ROOT / "tests" / "fixtures" / "paris_eiffel_day" / "class_mask.png"
TARGETS_OUT = REPO_ROOT / "tests" / "fixtures" / "paris_eiffel_day" / "class_targets.json"

REFERENCE_SHA256 = "CA13E48F90F8C3ACDA40BC086C76E3F1C0CF0D7B456228D7F3D2EA92ED22A067"

UNASSIGNED = 255

# Class IDs from the design spec, section 9.
CLASS_NAMES = {
    0: "background",
    1: "base",
    2: "landuse",
    3: "park",
    4: "water",
    5: "road",
    6: "road_hi",
    7: "footpath",
    8: "building_roof",
    9: "building_wall",
    10: "tower",
    11: "tower_deck",
    12: "tree",
    13: "pitch",
    14: "car",
}

# Classes the reference cannot distinguish, with the reason recorded so the
# abstention is reviewable rather than an unexplained gap.
ABSTAINED = {
    0: "reference is full-bleed ground; no sky is visible in the frame",
    2: "landuse and park are one green family in the reference; no visible boundary",
    6: "road_hi is the same paved colour as road; the reference draws no distinction",
    7: "footpath is the same paved colour as road at 2-3 px width",
    11: "no distinct deck band is visible; the teal in this frame is garden ponds "
    "(class 4), not tower structure -- see notes",
    14: "vehicles are 4-10 px specks at screen-recording compression; a "
    "colour/size detector cannot separate them from tower lattice and codec "
    "artefacts (a first attempt returned 4,007 px spread over the whole frame, "
    "concentrated on the tower, with a muddy violet median). Gated on "
    "reachability in the render's own raster instead",
}


def _predicates():
    """Expectation predicates, one per assigned class.

    Each takes float arrays R, G, B, L (luma) and returns a per-pixel bool.
    Used to subtract contaminants inside a hand-placed region, never to assign
    a class. See the filtering block in main().
    """
    return {
        1: lambda R, G, B, L: (L > 165) & (np.abs(B - R) < 25),
        3: lambda R, G, B, L: (G >= np.maximum(R, B) - 4) & (L > 130) & (L < 205),
        4: lambda R, G, B, L: (B - R) > 40,
        5: lambda R, G, B, L: (L > 165) & (np.abs(B - R) < 25),
        8: lambda R, G, B, L: (L > 180) & ((B - R) < 22),
        9: lambda R, G, B, L: ((B - R) > 15) & (L > 110) & (L < 200),
        10: lambda R, G, B, L: ((R - G) > 60) & (R > 170),
        12: lambda R, G, B, L: (G > R) & (G > B) & (L < 155),
        13: lambda R, G, B, L: (G - B) > 45,
    }


# Hand-placed regions, image coordinates on the 960x720 reference.
# Polygons are kept well inside their surface so anti-aliased edges and the
# recording's chroma subsampling do not bleed a neighbouring class in.
REGIONS: dict[int, list[list[tuple[int, int]]]] = {
    # Seine, wide reaches only. NOTE: the garden ponds are also class 4 but are
    # a much lighter tint; see the bimodality note emitted by this script.
    4: [
        [(35, 470), (150, 405), (215, 452), (95, 528)],
        [(190, 300), (300, 250), (330, 288), (215, 340)],
        [(600, 70), (690, 40), (705, 72), (615, 100)],
    ],
    # Paved esplanade / quay. Same material family as road; kept separate only
    # because the spec declares separate classes.
    1: [
        [(300, 330), (360, 300), (378, 330), (318, 360)],
        [(120, 560), (180, 528), (196, 556), (136, 588)],
    ],
    5: [
        [(250, 250), (290, 232), (300, 250), (260, 268)],
        [(392, 300), (430, 284), (438, 300), (400, 316)],
    ],
    # Building roofs: warm-white lit tops. Roof and paved ground are the same
    # colour in this reference, so these are placed only inside blocks
    # visually confirmed to be buildings -- never by colour search alone.
    8: [
        [(736, 150), (746, 150), (746, 160), (736, 160)],
        [(868, 150), (878, 150), (878, 160), (868, 160)],
        [(832, 204), (842, 204), (842, 214), (832, 214)],
        [(700, 231), (710, 231), (710, 241), (700, 241)],
        [(344, 517), (354, 517), (354, 527), (344, 527)],
        [(198, 9), (208, 9), (208, 19), (198, 19)],
        [(240, 15), (250, 15), (250, 25), (240, 25)],
        [(156, 18), (166, 18), (166, 28), (156, 28)],
    ],
    # Building walls: violet shadow faces. Placed on verified-pure blocks.
    9: [
        [(624, 0), (640, 0), (640, 16), (624, 16)],
        [(496, 688), (512, 688), (512, 704), (496, 704)],
        [(320, 48), (336, 48), (336, 64), (320, 64)],
        [(64, 368), (80, 368), (80, 384), (64, 384)],
        [(848, 128), (864, 128), (864, 144), (848, 144)],
        [(448, 560), (464, 560), (464, 576), (448, 576)],
    ],
    # Tower: SUNLIT faces only. The tower is strongly bimodal -- lit faces
    # around (215, 135, 125) against shaded ones far darker -- so a single
    # region spanning both yields an IQR of 65, a "tolerance" loose enough to
    # accept almost any salmon. Measuring the lit faces gives a target that can
    # actually fail. Shaded tower is left unassigned and does not gate.
    # Lattice openings are excluded: background shows through them.
    10: [
        [(504, 144), (510, 144), (510, 150), (504, 150)],
        [(504, 291), (510, 291), (510, 297), (504, 297)],
        [(483, 321), (489, 321), (489, 327), (483, 327)],
        [(456, 354), (462, 354), (462, 360), (456, 360)],
        [(483, 378), (489, 378), (489, 384), (483, 384)],
        [(513, 390), (519, 390), (519, 396), (513, 396)],
    ],
    # Tree canopy. Individual trees are only a few pixels across, so these are
    # small blocks on verified-pure canopy rather than large polygons.
    12: [
        [(656, 384), (672, 384), (672, 400), (656, 400)],
        [(184, 56), (192, 56), (192, 64), (184, 64)],
        [(244, 80), (252, 80), (252, 88), (244, 88)],
        [(28, 188), (36, 188), (36, 196), (28, 196)],
        [(80, 316), (88, 316), (88, 324), (80, 324)],
        [(620, 372), (628, 372), (628, 380), (620, 380)],
        [(616, 440), (624, 440), (624, 448), (616, 448)],
        [(376, 496), (384, 496), (384, 504), (376, 504)],
        [(244, 544), (252, 544), (252, 552), (244, 552)],
    ],
    # Lawn / park sward.
    3: [
        [(600, 508), (660, 508), (660, 540), (600, 540)],
        [(120, 120), (176, 120), (176, 152), (120, 152)],
    ],
    # Football pitch.
    13: [
        [(228, 578), (300, 578), (300, 634), (228, 634)],
    ],
}


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _saturation(rgb: np.ndarray) -> np.ndarray:
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    return np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)


def _rasterize(regions: dict[int, list[list[tuple[int, int]]]], size) -> np.ndarray:
    mask = Image.new("L", size, UNASSIGNED)
    draw = ImageDraw.Draw(mask)
    for class_id, polys in sorted(regions.items()):
        for poly in polys:
            draw.polygon(poly, fill=class_id)
    return np.array(mask, dtype=np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qc-dir", type=Path, default=None)
    args = parser.parse_args()

    raw = REFERENCE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest().upper()
    if digest != REFERENCE_SHA256:
        raise SystemExit(
            f"reference frame hash mismatch\n  expected {REFERENCE_SHA256}\n  found    {digest}"
        )

    image = Image.open(REFERENCE).convert("RGB")
    rgb = np.asarray(image, dtype=np.float32)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    L = _luma(rgb)

    mask = _rasterize(REGIONS, image.size)

    # --- Filter each region by its expectation ------------------------------
    #
    # A hand-placed region declares *where* a class is -- which is the part
    # colour cannot decide, since roof and road share a colour. The predicate
    # then removes contaminants *within* that region: a car parked on the
    # esplanade, a line marking on the pitch, a shadow across a roof. Those
    # pixels belong to other classes, so they are returned to UNASSIGNED rather
    # than averaged into this class's target.
    #
    # This is not a colour classifier: the predicate never assigns a class, it
    # only subtracts. A region that survives below the retention floor is a
    # placement error, because a correctly placed region is mostly its own
    # class.
    RETENTION_FLOOR = 0.60
    predicates = _predicates()
    failures = []
    retention = {}
    for class_id, predicate in predicates.items():
        sel = mask == class_id
        if not sel.any():
            failures.append(f"class {class_id} ({CLASS_NAMES[class_id]}): no pixels assigned")
            continue
        ys, xs = np.nonzero(sel)
        ok = predicate(R[sel], G[sel], B[sel], L[sel])
        frac = float(ok.mean())
        retention[class_id] = (frac, int(sel.sum()), int(ok.sum()))
        if frac < RETENTION_FLOOR:
            median = tuple(np.median(rgb[sel], axis=0).round().astype(int))
            failures.append(
                f"class {class_id} ({CLASS_NAMES[class_id]}): only {frac:.1%} of "
                f"{int(sel.sum())} px satisfy the expectation; median {median} "
                "-- region is probably on the wrong surface"
            )
            continue
        mask[ys[~ok], xs[~ok]] = UNASSIGNED
    if failures:
        raise SystemExit("region placement check FAILED:\n  " + "\n  ".join(failures))

    # --- Derive targets and tolerances -------------------------------------
    targets = {}
    for class_id in sorted(CLASS_NAMES):
        name = CLASS_NAMES[class_id]
        sel = mask == class_id
        count = int(sel.sum())
        if count == 0:
            targets[name] = {
                "class_id": class_id,
                "assigned_px": 0,
                "status": "abstained",
                "reason": ABSTAINED.get(class_id, "no region placed"),
            }
            continue
        px = rgb[sel]
        median = np.median(px, axis=0)
        q1 = np.percentile(px, 25, axis=0)
        q3 = np.percentile(px, 75, axis=0)
        iqr = float(np.max(q3 - q1))
        targets[name] = {
            "class_id": class_id,
            "assigned_px": count,
            "status": "measured",
            "target_rgb": [round(float(v), 1) for v in median],
            "iqr_max": round(iqr, 1),
            # Spec section 12.2: tolerance = max(6, IQR), so a class with
            # genuinely variable colour is not held tighter than its own spread.
            "tolerance": round(max(6.0, iqr), 1),
        }

    assigned_frac = float((mask != UNASSIGNED).mean())
    payload = {
        "schema_version": 1,
        "reference": str(REFERENCE.relative_to(REPO_ROOT)).replace("\\", "/"),
        "reference_sha256": REFERENCE_SHA256,
        "assigned_fraction": round(assigned_frac, 4),
        "classes": targets,
    }

    Image.fromarray(mask, mode="L").save(MASK_OUT)
    TARGETS_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    mask_digest = hashlib.sha256(MASK_OUT.read_bytes()).hexdigest().upper()

    print(f"[mask] assigned {assigned_frac:.2%} of pixels")
    for name, entry in sorted(targets.items(), key=lambda kv: kv[1]["class_id"]):
        if entry["status"] == "measured":
            rgb_txt = ", ".join(f"{v:6.1f}" for v in entry["target_rgb"])
            kept = retention.get(entry["class_id"])
            keep_txt = f"  kept {kept[0]:.0%}" if kept else ""
            print(
                f"  {entry['class_id']:>2} {name:<14} {entry['assigned_px']:>7,} px "
                f"({rgb_txt})  tol {entry['tolerance']:.1f}{keep_txt}"
            )
        else:
            print(f"  {entry['class_id']:>2} {name:<14} {'abstained':>10}  {entry['reason']}")
    print(f"[mask] wrote {MASK_OUT}")
    print(f"[mask] sha256 {mask_digest}")
    print(f"[mask] wrote {TARGETS_OUT}")

    if args.qc_dir:
        args.qc_dir.mkdir(parents=True, exist_ok=True)
        palette = {
            1: (255, 170, 0), 3: (120, 255, 120), 4: (0, 120, 255), 5: (255, 90, 0),
            8: (255, 255, 0), 9: (255, 0, 255), 10: (255, 0, 0), 12: (0, 160, 0),
            13: (0, 255, 200), 14: (255, 255, 255),
        }
        overlay = rgb.copy()
        for class_id, colour in palette.items():
            sel = mask == class_id
            overlay[sel] = 0.45 * overlay[sel] + 0.55 * np.asarray(colour, dtype=np.float32)
        Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(
            args.qc_dir / "mask_overlay.png"
        )
        print(f"[mask] wrote {args.qc_dir / 'mask_overlay.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
