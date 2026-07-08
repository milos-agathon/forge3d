#!/usr/bin/env python3
# tools/verify_provenance.py
# VERITAS: standalone offline verifier for a forge3d provenance triple
# (image.png, source_map.npy, provenance.json). Runs WITHOUT the compiled
# _forge3d extension: numpy + stdlib hashlib + the pure-Python Ed25519
# fallback are the only requirements.
# RELEVANT FILES: python/forge3d/provenance.py, python/forge3d/_ed25519.py,
# src/core/provenance.rs

"""Re-verify which source dataset produced each rendered pixel.

Usage::

    python tools/verify_provenance.py image.png source_map.npy provenance.json

Checks, in order:

1. reconstructs the per-pixel source map from ``source_map.npy``;
2. recomputes every contributing-tile leaf and the source-map leaf;
3. rebuilds the SHA256 Merkle root and compares it to the signed root;
4. verifies the Ed25519 signature against the embedded public key;
5. reports per-source pixel coverage;
6. runs a single-texel tamper probe (in memory) and confirms the root breaks.

Exit code 0 iff the root matches, the signature verifies, the image/source-map
dimensions agree, and the tamper probe is detected.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path

_REPO_PYTHON = Path(__file__).resolve().parents[1] / "python"
if _REPO_PYTHON.exists() and str(_REPO_PYTHON) not in sys.path:
    sys.path.insert(0, str(_REPO_PYTHON))


def _read_png_dims(path: Path) -> tuple[int, int]:
    """Width/height from the PNG IHDR chunk — no image decoder needed."""
    data = path.read_bytes()
    signature = b"\x89PNG\r\n\x1a\n"
    if not data.startswith(signature) or len(data) < 33:
        raise ValueError(f"{path} is not a PNG file")
    length, chunk_type = struct.unpack(">I4s", data[8:16])
    if chunk_type != b"IHDR" or length < 13:
        raise ValueError(f"{path} has no IHDR chunk")
    ihdr = data[16 : 16 + 13]
    expected_crc = struct.unpack(">I", data[16 + 13 : 16 + 17])[0]
    if zlib.crc32(b"IHDR" + ihdr) & 0xFFFFFFFF != expected_crc:
        raise ValueError(f"{path}: IHDR checksum mismatch")
    width, height = struct.unpack(">2I", ihdr[:8])
    return int(width), int(height)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path, help="rendered image (PNG)")
    parser.add_argument("source_map", type=Path, help="per-pixel source-id map (.npy, uint32)")
    parser.add_argument("manifest", type=Path, help="provenance.json manifest")
    args = parser.parse_args(argv)

    import numpy as np

    from forge3d.bundle import _compute_sha256  # shared bundle-integrity SHA256
    from forge3d.provenance import (
        build_merkle_root,
        manifest_leaf_encodings,
        verify_provenance_offline,
    )

    source_map = np.asarray(np.load(args.source_map), dtype=np.uint32)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = verify_provenance_offline(source_map, manifest)

    image_width, image_height = _read_png_dims(args.image)
    map_height, map_width = source_map.shape
    image_dims_match = (image_width, image_height) == (map_width, map_height)

    print(f"image: {args.image} ({image_width}x{image_height})")
    print(f"image_sha256: {_compute_sha256(args.image)}")
    print(f"source_map: {args.source_map} ({map_width}x{map_height})")
    print(f"image_dims_match: {image_dims_match}")

    total = int(source_map.size)
    attributed = total - report["unattributed_pixels"]
    print(f"pixels_total: {total}")
    print(
        f"pixels_attributed: {attributed} "
        f"({100.0 * attributed / max(total, 1):.2f}%)"
    )
    source_table = {
        int(entry["source_id"]): entry for entry in manifest.get("source_table", ())
    }
    for source_id, count in sorted(report["coverage"].items()):
        entry = source_table.get(source_id)
        family = entry["family"] if entry else "UNKNOWN"
        content = entry["content_hash"][:16] + "…" if entry else "?"
        print(
            f"coverage source_id={source_id} family={family} "
            f"pixels={count} ({100.0 * count / max(total, 1):.2f}%) "
            f"content_hash={content}"
        )
    if report["unknown_source_ids"]:
        print(f"unknown_source_ids: {report['unknown_source_ids']}")

    print(f"merkle_root_match: {report['root_match']}")
    print(f"signature_valid: {report['signature_valid']}")

    # Single-texel tamper probe: flipping one source-map texel must break the
    # recomputed root. SOURCE_ID_NONE never gains attribution — the probe uses
    # an XOR so the sentinel flips to a nonzero id and vice versa.
    tampered = source_map.copy()
    tampered[0, 0] ^= 1
    tampered_root = build_merkle_root(manifest_leaf_encodings(manifest, tampered))
    tamper_detected = tampered_root.hex() != report["signed_root"]
    print(f"tamper_probe_single_texel_detected: {tamper_detected}")

    ok = bool(
        report["root_match"]
        and report["signature_valid"]
        and report["dims_match"]
        and image_dims_match
        and not report["unknown_source_ids"]
        and tamper_detected
    )
    print(f"verified: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
