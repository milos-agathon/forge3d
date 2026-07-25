#!/usr/bin/env python3
"""Reproduce the committed F3DZ byte-stream hashes on the current platform."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tomllib
from pathlib import Path

import numpy as np

from forge3d.codec import compress_dem


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "data" / "codec_corpus"
EPSILONS = (0.05, 0.1, 0.5, 1.0)
TILES = ("alpine", "rolling", "coastal_flat_nodata", "canyon")


def build_report() -> dict[str, object]:
    expected = tomllib.loads((CORPUS / "DETERMINISM.toml").read_text(encoding="utf-8"))
    streams: list[dict[str, object]] = []
    for tile in TILES:
        source = np.load(CORPUS / f"{tile}.npy", allow_pickle=False)
        for epsilon in EPSILONS:
            first = compress_dem(source, epsilon, progressive=True)
            second = compress_dem(source, epsilon, progressive=True)
            first_sha = hashlib.sha256(first).hexdigest()
            second_sha = hashlib.sha256(second).hexdigest()
            key = f"eps_{str(epsilon).replace('.', '_')}"
            locked_sha = expected[tile][key]
            if first != second:
                raise RuntimeError(f"same-platform stream mismatch for {tile} eps={epsilon}")
            if first_sha != locked_sha:
                raise RuntimeError(
                    f"cross-platform stream mismatch for {tile} eps={epsilon}: "
                    f"actual={first_sha} locked={locked_sha}"
                )
            streams.append(
                {
                    "tile": tile,
                    "epsilon": epsilon,
                    "bytes": len(first),
                    "first_sha256": first_sha,
                    "second_sha256": second_sha,
                    "locked_sha256": locked_sha,
                }
            )
    return {
        "format": "forge3d-f3dz-determinism/1",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "streams": streams,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
