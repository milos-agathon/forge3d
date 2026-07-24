#!/usr/bin/env python3
"""Reproduce the committed F3DZ real-terrain corpus from public HGT tiles."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CorpusTile:
    name: str
    source: str
    sha256: str
    row: int
    column: int
    water_is_nodata: bool = False


TILES = (
    CorpusTile(
        "alpine",
        "N46W122.hgt.gz",
        "55866d108c7b55b911c226e457d727d5558eb418ba2328920cf6ec8149f45bbf",
        256,
        768,
    ),
    CorpusTile(
        "rolling",
        "N38W079.hgt.gz",
        "0aa4afc1985773cd9f9bc9197a9b8cd784cf420242924279cc0146de4103a33a",
        1536,
        2304,
    ),
    CorpusTile(
        "coastal_flat_nodata",
        "N25W081.hgt.gz",
        "903135e276b27ad8409249a555d411080b2932779682975234eb47b1e09b61b4",
        0,
        2304,
        water_is_nodata=True,
    ),
    CorpusTile(
        "canyon",
        "N37W113.hgt.gz",
        "73b01c2a33e4a2ae66d2e817445a00c74e8a505713f5f5e71c7de4836b8767aa",
        2048,
        768,
    ),
)

HGT_SIDE = 3601
CROP_SIDE = 256
VOID = -32768
CANONICAL_NAN = np.array([0x7FC00000], dtype="<u4").view("<f4")[0]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _npy_v1_bytes(array: np.ndarray) -> bytes:
    """Write a byte-stable NumPy v1 file without NumPy-version header drift."""
    array = np.ascontiguousarray(array, dtype="<f4")
    header = (
        "{'descr': '<f4', 'fortran_order': False, "
        f"'shape': ({array.shape[0]}, {array.shape[1]}), }}"
    )
    prefix_len = 10
    padding = (-((prefix_len + len(header) + 1) % 64)) % 64
    encoded_header = (header + (" " * padding) + "\n").encode("latin1")
    if len(encoded_header) > 0xFFFF:
        raise ValueError("NumPy v1 header exceeds u16")
    return (
        b"\x93NUMPY"
        + bytes((1, 0))
        + struct.pack("<H", len(encoded_header))
        + encoded_header
        + array.tobytes(order="C")
    )


def generate(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for tile in TILES:
        source_path = source_dir / tile.source
        compressed = source_path.read_bytes()
        actual_sha = _sha256(compressed)
        if actual_sha != tile.sha256:
            raise ValueError(
                f"{source_path}: expected SHA-256 {tile.sha256}, got {actual_sha}"
            )
        with gzip.GzipFile(fileobj=source_path.open("rb")) as stream:
            raw = stream.read()
        expected_bytes = HGT_SIDE * HGT_SIDE * 2
        if len(raw) != expected_bytes:
            raise ValueError(
                f"{source_path}: expected {expected_bytes} HGT bytes, got {len(raw)}"
            )
        heights = np.frombuffer(raw, dtype=">i2").reshape(HGT_SIDE, HGT_SIDE)
        crop = heights[
            tile.row : tile.row + CROP_SIDE,
            tile.column : tile.column + CROP_SIDE,
        ]
        if crop.shape != (CROP_SIDE, CROP_SIDE):
            raise ValueError(f"{source_path}: crop is out of bounds")
        if np.any(crop == VOID):
            raise ValueError(f"{source_path}: source crop contains an HGT void")
        output = crop.astype("<f4")
        if tile.water_is_nodata:
            # The real source uses elevation <= 0 for sea. This corpus models
            # the terrain consumer's water mask as explicit canonical NaN
            # nodata while preserving every positive land sample verbatim.
            output[output <= 0.0] = CANONICAL_NAN
        npy = _npy_v1_bytes(output)
        destination = output_dir / f"{tile.name}.npy"
        destination.write_bytes(npy)
        finite = output[np.isfinite(output)]
        print(
            f"{destination.name}: sha256={_sha256(npy)} "
            f"min={finite.min():.1f} max={finite.max():.1f} "
            f"nodata={np.count_nonzero(np.isnan(output))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="directory containing the four downloaded .hgt.gz source tiles",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/data/codec_corpus"),
    )
    args = parser.parse_args()
    generate(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
