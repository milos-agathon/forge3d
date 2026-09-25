"""Build SIDERA's display-only lunar disc texture from NASA SVS mosaic 5001.

NASA SVS source: https://svs.gsfc.nasa.gov/5001/
The source is a full-Moon nearside mosaic, not a libration or reflectance model.
Regeneration requires Pillow and either network access or ``--source``.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

from PIL import Image


SOURCE_URL = "https://svs.gsfc.nasa.gov/vis/a000000/a005000/a005001/moon_mosaic_print.jpg"
SOURCE_SHA256 = "36fbe604043f1403acbc4db6fbd36d04db1db1c8a3749aa81d1e6ffea072a5fa"
SIZE = 256
SOURCE_LIMB_RADIUS = 0.96
INTERIOR_RADIUS = 0.9
TARGET_INTERIOR_ALBEDO = 0.84
OUTPUT = Path(__file__).resolve().parents[1] / "data/sidera/moon_albedo.bin"


def downsample(source: Image.Image) -> Image.Image:
    """Reduce the pinned mosaic after removing its photographic outer rim."""
    if source.width != source.height:
        raise ValueError(f"NASA lunar mosaic must be square, got {source.size}")
    if not 0.0 < SOURCE_LIMB_RADIUS < 1.0:
        raise ValueError("invalid source limb radius")
    # The source mosaic includes a dark photographic rim outside the useful
    # lunar surface. Crop the inner r<=0.96 square and map it to the runtime
    # unit disc; this keeps markings while avoiding a second dark limb when the
    # shader applies its own antialiased disc coverage.
    margin = round((1.0 - SOURCE_LIMB_RADIUS) * (source.width - 1) / 2.0)
    if margin <= 0 or 2 * margin >= source.width:
        raise ValueError("source limb crop is empty")
    cropped = source.crop((margin, margin, source.width - margin, source.height - margin))
    return cropped.resize((SIZE, SIZE), Image.Resampling.LANCZOS)


def tone_map(image: Image.Image) -> bytes:
    """Compress highlights while keeping the interior median at display albedo."""
    # Grayscale bytes avoid Pillow's version-dependent pixel sequence API.
    pixels = image.tobytes()
    width, height = image.size
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    radius = min(cx, cy) * INTERIOR_RADIUS
    interior = [
        value
        for index, value in enumerate(pixels)
        if ((index % width - cx) ** 2 + (index // width - cy) ** 2) <= radius * radius
    ]
    if not interior:
        raise ValueError("NASA lunar mosaic has no interior pixels")
    interior.sort()
    median = interior[len(interior) // 2] / 255.0
    if not 0.0 < median < TARGET_INTERIOR_ALBEDO < 1.0:
        raise ValueError(f"invalid lunar interior median {median}")

    # A rational shoulder maps black to black, the source white point to one,
    # and the measured interior median to the declared relative albedo. Unlike
    # a scale followed by min(255, ...), bright craters retain ordering instead
    # of collapsing into a large saturated patch.
    shoulder = median * (1.0 - TARGET_INTERIOR_ALBEDO) / (TARGET_INTERIOR_ALBEDO - median)
    encoded = bytearray()
    for value in pixels:
        normalized = value / 255.0
        mapped = 0.0 if normalized == 0.0 else normalized * (1.0 + shoulder) / (normalized + shoulder)
        encoded.append(round(255.0 * min(1.0, mapped)))
    return bytes(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="local copy of the NASA JPEG")
    args = parser.parse_args()
    source = args.source.read_bytes() if args.source else urlopen(SOURCE_URL).read()
    digest = sha256(source).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"NASA lunar mosaic SHA-256 mismatch: {digest}")
    image = downsample(Image.open(BytesIO(source)).convert("L"))
    albedo = tone_map(image)
    if len(albedo) != SIZE * SIZE:
        raise ValueError(f"unexpected lunar texture size {len(albedo)}")
    OUTPUT.write_bytes(albedo)
    print(f"{OUTPUT.name}: {len(albedo)} bytes, SHA-256 {sha256(albedo).hexdigest()}")


if __name__ == "__main__":
    main()
