# examples/xyz_tile_compose_demo.py
# Demo: compose an XYZ tile mosaic for a small bbox and save PNG.
# Useful for validating U1: fetch count, cache behavior, and provider attribution.
# RELEVANT FILES:python/forge3d/tiles/client.py,python/forge3d/tiles/overlay.py,tests/test_tiles_client.py

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose XYZ tile mosaic and save.")
    parser.add_argument("--out", type=str, default="reports/u1_tiles.png")
    parser.add_argument("--provider", type=str, default="osm")
    parser.add_argument("--zoom", type=int, default=12)
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"), default=(-0.2, 51.45, 0.2, 51.7))
    parser.add_argument("--offline", action="store_true", help="Use cache-only mode")
    args = parser.parse_args()

    from forge3d.tiles import TileClient, TileProvider, draw_attribution

    if args.provider.lower() == "osm":
        provider = TileProvider(
            name="OpenStreetMap",
            url_template="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            license="ODbL",
            homepage="https://www.openstreetmap.org",
            tile_size=256,
        )
    else:
        provider = TileProvider(name=args.provider, url_template=args.provider)

    # Provide a polite User-Agent per OSM policy; override via FORGE3D_TILE_USER_AGENT if needed.
    client = TileClient(default_headers={
        "User-Agent": "forge3d/0.12 (https://github.com/example/forge3d; contact: tiles@example.com)",
        "Referer": "https://forge3d.readthedocs.io/",
    })
    img = client.compose_mosaic(provider, tuple(args.bbox), args.zoom, online=not args.offline)

    meta = provider.attribution()
    text = "© OpenStreetMap contributors" if provider.name.lower().startswith("openstreetmap") else f"{meta['name']} ({meta.get('license','')})"
    draw_attribution(img, text, position="br", dpi=96, margin=8)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Pillow is required to save images: {e}")
    img.save(str(out_path))
    print(f"Saved mosaic: {out_path}")


if __name__ == "__main__":
    main()
