# examples/cartopy_overlay.py
# Demo: Cartopy interop by drawing a forge3d tile mosaic and attribution in a GeoAxes scene.
# Runs headless (Agg) and saves a composite PNG, if cartopy/matplotlib are available.
# RELEVANT FILES:python/forge3d/tiles/client.py,python/forge3d/tiles/overlay.py,docs/integration/cartopy.md

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Cartopy interop demo with forge3d tiles.")
    parser.add_argument("--out", type=str, default="reports/u3_cartopy.png")
    parser.add_argument("--provider", type=str, default="osm")
    parser.add_argument("--zoom", type=int, default=6)
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"), default=(-3.0, 50.0, 2.0, 53.0))
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import cartopy.crs as ccrs  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Cartopy/Matplotlib required for this demo: {e}")

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
    img = client.compose_mosaic(provider, tuple(args.bbox), args.zoom, online=True)
    meta = provider.attribution()
    # Use recommended attribution text
    text = "© OpenStreetMap contributors" if provider.name.lower().startswith("openstreetmap") else f"{meta['name']} ({meta.get('license','')})"
    draw_attribution(img, text, position="br", dpi=96, margin=8)

    # Plot with Cartopy GeoAxes and match extent
    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(args.bbox, crs=ccrs.PlateCarree())
    ax.coastlines(resolution="50m", color="gray")
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # Draw mosaic as an image with the same extent
    ax.imshow(img, origin="upper", extent=args.bbox, transform=ccrs.PlateCarree())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    print(f"Saved cartopy composite: {out_path}")


if __name__ == "__main__":
    main()
