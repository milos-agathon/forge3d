# docs/tiles/xyz_wmts.md
# Basemap tile fetching, caching, and composition (XYZ/WMTS) in forge3d.
# Documents the Python tiles API, cache layout, and attribution requirements.
# RELEVANT FILES:python/forge3d/tiles/client.py,python/forge3d/tiles/overlay.py,examples/xyz_tile_compose_demo.py

## Overview

This page describes the lightweight Python-side tile client used to fetch and compose XYZ/WMTS basemap tiles into RGBA mosaics.

It also covers on-disk caching, offline mode, and attribution overlay options.


## Installation

Tiles functionality lives in `forge3d.tiles` and uses Pillow for image composition.

Install extras:

```
pip install forge3d[tiles]
```


## API

- `forge3d.tiles.TileProvider(name, url_template, subdomains=None, license=None, homepage=None, tile_size=256)`

- `forge3d.tiles.TileClient(cache_root=None)`

- `TileClient.compose_mosaic(provider, bbox, zoom, online=True) -> PIL.Image`

- `forge3d.tiles.draw_attribution(image, text, logo=None, position="br", dpi=96, margin=8)`


## Cache layout

Tiles are cached under the user cache directory, typically:

`<user cache>/forge3d/tiles/{provider}/{z}/{x}/{y}.png`

Metadata is stored next to the tile as `{y}.json` with ETag/Last-Modified when available.


## Attribution

Providers require attribution and a polite HTTP User-Agent.

Use `draw_attribution` to render text and optional logos in a consistent, readable way.

For OpenStreetMap, use the recommended string: `© OpenStreetMap contributors`.

Set a User-Agent with contact info via environment variable:

```
$env:FORGE3D_TILE_USER_AGENT = "forge3d/0.12 (https://your-site; contact: you@example.org)"
```

The OSM provider example:

```
TileProvider(
  name="OpenStreetMap",
  url_template="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
  license="ODbL",
  homepage="https://www.openstreetmap.org",
)
```
