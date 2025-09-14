# docs/integration/cartopy.md
# Cartopy integration example using forge3d tiles as a background.
# Shows headless (Agg) rendering and extent alignment with GeoAxes.
# RELEVANT FILES:examples/cartopy_overlay.py,python/forge3d/tiles/client.py,python/forge3d/tiles/overlay.py

## Overview

This guide demonstrates how to use `forge3d.tiles` to compose a basemap mosaic and plot it as a background image in a Cartopy `GeoAxes`.

It runs headless with the Agg backend and saves a PNG.


## Requirements

```
pip install forge3d[cartopy]
```


## Example

Run the example:

```
python examples/cartopy_overlay.py --out reports/u3_cartopy.png --provider osm
```

This fetches tiles for a small UK extent, draws an attribution overlay, and saves the result.

