Vector Overlays (Option B)
==========================

Vector overlays render GPU geometry (points, lines, polygons) in world space,
optionally draped onto the terrain heightfield, with proper lighting and shadow
integration.

Overview
--------

Vector overlays are ideal for:

- Annotation markers and labels
- Contour lines
- Route paths and trails
- Polygon boundaries (parcels, zones)
- Point features (POIs, markers)

Quick Start
-----------

Python API
~~~~~~~~~~

.. code-block:: python

    from forge3d.terrain_params import (
        VectorOverlayConfig,
        VectorVertex,
        PrimitiveType,
    )
    
    config = VectorOverlayConfig(
        name="marker",
        vertices=[
            VectorVertex(100, 0, 100, r=1, g=0, b=0),
            VectorVertex(200, 0, 100, r=0, g=1, b=0),
            VectorVertex(150, 0, 200, r=0, g=0, b=1),
        ],
        indices=[0, 1, 2],
        primitive=PrimitiveType.TRIANGLES,
        drape=True,
        drape_offset=1.0,
    )

IPC Commands
~~~~~~~~~~~~

.. code-block:: json

    {
        "cmd": "add_vector_overlay",
        "name": "marker",
        "primitive": "triangles",
        "vertices": [[100, 0, 100, 1, 0, 0, 1], ...],
        "indices": [0, 1, 2],
        "drape": true,
        "drape_offset": 1.0
    }

Configuration
-------------

VectorVertex attributes: x, y, z (position), r, g, b, a (color 0.0-1.0).

VectorOverlayConfig attributes:

- **name**: Unique layer identifier (required)
- **vertices**: List of VectorVertex (required)
- **indices**: Index buffer for drawing (required)
- **primitive**: PrimitiveType (default: TRIANGLES)
- **drape**: Drape onto terrain (default: False)
- **drape_offset**: Height above terrain in meters (default: 0.5)
- **opacity**: Layer opacity 0.0-1.0 (default: 1.0)
- **depth_bias**: Z-fighting prevention 0.01-1.0 (default: 0.1)
- **line_width**: Line width 1.0-10.0 (default: 2.0)
- **point_size**: Point size 1.0-20.0 (default: 5.0)
- **visible**: Layer visibility (default: True)
- **z_order**: Stacking order, lower = behind (default: 0)

PrimitiveType: POINTS, LINES, LINE_STRIP, TRIANGLES, TRIANGLE_STRIP

Draping
-------

When ``drape=True``, vertices are projected onto terrain:

1. World XZ converted to terrain UV
2. Heightmap sampled with bilinear interpolation
3. Vertex Y set to terrain_height + drape_offset
4. Normal computed from terrain gradient

Lighting
--------

Overlays use the same lighting model as terrain:

- Diffuse: albedo * sun_color * NdotL * sun_intensity * shadow_term
- Shadow: Samples sun_vis_tex at terrain UV
- Ambient: albedo * ambient_intensity

This ensures overlays receive identical lighting and shadows as terrain.

IPC Commands Reference
----------------------

- **add_vector_overlay**: Add new overlay layer
- **remove_vector_overlay**: Remove by ID
- **set_vector_overlay_visible**: Set visibility
- **set_vector_overlay_opacity**: Set opacity
- **list_vector_overlays**: List all IDs
- **set_vector_overlays_enabled**: Enable/disable system
- **set_global_vector_overlay_opacity**: Set global opacity multiplier

Feature Flag
------------

Vector overlays are disabled by default (``vector_overlays_enabled: false``).
Enable via IPC or by adding an overlay layer.
