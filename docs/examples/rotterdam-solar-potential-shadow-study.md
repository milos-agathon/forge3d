# Rotterdam Solar Potential And Shadow Study

Runnable example:

```bash
python examples/rotterdam_solar_potential_shadow_study.py
```

The example renders a 3D planning map centered on Erasmus Bridge / Kop van Zuid
in Rotterdam. Roofs are colored by estimated solar suitability, and shadows are
computed for the selected local date/time. Add `--animate` to export a day-cycle
MP4.

## Data Sources

- **3D buildings:** 3D BAG public WFS, layer `BAG3D:lod22`.
  The script uses real LoD2.2 roof-surface polygons with AHN-derived height,
  slope, and azimuth attributes, then reconstructs roof and wall mesh geometry
  for rendering. If a building has no LoD2.2 roof facets in the clipped AOI,
  a 3D BAG `BAG3D:lod12` fallback surface is added so the building is still
  represented. Responses are cached under `examples/.cache/rotterdam_solar_potential/`.
- **Map context:** OpenStreetMap via Overpass for water, roads, rail, parks, and
  bridge context. Park and water layers use lighter way-only queries to avoid
  relation-heavy Overpass timeouts. These layers are visual context only and
  remain secondary to the roof solar layer.
- **Solar irradiation:** PVGIS/JRC `PVcalc` is used when reachable. If it cannot
  be fetched, the example falls back to a documented Rotterdam annual irradiation
  value of `1030 kWh/m2/year`.

## Assumptions

The estimate uses configurable constants for panel efficiency, performance
ratio, minimum usable roof area, roof usable fraction, slope/orientation
penalties, and a selected-time shadow penalty. Defaults are intentionally simple:

```bash
python examples/rotterdam_solar_potential_shadow_study.py \
  --panel-efficiency 0.20 \
  --performance-ratio 0.82 \
  --min-roof-area 18 \
  --shadow-penalty 0.25
```

The output is a planning and visualization estimate. It does not account for
structural capacity, ownership, fire setbacks, monument permits, detailed HVAC
obstructions, inverter design, or current grid-connection capacity.

The legend and assumptions panel is rendered in compact reserved side or bottom
space, not over the map viewport. The `excluded` legend class marks roof facets
that were not counted in the PV total because they hit a screening constraint
such as small usable area, glass roof tagging, or a 3D BAG quality flag.

## Optional Layers

AHN terrain is represented indirectly through the 3D BAG roof and ground-height
attributes. CBS neighborhood summaries, RCE heritage constraints, municipal tree
data, and grid-capacity overlays are not fetched by default; add those as cached
local preprocessing layers if a project needs legal or grid-context screening.
