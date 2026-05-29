# 3D Map Project Ideas

These are recommended example projects that are not already covered by the
current `examples/` catalog. The emphasis is on ideas that exercise different
parts of forge3d: terrain, overlays, buildings, animation, labels, lighting,
and temporal data.

## Recommended Next Projects

| Project | What it would demonstrate | Why it is distinct |
| --- | --- | --- |
| Coastal Storm Surge And Evacuation Routes | Animated coastal water levels, road closures, shelters, evacuation corridors, and exposed neighborhoods. | Extends flood mapping beyond a single city scene into emergency-routing and coastal-risk storytelling. |
| Wildfire Spread And Smoke Exposure | Terrain-based fire perimeter growth, wind direction, smoke plume overlays, and population or asset exposure. | Adds a hazard workflow that combines time, terrain, atmospheric effects, and thematic overlays. |
| Urban Heat Island Map | Building and block-level surface temperature, tree canopy, impervious surface, and shade patterns. | Uses 3D buildings and raster overlays for a city-climate story that differs from flood and transit examples. |
| Bathymetry And Coastal Terrain Viewer | Land elevation, seafloor relief, depth contours, ports, coastlines, and above/below sea-level styling. | Adds a water-and-seafloor terrain example, while existing terrain demos focus mostly on land. |
| Solar Potential And Shadow Study | Sun path animation, roof suitability colors, shadows over time, and candidate solar surfaces. | Exercises lighting, day-cycle logic, buildings, and practical urban analysis in one workflow. |
| Earthquake Shake And Landslide Risk Map | Fault lines, intensity rasters, landslide susceptibility, infrastructure, and terrain exaggeration. | Adds geological risk mapping with line, raster, terrain, and infrastructure layers. |
| 3D Flight Paths And Airspace Map | Airport context, altitude-colored tracks, approach corridors, noise contours, and labels. | Uses the vertical dimension directly, unlike rail and road examples that are mostly surface-bound. |
| Watershed And Rainfall Runoff Simulation | Animated rainfall, flow accumulation, streams, basin boundaries, and downstream gauges. | Turns terrain into a process-oriented hydrology example rather than a static river view. |
| Archaeology Or Historical City Layers | Modern terrain or city context with a time slider for historical roads, walls, sites, and land use. | Demonstrates temporal layer management and label-heavy storytelling. |
| Bike Comfort And Street Slope Map | Street network colored by grade, traffic stress, protected lanes, climb effort, and access. | Adds a human-scale mobility example that is distinct from transit and travel-time demos. |

## Suggested Top Three

1. **Wildfire Spread And Smoke Exposure**: visually strong, clearly different
   from the current examples, and a good test of temporal hazard overlays.
2. **Solar Potential And Shadow Study**: uses existing city, building, lighting,
   and day-cycle capabilities in a practical urban analysis workflow.
3. **Bathymetry And Coastal Terrain Viewer**: fills a terrain gap by showing
   seafloor relief and coastal cartography.

## Selection Notes

- Prefer projects with a clear visual result and a concrete map-reading task.
- Avoid duplicating the existing OSM city, flood, terrain landcover, rail,
  transit, point cloud, and population spike examples.
- Use temporal animation where it adds interpretation, not just motion.
- Keep each example focused on one primary story so it remains runnable and
  understandable from the command line.
