# Feature coverage: what Forge3D can do, and what Studio brings of it

The engine's own inventory is `docs/guides/feature_map.md` (its workflows, not
its module layout). This page is the other half: for each workflow, what the
Studio exposes, what it cannot reach, and why. It is the answer to "are all
Forge3D's features brought to Studio?" — the honest answer is *no, and here is
exactly which ones and what each one costs*.

Two rules govern the table:

* **A control must change pixels.** A parameter measured at 0 changed samples
  on Studio's render path is not exposed; the audit gate
  (`tests/test_studio_surface_audit.py`) keeps the model free of them.
* **A shell must not re-implement an engine mechanism.** Where the engine has
  a path, Studio uses it (supersampling is `render_scale`, the class palette is
  `thematic.apply_palette`, the camera is `camera_render_view`, the plate's
  legend is `forge3d.legend.Legend`).

## 1. The stage

| Phase | What it was | State |
|---|---|---|
| 0–2 | Shell, project model, Solid viewport | done |
| 3 | Workspace switching | dropped on purpose (rendering lives in the viewport) |
| 4–7 | Rendered mode, materials, timeline, hardening | done |
| 8–9 | Engine facts in the probe, sun from date/place | done |
| 10 | AOV + OIDN final stills | done |
| 11 | Colormaps, sun visibility, vertical scale, despeckle | done |
| 12 | Map plate, map furniture, categorical legend | done |
| 13 | The map look (one button), the release audit as a CI gate | done |
| 14 | **The engine's cartographic tier** (vectors, labels, furniture) | shipped: lines, labels, furniture and the graticule |
| 15 | Convergence on `MapScene` as the scene model | the still is the recipe; step 4 remains (typed raster layers) |

Studio exposes **85 controls** — the app's 55 numeric fields, 3 range sliders,
18 toggles, 5 choice fields and 4 text fields (the earlier count, 68,
predated the vector layer's colour, width and label controls and the
graticule's six came with the grid; the MapScene preview toggle came and went
with it) — over 28
render settings, the scene model (terrain, mesh, camera, light), the timeline,
the map plate and the map's layer stack. The engine's
public Python surface is **383 names** (the label compositor joined it); the
table below is the honest breakdown of what a *terrain and cartography studio*
needs from those.

## 2. The inventory

| Engine workflow (from its feature map) | Studio | Note |
|---|---|---|
| Interactive terrain viewing | **own path** | Studio's viewport is three.js; the still is the engine's offscreen terrain pass |
| Terrain inputs (`mini_dem`, `fetch_dem`, datasets, COG) | partial | local GeoTIFF only; no dataset fetching, no COG streaming |
| Raster overlays (`ViewerHandle.load_overlay`) | **full** | the drape, in the Material preview and in Rendered, at the source's resolution |
| Thematic rasters (categorical/bivariate styles) | partial | the class palette and the elevation ramp; no bivariate styles, no `MapScene` styles |
| Vector overlays and labels | **lines, labels, furniture and the graticule** | the engine's projection + OIT rasterizer for the lines (with the lines' own depth, so a ridge hides them); the engine's label plan + text compositor for the labels; on the quality surface both are drawn by the engine in the terrain's own pass, with the ground's depth; the engine's `ScaleBar`, `NorthArrow` and `generate_graticule` for the furniture and the grid |
| The map as a layer stack | **full** | the studio's own ordered stack (relief, ramp, imagery, vector, labels, graticule, plate), drawn back to front by the render service |
| The gallery's recipes as starter scenes | **full** | the Swiss land-cover, the Luxembourg rail network and the Mount Fuji labels, with their layers, camera and sun |
| Map furniture (title, caption, north arrow, scale bar, legend) | **full** | the plate layer's own switches; the arrow and the bar are the engine's own (`NorthArrow`, `ScaleBar`), the page itself the studio's compositor |
| Picking and selection | partial | the Solid view picks objects; the engine's terrain picking is not used |
| Point clouds | none | a different product tier |
| Camera automation (`animation`, `camera_rigs`) | partial | the timeline's camera path and FOV reach the render; **object channels do not** |
| Terrain quality controls (`terrain_params`, `presets`) | partial | 28 settings are wired; the engine's presets are not exposed |
| Water surfaces and optics | **full** | the mask, the physical surface, the roughness |
| Screen-space effects (SSAO, SSGI, SSR, clouds) | **full** | the engine's quality surface; the still renders on it (the AOV finals cannot) |
| Native/offscreen rendering | **full** | `TerrainRenderer.render_terrain_pbr_pom` |
| Reference-scene matching (`recover_scene`) | none | a research tool |
| Typed map scenes (`MapScene`, `LabelLayer`, buildings, 3D Tiles) | **none** | the phase-14/15 target |
| Geographic preparation (`PreparationRecipe`) | none | the engine prepares; Studio opens what is prepared |
| Ordered map composition (`MapScene.render_passes`) | none | the plate is a Studio-side composite |
| Geometry, mesh, vector, SDF, path tracing | none | tier-2 (different products) |
| Device and memory diagnostics | **full** | the probe and the status bar |
| Map plates and furniture (`MapPlate`, `Legend`, `ScaleBar`, `NorthArrow`) | **own plate** | `MapPlate` is Pro-gated, so Studio composes its own page from the engine's `Legend` |
| Linked publications, vector export, buildings, Mapbox styles, bundles | none | Pro or tier-2 |

## 3. The tools that are not connected

Each of these is a place where the Studio shows something the engine never
sees. They are the reason the app feels loose.

1. **Timeline channels that do not reach the render are now marked.** The
   matrix is derived from the handoff (`workspace::handoff`), not assumed:
   the camera's translation, rotation and FOV reach the export via
   `camera_render_view`; the directional light's rotation reaches it via
   `sun_angles` (azimuth/elevation); everything else — meshes, camera scale,
   light translation/scale, point lights — is preview-only. The Timeline
   shows a precise per-object hint instead of a blanket claim;
   `src/state/animationReach.ts` (pinned by `animationReach.test.ts` and by a
   studio-core test proving the evaluate → handoff sun path) is the one
   source for it. Meshes still cannot reach Rendered (the capability table
   says so); this closes the honesty gap, not the capability gap.
2. **The Material preview matches the still's shadow rig and its diffuse-only
   terrain light.** The terrain's Material material is now
   `MeshPhysicalMaterial` with `specularIntensity = 0` — the engine's
   `lambert_physical` branch (an albedo/π direct term × nDotL × PCF visibility,
   plus the environment irradiance term) has no specular lobe, dielectric or
   otherwise, and neither does the preview anymore. The project's shadow
   settings now reach the preview's shadow rig too: `shadowResolution` sets
   the map size, and `shadowTechnique` maps to `PCFShadowMap` /
   `PCFSoftShadowMap` for `pcf` / `pcss` (three has no blocker search — the
   pcss case is the softer PCF variant, documented, not a real PCSS). The
   slope-scale bias maps to a −1e-04 depth bias; a small world-space normal
   bias keeps grazing light acne-free at viewer scale. The engine's
   ray-marched **sun visibility** (a ridge shadowing the ground behind it)
   still has no three.js counterpart and is a documented gap, like the
   terrain-occluded labels. The terrain editor's Terrain section also offers a
   `Hillshade` raster (same ESRI/QGIS formula, preview aid only).
3. **The preview and the still are two renderers.** They agree on the lighting
   rig, the exposure and the tone-map by construction, but the engine's terrain
   shading (triplanar layers, AO, sun visibility, shadows) only exists in the
   still. The ramp and the drape *are* in both (that is why Phase 8 put the
   ramp in the preview).
4. **The Studio has its own scene model and camera mapping.** The camera's
   Solid-to-renderer mapping is implemented twice — `viewMath.computeRenderView`
   in TypeScript and `workspace::compute_render_view` in Rust — and pinned to
   one shared golden vector so the two cannot drift. Duplication, but a
   *checked* one.
5. **The Studio README links to an untracked file.** `studio-tauri/README.md`
   points at `../FORGE3D_STUDIO_IMPLEMENTATION_PLAN.md`, which was not in the
   repository: the plan the product is built from was invisible to anyone who
   cloned it. Now tracked.

## 4. Cleanup

Done in this pass:

* the dead ambient-occlusion control (0 changed samples, 2 s per frame) —
  deleted, not documented;
* the zScale and frame-size ranges the engine refuses — the model now
  rejects them like the engine does;
* the duplicated camera mapping — kept, but pinned by a golden vector instead
  of by convention;
* the plan file the README links to — tracked.

Still open, in order of value:

* the **Material preview's shadows** (a flat preview against a shadowed still);
* the **timeline's preview-only channels** (a control that cannot reach the
  output);
* the engine's **presets** and **dataset fetching** (Studio opens what is
  prepared; the engine can fetch it);
* the studio's own **plate compositor** versus the engine's Pro `MapPlate` —
  documented, and the reason the plate is not in the Pro tier.

## 5. The phases

**Phase 15 — the map is a layer stack.** The map's decisions used to live in
three places (the terrain object's draped raster and vector layer, the render
settings' colormap and plate). They are one ordered stack now
(`Project::cartography`): each layer has a kind, a name, an eye and its own
style, and the Studio has a **Cartography workspace** — a Photoshop-like layer
panel on the left, the map in the middle, the selected layer's properties on
the right. The shell's own model, but the shape matches MapScene's recipe
layers, so the convergence below is a mapping rather than a rewrite.

**Phase 14 — the engine's cartographic tier** — **shipped: lines, labels,
furniture and the graticule**
(`docs/goal-phase-14-cartographic-tier.md`): the rail network draws on the
Luxembourg DEM and on the plate through the engine's own projection and OIT
rasterizer, and the place names draw through the engine's label plan and its
text compositor (`forge3d.composite_label_layers`, made public for callers
that own their pixels). The lines carry their own depth (the engine's public
`forge3d.vector.projected_depth`), so a ridge in front of a line hides it;
the plate's scale bar and north arrow are the engine's own (`ScaleBar`,
`NorthArrow`), and the graticule is the engine's grid
(`forge3d.graticule.generate_graticule`) with its annotations placed by the
engine's label plan. Still open in the tier: polygons, a legend for the
vector attributes, and labels occluded by the terrain's ridges.

**Phase 15 (MapScene — the still *is* the recipe now).** Every still — and
every viewport frame — goes through the engine's `MapScene` recipe: the studio
composes a `TerrainSource` from the scaled DEM, an `OrbitCamera` and a
`LightingPreset` (sun, IBL path and intensity, sky/fog/water/sun-visibility
settings) from the request, and the engine compiles and renders the frame
with the same TAA warm-up the old default used. Studio composites the numpy
cartography (vectors, labels, graticule) and the plate on top, exactly as
before. The reply reports `renderPath: "mapscene"`, `plate: true | null`, and
`title`/`caption`/etc. unchanged; the shell names the renderer
`forge3d.mapscene.render`. Measured on the Swiss DEM and Mount Fuji at
900×700: **byte-identical to the pre-switch default** — pixel counts from the
roadmap §5.29. The overlays (imagery drape, elevation ramp) ride the
recipe's own shader overlay channel via a baked drape PNG
(`LightingPreset.settings["overlays"]` — JSON-serializable, one colour
source per frame). The AOV/denoise path is the same one-shot
`render_with_aov` route on both sides, byte-identical as well.
**Converge on `MapScene`.** The engine's typed map scene carries labels,
buildings, 3D Tiles, furniture, graticules and bundles; Studio has its own
model and its own service. The map's layer stack (above) is shaped like
MapScene's recipe layers, so the convergence is now a mapping: the stack's
layers become the recipe's layers, and the engine's own compositors draw
them. *Trigger:* the labels already adopted the engine's label plan and text
compositor rather than a second implementation — the rest of the tier can
follow the same road. Step 4 closed the loop — the imagery drape is a typed
`RasterOverlay` layer rendered by the recipe's own UV drape; the elevation
ramp rides the engine's own colormap overlay (the recipe builds a
`Colormap1D` from the project LUT); the baked-PNG compositor channel is
gone (the native UV drape now composites in row bands under the host
budget).

**Phase 16 — the preview matches the still.** Done: the preview's hillshade
raster, the map overlay (vector lines, feature labels and the graticule,
draped on the terrain and read through the still's own reader), the plate's
crop frame, and the Material preview's lighting/shadow match landed, so
authoring shows the output. The one unmatched term is the engine's
ray-marched sun visibility (a ridge shadowing the ground behind it) — no
three.js equivalent, documented.

**Phase 17 — the engine's inputs.** Dataset fetching, the engine's presets,
COG streaming: Studio opens what the engine can fetch.

**Not planned:** path tracing, splats, inverse rendering, point clouds,
volumetrics, the globe. Different product tiers, and the roadmap's §8 already
says so.
