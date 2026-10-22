# Native interactive MapScene frames

MapScene requests enabled effects before TerrainRenderer renders. Screen, mesh and
clipmap cameras use one shared HDR frame when SSAO/GTAO, SSGI, SSR, TAA, cloud
shadows or volumetrics are requested. Fog, sky and height AO alone retain their
existing native path, including existing high-bit-depth and AOV/provenance outputs. Native rendering never calls the NumPy screen-space
or cloud utilities. Those private helpers remain only for non-rendering tests.

The fixed order is:

1. Streaming, clipmap and virtual texture resources; generate the shared cloud field.
2. Cascaded terrain/building shadow depth and moments, and terrain sun visibility.
3. Terrain and opaque building instances into linear HDR and one G-buffer.
4. Heightfield horizon visibility and the existing GTAO, resolved once as
   an energy-bounded visibility product. In combined mode GTAO excludes
   terrain-to-terrain samples, so height AO owns terrain horizons and GTAO owns
   building contacts; each term is applied once. GTAO-only retains all shared
   depth samples, and buildings use GTAO alone.
5. Existing SSGI and SSR. Misses retain the real surface IBL/probe contribution.
6. The existing terrain atmosphere evaluator and viewer froxel build/apply in HDR.
7. Existing equal-resolution TAA.
8. Tone mapping and output transform.
9. Depth-aware unlit point OIT/EDL and cartographic vectors at output resolution,
   preserving authored display colors.
10. Frozen compiled labels and map furniture at output resolution.

Shared depth is positive linear view distance; zero denotes background. Terrain
and scalar-PBR building/scatter instances write compatible encoded view normals,
roughness, material/albedo, velocity and surface identity. Terrain AO uses the
actual surface normal rather than the forward shader's stylized relief normal;
water retains its authored wave normal. Motion is current minus previous UV,
with positive Y down. The existing TAA kernel is specialized to consume that
motion once; previous-camera-plane crossings reject history with finite motion.
Transparent points
sample opaque depth and own a separate representative device-depth texture for
EDL. Their draws cannot write opaque depth. Direct sun shading on terrain and
buildings, aerial perspective and froxels all sample the same cloud field and sun.

Keep existing `screen_space`, `height_ao`, `clouds`, `fog`, `sky`, `volumetrics`
and terrain recipe keys. The existing separate height-AO and SSAO enable flags
express height-only, GTAO-only and combined AO. `screen_space.gtao` is an alias
for the existing `ssao` configuration; conflicting simultaneously enabled
aliases are diagnosed. Effects remain off by default.
Use the default screen camera, `camera.camera_mode="mesh:zup"`, or the clipmap
recipe settings. The interactive screen path draws the existing terrain mesh under an
orthographic camera, so normal hardware raster depth occludes buildings.
Perspective and orthographic effect reconstruction
are specialized without changing the authoritative legacy shaders. Offline accumulation and AOV/provenance exports
retain their separate existing entry points; combining them with interactive
effects is diagnosed. TAAU/render scaling, visibility-buffer shading, localized
density volumes, independently projected tiles, and world-space textured GLTF
geometry currently lack matching shared attachments and are diagnosed explicitly.
Screen-image landmarks use the cartographic route described below.
Nonlinear height curves with height AO are diagnosed because the horizon pass
has no curved-height binding.

Inline point layers need `metadata.coordinate_space="world"` and finite Nx3
`metadata.positions` in the terrain CRS. Color, point size and existing EDL
settings are supported. Cloud `shadow_offset_x/y` (and existing
`wind_offset_x/y` aliases) move the shared GPU field. All participating clouds share one weighted-OIT
accumulator and EDL configuration. Build with the `weighted-oit` feature; missing
features or a source without an XYZ/depth command stream produce diagnostics.

The lower-level `TerrainRenderer.render_interactive(material_set, env_maps,
params, heightmap, water_mask=None, time_seconds=0.0, clouds=None,
label_batches=None, vector_batches=None, furniture_batches=None,
point_batches=None, certificate=None, cache=None)` returns the final `Frame`.
The native ANAMNESIS cache does not yet serialize the shared-depth HDR graph;
passing a non-`None` cache raises before rendering. Call
`interactive_pipeline_info()` for effect enablement, actual stage/resource sizes,
layer stages, depth convention, history invalidation, fallback models, timestamp
availability and readback counters. MapScene embeds that data plus the CENSOR
execution report in `last_render_metadata["native_pipeline"]`.

The routine frame records zero intermediate image transfers and one completed
final beauty read. Label visibility is resolved by `compile_plan()` against the
same camera and terrain/building geometry, then consumed without mutation.
`_interactive_validation_data()` is an explicit internal measurement tool: it
permits additional AOV reads only after the beauty frame was read, and never
feeds those images back to rendering. GPU times come from executed timestamp
queries; `gpu_timestamps_available=false` identifies unavailable measurements.

Validation uses `tests/_concordia_scene.py`, the native-pipeline, GTAO/height-AO,
atmosphere and certificate-order tests. Canonical golden/certificate updates
require owner review. A successful GPU run proves only its recorded adapter and
backend; other backends remain ABSENT until independently rendered.


The private after-beauty validation capture also replays the submitted label and
vector batches into transparent output-resolution targets. `label_coverage` is
RGBA8; `vector_coverage` is also RGBA8. These explicit
validation draws and transfers do not run during `render_interactive`, do not
feed beauty processing, and are excluded from its single-readback certificate.
They allow byte comparisons of real GPU coverage with temporal resolve on and off.


Interactive `MapScene.render` uses the completed native capture directly, so
pass support checks and the saved execution report describe the current frame.
Effects-off rendering keeps its existing Python finalization capture. Ordinary
report inspection is unsigned; signing or replacing canonical certificates
requires the owner's separate approval.

The runnable `examples/mapscene_water_clouds.py` preset uses a mesh camera and
animates the shared cloud field through existing recipe offsets.

The interactive shared attachments render one sample. Requests for
`msaa_samples != 1` are rejected with a shared-G-buffer diagnostic. The report
records both `requested_geometry_sample_count` and `geometry_sample_count`;
MapScene supplies one.


`stage_sizes` identifies each stage's primary image input/output: resource
uploads have no single image output, geometry consumes the DEM, and shadows
publish the actual atlas size. `auxiliary_sizes` records actual array-layer
extents for shadow depth/moments, sun/height visibility, depth-pyramid mip sizes,
AO/SSR targets, froxel volume/apply targets and temporal history. A null entry
denotes a resource not written by the selected frame.


Native vector input authoring evaluates stroke, join and fill coverage only
inside each primitive's geometric support, retaining global pixel coordinates
and the existing antialiasing math. It rebuilds inputs for each frame; camera
motion does not require a cached image. The legacy rasterizer keeps its default
full-canvas behavior for existing output parity.


Temporal history belongs to a persistent TerrainRenderer / shared render context.
A fresh one-shot render has no previous frame: it reports
`temporal_history_used=false`, zero raster jitter, and `mapscene.taa` as
`underdeveloped`. Subsequent frames use the existing Halton jitter sequence and
real history; history resets on resolution, geometry, AO configuration, lighting,
material, environment, fog, sky, volumetrics, water-mask contents, draped imagery
or cloud changes and discontinuous CHRONOS identity. Image caching is bypassed
when TAA is enabled because live history is not part of the scene cache key.
Camera motion is measured in unjittered UV coordinates. Equal-resolution TAA
does not satisfy the separate TEMPORA/TAAU dependency, which remains unavailable.

The authoritative height-AO, SSAO/GTAO and bilateral-filter source files remain
unchanged. CONCORDIA uses checked source specializations; viewer, offline, AOV
and default native renderers retain their original kernels. The visibility
product is a bounded independent-visibility approximation: disjoint geometry
can still block correlated solid angles. Acceptance measures the actual GPU
composition against analytic intervals and a mixed terrain/building PT reference.


Native terrain projection methods accept the additive `shared_frame=False`
argument. MapScene world inputs pass `True` for the shared orthographic camera;
existing callers retain the original projection behavior by default.


When vectors or point commands need visibility, the geometry stage first uses
the existing production terrain/scatter entry with an unjittered camera and
direct resident geometry. It copies the resulting positive-view-depth and
surface-identity textures on GPU, then clears and reuses that same G-buffer for the jittered lighting
frame. Independent existing instance-uniform buffers preserve each encoded
camera. This repeats geometry/shading work for cartographic visibility; it does
not allocate a second G-buffer or introduce another shader implementation.
Both draws and the depth copy stay in the same encoder and submission. The
model is `unjittered_production_geometry_replay`;
`concordia.cartography_depth` records executed time and
`auxiliary_sizes.cartography_depth` records the actual output extent. There is
no CPU depth-image input or intermediate image readback. This extra work is
included in the end-to-end performance measurement.

Existing glTF landmarks with `screen_rect`, `landmark_screen_rect`, or the
legacy default rectangle and no world-geometry features are designated
`post_resolve_cartographic` image assets. Their existing sampled-texture color
style is authored before rendering, uploaded once, and blended in the native
cartographic pass before vectors and labels. Metadata records
`gltf_textured_backend="native_cartographic_landmark"`, `gltf_layer_stages` and
`gltf_material_model="authored_screen_image"`. This supports the existing
screen-image representation; `buildings.textured_pbr` is `underdeveloped`.
The legacy screen-image route reports the same capability level.
World-space textured geometry still blocks until it has shared UV, material
and velocity outputs. No rendered image enters the CPU landmark authoring step.

World labels on the interactive effects route use its camera and terrain/building
occlusion; screen-camera world labels use the shared orthographic camera.
Effects-off recipes retain legacy terrain-only occlusion and project world labels
only with a 3D camera. Existing explicit screen coordinates and frozen projected
anchors retain their coordinate meaning. The effects-off image-cache namespace
invalidates frames produced with the superseded label-visibility rules.

Interactive effects require an output viewport of at least 64x64 pixels. Smaller
requests raise `MapSceneNativeUnavailable` with the requested size and required
native entry point, before terrain parameter validation. Effects-off recipes keep
their existing small-output rendering behavior.

Draped vector depth uses the production procedural grid's bilinear sampling and
triangle diagonal rather than coarse DEM-cell triangles. Ground drapes freeze
terrain-only visibility with a bias derived from each covered pixel's stroke offset and
terrain depth slope. The same unjittered geometry replay copies surface identity
on GPU alongside depth. The cartographic shader uses that identity to hide
ground drapes on buildings and background without a terrain coplanarity test.
Elevated geometry retains the ordinary live depth comparison. Clipmap terrain
visibility remains a deterministic proxy. CPU authoring consumes no rendered image.
Fills and outlines are separate GPU batches, so an occluded outline does not
erase a visible fill beneath it; caps and joins keep their authored raster coverage.

The shared cloud-shadow shader also contains necessary WGSL type/mutability
repairs. Its separate `Scene.enable_cloud_shadows` consumer is covered by a
physical-GPU regression, rather than inferred from the MapScene pipeline.
The single-readback counters count image transfers. Timing-query, LOD and
texture-feedback buffer waits may occur after submission; no such buffer feeds
the geometry-to-post-processing sequence in the routine frame.

Both default-screen routes now frame the full terrain tile. The legacy fullscreen
shader preserves oversized vertex UVs until interpolation, and raster imagery
keeps its top-to-bottom convention without the old factor-of-two compensation.
The physical framing fixture differs at one colour-threshold edge (legacy left
edge 39, shared-mesh left edge 38). The owner-approved parity gate allows at most
one output pixel per bounding-box edge. It still rejects the old zoomed-corner
framing. Protected golden and certificate updates require separate review;
production signing remains on protected main. The mixed 3D camera is unchanged.

The cloud-shadow recipe uses stronger direct sun, lower IBL and stronger
cloud attenuation. The canonical recipe enables no SSR. Its measurement helper
clones the input and enables zero-intensity SSR in both comparison variants to
select the same interactive route with shadows disabled. It requires more than the
unchanged 2.0 pixel mean-difference tolerance on at least 5% of lit terrain,
and requires the whole-image mean difference to exceed 2.0 as well.
